"""Build checksum-closed static-correctness evidence for thesis-v2."""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


_OUTPUT_RELATIVE = Path("artifacts/preregistration/v2/correctness/v2")
_FIXED_INTERPRETER = Path("/home/link/miniconda/envs/llm-ops-torch26/bin/python")
_INTERPRETER_PROPERTY = "correctness_python_executable"
_PYTHON_VERSION_PROPERTY = "correctness_python_version"
_SOURCE_COMMIT_PROPERTY = "correctness_source_commit"
_RESEARCH_TREE_PROPERTY = "correctness_research_tree_sha256"
_SOURCE_HASH_PREFIX = "correctness_source_sha256::"
_HEX = frozenset("0123456789abcdef")

_GATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "plan2_algorithm_audit_record",
        (
            "tests/test_runtime_audit.py::test_algorithm_audit_record_is_frozen_and_has_only_contract_fields",
            "tests/test_runtime_audit.py::test_claim_gate_accepts_complete_static_evidence",
        ),
    ),
    (
        "incremental_equivalence_10000_actions",
        (
            "tests/test_incremental_evaluation.py::test_ten_thousand_random_actions_match_full_closure",
        ),
    ),
    (
        "externality_three_layer_conservation",
        (
            "tests/test_externality_analysis.py::test_certificate_conserves_two_victim_links_and_marks_move_impacts",
            "tests/test_externality_analysis.py::test_realized_admission_and_decision_drive_allocated_cpu_fields",
        ),
    ),
    (
        "dd_exact_admission_agreement",
        (
            "tests/test_exact_admission.py::test_dd_matches_exact_admission_on_small_random_instances",
        ),
    ),
    (
        "method_spec_two_by_two_factorial",
        (
            "tests/test_mechanism_controls.py::test_joint_methods_form_an_exact_two_by_two_factorial",
        ),
    ),
    (
        "closest_work_adaptation_mapping",
        (
            "tests/test_baselines_adapted.py::test_guo_mapping_audit_keeps_adaptation_distinct_from_reproduction",
        ),
    ),
)
_REQUIRED_NODEIDS = tuple(nodeid for _gate, nodeids in _GATES for nodeid in nodeids)
_REQUIRED_SOURCES = tuple(sorted({nodeid.split("::", 1)[0] for nodeid in _REQUIRED_NODEIDS}))
_RESEARCH_ROOTS = ("uav_mec/", "tests/", "configs/thesis/")
_RESEARCH_FILES = frozenset(
    {
        "Makefile",
        "pyproject.toml",
        "requirements.txt",
        "requirements-lock.txt",
        "docs/audits/2026-07-19-adapted-baseline-mapping.md",
        "docs/literature_baseline_audit_2024_2026.md",
    }
)
_EXTERNALITY_SOURCE = "tests/test_externality_analysis.py"


@dataclass(frozen=True, slots=True)
class _ResearchSnapshot:
    sha256: str
    files: Mapping[str, bytes]
    mtimes_ns: Mapping[str, int]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _regular_file(path: Path, label: str) -> Path:
    raw = Path(path)
    try:
        metadata = raw.lstat()
    except OSError as exc:
        raise ValueError(f"{label} must exist as a regular non-symlink file") from exc
    if raw.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    return raw.resolve(strict=True)


def _repo_file(repo_root: Path, relative: str, label: str) -> Path:
    raw = repo_root / relative
    current = raw
    while current != repo_root:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ValueError(f"{label} path cannot be inspected") from exc
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{label} path ancestors must not be symlinks")
        current = current.parent
    resolved = _regular_file(raw, label)
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes repo_root") from exc
    return resolved


def _directory(path: Path, label: str) -> Path:
    raw = Path(path)
    try:
        metadata = raw.lstat()
    except OSError as exc:
        raise ValueError(f"{label} must be an existing directory") from exc
    if raw.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a non-symlink directory")
    return raw.resolve(strict=True)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _assert_finite(value: object, field: str = "document") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} contains a non-finite value")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{field}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{field}[{index}]")


def _safe_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or value != pure.as_posix():
        raise ValueError(f"{field} must be a safe repository-relative path")
    return value


def _registry_inputs(
    repo_root: Path,
    registry_path: Path,
    snapshot: _ResearchSnapshot,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    registry = _regular_file(registry_path, "registry")
    try:
        registry_relative = registry.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError("registry must be inside repo_root") from exc
    _repo_file(repo_root, registry_relative, "registry")
    registry_bytes = snapshot.files.get(registry_relative)
    if registry_bytes is None:
        raise ValueError("registry must be Git tracked research source")
    try:
        parsed = tomllib.loads(registry_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("registry must be valid UTF-8 TOML") from exc
    _assert_finite(parsed, "registry")
    families = parsed.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("registry must contain at least one family")
    config_paths: list[str] = []
    for index, family in enumerate(families):
        if not isinstance(family, Mapping):
            raise ValueError(f"registry.families[{index}] must be a table")
        config_paths.append(
            _safe_relative_path(family.get("config_path"), f"registry.families[{index}].config_path")
        )
    if len(config_paths) != len(set(config_paths)):
        raise ValueError("registry config_path values must be unique")
    configs: list[dict[str, object]] = []
    for relative in sorted(config_paths):
        _repo_file(repo_root, relative, f"config {relative}")
        config_bytes = snapshot.files.get(relative)
        if config_bytes is None:
            raise ValueError(f"config {relative} must be Git tracked research source")
        try:
            parsed_config = tomllib.loads(config_bytes.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"config {relative} must be valid UTF-8 TOML") from exc
        _assert_finite(parsed_config, f"config {relative}")
        configs.append({"path": relative, "sha256": _sha256_bytes(config_bytes)})
    return (
        {
            "path": registry_relative,
            "sha256": _sha256_bytes(registry_bytes),
        },
        configs,
    )


def _source_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40 or any(c not in _HEX for c in commit):
        raise ValueError("repo_root must have a valid Git source commit")
    return commit


def _git_tracked_paths(repo_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError("repo_root must expose its Git tracked file set")
    try:
        values = completed.stdout.decode("utf-8").split("\0")
    except UnicodeDecodeError as exc:
        raise ValueError("Git tracked research paths must be UTF-8") from exc
    return tuple(value for value in values if value)


def _research_tree_snapshot(repo_root: Path) -> _ResearchSnapshot:
    selected = tuple(
        sorted(
            relative
            for relative in _git_tracked_paths(repo_root)
            if relative in _RESEARCH_FILES
            or relative.startswith(_RESEARCH_ROOTS)
            or (
                "/" not in relative
                and relative.startswith("requirements")
                and relative.endswith(".txt")
            )
        )
    )
    required_exact = _RESEARCH_FILES - {"requirements.txt", "requirements-lock.txt"}
    missing = required_exact - set(selected)
    if missing or not any(path.startswith("uav_mec/") for path in selected):
        raise ValueError(f"research source tree is incomplete: {sorted(missing)}")
    digest = hashlib.sha256()
    files: dict[str, bytes] = {}
    mtimes: dict[str, int] = {}
    for relative in selected:
        source = _repo_file(repo_root, relative, f"research source {relative}")
        encoded = relative.encode("utf-8")
        value = source.read_bytes()
        files[relative] = value
        mtimes[relative] = source.stat().st_mtime_ns
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return _ResearchSnapshot(digest.hexdigest(), files, mtimes)


def _assert_attribute_names(function: ast.FunctionDef) -> set[str]:
    return {
        node.attr
        for statement in function.body
        if isinstance(statement, ast.Assert)
        for node in ast.walk(statement.test)
        if isinstance(node, ast.Attribute)
    }


def _validate_externality_conservation_source(snapshot: _ResearchSnapshot) -> None:
    try:
        tree = ast.parse(snapshot.files[_EXTERNALITY_SOURCE].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, SyntaxError) as exc:
        raise ValueError("externality conservation source must be valid tracked Python") from exc
    functions = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    expected = {
        "test_certificate_conserves_two_victim_links_and_marks_move_impacts": {"exposure_delta_w"},
        "test_realized_admission_and_decision_drive_allocated_cpu_fields": {
            "completion_delta",
            "actual_allocated_cpu_delta_hz",
        },
    }
    missing: list[str] = []
    for function_name, attributes in expected.items():
        function = functions.get(function_name)
        if not isinstance(function, ast.FunctionDef):
            missing.append(function_name)
            continue
        missing.extend(sorted(attributes - _assert_attribute_names(function)))
    if missing:
        raise ValueError(
            "externality three-layer conservation source evidence is missing " + ", ".join(missing)
        )


def _property_table(suite: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    properties_elements = suite.findall("properties")
    if len(properties_elements) != 1:
        raise ValueError("pytest testsuite must contain exactly one direct properties element")
    for property_element in properties_elements[0].findall("property"):
        name = property_element.get("name")
        value = property_element.get("value")
        if not name or value is None:
            raise ValueError("JUnit properties must have name and value")
        if name in result and result[name] != value:
            raise ValueError(f"JUnit property {name!r} is ambiguous")
        result[name] = value
    return result


def _testcase_nodeid(case: ET.Element) -> str:
    explicit = case.get("nodeid")
    if explicit:
        return explicit
    name = case.get("name")
    file_name = case.get("file")
    if not name:
        raise ValueError("JUnit testcase is missing its name")
    if file_name:
        normalized_file = PurePosixPath(file_name.replace("\\", "/")).as_posix()
    else:
        classname = case.get("classname")
        if not classname:
            raise ValueError("JUnit testcase is missing both file and classname")
        normalized_file = classname.replace(".", "/") + ".py"
    return f"{normalized_file}::{name}"


def _finite_duration(element: ET.Element, field: str) -> float:
    raw = element.get("time", "0")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{field} duration must be finite") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field} duration must be finite and non-negative")
    return value


def _report_evidence(
    report_bytes: bytes,
    report_mtime_ns: int,
    snapshot: _ResearchSnapshot,
    *,
    source_commit: str,
    research_tree_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, int]]:
    try:
        root = ET.fromstring(report_bytes)
    except ET.ParseError as exc:
        raise ValueError("test report must be valid JUnit XML") from exc
    if root.tag == "testsuite":
        suites = (root,)
    elif root.tag == "testsuites":
        suites = tuple(root.findall("testsuite"))
        if len(root) != len(suites):
            raise ValueError("testsuites may contain only direct testsuite children")
    else:
        raise ValueError("test report root must be testsuite or testsuites")
    if len(suites) != 1 or suites[0].get("name") != "pytest":
        raise ValueError("JUnit report must contain exactly one direct pytest testsuite")
    suite = suites[0]
    _finite_duration(suite, "JUnit suite")
    properties = _property_table(suite)
    expected_executable = str(_FIXED_INTERPRETER.resolve(strict=True))
    if Path(sys.executable).resolve() != _FIXED_INTERPRETER.resolve(strict=True):
        raise ValueError("correctness bundles must be built by the fixed thesis interpreter")
    if properties.get(_INTERPRETER_PROPERTY) != expected_executable:
        raise ValueError("JUnit report interpreter does not match the fixed current interpreter")
    if properties.get(_PYTHON_VERSION_PROPERTY) != platform.python_version():
        raise ValueError("JUnit report Python version does not match the fixed current interpreter")
    if properties.get(_SOURCE_COMMIT_PROPERTY) != source_commit:
        raise ValueError("JUnit report source commit does not match repo_root HEAD")
    reported_tree = properties.get(_RESEARCH_TREE_PROPERTY)
    if not _is_sha256(reported_tree) or reported_tree != research_tree_sha256:
        raise ValueError("JUnit report research source tree hash does not match repo_root")

    source_hashes: dict[str, str] = {}
    newest_source_mtime_ns = 0
    for relative in _REQUIRED_SOURCES:
        source_bytes = snapshot.files.get(relative)
        if source_bytes is None:
            raise ValueError(f"required test source {relative} is outside research snapshot")
        digest = _sha256_bytes(source_bytes)
        reported_digest = properties.get(f"{_SOURCE_HASH_PREFIX}{relative}")
        if not _is_sha256(reported_digest) or reported_digest != digest:
            raise ValueError(f"JUnit source hash mismatch for {relative}")
        source_hashes[relative] = digest
        newest_source_mtime_ns = max(newest_source_mtime_ns, snapshot.mtimes_ns[relative])
    if report_mtime_ns < newest_source_mtime_ns:
        raise ValueError("JUnit report is stale relative to its required test sources")

    all_cases = suite.findall("testcase")
    if len(suite.findall(".//testcase")) != len(all_cases):
        raise ValueError("pytest testcases must be direct testsuite children")
    summary = {"passed": 0, "failure": 0, "error": 0, "skipped": 0}
    cases_by_nodeid: dict[str, list[ET.Element]] = {}
    for case in all_cases:
        status = "passed"
        for candidate in ("failure", "error", "skipped"):
            if case.find(candidate) is not None:
                status = candidate
                break
        summary[status] += 1
        cases_by_nodeid.setdefault(_testcase_nodeid(case), []).append(case)
    for status in ("failure", "error", "skipped"):
        if summary[status]:
            suffix = "; test failed" if status == "failure" else ""
            raise ValueError(
                f"JUnit report contains {summary[status]} {status} testcase(s){suffix}"
            )
    rows: list[dict[str, object]] = []
    for nodeid in _REQUIRED_NODEIDS:
        cases = cases_by_nodeid.get(nodeid, [])
        if not cases:
            raise ValueError(f"JUnit report is missing required test {nodeid}")
        if len(cases) != 1:
            raise ValueError(f"JUnit report contains duplicate required test {nodeid}")
        case = cases[0]
        if case.find("failure") is not None or case.find("error") is not None:
            raise ValueError(f"required test failed: {nodeid}")
        if case.find("skipped") is not None:
            raise ValueError(f"required test skipped: {nodeid}")
        source = nodeid.split("::", 1)[0]
        rows.append(
            {
                "nodeid": nodeid,
                "source_path": source,
                "source_sha256": source_hashes[source],
                "status": "passed",
                "duration_s": _finite_duration(case, f"required test {nodeid}"),
            }
        )
    return (
        rows,
        {
            "executable": expected_executable,
            "python_version": platform.python_version(),
        },
        {"total": len(all_cases), **summary},
    )


def _validate_output(repo_root: Path, output_dir: Path) -> Path:
    output = Path(os.path.abspath(os.fspath(output_dir)))
    expected = repo_root / _OUTPUT_RELATIVE
    if output != expected:
        raise ValueError(f"output directory must be exactly {expected}")
    if output.exists() or output.is_symlink():
        raise FileExistsError("correctness output directory must not already exist")
    current = output.parent
    while current != repo_root:
        if current.exists() and current.is_symlink():
            raise ValueError("correctness output ancestors must not be symlinks")
        if current == current.parent:
            raise ValueError("correctness output must remain inside repo_root")
        current = current.parent
    return output


def _write_file(path: Path, value: bytes) -> None:
    path.write_bytes(value)
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"bundle payload {path.name} is not a regular file")


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without ever replacing a target."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Linux renameat2 is required for no-replace correctness publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "correctness output directory must not already exist",
            target,
        )
    if error_number in {errno.ENOSYS, errno.EINVAL}:
        raise RuntimeError("filesystem does not support atomic renameat2 RENAME_NOREPLACE")
    raise OSError(error_number, os.strerror(error_number), target)


def build_correctness_bundle(
    test_report_path: Path | str,
    repo_root: Path | str,
    registry_path: Path | str,
    output_dir: Path | str,
) -> Path:
    """Build a new immutable correctness bundle, refusing overwrite or drift."""

    root = _directory(Path(repo_root), "repo_root")
    output = _validate_output(root, Path(output_dir))
    report = _regular_file(Path(test_report_path), "test report")
    source_commit = _source_commit(root)
    research_snapshot = _research_tree_snapshot(root)
    registry, configs = _registry_inputs(root, Path(registry_path), research_snapshot)
    _validate_externality_conservation_source(research_snapshot)
    report_bytes = report.read_bytes()
    required_tests, interpreter, test_summary = _report_evidence(
        report_bytes,
        report.stat().st_mtime_ns,
        research_snapshot,
        source_commit=source_commit,
        research_tree_sha256=research_snapshot.sha256,
    )
    evidence = {
        "schema_version": 1,
        "source_commit": source_commit,
        "research_tree_sha256": research_snapshot.sha256,
        "registry": registry,
        "configs": configs,
        "test_report_sha256": _sha256_bytes(report_bytes),
        "interpreter": interpreter,
        "test_summary": test_summary,
        "required_tests": required_tests,
        "static_gates": {
            name: list(nodeids) for name, nodeids in _GATES
        }
        | {
            "externality_conservation_layers": {
                "exposure": _GATES[2][1][0],
                "completion": _GATES[2][1][1],
                "allocated_cpu": _GATES[2][1][1],
            }
        },
    }
    evidence_bytes = _canonical_json(evidence)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".correctness-v2-", dir=output.parent))
    try:
        payloads = {
            "EVIDENCE.json": evidence_bytes,
            "TEST_REPORT.xml": report_bytes,
        }
        for name, value in payloads.items():
            _write_file(temporary / name, value)
        manifest = {
            "schema_version": 1,
            "payload_files": [
                {
                    "path": name,
                    "sha256": _sha256_file(temporary / name),
                    "size_bytes": (temporary / name).stat().st_size,
                }
                for name in sorted(payloads)
            ],
        }
        _write_file(temporary / "MANIFEST.json", _canonical_json(manifest))
        checksum_names = (*sorted(payloads), "MANIFEST.json")
        checksum_bytes = "".join(
            f"{_sha256_file(temporary / name)}  {name}\n" for name in checksum_names
        ).encode("ascii")
        _write_file(temporary / "SHA256SUMS", checksum_bytes)
        if _source_commit(root) != source_commit:
            raise ValueError("source commit drifted while building correctness bundle")
        final_snapshot = _research_tree_snapshot(root)
        if final_snapshot.sha256 != research_snapshot.sha256:
            raise ValueError("research source tree drifted while building correctness bundle")
        _rename_noreplace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build",
        help="build a checksum-closed correctness bundle",
        allow_abbrev=False,
    )
    build.add_argument("--test-report", type=Path, required=True)
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--registry", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        build_correctness_bundle(args.test_report, args.repo_root, args.registry, args.output)
        return 0
    raise AssertionError(f"unreachable command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
