from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import uav_mec.correctness_bundle as correctness_bundle
from uav_mec.correctness_bundle import build_correctness_bundle, main


REQUIRED_NODEIDS = (
    "tests/test_runtime_audit.py::test_algorithm_audit_record_is_frozen_and_has_only_contract_fields",
    "tests/test_runtime_audit.py::test_claim_gate_accepts_complete_static_evidence",
    "tests/test_incremental_evaluation.py::test_ten_thousand_random_actions_match_full_closure",
    "tests/test_externality_analysis.py::test_certificate_conserves_two_victim_links_and_marks_move_impacts",
    "tests/test_externality_analysis.py::test_realized_admission_and_decision_drive_allocated_cpu_fields",
    "tests/test_exact_admission.py::test_dd_matches_exact_admission_on_small_random_instances",
    "tests/test_mechanism_controls.py::test_joint_methods_form_an_exact_two_by_two_factorial",
    "tests/test_baselines_adapted.py::test_guo_mapping_audit_keeps_adaptation_distinct_from_reproduction",
)
RESEARCH_ROOTS = ("uav_mec/", "tests/", "configs/thesis/")
RESEARCH_FILES = {
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
    "docs/audits/2026-07-19-adapted-baseline-mapping.md",
    "docs/literature_baseline_audit_2024_2026.md",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def _bind_real_junit_report_to_interpreter_and_sources(
    record_testsuite_property: pytest.FixtureRequest,
) -> None:
    """Make a full-suite JUnit report usable as checksum-closed evidence."""

    record_testsuite_property("correctness_python_executable", str(Path(sys.executable).resolve()))
    record_testsuite_property("correctness_python_version", platform.python_version())
    root = Path(__file__).resolve().parents[1]
    record_testsuite_property("correctness_source_commit", _git(root, "rev-parse", "HEAD"))
    record_testsuite_property(
        "correctness_research_tree_sha256", _research_tree_sha256(root)
    )
    for relative in sorted({nodeid.split("::", 1)[0] for nodeid in REQUIRED_NODEIDS}):
        record_testsuite_property(f"correctness_source_sha256::{relative}", _sha256(root / relative))


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _research_tree_sha256(repo: Path) -> str:
    tracked = _git(repo, "ls-files", "-z").split("\0")
    selected = sorted(
        relative
        for relative in tracked
        if relative
        and (
            relative in RESEARCH_FILES
            or relative.startswith(RESEARCH_ROOTS)
            or (
                "/" not in relative
                and relative.startswith("requirements")
                and relative.endswith(".txt")
            )
        )
    )
    digest = hashlib.sha256()
    for relative in selected:
        encoded = relative.encode("utf-8")
        value = (repo / relative).read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    source_paths = sorted({nodeid.split("::", 1)[0] for nodeid in REQUIRED_NODEIDS})
    for relative in source_paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "tests/test_externality_analysis.py":
            path.write_text(
                "def test_certificate_conserves_two_victim_links_and_marks_move_impacts():\n"
                "    assert certificate.exposure_delta_w == sum(row.exposure_delta_w for row in certificate.block_deltas)\n"
                "\n"
                "def test_realized_admission_and_decision_drive_allocated_cpu_fields():\n"
                "    assert certificate.completion_delta == sum(int(row.completed_after) - int(row.completed_before) for row in certificate.task_impacts)\n"
                "    assert certificate.actual_allocated_cpu_delta_hz == sum(row.allocated_cpu_after_hz - row.allocated_cpu_before_hz for row in certificate.task_impacts)\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"# immutable evidence source: {relative}\n", encoding="utf-8")
    production = repo / "uav_mec/kernel.py"
    production.parent.mkdir(parents=True)
    production.write_text("VALUE = 1\n", encoding="utf-8")
    for relative in (
        "docs/audits/2026-07-19-adapted-baseline-mapping.md",
        "docs/literature_baseline_audit_2024_2026.md",
    ):
        document = repo / relative
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(f"# {relative}\n", encoding="utf-8")
    (repo / "Makefile").write_text("test:\n\tpython -m pytest\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        "[project]\nname = \"correctness-fixture\"\nversion = \"1.0.0\"\n",
        encoding="utf-8",
    )
    config = repo / "configs/thesis/confirmatory.toml"
    config.parent.mkdir(parents=True)
    config.write_text("schema_version = 2\nvalue = 1.0\n", encoding="utf-8")
    registry = repo / "configs/thesis/experiment_registry_v2.toml"
    registry.write_text(
        "schema_version = 2\n"
        "[[families]]\n"
        "name = \"main\"\n"
        "config_path = \"configs/thesis/confirmatory.toml\"\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Correctness Test",
        "-c",
        "user.email=correctness@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return repo, registry


def _write_report(
    repo: Path,
    path: Path,
    *,
    nodeids: tuple[str, ...] = REQUIRED_NODEIDS,
    failed_nodeid: str | None = None,
    skipped_nodeid: str | None = None,
    executable: str | None = None,
    source_hash_overrides: dict[str, str] | None = None,
    testcase_time: str = "0.01",
) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "pytest",
            "tests": str(len(nodeids)),
            "failures": str(int(failed_nodeid is not None)),
            "errors": "0",
            "skipped": str(int(skipped_nodeid is not None)),
            "time": testcase_time,
        },
    )
    properties = ET.SubElement(suite, "properties")
    values = {
        "correctness_python_executable": executable or str(Path(sys.executable).resolve()),
        "correctness_python_version": platform.python_version(),
        "correctness_source_commit": _git(repo, "rev-parse", "HEAD"),
        "correctness_research_tree_sha256": _research_tree_sha256(repo),
    }
    overrides = source_hash_overrides or {}
    for relative in sorted({nodeid.split("::", 1)[0] for nodeid in REQUIRED_NODEIDS}):
        values[f"correctness_source_sha256::{relative}"] = overrides.get(
            relative, _sha256(repo / relative)
        )
    for name, value in values.items():
        ET.SubElement(properties, "property", {"name": name, "value": value})
    for nodeid in nodeids:
        relative, name = nodeid.split("::", 1)
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": relative.removesuffix(".py").replace("/", "."),
                "name": name,
                "file": relative,
                "time": testcase_time,
            },
        )
        if nodeid == failed_nodeid:
            ET.SubElement(case, "failure", {"message": "assertion failed"})
        if nodeid == skipped_nodeid:
            ET.SubElement(case, "skipped", {"message": "not run"})
    path.write_bytes(ET.tostring(suite, encoding="utf-8", xml_declaration=True))


def _output(repo: Path) -> Path:
    return repo / "artifacts/preregistration/v2/correctness/v2"


def test_build_correctness_bundle_is_checksum_closed_and_records_static_gates(
    tmp_path: Path,
) -> None:
    repo, registry = _make_repo(tmp_path)
    report = repo / "fresh-junit.xml"
    _write_report(repo, report)

    result = build_correctness_bundle(report, repo, registry, _output(repo))

    assert result == _output(repo)
    assert tuple(path.name for path in sorted(result.iterdir())) == (
        "EVIDENCE.json",
        "MANIFEST.json",
        "SHA256SUMS",
        "TEST_REPORT.xml",
    )
    evidence = json.loads((result / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert evidence["source_commit"] == _git(repo, "rev-parse", "HEAD")
    assert evidence["research_tree_sha256"] == _research_tree_sha256(repo)
    assert evidence["registry"]["sha256"] == _sha256(registry)
    assert evidence["configs"] == [
        {
            "path": "configs/thesis/confirmatory.toml",
            "sha256": _sha256(repo / "configs/thesis/confirmatory.toml"),
        }
    ]
    assert tuple(row["nodeid"] for row in evidence["required_tests"]) == REQUIRED_NODEIDS
    assert all(row["status"] == "passed" for row in evidence["required_tests"])
    assert evidence["test_summary"] == {
        "error": 0,
        "failure": 0,
        "passed": len(REQUIRED_NODEIDS),
        "skipped": 0,
        "total": len(REQUIRED_NODEIDS),
    }
    assert evidence["interpreter"] == {
        "executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
    }
    assert evidence["static_gates"]["externality_conservation_layers"] == {
        "allocated_cpu": REQUIRED_NODEIDS[4],
        "completion": REQUIRED_NODEIDS[4],
        "exposure": REQUIRED_NODEIDS[3],
    }
    manifest = json.loads((result / "MANIFEST.json").read_text(encoding="utf-8"))
    assert [row["path"] for row in manifest["payload_files"]] == [
        "EVIDENCE.json",
        "TEST_REPORT.xml",
    ]
    for row in manifest["payload_files"]:
        assert row["sha256"] == _sha256(result / row["path"])
        assert row["size_bytes"] == (result / row["path"]).stat().st_size
    checksum_rows = (result / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    assert checksum_rows == [
        f"{_sha256(result / name)}  {name}"
        for name in ("EVIDENCE.json", "TEST_REPORT.xml", "MANIFEST.json")
    ]


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_bundle_rejects_missing_failed_or_skipped_required_tests(
    tmp_path: Path, status: str
) -> None:
    repo, registry = _make_repo(tmp_path)
    report = repo / "fresh-junit.xml"
    if status == "failed":
        _write_report(repo, report, failed_nodeid=REQUIRED_NODEIDS[0])
    else:
        _write_report(repo, report, skipped_nodeid=REQUIRED_NODEIDS[0])
    with pytest.raises(ValueError, match=status):
        build_correctness_bundle(report, repo, registry, _output(repo))
    assert not _output(repo).exists()

    _write_report(repo, report, nodeids=REQUIRED_NODEIDS[1:])
    with pytest.raises(ValueError, match="missing required test"):
        build_correctness_bundle(report, repo, registry, _output(repo))
    assert not _output(repo).exists()


def test_bundle_rejects_stale_or_mismatched_test_source_hashes(tmp_path: Path) -> None:
    repo, registry = _make_repo(tmp_path)
    report = repo / "fresh-junit.xml"
    relative = REQUIRED_NODEIDS[0].split("::", 1)[0]
    _write_report(repo, report, source_hash_overrides={relative: "0" * 64})
    with pytest.raises(ValueError, match="source hash"):
        build_correctness_bundle(report, repo, registry, _output(repo))

    _write_report(repo, report)
    source = repo / relative
    source_stat = source.stat()
    os.utime(source, ns=(source_stat.st_atime_ns, report.stat().st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="stale"):
        build_correctness_bundle(report, repo, registry, _output(repo))
    assert not _output(repo).exists()


def test_bundle_rejects_production_tree_or_source_commit_drift_after_report(
    tmp_path: Path,
) -> None:
    repo, registry = _make_repo(tmp_path / "production-drift")
    report = repo / "fresh-junit.xml"
    _write_report(repo, report)
    (repo / "uav_mec/kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="research source tree"):
        build_correctness_bundle(report, repo, registry, _output(repo))

    repo, registry = _make_repo(tmp_path / "commit-drift")
    report = repo / "fresh-junit.xml"
    _write_report(repo, report)
    _git(
        repo,
        "-c",
        "user.name=Correctness Test",
        "-c",
        "user.email=correctness@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "new commit",
    )
    with pytest.raises(ValueError, match="source commit"):
        build_correctness_bundle(report, repo, registry, _output(repo))


def test_bundle_rejects_externality_source_without_completion_or_cpu_conservation(
    tmp_path: Path,
) -> None:
    for missing in ("completion_delta", "actual_allocated_cpu_delta_hz"):
        repo, registry = _make_repo(tmp_path / missing)
        source = repo / "tests/test_externality_analysis.py"
        source.write_text(
            "\n".join(
                line
                for line in source.read_text(encoding="utf-8").splitlines()
                if missing not in line
            )
            + "\n",
            encoding="utf-8",
        )
        _git(repo, "add", ".")
        _git(
            repo,
            "-c",
            "user.name=Correctness Test",
            "-c",
            "user.email=correctness@example.invalid",
            "commit",
            "-qm",
            f"remove {missing}",
        )
        report = repo / "fresh-junit.xml"
        _write_report(repo, report)
        with pytest.raises(ValueError, match="externality.*conservation"):
            build_correctness_bundle(report, repo, registry, _output(repo))


def test_bundle_rejects_any_nonrequired_failed_error_or_skipped_test(tmp_path: Path) -> None:
    for status in ("failure", "error", "skipped"):
        repo, registry = _make_repo(tmp_path / status)
        report = repo / "fresh-junit.xml"
        _write_report(repo, report)
        root = ET.fromstring(report.read_bytes())
        case = ET.SubElement(
            root,
            "testcase",
            {"classname": "tests.test_other", "name": "test_other", "time": "0.0"},
        )
        ET.SubElement(case, status, {"message": status})
        report.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
        with pytest.raises(ValueError, match=f"JUnit report contains.*{status}"):
            build_correctness_bundle(report, repo, registry, _output(repo))


def test_bundle_rejects_wrong_interpreter_nan_and_infinite_durations(tmp_path: Path) -> None:
    repo, registry = _make_repo(tmp_path)
    report = repo / "fresh-junit.xml"
    _write_report(repo, report, executable="/different/python")
    with pytest.raises(ValueError, match="interpreter"):
        build_correctness_bundle(report, repo, registry, _output(repo))
    for value in ("NaN", "inf", "-Inf"):
        _write_report(repo, report, testcase_time=value)
        with pytest.raises(ValueError, match="finite"):
            build_correctness_bundle(report, repo, registry, _output(repo))
    assert not _output(repo).exists()


def test_bundle_refuses_wrong_output_existing_extra_files_and_symlinks(tmp_path: Path) -> None:
    repo, registry = _make_repo(tmp_path)
    report = repo / "fresh-junit.xml"
    _write_report(repo, report)
    with pytest.raises(ValueError, match="output directory"):
        build_correctness_bundle(report, repo, registry, repo / "runs/correctness")

    output = _output(repo)
    output.mkdir(parents=True)
    (output / "EXTRA").write_text("not admissible\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must not already exist"):
        build_correctness_bundle(report, repo, registry, output)

    link_repo, link_registry = _make_repo(tmp_path / "symlink-case")
    link_report = link_repo / "fresh-junit.xml"
    _write_report(link_repo, link_report)
    symlink_report = link_repo / "report-link.xml"
    symlink_report.symlink_to(link_report)
    with pytest.raises(ValueError, match="regular non-symlink"):
        build_correctness_bundle(symlink_report, link_repo, link_registry, _output(link_repo))


def test_correctness_bundle_cli_build_matches_shared_contract(tmp_path: Path) -> None:
    repo, registry = _make_repo(tmp_path)
    report = repo / "fresh-junit.xml"
    _write_report(repo, report)
    output = _output(repo)
    assert (
        main(
            [
                "build",
                "--test-report",
                str(report),
                "--repo-root",
                str(repo),
                "--registry",
                str(registry),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "SHA256SUMS").is_file()


def test_correctness_bundle_cli_rejects_abbreviated_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "build",
                "--test-re",
                "report.xml",
                "--test-report",
                "report.xml",
                "--repo-root",
                ".",
                "--registry",
                "registry.toml",
                "--output",
                "output",
            ]
        )
    assert "unrecognized arguments: --test-re" in capsys.readouterr().err


def test_bundle_validates_hashes_and_archives_one_junit_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, registry = _make_repo(tmp_path)
    report = repo / "fresh-junit.xml"
    _write_report(repo, report)
    expected = report.read_bytes()
    original_read_bytes = Path.read_bytes
    reads = 0

    def adversarial_read(path: Path) -> bytes:
        nonlocal reads
        if path == report:
            reads += 1
            if reads > 1:
                return b"<testsuite name='attacker'/>"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", adversarial_read)
    result = build_correctness_bundle(report, repo, registry, _output(repo))
    assert reads == 1
    assert (result / "TEST_REPORT.xml").read_bytes() == expected


def test_bundle_accepts_one_pytest_suite_wrapper_and_rejects_mixed_suites(
    tmp_path: Path,
) -> None:
    repo, registry = _make_repo(tmp_path / "one")
    report = repo / "fresh-junit.xml"
    _write_report(repo, report)
    suite = ET.fromstring(report.read_bytes())
    wrapper = ET.Element("testsuites")
    wrapper.append(suite)
    report.write_bytes(ET.tostring(wrapper, encoding="utf-8", xml_declaration=True))
    assert build_correctness_bundle(report, repo, registry, _output(repo)).is_dir()

    repo, registry = _make_repo(tmp_path / "two")
    report = repo / "fresh-junit.xml"
    _write_report(repo, report)
    trusted = ET.fromstring(report.read_bytes())
    wrapper = ET.Element("testsuites")
    wrapper.extend((trusted, ET.Element("testsuite", {"name": "attacker", "time": "0"})))
    report.write_bytes(ET.tostring(wrapper, encoding="utf-8", xml_declaration=True))
    with pytest.raises(ValueError, match="exactly one.*pytest testsuite"):
        build_correctness_bundle(report, repo, registry, _output(repo))


def test_externality_comments_or_other_functions_cannot_impersonate_assertions(
    tmp_path: Path,
) -> None:
    repo, registry = _make_repo(tmp_path)
    source = repo / "tests/test_externality_analysis.py"
    source.write_text(
        "def test_certificate_conserves_two_victim_links_and_marks_move_impacts():\n"
        "    assert certificate.exposure_delta_w == 0\n"
        "\n"
        "def test_realized_admission_and_decision_drive_allocated_cpu_fields():\n"
        "    assert True\n"
        "    # completion_delta actual_allocated_cpu_delta_hz\n"
        "\n"
        "def decoy():\n"
        "    assert certificate.completion_delta == 0\n"
        "    assert certificate.actual_allocated_cpu_delta_hz == 0\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=T", "-c", "user.email=t@invalid", "commit", "-qm", "decoy")
    report = repo / "fresh-junit.xml"
    _write_report(repo, report)
    with pytest.raises(ValueError, match="externality.*conservation"):
        build_correctness_bundle(report, repo, registry, _output(repo))


def test_rename_noreplace_refuses_an_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    with pytest.raises(FileExistsError):
        correctness_bundle._rename_noreplace(source, target)
    assert source.is_dir() and target.is_dir()


def test_bundle_rejects_config_parent_symlink_escape(tmp_path: Path) -> None:
    repo, registry = _make_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "confirmatory.toml").write_text(
        "schema_version = 2\nvalue = 1.0\n", encoding="utf-8"
    )
    (repo / "configs/escape").symlink_to(outside, target_is_directory=True)
    registry.write_text(
        "schema_version = 2\n"
        "[[families]]\n"
        "name = \"main\"\n"
        "config_path = \"configs/escape/confirmatory.toml\"\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=T", "-c", "user.email=t@invalid", "commit", "-qm", "escape")
    report = repo / "fresh-junit.xml"
    _write_report(repo, report)
    with pytest.raises(ValueError, match="config.*symlink"):
        build_correctness_bundle(report, repo, registry, _output(repo))
