"""Publish lightweight thesis evidence from checksummed experiment bundles."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from .analyze_results import verify_checksums
from .artifacts import RunArtifactWriter, sha256_file


REQUIRED_RUNS = frozenset({"main", "pairing", "resource", "exact"})
REQUIRED_SUITES = frozenset({"mechanism", "scalability", "robustness"})


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_keys(values: Mapping[str, Path], expected: frozenset[str]) -> None:
    if set(values) != expected:
        raise ValueError(f"expected exactly these inputs: {sorted(expected)}")


def _publish_json_files(
    writer: RunArtifactWriter,
    source: Path,
    destination: str,
    names: tuple[str, ...],
) -> None:
    for name in names:
        writer.write_json(f"{destination}/{name}", _json(source / name))


def _suite_cases(root: Path) -> tuple[Path, ...]:
    cases = tuple(sorted(path for path in root.iterdir() if path.is_dir()))
    if not cases:
        raise ValueError(f"suite contains no case bundles: {root}")
    return cases


def publish_canonical(
    runs: Mapping[str, Path],
    analysis: Path,
    suite_roots: Mapping[str, Path],
    output_dir: Path,
) -> None:
    """Copy only checksummed summaries, manifests, and analysis JSON."""
    _require_keys(runs, REQUIRED_RUNS)
    _require_keys(suite_roots, REQUIRED_SUITES)

    normalized_runs = {name: Path(path) for name, path in runs.items()}
    normalized_suites = {
        name: Path(path) for name, path in suite_roots.items()
    }
    analysis = Path(analysis)

    for source in normalized_runs.values():
        verify_checksums(source)
    verify_checksums(analysis)
    cases_by_suite = {
        name: _suite_cases(root)
        for name, root in normalized_suites.items()
    }
    for cases in cases_by_suite.values():
        for case in cases:
            verify_checksums(case)

    acceptance = _json(analysis / "acceptance.json")
    if not isinstance(acceptance, dict) or not acceptance.get("all_gates_pass"):
        raise RuntimeError("refusing to publish evidence with failed evidence gates")

    writer = RunArtifactWriter(Path(output_dir))
    provenance: dict[str, str] = {}
    run_index: dict[str, str] = {}
    for name, source in sorted(normalized_runs.items()):
        destination = f"runs/{name}"
        _publish_json_files(
            writer,
            source,
            destination,
            ("summary.json", "manifest.json"),
        )
        run_index[name] = destination
        provenance[f"run:{name}"] = sha256_file(
            source / "checksums.sha256"
        )

    _publish_json_files(
        writer,
        analysis,
        "analysis",
        (
            "statistics.json",
            "metrics.json",
            "acceptance.json",
            "manifest.json",
        ),
    )
    provenance["analysis"] = sha256_file(analysis / "checksums.sha256")

    suite_index: dict[str, dict[str, str]] = {}
    for suite_name, cases in sorted(cases_by_suite.items()):
        case_index: dict[str, str] = {}
        for case in cases:
            destination = f"suites/{suite_name}/{case.name}"
            _publish_json_files(
                writer,
                case,
                destination,
                ("summary.json", "manifest.json"),
            )
            case_index[case.name] = destination
            provenance[f"suite:{suite_name}:{case.name}"] = sha256_file(
                case / "checksums.sha256"
            )
        suite_index[suite_name] = case_index

    writer.write_json(
        "index.json",
        {
            "schema_version": 1,
            "runs": run_index,
            "analysis": "analysis",
            "suites": suite_index,
        },
    )
    writer.finalize(
        {
            "schema_version": 1,
            "source_bundle_checksums": provenance,
            "policy": (
                "JSON summary/statistics/manifest only; raw task and slot "
                "rows excluded"
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish passed, lightweight canonical thesis evidence",
    )
    for name in ("main", "pairing", "resource", "exact", "analysis"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("mechanism", "scalability", "robustness", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    publish_canonical(
        {
            "main": args.main,
            "pairing": args.pairing,
            "resource": args.resource,
            "exact": args.exact,
        },
        args.analysis,
        {
            "mechanism": args.mechanism,
            "scalability": args.scalability,
            "robustness": args.robustness,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
