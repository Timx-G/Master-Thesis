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
V2_REQUIRED_FAMILIES = frozenset(
    {
        "main",
        "externality",
        "resource",
        "exact",
        "joint",
        "scaling",
        "robustness",
        "trace",
    }
)
V2_REQUIRED_HARD_GATES = frozenset(
    {
        "constraint_violations_zero",
        "incremental_evaluation_exact",
        "dd_exact_oracle_agreement",
        "shared_trace_identity",
        "complete_provenance",
        "runtime_audit_pass",
    }
)


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


def _validate_v2_acceptance(acceptance: object) -> list[dict[str, object]]:
    if not isinstance(acceptance, dict):
        raise RuntimeError("refusing to publish invalid v2 acceptance data")
    gates = acceptance.get("hard_gates")
    if (
        not acceptance.get("all_gates_pass")
        or not isinstance(gates, dict)
        or set(gates) != V2_REQUIRED_HARD_GATES
        or not all(value is True for value in gates.values())
    ):
        raise RuntimeError("refusing to publish failed hard evidence gates")
    claims = acceptance.get("claims")
    if not isinstance(claims, list):
        raise RuntimeError("v2 acceptance must contain a claim matrix")
    required = {"claim_id", "status", "evidence"}
    normalized: list[dict[str, object]] = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != required:
            raise RuntimeError("invalid v2 claim matrix row")
        normalized.append({name: claim[name] for name in sorted(required)})
    return normalized


def publish_canonical_v2(
    family_bundles: Mapping[str, Path],
    analysis: Path,
    output_dir: Path,
) -> None:
    """Publish a complete, checksummed v2 thesis evidence bundle.

    A family input is one finalized run bundle. Raw task and slot rows are
    deliberately excluded; the canonical package contains only immutable
    summaries, manifests, analysis outputs, and their source-bundle hashes.
    """
    _require_keys(family_bundles, V2_REQUIRED_FAMILIES)
    normalized = {
        name: Path(path) for name, path in family_bundles.items()
    }
    analysis = Path(analysis)
    for source in normalized.values():
        verify_checksums(source)
    verify_checksums(analysis)

    acceptance = _json(analysis / "acceptance.json")
    claim_rows = _validate_v2_acceptance(acceptance)
    writer = RunArtifactWriter(Path(output_dir))
    family_index: dict[str, str] = {}
    provenance: dict[str, str] = {}
    for name, source in sorted(normalized.items()):
        destination = f"families/{name}"
        _publish_json_files(
            writer,
            source,
            destination,
            ("summary.json", "manifest.json"),
        )
        family_index[name] = destination
        provenance[f"family:{name}"] = sha256_file(
            source / "checksums.sha256"
        )

    _publish_json_files(
        writer,
        analysis,
        "analysis",
        ("statistics.json", "metrics.json", "acceptance.json", "manifest.json"),
    )
    writer.write_csv(
        "claim_matrix.csv",
        claim_rows,
        fieldnames=("claim_id", "evidence", "status"),
    )
    provenance["analysis"] = sha256_file(analysis / "checksums.sha256")
    writer.write_json(
        "index.json",
        {
            "schema_version": 2,
            "families": family_index,
            "analysis": "analysis",
            "claim_matrix": "claim_matrix.csv",
        },
    )
    writer.finalize(
        {
            "schema_version": 2,
            "source_bundle_checksums": provenance,
            "hard_gates": sorted(V2_REQUIRED_HARD_GATES),
            "policy": "summary/statistics evidence only; raw rows excluded",
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
