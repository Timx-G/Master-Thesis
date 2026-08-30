import json
from pathlib import Path

import pytest

from uav_mec.analyze_results import verify_checksums
from uav_mec.artifacts import RunArtifactWriter
from uav_mec.publish_canonical import (
    V2_REQUIRED_FAMILIES,
    V2_REQUIRED_HARD_GATES,
    publish_canonical,
    publish_canonical_v2,
)


def source_bundle(path: Path, run_id: str) -> None:
    writer = RunArtifactWriter(path)
    writer.write_json(
        "summary.json",
        {"run_id": run_id, "seeds": [1000, 1001], "methods": {}},
    )
    writer.write_csv("tasks.csv", [{"task_id": 1, "completed": True}])
    writer.write_csv("slots.csv", [{"slot": 0, "runtime_s": 0.01}])
    writer.finalize({"run_id": run_id, "resolved_definition": {}})


def make_sources(tmp_path: Path, gates_pass: bool = True):
    sources = {}
    for name in ("main", "pairing", "resource", "exact"):
        path = tmp_path / f"source-{name}"
        source_bundle(path, name)
        sources[name] = path

    analysis = tmp_path / "source-analysis"
    writer = RunArtifactWriter(analysis)
    writer.write_json("statistics.json", {})
    writer.write_json("metrics.json", {})
    writer.write_json("acceptance.json", {"all_gates_pass": gates_pass})
    writer.finalize({"inputs": {}})

    suite_roots = {}
    for suite in ("mechanism", "scalability", "robustness"):
        case = tmp_path / f"suite-{suite}" / "case0"
        source_bundle(case, f"{suite}-case0")
        suite_roots[suite] = case.parent
    return sources, analysis, suite_roots


def test_publisher_keeps_only_whitelisted_lightweight_evidence(
    tmp_path: Path,
):
    sources, analysis, suite_roots = make_sources(tmp_path)
    output = tmp_path / "canonical-v1"

    publish_canonical(sources, analysis, suite_roots, output)

    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["runs"]["main"] == "runs/main"
    assert index["suites"]["mechanism"]["case0"] == (
        "suites/mechanism/case0"
    )
    assert (output / "analysis/statistics.json").exists()
    assert not list(output.rglob("tasks.csv"))
    assert not list(output.rglob("slots.csv"))
    verify_checksums(output)


def test_publisher_refuses_failed_evidence_gates(tmp_path: Path):
    sources, analysis, suite_roots = make_sources(
        tmp_path,
        gates_pass=False,
    )

    with pytest.raises(RuntimeError, match="failed evidence gates"):
        publish_canonical(
            sources,
            analysis,
            suite_roots,
            tmp_path / "canonical-v1",
        )


def make_v2_sources(tmp_path: Path, *, failed_gate: str | None = None):
    families = {}
    for name in sorted(V2_REQUIRED_FAMILIES):
        path = tmp_path / f"v2-{name}"
        source_bundle(path, name)
        families[name] = path

    analysis = tmp_path / "v2-analysis"
    writer = RunArtifactWriter(analysis)
    hard_gates = {name: True for name in V2_REQUIRED_HARD_GATES}
    if failed_gate is not None:
        hard_gates[failed_gate] = False
    writer.write_json("statistics.json", {"hypotheses": {}})
    writer.write_json("metrics.json", {})
    writer.write_json(
        "acceptance.json",
        {
            "all_gates_pass": failed_gate is None,
            "hard_gates": hard_gates,
            "claims": [
                {
                    "claim_id": "H1",
                    "status": "supported",
                    "evidence": "statistics/H1",
                }
            ],
        },
    )
    writer.finalize({"schema_version": 2})
    return families, analysis


def test_v2_publisher_requires_all_families_and_writes_claim_matrix(
    tmp_path: Path,
):
    families, analysis = make_v2_sources(tmp_path)
    output = tmp_path / "canonical-v2"

    publish_canonical_v2(families, analysis, output)

    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == 2
    assert set(index["families"]) == V2_REQUIRED_FAMILIES
    assert (output / "claim_matrix.csv").read_text(encoding="utf-8") == (
        "claim_id,evidence,status\nH1,statistics/H1,supported\n"
    )
    assert not list(output.rglob("tasks.csv"))
    assert not list(output.rglob("slots.csv"))
    verify_checksums(output)


def test_v2_publisher_refuses_missing_family(tmp_path: Path):
    families, analysis = make_v2_sources(tmp_path)
    families.pop("trace")

    with pytest.raises(ValueError, match="expected exactly these inputs"):
        publish_canonical_v2(
            families,
            analysis,
            tmp_path / "canonical-v2",
        )


def test_v2_publisher_refuses_failed_or_missing_hard_gate(tmp_path: Path):
    gate = next(iter(V2_REQUIRED_HARD_GATES))
    families, analysis = make_v2_sources(tmp_path, failed_gate=gate)

    with pytest.raises(RuntimeError, match="failed hard evidence gates"):
        publish_canonical_v2(
            families,
            analysis,
            tmp_path / "canonical-v2-failed",
        )
