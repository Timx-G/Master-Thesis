import json
from pathlib import Path

import pytest

from uav_mec.analyze_results import verify_checksums
from uav_mec.artifacts import RunArtifactWriter
from uav_mec.publish_canonical import publish_canonical


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
