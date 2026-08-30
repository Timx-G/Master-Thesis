import csv
import json
from dataclasses import replace
from pathlib import Path

from uav_mec.experiment import load_definition
from uav_mec.run_exact_benchmark import run_exact_definition


def test_exact_benchmark_never_reports_heuristic_above_optimum(
    tmp_path: Path,
):
    config_path = Path("configs/thesis/confirmatory_exact.toml")
    base = load_definition(config_path)
    definition = replace(
        base,
        seeds=(42,),
        scenario=replace(
            base.scenario,
            num_slots=2,
            num_users=4,
        ),
    )
    output = tmp_path / "exact"

    run_exact_definition(definition, config_path, output)

    with (output / "exact_slots.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["complete_search"] == "True" for row in rows)
    assert all(
        int(row["full_completed"]) <= int(row["exact_completed"])
        for row in rows
    )
    assert all(int(row["evaluated_schedules"]) > 0 for row in rows)
    assert all(row["cardinality_optimal"] in {"True", "False"} for row in rows)
    assert all(row["lexicographic_optimal"] in {"True", "False"} for row in rows)
    for row in rows:
        if row["cardinality_optimal"] == "True":
            assert row["cpu_gap_hz"] != ""
        else:
            assert row["cpu_gap_hz"] == ""
        if row["cpu_layer_optimal"] == "True":
            assert row["interference_gap_w"] != ""
        else:
            assert row["interference_gap_w"] == ""

    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert 0.0 <= summary["optimal_slot_fraction"] <= 1.0
    assert summary["optimal_slot_fraction"] == summary[
        "lexicographic_optimal_slot_fraction"
    ]
    assert 0.0 <= summary["cardinality_optimal_slot_fraction"] <= 1.0
    assert 0.0 <= summary["nontrivial_slot_fraction"] <= 1.0
    assert summary["mean_completion_gap"] >= 0.0
    assert summary["optimality_metric"] == "quantized lexicographic Phi"
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["resolved_definition"]["methods"] == ["full", "exact"]
