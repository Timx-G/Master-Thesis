import json
from dataclasses import replace
from pathlib import Path

import pytest

from uav_mec.analyze_results import verify_checksums
from uav_mec.capacity_analysis import (
    capacity_case_metrics,
    write_capacity_analysis,
)
from uav_mec.experiment import run_definition
from uav_mec.run_thesis_suite import CAPACITY_BASE_PATH, build_suite


def test_capacity_analysis_reports_realized_offered_load(tmp_path: Path):
    case = {item.name: item for item in build_suite("capacity")}["load_k3_n45"]
    definition = replace(
        case.definition,
        seeds=(4000, 4001),
        scenario=replace(case.definition.scenario, num_slots=2),
    )
    run_path = tmp_path / "case"
    run_definition(definition, CAPACITY_BASE_PATH, run_path)

    metrics = capacity_case_metrics(definition, run_path)

    assert metrics["scenario"]["registered_users_per_uav"] == 15.0
    assert metrics["scenario"]["burst_rate"] == 0.9
    assert metrics["scenario"]["cool_rate"] == 0.2
    assert metrics["load"]["noma_positions_per_uav_slot"] == 6
    assert metrics["load"]["noma_positions_per_slot"] == 18
    assert metrics["scenario"]["arrival_mode"] == "periodic"
    assert metrics["load"]["expected_arrival_probability_per_user"] == pytest.approx(
        sum(1.0 / period for period in (2, 2, 3)) / 3
    )
    assert 0.0 <= metrics["load"]["realized_arrival_probability_per_user"] <= 1.0
    assert metrics["load"]["mean_arrivals_per_uav_slot"] > 0.0
    assert 0.9 < metrics["load"]["local_infeasible_task_fraction"] <= 1.0
    assert metrics["load"]["mean_radio_offered_load"] > 0.0
    assert 0.0 <= metrics["load"]["overloaded_slot_fraction"] <= 1.0
    assert set(metrics["methods"]) == {
        "intra_global",
        "cross_bilateral",
        "full",
    }
    assert metrics["methods"]["full"]["runtime_p95_s"] >= 0.0
    assert metrics["methods"]["full"]["sic_violations"] == 0


def test_capacity_analysis_writes_checksummed_bundle(tmp_path: Path):
    case = build_suite("capacity")[0]
    definition = replace(
        case.definition,
        seeds=(4000,),
        scenario=replace(case.definition.scenario, num_slots=1),
    )
    run_path = tmp_path / "case"
    run_definition(definition, CAPACITY_BASE_PATH, run_path)

    output_path = tmp_path / "analysis"
    write_capacity_analysis(
        {case.name: (definition, run_path)},
        output_path,
    )

    verify_checksums(output_path)
    report = json.loads((output_path / "capacity_metrics.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert set(report["cases"]) == {case.name}
