import json
from dataclasses import replace
from pathlib import Path

import pytest

from uav_mec.analyze_results import verify_checksums
from uav_mec.domain import SlotSnapshot, Task
from uav_mec.experiment import run_definition
from uav_mec.externality_analysis import (
    association_overlap_ratios,
    bilateral_externality_metrics,
    qualify_overlap_fraction,
    write_externality_analysis,
)
from uav_mec.run_thesis_suite import EXTERNALITY_BASE_PATH, build_suite


def test_association_overlap_ratio_uses_second_best_over_best_gain():
    snapshot = SlotSnapshot(
        0,
        (
            Task(1, 0, 0, 1e6, 1e9, 0.8, 1e9),
            Task(2, 1, 0, 1e6, 1e9, 0.8, 1e9),
        ),
        [[8e-8, 4e-8, 1e-8], [2e-8, 2e-8, 1e-9]],
        3,
    )

    assert association_overlap_ratios(snapshot) == pytest.approx({1: 0.5, 2: 1.0})


@pytest.mark.parametrize(
    ("band", "fraction"),
    (("low", 0.15), ("medium", 0.30), ("medium", 0.45), ("high", 0.60)),
)
def test_overlap_fraction_qualification_accepts_frozen_band_boundaries(band, fraction):
    qualify_overlap_fraction(band, fraction)


@pytest.mark.parametrize(
    ("band", "fraction"),
    (("low", 0.16), ("medium", 0.29), ("medium", 0.46), ("high", 0.59)),
)
def test_overlap_fraction_qualification_rejects_out_of_band_values(band, fraction):
    with pytest.raises(ValueError, match="overlap"):
        qualify_overlap_fraction(band, fraction)


def test_bilateral_metrics_count_only_acceptances_rejected_by_global_objective():
    rows = [
        {
            "method": "cross_bilateral",
            "move_type": "cross",
            "accepted": True,
            "before": [3, 0.2, -1e-9],
            "after": [2, 0.3, -3e-9],
            "external_rate_affected_task_ids": [7, 8],
        },
        {
            "method": "cross_bilateral",
            "move_type": "position",
            "accepted": True,
            "before": [2, 0.2, -3e-9],
            "after": [3, 0.1, -2e-9],
            "external_rate_affected_task_ids": [9],
        },
        {
            "method": "cross_bilateral",
            "move_type": "replace",
            "accepted": False,
            "before": [3, 0.1, -2e-9],
            "after": [2, 0.1, -4e-9],
            "external_rate_affected_task_ids": [10],
        },
        {
            "method": "cross_bilateral",
            "move_type": "position",
            "accepted": True,
            "before": [3, 0.2, -2e-9],
            "after": [3, 0.1, -2e-9],
            "external_rate_affected_task_ids": [],
        },
        {
            "method": "full",
            "move_type": "cross",
            "accepted": True,
            "before": [2, 0.2, -3e-9],
            "after": [3, 0.1, -2e-9],
            "external_rate_affected_task_ids": [11],
        },
    ]

    metrics = bilateral_externality_metrics(rows)

    assert metrics["evaluated_candidates"] == 4
    assert metrics["accepted_candidates"] == 3
    assert metrics["accepted_global_rejected"] == 2
    assert metrics["accepted_global_rejected_rate"] == pytest.approx(2 / 3)
    assert metrics["global_rejections_with_external_rate_effect"] == 1
    assert metrics["completion_harming_acceptances"] == 1
    assert metrics["completion_tasks_lost"] == 1
    assert metrics["interference_increasing_acceptances"] == 1
    assert metrics["total_interference_increase_w"] == pytest.approx(2e-9)
    assert metrics["external_affected_task_event_count"] == 2


def test_externality_analysis_writes_paired_and_overlap_evidence(tmp_path: Path):
    case = build_suite("externality")[0]
    definition = replace(
        case.definition,
        seeds=(2000, 2001),
        scenario=replace(case.definition.scenario, num_slots=2),
    )
    run_path = tmp_path / "case"
    run_definition(definition, EXTERNALITY_BASE_PATH, run_path)

    analysis_path = tmp_path / "analysis"
    write_externality_analysis(
        {"tiny": (definition, run_path)},
        analysis_path,
    )

    verify_checksums(analysis_path)
    report = json.loads((analysis_path / "stress_metrics.json").read_text(encoding="utf-8"))
    assert report["overlap_region_definition"] == {
        "criterion": "best_two_large_scale_gain_gap_db <= threshold_db",
        "threshold_db": 3.0,
    }
    metrics = report["cases"]["tiny"]
    assert metrics["actual_arrivals_per_uav_slot"] > 0.0
    assert set(metrics["comparisons"]) == {
        "full_minus_cross_bilateral",
        "full_minus_intra_global",
    }
    assert set(metrics["overlap_comparisons"]) == {
        "full_minus_cross_bilateral",
        "full_minus_intra_global",
    }
    assert set(metrics["methods"]) == {
        "intra_global",
        "cross_bilateral",
        "full",
    }
    assert "bilateral_externality" in metrics

    event_rows = [json.loads(line) for line in (run_path / "events.jsonl").read_text().splitlines()]
    evaluated = [row for row in event_rows if isinstance(row.get("before"), list)]
    assert evaluated
    assert all("external_rate_affected_task_ids" in row for row in evaluated)
