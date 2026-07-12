import copy
from pathlib import Path

import pytest

from uav_mec.analyze_results import (
    Bundle,
    comparison_report,
    validate_bundle,
    verify_checksums,
)
from uav_mec.artifacts import RunArtifactWriter


def bundle() -> Bundle:
    methods = ("full", "intra_global")
    tasks = []
    slots = []
    values = {
        "full": (1.0, 1.0, 1.0, 1.0),
        "intra_global": (0.5, 0.5, 0.5, 0.5),
    }
    for seed in range(4):
        for method in methods:
            completed_count = 0
            for task_id in (1, 2):
                completed = method == "full" or task_id == 1
                completed_count += completed
                tasks.append(
                    {
                        "method": method,
                        "seed": str(seed),
                        "slot": "0",
                        "task_id": str(task_id),
                        "user_id": str(task_id),
                        "arrival_slot": "0",
                        "data_bits": "1000000",
                        "comp_cycles": "1200000000",
                        "deadline_s": "0.8",
                        "local_cpu_hz": "1000000000",
                        "completed": str(completed),
                        "delay_s": "0.7" if completed else "",
                        "alpha": "0.4" if completed else "0",
                        "cpu_hz": "1000000000" if completed else "0",
                        "rate_bps": "1000000" if completed else "",
                        "sinr": "1" if completed else "",
                        "serving_uav": "0" if completed else "",
                        "subcarrier": "0" if completed else "",
                        "sic_feasible": "True" if completed else "",
                        "cross_interference_w": "0",
                        "reason": "complete" if completed else "unscheduled",
                    }
                )
            slots.append(
                {
                    "method": method,
                    "seed": str(seed),
                    "slot": "0",
                    "arrivals": "2",
                    "completed": str(completed_count),
                    "slack_s": "0.2",
                    "cross_interference_w": "0",
                    "runtime_s": "0.01",
                    "sic_violations": "0",
                }
            )
    summary = {
        "seeds": list(range(4)),
        "initializer_status_counts": {"optimal": 8},
        "search_termination_counts": {"local_optimum": 8},
        "methods": {
            name: {"on_time_rate_by_seed": list(values[name])}
            for name in methods
        },
    }
    manifest = {
        "resolved_definition": {
            "evaluation": {"cpu_per_uav_hz": 4e9},
            "scenario": {"num_uavs": 1, "num_slots": 1},
        }
    }
    return Bundle("pairing", None, manifest, summary, tasks, slots)


def test_duplicate_task_result_is_a_hard_validation_failure():
    value = bundle()
    duplicated = copy.deepcopy(value.tasks)
    duplicated.append(copy.deepcopy(duplicated[0]))

    with pytest.raises(ValueError, match="duplicate task outcome"):
        validate_bundle(
            Bundle(
                value.name,
                value.path,
                value.manifest,
                value.summary,
                duplicated,
                value.slots,
            )
        )


def test_cpu_budget_violation_is_a_hard_validation_failure():
    value = bundle()
    tasks = copy.deepcopy(value.tasks)
    tasks[0]["cpu_hz"] = "5000000000"

    with pytest.raises(ValueError, match="CPU budget exceeded"):
        validate_bundle(
            Bundle(
                value.name,
                value.path,
                value.manifest,
                value.summary,
                tasks,
                value.slots,
            )
        )


def test_positive_paired_effect_has_ci_and_holm_fields():
    report = comparison_report({"pairing": bundle().summary})
    result = report["innovation1_cross"]

    assert result["mean_difference_pp"] == pytest.approx(50.0)
    assert result["ci_low_pp"] > 0.0
    assert 0.0 <= result["p_holm"] <= 1.0


def test_checksum_verification_rejects_unlisted_files(tmp_path: Path):
    path = tmp_path / "bundle"
    writer = RunArtifactWriter(path)
    writer.write_json("summary.json", {})
    writer.finalize({"resolved_definition": {}})
    (path / "rogue.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unchecksummed"):
        verify_checksums(path)


def test_checksum_verification_rejects_tampering(tmp_path: Path):
    path = tmp_path / "bundle"
    writer = RunArtifactWriter(path)
    writer.write_json("summary.json", {"value": 1})
    writer.finalize({"resolved_definition": {}})
    (path / "summary.json").write_text('{"value": 2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums(path)
