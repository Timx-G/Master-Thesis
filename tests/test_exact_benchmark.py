import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.exact import ExactScheduleResult, exact_schedule
from uav_mec.experiment import load_definition
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy, ExactAdmissionResult
from uav_mec.run_exact_benchmark import run_exact_definition


def _tiny_exact_case() -> tuple[SlotSnapshot, NOMAChannel, EvaluationConfig]:
    tasks = tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=700_000.0,
            comp_cycles=1_500_000_000.0,
            deadline_s=0.8,
            local_cpu_hz=400_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in (1, 2, 3)
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[9e-8], [5e-8], [2e-8]]), 1)
    cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.01,
        cpu_per_uav_hz=2_000_000_000.0,
        allocation_policy=AllocationPolicy.DEADLINE,
    )
    return snapshot, NOMAChannel(ChannelConfig(), 100.0), cfg


def test_exact_schedule_has_a_distinct_complete_certificate() -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    result = exact_schedule(snapshot, channel, cfg)
    assert isinstance(result, ExactScheduleResult)
    assert not isinstance(result, ExactAdmissionResult)
    assert result.certificate.complete
    assert result.certificate.raw_states_enumerated > result.certificate.unique_closed_states
    assert result.certificate.best_bound == result.objective
    assert result.certificate.lexicographic_gap == (0, 0.0, 0.0)
    assert set(result.plan.schedule.task_ids) == set(result.plan.allocation.completed_task_ids)
    assert result.certificate.cpu_nontrivial
    assert result.certificate.schedule_nontrivial


def test_capped_exact_schedule_has_a_bounded_incomplete_certificate() -> None:
    snapshot, channel, cfg = _tiny_exact_case()

    result = exact_schedule(snapshot, channel, cfg, state_cap=1)

    assert not result.certificate.complete
    assert result.certificate.raw_states_enumerated == 1
    assert result.certificate.gap_status == "bounded"
    assert all(component is not None for component in result.certificate.best_bound)
    assert all(component is not None for component in result.certificate.lexicographic_gap)


def test_incomplete_exact_schedule_does_not_claim_schedule_nontriviality() -> None:
    base_snapshot, channel, cfg = _tiny_exact_case()
    snapshot = SlotSnapshot(
        0,
        base_snapshot.tasks[:2],
        base_snapshot.planning_channel_gains[:2],
        1,
    )
    cfg = replace(cfg, cpu_per_uav_hz=8_000_000_000.0)

    capped = exact_schedule(snapshot, channel, cfg, state_cap=2)
    exhaustive = exact_schedule(snapshot, channel, cfg)

    assert not capped.certificate.complete
    assert capped.objective[0] == 1
    assert exhaustive.objective[0] == 2
    assert not exhaustive.certificate.schedule_nontrivial
    assert not capped.certificate.schedule_nontrivial


def test_exact_benchmark_never_reports_heuristic_above_optimum(
    tmp_path: Path,
    development_seed_access,
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

    run_exact_definition(
        definition,
        config_path,
        output,
        seed_access=development_seed_access,
    )

    with (output / "exact_slots.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["complete_search"] == "True" for row in rows)
    assert all(row["result_schema"] == "exact_schedule" for row in rows)
    assert all(row["certificate_label"] == "Exact" for row in rows)
    assert all(int(row["full_completed"]) <= int(row["exact_completed"]) for row in rows)
    assert all(int(row["evaluated_schedules"]) > 0 for row in rows)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert 0.0 <= summary["optimal_slot_fraction"] <= 1.0
    assert summary["mean_completion_gap"] >= 0.0
    assert summary["result_schema"] == "exact_schedule"
    assert summary["certificate_label"] == "Exact"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolved_definition"]["methods"] == ["full", "exact"]
