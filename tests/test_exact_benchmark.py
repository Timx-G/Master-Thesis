import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.exact import (
    ExactScheduleCertificate,
    ExactScheduleResult,
    exact_schedule,
)
from uav_mec.experiment import (
    load_definition,
    method_spec_sha256,
    method_spec_to_dict,
    thesis_method_specs,
)
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


def test_state_cap_at_iterator_exhaustion_is_complete() -> None:
    base_snapshot, channel, cfg = _tiny_exact_case()
    snapshot = SlotSnapshot(
        0,
        base_snapshot.tasks[:1],
        base_snapshot.planning_channel_gains[:1],
        1,
    )

    result = exact_schedule(snapshot, channel, cfg, state_cap=2)

    assert result.certificate.raw_states_enumerated == 2
    assert result.certificate.complete
    assert result.certificate.gap_status == "exact"


def test_timeout_returns_a_bounded_incomplete_certificate(monkeypatch) -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    monotonic_values = iter((0.0, 1.0))
    monkeypatch.setattr(
        "uav_mec.exact.time.monotonic",
        lambda: next(monotonic_values),
    )

    result = exact_schedule(snapshot, channel, cfg, timeout_s=0.5)

    assert result.certificate.raw_states_enumerated == 1
    assert not result.certificate.complete
    assert result.certificate.gap_status == "bounded"
    assert not result.certificate.schedule_nontrivial


def test_exact_schedule_propagates_independent_downlink_rates() -> None:
    base_snapshot, channel, cfg = _tiny_exact_case()
    task = replace(base_snapshot.tasks[0], result_ratio=0.25)
    snapshot = SlotSnapshot(
        0,
        (task,),
        base_snapshot.planning_channel_gains[:1],
        1,
    )

    result = exact_schedule(
        snapshot,
        channel,
        replace(cfg, cpu_per_uav_hz=8_000_000_000.0),
        downlink_rates_bps={task.task_id: 5_000_000.0},
    )

    assert result.certificate.complete
    assert result.plan.schedule.task_ids == (task.task_id,)


def test_exact_schedule_counts_only_unique_user_raw_assignments() -> None:
    base_snapshot, channel, cfg = _tiny_exact_case()
    tasks = (
        replace(base_snapshot.tasks[0], user_id=7, job_index=0),
        replace(base_snapshot.tasks[1], user_id=7, job_index=1),
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        base_snapshot.planning_channel_gains[:2],
        1,
    )

    result = exact_schedule(
        snapshot,
        channel,
        replace(cfg, cpu_per_uav_hz=8_000_000_000.0),
    )

    assert result.certificate.complete
    assert result.certificate.raw_states_enumerated == 3


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("best_bound", (1.5, 0.0, 0.0)),
        ("lexicographic_gap", (0.5, 0.0, 0.0)),
    ),
)
def test_exact_certificate_requires_integer_completion_components(
    field: str,
    value: tuple[float, float, float],
) -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    certificate = exact_schedule(
        snapshot,
        channel,
        cfg,
        state_cap=1,
    ).certificate
    assert isinstance(certificate, ExactScheduleCertificate)

    with pytest.raises(ValueError, match="completion component"):
        replace(certificate, **{field: value})


def test_bounded_exact_certificate_cannot_be_below_incumbent() -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    result = exact_schedule(snapshot, channel, cfg, state_cap=1)
    invalid = replace(
        result.certificate,
        best_bound=(-1, 0.0, 0.0),
        lexicographic_gap=(-1, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="bound.*objective"):
        replace(result, certificate=invalid)


def test_bounded_exact_certificate_gap_matches_bound_minus_objective() -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    result = exact_schedule(snapshot, channel, cfg, state_cap=1)
    invalid = replace(
        result.certificate,
        lexicographic_gap=(0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="gap.*bound.*objective"):
        replace(result, certificate=invalid)


def test_exact_benchmark_rejects_an_incomplete_certificate(
    tmp_path: Path,
    development_seed_access,
    monkeypatch,
) -> None:
    config_path = Path("configs/thesis/confirmatory_exact.toml")
    base = load_definition(config_path)
    definition = replace(
        base,
        seeds=(42,),
        scenario=replace(
            base.scenario,
            num_slots=1,
            num_users=1,
            random_burst=False,
            task=replace(
                base.scenario.task,
                arrival_prob=1.0,
                data_min=2_000_000.0,
                data_max=2_000_000.0,
                comp_min=1_000.0,
                comp_max=1_000.0,
            ),
        ),
    )

    def capped_exact(snapshot, channel, cfg, max_heavy_tasks):
        return exact_schedule(
            snapshot,
            channel,
            cfg,
            max_heavy_tasks=max_heavy_tasks,
            state_cap=1,
        )

    monkeypatch.setattr(
        "uav_mec.run_exact_benchmark.exact_schedule",
        capped_exact,
    )

    with pytest.raises(RuntimeError, match="incomplete exact certificate"):
        run_exact_definition(
            definition,
            config_path,
            tmp_path / "incomplete-exact",
            seed_access=development_seed_access,
        )


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
    specs = thesis_method_specs()
    for method_id in ("full", "exact"):
        serialized = method_spec_to_dict(specs[method_id])
        factor_hash = method_spec_sha256(specs[method_id])
        assert summary["methods"][method_id] == {
            "method_spec": serialized,
            "method_spec_sha256": factor_hash,
        }
        assert manifest["method_specs"][method_id] == {
            "factor_hash": factor_hash,
            "spec": serialized,
        }


def test_exact_benchmark_rejects_allocation_factor_drift_before_seed_access(
    tmp_path: Path,
) -> None:
    config_path = Path("configs/thesis/confirmatory_exact.toml")
    base = load_definition(config_path)
    incompatible = replace(
        base,
        evaluation=replace(
            base.evaluation,
            allocation_policy=AllocationPolicy.EDF_MIN_CPU,
        ),
    )

    with pytest.raises(ValueError, match="allocation_policy"):
        run_exact_definition(
            incompatible,
            config_path,
            tmp_path / "incompatible-exact",
            seed_access=object(),  # type: ignore[arg-type]
        )


def test_exact_benchmark_rejects_complete_spec_hash_drift_before_seed_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = Path("configs/thesis/confirmatory_exact.toml")
    definition = load_definition(config_path)
    specs = thesis_method_specs()
    specs["exact"] = replace(specs["exact"], display_name="drifted exact")
    monkeypatch.setattr(
        "uav_mec.run_exact_benchmark.thesis_method_specs",
        lambda: specs,
    )

    with pytest.raises(ValueError, match="factor hash"):
        run_exact_definition(
            definition,
            config_path,
            tmp_path / "drifted-exact",
            seed_access=object(),  # type: ignore[arg-type]
        )
