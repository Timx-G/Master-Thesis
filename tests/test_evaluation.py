import dataclasses
import inspect

import numpy as np
import pytest

from uav_mec.closure import close_assignment, freeze_plan, rate_raw_assignment
from uav_mec.domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import (
    EvaluationConfig,
    InfeasibleScheduleError,
    build_outcome_from_rates,
    compute_rate_records,
    evaluate_frozen_plan,
    evaluate_schedule,
    semantic_fingerprint,
)
from uav_mec.externality import aggregate_victim_link_exposure
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy


def config(
    gamma: float = 1.0,
    *,
    n_subcarriers: int = 1,
    cpu_per_uav_hz: float | tuple[float, ...] = 4e9,
) -> EvaluationConfig:
    return EvaluationConfig(
        n_subcarriers=n_subcarriers,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=gamma,
        cpu_per_uav_hz=cpu_per_uav_hz,
    )


def _basis_case() -> tuple[SlotSnapshot, Schedule, NOMAChannel, EvaluationConfig]:
    tasks = (
        Task(
            1,
            0,
            0,
            1e6,
            1.4e9,
            0.8,
            1e9,
            slot_duration_s=1.0,
            result_ratio=0.0,
        ),
        Task(
            2,
            1,
            0,
            1e6,
            1.4e9,
            0.8,
            1e9,
            slot_duration_s=1.0,
            result_ratio=0.0,
        ),
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[1e-9], [8e-9]]),
        1,
        estimated_channel_gains=np.array([[9e-9], [2e-9]]),
    )
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "strong"),
            ScheduleEntry(2, 1, 0, 0, "weak"),
        )
    )
    evaluation = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.01,
        cpu_per_uav_hz=4e9,
    )
    return snapshot, schedule, NOMAChannel(ChannelConfig(), 100.0), evaluation


def test_semantic_cache_key_separates_basis_policy_gain_and_task_rho() -> None:
    snapshot, _schedule, channel, evaluation = _basis_case()
    planning = semantic_fingerprint(
        snapshot,
        channel,
        evaluation,
        ChannelBasis.PLANNING,
    )
    realized = semantic_fingerprint(
        snapshot,
        channel,
        evaluation,
        ChannelBasis.REALIZED,
    )
    changed_policy = dataclasses.replace(
        evaluation,
        allocation_policy=AllocationPolicy.CROSSING_EQUAL,
    )
    policy_key = semantic_fingerprint(
        snapshot,
        channel,
        changed_policy,
        ChannelBasis.PLANNING,
    )
    changed_task = dataclasses.replace(snapshot.tasks[0], result_ratio=0.2)
    rho_snapshot = SlotSnapshot(
        snapshot.slot_index,
        (changed_task, snapshot.tasks[1]),
        snapshot.channel_gains,
        snapshot.num_uavs,
        snapshot.estimated_channel_gains,
    )
    rho_key = semantic_fingerprint(
        rho_snapshot,
        channel,
        evaluation,
        ChannelBasis.PLANNING,
        downlink_rates_bps={1: 500_000.0, 2: 1_000_000.0},
    )
    downlink_a = semantic_fingerprint(
        rho_snapshot,
        channel,
        evaluation,
        ChannelBasis.PLANNING,
        downlink_rates_bps={1: 1_000_000.0, 2: 1_000_000.0},
    )
    downlink_b = semantic_fingerprint(
        rho_snapshot,
        channel,
        evaluation,
        ChannelBasis.PLANNING,
        downlink_rates_bps={1: 2_000_000.0, 2: 1_000_000.0},
    )
    assert len({planning, realized, policy_key, rho_key, downlink_a, downlink_b}) == 6


def test_semantic_cache_key_is_stable_across_fresh_equivalent_objects() -> None:
    snapshot, _schedule, channel, evaluation = _basis_case()
    fresh_snapshot = SlotSnapshot(
        slot_index=snapshot.slot_index,
        tasks=tuple(dataclasses.replace(task) for task in snapshot.tasks),
        channel_gains=np.array(snapshot.channel_gains, copy=True),
        num_uavs=snapshot.num_uavs,
        estimated_channel_gains=np.array(
            snapshot.estimated_channel_gains,
            copy=True,
        ),
    )
    fresh_channel = NOMAChannel(dataclasses.replace(channel.cfg), channel.H)
    fresh_evaluation = dataclasses.replace(evaluation)

    expected = semantic_fingerprint(
        snapshot,
        channel,
        evaluation,
        ChannelBasis.PLANNING,
        downlink_rates_bps={2: 1_000_000.0, 1: 500_000.0},
    )
    actual = semantic_fingerprint(
        fresh_snapshot,
        fresh_channel,
        fresh_evaluation,
        ChannelBasis.PLANNING,
        downlink_rates_bps={1: 500_000.0, 2: 1_000_000.0},
    )

    assert actual == expected


def test_realized_replay_never_reallocates_or_recloses() -> None:
    snapshot, schedule, channel, evaluation = _basis_case()
    rated = rate_raw_assignment(
        snapshot,
        schedule,
        channel,
        evaluation,
        channel_basis=ChannelBasis.PLANNING,
    )
    plan = freeze_plan(close_assignment(snapshot, rated, channel, evaluation))
    planned_decisions = tuple(
        (decision.task_id, decision.alpha, decision.cpu_hz)
        for decision in plan.allocation.decisions
    )
    replay_source = inspect.getsource(evaluate_frozen_plan)
    assert "allocate_with_policy" not in replay_source
    assert "close_assignment" not in replay_source
    realized = evaluate_frozen_plan(snapshot, plan, channel, evaluation)
    assert realized.plan_key == plan.stable_key()
    assert realized.schedule == plan.schedule
    assert tuple(
        (decision.task_id, decision.alpha, decision.cpu_hz)
        for decision in realized.decisions
    ) == planned_decisions
    assert {record.channel_basis for record in realized.rates} == {
        ChannelBasis.REALIZED
    }


def test_evaluation_config_has_no_global_result_ratio() -> None:
    cfg = config()

    assert not hasattr(cfg, "result_ratio")


def test_evaluation_uses_independent_downlink_rate_for_positive_task_rho() -> None:
    task = Task(
        99,
        0,
        0,
        1e6,
        1.2e9,
        0.8,
        1e9,
        result_ratio=0.5,
    )
    snapshot = SlotSnapshot(0, (task,), np.array([[1e-7]]), 1)
    schedule = Schedule((ScheduleEntry(99, 0, 0, 0, "solo"),))
    channel = NOMAChannel(ChannelConfig(), 100.0)

    fast = evaluate_schedule(
        snapshot,
        schedule,
        channel,
        config(),
        downlink_rates_bps={99: 10e6},
    )
    slow = evaluate_schedule(
        snapshot,
        schedule,
        channel,
        config(),
        downlink_rates_bps={99: 1e6},
    )

    assert fast.decisions[0].cpu_hz < slow.decisions[0].cpu_hz


def test_cross_uav_user_uses_schedule_serving_channel_and_cpu_pool():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[1e-10, 1e-7]]), 2)
    schedule = Schedule((ScheduleEntry(1, 0, 1, 0, "solo"),))

    outcome = evaluate_schedule(
        snapshot,
        schedule,
        NOMAChannel(ChannelConfig(), 100.0),
        config(),
    )

    rate = outcome.rates[0]
    decision = outcome.decisions[0]
    assert rate.task_id == 1
    assert rate.serving_uav == 1
    assert rate.rate_bps > 1e6
    assert decision.completed
    assert decision.cpu_hz <= config().cpu_per_uav_hz


def test_realized_basis_preserves_schedule_order_and_records_sic_layer() -> None:
    tasks = (
        Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.4e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(
        0, tasks, np.array([[1e-9], [8e-9]]), 1,
        estimated_channel_gains=np.array([[9e-9], [2e-9]]),
    )
    schedule = Schedule((
        ScheduleEntry(1, 0, 0, 0, "strong"),
        ScheduleEntry(2, 1, 0, 0, "weak"),
    ))
    records = compute_rate_records(
        snapshot, schedule, NOMAChannel(ChannelConfig(), 100.0), config(),
        channel_basis=ChannelBasis.REALIZED,
    )
    assert [row.task_id for row in records] == [1, 2]
    assert all(row.channel_basis is ChannelBasis.REALIZED for row in records)
    assert {(layer.task_id, layer.decode_rank)
            for row in records for layer in row.sic_layers} == {(1, 0), (2, 1)}


def test_cross_interference_uses_interferer_gain_at_target_uav():
    tasks = (
        Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.4e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array(
            [
                [1e-7, 1e-12],
                [4e-8, 1e-7],
            ]
        ),
        2,
    )
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "solo"),
            ScheduleEntry(2, 1, 1, 0, "solo"),
        )
    )

    rates = compute_rate_records(
        snapshot,
        schedule,
        NOMAChannel(ChannelConfig(), 100.0),
        config(),
        channel_basis=ChannelBasis.PLANNING,
    )

    by_task = {record.task_id: record for record in rates}
    assert by_task[1].cross_interference_w == pytest.approx(0.2 * 4e-8)
    assert by_task[2].cross_interference_w == pytest.approx(0.2 * 1e-12)


def test_evaluator_exposes_canonical_victim_link_aggregate():
    tasks = (
        Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.4e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[1e-7, 1e-12], [4e-8, 1e-7]]),
        2,
    )
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "solo"),
            ScheduleEntry(2, 1, 1, 0, "solo"),
        )
    )
    evaluation = config()
    rates = compute_rate_records(
        snapshot,
        schedule,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
        channel_basis=ChannelBasis.PLANNING,
    )

    outcome = build_outcome_from_rates(
        snapshot,
        {1: 0, 2: 1},
        rates,
        evaluation,
    )

    expected = aggregate_victim_link_exposure(rates)
    assert outcome.aggregate_victim_link_exposure_w == pytest.approx(expected)
    assert outcome.total_cross_interference_w == pytest.approx(expected)


def test_positive_tiny_sinr_keeps_a_positive_rate():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[5e-33]]), 1)
    schedule = Schedule((ScheduleEntry(1, 0, 0, 0, "solo"),))

    (rate,) = compute_rate_records(
        snapshot,
        schedule,
        NOMAChannel(ChannelConfig(), 100.0),
        config(),
        channel_basis=ChannelBasis.PLANNING,
    )

    assert 0.0 < rate.rate_bps < 1e-6


def test_cpu_budgets_are_independent_on_actual_serving_uavs():
    tasks = (
        Task(1, 0, 0, 1e6, 1.5e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.5e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.full((2, 2), 1e-7), 2)
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "solo"),
            ScheduleEntry(2, 1, 1, 1, "solo"),
        )
    )
    evaluation = config(n_subcarriers=2, cpu_per_uav_hz=1e9)

    outcome = evaluate_schedule(
        snapshot,
        schedule,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
    )

    assert outcome.completed_task_ids == (1, 2)
    assert all(decision.cpu_hz <= evaluation.cpu_per_uav_hz for decision in outcome.decisions)
    assert sum(decision.cpu_hz for decision in outcome.decisions) > evaluation.cpu_per_uav_hz


def test_heterogeneous_cpu_vector_is_applied_to_each_actual_serving_uav():
    tasks = (
        Task(1, 0, 0, 1e6, 1.5e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.5e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.full((2, 2), 1e-7), 2)
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "solo"),
            ScheduleEntry(2, 1, 1, 1, "solo"),
        )
    )
    evaluation = config(
        n_subcarriers=2,
        cpu_per_uav_hz=(0.5e9, 3e9),
    )

    outcome = evaluate_schedule(
        snapshot,
        schedule,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
    )

    by_task = {decision.task_id: decision for decision in outcome.decisions}
    assert not by_task[1].completed
    assert by_task[2].completed
    assert by_task[2].cpu_hz <= evaluation.cpu_budget_for(1, 2)


def test_cpu_vector_length_must_match_snapshot_uav_count():
    evaluation = config(cpu_per_uav_hz=(2e9, 4e9, 6e9))

    with pytest.raises(ValueError, match="CPU budget vector length"):
        evaluation.cpu_budget_for(0, 2)


def test_pair_below_hard_sic_threshold_is_rejected():
    tasks = (
        Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.4e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[1e-8], [1e-8]]), 1)
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "strong"),
            ScheduleEntry(2, 1, 0, 0, "weak"),
        )
    )

    with pytest.raises(InfeasibleScheduleError, match="SIC"):
        evaluate_schedule(
            snapshot,
            schedule,
            NOMAChannel(ChannelConfig(), 100.0),
            config(),
        )


def test_every_arrived_task_has_exactly_one_outcome():
    tasks = (
        Task(1, 0, 0, 1e6, 0.4e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.8e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[1e-8], [1e-8]]), 1)

    outcome = evaluate_schedule(
        snapshot,
        Schedule(),
        NOMAChannel(ChannelConfig(), 100.0),
        config(),
    )

    assert outcome.arrived_task_ids == (1, 2)
    assert outcome.completed_task_ids == (1,)
    assert outcome.missed_task_ids == (2,)
    assert {decision.task_id for decision in outcome.decisions} == {1, 2}


def test_invalid_schedule_is_reported_as_infeasible_evaluation():
    local_task = Task(1, 0, 0, 1e6, 0.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (local_task,), np.array([[1e-8]]), 1)
    schedule = Schedule((ScheduleEntry(1, 0, 0, 0, "solo"),))

    with pytest.raises(InfeasibleScheduleError, match="local-feasible"):
        evaluate_schedule(
            snapshot,
            schedule,
            NOMAChannel(ChannelConfig(), 100.0),
            config(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sic_residual", 1.1),
        ("sic_residual", float("nan")),
        ("gamma_sic_min", 0.0),
    ),
)
def test_evaluation_config_rejects_nonphysical_values(field, value):
    values = {
        "n_subcarriers": 1,
        "bandwidth_hz": 5e6,
        "transmit_power_w": 0.2,
        "sic_residual": 0.02,
        "gamma_sic_min": 1.0,
        "cpu_per_uav_hz": 4e9,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        EvaluationConfig(**values)
