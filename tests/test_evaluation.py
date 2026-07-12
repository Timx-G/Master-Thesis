import numpy as np
import pytest

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import (
    EvaluationConfig,
    InfeasibleScheduleError,
    compute_rate_records,
    evaluate_schedule,
)
from uav_mec.model import ChannelConfig, NOMAChannel


def config(
    gamma: float = 1.0,
    *,
    n_subcarriers: int = 1,
    cpu_per_uav_hz: float = 4e9,
) -> EvaluationConfig:
    return EvaluationConfig(
        n_subcarriers=n_subcarriers,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=gamma,
        cpu_per_uav_hz=cpu_per_uav_hz,
        result_ratio=0.1,
    )


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
    )

    by_task = {record.task_id: record for record in rates}
    assert by_task[1].cross_interference_w == pytest.approx(0.2 * 4e-8)
    assert by_task[2].cross_interference_w == pytest.approx(0.2 * 1e-12)


def test_positive_tiny_sinr_keeps_a_positive_rate():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[5e-33]]), 1)
    schedule = Schedule((ScheduleEntry(1, 0, 0, 0, "solo"),))

    (rate,) = compute_rate_records(
        snapshot,
        schedule,
        NOMAChannel(ChannelConfig(), 100.0),
        config(),
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
        ("result_ratio", -0.1),
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
        "result_ratio": 0.1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        EvaluationConfig(**values)
