from types import SimpleNamespace

import numpy as np
import pytest

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, evaluate_schedule
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.scheduler import (
    InitializationError,
    build_initial_schedule,
    repair_sic,
)


def snapshot() -> SlotSnapshot:
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(6))
    gains = np.array(
        [
            [8e-8, 1e-8],
            [6e-8, 2e-8],
            [4e-8, 3e-8],
            [3e-8, 5e-8],
            [2e-8, 7e-8],
            [1e-8, 9e-8],
        ]
    )
    return SlotSnapshot(0, tasks, gains, 2)


def eval_cfg() -> EvaluationConfig:
    return EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)


def test_initializer_returns_a_fully_feasible_schedule():
    channel = NOMAChannel(ChannelConfig(), 100.0)

    result = build_initial_schedule(snapshot(), channel, eval_cfg())
    outcome = evaluate_schedule(snapshot(), result.schedule, channel, eval_cfg())

    assert result.solver_status == "optimal"
    assert all(record.sic_feasible for record in outcome.rates)
    assert len({entry.task_id for entry in result.schedule.entries}) == len(
        result.schedule.entries
    )
    assert len(
        {(entry.serving_uav, entry.subcarrier) for entry in result.schedule.entries}
    ) <= snapshot().num_uavs * eval_cfg().n_subcarriers


def test_initializer_never_silently_falls_back_when_milp_fails(monkeypatch):
    monkeypatch.setattr(
        "uav_mec.scheduler.milp",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            message="forced failure",
        ),
    )

    with pytest.raises(InitializationError, match="forced failure"):
        build_initial_schedule(
            snapshot(),
            NOMAChannel(ChannelConfig(), 100.0),
            eval_cfg(),
        )


def test_initializer_does_not_schedule_zero_benefit_candidates():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    target = SlotSnapshot(0, (task,), np.zeros((1, 1)), 1)

    result = build_initial_schedule(
        target,
        NOMAChannel(ChannelConfig(), 100.0),
        eval_cfg(),
    )

    assert result.schedule.entries == ()


def test_sic_repair_keeps_the_strong_user_as_solo():
    tasks = (
        Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.4e9, 0.8, 1e9),
    )
    target = SlotSnapshot(0, tasks, np.array([[2e-8], [1e-8]]), 1)
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "strong"),
            ScheduleEntry(2, 1, 0, 0, "weak"),
        )
    )
    strict = EvaluationConfig(1, 5e6, 0.2, 0.02, 3.0, 4e9, 0.1)

    repaired = repair_sic(
        schedule,
        target,
        NOMAChannel(ChannelConfig(), 100.0),
        strict,
    )

    assert repaired.entries == (ScheduleEntry(1, 0, 0, 0, "solo"),)


def test_initializer_is_deterministic_across_repeated_solves():
    channel = NOMAChannel(ChannelConfig(), 100.0)

    keys = {
        build_initial_schedule(snapshot(), channel, eval_cfg()).schedule.stable_key()
        for _ in range(5)
    }

    assert len(keys) == 1
