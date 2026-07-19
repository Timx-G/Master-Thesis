import numpy as np
import pytest

from uav_mec.baselines_v2 import greedy_schedule
from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, evaluate_schedule
from uav_mec.exact import exact_schedule
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.scheduler import network_objective


def test_complete_enumeration_is_never_worse_than_greedy():
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(5))
    gains = np.array([[9e-8], [7e-8], [5e-8], [3e-8], [1e-8]])
    snapshot = SlotSnapshot(0, tasks, gains, 1)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9)

    greedy = evaluate_schedule(
        snapshot,
        greedy_schedule(snapshot, 2),
        channel,
        cfg,
    )
    exact = exact_schedule(snapshot, channel, cfg, max_heavy_tasks=6)

    assert network_objective(exact.outcome) >= network_objective(greedy)
    assert exact.complete_search


def test_exact_search_enumerates_non_nearest_uav_associations():
    tasks = tuple(Task(i, i, 0, 0.2e6, 1.2e9, 0.8, 1e9) for i in range(4))
    gains = np.array(
        [
            [1e-6, 1e-12],
            [2e-7, 1e-12],
            [9e-8, 8e-8],
            [9e-9, 8e-9],
        ]
    )
    snapshot = SlotSnapshot(0, tasks, gains, 2)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 8e9)

    result = exact_schedule(snapshot, channel, cfg, max_heavy_tasks=4)

    serving = {
        entry.task_id: entry.serving_uav for entry in result.schedule.entries
    }
    assert result.outcome.completed_count == 4
    assert any(
        serving[task.task_id] != int(np.argmax(gains[row]))
        for row, task in enumerate(tasks)
    )
    assert len(serving) == len(result.schedule.entries)
    verified = evaluate_schedule(snapshot, result.schedule, channel, cfg)
    assert all(record.sic_feasible for record in verified.rates)


def test_exact_evaluation_count_has_no_duplicate_empty_schedule():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[1e-8]]), 1)
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9)

    result = exact_schedule(
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        cfg,
        max_heavy_tasks=1,
    )

    assert result.evaluated_schedules == 2


def test_exact_search_rejects_instances_above_declared_limit():
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(2))
    snapshot = SlotSnapshot(0, tasks, np.ones((2, 1)), 1)
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9)

    with pytest.raises(ValueError, match="limited to 1"):
        exact_schedule(
            snapshot,
            NOMAChannel(ChannelConfig(), 100.0),
            cfg,
            max_heavy_tasks=1,
        )
