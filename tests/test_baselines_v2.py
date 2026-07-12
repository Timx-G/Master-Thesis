import numpy as np
import pytest

from uav_mec.baselines_v2 import greedy_schedule, local_only_schedule
from uav_mec.domain import SlotSnapshot, Task
from uav_mec.schedule import validate_schedule


def test_local_only_never_schedules_radio_users():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[1e-8]]), 1)

    assert local_only_schedule(snapshot).entries == ()


def test_greedy_schedule_is_capacity_feasible_and_deterministic():
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(6))
    gains = np.array(
        [
            [9e-8, 1e-8],
            [7e-8, 2e-8],
            [5e-8, 3e-8],
            [3e-8, 5e-8],
            [2e-8, 7e-8],
            [1e-8, 9e-8],
        ]
    )
    snapshot = SlotSnapshot(0, tasks, gains, 2)

    schedules = [greedy_schedule(snapshot, 2) for _ in range(3)]

    assert len({schedule.stable_key() for schedule in schedules}) == 1
    validate_schedule(schedules[0], snapshot, 2)
    assert all(
        entry.serving_uav == int(np.argmax(gains[entry.task_id]))
        for entry in schedules[0].entries
    )


def test_greedy_rejects_nonpositive_subcarrier_count():
    snapshot = SlotSnapshot(0, (), np.empty((0, 1)), 1)

    with pytest.raises(ValueError, match="n_subcarriers"):
        greedy_schedule(snapshot, 0)
