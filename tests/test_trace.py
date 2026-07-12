import numpy as np
import pytest

from uav_mec.domain import ExperimentTrace, SlotSnapshot
from uav_mec.model import ScenarioConfig
from uav_mec.trace import generate_trace


def test_trace_is_identical_for_the_same_seed():
    cfg = ScenarioConfig(
        num_users=6,
        num_slots=4,
        random_burst=True,
        user_mobility=True,
    )

    first = generate_trace(cfg, 123)
    second = generate_trace(cfg, 123)

    assert first.seed == second.seed
    assert [slot.tasks for slot in first.slots] == [slot.tasks for slot in second.slots]
    for left, right in zip(first.slots, second.slots, strict=True):
        np.testing.assert_array_equal(left.channel_gains, right.channel_gains)


def test_trace_assigns_each_arrival_one_stable_task_id():
    cfg = ScenarioConfig(num_users=8, num_slots=6, random_burst=True)

    trace = generate_trace(cfg, 456)

    tasks = [task for snapshot in trace.slots for task in snapshot.tasks]
    task_ids = [task.task_id for task in tasks]
    assert len(task_ids) == len(set(task_ids))
    for snapshot in trace.slots:
        assert all(task.arrival_slot == snapshot.slot_index for task in snapshot.tasks)
        assert all(
            task.task_id == snapshot.slot_index * cfg.num_users + task.user_id
            for task in snapshot.tasks
        )
        assert snapshot.channel_gains.shape == (len(snapshot.tasks), cfg.num_uavs)


def test_experiment_trace_freezes_a_supplied_slot_list():
    source = [SlotSnapshot(0, (), np.empty((0, 1)), 1)]

    trace = ExperimentTrace(1, source)
    source.clear()

    assert isinstance(trace.slots, tuple)
    assert len(trace.slots) == 1


def test_trace_rejects_unsupported_nonfixed_cluster_layout():
    cfg = ScenarioConfig(fixed_cluster_centers=False)

    with pytest.raises(ValueError, match="fixed_cluster_centers"):
        generate_trace(cfg, 1)
