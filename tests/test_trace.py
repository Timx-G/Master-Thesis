import numpy as np
import pytest

from uav_mec.domain import ExperimentTrace, SlotSnapshot
from uav_mec.model import ScenarioConfig, TaskConfig
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


def test_v1_fixed_deadline_trace_keeps_its_frozen_random_stream():
    cfg = ScenarioConfig(
        num_users=6,
        num_slots=2,
        random_burst=True,
        user_mobility=True,
        task=TaskConfig(t_max=0.8, t_min=None),
    )

    trace = generate_trace(cfg, 123)
    fingerprint = [
        (
            task.task_id,
            task.data_bits,
            task.comp_cycles,
            task.deadline_s,
        )
        for snapshot in trace.slots
        for task in snapshot.tasks
    ]

    assert fingerprint == [
        (1, 1678600.2057921325, 1274196071.6436934, 0.8),
        (3, 1196462.3495964701, 946835221.1374172, 0.8),
        (4, 423831.8668163716, 311756571.3681203, 0.8),
        (7, 1675269.6850935433, 1562374695.752683, 0.8),
        (9, 514378.89059743506, 407192208.482374, 0.8),
    ]


def test_v2_heterogeneous_deadlines_are_reproducible_and_bounded():
    cfg = ScenarioConfig(
        num_users=20,
        num_slots=3,
        random_burst=True,
        task=TaskConfig(t_min=0.6, t_max=1.0),
    )

    first = generate_trace(cfg, 2000)
    second = generate_trace(cfg, 2000)
    first_deadlines = tuple(task.deadline_s for snapshot in first.slots for task in snapshot.tasks)
    second_deadlines = tuple(
        task.deadline_s for snapshot in second.slots for task in snapshot.tasks
    )

    assert first_deadlines == second_deadlines
    assert first_deadlines
    assert all(0.6 <= deadline <= 1.0 for deadline in first_deadlines)
    assert len(set(first_deadlines)) > 1


def test_uav_overlap_layout_places_hotspots_on_two_uav_equal_gain_line():
    cfg = ScenarioConfig(
        num_users=3,
        num_slots=1,
        num_uavs=2,
        num_clusters=3,
        cluster_std=0.0,
        cluster_layout="uav_overlap",
        task=TaskConfig(arrival_prob=1.0),
    )

    snapshot = generate_trace(cfg, 3000).slots[0]

    assert len(snapshot.tasks) == 3
    np.testing.assert_allclose(
        snapshot.channel_gains[:, 0],
        snapshot.channel_gains[:, 1],
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize("invalid_t_min", [0.0, -1.0, 1.1, float("nan")])
def test_task_config_rejects_invalid_heterogeneous_deadline_floor(
    invalid_t_min: float,
):
    with pytest.raises(ValueError, match="t_min"):
        TaskConfig(t_min=invalid_t_min, t_max=1.0)


@pytest.mark.parametrize("invalid_t_max", [0.0, -1.0, float("nan"), float("inf")])
def test_task_config_rejects_invalid_heterogeneous_deadline_ceiling(
    invalid_t_max: float,
):
    with pytest.raises(ValueError, match="t_max"):
        TaskConfig(t_min=0.6, t_max=invalid_t_max)


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


def test_periodic_trace_freezes_user_profiles_and_job_identity():
    cfg = ScenarioConfig(
        num_users=6,
        num_slots=9,
        task=TaskConfig(
            arrival_mode="periodic",
            period_choices=(2, 2, 3),
            period_jitter_slots=0,
            t_min=0.6,
            t_max=1.0,
        ),
    )

    trace = generate_trace(cfg, 1000)
    tasks = tuple(task for snapshot in trace.slots for task in snapshot.tasks)
    by_user = {
        user: tuple(task for task in tasks if task.user_id == user)
        for user in range(cfg.num_users)
    }

    for user_tasks in by_user.values():
        assert user_tasks
        assert [task.job_index for task in user_tasks] == list(range(len(user_tasks)))
        assert len({task.data_bits for task in user_tasks}) == 1
        assert len({task.comp_cycles / task.data_bits for task in user_tasks}) == 1
        assert len({task.deadline_s for task in user_tasks}) == 1
        gaps = [
            right.arrival_slot - left.arrival_slot
            for left, right in zip(user_tasks, user_tasks[1:])
        ]
        assert len(set(gaps)) <= 1

    assert len({(task.user_id, task.job_index) for task in tasks}) == len(tasks)
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert any(
        task.task_id != task.arrival_slot * cfg.num_users + task.user_id
        for task in tasks
    )


def test_periodic_jitter_is_reproducible_and_keeps_releases_strictly_increasing():
    cfg = ScenarioConfig(
        num_users=8,
        num_slots=12,
        task=TaskConfig(
            arrival_mode="periodic",
            period_choices=(2, 3),
            period_jitter_slots=1,
            t_min=0.6,
            t_max=1.0,
        ),
    )

    first = generate_trace(cfg, 1001)
    second = generate_trace(cfg, 1001)

    assert [slot.tasks for slot in first.slots] == [slot.tasks for slot in second.slots]
    for user in range(cfg.num_users):
        arrivals = [
            task.arrival_slot
            for slot in first.slots
            for task in slot.tasks
            if task.user_id == user
        ]
        assert arrivals == sorted(set(arrivals))


@pytest.mark.parametrize(
    "kwargs",
    (
        {"arrival_mode": "unknown"},
        {"period_choices": ()},
        {"period_choices": (0, 2)},
        {"period_choices": (True, 2)},
        {"period_jitter_slots": -1},
        {"period_jitter_slots": 2},
        {"period_choices": (1,), "period_jitter_slots": 1},
    ),
)
def test_task_config_rejects_invalid_periodic_settings(kwargs):
    with pytest.raises(ValueError):
        TaskConfig(**kwargs)


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


def test_trace_rejects_unknown_cluster_layout():
    cfg = ScenarioConfig(cluster_layout="unknown")

    with pytest.raises(ValueError, match="cluster_layout"):
        generate_trace(cfg, 1)
