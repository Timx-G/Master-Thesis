from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_mec.domain import ExperimentTrace, SlotSnapshot
from uav_mec.experiment import load_definition
from uav_mec.experiment_registry import load_registry, registered_family
from uav_mec.model import ScenarioConfig, TaskConfig
from uav_mec.trace import generate_trace


REGISTRY_PATH = Path("configs/thesis/experiment_registry_v2.toml")


def test_trace_is_identical_for_the_same_seed(development_seed_access):
    cfg = ScenarioConfig(
        num_users=6,
        num_slots=4,
        random_burst=True,
        user_mobility=True,
    )

    first = generate_trace(cfg, 123, seed_access=development_seed_access)
    second = generate_trace(cfg, 123, seed_access=development_seed_access)

    assert first.seed == second.seed
    assert [slot.tasks for slot in first.slots] == [slot.tasks for slot in second.slots]
    for left, right in zip(first.slots, second.slots, strict=True):
        np.testing.assert_array_equal(left.channel_gains, right.channel_gains)


def test_trace_assigns_each_arrival_one_stable_task_id(development_seed_access):
    cfg = ScenarioConfig(num_users=8, num_slots=6, random_burst=True)

    trace = generate_trace(cfg, 456, seed_access=development_seed_access)

    tasks = [task for snapshot in trace.slots for task in snapshot.tasks]
    task_ids = [task.task_id for task in tasks]
    assert len(task_ids) == len(set(task_ids))
    for snapshot in trace.slots:
        assert all(task.arrival_slot == snapshot.slot_index for task in snapshot.tasks)
        assert all(task.task_id == (task.user_id << 32) | task.job_index for task in snapshot.tasks)
        assert snapshot.channel_gains.shape == (len(snapshot.tasks), cfg.num_uavs)


def test_experiment_trace_freezes_a_supplied_slot_list():
    source = [SlotSnapshot(0, (), np.empty((0, 1)), 1)]

    trace = ExperimentTrace(1, source)
    source.clear()

    assert isinstance(trace.slots, tuple)
    assert len(trace.slots) == 1


def test_trace_rejects_unsupported_nonfixed_cluster_layout(development_seed_access):
    cfg = ScenarioConfig(fixed_cluster_centers=False)

    with pytest.raises(ValueError, match="fixed_cluster_centers"):
        generate_trace(cfg, 1, seed_access=development_seed_access)


def _arrivals_by_user(trace: ExperimentTrace) -> dict[int, list]:
    arrivals: dict[int, list] = {}
    for snapshot in trace.slots:
        for task in snapshot.tasks:
            arrivals.setdefault(task.user_id, []).append(task)
    return arrivals


def test_periodic_and_bernoulli_modes_materialize_different_arrivals(
    development_seed_access,
):
    periodic = ScenarioConfig(
        num_users=6,
        num_slots=12,
        task=TaskConfig(
            arrival_mode="periodic",
            arrival_prob=0.0,
            period_choices=(2,),
            period_jitter_slots=0,
        ),
    )
    bernoulli = replace(
        periodic,
        task=replace(periodic.task, arrival_mode="bernoulli"),
    )

    periodic_trace = generate_trace(
        periodic,
        17,
        seed_access=development_seed_access,
    )
    bernoulli_trace = generate_trace(
        bernoulli,
        17,
        seed_access=development_seed_access,
    )

    periodic_arrivals = _arrivals_by_user(periodic_trace)
    assert set(periodic_arrivals) == set(range(periodic.num_users))
    for tasks in periodic_arrivals.values():
        slots = [task.arrival_slot for task in tasks]
        assert np.diff(slots).tolist() == [2] * (len(slots) - 1)
        assert [task.job_index for task in tasks] == list(range(len(tasks)))
    assert not any(snapshot.tasks for snapshot in bernoulli_trace.slots)


def test_periodic_jitter_is_seed_stable_and_keeps_positive_intervals(
    development_seed_access,
):
    cfg = ScenarioConfig(
        num_users=8,
        num_slots=30,
        task=TaskConfig(
            arrival_mode="periodic",
            period_choices=(3,),
            period_jitter_slots=2,
            t_min=0.5,
            t_max=0.9,
        ),
    )

    first = generate_trace(cfg, 23, seed_access=development_seed_access)
    second = generate_trace(cfg, 23, seed_access=development_seed_access)

    assert [slot.tasks for slot in first.slots] == [
        slot.tasks for slot in second.slots
    ]
    intervals = []
    for tasks in _arrivals_by_user(first).values():
        slots = [task.arrival_slot for task in tasks]
        user_intervals = np.diff(slots).tolist()
        assert all(1 <= interval <= 5 for interval in user_intervals)
        assert [task.job_index for task in tasks] == list(range(len(tasks)))
        intervals.extend(user_intervals)
    assert any(interval != 3 for interval in intervals)

    deadlines = [
        task.deadline_s
        for snapshot in first.slots
        for task in snapshot.tasks
    ]
    assert all(0.5 <= deadline <= 0.9 for deadline in deadlines)
    assert len(set(deadlines)) > 1


@pytest.mark.parametrize(
    "family_name,expected_intervals,seed",
    (
        ("main", {2, 3}, 17),
        ("externality", {2}, 18),
    ),
)
def test_registered_periodic_and_deadline_fields_change_materialized_traces(
    family_name,
    expected_intervals,
    seed,
    development_seed_access,
) -> None:
    registry = load_registry(REGISTRY_PATH)
    definition = load_definition(
        registered_family(registry, family_name).config_path
    )

    trace = generate_trace(
        definition.scenario,
        seed,
        seed_access=development_seed_access,
    )

    tasks_by_user = _arrivals_by_user(trace)
    observed_intervals = {
        right.arrival_slot - left.arrival_slot
        for tasks in tasks_by_user.values()
        for left, right in zip(tasks, tasks[1:])
    }
    assert observed_intervals == expected_intervals
    assert all(
        [task.job_index for task in tasks] == list(range(len(tasks)))
        for tasks in tasks_by_user.values()
    )

    deadlines = {
        task.deadline_s
        for tasks in tasks_by_user.values()
        for task in tasks
    }
    assert min(deadlines) >= definition.scenario.task.t_min
    assert max(deadlines) <= definition.scenario.task.t_max
    assert len(deadlines) > 1


@pytest.mark.parametrize("t_min", (None, 0.7))
def test_deadline_is_exact_when_no_range_is_requested(
    t_min,
    development_seed_access,
):
    cfg = ScenarioConfig(
        num_users=4,
        num_slots=4,
        task=TaskConfig(
            arrival_mode="bernoulli",
            arrival_prob=1.0,
            t_min=t_min,
            t_max=0.7,
        ),
    )

    trace = generate_trace(cfg, 29, seed_access=development_seed_access)

    deadlines = {
        task.deadline_s
        for snapshot in trace.slots
        for task in snapshot.tasks
    }
    assert deadlines == {0.7}


@pytest.mark.parametrize(
    "overrides,match",
    (
        ({"arrival_mode": "jitter"}, "arrival_mode"),
        ({"period_choices": (0,)}, "period_choices"),
        ({"period_jitter_slots": -1}, "period_jitter_slots"),
        ({"t_max": 0.0}, "t_max"),
        ({"t_max": float("inf")}, "t_max"),
        ({"t_max": float("nan")}, "t_max"),
        ({"t_min": 0.9, "t_max": 0.8}, "t_min"),
        ({"t_min": float("nan")}, "t_min"),
    ),
)
def test_task_config_rejects_unsupported_arrival_or_deadline_semantics(
    overrides,
    match,
):
    with pytest.raises(ValueError, match=match):
        TaskConfig(**overrides)


def test_trace_fails_closed_if_arrival_mode_is_mutated_after_validation(
    development_seed_access,
):
    cfg = ScenarioConfig(num_users=1, num_slots=1)
    cfg.task.arrival_mode = "jitter"

    with pytest.raises(ValueError, match="arrival_mode"):
        generate_trace(cfg, 31, seed_access=development_seed_access)
