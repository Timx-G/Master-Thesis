import itertools
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from uav_mec.domain import Task
from uav_mec.exact import ExactScheduleResult
from uav_mec.resource_allocation import (
    ExactAdmissionResult,
    exact_admission,
    minimum_cpu_prefix_admission,
    required_resources,
)


def _admission_tasks(seed: int, count: int = 6) -> tuple[Task, ...]:
    rng = np.random.default_rng(seed)
    return tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=float(rng.uniform(300_000.0, 900_000.0)),
            comp_cycles=float(rng.uniform(8e8, 1.8e9)),
            deadline_s=float(rng.uniform(0.6, 0.9)),
            local_cpu_hz=400_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in range(count)
    )


def _reference_subset(
    tasks: tuple[Task, ...],
    rates: dict[int, float],
    budget: float,
) -> tuple[int, ...]:
    requirements = {
        task.task_id: required_resources(task, rates[task.task_id])
        for task in tasks
    }
    candidates: list[tuple[tuple[int, ...], float]] = []
    for width in range(len(tasks) + 1):
        for task_subset in itertools.combinations(tasks, width):
            subset = tuple(sorted(task.task_id for task in task_subset))
            subset_requirements = tuple(requirements[task_id] for task_id in subset)
            if not all(requirement.feasible for requirement in subset_requirements):
                continue
            cpu_hz = math.fsum(
                requirement.cpu_hz for requirement in subset_requirements
            )
            if cpu_hz <= budget:
                candidates.append((subset, cpu_hz))
    selected, _cpu_hz = min(
        candidates,
        key=lambda row: (-len(row[0]), row[1], row[0]),
    )
    return selected


def test_dd_matches_exact_admission_on_small_random_instances() -> None:
    for seed in range(100):
        tasks = _admission_tasks(seed)
        rates = {
            task.task_id: 1_000_000.0 + 50_000.0 * task.task_id
            for task in tasks
        }
        budget = 4_000_000_000.0

        dd = minimum_cpu_prefix_admission(tasks, rates, budget)
        exact = exact_admission(tasks, rates, budget)
        reference = _reference_subset(tasks, rates, budget)

        assert isinstance(exact, ExactAdmissionResult)
        assert tuple(sorted(dd.completed_task_ids)) == exact.admitted_task_ids == reference
        assert dd.total_cpu_hz == pytest.approx(exact.total_cpu_hz)
        assert exact.enumerated_subsets == 2 ** len(tasks)


def test_minimum_cpu_occurs_at_alpha_min_boundary() -> None:
    task = _admission_tasks(7, count=1)[0]
    requirement = required_resources(task, 2_000_000.0)
    next_alpha = math.nextafter(requirement.alpha, 1.0)

    next_requirement = required_resources(
        task,
        2_000_000.0,
        alpha=next_alpha,
    )

    assert next_requirement.cpu_hz >= requirement.cpu_hz


def test_exact_admission_is_a_frozen_schedule_free_result() -> None:
    tasks = _admission_tasks(3, count=3)
    rates = {task.task_id: 2_000_000.0 for task in tasks}

    result = exact_admission(tasks, rates, 4_000_000_000.0)

    assert not hasattr(result, "__dict__")
    assert not hasattr(result, "schedule")
    assert tuple(result.requirements) == tuple(sorted(task.task_id for task in tasks))
    with pytest.raises(FrozenInstanceError):
        result.total_cpu_hz = 0.0
    with pytest.raises(TypeError):
        result.requirements[tasks[0].task_id] = result.requirements[tasks[0].task_id]


def test_exact_admission_enumerates_infeasible_tasks_without_admitting_them() -> None:
    local = Task(10, 10, 0, 1e6, 0.5e9, 1.0, 1e9)
    feasible = Task(11, 11, 0, 1e6, 1.2e9, 0.8, 1e9)
    link_infeasible = Task(12, 12, 0, 1e6, 2.0e9, 0.8, 1e9)
    tasks = (link_infeasible, local, feasible)
    rates = {11: 4e6, 12: 1e3}

    result = exact_admission(tasks, rates, 4e9)

    assert result.admitted_task_ids == (10, 11)
    assert result.rejected_task_ids == (12,)
    assert result.enumerated_subsets == 8
    assert result.total_cpu_hz == result.requirements[11].cpu_hz


def test_exact_admission_local_task_does_not_require_unused_downlink_rate() -> None:
    local = Task(20, 20, 0, 1e6, 0.5e9, 1.0, 1e9, result_ratio=0.5)

    result = exact_admission((local,), {}, 0.0)

    assert not isinstance(result, ExactScheduleResult)
    assert result.admitted_task_ids == (local.task_id,)
    assert result.total_cpu_hz == 0.0
    assert result.requirements[local.task_id].local_only


@pytest.mark.parametrize("invalid_alpha", [-0.1, 1.1, math.nan, math.inf])
def test_explicit_requirement_alpha_is_validated(invalid_alpha: float) -> None:
    task = _admission_tasks(11, count=1)[0]

    with pytest.raises(ValueError, match="alpha"):
        required_resources(task, 2_000_000.0, alpha=invalid_alpha)
