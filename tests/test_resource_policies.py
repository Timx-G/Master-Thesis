from dataclasses import replace

import numpy as np
import pytest

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, evaluate_schedule
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import (
    AllocationPolicy,
    allocate_with_policy,
)


def test_equal_cpu_policy_assigns_identical_cpu_to_all_scheduled_tasks():
    tasks = (
        Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.6e9, 0.8, 1e9),
    )

    result = allocate_with_policy(
        AllocationPolicy.CROSSING_EQUAL,
        tasks,
        {1: 4e6, 2: 4e6},
        4e9,
        0.1,
    )

    assert {decision.cpu_hz for decision in result.decisions} == {2e9}
    assert result.total_cpu_hz == 4e9


def test_fixed_half_policy_really_uses_alpha_half():
    task = Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9)

    result = allocate_with_policy(
        AllocationPolicy.FIXED_HALF_EQUAL,
        (task,),
        {1: 5e6},
        4e9,
        0.1,
    )

    assert result.decisions[0].alpha == 0.5


def test_deadline_policy_matches_the_proven_allocator():
    tasks = (
        Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.8e9, 0.8, 1e9),
    )

    result = allocate_with_policy(
        AllocationPolicy.DEADLINE,
        tasks,
        {1: 3e6, 2: 3e6},
        2e9,
        0.1,
    )

    assert result.total_cpu_hz <= 2e9
    by_id = {decision.task_id: decision for decision in result.decisions}
    assert by_id[1].alpha == pytest.approx(1.0 / 3.0)
    assert by_id[2].alpha == pytest.approx(5.0 / 9.0)
    assert result.completed_task_ids


@pytest.mark.parametrize("invalid_rate", [0.0, -1.0, float("nan")])
def test_equal_policy_rejects_invalid_rates_without_nan_decisions(invalid_rate):
    task = Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9)

    result = allocate_with_policy(
        AllocationPolicy.CROSSING_EQUAL,
        (task,),
        {1: invalid_rate},
        4e9,
        0.1,
    )

    (decision,) = result.decisions
    assert not decision.completed
    assert decision.reason == "invalid_rate"
    assert decision.delay_s is None
    assert decision.cpu_hz == 0.0
    assert result.total_cpu_hz == 0.0


def test_equal_policy_handles_zero_cpu_budget_as_a_miss():
    task = Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9)

    result = allocate_with_policy(
        AllocationPolicy.FIXED_HALF_EQUAL,
        (task,),
        {1: 5e6},
        0.0,
        0.1,
    )

    assert not result.decisions[0].completed
    assert result.decisions[0].reason == "cpu_budget"
    assert result.total_cpu_hz == 0.0


def test_evaluator_routes_tasks_through_selected_policy():
    task = Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[1e-7]]), 1)
    schedule = Schedule((ScheduleEntry(1, 0, 0, 0, "solo"),))
    deadline = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9)
    fixed = replace(
        deadline,
        allocation_policy=AllocationPolicy.FIXED_HALF_EQUAL,
    )

    outcome = evaluate_schedule(
        snapshot,
        schedule,
        NOMAChannel(ChannelConfig(), 100.0),
        fixed,
    )

    assert outcome.decisions[0].alpha == 0.5
    assert outcome.decisions[0].reason == "fixed_half_equal"


def test_evaluation_config_rejects_raw_policy_strings():
    with pytest.raises(ValueError, match="allocation_policy"):
        EvaluationConfig(
            1,
            5e6,
            0.2,
            0.02,
            1.0,
            4e9,
            allocation_policy="deadline",
        )
