import math
from dataclasses import replace

import numpy as np
import pytest

from uav_mec.closure import close_assignment, freeze_plan, rate_raw_assignment
from uav_mec.domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import (
    EvaluationConfig,
    evaluate_frozen_plan,
    evaluate_schedule,
)
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import (
    AllocationPolicy,
    allocate_with_policy,
    required_resources,
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
    )

    assert {decision.cpu_hz for decision in result.decisions} == {2e9}
    assert result.total_cpu_hz == 4e9


def test_crossing_equal_avoids_overflow_when_crossing_terms_are_large():
    task = Task(
        1,
        0,
        0,
        math.nextafter(0.0, math.inf),
        1e308,
        9e307,
        1.0,
    )

    result = allocate_with_policy(
        AllocationPolicy.CROSSING_EQUAL,
        (task,),
        {task.task_id: 1.0},
        1.0,
    )

    (decision,) = result.decisions
    assert decision.alpha == 0.5
    assert decision.completed
    assert decision.delay_s == pytest.approx(5e307)
    assert decision.reason == AllocationPolicy.CROSSING_EQUAL.value
    assert result.completed_task_ids == (task.task_id,)
    assert result.total_cpu_hz == 1.0


def test_fixed_half_policy_really_uses_alpha_half():
    task = Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9)

    result = allocate_with_policy(
        AllocationPolicy.FIXED_HALF_EQUAL,
        (task,),
        {1: 5e6},
        4e9,
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
    )

    assert result.total_cpu_hz <= 2e9
    by_id = {decision.task_id: decision for decision in result.decisions}
    assert by_id[1].alpha == pytest.approx(1.0 / 3.0)
    assert by_id[2].alpha == pytest.approx(5.0 / 9.0)
    assert result.completed_task_ids


def test_deadline_first_min_cpu_is_distinct_from_dd_mcaa_cpu_order() -> None:
    smallest_positive = math.nextafter(0.0, math.inf)
    early_expensive = Task(10, 10, 0, smallest_positive, 0.65, 0.5, 1.0)
    later_cheap = Task(20, 20, 0, smallest_positive, 1.1, 1.0, 1.0)
    tasks = (later_cheap, early_expensive)
    rates = {task.task_id: 1.0 for task in tasks}
    early_requirement = required_resources(early_expensive, 1.0)

    dd = allocate_with_policy(
        AllocationPolicy.DEADLINE,
        tasks,
        rates,
        early_requirement.cpu_hz,
    )
    deadline_first = allocate_with_policy(
        AllocationPolicy.EDF_MIN_CPU,
        tasks,
        rates,
        early_requirement.cpu_hz,
    )

    assert dd.completed_task_ids == (later_cheap.task_id,)
    assert deadline_first.completed_task_ids == (early_expensive.task_id,)
    assert AllocationPolicy.EDF_MIN_CPU.value == "edf_min_cpu"


@pytest.mark.parametrize("invalid_rate", [0.0, -1.0, float("nan")])
def test_equal_policy_rejects_invalid_rates_without_nan_decisions(invalid_rate):
    task = Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9)

    result = allocate_with_policy(
        AllocationPolicy.CROSSING_EQUAL,
        (task,),
        {1: invalid_rate},
        4e9,
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
    )

    assert not result.decisions[0].completed
    assert result.decisions[0].reason == "cpu_budget"
    assert result.total_cpu_hz == 0.0


@pytest.mark.parametrize(
    "policy",
    (
        AllocationPolicy.CROSSING_EQUAL,
        AllocationPolicy.FIXED_HALF_EQUAL,
    ),
)
def test_equal_policy_handles_underflowed_per_task_cpu_as_misses(
    policy: AllocationPolicy,
):
    tasks = (
        Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.2e9, 0.8, 1e9),
    )
    smallest_positive = math.nextafter(0.0, math.inf)

    result = allocate_with_policy(
        policy,
        tasks,
        {1: 5e6, 2: 5e6},
        smallest_positive,
    )

    assert not result.completed_task_ids
    assert all(not decision.completed for decision in result.decisions)
    assert all(decision.reason == "cpu_budget" for decision in result.decisions)
    assert all(decision.cpu_hz == 0.0 for decision in result.decisions)
    assert result.total_cpu_hz == 0.0


@pytest.mark.parametrize("policy", tuple(AllocationPolicy))
def test_policy_allocator_rejects_removed_scalar_result_ratio(
    policy: AllocationPolicy,
):
    with pytest.raises(TypeError, match="downlink_rates_bps"):
        allocate_with_policy(policy, (), {}, 0.0, 0.1)


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


def test_realized_replay_preserves_fixed_policy_cpu_decisions() -> None:
    task = Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9)
    snapshot = SlotSnapshot(
        0,
        (task,),
        np.array([[2e-8]]),
        1,
        estimated_channel_gains=np.array([[1e-7]]),
    )
    schedule = Schedule((ScheduleEntry(1, 0, 0, 0, "solo"),))
    cfg = EvaluationConfig(
        1,
        5e6,
        0.2,
        0.02,
        0.01,
        4e9,
        allocation_policy=AllocationPolicy.FIXED_HALF_EQUAL,
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)
    closed = close_assignment(
        snapshot,
        rate_raw_assignment(
            snapshot,
            schedule,
            channel,
            cfg,
            channel_basis=ChannelBasis.PLANNING,
        ),
        channel,
        cfg,
    )

    realized = evaluate_frozen_plan(
        snapshot,
        freeze_plan(closed),
        channel,
        cfg,
    )

    planned = closed.allocation.decisions[0]
    replayed = realized.decisions[0]
    assert replayed.alpha == planned.alpha == 0.5
    assert replayed.cpu_hz == planned.cpu_hz
