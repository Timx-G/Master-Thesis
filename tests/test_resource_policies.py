from dataclasses import replace
from itertools import combinations
import math
from pathlib import Path

import numpy as np
import pytest

import uav_mec.experiment as experiment_module
from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, evaluate_schedule
from uav_mec.experiment import _slot_method, load_definition
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import (
    AllocationPolicy,
    allocation_score_with_policy,
    allocate_with_policy,
    required_resources,
)
from uav_mec.scheduler import SearchResult
from uav_mec.trace import generate_trace


def _negligible_radio_task(
    task_id: int,
    *,
    comp_cycles: float,
    deadline_s: float,
) -> Task:
    return Task(
        task_id,
        task_id,
        0,
        math.nextafter(0.0, math.inf),
        comp_cycles,
        deadline_s,
        1.0,
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


@pytest.mark.parametrize("policy", tuple(AllocationPolicy))
def test_lightweight_allocation_score_matches_materialized_result(policy: AllocationPolicy):
    tasks = (
        Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9),
        Task(2, 1, 0, 1.5e6, 1.8e9, 0.9, 1e9),
        Task(3, 2, 0, 0.8e6, 1.1e9, 0.7, 1e9),
    )
    rates = {1: 3e6, 2: 4e6, 3: 2.5e6}

    score = allocation_score_with_policy(policy, tasks, rates, 4e9, 0.1)
    trusted_score = allocation_score_with_policy(
        policy,
        tasks,
        rates,
        4e9,
        0.1,
        prevalidated_offload_only=True,
    )
    result = allocate_with_policy(policy, tasks, rates, 4e9, 0.1)

    assert trusted_score == score
    assert score.completed_count == len(result.completed_task_ids)
    assert score.total_cpu_hz == pytest.approx(
        sum(
            decision.cpu_hz
            for decision in result.decisions
            if decision.completed and decision.alpha > 0.0
        )
    )


def test_edf_min_cpu_prioritizes_absolute_deadline_then_cpu_cost():
    early_expensive = _negligible_radio_task(
        10,
        comp_cycles=2.0,
        deadline_s=0.5,
    )
    late_cheap = (
        _negligible_radio_task(20, comp_cycles=2.0, deadline_s=1.0),
        _negligible_radio_task(21, comp_cycles=2.0, deadline_s=1.0),
    )
    tasks = (early_expensive, *late_cheap)
    rates = {task.task_id: 1.0 for task in tasks}
    early_cost = required_resources(early_expensive, 1.0, 0.0).cpu_hz

    result = allocate_with_policy(
        AllocationPolicy.EDF_MIN_CPU,
        tasks,
        rates,
        early_cost,
        0.0,
    )

    assert result.completed_task_ids == (early_expensive.task_id,)


def test_min_slack_min_cpu_prioritizes_most_negative_local_slack():
    urgent_expensive = _negligible_radio_task(
        30,
        comp_cycles=4.0,
        deadline_s=1.0,
    )
    less_urgent_cheap = (
        _negligible_radio_task(40, comp_cycles=2.0, deadline_s=1.0),
        _negligible_radio_task(41, comp_cycles=2.0, deadline_s=1.0),
    )
    tasks = (urgent_expensive, *less_urgent_cheap)
    rates = {task.task_id: 1.0 for task in tasks}
    urgent_cost = required_resources(urgent_expensive, 1.0, 0.0).cpu_hz

    result = allocate_with_policy(
        AllocationPolicy.MIN_SLACK_MIN_CPU,
        tasks,
        rates,
        urgent_cost,
        0.0,
    )

    assert result.completed_task_ids == (urgent_expensive.task_id,)


def test_exact_oracle_matches_complete_subset_enumeration():
    tasks = tuple(
        _negligible_radio_task(
            50 + index,
            comp_cycles=cycles,
            deadline_s=1.0,
        )
        for index, cycles in enumerate((2.0, 2.5, 3.0, 5.0))
    )
    rates = {task.task_id: 1.0 for task in tasks}
    requirements = {
        task.task_id: required_resources(task, 1.0, 0.0)
        for task in tasks
    }
    budget = math.fsum(
        (requirements[50].cpu_hz, requirements[52].cpu_hz)
    )
    feasible_subsets = [
        subset
        for size in range(len(tasks) + 1)
        for subset in combinations(tasks, size)
        if math.fsum(requirements[task.task_id].cpu_hz for task in subset)
        <= budget
    ]
    expected = min(
        feasible_subsets,
        key=lambda subset: (
            -len(subset),
            math.fsum(
                requirements[task.task_id].cpu_hz for task in subset
            ),
            tuple(sorted(task.task_id for task in subset)),
        ),
    )

    result = allocate_with_policy(
        AllocationPolicy.EXACT_ORACLE,
        tasks,
        rates,
        budget,
        0.0,
    )

    assert result.completed_task_ids == tuple(
        sorted(task.task_id for task in expected)
    )


def test_exact_oracle_refuses_an_unbounded_exponential_instance():
    tasks = tuple(
        _negligible_radio_task(
            100 + index,
            comp_cycles=2.0,
            deadline_s=1.0,
        )
        for index in range(21)
    )

    with pytest.raises(ValueError, match="at most 20"):
        allocate_with_policy(
            AllocationPolicy.EXACT_ORACLE,
            tasks,
            {task.task_id: 1.0 for task in tasks},
            100.0,
            0.0,
        )


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
    deadline = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)
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


def test_wireless_search_embeds_the_selected_resource_policy(monkeypatch):
    definition = load_definition(Path("configs/thesis/confirmatory_resource_strong.toml"))
    snapshot = generate_trace(
        definition.scenario,
        definition.seeds[0],
        definition.csi_error_std,
    ).slots[0]
    captured: list[AllocationPolicy] = []

    def fake_multistart(initializers, planning, channel, evaluation, search):
        captured.append(evaluation.allocation_policy)
        initializer_id, schedule = initializers[0]
        outcome = evaluate_schedule(planning, schedule, channel, evaluation)
        return SearchResult(
            schedule,
            outcome,
            (),
            1,
            True,
            "local_optimum",
            initializer_id,
        )

    monkeypatch.setattr(
        experiment_module,
        "optimize_multistart",
        fake_multistart,
    )

    _slot_method(
        "full_edf_min_cpu",
        definition.scenario,
        replace(
            definition.evaluation,
            allocation_policy=AllocationPolicy.EDF_MIN_CPU,
        ),
        definition.max_iterations,
        [],
    )(snapshot)

    assert captured == [AllocationPolicy.EDF_MIN_CPU]


@pytest.mark.parametrize(
    ("method", "final_policy"),
    (
        ("fixed_radio_dd", AllocationPolicy.DEADLINE),
        ("fixed_radio_edf", AllocationPolicy.EDF_MIN_CPU),
        ("fixed_radio_min_slack", AllocationPolicy.MIN_SLACK_MIN_CPU),
        ("fixed_radio_exact", AllocationPolicy.EXACT_ORACLE),
    ),
)
def test_fixed_radio_resource_controls_change_only_final_allocation_policy(
    method,
    final_policy,
    monkeypatch,
):
    definition = load_definition(Path("configs/thesis/confirmatory_main_v2.toml"))
    snapshot = generate_trace(
        replace(definition.scenario, num_slots=1),
        42,
        definition.csi_error_std,
    ).slots[0]
    search_policies: list[AllocationPolicy] = []
    final_policies: list[AllocationPolicy] = []

    def fake_multistart(initializers, planning, channel, evaluation, search):
        search_policies.append(evaluation.allocation_policy)
        initializer_id, schedule = initializers[0]
        outcome = evaluate_schedule(planning, schedule, channel, evaluation)
        return SearchResult(
            schedule,
            outcome,
            (),
            1,
            True,
            "local_optimum",
            initializer_id,
        )

    def capture_final(snapshot, schedule, channel, evaluation):
        final_policies.append(evaluation.allocation_policy)
        return evaluate_schedule(snapshot, schedule, channel, evaluation)

    monkeypatch.setattr(experiment_module, "optimize_multistart", fake_multistart)
    monkeypatch.setattr(
        experiment_module,
        "evaluate_realized_schedule",
        capture_final,
    )

    _slot_method(
        method,
        definition.scenario,
        replace(definition.evaluation, allocation_policy=final_policy),
        definition.max_iterations,
        [],
    )(snapshot)

    assert search_policies == [AllocationPolicy.DEADLINE]
    assert final_policies == [final_policy]


def test_evaluation_config_rejects_raw_policy_strings():
    with pytest.raises(ValueError, match="allocation_policy"):
        EvaluationConfig(
            1,
            5e6,
            0.2,
            0.02,
            1.0,
            4e9,
            0.1,
            allocation_policy="deadline",
        )
