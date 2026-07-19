import math
from dataclasses import FrozenInstanceError
from itertools import combinations, permutations

import pytest

import uav_mec.resource_allocation as resource_allocation
from uav_mec.domain import Task
from uav_mec.resource_allocation import (
    AllocationResult,
    RequirementReason,
    ResourceRequirement,
    allocate_deadline_tasks,
    completion_delay_s,
    required_resources,
)


def tasks() -> tuple[Task, ...]:
    return (
        Task(10, 0, 0, 1e6, 1.1e9, 0.8, 1e9),
        Task(11, 1, 0, 1e6, 1.4e9, 0.8, 1e9),
        Task(12, 2, 0, 1e6, 1.9e9, 0.8, 1e9),
    )


def test_allocator_selects_smallest_cpu_costs_within_budget():
    rates = {10: 4e6, 11: 4e6, 12: 4e6}
    reqs = [required_resources(task, rates[task.task_id], 0.1) for task in tasks()]
    budget = reqs[0].cpu_hz + reqs[1].cpu_hz + 1.0

    result = allocate_deadline_tasks(tasks(), rates, budget, 0.1)

    assert result.completed_task_ids == (10, 11)
    assert result.total_cpu_hz <= budget


def test_allocator_cardinality_matches_brute_force():
    rates = {10: 3e6, 11: 2e6, 12: 5e6}
    task_list = tasks()
    requirements = {
        task.task_id: required_resources(task, rates[task.task_id], 0.1) for task in task_list
    }
    budget = 2.0e9
    brute_best = 0
    for size in range(len(task_list) + 1):
        for subset in combinations(task_list, size):
            if sum(requirements[task.task_id].cpu_hz for task in subset) <= budget:
                brute_best = max(brute_best, size)

    result = allocate_deadline_tasks(task_list, rates, budget, 0.1)

    assert len(result.completed_task_ids) == brute_best


def test_completion_delay_matches_normalized_parallel_branch_model():
    task = Task(
        20,
        0,
        0,
        1e6,
        1.6e9,
        0.8,
        1e9,
        result_ratio=0.1,
    )
    rate_bps = 4e6
    result_ratio = 0.1
    requirement = required_resources(task, rate_bps, rate_bps)
    local_delay_s = (1.0 - requirement.alpha) * task.comp_cycles / task.local_cpu_hz
    edge_delay_s = requirement.alpha * (
        (1.0 + result_ratio) * task.data_bits / rate_bps + task.comp_cycles / requirement.cpu_hz
    )

    delay_s = completion_delay_s(
        task,
        requirement.alpha,
        rate_bps,
        requirement.cpu_hz,
        rate_bps,
    )

    assert delay_s == pytest.approx(max(local_delay_s, edge_delay_s))


@pytest.mark.parametrize("invalid_budget", [-1.0, math.nan, math.inf])
def test_allocator_validates_cpu_budget_even_without_tasks(invalid_budget: float):
    with pytest.raises(ValueError, match="cpu_budget_hz"):
        allocate_deadline_tasks((), {}, invalid_budget, 0.1)


def test_allocator_requires_downlink_map_for_positive_task_ratio():
    task = Task(
        21,
        0,
        0,
        1e6,
        1.2e9,
        0.8,
        1e9,
        result_ratio=0.1,
    )

    with pytest.raises(KeyError, match="downlink_rate_bps"):
        allocate_deadline_tasks((task,), {task.task_id: 4e6}, 4e9)


def test_allocator_rejects_duplicate_task_ids():
    duplicate_tasks = (
        Task(30, 0, 0, 1e6, 600e6, 0.8, 1e9),
        Task(30, 1, 0, 2e6, 700e6, 0.8, 1e9),
    )

    with pytest.raises(ValueError, match="duplicate"):
        allocate_deadline_tasks(duplicate_tasks, {}, 0.0, 0.1)


def test_local_task_allows_missing_rate_entry():
    task = Task(40, 0, 0, 1e6, 600e6, 0.8, 1e9)

    result = allocate_deadline_tasks((task,), {}, 0.0, 0.1)

    assert result.completed_task_ids == (task.task_id,)
    assert result.decisions[0].reason == RequirementReason.LOCAL.value


def test_offloaded_task_rejects_missing_rate_entry():
    task = Task(41, 0, 0, 1e6, 1.2e9, 0.8, 1e9)

    with pytest.raises(KeyError, match="rate"):
        allocate_deadline_tasks((task,), {}, 2e9, 0.1)


def test_missing_rate_is_rejected_before_any_resource_derivation(
    monkeypatch: pytest.MonkeyPatch,
):
    first_task = Task(42, 0, 0, 1e6, 1.2e9, 0.8, 1e9)
    missing_task = Task(43, 1, 0, 1e6, 1.3e9, 0.8, 1e9)

    def unexpected_derivation(*args: object) -> ResourceRequirement:
        raise AssertionError("resource derivation must not start before rate validation")

    monkeypatch.setattr(resource_allocation, "required_resources", unexpected_derivation)

    with pytest.raises(KeyError, match="rate"):
        allocate_deadline_tasks(
            (first_task, missing_task),
            {first_task.task_id: 4e6},
            2e9,
            0.1,
        )


def equal_tenth_cost_tasks() -> tuple[Task, ...]:
    smallest_positive = math.nextafter(0.0, math.inf)
    return tuple(
        Task(task_id, task_id, 0, smallest_positive, 0.2, 1.0, 0.1) for task_id in range(100, 110)
    )


@pytest.mark.parametrize(
    ("budget", "expected_count"),
    [
        (1.0, 10),
        (math.nextafter(1.0, 0.0), 9),
        (math.nextafter(1.0, math.inf), 10),
    ],
)
def test_allocator_enforces_hard_budget_at_prefix_sum_boundaries(
    budget: float,
    expected_count: int,
):
    task_list = equal_tenth_cost_tasks()
    rates = {task.task_id: 1.0 for task in task_list}

    result = allocate_deadline_tasks(task_list, rates, budget, 0.0)
    completed_mec_cpu = math.fsum(
        decision.cpu_hz for decision in result.decisions if decision.completed
    )

    assert len(result.completed_task_ids) == expected_count
    assert result.total_cpu_hz == completed_mec_cpu
    assert completed_mec_cpu <= budget


def test_allocator_has_no_absolute_budget_tolerance_at_low_scale():
    smallest_positive = math.nextafter(0.0, math.inf)
    task = Task(120, 0, 0, smallest_positive, 2e-10, 1.0, 1e-10)
    requirement = required_resources(task, 1.0, 0.0)
    assert 0.0 < requirement.cpu_hz < 1e-9

    result = allocate_deadline_tasks((task,), {task.task_id: 1.0}, 0.0, 0.0)

    assert result.completed_task_ids == ()
    assert result.total_cpu_hz == 0.0


def mec_task_with_cpu_cost(task_id: int, cpu_hz: float) -> Task:
    return Task(
        task_id,
        task_id,
        0,
        math.nextafter(0.0, math.inf),
        2.0 * cpu_hz,
        1.0,
        cpu_hz,
    )


def test_allocator_breaks_a_real_equal_cost_tie_by_smallest_task_id():
    task_list = (
        mec_task_with_cpu_cost(201, 0.1),
        mec_task_with_cpu_cost(200, 0.1),
    )
    rates = {task.task_id: 1.0 for task in task_list}

    result = allocate_deadline_tasks(task_list, rates, 0.1, 0.0)

    assert result.completed_task_ids == (200,)


def test_allocator_prioritizes_cpu_cost_when_cost_and_id_orders_are_opposite():
    task_list = (
        mec_task_with_cpu_cost(210, 0.3),
        mec_task_with_cpu_cost(220, 0.2),
        mec_task_with_cpu_cost(230, 0.1),
    )
    rates = {task.task_id: 1.0 for task in task_list}
    budget = math.fsum((0.1, 0.2))

    result = allocate_deadline_tasks(task_list, rates, budget, 0.0)

    assert result.completed_task_ids == (220, 230)


def test_allocator_result_is_invariant_under_all_input_permutations():
    task_list = (
        mec_task_with_cpu_cost(240, 0.1),
        mec_task_with_cpu_cost(241, 0.2),
        mec_task_with_cpu_cost(242, 0.3),
    )
    rates = {task.task_id: 1.0 for task in task_list}
    budget = math.fsum((0.1, 0.2))
    expected = allocate_deadline_tasks(task_list, rates, budget, 0.0)

    for permutation in permutations(task_list):
        assert allocate_deadline_tasks(permutation, rates, budget, 0.0) == expected


def test_empty_allocation_has_empty_ordered_outputs():
    assert allocate_deadline_tasks((), {}, 0.0, 0.0) == AllocationResult((), (), 0.0)


def test_all_local_tasks_complete_without_rates_or_mec_cpu():
    task_list = (
        Task(301, 1, 0, 1e6, 700e6, 0.8, 1e9),
        Task(300, 0, 0, 1e6, 600e6, 0.8, 1e9),
    )

    result = allocate_deadline_tasks(task_list, {}, 0.0, 0.1)

    assert result.completed_task_ids == (300, 301)
    assert tuple(decision.task_id for decision in result.decisions) == (300, 301)
    assert all(decision.reason == RequirementReason.LOCAL.value for decision in result.decisions)
    assert result.total_cpu_hz == 0.0


@pytest.mark.parametrize("invalid_rate", [0.0, -1.0, math.nan, math.inf])
def test_present_invalid_rate_is_an_invalid_rate_decision(invalid_rate: float):
    task = Task(310, 0, 0, 1e6, 1.2e9, 0.8, 1e9)

    result = allocate_deadline_tasks(
        (task,),
        {task.task_id: invalid_rate},
        2e9,
        0.1,
    )

    assert result.completed_task_ids == ()
    assert result.decisions[0].reason == RequirementReason.INVALID_RATE.value


def test_allocator_preserves_link_and_invalid_rate_rejection_reasons():
    invalid_rate_task = Task(320, 0, 0, 1e6, 1.2e9, 0.8, 1e9)
    link_infeasible_task = Task(321, 1, 0, 1e6, 2e9, 0.8, 1e9)

    result = allocate_deadline_tasks(
        (link_infeasible_task, invalid_rate_task),
        {invalid_rate_task.task_id: 0.0, link_infeasible_task.task_id: 100e3},
        10e9,
        0.1,
    )

    assert result.completed_task_ids == ()
    assert tuple(decision.reason for decision in result.decisions) == (
        RequirementReason.INVALID_RATE.value,
        RequirementReason.LINK_INFEASIBLE.value,
    )


def test_single_feasible_cost_over_budget_is_rejected_for_cpu_budget():
    task = mec_task_with_cpu_cost(330, 0.1)

    result = allocate_deadline_tasks(
        (task,),
        {task.task_id: 1.0},
        math.nextafter(0.1, 0.0),
        0.0,
    )

    assert result.completed_task_ids == ()
    assert result.decisions[0].reason == RequirementReason.CPU_BUDGET.value


def test_mixed_allocation_has_one_ordered_consistent_decision_per_task():
    local_task = Task(343, 3, 0, 1e6, 600e6, 0.8, 1e9)
    chosen_task = mec_task_with_cpu_cost(342, 0.1)
    budget_task = mec_task_with_cpu_cost(341, 0.2)
    invalid_task = Task(340, 0, 0, 1e6, 1.2e9, 0.8, 1e9)
    task_list = (local_task, chosen_task, budget_task, invalid_task)
    rates = {
        chosen_task.task_id: 1.0,
        budget_task.task_id: 1.0,
        invalid_task.task_id: 0.0,
    }

    result = allocate_deadline_tasks(task_list, rates, 0.1, 0.0)
    decisions_by_id = {decision.task_id: decision for decision in result.decisions}
    completed_from_decisions = tuple(
        decision.task_id for decision in result.decisions if decision.completed
    )
    completed_mec_cpu = math.fsum(
        decision.cpu_hz
        for decision in result.decisions
        if decision.completed and decision.cpu_hz > 0.0
    )

    assert tuple(decisions_by_id) == (340, 341, 342, 343)
    assert len(result.decisions) == len(task_list)
    assert result.completed_task_ids == completed_from_decisions == (342, 343)
    assert result.total_cpu_hz == completed_mec_cpu == 0.1
    assert decisions_by_id[340].reason == RequirementReason.INVALID_RATE.value
    assert decisions_by_id[341].reason == RequirementReason.CPU_BUDGET.value
    assert decisions_by_id[342].reason == RequirementReason.MEC.value
    assert decisions_by_id[343].reason == RequirementReason.LOCAL.value


def test_completed_mec_decision_uses_task4_requirement_and_canonical_delay():
    task = Task(
        350,
        0,
        0,
        1e6,
        1.6e9,
        0.8,
        1e9,
        result_ratio=0.1,
    )
    rate_bps = 4e6
    result_ratio = 0.1
    requirement = required_resources(task, rate_bps, rate_bps)

    result = allocate_deadline_tasks(
        (task,),
        {task.task_id: rate_bps},
        requirement.cpu_hz,
        {task.task_id: rate_bps},
    )
    decision = result.decisions[0]
    expected_delay_s = completion_delay_s(
        task,
        requirement.alpha,
        rate_bps,
        requirement.cpu_hz,
        rate_bps,
    )

    assert decision.alpha == requirement.alpha
    assert decision.cpu_hz == requirement.cpu_hz
    assert decision.delay_s == expected_delay_s
    assert math.isfinite(decision.delay_s)
    assert decision.delay_s <= task.deadline_s
    assert decision.slack_s == task.deadline_s - decision.delay_s


def test_completion_delay_local_path_ignores_unused_radio_and_mec_values():
    task = Task(360, 0, 0, 1e6, 600e6, 0.8, 1e9)

    assert completion_delay_s(task, 0.0, math.nan, 0.0) == task.local_delay_s


def test_local_deadline_equality_uses_same_ieee_path_for_classification_and_decision():
    comp_cycles = float.fromhex("0x1.c8df1bf7731c0p-842")
    local_cpu_hz = float.fromhex("0x1.f63e461756020p+181")
    deadline_s = comp_cycles / local_cpu_hz
    task = Task(361, 0, 0, 1.0, comp_cycles, deadline_s, local_cpu_hz)
    assert task.local_delay_s == task.deadline_s

    result = allocate_deadline_tasks((task,), {}, 0.0, 0.0)

    assert result.completed_task_ids == (task.task_id,)
    assert result.decisions[0].delay_s == task.deadline_s
    assert result.decisions[0].slack_s == 0.0


def test_canonical_delay_preserves_task4_grouped_ieee_feasibility_check():
    task = Task(
        362,
        0,
        0,
        float.fromhex("0x1.5b75d74482a62p+423"),
        float.fromhex("0x1.ba18e2de49403p+118"),
        float.fromhex("0x1.b7fe4b6a84df2p+140"),
        float.fromhex("0x1.22499787aab76p-24"),
        result_ratio=float.fromhex("0x1.999999999999ap-4"),
    )
    rate_bps = float.fromhex("0x1.8a0269110889ap+283")
    result_ratio = float.fromhex("0x1.999999999999ap-4")

    requirement = required_resources(task, rate_bps, rate_bps)
    grouped_delay_s = requirement.alpha * (
        (1.0 + result_ratio) * (task.data_bits / rate_bps) + task.comp_cycles / requirement.cpu_hz
    )
    delay_s = completion_delay_s(
        task,
        requirement.alpha,
        rate_bps,
        requirement.cpu_hz,
        rate_bps,
    )

    assert grouped_delay_s <= task.deadline_s
    assert delay_s <= task.deadline_s


def test_allocator_fails_fast_if_a_selected_requirement_is_not_deadline_feasible(
    monkeypatch: pytest.MonkeyPatch,
):
    task = mec_task_with_cpu_cost(370, 0.1)
    corrupted = ResourceRequirement(
        task.task_id,
        True,
        False,
        0.5,
        0.01,
        RequirementReason.MEC,
    )
    monkeypatch.setattr(resource_allocation, "required_resources", lambda *_: corrupted)

    with pytest.raises(RuntimeError, match="deadline"):
        allocate_deadline_tasks((task,), {task.task_id: 1.0}, 1.0, 0.0)


def test_allocation_result_is_frozen_and_slotted():
    result = AllocationResult((), (), 0.0)

    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.total_cpu_hz = 1.0


def brute_force_completed_count(
    task_list: tuple[Task, ...],
    rates: dict[int, float],
    budget: float,
    result_ratio: float,
) -> int:
    requirements = {
        task.task_id: required_resources(
            task,
            rates.get(task.task_id, math.nan),
            result_ratio,
        )
        for task in task_list
    }
    best = 0
    for size in range(len(task_list) + 1):
        for subset in combinations(task_list, size):
            subset_requirements = [requirements[task.task_id] for task in subset]
            if not all(requirement.feasible for requirement in subset_requirements):
                continue
            if math.fsum(requirement.cpu_hz for requirement in subset_requirements) <= budget:
                best = max(best, size)
    return best


def test_allocator_cardinality_matches_brute_force_across_instances_and_budgets():
    ordinary_tasks = tasks()
    instances = (
        (
            tuple(
                mec_task_with_cpu_cost(400 + index, cost)
                for index, cost in enumerate((0.05, 0.1, 0.2, 0.4))
            ),
            {400 + index: 1.0 for index in range(4)},
            0.0,
        ),
        (
            ordinary_tasks,
            {10: 3e6, 11: 2e6, 12: 5e6},
            0.1,
        ),
        (
            (
                Task(420, 0, 0, 1e6, 600e6, 0.8, 1e9),
                mec_task_with_cpu_cost(421, 0.1),
                mec_task_with_cpu_cost(422, 0.3),
                Task(423, 3, 0, 1e6, 1.2e9, 0.8, 1e9),
            ),
            {421: 1.0, 422: 1.0, 423: 0.0},
            0.0,
        ),
    )

    for task_list, rates, result_ratio in instances:
        requirements = [
            required_resources(
                task,
                rates.get(task.task_id, math.nan),
                result_ratio,
            )
            for task in task_list
        ]
        costs = sorted(
            requirement.cpu_hz
            for requirement in requirements
            if requirement.feasible and not requirement.local_only
        )
        first_two = math.fsum(costs[:2])
        total = math.fsum(costs)
        budgets = (
            0.0,
            costs[0],
            math.nextafter(first_two, 0.0),
            first_two,
            total,
            math.nextafter(total, math.inf),
        )

        for budget in budgets:
            result = allocate_deadline_tasks(task_list, rates, budget, result_ratio)
            assert len(result.completed_task_ids) == brute_force_completed_count(
                task_list,
                rates,
                budget,
                result_ratio,
            )
