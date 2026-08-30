"""Deadline-derived radio and MEC resource requirements."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import combinations

from .domain import AllocationDecision, Task

_MAX_ULP_CORRECTIONS = 4


class RequirementReason(str, Enum):
    LOCAL = "local"
    MEC = "mec"
    INVALID_RATE = "invalid_rate"
    LINK_INFEASIBLE = "link_infeasible"
    CPU_BUDGET = "cpu_budget"


class AllocationPolicy(str, Enum):
    DEADLINE = "deadline"
    CROSSING_EQUAL = "crossing_equal"
    FIXED_HALF_EQUAL = "fixed_half_equal"
    EDF_MIN_CPU = "edf_min_cpu"
    MIN_SLACK_MIN_CPU = "min_slack_min_cpu"
    EXACT_ORACLE = "exact_oracle"


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    task_id: int
    feasible: bool
    local_only: bool
    alpha: float
    cpu_hz: float
    reason: RequirementReason


@dataclass(frozen=True, slots=True)
class AllocationResult:
    decisions: tuple[AllocationDecision, ...]
    completed_task_ids: tuple[int, ...]
    total_cpu_hz: float


@dataclass(frozen=True, slots=True)
class AllocationScore:
    """Objective fields needed while screening a wireless schedule."""

    completed_count: int
    total_cpu_hz: float


@dataclass(frozen=True, slots=True)
class _PreparedAllocation:
    task_by_id: dict[int, Task]
    local_decisions: tuple[AllocationDecision, ...]
    mec_requirements: tuple[ResourceRequirement, ...]
    rejected: dict[int, ResourceRequirement]


def _positive_product_ratio(
    numerators: tuple[float, ...],
    denominators: tuple[float, ...] = (),
) -> float:
    """Evaluate a positive product ratio without avoidable intermediate overflow."""
    if any(value == 0.0 for value in numerators):
        return 0.0

    significand = 1.0
    exponent = 0
    for value in numerators:
        fraction, value_exponent = math.frexp(value)
        significand *= fraction
        exponent += value_exponent
        significand, normalization = math.frexp(significand)
        exponent += normalization
    for value in denominators:
        fraction, value_exponent = math.frexp(value)
        significand /= fraction
        exponent -= value_exponent
        significand, normalization = math.frexp(significand)
        exponent += normalization

    try:
        return math.ldexp(significand, exponent)
    except OverflowError:
        return math.inf


def _cpu_sum(cpu_costs: Sequence[float]) -> float:
    """Return a deterministic CPU sum, treating unrepresentable totals as infinite."""
    try:
        return math.fsum(cpu_costs)
    except OverflowError:
        return math.inf


def _link_infeasible(task: Task, alpha: float) -> ResourceRequirement:
    return ResourceRequirement(
        task.task_id,
        False,
        False,
        alpha,
        math.inf,
        RequirementReason.LINK_INFEASIBLE,
    )


def _local_branch_delay(task: Task, alpha: float) -> float:
    return (1.0 - alpha) * task.comp_cycles / task.local_cpu_hz


@lru_cache(maxsize=262_144)
def _normalized_local_branch_delay(task: Task, alpha: float) -> float:
    return _positive_product_ratio(
        (1.0 - alpha, task.comp_cycles),
        (task.local_cpu_hz,),
    )


@lru_cache(maxsize=262_144)
def _normalized_transmission_s(
    task: Task,
    rate_bps: float,
    result_ratio: float,
    alpha: float,
) -> float:
    return _positive_product_ratio(
        (alpha, 1.0 + result_ratio, task.data_bits),
        (rate_bps,),
    )


@lru_cache(maxsize=262_144)
def _transfer_per_alpha_s(
    task: Task,
    rate_bps: float,
    result_ratio: float,
) -> float:
    return _positive_product_ratio(
        (1.0 + result_ratio, task.data_bits),
        (rate_bps,),
    )


@lru_cache(maxsize=65_536)
def _deadline_per_alpha_s(task: Task, alpha: float) -> float:
    return _positive_product_ratio(
        (task.deadline_s,),
        (alpha,),
    )


def completion_delay_s(
    task: Task,
    rate_bps: float,
    result_ratio: float,
    alpha: float,
    cpu_hz: float,
) -> float:
    """Return the normalized parallel local/MEC completion delay.

    ``alpha == 0`` is the explicit local-only path and does not consume or
    validate radio and MEC resources. For positive ``alpha``, normalized product
    ratios avoid intermediate overflow. The returned edge delay conservatively
    covers both the normalized split and finite grouped IEEE evaluations so Task 4
    corrections only move CPU toward deadline feasibility.
    """
    if not math.isfinite(result_ratio) or result_ratio < 0.0:
        raise ValueError("result_ratio must be finite and non-negative")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and between zero and one")

    if alpha == 0.0:
        return task.local_delay_s
    if not math.isfinite(rate_bps) or rate_bps <= 0.0:
        raise ValueError("rate_bps must be finite and positive when alpha is positive")
    if not math.isfinite(cpu_hz) or cpu_hz <= 0.0:
        raise ValueError("cpu_hz must be finite and positive when alpha is positive")

    local_delay_s = _normalized_local_branch_delay(task, alpha)
    transmission_s = _normalized_transmission_s(
        task,
        rate_bps,
        result_ratio,
        alpha,
    )
    computation_s = _positive_product_ratio(
        (alpha, task.comp_cycles),
        (cpu_hz,),
    )
    try:
        edge_delay_s = math.fsum((transmission_s, computation_s))
    except OverflowError:
        edge_delay_s = math.inf

    grouped_terms = (1.0 + result_ratio) * (task.data_bits / rate_bps) + (task.comp_cycles / cpu_hz)
    if math.isfinite(grouped_terms):
        edge_delay_s = max(edge_delay_s, alpha * grouped_terms)
    return max(local_delay_s, edge_delay_s)


@lru_cache(maxsize=65_536)
def _deadline_feasible_alpha(task: Task) -> tuple[float, bool]:
    """Return a conservative deadline-feasible alpha under IEEE arithmetic."""
    local_fraction = _positive_product_ratio(
        (task.deadline_s, task.local_cpu_hz),
        (task.comp_cycles,),
    )
    alpha = min(max(1.0 - local_fraction, 0.0), 1.0)
    if alpha == 0.0:
        alpha = math.nextafter(0.0, 1.0)

    for _ in range(_MAX_ULP_CORRECTIONS + 1):
        local_delay = _local_branch_delay(task, alpha)
        if local_delay <= task.deadline_s:
            return alpha, True
        if alpha == 1.0:
            break

        if math.isfinite(local_delay):
            correction = _positive_product_ratio(
                (local_delay - task.deadline_s, task.local_cpu_hz),
                (task.comp_cycles,),
            )
            candidate = min(alpha + correction, 1.0)
        else:
            candidate = 1.0
        if candidate > alpha:
            alpha = candidate
        else:
            alpha = math.nextafter(alpha, 1.0)

    return alpha, _local_branch_delay(task, alpha) <= task.deadline_s


def _deadline_feasible_cpu_hz(
    task: Task,
    alpha: float,
    rate_bps: float,
    result_ratio: float,
    transmission_s: float,
) -> float:
    """Return a conservative deadline-feasible CPU requirement."""
    denominator = task.deadline_s - transmission_s
    denominator_candidate = _positive_product_ratio(
        (alpha, task.comp_cycles),
        (denominator,),
    )

    transfer_per_alpha = _transfer_per_alpha_s(task, rate_bps, result_ratio)
    deadline_per_alpha = _deadline_per_alpha_s(task, alpha)
    grouped_window = deadline_per_alpha - transfer_per_alpha
    grouped_candidate = 0.0
    if math.isfinite(grouped_window) and grouped_window > 0.0:
        grouped_candidate = _positive_product_ratio(
            (task.comp_cycles,),
            (grouped_window,),
        )

    cpu_hz = max(denominator_candidate, grouped_candidate)
    if cpu_hz == 0.0:
        cpu_hz = math.nextafter(0.0, math.inf)
    return cpu_hz


@lru_cache(maxsize=262_144)
def required_resources(
    task: Task,
    rate_bps: float,
    result_ratio: float,
) -> ResourceRequirement:
    """Return the task's pre-budget deadline-feasible resource requirements.

    A task that meets its deadline locally needs neither resource, so ``rate_bps``
    is intentionally unused and may be invalid on that path. At ordinary
    representable scales, ``alpha`` and ``cpu_hz`` are the closed-form analytical
    minima. Near IEEE overflow, underflow, or cancellation, bounded corrections
    move only toward feasibility and may conservatively exceed an adjacent feasible
    float; the function never searches downward through quantization plateaus because
    doing so could understate the real-valued requirement.

    ``feasible`` means the module's normalized floating-point delay checks meet the
    deadline. MEC CPU-budget admission is handled separately by the allocator.
    """
    if not math.isfinite(result_ratio) or result_ratio < 0:
        raise ValueError("result_ratio must be finite and non-negative")
    if task.local_delay_s <= task.deadline_s:
        return ResourceRequirement(
            task.task_id,
            True,
            True,
            0.0,
            0.0,
            RequirementReason.LOCAL,
        )
    if not math.isfinite(rate_bps) or rate_bps <= 0:
        return ResourceRequirement(
            task.task_id,
            False,
            False,
            0.0,
            math.inf,
            RequirementReason.INVALID_RATE,
        )
    alpha, local_branch_feasible = _deadline_feasible_alpha(task)
    if not local_branch_feasible:
        return _link_infeasible(task, alpha)
    transmission_s = _normalized_transmission_s(
        task,
        rate_bps,
        result_ratio,
        alpha,
    )
    denominator = task.deadline_s - transmission_s
    if not math.isfinite(denominator) or denominator <= 0.0:
        return _link_infeasible(task, alpha)

    cpu_hz = _deadline_feasible_cpu_hz(
        task,
        alpha,
        rate_bps,
        result_ratio,
        transmission_s,
    )
    if not math.isfinite(cpu_hz) or cpu_hz <= 0.0:
        return _link_infeasible(task, alpha)

    for _ in range(_MAX_ULP_CORRECTIONS + 1):
        delay_s = completion_delay_s(
            task,
            rate_bps,
            result_ratio,
            alpha,
            cpu_hz,
        )
        if math.isfinite(delay_s) and delay_s <= task.deadline_s:
            break
        cpu_hz = math.nextafter(cpu_hz, math.inf)
        if not math.isfinite(cpu_hz):
            return _link_infeasible(task, alpha)
    else:
        return _link_infeasible(task, alpha)

    return ResourceRequirement(
        task.task_id,
        True,
        False,
        alpha,
        cpu_hz,
        RequirementReason.MEC,
    )


@lru_cache(maxsize=262_144)
def screening_required_resources(
    task: Task,
    rate_bps: float,
    result_ratio: float,
) -> ResourceRequirement:
    """Evaluate the registered finite-scale formula for candidate screening."""

    if not math.isfinite(result_ratio) or result_ratio < 0.0:
        raise ValueError("result_ratio must be finite and non-negative")
    if task.local_delay_s <= task.deadline_s:
        return ResourceRequirement(
            task.task_id,
            True,
            True,
            0.0,
            0.0,
            RequirementReason.LOCAL,
        )
    if not math.isfinite(rate_bps) or rate_bps <= 0.0:
        return ResourceRequirement(
            task.task_id,
            False,
            False,
            0.0,
            math.inf,
            RequirementReason.INVALID_RATE,
        )
    alpha, local_branch_feasible = _deadline_feasible_alpha(task)
    if not local_branch_feasible:
        return _link_infeasible(task, alpha)

    transfer_per_alpha = (1.0 + result_ratio) * task.data_bits / rate_bps
    transmission_s = alpha * transfer_per_alpha
    denominator = task.deadline_s - transmission_s
    grouped_window = task.deadline_s / alpha - transfer_per_alpha
    if denominator <= 0.0 or grouped_window <= 0.0:
        return _link_infeasible(task, alpha)
    cpu_hz = max(
        alpha * task.comp_cycles / denominator,
        task.comp_cycles / grouped_window,
    )
    if not math.isfinite(cpu_hz) or cpu_hz <= 0.0:
        return _link_infeasible(task, alpha)

    for _ in range(_MAX_ULP_CORRECTIONS + 1):
        local_delay_s = (1.0 - alpha) * task.comp_cycles / task.local_cpu_hz
        edge_delay_s = alpha * (transfer_per_alpha + task.comp_cycles / cpu_hz)
        if max(local_delay_s, edge_delay_s) <= task.deadline_s:
            break
        cpu_hz = math.nextafter(cpu_hz, math.inf)
    else:
        return _link_infeasible(task, alpha)
    return ResourceRequirement(
        task.task_id,
        True,
        False,
        alpha,
        cpu_hz,
        RequirementReason.MEC,
    )


def _prepare_deadline_allocation(
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    result_ratio: float,
) -> _PreparedAllocation:
    if not math.isfinite(cpu_budget_hz) or cpu_budget_hz < 0.0:
        raise ValueError("cpu_budget_hz must be finite and non-negative")
    if not math.isfinite(result_ratio) or result_ratio < 0.0:
        raise ValueError("result_ratio must be finite and non-negative")

    task_list = tuple(tasks)
    seen_task_ids: set[int] = set()
    for task in task_list:
        if task.task_id in seen_task_ids:
            raise ValueError(f"duplicate task_id {task.task_id}")
        seen_task_ids.add(task.task_id)

    missing_rate_task_ids = sorted(
        task.task_id
        for task in task_list
        if task.requires_offload and task.task_id not in rates_bps
    )
    if missing_rate_task_ids:
        raise KeyError(f"missing rate_bps for task {missing_rate_task_ids[0]}")

    local_decisions: list[AllocationDecision] = []
    mec_requirements: list[ResourceRequirement] = []
    rejected: dict[int, ResourceRequirement] = {}
    task_by_id = {task.task_id: task for task in task_list}

    for task in task_list:
        if task.requires_offload:
            rate_bps = rates_bps[task.task_id]
        else:
            rate_bps = math.nan
        requirement = required_resources(
            task,
            rate_bps,
            result_ratio,
        )
        if requirement.local_only:
            delay_s = completion_delay_s(
                task,
                math.nan,
                result_ratio,
                0.0,
                0.0,
            )
            if not math.isfinite(delay_s) or delay_s > task.deadline_s:
                raise RuntimeError(f"local task {task.task_id} violates its deadline requirement")
            local_decisions.append(
                AllocationDecision(
                    task.task_id,
                    0.0,
                    0.0,
                    True,
                    delay_s,
                    task.deadline_s - delay_s,
                    requirement.reason.value,
                )
            )
        elif requirement.feasible:
            mec_requirements.append(requirement)
        else:
            rejected[task.task_id] = requirement

    return _PreparedAllocation(
        task_by_id,
        tuple(local_decisions),
        tuple(mec_requirements),
        rejected,
    )


def _ranked_admission(
    requirements: Sequence[ResourceRequirement],
    task_by_id: Mapping[int, Task],
    cpu_budget_hz: float,
    policy: AllocationPolicy,
) -> set[int]:
    """Admit feasible tasks in one reproducible priority order.

    EDF-MinCPU orders by ``(absolute deadline, f_min, task_id)``.
    Min-Slack-MinCPU uses pre-offloading local slack
    ``deadline - C / f_local`` before the same deterministic tie breakers.
    A task that does not fit is rejected without blocking cheaper later tasks.
    """
    if policy is AllocationPolicy.DEADLINE:
        ordered = sorted(
            requirements,
            key=lambda requirement: (
                requirement.cpu_hz,
                requirement.task_id,
            ),
        )
    elif policy is AllocationPolicy.EDF_MIN_CPU:
        ordered = sorted(
            requirements,
            key=lambda requirement: (
                task_by_id[requirement.task_id].absolute_deadline_s,
                requirement.cpu_hz,
                requirement.task_id,
            ),
        )
    elif policy is AllocationPolicy.MIN_SLACK_MIN_CPU:
        ordered = sorted(
            requirements,
            key=lambda requirement: (
                task_by_id[requirement.task_id].deadline_s
                - task_by_id[requirement.task_id].local_delay_s,
                requirement.cpu_hz,
                requirement.task_id,
            ),
        )
    else:
        raise ValueError(f"unsupported ranked allocation policy {policy}")

    chosen: set[int] = set()
    chosen_cpu_costs: list[float] = []
    for requirement in ordered:
        candidate_cpu_hz = _cpu_sum(
            (*chosen_cpu_costs, requirement.cpu_hz)
        )
        if candidate_cpu_hz <= cpu_budget_hz:
            chosen.add(requirement.task_id)
            chosen_cpu_costs.append(requirement.cpu_hz)
        elif policy is AllocationPolicy.DEADLINE:
            # The remaining requirements are no cheaper under MinCPU order.
            break
    return chosen


def _exact_admission(
    requirements: Sequence[ResourceRequirement],
    cpu_budget_hz: float,
) -> set[int]:
    """Enumerate fixed-schedule per-UAV admission subsets exactly.

    The objective is maximum completed-task cardinality, then minimum CPU
    consumption, then the lexicographically smallest task-ID tuple. Radio
    scheduling and serving-UAV association are fixed inputs to this oracle.
    """
    ordered = tuple(sorted(requirements, key=lambda item: item.task_id))
    if len(ordered) > 20:
        raise ValueError(
            "exact resource oracle supports at most 20 feasible MEC tasks"
        )

    for size in range(len(ordered), -1, -1):
        best: tuple[float, tuple[int, ...]] | None = None
        for subset in combinations(ordered, size):
            total_cpu_hz = _cpu_sum(
                tuple(requirement.cpu_hz for requirement in subset)
            )
            if total_cpu_hz <= cpu_budget_hz:
                candidate = (
                    total_cpu_hz,
                    tuple(
                        requirement.task_id
                        for requirement in subset
                    ),
                )
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            return set(best[1])
    return set()


def _materialize_allocation(
    prepared: _PreparedAllocation,
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    result_ratio: float,
    chosen: set[int],
) -> AllocationResult:
    task_by_id = prepared.task_by_id
    decisions = list(prepared.local_decisions)

    requirement_by_id = {
        requirement.task_id: requirement
        for requirement in prepared.mec_requirements
    }
    for task_id, requirement in requirement_by_id.items():
        task = task_by_id[task_id]
        if task_id in chosen:
            rate_bps = rates_bps[task_id]
            delay_s = completion_delay_s(
                task,
                rate_bps,
                result_ratio,
                requirement.alpha,
                requirement.cpu_hz,
            )
            if not math.isfinite(delay_s) or delay_s > task.deadline_s:
                raise RuntimeError(f"selected task {task_id} violates its deadline requirement")
            decisions.append(
                AllocationDecision(
                    task_id,
                    requirement.alpha,
                    requirement.cpu_hz,
                    True,
                    delay_s,
                    task.deadline_s - delay_s,
                    requirement.reason.value,
                )
            )
        else:
            decisions.append(
                AllocationDecision(
                    task_id,
                    requirement.alpha,
                    0.0,
                    False,
                    None,
                    0.0,
                    RequirementReason.CPU_BUDGET.value,
                )
            )

    for task_id, requirement in prepared.rejected.items():
        decisions.append(
            AllocationDecision(
                task_id,
                requirement.alpha,
                0.0,
                False,
                None,
                0.0,
                requirement.reason.value,
            )
        )

    ordered = tuple(sorted(decisions, key=lambda decision: decision.task_id))
    completed = tuple(decision.task_id for decision in ordered if decision.completed)
    total_cpu_hz = _cpu_sum(
        tuple(
            decision.cpu_hz for decision in ordered if decision.completed and decision.cpu_hz > 0.0
        )
    )
    if total_cpu_hz > cpu_budget_hz:
        raise RuntimeError("completed MEC decisions exceed cpu_budget_hz")
    return AllocationResult(ordered, completed, total_cpu_hz)


def _allocate_requirement_policy(
    policy: AllocationPolicy,
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    result_ratio: float,
) -> AllocationResult:
    prepared = _prepare_deadline_allocation(
        tasks,
        rates_bps,
        cpu_budget_hz,
        result_ratio,
    )
    if policy is AllocationPolicy.EXACT_ORACLE:
        chosen = _exact_admission(
            prepared.mec_requirements,
            cpu_budget_hz,
        )
    else:
        chosen = _ranked_admission(
            prepared.mec_requirements,
            prepared.task_by_id,
            cpu_budget_hz,
            policy,
        )
    return _materialize_allocation(
        prepared,
        rates_bps,
        cpu_budget_hz,
        result_ratio,
        chosen,
    )


def allocate_deadline_tasks(
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    result_ratio: float,
) -> AllocationResult:
    """Return the canonical maximum-cardinality deadline allocation.

    Local-feasible tasks complete at zero MEC cost. Feasible offloaded tasks are
    ordered by ``(required CPU, task_id)`` and the largest hard-budget prefix is
    selected. This maximizes completion cardinality, then minimizes total MEC CPU
    among maximum-cardinality subsets, with ``task_id`` resolving equal-cost
    choices. The prefix is the deterministic representative of that objective
    hierarchy; other maximum-cardinality feasible subsets can still exist.

    ``total_cpu_hz`` includes only completed MEC decisions. A missing rate for a
    task that requires offloading is an input data-flow error and raises ``KeyError``.
    """
    return _allocate_requirement_policy(
        AllocationPolicy.DEADLINE,
        tasks,
        rates_bps,
        cpu_budget_hz,
        result_ratio,
    )


def _equal_cpu_allocation(
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    result_ratio: float,
    fixed_alpha: float | None,
) -> AllocationResult:
    if not math.isfinite(cpu_budget_hz) or cpu_budget_hz < 0.0:
        raise ValueError("cpu_budget_hz must be finite and non-negative")
    if not math.isfinite(result_ratio) or result_ratio < 0.0:
        raise ValueError("result_ratio must be finite and non-negative")
    if fixed_alpha is not None and (
        not math.isfinite(fixed_alpha) or not 0.0 <= fixed_alpha <= 1.0
    ):
        raise ValueError("fixed_alpha must be finite and in [0, 1]")

    ordered_tasks = tuple(sorted(tasks, key=lambda task: task.task_id))
    if len({task.task_id for task in ordered_tasks}) != len(ordered_tasks):
        raise ValueError("duplicate task_id in allocation input")
    if not ordered_tasks:
        return AllocationResult((), (), 0.0)

    if cpu_budget_hz == 0.0:
        decisions = tuple(
            AllocationDecision(
                task.task_id,
                fixed_alpha if fixed_alpha is not None else 0.0,
                0.0,
                False,
                None,
                0.0,
                RequirementReason.CPU_BUDGET.value,
            )
            for task in ordered_tasks
        )
        return AllocationResult(decisions, (), 0.0)

    cpu_hz = cpu_budget_hz / len(ordered_tasks)
    decisions: list[AllocationDecision] = []
    for task in ordered_tasks:
        rate_bps = rates_bps.get(task.task_id, 0.0)
        if not math.isfinite(rate_bps) or rate_bps <= 0.0:
            decisions.append(
                AllocationDecision(
                    task.task_id,
                    0.0,
                    0.0,
                    False,
                    None,
                    0.0,
                    RequirementReason.INVALID_RATE.value,
                )
            )
            continue

        if fixed_alpha is None:
            local_full_s = task.comp_cycles / task.local_cpu_hz
            try:
                edge_unit_s = math.fsum(
                    (
                        (1.0 + result_ratio) * task.data_bits / rate_bps,
                        task.comp_cycles / cpu_hz,
                    )
                )
            except OverflowError:
                edge_unit_s = math.inf
            denominator = local_full_s + edge_unit_s
            alpha = local_full_s / denominator if math.isfinite(denominator) else math.nan
        else:
            alpha = fixed_alpha

        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            decisions.append(
                AllocationDecision(
                    task.task_id,
                    0.0,
                    0.0,
                    False,
                    None,
                    0.0,
                    RequirementReason.LINK_INFEASIBLE.value,
                )
            )
            continue

        delay_s = completion_delay_s(
            task,
            rate_bps,
            result_ratio,
            alpha,
            cpu_hz,
        )
        completed = math.isfinite(delay_s) and delay_s <= task.deadline_s
        reason = (
            AllocationPolicy.CROSSING_EQUAL.value
            if fixed_alpha is None
            else AllocationPolicy.FIXED_HALF_EQUAL.value
        )
        decisions.append(
            AllocationDecision(
                task.task_id,
                alpha,
                cpu_hz,
                completed,
                delay_s if math.isfinite(delay_s) else None,
                max(task.deadline_s - delay_s, 0.0) if completed else 0.0,
                reason,
            )
        )

    ordered = tuple(decisions)
    completed_ids = tuple(
        decision.task_id for decision in ordered if decision.completed
    )
    total_cpu_hz = _cpu_sum(
        tuple(decision.cpu_hz for decision in ordered if decision.cpu_hz > 0.0)
    )
    if total_cpu_hz > cpu_budget_hz:
        raise RuntimeError("equal allocation exceeds cpu_budget_hz")
    return AllocationResult(ordered, completed_ids, total_cpu_hz)


def allocate_with_policy(
    policy: AllocationPolicy,
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    result_ratio: float,
) -> AllocationResult:
    """Route a fixed task/rate set through one controlled allocation policy."""
    if not isinstance(policy, AllocationPolicy):
        raise ValueError("policy must be an AllocationPolicy value")
    if policy is AllocationPolicy.DEADLINE:
        return allocate_deadline_tasks(
            tasks,
            rates_bps,
            cpu_budget_hz,
            result_ratio,
        )
    if policy in {
        AllocationPolicy.EDF_MIN_CPU,
        AllocationPolicy.MIN_SLACK_MIN_CPU,
        AllocationPolicy.EXACT_ORACLE,
    }:
        return _allocate_requirement_policy(
            policy,
            tasks,
            rates_bps,
            cpu_budget_hz,
            result_ratio,
        )
    if policy is AllocationPolicy.CROSSING_EQUAL:
        return _equal_cpu_allocation(
            tasks,
            rates_bps,
            cpu_budget_hz,
            result_ratio,
            None,
        )
    if policy is AllocationPolicy.FIXED_HALF_EQUAL:
        return _equal_cpu_allocation(
            tasks,
            rates_bps,
            cpu_budget_hz,
            result_ratio,
            0.5,
        )
    raise ValueError(f"unsupported allocation policy {policy}")


def allocation_score_with_policy(
    policy: AllocationPolicy,
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    result_ratio: float,
    *,
    prevalidated_offload_only: bool = False,
    screening_arithmetic: bool = False,
) -> AllocationScore:
    """Return allocation objective fields without materializing task decisions."""

    if policy in {
        AllocationPolicy.DEADLINE,
        AllocationPolicy.EDF_MIN_CPU,
        AllocationPolicy.MIN_SLACK_MIN_CPU,
        AllocationPolicy.EXACT_ORACLE,
    }:
        if not math.isfinite(cpu_budget_hz) or cpu_budget_hz < 0.0:
            raise ValueError("cpu_budget_hz must be finite and non-negative")
        if not math.isfinite(result_ratio) or result_ratio < 0.0:
            raise ValueError("result_ratio must be finite and non-negative")
        task_list = tuple(tasks)
        if (
            prevalidated_offload_only
            and screening_arithmetic
            and policy is AllocationPolicy.DEADLINE
        ):
            cpu_costs = []
            for task in task_list:
                requirement = screening_required_resources(
                    task,
                    rates_bps[task.task_id],
                    result_ratio,
                )
                if requirement.feasible and not requirement.local_only:
                    cpu_costs.append((requirement.cpu_hz, requirement.task_id))
            cpu_costs.sort()
            chosen_cpu_costs: list[float] = []
            for cpu_hz, _task_id in cpu_costs:
                candidate_cpu_hz = _cpu_sum((*chosen_cpu_costs, cpu_hz))
                if candidate_cpu_hz > cpu_budget_hz:
                    break
                chosen_cpu_costs.append(cpu_hz)
            return AllocationScore(
                len(chosen_cpu_costs),
                _cpu_sum(tuple(chosen_cpu_costs)),
            )
        if not prevalidated_offload_only:
            if len({task.task_id for task in task_list}) != len(task_list):
                raise ValueError("duplicate task_id in allocation input")
            missing = sorted(
                task.task_id
                for task in task_list
                if task.requires_offload and task.task_id not in rates_bps
            )
            if missing:
                raise KeyError(f"missing rate_bps for task {missing[0]}")
        if screening_arithmetic and not prevalidated_offload_only:
            raise ValueError(
                "screening_arithmetic requires prevalidated_offload_only"
            )
        requirement_function = (
            screening_required_resources
            if screening_arithmetic
            else required_resources
        )
        requirements = tuple(
            requirement_function(
                task,
                rates_bps[task.task_id]
                if prevalidated_offload_only or task.requires_offload
                else math.nan,
                result_ratio,
            )
            for task in task_list
        )
        if prevalidated_offload_only and policy is AllocationPolicy.DEADLINE:
            ordered_requirements = sorted(
                (
                    requirement
                    for requirement in requirements
                    if requirement.feasible and not requirement.local_only
                ),
                key=lambda requirement: (
                    requirement.cpu_hz,
                    requirement.task_id,
                ),
            )
            chosen_cpu_costs: list[float] = []
            for requirement in ordered_requirements:
                candidate_cpu_hz = _cpu_sum(
                    (*chosen_cpu_costs, requirement.cpu_hz)
                )
                if candidate_cpu_hz > cpu_budget_hz:
                    break
                chosen_cpu_costs.append(requirement.cpu_hz)
            return AllocationScore(
                len(chosen_cpu_costs),
                _cpu_sum(tuple(chosen_cpu_costs)),
            )
        local_count = (
            0
            if prevalidated_offload_only
            else sum(requirement.local_only for requirement in requirements)
        )
        mec_requirements = tuple(
            requirement
            for requirement in requirements
            if requirement.feasible and not requirement.local_only
        )
        task_by_id = {task.task_id: task for task in task_list}
        if policy is AllocationPolicy.EXACT_ORACLE:
            chosen = _exact_admission(mec_requirements, cpu_budget_hz)
        else:
            chosen = _ranked_admission(
                mec_requirements,
                task_by_id,
                cpu_budget_hz,
                policy,
            )
        requirement_by_id = {
            requirement.task_id: requirement
            for requirement in mec_requirements
        }
        return AllocationScore(
            local_count + len(chosen),
            _cpu_sum(tuple(requirement_by_id[task_id].cpu_hz for task_id in chosen)),
        )

    result = allocate_with_policy(
        policy,
        tasks,
        rates_bps,
        cpu_budget_hz,
        result_ratio,
    )
    return AllocationScore(
        len(result.completed_task_ids),
        _cpu_sum(
            tuple(
                decision.cpu_hz
                for decision in result.decisions
                if decision.completed and decision.alpha > 0.0
            )
        ),
    )
