"""Deadline-derived radio and MEC resource requirements."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from types import MappingProxyType

from .domain import AllocationDecision, Task
from .latency import completion_delay_s

_MAX_ULP_CORRECTIONS = 4


class RequirementReason(str, Enum):
    LOCAL = "local"
    MEC = "mec"
    INVALID_RATE = "invalid_rate"
    LINK_INFEASIBLE = "link_infeasible"
    CPU_BUDGET = "cpu_budget"


class AllocationPolicy(str, Enum):
    DEADLINE = "deadline"
    EDF_MIN_CPU = "edf_min_cpu"
    MIN_SLACK_MIN_CPU = "min_slack_min_cpu"
    CROSSING_EQUAL = "crossing_equal"
    FIXED_HALF_EQUAL = "fixed_half_equal"


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
class ExactAdmissionResult:
    """Exhaustive fixed-radio CPU-admission result.

    This certificate deliberately has no radio schedule: it conditions on the
    caller's closure-before task set and frozen rates, then enumerates CPU subsets
    under one static virtual-share capacity.
    """

    admitted_task_ids: tuple[int, ...]
    rejected_task_ids: tuple[int, ...]
    requirements: Mapping[int, ResourceRequirement]
    total_cpu_hz: float
    enumerated_subsets: int

    def __post_init__(self) -> None:
        admitted = tuple(self.admitted_task_ids)
        rejected = tuple(self.rejected_task_ids)
        if admitted != tuple(sorted(admitted)) or len(set(admitted)) != len(admitted):
            raise ValueError("admitted_task_ids must be unique and sorted")
        if rejected != tuple(sorted(rejected)) or len(set(rejected)) != len(rejected):
            raise ValueError("rejected_task_ids must be unique and sorted")
        if set(admitted).intersection(rejected):
            raise ValueError("admitted and rejected task IDs must be disjoint")

        requirements = dict(self.requirements)
        if set(requirements) != set(admitted).union(rejected):
            raise ValueError("requirements must cover every admitted and rejected task")
        if any(task_id != requirement.task_id for task_id, requirement in requirements.items()):
            raise ValueError("requirement keys must match requirement task IDs")
        if not math.isfinite(self.total_cpu_hz) or self.total_cpu_hz < 0.0:
            raise ValueError("total_cpu_hz must be finite and non-negative")
        if (
            not isinstance(self.enumerated_subsets, int)
            or isinstance(self.enumerated_subsets, bool)
            or self.enumerated_subsets < 1
        ):
            raise ValueError("enumerated_subsets must be a positive integer")

        object.__setattr__(self, "admitted_task_ids", admitted)
        object.__setattr__(self, "rejected_task_ids", rejected)
        object.__setattr__(
            self,
            "requirements",
            MappingProxyType(dict(sorted(requirements.items()))),
        )


def _positive_product_ratio(
    numerators: tuple[float, ...],
    denominators: tuple[float, ...] = (),
) -> float:
    """Evaluate a positive product ratio without avoidable intermediate overflow."""
    significand, exponent = _positive_product_ratio_parts(
        numerators,
        denominators,
    )
    try:
        return math.ldexp(significand, exponent)
    except OverflowError:
        return math.inf


def _positive_product_ratio_parts(
    numerators: tuple[float, ...],
    denominators: tuple[float, ...] = (),
) -> tuple[float, int]:
    """Keep a positive product ratio scaled when its float value is unrepresentable."""
    if any(value == 0.0 for value in numerators):
        return 0.0, 0

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

    return significand, exponent


def _crossing_alpha(
    task: Task,
    uplink_rate_bps: float,
    mec_cpu_hz: float,
    downlink_rate_bps: float | None,
) -> float:
    """Return A / sum(A, edge terms) without overflowing positive terms."""
    terms = [
        _positive_product_ratio_parts(
            (task.comp_cycles,),
            (task.local_cpu_hz,),
        ),
        _positive_product_ratio_parts(
            (task.data_bits,),
            (uplink_rate_bps,),
        ),
        _positive_product_ratio_parts(
            (task.comp_cycles,),
            (mec_cpu_hz,),
        ),
    ]
    if task.result_ratio > 0.0:
        terms.append(
            _positive_product_ratio_parts(
                (task.result_ratio, task.data_bits),
                (float(downlink_rate_bps),),
            )
        )

    scale_exponent = max(exponent for _, exponent in terms)
    scaled_denominator = math.fsum(
        math.ldexp(significand, exponent - scale_exponent) for significand, exponent in terms
    )
    local_significand, local_exponent = terms[0]
    ratio_significand, ratio_exponent = math.frexp(local_significand / scaled_denominator)
    return math.ldexp(
        ratio_significand,
        local_exponent - scale_exponent + ratio_exponent,
    )


def _cpu_sum(cpu_costs: Sequence[float]) -> float:
    """Return a deterministic CPU sum, treating unrepresentable totals as infinite."""
    try:
        return math.fsum(cpu_costs)
    except OverflowError:
        return math.inf


def _validate_downlink_rates_bps(
    downlink_rates_bps: Mapping[int, float] | None,
) -> None:
    if downlink_rates_bps is not None and not isinstance(downlink_rates_bps, Mapping):
        raise TypeError("downlink_rates_bps must be a mapping or None")


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


def _deadline_feasible_alpha(task: Task) -> tuple[float, bool]:
    """Return a conservative deadline-feasible alpha under IEEE arithmetic."""
    local_fraction = _positive_product_ratio(
        (task.remaining_time_s, task.local_cpu_hz),
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
    transmission_s: float,
    transfer_per_alpha_s: float,
) -> float:
    """Return a conservative deadline-feasible CPU requirement."""
    denominator = task.remaining_time_s - transmission_s
    denominator_candidate = _positive_product_ratio(
        (alpha, task.comp_cycles),
        (denominator,),
    )

    deadline_per_alpha = _positive_product_ratio(
        (task.remaining_time_s,),
        (alpha,),
    )
    grouped_window = deadline_per_alpha - transfer_per_alpha_s
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


def required_resources(
    task: Task,
    rate_bps: float,
    *,
    downlink_rate_bps: float | None = None,
    alpha: float | None = None,
) -> ResourceRequirement:
    """Return the task's pre-budget deadline-feasible resource requirements.

    With ``alpha=None``, a task that meets its deadline locally needs neither
    resource, so ``rate_bps`` is intentionally unused and may be invalid on that
    path. Otherwise the explicit, validated alpha is evaluated without changing
    the admission policy. At ordinary representable scales, the default ``alpha``
    and ``cpu_hz`` are the closed-form analytical minima. Near IEEE overflow,
    underflow, or cancellation, bounded corrections move only toward feasibility
    and may conservatively exceed an adjacent feasible float; the function never
    searches downward through quantization plateaus because doing so could
    understate the real-valued requirement.

    ``feasible`` means the module's normalized floating-point delay checks meet the
    deadline. MEC CPU-budget admission is handled separately by the allocator.
    """
    if alpha is not None:
        if isinstance(alpha, bool):
            raise ValueError("alpha must be finite and in [0, 1]")
        try:
            alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError("alpha must be finite and in [0, 1]") from exc
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1]")

    if alpha is None and task.local_delay_s <= task.remaining_time_s:
        return ResourceRequirement(
            task.task_id,
            True,
            True,
            0.0,
            0.0,
            RequirementReason.LOCAL,
        )
    if alpha == 0.0:
        if task.local_delay_s <= task.remaining_time_s:
            return ResourceRequirement(
                task.task_id,
                True,
                True,
                0.0,
                0.0,
                RequirementReason.LOCAL,
            )
        return _link_infeasible(task, 0.0)
    if not math.isfinite(rate_bps) or rate_bps <= 0:
        return ResourceRequirement(
            task.task_id,
            False,
            False,
            0.0,
            math.inf,
            RequirementReason.INVALID_RATE,
        )
    if task.result_ratio > 0.0 and downlink_rate_bps is None:
        raise ValueError("downlink_rate_bps is required for positive result_ratio")
    if task.result_ratio > 0.0 and (
        not math.isfinite(float(downlink_rate_bps)) or float(downlink_rate_bps) <= 0.0
    ):
        return ResourceRequirement(
            task.task_id,
            False,
            False,
            0.0,
            math.inf,
            RequirementReason.INVALID_RATE,
        )
    if alpha is None:
        selected_alpha, local_branch_feasible = _deadline_feasible_alpha(task)
    else:
        selected_alpha = float(alpha)
        local_branch_feasible = _local_branch_delay(task, selected_alpha) <= task.remaining_time_s
    alpha = selected_alpha
    if not local_branch_feasible:
        return _link_infeasible(task, alpha)
    uplink_s = _positive_product_ratio(
        (alpha, task.data_bits),
        (rate_bps,),
    )
    downlink_s = (
        0.0
        if task.result_ratio == 0.0
        else _positive_product_ratio(
            (alpha, task.result_ratio, task.data_bits),
            (float(downlink_rate_bps),),
        )
    )
    transmission_s = _cpu_sum((uplink_s, downlink_s))
    denominator = task.remaining_time_s - transmission_s
    if not math.isfinite(denominator) or denominator <= 0.0:
        return _link_infeasible(task, alpha)

    transfer_per_alpha_s = _cpu_sum(
        (
            _positive_product_ratio(
                (task.data_bits,),
                (rate_bps,),
            ),
            (
                0.0
                if task.result_ratio == 0.0
                else _positive_product_ratio(
                    (task.result_ratio, task.data_bits),
                    (float(downlink_rate_bps),),
                )
            ),
        )
    )
    cpu_hz = _deadline_feasible_cpu_hz(
        task,
        alpha,
        transmission_s,
        transfer_per_alpha_s,
    )
    if not math.isfinite(cpu_hz) or cpu_hz <= 0.0:
        return _link_infeasible(task, alpha)

    for _ in range(_MAX_ULP_CORRECTIONS + 1):
        delay_s = completion_delay_s(
            task,
            alpha,
            rate_bps,
            cpu_hz,
            downlink_rate_bps,
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


def _static_virtual_cpu_admission(
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    downlink_rates_bps: Mapping[int, float] | None,
    *,
    ordering: str,
) -> AllocationResult:
    """Allocate persistent virtual CPU shares in one frozen radio snapshot.

    Each admitted task keeps its derived CPU rate for the complete edge branch;
    the budget is therefore a simultaneous capacity constraint, not a preemptive
    single-processor demand-bound test.

    ``total_cpu_hz`` includes only completed MEC decisions. A missing rate for a
    task that requires offloading is an input data-flow error and raises ``KeyError``.
    """
    if not math.isfinite(cpu_budget_hz) or cpu_budget_hz < 0.0:
        raise ValueError("cpu_budget_hz must be finite and non-negative")
    _validate_downlink_rates_bps(downlink_rates_bps)

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
    missing_downlink_task_ids = sorted(
        task.task_id
        for task in task_list
        if task.requires_offload
        and task.result_ratio > 0.0
        and (downlink_rates_bps is None or task.task_id not in downlink_rates_bps)
    )
    if missing_downlink_task_ids:
        raise KeyError("missing downlink_rate_bps for task " f"{missing_downlink_task_ids[0]}")

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
            downlink_rate_bps=(
                None
                if not task.requires_offload or task.result_ratio == 0.0
                else downlink_rates_bps[task.task_id]
            ),
        )
        if requirement.local_only:
            delay_s = completion_delay_s(
                task,
                0.0,
                math.nan,
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

    if ordering == "deadline_first":
        mec_requirements.sort(
            key=lambda requirement: (
                task_by_id[requirement.task_id].absolute_deadline_s,
                requirement.cpu_hz,
                requirement.task_id,
            )
        )
    elif ordering == "minimum_slack":

        def slack_key(requirement: ResourceRequirement) -> tuple[float, float, int]:
            task = task_by_id[requirement.task_id]
            delay_s = completion_delay_s(
                task,
                requirement.alpha,
                rates_bps[requirement.task_id],
                cpu_budget_hz,
                None if task.result_ratio == 0.0 else downlink_rates_bps[requirement.task_id],
            )
            slack_s = task.deadline_s - delay_s
            return (slack_s, requirement.cpu_hz, requirement.task_id)

        mec_requirements.sort(key=slack_key)
    elif ordering == "minimum_cpu":
        mec_requirements.sort(key=lambda requirement: (requirement.cpu_hz, requirement.task_id))
    else:
        raise ValueError(f"unsupported static admission ordering {ordering}")
    chosen: set[int] = set()
    chosen_cpu_costs: list[float] = []
    for requirement in mec_requirements:
        candidate_cpu_hz = _cpu_sum((*chosen_cpu_costs, requirement.cpu_hz))
        if candidate_cpu_hz > cpu_budget_hz:
            break
        chosen.add(requirement.task_id)
        chosen_cpu_costs.append(requirement.cpu_hz)

    decisions = list(local_decisions)
    requirement_by_id = {requirement.task_id: requirement for requirement in mec_requirements}
    for task_id, requirement in requirement_by_id.items():
        task = task_by_id[task_id]
        if task_id in chosen:
            rate_bps = rates_bps[task_id]
            delay_s = completion_delay_s(
                task,
                requirement.alpha,
                rate_bps,
                requirement.cpu_hz,
                None if task.result_ratio == 0.0 else downlink_rates_bps[task_id],
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

    for task_id, requirement in rejected.items():
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


def minimum_cpu_prefix_admission(
    tasks: Sequence[Task],
    uplink_rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    downlink_rates_bps: Mapping[int, float] | None = None,
) -> AllocationResult:
    """Return the DD-MCAA static-share response for frozen radio rates.

    Feasible offloaded tasks are ordered by ``(minimum CPU, task_id)`` and the
    longest hard-capacity prefix is admitted. Under unit task value and independent
    static virtual CPU shares, this maximizes cardinality and then minimizes total
    CPU within that cardinality.
    """
    return _static_virtual_cpu_admission(
        tasks,
        uplink_rates_bps,
        cpu_budget_hz,
        downlink_rates_bps,
        ordering="minimum_cpu",
    )


def deadline_first_minimum_cpu_admission(
    tasks: Sequence[Task],
    uplink_rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    downlink_rates_bps: Mapping[int, float] | None = None,
) -> AllocationResult:
    """Return the distinct Deadline-First-MinCPU static-share prefix policy.

    This compatibility control orders by absolute deadline, minimum CPU, and task
    ID. It reserves persistent parallel virtual CPU shares and is not preemptive
    EDF scheduling.
    """
    return _static_virtual_cpu_admission(
        tasks,
        uplink_rates_bps,
        cpu_budget_hz,
        downlink_rates_bps,
        ordering="deadline_first",
    )


def minimum_slack_minimum_cpu_admission(
    tasks: Sequence[Task],
    uplink_rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    downlink_rates_bps: Mapping[int, float] | None = None,
) -> AllocationResult:
    """Admit the most urgent full-budget-laxity tasks, then minimum CPU.

    Laxity is evaluated on the frozen radio state with the task temporarily
    receiving the full per-UAV CPU budget.  The policy then reserves each
    selected task's analytical minimum static CPU share.
    """
    return _static_virtual_cpu_admission(
        tasks,
        uplink_rates_bps,
        cpu_budget_hz,
        downlink_rates_bps,
        ordering="minimum_slack",
    )


def allocate_deadline_tasks(
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    downlink_rates_bps: Mapping[int, float] | None = None,
) -> AllocationResult:
    """Compatibility name for :func:`minimum_cpu_prefix_admission`."""
    return minimum_cpu_prefix_admission(
        tasks,
        rates_bps,
        cpu_budget_hz,
        downlink_rates_bps,
    )


def exact_admission(
    tasks: Sequence[Task],
    uplink_rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    downlink_rates_bps: Mapping[int, float] | None = None,
) -> ExactAdmissionResult:
    """Exhaustively optimize fixed-radio static-share CPU admission.

    Every subset of the supplied closure-before task set is visited. The objective
    is maximum admitted cardinality, then minimum total CPU, then lexicographically
    smallest sorted task-ID tuple. This oracle neither receives nor returns a radio
    schedule and does not apply admission closure.
    """
    if not math.isfinite(cpu_budget_hz) or cpu_budget_hz < 0.0:
        raise ValueError("cpu_budget_hz must be finite and non-negative")
    _validate_downlink_rates_bps(downlink_rates_bps)

    task_list = tuple(tasks)
    seen_task_ids: set[int] = set()
    for task in task_list:
        if task.task_id in seen_task_ids:
            raise ValueError(f"duplicate task_id {task.task_id}")
        seen_task_ids.add(task.task_id)

    missing_rate_task_ids = sorted(
        task.task_id
        for task in task_list
        if task.requires_offload and task.task_id not in uplink_rates_bps
    )
    if missing_rate_task_ids:
        raise KeyError(f"missing uplink_rate_bps for task {missing_rate_task_ids[0]}")
    missing_downlink_task_ids = sorted(
        task.task_id
        for task in task_list
        if task.requires_offload
        and task.result_ratio > 0.0
        and (downlink_rates_bps is None or task.task_id not in downlink_rates_bps)
    )
    if missing_downlink_task_ids:
        raise KeyError("missing downlink_rate_bps for task " f"{missing_downlink_task_ids[0]}")

    requirements = {
        task.task_id: required_resources(
            task,
            (uplink_rates_bps[task.task_id] if task.requires_offload else math.nan),
            downlink_rate_bps=(
                None
                if not task.requires_offload or task.result_ratio == 0.0
                else downlink_rates_bps[task.task_id]
            ),
        )
        for task in sorted(task_list, key=lambda item: item.task_id)
    }
    ordered_task_ids = tuple(requirements)
    best_task_ids: tuple[int, ...] = ()
    best_cpu_hz = 0.0
    best_key: tuple[int, float, tuple[int, ...]] = (0, 0.0, ())
    enumerated_subsets = 0

    for width in range(len(ordered_task_ids) + 1):
        for subset in combinations(ordered_task_ids, width):
            enumerated_subsets += 1
            subset_requirements = tuple(requirements[task_id] for task_id in subset)
            if not all(requirement.feasible for requirement in subset_requirements):
                continue
            total_cpu_hz = _cpu_sum(
                tuple(requirement.cpu_hz for requirement in subset_requirements)
            )
            if total_cpu_hz > cpu_budget_hz:
                continue
            candidate_key = (-len(subset), total_cpu_hz, subset)
            if candidate_key < best_key:
                best_key = candidate_key
                best_task_ids = subset
                best_cpu_hz = total_cpu_hz

    admitted = tuple(best_task_ids)
    admitted_set = set(admitted)
    rejected = tuple(task_id for task_id in ordered_task_ids if task_id not in admitted_set)
    return ExactAdmissionResult(
        admitted_task_ids=admitted,
        rejected_task_ids=rejected,
        requirements=requirements,
        total_cpu_hz=best_cpu_hz,
        enumerated_subsets=enumerated_subsets,
    )


def _equal_cpu_allocation(
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    downlink_rates_bps: Mapping[int, float] | None,
    fixed_alpha: float | None,
) -> AllocationResult:
    if not math.isfinite(cpu_budget_hz) or cpu_budget_hz < 0.0:
        raise ValueError("cpu_budget_hz must be finite and non-negative")
    if fixed_alpha is not None and (
        not math.isfinite(fixed_alpha) or not 0.0 <= fixed_alpha <= 1.0
    ):
        raise ValueError("fixed_alpha must be finite and in [0, 1]")

    ordered_tasks = tuple(sorted(tasks, key=lambda task: task.task_id))
    if len({task.task_id for task in ordered_tasks}) != len(ordered_tasks):
        raise ValueError("duplicate task_id in allocation input")
    if not ordered_tasks:
        return AllocationResult((), (), 0.0)

    missing_downlink_task_ids = sorted(
        task.task_id
        for task in ordered_tasks
        if task.requires_offload
        and task.result_ratio > 0.0
        and (downlink_rates_bps is None or task.task_id not in downlink_rates_bps)
    )
    if missing_downlink_task_ids:
        raise KeyError("missing downlink_rate_bps for task " f"{missing_downlink_task_ids[0]}")

    cpu_hz = cpu_budget_hz / len(ordered_tasks)
    if cpu_hz == 0.0:
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

    decisions: list[AllocationDecision] = []
    for task in ordered_tasks:
        rate_bps = rates_bps.get(task.task_id, 0.0)
        downlink_rate_bps = None if task.result_ratio == 0.0 else downlink_rates_bps[task.task_id]
        if (
            not math.isfinite(rate_bps)
            or rate_bps <= 0.0
            or (
                task.result_ratio > 0.0
                and (not math.isfinite(downlink_rate_bps) or downlink_rate_bps <= 0.0)
            )
        ):
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
            alpha = _crossing_alpha(
                task,
                rate_bps,
                cpu_hz,
                downlink_rate_bps,
            )
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
            alpha,
            rate_bps,
            cpu_hz,
            downlink_rate_bps,
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
    completed_ids = tuple(decision.task_id for decision in ordered if decision.completed)
    total_cpu_hz = _cpu_sum(tuple(decision.cpu_hz for decision in ordered if decision.cpu_hz > 0.0))
    if total_cpu_hz > cpu_budget_hz:
        raise RuntimeError("equal allocation exceeds cpu_budget_hz")
    return AllocationResult(ordered, completed_ids, total_cpu_hz)


def allocate_with_policy(
    policy: AllocationPolicy,
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    downlink_rates_bps: Mapping[int, float] | None = None,
) -> AllocationResult:
    """Route a fixed task/rate set through one controlled allocation policy."""
    if not isinstance(policy, AllocationPolicy):
        raise ValueError("policy must be an AllocationPolicy value")
    _validate_downlink_rates_bps(downlink_rates_bps)
    if policy is AllocationPolicy.DEADLINE:
        return minimum_cpu_prefix_admission(
            tasks,
            rates_bps,
            cpu_budget_hz,
            downlink_rates_bps,
        )
    if policy is AllocationPolicy.EDF_MIN_CPU:
        return deadline_first_minimum_cpu_admission(
            tasks,
            rates_bps,
            cpu_budget_hz,
            downlink_rates_bps,
        )
    if policy is AllocationPolicy.MIN_SLACK_MIN_CPU:
        return minimum_slack_minimum_cpu_admission(
            tasks,
            rates_bps,
            cpu_budget_hz,
            downlink_rates_bps,
        )
    if policy is AllocationPolicy.CROSSING_EQUAL:
        return _equal_cpu_allocation(
            tasks,
            rates_bps,
            cpu_budget_hz,
            downlink_rates_bps,
            None,
        )
    if policy is AllocationPolicy.FIXED_HALF_EQUAL:
        return _equal_cpu_allocation(
            tasks,
            rates_bps,
            cpu_budget_hz,
            downlink_rates_bps,
            0.5,
        )
    raise ValueError(f"unsupported allocation policy {policy}")
