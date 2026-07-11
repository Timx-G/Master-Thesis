"""Deadline-derived radio and MEC resource requirements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .domain import Task

_MAX_ULP_CORRECTIONS = 4


class RequirementReason(str, Enum):
    LOCAL = "local"
    MEC = "mec"
    INVALID_RATE = "invalid_rate"
    LINK_INFEASIBLE = "link_infeasible"
    CPU_BUDGET = "cpu_budget"


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    task_id: int
    feasible: bool
    local_only: bool
    alpha: float
    cpu_hz: float
    reason: RequirementReason


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


def _minimum_representable_alpha(task: Task) -> tuple[float, bool]:
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


def _edge_branch_meets_deadline(
    task: Task,
    alpha: float,
    cpu_hz: float,
    rate_bps: float,
    result_ratio: float,
    transmission_s: float,
) -> bool:
    computation_s = _positive_product_ratio(
        (alpha, task.comp_cycles),
        (cpu_hz,),
    )
    if transmission_s + computation_s > task.deadline_s:
        return False

    grouped_terms = (1.0 + result_ratio) * (task.data_bits / rate_bps) + (task.comp_cycles / cpu_hz)
    if math.isfinite(grouped_terms):
        return alpha * grouped_terms <= task.deadline_s
    return True


def _minimum_cpu_hz(
    task: Task,
    alpha: float,
    rate_bps: float,
    result_ratio: float,
    transmission_s: float,
) -> float:
    denominator = task.deadline_s - transmission_s
    denominator_candidate = _positive_product_ratio(
        (alpha, task.comp_cycles),
        (denominator,),
    )

    transfer_per_alpha = _positive_product_ratio(
        (1.0 + result_ratio, task.data_bits),
        (rate_bps,),
    )
    deadline_per_alpha = _positive_product_ratio(
        (task.deadline_s,),
        (alpha,),
    )
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


def required_resources(
    task: Task,
    rate_bps: float,
    result_ratio: float,
) -> ResourceRequirement:
    """Return the task's pre-budget minimum radio and MEC requirements.

    A task that meets its deadline locally needs neither resource, so ``rate_bps``
    is intentionally unused and may be invalid on that path. MEC CPU-budget
    admission is handled by the allocator after this mathematical/link check.
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
    alpha, local_branch_feasible = _minimum_representable_alpha(task)
    if not local_branch_feasible:
        return _link_infeasible(task, alpha)
    transmission_s = _positive_product_ratio(
        (alpha, 1.0 + result_ratio, task.data_bits),
        (rate_bps,),
    )
    denominator = task.deadline_s - transmission_s
    if not math.isfinite(denominator) or denominator <= 0.0:
        return _link_infeasible(task, alpha)

    cpu_hz = _minimum_cpu_hz(
        task,
        alpha,
        rate_bps,
        result_ratio,
        transmission_s,
    )
    if not math.isfinite(cpu_hz) or cpu_hz <= 0.0:
        return _link_infeasible(task, alpha)

    for _ in range(_MAX_ULP_CORRECTIONS + 1):
        if _edge_branch_meets_deadline(
            task,
            alpha,
            cpu_hz,
            rate_bps,
            result_ratio,
            transmission_s,
        ):
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
