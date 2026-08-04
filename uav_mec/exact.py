"""Complete small-instance schedule enumeration for exact references."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product

from .closure import (
    ClosedExecutableSchedule,
    close_assignment,
    closed_semantic_key,
    freeze_plan,
    rate_raw_assignment,
)
from .domain import (
    ChannelBasis,
    FrozenSlotPlan,
    Schedule,
    ScheduleEntry,
    SlotSnapshot,
    Task,
)
from .evaluation import (
    EvaluationConfig,
    InfeasibleScheduleError,
    build_outcome_from_closed,
    semantic_fingerprint,
)
from .externality import aggregate_victim_link_exposure
from .model import NOMAChannel
from .resource_allocation import required_resources
from .schedule import ScheduleValidationError
from .scheduler import network_objective


@dataclass(frozen=True, slots=True)
class ExactScheduleCertificate:
    complete: bool
    raw_states_enumerated: int
    unique_closed_states: int
    best_bound: tuple[int | None, float | None, float | None]
    lexicographic_gap: tuple[int | None, float | None, float | None]
    gap_status: str
    cpu_nontrivial: bool
    schedule_nontrivial: bool
    semantic_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.complete, bool):
            raise ValueError("complete must be bool")
        for name, value in (
            ("raw_states_enumerated", self.raw_states_enumerated),
            ("unique_closed_states", self.unique_closed_states),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.unique_closed_states > self.raw_states_enumerated:
            raise ValueError("unique closed states cannot exceed raw states")
        if self.gap_status not in {"exact", "bounded", "unknown"}:
            raise ValueError("gap_status must be exact, bounded, or unknown")
        if not isinstance(self.cpu_nontrivial, bool) or not isinstance(
            self.schedule_nontrivial,
            bool,
        ):
            raise ValueError("nontriviality flags must be bool")
        if not isinstance(self.semantic_fingerprint, str) or not self.semantic_fingerprint:
            raise ValueError("semantic_fingerprint must be a non-empty string")

        _validate_certificate_tuple("best_bound", self.best_bound)
        _validate_certificate_tuple("lexicographic_gap", self.lexicographic_gap)
        if self.complete:
            if self.gap_status != "exact":
                raise ValueError("a complete certificate must have exact gap status")
            if self.lexicographic_gap != (0, 0.0, 0.0):
                raise ValueError("a complete certificate must have zero gap")
        elif self.gap_status == "exact":
            raise ValueError("an incomplete certificate cannot be exact")
        if self.gap_status == "bounded" and any(
            value is None for value in self.best_bound + self.lexicographic_gap
        ):
            raise ValueError("a bounded certificate requires finite components")


@dataclass(frozen=True, slots=True)
class _ExactOutcomeView:
    """Compatibility projection for callers predating frozen exact plans."""

    completed_count: int
    decisions: tuple
    aggregate_victim_link_exposure_w: float


@dataclass(frozen=True, slots=True)
class ExactScheduleResult:
    plan: FrozenSlotPlan
    objective: tuple[int, float, float]
    certificate: ExactScheduleCertificate

    def __post_init__(self) -> None:
        if not isinstance(self.plan, FrozenSlotPlan):
            raise TypeError("plan must be a FrozenSlotPlan")
        _validate_certificate_tuple("objective", self.objective)
        if any(value is None for value in self.objective):
            raise ValueError("objective components must be finite")
        if not isinstance(self.certificate, ExactScheduleCertificate):
            raise TypeError("certificate must be an ExactScheduleCertificate")
        if self.plan.semantic_fingerprint != self.certificate.semantic_fingerprint:
            raise ValueError("plan and certificate fingerprints must agree")
        if self.certificate.complete and self.certificate.best_bound != self.objective:
            raise ValueError("a complete certificate bound must equal objective")
        if self.certificate.gap_status == "bounded":
            bound = self.certificate.best_bound
            if bound < self.objective:
                raise ValueError("a bounded certificate bound must cover objective")
            expected_gap = tuple(
                bound_value - objective_value
                for bound_value, objective_value in zip(
                    bound,
                    self.objective,
                    strict=True,
                )
            )
            if self.certificate.lexicographic_gap != expected_gap:
                raise ValueError(
                    "certificate gap must equal bound minus objective"
                )

    @property
    def schedule(self) -> Schedule:
        """Compatibility view of the certified frozen schedule."""
        return self.plan.schedule

    @property
    def outcome(self) -> _ExactOutcomeView:
        """Compatibility objective view without reopening the frozen plan."""
        return _ExactOutcomeView(
            self.objective[0],
            self.plan.allocation.decisions,
            aggregate_victim_link_exposure(self.plan.planning_rates),
        )

    @property
    def evaluated_schedules(self) -> int:
        """Compatibility alias for the number of legal raw states visited."""
        return self.certificate.raw_states_enumerated

    @property
    def complete_search(self) -> bool:
        """Compatibility alias for certificate completeness."""
        return self.certificate.complete


def _validate_certificate_tuple(
    name: str,
    values: tuple[int | None, float | None, float | None],
) -> None:
    if not isinstance(values, tuple) or len(values) != 3:
        raise ValueError(f"{name} must contain three components")
    if values[0] is not None and (
        not isinstance(values[0], int) or isinstance(values[0], bool)
    ):
        raise ValueError(f"{name} completion component must be an integer or None")
    for value in values:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} components must be finite or None")


def _legal_raw_assignments(
    tasks: tuple[Task, ...],
    resource_blocks: int,
) -> Iterator[tuple[int, ...]]:
    choices = range(-1, resource_blocks)
    for assignment in product(choices, repeat=len(tasks)):
        counts = [0] * resource_blocks
        scheduled_users: set[int] = set()
        legal = True
        for task, block in zip(tasks, assignment, strict=True):
            if block < 0:
                continue
            if task.user_id in scheduled_users:
                legal = False
                break
            scheduled_users.add(task.user_id)
            counts[block] += 1
            if counts[block] > 2:
                legal = False
                break
        if legal:
            yield assignment


def _raw_schedule(
    tasks: tuple,
    assignment: tuple[int, ...],
    n_subcarriers: int,
) -> Schedule:
    entries = []
    for task, block in zip(tasks, assignment, strict=True):
        if block < 0:
            continue
        uav, subcarrier = divmod(block, n_subcarriers)
        entries.append(
            ScheduleEntry(
                task.task_id,
                task.user_id,
                uav,
                subcarrier,
                "solo",
            )
        )
    return Schedule(tuple(entries))


def _cpu_requirement_audit(
    snapshot: SlotSnapshot,
    rated,
    cfg: EvaluationConfig,
    downlink_rates_bps: Mapping[int, float] | None,
) -> tuple[tuple[int, ...], float | None]:
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    requirements_by_uav: dict[int, list] = {}
    for record in rated.rates:
        task = task_by_id[record.task_id]
        requirement = required_resources(
            task,
            record.rate_bps,
            downlink_rate_bps=(
                None
                if task.result_ratio == 0.0
                else downlink_rates_bps[record.task_id]
            ),
        )
        requirements_by_uav.setdefault(record.serving_uav, []).append(requirement)
    overloaded = tuple(
        uav
        for uav, requirements in sorted(requirements_by_uav.items())
        if all(requirement.feasible for requirement in requirements)
        and math.fsum(requirement.cpu_hz for requirement in requirements)
        > cfg.cpu_budget_for(uav, snapshot.num_uavs)
    )
    all_requirements = tuple(
        requirement
        for requirements in requirements_by_uav.values()
        for requirement in requirements
    )
    total_cpu_hz = (
        math.fsum(requirement.cpu_hz for requirement in all_requirements)
        if all(requirement.feasible for requirement in all_requirements)
        else None
    )
    return overloaded, total_cpu_hz


def _certificate_bound(
    snapshot: SlotSnapshot,
    objective: tuple[int, float, float],
    complete: bool,
) -> tuple[
    tuple[int, float, float],
    tuple[int, float, float],
    str,
]:
    if complete:
        return objective, (0, 0.0, 0.0), "exact"
    bound = (len(snapshot.tasks), 0.0, 0.0)
    gap = (
        bound[0] - objective[0],
        bound[1] - objective[1],
        bound[2] - objective[2],
    )
    return bound, gap, "bounded"


def exact_schedule(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    max_heavy_tasks: int = 5,
    *,
    state_cap: int | None = None,
    timeout_s: float | None = None,
    downlink_rates_bps: Mapping[int, float] | None = None,
    audit: dict[str, object] | None = None,
) -> ExactScheduleResult:
    """Enumerate, close, and certify every small-instance radio assignment."""
    if not isinstance(max_heavy_tasks, int) or isinstance(max_heavy_tasks, bool):
        raise ValueError("max_heavy_tasks must be an integer")
    if max_heavy_tasks < 0:
        raise ValueError("max_heavy_tasks must be non-negative")
    if state_cap is not None and (
        not isinstance(state_cap, int) or isinstance(state_cap, bool) or state_cap < 1
    ):
        raise ValueError("state_cap must be a positive integer or None")
    if timeout_s is not None and (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0.0
    ):
        raise ValueError("timeout_s must be finite and positive or None")
    if audit is not None and not isinstance(audit, dict):
        raise TypeError("audit must be a dict or None")

    heavy = tuple(
        sorted(
            (task for task in snapshot.tasks if task.requires_offload),
            key=lambda task: task.task_id,
        )
    )
    if len(heavy) > max_heavy_tasks:
        raise ValueError(f"exact search limited to {max_heavy_tasks} heavy tasks")

    resource_blocks = snapshot.num_uavs * cfg.n_subcarriers
    fingerprint = semantic_fingerprint(
        snapshot,
        channel,
        cfg,
        ChannelBasis.PLANNING,
        downlink_rates_bps=downlink_rates_bps,
    )
    best_closed: ClosedExecutableSchedule | None = None
    best_objective: tuple[int, float, float] | None = None
    unique_closed: dict[tuple, ClosedExecutableSchedule] = {}
    raw_states_enumerated = 0
    cpu_overloaded_uav_ids: set[int] = set()
    wireless_feasible_states = 0
    full_service_wireless_state_exists = False
    full_service_secondary_objectives: set[tuple[float, float]] = set()
    complete = False
    started = time.monotonic()
    assignments = iter(_legal_raw_assignments(heavy, resource_blocks))
    while True:
        try:
            assignment = next(assignments)
        except StopIteration:
            complete = True
            break
        if state_cap is not None and raw_states_enumerated >= state_cap:
            break
        if (
            timeout_s is not None
            and raw_states_enumerated > 0
            and time.monotonic() - started >= timeout_s
        ):
            break

        raw_states_enumerated += 1
        raw = _raw_schedule(heavy, assignment, cfg.n_subcarriers)
        try:
            rated = rate_raw_assignment(
                snapshot,
                raw,
                channel,
                cfg,
                channel_basis=ChannelBasis.PLANNING,
                downlink_rates_bps=downlink_rates_bps,
            )
            if not all(record.sic_feasible for record in rated.rates):
                continue
            wireless_feasible_states += 1
            overloaded_uavs, raw_minimum_cpu_hz = _cpu_requirement_audit(
                snapshot,
                rated,
                cfg,
                downlink_rates_bps,
            )
            cpu_overloaded_uav_ids.update(overloaded_uavs)
            if set(rated.assignment.task_ids) == {
                task.task_id for task in heavy
            }:
                full_service_wireless_state_exists = True
                if raw_minimum_cpu_hz is not None:
                    full_service_secondary_objectives.add(
                        (
                            round(raw_minimum_cpu_hz, 6),
                            round(
                                aggregate_victim_link_exposure(rated.rates),
                                18,
                            ),
                        )
                    )
            closed = close_assignment(
                snapshot,
                rated,
                channel,
                cfg,
                downlink_rates_bps=downlink_rates_bps,
            )
        except (InfeasibleScheduleError, ScheduleValidationError):
            continue
        key = closed_semantic_key(closed)
        if key in unique_closed:
            continue
        unique_closed[key] = closed
        outcome = build_outcome_from_closed(snapshot, closed)
        objective = network_objective(outcome)
        if (
            best_closed is None
            or best_objective is None
            or objective > best_objective
            or (objective == best_objective and closed.stable_key() < best_closed.stable_key())
        ):
            best_closed = closed
            best_objective = objective

    if best_closed is None or best_objective is None:
        raise RuntimeError("exact search found no feasible schedule")
    best_bound, gap, gap_status = _certificate_bound(
        snapshot,
        best_objective,
        complete,
    )
    schedule_nontrivial = (
        complete
        and wireless_feasible_states >= 2
        and (
            not full_service_wireless_state_exists
            or len(full_service_secondary_objectives) > 1
        )
    )
    reasons: list[str] = []
    if cpu_overloaded_uav_ids:
        reasons.append("cpu_budget_overload")
    if schedule_nontrivial and not full_service_wireless_state_exists:
        reasons.append("no_full_service_wireless_state")
    if schedule_nontrivial and len(full_service_secondary_objectives) > 1:
        reasons.append("distinct_full_service_secondary_objectives")
    if not complete:
        reasons.append("incomplete_enumeration")
    if audit is not None:
        audit.clear()
        audit.update(
            {
                "cpu_overloaded_uav_ids": tuple(sorted(cpu_overloaded_uav_ids)),
                "wireless_feasible_states": wireless_feasible_states,
                "full_service_wireless_state_exists": (
                    full_service_wireless_state_exists
                ),
                "full_service_closed_state_exists": (
                    full_service_wireless_state_exists
                ),
                "distinct_full_service_secondary_objectives": len(
                    full_service_secondary_objectives
                ),
                "nontriviality_reason": ";".join(reasons) if reasons else "none",
            }
        )
    certificate = ExactScheduleCertificate(
        complete=complete,
        raw_states_enumerated=raw_states_enumerated,
        unique_closed_states=len(unique_closed),
        best_bound=best_bound,
        lexicographic_gap=gap,
        gap_status=gap_status,
        cpu_nontrivial=bool(cpu_overloaded_uav_ids),
        schedule_nontrivial=schedule_nontrivial,
        semantic_fingerprint=fingerprint,
    )
    return ExactScheduleResult(
        plan=freeze_plan(best_closed),
        objective=best_objective,
        certificate=certificate,
    )
