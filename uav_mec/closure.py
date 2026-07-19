"""Canonical raw-radio-assignment to executable-schedule boundary."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from .domain import (
    ChannelBasis,
    FrozenSlotPlan,
    RateRecord,
    Schedule,
    SlotSnapshot,
    Task,
)
from .evaluation import (
    EvaluationConfig,
    InfeasibleScheduleError,
    compute_rate_records,
    semantic_fingerprint,
)
from .model import NOMAChannel
from .resource_allocation import (
    AllocationPolicy,
    AllocationResult,
    allocate_with_policy,
)
from .schedule import normalize_roles


class ClosureConvergenceError(RuntimeError):
    """Raised only when deletion-only admission closure fails to converge."""


@dataclass(frozen=True, slots=True)
class RawRateSnapshot:
    """Rated raw assignment bound to its complete semantic fingerprint."""

    assignment: Schedule
    rates: tuple[RateRecord, ...]
    channel_basis: ChannelBasis
    semantic_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rates", tuple(self.rates))

    def stable_key(self) -> tuple:
        return (
            self.assignment.stable_key(),
            self.channel_basis.value,
            self.semantic_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class ClosedExecutableSchedule:
    """Deletion-closed allocation whose stable key carries full semantics."""

    schedule: Schedule
    rate_snapshot: RawRateSnapshot
    allocation: AllocationResult
    closure_rounds: int
    allocation_policy: AllocationPolicy

    def stable_key(self) -> tuple:
        return (
            self.schedule.stable_key(),
            self.rate_snapshot.channel_basis.value,
            self.rate_snapshot.semantic_fingerprint,
            self.allocation_policy.value,
        )


def freeze_plan(closed: ClosedExecutableSchedule) -> FrozenSlotPlan:
    """Seal one planning-basis executable schedule for realized replay."""
    if not isinstance(closed, ClosedExecutableSchedule):
        raise TypeError("closed must be a ClosedExecutableSchedule")
    if closed.rate_snapshot.channel_basis is not ChannelBasis.PLANNING:
        raise ValueError("only a planning-basis closed schedule can be frozen")
    if closed.rate_snapshot.assignment != closed.schedule:
        raise ValueError("closed schedule and planning rate snapshot disagree")

    scheduled_ids = tuple(closed.schedule.task_ids)
    rate_ids = tuple(sorted(record.task_id for record in closed.rate_snapshot.rates))
    decision_ids = tuple(
        sorted(decision.task_id for decision in closed.allocation.decisions)
    )
    admitted_ids = tuple(sorted(closed.allocation.completed_task_ids))
    if not (
        scheduled_ids == rate_ids == decision_ids == admitted_ids
    ):
        raise ValueError(
            "closed schedule, rates, decisions, and admission must agree"
        )
    if any(
        record.channel_basis is not ChannelBasis.PLANNING
        for record in closed.rate_snapshot.rates
    ):
        raise ValueError("frozen planning rates must use the planning basis")
    if any(
        not decision.completed
        for decision in closed.allocation.decisions
    ):
        raise ValueError("a frozen closed allocation must be executable")

    return FrozenSlotPlan(
        schedule=closed.schedule,
        planning_rates=closed.rate_snapshot.rates,
        allocation=closed.allocation,
        semantic_fingerprint=closed.rate_snapshot.semantic_fingerprint,
    )


def rate_raw_assignment(
    snapshot: SlotSnapshot,
    assignment: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    *,
    channel_basis: ChannelBasis,
    downlink_rates_bps: Mapping[int, float] | None = None,
) -> RawRateSnapshot:
    """Normalize and rate one raw assignment without applying admission."""
    normalized = normalize_roles(
        assignment,
        snapshot,
        channel_basis=channel_basis,
    )
    rates = compute_rate_records(
        snapshot,
        normalized,
        channel,
        cfg,
        channel_basis=channel_basis,
    )
    return RawRateSnapshot(
        normalized,
        rates,
        channel_basis,
        semantic_fingerprint(
            snapshot,
            channel,
            cfg,
            channel_basis,
            downlink_rates_bps=downlink_rates_bps,
        ),
    )


def _validate_rate_snapshot(rated: RawRateSnapshot) -> None:
    entry_by_id = {
        entry.task_id: entry for entry in rated.assignment.entries
    }
    rate_by_id = {record.task_id: record for record in rated.rates}
    if len(entry_by_id) != len(rated.assignment.entries):
        raise ValueError("raw assignment contains duplicate task IDs")
    if len(rate_by_id) != len(rated.rates):
        raise ValueError("raw rate snapshot contains duplicate task IDs")
    if set(entry_by_id) != set(rate_by_id):
        raise ValueError(
            "raw assignment and rate records must have identical task IDs"
        )
    for task_id, record in rate_by_id.items():
        entry = entry_by_id[task_id]
        if record.channel_basis is not rated.channel_basis:
            raise ValueError("raw rate record has the wrong channel basis")
        if (
            record.serving_uav != entry.serving_uav
            or record.subcarrier != entry.subcarrier
        ):
            raise ValueError("raw rate record disagrees with its assignment")
    sic_violations = sorted(
        record.task_id for record in rated.rates if not record.sic_feasible
    )
    if sic_violations:
        raise InfeasibleScheduleError(
            f"SIC threshold violated by tasks {sic_violations}"
        )


def _allocate_per_serving_uav(
    snapshot: SlotSnapshot,
    rated: RawRateSnapshot,
    cfg: EvaluationConfig,
    downlink_rates_bps: Mapping[int, float] | None,
) -> AllocationResult:
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    rate_by_id = {
        record.task_id: record.rate_bps for record in rated.rates
    }
    task_ids_by_uav: dict[int, list[int]] = defaultdict(list)
    for entry in rated.assignment.entries:
        task_ids_by_uav[entry.serving_uav].append(entry.task_id)

    decisions = []
    completed_task_ids = []
    cpu_totals = []
    for uav in range(snapshot.num_uavs):
        task_ids = tuple(sorted(task_ids_by_uav.get(uav, ())))
        tasks: tuple[Task, ...] = tuple(task_by_id[task_id] for task_id in task_ids)
        allocation = allocate_with_policy(
            cfg.allocation_policy,
            tasks,
            rate_by_id,
            cfg.cpu_per_uav_hz,
            downlink_rates_bps,
        )
        decisions.extend(allocation.decisions)
        completed_task_ids.extend(allocation.completed_task_ids)
        cpu_totals.append(allocation.total_cpu_hz)

    try:
        total_cpu_hz = math.fsum(cpu_totals)
    except OverflowError as exc:
        raise ValueError("aggregate MEC allocation is not representable") from exc
    return AllocationResult(
        tuple(sorted(decisions, key=lambda decision: decision.task_id)),
        tuple(sorted(completed_task_ids)),
        total_cpu_hz,
    )


def close_assignment(
    snapshot: SlotSnapshot,
    rated: RawRateSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    *,
    downlink_rates_bps: Mapping[int, float] | None = None,
) -> ClosedExecutableSchedule:
    """Delete unadmitted entries and re-rate until every entry is executable."""
    if not isinstance(rated, RawRateSnapshot):
        raise TypeError("rated must be a RawRateSnapshot")
    expected_fingerprint = semantic_fingerprint(
        snapshot,
        channel,
        cfg,
        rated.channel_basis,
        downlink_rates_bps=downlink_rates_bps,
    )
    if rated.semantic_fingerprint != expected_fingerprint:
        raise ValueError("raw rate snapshot semantic fingerprint mismatch")

    original_task_ids = frozenset(rated.assignment.task_ids)
    maximum_rounds = len(rated.assignment.entries)
    current = rated
    closure_rounds = 0
    while True:
        _validate_rate_snapshot(current)
        allocation = _allocate_per_serving_uav(
            snapshot,
            current,
            cfg,
            downlink_rates_bps,
        )
        scheduled_ids = frozenset(current.assignment.task_ids)
        admitted_ids = frozenset(allocation.completed_task_ids)
        if not admitted_ids <= scheduled_ids:
            raise ClosureConvergenceError(
                "allocation inserted a task outside the raw assignment"
            )
        if scheduled_ids == admitted_ids:
            if not scheduled_ids <= original_task_ids:
                raise ClosureConvergenceError(
                    "closure inserted a task outside the original assignment"
                )
            return ClosedExecutableSchedule(
                current.assignment,
                current,
                allocation,
                closure_rounds,
                cfg.allocation_policy,
            )

        if closure_rounds >= maximum_rounds or not admitted_ids < scheduled_ids:
            raise ClosureConvergenceError(
                "deletion-only admission closure failed to make progress"
            )
        reduced = Schedule(
            tuple(
                entry
                for entry in current.assignment.entries
                if entry.task_id in admitted_ids
            )
        )
        closure_rounds += 1
        current = rate_raw_assignment(
            snapshot,
            reduced,
            channel,
            cfg,
            channel_basis=current.channel_basis,
            downlink_rates_bps=downlink_rates_bps,
        )
