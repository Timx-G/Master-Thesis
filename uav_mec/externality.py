"""Victim-link exposure accounting and auditable move certificates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .domain import AllocationDecision, ChannelBasis, RateRecord, ScheduleEntry


ReceiverBlock = tuple[int, int]


@dataclass(frozen=True, slots=True)
class BlockExposureDelta:
    receiver_uav: int
    subcarrier: int
    victim_links_before: int
    victim_links_after: int
    exposure_before_w: float
    exposure_after_w: float
    exposure_delta_w: float


@dataclass(frozen=True, slots=True)
class TaskImpact:
    task_id: int
    source_block: ReceiverBlock | None
    target_block: ReceiverBlock | None
    role_before: str | None
    role_after: str | None
    planned_rate_before_bps: float | None
    planned_rate_after_bps: float | None
    planned_sinr_before: float | None
    planned_sinr_after: float | None
    planned_sic_before: bool | None
    planned_sic_after: bool | None
    realized_rate_before_bps: float | None
    realized_rate_after_bps: float | None
    realized_sinr_before: float | None
    realized_sinr_after: float | None
    realized_sic_before: bool | None
    realized_sic_after: bool | None
    minimum_cpu_before_hz: float | None
    minimum_cpu_after_hz: float | None
    allocated_cpu_before_hz: float
    allocated_cpu_after_hz: float
    admitted_before: bool
    admitted_after: bool
    completed_before: bool
    completed_after: bool
    third_party: bool
    affected_receiver_blocks: tuple[ReceiverBlock, ...]


@dataclass(frozen=True, slots=True)
class ExternalityCertificate:
    completion_delta: int
    admitted_cpu_delta_hz: float
    actual_allocated_cpu_delta_hz: float
    exposure_delta_w: float
    block_deltas: tuple[BlockExposureDelta, ...]
    task_impacts: tuple[TaskImpact, ...]
    affected_receiver_blocks: tuple[ReceiverBlock, ...]


@dataclass(frozen=True, slots=True)
class _BlockExposure:
    victim_links: int
    exposure_w: float


def _finite_float(value: object, *, field: str) -> float:
    """Return one finite scalar or reject the certificate input."""
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{field} must be finite")
    return scalar


def _finite_sum(values: Iterable[float], *, field: str) -> float:
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(total):
        raise ValueError(f"{field} must be finite")
    return total


def aggregate_victim_link_exposure(rates: Iterable[RateRecord]) -> float:
    """Sum interference once for every victim task link.

    A two-user NOMA block contributes two records because both decoded task
    links experience the receiver's cross-UAV interference.  The records are
    therefore summed directly; block occupancy must never be applied again.
    """
    return _finite_sum(
        (
            _finite_float(
                record.cross_interference_w,
                field=f"rate[{record.task_id}].cross_interference_w",
            )
            for record in rates
        ),
        field="aggregate victim-link exposure",
    )


def _validate_rate_records(
    rates: Sequence[RateRecord],
    *,
    label: str,
    expected_basis: ChannelBasis,
) -> None:
    _index_by_task_id(rates, label=label)
    for record in rates:
        if record.channel_basis is not expected_basis:
            raise ValueError(
                f"{label} rate {record.task_id} must use the "
                f"{expected_basis.value} channel basis"
            )
        _finite_float(record.rate_bps, field=f"{label} rate_bps")
        _finite_float(record.sinr, field=f"{label} sinr")
        _finite_float(
            record.cross_interference_w,
            field=f"{label} cross_interference_w",
        )
        for layer in record.sic_layers:
            _finite_float(layer.sinr, field=f"{label} SIC sinr")
            _finite_float(layer.threshold, field=f"{label} SIC threshold")


def _validate_decisions(
    decisions: Sequence[AllocationDecision],
    *,
    label: str,
) -> None:
    _index_by_task_id(decisions, label=label)
    for decision in decisions:
        _finite_float(decision.alpha, field=f"{label} alpha")
        _finite_float(decision.cpu_hz, field=f"{label} cpu_hz")
        _finite_float(decision.slack_s, field=f"{label} slack_s")
        if decision.delay_s is not None:
            _finite_float(decision.delay_s, field=f"{label} delay_s")


def _index_by_task_id(rows: Sequence[Any], *, label: str) -> dict[int, Any]:
    indexed: dict[int, Any] = {}
    for row in rows:
        task_id = row.task_id
        if task_id in indexed:
            raise ValueError(f"{label} contains duplicate task_id {task_id}")
        indexed[task_id] = row
    return indexed


def _task_id_set(values: Sequence[int], *, label: str) -> set[int]:
    task_ids = set(values)
    if len(task_ids) != len(values):
        raise ValueError(f"{label} contains duplicate task IDs")
    return task_ids


def _block_exposure(rates: Sequence[RateRecord]) -> dict[ReceiverBlock, _BlockExposure]:
    grouped: dict[ReceiverBlock, list[float]] = {}
    for record in rates:
        block = (record.serving_uav, record.subcarrier)
        grouped.setdefault(block, []).append(
            _finite_float(
                record.cross_interference_w,
                field=f"rate[{record.task_id}].cross_interference_w",
            )
        )
    return {
        block: _BlockExposure(
            victim_links=len(exposures),
            exposure_w=_finite_sum(
                exposures,
                field=f"receiver block {block} exposure",
            ),
        )
        for block, exposures in grouped.items()
    }


def _delta_row(
    block: ReceiverBlock,
    before: _BlockExposure | None,
    after: _BlockExposure | None,
) -> BlockExposureDelta:
    before = before or _BlockExposure(0, 0.0)
    after = after or _BlockExposure(0, 0.0)
    delta = _finite_float(
        after.exposure_w - before.exposure_w,
        field=f"receiver block {block} exposure delta",
    )
    return BlockExposureDelta(
        receiver_uav=block[0],
        subcarrier=block[1],
        victim_links_before=before.victim_links,
        victim_links_after=after.victim_links,
        exposure_before_w=before.exposure_w,
        exposure_after_w=after.exposure_w,
        exposure_delta_w=delta,
    )


def _entry_block(entry: ScheduleEntry | None) -> ReceiverBlock | None:
    if entry is None:
        return None
    return (entry.serving_uav, entry.subcarrier)


def _rate_scalar(record: RateRecord | None, field: str) -> float | None:
    if record is None:
        return None
    return float(getattr(record, field))


def _rate_sic(record: RateRecord | None) -> bool | None:
    return None if record is None else bool(record.sic_feasible)


def _decision_cpu(decision: AllocationDecision | None) -> float | None:
    return None if decision is None else float(decision.cpu_hz)


def _allocated_cpu(
    decision: AllocationDecision | None,
    *,
    admitted: bool,
) -> float:
    # Missing and non-admitted decisions consume no actual MEC budget even if
    # a policy retained a positive requested/minimum CPU value for auditing.
    if decision is None or not admitted:
        return 0.0
    return float(decision.cpu_hz)


def _task_impacts(
    before: Any,
    after: Any,
    realized_before: Any | None,
    realized_after: Any | None,
    affected_receiver_blocks: tuple[ReceiverBlock, ...],
) -> tuple[TaskImpact, ...]:
    entries_before = _index_by_task_id(
        tuple(before.schedule.entries),
        label="before schedule",
    )
    entries_after = _index_by_task_id(
        tuple(after.schedule.entries),
        label="after schedule",
    )
    planned_rates_before = _index_by_task_id(
        tuple(before.rate_snapshot.rates),
        label="before planning rates",
    )
    planned_rates_after = _index_by_task_id(
        tuple(after.rate_snapshot.rates),
        label="after planning rates",
    )
    minimum_decisions_before = _index_by_task_id(
        tuple(before.allocation.decisions),
        label="before allocation decisions",
    )
    minimum_decisions_after = _index_by_task_id(
        tuple(after.allocation.decisions),
        label="after allocation decisions",
    )
    admitted_before_ids = _task_id_set(
        tuple(before.allocation.completed_task_ids),
        label="before admitted tasks",
    )
    admitted_after_ids = _task_id_set(
        tuple(after.allocation.completed_task_ids),
        label="after admitted tasks",
    )

    if realized_before is None:
        realized_rates_before: dict[int, RateRecord] = {}
        realized_rates_after: dict[int, RateRecord] = {}
        actual_decisions_before = minimum_decisions_before
        actual_decisions_after = minimum_decisions_after
        completed_before_ids = admitted_before_ids
        completed_after_ids = admitted_after_ids
    else:
        realized_rates_before = _index_by_task_id(
            tuple(realized_before.rates),
            label="before realized rates",
        )
        realized_rates_after = _index_by_task_id(
            tuple(realized_after.rates),
            label="after realized rates",
        )
        actual_decisions_before = _index_by_task_id(
            tuple(realized_before.decisions),
            label="before realized decisions",
        )
        actual_decisions_after = _index_by_task_id(
            tuple(realized_after.decisions),
            label="after realized decisions",
        )
        completed_before_ids = _task_id_set(
            tuple(realized_before.completed_task_ids),
            label="before completed tasks",
        )
        completed_after_ids = _task_id_set(
            tuple(realized_after.completed_task_ids),
            label="after completed tasks",
        )

    task_ids = tuple(
        sorted(
            set(entries_before)
            | set(entries_after)
            | set(planned_rates_before)
            | set(planned_rates_after)
            | set(minimum_decisions_before)
            | set(minimum_decisions_after)
            | set(realized_rates_before)
            | set(realized_rates_after)
            | set(actual_decisions_before)
            | set(actual_decisions_after)
            | admitted_before_ids
            | admitted_after_ids
            | completed_before_ids
            | completed_after_ids
        )
    )

    impacts: list[TaskImpact] = []
    for task_id in task_ids:
        entry_before = entries_before.get(task_id)
        entry_after = entries_after.get(task_id)
        planned_before = planned_rates_before.get(task_id)
        planned_after = planned_rates_after.get(task_id)
        realized_rate_before = realized_rates_before.get(task_id)
        realized_rate_after = realized_rates_after.get(task_id)
        minimum_before = minimum_decisions_before.get(task_id)
        minimum_after = minimum_decisions_after.get(task_id)
        actual_before = actual_decisions_before.get(task_id)
        actual_after = actual_decisions_after.get(task_id)

        source_block = _entry_block(entry_before)
        target_block = _entry_block(entry_after)
        role_before = None if entry_before is None else entry_before.noma_role
        role_after = None if entry_after is None else entry_after.noma_role
        admitted_before = task_id in admitted_before_ids
        admitted_after = task_id in admitted_after_ids
        completed_before = task_id in completed_before_ids
        completed_after = task_id in completed_after_ids
        allocated_cpu_before_hz = _allocated_cpu(
            actual_before,
            admitted=completed_before,
        )
        allocated_cpu_after_hz = _allocated_cpu(
            actual_after,
            admitted=completed_after,
        )

        changed_fields_before = (
            _rate_scalar(planned_before, "rate_bps"),
            _rate_scalar(planned_before, "sinr"),
            _rate_sic(planned_before),
            _rate_scalar(realized_rate_before, "rate_bps"),
            _rate_scalar(realized_rate_before, "sinr"),
            _rate_sic(realized_rate_before),
            _decision_cpu(minimum_before),
            allocated_cpu_before_hz,
            admitted_before,
            completed_before,
        )
        changed_fields_after = (
            _rate_scalar(planned_after, "rate_bps"),
            _rate_scalar(planned_after, "sinr"),
            _rate_sic(planned_after),
            _rate_scalar(realized_rate_after, "rate_bps"),
            _rate_scalar(realized_rate_after, "sinr"),
            _rate_sic(realized_rate_after),
            _decision_cpu(minimum_after),
            allocated_cpu_after_hz,
            admitted_after,
            completed_after,
        )
        third_party = (
            source_block == target_block
            and role_before == role_after
            and changed_fields_before != changed_fields_after
        )

        impacts.append(
            TaskImpact(
                task_id=task_id,
                source_block=source_block,
                target_block=target_block,
                role_before=role_before,
                role_after=role_after,
                planned_rate_before_bps=_rate_scalar(planned_before, "rate_bps"),
                planned_rate_after_bps=_rate_scalar(planned_after, "rate_bps"),
                planned_sinr_before=_rate_scalar(planned_before, "sinr"),
                planned_sinr_after=_rate_scalar(planned_after, "sinr"),
                planned_sic_before=_rate_sic(planned_before),
                planned_sic_after=_rate_sic(planned_after),
                realized_rate_before_bps=_rate_scalar(
                    realized_rate_before,
                    "rate_bps",
                ),
                realized_rate_after_bps=_rate_scalar(
                    realized_rate_after,
                    "rate_bps",
                ),
                realized_sinr_before=_rate_scalar(realized_rate_before, "sinr"),
                realized_sinr_after=_rate_scalar(realized_rate_after, "sinr"),
                realized_sic_before=_rate_sic(realized_rate_before),
                realized_sic_after=_rate_sic(realized_rate_after),
                minimum_cpu_before_hz=_decision_cpu(minimum_before),
                minimum_cpu_after_hz=_decision_cpu(minimum_after),
                allocated_cpu_before_hz=allocated_cpu_before_hz,
                allocated_cpu_after_hz=allocated_cpu_after_hz,
                admitted_before=admitted_before,
                admitted_after=admitted_after,
                completed_before=completed_before,
                completed_after=completed_after,
                third_party=third_party,
                affected_receiver_blocks=affected_receiver_blocks,
            )
        )
    return tuple(impacts)


def _validate_side(closed: Any, *, label: str) -> None:
    rates = tuple(closed.rate_snapshot.rates)
    decisions = tuple(closed.allocation.decisions)
    _validate_rate_records(
        rates,
        label=f"{label} planning",
        expected_basis=ChannelBasis.PLANNING,
    )
    _validate_decisions(decisions, label=f"{label} allocation")
    _finite_float(
        closed.allocation.total_cpu_hz,
        field=f"{label} admitted CPU total",
    )
    _index_by_task_id(tuple(closed.schedule.entries), label=f"{label} schedule")
    _task_id_set(
        tuple(closed.allocation.completed_task_ids),
        label=f"{label} admitted tasks",
    )


def _validate_realized_side(realized: Any, *, label: str) -> None:
    _validate_rate_records(
        tuple(realized.rates),
        label=f"{label} realized",
        expected_basis=ChannelBasis.REALIZED,
    )
    _validate_decisions(
        tuple(realized.decisions),
        label=f"{label} realized allocation",
    )
    _task_id_set(
        tuple(realized.completed_task_ids),
        label=f"{label} completed tasks",
    )


def externality_certificate(
    before: Any,
    after: Any,
    *,
    realized_before: Any | None = None,
    realized_after: Any | None = None,
) -> ExternalityCertificate:
    """Build a completion/CPU/exposure-conserving move certificate."""
    if (realized_before is None) != (realized_after is None):
        raise ValueError("both realized outcomes are required for a result audit")

    _validate_side(before, label="before")
    _validate_side(after, label="after")
    if realized_before is not None:
        _validate_realized_side(realized_before, label="before")
        _validate_realized_side(realized_after, label="after")

    before_rates = tuple(before.rate_snapshot.rates)
    after_rates = tuple(after.rate_snapshot.rates)
    before_blocks = _block_exposure(before_rates)
    after_blocks = _block_exposure(after_rates)
    affected_receiver_blocks = tuple(
        sorted(before_blocks.keys() | after_blocks.keys())
    )
    block_deltas = tuple(
        _delta_row(
            block,
            before_blocks.get(block),
            after_blocks.get(block),
        )
        for block in affected_receiver_blocks
    )

    exposure_delta_w = _finite_float(
        aggregate_victim_link_exposure(after_rates)
        - aggregate_victim_link_exposure(before_rates),
        field="victim-link exposure delta",
    )
    block_delta_w = _finite_sum(
        (row.exposure_delta_w for row in block_deltas),
        field="block exposure delta",
    )
    if not math.isclose(
        block_delta_w,
        exposure_delta_w,
        rel_tol=1e-12,
        abs_tol=1e-18,
    ):
        raise ValueError(
            "externality certificate does not conserve victim-link exposure"
        )

    task_impacts = _task_impacts(
        before,
        after,
        realized_before,
        realized_after,
        affected_receiver_blocks,
    )
    actual_allocated_cpu_delta_hz = _finite_sum(
        (
            impact.allocated_cpu_after_hz - impact.allocated_cpu_before_hz
            for impact in task_impacts
        ),
        field="actual allocated CPU delta",
    )
    independently_summed_cpu_delta_hz = _finite_float(
        _finite_sum(
            (impact.allocated_cpu_after_hz for impact in task_impacts),
            field="after actual allocated CPU",
        )
        - _finite_sum(
            (impact.allocated_cpu_before_hz for impact in task_impacts),
            field="before actual allocated CPU",
        ),
        field="independent actual allocated CPU delta",
    )
    if not math.isclose(
        actual_allocated_cpu_delta_hz,
        independently_summed_cpu_delta_hz,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "externality certificate does not conserve per-task allocated CPU"
        )

    if realized_before is None:
        completed_before_ids = set(before.allocation.completed_task_ids)
        completed_after_ids = set(after.allocation.completed_task_ids)
    else:
        completed_before_ids = set(realized_before.completed_task_ids)
        completed_after_ids = set(realized_after.completed_task_ids)
    completion_delta = len(completed_after_ids) - len(completed_before_ids)
    per_task_completion_delta = sum(
        int(impact.completed_after) - int(impact.completed_before)
        for impact in task_impacts
    )
    if completion_delta != per_task_completion_delta:
        raise ValueError(
            "externality certificate does not conserve per-task completion"
        )

    admitted_cpu_delta_hz = _finite_float(
        float(after.allocation.total_cpu_hz)
        - float(before.allocation.total_cpu_hz),
        field="admitted CPU delta",
    )
    return ExternalityCertificate(
        completion_delta=completion_delta,
        admitted_cpu_delta_hz=admitted_cpu_delta_hz,
        actual_allocated_cpu_delta_hz=actual_allocated_cpu_delta_hz,
        exposure_delta_w=exposure_delta_w,
        block_deltas=block_deltas,
        task_impacts=task_impacts,
        affected_receiver_blocks=affected_receiver_blocks,
    )
