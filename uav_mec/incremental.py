"""Dependency-complete incremental evaluation of raw schedule moves."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from .closure import (
    ClosedExecutableSchedule,
    RawRateSnapshot,
    close_assignment,
    rate_raw_assignment,
)
from .domain import ChannelBasis, RateRecord, Schedule, ScheduleEntry, SlotSnapshot
from .evaluation import EvaluationConfig, compute_rate_records, semantic_fingerprint
from .model import NOMAChannel
from .schedule import normalize_roles, validate_schedule


@dataclass(frozen=True, slots=True)
class IncrementalDependencyClosure:
    """Auditable radio, role, association, and CPU dependency closure."""

    affected_subcarriers: tuple[int, ...]
    receiver_blocks: tuple[tuple[int, int], ...]
    role_task_ids: tuple[int, ...]
    reassigned_task_ids: tuple[int, ...]
    cpu_uavs: tuple[int, ...]
    cpu_task_ids: tuple[int, ...]


def _normalized_planning_schedule(
    snapshot: SlotSnapshot,
    schedule: Schedule,
    cfg: EvaluationConfig,
) -> Schedule:
    if not isinstance(schedule, Schedule):
        raise TypeError("schedule must be a Schedule")
    normalized = normalize_roles(
        schedule,
        snapshot,
        channel_basis=ChannelBasis.PLANNING,
    )
    validate_schedule(normalized, snapshot, cfg.n_subcarriers)
    return normalized


def _entries_by_block(
    schedule: Schedule,
) -> dict[tuple[int, int], tuple[ScheduleEntry, ...]]:
    grouped: dict[tuple[int, int], list[ScheduleEntry]] = defaultdict(list)
    for entry in schedule.entries:
        grouped[(entry.serving_uav, entry.subcarrier)].append(entry)
    return {
        block: tuple(
            sorted(
                entries,
                key=lambda entry: (entry.noma_role, entry.task_id),
            )
        )
        for block, entries in grouped.items()
    }


def dependency_closure(
    snapshot: SlotSnapshot,
    before: Schedule,
    candidate: Schedule,
    cfg: EvaluationConfig,
) -> IncrementalDependencyClosure:
    """Return every dependency that can change under one raw schedule move.

    A changed transmitter affects every UAV receiver on each source or target
    subcarrier.  Membership/order changes cover both versions of each touched
    NOMA group.  Rate- or association-affected tasks pull in their old and new
    MEC servers, and those servers pull in every task in their allocation
    ordering.
    """
    if not isinstance(snapshot, SlotSnapshot):
        raise TypeError("snapshot must be a SlotSnapshot")
    if not isinstance(cfg, EvaluationConfig):
        raise TypeError("cfg must be an EvaluationConfig")

    normalized_before = _normalized_planning_schedule(snapshot, before, cfg)
    normalized_candidate = _normalized_planning_schedule(snapshot, candidate, cfg)
    before_by_task = {
        entry.task_id: entry for entry in normalized_before.entries
    }
    candidate_by_task = {
        entry.task_id: entry for entry in normalized_candidate.entries
    }
    changed_task_ids = {
        task_id
        for task_id in before_by_task.keys() | candidate_by_task.keys()
        if before_by_task.get(task_id) != candidate_by_task.get(task_id)
    }

    affected_subcarriers: set[int] = set()
    for task_id in changed_task_ids:
        old = before_by_task.get(task_id)
        new = candidate_by_task.get(task_id)
        if old is not None:
            affected_subcarriers.add(old.subcarrier)
        if new is not None:
            affected_subcarriers.add(new.subcarrier)

    receiver_blocks = {
        (uav, subcarrier)
        for subcarrier in affected_subcarriers
        for uav in range(snapshot.num_uavs)
    }

    before_groups = _entries_by_block(normalized_before)
    candidate_groups = _entries_by_block(normalized_candidate)
    role_task_ids: set[int] = set()
    for block in before_groups.keys() | candidate_groups.keys():
        old_group = before_groups.get(block, ())
        new_group = candidate_groups.get(block, ())
        if old_group != new_group:
            role_task_ids.update(entry.task_id for entry in old_group)
            role_task_ids.update(entry.task_id for entry in new_group)

    reassigned_task_ids = {
        task_id
        for task_id in before_by_task.keys() & candidate_by_task.keys()
        if (
            before_by_task[task_id].serving_uav
            != candidate_by_task[task_id].serving_uav
        )
    }

    rate_task_ids = {
        entry.task_id
        for entry in (
            normalized_before.entries + normalized_candidate.entries
        )
        if entry.subcarrier in affected_subcarriers
    }
    cpu_uavs = {
        entry.serving_uav
        for task_id in rate_task_ids | reassigned_task_ids
        for entry in (
            before_by_task.get(task_id),
            candidate_by_task.get(task_id),
        )
        if entry is not None
    }
    cpu_task_ids = {
        entry.task_id
        for entry in (
            normalized_before.entries + normalized_candidate.entries
        )
        if entry.serving_uav in cpu_uavs
    }

    return IncrementalDependencyClosure(
        affected_subcarriers=tuple(sorted(affected_subcarriers)),
        receiver_blocks=tuple(sorted(receiver_blocks)),
        role_task_ids=tuple(sorted(role_task_ids)),
        reassigned_task_ids=tuple(sorted(reassigned_task_ids)),
        cpu_uavs=tuple(sorted(cpu_uavs)),
        cpu_task_ids=tuple(sorted(cpu_task_ids)),
    )


def _rate_record_matches_entry(
    record: RateRecord,
    entry: ScheduleEntry,
    channel_basis: ChannelBasis,
) -> bool:
    return (
        record.task_id == entry.task_id
        and record.serving_uav == entry.serving_uav
        and record.subcarrier == entry.subcarrier
        and record.channel_basis is channel_basis
    )


def evaluate_move_incremental(
    snapshot: SlotSnapshot,
    before: ClosedExecutableSchedule,
    candidate: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    *,
    downlink_rates_bps: Mapping[int, float] | None = None,
) -> ClosedExecutableSchedule:
    """Close ``candidate`` while reusing only dependency-independent rates.

    Reuse is permitted only when the previous raw-rate snapshot has the exact
    current semantic fingerprint.  Allocation and deletion-only closure still
    use the canonical full-policy path, so the returned object has precisely
    the same semantics and type as a full evaluation.
    """
    if not isinstance(before, ClosedExecutableSchedule):
        raise TypeError("before must be a ClosedExecutableSchedule")
    if before.rate_snapshot.channel_basis is not ChannelBasis.PLANNING:
        raise ValueError("incremental move evaluation requires the planning basis")

    normalized_candidate = _normalized_planning_schedule(snapshot, candidate, cfg)
    expected_fingerprint = semantic_fingerprint(
        snapshot,
        channel,
        cfg,
        ChannelBasis.PLANNING,
        downlink_rates_bps=downlink_rates_bps,
    )
    if (
        before.rate_snapshot.semantic_fingerprint != expected_fingerprint
        or before.rate_snapshot.assignment != before.schedule
    ):
        return close_assignment(
            snapshot,
            rate_raw_assignment(
                snapshot,
                normalized_candidate,
                channel,
                cfg,
                channel_basis=ChannelBasis.PLANNING,
                downlink_rates_bps=downlink_rates_bps,
            ),
            channel,
            cfg,
            downlink_rates_bps=downlink_rates_bps,
        )

    dependencies = dependency_closure(
        snapshot,
        before.schedule,
        normalized_candidate,
        cfg,
    )
    dependency_blocks = frozenset(dependencies.receiver_blocks)
    before_entry_by_task = {
        entry.task_id: entry for entry in before.schedule.entries
    }
    before_rate_by_task = {
        record.task_id: record for record in before.rate_snapshot.rates
    }

    reused: dict[int, RateRecord] = {}
    recompute_blocks = set(dependency_blocks)
    for entry in normalized_candidate.entries:
        block = (entry.serving_uav, entry.subcarrier)
        old_entry = before_entry_by_task.get(entry.task_id)
        old_rate = before_rate_by_task.get(entry.task_id)
        if (
            block not in dependency_blocks
            and old_entry == entry
            and old_rate is not None
            and _rate_record_matches_entry(
                old_rate,
                entry,
                ChannelBasis.PLANNING,
            )
        ):
            reused[entry.task_id] = old_rate
        else:
            recompute_blocks.add(block)

    recomputed = compute_rate_records(
        snapshot,
        normalized_candidate,
        channel,
        cfg,
        channel_basis=ChannelBasis.PLANNING,
        receiver_blocks=recompute_blocks,
    )
    merged = dict(reused)
    merged.update((record.task_id, record) for record in recomputed)
    candidate_task_ids = set(normalized_candidate.task_ids)
    if set(merged) != candidate_task_ids:
        return close_assignment(
            snapshot,
            rate_raw_assignment(
                snapshot,
                normalized_candidate,
                channel,
                cfg,
                channel_basis=ChannelBasis.PLANNING,
                downlink_rates_bps=downlink_rates_bps,
            ),
            channel,
            cfg,
            downlink_rates_bps=downlink_rates_bps,
        )

    rated = RawRateSnapshot(
        assignment=normalized_candidate,
        rates=tuple(merged[task_id] for task_id in sorted(merged)),
        channel_basis=ChannelBasis.PLANNING,
        semantic_fingerprint=expected_fingerprint,
    )
    return close_assignment(
        snapshot,
        rated,
        channel,
        cfg,
        downlink_rates_bps=downlink_rates_bps,
    )
