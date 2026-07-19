"""Deterministic schedule initialization and local-search building blocks."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .closure import (
    ClosedExecutableSchedule,
    close_assignment,
    rate_raw_assignment,
)
from .domain import ChannelBasis, Schedule, ScheduleEntry, SlotOutcome, SlotSnapshot
from .evaluation import (
    EvaluationConfig,
    InfeasibleScheduleError,
    build_outcome_from_closed,
    compute_rate_records,
)
from .incremental import evaluate_move_incremental
from .model import NOMAChannel
from .schedule import normalize_roles, validate_schedule


class InitializationError(RuntimeError):
    """Raised when the declared MILP initializer cannot produce a result."""


@dataclass(frozen=True, slots=True)
class InitializationResult:
    schedule: Schedule
    solver_status: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    uav: int
    task_ids: tuple[int, ...]
    benefit: int


def _close_planning_schedule(
    snapshot: SlotSnapshot,
    schedule: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> ClosedExecutableSchedule:
    return close_assignment(
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


def _candidate_schedule(
    candidate: _Candidate,
    subcarrier: int,
    snapshot: SlotSnapshot,
) -> Schedule:
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    raw = Schedule(
        tuple(
            ScheduleEntry(
                task_id,
                task_by_id[task_id].user_id,
                candidate.uav,
                subcarrier,
                "solo",
            )
            for task_id in candidate.task_ids
        )
    )
    return normalize_roles(raw, snapshot, channel_basis=ChannelBasis.PLANNING)


def _candidates_for_uav(
    snapshot: SlotSnapshot,
    uav: int,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> list[_Candidate]:
    row_by_task = {
        task.task_id: row for row, task in enumerate(snapshot.tasks)
    }
    planning_gains = snapshot.planning_channel_gains
    offload_tasks = [task for task in snapshot.tasks if task.requires_offload]
    assigned = sorted(
        (
            task
            for task in offload_tasks
            if int(
                np.asarray(
                    planning_gains[row_by_task[task.task_id]]
                ).argmax()
            )
            == uav
        ),
        key=lambda task: task.task_id,
    )
    local_count = sum(not task.requires_offload for task in snapshot.tasks)
    task_groups = [(task.task_id,) for task in assigned]
    task_groups.extend(
        tuple(task.task_id for task in pair)
        for pair in combinations(assigned, 2)
    )

    candidates: list[_Candidate] = []
    for task_ids in task_groups:
        ordered_ids = tuple(sorted(task_ids))
        trial = _Candidate(uav, ordered_ids, 0)
        schedule = _candidate_schedule(trial, 0, snapshot)
        try:
            closed = _close_planning_schedule(
                snapshot,
                schedule,
                channel,
                cfg,
            )
        except InfeasibleScheduleError:
            continue
        outcome = build_outcome_from_closed(snapshot, closed)
        benefit = max(outcome.completed_count - local_count, 0)
        if benefit > 0:
            candidates.append(_Candidate(uav, ordered_ids, benefit))
    return candidates


def _solve_uav_candidates(
    candidates: list[_Candidate],
    n_subcarriers: int,
) -> list[_Candidate]:
    if not candidates:
        return []

    task_ids = sorted(
        {task_id for candidate in candidates for task_id in candidate.task_ids}
    )
    rows = []
    upper_bounds = []
    for task_id in task_ids:
        rows.append(
            [float(task_id in candidate.task_ids) for candidate in candidates]
        )
        upper_bounds.append(1.0)
    rows.append([1.0] * len(candidates))
    upper_bounds.append(float(n_subcarriers))

    objective = np.array(
        [
            -float(candidate.benefit)
            - 1e-6 / (1 + sum(candidate.task_ids))
            for candidate in candidates
        ]
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(candidates)),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=LinearConstraint(
            np.asarray(rows),
            -np.inf,
            np.asarray(upper_bounds),
        ),
    )
    if not result.success:
        raise InitializationError(f"per-UAV MILP failed: {result.message}")
    return [
        candidate
        for candidate, chosen in zip(candidates, result.x, strict=True)
        if chosen > 0.5
    ]


def _drop_weak_user(schedule: Schedule, block: tuple[int, int]) -> Schedule:
    return Schedule(
        tuple(
            entry
            for entry in schedule.entries
            if (entry.serving_uav, entry.subcarrier) != block
            or entry.noma_role != "weak"
        )
    )


def repair_sic(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> Schedule:
    """Deterministically downgrade violating pairs until all SIC checks pass."""
    current = normalize_roles(schedule, snapshot, channel_basis=ChannelBasis.PLANNING)
    while current.entries:
        records = compute_rate_records(snapshot, current, channel, cfg, channel_basis=ChannelBasis.PLANNING)
        violating_ids = {
            record.task_id for record in records if not record.sic_feasible
        }
        if not violating_ids:
            return current
        violating_blocks = sorted(
            {
                (entry.serving_uav, entry.subcarrier)
                for entry in current.entries
                if entry.task_id in violating_ids
            }
        )
        current = normalize_roles(
            _drop_weak_user(current, violating_blocks[-1]),
            snapshot,
            channel_basis=ChannelBasis.PLANNING,
        )
    return current


def build_initial_schedule(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> InitializationResult:
    """Build a feasible strongest-UAV initial schedule using per-UAV MILPs."""
    entries: list[ScheduleEntry] = []
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    for uav in range(snapshot.num_uavs):
        selected = _solve_uav_candidates(
            _candidates_for_uav(snapshot, uav, channel, cfg),
            cfg.n_subcarriers,
        )
        selected.sort(key=lambda candidate: (-candidate.benefit, candidate.task_ids))
        for subcarrier, candidate in enumerate(selected):
            entries.extend(
                ScheduleEntry(
                    task_id,
                    task_by_id[task_id].user_id,
                    uav,
                    subcarrier,
                    "solo",
                )
                for task_id in candidate.task_ids
            )

    repaired = repair_sic(
        Schedule(tuple(entries)),
        snapshot,
        channel,
        cfg,
    )
    closed = _close_planning_schedule(snapshot, repaired, channel, cfg)
    return InitializationResult(closed.schedule, "optimal")


MoveType = Literal["replace", "position", "cross"]


@dataclass(frozen=True, slots=True)
class Neighbor:
    schedule: Schedule
    move_type: MoveType
    affected_blocks: tuple[tuple[int, int], ...]


def _with_task(
    entry: ScheduleEntry,
    task_id: int,
    user_id: int,
) -> ScheduleEntry:
    return ScheduleEntry(
        task_id,
        user_id,
        entry.serving_uav,
        entry.subcarrier,
        entry.noma_role,
    )


def _nearest_uav(snapshot: SlotSnapshot, task_id: int) -> int:
    row = next(
        row
        for row, task in enumerate(snapshot.tasks)
        if task.task_id == task_id
    )
    return int(np.asarray(snapshot.planning_channel_gains[row]).argmax())


def _normalized_neighbor(
    entries: list[ScheduleEntry],
    move_type: MoveType,
    affected: set[tuple[int, int]],
    snapshot: SlotSnapshot,
    n_subcarriers: int,
) -> Neighbor:
    schedule = normalize_roles(Schedule(tuple(entries)), snapshot, channel_basis=ChannelBasis.PLANNING)
    validate_schedule(schedule, snapshot, n_subcarriers)
    return Neighbor(schedule, move_type, tuple(sorted(affected)))


def _entry_at_block(
    entry: ScheduleEntry,
    block: tuple[int, int],
) -> ScheduleEntry:
    return ScheduleEntry(
        entry.task_id,
        entry.user_id,
        block[0],
        block[1],
        entry.noma_role,
    )


def _enumerate_neighbor_moves(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    n_subcarriers: int,
    allow_cross_uav: bool,
) -> tuple[Neighbor, ...]:
    """Return every legal raw action without pre-closure deduplication."""
    validate_schedule(schedule, snapshot, n_subcarriers)
    entries = list(schedule.entries)
    scheduled_ids = {entry.task_id for entry in entries}
    scheduled_users = {entry.user_id for entry in entries}
    unscheduled = tuple(
        sorted(
            (
                task
                for task in snapshot.tasks
                if task.requires_offload and task.task_id not in scheduled_ids
            ),
            key=lambda task: task.task_id,
        )
    )
    blocks = tuple(
        (uav, subcarrier)
        for uav in range(snapshot.num_uavs)
        for subcarrier in range(n_subcarriers)
    )
    occupancy = {
        block: sum(
            (entry.serving_uav, entry.subcarrier) == block
            for entry in entries
        )
        for block in blocks
    }
    neighbors: list[Neighbor] = []

    def append(
        changed: list[ScheduleEntry],
        move_type: MoveType,
        affected: set[tuple[int, int]],
    ) -> None:
        neighbors.append(
            _normalized_neighbor(
                changed,
                move_type,
                affected,
                snapshot,
                n_subcarriers,
            )
        )

    # Replace family: explicit removal, replacement, and insertion.  An
    # intra-only search gives a previously unscheduled task its deterministic
    # nearest-UAV association; the full search exposes every serving UAV.
    for index, old in enumerate(entries):
        append(
            entries[:index] + entries[index + 1 :],
            "replace",
            {(old.serving_uav, old.subcarrier)},
        )
        other_users = scheduled_users - {old.user_id}
        for task in unscheduled:
            if task.user_id in other_users:
                continue
            if (
                not allow_cross_uav
                and _nearest_uav(snapshot, task.task_id) != old.serving_uav
            ):
                continue
            changed = entries.copy()
            changed[index] = _with_task(old, task.task_id, task.user_id)
            append(
                changed,
                "replace",
                {(old.serving_uav, old.subcarrier)},
            )

    for task in unscheduled:
        if task.user_id in scheduled_users:
            continue
        nearest = _nearest_uav(snapshot, task.task_id)
        for block in blocks:
            if occupancy[block] >= 2:
                continue
            if not allow_cross_uav and block[0] != nearest:
                continue
            append(
                entries
                + [
                    ScheduleEntry(
                        task.task_id,
                        task.user_id,
                        block[0],
                        block[1],
                        "solo",
                    )
                ],
                "replace",
                {block},
            )

    # Position family: exchange two occupied positions or relocate one task to
    # a free position on the same UAV.
    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            first, second = entries[left], entries[right]
            first_block = (first.serving_uav, first.subcarrier)
            second_block = (second.serving_uav, second.subcarrier)
            if first.serving_uav != second.serving_uav or first_block == second_block:
                continue
            changed = entries.copy()
            changed[left] = _entry_at_block(first, second_block)
            changed[right] = _entry_at_block(second, first_block)
            append(changed, "position", {first_block, second_block})

    for index, entry in enumerate(entries):
        source = (entry.serving_uav, entry.subcarrier)
        for target in blocks:
            if (
                target == source
                or target[0] != entry.serving_uav
                or occupancy[target] >= 2
            ):
                continue
            changed = entries.copy()
            changed[index] = _entry_at_block(entry, target)
            append(changed, "position", {source, target})

    if allow_cross_uav:
        # CrossUAV family: exchange occupied positions or relocate into any
        # free position at another receiver UAV.
        for left in range(len(entries)):
            for right in range(left + 1, len(entries)):
                first, second = entries[left], entries[right]
                if first.serving_uav == second.serving_uav:
                    continue
                first_block = (first.serving_uav, first.subcarrier)
                second_block = (second.serving_uav, second.subcarrier)
                changed = entries.copy()
                changed[left] = _entry_at_block(first, second_block)
                changed[right] = _entry_at_block(second, first_block)
                append(changed, "cross", {first_block, second_block})

        for index, entry in enumerate(entries):
            source = (entry.serving_uav, entry.subcarrier)
            for target in blocks:
                if target[0] == entry.serving_uav or occupancy[target] >= 2:
                    continue
                changed = entries.copy()
                changed[index] = _entry_at_block(entry, target)
                append(changed, "cross", {source, target})

    return tuple(
        sorted(
            neighbors,
            key=lambda neighbor: (
                neighbor.move_type,
                neighbor.schedule.stable_key(),
                neighbor.affected_blocks,
            ),
        )
    )


def enumerate_raw_moves(
    snapshot: SlotSnapshot,
    schedule: Schedule,
    cfg: EvaluationConfig,
) -> tuple[Schedule, ...]:
    """Enumerate the complete Replace/Position/CrossUAV raw neighborhood.

    Distinct legal actions are intentionally retained even when they happen to
    produce the same raw schedule.  Search-state identity is defined only after
    rating and policy closure.
    """
    return tuple(
        neighbor.schedule
        for neighbor in _enumerate_neighbor_moves(
            schedule,
            snapshot,
            cfg.n_subcarriers,
            True,
        )
    )


def generate_neighbors(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    n_subcarriers: int,
    allow_cross_uav: bool,
) -> tuple[Neighbor, ...]:
    """Compatibility view of unique raw schedules in the configured scope."""
    ordered = _enumerate_neighbor_moves(
        schedule,
        snapshot,
        n_subcarriers,
        allow_cross_uav,
    )
    unique: dict[tuple, Neighbor] = {}
    for neighbor in ordered:
        unique.setdefault(neighbor.schedule.stable_key(), neighbor)
    return tuple(unique.values())


def _closed_candidate_key(closed: ClosedExecutableSchedule) -> tuple:
    """Bind closed-state identity to the complete Plan 1 semantics."""
    return (
        closed.stable_key(),
        closed.rate_snapshot.semantic_fingerprint,
    )


def close_and_deduplicate_moves(
    snapshot: SlotSnapshot,
    raw_moves: Iterable[Schedule],
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    *,
    channel_basis: ChannelBasis,
) -> tuple[ClosedExecutableSchedule, ...]:
    """Close every raw move and deduplicate only by its semantic closed key."""
    unique: dict[tuple, ClosedExecutableSchedule] = {}
    for raw_move in raw_moves:
        try:
            closed = close_assignment(
                snapshot,
                rate_raw_assignment(
                    snapshot,
                    raw_move,
                    channel,
                    cfg,
                    channel_basis=channel_basis,
                ),
                channel,
                cfg,
            )
        except InfeasibleScheduleError:
            continue
        unique.setdefault(_closed_candidate_key(closed), closed)
    return tuple(unique[key] for key in sorted(unique))


class Acceptance(str, Enum):
    GLOBAL = "global"
    BILATERAL = "bilateral"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    allow_cross_uav: bool = True
    acceptance: Acceptance = Acceptance.GLOBAL
    max_iterations: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.allow_cross_uav, bool):
            raise ValueError("allow_cross_uav must be bool")
        if not isinstance(self.acceptance, Acceptance):
            raise ValueError("acceptance must be an Acceptance value")
        if not isinstance(self.max_iterations, int) or isinstance(
            self.max_iterations, bool
        ):
            raise ValueError("max_iterations must be an integer")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")


@dataclass(frozen=True, slots=True)
class SearchEvent:
    move_type: MoveType
    accepted: bool
    reason: str
    before_objective: tuple[int, float, float]
    after_objective: tuple[int, float, float]
    schedule_key: tuple


@dataclass(frozen=True, slots=True)
class SearchResult:
    schedule: Schedule
    outcome: SlotOutcome
    events: tuple[SearchEvent, ...]
    iterations: int
    stop_reason: Literal["no_improving_move", "iteration_cap"]
    local_optimum_certified: bool
    raw_move_count: int
    unique_closed_candidate_count: int
    cache_hits: int
    accepted_moves: int
    closure_rounds: int

    @property
    def converged(self) -> bool:
        """Compatibility alias for code predating explicit certification."""
        return self.stop_reason == "no_improving_move"

    @property
    def termination_reason(self) -> str:
        """Compatibility label retained for legacy result writers."""
        if self.stop_reason == "iteration_cap":
            return "max_iterations"
        if self.local_optimum_certified:
            return "local_optimum"
        return "cycle_free_fixed_point"


def network_objective(outcome: SlotOutcome) -> tuple[int, float, float]:
    """Return the canonical network-level lexicographic objective."""
    total_mec_cpu_hz = math.fsum(
        decision.cpu_hz for decision in outcome.decisions
    )
    exposure_w = outcome.aggregate_victim_link_exposure_w
    if (
        not math.isfinite(total_mec_cpu_hz)
        or total_mec_cpu_hz < 0.0
        or not math.isfinite(exposure_w)
        or exposure_w < 0.0
    ):
        raise ValueError("network objective requires finite non-negative costs")
    return (
        outcome.completed_count,
        round(-total_mec_cpu_hz, 6),
        round(-exposure_w, 18),
    )


def _block_utility(
    schedule: Schedule,
    outcome: SlotOutcome,
    block: tuple[int, int],
) -> tuple[int, float]:
    task_ids = {
        entry.task_id
        for entry in schedule.entries
        if (entry.serving_uav, entry.subcarrier) == block
    }
    decisions = [
        decision
        for decision in outcome.decisions
        if decision.task_id in task_ids
    ]
    return (
        sum(decision.completed for decision in decisions),
        round(
            sum(
                decision.slack_s
                for decision in decisions
                if decision.completed
            ),
            12,
        ),
    )


def _bilateral_accepts(
    before_schedule: Schedule,
    before: SlotOutcome,
    neighbor: Neighbor,
    after: SlotOutcome,
) -> bool:
    comparisons = [
        (
            _block_utility(before_schedule, before, block),
            _block_utility(neighbor.schedule, after, block),
        )
        for block in neighbor.affected_blocks
    ]
    return (
        bool(comparisons)
        and all(new >= old for old, new in comparisons)
        and any(new > old for old, new in comparisons)
    )


def _changed_receiver_blocks(
    before: Schedule,
    after: Schedule,
) -> tuple[tuple[int, int], ...]:
    before_by_task = {entry.task_id: entry for entry in before.entries}
    after_by_task = {entry.task_id: entry for entry in after.entries}
    affected: set[tuple[int, int]] = set()
    for task_id in before_by_task.keys() | after_by_task.keys():
        old = before_by_task.get(task_id)
        new = after_by_task.get(task_id)
        if old == new:
            continue
        if old is not None:
            affected.add((old.serving_uav, old.subcarrier))
        if new is not None:
            affected.add((new.serving_uav, new.subcarrier))
    return tuple(sorted(affected))


def _validated_initial_closure(
    snapshot: SlotSnapshot,
    initial: Schedule | ClosedExecutableSchedule,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
) -> ClosedExecutableSchedule:
    if isinstance(initial, ClosedExecutableSchedule):
        if initial.rate_snapshot.channel_basis is not ChannelBasis.PLANNING:
            raise ValueError("search requires a planning-basis closed schedule")
        validated = close_assignment(
            snapshot,
            initial.rate_snapshot,
            channel,
            evaluation,
        )
        if (
            validated.stable_key() != initial.stable_key()
            or validated.rate_snapshot.rates != initial.rate_snapshot.rates
            or validated.allocation.decisions != initial.allocation.decisions
        ):
            raise ValueError("initial closed schedule is inconsistent")
        return initial
    if not isinstance(initial, Schedule):
        raise TypeError("initial must be a Schedule or ClosedExecutableSchedule")
    normalized = normalize_roles(
        initial,
        snapshot,
        channel_basis=ChannelBasis.PLANNING,
    )
    return _close_planning_schedule(snapshot, normalized, channel, evaluation)


def optimize_schedule(
    initial: Schedule | ClosedExecutableSchedule | SlotSnapshot,
    snapshot: SlotSnapshot | Schedule | ClosedExecutableSchedule,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
    search: SearchConfig,
) -> SearchResult:
    """Search closed executable states and explicitly certify termination.

    The historical ``(initial, snapshot, ...)`` argument order remains valid;
    the normative ``(snapshot, initial, ...)`` order is accepted as well.
    """
    if isinstance(initial, SlotSnapshot):
        initial, snapshot = snapshot, initial
    if not isinstance(snapshot, SlotSnapshot):
        raise TypeError("snapshot must be a SlotSnapshot")

    initial_closed = _validated_initial_closure(
        snapshot,
        initial,
        channel,
        evaluation,
    )
    current_closed = initial_closed
    current_outcome = build_outcome_from_closed(snapshot, current_closed)
    current_key = _closed_candidate_key(current_closed)

    events: list[SearchEvent] = []
    outcome_cache: dict[
        tuple,
        tuple[ClosedExecutableSchedule, SlotOutcome],
    ] = {current_key: (current_closed, current_outcome)}
    visited = {current_key}
    discovered_candidate_keys: set[tuple] = set()
    raw_move_count = 0
    cache_hits = 0
    accepted_moves = 0
    closure_rounds = initial_closed.closure_rounds
    iterations = 0
    stop_reason: Literal["no_improving_move", "iteration_cap"] = (
        "iteration_cap"
    )

    while iterations < search.max_iterations:
        iterations += 1
        accepted_this_iteration = False
        before_objective = network_objective(current_outcome)
        unique_this_iteration: set[tuple] = set()

        for raw_neighbor in _enumerate_neighbor_moves(
            current_closed.schedule,
            snapshot,
            evaluation.n_subcarriers,
            search.allow_cross_uav,
        ):
            raw_move_count += 1
            raw_schedule_key = raw_neighbor.schedule.stable_key()
            try:
                candidate_closed = evaluate_move_incremental(
                    snapshot,
                    current_closed,
                    raw_neighbor.schedule,
                    channel,
                    evaluation,
                )
            except InfeasibleScheduleError as exc:
                events.append(
                    SearchEvent(
                        raw_neighbor.move_type,
                        False,
                        str(exc),
                        before_objective,
                        before_objective,
                        raw_schedule_key,
                    )
                )
                continue

            closure_rounds += candidate_closed.closure_rounds
            candidate_key = _closed_candidate_key(candidate_closed)
            discovered_candidate_keys.add(candidate_key)
            if candidate_key in unique_this_iteration:
                cache_hits += 1
                continue
            unique_this_iteration.add(candidate_key)

            cached = outcome_cache.get(candidate_key)
            if cached is None:
                candidate_outcome = build_outcome_from_closed(
                    snapshot,
                    candidate_closed,
                )
                outcome_cache[candidate_key] = (
                    candidate_closed,
                    candidate_outcome,
                )
            else:
                cache_hits += 1
                candidate_closed, candidate_outcome = cached

            candidate_schedule = candidate_closed.schedule
            schedule_key = candidate_schedule.stable_key()
            affected_blocks = _changed_receiver_blocks(
                current_closed.schedule,
                candidate_schedule,
            )
            closed_neighbor = Neighbor(
                candidate_schedule,
                raw_neighbor.move_type,
                affected_blocks,
            )
            after_objective = network_objective(candidate_outcome)

            if search.acceptance is Acceptance.GLOBAL:
                accepted = after_objective > before_objective
                accepted_reason = "global strict improvement"
            elif candidate_key in visited:
                accepted = False
                accepted_reason = "visited closed schedule"
            else:
                accepted = _bilateral_accepts(
                    current_closed.schedule,
                    current_outcome,
                    closed_neighbor,
                    candidate_outcome,
                )
                accepted_reason = "bilateral Pareto improvement"

            events.append(
                SearchEvent(
                    raw_neighbor.move_type,
                    accepted,
                    (
                        accepted_reason
                        if accepted
                        else "acceptance criterion not met"
                    ),
                    before_objective,
                    after_objective,
                    schedule_key,
                )
            )
            if accepted:
                current_closed = candidate_closed
                current_outcome = candidate_outcome
                current_key = candidate_key
                visited.add(current_key)
                accepted_moves += 1
                accepted_this_iteration = True
                break

        if not accepted_this_iteration:
            stop_reason = "no_improving_move"
            break

    local_optimum_certified = (
        stop_reason == "no_improving_move"
        and search.acceptance is Acceptance.GLOBAL
    )
    return SearchResult(
        schedule=current_closed.schedule,
        outcome=current_outcome,
        events=tuple(events),
        iterations=iterations,
        stop_reason=stop_reason,
        local_optimum_certified=local_optimum_certified,
        raw_move_count=raw_move_count,
        unique_closed_candidate_count=len(discovered_candidate_keys),
        cache_hits=cache_hits,
        accepted_moves=accepted_moves,
        closure_rounds=closure_rounds,
    )
