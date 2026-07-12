"""Deterministic schedule initialization and local-search building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .domain import Schedule, ScheduleEntry, SlotOutcome, SlotSnapshot
from .evaluation import (
    EvaluationConfig,
    InfeasibleScheduleError,
    compute_rate_records,
    evaluate_schedule,
)
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
    return normalize_roles(raw, snapshot)


def _candidates_for_uav(
    snapshot: SlotSnapshot,
    uav: int,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> list[_Candidate]:
    row_by_task = {
        task.task_id: row for row, task in enumerate(snapshot.tasks)
    }
    offload_tasks = [task for task in snapshot.tasks if task.requires_offload]
    assigned = sorted(
        (
            task
            for task in offload_tasks
            if int(
                np.asarray(
                    snapshot.channel_gains[row_by_task[task.task_id]]
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
            outcome = evaluate_schedule(snapshot, schedule, channel, cfg)
        except InfeasibleScheduleError:
            continue
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
    current = normalize_roles(schedule, snapshot)
    while current.entries:
        records = compute_rate_records(snapshot, current, channel, cfg)
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
    evaluate_schedule(snapshot, repaired, channel, cfg)
    return InitializationResult(repaired, "optimal")


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
    return int(np.asarray(snapshot.channel_gains[row]).argmax())


def _normalized_neighbor(
    entries: list[ScheduleEntry],
    move_type: MoveType,
    affected: set[tuple[int, int]],
    snapshot: SlotSnapshot,
    n_subcarriers: int,
) -> Neighbor:
    schedule = normalize_roles(Schedule(tuple(entries)), snapshot)
    validate_schedule(schedule, snapshot, n_subcarriers)
    return Neighbor(schedule, move_type, tuple(sorted(affected)))


def generate_neighbors(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    n_subcarriers: int,
    allow_cross_uav: bool,
) -> tuple[Neighbor, ...]:
    """Enumerate canonical Replace, Position, and optional CrossUAV moves."""
    validate_schedule(schedule, snapshot, n_subcarriers)
    scheduled_ids = {entry.task_id for entry in schedule.entries}
    unscheduled = [
        task
        for task in snapshot.tasks
        if task.requires_offload and task.task_id not in scheduled_ids
    ]
    neighbors: list[Neighbor] = []
    entries = list(schedule.entries)

    for index, old in enumerate(entries):
        for task in unscheduled:
            if _nearest_uav(snapshot, task.task_id) != old.serving_uav:
                continue
            changed = entries.copy()
            changed[index] = _with_task(old, task.task_id, task.user_id)
            neighbors.append(
                _normalized_neighbor(
                    changed,
                    "replace",
                    {(old.serving_uav, old.subcarrier)},
                    snapshot,
                    n_subcarriers,
                )
            )

    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            first, second = entries[left], entries[right]
            if (
                first.serving_uav != second.serving_uav
                or first.subcarrier == second.subcarrier
            ):
                continue
            changed = entries.copy()
            changed[left] = _with_task(
                first,
                second.task_id,
                second.user_id,
            )
            changed[right] = _with_task(
                second,
                first.task_id,
                first.user_id,
            )
            neighbors.append(
                _normalized_neighbor(
                    changed,
                    "position",
                    {
                        (first.serving_uav, first.subcarrier),
                        (second.serving_uav, second.subcarrier),
                    },
                    snapshot,
                    n_subcarriers,
                )
            )

    if allow_cross_uav:
        for left in range(len(entries)):
            for right in range(left + 1, len(entries)):
                first, second = entries[left], entries[right]
                if first.serving_uav == second.serving_uav:
                    continue
                changed = entries.copy()
                changed[left] = _with_task(
                    first,
                    second.task_id,
                    second.user_id,
                )
                changed[right] = _with_task(
                    second,
                    first.task_id,
                    first.user_id,
                )
                neighbors.append(
                    _normalized_neighbor(
                        changed,
                        "cross",
                        {
                            (first.serving_uav, first.subcarrier),
                            (second.serving_uav, second.subcarrier),
                        },
                        snapshot,
                        n_subcarriers,
                    )
                )

    ordered = sorted(
        neighbors,
        key=lambda neighbor: (
            neighbor.move_type,
            neighbor.schedule.stable_key(),
        ),
    )
    unique: dict[tuple, Neighbor] = {}
    for neighbor in ordered:
        unique.setdefault(neighbor.schedule.stable_key(), neighbor)
    return tuple(unique.values())


class Acceptance(str, Enum):
    GLOBAL = "global"
    BILATERAL = "bilateral"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    allow_cross_uav: bool
    acceptance: Acceptance
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
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


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
    converged: bool
    termination_reason: str


def network_objective(outcome: SlotOutcome) -> tuple[int, float, float]:
    """Return the canonical network-level lexicographic objective."""
    return (
        outcome.completed_count,
        round(outcome.total_slack_s, 12),
        round(-outcome.total_cross_interference_w, 18),
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


def optimize_schedule(
    initial: Schedule,
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
    search: SearchConfig,
) -> SearchResult:
    """Run deterministic first-improvement search over feasible schedules."""
    current = normalize_roles(initial, snapshot)
    current_outcome = evaluate_schedule(
        snapshot,
        current,
        channel,
        evaluation,
    )
    events: list[SearchEvent] = []
    visited = {current.stable_key()}
    iterations = 0
    improved = False

    while iterations < search.max_iterations:
        iterations += 1
        improved = False
        for neighbor in generate_neighbors(
            current,
            snapshot,
            evaluation.n_subcarriers,
            search.allow_cross_uav,
        ):
            before_objective = network_objective(current_outcome)
            schedule_key = neighbor.schedule.stable_key()
            if schedule_key in visited:
                events.append(
                    SearchEvent(
                        neighbor.move_type,
                        False,
                        "visited schedule",
                        before_objective,
                        before_objective,
                        schedule_key,
                    )
                )
                continue

            try:
                candidate_outcome = evaluate_schedule(
                    snapshot,
                    neighbor.schedule,
                    channel,
                    evaluation,
                )
            except InfeasibleScheduleError as exc:
                events.append(
                    SearchEvent(
                        neighbor.move_type,
                        False,
                        str(exc),
                        before_objective,
                        before_objective,
                        schedule_key,
                    )
                )
                continue

            after_objective = network_objective(candidate_outcome)
            if search.acceptance is Acceptance.GLOBAL:
                accepted = after_objective > before_objective
                accepted_reason = "global strict improvement"
            else:
                accepted = _bilateral_accepts(
                    current,
                    current_outcome,
                    neighbor,
                    candidate_outcome,
                )
                accepted_reason = "bilateral Pareto improvement"
            events.append(
                SearchEvent(
                    neighbor.move_type,
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
                current = neighbor.schedule
                current_outcome = candidate_outcome
                visited.add(schedule_key)
                improved = True
                break
        if not improved:
            break

    converged = not improved
    if not converged:
        termination_reason = "max_iterations"
    elif search.acceptance is Acceptance.GLOBAL:
        termination_reason = "local_optimum"
    else:
        termination_reason = "cycle_free_fixed_point"
    return SearchResult(
        current,
        current_outcome,
        tuple(events),
        iterations,
        converged,
        termination_reason,
    )
