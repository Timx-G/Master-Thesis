"""Deterministic schedule initialization and local-search building blocks."""

from __future__ import annotations

import math
from concurrent.futures import Executor, ProcessPoolExecutor
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .domain import Schedule, ScheduleEntry, SlotOutcome, SlotSnapshot, Task
from .evaluation import (
    EvaluationConfig,
    InfeasibleScheduleError,
    ScheduleScore,
    build_evaluation_context,
    build_outcome_from_rates,
    compute_rate_records,
    evaluate_schedule,
    evaluate_schedule_incremental,
    score_schedule_incremental,
)
from .model import NOMAChannel
from .schedule import (
    normalize_roles,
    normalize_roles_for_blocks,
    serving_uav_by_task,
    validate_schedule,
)


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
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    planning_gains = snapshot.planning_channel_gains
    offload_tasks = [task for task in snapshot.tasks if task.requires_offload]
    assigned = sorted(
        (
            task
            for task in offload_tasks
            if int(np.asarray(planning_gains[row_by_task[task.task_id]]).argmax()) == uav
        ),
        key=lambda task: task.task_id,
    )
    local_count = sum(not task.requires_offload for task in snapshot.tasks)
    task_groups = [(task.task_id,) for task in assigned]
    task_groups.extend(tuple(task.task_id for task in pair) for pair in combinations(assigned, 2))

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

    task_ids = sorted({task_id for candidate in candidates for task_id in candidate.task_ids})
    rows = []
    upper_bounds = []
    for task_id in task_ids:
        rows.append([float(task_id in candidate.task_ids) for candidate in candidates])
        upper_bounds.append(1.0)
    rows.append([1.0] * len(candidates))
    upper_bounds.append(float(n_subcarriers))

    objective = np.array(
        [
            -float(candidate.benefit) - 1e-6 / (1 + sum(candidate.task_ids))
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
        candidate for candidate, chosen in zip(candidates, result.x, strict=True) if chosen > 0.5
    ]


def _drop_weak_user(schedule: Schedule, block: tuple[int, int]) -> Schedule:
    return Schedule(
        tuple(
            entry
            for entry in schedule.entries
            if (entry.serving_uav, entry.subcarrier) != block or entry.noma_role != "weak"
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
        violating_ids = {record.task_id for record in records if not record.sic_feasible}
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
OperationType = Literal["replace", "insert", "drop", "swap", "relocate"]


@dataclass(frozen=True, slots=True)
class Neighbor:
    schedule: Schedule
    move_type: MoveType
    affected_blocks: tuple[tuple[int, int], ...]
    operation: OperationType = "swap"


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


def _at_block(
    entry: ScheduleEntry,
    serving_uav: int,
    subcarrier: int,
) -> ScheduleEntry:
    return ScheduleEntry(
        entry.task_id,
        entry.user_id,
        serving_uav,
        subcarrier,
        entry.noma_role,
    )


def _normalized_neighbor(
    entries: list[ScheduleEntry],
    move_type: MoveType,
    affected: set[tuple[int, int]],
    snapshot: SlotSnapshot,
    n_subcarriers: int,
    operation: OperationType,
    task_by_id: dict[int, Task],
    row_by_task: dict[int, int],
) -> Neighbor:
    del n_subcarriers
    schedule = normalize_roles_for_blocks(
        Schedule(tuple(entries)),
        snapshot,
        tuple(sorted(affected)),
        task_by_id=task_by_id,
        row_by_task=row_by_task,
        trusted=True,
    )
    return Neighbor(schedule, move_type, tuple(sorted(affected)), operation)


def generate_neighbors(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    n_subcarriers: int,
    allow_cross_uav: bool,
    *,
    stable_order: bool = True,
) -> tuple[Neighbor, ...]:
    """Enumerate canonical Replace, Position, and optional CrossUAV moves."""
    if not isinstance(stable_order, bool):
        raise ValueError("stable_order must be bool")
    validate_schedule(schedule, snapshot, n_subcarriers)
    schedule = normalize_roles(schedule, snapshot)
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    scheduled_ids = {entry.task_id for entry in schedule.entries}
    unscheduled = [
        task
        for task in snapshot.tasks
        if task.requires_offload and task.task_id not in scheduled_ids
    ]
    neighbors: list[Neighbor] = []
    entries = list(schedule.entries)
    occupancy: dict[tuple[int, int], int] = {
        (uav, subcarrier): 0
        for uav in range(snapshot.num_uavs)
        for subcarrier in range(n_subcarriers)
    }
    for entry in entries:
        occupancy[(entry.serving_uav, entry.subcarrier)] += 1

    for index, old in enumerate(entries):
        for task in unscheduled:
            changed = entries.copy()
            changed[index] = _with_task(old, task.task_id, task.user_id)
            neighbors.append(
                _normalized_neighbor(
                    changed,
                    "replace",
                    {(old.serving_uav, old.subcarrier)},
                    snapshot,
                    n_subcarriers,
                    "replace",
                    task_by_id,
                    row_by_task,
                )
            )

    for task in unscheduled:
        for block, count in sorted(occupancy.items()):
            if count >= 2:
                continue
            uav, subcarrier = block
            changed = entries + [
                ScheduleEntry(
                    task.task_id,
                    task.user_id,
                    uav,
                    subcarrier,
                    "solo",
                )
            ]
            neighbors.append(
                _normalized_neighbor(
                    changed,
                    "replace",
                    {block},
                    snapshot,
                    n_subcarriers,
                    "insert",
                    task_by_id,
                    row_by_task,
                )
            )

    for index, old in enumerate(entries):
        changed = entries.copy()
        del changed[index]
        neighbors.append(
            _normalized_neighbor(
                changed,
                "replace",
                {(old.serving_uav, old.subcarrier)},
                snapshot,
                n_subcarriers,
                "drop",
                task_by_id,
                row_by_task,
            )
        )

    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            first, second = entries[left], entries[right]
            if first.serving_uav != second.serving_uav or first.subcarrier == second.subcarrier:
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
                    "swap",
                    task_by_id,
                    row_by_task,
                )
            )

    for index, entry in enumerate(entries):
        source = (entry.serving_uav, entry.subcarrier)
        for target_subcarrier in range(n_subcarriers):
            target = (entry.serving_uav, target_subcarrier)
            if target == source or occupancy[target] >= 2:
                continue
            changed = entries.copy()
            changed[index] = _at_block(
                entry,
                entry.serving_uav,
                target_subcarrier,
            )
            neighbors.append(
                _normalized_neighbor(
                    changed,
                    "position",
                    {source, target},
                    snapshot,
                    n_subcarriers,
                    "relocate",
                    task_by_id,
                    row_by_task,
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
                        "swap",
                        task_by_id,
                        row_by_task,
                    )
                )

        for index, entry in enumerate(entries):
            source = (entry.serving_uav, entry.subcarrier)
            for target_uav in range(snapshot.num_uavs):
                if target_uav == entry.serving_uav:
                    continue
                for target_subcarrier in range(n_subcarriers):
                    target = (target_uav, target_subcarrier)
                    if occupancy[target] >= 2:
                        continue
                    changed = entries.copy()
                    changed[index] = _at_block(
                        entry,
                        target_uav,
                        target_subcarrier,
                    )
                    neighbors.append(
                        _normalized_neighbor(
                            changed,
                            "cross",
                            {source, target},
                            snapshot,
                            n_subcarriers,
                            "relocate",
                            task_by_id,
                            row_by_task,
                        )
                    )

    ordered = (
        sorted(
            neighbors,
            key=lambda neighbor: (
                neighbor.move_type,
                neighbor.schedule.stable_key(),
                neighbor.operation,
            ),
        )
        if stable_order
        else neighbors
    )
    unique: dict[tuple, Neighbor] = {}
    for neighbor in ordered:
        unique.setdefault(neighbor.schedule.stable_key(), neighbor)
    return tuple(unique.values())


class Acceptance(str, Enum):
    GLOBAL = "global"
    BILATERAL = "bilateral"


class Improvement(str, Enum):
    FIRST = "first"
    BEST = "best"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    allow_cross_uav: bool
    acceptance: Acceptance
    max_iterations: int = 50
    improvement: Improvement = Improvement.FIRST
    initializer_id: str = "initial"
    record_rejected_events: bool = True
    multistart_workers: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.allow_cross_uav, bool):
            raise ValueError("allow_cross_uav must be bool")
        if not isinstance(self.acceptance, Acceptance):
            raise ValueError("acceptance must be an Acceptance value")
        if not isinstance(self.max_iterations, int) or isinstance(self.max_iterations, bool):
            raise ValueError("max_iterations must be an integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not isinstance(self.improvement, Improvement):
            raise ValueError("improvement must be an Improvement value")
        if not isinstance(self.initializer_id, str) or not self.initializer_id:
            raise ValueError("initializer_id must be a non-empty string")
        if not isinstance(self.record_rejected_events, bool):
            raise ValueError("record_rejected_events must be bool")
        if (
            not isinstance(self.multistart_workers, int)
            or isinstance(self.multistart_workers, bool)
            or self.multistart_workers <= 0
        ):
            raise ValueError("multistart_workers must be a positive integer")


@dataclass(frozen=True, slots=True)
class SearchEvent:
    move_type: MoveType
    accepted: bool
    reason: str
    before_objective: tuple[int, int, int]
    after_objective: tuple[int, int, int]
    schedule_key: tuple
    external_rate_affected_task_ids: tuple[int, ...] = ()
    certificate: ExternalityCertificate | None = None
    operation: OperationType = "swap"


@dataclass(frozen=True, slots=True)
class SearchResult:
    schedule: Schedule
    outcome: SlotOutcome
    events: tuple[SearchEvent, ...]
    iterations: int
    converged: bool
    termination_reason: str
    initializer_id: str = "initial"
    candidate_evaluations: int = 0


@dataclass(frozen=True, slots=True)
class TaskImpact:
    task_id: int
    completed_before: bool
    completed_after: bool
    cpu_before_hz: float
    cpu_after_hz: float
    rate_before_bps: float | None
    rate_after_bps: float | None
    sinr_before: float | None
    sinr_after: float | None
    sic_before: bool | None
    sic_after: bool | None


@dataclass(frozen=True, slots=True)
class BlockInterferenceDelta:
    receiver_uav: int
    subcarrier: int
    before_w: float
    after_w: float
    delta_w: float


@dataclass(frozen=True, slots=True)
class ExternalityCertificate:
    participants: tuple[TaskImpact, ...]
    third_parties: tuple[TaskImpact, ...]
    interference_deltas: tuple[BlockInterferenceDelta, ...]


@dataclass(frozen=True, slots=True)
class _EvaluatedNeighbor:
    neighbor: Neighbor
    outcome: SlotOutcome | None
    after_objective: tuple[int, int, int]
    eligible: bool
    accepted_reason: str
    score: ScheduleScore | None = None


def network_objective(outcome: SlotOutcome | ScheduleScore) -> tuple[int, int, int]:
    """Return the canonical network-level lexicographic objective."""
    return (
        outcome.completed_count,
        -round(outcome.total_mec_cpu_hz),
        -round(outcome.total_cross_interference_w / 1e-18),
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
    decisions = [decision for decision in outcome.decisions if decision.task_id in task_ids]
    return (
        sum(decision.completed for decision in decisions),
        round(
            sum(decision.slack_s for decision in decisions if decision.completed),
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


def _bilateral_rank(
    neighbor: Neighbor,
    outcome: SlotOutcome,
) -> tuple[int, float]:
    utilities = [
        _block_utility(neighbor.schedule, outcome, block)
        for block in neighbor.affected_blocks
    ]
    return (
        sum(utility[0] for utility in utilities),
        round(sum(utility[1] for utility in utilities), 12),
    )


def _best_evaluated_neighbor(
    evaluated: Sequence[_EvaluatedNeighbor],
    acceptance: Acceptance,
) -> _EvaluatedNeighbor | None:
    eligible = [candidate for candidate in evaluated if candidate.eligible]
    if not eligible:
        return None
    if acceptance is Acceptance.GLOBAL:
        best_rank = max(candidate.after_objective for candidate in eligible)
        finalists = [
            candidate
            for candidate in eligible
            if candidate.after_objective == best_rank
        ]
    else:
        if any(candidate.outcome is None for candidate in eligible):
            raise RuntimeError("bilateral ranking requires materialized outcomes")
        best_rank = max(
            _bilateral_rank(candidate.neighbor, candidate.outcome)
            for candidate in eligible
        )
        finalists = [
            candidate
            for candidate in eligible
            if _bilateral_rank(candidate.neighbor, candidate.outcome) == best_rank
        ]
    return min(
        finalists,
        key=lambda candidate: (
            candidate.neighbor.schedule.stable_key(),
            candidate.neighbor.operation,
        ),
    )


def _prefer_evaluated_neighbor(
    current: _EvaluatedNeighbor | None,
    candidate: _EvaluatedNeighbor,
    acceptance: Acceptance,
) -> _EvaluatedNeighbor:
    """Return the deterministic best of two eligible candidates."""

    if current is None:
        return candidate
    if acceptance is Acceptance.GLOBAL:
        current_rank = current.after_objective
        candidate_rank = candidate.after_objective
    else:
        if current.outcome is None or candidate.outcome is None:
            raise RuntimeError("bilateral ranking requires materialized outcomes")
        current_rank = _bilateral_rank(current.neighbor, current.outcome)
        candidate_rank = _bilateral_rank(candidate.neighbor, candidate.outcome)
    if candidate_rank != current_rank:
        return candidate if candidate_rank > current_rank else current
    current_tie = (
        current.neighbor.schedule.stable_key(),
        current.neighbor.operation,
    )
    candidate_tie = (
        candidate.neighbor.schedule.stable_key(),
        candidate.neighbor.operation,
    )
    return candidate if candidate_tie < current_tie else current


def _external_rate_affected_task_ids(
    before_schedule: Schedule,
    before: SlotOutcome,
    neighbor: Neighbor,
    after: SlotOutcome,
) -> tuple[int, ...]:
    """Return non-participant tasks whose realized radio record changed."""
    affected_blocks = set(neighbor.affected_blocks)
    direct_task_ids = {
        entry.task_id
        for schedule in (before_schedule, neighbor.schedule)
        for entry in schedule.entries
        if (entry.serving_uav, entry.subcarrier) in affected_blocks
    }
    before_rates = {record.task_id: record for record in before.rates}
    after_rates = {record.task_id: record for record in after.rates}
    external: list[int] = []
    for task_id in sorted(set(before_rates) | set(after_rates)):
        if task_id in direct_task_ids:
            continue
        left = before_rates.get(task_id)
        right = after_rates.get(task_id)
        if left is None or right is None:
            external.append(task_id)
            continue
        changed = (
            left.serving_uav != right.serving_uav
            or left.subcarrier != right.subcarrier
            or left.sic_feasible != right.sic_feasible
            or not math.isclose(
                left.rate_bps,
                right.rate_bps,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            or not math.isclose(
                left.sinr,
                right.sinr,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            or not math.isclose(
                left.cross_interference_w,
                right.cross_interference_w,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        )
        if changed:
            external.append(task_id)
    return tuple(external)


def _impact_for_task(
    task_id: int,
    before: SlotOutcome,
    after: SlotOutcome,
) -> TaskImpact:
    before_decisions = {decision.task_id: decision for decision in before.decisions}
    after_decisions = {decision.task_id: decision for decision in after.decisions}
    before_rates = {record.task_id: record for record in before.rates}
    after_rates = {record.task_id: record for record in after.rates}
    before_decision = before_decisions.get(task_id)
    after_decision = after_decisions.get(task_id)
    before_rate = before_rates.get(task_id)
    after_rate = after_rates.get(task_id)
    return TaskImpact(
        task_id=task_id,
        completed_before=(False if before_decision is None else before_decision.completed),
        completed_after=(False if after_decision is None else after_decision.completed),
        cpu_before_hz=(0.0 if before_decision is None else before_decision.cpu_hz),
        cpu_after_hz=(0.0 if after_decision is None else after_decision.cpu_hz),
        rate_before_bps=(None if before_rate is None else before_rate.rate_bps),
        rate_after_bps=(None if after_rate is None else after_rate.rate_bps),
        sinr_before=(None if before_rate is None else before_rate.sinr),
        sinr_after=(None if after_rate is None else after_rate.sinr),
        sic_before=(None if before_rate is None else before_rate.sic_feasible),
        sic_after=(None if after_rate is None else after_rate.sic_feasible),
    )


def _optional_float_changed(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is not right
    return not math.isclose(left, right, rel_tol=1e-12, abs_tol=0.0)


def _impact_changed(impact: TaskImpact) -> bool:
    return (
        impact.completed_before != impact.completed_after
        or _optional_float_changed(impact.cpu_before_hz, impact.cpu_after_hz)
        or _optional_float_changed(impact.rate_before_bps, impact.rate_after_bps)
        or _optional_float_changed(impact.sinr_before, impact.sinr_after)
        or impact.sic_before != impact.sic_after
    )


def _interference_by_block(outcome: SlotOutcome) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for record in outcome.rates:
        block = (record.serving_uav, record.subcarrier)
        previous = values.setdefault(block, record.cross_interference_w)
        if not math.isclose(
            previous,
            record.cross_interference_w,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("rate records disagree on block-level interference")
    return values


def externality_certificate(
    before_schedule: Schedule,
    before: SlotOutcome,
    neighbor: Neighbor,
    after: SlotOutcome,
) -> ExternalityCertificate:
    """Return task-level participant/third-party changes and block interference deltas."""
    affected_blocks = set(neighbor.affected_blocks)
    participant_ids = {
        entry.task_id
        for schedule in (before_schedule, neighbor.schedule)
        for entry in schedule.entries
        if (entry.serving_uav, entry.subcarrier) in affected_blocks
    }
    all_task_ids = set(before.arrived_task_ids) | set(after.arrived_task_ids)
    participants = tuple(
        _impact_for_task(task_id, before, after)
        for task_id in sorted(participant_ids)
    )
    third_parties = tuple(
        impact
        for task_id in sorted(all_task_ids - participant_ids)
        if _impact_changed(impact := _impact_for_task(task_id, before, after))
    )

    before_interference = _interference_by_block(before)
    after_interference = _interference_by_block(after)
    deltas = []
    for uav, subcarrier in sorted(set(before_interference) | set(after_interference)):
        left = before_interference.get((uav, subcarrier), 0.0)
        right = after_interference.get((uav, subcarrier), 0.0)
        if math.isclose(left, right, rel_tol=1e-12, abs_tol=0.0):
            continue
        deltas.append(
            BlockInterferenceDelta(
                uav,
                subcarrier,
                left,
                right,
                right - left,
            )
        )
    return ExternalityCertificate(participants, third_parties, tuple(deltas))


def optimize_schedule(
    initial: Schedule,
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
    search: SearchConfig,
    _outcome_cache: dict[tuple, SlotOutcome] | None = None,
    _score_cache: dict[tuple, ScheduleScore] | None = None,
) -> SearchResult:
    """Run deterministic first- or best-improvement schedule search."""
    current = normalize_roles(initial, snapshot)
    evaluation_context = build_evaluation_context(snapshot, channel, evaluation)
    outcome_cache = {} if _outcome_cache is None else _outcome_cache
    score_cache = {} if _score_cache is None else _score_cache
    current_key = current.stable_key()
    current_outcome = outcome_cache.get(current_key)
    if current_outcome is None:
        current_outcome = evaluate_schedule(
            snapshot,
            current,
            channel,
            evaluation,
        )
        outcome_cache[current_key] = current_outcome
    events: list[SearchEvent] = []
    visited = {current.stable_key()}
    iterations = 0
    improved = False
    candidate_evaluations = 0

    while iterations < search.max_iterations:
        iterations += 1
        improved = False
        before_objective = network_objective(current_outcome)
        evaluated: list[_EvaluatedNeighbor] = []
        selected: _EvaluatedNeighbor | None = None
        neighbors = generate_neighbors(
            current,
            snapshot,
            evaluation.n_subcarriers,
            search.allow_cross_uav,
            stable_order=(
                search.improvement is not Improvement.BEST
                or search.record_rejected_events
            ),
        )
        for neighbor in neighbors:
            candidate_evaluations += 1
            schedule_key = neighbor.schedule.stable_key()
            if schedule_key in visited:
                if search.record_rejected_events:
                    events.append(
                        SearchEvent(
                            neighbor.move_type,
                            False,
                            "visited schedule",
                            before_objective,
                            before_objective,
                            schedule_key,
                            operation=neighbor.operation,
                        )
                    )
                continue

            try:
                candidate_outcome = outcome_cache.get(schedule_key)
                candidate_score: ScheduleScore | None = None
                if candidate_outcome is None and search.acceptance is Acceptance.GLOBAL:
                    candidate_score = score_cache.get(schedule_key)
                    if candidate_score is None:
                        candidate_score = score_schedule_incremental(
                            snapshot,
                            current,
                            current_outcome,
                            neighbor.schedule,
                            neighbor.affected_blocks,
                            channel,
                            evaluation,
                            validate_structure=False,
                            context=evaluation_context,
                            retain_rates=False,
                        )
                        score_cache[schedule_key] = candidate_score
                elif candidate_outcome is None:
                    candidate_outcome = evaluate_schedule_incremental(
                        snapshot,
                        current,
                        current_outcome,
                        neighbor.schedule,
                        neighbor.affected_blocks,
                        channel,
                        evaluation,
                        validate_structure=False,
                    )
                    outcome_cache[schedule_key] = candidate_outcome
            except InfeasibleScheduleError as exc:
                if search.record_rejected_events:
                    events.append(
                        SearchEvent(
                            neighbor.move_type,
                            False,
                            str(exc),
                            before_objective,
                            before_objective,
                            schedule_key,
                            operation=neighbor.operation,
                        )
                    )
                continue

            objective_state = (
                candidate_outcome if candidate_outcome is not None else candidate_score
            )
            if objective_state is None:
                raise RuntimeError("candidate evaluation produced no objective state")
            after_objective = network_objective(objective_state)
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
            if (
                search.improvement is Improvement.BEST
                and not search.record_rejected_events
            ):
                if accepted:
                    candidate = _EvaluatedNeighbor(
                        neighbor,
                        candidate_outcome,
                        after_objective,
                        True,
                        accepted_reason,
                        candidate_score,
                    )
                    selected = _prefer_evaluated_neighbor(
                        selected,
                        candidate,
                        search.acceptance,
                    )
            else:
                candidate = _EvaluatedNeighbor(
                    neighbor,
                    candidate_outcome,
                    after_objective,
                    accepted,
                    accepted_reason,
                    candidate_score,
                )
                evaluated.append(candidate)
                if accepted and search.improvement is Improvement.FIRST:
                    selected = candidate
                    break

        if search.improvement is Improvement.BEST and search.record_rejected_events:
            selected = _best_evaluated_neighbor(evaluated, search.acceptance)

        selected_external_task_ids: tuple[int, ...] = ()
        selected_certificate: ExternalityCertificate | None = None
        if selected is not None:
            if selected.outcome is None:
                if selected.score is None:
                    raise RuntimeError("selected candidate has no score or outcome")
                if selected.score.rates:
                    materialized = build_outcome_from_rates(
                        snapshot,
                        serving_uav_by_task(selected.neighbor.schedule),
                        selected.score.rates,
                        evaluation,
                    )
                else:
                    materialized = evaluate_schedule_incremental(
                        snapshot,
                        current,
                        current_outcome,
                        selected.neighbor.schedule,
                        selected.neighbor.affected_blocks,
                        channel,
                        evaluation,
                        validate_structure=False,
                    )
                if network_objective(materialized) != selected.after_objective:
                    raise RuntimeError(
                        "lightweight candidate score differs from materialized objective"
                    )
                selected = replace(selected, outcome=materialized)
            if selected.outcome is None:
                raise RuntimeError("selected candidate outcome was not materialized")
            selected_external_task_ids = _external_rate_affected_task_ids(
                current,
                current_outcome,
                selected.neighbor,
                selected.outcome,
            )
            selected_certificate = externality_certificate(
                current,
                current_outcome,
                selected.neighbor,
                selected.outcome,
            )

        if not search.record_rejected_events:
            evaluated = [] if selected is None else [selected]

        for candidate in evaluated:
            accepted = (
                selected is not None
                and candidate.neighbor.schedule.stable_key()
                == selected.neighbor.schedule.stable_key()
                and candidate.neighbor.operation == selected.neighbor.operation
            )
            if not accepted and not search.record_rejected_events:
                continue
            if accepted:
                reason = candidate.accepted_reason
            elif candidate.eligible and search.improvement is Improvement.BEST:
                reason = "not selected by best-improvement"
            else:
                reason = "acceptance criterion not met"
            events.append(
                SearchEvent(
                    candidate.neighbor.move_type,
                    accepted,
                    reason,
                    before_objective,
                    candidate.after_objective,
                    candidate.neighbor.schedule.stable_key(),
                    selected_external_task_ids if accepted else (),
                    selected_certificate if accepted else None,
                    candidate.neighbor.operation,
                )
            )

        if selected is not None:
            if selected.outcome is None:
                raise RuntimeError("selected candidate outcome was not materialized")
            verified_outcome = evaluate_schedule(
                snapshot,
                selected.neighbor.schedule,
                channel,
                evaluation,
            )
            if verified_outcome != selected.outcome:
                raise RuntimeError(
                    "incremental candidate evaluation differs from full recomputation"
                )
            current = selected.neighbor.schedule
            current_outcome = verified_outcome
            outcome_cache[current.stable_key()] = verified_outcome
            visited.add(current.stable_key())
            improved = True
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
        search.initializer_id,
        candidate_evaluations,
    )


def _optimize_one_start(
    arguments: tuple[
        str,
        Schedule,
        SlotSnapshot,
        NOMAChannel,
        EvaluationConfig,
        SearchConfig,
    ],
) -> SearchResult:
    initializer_id, schedule, snapshot, channel, evaluation, search = arguments
    return optimize_schedule(
        schedule,
        snapshot,
        channel,
        evaluation,
        replace(search, initializer_id=initializer_id, multistart_workers=1),
    )


def optimize_multistart(
    initializers: Iterable[tuple[str, Schedule]],
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
    search: SearchConfig,
    *,
    executor: Executor | None = None,
) -> SearchResult:
    """Optimize every deterministic initializer and return the best terminal result."""
    if executor is not None and search.multistart_workers > 1:
        seen_initializer_ids: set[str] = set()
        futures = []
        for initializer_id, schedule in initializers:
            if not isinstance(initializer_id, str) or not initializer_id:
                raise ValueError("initializer IDs must be non-empty strings")
            if initializer_id in seen_initializer_ids:
                raise ValueError("initializer IDs must be unique")
            seen_initializer_ids.add(initializer_id)
            arguments = (
                initializer_id,
                schedule,
                snapshot,
                channel,
                evaluation,
                search,
            )
            futures.append(executor.submit(_optimize_one_start, arguments))
        if not futures:
            raise ValueError("initializers must be non-empty")
        results = tuple(future.result() for future in futures)
    else:
        ordered = tuple(initializers)
        if not ordered:
            raise ValueError("initializers must be non-empty")
        initializer_ids = [initializer_id for initializer_id, _schedule in ordered]
        if any(not isinstance(value, str) or not value for value in initializer_ids):
            raise ValueError("initializer IDs must be non-empty strings")
        if len(set(initializer_ids)) != len(initializer_ids):
            raise ValueError("initializer IDs must be unique")
        ordered_starts = tuple(sorted(ordered, key=lambda item: item[0]))

    if executor is None and (
        search.multistart_workers == 1 or len(ordered_starts) == 1
    ):
        outcome_cache: dict[tuple, SlotOutcome] = {}
        score_cache: dict[tuple, ScheduleScore] = {}
        results = tuple(
            optimize_schedule(
                schedule,
                snapshot,
                channel,
                evaluation,
                replace(search, initializer_id=initializer_id),
                outcome_cache,
                score_cache,
            )
            for initializer_id, schedule in ordered_starts
        )
    elif executor is None:
        arguments = tuple(
            (initializer_id, schedule, snapshot, channel, evaluation, search)
            for initializer_id, schedule in ordered_starts
        )
        with ProcessPoolExecutor(
            max_workers=min(search.multistart_workers, len(arguments))
        ) as owned_executor:
            results = tuple(owned_executor.map(_optimize_one_start, arguments))
    best_objective = max(network_objective(result.outcome) for result in results)
    best = min(
        (result for result in results if network_objective(result.outcome) == best_objective),
        key=lambda result: (
            result.initializer_id,
            result.schedule.stable_key(),
        ),
    )
    return replace(
        best,
        candidate_evaluations=sum(
            result.candidate_evaluations for result in results
        ),
    )
