"""Deterministic schedule initialization and local-search building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .domain import Schedule, ScheduleEntry, SlotSnapshot
from .evaluation import (
    EvaluationConfig,
    InfeasibleScheduleError,
    compute_rate_records,
    evaluate_schedule,
)
from .model import NOMAChannel
from .schedule import normalize_roles


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
