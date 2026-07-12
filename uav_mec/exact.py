"""Complete small-instance schedule enumeration for exact references."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .domain import Schedule, ScheduleEntry, SlotOutcome, SlotSnapshot
from .evaluation import (
    EvaluationConfig,
    InfeasibleScheduleError,
    evaluate_schedule,
)
from .model import NOMAChannel
from .schedule import normalize_roles
from .scheduler import network_objective


@dataclass(frozen=True, slots=True)
class ExactResult:
    schedule: Schedule
    outcome: SlotOutcome
    evaluated_schedules: int
    complete_search: bool


def exact_schedule(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    max_heavy_tasks: int,
) -> ExactResult:
    """Enumerate unserved and every UAV-subcarrier assignment completely."""
    if not isinstance(max_heavy_tasks, int) or isinstance(max_heavy_tasks, bool):
        raise ValueError("max_heavy_tasks must be an integer")
    if max_heavy_tasks < 0:
        raise ValueError("max_heavy_tasks must be non-negative")

    heavy = tuple(
        sorted(
            (task for task in snapshot.tasks if task.requires_offload),
            key=lambda task: task.task_id,
        )
    )
    if len(heavy) > max_heavy_tasks:
        raise ValueError(
            f"exact search limited to {max_heavy_tasks} heavy tasks"
        )

    resource_blocks = snapshot.num_uavs * cfg.n_subcarriers
    choices = range(-1, resource_blocks)
    best_schedule: Schedule | None = None
    best_outcome: SlotOutcome | None = None
    evaluated = 0

    for assignment in product(choices, repeat=len(heavy)):
        counts = {
            block: assignment.count(block)
            for block in range(resource_blocks)
        }
        if any(count > 2 for count in counts.values()):
            continue

        entries = []
        for task, block in zip(heavy, assignment, strict=True):
            if block < 0:
                continue
            uav, subcarrier = divmod(block, cfg.n_subcarriers)
            entries.append(
                ScheduleEntry(
                    task.task_id,
                    task.user_id,
                    uav,
                    subcarrier,
                    "solo",
                )
            )
        schedule = normalize_roles(Schedule(tuple(entries)), snapshot)
        evaluated += 1
        try:
            outcome = evaluate_schedule(snapshot, schedule, channel, cfg)
        except InfeasibleScheduleError:
            continue

        if best_schedule is None or best_outcome is None:
            best_schedule = schedule
            best_outcome = outcome
            continue
        candidate = (network_objective(outcome), schedule.stable_key())
        incumbent = (
            network_objective(best_outcome),
            best_schedule.stable_key(),
        )
        if candidate[0] > incumbent[0] or (
            candidate[0] == incumbent[0]
            and candidate[1] < incumbent[1]
        ):
            best_schedule = schedule
            best_outcome = outcome

    if best_schedule is None or best_outcome is None:
        raise RuntimeError("exact search found no feasible schedule")
    return ExactResult(best_schedule, best_outcome, evaluated, True)
