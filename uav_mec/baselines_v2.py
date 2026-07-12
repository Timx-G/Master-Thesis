"""Controlled scheduling baselines on the trusted data path."""

from __future__ import annotations

import numpy as np

from .domain import Schedule, ScheduleEntry, SlotSnapshot
from .schedule import normalize_roles


def local_only_schedule(snapshot: SlotSnapshot) -> Schedule:
    """Return an empty radio schedule; local-feasible tasks remain evaluable."""
    return Schedule()


def greedy_schedule(
    snapshot: SlotSnapshot,
    n_subcarriers: int,
) -> Schedule:
    """Pair strongest and weakest nearest-UAV users without joint search."""
    if not isinstance(n_subcarriers, int) or isinstance(n_subcarriers, bool):
        raise ValueError("n_subcarriers must be an integer")
    if n_subcarriers <= 0:
        raise ValueError("n_subcarriers must be positive")

    row_by_task = {
        task.task_id: row for row, task in enumerate(snapshot.tasks)
    }
    entries: list[ScheduleEntry] = []
    for uav in range(snapshot.num_uavs):
        candidates = [
            task
            for task in snapshot.tasks
            if task.requires_offload
            and int(
                np.asarray(
                    snapshot.channel_gains[row_by_task[task.task_id]]
                ).argmax()
            )
            == uav
        ]
        candidates.sort(
            key=lambda task: (
                -snapshot.channel_gains[
                    row_by_task[task.task_id],
                    uav,
                ],
                task.task_id,
            )
        )
        selected = candidates[: 2 * n_subcarriers]
        for subcarrier in range(n_subcarriers):
            left = subcarrier
            right = len(selected) - 1 - subcarrier
            if left > right:
                break
            for index in sorted({left, right}):
                task = selected[index]
                entries.append(
                    ScheduleEntry(
                        task.task_id,
                        task.user_id,
                        uav,
                        subcarrier,
                        "solo",
                    )
                )

    return normalize_roles(Schedule(tuple(entries)), snapshot)
