from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .domain import Schedule, ScheduleEntry, SlotSnapshot, Task


class ScheduleValidationError(ValueError):
    """Raised when a schedule violates a structural system constraint."""


def serving_uav_by_task(schedule: Schedule) -> dict[int, int]:
    return {entry.task_id: entry.serving_uav for entry in schedule.entries}


def validate_schedule(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    n_subcarriers: int,
) -> None:
    if n_subcarriers <= 0:
        raise ScheduleValidationError("n_subcarriers must be positive")

    task_by_id = {task.task_id: task for task in snapshot.tasks}
    seen_tasks: set[int] = set()
    seen_users: set[int] = set()
    groups: dict[tuple[int, int], list[ScheduleEntry]] = defaultdict(list)

    for entry in schedule.entries:
        if entry.task_id in seen_tasks:
            raise ScheduleValidationError(f"duplicate task {entry.task_id}")
        if entry.user_id in seen_users:
            raise ScheduleValidationError(f"duplicate user {entry.user_id}")

        task = task_by_id.get(entry.task_id)
        if task is None or task.user_id != entry.user_id:
            raise ScheduleValidationError(f"task/user mismatch for task {entry.task_id}")
        if not task.requires_offload:
            raise ScheduleValidationError(f"local-feasible task {entry.task_id} consumes radio")
        if not 0 <= entry.serving_uav < snapshot.num_uavs:
            raise ScheduleValidationError(f"invalid UAV {entry.serving_uav}")
        if not 0 <= entry.subcarrier < n_subcarriers:
            raise ScheduleValidationError(f"invalid subcarrier {entry.subcarrier}")

        seen_tasks.add(entry.task_id)
        seen_users.add(entry.user_id)
        groups[(entry.serving_uav, entry.subcarrier)].append(entry)

    for resource_block, entries in groups.items():
        if len(entries) > 2:
            raise ScheduleValidationError(f"capacity exceeded at {resource_block}")
        roles = {entry.noma_role for entry in entries}
        expected = {"solo"} if len(entries) == 1 else {"strong", "weak"}
        if roles != expected:
            raise ScheduleValidationError(
                f"roles {sorted(roles)} invalid at {resource_block}; "
                f"expected {sorted(expected)}"
            )


def normalize_roles(schedule: Schedule, snapshot: SlotSnapshot) -> Schedule:
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    groups: dict[tuple[int, int], list[ScheduleEntry]] = defaultdict(list)
    for entry in schedule.entries:
        task = task_by_id.get(entry.task_id)
        if task is None or task.user_id != entry.user_id:
            raise ScheduleValidationError(f"task/user mismatch for task {entry.task_id}")
        if not 0 <= entry.serving_uav < snapshot.num_uavs:
            raise ScheduleValidationError(f"invalid UAV {entry.serving_uav}")
        if entry.subcarrier < 0:
            raise ScheduleValidationError(f"invalid subcarrier {entry.subcarrier}")
        groups[(entry.serving_uav, entry.subcarrier)].append(entry)

    normalized: list[ScheduleEntry] = []
    for (uav, subcarrier), entries in sorted(groups.items()):
        if len(entries) > 2:
            raise ScheduleValidationError(f"capacity exceeded at {(uav, subcarrier)}")
        if len(entries) == 1:
            entry = entries[0]
            normalized.append(
                ScheduleEntry(entry.task_id, entry.user_id, uav, subcarrier, "solo")
            )
            continue

        try:
            ordered = sorted(
                entries,
                key=lambda entry: (
                    -snapshot.channel_gains[row_by_task[entry.task_id], uav],
                    entry.task_id,
                ),
            )
        except (IndexError, KeyError) as error:
            raise ScheduleValidationError(
                f"cannot normalize resource block {(uav, subcarrier)}"
            ) from error

        strong, weak = ordered
        normalized.extend(
            (
                ScheduleEntry(strong.task_id, strong.user_id, uav, subcarrier, "strong"),
                ScheduleEntry(weak.task_id, weak.user_id, uav, subcarrier, "weak"),
            )
        )

    return Schedule(tuple(normalized))


def normalize_roles_for_blocks(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    affected_blocks: tuple[tuple[int, int], ...],
    *,
    task_by_id: Mapping[int, Task] | None = None,
    row_by_task: Mapping[int, int] | None = None,
    trusted: bool = False,
) -> Schedule:
    """Normalize changed blocks while preserving an already-normalized remainder."""

    affected = frozenset(affected_blocks)
    if not affected:
        raise ValueError("affected_blocks must be non-empty")
    tasks = (
        {task.task_id: task for task in snapshot.tasks}
        if task_by_id is None
        else task_by_id
    )
    rows = (
        {task.task_id: row for row, task in enumerate(snapshot.tasks)}
        if row_by_task is None
        else row_by_task
    )
    changed: dict[tuple[int, int], list[ScheduleEntry]] = defaultdict(list)
    normalized = []
    for entry in schedule.entries:
        block = (entry.serving_uav, entry.subcarrier)
        if block not in affected:
            normalized.append(entry)
            continue
        if not trusted:
            task = tasks.get(entry.task_id)
            if task is None or task.user_id != entry.user_id:
                raise ScheduleValidationError(f"task/user mismatch for task {entry.task_id}")
            if not 0 <= entry.serving_uav < snapshot.num_uavs:
                raise ScheduleValidationError(f"invalid UAV {entry.serving_uav}")
            if entry.subcarrier < 0:
                raise ScheduleValidationError(f"invalid subcarrier {entry.subcarrier}")
        changed[block].append(entry)

    for (uav, subcarrier), entries in changed.items():
        if len(entries) > 2:
            raise ScheduleValidationError(f"capacity exceeded at {(uav, subcarrier)}")
        if len(entries) == 1:
            entry = entries[0]
            normalized.append(
                ScheduleEntry(entry.task_id, entry.user_id, uav, subcarrier, "solo")
            )
            continue
        ordered = sorted(
            entries,
            key=lambda entry: (
                -snapshot.channel_gains[rows[entry.task_id], uav],
                entry.task_id,
            ),
        )
        strong, weak = ordered
        normalized.extend(
            (
                ScheduleEntry(strong.task_id, strong.user_id, uav, subcarrier, "strong"),
                ScheduleEntry(weak.task_id, weak.user_id, uav, subcarrier, "weak"),
            )
        )

    role_order = {"solo": 0, "strong": 0, "weak": 1}
    normalized.sort(
        key=lambda entry: (
            entry.serving_uav,
            entry.subcarrier,
            role_order[entry.noma_role],
        )
    )
    return Schedule(tuple(normalized))
