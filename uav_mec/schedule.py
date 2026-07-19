from __future__ import annotations

from collections import defaultdict

from .domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot


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


def normalize_roles(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    *,
    channel_basis: ChannelBasis = ChannelBasis.PLANNING,
) -> Schedule:
    if channel_basis is not ChannelBasis.PLANNING:
        raise ValueError("role normalization is only valid on the planning basis")
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
                    -snapshot.channel_gains_for(channel_basis)[row_by_task[entry.task_id], uav],
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
