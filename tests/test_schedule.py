import numpy as np
import pytest

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.schedule import (
    ScheduleValidationError,
    normalize_roles,
    serving_uav_by_task,
    validate_schedule,
)


def snapshot() -> SlotSnapshot:
    tasks = (
        Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.2e9, 0.8, 1e9),
        Task(3, 2, 0, 1e6, 0.5e9, 0.8, 1e9),
    )
    gains = np.array([[1e-8, 4e-8], [3e-8, 2e-8], [2e-8, 1e-8]])
    return SlotSnapshot(0, tasks, gains, 2)


def test_duplicate_task_is_rejected():
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "solo"),
            ScheduleEntry(1, 0, 1, 0, "solo"),
        )
    )

    with pytest.raises(ScheduleValidationError, match="duplicate task"):
        validate_schedule(schedule, snapshot(), n_subcarriers=1)


def test_local_feasible_task_must_not_consume_radio_resource():
    schedule = Schedule((ScheduleEntry(3, 2, 0, 0, "solo"),))

    with pytest.raises(ScheduleValidationError, match="local-feasible"):
        validate_schedule(schedule, snapshot(), n_subcarriers=1)


def test_duplicate_user_is_rejected():
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "solo"),
            ScheduleEntry(2, 0, 1, 0, "solo"),
        )
    )

    with pytest.raises(ScheduleValidationError, match="duplicate user"):
        validate_schedule(schedule, snapshot(), n_subcarriers=1)


@pytest.mark.parametrize(
    ("entry", "message"),
    (
        (ScheduleEntry(1, 2, 0, 0, "solo"), "task/user mismatch"),
        (ScheduleEntry(1, 0, -1, 0, "solo"), "invalid UAV"),
        (ScheduleEntry(1, 0, 2, 0, "solo"), "invalid UAV"),
        (ScheduleEntry(1, 0, 0, -1, "solo"), "invalid subcarrier"),
        (ScheduleEntry(1, 0, 0, 1, "solo"), "invalid subcarrier"),
    ),
)
def test_task_identity_and_resource_indices_are_validated(entry, message):
    with pytest.raises(ScheduleValidationError, match=message):
        validate_schedule(Schedule((entry,)), snapshot(), n_subcarriers=1)


def test_roles_are_normalized_on_actual_serving_channel():
    raw = Schedule(
        (
            ScheduleEntry(1, 0, 1, 0, "weak"),
            ScheduleEntry(2, 1, 1, 0, "strong"),
        )
    )

    normalized = normalize_roles(raw, snapshot())

    assert normalized.entries == (
        ScheduleEntry(1, 0, 1, 0, "strong"),
        ScheduleEntry(2, 1, 1, 0, "weak"),
    )
    assert serving_uav_by_task(normalized) == {1: 1, 2: 1}


def test_role_normalization_rejects_an_invalid_serving_uav():
    raw = Schedule((ScheduleEntry(1, 0, -1, 0, "solo"),))

    with pytest.raises(ScheduleValidationError, match="invalid UAV"):
        normalize_roles(raw, snapshot())


def test_equal_channel_gain_uses_task_id_as_the_role_tie_breaker():
    target = snapshot()
    tied = SlotSnapshot(
        target.slot_index,
        target.tasks,
        np.array([[2e-8, 4e-8], [2e-8, 2e-8], [1e-8, 1e-8]]),
        target.num_uavs,
    )
    raw = Schedule(
        (
            ScheduleEntry(2, 1, 0, 0, "strong"),
            ScheduleEntry(1, 0, 0, 0, "weak"),
        )
    )

    assert normalize_roles(raw, tied).entries == (
        ScheduleEntry(1, 0, 0, 0, "strong"),
        ScheduleEntry(2, 1, 0, 0, "weak"),
    )


@pytest.mark.parametrize(
    ("entries", "message"),
    (
        (
            (
                ScheduleEntry(1, 0, 0, 0, "strong"),
                ScheduleEntry(2, 1, 0, 0, "weak"),
                ScheduleEntry(3, 2, 0, 0, "solo"),
            ),
            "capacity exceeded",
        ),
        (
            (
                ScheduleEntry(1, 0, 0, 0, "strong"),
                ScheduleEntry(2, 1, 0, 0, "strong"),
            ),
            "roles",
        ),
    ),
)
def test_resource_block_capacity_and_roles_are_hard_constraints(entries, message):
    target = snapshot()
    if message == "capacity exceeded":
        target = SlotSnapshot(
            target.slot_index,
            target.tasks[:2] + (Task(3, 2, 0, 1e6, 1.2e9, 0.8, 1e9),),
            target.channel_gains,
            target.num_uavs,
        )

    with pytest.raises(ScheduleValidationError, match=message):
        validate_schedule(Schedule(entries), target, n_subcarriers=1)
