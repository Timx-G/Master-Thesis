import numpy as np

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task


def test_task_local_delay_and_deadline_identity():
    task = Task(
        task_id=1203,
        user_id=3,
        arrival_slot=12,
        data_bits=800_000.0,
        comp_cycles=1_200_000_000.0,
        deadline_s=0.8,
        local_cpu_hz=1_000_000_000.0,
    )
    assert task.local_delay_s == 1.2
    assert task.absolute_deadline_s == 12.8
    assert task.requires_offload


def test_schedule_key_is_independent_of_input_order():
    a = ScheduleEntry(7, 7, 1, 0, "weak")
    b = ScheduleEntry(2, 2, 0, 1, "solo")
    assert Schedule((a, b)).stable_key() == Schedule((b, a)).stable_key()


def test_slot_snapshot_rejects_wrong_channel_shape():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    try:
        SlotSnapshot(0, (task,), np.ones((2, 2)), 2)
    except ValueError as exc:
        assert "channel_gains" in str(exc)
    else:
        raise AssertionError("SlotSnapshot accepted a channel matrix with the wrong user count")


def test_slot_snapshot_owns_a_read_only_channel_copy():
    source = np.ones((1, 1))
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    snapshot = SlotSnapshot(0, (task,), source, 1)
    source[0, 0] = 9.0
    assert snapshot.channel_gains[0, 0] == 1.0
    try:
        snapshot.channel_gains[0, 0] = 2.0
    except ValueError:
        return
    raise AssertionError("SlotSnapshot channel matrix remained mutable")
