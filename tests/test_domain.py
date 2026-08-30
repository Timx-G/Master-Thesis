import copy
import pickle
import warnings

import numpy as np
import pytest

from uav_mec.domain import (
    AllocationDecision,
    RateRecord,
    Schedule,
    ScheduleEntry,
    SlotOutcome,
    SlotSnapshot,
    Task,
)


def assert_hardened_channel_gains(gains):
    expected = np.array(gains, copy=True, subok=False)

    with pytest.raises(ValueError):
        gains[0, 0] = 9.0
    with pytest.raises(ValueError):
        gains.setflags(write=True)
    with pytest.raises((AttributeError, ValueError)):
        gains.shape = (2, 1)
    with pytest.raises(ValueError):
        gains.resize((2, 1))
    replacement = np.zeros(gains.shape, dtype=gains.dtype)
    with pytest.raises((AttributeError, ValueError)):
        gains.data = replacement.data

    assert gains.shape == (1, 2)
    assert not gains.flags.writeable
    assert np.array_equal(gains, expected)


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


@pytest.mark.parametrize(
    "field_name",
    ("data_bits", "comp_cycles", "deadline_s", "local_cpu_hz"),
)
def test_task_rejects_non_positive_and_non_finite_quantities(field_name):
    quantities = {
        "data_bits": 1.0,
        "comp_cycles": 1.0,
        "deadline_s": 1.0,
        "local_cpu_hz": 1.0,
    }
    for invalid in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        invalid_quantities = quantities | {field_name: invalid}
        with pytest.raises(ValueError, match="task quantities"):
            Task(1, 0, 0, **invalid_quantities)


@pytest.mark.parametrize("job_index", (-1, 1.5, True))
def test_task_rejects_invalid_job_index(job_index):
    with pytest.raises(ValueError, match="job_index"):
        Task(1, 0, 0, 1.0, 1.0, 1.0, 1.0, job_index=job_index)


def test_schedule_key_is_independent_of_input_order():
    a = ScheduleEntry(7, 7, 1, 0, "weak")
    b = ScheduleEntry(2, 2, 0, 1, "solo")
    expected = ((0, 1, 2, 2, "solo"), (1, 0, 7, 7, "weak"))
    assert Schedule((a, b)).stable_key() == expected
    assert Schedule((b, a)).stable_key() == expected


def test_schedule_reuses_its_immutable_stable_key():
    schedule = Schedule(
        (
            ScheduleEntry(7, 7, 1, 0, "weak"),
            ScheduleEntry(2, 2, 0, 1, "solo"),
        )
    )

    assert schedule.stable_key() is schedule.stable_key()


def test_schedule_copies_entries_to_a_tuple():
    entry = ScheduleEntry(1, 1, 0, 0, "solo")
    source = [entry]
    schedule = Schedule(source)

    source.clear()

    assert schedule.entries == (entry,)
    assert isinstance(schedule.entries, tuple)


def test_slot_snapshot_rejects_wrong_channel_shape():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    try:
        SlotSnapshot(0, (task,), np.ones((2, 2)), 2)
    except ValueError as exc:
        assert "channel_gains" in str(exc)
    else:
        raise AssertionError("SlotSnapshot accepted a channel matrix with the wrong user count")


def test_slot_snapshot_rejects_invalid_real_channel_gains():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    for invalid in (float("nan"), float("inf"), float("-inf"), -1.0):
        with pytest.raises(ValueError, match="channel_gains"):
            SlotSnapshot(0, (task,), np.array([[invalid]]), 1)


def test_slot_snapshot_rejects_complex_channel_gains_without_warning():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="channel_gains"):
            SlotSnapshot(0, (task,), np.array([[1.0 + 2.0j]]), 1)


def test_slot_snapshot_copies_tasks_to_a_tuple():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    source = [task]
    snapshot = SlotSnapshot(0, source, np.ones((1, 1)), 1)

    source.clear()

    assert snapshot.tasks == (task,)
    assert isinstance(snapshot.tasks, tuple)


def test_slot_snapshot_equality_compares_values():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    equal_task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    snapshot = SlotSnapshot(0, (task,), np.array([[1.0, 2.0]]), 2)

    assert snapshot == SlotSnapshot(0, (equal_task,), np.array([[1.0, 2.0]]), 2)
    assert snapshot != SlotSnapshot(1, (equal_task,), np.array([[1.0, 2.0]]), 2)
    assert snapshot != SlotSnapshot(0, (equal_task,), np.array([[1.0, 3.0]]), 2)
    assert snapshot != SlotSnapshot(
        0,
        (Task(2, 0, 0, 1.0, 1.0, 0.8, 1.0),),
        np.array([[1.0, 2.0]]),
        2,
    )


def test_slot_snapshot_is_explicitly_unhashable():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    snapshot = SlotSnapshot(0, (task,), np.ones((1, 1)), 1)

    assert SlotSnapshot.__hash__ is None
    with pytest.raises(TypeError):
        hash(snapshot)


def test_slot_snapshot_owns_a_read_only_channel_copy():
    source = np.ones((1, 1))
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    snapshot = SlotSnapshot(0, (task,), source, 1)
    source[0, 0] = 9.0
    assert snapshot.channel_gains[0, 0] == 1.0
    with pytest.raises(ValueError):
        snapshot.channel_gains[0, 0] = 2.0


def test_slot_snapshot_channel_copy_has_expected_array_contract():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    gains = SlotSnapshot(0, (task,), np.ones((1, 2)), 2).channel_gains

    assert isinstance(gains, np.ndarray)
    assert gains.shape == (1, 2)
    assert gains.dtype == np.dtype(float)
    assert not gains.flags.writeable


@pytest.mark.parametrize(
    "flags",
    (
        pytest.param({"write": True}, id="write"),
        pytest.param({"align": False}, id="align"),
        pytest.param({"uic": True}, id="writeback-if-copy"),
    ),
)
def test_slot_snapshot_channel_copy_rejects_flag_changes(flags):
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    gains = SlotSnapshot(0, (task,), np.ones((1, 1)), 1).channel_gains

    with pytest.raises(ValueError, match="channel_gains"):
        gains.setflags(**flags)


def test_slot_snapshot_channel_copy_shape_cannot_be_reassigned():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    gains = SlotSnapshot(0, (task,), np.ones((1, 2)), 2).channel_gains

    with pytest.raises((AttributeError, ValueError)):
        gains.shape = (2, 1)


def test_slot_snapshot_channel_copy_cannot_be_resized():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    gains = SlotSnapshot(0, (task,), np.ones((1, 2)), 2).channel_gains

    with pytest.raises(ValueError):
        gains.resize((2, 1))
    assert gains.shape == (1, 2)


def test_slot_snapshot_deepcopy_preserves_equality_and_immutability():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    snapshot = SlotSnapshot(0, (task,), np.array([[1.0, 2.0]]), 2)

    copied = copy.deepcopy(snapshot)

    assert copied == snapshot
    assert copied is not snapshot
    assert_hardened_channel_gains(copied.channel_gains)


def test_slot_snapshot_pickle_roundtrip_preserves_equality_and_immutability():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    snapshot = SlotSnapshot(0, (task,), np.array([[1.0, 2.0]]), 2)

    restored = pickle.loads(pickle.dumps(snapshot))

    assert restored == snapshot
    assert_hardened_channel_gains(restored.channel_gains)


def test_slot_snapshot_channel_calculations_return_plain_numpy_values():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    gains = SlotSnapshot(0, (task,), np.array([[1.0, 4.0]]), 2).channel_gains

    result = np.sqrt(gains)
    total = np.sum(gains)

    assert type(result) is np.ndarray
    assert np.array_equal(result, np.array([[1.0, 2.0]]))
    result[0, 0] = 3.0
    result.shape = (2, 1)
    assert result.shape == (2, 1)
    assert np.isscalar(total)
    assert total == 5.0


def test_slot_snapshot_channel_rejects_ufunc_output_target():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    gains = SlotSnapshot(0, (task,), np.array([[1.0, 2.0]]), 2).channel_gains

    with pytest.raises(ValueError, match="channel_gains"):
        np.add(gains, 1.0, out=gains)

    output = np.empty((1, 2))
    assert np.add(gains, 1.0, out=output) is output
    assert np.array_equal(output, np.array([[2.0, 3.0]]))


def test_slot_outcome_copies_collection_fields_to_tuples():
    decision = AllocationDecision(1, 0.5, 1.0, True, 0.5, 0.3, "completed")
    rate = RateRecord(1, 1.0, 1.0, True, 0.0, 0, 0)
    arrived = [1]
    completed = [1]
    missed = [2]
    decisions = [decision]
    rates = [rate]
    outcome = SlotOutcome(
        0,
        arrived,
        completed,
        missed,
        decisions,
        rates,
        0.3,
        0.0,
    )

    arrived.append(3)
    completed.clear()
    missed.clear()
    decisions.clear()
    rates.clear()

    assert outcome.arrived_task_ids == (1,)
    assert outcome.completed_task_ids == (1,)
    assert outcome.missed_task_ids == (2,)
    assert outcome.decisions == (decision,)
    assert outcome.rates == (rate,)
    assert outcome.completed_count == 1
