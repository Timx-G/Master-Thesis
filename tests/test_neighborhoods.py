import numpy as np

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.schedule import normalize_roles, normalize_roles_for_blocks, validate_schedule
from uav_mec.scheduler import generate_neighbors


def snapshot() -> SlotSnapshot:
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(5))
    gains = np.array(
        [
            [9e-8, 1e-8],
            [7e-8, 2e-8],
            [6e-8, 4e-8],
            [2e-8, 8e-8],
            [1e-8, 9e-8],
        ]
    )
    return SlotSnapshot(0, tasks, gains, 2)


def initial() -> Schedule:
    return Schedule(
        (
            ScheduleEntry(0, 0, 0, 0, "strong"),
            ScheduleEntry(1, 1, 0, 0, "weak"),
            ScheduleEntry(4, 4, 1, 0, "strong"),
            ScheduleEntry(3, 3, 1, 0, "weak"),
        )
    )


def test_affected_block_normalization_matches_full_normalization():
    raw = Schedule(
        (
            ScheduleEntry(2, 2, 0, 0, "weak"),
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(4, 4, 1, 0, "strong"),
            ScheduleEntry(3, 3, 1, 0, "weak"),
        )
    )

    assert normalize_roles_for_blocks(raw, snapshot(), ((0, 0),)) == normalize_roles(
        raw,
        snapshot(),
    )


def test_intra_neighborhood_has_no_cross_uav_reassignment():
    neighbors = generate_neighbors(
        initial(),
        snapshot(),
        n_subcarriers=2,
        allow_cross_uav=False,
    )
    original_uav = {
        entry.task_id: entry.serving_uav for entry in initial().entries
    }

    for neighbor in neighbors:
        for entry in neighbor.schedule.entries:
            if entry.task_id in original_uav:
                assert entry.serving_uav == original_uav[entry.task_id]


def test_replace_neighborhood_can_admit_an_unscheduled_nearest_uav_task():
    neighbors = generate_neighbors(
        initial(),
        snapshot(),
        n_subcarriers=2,
        allow_cross_uav=False,
    )

    assert any(
        neighbor.move_type == "replace"
        and neighbor.operation == "replace"
        and any(entry.task_id == 2 for entry in neighbor.schedule.entries)
        for neighbor in neighbors
    )


def test_replace_can_use_a_non_nearest_existing_position():
    neighbors = generate_neighbors(
        initial(),
        snapshot(),
        n_subcarriers=2,
        allow_cross_uav=False,
    )

    assert any(
        neighbor.move_type == "replace"
        and neighbor.operation == "replace"
        and any(
            entry.task_id == 2 and entry.serving_uav == 1
            for entry in neighbor.schedule.entries
        )
        for neighbor in neighbors
    )


def test_replace_neighborhood_includes_insert_and_drop_empty_position_moves():
    neighbors = generate_neighbors(
        initial(),
        snapshot(),
        n_subcarriers=2,
        allow_cross_uav=False,
    )

    assert any(
        neighbor.move_type == "replace"
        and neighbor.operation == "insert"
        and len(neighbor.schedule.entries) == len(initial().entries) + 1
        and any(entry.task_id == 2 for entry in neighbor.schedule.entries)
        for neighbor in neighbors
    )
    assert any(
        neighbor.move_type == "replace"
        and neighbor.operation == "drop"
        and len(neighbor.schedule.entries) == len(initial().entries) - 1
        for neighbor in neighbors
    )


def test_position_neighborhood_swaps_users_between_same_uav_blocks():
    schedule = Schedule(
        (
            ScheduleEntry(0, 0, 0, 0, "strong"),
            ScheduleEntry(1, 1, 0, 0, "weak"),
            ScheduleEntry(2, 2, 0, 1, "solo"),
        )
    )

    neighbors = generate_neighbors(
        schedule,
        snapshot(),
        n_subcarriers=2,
        allow_cross_uav=False,
    )

    assert any(
        neighbor.move_type == "position"
        and neighbor.operation == "swap"
        and neighbor.affected_blocks == ((0, 0), (0, 1))
        and any(
            entry.task_id == 0 and entry.subcarrier == 1
            for entry in neighbor.schedule.entries
        )
        for neighbor in neighbors
    )


def test_position_neighborhood_relocates_to_an_empty_same_uav_position():
    neighbors = generate_neighbors(
        initial(),
        snapshot(),
        n_subcarriers=2,
        allow_cross_uav=False,
    )

    assert any(
        neighbor.move_type == "position"
        and neighbor.operation == "relocate"
        and any(
            entry.task_id == 0
            and entry.serving_uav == 0
            and entry.subcarrier == 1
            for entry in neighbor.schedule.entries
        )
        for neighbor in neighbors
    )


def test_cross_neighborhood_contains_actual_service_exchange():
    neighbors = generate_neighbors(
        initial(),
        snapshot(),
        n_subcarriers=2,
        allow_cross_uav=True,
    )

    assert any(
        neighbor.move_type == "cross"
        and neighbor.operation == "swap"
        and any(
            entry.task_id == 0 and entry.serving_uav == 1
            for entry in neighbor.schedule.entries
        )
        for neighbor in neighbors
    )


def test_cross_neighborhood_relocates_to_an_empty_other_uav_position():
    neighbors = generate_neighbors(
        initial(),
        snapshot(),
        n_subcarriers=2,
        allow_cross_uav=True,
    )

    assert any(
        neighbor.move_type == "cross"
        and neighbor.operation == "relocate"
        and any(
            entry.task_id == 0
            and entry.serving_uav == 1
            and entry.subcarrier == 1
            for entry in neighbor.schedule.entries
        )
        for neighbor in neighbors
    )


def test_neighbors_are_valid_unique_and_stably_ordered():
    neighbors = generate_neighbors(
        initial(),
        snapshot(),
        n_subcarriers=2,
        allow_cross_uav=True,
    )
    keys = [neighbor.schedule.stable_key() for neighbor in neighbors]

    for neighbor in neighbors:
        validate_schedule(neighbor.schedule, snapshot(), 2)
    assert len(keys) == len(set(keys))
    assert [
        (neighbor.move_type, neighbor.schedule.stable_key())
        for neighbor in neighbors
    ] == sorted(
        (neighbor.move_type, neighbor.schedule.stable_key())
        for neighbor in neighbors
    )


def test_unsorted_neighbor_fast_path_preserves_the_complete_unique_set():
    stable = generate_neighbors(initial(), snapshot(), 2, True)
    unsorted = generate_neighbors(
        initial(),
        snapshot(),
        2,
        True,
        stable_order=False,
    )

    assert {neighbor.schedule.stable_key() for neighbor in unsorted} == {
        neighbor.schedule.stable_key() for neighbor in stable
    }
    assert len(unsorted) == len(stable)
