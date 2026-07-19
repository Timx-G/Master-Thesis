import numpy as np

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.schedule import validate_schedule
from uav_mec.evaluation import EvaluationConfig
from uav_mec.scheduler import enumerate_raw_moves, generate_neighbors


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
            ScheduleEntry(3, 3, 1, 0, "strong"),
            ScheduleEntry(4, 4, 1, 0, "weak"),
        )
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
        and any(entry.task_id == 2 for entry in neighbor.schedule.entries)
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
        and neighbor.affected_blocks == ((0, 0), (0, 1))
        and any(
            entry.task_id == 0 and entry.subcarrier == 1
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
        and any(
            entry.task_id == 0 and entry.serving_uav == 1
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


def test_full_raw_neighborhood_includes_empty_position_actions_and_removal():
    current = initial()
    raw_moves = enumerate_raw_moves(
        snapshot(),
        current,
        EvaluationConfig(2, 5e6, 0.2, 0.02, 0.01, 8e9),
    )

    # Insert an unscheduled task into a genuinely empty block.
    assert any(
        set(move.task_ids) == {0, 1, 2, 3, 4}
        and any(
            entry.task_id == 2 and entry.subcarrier == 1
            for entry in move.entries
        )
        for move in raw_moves
    )
    # Drop is an explicit raw action; closure is not responsible for creating it.
    assert any(set(move.task_ids) == {1, 3, 4} for move in raw_moves)
    # Intra- and cross-UAV relocations may target an empty position.
    assert any(
        any(
            entry.task_id == 0
            and entry.serving_uav == 0
            and entry.subcarrier == 1
            for entry in move.entries
        )
        and set(move.task_ids) == set(current.task_ids)
        for move in raw_moves
    )
    assert any(
        any(
            entry.task_id == 0
            and entry.serving_uav == 1
            and entry.subcarrier == 1
            for entry in move.entries
        )
        and set(move.task_ids) == set(current.task_ids)
        for move in raw_moves
    )
    for move in raw_moves:
        validate_schedule(move, snapshot(), 2)
