import dataclasses

import numpy as np

from uav_mec.closure import close_assignment, rate_raw_assignment
from uav_mec.domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, InfeasibleScheduleError
from uav_mec.incremental import dependency_closure, evaluate_move_incremental
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy
from uav_mec.scheduler import enumerate_raw_moves


def _random_incremental_case(
    seed: int,
) -> tuple[SlotSnapshot, Schedule, NOMAChannel, EvaluationConfig]:
    rng = np.random.default_rng(seed)
    tasks = tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=float(rng.uniform(600_000.0, 1_200_000.0)),
            comp_cycles=float(rng.uniform(1.2e9, 1.8e9)),
            deadline_s=0.8,
            local_cpu_hz=400_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in range(12)
    )
    gains = rng.uniform(2e-8, 9e-8, size=(12, 2))
    snapshot = SlotSnapshot(0, tasks, gains, 2)
    entries = []
    for index, task in enumerate(tasks):
        uav = (index // 2) % 2
        subcarrier = (index // 4) % 3
        role = "strong" if index % 2 == 0 else "weak"
        entries.append(
            ScheduleEntry(task.task_id, task.user_id, uav, subcarrier, role)
        )
    cfg = EvaluationConfig(
        n_subcarriers=3,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.001,
        cpu_per_uav_hz=20e9,
        allocation_policy=AllocationPolicy.DEADLINE,
    )
    return snapshot, Schedule(tuple(entries)), NOMAChannel(ChannelConfig(), 100.0), cfg


def _full_close(
    snapshot: SlotSnapshot,
    raw: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
):
    return close_assignment(
        snapshot,
        rate_raw_assignment(
            snapshot,
            raw,
            channel,
            cfg,
            channel_basis=ChannelBasis.PLANNING,
        ),
        channel,
        cfg,
    )


def _assert_closed_equal(left, right) -> None:
    assert left.stable_key() == right.stable_key()
    assert left.rate_snapshot.rates == right.rate_snapshot.rates
    assert left.allocation.decisions == right.allocation.decisions
    assert left.allocation.completed_task_ids == right.allocation.completed_task_ids


def test_ten_thousand_random_actions_match_full_closure() -> None:
    action_count = 0
    for seed in range(100):
        snapshot, initial_raw, channel, cfg = _random_incremental_case(seed)
        before = _full_close(snapshot, initial_raw, channel, cfg)
        rng = np.random.default_rng(seed + 100_000)
        for _move_index in range(100):
            moves = tuple(enumerate_raw_moves(snapshot, before.schedule, cfg))
            assert moves
            start = int(rng.integers(0, len(moves)))
            chosen = before.schedule
            full = before
            for offset in range(len(moves)):
                candidate = moves[(start + offset) % len(moves)]
                try:
                    full = _full_close(snapshot, candidate, channel, cfg)
                except InfeasibleScheduleError:
                    continue
                chosen = candidate
                break
            incremental = evaluate_move_incremental(
                snapshot,
                before,
                chosen,
                channel,
                cfg,
            )
            _assert_closed_equal(incremental, full)
            before = full
            action_count += 1
    assert action_count == 10_000


def test_cross_uav_move_dependency_contains_receivers_roles_and_cpu_peers() -> None:
    snapshot, before, _channel, cfg = _random_incremental_case(301)
    candidate_entries = []
    for entry in before.entries:
        if entry.task_id == 0:
            candidate_entries.append(dataclasses.replace(entry, serving_uav=1))
        elif entry.task_id == 2:
            candidate_entries.append(dataclasses.replace(entry, serving_uav=0))
        else:
            candidate_entries.append(entry)
    candidate = Schedule(tuple(candidate_entries))
    dependencies = dependency_closure(snapshot, before, candidate, cfg)
    assert dependencies.affected_subcarriers == (0,)
    assert dependencies.receiver_blocks == ((0, 0), (1, 0))
    assert dependencies.reassigned_task_ids == (0, 2)
    assert {0, 1, 2, 3} <= set(dependencies.role_task_ids)
    assert dependencies.cpu_uavs == (0, 1)
    assert set(dependencies.cpu_task_ids) == set(range(12))
