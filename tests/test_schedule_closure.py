from dataclasses import replace

import numpy as np
import pytest

from uav_mec.closure import (
    ClosedExecutableSchedule,
    closed_semantic_key,
    close_assignment,
    freeze_plan,
    rate_raw_assignment,
)
from uav_mec.domain import (
    ChannelBasis,
    FrozenSlotPlan,
    Schedule,
    ScheduleEntry,
    SlotSnapshot,
    Task,
)
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy


def _overloaded_case() -> tuple[
    SlotSnapshot,
    Schedule,
    NOMAChannel,
    EvaluationConfig,
]:
    tasks = tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=1_000_000.0,
            comp_cycles=1_600_000_000.0,
            deadline_s=0.8,
            local_cpu_hz=500_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in (1, 2)
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[9e-8], [3e-8]]), 1)
    raw = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(2, 2, 0, 0, "weak"),
        )
    )
    cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.01,
        cpu_per_uav_hz=1_200_000_000.0,
        allocation_policy=AllocationPolicy.DEADLINE,
    )
    return snapshot, raw, NOMAChannel(ChannelConfig(), 100.0), cfg


def test_raw_snapshot_is_distinct_from_closed_executable_schedule() -> None:
    snapshot, raw, channel, cfg = _overloaded_case()
    rated = rate_raw_assignment(
        snapshot,
        raw,
        channel,
        cfg,
        channel_basis=ChannelBasis.PLANNING,
    )
    closed = close_assignment(snapshot, rated, channel, cfg)
    assert isinstance(closed, ClosedExecutableSchedule)
    assert set(closed.schedule.task_ids) == set(
        closed.allocation.completed_task_ids
    )
    assert set(closed.schedule.task_ids) < set(rated.assignment.task_ids)
    assert closed.rate_snapshot.assignment == closed.schedule


def test_closure_is_idempotent_and_deletion_only() -> None:
    snapshot, raw, channel, cfg = _overloaded_case()
    first = close_assignment(
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
    second = close_assignment(snapshot, first.rate_snapshot, channel, cfg)
    assert second.stable_key() == first.stable_key()
    assert first.closure_rounds <= len(raw.entries)
    assert set(first.schedule.task_ids) <= set(raw.task_ids)


def test_closed_semantic_key_binds_stable_state_and_fingerprint() -> None:
    snapshot, raw, channel, cfg = _overloaded_case()
    closed = close_assignment(
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

    assert closed_semantic_key(closed) == (
        closed.stable_key(),
        closed.rate_snapshot.semantic_fingerprint,
    )


def test_closure_rejects_a_semantic_fingerprint_mismatch() -> None:
    snapshot, raw, channel, cfg = _overloaded_case()
    rated = rate_raw_assignment(
        snapshot,
        raw,
        channel,
        cfg,
        channel_basis=ChannelBasis.PLANNING,
    )

    with pytest.raises(ValueError, match="semantic fingerprint"):
        close_assignment(
            snapshot,
            rated,
            channel,
            replace(cfg, cpu_per_uav_hz=cfg.cpu_per_uav_hz * 2.0),
        )


def test_freeze_plan_accepts_only_planning_basis_closure() -> None:
    snapshot, raw, channel, cfg = _overloaded_case()
    planning = close_assignment(
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
    plan = freeze_plan(planning)

    assert isinstance(plan, FrozenSlotPlan)
    assert plan.schedule == planning.schedule
    assert plan.planning_rates == planning.rate_snapshot.rates
    assert plan.semantic_fingerprint == planning.rate_snapshot.semantic_fingerprint

    realized = replace(
        planning,
        rate_snapshot=replace(
            planning.rate_snapshot,
            channel_basis=ChannelBasis.REALIZED,
        ),
    )
    with pytest.raises(ValueError, match="planning-basis"):
        freeze_plan(realized)
