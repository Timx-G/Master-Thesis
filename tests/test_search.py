from dataclasses import replace

import numpy as np
import pytest

import uav_mec.scheduler as scheduler_module
from uav_mec.closure import close_assignment, rate_raw_assignment
from uav_mec.domain import (
    AllocationDecision,
    ChannelBasis,
    RateRecord,
    Schedule,
    ScheduleEntry,
    SicLayerCheck,
    SlotOutcome,
    SlotSnapshot,
    Task,
)
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy
from uav_mec.scheduler import (
    Acceptance,
    Neighbor,
    SearchConfig,
    _bilateral_accepts,
    close_and_deduplicate_moves,
    generate_neighbors,
    network_objective,
    optimize_schedule,
)


def replace_case():
    tasks = (
        Task(1, 0, 0, 0.25e6, 0.218e9, 0.30, 0.597e9),
        Task(2, 1, 0, 0.49e6, 1.67e9, 1.17, 0.994e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[4.9e-8], [3.6e-8]]), 1)
    initial = Schedule((ScheduleEntry(1, 0, 0, 0, "solo"),))
    evaluation = EvaluationConfig(
        1,
        5e6,
        0.2,
        0.02,
        0.01,
        0.73e9,
        allocation_policy=AllocationPolicy.CROSSING_EQUAL,
    )
    return snapshot, initial, evaluation


def cpu_replace_case():
    snapshot, _initial, evaluation = replace_case()
    task = next(task for task in snapshot.tasks if task.task_id == 2)
    initial = Schedule(
        (ScheduleEntry(task.task_id, task.user_id, 0, 0, "solo"),)
    )
    return (
        snapshot,
        initial,
        replace(evaluation, allocation_policy=AllocationPolicy.DEADLINE),
    )


def _closure_search_case() -> tuple[
    SlotSnapshot,
    Schedule,
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
            comp_cycles=1_800_000_000.0,
            deadline_s=0.8,
            local_cpu_hz=500_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in (1, 2, 3)
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[9e-8], [4e-8], [2e-8]]),
        1,
    )
    left = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(2, 2, 0, 0, "weak"),
        )
    )
    right = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(3, 3, 0, 0, "weak"),
        )
    )
    cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.01,
        cpu_per_uav_hz=1_100_000_000.0,
        allocation_policy=AllocationPolicy.DEADLINE,
    )
    return snapshot, left, right, NOMAChannel(ChannelConfig(), 100.0), cfg


def test_raw_moves_are_deduplicated_only_after_closure() -> None:
    snapshot, left, right, channel, cfg = _closure_search_case()
    closed = close_and_deduplicate_moves(
        snapshot,
        (left, right),
        channel,
        cfg,
        channel_basis=ChannelBasis.PLANNING,
    )
    expected = {
        close_assignment(
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
        ).stable_key()
        for raw in (left, right)
    }
    assert {candidate.stable_key() for candidate in closed} == expected
    assert len(closed) == len(expected)


def test_iteration_cap_is_not_a_local_optimum_certificate() -> None:
    snapshot, left, _right, channel, cfg = _closure_search_case()
    initial = close_assignment(
        snapshot,
        rate_raw_assignment(
            snapshot,
            left,
            channel,
            cfg,
            channel_basis=ChannelBasis.PLANNING,
        ),
        channel,
        cfg,
    )
    result = optimize_schedule(
        snapshot,
        initial,
        channel,
        cfg,
        SearchConfig(max_iterations=0),
    )
    assert result.stop_reason == "iteration_cap"
    assert not result.local_optimum_certified


def test_replace_search_strictly_improves_and_converges():
    snapshot, initial, evaluation = cpu_replace_case()

    result = optimize_schedule(
        initial,
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
        SearchConfig(False, Acceptance.GLOBAL, 20),
    )

    assert result.outcome.completed_count == 2
    assert result.schedule.task_ids == (1, 2)
    assert all(
        event.after_objective > event.before_objective
        for event in result.events
        if event.accepted
    )
    assert result.iterations <= 20
    assert result.converged
    assert result.termination_reason == "local_optimum"


def test_search_returns_a_closed_executable_schedule() -> None:
    tasks = tuple(
        Task(i, i, 0, 1e6, 1.6e9, 0.8, 0.5e9)
        for i in (1, 2)
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[9e-8], [3e-8]]),
        1,
    )
    initial = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(2, 2, 0, 0, "weak"),
        )
    )
    evaluation = EvaluationConfig(
        1,
        5e6,
        0.2,
        0.02,
        0.01,
        1.2e9,
    )

    result = optimize_schedule(
        initial,
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
        SearchConfig(False, Acceptance.GLOBAL, 20),
    )

    assert set(result.schedule.task_ids) == set(
        result.outcome.completed_task_ids
    )
    assert set(result.schedule.task_ids) < set(initial.task_ids)


def test_search_enumerates_each_iteration_from_the_closed_current_schedule() -> None:
    tasks = (
        Task(1, 1, 0, 0.49e6, 1.67e9, 1.17, 0.994e9),
        Task(2, 2, 0, 0.25e6, 0.218e9, 0.30, 0.597e9),
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[3.6e-8], [4.9e-8]]),
        1,
    )
    raw = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "weak"),
            ScheduleEntry(2, 2, 0, 0, "strong"),
        )
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)
    evaluation = EvaluationConfig(
        1,
        5e6,
        0.2,
        0.02,
        0.01,
        0.73e9,
        allocation_policy=AllocationPolicy.CROSSING_EQUAL,
    )
    closed = close_assignment(
        snapshot,
        rate_raw_assignment(
            snapshot,
            raw,
            channel,
            evaluation,
            channel_basis=ChannelBasis.PLANNING,
        ),
        channel,
        evaluation,
    )

    assert raw.task_ids == (1, 2)
    assert closed.schedule.task_ids == (2,)
    assert all(
        len(neighbor.schedule.entries) < len(raw.entries)
        for neighbor in generate_neighbors(raw, snapshot, 1, False)
    )
    assert any(
        neighbor.move_type == "replace"
        and neighbor.schedule.task_ids == (1,)
        for neighbor in generate_neighbors(
            closed.schedule,
            snapshot,
            1,
            False,
        )
    )

    result = optimize_schedule(
        raw,
        snapshot,
        channel,
        evaluation,
        SearchConfig(False, Acceptance.GLOBAL, 20),
    )

    assert result.schedule.task_ids == (2,)
    assert any(
        event.move_type == "replace" and event.schedule_key
        for event in result.events
    )


def test_search_reports_iteration_limit_after_accepting_last_allowed_move():
    snapshot, initial, evaluation = cpu_replace_case()

    result = optimize_schedule(
        initial,
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
        SearchConfig(False, Acceptance.GLOBAL, 1),
    )

    assert result.iterations == 1
    assert not result.converged
    assert result.termination_reason == "max_iterations"
    assert any(event.accepted for event in result.events)


def test_search_is_deterministic():
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(4))
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[9e-8], [7e-8], [5e-8], [3e-8]]),
        1,
    )
    initial = Schedule(
        (
            ScheduleEntry(0, 0, 0, 0, "strong"),
            ScheduleEntry(3, 3, 0, 0, "weak"),
        )
    )
    evaluation = EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = SearchConfig(False, Acceptance.GLOBAL, 20)

    keys = {
        optimize_schedule(
            initial,
            snapshot,
            channel,
            evaluation,
            cfg,
        ).schedule.stable_key()
        for _ in range(5)
    }

    assert len(keys) == 1


def test_bilateral_search_uses_pairwise_acceptance_and_cycle_free_termination():
    snapshot, initial, evaluation = replace_case()

    result = optimize_schedule(
        initial,
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
        SearchConfig(False, Acceptance.BILATERAL, 20),
    )

    assert result.outcome.completed_count == 1
    assert result.converged
    assert result.termination_reason == "cycle_free_fixed_point"
    assert any(
        event.accepted and event.reason == "bilateral Pareto improvement"
        for event in result.events
    )


def test_global_gain_can_fail_bilateral_acceptance():
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "solo"),
            ScheduleEntry(2, 1, 0, 1, "strong"),
            ScheduleEntry(3, 2, 0, 1, "weak"),
        )
    )
    before = SlotOutcome(
        0,
        (1, 2, 3),
        (1,),
        (2, 3),
        (
            AllocationDecision(1, 0.4, 1e9, True, 0.7, 0.1, "complete"),
            AllocationDecision(2, 0.4, 0.0, False, None, 0.0, "cpu_budget"),
            AllocationDecision(3, 0.4, 0.0, False, None, 0.0, "cpu_budget"),
        ),
        (),
        0.1,
        2e-9,
    )
    after = SlotOutcome(
        0,
        (1, 2, 3),
        (2, 3),
        (1,),
        (
            AllocationDecision(1, 0.4, 0.0, False, None, 0.0, "cpu_budget"),
            AllocationDecision(2, 0.4, 1e9, True, 0.75, 0.05, "complete"),
            AllocationDecision(3, 0.4, 1e9, True, 0.75, 0.05, "complete"),
        ),
        (),
        0.1,
        1e-9,
    )
    neighbor = Neighbor(schedule, "position", ((0, 0), (0, 1)))

    assert network_objective(after) > network_objective(before)
    assert not _bilateral_accepts(schedule, before, neighbor, after)


def test_network_objective_uses_aggregate_victim_link_exposure():
    def outcome(cross_interference_w: float, compatibility_total: float):
        layer = SicLayerCheck(1, 0, 0, 0, 2.0, 1.0, True)
        rate = RateRecord(
            1,
            1_000_000.0,
            2.0,
            True,
            cross_interference_w,
            0,
            0,
            ChannelBasis.PLANNING,
            (layer,),
        )
        return SlotOutcome(
            0,
            (1,),
            (1,),
            (),
            (AllocationDecision(1, 0.5, 1e9, True, 0.5, 0.3, "mec"),),
            (rate,),
            0.3,
            compatibility_total,
        )

    before = outcome(0.4, 0.0)
    after = outcome(0.1, 99.0)

    assert after.aggregate_victim_link_exposure_w < (
        before.aggregate_victim_link_exposure_w
    )
    assert network_objective(after) > network_objective(before)


def test_network_objective_minimizes_mec_cpu_before_exposure() -> None:
    def outcome(cpu_hz: float, exposure_w: float) -> SlotOutcome:
        layer = SicLayerCheck(1, 0, 0, 0, 2.0, 1.0, True)
        rate = RateRecord(
            1,
            1_000_000.0,
            2.0,
            True,
            exposure_w,
            0,
            0,
            ChannelBasis.PLANNING,
            (layer,),
        )
        return SlotOutcome(
            0,
            (1,),
            (1,),
            (),
            (AllocationDecision(1, 0.5, cpu_hz, True, 0.5, 0.3, "mec"),),
            (rate,),
            0.3,
            exposure_w,
        )

    lower_cpu_higher_exposure = outcome(1e9, 0.4)
    higher_cpu_lower_exposure = outcome(2e9, 0.1)

    assert network_objective(lower_cpu_higher_exposure) > network_objective(
        higher_cpu_lower_exposure
    )


def test_search_result_preserves_closed_neighborhood_audit_counts() -> None:
    snapshot, initial, evaluation = replace_case()
    result = optimize_schedule(
        initial,
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
        SearchConfig(False, Acceptance.GLOBAL, 20),
    )

    assert result.stop_reason == "no_improving_move"
    assert result.local_optimum_certified
    assert result.raw_move_count >= result.unique_closed_candidate_count > 0
    assert result.cache_hits >= 0
    assert result.accepted_moves == sum(event.accepted for event in result.events)
    assert result.closure_rounds >= 0


def test_search_evaluates_every_raw_candidate_through_incremental_boundary(
    monkeypatch,
) -> None:
    snapshot, initial, evaluation = replace_case()
    calls = 0
    implementation = scheduler_module.evaluate_move_incremental

    def audited_incremental(*args, **kwargs):
        nonlocal calls
        calls += 1
        return implementation(*args, **kwargs)

    monkeypatch.setattr(
        scheduler_module,
        "evaluate_move_incremental",
        audited_incremental,
    )
    result = optimize_schedule(
        initial,
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
        SearchConfig(False, Acceptance.GLOBAL, 1),
    )

    assert calls == result.raw_move_count
    assert calls > 0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"allow_cross_uav": False, "acceptance": "global"},
        {"allow_cross_uav": 1, "acceptance": Acceptance.GLOBAL},
        {
            "allow_cross_uav": False,
            "acceptance": Acceptance.GLOBAL,
            "max_iterations": 1.5,
        },
    ),
)
def test_search_config_rejects_ambiguous_types(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)
