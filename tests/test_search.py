import numpy as np
import pytest

from uav_mec.domain import (
    AllocationDecision,
    Schedule,
    ScheduleEntry,
    SlotOutcome,
    SlotSnapshot,
    Task,
)
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.scheduler import (
    Acceptance,
    Neighbor,
    SearchConfig,
    _bilateral_accepts,
    network_objective,
    optimize_schedule,
)


def replace_case():
    tasks = (
        Task(1, 0, 0, 2e6, 1.9e9, 0.8, 1e9),
        Task(2, 1, 0, 0.4e6, 1.1e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[2e-16], [9e-8]]), 1)
    initial = Schedule((ScheduleEntry(1, 0, 0, 0, "solo"),))
    evaluation = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9)
    return snapshot, initial, evaluation


def test_replace_search_strictly_improves_and_converges():
    snapshot, initial, evaluation = replace_case()

    result = optimize_schedule(
        initial,
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
        SearchConfig(False, Acceptance.GLOBAL, 20),
    )

    assert result.outcome.completed_count == 1
    assert any(entry.task_id == 2 for entry in result.schedule.entries)
    assert all(
        event.after_objective > event.before_objective
        for event in result.events
        if event.accepted
    )
    assert result.iterations <= 20
    assert result.converged
    assert result.termination_reason == "local_optimum"


def test_search_reports_iteration_limit_after_accepting_last_allowed_move():
    snapshot, initial, evaluation = replace_case()

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
