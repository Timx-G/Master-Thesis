import numpy as np
import pytest
import uav_mec.scheduler as scheduler_module

from uav_mec.domain import (
    AllocationDecision,
    RateRecord,
    Schedule,
    ScheduleEntry,
    SlotOutcome,
    SlotSnapshot,
    Task,
)
from uav_mec.evaluation import EvaluationConfig
from uav_mec.evaluation import InfeasibleScheduleError, evaluate_schedule
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.scheduler import (
    Acceptance,
    Improvement,
    Neighbor,
    SearchConfig,
    _bilateral_accepts,
    _external_rate_affected_task_ids,
    externality_certificate,
    network_objective,
    generate_neighbors,
    optimize_multistart,
    optimize_schedule,
)


def replace_case():
    tasks = (
        Task(1, 0, 0, 2e6, 1.9e9, 0.8, 1e9),
        Task(2, 1, 0, 0.4e6, 1.1e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[2e-16], [9e-8]]), 1)
    initial = Schedule((ScheduleEntry(1, 0, 0, 0, "solo"),))
    evaluation = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)
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
        event.after_objective > event.before_objective for event in result.events if event.accepted
    )
    assert result.iterations <= 20
    assert result.converged
    assert result.termination_reason == "local_optimum"
    assert all(isinstance(event.external_rate_affected_task_ids, tuple) for event in result.events)
    assert all(event.certificate is not None for event in result.events if event.accepted)
    assert all(event.certificate is None for event in result.events if not event.accepted)


def test_global_best_improvement_screens_candidates_with_lightweight_scores(monkeypatch):
    snapshot, initial, evaluation = replace_case()
    calls = {"score": 0, "materialized_incremental": 0}
    original_score = scheduler_module.score_schedule_incremental
    original_incremental = scheduler_module.evaluate_schedule_incremental

    def counted_score(*args, **kwargs):
        calls["score"] += 1
        return original_score(*args, **kwargs)

    def counted_incremental(*args, **kwargs):
        calls["materialized_incremental"] += 1
        return original_incremental(*args, **kwargs)

    monkeypatch.setattr(scheduler_module, "score_schedule_incremental", counted_score)
    monkeypatch.setattr(scheduler_module, "evaluate_schedule_incremental", counted_incremental)

    result = optimize_schedule(
        initial,
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
        SearchConfig(False, Acceptance.GLOBAL, 20, Improvement.BEST),
    )

    assert result.converged
    assert calls["score"] > 0
    assert calls["materialized_incremental"] == sum(
        event.accepted for event in result.events
    )
    assert calls["materialized_incremental"] < calls["score"]


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
    evaluation = EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)
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


def test_best_improvement_selects_the_best_candidate_in_the_iteration():
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(4))
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[9e-8], [7e-8], [5e-8], [3e-8]]),
        1,
    )
    initial = Schedule((ScheduleEntry(3, 3, 0, 0, "solo"),))
    evaluation = EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    before = evaluate_schedule(snapshot, initial, channel, evaluation)
    eligible = []
    for neighbor in generate_neighbors(initial, snapshot, 2, False):
        try:
            outcome = evaluate_schedule(snapshot, neighbor.schedule, channel, evaluation)
        except InfeasibleScheduleError:
            continue
        if network_objective(outcome) > network_objective(before):
            eligible.append((network_objective(outcome), neighbor.schedule.stable_key()))
    best_objective = max(objective for objective, _key in eligible)
    expected_key = min(key for objective, key in eligible if objective == best_objective)

    result = optimize_schedule(
        initial,
        snapshot,
        channel,
        evaluation,
        SearchConfig(
            False,
            Acceptance.GLOBAL,
            1,
            improvement=Improvement.BEST,
        ),
    )

    assert network_objective(result.outcome) == best_objective
    assert result.schedule.stable_key() == expected_key


def test_multistart_selects_the_best_terminal_result_deterministically():
    snapshot, initial, evaluation = replace_case()
    channel = NOMAChannel(ChannelConfig(), 100.0)
    empty = Schedule()
    search = SearchConfig(
        False,
        Acceptance.GLOBAL,
        1,
        improvement=Improvement.BEST,
    )
    individual = {
        initializer_id: optimize_schedule(
            schedule,
            snapshot,
            channel,
            evaluation,
            search,
        )
        for initializer_id, schedule in (("empty", empty), ("ilp", initial))
    }
    best_objective = max(network_objective(result.outcome) for result in individual.values())
    expected_id = min(
        initializer_id
        for initializer_id, result in individual.items()
        if network_objective(result.outcome) == best_objective
    )

    result = optimize_multistart(
        (("ilp", initial), ("empty", empty)),
        snapshot,
        channel,
        evaluation,
        search,
    )

    assert result.initializer_id == expected_id
    assert network_objective(result.outcome) == best_objective


def test_parallel_multistart_matches_serial_result_exactly():
    snapshot, initial, evaluation = replace_case()
    empty = Schedule(())
    initializers = (("empty", empty), ("scheduled", initial))
    channel = NOMAChannel(ChannelConfig(), 100.0)

    serial = optimize_multistart(
        initializers,
        snapshot,
        channel,
        evaluation,
        SearchConfig(
            False,
            Acceptance.GLOBAL,
            20,
            Improvement.BEST,
            multistart_workers=1,
        ),
    )
    parallel = optimize_multistart(
        initializers,
        snapshot,
        channel,
        evaluation,
        SearchConfig(
            False,
            Acceptance.GLOBAL,
            20,
            Improvement.BEST,
            multistart_workers=2,
        ),
    )

    assert parallel.schedule == serial.schedule
    assert parallel.outcome == serial.outcome
    assert parallel.initializer_id == serial.initializer_id
    assert parallel.candidate_evaluations == serial.candidate_evaluations


def test_parallel_multistart_reuses_a_caller_owned_executor(monkeypatch):
    snapshot, initial, evaluation = replace_case()
    initializers = (("empty", Schedule()), ("scheduled", initial))
    channel = NOMAChannel(ChannelConfig(), 100.0)

    class RecordingExecutor:
        def __init__(self):
            self.timeline = []

        def submit(self, function, arguments):
            initializer_id = arguments[0]
            self.timeline.append(f"submit:{initializer_id}")

            class ImmediateFuture:
                def result(self):
                    return function(arguments)

            return ImmediateFuture()

    executor = RecordingExecutor()

    def unexpected_pool(*_args, **_kwargs):
        raise AssertionError("a caller-owned executor must be reused")

    monkeypatch.setattr(scheduler_module, "ProcessPoolExecutor", unexpected_pool)
    def lazy_initializers():
        for initializer_id, schedule in initializers:
            executor.timeline.append(f"build:{initializer_id}")
            yield initializer_id, schedule

    result = optimize_multistart(
        lazy_initializers(),
        snapshot,
        channel,
        evaluation,
        SearchConfig(
            False,
            Acceptance.GLOBAL,
            20,
            Improvement.BEST,
            multistart_workers=2,
        ),
        executor=executor,
    )

    assert result.converged
    assert executor.timeline == [
        "build:empty",
        "submit:empty",
        "build:scheduled",
        "submit:scheduled",
    ]


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
        event.accepted and event.reason == "bilateral Pareto improvement" for event in result.events
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


def test_network_objective_prefers_lower_mec_cpu_after_cardinality():
    expensive = SlotOutcome(
        0,
        (1, 2),
        (1, 2),
        (),
        (),
        (),
        100.0,
        1e-9,
        total_mec_cpu_hz=2_000_001.0,
    )
    efficient = SlotOutcome(
        0,
        (1, 2),
        (1, 2),
        (),
        (),
        (),
        0.0,
        2e-9,
        total_mec_cpu_hz=2_000_000.0,
    )

    assert network_objective(efficient) > network_objective(expensive)


def test_network_objective_uses_frozen_cpu_and_interference_quantization():
    left = SlotOutcome(
        0,
        (1,),
        (1,),
        (),
        (),
        (),
        0.0,
        2.4e-18,
        total_mec_cpu_hz=1_000_000.4,
    )
    right = SlotOutcome(
        0,
        (1,),
        (1,),
        (),
        (),
        (),
        999.0,
        2.49e-18,
        total_mec_cpu_hz=1_000_000.49,
    )

    assert network_objective(left) == network_objective(right)


def test_external_rate_effect_excludes_tasks_in_directly_modified_blocks():
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "strong"),
            ScheduleEntry(2, 1, 0, 0, "weak"),
            ScheduleEntry(3, 2, 1, 0, "solo"),
            ScheduleEntry(4, 3, 1, 1, "solo"),
        )
    )
    neighbor = Neighbor(schedule, "position", ((0, 0),))
    before = SlotOutcome(
        0,
        (1, 2, 3, 4),
        (),
        (1, 2, 3, 4),
        (),
        (
            RateRecord(1, 10.0, 2.0, True, 1e-9, 0, 0),
            RateRecord(2, 8.0, 1.5, True, 1e-9, 0, 0),
            RateRecord(3, 7.0, 1.0, True, 2e-9, 1, 0),
            RateRecord(4, 6.0, 0.8, True, 0.0, 1, 1),
        ),
        0.0,
        4e-9,
    )
    after = SlotOutcome(
        0,
        (1, 2, 3, 4),
        (),
        (1, 2, 3, 4),
        (),
        (
            RateRecord(1, 11.0, 2.2, True, 0.8e-9, 0, 0),
            RateRecord(2, 9.0, 1.7, True, 0.8e-9, 0, 0),
            RateRecord(3, 5.0, 0.6, True, 3e-9, 1, 0),
            RateRecord(4, 6.0, 0.8, True, 0.0, 1, 1),
        ),
        0.0,
        4.6e-9,
    )

    assert _external_rate_affected_task_ids(
        schedule,
        before,
        neighbor,
        after,
    ) == (3,)


def test_externality_certificate_separates_participants_and_third_parties():
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "strong"),
            ScheduleEntry(2, 1, 0, 0, "weak"),
            ScheduleEntry(3, 2, 1, 0, "solo"),
            ScheduleEntry(4, 3, 1, 1, "solo"),
        )
    )
    neighbor = Neighbor(schedule, "position", ((0, 0),), "swap")
    before = SlotOutcome(
        0,
        (1, 2, 3, 4),
        (1, 3, 4),
        (2,),
        (
            AllocationDecision(1, 0.4, 1.0e9, True, 0.7, 0.1, "complete"),
            AllocationDecision(2, 0.4, 0.0, False, None, 0.0, "cpu_budget"),
            AllocationDecision(3, 0.4, 1.2e9, True, 0.7, 0.1, "complete"),
            AllocationDecision(4, 0.4, 0.8e9, True, 0.7, 0.1, "complete"),
        ),
        (
            RateRecord(1, 10.0, 2.0, True, 1e-9, 0, 0),
            RateRecord(2, 8.0, 1.5, True, 1e-9, 0, 0),
            RateRecord(3, 7.0, 1.0, True, 2e-9, 1, 0),
            RateRecord(4, 6.0, 0.8, True, 0.0, 1, 1),
        ),
        0.3,
        4e-9,
        total_mec_cpu_hz=3.0e9,
    )
    after = SlotOutcome(
        0,
        (1, 2, 3, 4),
        (1, 4),
        (2, 3),
        (
            AllocationDecision(1, 0.4, 0.9e9, True, 0.7, 0.1, "complete"),
            AllocationDecision(2, 0.4, 0.0, False, None, 0.0, "cpu_budget"),
            AllocationDecision(3, 0.4, 0.0, False, None, 0.0, "cpu_budget"),
            AllocationDecision(4, 0.4, 0.8e9, True, 0.7, 0.1, "complete"),
        ),
        (
            RateRecord(1, 11.0, 2.2, True, 0.8e-9, 0, 0),
            RateRecord(2, 9.0, 1.7, True, 0.8e-9, 0, 0),
            RateRecord(3, 5.0, 0.6, True, 3e-9, 1, 0),
            RateRecord(4, 6.0, 0.8, True, 0.0, 1, 1),
        ),
        0.2,
        4.6e-9,
        total_mec_cpu_hz=1.7e9,
    )

    certificate = externality_certificate(
        schedule,
        before,
        neighbor,
        after,
    )

    assert tuple(impact.task_id for impact in certificate.participants) == (1, 2)
    assert tuple(impact.task_id for impact in certificate.third_parties) == (3,)
    participant = certificate.participants[0]
    assert participant.cpu_before_hz == 1.0e9
    assert participant.cpu_after_hz == 0.9e9
    third_party = certificate.third_parties[0]
    assert third_party.completed_before
    assert not third_party.completed_after
    assert third_party.rate_before_bps == 7.0
    assert third_party.rate_after_bps == 5.0
    deltas = {
        (delta.receiver_uav, delta.subcarrier): delta.delta_w
        for delta in certificate.interference_deltas
    }
    assert deltas[(0, 0)] == pytest.approx(-0.2e-9)
    assert deltas[(1, 0)] == pytest.approx(1.0e-9)


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
        {
            "allow_cross_uav": False,
            "acceptance": Acceptance.GLOBAL,
            "improvement": "best",
        },
    ),
)
def test_search_config_rejects_ambiguous_types(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)
