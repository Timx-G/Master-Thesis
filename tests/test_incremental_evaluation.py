import numpy as np

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import (
    build_evaluation_context,
    EvaluationConfig,
    InfeasibleScheduleError,
    evaluate_schedule,
    evaluate_schedule_incremental,
    score_schedule_incremental,
)
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.scheduler import generate_neighbors


def test_incremental_neighbor_evaluation_matches_full_recomputation():
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(6))
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array(
            [
                [9e-8, 2e-8],
                [7e-8, 3e-8],
                [6e-8, 5e-8],
                [5e-8, 6e-8],
                [3e-8, 7e-8],
                [2e-8, 9e-8],
            ]
        ),
        2,
    )
    schedule = Schedule(
        (
            ScheduleEntry(0, 0, 0, 0, "strong"),
            ScheduleEntry(1, 1, 0, 0, "weak"),
            ScheduleEntry(4, 4, 1, 1, "weak"),
            ScheduleEntry(5, 5, 1, 1, "strong"),
        )
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)
    evaluation = EvaluationConfig(
        2,
        5e6,
        0.2,
        0.02,
        0.1,
        4e9,
        0.1,
        cross_interference_scale=0.5,
    )
    before = evaluate_schedule(snapshot, schedule, channel, evaluation)
    context = build_evaluation_context(snapshot, channel, evaluation)

    checked = 0
    for neighbor in generate_neighbors(schedule, snapshot, 2, True):
        try:
            full = evaluate_schedule(snapshot, neighbor.schedule, channel, evaluation)
        except InfeasibleScheduleError:
            continue
        incremental = evaluate_schedule_incremental(
            snapshot,
            schedule,
            before,
            neighbor.schedule,
            neighbor.affected_blocks,
            channel,
            evaluation,
        )
        assert incremental == full
        score = score_schedule_incremental(
            snapshot,
            schedule,
            before,
            neighbor.schedule,
            neighbor.affected_blocks,
            channel,
            evaluation,
            validate_structure=False,
            context=context,
        )
        screening_score = score_schedule_incremental(
            snapshot,
            schedule,
            before,
            neighbor.schedule,
            neighbor.affected_blocks,
            channel,
            evaluation,
            validate_structure=False,
            context=context,
            retain_rates=False,
        )
        assert score.completed_count == full.completed_count
        assert round(score.total_mec_cpu_hz) == round(full.total_mec_cpu_hz)
        assert round(score.total_cross_interference_w / 1e-18) == round(
            full.total_cross_interference_w / 1e-18
        )
        assert score.rates == full.rates
        assert screening_score.completed_count == score.completed_count
        assert round(screening_score.total_mec_cpu_hz) == round(score.total_mec_cpu_hz)
        assert round(screening_score.total_cross_interference_w / 1e-18) == round(
            score.total_cross_interference_w / 1e-18
        )
        assert screening_score.rates == ()
        checked += 1

    assert checked >= 20
