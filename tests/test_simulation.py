import numpy as np
import pytest

from uav_mec.domain import (
    AllocationDecision,
    ExperimentTrace,
    Schedule,
    SlotOutcome,
    SlotSnapshot,
    Task,
)
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ScenarioConfig
from uav_mec.simulation import run_episode, run_outcome_episode
from uav_mec.trace import generate_trace


def evaluation_config(cfg: ScenarioConfig) -> EvaluationConfig:
    return EvaluationConfig(
        cfg.n_subcarriers,
        cfg.channel.bandwidth,
        cfg.channel.p_max,
        0.02,
        1.0,
        cfg.f_per_uav,
    )


def test_episode_denominator_equals_unique_arrivals_without_retries():
    cfg = ScenarioConfig(
        num_users=6,
        num_slots=5,
        random_burst=True,
        user_mobility=True,
    )
    trace = generate_trace(cfg, 321)
    calls = []

    def empty_scheduler(snapshot):
        calls.append(snapshot.slot_index)
        return Schedule()

    result = run_episode(trace, empty_scheduler, cfg, evaluation_config(cfg))

    expected_ids = {task.task_id for slot in trace.slots for task in slot.tasks}
    assert result.total_arrivals == len(expected_ids)
    assert len(result.task_ids) == len(set(result.task_ids))
    assert result.total_completed + result.total_missed == result.total_arrivals
    assert calls == list(range(cfg.num_slots))
    assert len(result.outcomes) == cfg.num_slots


@pytest.mark.parametrize("placement", ["same_slot", "later_slot"])
def test_episode_rejects_every_repeated_task_id(placement):
    cfg = ScenarioConfig(num_users=1, num_slots=2, num_uavs=1)
    task = Task(7, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    if placement == "same_slot":
        with pytest.raises(ValueError, match="unique task identity"):
            SlotSnapshot(0, (task, task), np.ones((2, 1)), 1)
        return
    else:
        slots = (
            SlotSnapshot(0, (task,), np.ones((1, 1)), 1),
            SlotSnapshot(1, (task,), np.ones((1, 1)), 1),
        )
    trace = ExperimentTrace(123, slots)

    with pytest.raises(ValueError, match="repeats a task_id"):
        run_episode(trace, lambda snapshot: Schedule(), cfg, evaluation_config(cfg))


def test_outcome_runner_rejects_a_method_that_changes_arrival_accounting():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    trace = ExperimentTrace(
        123,
        (SlotSnapshot(0, (task,), np.ones((1, 1)), 1),),
    )
    wrong = SlotOutcome(0, (), (), (), (), (), 0.0, 0.0)

    with pytest.raises(ValueError, match="arrived_task_ids"):
        run_outcome_episode(trace, lambda snapshot: wrong)


def test_outcome_runner_rejects_duplicate_completion_ids():
    task = Task(1, 0, 0, 1e6, 0.4e9, 0.8, 1e9)
    trace = ExperimentTrace(
        123,
        (SlotSnapshot(0, (task,), np.ones((1, 1)), 1),),
    )
    decision = AllocationDecision(1, 0.0, 0.0, True, 0.4, 0.4, "local")
    wrong = SlotOutcome(0, (1,), (1, 1), (), (decision,), (), 0.4, 0.0)

    with pytest.raises(ValueError, match="completion partition"):
        run_outcome_episode(trace, lambda snapshot: wrong)
