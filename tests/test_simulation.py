import numpy as np
import pytest

from uav_mec.domain import ExperimentTrace, Schedule, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ScenarioConfig
from uav_mec.simulation import run_episode
from uav_mec.trace import generate_trace


def evaluation_config(cfg: ScenarioConfig) -> EvaluationConfig:
    return EvaluationConfig(
        cfg.n_subcarriers,
        cfg.channel.bandwidth,
        cfg.channel.p_max,
        0.02,
        1.0,
        cfg.f_per_uav,
        cfg.task.result_ratio,
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
        slots = (SlotSnapshot(0, (task, task), np.ones((2, 1)), 1),)
    else:
        slots = (
            SlotSnapshot(0, (task,), np.ones((1, 1)), 1),
            SlotSnapshot(1, (task,), np.ones((1, 1)), 1),
        )
    trace = ExperimentTrace(123, slots)

    with pytest.raises(ValueError, match="repeats a task_id"):
        run_episode(trace, lambda snapshot: Schedule(), cfg, evaluation_config(cfg))
