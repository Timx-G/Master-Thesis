from pathlib import Path

import numpy as np

from uav_mec.baselines_v2 import ia_greedy_schedule
from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, evaluate_schedule
from uav_mec.experiment import _slot_method, load_definition
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.trace import generate_trace


def test_ia_greedy_returns_a_feasible_schedule_on_shared_spectrum():
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(6))
    gains = np.array(
        [
            [9e-8, 2e-8],
            [7e-8, 3e-8],
            [5e-8, 4e-8],
            [4e-8, 5e-8],
            [3e-8, 7e-8],
            [2e-8, 9e-8],
        ]
    )
    snapshot = SlotSnapshot(0, tasks, gains, 2)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)

    schedule = ia_greedy_schedule(snapshot, channel, cfg)
    outcome = evaluate_schedule(snapshot, schedule, channel, cfg)

    assert all(record.sic_feasible for record in outcome.rates)
    assert len({entry.task_id for entry in schedule.entries}) == len(
        schedule.entries
    )


def test_ia_greedy_does_not_fill_blocks_with_zero_benefit_users():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.zeros((1, 1)), 1)
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)

    schedule = ia_greedy_schedule(
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        cfg,
    )

    assert schedule.entries == ()


def test_registered_controlled_baselines_execute_on_shared_trace():
    definition = load_definition(Path("configs/thesis/smoke.toml"))
    snapshot = generate_trace(
        definition.scenario,
        definition.seeds[0],
    ).slots[0]

    for name in ("ia_greedy", "exchange_stable", "intra_bilateral"):
        outcome = _slot_method(
            name,
            definition.scenario,
            definition.evaluation,
            definition.max_iterations,
            [],
        )(snapshot)
        assert outcome.arrived_task_ids == tuple(
            sorted(task.task_id for task in snapshot.tasks)
        )
