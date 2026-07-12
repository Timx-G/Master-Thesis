import math

import numpy as np
import pytest

from uav_mec.baselines_v2 import evaluate_fair_ofdma
from uav_mec.domain import RateRecord, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, build_outcome_from_rates
from uav_mec.model import ChannelConfig, NOMAChannel


def evaluation_config(n_subcarriers: int = 2) -> EvaluationConfig:
    return EvaluationConfig(
        n_subcarriers,
        5e6,
        0.2,
        0.02,
        1.0,
        4e9,
        0.1,
    )


def test_fair_ofdma_uses_shared_task_and_cpu_semantics():
    tasks = tuple(Task(i, i, 0, 1e6, 1.3e9, 0.8, 1e9) for i in range(8))
    gains = np.array(
        [
            [9e-8, 1e-8],
            [8e-8, 2e-8],
            [7e-8, 3e-8],
            [6e-8, 4e-8],
            [4e-8, 6e-8],
            [3e-8, 7e-8],
            [2e-8, 8e-8],
            [1e-8, 9e-8],
        ]
    )
    snapshot = SlotSnapshot(0, tasks, gains, 2)

    outcome = evaluate_fair_ofdma(
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation_config(),
    )

    assert outcome.arrived_task_ids == tuple(range(8))
    assert len(outcome.decisions) == 8
    assert len(outcome.rates) == 8
    assert all(record.sic_feasible for record in outcome.rates)
    assert all(record.rate_bps > 0.0 for record in outcome.rates)


def test_two_users_on_one_ofdma_subcarrier_each_receive_half_slot_rate():
    tasks = (
        Task(1, 0, 0, 1e6, 1.3e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.3e9, 0.8, 1e9),
    )
    gains = np.array([[1e-7], [5e-8]])
    snapshot = SlotSnapshot(0, tasks, gains, 1)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = evaluation_config(n_subcarriers=1)

    outcome = evaluate_fair_ofdma(snapshot, channel, cfg)

    noise_w = channel.noise_for_bandwidth(cfg.bandwidth_hz)
    by_task = {record.task_id: record for record in outcome.rates}
    for row, task in enumerate(tasks):
        sinr = cfg.transmit_power_w * gains[row, 0] / noise_w
        full_slot_rate = cfg.bandwidth_hz * math.log1p(sinr) / math.log(2.0)
        assert by_task[task.task_id].rate_bps == pytest.approx(
            0.5 * full_slot_rate
        )


def test_fair_ofdma_keeps_local_tasks_out_of_radio_records():
    tasks = (
        Task(1, 0, 0, 1e6, 0.4e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.3e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[1e-7], [1e-7]]), 1)

    outcome = evaluate_fair_ofdma(
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation_config(1),
    )

    assert {record.task_id for record in outcome.rates} == {2}
    assert outcome.decisions[0].reason == "local"


def test_shared_outcome_builder_rejects_duplicate_rate_records():
    task = Task(1, 0, 0, 1e6, 1.3e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[1e-7]]), 1)
    record = RateRecord(1, 1e6, 1.0, True, 0.0, 0, 0)

    with pytest.raises(ValueError, match="duplicate rate"):
        build_outcome_from_rates(
            snapshot,
            {1: 0},
            (record, record),
            evaluation_config(1),
        )


def test_shared_outcome_builder_rejects_rate_cpu_service_mismatch():
    task = Task(1, 0, 0, 1e6, 1.3e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[1e-7, 1e-7]]), 2)
    record = RateRecord(1, 1e6, 1.0, True, 0.0, 1, 0)

    with pytest.raises(ValueError, match="serving UAV"):
        build_outcome_from_rates(
            snapshot,
            {1: 0},
            (record,),
            evaluation_config(1),
        )


def test_shared_outcome_builder_rejects_noninteger_uav_indices():
    task = Task(1, 0, 0, 1e6, 1.3e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[1e-7, 1e-7]]), 2)
    record = RateRecord(1, 1e6, 1.0, True, 0.0, 0.5, 0)

    with pytest.raises(ValueError, match="invalid UAV"):
        build_outcome_from_rates(
            snapshot,
            {1: 0.5},
            (record,),
            evaluation_config(1),
        )
