import numpy as np
import pytest

from uav_mec.domain import ChannelBasis, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.experiment import _slot_method
from uav_mec.model import ChannelConfig, NOMAChannel, ScenarioConfig
from uav_mec.scheduler import _nearest_uav
from uav_mec.trace import generate_trace


def test_scheduler_uses_estimated_csi_for_nearest_uav():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(
        0,
        (task,),
        np.array([[9e-8, 1e-8]]),
        2,
        estimated_channel_gains=np.array([[1e-8, 9e-8]]),
    )

    assert _nearest_uav(snapshot, 1) == 1
    assert snapshot.channel_gains[0, 0] > snapshot.channel_gains[0, 1]


def test_csi_error_is_deterministic_and_does_not_change_exogenous_truth(
    development_seed_access,
):
    cfg = ScenarioConfig(num_users=6, num_slots=2, random_burst=True)
    perfect = generate_trace(
        cfg,
        99,
        csi_error_std=0.0,
        seed_access=development_seed_access,
    )
    noisy_a = generate_trace(
        cfg,
        99,
        csi_error_std=0.1,
        seed_access=development_seed_access,
    )
    noisy_b = generate_trace(
        cfg,
        99,
        csi_error_std=0.1,
        seed_access=development_seed_access,
    )

    for clean, left, right in zip(
        perfect.slots,
        noisy_a.slots,
        noisy_b.slots,
        strict=True,
    ):
        assert clean.tasks == left.tasks == right.tasks
        np.testing.assert_array_equal(clean.channel_gains, left.channel_gains)
        np.testing.assert_array_equal(
            left.estimated_channel_gains,
            right.estimated_channel_gains,
        )


def test_estimated_csi_is_immutable_and_changes_snapshot_equality():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    true = np.array([[9e-8, 1e-8]])
    first = SlotSnapshot(
        0,
        (task,),
        true,
        2,
        estimated_channel_gains=np.array([[8e-8, 2e-8]]),
    )
    second = SlotSnapshot(
        0,
        (task,),
        true,
        2,
        estimated_channel_gains=np.array([[7e-8, 3e-8]]),
    )

    assert first != second
    with pytest.raises(ValueError):
        first.estimated_channel_gains.setflags(write=True)


def test_explicit_channel_basis_selects_estimates_without_mutating_truth():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(
        0,
        (task,),
        np.array([[9e-8, 1e-8]]),
        2,
        estimated_channel_gains=np.array([[1e-8, 9e-8]]),
    )

    planning_gains = snapshot.channel_gains_for(ChannelBasis.PLANNING)
    realized_gains = snapshot.channel_gains_for(ChannelBasis.REALIZED)

    np.testing.assert_array_equal(
        planning_gains,
        snapshot.estimated_channel_gains,
    )
    np.testing.assert_array_equal(realized_gains, snapshot.channel_gains)
    np.testing.assert_array_equal(
        snapshot.channel_gains,
        np.array([[9e-8, 1e-8]]),
    )


def test_final_rate_uses_true_channel_after_estimated_csi_planning():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(
        0,
        (task,),
        np.array([[9e-8, 1e-8]]),
        2,
        estimated_channel_gains=np.array([[1e-8, 9e-8]]),
    )
    scenario = ScenarioConfig(
        num_users=1,
        num_slots=1,
        num_uavs=2,
        n_subcarriers=1,
    )
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9)

    outcome = _slot_method("greedy", scenario, cfg, 5, [])(snapshot)

    record = outcome.rates[0]
    channel = NOMAChannel(ChannelConfig(), scenario.uav.height)
    noise_w = channel.noise_for_bandwidth(cfg.bandwidth_hz)
    counterfactual_nearest_rate = cfg.bandwidth_hz * np.log2(
        1.0 + cfg.transmit_power_w * 9e-8 / noise_w
    )
    assert record.serving_uav == 1
    assert record.rate_bps < counterfactual_nearest_rate


def test_true_sic_outage_is_counted_as_a_miss_instead_of_crashing():
    tasks = (
        Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.4e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[1e-8], [1e-8]]),
        1,
        estimated_channel_gains=np.array([[9e-8], [1e-8]]),
    )
    scenario = ScenarioConfig(
        num_users=2,
        num_slots=1,
        num_uavs=1,
        n_subcarriers=1,
    )
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9)

    outcome = _slot_method("greedy", scenario, cfg, 5, [])(snapshot)

    assert outcome.completed_task_ids == ()
    assert outcome.missed_task_ids == (1, 2)
    assert outcome.sic_outage_task_ids == (1, 2)
    assert outcome.plan_key
    assert all(not record.sic_feasible for record in outcome.rates)
    assert {decision.reason for decision in outcome.decisions} == {"sic_outage"}
    assert all(decision.alpha > 0.0 for decision in outcome.decisions)
    assert all(decision.cpu_hz > 0.0 for decision in outcome.decisions)
    assert {
        (layer.task_id, layer.decode_rank)
        for record in outcome.rates
        for layer in record.sic_layers
    } == {(1, 0), (2, 1)}


@pytest.mark.parametrize("error", [-0.1, float("nan")])
def test_trace_rejects_nonphysical_csi_error(error, development_seed_access):
    with pytest.raises(ValueError, match="csi_error_std"):
        generate_trace(
            ScenarioConfig(num_users=1, num_slots=1),
            1,
            error,
            seed_access=development_seed_access,
        )
