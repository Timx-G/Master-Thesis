import math
from pathlib import Path

import numpy as np
import pytest

from uav_mec.baselines_adapted import (
    guo_jagp_adapted,
    guo_required_powers,
    guo_total_required_power,
)
from uav_mec.domain import (
    ChannelBasis,
    Schedule,
    ScheduleEntry,
    SlotSnapshot,
    Task,
)
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ChannelConfig, NOMAChannel


def _offload_task(index: int) -> Task:
    return Task(
        task_id=index,
        user_id=index,
        arrival_slot=0,
        data_bits=600_000.0,
        comp_cycles=1_200_000_000.0,
        deadline_s=0.8,
        local_cpu_hz=400_000_000.0,
        slot_duration_s=1.0,
        result_ratio=0.0,
    )


def test_guo_power_recursion_matches_hand_calculation() -> None:
    powers = guo_required_powers(
        channel_power_gains=(4.0, 1.0),
        qos_spectral_efficiency=(1.0, 1.0),
        noise_w=1.0,
    )

    assert powers == pytest.approx((0.5, 1.0))


@pytest.mark.parametrize(
    ("gains", "targets", "noise_w", "message"),
    (
        ((), (), 1.0, "same positive length"),
        ((1.0,), (1.0, 2.0), 1.0, "same positive length"),
        ((0.0,), (1.0,), 1.0, "gains"),
        ((math.inf,), (1.0,), 1.0, "gains"),
        ((1.0,), (-1.0,), 1.0, "spectral efficiencies"),
        ((1.0,), (math.nan,), 1.0, "spectral efficiencies"),
        ((1.0,), (1.0,), 0.0, "noise_w"),
    ),
)
def test_guo_power_recursion_rejects_invalid_inputs(
    gains: tuple[float, ...],
    targets: tuple[float, ...],
    noise_w: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        guo_required_powers(gains, targets, noise_w)


def test_guo_potential_is_required_power_only_and_enforces_ceiling() -> None:
    tasks = (_offload_task(0), _offload_task(1))
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[4.0], [1.0]]),
        1,
    )
    schedule = Schedule(
        (
            ScheduleEntry(0, 0, 0, 0, "strong"),
            ScheduleEntry(1, 1, 0, 0, "weak"),
        )
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)
    channel.noise_linear = 1.0
    cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=channel.cfg.bandwidth,
        transmit_power_w=2.0,
        sic_residual=0.02,
        gamma_sic_min=0.1,
        cpu_per_uav_hz=4e9,
        cross_interference_scale=1.0,
    )

    potential = guo_total_required_power(
        snapshot,
        schedule,
        channel,
        cfg,
        ChannelBasis.PLANNING,
        1.0,
    )
    changed_thesis_terms = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=channel.cfg.bandwidth,
        transmit_power_w=2.0,
        sic_residual=0.9,
        gamma_sic_min=100.0,
        cpu_per_uav_hz=1.0,
        cross_interference_scale=0.0,
    )
    unchanged = guo_total_required_power(
        snapshot,
        schedule,
        channel,
        changed_thesis_terms,
        ChannelBasis.PLANNING,
        1.0,
    )

    assert potential == pytest.approx(1.5)
    assert unchanged == pytest.approx(potential)

    capped_cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=channel.cfg.bandwidth,
        transmit_power_w=0.75,
        sic_residual=0.02,
        gamma_sic_min=0.1,
        cpu_per_uav_hz=4e9,
    )
    assert math.isinf(
        guo_total_required_power(
            snapshot,
            schedule,
            channel,
            capped_cfg,
            ChannelBasis.PLANNING,
            1.0,
        )
    )


def test_guo_adaptation_is_deterministic_and_returns_a_raw_schedule() -> None:
    tasks = tuple(_offload_task(index) for index in range(4))
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array(
            [
                [9e-8, 2e-8],
                [6e-8, 4e-8],
                [3e-8, 8e-8],
                [2e-8, 7e-8],
            ]
        ),
        2,
    )
    cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.1,
        cpu_per_uav_hz=4e9,
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)

    first = guo_jagp_adapted(
        snapshot,
        channel,
        cfg,
        channel_basis=ChannelBasis.PLANNING,
        qos_spectral_efficiency=0.5,
    )
    second = guo_jagp_adapted(
        snapshot,
        channel,
        cfg,
        channel_basis=ChannelBasis.PLANNING,
        qos_spectral_efficiency=0.5,
    )

    assert first == second
    assert len(first.entries) == 4
    assert len(first.entries) <= (
        2 * snapshot.num_uavs * cfg.n_subcarriers
    )
    assert len(first.task_ids) == len(set(first.task_ids))


def test_guo_mapping_audit_keeps_adaptation_distinct_from_reproduction() -> None:
    mapping = Path(
        "docs/audits/2026-07-19-adapted-baseline-mapping.md"
    ).read_text(encoding="utf-8")
    closest_work = Path(
        "docs/literature_baseline_audit_2024_2026.md"
    ).read_text(encoding="utf-8")

    for component in (
        "decision variables",
        "objective/potential",
        "association",
        "grouping",
        "rate/SIC",
        "power",
        "channel",
        "CPU",
        "task/deadline",
        "mobility",
        "stopping rule",
    ):
        assert f"| {component} |" in mapping
    for evidence in (
        "10.1109/ICC.2019.8761779",
        "Eq. 5",
        "Eqs. 9--10",
        "Eq. 11",
        "Algorithm 1",
        "not a reproduction",
    ):
        assert evidence in mapping
    assert "unavailable author code" in closest_work
    assert "two-user block cap" in closest_work
