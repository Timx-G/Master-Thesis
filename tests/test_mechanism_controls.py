from dataclasses import replace

import numpy as np
import pytest

from uav_mec.domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, compute_rate_records
from uav_mec.model import ChannelConfig, NOMAChannel


def base_config() -> EvaluationConfig:
    return EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)


def test_zero_cross_scale_removes_only_cross_uav_interference():
    tasks = (
        Task(1, 0, 0, 1e6, 1.3e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.3e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[9e-8, 8e-8], [8e-8, 9e-8]]),
        2,
    )
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "solo"),
            ScheduleEntry(2, 1, 1, 0, "solo"),
        )
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)
    isolated = compute_rate_records(
        snapshot,
        schedule,
        channel,
        replace(base_config(), cross_interference_scale=0.0),
        channel_basis=ChannelBasis.PLANNING,
    )
    half = compute_rate_records(
        snapshot,
        schedule,
        channel,
        replace(base_config(), cross_interference_scale=0.5),
        channel_basis=ChannelBasis.PLANNING,
    )
    coupled = compute_rate_records(
        snapshot,
        schedule,
        channel,
        base_config(),
        channel_basis=ChannelBasis.PLANNING,
    )

    assert all(record.cross_interference_w == 0.0 for record in isolated)
    for zero, middle, full in zip(isolated, half, coupled, strict=True):
        assert middle.cross_interference_w == pytest.approx(
            0.5 * full.cross_interference_w
        )
        assert zero.rate_bps > middle.rate_bps > full.rate_bps


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("nan")])
def test_cross_interference_scale_rejects_nonphysical_values(scale):
    with pytest.raises(ValueError, match="cross_interference_scale"):
        replace(base_config(), cross_interference_scale=scale)
