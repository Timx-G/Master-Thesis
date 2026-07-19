"""Canonical, unit-safe task latency formulas for thesis-v2."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .domain import Task


@dataclass(frozen=True, slots=True)
class BranchDelays:
    local_s: float
    uplink_s: float
    mec_s: float
    downlink_s: float

    @property
    def edge_s(self) -> float:
        return math.fsum((self.uplink_s, self.mec_s, self.downlink_s))

    @property
    def completion_s(self) -> float:
        return max(self.local_s, self.edge_s)


def branch_delays_s(task: Task, alpha: float, uplink_rate_bps: float,
                    mec_cpu_hz: float, downlink_rate_bps: float | None) -> BranchDelays:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if alpha == 0.0:
        return BranchDelays(task.local_delay_s, 0.0, 0.0, 0.0)
    if not math.isfinite(uplink_rate_bps) or uplink_rate_bps <= 0.0:
        raise ValueError("uplink_rate_bps must be finite and positive")
    if not math.isfinite(mec_cpu_hz) or mec_cpu_hz <= 0.0:
        raise ValueError("mec_cpu_hz must be finite and positive")
    if task.result_ratio > 0.0 and (downlink_rate_bps is None or
        not math.isfinite(downlink_rate_bps) or downlink_rate_bps <= 0.0):
        raise ValueError("downlink_rate_bps is required for positive result_ratio")
    return BranchDelays(
        local_s=(1.0 - alpha) * task.comp_cycles / task.local_cpu_hz,
        uplink_s=alpha * task.data_bits / uplink_rate_bps,
        mec_s=alpha * task.comp_cycles / mec_cpu_hz,
        downlink_s=0.0 if task.result_ratio == 0.0 else
            alpha * task.result_ratio * task.data_bits / float(downlink_rate_bps),
    )


def completion_delay_s(task: Task, alpha: float, uplink_rate_bps: float,
                       mec_cpu_hz: float, downlink_rate_bps: float | None = None) -> float:
    return branch_delays_s(task, alpha, uplink_rate_bps, mec_cpu_hz, downlink_rate_bps).completion_s
