from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

NomaRole = Literal["strong", "weak", "solo"]


@dataclass(frozen=True, slots=True)
class Task:
    task_id: int
    user_id: int
    arrival_slot: int
    data_bits: float
    comp_cycles: float
    deadline_s: float
    local_cpu_hz: float

    def __post_init__(self) -> None:
        if min(self.data_bits, self.comp_cycles, self.deadline_s, self.local_cpu_hz) <= 0:
            raise ValueError("task quantities must be positive")

    @property
    def local_delay_s(self) -> float:
        return self.comp_cycles / self.local_cpu_hz

    @property
    def absolute_deadline_s(self) -> float:
        return self.arrival_slot + self.deadline_s

    @property
    def requires_offload(self) -> bool:
        return self.local_delay_s > self.deadline_s


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    task_id: int
    user_id: int
    serving_uav: int
    subcarrier: int
    noma_role: NomaRole


@dataclass(frozen=True, slots=True)
class Schedule:
    entries: tuple[ScheduleEntry, ...] = ()

    def stable_key(self) -> tuple[tuple[int, int, int, int, str], ...]:
        return tuple(
            sorted(
                (
                    entry.serving_uav,
                    entry.subcarrier,
                    entry.user_id,
                    entry.task_id,
                    entry.noma_role,
                )
                for entry in self.entries
            )
        )


@dataclass(frozen=True, slots=True)
class SlotSnapshot:
    slot_index: int
    tasks: tuple[Task, ...]
    channel_gains: np.ndarray
    num_uavs: int

    def __post_init__(self) -> None:
        expected = (len(self.tasks), self.num_uavs)
        gains = np.array(self.channel_gains, dtype=float, copy=True)
        if gains.shape != expected:
            raise ValueError(f"channel_gains must have shape {expected}")
        if not np.isfinite(gains).all() or (gains < 0).any():
            raise ValueError("channel_gains must be finite and non-negative")
        gains.setflags(write=False)
        object.__setattr__(self, "channel_gains", gains)


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    task_id: int
    alpha: float
    cpu_hz: float
    completed: bool
    delay_s: float | None
    slack_s: float
    reason: str


@dataclass(frozen=True, slots=True)
class RateRecord:
    task_id: int
    rate_bps: float
    sinr: float
    sic_feasible: bool
    cross_interference_w: float
    serving_uav: int
    subcarrier: int


@dataclass(frozen=True, slots=True)
class SlotOutcome:
    slot_index: int
    arrived_task_ids: tuple[int, ...]
    completed_task_ids: tuple[int, ...]
    missed_task_ids: tuple[int, ...]
    decisions: tuple[AllocationDecision, ...]
    rates: tuple[RateRecord, ...]
    total_slack_s: float
    total_cross_interference_w: float
    runtime_s: float = 0.0

    @property
    def completed_count(self) -> int:
        return len(self.completed_task_ids)
