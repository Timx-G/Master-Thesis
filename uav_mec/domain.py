from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

NomaRole = Literal["strong", "weak", "solo"]


class _ImmutableChannelGains(np.ndarray):
    def __new__(cls, values: np.ndarray) -> _ImmutableChannelGains:
        immutable_bytes = values.tobytes(order="C")
        array = np.frombuffer(immutable_bytes, dtype=float).reshape(values.shape)
        return array.view(cls)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"dtype", "shape", "strides"}:
            raise AttributeError("channel_gains metadata is immutable")
        super().__setattr__(name, value)

    def setflags(
        self,
        write: bool | None = None,
        align: bool | None = None,
        uic: bool | None = None,
    ) -> None:
        if write:
            raise ValueError("channel_gains is immutable")
        super().setflags(write=write, align=align, uic=uic)

    def resize(self, *args: object, **kwargs: object) -> None:
        raise ValueError("channel_gains shape is immutable")


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
        quantities = (self.data_bits, self.comp_cycles, self.deadline_s, self.local_cpu_hz)
        if any(not math.isfinite(quantity) or quantity <= 0 for quantity in quantities):
            raise ValueError("task quantities must be positive and finite")

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))

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


@dataclass(frozen=True, slots=True, eq=False)
class SlotSnapshot:
    slot_index: int
    tasks: tuple[Task, ...]
    channel_gains: np.ndarray
    num_uavs: int

    __hash__ = None

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        raw_gains = np.asarray(self.channel_gains)
        expected = (len(tasks), self.num_uavs)
        if raw_gains.shape != expected:
            raise ValueError(f"channel_gains must have shape {expected}")
        if np.iscomplexobj(raw_gains) or (
            raw_gains.dtype.kind == "O"
            and any(np.iscomplexobj(value) for value in raw_gains.flat)
        ):
            raise ValueError("channel_gains must be real")
        gains = np.array(raw_gains, dtype=float, copy=True)
        if not np.isfinite(gains).all() or (gains < 0).any():
            raise ValueError("channel_gains must be finite and non-negative")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "channel_gains", _ImmutableChannelGains(gains))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SlotSnapshot):
            return NotImplemented
        return (
            self.slot_index == other.slot_index
            and self.tasks == other.tasks
            and self.num_uavs == other.num_uavs
            and np.array_equal(self.channel_gains, other.channel_gains)
        )


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "arrived_task_ids", tuple(self.arrived_task_ids))
        object.__setattr__(self, "completed_task_ids", tuple(self.completed_task_ids))
        object.__setattr__(self, "missed_task_ids", tuple(self.missed_task_ids))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "rates", tuple(self.rates))

    @property
    def completed_count(self) -> int:
        return len(self.completed_task_ids)
