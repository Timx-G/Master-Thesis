from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
from enum import Enum

import numpy as np

NomaRole = Literal["strong", "weak", "solo"]

class ChannelBasis(str, Enum):
    PLANNING = "planning"
    REALIZED = "realized"

@dataclass(frozen=True, slots=True)
class SicLayerCheck:
    task_id: int
    receiver_uav: int
    subcarrier: int
    decode_rank: int
    sinr: float
    threshold: float
    feasible: bool


@dataclass(frozen=True, order=True, slots=True)
class TaskIdentity:
    user_id: int
    job_index: int


def _reconstruct_channel_gains(
    shape: tuple[int, ...], immutable_bytes: bytes
) -> _ImmutableChannelGains:
    values = np.frombuffer(immutable_bytes, dtype=float).reshape(shape)
    return _ImmutableChannelGains(values)


class _ImmutableChannelGains(np.ndarray):
    def __new__(cls, values: np.ndarray) -> _ImmutableChannelGains:
        immutable_bytes = values.tobytes(order="C")
        array = np.frombuffer(immutable_bytes, dtype=float).reshape(values.shape)
        return array.view(cls)

    def __array_finalize__(self, source: object) -> None:
        if source is not None:
            np.ndarray.setflags(self, write=False)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> object:
        outputs = kwargs.get("out")
        if outputs is not None:
            output_values = outputs if isinstance(outputs, tuple) else (outputs,)
            if any(isinstance(output, _ImmutableChannelGains) for output in output_values):
                raise ValueError("channel_gains cannot be used as a ufunc output")
        plain_inputs = tuple(
            np.asarray(value) if isinstance(value, _ImmutableChannelGains) else value
            for value in inputs
        )
        return getattr(ufunc, method)(*plain_inputs, **kwargs)

    def __deepcopy__(self, memo: dict[int, object]) -> _ImmutableChannelGains:
        copied = _ImmutableChannelGains(np.asarray(self))
        memo[id(self)] = copied
        return copied

    def __reduce_ex__(self, protocol: int) -> tuple[object, tuple[tuple[int, ...], bytes]]:
        return (_reconstruct_channel_gains, (self.shape, self.tobytes(order="C")))

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"data", "dtype", "shape", "strides"}:
            raise AttributeError("channel_gains metadata is immutable")
        super().__setattr__(name, value)

    def setflags(
        self,
        write: bool | None = None,
        align: bool | None = None,
        uic: bool | None = None,
    ) -> None:
        if write is not None or align is not None or uic is not None:
            raise ValueError("channel_gains flags are immutable")

    def resize(self, *args: object, **kwargs: object) -> None:
        raise ValueError("channel_gains shape is immutable")


def _validated_channel_gains(
    values: np.ndarray,
    expected: tuple[int, int],
    name: str,
) -> _ImmutableChannelGains:
    raw_gains = np.asarray(values)
    if raw_gains.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    if np.iscomplexobj(raw_gains) or (
        raw_gains.dtype.kind == "O"
        and any(np.iscomplexobj(value) for value in raw_gains.flat)
    ):
        raise ValueError(f"{name} must be real")
    gains = np.array(raw_gains, dtype=float, copy=True)
    if not np.isfinite(gains).all() or (gains < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    return _ImmutableChannelGains(gains)


@dataclass(frozen=True, slots=True)
class Task:
    task_id: int
    user_id: int
    arrival_slot: int
    data_bits: float
    comp_cycles: float
    deadline_s: float
    local_cpu_hz: float
    job_index: int = 0
    slot_duration_s: float = 1.0
    result_ratio: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("task_id", self.task_id), ("user_id", self.user_id), ("arrival_slot", self.arrival_slot), ("job_index", self.job_index)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        quantities = (self.data_bits, self.comp_cycles, self.deadline_s, self.local_cpu_hz)
        if any(not math.isfinite(quantity) or quantity <= 0 for quantity in quantities):
            raise ValueError("task quantities must be positive and finite")
        if not math.isfinite(self.slot_duration_s) or self.slot_duration_s <= 0:
            raise ValueError("slot_duration_s must be positive and finite")
        if not math.isfinite(self.result_ratio) or not 0 <= self.result_ratio <= 1:
            raise ValueError("result_ratio must be finite in [0, 1]")

    @property
    def identity(self) -> TaskIdentity:
        return TaskIdentity(self.user_id, self.job_index)

    @property
    def release_time_s(self) -> float:
        return self.arrival_slot * self.slot_duration_s

    @property
    def remaining_time_s(self) -> float:
        return self.deadline_s

    @property
    def local_delay_s(self) -> float:
        return self.comp_cycles / self.local_cpu_hz

    @property
    def absolute_deadline_s(self) -> float:
        return self.release_time_s + self.remaining_time_s

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
    estimated_channel_gains: np.ndarray | None = None

    __hash__ = None

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        if len({task.task_id for task in tasks}) != len(tasks) or len({task.identity for task in tasks}) != len(tasks):
            raise ValueError("tasks must have unique task identity")
        expected = (len(tasks), self.num_uavs)
        gains = _validated_channel_gains(
            self.channel_gains,
            expected,
            "channel_gains",
        )
        estimated = (
            None
            if self.estimated_channel_gains is None
            else _validated_channel_gains(
                self.estimated_channel_gains,
                expected,
                "estimated_channel_gains",
            )
        )
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "channel_gains", gains)
        object.__setattr__(self, "estimated_channel_gains", estimated)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SlotSnapshot):
            return NotImplemented
        return (
            self.slot_index == other.slot_index
            and self.tasks == other.tasks
            and self.num_uavs == other.num_uavs
            and np.array_equal(self.channel_gains, other.channel_gains)
            and (
                self.estimated_channel_gains is None
                and other.estimated_channel_gains is None
                or self.estimated_channel_gains is not None
                and other.estimated_channel_gains is not None
                and np.array_equal(
                    self.estimated_channel_gains,
                    other.estimated_channel_gains,
                )
            )
        )

    @property
    def planning_channel_gains(self) -> np.ndarray:
        if self.estimated_channel_gains is None:
            return self.channel_gains
        return self.estimated_channel_gains

    def as_planning_snapshot(self) -> SlotSnapshot:
        return SlotSnapshot(
            self.slot_index,
            self.tasks,
            self.planning_channel_gains,
            self.num_uavs,
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
    channel_basis: ChannelBasis = ChannelBasis.PLANNING
    sic_layers: tuple[SicLayerCheck, ...] = ()


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

@dataclass(frozen=True, slots=True)
class RealizedSlotOutcome:
    plan_key: tuple = ()
    schedule: Schedule = Schedule()
    decisions: tuple[AllocationDecision, ...] = ()
    rates: tuple[RateRecord, ...] = ()
    completed_task_ids: tuple[int, ...] = ()
    sic_outage_task_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ExperimentTrace:
    seed: int
    slots: tuple[SlotSnapshot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots))


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int
    outcomes: tuple[SlotOutcome, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(
            task_id
            for outcome in self.outcomes
            for task_id in outcome.arrived_task_ids
        )

    @property
    def total_arrivals(self) -> int:
        return len(self.task_ids)

    @property
    def total_completed(self) -> int:
        return sum(outcome.completed_count for outcome in self.outcomes)

    @property
    def total_missed(self) -> int:
        return self.total_arrivals - self.total_completed

    @property
    def on_time_rate(self) -> float:
        return self.total_completed / max(self.total_arrivals, 1)
