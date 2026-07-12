"""Single-slot radio, SIC, deadline, and MEC evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .domain import (
    AllocationDecision,
    RateRecord,
    Schedule,
    ScheduleEntry,
    SlotOutcome,
    SlotSnapshot,
)
from .model import NOMAChannel
from .resource_allocation import allocate_deadline_tasks
from .schedule import ScheduleValidationError, validate_schedule


class InfeasibleScheduleError(ValueError):
    """Raised when a schedule cannot satisfy structural or hard SIC constraints."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    n_subcarriers: int
    bandwidth_hz: float
    transmit_power_w: float
    sic_residual: float
    gamma_sic_min: float
    cpu_per_uav_hz: float
    result_ratio: float

    def __post_init__(self) -> None:
        if not isinstance(self.n_subcarriers, int) or isinstance(self.n_subcarriers, bool):
            raise ValueError("n_subcarriers must be an integer")
        if self.n_subcarriers <= 0:
            raise ValueError("n_subcarriers must be positive")
        self._require_positive("bandwidth_hz", self.bandwidth_hz)
        self._require_positive("transmit_power_w", self.transmit_power_w)
        self._require_positive("gamma_sic_min", self.gamma_sic_min)
        self._require_positive("cpu_per_uav_hz", self.cpu_per_uav_hz)
        if not math.isfinite(self.sic_residual) or not 0.0 <= self.sic_residual <= 1.0:
            raise ValueError("sic_residual must be finite and in [0, 1]")
        if not math.isfinite(self.result_ratio) or self.result_ratio < 0.0:
            raise ValueError("result_ratio must be finite and non-negative")

    @staticmethod
    def _require_positive(name: str, value: float) -> None:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")


def _cross_interference_w(
    target_uav: int,
    subcarrier: int,
    groups: dict[tuple[int, int], list[ScheduleEntry]],
    row_by_task: dict[int, int],
    snapshot: SlotSnapshot,
    transmit_power_w: float,
) -> float:
    powers = (
        transmit_power_w * snapshot.channel_gains[row_by_task[entry.task_id], target_uav]
        for (other_uav, other_subcarrier), entries in groups.items()
        if other_uav != target_uav and other_subcarrier == subcarrier
        for entry in entries
    )
    try:
        return math.fsum(powers)
    except OverflowError as exc:
        raise InfeasibleScheduleError("cross-UAV interference is not representable") from exc


def _rate_bps(bandwidth_hz: float, sinr: float) -> float:
    return bandwidth_hz * math.log1p(sinr) / math.log(2.0)


def compute_rate_records(
    snapshot: SlotSnapshot,
    schedule: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> tuple[RateRecord, ...]:
    """Evaluate uplink rates on each entry's actual serving UAV."""
    validate_schedule(schedule, snapshot, cfg.n_subcarriers)
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    groups: dict[tuple[int, int], list[ScheduleEntry]] = defaultdict(list)
    for entry in schedule.entries:
        groups[(entry.serving_uav, entry.subcarrier)].append(entry)

    bandwidth_hz = cfg.bandwidth_hz / cfg.n_subcarriers
    noise_w = channel.noise_for_bandwidth(bandwidth_hz)
    records: list[RateRecord] = []
    for (uav, subcarrier), own_entries in sorted(groups.items()):
        cross_w = _cross_interference_w(
            uav,
            subcarrier,
            groups,
            row_by_task,
            snapshot,
            cfg.transmit_power_w,
        )
        if len(own_entries) == 1:
            entry = own_entries[0]
            received_w = (
                cfg.transmit_power_w
                * snapshot.channel_gains[row_by_task[entry.task_id], uav]
            )
            sinr = received_w / (cross_w + noise_w)
            records.append(
                RateRecord(
                    entry.task_id,
                    _rate_bps(bandwidth_hz, sinr),
                    sinr,
                    True,
                    cross_w,
                    uav,
                    subcarrier,
                )
            )
            continue

        by_role = {entry.noma_role: entry for entry in own_entries}
        strong = by_role["strong"]
        weak = by_role["weak"]
        received_strong_w = (
            cfg.transmit_power_w
            * snapshot.channel_gains[row_by_task[strong.task_id], uav]
        )
        received_weak_w = (
            cfg.transmit_power_w
            * snapshot.channel_gains[row_by_task[weak.task_id], uav]
        )
        if received_strong_w < received_weak_w:
            raise InfeasibleScheduleError(
                "strong/weak order does not match the serving-UAV channel"
            )

        sinr_strong = received_strong_w / (received_weak_w + cross_w + noise_w)
        sinr_weak = received_weak_w / (
            cfg.sic_residual * received_strong_w + cross_w + noise_w
        )
        sic_feasible = sinr_strong >= cfg.gamma_sic_min
        records.extend(
            (
                RateRecord(
                    strong.task_id,
                    _rate_bps(bandwidth_hz, sinr_strong),
                    sinr_strong,
                    sic_feasible,
                    cross_w,
                    uav,
                    subcarrier,
                ),
                RateRecord(
                    weak.task_id,
                    _rate_bps(bandwidth_hz, sinr_weak),
                    sinr_weak,
                    sic_feasible,
                    cross_w,
                    uav,
                    subcarrier,
                ),
            )
        )

    return tuple(sorted(records, key=lambda record: record.task_id))


def evaluate_schedule(
    snapshot: SlotSnapshot,
    schedule: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> SlotOutcome:
    """Return one decision per arrived task for a candidate schedule."""
    try:
        rates = compute_rate_records(snapshot, schedule, channel, cfg)
    except ScheduleValidationError as exc:
        raise InfeasibleScheduleError(str(exc)) from exc

    sic_violations = [record.task_id for record in rates if not record.sic_feasible]
    if sic_violations:
        raise InfeasibleScheduleError(f"SIC threshold violated by tasks {sic_violations}")

    task_by_id = {task.task_id: task for task in snapshot.tasks}
    rate_by_id = {record.task_id: record.rate_bps for record in rates}
    entries_by_uav: dict[int, list[int]] = defaultdict(list)
    for entry in schedule.entries:
        entries_by_uav[entry.serving_uav].append(entry.task_id)

    decisions: dict[int, AllocationDecision] = {}
    for task in snapshot.tasks:
        if not task.requires_offload:
            decisions[task.task_id] = AllocationDecision(
                task.task_id,
                0.0,
                0.0,
                True,
                task.local_delay_s,
                task.deadline_s - task.local_delay_s,
                "local",
            )

    for uav in range(snapshot.num_uavs):
        scheduled_tasks = tuple(
            task_by_id[task_id] for task_id in entries_by_uav.get(uav, ())
        )
        allocation = allocate_deadline_tasks(
            scheduled_tasks,
            rate_by_id,
            cfg.cpu_per_uav_hz,
            cfg.result_ratio,
        )
        decisions.update(
            {decision.task_id: decision for decision in allocation.decisions}
        )

    for task in snapshot.tasks:
        if task.task_id not in decisions:
            decisions[task.task_id] = AllocationDecision(
                task.task_id,
                0.0,
                0.0,
                False,
                None,
                0.0,
                "unscheduled",
            )

    ordered = tuple(decisions[task_id] for task_id in sorted(decisions))
    completed = tuple(
        decision.task_id for decision in ordered if decision.completed
    )
    missed = tuple(
        decision.task_id for decision in ordered if not decision.completed
    )
    # This is a task-link aggregate: both users in a NOMA pair experience the
    # receiver's external interference through their respective rate equations.
    total_cross_interference_w = math.fsum(
        record.cross_interference_w for record in rates
    )
    return SlotOutcome(
        slot_index=snapshot.slot_index,
        arrived_task_ids=tuple(sorted(task_by_id)),
        completed_task_ids=completed,
        missed_task_ids=missed,
        decisions=ordered,
        rates=rates,
        total_slack_s=math.fsum(
            decision.slack_s for decision in ordered if decision.completed
        ),
        total_cross_interference_w=total_cross_interference_w,
    )
