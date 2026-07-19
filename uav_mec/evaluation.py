"""Single-slot radio, SIC, deadline, and MEC evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

from .domain import (
    AllocationDecision,
    RateRecord,
    Schedule,
    ScheduleEntry,
    SlotOutcome,
    SlotSnapshot,
    ChannelBasis,
    SicLayerCheck,
)
from .model import NOMAChannel
from .resource_allocation import AllocationPolicy, allocate_with_policy
from .schedule import (
    ScheduleValidationError,
    normalize_roles,
    serving_uav_by_task,
    validate_schedule,
)


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
    allocation_policy: AllocationPolicy = AllocationPolicy.DEADLINE
    cross_interference_scale: float = 1.0

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
        if not isinstance(self.allocation_policy, AllocationPolicy):
            raise ValueError("allocation_policy must be an AllocationPolicy value")
        if (
            not math.isfinite(self.cross_interference_scale)
            or not 0.0 <= self.cross_interference_scale <= 1.0
        ):
            raise ValueError("cross_interference_scale must be in [0, 1]")

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
    gains,
) -> float:
    powers = (
        transmit_power_w * gains[row_by_task[entry.task_id], target_uav]
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
    *,
    channel_basis: ChannelBasis = ChannelBasis.PLANNING,
) -> tuple[RateRecord, ...]:
    """Evaluate uplink rates on each entry's actual serving UAV."""
    if not isinstance(channel_basis, ChannelBasis):
        raise ValueError("channel_basis must be a ChannelBasis value")
    validate_schedule(schedule, snapshot, cfg.n_subcarriers)
    gains = snapshot.channel_gains_for(channel_basis)
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
            gains,
        )
        cross_w *= cfg.cross_interference_scale
        if len(own_entries) == 1:
            entry = own_entries[0]
            received_w = (
                cfg.transmit_power_w
                * gains[row_by_task[entry.task_id], uav]
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
                    channel_basis,
                    (SicLayerCheck(entry.task_id, uav, subcarrier, 0, sinr, cfg.gamma_sic_min, sinr >= cfg.gamma_sic_min),),
                )
            )
            continue

        by_role = {entry.noma_role: entry for entry in own_entries}
        strong = by_role["strong"]
        weak = by_role["weak"]
        received_strong_w = (
            cfg.transmit_power_w
            * gains[row_by_task[strong.task_id], uav]
        )
        received_weak_w = (
            cfg.transmit_power_w
            * gains[row_by_task[weak.task_id], uav]
        )
        if channel_basis is ChannelBasis.PLANNING and received_strong_w < received_weak_w:
            raise InfeasibleScheduleError(
                "strong/weak order does not match the serving-UAV channel"
            )

        received_w = (received_strong_w, received_weak_w)
        layers = tuple(
            SicLayerCheck(
                entry.task_id, uav, subcarrier, q,
                received_w[q] / math.fsum((noise_w, cross_w, math.fsum(received_w[q + 1:]), cfg.sic_residual * math.fsum(received_w[:q]))),
                cfg.gamma_sic_min,
                received_w[q] / math.fsum((noise_w, cross_w, math.fsum(received_w[q + 1:]), cfg.sic_residual * math.fsum(received_w[:q]))) >= cfg.gamma_sic_min,
            )
            for q, entry in enumerate((strong, weak))
        )
        sinr_strong, sinr_weak = (layer.sinr for layer in layers)
        sic_feasible = all(layer.feasible for layer in layers)
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
                    channel_basis,
                    (layers[0],),
                ),
                RateRecord(
                    weak.task_id,
                    _rate_bps(bandwidth_hz, sinr_weak),
                    sinr_weak,
                    sic_feasible,
                    cross_w,
                    uav,
                    subcarrier,
                    channel_basis,
                    (layers[1],),
                ),
            )
        )

    return tuple(sorted(records, key=lambda record: record.task_id))


def build_outcome_from_rates(
    snapshot: SlotSnapshot,
    uav_by_task: Mapping[int, int],
    rates: tuple[RateRecord, ...],
    cfg: EvaluationConfig,
) -> SlotOutcome:
    """Apply shared task accounting and per-serving-UAV CPU allocation."""
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    ordered_rates = tuple(sorted(rates, key=lambda record: record.task_id))
    rate_by_id = {
        record.task_id: record.rate_bps for record in ordered_rates
    }
    if len(rate_by_id) != len(ordered_rates):
        raise ValueError("duplicate rate record")
    if set(rate_by_id) != set(uav_by_task):
        raise ValueError(
            "rate records and serving-UAV mapping must have identical task IDs"
        )
    if any(task_id not in task_by_id for task_id in uav_by_task):
        raise ValueError("serving-UAV mapping contains an unknown task")
    if any(
        not isinstance(serving_uav, Integral)
        or isinstance(serving_uav, bool)
        or not 0 <= serving_uav < snapshot.num_uavs
        for serving_uav in uav_by_task.values()
    ):
        raise ValueError("serving-UAV mapping contains an invalid UAV")
    for record in ordered_rates:
        if record.serving_uav != uav_by_task[record.task_id]:
            raise ValueError("rate record and CPU mapping disagree on serving UAV")
        if (
            not isinstance(record.subcarrier, Integral)
            or isinstance(record.subcarrier, bool)
            or not 0 <= record.subcarrier < cfg.n_subcarriers
        ):
            raise ValueError("rate record contains an invalid subcarrier")
        if not record.sic_feasible:
            raise ValueError("rate record violates the hard SIC constraint")
        if (
            not math.isfinite(record.rate_bps)
            or record.rate_bps < 0.0
            or not math.isfinite(record.sinr)
            or record.sinr < 0.0
            or not math.isfinite(record.cross_interference_w)
            or record.cross_interference_w < 0.0
        ):
            raise ValueError("rate record contains a nonphysical value")
        if not task_by_id[record.task_id].requires_offload:
            raise ValueError("local-feasible task must not have a radio rate record")

    entries_by_uav: dict[int, list[int]] = defaultdict(list)
    for task_id, serving_uav in uav_by_task.items():
        entries_by_uav[serving_uav].append(task_id)

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
        allocation = allocate_with_policy(
            cfg.allocation_policy,
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
        record.cross_interference_w for record in ordered_rates
    )
    return SlotOutcome(
        slot_index=snapshot.slot_index,
        arrived_task_ids=tuple(sorted(task_by_id)),
        completed_task_ids=completed,
        missed_task_ids=missed,
        decisions=ordered,
        rates=ordered_rates,
        total_slack_s=math.fsum(
            decision.slack_s for decision in ordered if decision.completed
        ),
        total_cross_interference_w=total_cross_interference_w,
    )


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

    sic_violations = [
        record.task_id for record in rates if not record.sic_feasible
    ]
    if sic_violations:
        raise InfeasibleScheduleError(
            f"SIC threshold violated by tasks {sic_violations}"
        )
    return build_outcome_from_rates(
        snapshot,
        serving_uav_by_task(schedule),
        rates,
        cfg,
    )


def evaluate_realized_schedule(
    snapshot: SlotSnapshot,
    planned_schedule: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> SlotOutcome:
    """Evaluate a planned schedule on truth, recording actual SIC outages.

    The receiver may reorder a planned pair using its realized channel, but no
    user association or resource-block decision is changed. A pair that still
    violates the hard threshold is excluded from MEC allocation and counted as
    a miss while its transmission remains in every other link's interference.
    """
    try:
        realized_schedule = normalize_roles(planned_schedule, snapshot)
        rates = compute_rate_records(
            snapshot,
            realized_schedule,
            channel,
            cfg,
        )
    except ScheduleValidationError as exc:
        raise InfeasibleScheduleError(str(exc)) from exc

    outage_ids = {
        record.task_id for record in rates if not record.sic_feasible
    }
    if not outage_ids:
        return build_outcome_from_rates(
            snapshot,
            serving_uav_by_task(realized_schedule),
            rates,
            cfg,
        )

    realized_uav = serving_uav_by_task(realized_schedule)
    feasible_rates = tuple(
        record for record in rates if record.task_id not in outage_ids
    )
    feasible_uav = {
        task_id: uav
        for task_id, uav in realized_uav.items()
        if task_id not in outage_ids
    }
    base = build_outcome_from_rates(
        snapshot,
        feasible_uav,
        feasible_rates,
        cfg,
    )
    decisions = tuple(
        AllocationDecision(
            decision.task_id,
            0.0,
            0.0,
            False,
            None,
            0.0,
            "sic_outage",
        )
        if decision.task_id in outage_ids
        else decision
        for decision in base.decisions
    )
    return SlotOutcome(
        slot_index=snapshot.slot_index,
        arrived_task_ids=base.arrived_task_ids,
        completed_task_ids=tuple(
            decision.task_id for decision in decisions if decision.completed
        ),
        missed_task_ids=tuple(
            decision.task_id for decision in decisions if not decision.completed
        ),
        decisions=decisions,
        rates=rates,
        total_slack_s=math.fsum(
            decision.slack_s for decision in decisions if decision.completed
        ),
        total_cross_interference_w=math.fsum(
            record.cross_interference_w for record in rates
        ),
    )
