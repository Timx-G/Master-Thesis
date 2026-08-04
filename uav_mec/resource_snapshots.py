"""Immutable raw-radio snapshots for resource-policy comparisons."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import Enum

from .domain import (
    AllocationDecision,
    ChannelBasis,
    RateRecord,
    Schedule,
    ScheduleEntry,
    SlotOutcome,
    SlotSnapshot,
)
from .evaluation import EvaluationConfig, build_outcome_from_rates, compute_rate_records
from .externality import aggregate_victim_link_exposure
from .latency import completion_delay_s
from .model import NOMAChannel
from .resource_allocation import AllocationPolicy, exact_admission
from .schedule import normalize_roles, serving_uav_by_task
from .scheduler import repair_sic


class RadioGenerator(str, Enum):
    DD_GENERATED = "dd_generated"
    RADIO_NEUTRAL = "radio_neutral"
    DEADLINE_FIRST_GENERATED = "deadline_first_generated"


@dataclass(frozen=True, slots=True)
class RawRadioSnapshot:
    snapshot_id: str
    seed: int
    slot: int
    generator: RadioGenerator
    schedule: Schedule
    planned_rates: tuple[RateRecord, ...]
    realized_rates: tuple[RateRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.slot, int) or isinstance(self.slot, bool) or self.slot < 0:
            raise ValueError("slot must be a non-negative integer")
        if not isinstance(self.generator, RadioGenerator):
            raise ValueError("generator must be a RadioGenerator")
        if not isinstance(self.schedule, Schedule):
            raise TypeError("schedule must be a Schedule")
        object.__setattr__(self, "planned_rates", tuple(self.planned_rates))
        object.__setattr__(self, "realized_rates", tuple(self.realized_rates))
        if len(self.snapshot_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.snapshot_id
        ):
            raise ValueError("snapshot_id must be a lowercase SHA-256 digest")
        schedule_ids = set(self.schedule.task_ids)
        if schedule_ids != {row.task_id for row in self.planned_rates} or schedule_ids != {
            row.task_id for row in self.realized_rates
        }:
            raise ValueError("schedule and rate snapshots must contain identical task IDs")
        if any(row.channel_basis is not ChannelBasis.PLANNING for row in self.planned_rates) or any(
            row.channel_basis is not ChannelBasis.REALIZED for row in self.realized_rates
        ):
            raise ValueError("rate snapshot channel basis mismatch")
        expected = hashlib.sha256(
            _snapshot_payload(
                self.seed,
                self.slot,
                self.generator,
                self.schedule,
                self.planned_rates,
                self.realized_rates,
            )
        ).hexdigest()
        if self.snapshot_id != expected:
            raise ValueError("snapshot_id does not match snapshot content")


def _wireless_only_schedule(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
    *,
    deadline_first: bool,
) -> Schedule:
    """Build a deterministic schedule without consulting any CPU policy."""
    gains = snapshot.planning_channel_gains
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}

    def task_key(task):
        row = row_by_task[task.task_id]
        best_gain = max(float(value) for value in gains[row])
        prefix = (task.absolute_deadline_s,) if deadline_first else ()
        return (*prefix, -best_gain, task.task_id)

    occupancy = {
        (uav, subcarrier): 0
        for uav in range(snapshot.num_uavs)
        for subcarrier in range(evaluation.n_subcarriers)
    }
    entries: list[ScheduleEntry] = []
    for task in sorted(
        (task for task in snapshot.tasks if task.requires_offload),
        key=task_key,
    ):
        row = row_by_task[task.task_id]
        uavs = sorted(
            range(snapshot.num_uavs),
            key=lambda uav: (-float(gains[row, uav]), uav),
        )
        available = [
            (occupancy[(uav, subcarrier)], uav, subcarrier)
            for uav in uavs
            for subcarrier in range(evaluation.n_subcarriers)
            if occupancy[(uav, subcarrier)] < 2
        ]
        if not available:
            break
        _load, uav, subcarrier = min(
            available,
            key=lambda item: (uavs.index(item[1]), item[0], item[2]),
        )
        entries.append(ScheduleEntry(task.task_id, task.user_id, uav, subcarrier, "solo"))
        occupancy[(uav, subcarrier)] += 1
    schedule = normalize_roles(
        Schedule(tuple(entries)), snapshot, channel_basis=ChannelBasis.PLANNING
    )
    return repair_sic(schedule, snapshot, channel, evaluation)


def build_radio_neutral_schedule(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
) -> Schedule:
    """Return the CPU-policy-neutral strongest-link wireless generator."""
    return _wireless_only_schedule(snapshot, channel, evaluation, deadline_first=False)


def build_deadline_first_schedule(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
) -> Schedule:
    """Return the CPU-policy-neutral earliest-deadline wireless generator."""
    return _wireless_only_schedule(snapshot, channel, evaluation, deadline_first=True)


def _rate_dict(record: RateRecord) -> dict[str, object]:
    return {
        "task_id": int(record.task_id),
        "rate_bps": float(record.rate_bps),
        "sinr": float(record.sinr),
        "sic_feasible": bool(record.sic_feasible),
        "cross_interference_w": float(record.cross_interference_w),
        "serving_uav": int(record.serving_uav),
        "subcarrier": int(record.subcarrier),
        "channel_basis": record.channel_basis.value,
        "sic_layers": [
            {
                "task_id": int(layer.task_id),
                "receiver_uav": int(layer.receiver_uav),
                "subcarrier": int(layer.subcarrier),
                "decode_rank": int(layer.decode_rank),
                "sinr": float(layer.sinr),
                "threshold": float(layer.threshold),
                "feasible": bool(layer.feasible),
            }
            for layer in record.sic_layers
        ],
    }


def _snapshot_payload(
    seed: int,
    slot: int,
    generator: RadioGenerator,
    schedule: Schedule,
    planned_rates: tuple[RateRecord, ...],
    realized_rates: tuple[RateRecord, ...],
) -> bytes:
    value = {
        "seed": seed,
        "slot": slot,
        "generator": generator.value,
        "schedule": [
            {
                "task_id": entry.task_id,
                "user_id": entry.user_id,
                "serving_uav": entry.serving_uav,
                "subcarrier": entry.subcarrier,
                "noma_role": entry.noma_role,
            }
            for entry in sorted(schedule.entries, key=lambda item: item.task_id)
        ],
        "planned_rates": [
            _rate_dict(row) for row in sorted(planned_rates, key=lambda row: row.task_id)
        ],
        "realized_rates": [
            _rate_dict(row) for row in sorted(realized_rates, key=lambda row: row.task_id)
        ],
    }
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def raw_snapshot_to_dict(raw: RawRadioSnapshot) -> dict[str, object]:
    """Return the exact canonical content represented by ``snapshot_id``."""
    value = json.loads(
        _snapshot_payload(
            raw.seed,
            raw.slot,
            raw.generator,
            raw.schedule,
            raw.planned_rates,
            raw.realized_rates,
        )
    )
    value["snapshot_id"] = raw.snapshot_id
    return value


def freeze_radio_snapshot(
    seed: int,
    snapshot: SlotSnapshot,
    generator: RadioGenerator,
    schedule: Schedule,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
) -> RawRadioSnapshot:
    """Evaluate both channel bases once and freeze their raw bytes."""
    if not isinstance(generator, RadioGenerator):
        raise ValueError("generator must be a RadioGenerator")
    planned = compute_rate_records(
        snapshot, schedule, channel, evaluation, channel_basis=ChannelBasis.PLANNING
    )
    realized = compute_rate_records(
        snapshot, schedule, channel, evaluation, channel_basis=ChannelBasis.REALIZED
    )
    if any(not row.sic_feasible for row in planned):
        raise ValueError("raw schedule is not wirelessly feasible on the planning basis")
    payload = _snapshot_payload(seed, snapshot.slot_index, generator, schedule, planned, realized)
    return RawRadioSnapshot(
        hashlib.sha256(payload).hexdigest(),
        seed,
        snapshot.slot_index,
        generator,
        schedule,
        planned,
        realized,
    )


def _exact_outcome(
    snapshot: SlotSnapshot,
    raw: RawRadioSnapshot,
    rates: tuple[RateRecord, ...],
    evaluation: EvaluationConfig,
) -> SlotOutcome:
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    rate_by_id = {record.task_id: record.rate_bps for record in rates}
    uav_by_task = serving_uav_by_task(raw.schedule)
    decisions: dict[int, AllocationDecision] = {}
    for uav in range(snapshot.num_uavs):
        tasks = tuple(
            task_by_id[task_id]
            for task_id, serving_uav in sorted(uav_by_task.items())
            if serving_uav == uav
        )
        result = exact_admission(tasks, rate_by_id, evaluation.cpu_per_uav_hz)
        admitted = set(result.admitted_task_ids)
        for task in tasks:
            requirement = result.requirements[task.task_id]
            if task.task_id in admitted:
                delay = completion_delay_s(
                    task,
                    requirement.alpha,
                    rate_by_id[task.task_id],
                    requirement.cpu_hz,
                )
                decisions[task.task_id] = AllocationDecision(
                    task.task_id,
                    requirement.alpha,
                    requirement.cpu_hz,
                    True,
                    delay,
                    task.deadline_s - delay,
                    requirement.reason.value,
                )
            else:
                decisions[task.task_id] = AllocationDecision(
                    task.task_id,
                    requirement.alpha,
                    0.0,
                    False,
                    None,
                    0.0,
                    "cpu_budget",
                )
    for task in snapshot.tasks:
        if task.task_id in decisions:
            continue
        completed = not task.requires_offload
        decisions[task.task_id] = AllocationDecision(
            task.task_id,
            0.0,
            0.0,
            completed,
            task.local_delay_s if completed else None,
            task.deadline_s - task.local_delay_s if completed else 0.0,
            "local" if completed else "unscheduled",
        )
    ordered = tuple(decisions[key] for key in sorted(decisions))
    return SlotOutcome(
        snapshot.slot_index,
        tuple(sorted(task_by_id)),
        tuple(row.task_id for row in ordered if row.completed),
        tuple(row.task_id for row in ordered if not row.completed),
        ordered,
        rates,
        math.fsum(row.slack_s for row in ordered if row.completed),
        aggregate_victim_link_exposure(rates),
    )


def evaluate_frozen_resource_snapshot(
    snapshot: SlotSnapshot,
    raw: RawRadioSnapshot,
    policy: AllocationPolicy | str,
    evaluation: EvaluationConfig,
    *,
    channel_basis: ChannelBasis = ChannelBasis.REALIZED,
) -> SlotOutcome:
    """Vary only CPU allocation over one immutable raw radio snapshot."""
    if snapshot.slot_index != raw.slot:
        raise ValueError("snapshot slot does not match raw radio snapshot")
    if channel_basis is ChannelBasis.PLANNING:
        rates = raw.planned_rates
    elif channel_basis is ChannelBasis.REALIZED:
        rates = raw.realized_rates
    else:
        raise ValueError("channel_basis must be planning or realized")
    if policy == "exact_admission":
        return _exact_outcome(snapshot, raw, rates, evaluation)
    if not isinstance(policy, AllocationPolicy):
        raise ValueError("policy must be an AllocationPolicy or exact_admission")
    outcome = build_outcome_from_rates(
        snapshot,
        serving_uav_by_task(raw.schedule),
        rates,
        replace(evaluation, allocation_policy=policy),
    )
    return replace(outcome, rates=rates)
