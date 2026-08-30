"""Single-slot radio, SIC, deadline, and MEC evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

from .domain import (
    AllocationDecision,
    RateRecord,
    Schedule,
    ScheduleEntry,
    SlotOutcome,
    SlotSnapshot,
    Task,
)
from .model import NOMAChannel
from .resource_allocation import (
    AllocationPolicy,
    allocate_with_policy,
    allocation_score_with_policy,
)
from .schedule import (
    ScheduleValidationError,
    normalize_roles,
    serving_uav_by_task,
    validate_schedule,
)


class InfeasibleScheduleError(ValueError):
    """Raised when a schedule cannot satisfy structural or hard SIC constraints."""


@lru_cache(maxsize=262_144)
def _cached_allocation(
    policy: AllocationPolicy,
    tasks: tuple,
    rate_items: tuple[tuple[int, float], ...],
    cpu_budget_hz: float,
    result_ratio: float,
    allocator_identity: int,
):
    del allocator_identity
    return allocate_with_policy(
        policy,
        tasks,
        dict(rate_items),
        cpu_budget_hz,
        result_ratio,
    )


@lru_cache(maxsize=262_144)
def _cached_allocation_score(
    policy: AllocationPolicy,
    tasks: tuple,
    rate_items: tuple[tuple[int, float], ...],
    cpu_budget_hz: float,
    result_ratio: float,
    allocator_identity: int,
    screening_arithmetic: bool,
):
    del allocator_identity
    return allocation_score_with_policy(
        policy,
        tasks,
        dict(rate_items),
        cpu_budget_hz,
        result_ratio,
        prevalidated_offload_only=True,
        screening_arithmetic=screening_arithmetic,
    )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    n_subcarriers: int
    bandwidth_hz: float
    transmit_power_w: float
    sic_residual: float
    gamma_sic_min: float
    cpu_per_uav_hz: float | tuple[float, ...]
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
        if isinstance(self.cpu_per_uav_hz, (tuple, list)):
            if not self.cpu_per_uav_hz:
                raise ValueError("cpu_per_uav_hz must be non-empty")
            budgets = tuple(float(value) for value in self.cpu_per_uav_hz)
            for value in budgets:
                self._require_positive("cpu_per_uav_hz", value)
            object.__setattr__(self, "cpu_per_uav_hz", budgets)
        else:
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

    def cpu_budget_for(self, uav: int, num_uavs: int) -> float:
        if isinstance(self.cpu_per_uav_hz, tuple):
            if len(self.cpu_per_uav_hz) != num_uavs:
                raise ValueError(
                    "CPU budget vector length must match the number of UAVs"
                )
            return self.cpu_per_uav_hz[uav]
        return float(self.cpu_per_uav_hz)


@dataclass(frozen=True, slots=True)
class ScheduleScore:
    """Minimal exact objective state used during candidate screening."""

    completed_count: int
    total_mec_cpu_hz: float
    total_cross_interference_w: float
    rates: tuple[RateRecord, ...]


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Static per-slot values reused across exact candidate evaluations."""

    snapshot: SlotSnapshot
    channel: NOMAChannel
    config: EvaluationConfig
    task_by_id: dict[int, Task]
    received_power_by_task: dict[int, tuple[float, ...]]
    local_completed_count: int
    bandwidth_hz: float
    noise_w: float
    cpu_budgets_hz: tuple[float, ...]


def build_evaluation_context(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> EvaluationContext:
    """Precompute candidate-invariant values for one slot and configuration."""

    bandwidth_hz = cfg.bandwidth_hz / cfg.n_subcarriers
    return EvaluationContext(
        snapshot=snapshot,
        channel=channel,
        config=cfg,
        task_by_id={task.task_id: task for task in snapshot.tasks},
        received_power_by_task={
            task.task_id: tuple(
                float(cfg.transmit_power_w * snapshot.channel_gains[row, uav])
                for uav in range(snapshot.num_uavs)
            )
            for row, task in enumerate(snapshot.tasks)
        },
        local_completed_count=sum(not task.requires_offload for task in snapshot.tasks),
        bandwidth_hz=bandwidth_hz,
        noise_w=channel.noise_for_bandwidth(bandwidth_hz),
        cpu_budgets_hz=tuple(
            cfg.cpu_budget_for(uav, snapshot.num_uavs)
            for uav in range(snapshot.num_uavs)
        ),
    )


def _require_matching_context(
    context: EvaluationContext,
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> None:
    if (
        context.snapshot is not snapshot
        or context.channel is not channel
        or context.config is not cfg
    ):
        raise ValueError("evaluation context belongs to another slot or configuration")


def _cross_interference_w(
    target_uav: int,
    subcarrier: int,
    groups: dict[tuple[int, int], list[ScheduleEntry]],
    row_by_task: dict[int, int],
    snapshot: SlotSnapshot,
    transmit_power_w: float,
    received_power_by_task: dict[int, tuple[float, ...]] | None = None,
    entries_by_subcarrier: dict[int, list[ScheduleEntry]] | None = None,
) -> float:
    if received_power_by_task is None:
        powers = (
            transmit_power_w * snapshot.channel_gains[row_by_task[entry.task_id], target_uav]
            for (other_uav, other_subcarrier), entries in groups.items()
            if other_uav != target_uav and other_subcarrier == subcarrier
            for entry in entries
        )
    elif entries_by_subcarrier is None:
        powers = (
            received_power_by_task[entry.task_id][target_uav]
            for (other_uav, other_subcarrier), entries in groups.items()
            if other_uav != target_uav and other_subcarrier == subcarrier
            for entry in entries
        )
    else:
        powers = (
            received_power_by_task[entry.task_id][target_uav]
            for entry in entries_by_subcarrier.get(subcarrier, ())
            if entry.serving_uav != target_uav
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
    subcarriers: frozenset[int] | None = None,
    validate_structure: bool = True,
    context: EvaluationContext | None = None,
) -> tuple[RateRecord, ...]:
    """Evaluate uplink rates on each entry's actual serving UAV."""
    if validate_structure:
        validate_schedule(schedule, snapshot, cfg.n_subcarriers)
    if subcarriers is not None and any(
        not isinstance(subcarrier, int)
        or isinstance(subcarrier, bool)
        or not 0 <= subcarrier < cfg.n_subcarriers
        for subcarrier in subcarriers
    ):
        raise ValueError("subcarriers contains an invalid index")
    if context is not None:
        _require_matching_context(context, snapshot, channel, cfg)
    row_by_task = (
        {}
        if context is not None
        else {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    )
    groups: dict[tuple[int, int], list[ScheduleEntry]] = defaultdict(list)
    entries_by_subcarrier: dict[int, list[ScheduleEntry]] | None = (
        defaultdict(list) if context is not None else None
    )
    for entry in schedule.entries:
        groups[(entry.serving_uav, entry.subcarrier)].append(entry)
        if entries_by_subcarrier is not None:
            entries_by_subcarrier[entry.subcarrier].append(entry)

    bandwidth_hz = (
        context.bandwidth_hz
        if context is not None
        else cfg.bandwidth_hz / cfg.n_subcarriers
    )
    noise_w = (
        context.noise_w
        if context is not None
        else channel.noise_for_bandwidth(bandwidth_hz)
    )
    records: list[RateRecord] = []
    for (uav, subcarrier), own_entries in sorted(groups.items()):
        if subcarriers is not None and subcarrier not in subcarriers:
            continue
        cross_w = _cross_interference_w(
            uav,
            subcarrier,
            groups,
            row_by_task,
            snapshot,
            cfg.transmit_power_w,
            None if context is None else context.received_power_by_task,
            entries_by_subcarrier,
        )
        cross_w *= cfg.cross_interference_scale
        if len(own_entries) == 1:
            entry = own_entries[0]
            received_w = (
                cfg.transmit_power_w
                * snapshot.channel_gains[row_by_task[entry.task_id], uav]
                if context is None
                else context.received_power_by_task[entry.task_id][uav]
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
            if context is None
            else context.received_power_by_task[strong.task_id][uav]
        )
        received_weak_w = (
            cfg.transmit_power_w
            * snapshot.channel_gains[row_by_task[weak.task_id], uav]
            if context is None
            else context.received_power_by_task[weak.task_id][uav]
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
            task_by_id[task_id]
            for task_id in sorted(entries_by_uav.get(uav, ()))
        )
        rate_items = tuple(
            (task.task_id, rate_by_id[task.task_id])
            for task in scheduled_tasks
        )
        allocation = _cached_allocation(
            cfg.allocation_policy,
            scheduled_tasks,
            rate_items,
            cfg.cpu_budget_for(uav, snapshot.num_uavs),
            cfg.result_ratio,
            id(allocate_with_policy),
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
        total_mec_cpu_hz=math.fsum(
            decision.cpu_hz
            for decision in ordered
            if decision.completed and decision.alpha > 0.0
        ),
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


def evaluate_schedule_incremental(
    snapshot: SlotSnapshot,
    before_schedule: Schedule,
    before_outcome: SlotOutcome,
    candidate_schedule: Schedule,
    affected_blocks: tuple[tuple[int, int], ...],
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    *,
    validate_structure: bool = True,
) -> SlotOutcome:
    """Recompute all receivers on affected subcarriers and reuse other rates."""
    rates = _incremental_rate_records(
        snapshot,
        before_schedule,
        before_outcome,
        candidate_schedule,
        affected_blocks,
        channel,
        cfg,
        validate_structure=validate_structure,
    )
    return build_outcome_from_rates(
        snapshot,
        serving_uav_by_task(candidate_schedule),
        rates,
        cfg,
    )


def _incremental_rate_records(
    snapshot: SlotSnapshot,
    before_schedule: Schedule,
    before_outcome: SlotOutcome,
    candidate_schedule: Schedule,
    affected_blocks: tuple[tuple[int, int], ...],
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    *,
    validate_structure: bool,
    context: EvaluationContext | None = None,
) -> tuple[RateRecord, ...]:
    if validate_structure:
        validate_schedule(before_schedule, snapshot, cfg.n_subcarriers)
        validate_schedule(candidate_schedule, snapshot, cfg.n_subcarriers)
    if before_outcome.slot_index != snapshot.slot_index:
        raise ValueError("before outcome belongs to another slot")
    if not affected_blocks:
        raise ValueError("affected_blocks must be non-empty")
    affected_subcarriers = frozenset(subcarrier for _uav, subcarrier in affected_blocks)
    if any(
        not 0 <= uav < snapshot.num_uavs
        or not 0 <= subcarrier < cfg.n_subcarriers
        for uav, subcarrier in affected_blocks
    ):
        raise ValueError("affected_blocks contains an invalid resource block")

    if validate_structure:
        def unaffected_entries(schedule: Schedule) -> tuple[tuple[int, int, int], ...]:
            return tuple(
                sorted(
                    (entry.task_id, entry.serving_uav, entry.subcarrier)
                    for entry in schedule.entries
                    if entry.subcarrier not in affected_subcarriers
                )
            )

        if unaffected_entries(before_schedule) != unaffected_entries(candidate_schedule):
            raise ValueError("candidate changed a subcarrier outside affected_blocks")

    changed_rates = compute_rate_records(
        snapshot,
        candidate_schedule,
        channel,
        cfg,
        subcarriers=affected_subcarriers,
        validate_structure=False,
        context=context,
    )
    reused_rates = tuple(
        record
        for record in before_outcome.rates
        if record.subcarrier not in affected_subcarriers
    )
    rates = tuple(
        sorted(
            (*reused_rates, *changed_rates),
            key=lambda record: record.task_id,
        )
    )
    sic_violations = [record.task_id for record in rates if not record.sic_feasible]
    if sic_violations:
        raise InfeasibleScheduleError(
            f"SIC threshold violated by tasks {sic_violations}"
        )
    return rates


def score_schedule_incremental(
    snapshot: SlotSnapshot,
    before_schedule: Schedule,
    before_outcome: SlotOutcome,
    candidate_schedule: Schedule,
    affected_blocks: tuple[tuple[int, int], ...],
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    *,
    validate_structure: bool = True,
    context: EvaluationContext | None = None,
    retain_rates: bool = True,
) -> ScheduleScore:
    """Compute the exact lexicographic objective without full task decisions."""

    if not isinstance(retain_rates, bool):
        raise ValueError("retain_rates must be bool")
    if not retain_rates:
        screening_context = (
            build_evaluation_context(snapshot, channel, cfg)
            if context is None
            else context
        )
        return _score_schedule_without_rate_records(
            snapshot,
            before_schedule,
            before_outcome,
            candidate_schedule,
            affected_blocks,
            channel,
            cfg,
            screening_context,
            validate_structure=validate_structure,
        )

    rates = _incremental_rate_records(
        snapshot,
        before_schedule,
        before_outcome,
        candidate_schedule,
        affected_blocks,
        channel,
        cfg,
        validate_structure=validate_structure,
        context=context,
    )
    if context is not None:
        _require_matching_context(context, snapshot, channel, cfg)
        task_by_id = context.task_by_id
        local_completed_count = context.local_completed_count
        cpu_budgets_hz = context.cpu_budgets_hz
    else:
        task_by_id = {task.task_id: task for task in snapshot.tasks}
        local_completed_count = sum(not task.requires_offload for task in snapshot.tasks)
        cpu_budgets_hz = tuple(
            cfg.cpu_budget_for(uav, snapshot.num_uavs)
            for uav in range(snapshot.num_uavs)
        )
    rate_by_id = {record.task_id: record.rate_bps for record in rates}
    if validate_structure:
        scheduled_ids = {entry.task_id for entry in candidate_schedule.entries}
        if set(rate_by_id) != scheduled_ids:
            raise ValueError("rate records and serving-UAV mapping must have identical task IDs")

    entries_by_uav: dict[int, list[int]] = defaultdict(list)
    for entry in candidate_schedule.entries:
        if validate_structure and not task_by_id[entry.task_id].requires_offload:
            raise ValueError("local-feasible task must not have a radio rate record")
        entries_by_uav[entry.serving_uav].append(entry.task_id)

    completed_count = local_completed_count
    cpu_costs: list[float] = []
    for uav in range(snapshot.num_uavs):
        scheduled_tasks = tuple(
            task_by_id[task_id]
            for task_id in sorted(entries_by_uav.get(uav, ()))
        )
        rate_items = tuple(
            (task.task_id, rate_by_id[task.task_id]) for task in scheduled_tasks
        )
        score = _cached_allocation_score(
            cfg.allocation_policy,
            scheduled_tasks,
            rate_items,
            cpu_budgets_hz[uav],
            cfg.result_ratio,
            id(allocation_score_with_policy),
            False,
        )
        completed_count += score.completed_count
        cpu_costs.append(score.total_cpu_hz)
    return ScheduleScore(
        completed_count,
        math.fsum(cpu_costs),
        math.fsum(record.cross_interference_w for record in rates),
        rates,
    )


def _score_schedule_without_rate_records(
    snapshot: SlotSnapshot,
    before_schedule: Schedule,
    before_outcome: SlotOutcome,
    candidate_schedule: Schedule,
    affected_blocks: tuple[tuple[int, int], ...],
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    context: EvaluationContext,
    *,
    validate_structure: bool,
) -> ScheduleScore:
    """Screen one trusted candidate without retaining task-level radio objects."""

    _require_matching_context(context, snapshot, channel, cfg)
    if validate_structure:
        validate_schedule(before_schedule, snapshot, cfg.n_subcarriers)
        validate_schedule(candidate_schedule, snapshot, cfg.n_subcarriers)
    if before_outcome.slot_index != snapshot.slot_index:
        raise ValueError("before outcome belongs to another slot")
    if not affected_blocks:
        raise ValueError("affected_blocks must be non-empty")
    affected_subcarriers = frozenset(
        subcarrier for _uav, subcarrier in affected_blocks
    )
    if any(
        not 0 <= uav < snapshot.num_uavs
        or not 0 <= subcarrier < cfg.n_subcarriers
        for uav, subcarrier in affected_blocks
    ):
        raise ValueError("affected_blocks contains an invalid resource block")
    if validate_structure:
        def unaffected_entries(schedule: Schedule) -> tuple[tuple[int, int, int], ...]:
            return tuple(
                sorted(
                    (entry.task_id, entry.serving_uav, entry.subcarrier)
                    for entry in schedule.entries
                    if entry.subcarrier not in affected_subcarriers
                )
            )

        if unaffected_entries(before_schedule) != unaffected_entries(candidate_schedule):
            raise ValueError("candidate changed a subcarrier outside affected_blocks")

    rate_by_id = {
        record.task_id: record.rate_bps
        for record in before_outcome.rates
        if record.subcarrier not in affected_subcarriers
    }
    cross_by_id = {
        record.task_id: record.cross_interference_w
        for record in before_outcome.rates
        if record.subcarrier not in affected_subcarriers
    }
    groups: dict[tuple[int, int], list[ScheduleEntry]] = defaultdict(list)
    entries_by_subcarrier: dict[int, list[ScheduleEntry]] = defaultdict(list)
    for entry in candidate_schedule.entries:
        if entry.subcarrier not in affected_subcarriers:
            continue
        groups[(entry.serving_uav, entry.subcarrier)].append(entry)
        entries_by_subcarrier[entry.subcarrier].append(entry)

    sic_violations: list[int] = []
    for (uav, subcarrier), own_entries in sorted(groups.items()):
        cross_w = _cross_interference_w(
            uav,
            subcarrier,
            groups,
            {},
            snapshot,
            cfg.transmit_power_w,
            context.received_power_by_task,
            entries_by_subcarrier,
        )
        cross_w *= cfg.cross_interference_scale
        if len(own_entries) == 1:
            entry = own_entries[0]
            received_w = context.received_power_by_task[entry.task_id][uav]
            sinr = received_w / (cross_w + context.noise_w)
            rate_by_id[entry.task_id] = _rate_bps(context.bandwidth_hz, sinr)
            cross_by_id[entry.task_id] = cross_w
            continue

        by_role = {entry.noma_role: entry for entry in own_entries}
        strong = by_role["strong"]
        weak = by_role["weak"]
        received_strong_w = context.received_power_by_task[strong.task_id][uav]
        received_weak_w = context.received_power_by_task[weak.task_id][uav]
        if received_strong_w < received_weak_w:
            raise InfeasibleScheduleError(
                "strong/weak order does not match the serving-UAV channel"
            )
        sinr_strong = received_strong_w / (
            received_weak_w + cross_w + context.noise_w
        )
        sinr_weak = received_weak_w / (
            cfg.sic_residual * received_strong_w + cross_w + context.noise_w
        )
        if sinr_strong < cfg.gamma_sic_min:
            sic_violations.extend((strong.task_id, weak.task_id))
        rate_by_id[strong.task_id] = _rate_bps(context.bandwidth_hz, sinr_strong)
        rate_by_id[weak.task_id] = _rate_bps(context.bandwidth_hz, sinr_weak)
        cross_by_id[strong.task_id] = cross_w
        cross_by_id[weak.task_id] = cross_w

    if sic_violations:
        raise InfeasibleScheduleError(
            f"SIC threshold violated by tasks {sorted(sic_violations)}"
        )
    if validate_structure:
        scheduled_ids = {entry.task_id for entry in candidate_schedule.entries}
        if set(rate_by_id) != scheduled_ids:
            raise ValueError("rate records and serving-UAV mapping must have identical task IDs")

    entries_by_uav: dict[int, list[int]] = defaultdict(list)
    for entry in candidate_schedule.entries:
        if (
            validate_structure
            and not context.task_by_id[entry.task_id].requires_offload
        ):
            raise ValueError("local-feasible task must not have a radio rate record")
        entries_by_uav[entry.serving_uav].append(entry.task_id)

    completed_count = context.local_completed_count
    cpu_costs: list[float] = []
    for uav in range(snapshot.num_uavs):
        scheduled_tasks = tuple(
            context.task_by_id[task_id]
            for task_id in sorted(entries_by_uav.get(uav, ()))
        )
        rate_items = tuple(
            (task.task_id, rate_by_id[task.task_id]) for task in scheduled_tasks
        )
        score = _cached_allocation_score(
            cfg.allocation_policy,
            scheduled_tasks,
            rate_items,
            context.cpu_budgets_hz[uav],
            cfg.result_ratio,
            id(allocation_score_with_policy),
            True,
        )
        completed_count += score.completed_count
        cpu_costs.append(score.total_cpu_hz)
    return ScheduleScore(
        completed_count,
        math.fsum(cpu_costs),
        math.fsum(cross_by_id[task_id] for task_id in sorted(cross_by_id)),
        (),
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
        total_mec_cpu_hz=math.fsum(
            decision.cpu_hz
            for decision in decisions
            if decision.completed and decision.alpha > 0.0
        ),
    )
