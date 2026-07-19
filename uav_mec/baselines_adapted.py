"""Audited literature adaptations kept separate from thesis-v2 methods.

The Guo-JAGP adaptation in this module preserves the source method's
sequential single-user best-response structure and required-power potential.
It returns only a raw association/grouping schedule.  Canonical radio, SIC,
MEC, deadline, closure, and realized-channel semantics remain the
responsibility of the common experiment evaluator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from .domain import (
    ChannelBasis,
    Schedule,
    ScheduleEntry,
    SlotSnapshot,
    Task,
)
from .evaluation import EvaluationConfig
from .model import NOMAChannel
from .schedule import normalize_roles, validate_schedule


GUO_DEFAULT_QOS_SPECTRAL_EFFICIENCY = 0.5


def _validate_qos_spectral_efficiency(value: float) -> float:
    qos = float(value)
    if not math.isfinite(qos) or qos < 0.0:
        raise ValueError(
            "qos_spectral_efficiency must be finite and non-negative"
        )
    return qos


def guo_required_powers(
    channel_power_gains: Sequence[float],
    qos_spectral_efficiency: Sequence[float],
    noise_w: float,
) -> tuple[float, ...]:
    """Return QoS-required powers ordered from strongest to weakest user.

    This is the Eq. 5 recursion used by the audited Guo-JAGP adaptation.  It
    deliberately has no thesis-v2 CPU, deadline, exposure, or closure term.
    """
    gains = tuple(float(value) for value in channel_power_gains)
    targets = tuple(float(value) for value in qos_spectral_efficiency)
    if not gains or len(gains) != len(targets):
        raise ValueError("gains and targets must have the same positive length")
    if any(not math.isfinite(value) or value <= 0.0 for value in gains):
        raise ValueError("channel gains must be finite and positive")
    if any(not math.isfinite(value) or value < 0.0 for value in targets):
        raise ValueError(
            "spectral efficiencies must be finite and non-negative"
        )
    if not math.isfinite(noise_w) or noise_w <= 0.0:
        raise ValueError("noise_w must be finite and positive")

    powers = [0.0] * len(gains)
    accumulated_received_w = 0.0
    for index in range(len(gains) - 1, -1, -1):
        beta = math.exp2(targets[index]) - 1.0
        powers[index] = (
            beta * (noise_w + accumulated_received_w) / gains[index]
        )
        accumulated_received_w = math.fsum(
            (accumulated_received_w, gains[index] * powers[index])
        )
    return tuple(powers)


def _guo_group_power(
    snapshot: SlotSnapshot,
    entries: Sequence[ScheduleEntry],
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    basis: ChannelBasis,
    qos: float,
) -> tuple[float, dict[int, float]]:
    """Return one group's required-power potential and per-task powers."""
    group = tuple(entries)
    if not group:
        raise ValueError("Guo group must be non-empty")
    if not isinstance(basis, ChannelBasis):
        raise ValueError("basis must be a ChannelBasis value")
    target = _validate_qos_spectral_efficiency(qos)
    block = (group[0].serving_uav, group[0].subcarrier)
    if any(
        (entry.serving_uav, entry.subcarrier) != block for entry in group
    ):
        raise ValueError("Guo group entries must share one resource block")

    row_by_task = {
        task.task_id: row for row, task in enumerate(snapshot.tasks)
    }
    gains = snapshot.channel_gains_for(basis)
    try:
        ordered = tuple(
            sorted(
                group,
                key=lambda entry: (
                    -gains[row_by_task[entry.task_id], entry.serving_uav],
                    entry.task_id,
                ),
            )
        )
        selected_gains = tuple(
            float(gains[row_by_task[entry.task_id], entry.serving_uav])
            for entry in ordered
        )
    except (IndexError, KeyError) as exc:
        raise ValueError("Guo group contains an invalid task or UAV") from exc

    if any(gain <= 0.0 for gain in selected_gains):
        return math.inf, {entry.task_id: math.inf for entry in ordered}

    bandwidth_hz = cfg.bandwidth_hz / cfg.n_subcarriers
    noise_w = channel.noise_for_bandwidth(bandwidth_hz)
    powers = guo_required_powers(
        selected_gains,
        (target,) * len(ordered),
        noise_w,
    )
    powers_by_task_id = {
        entry.task_id: power for entry, power in zip(ordered, powers)
    }
    if any(power > cfg.transmit_power_w for power in powers):
        return math.inf, powers_by_task_id
    return math.fsum(powers), powers_by_task_id


def guo_total_required_power(
    snapshot: SlotSnapshot,
    schedule: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    basis: ChannelBasis,
    qos: float,
) -> float:
    """Return the adapted Guo exact potential for a raw schedule.

    The value includes receiver noise and within-group power recursion only.
    In particular, cross-UAV exposure, MEC CPU, deadlines, completion, and
    closure do not enter this literature potential.
    """
    if not isinstance(basis, ChannelBasis):
        raise ValueError("basis must be a ChannelBasis value")
    target = _validate_qos_spectral_efficiency(qos)
    validate_schedule(schedule, snapshot, cfg.n_subcarriers)
    groups: dict[tuple[int, int], list[ScheduleEntry]] = defaultdict(list)
    for entry in schedule.entries:
        groups[(entry.serving_uav, entry.subcarrier)].append(entry)

    group_powers = (
        _guo_group_power(
            snapshot,
            groups[block],
            channel,
            cfg,
            basis,
            target,
        )[0]
        for block in sorted(groups)
    )
    return math.fsum(group_powers)


def _place_guo_task(
    schedule: Schedule,
    task: Task,
    uav: int,
    subcarrier: int,
    snapshot: SlotSnapshot,
    basis: ChannelBasis,
) -> Schedule | None:
    """Move one selected task, returning ``None`` for a full target block."""
    if basis is not ChannelBasis.PLANNING:
        raise ValueError("Guo schedule search is permitted only on planning channels")
    task_by_id = {item.task_id: item for item in snapshot.tasks}
    if task_by_id.get(task.task_id) != task:
        raise ValueError("selected Guo task is not present in the snapshot")
    if not task.requires_offload:
        raise ValueError("selected Guo task must require offloading")
    if not isinstance(uav, int) or isinstance(uav, bool):
        raise ValueError("uav must be an integer")
    if not 0 <= uav < snapshot.num_uavs:
        raise ValueError("uav is outside the snapshot")
    if (
        not isinstance(subcarrier, int)
        or isinstance(subcarrier, bool)
        or subcarrier < 0
    ):
        raise ValueError("subcarrier must be a non-negative integer")

    current_entries = tuple(
        entry for entry in schedule.entries if entry.task_id == task.task_id
    )
    if len(current_entries) > 1:
        raise ValueError("raw Guo schedule contains a duplicate task")
    retained = tuple(
        entry for entry in schedule.entries if entry.task_id != task.task_id
    )
    if any(
        entry.user_id == task.user_id and entry.task_id != task.task_id
        for entry in retained
    ):
        return None
    target_entries = tuple(
        entry
        for entry in retained
        if entry.serving_uav == uav and entry.subcarrier == subcarrier
    )
    if len(target_entries) >= 2:
        return None

    inserted = ScheduleEntry(
        task.task_id,
        task.user_id,
        uav,
        subcarrier,
        "solo",
    )
    return normalize_roles(
        Schedule(retained + (inserted,)),
        snapshot,
        channel_basis=basis,
    )


def _guo_selected_tasks(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    basis: ChannelBasis,
    qos: float,
) -> tuple[Task, ...]:
    """Freeze a feasible, capacity-limited task set before best response."""
    if basis is not ChannelBasis.PLANNING:
        raise ValueError("Guo task selection is permitted only on planning channels")
    target = _validate_qos_spectral_efficiency(qos)
    beta = math.exp2(target) - 1.0
    noise_w = channel.noise_for_bandwidth(
        cfg.bandwidth_hz / cfg.n_subcarriers
    )
    gains = snapshot.channel_gains_for(basis)
    candidates: list[tuple[float, int, Task]] = []
    for row, task in enumerate(snapshot.tasks):
        if not task.requires_offload:
            continue
        feasible_solo_powers = tuple(
            beta * noise_w / float(gain)
            for gain in gains[row]
            if float(gain) > 0.0
            and beta * noise_w / float(gain) <= cfg.transmit_power_w
        )
        if feasible_solo_powers:
            candidates.append(
                (min(feasible_solo_powers), task.task_id, task)
            )

    candidates.sort(key=lambda row: (row[0], row[1]))
    capacity = 2 * snapshot.num_uavs * cfg.n_subcarriers
    return tuple(row[2] for row in candidates[:capacity])


def guo_jagp_adapted(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    *,
    channel_basis: ChannelBasis,
    qos_spectral_efficiency: float = GUO_DEFAULT_QOS_SPECTRAL_EFFICIENCY,
) -> Schedule:
    """Return a deterministic Guo-JAGP adapted raw schedule.

    This is an adaptation, not a reproduction.  It retains sequential
    single-user best response under strict decreases of total QoS-required
    power, while adapting source grouping to the thesis two-user block cap.
    """
    if channel_basis is not ChannelBasis.PLANNING:
        raise ValueError("Guo-JAGP adaptation may optimize planning channels only")
    qos = _validate_qos_spectral_efficiency(qos_spectral_efficiency)
    selected = _guo_selected_tasks(
        snapshot,
        channel,
        cfg,
        channel_basis,
        qos,
    )

    state = Schedule()
    for task in selected:
        placements: list[tuple[float, int, int, Schedule]] = []
        for serving_uav in range(snapshot.num_uavs):
            for subcarrier in range(cfg.n_subcarriers):
                candidate = _place_guo_task(
                    state,
                    task,
                    serving_uav,
                    subcarrier,
                    snapshot,
                    channel_basis,
                )
                if candidate is None:
                    continue
                potential = guo_total_required_power(
                    snapshot,
                    candidate,
                    channel,
                    cfg,
                    channel_basis,
                    qos,
                )
                placements.append(
                    (potential, serving_uav, subcarrier, candidate)
                )
        if not placements:
            raise ValueError("selected Guo task has no radio position")
        state = min(
            placements,
            key=lambda row: (row[0], row[1], row[2], task.task_id),
        )[3]

    visited = {state.stable_key()}
    while True:
        accepted_in_sweep = False
        for task in selected:
            current_potential = guo_total_required_power(
                snapshot,
                state,
                channel,
                cfg,
                channel_basis,
                qos,
            )
            responses: list[tuple[float, int, int, Schedule]] = []
            for serving_uav in range(snapshot.num_uavs):
                for subcarrier in range(cfg.n_subcarriers):
                    candidate = _place_guo_task(
                        state,
                        task,
                        serving_uav,
                        subcarrier,
                        snapshot,
                        channel_basis,
                    )
                    if candidate is None:
                        continue
                    candidate_potential = guo_total_required_power(
                        snapshot,
                        candidate,
                        channel,
                        cfg,
                        channel_basis,
                        qos,
                    )
                    responses.append(
                        (
                            candidate_potential,
                            serving_uav,
                            subcarrier,
                            candidate,
                        )
                    )
            if not responses:
                raise ValueError("selected Guo task has no radio position")
            best = min(
                responses,
                key=lambda row: (row[0], row[1], row[2], task.task_id),
            )
            if best[0] < current_potential:
                state = best[3]
                if state.stable_key() in visited:
                    raise RuntimeError(
                        "strict Guo potential decrease repeated a state"
                    )
                visited.add(state.stable_key())
                accepted_in_sweep = True
        if not accepted_in_sweep:
            return state
