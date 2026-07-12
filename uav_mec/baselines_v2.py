"""Controlled scheduling baselines on the trusted data path."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .domain import (
    RateRecord,
    Schedule,
    ScheduleEntry,
    SlotOutcome,
    SlotSnapshot,
    Task,
)
from .evaluation import EvaluationConfig, build_outcome_from_rates
from .model import NOMAChannel
from .schedule import normalize_roles


def local_only_schedule(snapshot: SlotSnapshot) -> Schedule:
    """Return an empty radio schedule; local-feasible tasks remain evaluable."""
    return Schedule()


def greedy_schedule(
    snapshot: SlotSnapshot,
    n_subcarriers: int,
) -> Schedule:
    """Pair strongest and weakest nearest-UAV users without joint search."""
    if not isinstance(n_subcarriers, int) or isinstance(n_subcarriers, bool):
        raise ValueError("n_subcarriers must be an integer")
    if n_subcarriers <= 0:
        raise ValueError("n_subcarriers must be positive")

    row_by_task = {
        task.task_id: row for row, task in enumerate(snapshot.tasks)
    }
    entries: list[ScheduleEntry] = []
    for uav in range(snapshot.num_uavs):
        candidates = [
            task
            for task in snapshot.tasks
            if task.requires_offload
            and int(
                np.asarray(
                    snapshot.channel_gains[row_by_task[task.task_id]]
                ).argmax()
            )
            == uav
        ]
        candidates.sort(
            key=lambda task: (
                -snapshot.channel_gains[
                    row_by_task[task.task_id],
                    uav,
                ],
                task.task_id,
            )
        )
        selected = candidates[: 2 * n_subcarriers]
        for subcarrier in range(n_subcarriers):
            left = subcarrier
            right = len(selected) - 1 - subcarrier
            if left > right:
                break
            for index in sorted({left, right}):
                task = selected[index]
                entries.append(
                    ScheduleEntry(
                        task.task_id,
                        task.user_id,
                        uav,
                        subcarrier,
                        "solo",
                    )
                )

    return normalize_roles(Schedule(tuple(entries)), snapshot)


def evaluate_fair_ofdma(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> SlotOutcome:
    """Evaluate equal-admission OFDMA with at most two time phases."""
    row_by_task = {
        task.task_id: row for row, task in enumerate(snapshot.tasks)
    }
    selected: dict[int, list[Task]] = defaultdict(list)
    for uav in range(snapshot.num_uavs):
        candidates = [
            task
            for task in snapshot.tasks
            if task.requires_offload
            and int(
                np.asarray(
                    snapshot.channel_gains[row_by_task[task.task_id]]
                ).argmax()
            )
            == uav
        ]
        candidates.sort(
            key=lambda task: (
                -snapshot.channel_gains[
                    row_by_task[task.task_id],
                    uav,
                ],
                task.task_id,
            )
        )
        selected[uav] = candidates[: 2 * cfg.n_subcarriers]

    assignments: list[tuple[int, int, int, int]] = []
    for uav, tasks in selected.items():
        for index, task in enumerate(tasks):
            subcarrier = index % cfg.n_subcarriers
            phase = index // cfg.n_subcarriers
            assignments.append(
                (task.task_id, uav, subcarrier, phase)
            )

    phases = (
        2
        if any(
            len(tasks) > cfg.n_subcarriers
            for tasks in selected.values()
        )
        else 1
    )
    active_phases = {
        task_id: (
            tuple(range(phases))
            if len(selected[uav]) <= cfg.n_subcarriers
            else (phase,)
        )
        for task_id, uav, _subcarrier, phase in assignments
    }
    bandwidth_hz = cfg.bandwidth_hz / cfg.n_subcarriers
    noise_w = channel.noise_for_bandwidth(bandwidth_hz)
    rates = []
    for task_id, serving_uav, subcarrier, _phase in assignments:
        row = row_by_task[task_id]
        received_w = (
            cfg.transmit_power_w
            * snapshot.channel_gains[row, serving_uav]
        )
        phase_sinrs = []
        phase_cross_w = []
        for phase in active_phases[task_id]:
            cross_w = math.fsum(
                cfg.transmit_power_w
                * snapshot.channel_gains[
                    row_by_task[other_id],
                    serving_uav,
                ]
                for (
                    other_id,
                    other_uav,
                    other_subcarrier,
                    _other_phase,
                ) in assignments
                if other_uav != serving_uav
                and other_subcarrier == subcarrier
                and phase in active_phases[other_id]
            )
            phase_cross_w.append(cross_w)
            phase_sinrs.append(received_w / (cross_w + noise_w))

        rate_bps = math.fsum(
            bandwidth_hz * math.log1p(sinr) / math.log(2.0)
            for sinr in phase_sinrs
        ) / phases
        rates.append(
            RateRecord(
                task_id,
                rate_bps,
                math.fsum(phase_sinrs) / len(phase_sinrs),
                True,
                math.fsum(phase_cross_w) / phases,
                serving_uav,
                subcarrier,
            )
        )

    uav_by_task = {
        task_id: uav
        for task_id, uav, _subcarrier, _phase in assignments
    }
    return build_outcome_from_rates(
        snapshot,
        uav_by_task,
        tuple(rates),
        cfg,
    )
