"""Deterministic exogenous trace generation for trusted experiments."""

from __future__ import annotations

import math

import numpy as np
from .seed_access import SeedAccessRecorder

from .domain import ExperimentTrace, SlotSnapshot, Task
from .model import (
    FIXED_CLUSTER_CENTERS_3,
    FIXED_CLUSTER_CENTERS_4,
    FIXED_CLUSTER_CENTERS_5,
    NOMAChannel,
    ScenarioConfig,
    UAV_HOVER_POSITIONS,
)


def _centers(cfg: ScenarioConfig) -> np.ndarray:
    known = {
        3: FIXED_CLUSTER_CENTERS_3,
        4: FIXED_CLUSTER_CENTERS_4,
        5: FIXED_CLUSTER_CENTERS_5,
    }
    if cfg.num_clusters not in known:
        raise ValueError("trusted trace supports 3, 4, or 5 fixed clusters")
    return known[cfg.num_clusters].copy()


def generate_trace(
    cfg: ScenarioConfig,
    seed: int,
    csi_error_std: float = 0.0,
    *,
    seed_access: SeedAccessRecorder,
) -> ExperimentTrace:
    """Generate all task arrivals and channels from one seed-owned RNG."""
    if not math.isfinite(csi_error_std) or csi_error_std < 0.0:
        raise ValueError("csi_error_std must be finite and non-negative")
    if cfg.num_users <= 0 or cfg.num_slots <= 0 or cfg.n_subcarriers <= 0:
        raise ValueError("trusted trace requires positive users, slots, and subcarriers")
    if cfg.num_uavs not in UAV_HOVER_POSITIONS:
        raise ValueError(f"trusted trace has no fixed layout for K={cfg.num_uavs}")
    if not cfg.fixed_cluster_centers:
        raise ValueError("trusted trace requires fixed_cluster_centers=True")

    seed_access.record("generate_trace", [seed], {"scenario": repr(cfg), "csi_error_std": csi_error_std})
    rng = np.random.default_rng(seed)
    csi_rng = np.random.default_rng(seed + 7_919)
    centers = _centers(cfg)
    cluster_ids = np.arange(cfg.num_users) % cfg.num_clusters
    rng.shuffle(cluster_ids)
    positions = np.vstack(
        [
            np.clip(
                rng.normal(centers[cluster_ids[user]], cfg.cluster_std),
                0.0,
                cfg.area_size,
            )
            for user in range(cfg.num_users)
        ]
    )
    uav_positions = np.asarray(UAV_HOVER_POSITIONS[cfg.num_uavs], dtype=float)
    channel = NOMAChannel(cfg.channel, cfg.uav.height)
    burst_cluster = int(rng.integers(0, cfg.num_clusters))
    next_switch = int(
        rng.integers(cfg.burst_period_min, cfg.burst_period_max + 1)
    )

    slots: list[SlotSnapshot] = []
    next_job_index = [0 for _ in range(cfg.num_users)]
    if cfg.task.t_max > cfg.uav.dt:
        raise ValueError("trusted no-queue trace requires deadline <= slot duration")
    for slot in range(cfg.num_slots):
        if slot > 0 and cfg.user_mobility:
            angles = rng.uniform(0.0, 2.0 * math.pi, cfg.num_users)
            distance = cfg.user_speed * cfg.uav.dt
            positions[:, 0] += distance * np.cos(angles)
            positions[:, 1] += distance * np.sin(angles)
            positions = np.clip(positions, 0.0, cfg.area_size)

        if cfg.random_burst and slot >= next_switch:
            burst_cluster = int(rng.integers(0, cfg.num_clusters))
            next_switch = slot + int(
                rng.integers(cfg.burst_period_min, cfg.burst_period_max + 1)
            )

        tasks: list[Task] = []
        for user in range(cfg.num_users):
            probability = cfg.task.arrival_prob
            if cfg.random_burst:
                probability = (
                    cfg.burst_rate
                    if cluster_ids[user] == burst_cluster
                    else cfg.cool_rate
                )
            if rng.random() >= probability:
                continue

            data_bits = float(rng.uniform(cfg.task.data_min, cfg.task.data_max))
            density = float(rng.uniform(cfg.task.comp_min, cfg.task.comp_max))
            job_index = next_job_index[user]
            next_job_index[user] += 1
            tasks.append(
                Task(
                    task_id=(user << 32) | job_index,
                    user_id=user,
                    arrival_slot=slot,
                    data_bits=data_bits,
                    comp_cycles=data_bits * density,
                    deadline_s=cfg.task.t_max,
                    local_cpu_hz=cfg.task.f_local,
                    job_index=job_index,
                    slot_duration_s=cfg.uav.dt,
                    result_ratio=cfg.task.result_ratio,
                )
            )

        gains = np.zeros((len(tasks), cfg.num_uavs), dtype=float)
        for row, task in enumerate(tasks):
            for uav in range(cfg.num_uavs):
                horizontal_m = float(
                    np.linalg.norm(positions[task.user_id] - uav_positions[uav])
                )
                gains[row, uav] = channel.path_loss(horizontal_m)
        estimated = None
        if csi_error_std > 0.0:
            relative_error = csi_rng.normal(
                0.0,
                csi_error_std,
                gains.shape,
            )
            estimated = np.maximum(
                gains * (1.0 + relative_error),
                0.0,
            )
        slots.append(
            SlotSnapshot(
                slot,
                tuple(tasks),
                gains,
                cfg.num_uavs,
                estimated_channel_gains=estimated,
            )
        )

    return ExperimentTrace(seed, tuple(slots))
