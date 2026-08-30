"""Deterministic exogenous trace generation for trusted experiments."""

from __future__ import annotations

import math

import numpy as np

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
    if cfg.cluster_layout == "pairwise_overlap":
        if cfg.num_uavs != 3 or cfg.num_clusters != 3:
            raise ValueError(
                "pairwise_overlap cluster_layout requires K=3 and 3 clusters"
            )
        uavs = np.asarray(UAV_HOVER_POSITIONS[3], dtype=float)
        return np.asarray(
            (
                (uavs[0] + uavs[1]) / 2.0,
                (uavs[1] + uavs[2]) / 2.0,
                (uavs[2] + uavs[0]) / 2.0,
            )
        )
    if cfg.cluster_layout == "uav_overlap":
        if cfg.num_uavs != 2 or cfg.num_clusters != 3:
            raise ValueError("uav_overlap cluster_layout requires K=2 and 3 clusters")
        uavs = np.asarray(UAV_HOVER_POSITIONS[2], dtype=float)
        midpoint = uavs.mean(axis=0)
        baseline = uavs[1] - uavs[0]
        normal = np.array([-baseline[1], baseline[0]], dtype=float)
        normal /= np.linalg.norm(normal)
        offsets = np.array([-0.2, 0.0, 0.2]) * cfg.area_size
        centers = midpoint + offsets[:, None] * normal
        if (centers < 0.0).any() or (centers > cfg.area_size).any():
            raise ValueError("uav_overlap cluster_layout falls outside the area")
        return centers
    if cfg.cluster_layout != "default":
        raise ValueError(f"unsupported cluster_layout {cfg.cluster_layout!r}")
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

    rng = np.random.default_rng(seed)
    csi_rng = np.random.default_rng(seed + 7_919)
    deadline_rng = None if cfg.task.t_min is None else np.random.default_rng(seed + 15_841)
    profile_rng = np.random.default_rng(seed + 23_761)
    jitter_rng = np.random.default_rng(seed + 31_693)
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
    next_switch = int(rng.integers(cfg.burst_period_min, cfg.burst_period_max + 1))

    periodic_profiles: tuple[tuple[int, int, float, float, float], ...] = ()
    periodic_releases: tuple[dict[int, int], ...] = ()
    if cfg.task.arrival_mode == "periodic":
        profiles: list[tuple[int, int, float, float, float]] = []
        releases: list[dict[int, int]] = []
        for _user in range(cfg.num_users):
            period = int(profile_rng.choice(cfg.task.period_choices))
            phase = int(profile_rng.integers(0, period))
            data_bits = float(profile_rng.uniform(cfg.task.data_min, cfg.task.data_max))
            density = float(profile_rng.uniform(cfg.task.comp_min, cfg.task.comp_max))
            deadline_s = cfg.task.t_max
            if cfg.task.t_min is not None:
                deadline_s = float(profile_rng.uniform(cfg.task.t_min, cfg.task.t_max))
            if deadline_s > min(period * cfg.uav.dt, cfg.uav.dt):
                raise ValueError("periodic task deadline must not exceed one slot")

            user_releases: dict[int, int] = {}
            job_index = 0
            nominal = phase
            previous = -1
            while nominal < cfg.num_slots:
                jitter = (
                    int(jitter_rng.integers(0, cfg.task.period_jitter_slots + 1))
                    if cfg.task.period_jitter_slots
                    else 0
                )
                release = nominal + jitter
                if release <= previous:
                    raise RuntimeError("periodic release slots must be strictly increasing")
                if release < cfg.num_slots:
                    user_releases[release] = job_index
                    previous = release
                job_index += 1
                nominal = phase + job_index * period
            profiles.append((period, phase, data_bits, density, deadline_s))
            releases.append(user_releases)
        periodic_profiles = tuple(profiles)
        periodic_releases = tuple(releases)

    slots: list[SlotSnapshot] = []
    for slot in range(cfg.num_slots):
        if slot > 0 and cfg.user_mobility:
            angles = rng.uniform(0.0, 2.0 * math.pi, cfg.num_users)
            distance = cfg.user_speed * cfg.uav.dt
            positions[:, 0] += distance * np.cos(angles)
            positions[:, 1] += distance * np.sin(angles)
            positions = np.clip(positions, 0.0, cfg.area_size)

        if cfg.random_burst and slot >= next_switch:
            burst_cluster = int(rng.integers(0, cfg.num_clusters))
            next_switch = slot + int(rng.integers(cfg.burst_period_min, cfg.burst_period_max + 1))

        tasks: list[Task] = []
        for user in range(cfg.num_users):
            if cfg.task.arrival_mode == "periodic":
                job_index = periodic_releases[user].get(slot)
                if job_index is None:
                    continue
                _period, _phase, data_bits, density, deadline_s = periodic_profiles[user]
                tasks.append(
                    Task(
                        task_id=(user << 32) | job_index,
                        user_id=user,
                        arrival_slot=slot,
                        data_bits=data_bits,
                        comp_cycles=data_bits * density,
                        deadline_s=deadline_s,
                        local_cpu_hz=cfg.task.f_local,
                        job_index=job_index,
                    )
                )
                continue

            probability = cfg.task.arrival_prob
            if cfg.random_burst:
                probability = (
                    cfg.burst_rate if cluster_ids[user] == burst_cluster else cfg.cool_rate
                )
            if rng.random() >= probability:
                continue

            data_bits = float(rng.uniform(cfg.task.data_min, cfg.task.data_max))
            density = float(rng.uniform(cfg.task.comp_min, cfg.task.comp_max))
            deadline_s = cfg.task.t_max
            if deadline_rng is not None:
                deadline_s = float(deadline_rng.uniform(cfg.task.t_min, cfg.task.t_max))
            tasks.append(
                Task(
                    task_id=slot * cfg.num_users + user,
                    user_id=user,
                    arrival_slot=slot,
                    data_bits=data_bits,
                    comp_cycles=data_bits * density,
                    deadline_s=deadline_s,
                    local_cpu_hz=cfg.task.f_local,
                )
            )

        gains = np.zeros((len(tasks), cfg.num_uavs), dtype=float)
        for row, task in enumerate(tasks):
            for uav in range(cfg.num_uavs):
                horizontal_m = float(np.linalg.norm(positions[task.user_id] - uav_positions[uav]))
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
