"""Materialize and record experiments from an immutable GeoLife trace lock."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifacts import RunArtifactWriter
from .domain import ExperimentTrace, SlotSnapshot, Task
from .experiment import (
    ExperimentDefinition,
    _slot_method,
    definition_to_dict,
    load_definition,
    method_spec_sha256,
    method_spec_to_dict,
    thesis_method_specs,
)
from .externality import aggregate_victim_link_exposure
from .geolife_lock import TraceLockSnapshot, load_trace_lock_snapshot
from .model import NOMAChannel, ScenarioConfig, UAV_HOVER_POSITIONS
from .seed_access import SeedAccessRecorder, SeedPurpose
from .simulation import run_outcome_episode
from .trace import _periodic_arrival_state, _periodic_interval, _sample_deadline_s


@dataclass(frozen=True, slots=True)
class MaterializedLockedScenario:
    scenario_id: str
    participant_id: str
    split: str
    mode: str
    source_path: str
    source_start: str
    source_end: str
    group_id: str
    timestamps_s: np.ndarray
    positions_enu_m: np.ndarray
    task_profile_draws: np.ndarray
    trace_lock_sha256: str

    def __post_init__(self) -> None:
        for field in ("timestamps_s", "positions_enu_m", "task_profile_draws"):
            values = np.array(getattr(self, field), dtype=float, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field, values)


def _materialize_locked_scenarios(
    trace_lock_path: Path,
    *,
    split: str,
    seed: int,
    scenario_ids: Sequence[str] | None,
    seed_access: SeedAccessRecorder,
    allowed_splits: frozenset[str],
    lock_snapshot: TraceLockSnapshot | None = None,
) -> tuple[MaterializedLockedScenario, ...]:
    if split not in allowed_splits:
        expected = " or ".join(sorted(allowed_splits))
        raise ValueError(f"locked trace runner permits only the {expected} split")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(seed_access, SeedAccessRecorder):
        raise TypeError("seed_access must be a SeedAccessRecorder")
    snapshot = load_trace_lock_snapshot(trace_lock_path) if lock_snapshot is None else lock_snapshot
    lock = snapshot.lock
    digest = snapshot.sha256
    available = {
        segment.scenario_id: segment for segment in lock.segments if segment.split == split
    }
    if not available:
        raise ValueError(f"trace lock has no {split} scenarios")
    selected_ids = tuple(available) if scenario_ids is None else tuple(scenario_ids)
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("scenario_ids must be non-empty and unique")
    missing = [scenario_id for scenario_id in selected_ids if scenario_id not in available]
    if missing:
        raise ValueError(f"unlocked scenario requested: {missing[0]}")

    seed_access.record(
        "materialize_locked_trace",
        [seed],
        {
            "split": split,
            "trace_lock_sha256": digest,
            "scenario_ids": list(selected_ids),
        },
    )
    document = json.loads(snapshot.document_json)
    rng = np.random.default_rng(seed)
    rows: list[MaterializedLockedScenario] = []
    for scenario_id in selected_ids:
        segment = available[scenario_id]
        sample = document["samples"][scenario_id]
        timestamps = np.asarray(sample["timestamps_s"], dtype=float)
        positions = np.asarray(sample["positions_enu_m"], dtype=float)
        # These draws stand for task size/density/deadline profiles.  They are
        # intentionally orthogonal to the locked participant and mobility path.
        task_profiles = rng.random((len(timestamps), 3))
        rows.append(
            MaterializedLockedScenario(
                scenario_id=scenario_id,
                participant_id=segment.participant_id,
                split=segment.split,
                mode=segment.mode,
                source_path=segment.source_path,
                source_start=segment.source_start,
                source_end=segment.source_end,
                group_id=segment.group_id,
                timestamps_s=timestamps,
                positions_enu_m=positions,
                task_profile_draws=task_profiles,
                trace_lock_sha256=digest,
            )
        )
    return tuple(rows)


def materialize_locked_scenarios(
    trace_lock_path: Path,
    *,
    split: str,
    seed: int,
    seed_access: SeedAccessRecorder,
    scenario_ids: Sequence[str] | None = None,
    _lock_snapshot: TraceLockSnapshot | None = None,
) -> tuple[MaterializedLockedScenario, ...]:
    """Materialize only final scenarios for a trace-driven experiment."""

    return _materialize_locked_scenarios(
        trace_lock_path,
        split=split,
        seed=seed,
        scenario_ids=scenario_ids,
        seed_access=seed_access,
        allowed_splits=frozenset({"final"}),
        lock_snapshot=_lock_snapshot,
    )


def materialize_calibration_scenarios(
    trace_lock_path: Path,
    *,
    seed: int,
    seed_access: SeedAccessRecorder,
    _lock_snapshot: TraceLockSnapshot | None = None,
) -> tuple[MaterializedLockedScenario, ...]:
    """Internal calibration-only view; train and final remain inaccessible."""

    return _materialize_locked_scenarios(
        trace_lock_path,
        split="calibration",
        seed=seed,
        scenario_ids=None,
        seed_access=seed_access,
        allowed_splits=frozenset({"calibration"}),
        lock_snapshot=_lock_snapshot,
    )


def _locked_user_positions(
    scenarios: Sequence[MaterializedLockedScenario],
    cfg: ScenarioConfig,
) -> np.ndarray:
    """Expand one locked participant group to a seed-invariant formation."""

    members = tuple(scenarios)
    if not members:
        raise ValueError("locked scenario group must be non-empty")
    for scenario in members:
        if len(scenario.positions_enu_m) < cfg.num_slots:
            raise ValueError(
                f"locked scenario {scenario.scenario_id} has fewer than {cfg.num_slots} slots"
            )
    user_index = np.arange(cfg.num_users, dtype=float)
    radii = cfg.cluster_std * np.sqrt((user_index + 0.5) / cfg.num_users)
    angles = user_index * (math.pi * (3.0 - math.sqrt(5.0)))
    offsets = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    positions = np.empty((cfg.num_slots, cfg.num_users, 2), dtype=float)
    for user in range(cfg.num_users):
        source = members[user % len(members)]
        positions[:, user, :] = source.positions_enu_m[: cfg.num_slots] + offsets[user]
    return np.clip(positions, 0.0, cfg.area_size)


def _locked_experiment_trace(
    scenario: MaterializedLockedScenario | Sequence[MaterializedLockedScenario],
    cfg: ScenarioConfig,
    seed: int,
    csi_error_std: float,
) -> ExperimentTrace:
    """Generate task/channel draws over one immutable source mobility path."""

    if not math.isfinite(csi_error_std) or csi_error_std < 0.0:
        raise ValueError("csi_error_std must be finite and non-negative")
    if cfg.num_uavs not in UAV_HOVER_POSITIONS:
        raise ValueError(f"trace runner has no fixed UAV layout for K={cfg.num_uavs}")
    if cfg.task.t_max > cfg.uav.dt:
        raise ValueError("trace runner requires deadline no greater than slot duration")
    members = (scenario,) if isinstance(scenario, MaterializedLockedScenario) else tuple(scenario)
    positions = _locked_user_positions(members, cfg)
    cluster_ids = np.arange(cfg.num_users) % cfg.num_clusters
    uav_positions = np.asarray(UAV_HOVER_POSITIONS[cfg.num_uavs], dtype=float)
    channel = NOMAChannel(cfg.channel, cfg.uav.height)
    rng = np.random.default_rng(seed)
    csi_rng = np.random.default_rng(seed + 7_919)
    burst_cluster = int(rng.integers(0, cfg.num_clusters))
    next_switch = int(rng.integers(cfg.burst_period_min, cfg.burst_period_max + 1))
    next_job_index = [0 for _ in range(cfg.num_users)]
    if cfg.task.arrival_mode == "periodic":
        periods, next_arrivals = _periodic_arrival_state(cfg, rng)
    elif cfg.task.arrival_mode == "bernoulli":
        periods, next_arrivals = [], []
    else:
        raise ValueError("trace runner arrival_mode must be bernoulli or periodic")

    slots: list[SlotSnapshot] = []
    for slot in range(cfg.num_slots):
        if cfg.random_burst and slot >= next_switch:
            burst_cluster = int(rng.integers(0, cfg.num_clusters))
            next_switch = slot + int(rng.integers(cfg.burst_period_min, cfg.burst_period_max + 1))
        tasks: list[Task] = []
        for user in range(cfg.num_users):
            if cfg.task.arrival_mode == "periodic":
                if slot != next_arrivals[user]:
                    continue
                next_arrivals[user] = slot + _periodic_interval(
                    periods[user], cfg.task.period_jitter_slots, rng
                )
            else:
                probability = cfg.task.arrival_prob
                if cfg.random_burst:
                    probability = (
                        cfg.burst_rate if cluster_ids[user] == burst_cluster else cfg.cool_rate
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
                    deadline_s=_sample_deadline_s(cfg, rng),
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
                    np.linalg.norm(positions[slot, task.user_id] - uav_positions[uav])
                )
                gains[row, uav] = channel.path_loss(horizontal_m)
        estimated = None
        if csi_error_std > 0.0:
            estimated = np.maximum(
                gains * (1.0 + csi_rng.normal(0.0, csi_error_std, gains.shape)),
                0.0,
            )
        slots.append(
            SlotSnapshot(
                slot_index=slot,
                tasks=tuple(tasks),
                channel_gains=gains,
                num_uavs=cfg.num_uavs,
                estimated_channel_gains=estimated,
            )
        )
    return ExperimentTrace(seed=seed, slots=tuple(slots))


def _group_provenance(
    group_id: str,
    scenarios: Sequence[MaterializedLockedScenario],
    seed: int,
) -> dict[str, object]:
    members = tuple(scenarios)
    return {
        "seed": seed,
        "scenario_id": group_id,
        "participant_id": json.dumps(
            [member.participant_id for member in members], separators=(",", ":")
        ),
        "source_path": json.dumps(
            [member.source_path for member in members], separators=(",", ":")
        ),
        "source_start": min(member.source_start for member in members),
        "source_end": max(member.source_end for member in members),
        "trace_lock_sha256": members[0].trace_lock_sha256,
    }


def _final_groups(
    lock_snapshot: TraceLockSnapshot,
    scenarios: Sequence[MaterializedLockedScenario],
) -> tuple[tuple[str, tuple[MaterializedLockedScenario, ...]], ...]:
    """Resolve the protocol's final composite scenario groups exactly."""

    document = json.loads(lock_snapshot.document_json)
    by_participant: dict[str, list[MaterializedLockedScenario]] = {}
    for scenario in scenarios:
        by_participant.setdefault(scenario.participant_id, []).append(scenario)
    groups: list[tuple[str, tuple[MaterializedLockedScenario, ...]]] = []
    for row in document["final_groups"]:
        participants = tuple(row["participants"])
        members = tuple(
            scenario
            for participant in participants
            for scenario in by_participant.get(participant, ())
        )
        if not members or {member.participant_id for member in members} != set(participants):
            raise ValueError(f"final group {row['group_id']} is not fully materialized")
        groups.append((str(row["group_id"]), members))
    return tuple(groups)


def run_trace_definition(
    definition: ExperimentDefinition,
    config_path: Path,
    trace_lock_path: Path,
    output_dir: Path,
    *,
    seed_access: SeedAccessRecorder,
    scenario_ids: Sequence[str] | None = None,
) -> Path:
    """Execute registered methods over final-split locked mobility scenarios."""

    seed_values = tuple(definition.seeds)
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("seeds must be non-empty and unique")
    lock_snapshot = load_trace_lock_snapshot(trace_lock_path)
    lock = lock_snapshot.lock
    if seed_access.permit.purpose is SeedPurpose.FORMAL and any(
        seed not in lock.seeds for seed in seed_values
    ):
        raise ValueError("formal trace seeds must belong to the locked trace seed pool")
    all_specs = thesis_method_specs()
    try:
        selected_specs = {method: all_specs[method] for method in definition.methods}
    except KeyError as exc:
        raise ValueError(f"unknown trace method {exc.args[0]}") from exc

    seed_access.record(
        "run_trace_experiment",
        seed_values,
        {
            "config_path": str(Path(config_path)),
            "trace_lock_sha256": lock_snapshot.sha256,
            "split": "final",
        },
    )

    writer = RunArtifactWriter(output_dir, bundle_version=2)
    provenance: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    for seed in seed_values:
        scenarios = materialize_locked_scenarios(
            trace_lock_path,
            split="final",
            seed=seed,
            scenario_ids=scenario_ids,
            seed_access=seed_access,
            _lock_snapshot=lock_snapshot,
        )
        for group_id, members in _final_groups(lock_snapshot, scenarios):
            trace = _locked_experiment_trace(
                members,
                definition.scenario,
                seed,
                definition.csi_error_std,
            )
            task_by_id = {task.task_id: task for snapshot in trace.slots for task in snapshot.tasks}
            base = _group_provenance(group_id, members, seed)
            member_by_user = {
                user: members[user % len(members)] for user in range(definition.scenario.num_users)
            }
            for member in members:
                for slot, (timestamp, position, task_draw) in enumerate(
                    zip(
                        member.timestamps_s[: definition.scenario.num_slots],
                        member.positions_enu_m[: definition.scenario.num_slots],
                        member.task_profile_draws[: definition.scenario.num_slots],
                        strict=True,
                    )
                ):
                    provenance.append(
                        {
                            **base,
                            "participant_id": member.participant_id,
                            "source_path": member.source_path,
                            "source_start": member.source_start,
                            "source_end": member.source_end,
                            "source_segment_id": member.scenario_id,
                            "slot": slot,
                            "timestamp_s": float(timestamp),
                            "x_m": float(position[0]),
                            "y_m": float(position[1]),
                            "task_profile_draws": task_draw.tolist(),
                        }
                    )
            for method in definition.methods:
                events: list[dict[str, object]] = []
                target = replace(
                    definition.evaluation,
                    allocation_policy=selected_specs[method].allocation_policy,
                )
                episode = run_outcome_episode(
                    trace,
                    _slot_method(
                        method,
                        definition.scenario,
                        target,
                        definition.max_iterations,
                        events,
                    ),
                )
                episode_rows.append(
                    {
                        **base,
                        "method": method,
                        "on_time_rate": episode.on_time_rate,
                    }
                )
                for outcome in episode.outcomes:
                    plan_key = json.dumps(outcome.plan_key, separators=(",", ":"))
                    decisions = {row.task_id: row for row in outcome.decisions}
                    rates = {row.task_id: row for row in outcome.rates}
                    for task_id in outcome.arrived_task_ids:
                        task = task_by_id[task_id]
                        member = member_by_user[task.user_id]
                        decision = decisions[task_id]
                        rate = rates.get(task_id)
                        task_rows.append(
                            {
                                **base,
                                "participant_id": member.participant_id,
                                "source_path": member.source_path,
                                "source_start": member.source_start,
                                "source_end": member.source_end,
                                "method": method,
                                "slot": outcome.slot_index,
                                "task_id": task_id,
                                "user_id": task.user_id,
                                "data_bits": task.data_bits,
                                "comp_cycles": task.comp_cycles,
                                "deadline_s": task.deadline_s,
                                "completed": decision.completed,
                                "delay_s": decision.delay_s,
                                "alpha": decision.alpha,
                                "cpu_hz": decision.cpu_hz,
                                "rate_bps": None if rate is None else rate.rate_bps,
                                "sinr": None if rate is None else rate.sinr,
                                "sic_feasible": None if rate is None else rate.sic_feasible,
                                "reason": decision.reason,
                                "plan_key": plan_key,
                            }
                        )
                    slot_rows.append(
                        {
                            **base,
                            "method": method,
                            "slot": outcome.slot_index,
                            "arrivals": len(outcome.arrived_task_ids),
                            "completed": outcome.completed_count,
                            "slack_s": outcome.total_slack_s,
                            "aggregate_victim_link_exposure_w": (
                                aggregate_victim_link_exposure(outcome.rates)
                            ),
                            "sic_violations": sum(not row.sic_feasible for row in outcome.rates),
                            "runtime_s": outcome.runtime_s,
                            "plan_key": plan_key,
                        }
                    )
                event_rows.extend({**base, "method": method, **event} for event in events)
    writer.write_jsonl("trace_provenance.jsonl", provenance)
    writer.write_csv(
        "tasks.csv",
        task_rows,
        fieldnames=(
            "seed",
            "scenario_id",
            "participant_id",
            "source_path",
            "source_start",
            "source_end",
            "trace_lock_sha256",
            "method",
            "slot",
            "task_id",
            "user_id",
            "data_bits",
            "comp_cycles",
            "deadline_s",
            "completed",
            "delay_s",
            "alpha",
            "cpu_hz",
            "rate_bps",
            "sinr",
            "sic_feasible",
            "reason",
            "plan_key",
        ),
    )
    writer.write_csv(
        "slots.csv",
        slot_rows,
        fieldnames=(
            "seed",
            "scenario_id",
            "participant_id",
            "source_path",
            "source_start",
            "source_end",
            "trace_lock_sha256",
            "method",
            "slot",
            "arrivals",
            "completed",
            "slack_s",
            "aggregate_victim_link_exposure_w",
            "sic_violations",
            "runtime_s",
            "plan_key",
        ),
    )
    writer.write_jsonl("events.jsonl", event_rows)
    writer.write_json(
        "summary.json",
        {
            "schema_version": 1,
            "split": "final",
            "seeds": list(seed_values),
            "scenario_ids": sorted({row["scenario_id"] for row in provenance}),
            "participant_ids": sorted({row["participant_id"] for row in provenance}),
            "mobility_row_count": len(provenance),
            "episode_results": episode_rows,
            "method_specs": {
                method: {
                    "factor_hash": method_spec_sha256(selected_specs[method]),
                    "spec": method_spec_to_dict(selected_specs[method]),
                }
                for method in definition.methods
            },
            "mobility_expansion": (
                "each source-participant path is the centroid of one deterministic "
                "seed-invariant simulated-user formation"
            ),
            "interpretation": "human-mobility external-validity supplement, not UAV-MEC workload data",
        },
    )
    config_file = Path(config_path)
    writer.finalize(
        {
            "schema_version": 1,
            "runner": "trace_driven",
            "trace_lock_sha256": lock_snapshot.sha256,
            "seeds": list(seed_values),
            "config_path": str(config_file),
            "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
            "resolved_definition": definition_to_dict(definition),
        }
    )
    return Path(output_dir)


def run_trace_experiment(
    config_path: Path,
    trace_lock_path: Path,
    output_dir: Path,
    *,
    seed_access: SeedAccessRecorder,
    scenario_ids: Sequence[str] | None = None,
) -> Path:
    """Load one trace-family definition and execute it over locked final paths."""

    return run_trace_definition(
        load_definition(config_path),
        config_path,
        trace_lock_path,
        output_dir,
        seed_access=seed_access,
        scenario_ids=scenario_ids,
    )
