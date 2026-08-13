"""Guarded development-only calibration for exactness and runtime capacity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .artifacts import RunArtifactWriter
from .experiment import _slot_method, load_definition, run_definition, thesis_method_specs
from .experiment_registry import (
    ExperimentFamily,
    ExperimentRegistry,
    RegisteredCase,
    load_registry,
)
from .geolife_lock import load_trace_lock_snapshot
from .model import NOMAChannel
from .resource_snapshots import RadioGenerator, raw_snapshot_to_dict
from .run_exact_benchmark import run_exact_definition
from .run_resource_experiment import (
    evaluate_resource_generator,
    materialize_resource_generator_slot,
)
from .run_thesis_suite import build_registered_case_definitions
from .run_trace_experiment import (
    MaterializedLockedScenario,
    _locked_experiment_trace,
    materialize_calibration_scenarios,
)
from .runtime_audit import DescriptiveRuntimeAudit, descriptive_runtime_audit_to_dict
from .seed_access import (
    SeedAccessRecorder,
    SeedPurpose,
    development_permit,
)
from .trace import generate_trace
from .simulation import run_outcome_episode


class DevelopmentCalibrationCase(StrEnum):
    EXACT_NONTRIVIALITY = "exact_nontriviality"
    RUNTIME_CAPACITY = "runtime_capacity"


@dataclass(frozen=True, slots=True)
class RuntimeCalibrationResult:
    audit: DescriptiveRuntimeAudit
    raw_bytes: int
    summary_bytes: int
    event_bytes: int


RuntimeAdapter = Callable[..., DescriptiveRuntimeAudit | RuntimeCalibrationResult]


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def _json_bytes(value: object) -> int:
    return len(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> int:
    return sum(_json_bytes(row) + 1 for row in rows)


def _runtime_audit(
    runtimes: Sequence[float],
    candidate_counts: Sequence[int],
    cache_hits: int,
    *,
    peak_memory_bytes: int,
    log_bytes: int,
    measured_cpu_seconds: float,
) -> DescriptiveRuntimeAudit:
    candidate_total = sum(candidate_counts)
    environment_sha256 = _environment_identity()[1]
    return DescriptiveRuntimeAudit(
        reference_hardware=(
            f"{platform.platform()}; architecture={platform.machine()}; "
            f"logical_cpus={os.cpu_count()}"
        ),
        python_environment_sha256=environment_sha256,
        p50_runtime_s=_percentile(runtimes, 50),
        p95_runtime_s=_percentile(runtimes, 95),
        mean_runtime_s=float(np.mean(runtimes)) if runtimes else 0.0,
        candidate_count_p50=_percentile(candidate_counts, 50),
        candidate_count_p95=_percentile(candidate_counts, 95),
        candidate_count_total=candidate_total,
        cache_hit_rate=(cache_hits / candidate_total if candidate_total else 0.0),
        peak_memory_bytes=peak_memory_bytes,
        log_bytes=log_bytes,
        measured_cpu_seconds=measured_cpu_seconds,
    )


def _resource_runtime_measurement(
    *,
    family: ExperimentFamily,
    case: RegisteredCase,
    seeds: tuple[int, ...],
    max_slots: int,
    workers: int,
    trace_lock_path: Path | None,
    seed_access: SeedAccessRecorder,
) -> RuntimeCalibrationResult:
    del workers
    if trace_lock_path is not None:
        raise ValueError("resource runtime adapter does not accept a trace lock")
    try:
        generator = RadioGenerator(case.case_id)
    except ValueError as exc:
        raise ValueError(f"unknown Resource generator case {case.case_id}") from exc
    definition = load_definition(family.config_path)
    calibrated = replace(
        definition,
        seeds=seeds,
        scenario=replace(definition.scenario, num_slots=max_slots),
    )
    channel = NOMAChannel(
        calibrated.scenario.channel,
        calibrated.scenario.uav.height,
    )
    runtimes: list[float] = []
    candidates: list[int] = []
    cache_hits = 0
    raw_rows: list[Mapping[str, object]] = []
    policy_rows: list[Mapping[str, object]] = []
    event_rows: list[Mapping[str, object]] = []
    cpu_started = time.process_time()
    for seed in seeds:
        trace = generate_trace(
            calibrated.scenario,
            seed,
            calibrated.csi_error_std,
            seed_access=seed_access,
        )
        for snapshot in trace.slots:
            events: list[dict[str, object]] = []
            started = time.perf_counter()
            raw = materialize_resource_generator_slot(
                seed,
                snapshot,
                generator,
                channel,
                calibrated.evaluation,
                calibrated.max_iterations,
                calibrated.scenario,
                event_sink=events,
            )
            outcomes = evaluate_resource_generator(
                snapshot,
                raw,
                calibrated.evaluation,
            )
            runtimes.append(time.perf_counter() - started)
            terminations = [event for event in events if event.get("move_type") == "termination"]
            candidates.append(sum(int(event.get("raw_move_count", 0)) for event in terminations))
            cache_hits += sum(int(event.get("cache_hits", 0)) for event in terminations)
            raw_rows.append(raw_snapshot_to_dict(raw))
            event_rows.extend(
                {"seed": seed, "slot": snapshot.slot_index, **event} for event in events
            )
            policy_rows.extend(
                {
                    "seed": seed,
                    "slot": snapshot.slot_index,
                    "generator": generator.value,
                    "policy": row.policy,
                    "completed": row.outcome.completed_count,
                    "arrived": len(row.outcome.arrived_task_ids),
                }
                for row in outcomes
            )
    measured_cpu_seconds = time.process_time() - cpu_started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    event_bytes = _jsonl_bytes(event_rows)
    summary = {
        "family": family.name,
        "case_id": case.case_id,
        "seed_count": len(seeds),
        "slot_count": max_slots,
        "policy_count": len(policy_rows) // max(len(runtimes), 1),
    }
    return RuntimeCalibrationResult(
        audit=_runtime_audit(
            runtimes,
            candidates,
            cache_hits,
            peak_memory_bytes=peak_memory_bytes,
            log_bytes=event_bytes,
            measured_cpu_seconds=measured_cpu_seconds,
        ),
        raw_bytes=_jsonl_bytes(raw_rows) + _jsonl_bytes(policy_rows),
        summary_bytes=_json_bytes(summary) + 1,
        event_bytes=event_bytes,
    )


def _generic_runtime_measurement(
    *,
    family: ExperimentFamily,
    case: RegisteredCase,
    seeds: tuple[int, ...],
    max_slots: int,
    workers: int,
    trace_lock_path: Path | None,
    seed_access: SeedAccessRecorder,
) -> RuntimeCalibrationResult:
    del workers
    if trace_lock_path is not None:
        raise ValueError("generic runtime adapter does not accept a trace lock")
    definitions = {
        item.name: item.definition for item in build_registered_case_definitions(family.name)
    }
    try:
        definition = definitions[case.case_id]
    except KeyError as exc:
        raise ValueError(f"unresolved registered case {family.name}/{case.case_id}") from exc
    calibrated = replace(
        definition,
        seeds=seeds,
        scenario=replace(definition.scenario, num_slots=max_slots),
    )
    temporary = Path(tempfile.mkdtemp(prefix=f"uav-mec-{family.name}-runtime-"))
    output = temporary / "run"
    cpu_started = time.process_time()
    try:
        run_definition(
            calibrated,
            family.config_path,
            output,
            seed_access=seed_access,
        )
        measured_cpu_seconds = time.process_time() - cpu_started
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_memory_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    try:
        runtime_by_slot: dict[tuple[int, int], float] = {}
        with (output / "slots.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["seed"]), int(row["slot"]))
                runtime_by_slot[key] = runtime_by_slot.get(key, 0.0) + float(row["runtime_s"])
        candidates_by_slot: dict[tuple[int, int], int] = {}
        cache_hits = 0
        event_path = output / "events.jsonl"
        for line in event_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("move_type") != "termination":
                continue
            key = (int(event["seed"]), int(event["slot"]))
            count = int(event.get("raw_move_count", 0))
            hit_count = int(event.get("cache_hits", 0))
            candidates_by_slot[key] = candidates_by_slot.get(key, 0) + count
            cache_hits += hit_count
        runtimes = list(runtime_by_slot.values())
        candidate_counts = [float(candidates_by_slot.get(key, 0)) for key in runtime_by_slot]
        event_bytes = event_path.stat().st_size
        return RuntimeCalibrationResult(
            audit=_runtime_audit(
                runtimes,
                [int(value) for value in candidate_counts],
                cache_hits,
                peak_memory_bytes=peak_memory_bytes,
                log_bytes=event_bytes,
                measured_cpu_seconds=measured_cpu_seconds,
            ),
            raw_bytes=(output / "tasks.csv").stat().st_size + (output / "slots.csv").stat().st_size,
            summary_bytes=(output / "summary.json").stat().st_size,
            event_bytes=event_bytes,
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_development_seed_range(
    value: str,
    registry: ExperimentRegistry,
) -> tuple[int, ...]:
    """Parse one inclusive development range and reject every other pool."""

    if not isinstance(value, str) or value.count(":") != 1:
        raise ValueError("seeds must be an inclusive START:END range")
    left, right = value.split(":", 1)
    try:
        start, end = int(left), int(right)
    except ValueError as exc:
        raise ValueError("seeds must contain decimal integers") from exc
    if str(start) != left or str(end) != right:
        raise ValueError("seeds must use canonical decimal integers")
    if start > end:
        raise ValueError("development seed range start must not exceed end")
    seeds = tuple(range(start, end + 1))
    allowed = set(registry.development_seeds)
    if any(seed not in allowed for seed in seeds):
        raise ValueError("seed range must remain inside the development pool")
    return seeds


def _run_exact_nontriviality(
    *,
    family: ExperimentFamily,
    case: RegisteredCase,
    seeds: tuple[int, ...],
    max_slots: int,
    workers: int,
    trace_lock_path: Path | None,
    seed_access: SeedAccessRecorder,
) -> tuple[dict[str, bool], ...]:
    """Run the real complete-enumeration adapter and return certificate flags."""

    del case, workers
    if trace_lock_path is not None:
        raise ValueError("exact_nontriviality does not accept a trace lock")
    definition = load_definition(family.config_path)
    calibrated = replace(
        definition,
        seeds=seeds,
        scenario=replace(
            definition.scenario,
            num_slots=min(max_slots, definition.scenario.num_slots),
        ),
    )
    temporary = Path(tempfile.mkdtemp(prefix="uav-mec-exact-calibration-"))
    output = temporary / "exact"
    try:
        run_exact_definition(
            calibrated,
            family.config_path,
            output,
            seed_access=seed_access,
        )
        with (output / "exact_slots.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = tuple(csv.DictReader(handle))
        return tuple(
            {
                "complete": row["certificate_complete"] == "True",
                "cpu_nontrivial": row["cpu_nontrivial"] == "True",
                "schedule_nontrivial": row["schedule_nontrivial"] == "True",
            }
            for row in rows
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _exact_runtime_measurement(
    *,
    family: ExperimentFamily,
    case: RegisteredCase,
    seeds: tuple[int, ...],
    max_slots: int,
    workers: int,
    trace_lock_path: Path | None,
    seed_access: SeedAccessRecorder,
) -> RuntimeCalibrationResult:
    del case, workers
    if trace_lock_path is not None:
        raise ValueError("exact runtime adapter does not accept a trace lock")
    definition = load_definition(family.config_path)
    calibrated = replace(
        definition,
        seeds=seeds,
        scenario=replace(definition.scenario, num_slots=max_slots),
    )
    temporary = Path(tempfile.mkdtemp(prefix="uav-mec-exact-runtime-"))
    output = temporary / "run"
    cpu_started = time.process_time()
    try:
        run_exact_definition(
            calibrated,
            family.config_path,
            output,
            seed_access=seed_access,
        )
        measured_cpu_seconds = time.process_time() - cpu_started
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_memory_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
        with (output / "exact_slots.csv").open("r", encoding="utf-8", newline="") as handle:
            rows = tuple(csv.DictReader(handle))
        if not rows or any(row["certificate_complete"] != "True" for row in rows):
            raise RuntimeError("exact runtime calibration requires complete certificates")
        runtimes = [float(row["full_runtime_s"]) + float(row["exact_runtime_s"]) for row in rows]
        candidates = [int(row["raw_states_enumerated"]) for row in rows]
        raw_bytes = (output / "exact_slots.csv").stat().st_size
        summary_bytes = (output / "summary.json").stat().st_size
        return RuntimeCalibrationResult(
            audit=_runtime_audit(
                runtimes,
                candidates,
                0,
                peak_memory_bytes=peak_memory_bytes,
                log_bytes=0,
                measured_cpu_seconds=measured_cpu_seconds,
            ),
            raw_bytes=raw_bytes,
            summary_bytes=summary_bytes,
            event_bytes=0,
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _measure_materialized_trace_scenarios(
    scenarios: Sequence[MaterializedLockedScenario],
    *,
    max_slots: int,
    definition: object,
    seed: int,
) -> tuple[list[float], list[int], int, int, int, int]:
    """Execute registered trace methods and return runtime/storage strata."""

    if not hasattr(definition, "scenario"):
        raise TypeError("trace calibration definition is invalid")
    runtimes: list[float] = []
    candidate_counts: list[int] = []
    raw_bytes = 0
    event_bytes = 0
    cache_hits = 0
    summary_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        execution_started = time.perf_counter()
        slot_count = min(len(scenario.timestamps_s), max_slots)
        calibrated_scenario = replace(definition.scenario, num_slots=slot_count)
        trace = _locked_experiment_trace(
            scenario,
            calibrated_scenario,
            seed,
            definition.csi_error_std,
        )
        task_by_id = {task.task_id: task for snapshot in trace.slots for task in snapshot.tasks}
        runtime_by_slot = [0.0 for _ in range(slot_count)]
        candidates_by_slot = [0 for _ in range(slot_count)]
        for method in definition.methods:
            events: list[dict[str, object]] = []
            target = replace(
                definition.evaluation,
                allocation_policy=thesis_method_specs()[method].allocation_policy,
            )
            episode = run_outcome_episode(
                trace,
                _slot_method(
                    method,
                    calibrated_scenario,
                    target,
                    definition.max_iterations,
                    events,
                ),
            )
            for outcome in episode.outcomes:
                runtime_by_slot[outcome.slot_index] += outcome.runtime_s
                raw_bytes += (
                    _json_bytes(
                        {
                            "scenario_id": scenario.scenario_id,
                            "method": method,
                            "slot": outcome.slot_index,
                            "arrivals": len(outcome.arrived_task_ids),
                            "completed": outcome.completed_count,
                        }
                    )
                    + 1
                )
                for task_id in outcome.arrived_task_ids:
                    task = task_by_id[task_id]
                    raw_bytes += (
                        _json_bytes(
                            {
                                "scenario_id": scenario.scenario_id,
                                "method": method,
                                "slot": outcome.slot_index,
                                "task_id": task_id,
                                "user_id": task.user_id,
                                "data_bits": task.data_bits,
                                "comp_cycles": task.comp_cycles,
                                "deadline_s": task.deadline_s,
                            }
                        )
                        + 1
                    )
            for event in events:
                event_bytes += (
                    _json_bytes(
                        {
                            "scenario_id": scenario.scenario_id,
                            "method": method,
                            **event,
                        }
                    )
                    + 1
                )
                if event.get("move_type") == "termination":
                    slot = int(event["slot"])
                    candidates_by_slot[slot] += int(event.get("raw_move_count", 0))
                    cache_hits += int(event.get("cache_hits", 0))
        end_to_end_runtime = time.perf_counter() - execution_started
        accounted_runtime = math.fsum(runtime_by_slot)
        if runtime_by_slot:
            runtime_by_slot[0] += max(0.0, end_to_end_runtime - accounted_runtime)
        runtimes.extend(runtime_by_slot)
        candidate_counts.extend(candidates_by_slot)
        summary_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "participant_id": scenario.participant_id,
                "split": scenario.split,
                "slot_count": slot_count,
                "methods": list(definition.methods),
            }
        )
    return (
        runtimes,
        candidate_counts,
        raw_bytes,
        _json_bytes(summary_rows) + 1,
        event_bytes,
        cache_hits,
    )


def _trace_runtime_measurement(
    *,
    family: ExperimentFamily,
    case: RegisteredCase,
    seeds: tuple[int, ...],
    max_slots: int,
    workers: int,
    trace_lock_path: Path | None,
    seed_access: SeedAccessRecorder,
) -> RuntimeCalibrationResult:
    del workers
    if family.name != "trace_driven" or case.case_id != "geolife_final":
        raise ValueError("trace runtime adapter received an unregistered family or case")
    if trace_lock_path is None:
        raise ValueError("trace runtime adapter requires a trace lock")
    lock_snapshot = load_trace_lock_snapshot(trace_lock_path)
    definition = load_definition(family.config_path)
    runtimes: list[float] = []
    sample_counts: list[int] = []
    raw_bytes = 0
    summary_bytes = 0
    event_bytes = 0
    cache_hits = 0
    cpu_started = time.process_time()
    for seed in seeds:
        materialize_started = time.perf_counter()
        scenarios = materialize_calibration_scenarios(
            trace_lock_path,
            seed=seed,
            seed_access=seed_access,
            _lock_snapshot=lock_snapshot,
        )
        materialize_seconds = time.perf_counter() - materialize_started
        measured, counts, raw_size, summary_size, event_size, measured_cache_hits = (
            _measure_materialized_trace_scenarios(
                scenarios,
                max_slots=max_slots,
                definition=definition,
                seed=seed,
            )
        )
        if measured:
            measured[0] += materialize_seconds
        runtimes.extend(measured)
        sample_counts.extend(counts)
        raw_bytes += raw_size
        summary_bytes += summary_size
        event_bytes += event_size
        cache_hits += measured_cache_hits
    measured_cpu_seconds = time.process_time() - cpu_started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    return RuntimeCalibrationResult(
        audit=_runtime_audit(
            runtimes,
            sample_counts,
            cache_hits,
            peak_memory_bytes=peak_memory_bytes,
            log_bytes=event_bytes,
            measured_cpu_seconds=measured_cpu_seconds,
        ),
        raw_bytes=raw_bytes,
        summary_bytes=summary_bytes,
        event_bytes=event_bytes,
    )


# Deliberately one independently patchable adapter per registered family.
_RUNTIME_ADAPTERS: dict[str, RuntimeAdapter] = {
    "resource": _resource_runtime_measurement,
    "exact": _exact_runtime_measurement,
    "externality": _generic_runtime_measurement,
    "main": _generic_runtime_measurement,
    "joint": _generic_runtime_measurement,
    "scaling": _generic_runtime_measurement,
    "robustness": _generic_runtime_measurement,
    "trace_driven": _trace_runtime_measurement,
}


def _validated_seeds(seeds: Sequence[int], registry: ExperimentRegistry) -> tuple[int, ...]:
    values = tuple(seeds)
    if not values:
        raise ValueError("development calibration requires at least one seed")
    if any(type(seed) is not int for seed in values):
        raise TypeError("development calibration seeds must be integers")
    if values != tuple(sorted(set(values))):
        raise ValueError("development calibration seeds must be sorted and unique")
    if any(seed not in set(registry.development_seeds) for seed in values):
        raise ValueError("seed outside development pool")
    return values


def _validate_trace_lock(path: Path) -> str:
    snapshot = load_trace_lock_snapshot(path)
    if not any(segment.split == "calibration" for segment in snapshot.lock.segments):
        raise ValueError("development runtime trace lock must contain calibration scenarios")
    return snapshot.sha256


def _ledger_head(path: Path) -> str:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("development seed ledger is empty")
    value = json.loads(lines[-1])
    digest = value.get("digest") if isinstance(value, dict) else None
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("development seed ledger head is invalid")
    return digest


def _source_git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _environment_identity() -> tuple[dict[str, object], str]:
    payload: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }
    return payload, _canonical_digest(payload)


def _publish_bundle(
    output_dir: Path,
    *,
    payload_name: str,
    payload: Mapping[str, object],
    registry_path: Path,
    requested_case: DevelopmentCalibrationCase,
    seeds: tuple[int, ...],
    trace_lock_sha256: str | None,
    seed_access: SeedAccessRecorder,
) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}."))
    bundle = staging_root / "bundle"
    try:
        writer = RunArtifactWriter(bundle, bundle_version=2)
        writer.write_json(payload_name, dict(payload))
        ledger_inside = seed_access.ledger_path.resolve().is_relative_to(bundle.resolve())
        if not ledger_inside:
            writer.copy_file("seed-access.jsonl", seed_access.ledger_path)
        environment, environment_sha256 = _environment_identity()
        writer.finalize(
            {
                "schema_version": 1,
                "registry_sha256": _sha256(registry_path),
                "source_git_sha": _source_git_sha(),
                "environment": environment,
                "environment_sha256": environment_sha256,
                "requested_case": requested_case.value,
                "seeds": list(seeds),
                "trace_lock_sha256": trace_lock_sha256,
                "seed_ledger_head_sha256": _ledger_head(seed_access.ledger_path),
            }
        )
        os.replace(bundle, destination)
        return destination
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def run_development_calibration(
    registry_path: Path,
    case: DevelopmentCalibrationCase,
    seeds: Sequence[int],
    output_dir: Path,
    *,
    workers: int,
    trace_lock_path: Path | None,
    seed_access: SeedAccessRecorder,
) -> Path:
    """Run one seed-guarded calibration and atomically publish a v2 bundle."""

    if not isinstance(case, DevelopmentCalibrationCase):
        raise TypeError("case must be a DevelopmentCalibrationCase")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not isinstance(seed_access, SeedAccessRecorder):
        raise TypeError("seed_access must be a SeedAccessRecorder")
    if seed_access.permit.purpose is not SeedPurpose.DEVELOPMENT:
        raise ValueError("development calibration requires a development permit")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)

    registry_path = Path(registry_path)
    registry = load_registry(registry_path)
    seed_values = _validated_seeds(seeds, registry)
    if seed_access.permit.registry_sha256 != _sha256(registry_path):
        raise ValueError("seed permit registry digest mismatch")

    if case is DevelopmentCalibrationCase.EXACT_NONTRIVIALITY:
        if trace_lock_path is not None:
            raise ValueError("exact_nontriviality forbids a trace lock")
        trace_digest = None
    else:
        if trace_lock_path is None:
            raise ValueError("runtime_capacity requires a trace lock")
        trace_digest = _validate_trace_lock(trace_lock_path)

    seed_access.record(
        "development_calibration",
        seed_values,
        {
            "case": case.value,
            "workers": workers,
            "trace_lock_sha256": trace_digest,
        },
    )

    if case is DevelopmentCalibrationCase.EXACT_NONTRIVIALITY:
        family = next(item for item in registry.families if item.name == "exact")
        exact_case = next(item for item in family.cases if item.case_id == "exact_schedule")
        rows = tuple(
            _run_exact_nontriviality(
                family=family,
                case=exact_case,
                seeds=seed_values,
                max_slots=min(20, family.num_slots),
                workers=workers,
                trace_lock_path=None,
                seed_access=seed_access,
            )
        )
        if not rows or any(row.get("complete") is not True for row in rows):
            raise RuntimeError("incomplete exact certificate in development calibration")
        denominator = len(rows)
        if denominator % len(seed_values) != 0:
            raise RuntimeError("exact calibration rows do not form complete seed-slot strata")
        cpu_count = sum(row.get("cpu_nontrivial") is True for row in rows)
        schedule_count = sum(row.get("schedule_nontrivial") is True for row in rows)
        payload: Mapping[str, object] = {
            "seed_count": len(seed_values),
            "slot_count": denominator // len(seed_values),
            "complete_certificate_count": denominator,
            "cpu_nontrivial_count": cpu_count,
            "cpu_nontrivial_fraction": cpu_count / denominator,
            "schedule_nontrivial_count": schedule_count,
            "schedule_nontrivial_fraction": schedule_count / denominator,
        }
        payload_name = "exact_nontriviality.json"
    else:
        audit_rows: list[dict[str, object]] = []
        for family in registry.families:
            try:
                adapter = _RUNTIME_ADAPTERS[family.name]
            except KeyError as exc:
                raise RuntimeError(f"missing runtime adapter for family {family.name}") from exc
            for registered_case in family.cases:
                measured = adapter(
                    family=family,
                    case=registered_case,
                    seeds=seed_values,
                    max_slots=min(20, family.num_slots),
                    workers=workers,
                    trace_lock_path=(
                        Path(trace_lock_path) if family.name == "trace_driven" else None
                    ),
                    seed_access=seed_access,
                )
                if isinstance(measured, RuntimeCalibrationResult):
                    audit = measured.audit
                    storage = {
                        "raw_bytes": measured.raw_bytes,
                        "summary_bytes": measured.summary_bytes,
                        "event_bytes": measured.event_bytes,
                    }
                elif isinstance(measured, DescriptiveRuntimeAudit):
                    audit = measured
                    storage = {
                        "raw_bytes": 0,
                        "summary_bytes": 0,
                        "event_bytes": audit.log_bytes,
                    }
                else:
                    raise TypeError(
                        "runtime adapter must return a descriptive audit or measurement"
                    )
                audit_rows.append(
                    {
                        "family": family.name,
                        "case_id": registered_case.case_id,
                        "runner_kind": family.name,
                        "seed_count": len(seed_values),
                        "slot_count": min(20, family.num_slots),
                        **descriptive_runtime_audit_to_dict(audit),
                        **storage,
                    }
                )
        payload = {"schema_version": 1, "rows": audit_rows}
        payload_name = "runtime_audit.json"

    if trace_lock_path is not None and _validate_trace_lock(trace_lock_path) != trace_digest:
        raise ValueError("trace lock changed during development calibration")
    return _publish_bundle(
        destination,
        payload_name=payload_name,
        payload=payload,
        registry_path=registry_path,
        requested_case=case,
        seeds=seed_values,
        trace_lock_sha256=trace_digest,
        seed_access=seed_access,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--case", choices=tuple(item.value for item in DevelopmentCalibrationCase), required=True
    )
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--trace-lock", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry)
        case = DevelopmentCalibrationCase(args.case)
        seeds = parse_development_seed_range(args.seeds, registry)
        if args.workers <= 0:
            raise ValueError("workers must be positive")
        if case is DevelopmentCalibrationCase.RUNTIME_CAPACITY:
            if args.trace_lock is None:
                raise ValueError("runtime_capacity requires --trace-lock")
        elif args.trace_lock is not None:
            raise ValueError("exact_nontriviality forbids --trace-lock")
        if args.output.exists():
            raise FileExistsError(args.output)
    except (FileExistsError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    ledger_root = Path(tempfile.mkdtemp(prefix="uav-mec-development-ledger-"))
    try:
        recorder = SeedAccessRecorder(
            ledger_root / "seed-access.jsonl",
            development_permit(args.registry),
        )
        run_development_calibration(
            args.registry,
            case,
            seeds,
            args.output,
            workers=args.workers,
            trace_lock_path=args.trace_lock,
            seed_access=recorder,
        )
    finally:
        shutil.rmtree(ledger_root, ignore_errors=True)


if __name__ == "__main__":
    main()
