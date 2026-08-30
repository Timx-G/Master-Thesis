"""Configuration-driven trusted experiment runner."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
import tomllib
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np
import scipy

from .artifacts import RunArtifactWriter
from .baselines_v2 import (
    evaluate_fair_ofdma,
    greedy_schedule,
    ia_greedy_schedule,
    local_only_schedule,
)
from .domain import Schedule, SlotOutcome, SlotSnapshot
from .evaluation import (
    EvaluationConfig,
    evaluate_realized_schedule,
)
from .model import (
    ChannelConfig,
    NOMAChannel,
    ScenarioConfig,
    TaskConfig,
    UAVConfig,
)
from .resource_allocation import AllocationPolicy
from .scheduler import (
    Acceptance,
    Improvement,
    SearchConfig,
    build_initial_schedule,
    optimize_multistart,
    optimize_schedule,
    repair_sic,
)
from .simulation import run_outcome_episode
from .trace import generate_trace

RESOURCE_POLICY_BY_METHOD = {
    "exchange_edf": AllocationPolicy.EDF_MIN_CPU,
    "exchange_dd": AllocationPolicy.DEADLINE,
    "iea_edf": AllocationPolicy.EDF_MIN_CPU,
    "iea_dd": AllocationPolicy.DEADLINE,
    "full_fixed_half": AllocationPolicy.FIXED_HALF_EQUAL,
    "full_crossing": AllocationPolicy.CROSSING_EQUAL,
    "full_edf_min_cpu": AllocationPolicy.EDF_MIN_CPU,
    "full_min_slack_min_cpu": AllocationPolicy.MIN_SLACK_MIN_CPU,
    "full": AllocationPolicy.DEADLINE,
    "full_exact_oracle": AllocationPolicy.EXACT_ORACLE,
    "fixed_radio_dd": AllocationPolicy.DEADLINE,
    "fixed_radio_edf": AllocationPolicy.EDF_MIN_CPU,
    "fixed_radio_min_slack": AllocationPolicy.MIN_SLACK_MIN_CPU,
    "fixed_radio_exact": AllocationPolicy.EXACT_ORACLE,
}

FIXED_RADIO_METHODS = frozenset(
    {
        "fixed_radio_dd",
        "fixed_radio_edf",
        "fixed_radio_min_slack",
        "fixed_radio_exact",
    }
)

KNOWN_METHODS = frozenset(
    {
        "local_only",
        "fair_ofdma",
        "greedy",
        "ia_greedy",
        "ilp",
        "exchange_stable",
        "exchange_edf",
        "exchange_dd",
        "intra_bilateral",
        "cross_bilateral",
        "intra_global",
        "full",
        "iea_edf",
        "iea_dd",
        "full_single_first",
        "full_fixed_half",
        "full_crossing",
        "full_edf_min_cpu",
        "full_min_slack_min_cpu",
        "full_exact_oracle",
        "fixed_radio_dd",
        "fixed_radio_edf",
        "fixed_radio_min_slack",
        "fixed_radio_exact",
        "exact",
    }
)

MULTISTART_METHODS = frozenset(
    {
        "exchange_edf",
        "exchange_dd",
        "cross_bilateral",
        "intra_global",
        "full",
        "full_fixed_half",
        "full_crossing",
        "full_edf_min_cpu",
        "full_min_slack_min_cpu",
        "full_exact_oracle",
        "iea_edf",
        "iea_dd",
        "fixed_radio_dd",
        "fixed_radio_edf",
        "fixed_radio_min_slack",
        "fixed_radio_exact",
    }
)


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    run_id: str
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    scenario: ScenarioConfig
    evaluation: EvaluationConfig
    max_iterations: int
    csi_error_std: float = 0.0
    event_log_level: str = "full"
    multistart_workers: int = 1


def load_definition(path: Path) -> ExperimentDefinition:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")

    has_explicit_seeds = "seeds" in raw
    has_seed_range = "seed_start" in raw or "seed_count" in raw
    if has_explicit_seeds == has_seed_range:
        raise ValueError("define seeds or seed_start/seed_count, but not both")
    if has_explicit_seeds:
        raw_seeds = raw["seeds"]
        if not isinstance(raw_seeds, list) or any(
            not isinstance(seed, int) or isinstance(seed, bool) for seed in raw_seeds
        ):
            raise ValueError("seeds must be an integer list")
        seeds = tuple(raw_seeds)
    else:
        seed_start = raw.get("seed_start")
        seed_count = raw.get("seed_count")
        if (
            not isinstance(seed_start, int)
            or isinstance(seed_start, bool)
            or not isinstance(seed_count, int)
            or isinstance(seed_count, bool)
            or seed_count <= 0
        ):
            raise ValueError("seed_start and seed_count must be valid integers")
        seeds = tuple(range(seed_start, seed_start + seed_count))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")

    raw_methods = raw.get("methods")
    if not isinstance(raw_methods, list) or any(
        not isinstance(method, str) or not method for method in raw_methods
    ):
        raise ValueError("methods must be a string list")
    methods = tuple(raw_methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be non-empty and unique")
    unknown = sorted(set(methods) - KNOWN_METHODS)
    if unknown:
        raise ValueError(f"unknown method {unknown[0]}")

    scenario_raw = raw.get("scenario")
    if not isinstance(scenario_raw, dict):
        raise ValueError("scenario section is required")
    scenario_values = dict(scenario_raw)
    for field_name, config_type in (
        ("uav", UAVConfig),
        ("channel", ChannelConfig),
        ("task", TaskConfig),
    ):
        if field_name not in scenario_values:
            continue
        nested = scenario_values[field_name]
        if not isinstance(nested, dict):
            raise ValueError(f"scenario.{field_name} must be a table")
        scenario_values[field_name] = config_type(**nested)
    scenario = ScenarioConfig(**scenario_values)

    evaluation_raw = raw.get("evaluation")
    if not isinstance(evaluation_raw, dict):
        raise ValueError("evaluation section is required")
    try:
        allocation_policy = AllocationPolicy(evaluation_raw["allocation_policy"])
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid evaluation allocation_policy") from exc
    evaluation = EvaluationConfig(
        n_subcarriers=scenario.n_subcarriers,
        bandwidth_hz=scenario.channel.bandwidth,
        transmit_power_w=scenario.channel.p_max,
        sic_residual=float(evaluation_raw["sic_residual"]),
        gamma_sic_min=float(evaluation_raw["gamma_sic_min"]),
        cpu_per_uav_hz=scenario.f_per_uav,
        result_ratio=scenario.task.result_ratio,
        allocation_policy=allocation_policy,
        cross_interference_scale=float(evaluation_raw.get("cross_interference_scale", 1.0)),
    )

    search_raw = raw.get("search")
    if not isinstance(search_raw, dict):
        raise ValueError("search section is required")
    max_iterations = search_raw.get("max_iterations")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise ValueError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    multistart_workers = search_raw.get("multistart_workers", 1)
    if (
        not isinstance(multistart_workers, int)
        or isinstance(multistart_workers, bool)
        or multistart_workers <= 0
    ):
        raise ValueError("search.multistart_workers must be a positive integer")
    trace_raw = raw.get("trace", {})
    if not isinstance(trace_raw, dict):
        raise ValueError("trace section must be a table")
    csi_error_std = float(trace_raw.get("csi_error_std", 0.0))
    if not np.isfinite(csi_error_std) or csi_error_std < 0.0:
        raise ValueError("csi_error_std must be finite and non-negative")
    event_log_level = raw.get("event_log_level", "full")
    if event_log_level not in {"full", "summary"}:
        raise ValueError("event_log_level must be full or summary")
    return ExperimentDefinition(
        run_id=run_id,
        seeds=seeds,
        methods=methods,
        scenario=scenario,
        evaluation=evaluation,
        max_iterations=max_iterations,
        csi_error_std=csi_error_std,
        event_log_level=event_log_level,
        multistart_workers=multistart_workers,
    )


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _git_worktree_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _source_snapshot() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parent.parent
    source_paths = sorted((repo_root / "uav_mec").rglob("*.py"))
    files = {
        str(path.relative_to(repo_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_paths
    }
    aggregate = hashlib.sha256()
    for relative, digest in files.items():
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "sha256": aggregate.hexdigest(),
        "files": files,
    }


def _slot_method(
    name: str,
    scenario: ScenarioConfig,
    target_evaluation: EvaluationConfig,
    max_iterations: int,
    event_sink: list[dict[str, object]],
    event_log_level: str = "full",
    multistart_workers: int = 1,
    multistart_executor: Executor | None = None,
):
    if name not in KNOWN_METHODS:
        raise ValueError(f"unknown method {name}")
    if name == "exact":
        raise ValueError("exact requires the dedicated exact benchmark runner")
    if event_log_level not in {"full", "summary"}:
        raise ValueError("event_log_level must be full or summary")
    channel = NOMAChannel(scenario.channel, scenario.uav.height)
    search_evaluation = (
        replace(
            target_evaluation,
            allocation_policy=AllocationPolicy.DEADLINE,
        )
        if name in FIXED_RADIO_METHODS
        else target_evaluation
    )

    def evaluate_noma(
        snapshot: SlotSnapshot,
        schedule: Schedule,
    ) -> SlotOutcome:
        return evaluate_realized_schedule(
            snapshot,
            schedule,
            channel,
            target_evaluation,
        )

    def method(snapshot: SlotSnapshot) -> SlotOutcome:
        planning = snapshot.as_planning_snapshot()
        if name == "local_only":
            return evaluate_noma(snapshot, local_only_schedule(planning))
        if name == "fair_ofdma":
            return evaluate_fair_ofdma(
                snapshot,
                channel,
                target_evaluation,
            )
        if name == "greedy":
            schedule = repair_sic(
                greedy_schedule(planning, scenario.n_subcarriers),
                planning,
                channel,
                search_evaluation,
            )
            return evaluate_noma(snapshot, schedule)
        if name == "ia_greedy":
            schedule = ia_greedy_schedule(
                planning,
                channel,
                search_evaluation,
            )
            return evaluate_noma(snapshot, schedule)

        if name == "ilp":
            initialization = build_initial_schedule(
                planning,
                channel,
                search_evaluation,
            )
            event_sink.append(
                {
                    "slot": snapshot.slot_index,
                    "move_type": "initialization",
                    "initializer_id": "per_uav_ilp",
                    "accepted": True,
                    "reason": initialization.solver_status,
                    "before": None,
                    "after": None,
                    "schedule_key": initialization.schedule.stable_key(),
                }
            )
            return evaluate_noma(snapshot, initialization.schedule)

        initializer_records: list[tuple[str, Schedule, str]] = []
        if (
            name in MULTISTART_METHODS
            and multistart_executor is not None
            and multistart_workers > 1
        ):
            def initializer_stream():
                fair_load = repair_sic(
                    greedy_schedule(planning, scenario.n_subcarriers),
                    planning,
                    channel,
                    search_evaluation,
                )
                initializer_records.append(
                    ("fair_load_greedy", fair_load, "deterministic")
                )
                yield "fair_load_greedy", fair_load

                initialization = build_initial_schedule(
                    planning,
                    channel,
                    search_evaluation,
                )
                initializer_records.append(
                    (
                        "per_uav_ilp",
                        initialization.schedule,
                        initialization.solver_status,
                    )
                )
                yield "per_uav_ilp", initialization.schedule

                interference_aware = ia_greedy_schedule(
                    planning,
                    channel,
                    search_evaluation,
                )
                initializer_records.append(
                    ("ia_greedy", interference_aware, "deterministic")
                )
                yield "ia_greedy", interference_aware

            initializers = initializer_stream()
        else:
            initialization = build_initial_schedule(
                planning,
                channel,
                search_evaluation,
            )
            initialization_schedule = initialization.schedule
            initializer_records.append(
                (
                    "per_uav_ilp",
                    initialization_schedule,
                    initialization.solver_status,
                )
            )
            if name in MULTISTART_METHODS:
                fair_load = repair_sic(
                    greedy_schedule(planning, scenario.n_subcarriers),
                    planning,
                    channel,
                    search_evaluation,
                )
                interference_aware = ia_greedy_schedule(
                    planning,
                    channel,
                    search_evaluation,
                )
                initializer_records.extend(
                    (
                        ("ia_greedy", interference_aware, "deterministic"),
                        ("fair_load_greedy", fair_load, "deterministic"),
                    )
                )
            initializers = tuple(
                (initializer_id, initializer_schedule)
                for initializer_id, initializer_schedule, _status in initializer_records
            )

        search_map = {
            "exchange_stable": SearchConfig(
                False,
                Acceptance.BILATERAL,
                max_iterations,
            ),
            "intra_bilateral": SearchConfig(
                False,
                Acceptance.BILATERAL,
                max_iterations,
            ),
            "cross_bilateral": SearchConfig(
                True,
                Acceptance.BILATERAL,
                max_iterations,
            ),
            "exchange_edf": SearchConfig(
                True,
                Acceptance.BILATERAL,
                max_iterations,
                Improvement.BEST,
            ),
            "exchange_dd": SearchConfig(
                True,
                Acceptance.BILATERAL,
                max_iterations,
                Improvement.BEST,
            ),
            "intra_global": SearchConfig(
                False,
                Acceptance.GLOBAL,
                max_iterations,
            ),
            "full": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
                Improvement.BEST,
            ),
            "iea_edf": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
                Improvement.BEST,
            ),
            "iea_dd": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
                Improvement.BEST,
            ),
            "full_single_first": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
                Improvement.FIRST,
            ),
            "full_fixed_half": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
                Improvement.BEST,
            ),
            "full_crossing": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
                Improvement.BEST,
            ),
            "full_edf_min_cpu": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
                Improvement.BEST,
            ),
            "full_min_slack_min_cpu": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
                Improvement.BEST,
            ),
            "full_exact_oracle": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
                Improvement.BEST,
            ),
        }
        if name in {"cross_bilateral", "intra_global"}:
            base = search_map[name]
            search_map[name] = replace(base, improvement=Improvement.BEST)
        search_method = "full" if name in FIXED_RADIO_METHODS else name
        search_config = replace(
            search_map[search_method],
            record_rejected_events=event_log_level == "full",
            multistart_workers=multistart_workers,
        )
        if name in MULTISTART_METHODS:
            if multistart_executor is None:
                result = optimize_multistart(
                    initializers,
                    planning,
                    channel,
                    search_evaluation,
                    search_config,
                )
            else:
                result = optimize_multistart(
                    initializers,
                    planning,
                    channel,
                    search_evaluation,
                    search_config,
                    executor=multistart_executor,
                )
        else:
            result = optimize_schedule(
                initialization_schedule,
                planning,
                channel,
                search_evaluation,
                replace(search_config, initializer_id="per_uav_ilp"),
            )
        initializer_order = {
            "per_uav_ilp": 0,
            "ia_greedy": 1,
            "fair_load_greedy": 2,
        }
        for initializer_id, initializer_schedule, status in sorted(
            initializer_records,
            key=lambda record: initializer_order[record[0]],
        ):
            event_sink.append(
                {
                    "slot": snapshot.slot_index,
                    "move_type": "initialization",
                    "initializer_id": initializer_id,
                    "accepted": True,
                    "reason": status,
                    "before": None,
                    "after": None,
                    "schedule_key": initializer_schedule.stable_key(),
                }
            )
        if not result.converged:
            raise RuntimeError(
                f"search hit max_iterations={max_iterations} " f"at slot {snapshot.slot_index}"
            )
        event_sink.extend(
            {
                "slot": snapshot.slot_index,
                "move_type": event.move_type,
                "operation": event.operation,
                "accepted": event.accepted,
                "reason": event.reason,
                "before": event.before_objective,
                "after": event.after_objective,
                "schedule_key": event.schedule_key,
                "external_rate_affected_task_ids": (event.external_rate_affected_task_ids),
                "externality_certificate": (
                    None
                    if event.certificate is None
                    else _jsonable(asdict(event.certificate))
                ),
            }
            for event in result.events
        )
        event_sink.append(
            {
                "slot": snapshot.slot_index,
                "move_type": "termination",
                "accepted": True,
                "reason": result.termination_reason,
                "initializer_id": result.initializer_id,
                "improvement": search_map[search_method].improvement.value,
                "candidate_evaluations": result.candidate_evaluations,
                "before": None,
                "after": result.iterations,
                "schedule_key": result.schedule.stable_key(),
            }
        )
        return evaluate_noma(snapshot, result.schedule)

    return method


def _jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _warm_multistart_worker(worker_index: int) -> int:
    """Start a persistent worker before per-slot online timing begins."""

    return worker_index


def definition_to_dict(
    definition: ExperimentDefinition,
) -> dict[str, object]:
    value = _jsonable(asdict(definition))
    if not isinstance(value, dict):
        raise TypeError("serialized experiment definition must be a mapping")
    return value


def run_definition(
    definition: ExperimentDefinition,
    config_path: Path,
    output_dir: Path,
) -> None:
    writer = RunArtifactWriter(output_dir)
    method_summary = {name: [] for name in definition.methods}
    initializer_status_counts: dict[str, int] = {}
    termination_status_counts: dict[str, int] = {}
    task_fieldnames = (
        "method",
        "seed",
        "slot",
        "task_id",
        "user_id",
        "arrival_slot",
        "data_bits",
        "comp_cycles",
        "deadline_s",
        "local_cpu_hz",
        "completed",
        "delay_s",
        "alpha",
        "cpu_hz",
        "rate_bps",
        "sinr",
        "sic_feasible",
        "cross_interference_w",
        "serving_uav",
        "subcarrier",
        "reason",
    )
    slot_fieldnames = (
        "method",
        "seed",
        "slot",
        "arrivals",
        "completed",
        "slack_s",
        "cross_interference_w",
        "sic_violations",
        "runtime_s",
    )

    with (
        writer.open_csv("tasks.csv", task_fieldnames) as write_task,
        writer.open_csv("slots.csv", slot_fieldnames) as write_slot,
        writer.open_jsonl("events.jsonl") as write_event,
    ):
        for seed in definition.seeds:
            trace = generate_trace(
                definition.scenario,
                seed,
                definition.csi_error_std,
            )
            task_by_id = {task.task_id: task for snapshot in trace.slots for task in snapshot.tasks}
            for name in definition.methods:
                target = definition.evaluation
                resource_policy = RESOURCE_POLICY_BY_METHOD.get(name)
                if resource_policy is not None:
                    target = replace(
                        target,
                        allocation_policy=resource_policy,
                    )

                events: list[dict[str, object]] = []
                worker_count = min(definition.multistart_workers, 3)
                pool_context = (
                    ProcessPoolExecutor(max_workers=worker_count)
                    if name in MULTISTART_METHODS and worker_count > 1
                    else nullcontext(None)
                )
                with pool_context as multistart_executor:
                    if multistart_executor is not None:
                        warmups = tuple(
                            multistart_executor.submit(
                                _warm_multistart_worker,
                                worker_index,
                            )
                            for worker_index in range(worker_count)
                        )
                        for warmup in warmups:
                            warmup.result()
                    episode = run_outcome_episode(
                        trace,
                        _slot_method(
                            name,
                            definition.scenario,
                            target,
                            definition.max_iterations,
                            events,
                            definition.event_log_level,
                            definition.multistart_workers,
                            multistart_executor,
                        ),
                    )
                method_summary[name].append(episode.on_time_rate)
                for outcome in episode.outcomes:
                    decision_by_id = {decision.task_id: decision for decision in outcome.decisions}
                    rate_by_id = {record.task_id: record for record in outcome.rates}
                    for task_id in outcome.arrived_task_ids:
                        task = task_by_id[task_id]
                        decision = decision_by_id[task_id]
                        rate = rate_by_id.get(task_id)
                        write_task(
                            {
                                "method": name,
                                "seed": seed,
                                "slot": outcome.slot_index,
                                "task_id": task_id,
                                "user_id": task.user_id,
                                "arrival_slot": task.arrival_slot,
                                "data_bits": task.data_bits,
                                "comp_cycles": task.comp_cycles,
                                "deadline_s": task.deadline_s,
                                "local_cpu_hz": task.local_cpu_hz,
                                "completed": decision.completed,
                                "delay_s": decision.delay_s,
                                "alpha": decision.alpha,
                                "cpu_hz": decision.cpu_hz,
                                "rate_bps": (None if rate is None else rate.rate_bps),
                                "sinr": None if rate is None else rate.sinr,
                                "sic_feasible": (None if rate is None else rate.sic_feasible),
                                "cross_interference_w": (
                                    None if rate is None else rate.cross_interference_w
                                ),
                                "serving_uav": (None if rate is None else rate.serving_uav),
                                "subcarrier": (None if rate is None else rate.subcarrier),
                                "reason": decision.reason,
                            }
                        )
                    write_slot(
                        {
                            "method": name,
                            "seed": seed,
                            "slot": outcome.slot_index,
                            "arrivals": len(outcome.arrived_task_ids),
                            "completed": outcome.completed_count,
                            "slack_s": outcome.total_slack_s,
                            "cross_interference_w": (outcome.total_cross_interference_w),
                            "sic_violations": sum(
                                not record.sic_feasible for record in outcome.rates
                            ),
                            "runtime_s": outcome.runtime_s,
                        }
                    )
                for event in events:
                    row = {"method": name, "seed": seed, **event}
                    move_type = row.get("move_type")
                    if (
                        definition.event_log_level == "full"
                        or move_type
                        in {
                            "initialization",
                            "termination",
                        }
                    ):
                        write_event(row)
                    if move_type not in {
                        "initialization",
                        "termination",
                    }:
                        continue
                    if (
                        move_type == "initialization"
                        and row.get("initializer_id") != "per_uav_ilp"
                    ):
                        continue
                    status = str(row["reason"])
                    counts = (
                        initializer_status_counts
                        if move_type == "initialization"
                        else termination_status_counts
                    )
                    counts[status] = counts.get(status, 0) + 1

    summary = {
        "run_id": definition.run_id,
        "seeds": list(definition.seeds),
        "initializer_status_counts": dict(sorted(initializer_status_counts.items())),
        "search_termination_counts": dict(sorted(termination_status_counts.items())),
        "methods": {
            name: {
                "on_time_rate_by_seed": values,
                "mean": float(np.mean(values)),
                "std": (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
            }
            for name, values in method_summary.items()
        },
    }
    writer.write_json("summary.json", summary)
    config_path = Path(config_path)
    writer.finalize(
        {
            "run_id": definition.run_id,
            "git_sha": _git_sha(),
            "git_worktree_dirty": _git_worktree_dirty(),
            "source_snapshot": _source_snapshot(),
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "seeds": list(definition.seeds),
            "csi_error_std": definition.csi_error_std,
            "resolved_definition": definition_to_dict(definition),
            "resource_policy_by_method": {
                name: RESOURCE_POLICY_BY_METHOD[name].value
                for name in definition.methods
                if name in RESOURCE_POLICY_BY_METHOD
            },
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver": {
                "name": "scipy.optimize.milp/HiGHS",
                "package_version": scipy.__version__,
            },
        }
    )


def run_experiment(config_path: Path, output_dir: Path) -> None:
    run_definition(
        load_definition(config_path),
        config_path,
        output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trusted UAV-MEC thesis experiment",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(args.config, args.output)


if __name__ == "__main__":
    main()
