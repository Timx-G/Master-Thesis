"""Configuration-driven trusted experiment runner."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
import tomllib
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
from .model import NOMAChannel, ScenarioConfig
from .resource_allocation import AllocationPolicy
from .scheduler import (
    Acceptance,
    SearchConfig,
    build_initial_schedule,
    optimize_schedule,
    repair_sic,
)
from .simulation import run_outcome_episode
from .trace import generate_trace

KNOWN_METHODS = frozenset(
    {
        "local_only",
        "fair_ofdma",
        "greedy",
        "ia_greedy",
        "ilp",
        "exchange_stable",
        "intra_bilateral",
        "cross_bilateral",
        "intra_global",
        "full",
        "full_fixed_half",
        "full_crossing",
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


def load_definition(path: Path) -> ExperimentDefinition:
    config_path = Path(path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")

    has_explicit_seeds = "seeds" in raw
    has_seed_range = "seed_start" in raw or "seed_count" in raw
    if has_explicit_seeds == has_seed_range:
        raise ValueError(
            "define seeds or seed_start/seed_count, but not both"
        )
    if has_explicit_seeds:
        raw_seeds = raw["seeds"]
        if not isinstance(raw_seeds, list) or any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in raw_seeds
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
        not isinstance(method, str) or not method
        for method in raw_methods
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
    scenario = ScenarioConfig(**scenario_raw)

    evaluation_raw = raw.get("evaluation")
    if not isinstance(evaluation_raw, dict):
        raise ValueError("evaluation section is required")
    try:
        allocation_policy = AllocationPolicy(
            evaluation_raw["allocation_policy"]
        )
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
        cross_interference_scale=float(
            evaluation_raw.get("cross_interference_scale", 1.0)
        ),
    )

    search_raw = raw.get("search")
    if not isinstance(search_raw, dict):
        raise ValueError("search section is required")
    max_iterations = search_raw.get("max_iterations")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise ValueError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    trace_raw = raw.get("trace", {})
    if not isinstance(trace_raw, dict):
        raise ValueError("trace section must be a table")
    csi_error_std = float(trace_raw.get("csi_error_std", 0.0))
    if not np.isfinite(csi_error_std) or csi_error_std < 0.0:
        raise ValueError("csi_error_std must be finite and non-negative")
    return ExperimentDefinition(
        run_id=run_id,
        seeds=seeds,
        methods=methods,
        scenario=scenario,
        evaluation=evaluation,
        max_iterations=max_iterations,
        csi_error_std=csi_error_std,
    )


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _slot_method(
    name: str,
    scenario: ScenarioConfig,
    target_evaluation: EvaluationConfig,
    max_iterations: int,
    event_sink: list[dict[str, object]],
):
    if name not in KNOWN_METHODS:
        raise ValueError(f"unknown method {name}")
    channel = NOMAChannel(scenario.channel, scenario.uav.height)
    search_evaluation = replace(
        target_evaluation,
        allocation_policy=AllocationPolicy.DEADLINE,
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

        initialization = build_initial_schedule(
            planning,
            channel,
            search_evaluation,
        )
        event_sink.append(
            {
                "slot": snapshot.slot_index,
                "move_type": "initialization",
                "accepted": True,
                "reason": initialization.solver_status,
                "before": None,
                "after": None,
                "schedule_key": initialization.schedule.stable_key(),
            }
        )
        if name == "ilp":
            return evaluate_noma(snapshot, initialization.schedule)

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
            "intra_global": SearchConfig(
                False,
                Acceptance.GLOBAL,
                max_iterations,
            ),
            "full": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
            ),
            "full_fixed_half": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
            ),
            "full_crossing": SearchConfig(
                True,
                Acceptance.GLOBAL,
                max_iterations,
            ),
        }
        result = optimize_schedule(
            initialization.schedule,
            planning,
            channel,
            search_evaluation,
            search_map[name],
        )
        if not result.converged:
            raise RuntimeError(
                f"search hit max_iterations={max_iterations} "
                f"at slot {snapshot.slot_index}"
            )
        event_sink.extend(
            {
                "slot": snapshot.slot_index,
                "move_type": event.move_type,
                "accepted": event.accepted,
                "reason": event.reason,
                "before": event.before_objective,
                "after": event.after_objective,
                "schedule_key": event.schedule_key,
            }
            for event in result.events
        )
        event_sink.append(
            {
                "slot": snapshot.slot_index,
                "move_type": "termination",
                "accepted": True,
                "reason": result.termination_reason,
                "before": None,
                "after": result.iterations,
                "schedule_key": result.schedule.stable_key(),
            }
        )
        return evaluate_noma(snapshot, result.schedule)

    return method


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


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
    task_rows: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    method_summary = {name: [] for name in definition.methods}

    for seed in definition.seeds:
        trace = generate_trace(
            definition.scenario,
            seed,
            definition.csi_error_std,
        )
        task_by_id = {
            task.task_id: task
            for snapshot in trace.slots
            for task in snapshot.tasks
        }
        for name in definition.methods:
            target = definition.evaluation
            if name == "full_fixed_half":
                target = replace(
                    target,
                    allocation_policy=AllocationPolicy.FIXED_HALF_EQUAL,
                )
            elif name == "full_crossing":
                target = replace(
                    target,
                    allocation_policy=AllocationPolicy.CROSSING_EQUAL,
                )

            events: list[dict[str, object]] = []
            episode = run_outcome_episode(
                trace,
                _slot_method(
                    name,
                    definition.scenario,
                    target,
                    definition.max_iterations,
                    events,
                ),
            )
            method_summary[name].append(episode.on_time_rate)
            for outcome in episode.outcomes:
                decision_by_id = {
                    decision.task_id: decision
                    for decision in outcome.decisions
                }
                rate_by_id = {
                    record.task_id: record for record in outcome.rates
                }
                for task_id in outcome.arrived_task_ids:
                    task = task_by_id[task_id]
                    decision = decision_by_id[task_id]
                    rate = rate_by_id.get(task_id)
                    task_rows.append(
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
                            "rate_bps": None if rate is None else rate.rate_bps,
                            "sinr": None if rate is None else rate.sinr,
                            "sic_feasible": (
                                None if rate is None else rate.sic_feasible
                            ),
                            "cross_interference_w": (
                                None
                                if rate is None
                                else rate.cross_interference_w
                            ),
                            "serving_uav": (
                                None if rate is None else rate.serving_uav
                            ),
                            "subcarrier": (
                                None if rate is None else rate.subcarrier
                            ),
                            "reason": decision.reason,
                        }
                    )
                slot_rows.append(
                    {
                        "method": name,
                        "seed": seed,
                        "slot": outcome.slot_index,
                        "arrivals": len(outcome.arrived_task_ids),
                        "completed": outcome.completed_count,
                        "slack_s": outcome.total_slack_s,
                        "cross_interference_w": (
                            outcome.total_cross_interference_w
                        ),
                        "sic_violations": sum(
                            not record.sic_feasible
                            for record in outcome.rates
                        ),
                        "runtime_s": outcome.runtime_s,
                    }
                )
            event_rows.extend(
                {"method": name, "seed": seed, **event}
                for event in events
            )

    initializer_statuses = sorted(
        {
            str(row["reason"])
            for row in event_rows
            if row.get("move_type") == "initialization"
        }
    )
    termination_statuses = sorted(
        {
            str(row["reason"])
            for row in event_rows
            if row.get("move_type") == "termination"
        }
    )
    summary = {
        "run_id": definition.run_id,
        "seeds": list(definition.seeds),
        "initializer_status_counts": {
            status: sum(
                row.get("move_type") == "initialization"
                and row.get("reason") == status
                for row in event_rows
            )
            for status in initializer_statuses
        },
        "search_termination_counts": {
            status: sum(
                row.get("move_type") == "termination"
                and row.get("reason") == status
                for row in event_rows
            )
            for status in termination_statuses
        },
        "methods": {
            name: {
                "on_time_rate_by_seed": values,
                "mean": float(np.mean(values)),
                "std": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                ),
            }
            for name, values in method_summary.items()
        },
    }
    writer.write_json("summary.json", summary)
    writer.write_csv(
        "tasks.csv",
        task_rows,
        fieldnames=(
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
        ),
    )
    writer.write_csv(
        "slots.csv",
        slot_rows,
        fieldnames=(
            "method",
            "seed",
            "slot",
            "arrivals",
            "completed",
            "slack_s",
            "cross_interference_w",
            "sic_violations",
            "runtime_s",
        ),
    )
    writer.write_jsonl("events.jsonl", event_rows)
    config_path = Path(config_path)
    writer.finalize(
        {
            "run_id": definition.run_id,
            "git_sha": _git_sha(),
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "seeds": list(definition.seeds),
            "csi_error_std": definition.csi_error_std,
            "resolved_definition": definition_to_dict(definition),
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
