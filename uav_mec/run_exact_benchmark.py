"""Complete-enumeration comparison against the full heuristic."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import scipy

from .artifacts import RunArtifactWriter
from .exact import exact_schedule
from .experiment import (
    ExperimentDefinition,
    _git_sha,
    _slot_method,
    definition_to_dict,
    load_definition,
)
from .model import NOMAChannel
from .scheduler import network_objective
from .trace import generate_trace


def run_exact_definition(
    definition: ExperimentDefinition,
    config_path: Path,
    output_dir: Path,
) -> None:
    if definition.methods != ("full", "exact"):
        raise ValueError("exact benchmark methods must be ('full', 'exact')")
    if definition.scenario.num_users > 5:
        raise ValueError(
            "complete benchmark is frozen at no more than five users"
        )

    writer = RunArtifactWriter(output_dir)
    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    initializer_statuses: list[str] = []
    search_terminations: list[str] = []
    channel = NOMAChannel(
        definition.scenario.channel,
        definition.scenario.uav.height,
    )

    for seed in definition.seeds:
        trace = generate_trace(
            definition.scenario,
            seed,
            definition.csi_error_std,
        )
        full_completed = 0
        exact_completed = 0
        arrivals = 0
        for snapshot in trace.slots:
            events: list[dict[str, object]] = []
            started = time.perf_counter()
            full = _slot_method(
                "full",
                definition.scenario,
                definition.evaluation,
                definition.max_iterations,
                events,
            )(snapshot)
            full_runtime_s = time.perf_counter() - started
            initializer_statuses.extend(
                str(event["reason"])
                for event in events
                if event.get("move_type") == "initialization"
            )
            search_terminations.extend(
                str(event["reason"])
                for event in events
                if event.get("move_type") == "termination"
            )

            started = time.perf_counter()
            exact = exact_schedule(
                snapshot,
                channel,
                definition.evaluation,
                max_heavy_tasks=5,
            )
            exact_runtime_s = time.perf_counter() - started
            if network_objective(full) > network_objective(exact.outcome):
                raise AssertionError(
                    "heuristic objective exceeds complete enumeration"
                )

            slot_arrivals = len(full.arrived_task_ids)
            arrivals += slot_arrivals
            full_completed += full.completed_count
            exact_completed += exact.outcome.completed_count
            rows.append(
                {
                    "seed": seed,
                    "slot": snapshot.slot_index,
                    "arrivals": slot_arrivals,
                    "full_completed": full.completed_count,
                    "exact_completed": exact.outcome.completed_count,
                    "completion_gap": (
                        exact.outcome.completed_count
                        - full.completed_count
                    ),
                    "full_runtime_s": full_runtime_s,
                    "exact_runtime_s": exact_runtime_s,
                    "evaluated_schedules": exact.evaluated_schedules,
                    "complete_search": exact.complete_search,
                }
            )
        seed_rows.append(
            {
                "seed": seed,
                "arrivals": arrivals,
                "full_on_time_rate": (
                    full_completed / arrivals if arrivals else 0.0
                ),
                "exact_on_time_rate": (
                    exact_completed / arrivals if arrivals else 0.0
                ),
                "completion_gap": exact_completed - full_completed,
            }
        )

    gaps = np.asarray(
        [int(row["completion_gap"]) for row in rows],
        dtype=float,
    )
    if len(gaps) == 0:
        raise ValueError("exact benchmark produced no slot outcomes")
    summary = {
        "run_id": definition.run_id,
        "seeds": list(definition.seeds),
        "optimality_metric": "completed task count per slot",
        "initializer_status_counts": {
            status: initializer_statuses.count(status)
            for status in sorted(set(initializer_statuses))
        },
        "search_termination_counts": {
            status: search_terminations.count(status)
            for status in sorted(set(search_terminations))
        },
        "optimal_slot_fraction": float(np.mean(gaps == 0.0)),
        "mean_completion_gap": float(np.mean(gaps)),
        "max_completion_gap": int(np.max(gaps)),
        "by_seed": seed_rows,
    }
    writer.write_csv(
        "exact_slots.csv",
        rows,
        fieldnames=(
            "seed",
            "slot",
            "arrivals",
            "full_completed",
            "exact_completed",
            "completion_gap",
            "full_runtime_s",
            "exact_runtime_s",
            "evaluated_schedules",
            "complete_search",
        ),
    )
    writer.write_json("summary.json", summary)
    config_path = Path(config_path)
    writer.finalize(
        {
            "run_id": definition.run_id,
            "git_sha": _git_sha(),
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "resolved_definition": definition_to_dict(definition),
            "search_space": (
                "all unscheduled or UAV/subcarrier assignments "
                "with block capacity two"
            ),
            "optimality_metric": "completed task count per slot",
            "solver": {
                "name": "scipy.optimize.milp/HiGHS",
                "package_version": scipy.__version__,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete small-instance benchmark",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_exact_definition(
        load_definition(args.config),
        args.config,
        args.output,
    )


if __name__ == "__main__":
    main()
