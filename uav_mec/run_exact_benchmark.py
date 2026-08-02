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
    MethodSpec,
    _git_sha,
    _slot_method,
    definition_to_dict,
    load_definition,
    method_spec_sha256,
    method_spec_to_dict,
    thesis_method_specs,
)
from .model import NOMAChannel
from .scheduler import network_objective
from .seed_access import (
    SeedAccessRecorder,
    development_permit,
    pilot_permit,
)
from .trace import generate_trace


def _validated_exact_method_specs(
    definition: ExperimentDefinition,
) -> tuple[MethodSpec, MethodSpec]:
    specs = thesis_method_specs()
    if definition.methods != ("full", "exact"):
        raise ValueError("exact benchmark methods must be ('full', 'exact')")
    full_spec = specs[definition.methods[0]]
    exact_spec = specs[definition.methods[1]]
    if full_spec.wireless_kernel != "iea_cap":
        raise ValueError("full exact-benchmark method must use iea_cap")
    if exact_spec.wireless_kernel != "exact_schedule":
        raise ValueError("exact method must use the exact_schedule kernel")
    if exact_spec.traversal_rule != "exhaustive":
        raise ValueError("exact method must use exhaustive traversal")
    if {
        full_spec.allocation_policy,
        exact_spec.allocation_policy,
    } != {definition.evaluation.allocation_policy}:
        raise ValueError(
            "exact benchmark evaluation allocation_policy must match MethodSpec"
        )
    return full_spec, exact_spec


def run_exact_definition(
    definition: ExperimentDefinition,
    config_path: Path,
    output_dir: Path,
    *,
    seed_access: SeedAccessRecorder,
) -> None:
    full_spec, exact_spec = _validated_exact_method_specs(definition)
    if definition.scenario.num_users > 5:
        raise ValueError(
            "complete benchmark is frozen at no more than five users"
        )

    if not isinstance(seed_access, SeedAccessRecorder):
        raise TypeError("seed_access must be a SeedAccessRecorder")
    seed_access.record(
        "run_exact_definition",
        definition.seeds,
        {
            "run_id": definition.run_id,
            "config_path": str(Path(config_path)),
        },
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
            seed_access=seed_access,
        )
        full_completed = 0
        exact_completed = 0
        arrivals = 0
        for snapshot in trace.slots:
            events: list[dict[str, object]] = []
            started = time.perf_counter()
            full = _slot_method(
                full_spec.method_id,
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
            if not exact.certificate.complete:
                raise RuntimeError(
                    "incomplete exact certificate cannot be reported"
                )
            full_objective = network_objective(full)
            if full_objective > exact.objective:
                raise AssertionError(
                    "heuristic objective exceeds complete enumeration"
                )

            slot_arrivals = len(full.arrived_task_ids)
            exact_slot_completed = exact.objective[0]
            arrivals += slot_arrivals
            full_completed += full.completed_count
            exact_completed += exact_slot_completed
            rows.append(
                {
                    "seed": seed,
                    "slot": snapshot.slot_index,
                    "result_schema": "exact_schedule",
                    "certificate_label": (
                        "Exact"
                        if exact.certificate.complete
                        else "Incomplete"
                    ),
                    "arrivals": slot_arrivals,
                    "full_completed": full.completed_count,
                    "exact_completed": exact_slot_completed,
                    "completion_gap": (
                        exact_slot_completed - full.completed_count
                    ),
                    "full_runtime_s": full_runtime_s,
                    "exact_runtime_s": exact_runtime_s,
                    "evaluated_schedules": (
                        exact.certificate.raw_states_enumerated
                    ),
                    "complete_search": exact.certificate.complete,
                    "raw_states_enumerated": (
                        exact.certificate.raw_states_enumerated
                    ),
                    "unique_closed_states": (
                        exact.certificate.unique_closed_states
                    ),
                    "gap_status": exact.certificate.gap_status,
                    "best_bound_completed": exact.certificate.best_bound[0],
                    "best_bound_cpu": exact.certificate.best_bound[1],
                    "best_bound_interference": (
                        exact.certificate.best_bound[2]
                    ),
                    "completion_bound_gap": (
                        exact.certificate.lexicographic_gap[0]
                    ),
                    "cpu_bound_gap": (
                        exact.certificate.lexicographic_gap[1]
                    ),
                    "interference_bound_gap": (
                        exact.certificate.lexicographic_gap[2]
                    ),
                    "cpu_nontrivial": exact.certificate.cpu_nontrivial,
                    "schedule_nontrivial": (
                        exact.certificate.schedule_nontrivial
                    ),
                    "semantic_fingerprint": (
                        exact.certificate.semantic_fingerprint
                    ),
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
        "result_schema": "exact_schedule",
        "certificate_label": (
            "Exact"
            if all(bool(row["complete_search"]) for row in rows)
            else "Incomplete"
        ),
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
        "methods": {
            spec.method_id: {
                "method_spec": method_spec_to_dict(spec),
                "method_spec_sha256": method_spec_sha256(spec),
            }
            for spec in (full_spec, exact_spec)
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
            "result_schema",
            "certificate_label",
            "arrivals",
            "full_completed",
            "exact_completed",
            "completion_gap",
            "full_runtime_s",
            "exact_runtime_s",
            "evaluated_schedules",
            "complete_search",
            "raw_states_enumerated",
            "unique_closed_states",
            "gap_status",
            "best_bound_completed",
            "best_bound_cpu",
            "best_bound_interference",
            "completion_bound_gap",
            "cpu_bound_gap",
            "interference_bound_gap",
            "cpu_nontrivial",
            "schedule_nontrivial",
            "semantic_fingerprint",
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
            "method_specs": {
                spec.method_id: {
                    "factor_hash": method_spec_sha256(spec),
                    "spec": method_spec_to_dict(spec),
                }
                for spec in (full_spec, exact_spec)
            },
            "search_space": (
                "all unscheduled or UAV/subcarrier assignments "
                "with block capacity two, normalized and admission-closed"
            ),
            "result_schema": "exact_schedule",
            "certificate_label": summary["certificate_label"],
            "optimality_metric": "completed task count per slot",
            "solver": {
                "name": "closure-complete enumeration",
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
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("development", "pilot"),
        required=True,
        help="formal execution is available only through the guarded v2 runner",
    )
    args = parser.parse_args()
    permit = (
        development_permit(args.registry)
        if args.purpose == "development"
        else pilot_permit(args.registry)
    )
    run_exact_definition(
        load_definition(args.config),
        args.config,
        args.output,
        seed_access=SeedAccessRecorder(args.ledger, permit),
    )


if __name__ == "__main__":
    main()
