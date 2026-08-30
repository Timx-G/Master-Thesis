"""Load and runtime evidence for reality-calibrated capacity experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .analyze_results import load_bundle, validate_bundle
from .artifacts import RunArtifactWriter, sha256_file
from .experiment import ExperimentDefinition


def _completed(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid completed value {value!r}")


def capacity_case_metrics(
    definition: ExperimentDefinition,
    bundle_path: Path,
) -> dict[str, object]:
    """Validate and summarize one capacity-suite bundle."""
    bundle = load_bundle(definition.run_id, Path(bundle_path))
    validate_bundle(bundle)
    reference_method = definition.methods[0]
    reference_tasks = [row for row in bundle.tasks if row["method"] == reference_method]
    reference_slots = [row for row in bundle.slots if row["method"] == reference_method]
    demand_by_slot: dict[tuple[int, int], int] = defaultdict(int)
    for row in reference_slots:
        demand_by_slot[(int(row["seed"]), int(row["slot"]))] = 0

    local_infeasible = 0
    for row in reference_tasks:
        local_delay_s = float(row["comp_cycles"]) / float(row["local_cpu_hz"])
        if local_delay_s <= float(row["deadline_s"]):
            continue
        local_infeasible += 1
        demand_by_slot[(int(row["seed"]), int(row["slot"]))] += 1

    scenario = definition.scenario
    expected_arrival_probability = scenario.task.arrival_prob
    if scenario.task.arrival_mode == "periodic":
        expected_arrival_probability = float(
            np.mean([1.0 / period for period in scenario.task.period_choices])
        )
    elif scenario.random_burst:
        expected_arrival_probability = (
            scenario.cool_rate + (scenario.burst_rate - scenario.cool_rate) / scenario.num_clusters
        )
    positions_per_uav = scenario.n_subcarriers * scenario.users_per_sub
    positions_per_slot = scenario.num_uavs * positions_per_uav
    slot_demands = np.asarray(tuple(demand_by_slot.values()), dtype=float)
    slot_count = len(reference_slots)
    method_metrics: dict[str, dict[str, float | int]] = {}
    for method in definition.methods:
        slots = [row for row in bundle.slots if row["method"] == method]
        tasks = [row for row in bundle.tasks if row["method"] == method]
        runtimes = np.asarray(
            [float(row["runtime_s"]) for row in slots],
            dtype=float,
        )
        method_metrics[method] = {
            "on_time_rate": (
                sum(_completed(row["completed"]) for row in tasks) / len(tasks) if tasks else 0.0
            ),
            "runtime_mean_s": float(np.mean(runtimes)) if len(runtimes) else 0.0,
            "runtime_p50_s": float(np.percentile(runtimes, 50)) if len(runtimes) else 0.0,
            "runtime_p95_s": float(np.percentile(runtimes, 95)) if len(runtimes) else 0.0,
            "sic_violations": sum(int(row["sic_violations"]) for row in slots),
            "mean_cross_interference_w": (
                float(np.mean([float(row["cross_interference_w"]) for row in slots]))
                if slots
                else 0.0
            ),
        }

    return {
        "scenario": {
            "num_uavs": scenario.num_uavs,
            "num_users": scenario.num_users,
            "registered_users_per_uav": scenario.num_users / scenario.num_uavs,
            "num_clusters": scenario.num_clusters,
            "burst_rate": scenario.burst_rate,
            "cool_rate": scenario.cool_rate,
            "n_subcarriers": scenario.n_subcarriers,
            "users_per_subcarrier": scenario.users_per_sub,
            "num_slots": scenario.num_slots,
            "arrival_mode": scenario.task.arrival_mode,
            "period_choices": list(scenario.task.period_choices),
        },
        "load": {
            "noma_positions_per_uav_slot": positions_per_uav,
            "noma_positions_per_slot": positions_per_slot,
            "expected_arrival_probability_per_user": (expected_arrival_probability),
            "realized_arrival_probability_per_user": (
                len(reference_tasks) / (slot_count * scenario.num_users) if slot_count else 0.0
            ),
            "mean_arrivals_per_uav_slot": (
                len(reference_tasks) / (slot_count * scenario.num_uavs) if slot_count else 0.0
            ),
            "local_infeasible_task_fraction": (
                local_infeasible / len(reference_tasks) if reference_tasks else 0.0
            ),
            "mean_radio_offered_load": (
                float(np.mean(slot_demands / positions_per_slot)) if len(slot_demands) else 0.0
            ),
            "overloaded_slot_fraction": (
                float(np.mean(slot_demands > positions_per_slot)) if len(slot_demands) else 0.0
            ),
            "mean_offload_demand_per_slot": (
                float(np.mean(slot_demands)) if len(slot_demands) else 0.0
            ),
        },
        "methods": method_metrics,
    }


def write_capacity_analysis(
    cases: Mapping[str, tuple[ExperimentDefinition, Path]],
    output_dir: Path,
) -> None:
    """Write one checksummed analysis bundle for completed capacity cases."""
    if not cases:
        raise ValueError("capacity analysis requires at least one case")
    report = {
        "schema_version": 1,
        "load_definition": (
            "tasks whose local execution time exceeds their deadline / "
            "(num_uavs * num_subcarriers * users_per_subcarrier)"
        ),
        "cases": {
            name: capacity_case_metrics(definition, path)
            for name, (definition, path) in sorted(cases.items())
        },
    }
    writer = RunArtifactWriter(Path(output_dir))
    writer.write_json("capacity_metrics.json", report)
    writer.finalize(
        {
            "schema_version": 1,
            "source_bundle_checksums": {
                name: sha256_file(Path(path) / "checksums.sha256")
                for name, (_definition, path) in sorted(cases.items())
            },
        }
    )
