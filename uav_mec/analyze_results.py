"""Validate canonical evidence bundles and compute thesis statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import RunArtifactWriter, sha256_file
from .statistics import holm_adjust, paired_comparison


@dataclass(frozen=True, slots=True)
class Bundle:
    name: str
    path: Path | None
    manifest: dict
    summary: dict
    tasks: list[dict[str, str]]
    slots: list[dict[str, str]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_checksums(path: Path) -> None:
    root = Path(path)
    checksum_path = root / "checksums.sha256"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    listed: set[Path] = set()
    for line in lines:
        try:
            expected, relative_text = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line: {line!r}") from exc
        relative = Path(relative_text)
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or relative.is_absolute()
            or relative == Path(".")
            or ".." in relative.parts
            or relative.name == "checksums.sha256"
        ):
            raise ValueError(f"invalid checksum entry: {line!r}")
        if relative in listed:
            raise ValueError(f"duplicate checksum entry: {relative}")
        listed.add(relative)
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"checksummed file is missing or unsafe: {target}")
        if sha256_file(target) != expected:
            raise ValueError(f"checksum mismatch: {target}")

    actual = {
        file.relative_to(root)
        for file in root.rglob("*")
        if file.is_file() and file.name != "checksums.sha256"
    }
    if actual != listed:
        unchecksummed = sorted(str(item) for item in actual - listed)
        missing = sorted(str(item) for item in listed - actual)
        raise ValueError(
            f"unchecksummed or missing files: "
            f"unchecksummed={unchecksummed}, missing={missing}"
        )


def load_bundle(name: str, path: Path) -> Bundle:
    verify_checksums(path)
    return Bundle(
        name=name,
        path=Path(path),
        manifest=json.loads(
            (path / "manifest.json").read_text(encoding="utf-8")
        ),
        summary=json.loads(
            (path / "summary.json").read_text(encoding="utf-8")
        ),
        tasks=_read_csv(path / "tasks.csv"),
        slots=_read_csv(path / "slots.csv"),
    )


def _parse_bool(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean in {field}: {value!r}")


def _is_true(value: str) -> bool:
    return _parse_bool(value, "CSV value")


def _resolved_cpu_budgets(resolved: dict, num_uavs: int) -> tuple[float, ...]:
    value = resolved["evaluation"]["cpu_per_uav_hz"]
    if isinstance(value, list):
        if len(value) != num_uavs:
            raise ValueError("CPU budget vector length does not match num_uavs")
        budgets = tuple(float(item) for item in value)
    else:
        budget = float(value)
        budgets = (budget,) * num_uavs
    if any(not math.isfinite(budget) or budget <= 0.0 for budget in budgets):
        raise ValueError("CPU budgets must be finite and positive")
    return budgets


def validate_bundle(bundle: Bundle) -> dict[str, object]:
    summary_methods = bundle.summary.get("methods")
    if not isinstance(summary_methods, dict) or not summary_methods:
        raise ValueError("summary methods must be a non-empty mapping")
    methods = tuple(summary_methods)
    seeds = tuple(int(seed) for seed in bundle.summary.get("seeds", ()))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("summary seeds must be non-empty and unique")

    initializer_methods = {
        "ilp",
        "exchange_stable",
        "intra_bilateral",
        "cross_bilateral",
        "intra_global",
        "full",
        "full_fixed_half",
        "full_crossing",
    }
    statuses = bundle.summary.get("initializer_status_counts", {})
    if initializer_methods.intersection(methods) and not statuses:
        raise ValueError("missing ILP initializer status evidence")
    failed_statuses = {
        name: count
        for name, count in statuses.items()
        if name != "optimal" and count
    }
    if failed_statuses:
        raise ValueError(
            f"non-optimal ILP initializer status: {failed_statuses}"
        )

    search_methods = {
        "exchange_stable",
        "intra_bilateral",
        "cross_bilateral",
        "intra_global",
        "full",
        "full_fixed_half",
        "full_crossing",
    }
    terminations = bundle.summary.get("search_termination_counts", {})
    if search_methods.intersection(methods) and not terminations:
        raise ValueError("missing search termination evidence")
    accepted_terminations = {
        "local_optimum",
        "cycle_free_fixed_point",
    }
    failed_terminations = {
        name: count
        for name, count in terminations.items()
        if name not in accepted_terminations and count
    }
    if failed_terminations:
        raise ValueError(
            f"non-converged search termination: {failed_terminations}"
        )

    method_set = set(methods)
    seed_set = set(seeds)
    seen_slots: set[tuple[str, int, int]] = set()
    seed_slots: dict[str, set[tuple[int, int]]] = defaultdict(set)
    slot_counts: dict[tuple[str, int, int], tuple[int, int]] = {}
    slot_sic_violations = 0
    for row in bundle.slots:
        method = row["method"]
        seed = int(row["seed"])
        slot = int(row["slot"])
        if method not in method_set or seed not in seed_set:
            raise ValueError("slot row is outside declared methods or seeds")
        key = (method, seed, slot)
        if key in seen_slots:
            raise ValueError(f"duplicate slot outcome {key}")
        seen_slots.add(key)
        seed_slots[method].add((seed, slot))
        arrivals = int(row["arrivals"])
        completed = int(row["completed"])
        if arrivals < 0 or not 0 <= completed <= arrivals:
            raise ValueError(f"invalid slot task counts {key}")
        slot_counts[key] = (arrivals, completed)
        violations = int(row["sic_violations"])
        if violations < 0:
            raise ValueError(f"negative SIC violation count {key}")
        slot_sic_violations += violations
        runtime_s = float(row["runtime_s"])
        if not math.isfinite(runtime_s) or runtime_s < 0.0:
            raise ValueError(f"invalid runtime {key}")
    if set(seed_slots) != method_set:
        raise ValueError("missing slot rows for one or more methods")
    if len({frozenset(value) for value in seed_slots.values()}) != 1:
        raise ValueError("methods do not share identical seed/slot traces")
    covered_seeds = {
        seed for slots in seed_slots.values() for seed, _slot in slots
    }
    if covered_seeds != seed_set:
        raise ValueError("slot rows do not cover declared seeds")

    seen_tasks: set[tuple[str, int, int]] = set()
    task_sets: dict[tuple[str, int], set[int]] = defaultdict(set)
    immutable: dict[tuple[int, int], tuple[str, ...]] = {}
    identity_fields = (
        "user_id",
        "arrival_slot",
        "data_bits",
        "comp_cycles",
        "deadline_s",
        "local_cpu_hz",
    )
    task_sic_violations = 0
    cpu_by_pool: dict[tuple[str, int, int, int], float] = defaultdict(float)
    task_slot_counts: dict[tuple[str, int, int], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    resolved = bundle.manifest.get("resolved_definition")
    if not isinstance(resolved, dict):
        raise ValueError("manifest lacks resolved_definition")
    scenario = resolved["scenario"]
    num_uavs = int(scenario["num_uavs"])
    cpu_budgets = _resolved_cpu_budgets(resolved, num_uavs)

    for row in bundle.tasks:
        method = row["method"]
        seed = int(row["seed"])
        slot = int(row["slot"])
        task_id = int(row["task_id"])
        if method not in method_set or seed not in seed_set:
            raise ValueError("task row is outside declared methods or seeds")
        if (method, seed, slot) not in seen_slots:
            raise ValueError("task row refers to a missing slot row")
        key = (method, seed, task_id)
        if key in seen_tasks:
            raise ValueError(f"duplicate task outcome {key}")
        seen_tasks.add(key)
        task_sets[(method, seed)].add(task_id)

        identity = tuple(row[field] for field in identity_fields)
        trace_key = (seed, task_id)
        if trace_key in immutable and immutable[trace_key] != identity:
            raise ValueError(f"task input changed across methods: {trace_key}")
        immutable[trace_key] = identity

        completed = _parse_bool(row["completed"], "completed")
        task_slot_counts[(method, seed, slot)][0] += 1
        task_slot_counts[(method, seed, slot)][1] += int(completed)
        if completed:
            if not row.get("delay_s", ""):
                raise ValueError(f"completed task lacks delay: {key}")
            delay_s = float(row["delay_s"])
            deadline_s = float(row["deadline_s"])
            if (
                not math.isfinite(delay_s)
                or delay_s < 0.0
                or delay_s > deadline_s + 1e-12
            ):
                raise ValueError(f"completed task violates deadline: {key}")

        sic_text = row.get("sic_feasible", "")
        if sic_text and not _parse_bool(sic_text, "sic_feasible"):
            task_sic_violations += 1
        cpu_hz = float(row.get("cpu_hz", "0") or 0.0)
        if not math.isfinite(cpu_hz) or cpu_hz < 0.0:
            raise ValueError(f"invalid CPU allocation: {key}")
        serving_text = row.get("serving_uav", "")
        if serving_text:
            serving_uav = int(serving_text)
            if not 0 <= serving_uav < num_uavs:
                raise ValueError(f"invalid serving UAV: {key}")
            cpu_by_pool[(method, seed, slot, serving_uav)] += cpu_hz
        elif cpu_hz > 0.0:
            raise ValueError(f"positive CPU lacks serving UAV: {key}")

    for seed in seeds:
        method_task_sets = [task_sets[(method, seed)] for method in methods]
        if len({frozenset(value) for value in method_task_sets}) != 1:
            raise ValueError(
                f"methods do not share identical task IDs for seed {seed}"
            )
    for key, expected in slot_counts.items():
        actual = tuple(task_slot_counts[key])
        if actual != expected:
            raise ValueError(
                f"slot/task count mismatch for {key}: {actual} != {expected}"
            )

    if task_sic_violations or slot_sic_violations:
        raise ValueError(
            "found SIC violations: "
            f"tasks={task_sic_violations}, slots={slot_sic_violations}"
        )
    exceeded = {
        key: value
        for key, value in cpu_by_pool.items()
        if value > cpu_budgets[key[3]] + 1e-6
    }
    if exceeded:
        raise ValueError(f"CPU budget exceeded: {exceeded}")

    for method in methods:
        expected = summary_methods[method]["on_time_rate_by_seed"]
        if len(expected) != len(seeds):
            raise ValueError(f"summary seed count mismatch for {method}")
        actual = []
        for seed in seeds:
            rows = [
                row
                for row in bundle.tasks
                if row["method"] == method and int(row["seed"]) == seed
            ]
            actual.append(
                sum(_is_true(row["completed"]) for row in rows) / len(rows)
                if rows
                else 0.0
            )
        if not np.allclose(actual, expected, atol=1e-12, rtol=0.0):
            raise ValueError(f"summary/task mismatch for {method}")

    return {
        "methods": list(methods),
        "seeds": list(seeds),
        "unique_tasks": len(immutable),
        "sic_violations": 0,
        "cpu_budget_violations": 0,
        "initializer_failures": 0,
        "search_nonconvergence": 0,
    }


COMPARISONS = (
    ("innovation1_cross", "pairing", "full", "intra_global"),
    ("innovation1_acceptance", "pairing", "full", "cross_bilateral"),
    ("innovation2_deadline", "resource", "full", "full_crossing"),
    ("resource_fixed_half", "resource", "full", "full_fixed_half"),
    ("baseline_greedy", "main", "full", "greedy"),
    ("baseline_ia_greedy", "main", "full", "ia_greedy"),
    ("baseline_ilp", "main", "full", "ilp"),
    ("baseline_exchange", "main", "full", "exchange_stable"),
    ("baseline_ofdma", "main", "full", "fair_ofdma"),
)


def comparison_report(
    summaries: Mapping[str, dict],
) -> dict[str, dict[str, object]]:
    raw: dict[str, dict[str, object]] = {}
    for name, bundle_name, proposed, baseline in COMPARISONS:
        if bundle_name not in summaries:
            continue
        methods = summaries[bundle_name]["methods"]
        if proposed not in methods or baseline not in methods:
            continue
        left = (
            np.asarray(
                methods[proposed]["on_time_rate_by_seed"],
                dtype=float,
            )
            * 100.0
        )
        right = (
            np.asarray(
                methods[baseline]["on_time_rate_by_seed"],
                dtype=float,
            )
            * 100.0
        )
        result = paired_comparison(left, right)
        raw[name] = {
            "proposed": proposed,
            "baseline": baseline,
            "mean_difference_pp": result.mean_difference,
            "ci_low_pp": result.ci_low,
            "ci_high_pp": result.ci_high,
            "wilcoxon_w": result.wilcoxon_w,
            "p_raw": result.p_raw,
            "rank_biserial": result.rank_biserial,
        }
    adjusted = holm_adjust(
        {name: float(value["p_raw"]) for name, value in raw.items()}
    )
    for name, value in raw.items():
        value["p_holm"] = adjusted[name]
    return raw


def _jain(values: list[float]) -> float:
    vector = np.asarray(values, dtype=float)
    denominator = len(vector) * float(np.square(vector).sum())
    if denominator <= 0.0:
        return 0.0
    return float(vector.sum() ** 2 / denominator)


def bundle_metrics(bundle: Bundle) -> dict[str, dict[str, float]]:
    methods = tuple(bundle.summary["methods"])
    seeds = tuple(int(seed) for seed in bundle.summary["seeds"])
    resolved = bundle.manifest["resolved_definition"]
    scenario = resolved["scenario"]
    num_uavs = int(scenario["num_uavs"])
    cpu_budgets = _resolved_cpu_budgets(resolved, num_uavs)
    metrics: dict[str, dict[str, float]] = {}
    for method in methods:
        seed_metrics = []
        method_slots = [
            row for row in bundle.slots if row["method"] == method
        ]
        for seed in seeds:
            rows = [
                row
                for row in bundle.tasks
                if row["method"] == method and int(row["seed"]) == seed
            ]
            completed = [_is_true(row["completed"]) for row in rows]
            delays = [
                float(row["delay_s"])
                for row in rows
                if _is_true(row["completed"]) and row["delay_s"]
            ]
            by_user: dict[int, list[bool]] = defaultdict(list)
            for row, done in zip(rows, completed, strict=True):
                by_user[int(row["user_id"])].append(done)
            user_rates = [
                sum(values) / len(values) for values in by_user.values()
            ]
            used_cpu = sum(
                float(row["cpu_hz"])
                for row in rows
                if row["cpu_hz"]
            )
            denominator = (
                int(scenario["num_slots"])
                * math.fsum(cpu_budgets)
            )
            on_time_rate = (
                sum(completed) / len(completed) if completed else 0.0
            )
            seed_metrics.append(
                {
                    "on_time_rate": on_time_rate,
                    "drop_rate": 1.0 - on_time_rate,
                    "mean_completed_delay_s": (
                        float(np.mean(delays)) if delays else 0.0
                    ),
                    "jain_user_on_time": _jain(user_rates),
                    "cpu_utilization": used_cpu / denominator,
                }
            )
        runtimes = np.asarray(
            [float(row["runtime_s"]) for row in method_slots],
            dtype=float,
        )
        metrics[method] = {
            key: float(np.mean([row[key] for row in seed_metrics]))
            for key in seed_metrics[0]
        }
        metrics[method]["runtime_p50_s"] = float(
            np.quantile(runtimes, 0.50)
        )
        metrics[method]["runtime_p95_s"] = float(
            np.quantile(runtimes, 0.95)
        )
    return metrics


def analyze(
    main_path: Path,
    pairing_path: Path,
    resource_path: Path,
    exact_path: Path,
    output_dir: Path,
    require_thesis_gates: bool = False,
) -> dict[str, object]:
    bundles = {
        name: load_bundle(name, path)
        for name, path in (
            ("main", main_path),
            ("pairing", pairing_path),
            ("resource", resource_path),
        )
    }
    validation = {
        name: validate_bundle(bundle) for name, bundle in bundles.items()
    }
    verify_checksums(exact_path)
    exact_rows = _read_csv(exact_path / "exact_slots.csv")
    if not exact_rows or not all(
        _is_true(row["complete_search"]) for row in exact_rows
    ):
        raise ValueError("exact benchmark contains an incomplete search")
    if any(
        int(row["full_completed"]) > int(row["exact_completed"])
        for row in exact_rows
    ):
        raise ValueError("exact benchmark is below the heuristic")
    exact_summary = json.loads(
        (exact_path / "summary.json").read_text(encoding="utf-8")
    )
    comparisons = comparison_report(
        {name: bundle.summary for name, bundle in bundles.items()}
    )
    metrics = {
        name: bundle_metrics(bundle) for name, bundle in bundles.items()
    }
    gates = {
        "constraint_violations_zero": all(
            row["sic_violations"] == 0
            and row["cpu_budget_violations"] == 0
            and row["initializer_failures"] == 0
            and row["search_nonconvergence"] == 0
            for row in validation.values()
        ),
        "innovation1_cross_ci_positive": (
            float(comparisons["innovation1_cross"]["ci_low_pp"]) > 0.0
        ),
        "innovation1_acceptance_ci_positive": (
            float(comparisons["innovation1_acceptance"]["ci_low_pp"]) > 0.0
        ),
        "innovation2_deadline_ci_positive": (
            float(comparisons["innovation2_deadline"]["ci_low_pp"]) > 0.0
        ),
        "full_beats_predeclared_strong_baseline": (
            float(comparisons["baseline_exchange"]["ci_low_pp"]) > 0.0
        ),
        "n20_runtime_p95_below_one_second": (
            metrics["main"]["full"]["runtime_p95_s"] < 1.0
        ),
        "exact_search_complete": all(
            _is_true(row["complete_search"]) for row in exact_rows
        ),
    }
    report = {
        "validation": validation,
        "comparisons": comparisons,
        "metrics": metrics,
        "exact": exact_summary,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    writer = RunArtifactWriter(output_dir)
    writer.write_json("statistics.json", comparisons)
    writer.write_json("metrics.json", metrics)
    writer.write_json(
        "acceptance.json",
        {
            "validation": validation,
            "exact": exact_summary,
            "gates": gates,
            "all_gates_pass": report["all_gates_pass"],
        },
    )
    writer.finalize(
        {
            "inputs": {
                "main": str(main_path),
                "pairing": str(pairing_path),
                "resource": str(resource_path),
                "exact": str(exact_path),
            }
        }
    )
    if require_thesis_gates and not report["all_gates_pass"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"thesis evidence gates failed: {failed}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate thesis evidence and compute paired statistics",
    )
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-thesis-gates", action="store_true")
    args = parser.parse_args()
    analyze(
        args.main,
        args.pairing,
        args.resource,
        args.exact,
        args.output,
        args.require_thesis_gates,
    )


if __name__ == "__main__":
    main()
