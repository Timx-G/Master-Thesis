"""Metrics for the independently frozen interference-externality stress suite."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from .analyze_results import load_bundle, validate_bundle
from .artifacts import RunArtifactWriter, sha256_file
from .domain import SlotSnapshot
from .experiment import ExperimentDefinition
from .statistics import paired_comparison
from .trace import generate_trace

OVERLAP_GAP_DB = 3.0
OVERLAP_RATIO_THRESHOLD = 10.0 ** (-OVERLAP_GAP_DB / 10.0)


def qualify_overlap_fraction(band: str, fraction: float) -> None:
    """Enforce the predeclared low/medium/high overlap bands."""
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("overlap fraction must be finite and in [0, 1]")
    qualified = {
        "low": fraction <= 0.15,
        "medium": 0.30 <= fraction <= 0.45,
        "high": fraction >= 0.60,
    }
    if band not in qualified:
        raise ValueError(f"unknown overlap band {band!r}")
    if not qualified[band]:
        raise ValueError(
            f"overlap fraction {fraction:.6f} is outside the {band} band"
        )


def association_overlap_ratios(snapshot: SlotSnapshot) -> dict[int, float]:
    """Return second-best/best realized UAV gain for every arrived task."""
    ratios: dict[int, float] = {}
    for task, gains in zip(
        snapshot.tasks,
        np.asarray(snapshot.channel_gains),
        strict=True,
    ):
        ordered = np.sort(np.asarray(gains, dtype=float))
        if len(ordered) < 2 or ordered[-1] <= 0.0:
            ratio = 0.0
        else:
            ratio = float(ordered[-2] / ordered[-1])
        ratios[task.task_id] = ratio
    return ratios


def _objective(row: Mapping[str, object], field: str) -> tuple[float, ...] | None:
    value = row.get(field)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        objective = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(objective).all():
        return None
    return objective


def bilateral_externality_metrics(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, int | float]:
    """Summarize bilateral acceptances that the network objective rejects."""
    evaluated = 0
    accepted = 0
    global_rejected = 0
    completion_harming = 0
    tasks_lost = 0
    interference_increasing = 0
    total_interference_increase_w = 0.0
    external_affected_task_event_count = 0
    global_rejections_with_external_rate_effect = 0

    for row in rows:
        if row.get("method") != "cross_bilateral":
            continue
        before = _objective(row, "before")
        after = _objective(row, "after")
        if before is None or after is None:
            continue
        evaluated += 1
        if row.get("accepted") is not True:
            continue
        accepted += 1
        if after > before:
            continue
        global_rejected += 1
        task_loss = max(int(round(before[0] - after[0])), 0)
        if task_loss:
            completion_harming += 1
            tasks_lost += task_loss
        interference_increase_w = max(before[2] - after[2], 0.0)
        if interference_increase_w > 0.0:
            interference_increasing += 1
            total_interference_increase_w += interference_increase_w
        affected = row.get("external_rate_affected_task_ids", ())
        if isinstance(affected, (list, tuple)):
            external_affected_task_event_count += len(affected)
            if affected:
                global_rejections_with_external_rate_effect += 1

    return {
        "evaluated_candidates": evaluated,
        "accepted_candidates": accepted,
        "accepted_global_rejected": global_rejected,
        "accepted_global_rejected_rate": (global_rejected / accepted if accepted else 0.0),
        "global_rejections_with_external_rate_effect": (
            global_rejections_with_external_rate_effect
        ),
        "completion_harming_acceptances": completion_harming,
        "completion_tasks_lost": tasks_lost,
        "interference_increasing_acceptances": interference_increasing,
        "total_interference_increase_w": total_interference_increase_w,
        "external_affected_task_event_count": (external_affected_task_event_count),
    }


def _completed(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid completed value {value!r}")


def _paired_report(
    proposed: list[float],
    baseline: list[float],
) -> dict[str, float]:
    result = paired_comparison(
        np.asarray(proposed, dtype=float) * 100.0,
        np.asarray(baseline, dtype=float) * 100.0,
    )
    return {
        "mean_difference_pp": result.mean_difference,
        "ci_low_pp": result.ci_low,
        "ci_high_pp": result.ci_high,
        "wilcoxon_w": result.wilcoxon_w,
        "p_raw": result.p_raw,
        "rank_biserial": result.rank_biserial,
    }


def _optional_paired_report(
    proposed: list[float | None],
    baseline: list[float | None],
) -> dict[str, object]:
    paired = [
        (left, right)
        for left, right in zip(proposed, baseline, strict=True)
        if left is not None and right is not None
    ]
    if len(paired) < 2:
        return {
            "available": False,
            "paired_seed_count": len(paired),
        }
    report: dict[str, object] = _paired_report(
        [left for left, _right in paired],
        [right for _left, right in paired],
    )
    report.update(
        {
            "available": True,
            "paired_seed_count": len(paired),
        }
    )
    return report


def _event_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in (Path(path) / "events.jsonl").read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("event row must be an object")
        rows.append(value)
    return rows


def _overlap_by_task(
    definition: ExperimentDefinition,
) -> dict[tuple[int, int], float]:
    ratios: dict[tuple[int, int], float] = {}
    for seed in definition.seeds:
        trace = generate_trace(
            definition.scenario,
            seed,
            definition.csi_error_std,
        )
        for snapshot in trace.slots:
            for task_id, ratio in association_overlap_ratios(snapshot).items():
                ratios[(seed, task_id)] = ratio
    return ratios


def externality_case_metrics(
    definition: ExperimentDefinition,
    bundle_path: Path,
) -> dict[str, object]:
    """Validate and summarize one stress-suite case bundle."""
    bundle = load_bundle(definition.run_id, Path(bundle_path))
    validate_bundle(bundle)
    overlap_by_task = _overlap_by_task(definition)
    seeds = tuple(definition.seeds)
    method_metrics: dict[str, dict[str, object]] = {}

    for method in definition.methods:
        overall = list(bundle.summary["methods"][method]["on_time_rate_by_seed"])
        overlap_rates: list[float | None] = []
        overlap_counts: list[int] = []
        for seed in seeds:
            rows = [
                row for row in bundle.tasks if row["method"] == method and int(row["seed"]) == seed
            ]
            overlap_rows = [
                row
                for row in rows
                if overlap_by_task[(seed, int(row["task_id"]))] >= OVERLAP_RATIO_THRESHOLD
            ]
            overlap_counts.append(len(overlap_rows))
            overlap_rates.append(
                sum(_completed(row["completed"]) for row in overlap_rows) / len(overlap_rows)
                if overlap_rows
                else None
            )
        slot_interference = [
            float(row["cross_interference_w"]) for row in bundle.slots if row["method"] == method
        ]
        method_metrics[method] = {
            "overall_on_time_rate_by_seed": overall,
            "overlap_on_time_rate_by_seed": overlap_rates,
            "overlap_task_count_by_seed": overlap_counts,
            "mean_slot_cross_interference_w": (
                float(np.mean(slot_interference)) if slot_interference else 0.0
            ),
        }

    comparisons = {
        f"full_minus_{baseline}": _paired_report(
            list(method_metrics["full"]["overall_on_time_rate_by_seed"]),
            list(method_metrics[baseline]["overall_on_time_rate_by_seed"]),
        )
        for baseline in ("cross_bilateral", "intra_global")
    }
    overlap_comparisons = {
        f"full_minus_{baseline}": _optional_paired_report(
            list(method_metrics["full"]["overlap_on_time_rate_by_seed"]),
            list(method_metrics[baseline]["overlap_on_time_rate_by_seed"]),
        )
        for baseline in ("cross_bilateral", "intra_global")
    }
    reference_method = definition.methods[0]
    reference_rows = [row for row in bundle.tasks if row["method"] == reference_method]
    unique_arrivals = {(int(row["seed"]), int(row["task_id"])) for row in reference_rows}
    overlap_arrivals = sum(
        overlap_by_task[key] >= OVERLAP_RATIO_THRESHOLD for key in unique_arrivals
    )
    denominator = len(seeds) * definition.scenario.num_slots * definition.scenario.num_uavs
    return {
        "scenario": {
            "num_uavs": definition.scenario.num_uavs,
            "num_users": definition.scenario.num_users,
            "num_clusters": definition.scenario.num_clusters,
            "n_subcarriers": definition.scenario.n_subcarriers,
            "cluster_std_m": definition.scenario.cluster_std,
            "cluster_layout": definition.scenario.cluster_layout,
            "burst_rate": definition.scenario.burst_rate,
            "cool_rate": definition.scenario.cool_rate,
            "cross_interference_scale": (definition.evaluation.cross_interference_scale),
        },
        "actual_arrivals_per_uav_slot": (
            len(unique_arrivals) / denominator if denominator else 0.0
        ),
        "overlap_task_fraction": (
            overlap_arrivals / len(unique_arrivals) if unique_arrivals else 0.0
        ),
        "methods": method_metrics,
        "comparisons": comparisons,
        "overlap_comparisons": overlap_comparisons,
        "bilateral_externality": bilateral_externality_metrics(_event_rows(bundle_path)),
    }


def write_externality_analysis(
    cases: Mapping[str, tuple[ExperimentDefinition, Path]],
    output_dir: Path,
) -> None:
    """Write a checksummed analysis bundle for completed stress cases."""
    if not cases:
        raise ValueError("externality analysis requires at least one case")
    report = {
        "schema_version": 1,
        "overlap_region_definition": {
            "criterion": "best_two_large_scale_gain_gap_db <= threshold_db",
            "threshold_db": OVERLAP_GAP_DB,
        },
        "cases": {
            name: externality_case_metrics(definition, path)
            for name, (definition, path) in sorted(cases.items())
        },
    }
    writer = RunArtifactWriter(Path(output_dir))
    writer.write_json("stress_metrics.json", report)
    writer.finalize(
        {
            "schema_version": 1,
            "source_bundle_checksums": {
                name: sha256_file(Path(path) / "checksums.sha256")
                for name, (_definition, path) in sorted(cases.items())
            },
            "overlap_gap_db": OVERLAP_GAP_DB,
        }
    )
