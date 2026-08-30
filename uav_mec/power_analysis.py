"""Pilot-only paired power simulation for the joint Holm decision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True, slots=True)
class PowerDecision:
    selected_n: int | None
    evaluated_n: int
    power_by_hypothesis: dict[str, float]
    simulations: int
    target_power: float


def holm_rejections(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, bool]:
    """Return Holm step-down rejections with deterministic ID tie-breaking."""
    if not p_values:
        raise ValueError("p_values must be non-empty")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    ordered = []
    for hypothesis_id, p_value in p_values.items():
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            raise ValueError("hypothesis IDs must be non-empty strings")
        value = float(p_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be finite and in [0, 1]")
        ordered.append((value, hypothesis_id))
    ordered.sort()

    rejected = {hypothesis_id: False for _value, hypothesis_id in ordered}
    count = len(ordered)
    for index, (p_value, hypothesis_id) in enumerate(ordered):
        if p_value > alpha / (count - index):
            break
        rejected[hypothesis_id] = True
    return rejected


def _validate_pilot(
    pilot_differences: Mapping[str, Sequence[float]],
    mde_by_hypothesis: Mapping[str, float],
) -> dict[str, np.ndarray]:
    if set(pilot_differences) != set(mde_by_hypothesis) or not pilot_differences:
        raise ValueError("pilot differences and MDEs must have identical non-empty IDs")
    validated = {}
    for hypothesis_id in sorted(pilot_differences):
        values = np.asarray(pilot_differences[hypothesis_id], dtype=float)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"pilot {hypothesis_id} must contain finite paired differences")
        mde = float(mde_by_hypothesis[hypothesis_id])
        if not math.isfinite(mde) or mde <= 0.0:
            raise ValueError(f"MDE for {hypothesis_id} must be positive")
        validated[hypothesis_id] = values - float(np.mean(values))
    return validated


def select_sample_size(
    pilot_differences: Mapping[str, Sequence[float]],
    mde_by_hypothesis: Mapping[str, float],
    *,
    candidates: Sequence[int] = (50, 100, 200),
    simulations: int = 2000,
    target_power: float = 0.8,
    alpha: float = 0.05,
    seed: int = 0,
) -> PowerDecision:
    """Select the first candidate giving every H1-H4 Holm-adjusted target power."""
    residuals = _validate_pilot(pilot_differences, mde_by_hypothesis)
    candidate_sizes = tuple(candidates)
    if (
        not candidate_sizes
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in candidate_sizes)
        or tuple(sorted(set(candidate_sizes))) != candidate_sizes
    ):
        raise ValueError("candidates must be unique increasing positive integers")
    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations <= 0:
        raise ValueError("simulations must be a positive integer")
    if not math.isfinite(target_power) or not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be between zero and one")
    rng = np.random.default_rng(seed)
    last_power: dict[str, float] = {}

    for sample_size in candidate_sizes:
        rejection_counts = {hypothesis_id: 0 for hypothesis_id in residuals}
        for _ in range(simulations):
            p_values = {}
            for hypothesis_id, centered in residuals.items():
                sampled = rng.choice(centered, size=sample_size, replace=True)
                shifted = sampled + float(mde_by_hypothesis[hypothesis_id])
                p_values[hypothesis_id] = float(
                    wilcoxon(
                        shifted,
                        alternative="greater",
                        zero_method="wilcox",
                    ).pvalue
                )
            for hypothesis_id, rejected in holm_rejections(
                p_values,
                alpha=alpha,
            ).items():
                rejection_counts[hypothesis_id] += int(rejected)
        last_power = {
            hypothesis_id: count / simulations
            for hypothesis_id, count in rejection_counts.items()
        }
        if all(power >= target_power for power in last_power.values()):
            return PowerDecision(
                sample_size,
                sample_size,
                last_power,
                simulations,
                target_power,
            )
    return PowerDecision(
        None,
        candidate_sizes[-1],
        last_power,
        simulations,
        target_power,
    )
