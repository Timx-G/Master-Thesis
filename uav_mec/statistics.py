"""Paired statistical inference for canonical thesis comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class PairedComparison:
    mean_difference: float
    ci_low: float
    ci_high: float
    wilcoxon_w: float
    p_raw: float
    rank_biserial: float


def _finite_vector(
    values: np.ndarray,
    name: str,
    minimum_size: int,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) < minimum_size or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector with at least " f"{minimum_size} values")
    return vector


def paired_bootstrap_ci(
    differences: np.ndarray,
    resamples: int = 10_000,
    seed: int = 20260711,
) -> tuple[float, float]:
    """Return a deterministic 95% percentile CI for paired mean differences."""
    values = _finite_vector(differences, "differences", 2)
    if not isinstance(resamples, int) or isinstance(resamples, bool):
        raise ValueError("resamples must be an integer")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(resamples, len(values)),
    )
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def participant_clustered_bootstrap_ci(
    differences: np.ndarray,
    participant_ids: Sequence[str],
    resamples: int = 10_000,
    seed: int = 20260719,
) -> tuple[float, float]:
    """Return a 95% paired-mean CI resampled at participant level.

    Each sampled participant contributes all of that participant's observations.
    Repeatedly sampled participants contribute repeated copies of their complete
    observation cluster, preserving unequal cluster sizes.
    """

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite one-dimensional vector")
    identifiers = tuple(participant_ids)
    if len(identifiers) != len(values):
        raise ValueError("participant_ids must match differences shape")
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("participant_ids must be non-empty strings")
    unique = tuple(dict.fromkeys(identifiers))
    if len(unique) < 2:
        raise ValueError("at least two source participants are required")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")

    clusters = {
        identifier: values[np.asarray([value == identifier for value in identifiers])]
        for identifier in unique
    }
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(unique), size=(resamples, len(unique)))
    means = np.empty(resamples, dtype=float)
    for index, selected in enumerate(selections):
        sampled = [clusters[unique[participant_index]] for participant_index in selected]
        means[index] = float(np.concatenate(sampled).mean())
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def rank_biserial(differences: np.ndarray) -> float:
    """Return the signed-rank matched-pairs rank-biserial effect size."""
    values = _finite_vector(differences, "differences", 1)
    nonzero = values[values != 0.0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    return (positive - negative) / (positive + negative)


def paired_comparison(
    proposed: np.ndarray,
    baseline: np.ndarray,
    bootstrap_seed: int = 20260711,
) -> PairedComparison:
    """Compare same-seed samples as proposed minus baseline."""
    left = np.asarray(proposed, dtype=float)
    right = np.asarray(baseline, dtype=float)
    if left.shape != right.shape:
        raise ValueError("paired samples must have the same shape")
    left = _finite_vector(left, "paired samples", 2)
    right = _finite_vector(right, "paired samples", 2)
    differences = left - right
    ci_low, ci_high = paired_bootstrap_ci(
        differences,
        10_000,
        bootstrap_seed,
    )

    if np.all(differences == 0.0):
        w_value, p_value = 0.0, 1.0
    else:
        test = stats.wilcoxon(
            left,
            right,
            alternative="two-sided",
            zero_method="wilcox",
        )
        w_value = float(test.statistic)
        p_value = float(test.pvalue)
    return PairedComparison(
        mean_difference=float(differences.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        wilcoxon_w=w_value,
        p_raw=p_value,
        rank_biserial=rank_biserial(differences),
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values using the monotone step-down procedure."""
    validated = []
    for name, raw_value in p_values.items():
        try:
            p_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid p-value for {name}: {raw_value}") from exc
        if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"invalid p-value for {name}: {raw_value}")
        validated.append((name, p_value))

    ordered = sorted(validated, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(
            running,
            min((total - rank) * p_value, 1.0),
        )
        adjusted[name] = running
    return adjusted
