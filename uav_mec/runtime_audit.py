"""Frozen runtime-cost audit for the formal K3N45 and K5N75 scales."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


RUNTIME_LIMITS_S = {"K3N45": 1.0, "K5N75": 10.0}
AUDIT_SAMPLE_COUNT = 5 * 20


def _nonnegative_finite(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or any(
        not math.isfinite(float(value)) or value < 0.0 for value in array
    ):
        raise ValueError(f"{name} must contain nonnegative finite values")
    return array


def summarize_runtime_audit(
    *,
    scale: str,
    runtimes_s: Sequence[float],
    candidate_counts: Sequence[int],
    peak_memory_bytes: int,
    log_bytes: int,
    formal_slot_evaluations: int,
) -> dict[str, object]:
    """Summarize the preregistered 5-seed by 20-slot audit design."""
    if scale not in RUNTIME_LIMITS_S:
        raise ValueError(f"unregistered runtime-audit scale: {scale!r}")
    runtime = _nonnegative_finite(runtimes_s, "runtimes_s")
    candidates = _nonnegative_finite(candidate_counts, "candidate_counts")
    if len(runtime) != AUDIT_SAMPLE_COUNT or len(candidates) != len(runtime):
        raise ValueError(
            "runtime audit requires exactly 100 paired measurements "
            "(5 seeds x 20 slots)"
        )
    for value, name in (
        (peak_memory_bytes, "peak_memory_bytes"),
        (log_bytes, "log_bytes"),
        (formal_slot_evaluations, "formal_slot_evaluations"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    p95 = float(np.percentile(runtime, 95))
    limit = RUNTIME_LIMITS_S[scale]
    return {
        "scale": scale,
        "design": {"seeds": 5, "slots_per_seed": 20},
        "sample_count": len(runtime),
        "p50_runtime_s": float(np.percentile(runtime, 50)),
        "p95_runtime_s": p95,
        "mean_runtime_s": float(np.mean(runtime)),
        "candidate_count_p50": float(np.percentile(candidates, 50)),
        "candidate_count_p95": float(np.percentile(candidates, 95)),
        "candidate_count_total": int(np.sum(candidates)),
        "peak_memory_bytes": peak_memory_bytes,
        "log_bytes": log_bytes,
        "formal_slot_evaluations": formal_slot_evaluations,
        "predicted_total_cpu_hours": (
            float(np.mean(runtime)) * formal_slot_evaluations / 3600.0
        ),
        "runtime_limit_s": limit,
        "runtime_gate_pass": p95 < limit,
    }
