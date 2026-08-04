"""Static evidence gate for implementation-correctness claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any

_LOWER_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class DescriptiveRuntimeAudit:
    """Hardware-bound measurements without a universal pass/fail threshold."""

    reference_hardware: str
    python_environment_sha256: str
    p50_runtime_s: float
    p95_runtime_s: float
    mean_runtime_s: float
    candidate_count_p50: float
    candidate_count_p95: float
    candidate_count_total: int
    cache_hit_rate: float
    peak_memory_bytes: int
    log_bytes: int
    measured_cpu_seconds: float

    def __post_init__(self) -> None:
        if not self.reference_hardware.strip():
            raise ValueError("reference_hardware must be non-empty")
        if not _is_sha256_digest(self.python_environment_sha256):
            raise ValueError("python_environment_sha256 must be a SHA-256 digest")
        for field_name in (
            "p50_runtime_s",
            "p95_runtime_s",
            "mean_runtime_s",
            "candidate_count_p50",
            "candidate_count_p95",
            "cache_hit_rate",
            "measured_cpu_seconds",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{field_name} must be finite and nonnegative")
        if self.p95_runtime_s < self.p50_runtime_s:
            raise ValueError("p95_runtime_s must be at least p50_runtime_s")
        if self.candidate_count_p95 < self.candidate_count_p50:
            raise ValueError("candidate_count_p95 must be at least candidate_count_p50")
        if self.cache_hit_rate > 1:
            raise ValueError("cache_hit_rate must be at most one")
        for field_name in (
            "candidate_count_total",
            "peak_memory_bytes",
            "log_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")


def descriptive_runtime_audit_to_dict(
    audit: DescriptiveRuntimeAudit,
) -> dict[str, Any]:
    """Return the stable JSON field order used by calibration and budgeting."""

    return asdict(audit)


@dataclass(frozen=True)
class AlgorithmAuditRecord:
    search_stop_reasons: tuple[str, ...]
    local_optimum_certified: tuple[bool, ...]
    incremental_actions_checked: int
    externality_conserved: bool
    exact_schedule_complete: bool
    exact_admission_type: str
    exact_schedule_type: str
    method_spec_hashes: tuple[str, ...]
    adapted_baseline_mapping_hash: str


def _is_sha256_digest(value: str) -> bool:
    return _LOWER_HEX_SHA256.fullmatch(value) is not None


def validate_algorithm_claim_gate(record: AlgorithmAuditRecord) -> tuple[str, ...]:
    """Return stable error codes for incomplete static algorithm evidence."""

    errors: list[str] = []
    if "iteration_cap" in record.search_stop_reasons:
        errors.append("iteration_cap")

    pairs_match = len(record.search_stop_reasons) == len(record.local_optimum_certified) and all(
        certified == (stop_reason == "no_improving_move")
        for stop_reason, certified in zip(
            record.search_stop_reasons,
            record.local_optimum_certified,
            strict=True,
        )
    )
    if not pairs_match:
        errors.append("stop_reason_certification_mismatch")
    if record.incremental_actions_checked < 10_000:
        errors.append("incremental_actions_checked")
    if not record.externality_conserved:
        errors.append("externality_conserved")
    if not record.exact_schedule_complete:
        errors.append("exact_schedule_complete")
    if record.exact_admission_type == record.exact_schedule_type:
        errors.append("exact_result_types_distinct")
    if not record.method_spec_hashes or any(
        not _is_sha256_digest(digest) for digest in record.method_spec_hashes
    ):
        errors.append("method_spec_hashes")
    if not _is_sha256_digest(record.adapted_baseline_mapping_hash):
        errors.append("adapted_baseline_mapping_hash")
    return tuple(errors)
