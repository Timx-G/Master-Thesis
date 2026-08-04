"""Static evidence gate for implementation-correctness claims."""

from __future__ import annotations

from dataclasses import dataclass
import re


_LOWER_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


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

    pairs_match = len(record.search_stop_reasons) == len(
        record.local_optimum_certified
    ) and all(
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
