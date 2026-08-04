from dataclasses import FrozenInstanceError, fields

import pytest

from uav_mec.runtime_audit import (
    AlgorithmAuditRecord,
    DescriptiveRuntimeAudit,
    descriptive_runtime_audit_to_dict,
    validate_algorithm_claim_gate,
)

VALID_DIGEST = "a" * 64


def _valid_runtime_audit(**overrides: object) -> DescriptiveRuntimeAudit:
    values = {
        "reference_hardware": "AMD Ryzen 7 7840HS; 5.4 GiB limit",
        "python_environment_sha256": VALID_DIGEST,
        "p50_runtime_s": 0.12,
        "p95_runtime_s": 0.23,
        "mean_runtime_s": 0.14,
        "candidate_count_p50": 18.0,
        "candidate_count_p95": 31.0,
        "candidate_count_total": 240,
        "cache_hit_rate": 0.25,
        "peak_memory_bytes": 12_345_678,
        "log_bytes": 4_321,
        "measured_cpu_seconds": 2.8,
    }
    values.update(overrides)
    return DescriptiveRuntimeAudit(**values)


def test_runtime_audit_is_descriptive_without_fixed_realtime_gate() -> None:
    audit = _valid_runtime_audit(p95_runtime_s=1000.0)
    payload = descriptive_runtime_audit_to_dict(audit)
    assert "runtime_gate_pass" not in payload
    assert "runtime_limit_s" not in payload
    assert not hasattr(audit, "runtime_gate_pass")


def test_runtime_audit_reports_reference_hardware_p50_p95_candidates_cache_memory_and_cpu_time() -> (
    None
):
    payload = descriptive_runtime_audit_to_dict(_valid_runtime_audit())
    assert tuple(payload) == (
        "reference_hardware",
        "python_environment_sha256",
        "p50_runtime_s",
        "p95_runtime_s",
        "mean_runtime_s",
        "candidate_count_p50",
        "candidate_count_p95",
        "candidate_count_total",
        "cache_hit_rate",
        "peak_memory_bytes",
        "log_bytes",
        "measured_cpu_seconds",
    )


def _valid_record(**overrides: object) -> AlgorithmAuditRecord:
    values = {
        "search_stop_reasons": ("no_improving_move",),
        "local_optimum_certified": (True,),
        "incremental_actions_checked": 10_000,
        "externality_conserved": True,
        "exact_schedule_complete": True,
        "exact_admission_type": "ExactAdmissionResult",
        "exact_schedule_type": "ExactScheduleResult",
        "method_spec_hashes": (VALID_DIGEST,),
        "adapted_baseline_mapping_hash": "b" * 64,
    }
    values.update(overrides)
    return AlgorithmAuditRecord(**values)


def test_claim_gate_rejects_caps_incomplete_exact_and_unverified_incremental_counts() -> None:
    invalid = AlgorithmAuditRecord(
        search_stop_reasons=("no_improving_move", "iteration_cap"),
        local_optimum_certified=(True, False),
        incremental_actions_checked=9_999,
        externality_conserved=True,
        exact_schedule_complete=False,
        exact_admission_type="ExactAdmissionResult",
        exact_schedule_type="ExactScheduleResult",
        method_spec_hashes=("a" * 64,),
        adapted_baseline_mapping_hash="b" * 64,
    )
    errors = validate_algorithm_claim_gate(invalid)
    assert "iteration_cap" in errors
    assert "incremental_actions_checked" in errors
    assert "exact_schedule_complete" in errors


def test_claim_gate_accepts_complete_static_evidence() -> None:
    valid = _valid_record()
    assert validate_algorithm_claim_gate(valid) == ()


def test_algorithm_audit_record_is_frozen_and_has_only_contract_fields() -> None:
    record = _valid_record()
    assert tuple(field.name for field in fields(record)) == (
        "search_stop_reasons",
        "local_optimum_certified",
        "incremental_actions_checked",
        "externality_conserved",
        "exact_schedule_complete",
        "exact_admission_type",
        "exact_schedule_type",
        "method_spec_hashes",
        "adapted_baseline_mapping_hash",
    )
    with pytest.raises(FrozenInstanceError):
        record.incremental_actions_checked = 10_001  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    [
        (
            {
                "search_stop_reasons": ("no_improving_move", "iteration_cap"),
                "local_optimum_certified": (True,),
            },
            "stop_reason_certification_mismatch",
        ),
        (
            {"local_optimum_certified": (False,)},
            "stop_reason_certification_mismatch",
        ),
        (
            {
                "search_stop_reasons": ("iteration_cap",),
                "local_optimum_certified": (True,),
            },
            "iteration_cap",
        ),
        ({"incremental_actions_checked": 9_999}, "incremental_actions_checked"),
        ({"externality_conserved": False}, "externality_conserved"),
        ({"exact_schedule_complete": False}, "exact_schedule_complete"),
        (
            {
                "exact_admission_type": "ExactResult",
                "exact_schedule_type": "ExactResult",
            },
            "exact_result_types_distinct",
        ),
        ({"method_spec_hashes": ()}, "method_spec_hashes"),
        ({"method_spec_hashes": ("A" * 64,)}, "method_spec_hashes"),
        ({"method_spec_hashes": ("a" * 63,)}, "method_spec_hashes"),
        (
            {"adapted_baseline_mapping_hash": "G" * 64},
            "adapted_baseline_mapping_hash",
        ),
        (
            {"adapted_baseline_mapping_hash": "b" * 65},
            "adapted_baseline_mapping_hash",
        ),
    ],
)
def test_claim_gate_rejects_invalid_static_evidence(
    overrides: dict[str, object], expected_error: str
) -> None:
    assert expected_error in validate_algorithm_claim_gate(_valid_record(**overrides))


def test_claim_gate_error_codes_are_stable_and_not_duplicated() -> None:
    errors = validate_algorithm_claim_gate(
        _valid_record(
            search_stop_reasons=("iteration_cap",),
            local_optimum_certified=(True,),
            incremental_actions_checked=0,
            externality_conserved=False,
            exact_schedule_complete=False,
            exact_admission_type="ExactResult",
            exact_schedule_type="ExactResult",
            method_spec_hashes=("INVALID", "INVALID"),
            adapted_baseline_mapping_hash="INVALID",
        )
    )
    assert errors == (
        "iteration_cap",
        "stop_reason_certification_mismatch",
        "incremental_actions_checked",
        "externality_conserved",
        "exact_schedule_complete",
        "exact_result_types_distinct",
        "method_spec_hashes",
        "adapted_baseline_mapping_hash",
    )
