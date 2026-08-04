from dataclasses import FrozenInstanceError, fields

import pytest

from uav_mec.runtime_audit import AlgorithmAuditRecord, validate_algorithm_claim_gate


VALID_DIGEST = "a" * 64


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
