from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from uav_mec.experiment_registry import load_registry
from uav_mec.runtime_budget import (
    CaseRuntimeBudget,
    FamilyRuntimeBudget,
    RuntimeBudget,
    build_runtime_budget,
    load_and_verify_runtime_budget,
    main,
)

REGISTRY = Path("configs/thesis/experiment_registry_v2.toml")
POLICY = Path("configs/thesis/execution_policy_v2.json")


def _inputs(tmp_path: Path, *, selected_n=50):
    registry = load_registry(REGISTRY)
    power = tmp_path / "power.json"
    power.write_text(
        json.dumps(
            {
                "selected_n": selected_n,
                "evaluated_n": 200,
                "evidence_insufficient": selected_n is None,
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    rows = [
        {
            "family": family.name,
            "case_id": case.case_id,
            "runner_kind": family.name,
            "mean_runtime_s": 0.2,
            "p50_runtime_s": 0.15,
            "p95_runtime_s": 0.4,
            "peak_memory_bytes": 1_000_000,
            "raw_bytes": 1000,
            "summary_bytes": 100,
            "event_bytes": 200,
            "log_bytes": 50,
            "measured_cpu_seconds": 0.18,
            "candidate_count_p50": 10,
            "candidate_count_p95": 20,
            "candidate_count_total": 100,
            "cache_hit_rate": 0.25,
            "reference_hardware": "synthetic",
            "python_environment_sha256": "a" * 64,
        }
        for family in registry.families
        for case in family.cases
    ]
    audit.write_text(json.dumps({"schema_version": 1, "rows": rows}), encoding="utf-8")
    return power, audit


def _build(tmp_path: Path, *, selected_n=50, policy: Path = POLICY):
    power, audit = _inputs(tmp_path, selected_n=selected_n)
    return build_runtime_budget(
        REGISTRY,
        power,
        audit,
        policy,
        tmp_path / "budget.json",
        retry_fraction=0.1,
    )


def test_runtime_budget_covers_every_registry_family(tmp_path: Path) -> None:
    budget = _build(tmp_path)
    assert isinstance(budget, RuntimeBudget)
    assert all(isinstance(family, FamilyRuntimeBudget) for family in budget.families)
    assert all(
        isinstance(case, CaseRuntimeBudget) for family in budget.families for case in family.cases
    )
    assert tuple(budget["families"]) == tuple(
        family.name for family in load_registry(REGISTRY).families
    )

    with pytest.raises(FrozenInstanceError):
        budget.primary_sample_size = 100


def test_runtime_budget_covers_every_registered_case_exactly_once(tmp_path: Path) -> None:
    budget = _build(tmp_path)
    expected = {
        (family.name, case.case_id)
        for family in load_registry(REGISTRY).families
        for case in family.cases
    }
    actual = {(row["family"], row["case_id"]) for row in budget["cases"]}
    assert actual == expected
    assert len(budget["cases"]) == len(expected)


def test_runtime_budget_includes_cpu_wall_memory_disk_log_concurrency_and_retry_fields(
    tmp_path: Path,
) -> None:
    row = _build(tmp_path)["cases"][0]
    assert {
        "cpu_hours",
        "wall_hours_p95",
        "peak_concurrent_memory_bytes",
        "disk_bytes",
        "log_bytes",
        "concurrency",
        "retry_fraction",
        "retry_cpu_hours",
    } <= set(row)


def test_runtime_budget_binds_positive_shard_size_and_execution_policy_hash_per_family(
    tmp_path: Path,
) -> None:
    budget = _build(tmp_path)
    assert len(budget["execution_policy_sha256"]) == 64
    assert all(row["shard_size"] > 0 for row in budget["cases"])


def test_runtime_budget_uses_power_selected_n_only_for_h1_h4_families(
    tmp_path: Path,
) -> None:
    budget = _build(tmp_path, selected_n=50)
    by_key = {(row["family"], row["case_id"]): row for row in budget["cases"]}
    assert by_key[("main", "main_k3_n45")]["seed_count"] == 50
    assert by_key[("joint", "joint_k3_n45")]["seed_count"] == 50
    assert by_key[("externality", "overlap_high_coupling1_load12")]["seed_count"] == 50
    assert by_key[("externality", "overlap_low_coupling0_load08")]["seed_count"] == 30


def test_runtime_budget_uses_evaluated_n_two_hundred_when_selected_n_is_null(
    tmp_path: Path,
) -> None:
    budget = _build(tmp_path, selected_n=None)
    assert budget["primary_sample_size"] == 200


def test_runtime_budget_marks_power_guarantee_false_for_null_selection_fallback(
    tmp_path: Path,
) -> None:
    assert _build(tmp_path, selected_n=None)["power_guarantee"] is False


def test_runtime_budget_uses_thirty_seeds_for_each_descriptive_externality_grid(
    tmp_path: Path,
) -> None:
    rows = [row for row in _build(tmp_path)["cases"] if row["family"] == "externality"]
    assert sum(row["seed_count"] == 30 for row in rows) == 8


def test_runtime_budget_uses_fixed_counts_for_exact_scaling_robustness_and_trace_driven(
    tmp_path: Path,
) -> None:
    rows = _build(tmp_path)["cases"]
    assert {row["seed_count"] for row in rows if row["family"] == "exact"} == {50}
    assert all(
        row["seed_count"] == 30
        for row in rows
        if row["family"] in {"scaling", "robustness", "trace_driven"}
    )


def test_runtime_budget_rejects_missing_family_audit(tmp_path: Path) -> None:
    power, audit = _inputs(tmp_path)
    raw = json.loads(audit.read_text(encoding="utf-8"))
    raw["rows"].pop()
    audit.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="audit.*coverage"):
        build_runtime_budget(REGISTRY, power, audit, POLICY, tmp_path / "out.json")


def test_runtime_budget_rejects_nonpositive_concurrency_shard_size_or_retry_outside_zero_one(
    tmp_path: Path,
) -> None:
    power, audit = _inputs(tmp_path)
    raw = json.loads(POLICY.read_text(encoding="utf-8"))
    raw["families"][0]["concurrency"] = 0
    invalid = tmp_path / "policy.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="concurrency"):
        build_runtime_budget(REGISTRY, power, audit, invalid, tmp_path / "bad.json")
    with pytest.raises(ValueError, match="retry_fraction"):
        build_runtime_budget(
            REGISTRY, power, audit, POLICY, tmp_path / "bad2.json", retry_fraction=1.1
        )


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate"))
def test_runtime_budget_rejects_missing_extra_or_duplicate_execution_policy_family(
    tmp_path: Path, mutation: str
) -> None:
    power, audit = _inputs(tmp_path)
    raw = json.loads(POLICY.read_text(encoding="utf-8"))
    if mutation == "missing":
        raw["families"].pop()
    elif mutation == "extra":
        raw["families"].append({"family": "invented", "concurrency": 1, "shard_size": 1})
    else:
        raw["families"][-1] = dict(raw["families"][0])
    invalid = tmp_path / "invalid-policy.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="policy.*family"):
        build_runtime_budget(REGISTRY, power, audit, invalid, tmp_path / "out.json")


def test_runtime_budget_checksum_changes_when_power_or_audit_input_changes(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _build(first_root, selected_n=50)
    second = _build(second_root, selected_n=100)
    assert first["content_sha256"] != second["content_sha256"]
    assert first["power_decision_sha256"] != second["power_decision_sha256"]


def test_resource_budget_counts_one_search_and_six_responses_per_generator_case(
    tmp_path: Path,
) -> None:
    rows = [row for row in _build(tmp_path)["cases"] if row["family"] == "resource"]
    assert len(rows) == 3
    assert all(row["wireless_searches"] == row["seed_count"] * row["slot_count"] for row in rows)
    assert all(row["resource_policy_evaluations"] == 6 * row["wireless_searches"] for row in rows)


def test_runtime_budget_load_verifies_input_and_content_digests(tmp_path: Path) -> None:
    _build(tmp_path)
    loaded = load_and_verify_runtime_budget(tmp_path / "budget.json")
    assert loaded["content_sha256"]
    power = tmp_path / "power.json"
    power.write_text(power.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="power.*digest"):
        load_and_verify_runtime_budget(tmp_path / "budget.json")


def test_runtime_budget_cli_build_and_verify_reject_unknown_override(
    tmp_path: Path,
) -> None:
    power, audit = _inputs(tmp_path)
    output = tmp_path / "budget.json"
    main(
        [
            "build",
            "--registry",
            str(REGISTRY),
            "--power-decision",
            str(power),
            "--runtime-audit",
            str(audit),
            "--execution-policy",
            str(POLICY),
            "--retry-fraction",
            "0.1",
            "--output",
            str(output),
        ]
    )
    main(["verify", "--budget", str(output)])
    with pytest.raises(SystemExit) as error:
        main(["verify", "--budget", str(output), "--workers", "2"])
    assert error.value.code == 2


def test_execution_policy_freezes_exact_one_other_two_and_all_shard_sizes_one() -> None:
    raw = json.loads(POLICY.read_text(encoding="utf-8"))
    assert [row["family"] for row in raw["families"]] == [
        "resource",
        "exact",
        "externality",
        "main",
        "joint",
        "scaling",
        "robustness",
        "trace_driven",
    ]
    assert {row["shard_size"] for row in raw["families"]} == {1}
    assert next(row for row in raw["families"] if row["family"] == "exact")["concurrency"] == 1
    assert all(row["concurrency"] == 2 for row in raw["families"] if row["family"] != "exact")
