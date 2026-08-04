import csv
import dataclasses
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.exact import (
    ExactScheduleCertificate,
    ExactScheduleResult,
    exact_schedule,
)
from uav_mec.experiment import (
    load_definition,
    method_spec_sha256,
    method_spec_to_dict,
    thesis_method_specs,
)
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy, ExactAdmissionResult
from uav_mec.run_exact_benchmark import (
    _conditional_gap_layers,
    exact_claim_gate,
    run_exact_definition,
)


def _tiny_exact_case() -> tuple[SlotSnapshot, NOMAChannel, EvaluationConfig]:
    tasks = tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=700_000.0,
            comp_cycles=1_500_000_000.0,
            deadline_s=0.8,
            local_cpu_hz=400_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in (1, 2, 3)
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[9e-8], [5e-8], [2e-8]]), 1)
    cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.01,
        cpu_per_uav_hz=2_000_000_000.0,
        allocation_policy=AllocationPolicy.DEADLINE,
    )
    return snapshot, NOMAChannel(ChannelConfig(), 100.0), cfg


def _audited_exact(snapshot, channel, cfg):
    audit = {}
    result = exact_schedule(snapshot, channel, cfg, audit=audit)
    return result, audit


def test_exact_classifies_cpu_and_schedule_nontriviality_independently() -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    result, audit = _audited_exact(snapshot, channel, cfg)

    assert result.certificate.cpu_nontrivial == bool(
        audit["cpu_overloaded_uav_ids"]
    )
    assert result.certificate.schedule_nontrivial == (
        not audit["full_service_wireless_state_exists"]
        or audit["distinct_full_service_secondary_objectives"] > 1
    )


def test_cpu_only_pressure_is_not_schedule_nontrivial() -> None:
    tasks = tuple(
        Task(index, index, 0, 0.2e6, 1.2e9, 0.8, 0.4e9)
        for index in (1, 2)
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[9e-8], [5e-8]]), 1)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 0.01, 1.3e9)

    result, audit = _audited_exact(snapshot, channel, cfg)

    assert audit["full_service_wireless_state_exists"]
    assert result.certificate.cpu_nontrivial
    assert not result.certificate.schedule_nontrivial


def test_radio_capacity_pressure_is_schedule_nontrivial_without_cpu_pressure() -> None:
    tasks = tuple(
        Task(index, index, 0, 0.2e6, 1.2e9, 0.8, 0.4e9)
        for index in (1, 2, 3)
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[9e-8], [6e-8], [3e-8]]), 1)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 0.01, 20e9)

    result, audit = _audited_exact(snapshot, channel, cfg)

    assert not audit["full_service_wireless_state_exists"]
    assert not result.certificate.cpu_nontrivial
    assert result.certificate.schedule_nontrivial


def test_multiple_wireless_states_with_distinct_secondary_objectives_are_schedule_nontrivial() -> None:
    tasks = tuple(
        Task(index, index, 0, 0.2e6, 1.2e9, 0.8, 0.4e9)
        for index in (1, 2)
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[9e-8, 2e-8], [3e-8, 8e-8]]),
        2,
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 0.01, 20e9)

    result, audit = _audited_exact(snapshot, channel, cfg)

    assert audit["full_service_wireless_state_exists"]
    assert audit["distinct_full_service_secondary_objectives"] > 1
    assert result.certificate.schedule_nontrivial


def test_exact_claim_gate_requires_thirty_percent_for_each_claim_type() -> None:
    assert exact_claim_gate(0.30, 0.30) == {
        "dd_optimality_interpretation": True,
        "iea_near_optimality_interpretation": True,
    }
    assert exact_claim_gate(0.29, 0.30)["dd_optimality_interpretation"] is False
    assert (
        exact_claim_gate(0.30, 0.29)["iea_near_optimality_interpretation"]
        is False
    )


def _run_audited_exact_bundle(tmp_path, development_seed_access):
    config_path = Path("configs/thesis/confirmatory_exact.toml")
    base = load_definition(config_path)
    definition = replace(
        base,
        seeds=(42,),
        scenario=replace(base.scenario, num_slots=1, num_users=3),
    )
    output = tmp_path / "audited-exact"
    run_exact_definition(
        definition,
        config_path,
        output,
        seed_access=development_seed_access,
    )
    return output


def test_exact_benchmark_writes_two_nontriviality_flags_and_reasons(
    tmp_path, development_seed_access
) -> None:
    output = _run_audited_exact_bundle(tmp_path, development_seed_access)
    with (output / "exact_slots.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    for row in rows:
        assert row["cpu_nontrivial"] in {"True", "False"}
        assert row["schedule_nontrivial"] in {"True", "False"}
        assert row["nontriviality_reason"]
        assert row["certificate_complete"] == "True"


def test_exact_summary_reports_cpu_and_schedule_nontrivial_fractions(
    tmp_path, development_seed_access
) -> None:
    output = _run_audited_exact_bundle(tmp_path, development_seed_access)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert 0.0 <= summary["cpu_nontrivial_fraction"] <= 1.0
    assert 0.0 <= summary["schedule_nontrivial_fraction"] <= 1.0
    assert summary["nontriviality_thresholds"] == {
        "cpu": 0.30,
        "schedule": 0.30,
    }
    assert set(summary["claim_gates"]) == {
        "dd_optimality_interpretation",
        "iea_near_optimality_interpretation",
    }


def test_exact_rows_keep_all_three_conditional_gap_layers(
    tmp_path, development_seed_access
) -> None:
    assert _conditional_gap_layers((2, -3.0, -4.0), (1, -2.0, -3.0)) == (
        1,
        None,
        None,
    )
    assert _conditional_gap_layers((2, -2.0, -4.0), (2, -3.0, -3.0)) == (
        0,
        1.0,
        None,
    )
    assert _conditional_gap_layers((2, -2.0, -2.0), (2, -2.0, -3.0)) == (
        0,
        0.0,
        1.0,
    )
    output = _run_audited_exact_bundle(tmp_path, development_seed_access)
    with (output / "exact_slots.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert all(
        {"completion_gap", "cpu_gap_hz", "interference_gap_w"} <= set(row)
        for row in rows
    )


def test_exact_schedule_uses_plan2_result_and_certificate_types() -> None:
    assert tuple(field.name for field in dataclasses.fields(ExactScheduleResult)) == (
        "plan",
        "objective",
        "certificate",
    )
    assert tuple(field.name for field in dataclasses.fields(ExactScheduleCertificate)) == (
        "complete",
        "raw_states_enumerated",
        "unique_closed_states",
        "best_bound",
        "lexicographic_gap",
        "gap_status",
        "cpu_nontrivial",
        "schedule_nontrivial",
        "semantic_fingerprint",
    )


def test_exact_schedule_never_serializes_as_exact_admission(
    tmp_path, development_seed_access
) -> None:
    output = _run_audited_exact_bundle(tmp_path, development_seed_access)
    payload = (output / "exact_slots.csv").read_text(encoding="utf-8")

    assert "exact_schedule" in payload
    assert "exact_admission" not in payload


def test_legacy_exact_result_and_nontriviality_types_are_absent() -> None:
    source = Path("uav_mec/exact.py").read_text(encoding="utf-8")
    assert "class ExactResult" not in source
    assert "class ExactNontriviality" not in source


def test_exact_schedule_has_a_distinct_complete_certificate() -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    result = exact_schedule(snapshot, channel, cfg)
    assert isinstance(result, ExactScheduleResult)
    assert not isinstance(result, ExactAdmissionResult)
    assert result.certificate.complete
    assert result.certificate.raw_states_enumerated > result.certificate.unique_closed_states
    assert result.certificate.best_bound == result.objective
    assert result.certificate.lexicographic_gap == (0, 0.0, 0.0)
    assert set(result.plan.schedule.task_ids) == set(result.plan.allocation.completed_task_ids)
    assert result.certificate.cpu_nontrivial
    assert result.certificate.schedule_nontrivial


def test_capped_exact_schedule_has_a_bounded_incomplete_certificate() -> None:
    snapshot, channel, cfg = _tiny_exact_case()

    result = exact_schedule(snapshot, channel, cfg, state_cap=1)

    assert not result.certificate.complete
    assert result.certificate.raw_states_enumerated == 1
    assert result.certificate.gap_status == "bounded"
    assert all(component is not None for component in result.certificate.best_bound)
    assert all(component is not None for component in result.certificate.lexicographic_gap)


def test_incomplete_exact_schedule_does_not_claim_schedule_nontriviality() -> None:
    base_snapshot, channel, cfg = _tiny_exact_case()
    snapshot = SlotSnapshot(
        0,
        base_snapshot.tasks[:2],
        base_snapshot.planning_channel_gains[:2],
        1,
    )
    cfg = replace(cfg, cpu_per_uav_hz=8_000_000_000.0)

    capped = exact_schedule(snapshot, channel, cfg, state_cap=2)
    exhaustive = exact_schedule(snapshot, channel, cfg)

    assert not capped.certificate.complete
    assert capped.objective[0] == 1
    assert exhaustive.objective[0] == 2
    assert not exhaustive.certificate.schedule_nontrivial
    assert not capped.certificate.schedule_nontrivial


def test_state_cap_at_iterator_exhaustion_is_complete() -> None:
    base_snapshot, channel, cfg = _tiny_exact_case()
    snapshot = SlotSnapshot(
        0,
        base_snapshot.tasks[:1],
        base_snapshot.planning_channel_gains[:1],
        1,
    )

    result = exact_schedule(snapshot, channel, cfg, state_cap=2)

    assert result.certificate.raw_states_enumerated == 2
    assert result.certificate.complete
    assert result.certificate.gap_status == "exact"


def test_timeout_returns_a_bounded_incomplete_certificate(monkeypatch) -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    monotonic_values = iter((0.0, 1.0))
    monkeypatch.setattr(
        "uav_mec.exact.time.monotonic",
        lambda: next(monotonic_values),
    )

    result = exact_schedule(snapshot, channel, cfg, timeout_s=0.5)

    assert result.certificate.raw_states_enumerated == 1
    assert not result.certificate.complete
    assert result.certificate.gap_status == "bounded"
    assert not result.certificate.schedule_nontrivial


def test_exact_schedule_propagates_independent_downlink_rates() -> None:
    base_snapshot, channel, cfg = _tiny_exact_case()
    task = replace(base_snapshot.tasks[0], result_ratio=0.25)
    snapshot = SlotSnapshot(
        0,
        (task,),
        base_snapshot.planning_channel_gains[:1],
        1,
    )

    result = exact_schedule(
        snapshot,
        channel,
        replace(cfg, cpu_per_uav_hz=8_000_000_000.0),
        downlink_rates_bps={task.task_id: 5_000_000.0},
    )

    assert result.certificate.complete
    assert result.plan.schedule.task_ids == (task.task_id,)


def test_exact_schedule_counts_only_unique_user_raw_assignments() -> None:
    base_snapshot, channel, cfg = _tiny_exact_case()
    tasks = (
        replace(base_snapshot.tasks[0], user_id=7, job_index=0),
        replace(base_snapshot.tasks[1], user_id=7, job_index=1),
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        base_snapshot.planning_channel_gains[:2],
        1,
    )

    result = exact_schedule(
        snapshot,
        channel,
        replace(cfg, cpu_per_uav_hz=8_000_000_000.0),
    )

    assert result.certificate.complete
    assert result.certificate.raw_states_enumerated == 3


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("best_bound", (1.5, 0.0, 0.0)),
        ("lexicographic_gap", (0.5, 0.0, 0.0)),
    ),
)
def test_exact_certificate_requires_integer_completion_components(
    field: str,
    value: tuple[float, float, float],
) -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    certificate = exact_schedule(
        snapshot,
        channel,
        cfg,
        state_cap=1,
    ).certificate
    assert isinstance(certificate, ExactScheduleCertificate)

    with pytest.raises(ValueError, match="completion component"):
        replace(certificate, **{field: value})


def test_bounded_exact_certificate_cannot_be_below_incumbent() -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    result = exact_schedule(snapshot, channel, cfg, state_cap=1)
    invalid = replace(
        result.certificate,
        best_bound=(-1, 0.0, 0.0),
        lexicographic_gap=(-1, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="bound.*objective"):
        replace(result, certificate=invalid)


def test_bounded_exact_certificate_gap_matches_bound_minus_objective() -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    result = exact_schedule(snapshot, channel, cfg, state_cap=1)
    invalid = replace(
        result.certificate,
        lexicographic_gap=(0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="gap.*bound.*objective"):
        replace(result, certificate=invalid)


def test_exact_benchmark_rejects_an_incomplete_certificate(
    tmp_path: Path,
    development_seed_access,
    monkeypatch,
) -> None:
    config_path = Path("configs/thesis/confirmatory_exact.toml")
    base = load_definition(config_path)
    definition = replace(
        base,
        seeds=(42,),
        scenario=replace(
            base.scenario,
            num_slots=1,
            num_users=1,
            random_burst=False,
            task=replace(
                base.scenario.task,
                arrival_prob=1.0,
                data_min=2_000_000.0,
                data_max=2_000_000.0,
                comp_min=1_000.0,
                comp_max=1_000.0,
            ),
        ),
    )

    def capped_exact(snapshot, channel, cfg, max_heavy_tasks, *, audit):
        return exact_schedule(
            snapshot,
            channel,
            cfg,
            max_heavy_tasks=max_heavy_tasks,
            state_cap=1,
            audit=audit,
        )

    monkeypatch.setattr(
        "uav_mec.run_exact_benchmark.exact_schedule",
        capped_exact,
    )

    with pytest.raises(RuntimeError, match="incomplete exact certificate"):
        run_exact_definition(
            definition,
            config_path,
            tmp_path / "incomplete-exact",
            seed_access=development_seed_access,
        )


def test_exact_benchmark_never_reports_heuristic_above_optimum(
    tmp_path: Path,
    development_seed_access,
):
    config_path = Path("configs/thesis/confirmatory_exact.toml")
    base = load_definition(config_path)
    definition = replace(
        base,
        seeds=(42,),
        scenario=replace(
            base.scenario,
            num_slots=2,
            num_users=4,
        ),
    )
    output = tmp_path / "exact"

    run_exact_definition(
        definition,
        config_path,
        output,
        seed_access=development_seed_access,
    )

    with (output / "exact_slots.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["complete_search"] == "True" for row in rows)
    assert all(row["result_schema"] == "exact_schedule" for row in rows)
    assert all(row["certificate_label"] == "Exact" for row in rows)
    assert all(int(row["full_completed"]) <= int(row["exact_completed"]) for row in rows)
    assert all(int(row["evaluated_schedules"]) > 0 for row in rows)

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert 0.0 <= summary["optimal_slot_fraction"] <= 1.0
    assert summary["mean_completion_gap"] >= 0.0
    assert summary["result_schema"] == "exact_schedule"
    assert summary["certificate_label"] == "Exact"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resolved_definition"]["methods"] == ["full", "exact"]
    specs = thesis_method_specs()
    for method_id in ("full", "exact"):
        serialized = method_spec_to_dict(specs[method_id])
        factor_hash = method_spec_sha256(specs[method_id])
        assert summary["methods"][method_id] == {
            "method_spec": serialized,
            "method_spec_sha256": factor_hash,
        }
        assert manifest["method_specs"][method_id] == {
            "factor_hash": factor_hash,
            "spec": serialized,
        }


def test_exact_benchmark_rejects_allocation_factor_drift_before_seed_access(
    tmp_path: Path,
) -> None:
    config_path = Path("configs/thesis/confirmatory_exact.toml")
    base = load_definition(config_path)
    incompatible = replace(
        base,
        evaluation=replace(
            base.evaluation,
            allocation_policy=AllocationPolicy.EDF_MIN_CPU,
        ),
    )

    with pytest.raises(ValueError, match="allocation_policy"):
        run_exact_definition(
            incompatible,
            config_path,
            tmp_path / "incompatible-exact",
            seed_access=object(),  # type: ignore[arg-type]
        )


def test_exact_benchmark_rejects_complete_spec_hash_drift_before_seed_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = Path("configs/thesis/confirmatory_exact.toml")
    definition = load_definition(config_path)
    specs = thesis_method_specs()
    specs["exact"] = replace(specs["exact"], display_name="drifted exact")
    monkeypatch.setattr(
        "uav_mec.run_exact_benchmark.thesis_method_specs",
        lambda: specs,
    )

    with pytest.raises(ValueError, match="factor hash"):
        run_exact_definition(
            definition,
            config_path,
            tmp_path / "drifted-exact",
            seed_access=object(),  # type: ignore[arg-type]
        )
