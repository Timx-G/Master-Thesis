import json
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_mec.domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy
from uav_mec.resource_snapshots import (
    RadioGenerator,
    build_deadline_first_schedule,
    build_radio_neutral_schedule,
    evaluate_frozen_resource_snapshot,
    freeze_radio_snapshot,
)
from uav_mec.schedule import validate_schedule
from uav_mec.run_resource_experiment import (
    RESOURCE_POLICIES,
    evaluate_resource_slot,
    materialize_resource_slot,
    run_resource_definition,
)
from uav_mec.experiment import load_definition


def _fixture():
    tasks = (
        Task(1, 1, 0, 1e6, 1.4e9, 0.8, 1e9),
        Task(2, 2, 0, 1e6, 1.8e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[1e-7], [6e-8]]), 1)
    schedule = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(2, 2, 0, 0, "weak"),
        )
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 0.01, 1.2e9)
    return snapshot, schedule, channel, cfg


def test_resource_policies_share_one_byte_identical_raw_assignment_and_rate_snapshot():
    snapshot, schedule, channel, cfg = _fixture()
    raw = freeze_radio_snapshot(42, snapshot, RadioGenerator.DD_GENERATED, schedule, channel, cfg)

    dd = evaluate_frozen_resource_snapshot(snapshot, raw, AllocationPolicy.DEADLINE, cfg)
    deadline_first = evaluate_frozen_resource_snapshot(
        snapshot, raw, AllocationPolicy.EDF_MIN_CPU, cfg
    )

    assert dd.rates is raw.realized_rates
    assert deadline_first.rates is raw.realized_rates
    assert raw.schedule is schedule


def test_rejected_resource_task_remains_in_raw_snapshot_without_rate_recomputation():
    snapshot, schedule, channel, cfg = _fixture()
    raw = freeze_radio_snapshot(42, snapshot, RadioGenerator.DD_GENERATED, schedule, channel, cfg)
    outcome = evaluate_frozen_resource_snapshot(
        snapshot,
        raw,
        AllocationPolicy.DEADLINE,
        replace(cfg, cpu_per_uav_hz=1.0),
    )

    assert {record.task_id for record in raw.realized_rates} == {1, 2}
    assert set(outcome.missed_task_ids) == {1, 2}
    assert outcome.rates is raw.realized_rates


def test_exact_admission_receives_the_preclosure_candidate_set():
    snapshot, schedule, channel, cfg = _fixture()
    raw = freeze_radio_snapshot(42, snapshot, RadioGenerator.DD_GENERATED, schedule, channel, cfg)
    outcome = evaluate_frozen_resource_snapshot(
        snapshot, raw, "exact_admission", replace(cfg, cpu_per_uav_hz=1.0)
    )

    assert {decision.task_id for decision in outcome.decisions} == {1, 2}
    assert {record.task_id for record in outcome.rates} == {1, 2}


def test_resource_snapshot_id_changes_when_assignment_or_rate_bytes_change():
    snapshot, schedule, channel, cfg = _fixture()
    raw = freeze_radio_snapshot(42, snapshot, RadioGenerator.DD_GENERATED, schedule, channel, cfg)
    changed_schedule = Schedule(
        tuple(replace(entry, serving_uav=entry.serving_uav) for entry in reversed(schedule.entries))
    )
    same = freeze_radio_snapshot(
        42, snapshot, RadioGenerator.DD_GENERATED, changed_schedule, channel, cfg
    )
    changed_rate = freeze_radio_snapshot(
        42,
        snapshot,
        RadioGenerator.DD_GENERATED,
        schedule,
        channel,
        replace(cfg, transmit_power_w=0.25),
    )

    assert same.snapshot_id == raw.snapshot_id
    assert changed_rate.snapshot_id != raw.snapshot_id
    with pytest.raises(ValueError, match="content"):
        replace(raw, seed=43)


def test_resource_runner_emits_dd_neutral_and_deadline_first_generators():
    assert tuple(generator.value for generator in RadioGenerator) == (
        "dd_generated",
        "radio_neutral",
        "deadline_first_generated",
    )


def test_resource_config_freezes_all_six_response_policies():
    raw = tomllib.loads(
        Path("configs/thesis/confirmatory_resource_fixed_v2.toml").read_text(encoding="utf-8")
    )
    assert raw["resource_response_policies"] == list(RESOURCE_POLICIES)


def test_resource_generators_return_wirelessly_valid_policy_independent_schedules():
    snapshot, _schedule, channel, cfg = _fixture()

    neutral = build_radio_neutral_schedule(snapshot, channel, cfg)
    deadline_first = build_deadline_first_schedule(snapshot, channel, cfg)

    validate_schedule(neutral, snapshot, cfg.n_subcarriers)
    validate_schedule(deadline_first, snapshot, cfg.n_subcarriers)


def test_resource_runner_searches_each_generator_once_per_seed_slot(monkeypatch):
    snapshot, schedule, channel, cfg = _fixture()
    calls = []
    rate_bases = []

    def fake_schedule(generator, *_args, **_kwargs):
        calls.append(generator)
        return schedule

    monkeypatch.setattr("uav_mec.run_resource_experiment._schedule_for_generator", fake_schedule)
    from uav_mec import resource_snapshots as snapshot_module

    original_compute = snapshot_module.compute_rate_records

    def counted_rates(*args, **kwargs):
        rate_bases.append(kwargs["channel_basis"])
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(snapshot_module, "compute_rate_records", counted_rates)
    raw = materialize_resource_slot(42, snapshot, channel, cfg, 5)

    assert tuple(item.generator for item in raw) == tuple(RadioGenerator)
    assert calls == list(RadioGenerator)
    assert rate_bases.count(ChannelBasis.PLANNING) == len(RadioGenerator)
    assert rate_bases.count(ChannelBasis.REALIZED) == len(RadioGenerator)


def test_resource_runner_emits_fixed_half_and_crossing_alpha_min_for_each_generator(
    monkeypatch,
):
    snapshot, schedule, channel, cfg = _fixture()
    monkeypatch.setattr(
        "uav_mec.run_resource_experiment._schedule_for_generator",
        lambda *_args, **_kwargs: schedule,
    )
    raw = materialize_resource_slot(42, snapshot, channel, cfg, 5)
    rows = evaluate_resource_slot(snapshot, raw, cfg)

    assert set(RESOURCE_POLICIES) == {
        "fixed_half",
        "crossing_alpha_min",
        "deadline_first_min_cpu",
        "min_slack_min_cpu",
        "dd_mcaa",
        "exact_admission",
    }
    assert len(rows) == len(RadioGenerator) * len(RESOURCE_POLICIES)
    assert {(row.generator, row.policy) for row in rows} == {
        (generator, policy) for generator in RadioGenerator for policy in RESOURCE_POLICIES
    }


def _run_small_resource_bundle(tmp_path, development_seed_access):
    config_path = Path("configs/thesis/confirmatory_resource_fixed_v2.toml")
    base = load_definition(config_path)
    definition = replace(
        base,
        seeds=(42,),
        scenario=replace(
            base.scenario,
            num_users=4,
            num_slots=1,
            num_uavs=3,
            num_clusters=3,
            n_subcarriers=1,
        ),
        max_iterations=5,
    )
    output = tmp_path / "resource"
    run_resource_definition(
        definition,
        config_path,
        output,
        seed_access=development_seed_access,
    )
    return output


def test_resource_summary_reports_policy_effect_by_generator(tmp_path, development_seed_access):
    output = _run_small_resource_bundle(tmp_path, development_seed_access)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert set(summary["by_generator"]) == {generator.value for generator in RadioGenerator}
    for value in summary["by_generator"].values():
        assert set(value["policies"]) == set(RESOURCE_POLICIES)
        assert "dd_minus_deadline_first_pp" in value


def test_resource_manifest_hashes_raw_snapshot_artifact(tmp_path, development_seed_access):
    output = _run_small_resource_bundle(tmp_path, development_seed_access)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    checksums = (output / "checksums.sha256").read_text(encoding="utf-8")

    assert manifest["raw_snapshot_artifact"] == "raw_radio_snapshots.jsonl"
    assert "raw_radio_snapshots.jsonl" in checksums
