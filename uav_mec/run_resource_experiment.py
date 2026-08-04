"""Dedicated fixed-radio Resource experiment runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tomllib

from .artifacts import RunArtifactWriter
from .domain import ChannelBasis, Schedule, ScheduleEntry, SlotOutcome, SlotSnapshot
from .evaluation import EvaluationConfig
from .experiment import ExperimentDefinition, _slot_method, definition_to_dict
from .model import NOMAChannel, ScenarioConfig
from .resource_allocation import AllocationPolicy
from .resource_snapshots import (
    RadioGenerator,
    RawRadioSnapshot,
    build_radio_neutral_schedule,
    evaluate_frozen_resource_snapshot,
    freeze_radio_snapshot,
    raw_snapshot_to_dict,
)
from .schedule import normalize_roles
from .seed_access import SeedAccessRecorder
from .trace import generate_trace

RESOURCE_POLICIES = {
    "fixed_half": AllocationPolicy.FIXED_HALF_EQUAL,
    "crossing_alpha_min": AllocationPolicy.CROSSING_EQUAL,
    "deadline_first_min_cpu": AllocationPolicy.EDF_MIN_CPU,
    "min_slack_min_cpu": AllocationPolicy.MIN_SLACK_MIN_CPU,
    "dd_mcaa": AllocationPolicy.DEADLINE,
    "exact_admission": "exact_admission",
}


@dataclass(frozen=True, slots=True)
class ResourcePolicyOutcome:
    generator: RadioGenerator
    policy: str
    snapshot_id: str
    outcome: SlotOutcome


def _validate_resource_config(config_path: Path) -> None:
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    if raw.get("formal_family") != "resource":
        raise ValueError("resource config formal_family mismatch")
    if raw.get("resource_response_policies") != list(RESOURCE_POLICIES):
        raise ValueError("resource_response_policies mismatch")


def _schedule_from_outcome(
    snapshot: SlotSnapshot,
    outcome: SlotOutcome,
) -> Schedule:
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    schedule = Schedule(
        tuple(
            ScheduleEntry(
                record.task_id,
                task_by_id[record.task_id].user_id,
                record.serving_uav,
                record.subcarrier,
                "solo",
            )
            for record in outcome.rates
        )
    )
    return normalize_roles(schedule, snapshot, channel_basis=ChannelBasis.PLANNING)


def _schedule_for_generator(
    generator: RadioGenerator,
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
    scenario: ScenarioConfig,
    max_iterations: int,
    *,
    event_sink: list[dict[str, object]] | None = None,
) -> Schedule:
    if generator is RadioGenerator.RADIO_NEUTRAL:
        return build_radio_neutral_schedule(snapshot, channel, evaluation)
    if generator is RadioGenerator.DEADLINE_FIRST_GENERATED:
        target = replace(evaluation, allocation_policy=AllocationPolicy.EDF_MIN_CPU)
    elif generator is RadioGenerator.DD_GENERATED:
        target = replace(evaluation, allocation_policy=AllocationPolicy.DEADLINE)
    else:
        raise ValueError(f"unsupported radio generator {generator}")
    outcome = _slot_method(
        "full",
        scenario,
        target,
        max_iterations,
        [] if event_sink is None else event_sink,
    )(snapshot)
    return _schedule_from_outcome(snapshot, outcome)


def materialize_resource_generator_slot(
    seed: int,
    snapshot: SlotSnapshot,
    generator: RadioGenerator,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
    max_iterations: int,
    scenario: ScenarioConfig | None = None,
    *,
    event_sink: list[dict[str, object]] | None = None,
) -> RawRadioSnapshot:
    """Search one generator once and freeze its policy-independent radio state."""

    resolved_scenario = scenario or ScenarioConfig(
        num_users=max(len(snapshot.tasks), 1),
        num_uavs=snapshot.num_uavs,
        n_subcarriers=evaluation.n_subcarriers,
    )
    return freeze_radio_snapshot(
        seed,
        snapshot,
        generator,
        _schedule_for_generator(
            generator,
            snapshot,
            channel,
            evaluation,
            resolved_scenario,
            max_iterations,
            event_sink=event_sink,
        ),
        channel,
        evaluation,
    )


def materialize_resource_slot(
    seed: int,
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
    max_iterations: int,
    scenario: ScenarioConfig | None = None,
) -> tuple[RawRadioSnapshot, ...]:
    """Search each generator once and freeze exactly one radio snapshot."""
    return tuple(
        materialize_resource_generator_slot(
            seed,
            snapshot,
            generator,
            channel,
            evaluation,
            max_iterations,
            scenario,
        )
        for generator in RadioGenerator
    )


def evaluate_resource_generator(
    snapshot: SlotSnapshot,
    raw_snapshot: RawRadioSnapshot,
    evaluation: EvaluationConfig,
) -> tuple[ResourcePolicyOutcome, ...]:
    """Replay all six CPU responses on one byte-identical radio snapshot."""

    return tuple(
        ResourcePolicyOutcome(
            raw_snapshot.generator,
            policy_name,
            raw_snapshot.snapshot_id,
            evaluate_frozen_resource_snapshot(
                snapshot,
                raw_snapshot,
                policy,
                evaluation,
            ),
        )
        for policy_name, policy in RESOURCE_POLICIES.items()
    )


def evaluate_resource_slot(
    snapshot: SlotSnapshot,
    raw_snapshots: tuple[RawRadioSnapshot, ...],
    evaluation: EvaluationConfig,
) -> tuple[ResourcePolicyOutcome, ...]:
    """Apply all six CPU responses without changing frozen radio state."""
    if {raw.generator for raw in raw_snapshots} != set(RadioGenerator):
        raise ValueError("raw_snapshots must contain every generator exactly once")
    if len(raw_snapshots) != len(RadioGenerator):
        raise ValueError("raw_snapshots contains a duplicate generator")
    return tuple(
        row
        for raw in raw_snapshots
        for row in evaluate_resource_generator(snapshot, raw, evaluation)
    )


def run_resource_definition(
    definition: ExperimentDefinition,
    config_path: Path,
    output_dir: Path,
    *,
    seed_access: SeedAccessRecorder,
) -> None:
    """Execute one Resource definition and publish a checksummed raw bundle."""
    if not isinstance(seed_access, SeedAccessRecorder):
        raise TypeError("seed_access must be a SeedAccessRecorder")
    _validate_resource_config(config_path)
    seed_access.record(
        "run_resource_definition",
        definition.seeds,
        {"run_id": definition.run_id, "config_path": str(config_path)},
    )
    writer = RunArtifactWriter(output_dir)
    channel = NOMAChannel(definition.scenario.channel, definition.scenario.uav.height)
    raw_rows: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    totals = {
        (generator.value, policy): [0, 0]
        for generator in RadioGenerator
        for policy in RESOURCE_POLICIES
    }
    for seed in definition.seeds:
        trace = generate_trace(
            definition.scenario,
            seed,
            definition.csi_error_std,
            seed_access=seed_access,
        )
        for snapshot in trace.slots:
            raw = materialize_resource_slot(
                seed,
                snapshot,
                channel,
                definition.evaluation,
                definition.max_iterations,
                definition.scenario,
            )
            raw_rows.extend(raw_snapshot_to_dict(item) for item in raw)
            evaluated = evaluate_resource_slot(snapshot, raw, definition.evaluation)
            for row in evaluated:
                outcome = row.outcome
                key = (row.generator.value, row.policy)
                totals[key][0] += len(outcome.completed_task_ids)
                totals[key][1] += len(outcome.arrived_task_ids)
                slot_rows.append(
                    {
                        "seed": seed,
                        "slot": snapshot.slot_index,
                        "generator": row.generator.value,
                        "policy": row.policy,
                        "snapshot_id": row.snapshot_id,
                        "arrived": len(outcome.arrived_task_ids),
                        "completed": len(outcome.completed_task_ids),
                        "on_time_rate": (
                            len(outcome.completed_task_ids) / max(len(outcome.arrived_task_ids), 1)
                        ),
                    }
                )
                for decision in outcome.decisions:
                    task_rows.append(
                        {
                            "seed": seed,
                            "slot": snapshot.slot_index,
                            "generator": row.generator.value,
                            "policy": row.policy,
                            "snapshot_id": row.snapshot_id,
                            "task_id": decision.task_id,
                            "alpha": decision.alpha,
                            "cpu_hz": decision.cpu_hz,
                            "completed": decision.completed,
                            "reason": decision.reason,
                        }
                    )
    summary_by_generator: dict[str, object] = {}
    for generator in RadioGenerator:
        rates = {
            policy: (
                totals[(generator.value, policy)][0] / max(totals[(generator.value, policy)][1], 1)
            )
            for policy in RESOURCE_POLICIES
        }
        summary_by_generator[generator.value] = {
            "policies": rates,
            "dd_minus_deadline_first_pp": 100.0
            * (rates["dd_mcaa"] - rates["deadline_first_min_cpu"]),
        }
    summary = {
        "primary_generator": RadioGenerator.DD_GENERATED.value,
        "by_generator": summary_by_generator,
        "raw_snapshot_count": len(raw_rows),
        "slot_policy_row_count": len(slot_rows),
    }
    writer.write_jsonl("raw_radio_snapshots.jsonl", raw_rows)
    writer.write_csv(
        "slots.csv",
        slot_rows,
        (
            "seed",
            "slot",
            "generator",
            "policy",
            "snapshot_id",
            "arrived",
            "completed",
            "on_time_rate",
        ),
    )
    writer.write_csv(
        "tasks.csv",
        task_rows,
        (
            "seed",
            "slot",
            "generator",
            "policy",
            "snapshot_id",
            "task_id",
            "alpha",
            "cpu_hz",
            "completed",
            "reason",
        ),
    )
    writer.write_json("summary.json", summary)
    writer.finalize(
        {
            "schema_version": 2,
            "run_kind": "fixed_radio_resource",
            "resolved_definition": definition_to_dict(definition),
            "config_path": str(config_path),
            "raw_snapshot_artifact": "raw_radio_snapshots.jsonl",
            "generators": [item.value for item in RadioGenerator],
            "resource_policies": list(RESOURCE_POLICIES),
        }
    )
