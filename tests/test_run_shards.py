from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from uav_mec.artifacts import RunArtifactWriter
from uav_mec.run_shards import (
    ArtifactSpec,
    ShardPlan,
    ShardSpec,
    create_shard_plan,
    merge_shards,
    resume_shards,
    run_one_shard,
    shard_plan_sha256,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64


def _plan(*, source: str = DIGEST_C, environment: str = DIGEST_D) -> ShardPlan:
    return ShardPlan(
        family="main",
        case_id="default",
        config_sha256=DIGEST_A,
        registry_sha256=DIGEST_A,
        lock_sha256=DIGEST_B,
        source_sha256=source,
        environment_sha256=environment,
        runner_kind="synthetic",
        bundle_version=2,
        shards=(
            ShardSpec("seed-0000000071-0000000071", (71,)),
            ShardSpec("seed-0000000072-0000000072", (72,)),
        ),
        artifacts=(
            ArtifactSpec("tasks.csv", "csv", "rows", ("seed", "task_id")),
            ArtifactSpec("events.jsonl", "jsonl", "rows", ("seed", "event")),
            ArtifactSpec("summary.json", "json", "json_list", ()),
        ),
    )


def _runner(*, conflict: bool = False, schema_drift: bool = False):
    def run(destination: Path, shard: ShardSpec, plan: ShardPlan, digest: str) -> None:
        seed = shard.seeds[0]
        writer = RunArtifactWriter(destination, bundle_version=2)
        task_seed = 71 if conflict and seed == 72 else seed
        row = {"seed": task_seed, "task_id": 1, "completed": seed == 71}
        if schema_drift and seed == 72:
            row["drift"] = True
        writer.write_csv("tasks.csv", [row])
        writer.write_jsonl("events.jsonl", [{"seed": seed, "event": "done"}])
        writer.write_json("summary.json", {"seed": seed})
        writer.finalize(
            {
                "plan_sha256": digest,
                "family": plan.family,
                "case_id": plan.case_id,
                "seeds": list(shard.seeds),
                "runner_kind": plan.runner_kind,
                "config_sha256": plan.config_sha256,
                "registry_sha256": plan.registry_sha256,
                "lock_sha256": plan.lock_sha256,
                "source_sha256": plan.source_sha256,
                "environment_sha256": plan.environment_sha256,
                "artifacts": {
                    item.relative: {
                        "kind": item.kind,
                        "merge_strategy": item.merge_strategy,
                        "primary_key": list(item.primary_key),
                    }
                    for item in plan.artifacts
                },
            }
        )

    return run


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_shard_plan_binds_family_case_registry_lock_source_environment_and_artifact_schema(
    tmp_path: Path,
) -> None:
    plan = _plan()
    index = create_shard_plan(tmp_path / "case", plan)
    raw = json.loads(index.read_text(encoding="utf-8"))

    assert raw["family"] == "main"
    assert raw["case_id"] == "default"
    assert raw["registry_sha256"] == DIGEST_A
    assert raw["config_sha256"] == DIGEST_A
    assert raw["lock_sha256"] == DIGEST_B
    assert raw["source_sha256"] == DIGEST_C
    assert raw["environment_sha256"] == DIGEST_D
    assert raw["artifacts"][0]["primary_key"] == ["seed", "task_id"]
    assert len(shard_plan_sha256(plan)) == 64


def test_run_one_shard_refuses_a_name_not_in_the_frozen_plan(tmp_path: Path) -> None:
    plan = _plan()
    create_shard_plan(tmp_path / "case", plan)
    with pytest.raises(KeyError):
        run_one_shard(tmp_path / "case", plan, "seed-unknown", _runner())


def test_resume_runs_only_missing_shards(tmp_path: Path) -> None:
    plan = _plan()
    root = tmp_path / "case"
    create_shard_plan(root, plan)
    run_one_shard(root, plan, plan.shards[0].name, _runner())

    report = resume_shards(root, plan, _runner())

    assert report.completed_shards == 1
    assert report.reused_shards == 1


def test_merge_refuses_missing_shard_without_creating_merged_bundle(tmp_path: Path) -> None:
    plan = _plan()
    root = tmp_path / "case"
    create_shard_plan(root, plan)
    run_one_shard(root, plan, plan.shards[0].name, _runner())

    with pytest.raises(FileNotFoundError):
        merge_shards(root, plan)
    assert not (root / "merged").exists()


def test_merge_rejects_duplicate_seed_with_conflicting_valid_payloads(tmp_path: Path) -> None:
    plan = _plan()
    root = tmp_path / "case"
    create_shard_plan(root, plan)
    resume_shards(root, plan, _runner(conflict=True))

    with pytest.raises(ValueError, match="conflicting.*primary key"):
        merge_shards(root, plan)


def test_merge_rejects_schema_drift_across_valid_shards(tmp_path: Path) -> None:
    plan = _plan()
    root = tmp_path / "case"
    create_shard_plan(root, plan)
    resume_shards(root, plan, _runner(schema_drift=True))

    with pytest.raises(ValueError, match="schema"):
        merge_shards(root, plan)


def test_merge_is_byte_identical_across_shard_completion_order(tmp_path: Path) -> None:
    plan = _plan()
    roots = (tmp_path / "forward", tmp_path / "reverse")
    for root, order in zip(roots, (plan.shards, tuple(reversed(plan.shards))), strict=True):
        create_shard_plan(root, plan)
        for shard in order:
            run_one_shard(root, plan, shard.name, _runner())
        merge_shards(root, plan)

    assert _tree_digests(roots[0] / "merged") == _tree_digests(roots[1] / "merged")


def test_generic_merge_preserves_standard_exact_resource_and_trace_artifacts(
    tmp_path: Path,
) -> None:
    plan = replace(
        _plan(),
        artifacts=(
            ArtifactSpec("exact_slots.csv", "csv", "rows", ("seed", "slot")),
            ArtifactSpec(
                "raw_radio_snapshots.jsonl",
                "jsonl",
                "rows",
                ("seed", "slot", "snapshot_id"),
            ),
            ArtifactSpec(
                "trace_provenance.jsonl",
                "jsonl",
                "rows",
                ("seed", "participant"),
            ),
            ArtifactSpec("summary.json", "json", "json_list", ()),
        ),
    )

    def standard_runner(destination, shard, frozen, digest):
        seed = shard.seeds[0]
        writer = RunArtifactWriter(destination, bundle_version=2)
        writer.write_csv("exact_slots.csv", [{"seed": seed, "slot": 0}])
        writer.write_jsonl(
            "raw_radio_snapshots.jsonl",
            [{"seed": seed, "slot": 0, "snapshot_id": f"snapshot-{seed}"}],
        )
        writer.write_jsonl(
            "trace_provenance.jsonl",
            [{"seed": seed, "participant": f"participant-{seed}"}],
        )
        writer.write_json("summary.json", {"seed": seed})
        writer.finalize(
            {
                "plan_sha256": digest,
                "family": frozen.family,
                "case_id": frozen.case_id,
                "seeds": list(shard.seeds),
                "runner_kind": frozen.runner_kind,
                "config_sha256": frozen.config_sha256,
                "registry_sha256": frozen.registry_sha256,
                "lock_sha256": frozen.lock_sha256,
                "source_sha256": frozen.source_sha256,
                "environment_sha256": frozen.environment_sha256,
                "artifacts": {
                    item.relative: {
                        "kind": item.kind,
                        "merge_strategy": item.merge_strategy,
                        "primary_key": list(item.primary_key),
                    }
                    for item in frozen.artifacts
                },
            }
        )

    root = tmp_path / "case"
    create_shard_plan(root, plan)
    resume_shards(root, plan, standard_runner)
    merged = merge_shards(root, plan)
    assert all((merged / item.relative).is_file() for item in plan.artifacts)


def test_resume_rejects_config_source_environment_or_lock_hash_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "case"
    create_shard_plan(root, _plan())

    for changed in (
        replace(_plan(), config_sha256="9" * 64),
        _plan(source="e" * 64),
        _plan(environment="f" * 64),
        replace(_plan(), lock_sha256=None),
    ):
        with pytest.raises(ValueError, match="plan"):
            create_shard_plan(root, changed)
