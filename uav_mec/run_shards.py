"""Immutable, registry-bound shard planning, resume, and generic merge."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy

from .artifacts import RunArtifactWriter, sha256_file


def _digest(value: str | None, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    relative: str
    kind: str
    merge_strategy: str
    primary_key: tuple[str, ...]

    def __post_init__(self) -> None:
        path = Path(self.relative)
        if path.is_absolute() or path == Path(".") or ".." in path.parts:
            raise ValueError("artifact relative path is unsafe")
        if self.kind not in {"csv", "jsonl", "json", "binary"}:
            raise ValueError("unsupported artifact kind")
        if self.merge_strategy not in {"rows", "json_list", "copy"}:
            raise ValueError("unsupported artifact merge_strategy")
        if self.merge_strategy == "rows" and (
            self.kind not in {"csv", "jsonl"} or not self.primary_key
        ):
            raise ValueError("row artifacts require a primary key")
        if len(set(self.primary_key)) != len(self.primary_key):
            raise ValueError("artifact primary_key must be unique")


@dataclass(frozen=True, slots=True)
class ShardSpec:
    name: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name or self.name in {".", ".."}:
            raise ValueError("invalid shard name")
        if not self.seeds or tuple(sorted(set(self.seeds))) != self.seeds:
            raise ValueError("shard seeds must be a sorted unique tuple")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in self.seeds
        ):
            raise ValueError("shard seeds must be non-negative integers")


@dataclass(frozen=True, slots=True)
class ShardPlan:
    family: str
    case_id: str
    config_sha256: str
    registry_sha256: str
    lock_sha256: str | None
    source_sha256: str
    environment_sha256: str
    runner_kind: str
    bundle_version: int
    shards: tuple[ShardSpec, ...]
    artifacts: tuple[ArtifactSpec, ...]

    def __post_init__(self) -> None:
        if not self.family or not self.case_id or not self.runner_kind:
            raise ValueError("family, case_id, and runner_kind are required")
        _digest(self.config_sha256, "config_sha256")
        _digest(self.registry_sha256, "registry_sha256")
        _digest(self.lock_sha256, "lock_sha256", optional=True)
        _digest(self.source_sha256, "source_sha256")
        _digest(self.environment_sha256, "environment_sha256")
        if self.bundle_version != 2:
            raise ValueError("shard bundle_version must be 2")
        if not self.shards or len({item.name for item in self.shards}) != len(self.shards):
            raise ValueError("shard names must be non-empty and unique")
        seeds = [seed for shard in self.shards for seed in shard.seeds]
        if len(seeds) != len(set(seeds)):
            raise ValueError("planned shard seeds must be globally unique")
        if not self.artifacts or len({item.relative for item in self.artifacts}) != len(
            self.artifacts
        ):
            raise ValueError("artifact declarations must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class ShardRunReport:
    completed_shards: int
    reused_shards: int
    merged_path: Path | None = None


def _plan_dict(plan: ShardPlan) -> dict[str, object]:
    return {
        "family": plan.family,
        "case_id": plan.case_id,
        "config_sha256": plan.config_sha256,
        "registry_sha256": plan.registry_sha256,
        "lock_sha256": plan.lock_sha256,
        "source_sha256": plan.source_sha256,
        "environment_sha256": plan.environment_sha256,
        "runner_kind": plan.runner_kind,
        "bundle_version": plan.bundle_version,
        "shards": [asdict(item) for item in sorted(plan.shards, key=lambda item: item.name)],
        "artifacts": [asdict(item) for item in plan.artifacts],
    }


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def shard_plan_sha256(plan: ShardPlan) -> str:
    return hashlib.sha256(_canonical(_plan_dict(plan))).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def create_shard_plan(root: Path, plan: ShardPlan) -> Path:
    case_root = Path(root)
    payload = _canonical(_plan_dict(plan))
    index = case_root / "shard_index.json"
    if index.exists():
        if index.read_bytes() != payload:
            raise ValueError("existing shard plan differs from requested plan")
        return index
    case_root.mkdir(parents=True, exist_ok=True)
    if any(case_root.iterdir()):
        raise ValueError("non-empty shard root is missing its frozen plan")
    _atomic_bytes(index, payload)
    return index


def _verify_v2_bundle(root: Path) -> dict[str, object]:
    manifest_path = root / "MANIFEST.json"
    sums_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        raise ValueError(f"bundle {root.name} lacks v2 closure files")
    declared: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in declared:
            raise ValueError("invalid SHA256SUMS entry")
        declared[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(declared) != actual:
        raise ValueError("SHA256SUMS file set mismatch")
    for relative, digest in declared.items():
        if sha256_file(root / relative) != digest:
            raise ValueError(f"checksum mismatch for {relative}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _artifact_manifest(plan: ShardPlan) -> dict[str, object]:
    return {
        item.relative: {
            "kind": item.kind,
            "merge_strategy": item.merge_strategy,
            "primary_key": list(item.primary_key),
        }
        for item in plan.artifacts
    }


def _validate_shard(path: Path, plan: ShardPlan, shard: ShardSpec) -> None:
    manifest = _verify_v2_bundle(path)
    expected = {
        "plan_sha256": shard_plan_sha256(plan),
        "family": plan.family,
        "case_id": plan.case_id,
        "seeds": list(shard.seeds),
        "runner_kind": plan.runner_kind,
        "config_sha256": plan.config_sha256,
        "registry_sha256": plan.registry_sha256,
        "lock_sha256": plan.lock_sha256,
        "source_sha256": plan.source_sha256,
        "environment_sha256": plan.environment_sha256,
        "artifacts": _artifact_manifest(plan),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"shard {path.name} {field} mismatch")
    for artifact in plan.artifacts:
        if not (path / artifact.relative).is_file():
            raise ValueError(f"shard {path.name} missing artifact {artifact.relative}")


ShardRunner = Callable[[Path, ShardSpec, ShardPlan, str], None]


def run_one_shard(root: Path, plan: ShardPlan, name: str, runner: ShardRunner) -> bool:
    create_shard_plan(root, plan)
    by_name = {item.name: item for item in plan.shards}
    if name not in by_name:
        raise KeyError(name)
    shard = by_name[name]
    destination = Path(root) / "shards" / name
    if destination.exists():
        _validate_shard(destination, plan, shard)
        return False
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.parent / f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        runner(temporary, shard, plan, shard_plan_sha256(plan))
        _validate_shard(temporary, plan, shard)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return True


def resume_shards(root: Path, plan: ShardPlan, runner: ShardRunner) -> ShardRunReport:
    completed = reused = 0
    for shard in sorted(plan.shards, key=lambda item: item.name):
        if run_one_shard(root, plan, shard.name, runner):
            completed += 1
        else:
            reused += 1
    return ShardRunReport(completed, reused)


def _row_key(row: dict[str, object], fields: tuple[str, ...]) -> tuple[object, ...]:
    try:
        return tuple(
            json.dumps(row[field], allow_nan=False, separators=(",", ":"), sort_keys=True)
            for field in fields
        )
    except KeyError as exc:
        raise ValueError(f"row lacks primary key field {exc.args[0]}") from exc


def merge_shards(root: Path, plan: ShardPlan) -> Path:
    create_shard_plan(root, plan)
    paths = []
    for shard in sorted(plan.shards, key=lambda item: item.name):
        path = Path(root) / "shards" / shard.name
        if not path.exists():
            raise FileNotFoundError(path)
        _validate_shard(path, plan, shard)
        paths.append((shard, path))
    merged = Path(root) / "merged"
    source_sums = {shard.name: sha256_file(path / "SHA256SUMS") for shard, path in paths}
    if merged.exists():
        manifest = _verify_v2_bundle(merged)
        if manifest.get("source_shards") != source_sums or manifest.get(
            "plan_sha256"
        ) != shard_plan_sha256(plan):
            raise ValueError("existing merged bundle does not match source shards")
        return merged
    temporary = Path(root) / f".merged.{uuid.uuid4().hex}.tmp"
    writer = RunArtifactWriter(temporary, bundle_version=2)
    try:
        for artifact in plan.artifacts:
            if artifact.merge_strategy == "rows":
                rows: dict[tuple[object, ...], dict[str, object]] = {}
                header: tuple[str, ...] | None = None
                for _shard, path in paths:
                    source = path / artifact.relative
                    if artifact.kind == "csv":
                        with source.open(encoding="utf-8", newline="") as handle:
                            reader = csv.DictReader(handle)
                            current = tuple(reader.fieldnames or ())
                            if header is None:
                                header = current
                            elif header != current:
                                raise ValueError(f"schema drift for {artifact.relative}")
                            materialized = [dict(row) for row in reader]
                    else:
                        materialized = [
                            json.loads(line)
                            for line in source.read_text(encoding="utf-8").splitlines()
                            if line
                        ]
                    for row in materialized:
                        key = _row_key(row, artifact.primary_key)
                        if key in rows and rows[key] != row:
                            raise ValueError(f"conflicting row primary key {key}")
                        rows[key] = row
                ordered = [
                    rows[key] for key in sorted(rows, key=lambda value: tuple(map(str, value)))
                ]
                if artifact.kind == "csv":
                    writer.write_csv(artifact.relative, ordered, header)
                else:
                    writer.write_jsonl(artifact.relative, ordered)
            elif artifact.merge_strategy == "json_list":
                writer.write_json(
                    artifact.relative,
                    [
                        json.loads((path / artifact.relative).read_text(encoding="utf-8"))
                        for _shard, path in paths
                    ],
                )
            else:
                for shard, path in paths:
                    writer.copy_file(
                        f"shards/{shard.name}/{artifact.relative}", path / artifact.relative
                    )
        writer.finalize(
            {
                "plan_sha256": shard_plan_sha256(plan),
                "family": plan.family,
                "case_id": plan.case_id,
                "source_shards": source_sums,
                "artifacts": _artifact_manifest(plan),
            }
        )
        _verify_v2_bundle(temporary)
        os.replace(temporary, merged)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return merged


def run_sharded_experiment(
    config_path: Path,
    output_dir: Path,
    *,
    shard_size: int = 1,
) -> ShardRunReport:
    """Development-only compatibility wrapper over the generic shard APIs."""
    from dataclasses import replace
    from .experiment import _source_snapshot, load_definition, run_definition
    from .seed_access import SeedAccessRecorder, development_permit

    if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size < 1:
        raise ValueError("shard_size must be a positive integer")
    config = Path(config_path)
    root = Path(output_dir)
    definition = load_definition(config)
    groups = tuple(
        tuple(definition.seeds[offset : offset + shard_size])
        for offset in range(0, len(definition.seeds), shard_size)
    )
    registry = Path("configs/thesis/experiment_registry_v2.toml")
    environment_payload = _canonical(
        {
            "python": os.sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
    )
    plan = ShardPlan(
        family="development",
        case_id=(definition.cases[0].case_id if len(definition.cases) == 1 else "default"),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),
        lock_sha256=None,
        source_sha256=str(_source_snapshot()["sha256"]),
        environment_sha256=hashlib.sha256(environment_payload).hexdigest(),
        runner_kind="generic_experiment",
        bundle_version=2,
        shards=tuple(
            ShardSpec(
                f"seed-{seeds[0]:010d}-{seeds[-1]:010d}",
                seeds,
            )
            for seeds in groups
        ),
        artifacts=(
            ArtifactSpec("tasks.csv", "csv", "rows", ("seed", "slot", "method", "task_id")),
            ArtifactSpec("slots.csv", "csv", "rows", ("seed", "slot", "method")),
            ArtifactSpec(
                "events.jsonl",
                "jsonl",
                "rows",
                (
                    "seed",
                    "slot",
                    "method",
                    "move_type",
                    "reason",
                    "before",
                    "after",
                    "schedule_key",
                ),
            ),
            ArtifactSpec("summary.json", "json", "json_list", ()),
        ),
    )
    create_shard_plan(root, plan)
    recorder = SeedAccessRecorder(
        root / "development-seed-access.jsonl",
        development_permit(registry),
    )

    def runner(destination: Path, shard: ShardSpec, frozen: ShardPlan, digest: str) -> None:
        legacy = destination.with_name(f"{destination.name}.legacy")
        try:
            run_definition(
                replace(definition, seeds=shard.seeds),
                config,
                legacy,
                seed_access=recorder,
            )
            writer = RunArtifactWriter(destination, bundle_version=2)
            for artifact in frozen.artifacts:
                writer.copy_file(artifact.relative, legacy / artifact.relative)
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
                    "artifacts": _artifact_manifest(frozen),
                }
            )
        finally:
            shutil.rmtree(legacy, ignore_errors=True)

    report = resume_shards(root, plan, runner)
    return ShardRunReport(
        report.completed_shards,
        report.reused_shards,
        merge_shards(root, plan),
    )
