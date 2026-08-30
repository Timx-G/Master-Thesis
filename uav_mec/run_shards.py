"""Resumable seed-sharded execution for confirmatory experiment definitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .analyze_results import verify_checksums
from .artifacts import RunArtifactWriter, sha256_file
from .experiment import (
    ExperimentDefinition,
    _source_snapshot,
    definition_to_dict,
    load_definition,
    run_definition,
)


@dataclass(frozen=True, slots=True)
class ShardRunReport:
    completed_shards: int
    reused_shards: int
    merged_path: Path


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seed_groups(seeds: tuple[int, ...], shard_size: int) -> tuple[tuple[int, ...], ...]:
    if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    return tuple(
        tuple(seeds[offset : offset + shard_size])
        for offset in range(0, len(seeds), shard_size)
    )


def _shard_name(seeds: tuple[int, ...]) -> str:
    return f"seed-{seeds[0]:010d}-{seeds[-1]:010d}"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _frozen_index(
    definition: ExperimentDefinition,
    config_sha256: str,
    groups: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    return {
        "run_id": definition.run_id,
        "config_sha256": config_sha256,
        "seeds": list(definition.seeds),
        "methods": list(definition.methods),
        "shards": [
            {"name": _shard_name(seeds), "seeds": list(seeds)} for seeds in groups
        ],
    }


def _prepare_root(output_dir: Path, expected_index: dict[str, object]) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "shard_index.json"
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing != expected_index:
            raise ValueError("existing shard root uses a different configuration")
        return
    if any(root.iterdir()):
        raise ValueError("non-empty shard root is missing its frozen configuration index")
    _atomic_json(index_path, expected_index)


def _validate_shard(
    path: Path,
    *,
    config_sha256: str,
    seeds: tuple[int, ...],
    source_sha256: str,
) -> None:
    verify_checksums(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != config_sha256:
        raise ValueError(f"shard {path.name} configuration hash mismatch")
    if manifest.get("seeds") != list(seeds):
        raise ValueError(f"shard {path.name} seed range mismatch")
    snapshot = manifest.get("source_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("sha256") != source_sha256:
        raise ValueError(f"shard {path.name} source snapshot mismatch")


def _copy_csv_rows(
    writer: RunArtifactWriter,
    relative: str,
    shard_paths: tuple[Path, ...],
) -> None:
    first_header: tuple[str, ...] | None = None
    readers: list[tuple[Path, tuple[str, ...]]] = []
    for shard in shard_paths:
        with (shard / relative).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = tuple(next(reader))
        if first_header is None:
            first_header = header
        elif header != first_header:
            raise ValueError(f"inconsistent {relative} schema across shards")
        readers.append((shard, header))
    if first_header is None:
        raise ValueError("at least one shard is required")
    with writer.open_csv(relative, first_header) as write_row:
        for shard, _header in readers:
            with (shard / relative).open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    write_row(row)


def _copy_jsonl_rows(
    writer: RunArtifactWriter,
    relative: str,
    shard_paths: tuple[Path, ...],
) -> None:
    with writer.open_jsonl(relative) as write_row:
        for shard in shard_paths:
            with (shard / relative).open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        write_row(json.loads(line))


def _merged_summary(
    definition: ExperimentDefinition,
    shard_paths: tuple[Path, ...],
) -> dict[str, object]:
    method_values = {method: [] for method in definition.methods}
    initializer_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    for shard in shard_paths:
        summary = json.loads((shard / "summary.json").read_text(encoding="utf-8"))
        initializer_counts.update(summary["initializer_status_counts"])
        termination_counts.update(summary["search_termination_counts"])
        for method in definition.methods:
            method_values[method].extend(summary["methods"][method]["on_time_rate_by_seed"])
    return {
        "run_id": definition.run_id,
        "seeds": list(definition.seeds),
        "initializer_status_counts": dict(sorted(initializer_counts.items())),
        "search_termination_counts": dict(sorted(termination_counts.items())),
        "methods": {
            method: {
                "on_time_rate_by_seed": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
            for method, values in method_values.items()
        },
    }


def _merge_shards(
    definition: ExperimentDefinition,
    config_path: Path,
    output_dir: Path,
    shard_paths: tuple[Path, ...],
    source_sha256: str,
) -> Path:
    merged = output_dir / "merged"
    shard_checksums = {
        shard.name: sha256_file(shard / "checksums.sha256") for shard in shard_paths
    }
    config_sha256 = _config_sha256(config_path)
    if merged.exists():
        verify_checksums(merged)
        manifest = json.loads((merged / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("config_sha256") != config_sha256
            or manifest.get("source_snapshot_sha256") != source_sha256
            or manifest.get("shard_checksums") != shard_checksums
        ):
            raise ValueError("existing merged bundle does not match current shards")
        return merged

    temporary = output_dir / f".merged.{uuid.uuid4().hex}.tmp"
    writer = RunArtifactWriter(temporary)
    try:
        _copy_csv_rows(writer, "tasks.csv", shard_paths)
        _copy_csv_rows(writer, "slots.csv", shard_paths)
        _copy_jsonl_rows(writer, "events.jsonl", shard_paths)
        writer.write_json("summary.json", _merged_summary(definition, shard_paths))
        writer.finalize(
            {
                "run_id": definition.run_id,
                "config_path": str(config_path),
                "config_sha256": config_sha256,
                "source_snapshot_sha256": source_sha256,
                "seeds": list(definition.seeds),
                "resolved_definition": definition_to_dict(definition),
                "shard_checksums": shard_checksums,
            }
        )
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
    """Run, resume, and deterministically merge immutable seed shards."""

    config = Path(config_path)
    output = Path(output_dir)
    definition = load_definition(config)
    groups = _seed_groups(definition.seeds, shard_size)
    config_sha256 = _config_sha256(config)
    source_sha256 = str(_source_snapshot()["sha256"])
    _prepare_root(output, _frozen_index(definition, config_sha256, groups))
    shard_root = output / "shards"
    shard_root.mkdir(exist_ok=True)

    completed = 0
    reused = 0
    shard_paths: list[Path] = []
    for seeds in groups:
        destination = shard_root / _shard_name(seeds)
        if destination.exists():
            _validate_shard(
                destination,
                config_sha256=config_sha256,
                seeds=seeds,
                source_sha256=source_sha256,
            )
            reused += 1
        else:
            temporary = shard_root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
            try:
                run_definition(replace(definition, seeds=seeds), config, temporary)
                _validate_shard(
                    temporary,
                    config_sha256=config_sha256,
                    seeds=seeds,
                    source_sha256=source_sha256,
                )
                os.replace(temporary, destination)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            completed += 1
        shard_paths.append(destination)

    merged = _merge_shards(
        definition,
        config,
        output,
        tuple(shard_paths),
        source_sha256,
    )
    return ShardRunReport(completed, reused, merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable seed-sharded experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1)
    args = parser.parse_args()
    report = run_sharded_experiment(
        args.config,
        args.output,
        shard_size=args.shard_size,
    )
    print(
        f"completed={report.completed_shards} reused={report.reused_shards} "
        f"merged={report.merged_path}"
    )


if __name__ == "__main__":
    main()
