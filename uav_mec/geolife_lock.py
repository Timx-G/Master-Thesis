"""Build and verify participant-disjoint GeoLife trace locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .experiment_registry import load_registry, registered_family
from .geolife import (
    load_geolife_segments_snapshot,
    place_segment_in_area,
    resample_segment,
    sha256_file,
)

_SPLITS = ("train", "calibration", "final")
_TRACE_SEEDS = tuple(range(29_000, 29_030))
_SHA256_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class LockedTraceSegment:
    scenario_id: str
    participant_id: str
    split: str
    mode: str
    source_path: str
    source_start: str
    source_end: str
    slot_seconds: float
    translation_m: tuple[float, float]
    rotation_deg: float
    group_id: str


@dataclass(frozen=True, slots=True)
class TraceLock:
    schema_version: int
    dataset_sha256: str
    dataset_version: str
    source_url: str
    retrieved_on: str
    license_text: str
    license_sha256: str
    protocol_sha256: str
    registry_sha256: str
    seeds: tuple[int, ...]
    segments: tuple[LockedTraceSegment, ...]


@dataclass(frozen=True, slots=True)
class TraceLockSnapshot:
    """One verified immutable byte snapshot of a trace lock file."""

    lock: TraceLock
    document_json: bytes
    sha256: str


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _content_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(value))).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _regular_file(path: Path, name: str) -> Path:
    raw = Path(path)
    try:
        metadata = raw.lstat()
    except OSError as exc:
        raise ValueError(f"{name} must exist") from exc
    if raw.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} must resolve to a regular file") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} must resolve to a regular file")
    return resolved


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a table")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_float(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def _finite_float(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a string sequence")
    result = tuple(_string(item, field) for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{field} must be non-empty and unique")
    return result


def _utc_timestamp(value: object, field: str) -> str:
    text = _string(value, field)
    if not text.endswith("Z"):
        raise ValueError(f"{field} must be ISO-8601 UTC")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO-8601 UTC") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0.0:
        raise ValueError(f"{field} must be ISO-8601 UTC")
    return text


def _protocol(path: Path) -> tuple[dict[str, object], str]:
    protocol_path = _regular_file(path, "protocol")
    try:
        raw = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("protocol TOML is invalid") from exc
    if raw.get("schema_version") != 2:
        raise ValueError("protocol schema_version must be 2")
    return raw, sha256_file(protocol_path)


def _split_contract(raw: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    table = _mapping(raw.get("splits"), "splits")
    if set(table) != set(_SPLITS):
        raise ValueError("splits must declare exactly train, calibration, and final")
    result = {split: _string_sequence(table[split], f"splits.{split}") for split in _SPLITS}
    owner: dict[str, str] = {}
    for split, participants in result.items():
        for participant in participants:
            previous = owner.setdefault(participant, split)
            if previous != split:
                raise ValueError(
                    f"participant {participant} split overlap between {previous} and {split}"
                )
    return result


def _group_contract(
    raw: Mapping[str, object], final_participants: tuple[str, ...]
) -> tuple[dict[str, object], ...]:
    groups = raw.get("final_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("final_groups must be a non-empty table array")
    result: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_participants: set[str] = set()
    for index, value in enumerate(groups):
        table = _mapping(value, f"final_groups[{index}]")
        if set(table) != {"group_id", "participants"}:
            raise ValueError("final group fields must be group_id and participants")
        group_id = _string(table["group_id"], f"final_groups[{index}].group_id")
        participants = _string_sequence(
            table["participants"], f"final_groups[{index}].participants"
        )
        if group_id in seen_ids:
            raise ValueError("final group IDs must be unique")
        if any(participant not in final_participants for participant in participants):
            raise ValueError("final groups may contain only final participants")
        overlap = seen_participants.intersection(participants)
        if overlap:
            raise ValueError("final group participants must be disjoint")
        seen_ids.add(group_id)
        seen_participants.update(participants)
        result.append({"group_id": group_id, "participants": list(participants)})
    if seen_participants != set(final_participants):
        raise ValueError("final groups must cover every final participant exactly once")
    return tuple(result)


def _dataset_contract(raw: Mapping[str, object]) -> dict[str, str]:
    table = _mapping(raw.get("dataset"), "dataset")
    required = {
        "version",
        "source_url",
        "retrieved_on",
        "license_text",
        "license_sha256",
    }
    if set(table) != required:
        raise ValueError("dataset table fields do not match the protocol contract")
    result = {key: _string(table[key], f"dataset.{key}") for key in required}
    if not result["source_url"].startswith("https://"):
        raise ValueError("dataset.source_url must use HTTPS")
    try:
        date.fromisoformat(result["retrieved_on"])
    except ValueError as exc:
        raise ValueError("dataset.retrieved_on must be an ISO date") from exc
    actual_license = hashlib.sha256(result["license_text"].encode("utf-8")).hexdigest()
    if result["license_sha256"] != actual_license:
        raise ValueError("dataset license SHA-256 mismatch")
    return result


def _lock_document(
    archive_path: Path,
    protocol_path: Path,
    registry_path: Path,
) -> dict[str, object]:
    archive = _regular_file(archive_path, "archive")
    protocol, protocol_digest = _protocol(protocol_path)
    registry_file = _regular_file(registry_path, "registry")
    registry = load_registry(registry_file)
    try:
        trace_family = registered_family(registry, "trace_driven")
    except KeyError as exc:
        raise ValueError("registry has no trace_driven family") from exc
    if trace_family.seeds != _TRACE_SEEDS:
        raise ValueError("trace seed pool must be exactly 29000 through 29029")

    dataset = _dataset_contract(protocol)
    splits = _split_contract(protocol)
    groups = _group_contract(protocol, splits["final"])
    selection = _mapping(protocol.get("selection"), "selection")
    if set(selection) != {
        "accepted_modes",
        "slot_seconds",
        "minimum_segment_samples",
        "max_segments_per_participant_mode",
    }:
        raise ValueError("selection fields do not match the protocol contract")
    modes = _string_sequence(selection["accepted_modes"], "selection.accepted_modes")
    if modes != ("walk", "bike"):
        raise ValueError("accepted modes must be exactly walk and bike in that order")
    slot_seconds = _positive_float(selection["slot_seconds"], "selection.slot_seconds")
    minimum_samples = _positive_int(
        selection["minimum_segment_samples"], "selection.minimum_segment_samples"
    )
    if minimum_samples < 2:
        raise ValueError("minimum_segment_samples must be at least two")
    maximum_per_mode = _positive_int(
        selection["max_segments_per_participant_mode"],
        "selection.max_segments_per_participant_mode",
    )

    placement = _mapping(protocol.get("placement"), "placement")
    if set(placement) != {"area_size_m", "translation_m", "rotation_deg"}:
        raise ValueError("placement fields do not match the protocol contract")
    area_size_m = _positive_float(placement["area_size_m"], "placement.area_size_m")
    raw_translation = placement["translation_m"]
    if (
        not isinstance(raw_translation, Sequence)
        or isinstance(raw_translation, (str, bytes, bytearray))
        or len(raw_translation) != 2
    ):
        raise ValueError("placement.translation_m must contain two values")
    translation_m = tuple(
        _finite_float(value, "placement.translation_m") for value in raw_translation
    )
    rotation_deg = _finite_float(placement["rotation_deg"], "placement.rotation_deg")

    descriptor = os.open(archive, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("archive must remain a regular file")
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            descriptor = -1
            dataset_digest, source_segments = load_geolife_segments_snapshot(
                stream,
                modes=modes,
            )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    participant_split = {
        participant: split for split, participants in splits.items() for participant in participants
    }
    candidates: dict[tuple[str, str], list] = {}
    for segment in source_segments:
        if segment.participant_id in participant_split:
            candidates.setdefault((segment.participant_id, segment.mode), []).append(segment)

    locked: list[LockedTraceSegment] = []
    samples: dict[str, dict[str, object]] = {}
    selected_participants: set[str] = set()
    for split in _SPLITS:
        for participant in splits[split]:
            for mode in modes:
                ordered = sorted(
                    candidates.get((participant, mode), ()),
                    key=lambda segment: (
                        segment.source_start,
                        segment.source_end,
                        segment.source_path,
                    ),
                )
                accepted = 0
                for source in ordered:
                    try:
                        transformed = place_segment_in_area(
                            resample_segment(source, slot_seconds=slot_seconds),
                            area_size_m=area_size_m,
                            translation_m=translation_m,
                            rotation_deg=rotation_deg,
                        )
                    except ValueError:
                        continue
                    if len(transformed.timestamps_s) < minimum_samples:
                        continue
                    identity = "|".join(
                        (
                            split,
                            participant,
                            mode,
                            source.source_path,
                            source.source_start,
                            source.source_end,
                        )
                    )
                    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                    scenario_id = f"geolife-{split}-{participant}-{mode}-{suffix}"
                    if scenario_id in samples:
                        raise ValueError("scenario IDs must be unique")
                    locked.append(
                        LockedTraceSegment(
                            scenario_id=scenario_id,
                            participant_id=participant,
                            split=split,
                            mode=mode,
                            source_path=source.source_path,
                            source_start=source.source_start,
                            source_end=source.source_end,
                            slot_seconds=slot_seconds,
                            translation_m=(translation_m[0], translation_m[1]),
                            rotation_deg=rotation_deg,
                            group_id=f"participant:{participant}",
                        )
                    )
                    samples[scenario_id] = {
                        "timestamps_s": transformed.timestamps_s.tolist(),
                        "positions_enu_m": transformed.positions_enu_m.tolist(),
                    }
                    selected_participants.add(participant)
                    accepted += 1
                    if accepted == maximum_per_mode:
                        break
            if participant not in selected_participants:
                raise ValueError(
                    f"declared participant {participant} has no eligible source segment"
                )

    split_order = {name: index for index, name in enumerate(_SPLITS)}
    locked.sort(
        key=lambda segment: (
            split_order[segment.split],
            segment.participant_id,
            segment.source_start,
            segment.source_end,
            segment.source_path,
            segment.mode,
        )
    )
    samples = {segment.scenario_id: samples[segment.scenario_id] for segment in locked}
    document: dict[str, object] = {
        "schema_version": 2,
        "dataset_sha256": dataset_digest,
        "dataset_version": dataset["version"],
        "source_url": dataset["source_url"],
        "retrieved_on": dataset["retrieved_on"],
        "license_text": dataset["license_text"],
        "license_sha256": dataset["license_sha256"],
        "protocol_sha256": protocol_digest,
        "registry_sha256": sha256_file(registry_file),
        "seeds": list(_TRACE_SEEDS),
        "accepted_modes": list(modes),
        "area_size_m": area_size_m,
        "segments": [asdict(segment) for segment in locked],
        "samples": samples,
        "final_groups": list(groups),
    }
    document["content_sha256"] = _content_sha256(document)
    return document


def _trace_lock_from_document(document: Mapping[str, object]) -> TraceLock:
    stored_digest = document.get("content_sha256")
    if not _is_sha256(stored_digest):
        raise ValueError("trace lock content SHA-256 is invalid")
    unsigned = dict(document)
    del unsigned["content_sha256"]
    if _content_sha256(unsigned) != stored_digest:
        raise ValueError("trace lock content SHA-256 mismatch")
    if document.get("schema_version") != 2:
        raise ValueError("trace lock schema_version must be 2")
    for field in (
        "dataset_sha256",
        "protocol_sha256",
        "registry_sha256",
        "license_sha256",
    ):
        if not _is_sha256(document.get(field)):
            raise ValueError(f"trace lock {field} is invalid")
    license_text = _string(document.get("license_text"), "license_text")
    if hashlib.sha256(license_text.encode("utf-8")).hexdigest() != document["license_sha256"]:
        raise ValueError("trace lock license SHA-256 mismatch")
    source_url = _string(document.get("source_url"), "source_url")
    if not source_url.startswith("https://"):
        raise ValueError("trace lock source_url must use HTTPS")
    retrieved_on = _string(document.get("retrieved_on"), "retrieved_on")
    try:
        date.fromisoformat(retrieved_on)
    except ValueError as exc:
        raise ValueError("trace lock retrieved_on must be an ISO date") from exc
    raw_seeds = document.get("seeds")
    if not isinstance(raw_seeds, list) or tuple(raw_seeds) != _TRACE_SEEDS:
        raise ValueError("trace seed pool must be exactly 29000 through 29029")
    if document.get("accepted_modes") != ["walk", "bike"]:
        raise ValueError("trace lock accepted modes must be walk and bike")
    area_size_m = _positive_float(document.get("area_size_m"), "area_size_m")

    raw_segments = document.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("trace lock segments must be non-empty")
    segments: list[LockedTraceSegment] = []
    owner: dict[str, str] = {}
    scenario_ids: set[str] = set()
    for index, raw in enumerate(raw_segments):
        row = _mapping(raw, f"segments[{index}]")
        required = {
            "scenario_id",
            "participant_id",
            "split",
            "mode",
            "source_path",
            "source_start",
            "source_end",
            "slot_seconds",
            "translation_m",
            "rotation_deg",
            "group_id",
        }
        if set(row) != required:
            raise ValueError("locked segment fields do not match the contract")
        scenario_id = _string(row["scenario_id"], "scenario_id")
        participant = _string(row["participant_id"], "participant_id")
        split = _string(row["split"], "split")
        if split not in _SPLITS:
            raise ValueError("locked segment split is invalid")
        previous = owner.setdefault(participant, split)
        if previous != split:
            raise ValueError("participant appears across trace lock splits")
        if scenario_id in scenario_ids:
            raise ValueError("scenario IDs must be unique")
        scenario_ids.add(scenario_id)
        mode = _string(row["mode"], "mode")
        if mode not in {"walk", "bike"}:
            raise ValueError("locked segment mode is invalid")
        source_start = _utc_timestamp(row["source_start"], "source_start")
        source_end = _utc_timestamp(row["source_end"], "source_end")
        if source_end <= source_start:
            raise ValueError("locked segment source times are not monotonic")
        translation = row["translation_m"]
        if (
            not isinstance(translation, Sequence)
            or isinstance(translation, (str, bytes, bytearray))
            or len(translation) != 2
        ):
            raise ValueError("locked segment translation must contain two values")
        translation_m = tuple(_finite_float(value, "translation_m") for value in translation)
        group_id = _string(row["group_id"], "group_id")
        if group_id != f"participant:{participant}":
            raise ValueError("group_id must identify the source participant")
        segments.append(
            LockedTraceSegment(
                scenario_id=scenario_id,
                participant_id=participant,
                split=split,
                mode=mode,
                source_path=_string(row["source_path"], "source_path"),
                source_start=source_start,
                source_end=source_end,
                slot_seconds=_positive_float(row["slot_seconds"], "slot_seconds"),
                translation_m=(translation_m[0], translation_m[1]),
                rotation_deg=_finite_float(row["rotation_deg"], "rotation_deg"),
                group_id=group_id,
            )
        )

    split_order = {name: index for index, name in enumerate(_SPLITS)}
    expected_order = sorted(
        segments,
        key=lambda segment: (
            split_order[segment.split],
            segment.participant_id,
            segment.source_start,
            segment.source_end,
            segment.source_path,
            segment.mode,
        ),
    )
    if segments != expected_order:
        raise ValueError("trace lock segments are not in canonical order")

    raw_samples = _mapping(document.get("samples"), "samples")
    if set(raw_samples) != scenario_ids:
        raise ValueError("trace lock samples must cover every scenario exactly")
    for segment in segments:
        sample = _mapping(raw_samples[segment.scenario_id], "sample")
        if set(sample) != {"timestamps_s", "positions_enu_m"}:
            raise ValueError("trace sample fields do not match the contract")
        times = np.asarray(sample["timestamps_s"], dtype=float)
        positions = np.asarray(sample["positions_enu_m"], dtype=float)
        if (
            times.ndim != 1
            or positions.shape != (len(times), 2)
            or len(times) < 2
            or not np.isfinite(times).all()
            or not np.isfinite(positions).all()
            or np.any(np.diff(times) <= 0.0)
            or np.any(positions < 0.0)
            or np.any(positions > area_size_m)
        ):
            raise ValueError("trace sample positions or times are invalid")

    raw_groups = document.get("final_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("trace lock final_groups must be non-empty")
    final_participants = {
        segment.participant_id for segment in segments if segment.split == "final"
    }
    covered: set[str] = set()
    group_ids: set[str] = set()
    for raw_group in raw_groups:
        group = _mapping(raw_group, "final_group")
        if set(group) != {"group_id", "participants"}:
            raise ValueError("trace lock final group fields are invalid")
        group_id = _string(group["group_id"], "final_group.group_id")
        participants = _string_sequence(group["participants"], "final_group.participants")
        if group_id in group_ids or covered.intersection(participants):
            raise ValueError("trace lock final groups are not disjoint")
        if any(participant not in final_participants for participant in participants):
            raise ValueError("trace lock final group contains a non-final participant")
        group_ids.add(group_id)
        covered.update(participants)
    if covered != final_participants:
        raise ValueError("trace lock final groups do not cover final participants")

    return TraceLock(
        schema_version=2,
        dataset_sha256=str(document["dataset_sha256"]),
        dataset_version=_string(document.get("dataset_version"), "dataset_version"),
        source_url=source_url,
        retrieved_on=retrieved_on,
        license_text=license_text,
        license_sha256=str(document["license_sha256"]),
        protocol_sha256=str(document["protocol_sha256"]),
        registry_sha256=str(document["registry_sha256"]),
        seeds=_TRACE_SEEDS,
        segments=tuple(segments),
    )


def build_trace_lock(
    archive_path: Path,
    protocol_path: Path,
    registry_path: Path,
    output_path: Path,
) -> TraceLock:
    """Validate all inputs and atomically publish one deterministic trace lock."""

    document = _lock_document(archive_path, protocol_path, registry_path)
    result = _trace_lock_from_document(document)
    payload = _canonical_bytes(document)
    output = Path(output_path)
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError(output)
        if output.read_bytes() == payload:
            return load_and_verify_trace_lock(output)
        raise FileExistsError("refusing to overwrite a non-identical trace lock")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return result


def load_and_verify_trace_lock(path: Path) -> TraceLock:
    """Load a trace lock and verify its complete internal digest closure."""

    return load_trace_lock_snapshot(path).lock


def load_trace_lock_snapshot(path: Path) -> TraceLockSnapshot:
    """Read and verify a lock exactly once, closing replacement races."""

    lock_path = _regular_file(path, "trace lock")
    try:
        payload = lock_path.read_bytes()
        document = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("trace lock JSON is invalid") from exc
    if not isinstance(document, Mapping):
        raise ValueError("trace lock must contain a JSON object")
    return TraceLockSnapshot(
        lock=_trace_lock_from_document(document),
        document_json=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def trace_lock_sha256(path: Path) -> str:
    """Return the byte digest after verifying the lock."""

    return load_trace_lock_snapshot(path).sha256


def verify_trace_lock_archive(lock_path: Path, archive_path: Path) -> TraceLock:
    """Verify internal closure and bind it again to supplied dataset bytes."""

    archive = _regular_file(archive_path, "archive")
    lock = load_trace_lock_snapshot(lock_path).lock
    if sha256_file(archive) != lock.dataset_sha256:
        raise ValueError("GeoLife archive SHA-256 does not match the trace lock")
    return lock


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--protocol", type=Path, required=True)
    build.add_argument("--registry", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--archive", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            build_trace_lock(args.archive, args.protocol, args.registry, args.output)
        else:
            verify_trace_lock_archive(args.lock, args.archive)
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
