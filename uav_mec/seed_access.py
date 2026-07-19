"""Append-only, hash-chained seed-access recording and phase audits.

The ledger is deliberately independent of the experiment implementation.  A
caller must first obtain a permit from the verified registry or confirmatory
lock and must then pass a recorder into every operation that opens seed-owned
data.  Ledger lines are canonical JSON and are serialized by an OS file lock.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path


_ZERO_DIGEST = "0" * 64
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_UTC_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z"
)
_PHASES = (
    "before-final-pilot",
    "before-formal-lock",
    "before-formal-run",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PREREGISTRATION_ROOT = (
    _REPOSITORY_ROOT / "artifacts" / "preregistration" / "v2"
)
_RECORD_KEYS = frozenset(
    {
        "sequence",
        "timestamp_utc",
        "operation",
        "purpose",
        "family",
        "case_id",
        "seeds",
        "registry_sha256",
        "lock_sha256",
        "metadata",
        "previous_digest",
        "digest",
    }
)


class SeedPurpose(StrEnum):
    DEVELOPMENT = "development"
    PILOT = "pilot"
    FORMAL = "formal"


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _strict_int(value: object) -> bool:
    return type(value) is int


def _validated_seed_tuple(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence of integers")
    seeds = tuple(value)
    if any(not _strict_int(seed) for seed in seeds):
        raise TypeError(f"{field} must contain only integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{field} contains duplicate seeds")
    if seeds != tuple(sorted(seeds)):
        raise ValueError(f"{field} must be sorted")
    return seeds


@dataclass(frozen=True, slots=True)
class SeedPermit:
    purpose: SeedPurpose
    allowed_seeds: tuple[int, ...]
    family: str | None = None
    case_id: str | None = None
    registry_sha256: str = ""
    lock_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, SeedPurpose):
            raise TypeError("purpose must be a SeedPurpose")
        object.__setattr__(
            self,
            "allowed_seeds",
            _validated_seed_tuple(self.allowed_seeds, field="allowed_seeds"),
        )
        if not _is_digest(self.registry_sha256):
            raise ValueError("registry_sha256 must be a lowercase SHA-256 digest")
        if self.purpose is SeedPurpose.FORMAL:
            if not isinstance(self.family, str) or not self.family.strip():
                raise ValueError("formal permit requires family")
            if not isinstance(self.case_id, str) or not self.case_id.strip():
                raise ValueError("formal permit requires case_id")
            if not _is_digest(self.lock_sha256):
                raise ValueError("formal permit requires lock_sha256")
        elif any(
            value is not None
            for value in (self.family, self.case_id, self.lock_sha256)
        ):
            raise ValueError(
                "development and pilot permits cannot bind family, case_id, or lock"
            )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _validate_json_value(value: object, *, field: str) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, field=field)
        return
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{field} object keys must be strings")
        for item in value.values():
            _validate_json_value(item, field=field)
        return
    raise TypeError(f"{field} contains a non-JSON value of type {type(value).__name__}")


def _validate_operation(operation: object) -> str:
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string")
    return operation


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _validate_timestamp(value: object) -> None:
    if not isinstance(value, str) or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise ValueError("timestamp_utc must be canonical UTC")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError("timestamp_utc must be canonical UTC") from exc
    if parsed.isoformat(timespec="microseconds").replace("+00:00", "Z") != value:
        raise ValueError("timestamp_utc must be canonical UTC")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_record(
    record: object,
    *,
    line_number: int,
    expected_sequence: int,
    expected_previous: str,
) -> dict[str, object]:
    prefix = f"line {line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{prefix}: record must be a JSON object")
    actual_keys = frozenset(record)
    if actual_keys != _RECORD_KEYS:
        missing = sorted(_RECORD_KEYS - actual_keys)
        extra = sorted(actual_keys - _RECORD_KEYS)
        raise ValueError(
            f"{prefix}: invalid schema (missing={missing}, extra={extra})"
        )
    sequence = record["sequence"]
    if not _strict_int(sequence) or sequence != expected_sequence:
        raise ValueError(f"{prefix}: invalid sequence")
    try:
        _validate_timestamp(record["timestamp_utc"])
        _validate_operation(record["operation"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}: {exc}") from exc
    purpose_value = record["purpose"]
    if type(purpose_value) is not str:
        raise ValueError(f"{prefix}: purpose must be a string")
    try:
        purpose = SeedPurpose(purpose_value)
    except ValueError as exc:
        raise ValueError(f"{prefix}: unknown purpose") from exc
    family = record["family"]
    case_id = record["case_id"]
    lock_sha256 = record["lock_sha256"]
    if purpose is SeedPurpose.FORMAL:
        if not isinstance(family, str) or not family.strip():
            raise ValueError(f"{prefix}: formal record requires family")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{prefix}: formal record requires case_id")
        if not _is_digest(lock_sha256):
            raise ValueError(f"{prefix}: formal record requires lock_sha256")
    elif any(value is not None for value in (family, case_id, lock_sha256)):
        raise ValueError(f"{prefix}: non-formal record contains formal binding")
    if not _is_digest(record["registry_sha256"]):
        raise ValueError(f"{prefix}: invalid registry_sha256")
    seeds_value = record["seeds"]
    if not isinstance(seeds_value, list):
        raise ValueError(f"{prefix}: seeds must be a JSON array")
    try:
        _validated_seed_tuple(seeds_value, field="seeds")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}: {exc}") from exc
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"{prefix}: metadata must be a JSON object")
    try:
        _validate_json_value(metadata, field="metadata")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}: {exc}") from exc
    if record["previous_digest"] != expected_previous:
        raise ValueError(f"{prefix}: broken previous-record digest")
    if not _is_digest(record["digest"]):
        raise ValueError(f"{prefix}: invalid record digest")
    body = {key: value for key, value in record.items() if key != "digest"}
    expected_digest = _sha256_bytes(_canonical_json(body).encode("utf-8"))
    if record["digest"] != expected_digest:
        raise ValueError(f"{prefix}: invalid record digest")
    return record


def _verify_ledger_bytes(data: bytes) -> tuple[dict[str, object], ...]:
    if not data:
        return ()
    if not data.endswith(b"\n"):
        line_number = data.count(b"\n") + 1
        raise ValueError(f"line {line_number}: ledger must end with a newline")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("line 1: ledger is not UTF-8") from exc
    records: list[dict[str, object]] = []
    previous = _ZERO_DIGEST
    for line_number, line in enumerate(text[:-1].split("\n"), start=1):
        if not line:
            raise ValueError(f"line {line_number}: blank ledger line")
        try:
            parsed = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"line {line_number}: malformed JSON") from exc
        record = _validate_record(
            parsed,
            line_number=line_number,
            expected_sequence=line_number,
            expected_previous=previous,
        )
        canonical_line = _canonical_json(record)
        if line != canonical_line:
            raise ValueError(f"line {line_number}: JSON is not canonical")
        records.append(record)
        previous = str(record["digest"])
    return tuple(records)


def _open_regular_file(path: Path, flags: int, mode: int = 0o600) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags | nofollow, mode)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"seed ledger is not a regular file: {path}")
    return descriptor


def _read_ledger_snapshot(
    path: Path,
    *,
    allow_missing: bool,
) -> tuple[bytes, tuple[dict[str, object], ...]]:
    ledger_path = Path(path)
    try:
        descriptor = _open_regular_file(ledger_path, os.O_RDONLY)
    except FileNotFoundError:
        if allow_missing:
            return b"", ()
        raise
    with os.fdopen(descriptor, "rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            data = handle.read()
            return data, _verify_ledger_bytes(data)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(slots=True)
class SeedAccessRecorder:
    ledger_path: Path
    permit: SeedPermit

    def __post_init__(self) -> None:
        self.ledger_path = Path(self.ledger_path)
        if not isinstance(self.permit, SeedPermit):
            raise TypeError("permit must be a SeedPermit")

    def record(
        self,
        operation: str,
        seeds: Sequence[int],
        metadata: Mapping[str, object],
    ) -> None:
        operation_value = _validate_operation(operation)
        if not isinstance(seeds, Sequence) or isinstance(
            seeds, (str, bytes, bytearray)
        ):
            raise TypeError("seeds must be a sequence of integers")
        supplied_seeds = tuple(seeds)
        if any(not _strict_int(seed) for seed in supplied_seeds):
            raise TypeError("seeds must contain only integers")
        if len(supplied_seeds) != len(set(supplied_seeds)):
            raise ValueError("seeds contains duplicate seeds")
        seed_values = tuple(sorted(supplied_seeds))
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        metadata_value = dict(metadata)
        _validate_json_value(metadata_value, field="metadata")
        bad_seeds = sorted(set(seed_values) - set(self.permit.allowed_seeds))
        if bad_seeds:
            raise ValueError(f"seed outside permit: {bad_seeds[0]}")

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = _open_regular_file(
            self.ledger_path,
            os.O_RDWR | os.O_CREAT,
        )
        with os.fdopen(descriptor, "r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                records = _verify_ledger_bytes(handle.read())
                previous = (
                    str(records[-1]["digest"]) if records else _ZERO_DIGEST
                )
                body: dict[str, object] = {
                    "sequence": len(records) + 1,
                    "timestamp_utc": _utc_timestamp(),
                    "operation": operation_value,
                    "purpose": self.permit.purpose.value,
                    "family": self.permit.family,
                    "case_id": self.permit.case_id,
                    "seeds": list(seed_values),
                    "registry_sha256": self.permit.registry_sha256,
                    "lock_sha256": self.permit.lock_sha256,
                    "metadata": metadata_value,
                    "previous_digest": previous,
                }
                digest = _sha256_bytes(_canonical_json(body).encode("utf-8"))
                record = {**body, "digest": digest}
                encoded = (_canonical_json(record) + "\n").encode("utf-8")
                handle.seek(0, os.SEEK_END)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_registry(path: Path) -> object:
    try:
        from .experiment_registry import load_registry
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("experiment registry verifier is unavailable") from exc
    return load_registry(Path(path))


def _load_confirmatory_lock(path: Path) -> object:
    try:
        from .confirmatory_lock import load_and_verify_confirmatory_lock
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("confirmatory lock verifier is unavailable") from exc
    return load_and_verify_confirmatory_lock(Path(path))


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"verified object is missing {name}")
        return value[name]
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise ValueError(f"verified object is missing {name}") from exc


def _sequence_field(value: object, name: str) -> tuple[object, ...]:
    field = _field(value, name)
    if not isinstance(field, Sequence) or isinstance(
        field, (str, bytes, bytearray)
    ):
        raise ValueError(f"verified object field {name} must be a sequence")
    return tuple(field)


def development_permit(registry_path: Path) -> SeedPermit:
    path = Path(registry_path)
    registry = _load_registry(path)
    seeds = _validated_seed_tuple(
        _sequence_field(registry, "development_seeds"),
        field="development_seeds",
    )
    return SeedPermit(
        purpose=SeedPurpose.DEVELOPMENT,
        allowed_seeds=seeds,
        registry_sha256=_sha256_file(path),
    )


def pilot_permit(registry_path: Path) -> SeedPermit:
    path = Path(registry_path)
    registry = _load_registry(path)
    seeds = _validated_seed_tuple(
        _sequence_field(registry, "pilot_seeds"),
        field="pilot_seeds",
    )
    return SeedPermit(
        purpose=SeedPurpose.PILOT,
        allowed_seeds=seeds,
        registry_sha256=_sha256_file(path),
    )


def _locked_case_map(lock: object) -> dict[tuple[str, str], tuple[int, ...]]:
    result: dict[tuple[str, str], tuple[int, ...]] = {}
    for family in _sequence_field(lock, "families"):
        family_name = _field(family, "family")
        if not isinstance(family_name, str) or not family_name:
            raise ValueError("verified lock contains an invalid family")
        for case in _sequence_field(family, "cases"):
            case_id = _field(case, "case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("verified lock contains an invalid case_id")
            key = (family_name, case_id)
            if key in result:
                raise ValueError(f"verified lock duplicates {family_name}/{case_id}")
            result[key] = _validated_seed_tuple(
                _sequence_field(case, "seeds"),
                field=f"locked seeds for {family_name}/{case_id}",
            )
    return result


def _validated_lock(
    lock_path: Path,
) -> tuple[object, str, dict[tuple[str, str], tuple[int, ...]]]:
    path = Path(lock_path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("confirmatory lock path must be a verified directory")
    lock_json = path / "confirmatory_lock.json"
    if lock_json.is_symlink() or not lock_json.is_file():
        raise ValueError("confirmatory lock directory lacks confirmatory_lock.json")
    digest_before_validation = _sha256_file(lock_json)
    lock = _load_confirmatory_lock(path)
    lock_sha256 = _sha256_file(lock_json)
    if lock_sha256 != digest_before_validation:
        raise ValueError("confirmatory lock changed while it was being verified")
    return lock, lock_sha256, _locked_case_map(lock)


def formal_permit(
    lock_path: Path,
    family_name: str,
    case_id: str,
) -> SeedPermit:
    lock, lock_sha256, cases = _validated_lock(Path(lock_path))
    key = (family_name, case_id)
    if key not in cases:
        raise KeyError(f"{family_name}/{case_id}")
    registry_sha256 = _field(lock, "registry_sha256")
    if not _is_digest(registry_sha256):
        raise ValueError("verified lock contains an invalid registry_sha256")
    return SeedPermit(
        purpose=SeedPurpose.FORMAL,
        allowed_seeds=cases[key],
        family=family_name,
        case_id=case_id,
        registry_sha256=str(registry_sha256),
        lock_sha256=lock_sha256,
    )


def verify_seed_ledger(path: Path) -> tuple[dict[str, object], ...]:
    """Return a verified immutable ledger snapshot."""

    _, records = _read_ledger_snapshot(Path(path), allow_missing=False)
    return records


def _registry_pool_contract(
    registry: object,
) -> tuple[
    set[int],
    set[int],
    dict[str, set[int]],
    dict[str, set[str]],
]:
    development = set(
        _validated_seed_tuple(
            _sequence_field(registry, "development_seeds"),
            field="development_seeds",
        )
    )
    pilot = set(
        _validated_seed_tuple(
            _sequence_field(registry, "pilot_seeds"),
            field="pilot_seeds",
        )
    )
    formal: dict[str, set[int]] = {}
    cases: dict[str, set[str]] = {}
    for family in _sequence_field(registry, "families"):
        name = _field(family, "name")
        start = _field(family, "seed_start")
        count = _field(family, "seed_count")
        if not isinstance(name, str) or not name or name in formal:
            raise ValueError("registry contains an invalid or duplicate family")
        if not _strict_int(start) or not _strict_int(count) or count <= 0:
            raise ValueError(f"registry family {name} has an invalid seed interval")
        formal[name] = set(range(start, start + count))
        case_ids: set[str] = set()
        for case in _sequence_field(family, "cases"):
            case_id = _field(case, "case_id")
            if not isinstance(case_id, str) or not case_id or case_id in case_ids:
                raise ValueError(f"registry family {name} has an invalid case_id")
            case_ids.add(case_id)
        cases[name] = case_ids
    pools = [development, pilot, *formal.values()]
    for index, left in enumerate(pools):
        for right in pools[index + 1 :]:
            if left.intersection(right):
                raise ValueError("registry seed pools are not pairwise disjoint")
    return development, pilot, formal, cases


def audit_candidate_pool_unseen(
    registry: object,
    ledger_records: Sequence[Mapping[str, object]],
) -> None:
    development, pilot, formal, cases = _registry_pool_contract(registry)
    del development, pilot
    formal_union = set().union(*formal.values()) if formal else set()
    for index, record in enumerate(ledger_records, start=1):
        purpose = record.get("purpose")
        seeds_value = record.get("seeds")
        if not isinstance(seeds_value, Sequence) or isinstance(
            seeds_value, (str, bytes, bytearray)
        ):
            raise ValueError(f"record {index} has invalid seeds")
        seeds = set(seeds_value)
        if purpose in (SeedPurpose.DEVELOPMENT.value, SeedPurpose.PILOT.value):
            opened = sorted(seeds.intersection(formal_union))
            if opened:
                raise ValueError(
                    f"formal candidate seed accessed before lock: {opened[0]}"
                )
        elif purpose == SeedPurpose.FORMAL.value:
            family = record.get("family")
            case_id = record.get("case_id")
            if not isinstance(family, str) or family not in formal:
                raise ValueError(f"formal record {index} has an unknown family")
            if not isinstance(case_id, str) or case_id not in cases[family]:
                raise ValueError(f"formal record {index} has an unknown case")
            if not _is_digest(record.get("lock_sha256")):
                raise ValueError(f"formal record {index} has no valid lock digest")
            outside = sorted(seeds - formal[family])
            if outside:
                raise ValueError(
                    f"formal record {index} contains seed outside {family} pool"
                )
        else:
            raise ValueError(f"record {index} has an unknown purpose")


def _purpose_and_pool_validation(
    records: Sequence[Mapping[str, object]],
    development: set[int],
    pilot: set[int],
) -> None:
    for index, record in enumerate(records, start=1):
        seeds = set(record["seeds"])
        purpose = record["purpose"]
        if purpose == SeedPurpose.DEVELOPMENT.value and not seeds.issubset(development):
            raise ValueError(f"development record {index} contains seed outside development pool")
        if purpose == SeedPurpose.PILOT.value and not seeds.issubset(pilot):
            raise ValueError(f"pilot record {index} contains seed outside pilot pool")


def _pool_for_seed(
    seed: int,
    development: set[int],
    pilot: set[int],
    formal: Mapping[str, set[int]],
) -> str:
    if seed in development:
        return "development"
    if seed in pilot:
        return "pilot"
    for family, pool in formal.items():
        if seed in pool:
            return f"formal/{family}"
    return "unregistered"


def _count_summary(
    records: Sequence[Mapping[str, object]],
    development: set[int],
    pilot: set[int],
    formal: Mapping[str, set[int]],
    cases: Mapping[str, set[str]],
) -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    purpose_keys = tuple(purpose.value for purpose in SeedPurpose)
    family_keys = tuple(formal)
    case_keys = tuple(
        f"{family}/{case_id}"
        for family in family_keys
        for case_id in sorted(cases[family])
    )
    pool_keys = ("development", "pilot", *(f"formal/{name}" for name in family_keys))

    def empty(keys: Sequence[str]) -> dict[str, dict[str, object]]:
        return {
            key: {"records": 0, "seed_accesses": 0, "unique_seeds": set()}
            for key in keys
        }

    by_purpose = empty(purpose_keys)
    by_family = empty(family_keys)
    by_case = empty(case_keys)
    by_pool = empty(pool_keys)
    accessed_by_pool = {key: set() for key in pool_keys}
    for record in records:
        purpose = str(record["purpose"])
        seeds = tuple(record["seeds"])
        purpose_row = by_purpose[purpose]
        purpose_row["records"] = int(purpose_row["records"]) + 1
        purpose_row["seed_accesses"] = int(purpose_row["seed_accesses"]) + len(seeds)
        purpose_row["unique_seeds"].update(seeds)  # type: ignore[union-attr]
        family = record["family"]
        case_id = record["case_id"]
        if isinstance(family, str):
            family_row = by_family[family]
            family_row["records"] = int(family_row["records"]) + 1
            family_row["seed_accesses"] = int(family_row["seed_accesses"]) + len(seeds)
            family_row["unique_seeds"].update(seeds)  # type: ignore[union-attr]
            case_key = f"{family}/{case_id}"
            case_row = by_case[case_key]
            case_row["records"] = int(case_row["records"]) + 1
            case_row["seed_accesses"] = int(case_row["seed_accesses"]) + len(seeds)
            case_row["unique_seeds"].update(seeds)  # type: ignore[union-attr]
        touched_pools: set[str] = set()
        for seed in seeds:
            pool = _pool_for_seed(seed, development, pilot, formal)
            if pool == "unregistered":
                raise ValueError(f"ledger contains unregistered seed {seed}")
            touched_pools.add(pool)
            accessed_by_pool[pool].add(seed)
            pool_row = by_pool[pool]
            pool_row["seed_accesses"] = int(pool_row["seed_accesses"]) + 1
            pool_row["unique_seeds"].add(seed)  # type: ignore[union-attr]
        for pool in touched_pools:
            by_pool[pool]["records"] = int(by_pool[pool]["records"]) + 1

    def finalize(rows: dict[str, dict[str, object]]) -> dict[str, dict[str, int]]:
        return {
            key: {
                "records": int(row["records"]),
                "seed_accesses": int(row["seed_accesses"]),
                "unique_seeds": len(row["unique_seeds"]),  # type: ignore[arg-type]
            }
            for key, row in rows.items()
        }

    pool_sizes = {
        "development": len(development),
        "pilot": len(pilot),
        **{f"formal/{name}": len(pool) for name, pool in formal.items()},
    }
    pool_counts = {
        key: {
            "size": size,
            "accessed": len(accessed_by_pool[key]),
            "unseen": size - len(accessed_by_pool[key]),
        }
        for key, size in pool_sizes.items()
    }
    return (
        {
            "by_purpose": finalize(by_purpose),
            "by_family": finalize(by_family),
            "by_case": finalize(by_case),
            "by_pool": finalize(by_pool),
        },
        pool_counts,
    )


def _strict_nonnegative_int(value: object, field: str) -> int:
    if not _strict_int(value) or value < 0:
        raise ValueError(f"verified lock contains invalid {field}")
    return value


def _strict_digest_field(lock: object, field: str) -> str:
    value = _field(lock, field)
    if not _is_digest(value):
        raise ValueError(f"verified lock contains invalid {field}")
    return str(value)


def _verify_formal_boundary(
    *,
    lock_path: Path,
    lock: object,
    lock_sha256: str,
    locked_cases: Mapping[tuple[str, str], tuple[int, ...]],
    registry_sha256: str,
    live_bytes: bytes,
    records: tuple[dict[str, object], ...],
    registry: object,
) -> tuple[int, int]:
    locked_registry_sha256 = _strict_digest_field(lock, "registry_sha256")
    if locked_registry_sha256 != registry_sha256:
        raise ValueError("confirmatory lock registry digest mismatch")
    prefix_bytes = _strict_nonnegative_int(
        _field(lock, "seed_ledger_prefix_bytes"),
        "seed_ledger_prefix_bytes",
    )
    prefix_sha256 = _strict_digest_field(lock, "seed_ledger_prefix_sha256")
    prefix_head = _strict_digest_field(lock, "seed_ledger_head_sha256")
    copied_path = lock_path / "seed-access-prelock.jsonl"
    if copied_path.is_symlink() or not copied_path.is_file():
        raise ValueError("confirmatory lock lacks the copied prelock ledger")
    copied_bytes = copied_path.read_bytes()
    if len(copied_bytes) != prefix_bytes:
        raise ValueError("copied prelock ledger byte length mismatch")
    if _sha256_bytes(copied_bytes) != prefix_sha256:
        raise ValueError("copied prelock ledger digest mismatch")
    prefix_records = _verify_ledger_bytes(copied_bytes)
    actual_head = (
        str(prefix_records[-1]["digest"]) if prefix_records else _ZERO_DIGEST
    )
    if actual_head != prefix_head:
        raise ValueError("copied prelock ledger head mismatch")
    if len(live_bytes) < prefix_bytes or live_bytes[:prefix_bytes] != copied_bytes:
        raise ValueError("live ledger does not begin with the exact locked prefix")
    if tuple(records[: len(prefix_records)]) != prefix_records:
        raise ValueError("live ledger record prefix mismatch")
    audit_candidate_pool_unseen(registry, prefix_records)
    development, pilot, _, _ = _registry_pool_contract(registry)
    _purpose_and_pool_validation(prefix_records, development, pilot)
    if any(record["purpose"] == SeedPurpose.FORMAL.value for record in prefix_records):
        raise ValueError("formal access exists inside the prelock ledger prefix")

    post_records = records[len(prefix_records) :]
    for offset, record in enumerate(post_records, start=len(prefix_records) + 1):
        if record["purpose"] != SeedPurpose.FORMAL.value:
            raise ValueError(f"post-prefix record {offset} is not formal")
        family = record["family"]
        case_id = record["case_id"]
        if not isinstance(family, str) or not isinstance(case_id, str):
            raise ValueError(f"post-prefix record {offset} lacks family/case")
        key = (family, case_id)
        if key not in locked_cases:
            raise ValueError(f"post-prefix record {offset} has undeclared family/case")
        if record["lock_sha256"] != lock_sha256:
            raise ValueError(f"post-prefix record {offset} has wrong lock digest")
        if record["registry_sha256"] != registry_sha256:
            raise ValueError(f"post-prefix record {offset} has wrong registry digest")
        outside = sorted(set(record["seeds"]) - set(locked_cases[key]))
        if outside:
            raise ValueError(
                f"post-prefix record {offset} contains seed outside exact permit"
            )
    return len(prefix_records), len(post_records)


def _absolute_lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path))))


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _reject_symlinked_ancestors(path: Path) -> None:
    current = _absolute_lexical_path(path).parent
    while True:
        status = _lstat(current)
        if status is not None:
            if stat.S_ISLNK(status.st_mode):
                raise ValueError(f"audit output path contains a symlink: {current}")
            if not stat.S_ISDIR(status.st_mode):
                raise ValueError(
                    f"audit output ancestor is not a directory: {current}"
                )
        if current == current.parent:
            break
        current = current.parent


def _preflight_new_output(path: Path) -> Path:
    destination = _absolute_lexical_path(path)
    _reject_symlinked_ancestors(destination)
    status = _lstat(destination)
    if status is not None:
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"audit output path is a symlink: {destination}")
        raise FileExistsError(f"audit output already exists: {destination}")
    return destination


def _resolve_cli_audit_output(path: Path) -> Path:
    root = _absolute_lexical_path(_PREREGISTRATION_ROOT)
    destination = _absolute_lexical_path(path)
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "--output must resolve beneath artifacts/preregistration/v2"
        ) from exc
    if relative == Path("."):
        raise ValueError(
            "--output must name a file beneath artifacts/preregistration/v2"
        )

    current = root
    for component in (Path("."), *relative.parts):
        if component != Path("."):
            current = current / component
        status = _lstat(current)
        if status is None:
            continue
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"--output path contains a symlink: {current}")
        if current != destination and not stat.S_ISDIR(status.st_mode):
            raise ValueError(f"--output ancestor is not a directory: {current}")

    resolved_root = root.resolve(strict=False)
    resolved_destination = destination.resolve(strict=False)
    try:
        resolved_destination.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            "--output must resolve beneath artifacts/preregistration/v2"
        ) from exc
    return _preflight_new_output(resolved_destination)


def _atomic_write_new(path: Path, payload: bytes) -> None:
    destination = _preflight_new_output(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlinked_ancestors(destination)
    if _lstat(destination) is not None:
        raise FileExistsError(f"audit output already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"audit output already exists: {destination}"
            ) from exc
        temporary.unlink()
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def audit_seed_access(
    registry_path: Path,
    ledger_path: Path,
    phase: str,
    output_path: Path,
    *,
    lock_path: Path | None = None,
) -> dict[str, object]:
    if phase not in _PHASES:
        raise ValueError(f"invalid seed-access audit phase: {phase}")
    if phase == "before-formal-run" and lock_path is None:
        raise ValueError("before-formal-run requires lock_path")
    if phase != "before-formal-run" and lock_path is not None:
        raise ValueError("lock_path is accepted only for before-formal-run")
    output_file = _preflight_new_output(Path(output_path))

    registry_file = Path(registry_path)
    registry = _load_registry(registry_file)
    registry_sha256 = _sha256_file(registry_file)
    development, pilot, formal, cases = _registry_pool_contract(registry)
    live_bytes, records = _read_ledger_snapshot(
        Path(ledger_path),
        allow_missing=True,
    )
    for index, record in enumerate(records, start=1):
        if record["registry_sha256"] != registry_sha256:
            raise ValueError(f"ledger record {index} has wrong registry digest")
    audit_candidate_pool_unseen(registry, records)
    _purpose_and_pool_validation(records, development, pilot)

    lock_sha256: str | None = None
    prelock_count = len(records)
    postlock_count = 0
    if phase == "before-final-pilot":
        if any(record["purpose"] != SeedPurpose.DEVELOPMENT.value for record in records):
            raise ValueError("before-final-pilot permits only development access")
    elif phase == "before-formal-lock":
        if any(record["purpose"] == SeedPurpose.FORMAL.value for record in records):
            raise ValueError("before-formal-lock forbids formal access")
    else:
        assert lock_path is not None
        lock_dir = Path(lock_path)
        lock, lock_sha256, locked_cases = _validated_lock(lock_dir)
        prelock_count, postlock_count = _verify_formal_boundary(
            lock_path=lock_dir,
            lock=lock,
            lock_sha256=lock_sha256,
            locked_cases=locked_cases,
            registry_sha256=registry_sha256,
            live_bytes=live_bytes,
            records=records,
            registry=registry,
        )

    access_summary, pool_counts = _count_summary(
        records,
        development,
        pilot,
        formal,
        cases,
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "phase": phase,
        "registry_sha256": registry_sha256,
        "ledger_sha256": _sha256_bytes(live_bytes),
        "ledger_head_sha256": (
            str(records[-1]["digest"]) if records else _ZERO_DIGEST
        ),
        "ledger_bytes": len(live_bytes),
        "record_count": len(records),
        "access_summary": access_summary,
        "pool_counts": pool_counts,
        "prelock_record_count": prelock_count,
        "postlock_record_count": postlock_count,
        "lock_sha256": lock_sha256,
    }
    payload = (_canonical_json(result) + "\n").encode("utf-8")
    _atomic_write_new(output_file, payload)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the append-only seed-access ledger",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit",
        help="audit a seed-access phase boundary without opening a seed",
    )
    audit.add_argument("--registry", type=Path, required=True)
    audit.add_argument("--ledger", type=Path, required=True)
    audit.add_argument("--phase", choices=_PHASES, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument(
        "--lock",
        type=Path,
        help="verified confirmatory lock DIR; required only before a formal run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "audit":
        parser.error("unknown command")
    if args.phase == "before-formal-run" and args.lock is None:
        parser.error("--lock DIR is required for before-formal-run")
    if args.phase != "before-formal-run" and args.lock is not None:
        parser.error("--lock is accepted only for before-formal-run")
    output = _resolve_cli_audit_output(args.output)
    audit_seed_access(
        args.registry,
        args.ledger,
        args.phase,
        output,
        lock_path=args.lock,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
