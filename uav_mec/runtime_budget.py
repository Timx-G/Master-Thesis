"""Registry-complete, lockable runtime and storage budget construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .experiment_registry import load_registry


@dataclass(frozen=True, slots=True)
class CaseRuntimeBudget:
    case_id: str
    role: str
    sample_size: int
    shard_size: int
    cpu_hours: float
    wall_clock_hours: float
    peak_memory_bytes: int
    disk_bytes: int
    event_log_bytes: int

    def __post_init__(self) -> None:
        if not self.case_id or not self.role:
            raise ValueError("runtime budget case identity must be non-empty")
        if self.sample_size <= 0 or self.shard_size <= 0:
            raise ValueError("runtime budget sample and shard sizes must be positive")
        if any(
            value < 0
            for value in (
                self.cpu_hours,
                self.wall_clock_hours,
                self.peak_memory_bytes,
                self.disk_bytes,
                self.event_log_bytes,
            )
        ):
            raise ValueError("runtime budget case estimates must be non-negative")


@dataclass(frozen=True, slots=True)
class FamilyRuntimeBudget:
    family: str
    concurrency: int
    retry_fraction: float
    cases: tuple[CaseRuntimeBudget, ...]
    cpu_hours: float
    wall_clock_hours: float
    peak_memory_bytes: int
    disk_bytes: int
    event_log_bytes: int

    def __post_init__(self) -> None:
        if not self.family or self.concurrency <= 0 or not self.cases:
            raise ValueError("runtime budget family identity, concurrency, and cases are required")
        if not 0.0 <= self.retry_fraction <= 1.0:
            raise ValueError("runtime budget retry fraction must be in [0, 1]")
        if any(
            value < 0
            for value in (
                self.cpu_hours,
                self.wall_clock_hours,
                self.peak_memory_bytes,
                self.disk_bytes,
                self.event_log_bytes,
            )
        ):
            raise ValueError("runtime budget family estimates must be non-negative")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("runtime budget family contains duplicate cases")
        expected_cpu = sum(case.cpu_hours for case in self.cases) * (1.0 + self.retry_fraction)
        expected_wall = sum(case.wall_clock_hours for case in self.cases) * (
            1.0 + self.retry_fraction
        )
        if not math.isclose(self.cpu_hours, expected_cpu, rel_tol=1e-12):
            raise ValueError("runtime budget family CPU total mismatch")
        if not math.isclose(self.wall_clock_hours, expected_wall, rel_tol=1e-12):
            raise ValueError("runtime budget family wall-clock total mismatch")
        if self.peak_memory_bytes != max(case.peak_memory_bytes for case in self.cases):
            raise ValueError("runtime budget family peak-memory total mismatch")
        if self.disk_bytes != sum(case.disk_bytes for case in self.cases):
            raise ValueError("runtime budget family disk total mismatch")
        if self.event_log_bytes != sum(case.event_log_bytes for case in self.cases):
            raise ValueError("runtime budget family event-log total mismatch")


@dataclass(frozen=True, slots=True)
class RuntimeBudget:
    schema_version: int
    registry_sha256: str
    power_decision_sha256: str
    audit_sha256: str
    execution_policy_sha256: str
    primary_sample_size: int
    power_guarantee: bool
    families: tuple[FamilyRuntimeBudget, ...]
    total_cpu_hours: float
    total_disk_bytes: int
    _document_json: str = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.primary_sample_size <= 0 or not self.families:
            raise ValueError("runtime budget schema, sample size, and families are required")
        if self.total_cpu_hours < 0.0 or self.total_disk_bytes < 0:
            raise ValueError("runtime budget totals must be non-negative")
        if len({family.family for family in self.families}) != len(self.families):
            raise ValueError("runtime budget contains duplicate families")
        if not math.isclose(
            self.total_cpu_hours,
            sum(family.cpu_hours for family in self.families),
            rel_tol=1e-12,
        ):
            raise ValueError("runtime budget total CPU mismatch")
        if self.total_disk_bytes != sum(family.disk_bytes for family in self.families):
            raise ValueError("runtime budget total disk mismatch")
        for digest in (
            self.registry_sha256,
            self.power_decision_sha256,
            self.audit_sha256,
            self.execution_policy_sha256,
        ):
            if len(digest) != 64:
                raise ValueError("runtime budget input digest is invalid")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self._document_json)

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


def _runtime_budget_from_dict(value: dict[str, object]) -> RuntimeBudget:
    raw_families = value.get("families")
    if not isinstance(raw_families, dict):
        raise ValueError("runtime budget family table is invalid")
    raw_cases = value.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("runtime budget case table is invalid")
    family_order = tuple(
        dict.fromkeys(
            str(row["family"]) for row in raw_cases if isinstance(row, dict) and "family" in row
        )
    )
    if set(family_order) != set(raw_families):
        raise ValueError("runtime budget family/case coverage is inconsistent")
    families: list[FamilyRuntimeBudget] = []
    for family_name in family_order:
        rows = raw_families[family_name]
        if not isinstance(family_name, str) or not isinstance(rows, list) or not rows:
            raise ValueError("runtime budget family rows are invalid")
        cases = tuple(
            CaseRuntimeBudget(
                case_id=str(row["case_id"]),
                role=str(row["role"]),
                sample_size=int(row["seed_count"]),
                shard_size=int(row["shard_size"]),
                cpu_hours=float(row["cpu_hours"]),
                wall_clock_hours=float(row["wall_hours_p95"]),
                peak_memory_bytes=int(row["peak_concurrent_memory_bytes"]),
                disk_bytes=int(row["disk_bytes"]),
                event_log_bytes=int(row["log_bytes"]),
            )
            for row in rows
        )
        retry_fraction = float(rows[0]["retry_fraction"])
        families.append(
            FamilyRuntimeBudget(
                family=family_name,
                concurrency=int(rows[0]["concurrency"]),
                retry_fraction=retry_fraction,
                cases=cases,
                cpu_hours=sum(
                    float(row["cpu_hours"]) + float(row["retry_cpu_hours"]) for row in rows
                ),
                wall_clock_hours=sum(
                    float(row["wall_hours_p95"]) * (1.0 + retry_fraction) for row in rows
                ),
                peak_memory_bytes=max(case.peak_memory_bytes for case in cases),
                disk_bytes=sum(case.disk_bytes for case in cases),
                event_log_bytes=sum(case.event_log_bytes for case in cases),
            )
        )
    ordered_document = dict(value)
    ordered_document["families"] = {
        family_name: raw_families[family_name] for family_name in family_order
    }
    document = json.dumps(
        ordered_document,
        allow_nan=False,
        separators=(",", ":"),
    )
    return RuntimeBudget(
        schema_version=int(value["schema_version"]),
        registry_sha256=str(value["registry_sha256"]),
        power_decision_sha256=str(value["power_decision_sha256"]),
        audit_sha256=str(value["runtime_audit_sha256"]),
        execution_policy_sha256=str(value["execution_policy_sha256"]),
        primary_sample_size=int(value["primary_sample_size"]),
        power_guarantee=bool(value["power_guarantee"]),
        families=tuple(families),
        total_cpu_hours=float(value["total_cpu_hours"]),
        total_disk_bytes=int(value["total_disk_bytes"]),
        _document_json=document,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _content_sha256(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field} must be finite and non-negative")
    return float(value)


def _atomic_json(path: Path, value: object) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _primary_sample_size(power: dict[str, object]) -> tuple[int, bool]:
    selected = power.get("selected_n")
    evaluated = power.get("evaluated_n")
    insufficient = power.get("evidence_insufficient")
    if selected is None:
        if evaluated != 200 or insufficient is not True:
            raise ValueError(
                "null selected_n requires evaluated_n=200 and evidence_insufficient=true"
            )
        return 200, False
    if selected not in {50, 100, 200} or insufficient is not False:
        raise ValueError("selected_n must be a registered candidate with sufficient evidence")
    return int(selected), True


def build_runtime_budget(
    registry_path: Path,
    power_decision_path: Path,
    audit_path: Path,
    execution_policy_path: Path,
    output_path: Path,
    *,
    retry_fraction: float = 0.1,
) -> RuntimeBudget:
    """Build and atomically write one complete per-family/case budget."""
    if (
        not isinstance(retry_fraction, (int, float))
        or isinstance(retry_fraction, bool)
        or not math.isfinite(float(retry_fraction))
        or not 0.0 <= float(retry_fraction) <= 1.0
    ):
        raise ValueError("retry_fraction must be in [0, 1]")
    registry = load_registry(Path(registry_path))
    power = _load_json(power_decision_path)
    primary_n, power_guarantee = _primary_sample_size(power)

    policy_raw = _load_json(execution_policy_path)
    if policy_raw.get("schema_version") != 1 or not isinstance(policy_raw.get("families"), list):
        raise ValueError("execution policy schema is invalid")
    policy_rows = policy_raw["families"]
    expected_families = [family.name for family in registry.families]
    if [row.get("family") for row in policy_rows if isinstance(row, dict)] != expected_families:
        raise ValueError("execution policy family coverage/order mismatch")
    policies: dict[str, tuple[int, int]] = {}
    for row in policy_rows:
        if not isinstance(row, dict) or set(row) != {"family", "concurrency", "shard_size"}:
            raise ValueError("execution policy rows have invalid fields")
        family = str(row["family"])
        if family in policies:
            raise ValueError("duplicate execution policy family")
        policies[family] = (
            _positive_int(row["concurrency"], f"{family}.concurrency"),
            _positive_int(row["shard_size"], f"{family}.shard_size"),
        )

    audit_raw = _load_json(audit_path)
    if audit_raw.get("schema_version") != 1 or not isinstance(audit_raw.get("rows"), list):
        raise ValueError("runtime audit schema is invalid")
    audits: dict[tuple[str, str], dict[str, object]] = {}
    for row in audit_raw["rows"]:
        if not isinstance(row, dict):
            raise ValueError("runtime audit rows must be objects")
        key = (str(row.get("family")), str(row.get("case_id")))
        if key in audits:
            raise ValueError("runtime audit coverage contains a duplicate case")
        audits[key] = row
    expected_cases = {
        (family.name, case.case_id) for family in registry.families for case in family.cases
    }
    if set(audits) != expected_cases:
        raise ValueError("runtime audit coverage does not match registry family/case pairs")

    case_rows: list[dict[str, object]] = []
    family_rows: dict[str, list[dict[str, object]]] = {
        family.name: [] for family in registry.families
    }
    for family in registry.families:
        concurrency, shard_size = policies[family.name]
        for case in family.cases:
            if family.name in {"resource", "main", "joint"}:
                seed_count = primary_n
            elif family.name == "externality" and case.role == "primary":
                seed_count = primary_n
            else:
                if case.fixed_sample_size is None:
                    raise ValueError(f"{family.name}/{case.case_id} lacks a fixed sample size")
                seed_count = case.fixed_sample_size
            audit = audits[(family.name, case.case_id)]
            mean_runtime = _finite_nonnegative(audit.get("mean_runtime_s"), "mean_runtime_s")
            p95_runtime = _finite_nonnegative(audit.get("p95_runtime_s"), "p95_runtime_s")
            peak_memory = _finite_nonnegative(audit.get("peak_memory_bytes"), "peak_memory_bytes")
            raw_bytes = _finite_nonnegative(audit.get("raw_bytes"), "raw_bytes")
            summary_bytes = _finite_nonnegative(audit.get("summary_bytes"), "summary_bytes")
            event_bytes = _finite_nonnegative(audit.get("event_bytes"), "event_bytes")
            log_bytes = _finite_nonnegative(audit.get("log_bytes"), "log_bytes")
            instances = seed_count * family.num_slots
            cpu_hours = mean_runtime * instances / 3600.0
            row = {
                "family": family.name,
                "case_id": case.case_id,
                "role": case.role,
                "runner_kind": audit.get("runner_kind"),
                "seed_count": seed_count,
                "slot_count": family.num_slots,
                "method_count": len(family.methods),
                "slot_method_evaluations": instances * len(family.methods),
                # Resource cases are the three raw-radio generators.  Each case
                # searches once and replays all six CPU response policies.
                "wireless_searches": instances,
                "resource_policy_evaluations": (instances * 6 if family.name == "resource" else 0),
                "cpu_hours": cpu_hours,
                "wall_hours_p95": p95_runtime * instances / concurrency / 3600.0,
                "peak_concurrent_memory_bytes": int(peak_memory * concurrency),
                "disk_bytes": int((raw_bytes + summary_bytes + event_bytes) * instances),
                "log_bytes": int(log_bytes * instances),
                "concurrency": concurrency,
                "shard_size": shard_size,
                "retry_fraction": float(retry_fraction),
                "retry_cpu_hours": cpu_hours * float(retry_fraction),
                "execution_policy_sha256": _sha256(execution_policy_path),
            }
            case_rows.append(row)
            family_rows[family.name].append(row)
    result: dict[str, object] = {
        "schema_version": 1,
        "inputs": {
            "registry": str(Path(registry_path).resolve()),
            "power_decision": str(Path(power_decision_path).resolve()),
            "runtime_audit": str(Path(audit_path).resolve()),
            "execution_policy": str(Path(execution_policy_path).resolve()),
        },
        "registry_sha256": _sha256(registry_path),
        "power_decision_sha256": _sha256(power_decision_path),
        "runtime_audit_sha256": _sha256(audit_path),
        "execution_policy_sha256": _sha256(execution_policy_path),
        "primary_sample_size": primary_n,
        "power_guarantee": power_guarantee,
        "retry_fraction": float(retry_fraction),
        "families": family_rows,
        "cases": case_rows,
        "total_cpu_hours": sum(
            float(row["cpu_hours"]) + float(row["retry_cpu_hours"]) for row in case_rows
        ),
        "total_wall_hours_p95": sum(
            float(row["wall_hours_p95"]) * (1.0 + float(retry_fraction)) for row in case_rows
        ),
        "total_disk_bytes": sum(int(row["disk_bytes"]) for row in case_rows),
        "total_log_bytes": sum(int(row["log_bytes"]) for row in case_rows),
    }
    result["content_sha256"] = _content_sha256(result)
    _atomic_json(output_path, result)
    return _runtime_budget_from_dict(result)


def load_and_verify_runtime_budget(path: Path) -> RuntimeBudget:
    """Verify the budget identity, bound inputs, coverage, and derived totals."""

    budget = _load_json(path)
    stored_content = budget.get("content_sha256")
    if not isinstance(stored_content, str) or len(stored_content) != 64:
        raise ValueError("runtime budget content digest is missing")
    unsigned = dict(budget)
    del unsigned["content_sha256"]
    if _content_sha256(unsigned) != stored_content:
        raise ValueError("runtime budget content digest mismatch")

    inputs = budget.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "registry",
        "power_decision",
        "runtime_audit",
        "execution_policy",
    }:
        raise ValueError("runtime budget input paths are invalid")
    digest_fields = {
        "registry": "registry_sha256",
        "power_decision": "power_decision_sha256",
        "runtime_audit": "runtime_audit_sha256",
        "execution_policy": "execution_policy_sha256",
    }
    for input_name, digest_field in digest_fields.items():
        raw_path = inputs[input_name]
        if not isinstance(raw_path, str):
            raise ValueError(f"{input_name} input path is invalid")
        input_path = Path(raw_path)
        if input_path.is_symlink() or not input_path.is_file():
            raise ValueError(f"{input_name} input is unavailable")
        if _sha256(input_path) != budget.get(digest_field):
            label = input_name.replace("_", " ")
            raise ValueError(f"{label} digest mismatch")

    registry = load_registry(Path(str(inputs["registry"])))
    cases = budget.get("cases")
    families = budget.get("families")
    if not isinstance(cases, list) or not isinstance(families, dict):
        raise ValueError("runtime budget family/case tables are invalid")
    expected_keys = [
        (family.name, case.case_id) for family in registry.families for case in family.cases
    ]
    actual_keys = [
        (row.get("family"), row.get("case_id")) for row in cases if isinstance(row, dict)
    ]
    if actual_keys != expected_keys or len(actual_keys) != len(cases):
        raise ValueError("runtime budget case coverage/order mismatch")
    if set(families) != {family.name for family in registry.families}:
        raise ValueError("runtime budget family coverage mismatch")

    expected_cpu = sum(
        float(row["cpu_hours"]) + float(row["retry_cpu_hours"])
        for row in cases
        if isinstance(row, dict)
    )
    expected_disk = sum(int(row["disk_bytes"]) for row in cases if isinstance(row, dict))
    expected_log = sum(int(row["log_bytes"]) for row in cases if isinstance(row, dict))
    if not math.isclose(float(budget.get("total_cpu_hours", -1)), expected_cpu, rel_tol=1e-12):
        raise ValueError("runtime budget CPU total mismatch")
    if budget.get("total_disk_bytes") != expected_disk:
        raise ValueError("runtime budget disk total mismatch")
    if budget.get("total_log_bytes") != expected_log:
        raise ValueError("runtime budget log total mismatch")
    return _runtime_budget_from_dict(budget)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--registry", type=Path, required=True)
    build_parser.add_argument("--power-decision", type=Path, required=True)
    build_parser.add_argument("--runtime-audit", type=Path, required=True)
    build_parser.add_argument("--execution-policy", type=Path, required=True)
    build_parser.add_argument("--retry-fraction", type=float, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--budget", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build":
        build_runtime_budget(
            args.registry,
            args.power_decision,
            args.runtime_audit,
            args.execution_policy,
            args.output,
            retry_fraction=args.retry_fraction,
        )
    else:
        load_and_verify_runtime_budget(args.budget)


if __name__ == "__main__":
    main()
