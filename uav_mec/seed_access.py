"""Append-only, hash-chained seed access ledger."""
from __future__ import annotations
import hashlib, json, os, time, tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Sequence

class SeedPurpose(StrEnum):
    DEVELOPMENT = "development"
    PILOT = "pilot"
    FORMAL = "formal"

@dataclass(frozen=True, slots=True)
class SeedPermit:
    purpose: SeedPurpose
    allowed_seeds: tuple[int, ...]
    family: str | None = None
    case_id: str | None = None
    registry_sha256: str = ""
    lock_sha256: str | None = None
    def __post_init__(self):
        object.__setattr__(self, "allowed_seeds", tuple(self.allowed_seeds))
        if self.purpose is SeedPurpose.FORMAL and (not self.family or not self.case_id or not self.lock_sha256):
            raise ValueError("formal permit requires family, case_id, and lock_sha256")
        if self.purpose is not SeedPurpose.FORMAL and any((self.family, self.case_id, self.lock_sha256)):
            raise ValueError("development and pilot permits cannot bind family/case/lock")

class SeedAccessRecorder:
    def __init__(self, ledger_path: Path, permit: SeedPermit):
        self.ledger_path, self.permit = Path(ledger_path), permit
    def record(self, operation: str, seeds: Sequence[int], metadata: Mapping[str, object]) -> None:
        vals = [int(x) for x in seeds]
        if len(set(vals)) != len(vals): raise ValueError("duplicate seed in access record")
        bad = sorted(set(vals) - set(self.permit.allowed_seeds))
        if bad: raise ValueError(f"seed outside permit: {bad[0]}")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        previous = "0" * 64
        if self.ledger_path.exists():
            records = verify_seed_ledger(self.ledger_path)
            if records: previous = records[-1]["digest"]
        body = {"timestamp": time.time_ns(), "purpose": self.permit.purpose.value,
                "family": self.permit.family, "case_id": self.permit.case_id,
                "seeds": sorted(vals), "operation": str(operation), "metadata": dict(metadata),
                "registry_sha256": self.permit.registry_sha256, "lock_sha256": self.permit.lock_sha256,
                "previous_digest": previous}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        body["digest"] = hashlib.sha256(encoded.encode()).hexdigest()
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())

def _digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _pools(path: Path):
    raw = tomllib.loads(path.read_text(encoding="utf-8")); pools = {"development": [], "pilot": []}
    for key in pools:
        value = raw.get(f"{key}_seeds", raw.get("seed_pools", {}).get(key, []))
        pools[key] = list(value) if isinstance(value, list) else []
    if not pools["development"]: pools["development"] = list(range(1000))
    if not pools["pilot"]: pools["pilot"] = list(range(1000,1100))
    return pools
def development_permit(registry_path: Path) -> SeedPermit: return SeedPermit(SeedPurpose.DEVELOPMENT, tuple(_pools(registry_path)["development"]), registry_sha256=_digest(registry_path))
def pilot_permit(registry_path: Path) -> SeedPermit: return SeedPermit(SeedPurpose.PILOT, tuple(_pools(registry_path)["pilot"]), registry_sha256=_digest(registry_path))
def formal_permit(lock_path: Path, family_name: str, case_id: str) -> SeedPermit:
    raw = json.loads(Path(lock_path).read_text(encoding="utf-8")); cases = raw.get("cases", raw.get("family_cases", {}))
    entry = cases.get(family_name, {}).get(case_id) if isinstance(cases.get(family_name), dict) else None
    if entry is None: raise KeyError(f"{family_name}/{case_id}")
    seeds = entry.get("seeds", entry.get("allowed_seeds"))
    return SeedPermit(SeedPurpose.FORMAL, tuple(seeds), family_name, case_id, raw.get("registry_sha256", ""), _digest(lock_path))
def verify_seed_ledger(path: Path) -> tuple[dict, ...]:
    out=[]; prev="0"*64
    for line_no,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        try: obj=json.loads(line)
        except Exception as e: raise ValueError(f"line {line_no}: malformed JSON") from e
        if not isinstance(obj,dict) or "digest" not in obj or obj.get("previous_digest") != prev: raise ValueError(f"line {line_no}: broken predecessor")
        seeds=obj.get("seeds",[])
        if len(seeds)!=len(set(seeds)): raise ValueError(f"line {line_no}: duplicate seeds")
        digest=obj["digest"]; body={k:v for k,v in obj.items() if k!="digest"}; calc=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        if digest!=calc: raise ValueError(f"line {line_no}: invalid digest")
        out.append(obj); prev=digest
    return tuple(out)
def audit_candidate_pool_unseen(registry, ledger_records):
    dev=set(getattr(registry,"development_seeds",range(1000)))|set(getattr(registry,"pilot_seeds",range(1000,1100)))
    for r in ledger_records:
        if r.get("purpose") in ("development","pilot") and dev.intersection(r.get("seeds",[])): raise ValueError("formal candidate seed accessed during development/pilot")
def audit_seed_access(registry_path, ledger_path, phase, output_path, *, lock_path=None):
    if phase not in {"before-final-pilot","before-formal-lock","before-formal-run"}: raise ValueError("invalid phase")
    records=verify_seed_ledger(ledger_path) if Path(ledger_path).exists() else tuple()
    if phase=="before-formal-run" and lock_path is None: raise ValueError("lock_path required")
    result={"phase":phase,"record_count":len(records),"ledger_sha256":_digest(ledger_path) if Path(ledger_path).exists() else None}
    Path(output_path).parent.mkdir(parents=True,exist_ok=True); Path(output_path).write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8"); return result
