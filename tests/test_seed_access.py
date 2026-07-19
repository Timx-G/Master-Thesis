from __future__ import annotations

import ast
import hashlib
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import uav_mec.seed_access as seed_access_module
from uav_mec.model import ScenarioConfig
from uav_mec.seed_access import (
    SeedAccessRecorder,
    SeedPermit,
    SeedPurpose,
    audit_candidate_pool_unseen,
    audit_seed_access,
    formal_permit,
    verify_seed_ledger,
)
from uav_mec.trace import generate_trace


ZERO_DIGEST = "0" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registry() -> SimpleNamespace:
    return SimpleNamespace(
        development_seeds=tuple(range(1_000)),
        pilot_seeds=tuple(range(1_000, 1_100)),
        families=(
            SimpleNamespace(
                name="main",
                seed_start=24_000,
                seed_count=200,
                cases=(SimpleNamespace(case_id="main_k3_n45"),),
            ),
            SimpleNamespace(
                name="exact",
                seed_start=26_000,
                seed_count=50,
                cases=(SimpleNamespace(case_id="exact_schedule"),),
            ),
        ),
    )


def _registry_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "registry.toml"
    path.write_text("schema_version = 2\n", encoding="utf-8")
    monkeypatch.setattr(seed_access_module, "_load_registry", lambda _: _registry())
    return path


def _permit(
    purpose: SeedPurpose,
    seeds: tuple[int, ...],
    registry_path: Path,
    *,
    family: str | None = None,
    case_id: str | None = None,
    lock_sha256: str | None = None,
) -> SeedPermit:
    return SeedPermit(
        purpose=purpose,
        allowed_seeds=seeds,
        family=family,
        case_id=case_id,
        registry_sha256=_sha256(registry_path),
        lock_sha256=lock_sha256,
    )


def _install_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ledger_path: Path,
    *,
    locked_seeds: tuple[int, ...] = (24_000, 24_001),
) -> tuple[Path, SimpleNamespace]:
    lock_dir = tmp_path / "confirmatory-lock"
    lock_dir.mkdir()
    lock_json = lock_dir / "confirmatory_lock.json"
    lock_json.write_text('{"schema_version":1}\n', encoding="utf-8")
    prefix = ledger_path.read_bytes() if ledger_path.exists() else b""
    (lock_dir / "seed-access-prelock.jsonl").write_bytes(prefix)
    prefix_records = verify_seed_ledger(ledger_path) if prefix else ()
    registry_sha256 = next(
        record["registry_sha256"] for record in prefix_records
    ) if prefix_records else "a" * 64
    lock = SimpleNamespace(
        registry_sha256=registry_sha256,
        seed_ledger_prefix_bytes=len(prefix),
        seed_ledger_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        seed_ledger_head_sha256=(
            prefix_records[-1]["digest"] if prefix_records else ZERO_DIGEST
        ),
        families=(
            SimpleNamespace(
                family="main",
                cases=(
                    SimpleNamespace(
                        case_id="main_k3_n45",
                        seeds=locked_seeds,
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        seed_access_module,
        "_load_confirmatory_lock",
        lambda path: lock,
    )
    return lock_dir, lock


def test_generate_trace_requires_verified_access_outside_development_pool(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text("schema_version = 2\n", encoding="utf-8")
    recorder = SeedAccessRecorder(
        tmp_path / "seed-access.jsonl",
        _permit(SeedPurpose.DEVELOPMENT, tuple(range(1_000)), registry_path),
    )

    signature = inspect.signature(generate_trace)
    assert signature.parameters["seed_access"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="seed_access"):
        generate_trace(ScenarioConfig(num_users=1, num_slots=1), 42)
    with pytest.raises(ValueError, match="outside permit"):
        generate_trace(
            ScenarioConfig(num_users=1, num_slots=1),
            24_000,
            seed_access=recorder,
        )
    assert not recorder.ledger_path.exists()


def test_seed_ledger_records_trace_generation_with_hash_chain(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text("schema_version = 2\n", encoding="utf-8")
    ledger = tmp_path / "seed-access.jsonl"
    recorder = SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.DEVELOPMENT, (41, 42), registry_path),
    )

    recorder.record("generate_trace", [42], {"scenario_fingerprint": "a" * 64})
    recorder.record("generate_trace", [41], {"csi_error_std": 0.0})

    records = verify_seed_ledger(ledger)
    assert [record["sequence"] for record in records] == [1, 2]
    assert all(str(record["timestamp_utc"]).endswith("Z") for record in records)
    assert records[0]["previous_digest"] == ZERO_DIGEST
    assert records[1]["previous_digest"] == records[0]["digest"]
    assert records[0]["seeds"] == [42]
    assert records[1]["seeds"] == [41]
    assert ledger.read_text(encoding="utf-8").splitlines() == [
        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]


def test_seed_ledger_serializes_concurrent_appends_under_one_file_lock(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text("schema_version = 2\n", encoding="utf-8")
    ledger = tmp_path / "seed-access.jsonl"
    permit = _permit(
        SeedPurpose.DEVELOPMENT,
        tuple(range(32)),
        registry_path,
    )

    def append(seed: int) -> None:
        SeedAccessRecorder(ledger, permit).record("concurrent", [seed], {})

    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(executor.map(append, range(32)))

    records = verify_seed_ledger(ledger)
    assert [record["sequence"] for record in records] == list(range(1, 33))
    assert {record["seeds"][0] for record in records} == set(range(32))


def test_seed_ledger_rejects_tampering(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text("schema_version = 2\n", encoding="utf-8")
    ledger = tmp_path / "seed-access.jsonl"
    SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.DEVELOPMENT, (42,), registry_path),
    ).record("generate_trace", [42], {})
    ledger.write_bytes(ledger.read_bytes().replace(b"generate_trace", b"analyze_results"))

    with pytest.raises(ValueError, match="line 1"):
        verify_seed_ledger(ledger)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda record: record.update(extra="undeclared"),
        lambda record: record.update(sequence=True),
        lambda record: record.update(seeds=[42.0]),
        lambda record: record.update(metadata={"bad": float("nan")}),
        lambda record: record.update(timestamp_utc="2026-07-19 00:00:00"),
    ),
)
def test_seed_ledger_rejects_noncanonical_or_untyped_records(
    tmp_path: Path,
    mutation,
) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text("schema_version = 2\n", encoding="utf-8")
    ledger = tmp_path / "seed-access.jsonl"
    SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.DEVELOPMENT, (42,), registry_path),
    ).record("generate_trace", [42], {})
    record = json.loads(ledger.read_text(encoding="utf-8"))
    mutation(record)
    ledger.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        verify_seed_ledger(ledger)


@pytest.mark.parametrize("bad_metadata", ({"x": float("inf")}, {"x": [float("nan")]}, {1: "x"}))
def test_seed_recorder_rejects_nonfinite_or_non_json_metadata_before_creation(
    tmp_path: Path,
    bad_metadata,
) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text("schema_version = 2\n", encoding="utf-8")
    ledger = tmp_path / "seed-access.jsonl"
    recorder = SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.DEVELOPMENT, (42,), registry_path),
    )

    with pytest.raises((TypeError, ValueError), match="metadata"):
        recorder.record("generate_trace", [42], bad_metadata)
    assert not ledger.exists()


def test_candidate_pool_audit_rejects_a_prelock_formal_access(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text("schema_version = 2\n", encoding="utf-8")
    ledger = tmp_path / "seed-access.jsonl"
    recorder = SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.DEVELOPMENT, (24_000,), registry_path),
    )
    recorder.record("generate_trace", [24_000], {})

    with pytest.raises(ValueError, match="formal candidate"):
        audit_candidate_pool_unseen(_registry(), verify_seed_ledger(ledger))


def test_candidate_pool_audit_accepts_normal_development_and_bound_formal_access(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text("schema_version = 2\n", encoding="utf-8")
    ledger = tmp_path / "seed-access.jsonl"
    SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.DEVELOPMENT, (42,), registry_path),
    ).record("generate_trace", [42], {})
    SeedAccessRecorder(
        ledger,
        _permit(
            SeedPurpose.FORMAL,
            (24_000,),
            registry_path,
            family="main",
            case_id="main_k3_n45",
            lock_sha256="b" * 64,
        ),
    ).record("run_definition", [24_000], {})

    audit_candidate_pool_unseen(_registry(), verify_seed_ledger(ledger))


def test_seed_access_audit_cli_writes_checksum_bound_pre_pilot_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry_file(tmp_path, monkeypatch)
    ledger = tmp_path / "seed-access.jsonl"
    SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.DEVELOPMENT, (42,), registry_path),
    ).record("generate_trace", [42], {})
    output = tmp_path / "audit.json"

    assert seed_access_module.main(
        [
            "audit",
            "--registry",
            str(registry_path),
            "--ledger",
            str(ledger),
            "--phase",
            "before-final-pilot",
            "--output",
            str(output),
        ]
    ) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["phase"] == "before-final-pilot"
    assert report["registry_sha256"] == _sha256(registry_path)
    assert report["ledger_sha256"] == _sha256(ledger)
    assert report["ledger_head_sha256"] == verify_seed_ledger(ledger)[-1]["digest"]
    assert report["pool_counts"]["development"] == {
        "accessed": 1,
        "size": 1_000,
        "unseen": 999,
    }
    assert report["pool_counts"]["pilot"]["accessed"] == 0
    assert output.read_bytes().endswith(b"\n")


def test_seed_access_audit_rejects_unknown_phase_or_boundary_violation_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry_file(tmp_path, monkeypatch)
    ledger = tmp_path / "seed-access.jsonl"
    SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.PILOT, (1_000,), registry_path),
    ).record("run_power_pilot", [1_000], {})
    output = tmp_path / "audit.json"

    with pytest.raises(ValueError, match="phase"):
        audit_seed_access(registry_path, ledger, "unknown", output)
    assert not output.exists()
    with pytest.raises(ValueError, match="before-final-pilot"):
        audit_seed_access(
            registry_path,
            ledger,
            "before-final-pilot",
            output,
        )
    assert not output.exists()


def test_before_formal_lock_allows_pilot_but_rejects_every_formal_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry_file(tmp_path, monkeypatch)
    ledger = tmp_path / "seed-access.jsonl"
    SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.PILOT, (1_000,), registry_path),
    ).record("run_power_pilot", [1_000], {})
    output = tmp_path / "audit.json"

    audit_seed_access(registry_path, ledger, "before-formal-lock", output)
    output.unlink()
    SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.PILOT, (24_000,), registry_path),
    ).record("generate_trace", [24_000], {})
    with pytest.raises(ValueError, match="formal candidate"):
        audit_seed_access(registry_path, ledger, "before-formal-lock", output)
    assert not output.exists()


def test_formal_permit_requires_verified_lock_directory_and_exact_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text("schema_version = 2\n", encoding="utf-8")
    ledger = tmp_path / "seed-access.jsonl"
    SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.DEVELOPMENT, (42,), registry_path),
    ).record("generate_trace", [42], {})
    lock_dir, lock = _install_lock(tmp_path, monkeypatch, ledger)
    lock.registry_sha256 = _sha256(registry_path)

    permit = formal_permit(lock_dir, "main", "main_k3_n45")
    assert permit.allowed_seeds == (24_000, 24_001)
    assert permit.registry_sha256 == _sha256(registry_path)
    assert permit.lock_sha256 == _sha256(lock_dir / "confirmatory_lock.json")
    with pytest.raises(KeyError, match="main/missing"):
        formal_permit(lock_dir, "main", "missing")
    with pytest.raises(ValueError, match="directory"):
        formal_permit(lock_dir / "confirmatory_lock.json", "main", "main_k3_n45")


def test_before_formal_run_verifies_prefix_and_every_post_prefix_permit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry_file(tmp_path, monkeypatch)
    ledger = tmp_path / "seed-access.jsonl"
    SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.DEVELOPMENT, (42,), registry_path),
    ).record("generate_trace", [42], {})
    lock_dir, lock = _install_lock(tmp_path, monkeypatch, ledger)
    lock.registry_sha256 = _sha256(registry_path)
    formal = formal_permit(lock_dir, "main", "main_k3_n45")
    SeedAccessRecorder(ledger, formal).record("run_definition", [24_000], {})
    output = tmp_path / "audit.json"

    report = audit_seed_access(
        registry_path,
        ledger,
        "before-formal-run",
        output,
        lock_path=lock_dir,
    )

    assert report["prelock_record_count"] == 1
    assert report["postlock_record_count"] == 1
    assert report["lock_sha256"] == formal.lock_sha256
    assert report["pool_counts"]["formal/main"]["accessed"] == 1


@pytest.mark.parametrize("failure", ("prefix", "lock_digest", "case", "seed"))
def test_before_formal_run_rejects_invalid_prefix_or_post_prefix_permit_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    registry_path = _registry_file(tmp_path, monkeypatch)
    ledger = tmp_path / "seed-access.jsonl"
    SeedAccessRecorder(
        ledger,
        _permit(SeedPurpose.DEVELOPMENT, (42,), registry_path),
    ).record("generate_trace", [42], {})
    lock_dir, lock = _install_lock(tmp_path, monkeypatch, ledger)
    lock.registry_sha256 = _sha256(registry_path)
    formal = formal_permit(lock_dir, "main", "main_k3_n45")
    if failure == "prefix":
        copied = lock_dir / "seed-access-prelock.jsonl"
        copied.write_bytes(copied.read_bytes().replace(b"generate_trace", b"analyze_results"))
    else:
        SeedAccessRecorder(
            ledger,
            _permit(
                SeedPurpose.FORMAL,
                (24_000 if failure != "seed" else 24_199,),
                registry_path,
                family="main",
                case_id=("main_k3_n45" if failure != "case" else "undeclared"),
                lock_sha256=(formal.lock_sha256 if failure != "lock_digest" else "c" * 64),
            ),
        ).record(
            "run_definition",
            [24_000 if failure != "seed" else 24_199],
            {},
        )
    output = tmp_path / "audit.json"

    with pytest.raises((KeyError, ValueError)):
        audit_seed_access(
            registry_path,
            ledger,
            "before-formal-run",
            output,
            lock_path=lock_dir,
        )
    assert not output.exists()


def test_before_formal_run_fails_closed_without_lock_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry_file(tmp_path, monkeypatch)
    lock_dir = tmp_path / "confirmatory-lock"
    lock_dir.mkdir()
    (lock_dir / "confirmatory_lock.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        seed_access_module,
        "_load_confirmatory_lock",
        lambda _: (_ for _ in ()).throw(RuntimeError("verifier unavailable")),
    )

    with pytest.raises(RuntimeError, match="verifier unavailable"):
        audit_seed_access(
            registry_path,
            tmp_path / "missing-ledger.jsonl",
            "before-formal-run",
            tmp_path / "audit.json",
            lock_path=lock_dir,
        )


def test_seed_access_audit_help_lists_the_three_exact_phases(capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        seed_access_module.main(["audit", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "before-final-pilot" in help_text
    assert "before-formal-lock" in help_text
    assert "before-formal-run" in help_text


def test_tests_generate_only_development_seeds() -> None:
    trace_generating_tests = (
        "test_trace.py",
        "test_simulation.py",
        "test_csi_error.py",
        "test_ia_greedy.py",
        "test_thesis_suites.py",
    )
    for name in trace_generating_tests:
        tree = ast.parse((Path("tests") / name).read_text(encoding="utf-8"))
        for call in (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "generate_trace"
        ):
            assert any(keyword.arg == "seed_access" for keyword in call.keywords), name
            if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
                assert type(call.args[1].value) is int
                assert 0 <= call.args[1].value <= 999
