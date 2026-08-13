from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

import pytest

from uav_mec.experiment_registry import load_registry
from uav_mec.geolife_lock import build_trace_lock, load_and_verify_trace_lock
from uav_mec.run_development_calibration import (
    DevelopmentCalibrationCase,
    main,
    parse_development_seed_range,
    run_development_calibration,
)
from uav_mec.runtime_audit import DescriptiveRuntimeAudit
from uav_mec.seed_access import SeedAccessRecorder, development_permit

REGISTRY = Path("configs/thesis/experiment_registry_v2.toml")


def _recorder(tmp_path: Path) -> SeedAccessRecorder:
    return SeedAccessRecorder(tmp_path / "seed-access.jsonl", development_permit(REGISTRY))


def _audit() -> DescriptiveRuntimeAudit:
    return DescriptiveRuntimeAudit(
        reference_hardware="synthetic",
        python_environment_sha256="a" * 64,
        p50_runtime_s=0.1,
        p95_runtime_s=0.2,
        mean_runtime_s=0.12,
        candidate_count_p50=10,
        candidate_count_p95=20,
        candidate_count_total=30,
        cache_hit_rate=0.25,
        peak_memory_bytes=1024,
        log_bytes=128,
        measured_cpu_seconds=0.24,
    )


def test_development_calibration_accepts_only_exact_nontriviality_and_runtime_capacity() -> None:
    assert tuple(item.value for item in DevelopmentCalibrationCase) == (
        "exact_nontriviality",
        "runtime_capacity",
    )
    with pytest.raises(ValueError):
        DevelopmentCalibrationCase("other")


@pytest.mark.parametrize("value", ("-1:0", "999:1000", "4:3", "1", "a:2"))
def test_development_calibration_rejects_any_seed_outside_zero_through_999_before_output(
    value: str,
) -> None:
    registry = load_registry(REGISTRY)
    with pytest.raises(ValueError):
        parse_development_seed_range(value, registry)


def test_exact_development_calibration_reports_both_nontriviality_counts_and_fractions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = (
        {"complete": True, "cpu_nontrivial": True, "schedule_nontrivial": False},
        {"complete": True, "cpu_nontrivial": False, "schedule_nontrivial": True},
        {"complete": True, "cpu_nontrivial": True, "schedule_nontrivial": True},
        {"complete": True, "cpu_nontrivial": False, "schedule_nontrivial": False},
    )
    monkeypatch.setattr(
        "uav_mec.run_development_calibration._run_exact_nontriviality",
        lambda **_kwargs: rows,
    )
    output = run_development_calibration(
        REGISTRY,
        DevelopmentCalibrationCase.EXACT_NONTRIVIALITY,
        (0, 1),
        tmp_path / "exact",
        workers=1,
        trace_lock_path=None,
        seed_access=_recorder(tmp_path),
    )
    result = json.loads((output / "exact_nontriviality.json").read_text(encoding="utf-8"))
    assert result == {
        "complete_certificate_count": 4,
        "cpu_nontrivial_count": 2,
        "cpu_nontrivial_fraction": 0.5,
        "schedule_nontrivial_count": 2,
        "schedule_nontrivial_fraction": 0.5,
        "seed_count": 2,
        "slot_count": 2,
    }


def test_runtime_development_calibration_emits_one_audit_row_per_registry_family_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, tuple[int, ...], int]] = []

    def adapter(**kwargs):
        calls.append(
            (
                kwargs["family"].name,
                kwargs["case"].case_id,
                kwargs["seeds"],
                kwargs["max_slots"],
            )
        )
        return _audit()

    monkeypatch.setattr(
        "uav_mec.run_development_calibration._RUNTIME_ADAPTERS",
        {
            family: adapter
            for family in (
                "resource",
                "exact",
                "externality",
                "main",
                "joint",
                "scaling",
                "robustness",
                "trace_driven",
            )
        },
    )
    monkeypatch.setattr(
        "uav_mec.run_development_calibration._validate_trace_lock",
        lambda _path: "f" * 64,
    )
    trace_lock = tmp_path / "trace-lock.json"
    trace_lock.write_text('{"split":"calibration"}\n', encoding="utf-8")
    output = run_development_calibration(
        REGISTRY,
        DevelopmentCalibrationCase.RUNTIME_CAPACITY,
        (0, 1),
        tmp_path / "runtime",
        workers=2,
        trace_lock_path=trace_lock,
        seed_access=_recorder(tmp_path),
    )
    registry = load_registry(REGISTRY)
    expected = [
        (family.name, case.case_id) for family in registry.families for case in family.cases
    ]
    payload = json.loads((output / "runtime_audit.json").read_text(encoding="utf-8"))
    assert [(row["family"], row["case_id"]) for row in payload["rows"]] == expected
    assert [(family, case) for family, case, _seeds, _slots in calls] == expected
    assert all(seeds == (0, 1) and slots <= 20 for _, _, seeds, slots in calls)


def _trace_lock_fixture(tmp_path: Path) -> Path:
    archive = tmp_path / "geolife.zip"
    header = (
        "Geolife trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n"
        "0,2,255,My Track,0,0,2,8421376\n0\n"
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for participant, mode, minute in (
            ("000", "walk", 53),
            ("001", "bike", 54),
            ("002", "walk", 55),
        ):
            bundle.writestr(
                f"Data/{participant}/labels.txt",
                "Start Time\tEnd Time\tTransportation Mode\n"
                f"2008/10/23 02:{minute}:04\t2008/10/23 02:{minute}:14\t{mode}\n",
            )
            points = "".join(
                f"39.984702,116.{318417 + 60 * index:06d},0,100,39744.12,"
                f"2008-10-23,02:{minute}:{4 + 5 * index:02d}\n"
                for index in range(3)
            )
            bundle.writestr(f"Data/{participant}/Trajectory/source.plt", header + points)
    license_text = "Synthetic GeoLife test license."
    license_sha = hashlib.sha256(license_text.encode()).hexdigest()
    protocol = tmp_path / "protocol.toml"
    protocol.write_text(
        f"""schema_version = 2
[dataset]
version = "synthetic"
source_url = "https://example.test/geolife"
retrieved_on = "2026-08-04"
license_text = "{license_text}"
license_sha256 = "{license_sha}"
[selection]
accepted_modes = ["walk", "bike"]
slot_seconds = 1.0
minimum_segment_samples = 3
max_segments_per_participant_mode = 1
[splits]
train = ["000"]
calibration = ["001"]
final = ["002"]
[placement]
area_size_m = 500.0
translation_m = [250.0, 250.0]
rotation_deg = 0.0
[[final_groups]]
group_id = "final"
participants = ["002"]
""",
        encoding="utf-8",
    )
    lock_path = tmp_path / "trace-lock.json"
    build_trace_lock(archive, protocol, REGISTRY, lock_path)
    return lock_path


def test_runtime_development_calibration_uses_only_geolife_calibration_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = _trace_lock_fixture(tmp_path)
    lock = load_and_verify_trace_lock(lock_path)
    calibration_ids = {
        segment.scenario_id for segment in lock.segments if segment.split == "calibration"
    }
    forbidden_ids = {
        segment.scenario_id for segment in lock.segments if segment.split != "calibration"
    }
    observed: list[str] = []

    def measure(scenarios, *, max_slots, definition, seed):
        del max_slots, definition, seed
        observed.extend(scenario.scenario_id for scenario in scenarios)
        return (
            [0.001 for _ in scenarios],
            [len(row.timestamps_s) for row in scenarios],
            1,
            1,
            1,
            0,
        )

    monkeypatch.setattr(
        "uav_mec.run_development_calibration._measure_materialized_trace_scenarios",
        measure,
    )
    real_trace_adapter = __import__(
        "uav_mec.run_development_calibration", fromlist=["_RUNTIME_ADAPTERS"]
    )._RUNTIME_ADAPTERS["trace_driven"]
    monkeypatch.setattr(
        "uav_mec.run_development_calibration._RUNTIME_ADAPTERS",
        {
            family: real_trace_adapter if family == "trace_driven" else (lambda **_kwargs: _audit())
            for family in (
                "resource",
                "exact",
                "externality",
                "main",
                "joint",
                "scaling",
                "robustness",
                "trace_driven",
            )
        },
    )

    run_development_calibration(
        REGISTRY,
        DevelopmentCalibrationCase.RUNTIME_CAPACITY,
        (0,),
        tmp_path / "runtime",
        workers=1,
        trace_lock_path=lock_path,
        seed_access=_recorder(tmp_path),
    )

    assert set(observed) == calibration_ids
    assert not forbidden_ids.intersection(observed)


def test_runtime_development_calibration_requires_trace_lock_for_trace_driven(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="trace.lock"):
        run_development_calibration(
            REGISTRY,
            DevelopmentCalibrationCase.RUNTIME_CAPACITY,
            (0,),
            tmp_path / "runtime",
            workers=1,
            trace_lock_path=None,
            seed_access=_recorder(tmp_path),
        )


def test_development_calibration_records_access_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path)

    def exact_adapter(**_kwargs):
        records = recorder.ledger_path.read_text(encoding="utf-8").splitlines()
        assert json.loads(records[0])["operation"] == "development_calibration"
        return ({"complete": True, "cpu_nontrivial": False, "schedule_nontrivial": False},)

    monkeypatch.setattr(
        "uav_mec.run_development_calibration._run_exact_nontriviality",
        exact_adapter,
    )
    run_development_calibration(
        REGISTRY,
        DevelopmentCalibrationCase.EXACT_NONTRIVIALITY,
        (0,),
        tmp_path / "exact",
        workers=1,
        trace_lock_path=None,
        seed_access=recorder,
    )


@pytest.mark.parametrize(
    "arguments",
    (
        ("--case", "bad", "--seeds", "0:1", "--workers", "1"),
        ("--case", "exact_nontriviality", "--seeds", "1:0", "--workers", "1"),
        ("--case", "exact_nontriviality", "--seeds", "0:1", "--workers", "0"),
        ("--case", "exact_nontriviality", "--seeds", "0:1", "--workers", "1", "--trace-lock", "x"),
        ("--case", "runtime_capacity", "--seeds", "0:1", "--workers", "1"),
    ),
)
def test_development_calibration_cli_validates_range_workers_case_and_trace_lock(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as error:
        main(["--registry", str(REGISTRY), "--output", str(output), *arguments])
    assert error.value.code == 2
    assert not output.exists()
