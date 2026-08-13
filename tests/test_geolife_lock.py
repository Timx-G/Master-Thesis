from __future__ import annotations

import hashlib
import json
import csv
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_mec.geolife_lock import (
    build_trace_lock,
    load_and_verify_trace_lock,
    main,
)
from uav_mec.experiment import load_definition
from uav_mec.run_trace_experiment import (
    materialize_locked_scenarios,
    run_trace_definition,
)

REGISTRY = Path("configs/thesis/experiment_registry_v2.toml")
_HEADER = (
    "Geolife trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n"
    "0,2,255,My Track,0,0,2,8421376\n0\n"
)


def _write_archive(path: Path) -> str:
    participants = (
        ("000", "walk", 0),
        ("001", "bike", 1),
        ("002", "walk", 2),
        ("003", "bike", 3),
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for participant, mode, minute in participants:
            start = f"2008/10/23 02:{53 + minute:02d}:04"
            end = f"2008/10/23 02:{53 + minute:02d}:14"
            labels = "Start Time\tEnd Time\tTransportation Mode\n" f"{start}\t{end}\t{mode}\n"
            points = "".join(
                f"39.984702,116.{318417 + 60 * index:06d},0,100,39744.12,"
                f"2008-10-23,02:{53 + minute:02d}:{4 + 5 * index:02d}\n"
                for index in range(3)
            )
            archive.writestr(f"Data/{participant}/labels.txt", labels)
            archive.writestr(
                f"Data/{participant}/Trajectory/2008102302{53 + minute:02d}04.plt",
                _HEADER + points,
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_protocol(
    path: Path,
    *,
    train: tuple[str, ...] = ("000",),
    calibration: tuple[str, ...] = ("001",),
    final: tuple[str, ...] = ("002",),
    comment: str = "",
) -> None:
    license_text = "GeoLife research dataset terms; research use only."
    license_sha = hashlib.sha256(license_text.encode()).hexdigest()

    def values(items: tuple[str, ...]) -> str:
        return json.dumps(list(items), separators=(",", ":"))

    path.write_text(
        f"""schema_version = 2
{comment}
[dataset]
version = "GeoLife Trajectories 1.3 synthetic fixture"
source_url = "https://www.microsoft.com/en-us/research/project/geolife/"
retrieved_on = "2026-08-04"
license_text = "{license_text}"
license_sha256 = "{license_sha}"

[selection]
accepted_modes = ["walk", "bike"]
slot_seconds = 1.0
minimum_segment_samples = 3
max_segments_per_participant_mode = 1

[splits]
train = {values(train)}
calibration = {values(calibration)}
final = {values(final)}

[placement]
area_size_m = 500.0
translation_m = [250.0, 250.0]
rotation_deg = 0.0

[[final_groups]]
group_id = "final-group-1"
participants = {values(final)}
""",
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    archive = tmp_path / "geolife.zip"
    protocol = tmp_path / "protocol.toml"
    _write_archive(archive)
    _write_protocol(protocol)
    return archive, protocol


def test_trace_lock_rejects_participant_overlap_across_train_calibration_and_final(
    tmp_path: Path,
) -> None:
    archive, protocol = _inputs(tmp_path)
    _write_protocol(protocol, calibration=("001", "002"), final=("002",))

    with pytest.raises(ValueError, match="participant.*split|split.*overlap"):
        build_trace_lock(archive, protocol, REGISTRY, tmp_path / "lock.json")
    assert not (tmp_path / "lock.json").exists()


def test_trace_lock_rejects_seed_pool_other_than_29000_through_29029(
    tmp_path: Path,
) -> None:
    archive, protocol = _inputs(tmp_path)
    registry = tmp_path / "registry.toml"
    registry.write_text(
        REGISTRY.read_text(encoding="utf-8").replace("seed_start = 29000", "seed_start = 29001"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="29000.*29029|trace.*seed"):
        build_trace_lock(archive, protocol, registry, tmp_path / "lock.json")


def test_trace_lock_records_dataset_source_license_segments_resampling_and_groups(
    tmp_path: Path,
) -> None:
    archive, protocol = _inputs(tmp_path)
    lock = build_trace_lock(archive, protocol, REGISTRY, tmp_path / "lock.json")

    assert lock.dataset_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert lock.source_url.startswith("https://www.microsoft.com/")
    assert lock.license_text
    assert lock.license_sha256 == hashlib.sha256(lock.license_text.encode()).hexdigest()
    assert lock.seeds == tuple(range(29_000, 29_030))
    assert {segment.split for segment in lock.segments} == {
        "train",
        "calibration",
        "final",
    }
    assert all(segment.slot_seconds == 1.0 for segment in lock.segments)
    assert all(
        segment.group_id == f"participant:{segment.participant_id}" for segment in lock.segments
    )
    payload = json.loads((tmp_path / "lock.json").read_text(encoding="utf-8"))
    assert payload["final_groups"] == [{"group_id": "final-group-1", "participants": ["002"]}]
    assert set(payload["samples"]) == {segment.scenario_id for segment in lock.segments}


def test_trace_lock_is_deterministic_for_the_same_archive_and_protocol(tmp_path: Path) -> None:
    archive, protocol = _inputs(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    build_trace_lock(archive, protocol, REGISTRY, first)
    build_trace_lock(archive, protocol, REGISTRY, second)

    assert first.read_bytes() == second.read_bytes()


def test_trace_lock_changes_when_archive_or_protocol_bytes_change(tmp_path: Path) -> None:
    archive, protocol = _inputs(tmp_path)
    first = tmp_path / "first.json"
    build_trace_lock(archive, protocol, REGISTRY, first)

    changed_protocol = tmp_path / "changed.toml"
    _write_protocol(changed_protocol, comment="# byte-level protocol revision")
    second = tmp_path / "second.json"
    build_trace_lock(archive, changed_protocol, REGISTRY, second)

    changed_archive = tmp_path / "changed.zip"
    changed_archive.write_bytes(archive.read_bytes())
    with zipfile.ZipFile(changed_archive, "a") as bundle:
        bundle.comment = b"byte-level archive revision"
    third = tmp_path / "third.json"
    build_trace_lock(changed_archive, protocol, REGISTRY, third)

    assert (
        len({hashlib.sha256(path.read_bytes()).hexdigest() for path in (first, second, third)}) == 3
    )


def test_trace_lock_rejects_archive_mutation_during_single_snapshot_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, protocol = _inputs(tmp_path)
    from uav_mec import geolife

    original = geolife._trajectory_points
    mutated = False

    def mutate_during_parse(bundle, info):
        nonlocal mutated
        result = original(bundle, info)
        if not mutated:
            with archive.open("r+b") as handle:
                handle.seek(-1, 2)
                final = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([final[0] ^ 1]))
            mutated = True
        return result

    monkeypatch.setattr(geolife, "_trajectory_points", mutate_during_parse)
    with pytest.raises(ValueError, match="changed during lock construction"):
        build_trace_lock(archive, protocol, REGISTRY, tmp_path / "lock.json")
    assert not (tmp_path / "lock.json").exists()


def test_exogenous_seed_does_not_change_locked_trace_participants_or_positions(
    tmp_path: Path, development_seed_access
) -> None:
    archive, protocol = _inputs(tmp_path)
    lock_path = tmp_path / "lock.json"
    build_trace_lock(archive, protocol, REGISTRY, lock_path)

    first = materialize_locked_scenarios(
        lock_path, split="final", seed=17, seed_access=development_seed_access
    )
    second = materialize_locked_scenarios(
        lock_path, split="final", seed=18, seed_access=development_seed_access
    )

    assert [row.participant_id for row in first] == [row.participant_id for row in second]
    assert [row.scenario_id for row in first] == [row.scenario_id for row in second]
    assert all(
        np.array_equal(left.positions_enu_m, right.positions_enu_m)
        for left, right in zip(first, second, strict=True)
    )
    assert any(
        not np.array_equal(left.task_profile_draws, right.task_profile_draws)
        for left, right in zip(first, second, strict=True)
    )


def test_trace_runner_refuses_an_unlocked_participant_or_segment(
    tmp_path: Path, development_seed_access
) -> None:
    archive, protocol = _inputs(tmp_path)
    lock_path = tmp_path / "lock.json"
    build_trace_lock(archive, protocol, REGISTRY, lock_path)

    with pytest.raises(ValueError, match="unlocked scenario"):
        materialize_locked_scenarios(
            lock_path,
            split="final",
            seed=19,
            scenario_ids=("not-in-the-lock",),
            seed_access=development_seed_access,
        )
    with pytest.raises(ValueError, match="final"):
        materialize_locked_scenarios(
            lock_path, split="train", seed=19, seed_access=development_seed_access
        )


def test_trace_runner_executes_registered_methods_and_emits_row_provenance(
    tmp_path: Path, development_seed_access
) -> None:
    archive, protocol = _inputs(tmp_path)
    _write_protocol(protocol, final=("002", "003"))
    lock_path = tmp_path / "lock.json"
    build_trace_lock(archive, protocol, REGISTRY, lock_path)
    config = Path("configs/thesis/confirmatory_trace_v2.toml")
    definition = load_definition(config)
    definition = replace(
        definition,
        seeds=(21,),
        max_iterations=20,
        scenario=replace(
            definition.scenario,
            num_users=3,
            num_slots=2,
            n_subcarriers=1,
            task=replace(definition.scenario.task, arrival_prob=1.0),
        ),
    )

    output = run_trace_definition(
        definition,
        config,
        lock_path,
        tmp_path / "run",
        seed_access=development_seed_access,
    )

    with (output / "slots.csv").open(encoding="utf-8", newline="") as handle:
        slots = tuple(csv.DictReader(handle))
    with (output / "tasks.csv").open(encoding="utf-8", newline="") as handle:
        tasks = tuple(csv.DictReader(handle))
    assert {row["method"] for row in slots} == {"full", "exchange_stable"}
    assert len(slots) == 4
    assert {row["scenario_id"] for row in slots} == {"final-group-1"}
    assert tasks
    assert {row["participant_id"] for row in tasks} == {"002", "003"}
    required = {
        "scenario_id",
        "participant_id",
        "source_path",
        "source_start",
        "source_end",
        "trace_lock_sha256",
    }
    assert all(required.issubset(row) and all(row[field] for field in required) for row in slots)
    assert all(required.issubset(row) and all(row[field] for field in required) for row in tasks)


def test_geolife_cli_build_and_verify_match_shared_contract(tmp_path: Path) -> None:
    archive, protocol = _inputs(tmp_path)
    output = tmp_path / "nested" / "lock.json"

    main(
        [
            "build",
            "--archive",
            str(archive),
            "--protocol",
            str(protocol),
            "--registry",
            str(REGISTRY),
            "--output",
            str(output),
        ]
    )
    main(["verify", "--lock", str(output), "--archive", str(archive)])
    assert load_and_verify_trace_lock(output).seeds == tuple(range(29_000, 29_030))


def test_geolife_cli_rejects_symlink_archive_before_creating_output(
    tmp_path: Path,
) -> None:
    archive, protocol = _inputs(tmp_path)
    symlink = tmp_path / "archive-link.zip"
    symlink.symlink_to(archive)
    output = tmp_path / "missing-parent" / "lock.json"

    with pytest.raises(SystemExit) as error:
        main(
            [
                "build",
                "--archive",
                str(symlink),
                "--protocol",
                str(protocol),
                "--registry",
                str(REGISTRY),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert not output.parent.exists()
