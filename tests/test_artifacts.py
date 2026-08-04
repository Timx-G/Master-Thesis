import json
from pathlib import Path

import pytest

from uav_mec.artifacts import RunArtifactWriter, sha256_file


def test_writer_creates_manifest_data_and_checksums(tmp_path: Path):
    writer = RunArtifactWriter(tmp_path / "run")
    writer.write_json("summary.json", {"method": "full", "rate": 80.0})
    writer.write_csv("tasks.csv", [{"task_id": 1, "completed": True}])
    writer.write_jsonl("events.jsonl", [])
    writer.finalize({"git_sha": "abc123", "seeds": [42]})

    assert json.loads((tmp_path / "run/summary.json").read_text())["rate"] == 80.0
    assert (tmp_path / "run/tasks.csv").exists()
    assert (tmp_path / "run/events.jsonl").read_text(encoding="utf-8") == ""
    assert (tmp_path / "run/manifest.json").exists()
    checksum_text = (tmp_path / "run/checksums.sha256").read_text()
    assert sha256_file(tmp_path / "run/summary.json") in checksum_text
    assert "summary.json" in checksum_text
    assert "manifest.json" in checksum_text


def test_writer_refuses_to_overwrite_existing_run(tmp_path: Path):
    path = tmp_path / "run"
    RunArtifactWriter(path)

    with pytest.raises(FileExistsError):
        RunArtifactWriter(path)


def test_writer_rejects_paths_outside_the_run_directory(tmp_path: Path):
    writer = RunArtifactWriter(tmp_path / "run")

    with pytest.raises(ValueError, match="relative"):
        writer.write_json("../escaped.json", {"unsafe": True})
    assert not (tmp_path / "escaped.json").exists()


def test_writer_rejects_nonstandard_nonfinite_json(tmp_path: Path):
    writer = RunArtifactWriter(tmp_path / "run")

    with pytest.raises(ValueError):
        writer.write_json("bad.json", {"value": float("nan")})
    assert not (tmp_path / "run/bad.json").exists()


def test_writer_refuses_duplicate_files_and_post_finalize_mutation(tmp_path: Path):
    writer = RunArtifactWriter(tmp_path / "run")
    writer.write_json("summary.json", {"value": 1})

    with pytest.raises(FileExistsError):
        writer.write_json("summary.json", {"value": 2})

    writer.finalize({"git_sha": "abc123"})
    with pytest.raises(RuntimeError, match="finalized"):
        writer.write_json("late.json", {})
    with pytest.raises(RuntimeError, match="finalized"):
        writer.finalize({"git_sha": "different"})


def test_empty_csv_requires_explicit_fieldnames(tmp_path: Path):
    writer = RunArtifactWriter(tmp_path / "run")

    with pytest.raises(ValueError, match="fieldnames"):
        writer.write_csv("empty.csv", [])
    writer.write_csv("valid-empty.csv", [], fieldnames=["task_id"])

    assert (tmp_path / "run/valid-empty.csv").read_text() == "task_id\n"


def test_artifact_writer_stream_copies_bytes_without_json_reserialization(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"\x00raw\xffbytes\n")
    writer = RunArtifactWriter(tmp_path / "run", bundle_version=2)

    writer.copy_file("nested/source.bin", source)
    writer.finalize({"schema": "v2"})

    assert (tmp_path / "run/nested/source.bin").read_bytes() == source.read_bytes()


def test_v2_artifact_writer_uses_uppercase_manifest_and_sha256sums_closure_names(
    tmp_path: Path,
) -> None:
    writer = RunArtifactWriter(tmp_path / "run", bundle_version=2)
    writer.write_json("summary.json", {"value": 1})
    writer.finalize({"schema": "v2"})

    assert (tmp_path / "run/MANIFEST.json").is_file()
    assert (tmp_path / "run/SHA256SUMS").is_file()
    assert not (tmp_path / "run/manifest.json").exists()
    assert not (tmp_path / "run/checksums.sha256").exists()
    sums = (tmp_path / "run/SHA256SUMS").read_text(encoding="utf-8")
    assert "MANIFEST.json" in sums
    assert "summary.json" in sums
