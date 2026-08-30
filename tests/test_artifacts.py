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


def test_streaming_writers_remain_atomic_and_accept_incremental_rows(
    tmp_path: Path,
):
    writer = RunArtifactWriter(tmp_path / "run")

    with writer.open_csv(
        "tasks.csv",
        fieldnames=("task_id", "completed"),
    ) as write_task:
        write_task({"task_id": 1, "completed": True})
        assert not (tmp_path / "run/tasks.csv").exists()
        write_task({"task_id": 2, "completed": False})
    with writer.open_jsonl("events.jsonl") as write_event:
        write_event({"event": "start", "slot": 0})
        assert not (tmp_path / "run/events.jsonl").exists()

    assert (tmp_path / "run/tasks.csv").read_text() == ("task_id,completed\n1,True\n2,False\n")
    assert json.loads((tmp_path / "run/events.jsonl").read_text().strip()) == {
        "event": "start",
        "slot": 0,
    }


def test_streaming_writer_removes_temporary_file_after_failure(
    tmp_path: Path,
):
    writer = RunArtifactWriter(tmp_path / "run")

    with pytest.raises(RuntimeError, match="stop"):
        with writer.open_jsonl("events.jsonl") as write_event:
            write_event({"event": "partial"})
            raise RuntimeError("stop")

    assert not (tmp_path / "run/events.jsonl").exists()
    assert list((tmp_path / "run").glob(".events.jsonl.*")) == []
