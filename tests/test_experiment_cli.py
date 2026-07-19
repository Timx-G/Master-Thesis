import csv
import hashlib
import json
from pathlib import Path

import pytest

from uav_mec.experiment import load_definition, run_experiment


def test_smoke_experiment_writes_complete_bundle(tmp_path: Path):
    config_path = Path("configs/thesis/smoke.toml")
    output = tmp_path / "smoke-run"

    run_experiment(config_path, output)

    expected_files = {
        "checksums.sha256",
        "events.jsonl",
        "manifest.json",
        "slots.csv",
        "summary.json",
        "tasks.csv",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["methods"]) == {"local_only", "greedy", "full"}
    assert summary["seeds"] == [42, 43]
    assert all(
        len(values["on_time_rate_by_seed"]) == 2
        for values in summary["methods"].values()
    )
    with (output / "slots.csv").open(newline="") as handle:
        slot_rows = list(csv.DictReader(handle))
    assert len(slot_rows) == 3 * 2 * 3
    assert all(int(row["sic_violations"]) == 0 for row in slot_rows)
    assert all(
        float(row["aggregate_victim_link_exposure_w"])
        == pytest.approx(float(row["cross_interference_w"]))
        for row in slot_rows
    )
    assert all(
        "mean_aggregate_victim_link_exposure_w_by_seed" in values
        for values in summary["methods"].values()
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config_sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    checksum_lines = (output / "checksums.sha256").read_text().splitlines()
    assert len(checksum_lines) == 5


@pytest.mark.parametrize(
    ("seeds", "methods", "message"),
    (
        ("[]", '["full"]', "seeds"),
        ("[42, 42]", '["full"]', "seeds"),
        ("[42]", '["full", "full"]', "methods"),
        ("[42]", '["unknown"]', "unknown method"),
    ),
)
def test_definition_rejects_ambiguous_experiment_axes(
    tmp_path: Path,
    seeds: str,
    methods: str,
    message: str,
):
    config = tmp_path / "invalid.toml"
    config.write_text(
        f"""
run_id = "invalid"
seeds = {seeds}
methods = {methods}

[scenario]
num_users = 2
num_slots = 1
num_uavs = 1
num_clusters = 3
n_subcarriers = 1

[evaluation]
gamma_sic_min = 1.0
sic_residual = 0.02
allocation_policy = "deadline"

[search]
max_iterations = 2
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_definition(config)
