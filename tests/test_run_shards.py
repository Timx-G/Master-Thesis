from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from uav_mec.analyze_results import verify_checksums
from uav_mec.run_shards import run_sharded_experiment


def _write_config(path: Path, *, methods: str = '"local_only"') -> None:
    path.write_text(
        f'''run_id = "shard-smoke"
seeds = [71, 72]
methods = [{methods}]

[scenario]
num_users = 4
num_slots = 1
num_uavs = 2
num_clusters = 3
n_subcarriers = 2
fixed_cluster_centers = true
random_burst = false
user_mobility = false

[evaluation]
gamma_sic_min = 1.0
sic_residual = 0.02
allocation_policy = "deadline"

[search]
max_iterations = 5
''',
        encoding="utf-8",
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sharded_run_resumes_immutable_seed_bundles_and_merges(tmp_path: Path):
    config = tmp_path / "experiment.toml"
    output = tmp_path / "run"
    _write_config(config)

    first = run_sharded_experiment(config, output, shard_size=1)

    assert first.completed_shards == 2
    assert first.reused_shards == 0
    assert first.merged_path == output / "merged"
    verify_checksums(first.merged_path)
    assert sorted(path.name for path in (output / "shards").iterdir()) == [
        "seed-0000000071-0000000071",
        "seed-0000000072-0000000072",
    ]
    shard_checksums = {
        path.name: _digest(path / "checksums.sha256")
        for path in (output / "shards").iterdir()
    }

    second = run_sharded_experiment(config, output, shard_size=1)

    assert second.completed_shards == 0
    assert second.reused_shards == 2
    assert shard_checksums == {
        path.name: _digest(path / "checksums.sha256")
        for path in (output / "shards").iterdir()
    }


def test_sharded_run_rejects_changed_config_and_tampered_shard(tmp_path: Path):
    config = tmp_path / "experiment.toml"
    output = tmp_path / "run"
    _write_config(config)
    run_sharded_experiment(config, output, shard_size=1)

    _write_config(config, methods='"local_only", "greedy"')
    with pytest.raises(ValueError, match="configuration"):
        run_sharded_experiment(config, output, shard_size=1)

    _write_config(config)
    shard = output / "shards" / "seed-0000000071-0000000071"
    with (shard / "tasks.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        run_sharded_experiment(config, output, shard_size=1)
