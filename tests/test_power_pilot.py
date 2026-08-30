from __future__ import annotations

import json
from pathlib import Path

import pytest

from uav_mec.analyze_results import verify_checksums
from uav_mec.artifacts import RunArtifactWriter
from uav_mec.experiment_registry import load_registry
from uav_mec.run_power_pilot import (
    PILOT_SEEDS,
    load_pilot_differences,
    write_power_pilot_analysis,
)
from uav_mec.run_power_pilot_suite import run_power_pilot_suite


REGISTRY_PATH = Path("configs/thesis/experiment_registry_v2.toml")


def _pilot_bundle(
    root: Path,
    family: str,
    treatment: str,
    control: str,
) -> Path:
    path = root / family
    writer = RunArtifactWriter(path)
    control_values = [0.40 + index * 1e-5 for index in range(len(PILOT_SEEDS))]
    treatment_values = [value + 0.01 for value in control_values]
    writer.write_json(
        "summary.json",
        {
            "run_id": f"pilot-{family}",
            "seeds": list(PILOT_SEEDS),
            "initializer_status_counts": {},
            "search_termination_counts": {},
            "methods": {
                treatment: {"on_time_rate_by_seed": treatment_values},
                control: {"on_time_rate_by_seed": control_values},
            },
        },
    )
    writer.finalize(
        {
            "run_id": f"pilot-{family}",
            "seeds": list(PILOT_SEEDS),
        }
    )
    return path


def _all_bundles(root: Path) -> dict[str, Path]:
    registry = load_registry(REGISTRY_PATH)
    return {
        hypothesis.family: _pilot_bundle(
            root,
            hypothesis.family,
            hypothesis.treatment,
            hypothesis.control,
        )
        for hypothesis in registry.hypotheses
    }


def test_power_pilot_loads_seed_paired_percentage_point_differences(tmp_path: Path):
    registry = load_registry(REGISTRY_PATH)
    differences, input_hashes = load_pilot_differences(
        registry,
        _all_bundles(tmp_path),
    )

    assert set(differences) == {"H1", "H2", "H3", "H4"}
    assert all(len(values) == 100 for values in differences.values())
    assert all(values == pytest.approx([1.0] * 100) for values in differences.values())
    assert set(input_hashes) == {"main", "externality", "resource", "joint"}
    assert all(len(digest) == 64 for digest in input_hashes.values())


def test_power_pilot_writes_a_hashed_non_overwriting_decision_bundle(tmp_path: Path):
    bundles = _all_bundles(tmp_path / "inputs")
    output = tmp_path / "decision"

    write_power_pilot_analysis(
        REGISTRY_PATH,
        bundles,
        output,
        simulations=50,
        analysis_seed=123,
    )

    verify_checksums(output)
    decision = json.loads((output / "power_decision.json").read_text())
    assert decision["selected_n"] == 50
    assert decision["pilot_seeds"] == [1000, 1001, *range(1002, 1100)]
    assert decision["simulations"] == 50
    assert all(
        power >= 0.8
        for power in decision["power_by_hypothesis"].values()
    )
    with pytest.raises(FileExistsError):
        write_power_pilot_analysis(
            REGISTRY_PATH,
            bundles,
            output,
            simulations=50,
            analysis_seed=123,
        )


def test_power_pilot_rejects_an_incomplete_seed_pool(tmp_path: Path):
    registry = load_registry(REGISTRY_PATH)
    bundles = _all_bundles(tmp_path)
    main = bundles["main"]
    summary = json.loads((main / "summary.json").read_text())
    summary["seeds"] = summary["seeds"][:-1]
    (main / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="pilot seeds|checksum"):
        load_pilot_differences(registry, bundles)


def test_power_pilot_suite_runs_all_families_then_freezes_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    class Report:
        def __init__(self, merged_path):
            self.merged_path = merged_path

    def fake_sharded(config, output, *, shard_size):
        family = Path(config).stem.removeprefix("pilot_").removesuffix("_v2")
        merged = Path(output) / "merged"
        calls.append(("run", family, shard_size))
        return Report(merged)

    def fake_analysis(registry, bundles, output, *, simulations, analysis_seed):
        calls.append(
            (
                "analyze",
                tuple(sorted(bundles)),
                simulations,
                analysis_seed,
                Path(output).name,
            )
        )

    monkeypatch.setattr(
        "uav_mec.run_power_pilot_suite.run_sharded_experiment",
        fake_sharded,
    )
    monkeypatch.setattr(
        "uav_mec.run_power_pilot_suite.write_power_pilot_analysis",
        fake_analysis,
    )

    run_power_pilot_suite(
        tmp_path / "suite",
        shard_size=1,
        simulations=2000,
        analysis_seed=17,
    )

    assert calls[:4] == [
        ("run", "main", 1),
        ("run", "externality", 1),
        ("run", "resource_fixed", 1),
        ("run", "joint", 1),
    ]
    assert calls[4] == (
        "analyze",
        ("externality", "joint", "main", "resource"),
        2000,
        17,
        "decision",
    )
