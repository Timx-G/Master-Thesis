from pathlib import Path

import pytest

from uav_mec.experiment_registry import load_registry


def test_v2_registry_freezes_disjoint_seed_pools_and_main_hypotheses():
    registry = load_registry(Path("configs/thesis/experiment_registry_v2.toml"))

    pools = {
        family.name: (family.seed_start, family.seed_count, family.num_slots)
        for family in registry.families
    }
    assert pools == {
        "resource": (12000, 200, 100),
        "externality": (13000, 200, 100),
        "main": (14000, 200, 100),
        "robustness": (15000, 30, 50),
        "exact": (16000, 50, 20),
        "joint": (17000, 200, 100),
        "scaling": (18000, 30, 50),
    }
    all_seeds = [seed for family in registry.families for seed in family.seeds]
    assert len(all_seeds) == len(set(all_seeds))
    assert tuple(hypothesis.hypothesis_id for hypothesis in registry.hypotheses) == (
        "H1",
        "H2",
        "H3",
        "H4",
    )
    assert tuple(hypothesis.mde_pp for hypothesis in registry.hypotheses) == (
        0.5,
        0.5,
        2.0,
        2.0,
    )
    hypothesis_by_id = {
        hypothesis.hypothesis_id: hypothesis
        for hypothesis in registry.hypotheses
    }
    assert hypothesis_by_id["H3"].treatment == "fixed_radio_dd"
    assert hypothesis_by_id["H3"].control == "fixed_radio_edf"


def test_registry_rejects_overlapping_seed_pools(tmp_path: Path):
    path = tmp_path / "overlap.toml"
    path.write_text(
        """
candidate_sample_sizes = [50, 100, 200]

[[families]]
name = "one"
seed_start = 100
seed_count = 10
num_slots = 1
methods = ["full"]

[[families]]
name = "two"
seed_start = 109
seed_count = 10
num_slots = 1
methods = ["full"]

[[hypotheses]]
hypothesis_id = "H1"
family = "one"
treatment = "full"
control = "full"
mde_pp = 0.5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap"):
        load_registry(path)


def test_registry_rejects_unknown_methods(tmp_path: Path):
    path = tmp_path / "unknown.toml"
    path.write_text(
        """
candidate_sample_sizes = [50, 100, 200]

[[families]]
name = "one"
seed_start = 100
seed_count = 10
num_slots = 1
methods = ["not-a-method"]

[[hypotheses]]
hypothesis_id = "H1"
family = "one"
treatment = "not-a-method"
control = "full"
mde_pp = 0.5
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown method"):
        load_registry(path)
