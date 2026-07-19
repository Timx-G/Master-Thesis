from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from uav_mec.experiment_registry import (
    load_registry,
    registered_family,
    validate_registered_definition,
)


REGISTRY_PATH = Path("configs/thesis/experiment_registry_v2.toml")


def _definition_for(family, **changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "seeds": family.seeds,
        "methods": family.methods,
        "scenario": SimpleNamespace(num_slots=family.num_slots),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_v2_registry_freezes_approved_candidate_pools_and_all_eight_families() -> None:
    registry = load_registry(REGISTRY_PATH)

    assert registry.development_seeds == tuple(range(0, 1_000))
    assert registry.pilot_seeds == tuple(range(1_000, 1_100))
    assert registry.candidate_sample_sizes == (50, 100, 200)
    assert tuple(family.name for family in registry.families) == (
        "resource",
        "exact",
        "externality",
        "main",
        "joint",
        "scaling",
        "robustness",
        "trace_driven",
    )
    assert {
        family.name: (
            family.seed_start,
            family.seed_count,
            family.num_slots,
            family.formal_order,
        )
        for family in registry.families
    } == {
        "resource": (22_000, 200, 100, 1),
        "exact": (26_000, 50, 20, 2),
        "externality": (23_000, 200, 100, 3),
        "main": (24_000, 200, 100, 4),
        "joint": (27_000, 200, 100, 5),
        "scaling": (28_000, 30, 50, 6),
        "robustness": (25_000, 30, 50, 7),
        "trace_driven": (29_000, 30, 50, 8),
    }


def test_v2_registry_freezes_externality_descriptive_n_at_thirty() -> None:
    registry = load_registry(REGISTRY_PATH)

    assert registry.descriptive_externality_n == 30
    externality = registered_family(registry, "externality")
    assert [case.fixed_sample_size for case in externality.cases] == [
        30,
        30,
        30,
        30,
        30,
        30,
        30,
        None,
        30,
    ]


def test_v2_registry_freezes_every_case_id_role_and_sample_policy() -> None:
    registry = load_registry(REGISTRY_PATH)

    expected = {
        "resource": (
            "power_selected",
            (
                ("dd_generated", "primary", None),
                ("radio_neutral", "supporting", None),
                ("deadline_first_generated", "supporting", None),
            ),
        ),
        "exact": (
            "fixed",
            (("exact_schedule", "correctness", 50),),
        ),
        "externality": (
            "primary_power_selected_supporting_fixed",
            (
                ("overlap_low_coupling0_load08", "supporting", 30),
                ("overlap_low_coupling0_load12", "supporting", 30),
                ("overlap_low_coupling1_load08", "supporting", 30),
                ("overlap_low_coupling1_load12", "supporting", 30),
                ("overlap_high_coupling0_load08", "supporting", 30),
                ("overlap_high_coupling0_load12", "supporting", 30),
                ("overlap_high_coupling1_load08", "supporting", 30),
                ("overlap_high_coupling1_load12", "primary", None),
                ("overlap_medium_coupling05_load10", "supporting", 30),
            ),
        ),
        "main": (
            "power_selected",
            (("main_k3_n45", "primary", None),),
        ),
        "joint": (
            "power_selected",
            (("joint_k3_n45", "primary", None),),
        ),
        "scaling": (
            "fixed",
            tuple(
                (case_id, "supporting", 30)
                for case_id in (
                    "load_k3_n30",
                    "load_k3_n45",
                    "load_k3_n60",
                    "equal_k2_n30",
                    "equal_k3_n45",
                    "equal_k4_n60",
                    "equal_k5_n75",
                )
            ),
        ),
        "robustness": (
            "fixed",
            tuple(
                (case_id, "supporting", 30)
                for case_id in (
                    "csi0",
                    "csi05",
                    "csi10",
                    "csi20",
                    "sic_residual0",
                    "sic_residual02",
                    "sic_residual05",
                    "sic_residual10",
                    "sic_threshold_m3",
                    "sic_threshold0",
                    "sic_threshold3",
                    "sic_threshold5",
                    "cpu_homogeneous",
                    "cpu_heterogeneous",
                    "deadline_light",
                    "deadline_default",
                    "deadline_heavy",
                    "arrival_periodic",
                    "arrival_jitter",
                    "arrival_burst",
                    "pathloss_al_hourani",
                    "pathloss_3gpp_uma_av_audited",
                )
            ),
        ),
        "trace_driven": (
            "fixed",
            (("geolife_final", "supporting", 30),),
        ),
    }
    assert {
        family.name: (
            family.sample_size_policy,
            tuple(
                (case.case_id, case.role, case.fixed_sample_size)
                for case in family.cases
            ),
        )
        for family in registry.families
    } == expected


def test_registry_declares_disjoint_development_pilot_and_formal_pools() -> None:
    registry = load_registry(REGISTRY_PATH)

    named_pools = {
        "development": set(registry.development_seeds),
        "pilot": set(registry.pilot_seeds),
        **{family.name: set(family.seeds) for family in registry.families},
    }
    names = tuple(named_pools)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            assert named_pools[left].isdisjoint(named_pools[right]), (left, right)


def test_registry_links_each_family_to_one_matching_config() -> None:
    registry = load_registry(REGISTRY_PATH)

    assert len({family.config_path for family in registry.families}) == 8
    for family in registry.families:
        assert family.config_path.is_absolute()
        assert family.config_path.is_file()
        assert family.config_path.name not in {
            "confirmatory_main.toml",
            "confirmatory_pairing.toml",
            "confirmatory_resource.toml",
        }
        validate_registered_definition(
            registry,
            family.name,
            _definition_for(family),
        )

    h2_raw = tomllib.loads(
        registered_family(registry, "externality").config_path.read_text(
            encoding="utf-8"
        )
    )
    assert h2_raw["scenario"]["num_uavs"] == 3
    assert h2_raw["scenario"]["num_users"] == 45
    assert h2_raw["scenario"]["n_subcarriers"] == 3
    assert h2_raw["scenario"]["cluster_layout"] == "pairwise_overlap"
    assert h2_raw["scenario"]["cluster_std"] == 5.0
    assert h2_raw["scenario"]["task"]["period_choices"] == [2]
    assert h2_raw["evaluation"]["cross_interference_scale"] == 1.0


def test_registry_primary_hypotheses_freeze_test_estimand_and_holm_family() -> None:
    registry = load_registry(REGISTRY_PATH)

    assert tuple(hypothesis.hypothesis_id for hypothesis in registry.hypotheses) == (
        "H1",
        "H2",
        "H3",
        "H4",
    )
    assert {
        hypothesis.hypothesis_id: (
            hypothesis.family,
            hypothesis.case_id,
            hypothesis.treatment,
            hypothesis.control,
            hypothesis.mde_pp,
        )
        for hypothesis in registry.hypotheses
    } == {
        "H1": ("main", "main_k3_n45", "full", "intra_global", 0.5),
        "H2": (
            "externality",
            "overlap_high_coupling1_load12",
            "full",
            "cross_bilateral",
            0.5,
        ),
        "H3": (
            "resource",
            "dd_generated",
            "fixed_radio_dd",
            "fixed_radio_edf",
            2.0,
        ),
        "H4": ("joint", "joint_k3_n45", "iea_dd", "exchange_edf", 2.0),
    }
    assert {hypothesis.outcome for hypothesis in registry.hypotheses} == {
        "seed_episode_on_time_completion_rate"
    }
    assert {hypothesis.estimand for hypothesis in registry.hypotheses} == {
        "mean_paired_difference_pp"
    }
    assert {hypothesis.primary_test for hypothesis in registry.hypotheses} == {
        "paired_t"
    }
    assert {hypothesis.sensitivity_test for hypothesis in registry.hypotheses} == {
        "two_sided_wilcoxon"
    }
    assert {hypothesis.multiplicity_family for hypothesis in registry.hypotheses} == {
        "primary_h1_h4"
    }


def test_registry_rejects_config_seed_method_or_slot_drift() -> None:
    registry = load_registry(REGISTRY_PATH)
    family = registered_family(registry, "main")

    with pytest.raises(ValueError, match="main.seeds"):
        validate_registered_definition(
            registry,
            "main",
            _definition_for(family, seeds=family.seeds[:-1]),
        )
    with pytest.raises(ValueError, match="main.methods"):
        validate_registered_definition(
            registry,
            "main",
            _definition_for(family, methods=family.methods[:-1]),
        )
    with pytest.raises(ValueError, match="main.num_slots"):
        validate_registered_definition(
            registry,
            "main",
            _definition_for(
                family,
                scenario=SimpleNamespace(num_slots=family.num_slots - 1),
            ),
        )
    with pytest.raises(KeyError, match="unregistered"):
        registered_family(registry, "unregistered")
