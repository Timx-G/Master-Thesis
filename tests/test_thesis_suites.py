import json
from dataclasses import replace
from pathlib import Path

import pytest

from uav_mec.experiment import _slot_method, load_definition, run_definition
from uav_mec.resource_allocation import AllocationPolicy
from uav_mec.run_thesis_suite import BASE_PATH, build_suite
from uav_mec.trace import generate_trace


def test_confirmatory_seed_range_is_frozen_and_unseen():
    exploratory = load_definition(Path("configs/thesis/exploratory.toml"))
    confirmatory = load_definition(
        Path("configs/thesis/confirmatory_main.toml")
    )

    assert exploratory.seeds == tuple(range(42, 62))
    assert confirmatory.seeds == tuple(range(1000, 1050))
    assert set(exploratory.seeds).isdisjoint(confirmatory.seeds)


def test_pairing_ablation_changes_only_topology_and_acceptance():
    definition = load_definition(
        Path("configs/thesis/confirmatory_pairing.toml")
    )

    assert definition.methods == (
        "intra_bilateral",
        "cross_bilateral",
        "intra_global",
        "full",
    )


def test_resource_ablation_freezes_schedule_search_policy(
    development_seed_access,
):
    definition = load_definition(
        Path("configs/thesis/confirmatory_resource.toml")
    )
    assert definition.methods == (
        "full_fixed_half",
        "full_crossing",
        "full",
    )
    snapshot = generate_trace(
        definition.scenario,
        42,
        seed_access=development_seed_access,
    ).slots[0]
    signatures = []
    for method in definition.methods:
        target = definition.evaluation
        if method == "full_fixed_half":
            target = replace(
                target,
                allocation_policy=AllocationPolicy.FIXED_HALF_EQUAL,
            )
        elif method == "full_crossing":
            target = replace(
                target,
                allocation_policy=AllocationPolicy.CROSSING_EQUAL,
            )
        outcome = _slot_method(
            method,
            definition.scenario,
            target,
            definition.max_iterations,
            [],
        )(snapshot)
        signatures.append(
            tuple(
                (record.task_id, record.serving_uav, record.subcarrier)
                for record in outcome.rates
            )
        )

    assert len(set(signatures)) == 1


def test_scalability_and_robustness_cases_are_complete():
    scalability = build_suite("scalability")
    robustness = build_suite("robustness")

    assert {case.name for case in scalability} == {
        "n10",
        "n15",
        "n20",
        "n25",
        "n30",
        "n40",
        "n50",
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
    }
    assert {
        "nf3",
        "nf7",
        "nf10",
        "csi0",
        "csi05",
        "csi10",
        "csi20",
        "sicm3",
        "sic0",
        "sic3",
        "sic5",
        "cpu2",
        "cpu4",
        "cpu8",
        "t06",
        "t08",
        "t10",
        "load_light",
        "load_default",
        "load_heavy",
    }.issubset({case.name for case in robustness})


@pytest.mark.parametrize(
    ("suite_name", "value_path", "expected"),
    (
        ("mechanism", ("evaluation", "cross_interference_scale"), 0.0),
        ("scalability", ("scenario", "num_users"), 10),
        (
            "robustness",
            ("scenario", "channel", "noise_figure_db"),
            3.0,
        ),
    ),
)
def test_suite_case_executes_with_resolved_manifest(
    tmp_path: Path,
    suite_name: str,
    value_path: tuple[str, ...],
    expected: object,
    development_seed_access,
):
    case = build_suite(suite_name)[0]
    definition = replace(
        case.definition,
        seeds=(42, 43),
        scenario=replace(case.definition.scenario, num_slots=2),
    )
    output = tmp_path / suite_name

    run_definition(
        definition,
        BASE_PATH,
        output,
        seed_access=development_seed_access,
    )

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    resolved = manifest["resolved_definition"]
    assert resolved["seeds"] == [42, 43]
    assert resolved["scenario"]["num_slots"] == 2
    value = resolved
    for key in value_path:
        value = value[key]
    assert value == expected
