import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.experiment import (
    _slot_method,
    load_definition,
    method_spec_sha256,
    method_spec_to_dict,
    run_definition,
    thesis_method_specs,
)
from uav_mec.model import ScenarioConfig
from uav_mec.resource_allocation import AllocationPolicy
from uav_mec.run_thesis_suite import (
    BASE_PATH,
    build_registered_case_definitions,
    build_suite,
)
from uav_mec.trace import generate_trace


def test_confirmatory_seed_range_is_frozen_and_unseen():
    exploratory = load_definition(Path("configs/thesis/exploratory.toml"))
    confirmatory = load_definition(Path("configs/thesis/confirmatory_main.toml"))

    assert exploratory.seeds == tuple(range(42, 62))
    assert confirmatory.seeds == tuple(range(1000, 1050))
    assert set(exploratory.seeds).isdisjoint(confirmatory.seeds)


def test_pairing_ablation_changes_only_topology_and_acceptance():
    definition = load_definition(Path("configs/thesis/confirmatory_pairing.toml"))

    assert definition.methods == (
        "intra_bilateral",
        "cross_bilateral",
        "intra_global",
        "full",
    )


def test_resource_ablation_freezes_schedule_search_policy(
    development_seed_access,
):
    definition = load_definition(Path("configs/thesis/confirmatory_resource.toml"))
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
    search_traces = []
    search_signatures = []
    executable_signatures = {}
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
        events = []
        outcome = _slot_method(
            method,
            definition.scenario,
            target,
            definition.max_iterations,
            events,
        )(snapshot)
        search_traces.append(events)
        termination = [event for event in events if event["move_type"] == "termination"]
        assert len(termination) == 1
        assert termination[0]["raw_move_count"] >= 0
        assert termination[0]["unique_closed_candidate_count"] >= 0
        assert termination[0]["cache_hits"] >= 0
        assert termination[0]["cache_hits"] <= termination[0]["raw_move_count"]
        search_signatures.append(termination[0]["schedule_key"])
        executable_signatures[method] = tuple(
            (record.task_id, record.serving_uav, record.subcarrier) for record in outcome.rates
        )

    assert search_traces[1:] == [search_traces[0]] * (len(search_traces) - 1)
    assert len(set(search_signatures)) == 1

    searched_assignment = {
        (task_id, serving_uav, subcarrier)
        for serving_uav, subcarrier, _user_id, task_id, _role in (search_signatures[0])
    }
    assert all(
        set(signature) <= searched_assignment for signature in executable_signatures.values()
    )
    assert set(executable_signatures["full_fixed_half"]) < set(executable_signatures["full"])
    assert executable_signatures["full_crossing"] == executable_signatures["full"]


def test_guo_adaptation_uses_the_common_closure_and_replay_path() -> None:
    tasks = tuple(Task(index, index, 0, 600_000.0, 1.2e9, 0.8, 0.4e9) for index in range(4))
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array(
            [
                [9e-8, 2e-8],
                [6e-8, 4e-8],
                [3e-8, 8e-8],
                [2e-8, 7e-8],
            ]
        ),
        2,
    )
    scenario = ScenarioConfig(num_uavs=2, n_subcarriers=1)
    evaluation = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.1,
        cpu_per_uav_hz=4e9,
    )
    events: list[dict[str, object]] = []

    outcome = _slot_method(
        "guo_jagp_adapted",
        scenario,
        evaluation,
        20,
        events,
    )(snapshot)

    assert outcome.slot_index == snapshot.slot_index
    assert outcome.plan_key is not None
    assert events == []


def test_guo_adaptation_projects_raw_grouping_through_canonical_sic_repair(
    development_seed_access,
) -> None:
    definition = load_definition(Path("configs/thesis/confirmatory_main_v2.toml"))
    snapshot = generate_trace(
        replace(definition.scenario, num_slots=1),
        0,
        seed_access=development_seed_access,
    ).slots[0]
    outcome = _slot_method(
        "guo_jagp_adapted",
        definition.scenario,
        definition.evaluation,
        definition.max_iterations,
        [],
    )(snapshot)
    assert all(record.sic_feasible for record in outcome.rates)


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


def test_registered_externality_cases_resolve_explicit_overlap_load_and_coupling() -> None:
    cases = {
        case.name: case.definition for case in build_registered_case_definitions("externality")
    }
    assert len(cases) == 9
    low = cases["overlap_low_coupling0_load08"]
    assert low.scenario.cluster_layout == "pairwise_overlap"
    assert low.scenario.cluster_std == 150.0
    assert low.scenario.task.period_choices == (3,)
    assert low.evaluation.cross_interference_scale == 0.0
    center = cases["overlap_medium_coupling05_load10"]
    assert center.scenario.cluster_std == 30.0
    assert center.scenario.task.period_choices == (2, 3, 3)
    assert center.evaluation.cross_interference_scale == 0.5
    primary = cases["overlap_high_coupling1_load12"]
    assert primary.scenario.cluster_std == 5.0
    assert primary.scenario.task.period_choices == (2,)
    assert primary.evaluation.cross_interference_scale == 1.0


def test_registered_scaling_cases_resolve_exact_k_n_matrix() -> None:
    cases = {case.name: case.definition for case in build_registered_case_definitions("scaling")}
    assert tuple(cases) == (
        "load_k3_n30",
        "load_k3_n45",
        "load_k3_n60",
        "equal_k2_n30",
        "equal_k3_n45",
        "equal_k4_n60",
        "equal_k5_n75",
    )
    assert (cases["equal_k5_n75"].scenario.num_uavs, cases["equal_k5_n75"].scenario.num_users) == (
        5,
        75,
    )


def test_registered_robustness_cases_resolve_locked_one_axis_parameters() -> None:
    cases = {
        case.name: case.definition for case in build_registered_case_definitions("robustness")
    }

    assert tuple(cases) == (
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
    assert [cases[name].csi_error_std for name in ("csi0", "csi05", "csi10", "csi20")] == [
        0.0,
        0.05,
        0.10,
        0.20,
    ]
    assert cases["sic_residual10"].evaluation.sic_residual == 0.10
    assert cases["sic_threshold_m3"].evaluation.gamma_sic_min == pytest.approx(10**-0.3)
    assert cases["sic_threshold5"].evaluation.gamma_sic_min == pytest.approx(10**0.5)
    assert cases["cpu_homogeneous"].evaluation.cpu_per_uav_hz == (4e9, 4e9, 4e9)
    assert cases["cpu_heterogeneous"].evaluation.cpu_per_uav_hz == (2e9, 4e9, 6e9)
    assert sum(cases["cpu_homogeneous"].evaluation.cpu_per_uav_hz) == sum(
        cases["cpu_heterogeneous"].evaluation.cpu_per_uav_hz
    )
    assert [
        cases[name].scenario.task.t_max
        for name in ("deadline_light", "deadline_default", "deadline_heavy")
    ] == [1.0, 0.8, 0.6]
    periodic = cases["arrival_periodic"].scenario
    jitter = cases["arrival_jitter"].scenario
    burst = cases["arrival_burst"].scenario
    assert (periodic.task.arrival_mode, periodic.task.period_choices, periodic.task.period_jitter_slots) == (
        "periodic",
        (2, 2, 3),
        0,
    )
    assert (jitter.task.arrival_mode, jitter.task.period_choices, jitter.task.period_jitter_slots) == (
        "periodic",
        (2, 2, 3),
        1,
    )
    assert burst.task.arrival_mode == "bernoulli"
    assert burst.random_burst
    assert (burst.burst_rate, burst.cool_rate, burst.burst_period_min, burst.burst_period_max) == (
        0.9,
        0.2,
        8,
        15,
    )
    assert cases["pathloss_al_hourani"].scenario.channel.pathloss_model == "al_hourani"
    assert (
        cases["pathloss_3gpp_uma_av_audited"].scenario.channel.pathloss_model
        == "3gpp_uma_av_audited"
    )


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

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    resolved = manifest["resolved_definition"]
    assert resolved["seeds"] == [42, 43]
    assert resolved["scenario"]["num_slots"] == 2
    value = resolved
    for key in value_path:
        value = value[key]
    assert value == expected
    specs = thesis_method_specs()
    for method_id in definition.methods:
        serialized = method_spec_to_dict(specs[method_id])
        factor_hash = method_spec_sha256(specs[method_id])
        assert manifest["method_specs"][method_id] == {
            "factor_hash": factor_hash,
            "spec": serialized,
        }
        assert summary["methods"][method_id]["method_spec"] == serialized
        assert summary["methods"][method_id]["method_spec_sha256"] == factor_hash


@pytest.mark.parametrize(
    "method_id",
    ("fixed_radio_dd", "exact_admission", "exact"),
)
def test_generic_runner_rejects_dedicated_methods_before_seed_recording(
    tmp_path: Path,
    development_seed_access,
    method_id: str,
) -> None:
    config_path = Path("configs/thesis/confirmatory_main.toml")
    base = load_definition(config_path)
    definition = replace(
        base,
        seeds=(22_000,),
        methods=(method_id,),
    )
    ledger = development_seed_access.ledger_path
    before = ledger.read_bytes() if ledger.exists() else b""
    output = tmp_path / method_id

    with pytest.raises(ValueError, match="dedicated"):
        run_definition(
            definition,
            config_path,
            output,
            seed_access=development_seed_access,
        )

    after = ledger.read_bytes() if ledger.exists() else b""
    assert after == before
    assert not output.exists()
