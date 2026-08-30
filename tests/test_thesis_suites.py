import json
from dataclasses import replace
from pathlib import Path

import pytest

from uav_mec.experiment import _slot_method, load_definition, run_definition
from uav_mec.resource_allocation import AllocationPolicy
from uav_mec import experiment as experiment_module
from uav_mec import run_thesis_suite as thesis_suites
from uav_mec.artifacts import RunArtifactWriter
from uav_mec.run_thesis_suite import (
    BASE_PATH,
    EXTERNALITY_BASE_PATH,
    JOINT_BASE_PATH,
    build_suite,
)
from uav_mec.trace import generate_trace


def test_confirmatory_seed_range_is_frozen_and_unseen():
    exploratory = load_definition(Path("configs/thesis/exploratory.toml"))
    confirmatory = load_definition(Path("configs/thesis/confirmatory_main.toml"))

    assert exploratory.seeds == tuple(range(42, 62))
    assert confirmatory.seeds == tuple(range(1000, 1050))
    assert set(exploratory.seeds).isdisjoint(confirmatory.seeds)


def test_reality_calibrated_main_v2_is_frozen():
    definition = load_definition(Path("configs/thesis/confirmatory_main_v2.toml"))

    assert definition.seeds == tuple(range(14000, 14200))
    assert definition.scenario.num_uavs == 3
    assert definition.scenario.num_users == 45
    assert definition.scenario.num_clusters == 3
    assert definition.scenario.n_subcarriers == 3
    assert definition.scenario.f_per_uav == 4e9
    assert definition.scenario.task.data_min == 0.8e6
    assert definition.scenario.task.data_max == 3.0e6
    assert definition.scenario.task.comp_min == 800.0
    assert definition.scenario.task.comp_max == 1500.0
    assert definition.scenario.task.t_min == 0.6
    assert definition.scenario.task.t_max == 1.0
    assert not definition.scenario.random_burst
    assert definition.scenario.task.arrival_mode == "periodic"
    assert definition.scenario.task.period_choices == (2, 2, 3)
    assert definition.scenario.task.period_jitter_slots == 0
    assert definition.event_log_level == "summary"
    assert definition.multistart_workers == 3
    assert definition.methods == (
        "local_only",
        "greedy",
        "ia_greedy",
        "ilp",
        "exchange_stable",
        "intra_global",
        "cross_bilateral",
        "full",
        "fair_ofdma",
    )


def test_fixed_radio_resource_v2_is_frozen_for_h3():
    definition = load_definition(
        Path("configs/thesis/confirmatory_resource_fixed_v2.toml")
    )

    assert definition.seeds == tuple(range(12000, 12200))
    assert definition.scenario.num_uavs == 3
    assert definition.scenario.num_users == 45
    assert definition.scenario.num_slots == 100
    assert definition.scenario.f_per_uav == 4e9
    assert definition.scenario.task.arrival_mode == "periodic"
    assert definition.scenario.task.period_choices == (2, 2, 3)
    assert definition.methods == (
        "fixed_radio_dd",
        "fixed_radio_edf",
        "fixed_radio_min_slack",
        "fixed_radio_exact",
    )


@pytest.mark.parametrize(
    "config_name",
    (
        "pilot_main_v2.toml",
        "pilot_externality_v2.toml",
        "pilot_resource_fixed_v2.toml",
        "pilot_joint_v2.toml",
    ),
)
def test_power_pilot_definitions_use_only_the_reserved_pilot_pool(config_name):
    definition = load_definition(Path("configs/thesis") / config_name)

    assert definition.seeds == tuple(range(1000, 1100))
    assert definition.scenario.num_slots == 100
    assert definition.event_log_level == "summary"


def test_pairing_ablation_changes_only_topology_and_acceptance():
    definition = load_definition(Path("configs/thesis/confirmatory_pairing.toml"))

    assert definition.methods == (
        "intra_bilateral",
        "cross_bilateral",
        "intra_global",
        "full",
    )


def test_joint_suite_freezes_the_two_by_two_embedded_resource_design():
    definition = load_definition(JOINT_BASE_PATH)
    cases = build_suite("joint")

    assert definition.seeds == tuple(range(17000, 17200))
    assert definition.methods == (
        "exchange_edf",
        "exchange_dd",
        "iea_edf",
        "iea_dd",
    )
    assert len(cases) == 1
    assert cases[0].name == "joint_k3_n45"
    assert cases[0].definition == definition


def test_canonical_resource_ablation_remains_frozen():
    definition = load_definition(Path("configs/thesis/confirmatory_resource.toml"))
    assert definition.seeds == tuple(range(1000, 1050))
    assert definition.methods == (
        "full_fixed_half",
        "full_crossing",
        "full",
    )


def test_strong_resource_comparison_embeds_each_policy_in_search():
    definition = load_definition(Path("configs/thesis/confirmatory_resource_strong.toml"))
    assert definition.seeds == tuple(range(12000, 12200))
    assert definition.scenario.f_per_uav == 4e9
    assert definition.scenario.task.data_min == 0.8e6
    assert definition.scenario.task.data_max == 3.0e6
    assert definition.scenario.task.comp_min == 800.0
    assert definition.scenario.task.comp_max == 1500.0
    assert definition.scenario.task.t_min == 0.6
    assert definition.scenario.task.t_max == 1.0
    assert definition.event_log_level == "summary"
    assert definition.methods == (
        "full_fixed_half",
        "full_crossing",
        "full_edf_min_cpu",
        "full_min_slack_min_cpu",
        "full",
        "full_exact_oracle",
    )
    snapshot = generate_trace(
        definition.scenario,
        definition.seeds[0],
    ).slots[0]
    outcomes = []
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
        elif method == "full_edf_min_cpu":
            target = replace(
                target,
                allocation_policy=AllocationPolicy.EDF_MIN_CPU,
            )
        elif method == "full_min_slack_min_cpu":
            target = replace(
                target,
                allocation_policy=AllocationPolicy.MIN_SLACK_MIN_CPU,
            )
        elif method == "full_exact_oracle":
            target = replace(
                target,
                allocation_policy=AllocationPolicy.EXACT_ORACLE,
            )
        outcome = _slot_method(
            method,
            definition.scenario,
            target,
            definition.max_iterations,
            [],
        )(snapshot)
        outcomes.append(outcome)

    assert len({outcome.arrived_task_ids for outcome in outcomes}) == 1
    for outcome in outcomes:
        assert len(outcome.arrived_task_ids) == len(set(outcome.arrived_task_ids))
        assert all(record.sic_feasible for record in outcome.rates)
        assert all(
            decision.cpu_hz >= 0.0
            for decision in outcome.decisions
        )


def test_resource_run_manifest_records_each_controlled_policy(
    tmp_path: Path,
):
    config_path = Path("configs/thesis/confirmatory_resource_strong.toml")
    base = load_definition(config_path)
    definition = replace(
        base,
        seeds=(42,),
        scenario=replace(base.scenario, num_slots=1),
    )
    output = tmp_path / "resource"

    run_definition(definition, config_path, output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resource_policy_by_method"] == {
        "full": "deadline",
        "full_crossing": "crossing_equal",
        "full_edf_min_cpu": "edf_min_cpu",
        "full_exact_oracle": "exact_oracle",
        "full_fixed_half": "fixed_half_equal",
        "full_min_slack_min_cpu": "min_slack_min_cpu",
    }
    assert isinstance(manifest["git_worktree_dirty"], bool)
    assert len(manifest["source_snapshot"]["sha256"]) == 64
    assert "uav_mec/experiment.py" in manifest["source_snapshot"]["files"]
    assert all(
        len(digest) == 64
        for digest in manifest["source_snapshot"]["files"].values()
    )
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text().splitlines()
    ]
    assert events
    assert {row["move_type"] for row in events} <= {
        "initialization",
        "termination",
    }


def test_run_definition_streams_row_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class StreamingOnlyWriter(RunArtifactWriter):
        def write_csv(self, *_args, **_kwargs):
            raise AssertionError("run_definition must stream CSV rows")

        def write_jsonl(self, *_args, **_kwargs):
            raise AssertionError("run_definition must stream JSONL rows")

    monkeypatch.setattr(
        experiment_module,
        "RunArtifactWriter",
        StreamingOnlyWriter,
    )
    base = load_definition(Path("configs/thesis/confirmatory_main.toml"))
    definition = replace(
        base,
        seeds=(42,),
        methods=("local_only",),
        scenario=replace(base.scenario, num_slots=1),
    )

    run_definition(
        definition,
        Path("configs/thesis/confirmatory_main.toml"),
        tmp_path / "streamed",
    )

    assert (tmp_path / "streamed/tasks.csv").exists()
    assert (tmp_path / "streamed/slots.csv").exists()
    assert (tmp_path / "streamed/events.jsonl").exists()


def test_run_definition_reuses_one_multistart_pool_across_episode_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pool_instances = []

    class ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class RecordingExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.map_calls = 0
            self.submit_calls = 0
            pool_instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, *args):
            self.submit_calls += 1
            return ImmediateFuture(function(*args))

        def map(self, function, arguments):
            self.map_calls += 1
            return map(function, arguments)

    monkeypatch.setattr(
        experiment_module,
        "ProcessPoolExecutor",
        RecordingExecutor,
        raising=False,
    )
    base = load_definition(Path("configs/thesis/smoke.toml"))
    definition = replace(
        base,
        seeds=(42,),
        methods=("full",),
        scenario=replace(base.scenario, num_slots=2),
        multistart_workers=2,
    )

    run_definition(
        definition,
        Path("configs/thesis/smoke.toml"),
        tmp_path / "persistent-pool",
    )

    assert len(pool_instances) == 1
    assert pool_instances[0].submit_calls == 8
    assert pool_instances[0].map_calls == 0


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


def test_externality_stress_suite_freezes_independent_axes():
    cases = {case.name: case.definition for case in build_suite("externality")}

    assert set(cases) == {
        f"overlap_{overlap}_coupling{coupling}_load{load}"
        for overlap in ("low", "high")
        for coupling in ("0", "1")
        for load in ("08", "12")
    } | {
        "overlap_medium_coupling05_load10",
    }
    for definition in cases.values():
        scenario = definition.scenario
        assert scenario.num_uavs == 3
        assert scenario.num_users == 45
        assert scenario.num_clusters == 3
        assert scenario.f_per_uav == 4e9
        assert scenario.cluster_layout == "pairwise_overlap"
        assert scenario.task.arrival_mode == "periodic"
        assert scenario.task.data_min == 0.8e6
        assert scenario.task.data_max == 3.0e6
        assert scenario.task.comp_min == 800.0
        assert scenario.task.comp_max == 1500.0
        assert definition.methods == (
            "intra_global",
            "cross_bilateral",
            "full",
        )
        assert definition.seeds == tuple(range(13000, 13200))

    for name, definition in cases.items():
        if "overlap_low" in name:
            assert definition.scenario.cluster_std == 150.0
        elif "overlap_medium" in name:
            assert definition.scenario.cluster_std == 30.0
        else:
            assert definition.scenario.cluster_std == 5.0
        if "load08" in name:
            assert definition.scenario.task.period_choices == (3,)
        elif "load10" in name:
            assert definition.scenario.task.period_choices == (2, 3, 3)
        else:
            assert definition.scenario.task.period_choices == (2,)
        expected_coupling = 0.5 if "coupling05" in name else float(
            name.split("coupling", 1)[1].split("_", 1)[0]
        )
        assert definition.evaluation.cross_interference_scale == expected_coupling


def test_registered_robustness_and_scaling_suites_use_their_disjoint_seed_pools():
    robustness = build_suite("robustness")
    scalability = build_suite("scalability")

    assert robustness
    assert scalability
    assert all(case.definition.seeds == tuple(range(15000, 15030)) for case in robustness)
    assert all(case.definition.scenario.num_slots == 50 for case in robustness)
    assert all(case.definition.seeds == tuple(range(18000, 18030)) for case in scalability)
    assert all(case.definition.scenario.num_slots == 50 for case in scalability)


def test_exact_protocol_uses_the_registered_seed_pool_and_slot_count():
    definition = load_definition(Path("configs/thesis/confirmatory_exact.toml"))

    assert definition.seeds == tuple(range(16000, 16050))
    assert definition.scenario.num_slots == 20


def test_capacity_suite_freezes_busy_hour_and_equal_load_matrix():
    cases = {case.name: case.definition for case in build_suite("capacity")}

    assert set(cases) == {
        "load_k3_n30",
        "load_k3_n45",
        "load_k3_n60",
        "scale_k2_n30",
        "scale_k3_n45",
        "scale_k4_n60",
        "scale_k5_n75",
    }
    for name in ("load_k3_n30", "load_k3_n45", "load_k3_n60"):
        scenario = cases[name].scenario
        assert scenario.num_uavs == 3
        assert scenario.num_users in {30, 45, 60}
        assert scenario.num_clusters == 3
    for name in (
        "scale_k2_n30",
        "scale_k3_n45",
        "scale_k4_n60",
        "scale_k5_n75",
    ):
        scenario = cases[name].scenario
        assert scenario.num_users / scenario.num_uavs == 15
        assert scenario.num_clusters == max(3, scenario.num_uavs)
    expected_arrival_probabilities = [
        scenario.cool_rate + (scenario.burst_rate - scenario.cool_rate) / scenario.num_clusters
        for scenario in (
            cases[name].scenario
            for name in (
                "scale_k2_n30",
                "scale_k3_n45",
                "scale_k4_n60",
                "scale_k5_n75",
            )
        )
    ]
    assert expected_arrival_probabilities == pytest.approx([13.0 / 30.0] * 4)
    for definition in cases.values():
        assert definition.seeds == tuple(range(14000, 14200))
        assert definition.scenario.task.data_min == 0.8e6
        assert definition.scenario.task.data_max == 3.0e6
        assert definition.scenario.task.comp_min == 800.0
        assert definition.scenario.task.comp_max == 1500.0
        assert definition.scenario.task.t_min == 0.6
        assert definition.scenario.task.t_max == 1.0
        assert definition.methods == (
            "intra_global",
            "cross_bilateral",
            "full",
        )


def test_externality_runner_uses_independent_config_and_writes_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    case = build_suite("externality")[0]
    run_calls = []
    analysis_calls = []

    def fake_run(definition, config_path, output_path):
        run_calls.append((definition, config_path, output_path))

    def fake_analysis(cases, output_path):
        analysis_calls.append((cases, output_path))

    monkeypatch.setattr(thesis_suites, "build_suite", lambda _name: (case,))
    monkeypatch.setattr(thesis_suites, "run_definition", fake_run)
    monkeypatch.setattr(
        thesis_suites,
        "write_externality_analysis",
        fake_analysis,
        raising=False,
    )

    output = tmp_path / "externality"
    thesis_suites.run_suite("externality", output)

    assert run_calls == [(case.definition, EXTERNALITY_BASE_PATH, output / case.name)]
    assert analysis_calls == [
        (
            {case.name: (case.definition, output / case.name)},
            output / "analysis",
        )
    ]


def test_capacity_runner_uses_v2_config_and_writes_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    case = build_suite("capacity")[0]
    run_calls = []
    analysis_calls = []

    monkeypatch.setattr(thesis_suites, "build_suite", lambda _name: (case,))
    monkeypatch.setattr(
        thesis_suites,
        "run_definition",
        lambda definition, config_path, output_path: run_calls.append(
            (definition, config_path, output_path)
        ),
    )
    monkeypatch.setattr(
        thesis_suites,
        "write_capacity_analysis",
        lambda cases, output_path: analysis_calls.append((cases, output_path)),
        raising=False,
    )

    output = tmp_path / "capacity"
    thesis_suites.run_suite("capacity", output)

    assert run_calls == [(case.definition, thesis_suites.CAPACITY_BASE_PATH, output / case.name)]
    assert analysis_calls == [
        (
            {case.name: (case.definition, output / case.name)},
            output / "analysis",
        )
    ]


def test_suite_runner_can_execute_cases_with_bounded_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cases = build_suite("externality")[:2]
    run_calls = []
    executor_workers = []

    class FakeExecutor:
        def __init__(self, max_workers):
            executor_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, function, arguments):
            return [function(argument) for argument in arguments]

    def fake_run(definition, config_path, output_path):
        run_calls.append((definition.run_id, config_path, output_path))

    monkeypatch.setattr(thesis_suites, "build_suite", lambda _name: cases)
    monkeypatch.setattr(thesis_suites, "run_definition", fake_run)
    monkeypatch.setattr(thesis_suites, "ProcessPoolExecutor", FakeExecutor, raising=False)
    monkeypatch.setattr(thesis_suites, "write_externality_analysis", lambda *_args: None)

    output = tmp_path / "parallel"
    thesis_suites.run_suite("externality", output, workers=2)

    assert executor_workers == [2]
    assert run_calls == [(case.name, EXTERNALITY_BASE_PATH, output / case.name) for case in cases]


def test_suite_runner_rejects_nonpositive_worker_count(tmp_path: Path):
    with pytest.raises(ValueError, match="workers must be positive"):
        thesis_suites.run_suite("externality", tmp_path / "invalid", workers=0)


def test_suite_runner_can_bound_dry_run_seeds_and_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    case = build_suite("capacity")[0]
    captured = []

    monkeypatch.setattr(thesis_suites, "build_suite", lambda _name: (case,))
    monkeypatch.setattr(
        thesis_suites,
        "run_definition",
        lambda definition, _config_path, _output_path: captured.append(definition),
    )
    monkeypatch.setattr(thesis_suites, "write_capacity_analysis", lambda *_args: None)

    thesis_suites.run_suite(
        "capacity",
        tmp_path / "dry-run",
        seed_count=2,
        num_slots=3,
        development_seed_start=42,
    )

    assert captured[0].seeds == (42, 43)
    assert captured[0].scenario.num_slots == 3


def test_suite_runner_requires_explicit_nonformal_development_seed_pool(tmp_path: Path):
    with pytest.raises(ValueError, match="development_seed_start"):
        thesis_suites.run_suite(
            "capacity",
            tmp_path / "missing-development-pool",
            seed_count=2,
            num_slots=3,
        )

    with pytest.raises(ValueError, match="overlaps a registered formal seed"):
        thesis_suites.run_suite(
            "capacity",
            tmp_path / "formal-overlap",
            seed_count=2,
            num_slots=3,
            development_seed_start=14000,
        )


@pytest.mark.parametrize(
    ("keyword", "message"),
    (("seed_count", "seed_count must be positive"), ("num_slots", "num_slots must be positive")),
)
def test_suite_runner_rejects_nonpositive_dry_run_bounds(
    tmp_path: Path,
    keyword: str,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        thesis_suites.run_suite(
            "capacity",
            tmp_path / keyword,
            **{keyword: 0},
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
):
    case = build_suite(suite_name)[0]
    definition = replace(
        case.definition,
        seeds=(42, 43),
        scenario=replace(case.definition.scenario, num_slots=2),
    )
    output = tmp_path / suite_name

    run_definition(definition, BASE_PATH, output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    resolved = manifest["resolved_definition"]
    assert resolved["seeds"] == [42, 43]
    assert resolved["scenario"]["num_slots"] == 2
    value = resolved
    for key in value_path:
        value = value[key]
    assert value == expected
