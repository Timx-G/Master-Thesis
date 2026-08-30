import csv
import hashlib
import json
from pathlib import Path

import pytest

from uav_mec.experiment import _slot_method, load_definition, run_experiment
from uav_mec.trace import generate_trace


def test_smoke_definition_uses_periodic_task_semantics():
    definition = load_definition(Path("configs/thesis/smoke.toml"))

    assert definition.scenario.random_burst is False
    assert definition.scenario.task.arrival_mode == "periodic"
    assert definition.scenario.task.period_choices == (2, 2, 3)


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


@pytest.mark.parametrize(
    "method",
    ("intra_global", "cross_bilateral", "full"),
)
def test_structural_ablation_methods_share_multistart_best_improvement(method):
    definition = load_definition(Path("configs/thesis/smoke.toml"))
    snapshot = generate_trace(
        definition.scenario,
        definition.seeds[0],
        definition.csi_error_std,
    ).slots[0]
    events: list[dict[str, object]] = []

    _slot_method(
        method,
        definition.scenario,
        definition.evaluation,
        definition.max_iterations,
        events,
    )(snapshot)

    initializers = [
        event for event in events if event["move_type"] == "initialization"
    ]
    assert {event["initializer_id"] for event in initializers} == {
        "fair_load_greedy",
        "ia_greedy",
        "per_uav_ilp",
    }
    (termination,) = [
        event for event in events if event["move_type"] == "termination"
    ]
    assert termination["improvement"] == "best"
    assert termination["initializer_id"] in {
        "fair_load_greedy",
        "ia_greedy",
        "per_uav_ilp",
    }


def test_single_first_ablation_uses_only_the_ilp_initializer():
    definition = load_definition(Path("configs/thesis/smoke.toml"))
    snapshot = generate_trace(
        definition.scenario,
        definition.seeds[0],
        definition.csi_error_std,
    ).slots[0]
    events: list[dict[str, object]] = []

    _slot_method(
        "full_single_first",
        definition.scenario,
        definition.evaluation,
        definition.max_iterations,
        events,
    )(snapshot)

    initializers = [
        event for event in events if event["move_type"] == "initialization"
    ]
    assert [event["initializer_id"] for event in initializers] == ["per_uav_ilp"]
    (termination,) = [
        event for event in events if event["move_type"] == "termination"
    ]
    assert termination["improvement"] == "first"
    assert termination["initializer_id"] == "per_uav_ilp"


def test_summary_event_mode_keeps_accepted_moves_and_candidate_count():
    definition = load_definition(Path("configs/thesis/smoke.toml"))
    snapshot = generate_trace(
        definition.scenario,
        definition.seeds[0],
        definition.csi_error_std,
    ).slots[0]
    events: list[dict[str, object]] = []

    _slot_method(
        "full",
        definition.scenario,
        definition.evaluation,
        definition.max_iterations,
        events,
        event_log_level="summary",
    )(snapshot)

    candidate_events = [
        event
        for event in events
        if event["move_type"] not in {"initialization", "termination"}
    ]
    assert all(event["accepted"] for event in candidate_events)
    (termination,) = [
        event for event in events if event["move_type"] == "termination"
    ]
    assert termination["candidate_evaluations"] >= len(candidate_events)
