import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from uav_mec.domain import Task
from uav_mec.latency import branch_delays_s, completion_delay_s
from uav_mec.model import ChannelConfig, NOMAChannel, offload_delay
from uav_mec.pairing import evaluate_pairing, kkt_alpha
from uav_mec.resource_allocation import required_resources


def _task(rho: float) -> Task:
    return Task(task_id=1, user_id=0, arrival_slot=2, data_bits=800.0,
                comp_cycles=1_200.0, deadline_s=0.8, local_cpu_hz=2_000.0,
                slot_duration_s=1.0, result_ratio=rho)


def test_canonical_delay_uses_distinct_uplink_and_downlink_rates() -> None:
    delays = branch_delays_s(_task(0.25), 0.5, 400.0, 600.0, 100.0)
    assert delays.local_s == pytest.approx(0.3)
    assert delays.uplink_s == pytest.approx(1.0)
    assert delays.mec_s == pytest.approx(1.0)
    assert delays.downlink_s == pytest.approx(1.0)
    assert delays.edge_s == pytest.approx(3.0)
    assert completion_delay_s(_task(0.25), 0.5, 400.0, 600.0, 100.0) == pytest.approx(3.0)


def test_zero_rho_needs_no_downlink_rate() -> None:
    assert completion_delay_s(_task(0.0), 0.5, 400.0, 600.0, None) == pytest.approx(2.0)


def test_positive_rho_requires_an_independent_downlink_rate() -> None:
    with pytest.raises(ValueError, match="downlink_rate_bps"):
        completion_delay_s(_task(0.25), 0.5, 400.0, 600.0, None)


def test_resource_requirement_defaults_to_task_level_ratio() -> None:
    task = _task(0.25)
    requirement = required_resources(
        task,
        rate_bps=10_000.0,
        downlink_rate_bps=5_000.0,
    )
    assert requirement.task_id == task.task_id


def test_resource_requirement_api_has_no_global_result_ratio() -> None:
    assert tuple(inspect.signature(required_resources).parameters) == (
        "task",
        "rate_bps",
        "downlink_rate_bps",
        "alpha",
    )


def test_canonical_delay_returns_infinity_for_unrepresentable_edge_sum() -> None:
    task = Task(
        task_id=2,
        user_id=0,
        arrival_slot=0,
        data_bits=1e308,
        comp_cycles=1e308,
        deadline_s=1.0,
        local_cpu_hz=1e308,
        result_ratio=0.0,
    )

    assert math.isinf(completion_delay_s(task, 1.0, 1.0, 1.0))


def test_formal_pipeline_does_not_call_legacy_delay_helpers() -> None:
    formal_paths = (
        Path("uav_mec/closure.py"),
        Path("uav_mec/evaluation.py"),
        Path("uav_mec/resource_allocation.py"),
        Path("uav_mec/scheduler.py"),
        Path("uav_mec/exact.py"),
        Path("uav_mec/baselines_v2.py"),
        Path("uav_mec/experiment.py"),
    )
    forbidden_calls = {"kkt_alpha", "evaluate_pairing", "offload_delay"}
    for path in formal_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        imported_pairing = any(
            isinstance(node, ast.ImportFrom)
            and (node.module or "").endswith("pairing")
            or isinstance(node, ast.Import)
            and any(alias.name.endswith("pairing") for alias in node.names)
            for node in ast.walk(tree)
        )
        assert not imported_pairing, path
        assert not called.intersection(forbidden_calls), (path, called)


def test_legacy_delay_entry_points_emit_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="legacy-v1 delay semantics"):
        kkt_alpha(1_000.0, 2_000.0, 5_000.0, 10_000.0, 4_000.0)
    with pytest.warns(DeprecationWarning, match="legacy-v1 delay semantics"):
        offload_delay(1_000.0, 2_000.0, 5_000.0, 10_000.0)
    with pytest.warns(DeprecationWarning, match="legacy-v1 delay semantics"):
        result = evaluate_pairing(
            {},
            np.empty((0, 1)),
            NOMAChannel(ChannelConfig(), 100.0),
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=int),
            1,
            4e9,
            1e9,
            0.8,
        )
    assert result[:2] == (0, 0)
