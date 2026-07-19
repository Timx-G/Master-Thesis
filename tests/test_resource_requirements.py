import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import uav_mec.resource_allocation as resource_allocation
from uav_mec.domain import Task
from uav_mec.resource_allocation import (
    RequirementReason,
    ResourceRequirement,
    required_resources,
)


def make_task(comp_cycles: float, result_ratio: float = 0.1) -> Task:
    return Task(
        1,
        0,
        0,
        1_000_000.0,
        comp_cycles,
        0.8,
        1_000_000_000.0,
        result_ratio=result_ratio,
    )


def test_local_feasible_task_requires_no_radio_or_mec():
    req = required_resources(make_task(600_000_000.0), rate_bps=1.0)
    assert req.feasible
    assert req.local_only
    assert req.alpha == 0.0
    assert req.cpu_hz == 0.0
    assert req.reason is RequirementReason.LOCAL


def test_heavy_task_uses_minimum_offload_and_cpu():
    task = make_task(1_200_000_000.0)
    req = required_resources(
        task,
        rate_bps=4_000_000.0,
        downlink_rate_bps=4_000_000.0,
    )
    expected_alpha = 1.0 - 0.8e9 / 1.2e9
    expected_cpu = expected_alpha * 1.2e9 / (0.8 - expected_alpha * 1.1e6 / 4e6)
    assert req.alpha == pytest.approx(expected_alpha)
    assert req.cpu_hz == pytest.approx(expected_cpu)
    assert req.feasible


def test_link_is_infeasible_when_transmission_alone_exceeds_deadline():
    req = required_resources(
        make_task(2_000_000_000.0),
        rate_bps=100_000.0,
        downlink_rate_bps=100_000.0,
    )
    assert not req.feasible
    assert req.reason is RequirementReason.LINK_INFEASIBLE


def test_positive_result_ratio_requires_an_independent_downlink_rate():
    with pytest.raises(ValueError, match="downlink_rate_bps"):
        required_resources(
            make_task(1_200_000_000.0),
            rate_bps=4_000_000.0,
        )


def test_downlink_rate_is_keyword_only_to_reject_legacy_scalar_ratio():
    with pytest.raises(TypeError, match="positional"):
        required_resources(make_task(1_200_000_000.0), 4_000_000.0, 0.1)


@pytest.mark.parametrize("unused_rate", [0.0, -1.0, math.nan, math.inf])
def test_local_deadline_equality_ignores_unused_rate(unused_rate: float):
    task = make_task(800_000_000.0)
    assert task.local_delay_s == task.deadline_s

    req = required_resources(task, rate_bps=unused_rate)

    assert req == ResourceRequirement(
        task_id=task.task_id,
        feasible=True,
        local_only=True,
        alpha=0.0,
        cpu_hz=0.0,
        reason=RequirementReason.LOCAL,
    )


@pytest.mark.parametrize("invalid_rate", [0.0, -1.0, math.nan, math.inf])
def test_nonlocal_task_rejects_invalid_rate(invalid_rate: float):
    req = required_resources(
        make_task(1_200_000_000.0),
        rate_bps=invalid_rate,
    )

    assert not req.feasible
    assert not req.local_only
    assert req.reason is RequirementReason.INVALID_RATE


@pytest.mark.parametrize("invalid_rate", [0.0, -1.0, math.nan, math.inf])
def test_nonlocal_task_rejects_invalid_downlink_rate(invalid_rate: float):
    req = required_resources(
        make_task(1_200_000_000.0),
        rate_bps=4_000_000.0,
        downlink_rate_bps=invalid_rate,
    )

    assert not req.feasible
    assert req.reason is RequirementReason.INVALID_RATE


def test_analytical_requirement_matches_independent_full_alpha_grid():
    task = make_task(1_600_000_000.0)
    uplink_rate_bps = 4_000_000.0
    downlink_rate_bps = 2_000_000.0

    req = required_resources(
        task,
        uplink_rate_bps,
        downlink_rate_bps=downlink_rate_bps,
    )

    alphas = np.linspace(0.0, 1.0, 20_001)
    local_delays = (1.0 - alphas) * task.comp_cycles / task.local_cpu_hz
    transfer_per_alpha_s = (
        task.data_bits / uplink_rate_bps
        + task.result_ratio * task.data_bits / downlink_rate_bps
    )
    edge_windows = task.deadline_s - alphas * transfer_per_alpha_s
    feasible_mask = (local_delays <= task.deadline_s) & (edge_windows > 0.0)
    feasible_alphas = alphas[feasible_mask]
    feasible_cpu = feasible_alphas * task.comp_cycles / edge_windows[feasible_mask]
    minimum_index = int(np.argmin(feasible_cpu))

    assert req.alpha == pytest.approx(float(feasible_alphas[minimum_index]), abs=1e-15)
    assert req.cpu_hz == pytest.approx(float(feasible_cpu[minimum_index]), rel=1e-12)


def test_ordinary_scale_requirement_is_minimal_and_meets_both_branches():
    task = make_task(1_600_000_000.0)
    uplink_rate_bps = 4_000_000.0
    downlink_rate_bps = 2_000_000.0
    epsilon = 1e-9

    req = required_resources(
        task,
        uplink_rate_bps,
        downlink_rate_bps=downlink_rate_bps,
    )

    returned_local_delay = (1.0 - req.alpha) * task.comp_cycles / task.local_cpu_hz
    returned_edge_delay = req.alpha * (
        task.data_bits / uplink_rate_bps
        + task.comp_cycles / req.cpu_hz
        + task.result_ratio * task.data_bits / downlink_rate_bps
    )
    smaller_alpha_local_delay = (1.0 - (req.alpha - epsilon)) * task.comp_cycles / task.local_cpu_hz
    smaller_cpu_edge_delay = req.alpha * (
        task.data_bits / uplink_rate_bps
        + task.comp_cycles / (req.cpu_hz * (1.0 - epsilon))
        + task.result_ratio * task.data_bits / downlink_rate_bps
    )
    predecessor_cpu = math.nextafter(req.cpu_hz, 0.0)
    predecessor_cpu_edge_delay = req.alpha * (
        task.data_bits / uplink_rate_bps
        + task.comp_cycles / predecessor_cpu
        + task.result_ratio * task.data_bits / downlink_rate_bps
    )

    assert returned_local_delay <= task.deadline_s + 1e-12
    assert returned_edge_delay <= task.deadline_s + 1e-12
    assert smaller_alpha_local_delay > task.deadline_s
    assert smaller_cpu_edge_delay > task.deadline_s
    assert predecessor_cpu_edge_delay > task.deadline_s


def test_link_denominator_exactly_zero_and_below_are_infeasible():
    task = Task(2, 0, 0, 2.0, 2.0, 1.0, 1.0)
    boundary_rate = 1.0

    at_boundary = required_resources(task, boundary_rate)
    below_boundary = required_resources(
        task,
        math.nextafter(boundary_rate, 0.0),
    )

    assert not at_boundary.feasible
    assert at_boundary.reason is RequirementReason.LINK_INFEASIBLE
    assert not below_boundary.feasible
    assert below_boundary.reason is RequirementReason.LINK_INFEASIBLE


def test_link_denominator_one_ulp_above_zero_is_feasible():
    task = Task(2, 0, 0, 2.0, 2.0, 1.0, 1.0)
    rate_bps = math.nextafter(1.0, math.inf)

    req = required_resources(task, rate_bps)
    independently_recomputed_delay = req.alpha * (
        task.data_bits / rate_bps + task.comp_cycles / req.cpu_hz
    )

    assert req.feasible
    assert req.reason is RequirementReason.MEC
    assert independently_recomputed_delay <= task.deadline_s


def test_cpu_is_adjusted_when_independent_delay_is_one_ulp_late():
    task = Task(
        3,
        0,
        0,
        1.0,
        3.0,
        0.1,
        1.0,
        result_ratio=0.1,
    )

    uplink_rate_bps = 100_000.0
    downlink_rate_bps = 80_000.0
    req = required_resources(
        task,
        rate_bps=uplink_rate_bps,
        downlink_rate_bps=downlink_rate_bps,
    )
    independently_recomputed_delay = req.alpha * (
        task.data_bits / uplink_rate_bps
        + task.comp_cycles / req.cpu_hz
        + task.result_ratio * task.data_bits / downlink_rate_bps
    )

    assert req.feasible
    assert independently_recomputed_delay <= task.deadline_s


def test_subnormal_rounding_is_conservative_bounded_and_feasible(monkeypatch):
    smallest_positive = math.nextafter(0.0, math.inf)
    real_nextafter = math.nextafter
    nextafter_calls = 0

    def counted_nextafter(value: float, direction: float) -> float:
        nonlocal nextafter_calls
        nextafter_calls += 1
        if nextafter_calls > 8:
            raise AssertionError("rounding correction must be bounded")
        return real_nextafter(value, direction)

    monkeypatch.setattr(resource_allocation.math, "nextafter", counted_nextafter)
    task = Task(4, 0, 0, smallest_positive, smallest_positive, 0.75, smallest_positive)

    req = required_resources(task, rate_bps=1.0)
    local_delay = (1.0 - req.alpha) * task.comp_cycles / task.local_cpu_hz
    expected_alpha = 0.5
    predecessor = real_nextafter(expected_alpha, 0.0)
    second_predecessor = real_nextafter(predecessor, 0.0)
    predecessor_delay = (1.0 - predecessor) * task.comp_cycles / task.local_cpu_hz
    second_predecessor_delay = (1.0 - second_predecessor) * task.comp_cycles / task.local_cpu_hz

    assert req.feasible
    assert req.reason is RequirementReason.MEC
    assert req.alpha == expected_alpha
    assert nextafter_calls <= 8
    assert local_delay <= task.deadline_s
    # Cancellation leaves a short feasibility plateau below the conservative alpha.
    assert predecessor_delay <= task.deadline_s
    assert second_predecessor_delay > task.deadline_s
    assert 0.0 < req.cpu_hz < math.inf


def test_transmission_ratio_avoids_intermediate_overflow():
    task = Task(
        task_id=3,
        user_id=0,
        arrival_slot=0,
        data_bits=1e308,
        comp_cycles=2e9,
        deadline_s=2.0,
        local_cpu_hz=1e8,
        result_ratio=1.0,
    )

    req = required_resources(
        task,
        rate_bps=1e308,
        downlink_rate_bps=1e308,
    )

    assert req.feasible
    assert req.reason is RequirementReason.MEC
    assert req.alpha == pytest.approx(0.9)
    assert req.cpu_hz == pytest.approx(9e9, rel=1e-12)


def test_unrepresentable_cpu_requirement_is_not_reported_feasible():
    task = Task(
        task_id=4,
        user_id=0,
        arrival_slot=0,
        data_bits=2.0,
        comp_cycles=1e308,
        deadline_s=1.0,
        local_cpu_hz=5e307,
    )

    req = required_resources(
        task,
        rate_bps=math.nextafter(1.0, math.inf),
    )

    assert not req.feasible
    assert req.reason is RequirementReason.LINK_INFEASIBLE


@pytest.mark.parametrize(
    ("task", "rate_bps", "downlink_rate_bps"),
    [
        (make_task(1_200_000_000.0), 4_000_000.0, 4_000_000.0),
        (make_task(1_600_000_000.0), 4_000_000.0, 4_000_000.0),
        (
            Task(
                5,
                0,
                0,
                1e308,
                2e9,
                2.0,
                1e8,
                result_ratio=1.0,
            ),
            1e308,
            1e308,
        ),
    ],
)
def test_feasible_mec_requirement_has_finite_positive_resources(
    task: Task,
    rate_bps: float,
    downlink_rate_bps: float,
):
    req = required_resources(
        task,
        rate_bps,
        downlink_rate_bps=downlink_rate_bps,
    )

    assert req.feasible
    assert not req.local_only
    assert 0.0 < req.alpha <= 1.0
    assert math.isfinite(req.alpha)
    assert req.cpu_hz > 0.0
    assert math.isfinite(req.cpu_hz)


def test_resource_requirement_is_immutable():
    req = required_resources(
        make_task(1_200_000_000.0),
        4_000_000.0,
        downlink_rate_bps=4_000_000.0,
    )

    with pytest.raises(FrozenInstanceError):
        req.cpu_hz = 1.0
