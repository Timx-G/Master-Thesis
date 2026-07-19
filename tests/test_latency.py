import pytest

from uav_mec.domain import Task
from uav_mec.latency import branch_delays_s, completion_delay_s


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
