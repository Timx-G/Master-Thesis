import pytest

from uav_mec.runtime_audit import summarize_runtime_audit


def test_runtime_audit_reports_cost_and_passes_scale_gate():
    report = summarize_runtime_audit(
        scale="K3N45",
        runtimes_s=[0.2] * 95 + [0.8] * 5,
        candidate_counts=list(range(100)),
        peak_memory_bytes=64 * 1024 * 1024,
        log_bytes=4096,
        formal_slot_evaluations=20_000,
    )

    assert report["sample_count"] == 100
    assert report["p50_runtime_s"] == pytest.approx(0.2)
    assert report["p95_runtime_s"] < 1.0
    assert report["candidate_count_p95"] == pytest.approx(94.05)
    assert report["predicted_total_cpu_hours"] > 1.0
    assert report["runtime_gate_pass"] is True


def test_runtime_audit_enforces_five_seed_twenty_slot_design():
    with pytest.raises(ValueError, match="exactly 100"):
        summarize_runtime_audit(
            scale="K5N75",
            runtimes_s=[1.0] * 99,
            candidate_counts=[1] * 99,
            peak_memory_bytes=1,
            log_bytes=1,
            formal_slot_evaluations=100,
        )


def test_runtime_audit_uses_distinct_k5_gate():
    report = summarize_runtime_audit(
        scale="K5N75",
        runtimes_s=[9.9] * 95 + [12.0] * 5,
        candidate_counts=[20] * 100,
        peak_memory_bytes=1,
        log_bytes=1,
        formal_slot_evaluations=100,
    )

    assert report["p95_runtime_s"] > 10.0
    assert report["runtime_gate_pass"] is False


@pytest.mark.parametrize("scale", ["K2N30", "k3n45", ""])
def test_runtime_audit_rejects_unregistered_scale(scale: str):
    with pytest.raises(ValueError, match="scale"):
        summarize_runtime_audit(
            scale=scale,
            runtimes_s=[1.0] * 100,
            candidate_counts=[1] * 100,
            peak_memory_bytes=1,
            log_bytes=1,
            formal_slot_evaluations=100,
        )
