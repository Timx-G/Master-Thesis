import numpy as np

from uav_mec.power_analysis import holm_rejections, select_sample_size


def test_holm_rejections_stop_after_the_first_failed_ordered_hypothesis():
    rejected = holm_rejections(
        {"H1": 0.001, "H2": 0.01, "H3": 0.04, "H4": 0.20},
        alpha=0.05,
    )

    assert rejected == {
        "H1": True,
        "H2": True,
        "H3": False,
        "H4": False,
    }


def test_power_simulation_selects_the_smallest_candidate_meeting_joint_holm_power():
    pilot = {
        hypothesis: np.linspace(-0.02, 0.02, 100)
        for hypothesis in ("H1", "H2", "H3", "H4")
    }
    mdes = {"H1": 0.5, "H2": 0.5, "H3": 2.0, "H4": 2.0}

    decision = select_sample_size(
        pilot,
        mdes,
        candidates=(50, 100, 200),
        simulations=200,
        target_power=0.8,
        seed=123,
    )

    assert decision.selected_n == 50
    assert all(power >= 0.8 for power in decision.power_by_hypothesis.values())


def test_power_simulation_reports_insufficient_evidence_at_the_largest_candidate():
    pilot = {
        hypothesis: np.linspace(-100.0, 100.0, 100)
        for hypothesis in ("H1", "H2", "H3", "H4")
    }
    mdes = {hypothesis: 0.01 for hypothesis in pilot}

    decision = select_sample_size(
        pilot,
        mdes,
        candidates=(50, 100, 200),
        simulations=100,
        target_power=0.8,
        seed=321,
    )

    assert decision.selected_n is None
    assert any(power < 0.8 for power in decision.power_by_hypothesis.values())
