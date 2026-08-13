import numpy as np
import pytest

from uav_mec.statistics import (
    holm_adjust,
    paired_bootstrap_ci,
    paired_comparison,
    participant_clustered_bootstrap_ci,
    rank_biserial,
)


def test_holm_step_down_uses_cumulative_maximum():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})

    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


@pytest.mark.parametrize("invalid", [-0.01, 1.01, float("nan")])
def test_holm_rejects_invalid_p_values(invalid):
    with pytest.raises(ValueError, match="invalid p-value"):
        holm_adjust({"bad": invalid})


def test_paired_bootstrap_is_deterministic():
    differences = np.array([1.0, 2.0, 3.0, 4.0])

    first = paired_bootstrap_ci(differences, 10_000, 20260711)
    second = paired_bootstrap_ci(differences, 10_000, 20260711)

    assert first == second


@pytest.mark.parametrize("resamples", [0, -1, 1.5, True])
def test_paired_bootstrap_rejects_invalid_resample_counts(resamples):
    with pytest.raises(ValueError, match="resamples"):
        paired_bootstrap_ci(np.array([1.0, 2.0]), resamples=resamples)


def test_paired_comparison_reports_positive_effect():
    proposed = np.array([80.0, 82.0, 84.0, 86.0, 88.0])
    baseline = np.array([75.0, 77.0, 78.0, 80.0, 82.0])

    result = paired_comparison(proposed, baseline, bootstrap_seed=7)

    assert result.mean_difference > 0
    assert result.rank_biserial == pytest.approx(1.0)
    assert result.ci_low > 0
    assert 0.0 <= result.p_raw <= 1.0


def test_all_equal_pairs_have_neutral_effect_and_unit_p_value():
    values = np.array([3.0, 3.0, 3.0])

    result = paired_comparison(values, values)

    assert result.mean_difference == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.wilcoxon_w == 0.0
    assert result.p_raw == 1.0
    assert result.rank_biserial == 0.0


def test_rank_biserial_uses_absolute_rank_ties_and_ignores_zeros():
    differences = np.array([2.0, -2.0, 1.0, 0.0])

    assert rank_biserial(differences) == pytest.approx(1.0 / 6.0)


def test_paired_comparison_rejects_unpaired_shapes():
    with pytest.raises(ValueError, match="same shape"):
        paired_comparison(np.array([1.0, 2.0]), np.array([1.0]))


def test_participant_clustered_bootstrap_resamples_source_participants():
    differences = np.asarray([10.0, 10.0, 10.0, 0.0])
    participants = ("participant-a", "participant-a", "participant-a", "participant-b")

    first = participant_clustered_bootstrap_ci(
        differences, participants, resamples=20_000, seed=20260719
    )
    second = participant_clustered_bootstrap_ci(
        differences, participants, resamples=20_000, seed=20260719
    )

    assert first == second
    assert first == pytest.approx((0.0, 10.0))


@pytest.mark.parametrize(
    ("differences", "participants", "resamples"),
    (
        (np.asarray([1.0]), ("only",), 10),
        (np.asarray([1.0, 2.0]), ("same", "same"), 10),
        (np.asarray([1.0, 2.0]), ("a", ""), 10),
        (np.asarray([1.0, 2.0]), ("a",), 10),
        (np.asarray([1.0, np.nan]), ("a", "b"), 10),
        (np.asarray([1.0, 2.0]), ("a", "b"), 0),
    ),
)
def test_participant_clustered_bootstrap_rejects_invalid_inputs(
    differences, participants, resamples
):
    with pytest.raises(ValueError):
        participant_clustered_bootstrap_ci(differences, participants, resamples=resamples)
