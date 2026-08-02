import dataclasses
from dataclasses import replace

import numpy as np
import pytest

from uav_mec.domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, compute_rate_records
from uav_mec.experiment import MethodSpec, thesis_method_specs
from uav_mec.experiment_registry import thesis_contrasts
from uav_mec.model import ChannelConfig, NOMAChannel


def _factor_differences(left: MethodSpec, right: MethodSpec) -> set[str]:
    ignored = {"method_id", "display_name"}
    return {
        field.name
        for field in dataclasses.fields(MethodSpec)
        if field.name not in ignored
        and getattr(left, field.name) != getattr(right, field.name)
    }


def test_h1_and_h2_structural_ablations_change_only_registered_factors() -> None:
    specs = thesis_method_specs()
    assert _factor_differences(specs["full"], specs["intra_global"]) == {
        "neighborhood_scope"
    }
    assert _factor_differences(specs["full"], specs["cross_bilateral"]) == {
        "acceptance_rule"
    }
    assert specs["full"].initializer_ids == specs["intra_global"].initializer_ids
    assert specs["full"].initializer_ids == specs["cross_bilateral"].initializer_ids


def test_joint_methods_form_an_exact_two_by_two_factorial() -> None:
    specs = thesis_method_specs()
    keys = ("iea_dd", "iea_edf", "exchange_dd", "exchange_edf")
    joint = {key: specs[key] for key in keys}
    assert {spec.wireless_kernel for spec in joint.values()} == {
        "iea_cap",
        "exchange_stable",
    }
    assert {spec.allocation_policy.value for spec in joint.values()} == {
        "deadline",
        "edf_min_cpu",
    }
    signatures = {
        (spec.wireless_kernel, spec.allocation_policy.value) for spec in joint.values()
    }
    assert len(signatures) == 4
    assert len({spec.initializer_ids for spec in joint.values()}) == 1
    h4 = thesis_contrasts()["H4"]
    assert (h4.treatment_id, h4.control_id) == ("iea_dd", "exchange_edf")


def test_fixed_radio_resource_methods_name_their_schedule_source() -> None:
    specs = thesis_method_specs()
    assert specs["fixed_radio_dd"].fixed_radio_source == "dd_generated"
    assert specs["fixed_radio_edf"].fixed_radio_source == "dd_generated"
    assert (
        specs["fixed_radio_edf"].display_name == "Deadline-First-MinCPU (static share)"
    )
    assert "EDF" not in specs["fixed_radio_edf"].display_name
    assert specs["fixed_radio_dd_radio_neutral"].fixed_radio_source == "radio_neutral"
    assert specs["fixed_radio_edf_radio_neutral"].fixed_radio_source == "radio_neutral"
    assert (
        specs["fixed_radio_dd_deadline_generated"].fixed_radio_source
        == "deadline_first_generated"
    )
    assert (
        specs["fixed_radio_edf_deadline_generated"].fixed_radio_source
        == "deadline_first_generated"
    )
    h3 = thesis_contrasts()["H3"]
    assert (h3.treatment_id, h3.control_id, h3.stratum) == (
        "fixed_radio_dd",
        "fixed_radio_edf",
        "dd_generated",
    )


def test_end_to_end_noma_methods_share_canonical_semantics() -> None:
    specs = thesis_method_specs()
    method_ids = (
        "greedy",
        "ia_greedy",
        "per_uav_ilp",
        "exchange_stable",
        "intra_global",
        "cross_bilateral",
        "single_start_first",
        "full",
        "guo_jagp_adapted",
        "iea_dd",
        "iea_edf",
        "exchange_dd",
        "exchange_edf",
    )
    for method_id in method_ids:
        spec = specs[method_id]
        assert spec.physics_model_id == "canonical_noma_v2"
        assert spec.latency_model_id == "canonical_task_latency_v2"
        assert spec.closure_mode == "policy_closure"
        assert spec.realized_replay


def base_config() -> EvaluationConfig:
    return EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9)


def test_zero_cross_scale_removes_only_cross_uav_interference():
    tasks = (
        Task(1, 0, 0, 1e6, 1.3e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.3e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[9e-8, 8e-8], [8e-8, 9e-8]]),
        2,
    )
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "solo"),
            ScheduleEntry(2, 1, 1, 0, "solo"),
        )
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)
    isolated = compute_rate_records(
        snapshot,
        schedule,
        channel,
        replace(base_config(), cross_interference_scale=0.0),
        channel_basis=ChannelBasis.PLANNING,
    )
    half = compute_rate_records(
        snapshot,
        schedule,
        channel,
        replace(base_config(), cross_interference_scale=0.5),
        channel_basis=ChannelBasis.PLANNING,
    )
    coupled = compute_rate_records(
        snapshot,
        schedule,
        channel,
        base_config(),
        channel_basis=ChannelBasis.PLANNING,
    )

    assert all(record.cross_interference_w == 0.0 for record in isolated)
    for zero, middle, full in zip(isolated, half, coupled, strict=True):
        assert middle.cross_interference_w == pytest.approx(
            0.5 * full.cross_interference_w
        )
        assert zero.rate_bps > middle.rate_bps > full.rate_bps


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("nan")])
def test_cross_interference_scale_rejects_nonphysical_values(scale):
    with pytest.raises(ValueError, match="cross_interference_scale"):
        replace(base_config(), cross_interference_scale=scale)
