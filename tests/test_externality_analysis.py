import math
from dataclasses import dataclass, replace

import pytest

from uav_mec.domain import (
    AllocationDecision,
    ChannelBasis,
    RateRecord,
    RealizedSlotOutcome,
    Schedule,
    ScheduleEntry,
    SicLayerCheck,
    SlotOutcome,
)
from uav_mec.externality import (
    aggregate_victim_link_exposure,
    externality_certificate,
)
from uav_mec.resource_allocation import AllocationResult


@dataclass(frozen=True)
class _RateSnapshot:
    rates: tuple[RateRecord, ...]


@dataclass(frozen=True)
class _ClosedSchedule:
    schedule: Schedule
    rate_snapshot: _RateSnapshot
    allocation: AllocationResult


def _rate(
    task_id: int,
    *,
    cross_interference_w: float = 0.2,
    rate_bps: float = 1_000_000.0,
    receiver_uav: int = 0,
    subcarrier: int = 0,
    channel_basis: ChannelBasis = ChannelBasis.PLANNING,
) -> RateRecord:
    layer = SicLayerCheck(
        task_id,
        receiver_uav,
        subcarrier,
        0,
        2.0,
        1.0,
        True,
    )
    return RateRecord(
        task_id=task_id,
        rate_bps=rate_bps,
        sinr=2.0,
        sic_feasible=True,
        cross_interference_w=cross_interference_w,
        serving_uav=receiver_uav,
        subcarrier=subcarrier,
        channel_basis=channel_basis,
        sic_layers=(layer,),
    )


def _decision(
    task_id: int,
    cpu_hz: float,
    *,
    completed: bool = True,
) -> AllocationDecision:
    return AllocationDecision(
        task_id,
        0.5,
        cpu_hz,
        completed,
        0.5 if completed else None,
        0.3 if completed else 0.0,
        "mec" if completed else "cpu_budget",
    )


def _closed(
    rate: RateRecord,
    decision: AllocationDecision,
    *,
    total_cpu_hz: float | None = None,
) -> _ClosedSchedule:
    schedule = Schedule((ScheduleEntry(rate.task_id, rate.task_id, 0, 0, "solo"),))
    return _ClosedSchedule(
        schedule,
        _RateSnapshot((rate,)),
        AllocationResult(
            (decision,),
            (decision.task_id,) if decision.completed else (),
            decision.cpu_hz if total_cpu_hz is None else total_cpu_hz,
        ),
    )


def _realized(
    closed: _ClosedSchedule,
    rate: RateRecord,
    decision: AllocationDecision,
) -> RealizedSlotOutcome:
    return RealizedSlotOutcome(
        plan_key=("unit",),
        schedule=closed.schedule,
        decisions=(decision,),
        rates=(rate,),
        completed_task_ids=(decision.task_id,) if decision.completed else (),
        sic_outage_task_ids=(),
    )


def test_certificate_conserves_two_victim_links_and_marks_move_impacts() -> None:
    before_rates = (
        _rate(1, cross_interference_w=0.3),
        _rate(2, cross_interference_w=0.3),
        _rate(3, cross_interference_w=0.2, receiver_uav=1),
    )
    after_rates = (
        _rate(1, cross_interference_w=0.0, rate_bps=1_200_000.0),
        _rate(2, cross_interference_w=0.0, rate_bps=1_200_000.0),
        _rate(
            3,
            cross_interference_w=0.0,
            receiver_uav=1,
            subcarrier=1,
        ),
    )
    before = _ClosedSchedule(
        Schedule(
            (
                ScheduleEntry(1, 1, 0, 0, "strong"),
                ScheduleEntry(2, 2, 0, 0, "weak"),
                ScheduleEntry(3, 3, 1, 0, "solo"),
            )
        ),
        _RateSnapshot(before_rates),
        AllocationResult(
            (_decision(1, 1e9), _decision(2, 1e9), _decision(3, 1e9)),
            (1, 2, 3),
            3e9,
        ),
    )
    after = _ClosedSchedule(
        Schedule(
            (
                ScheduleEntry(1, 1, 0, 0, "strong"),
                ScheduleEntry(2, 2, 0, 0, "weak"),
                ScheduleEntry(3, 3, 1, 1, "solo"),
            )
        ),
        _RateSnapshot(after_rates),
        AllocationResult(
            (_decision(1, 1e9), _decision(2, 1e9), _decision(3, 1e9)),
            (1, 2, 3),
            3e9,
        ),
    )

    certificate = externality_certificate(before, after)

    assert aggregate_victim_link_exposure(before_rates) == pytest.approx(0.8)
    assert certificate.exposure_delta_w == pytest.approx(-0.8)
    assert math.fsum(
        row.exposure_delta_w for row in certificate.block_deltas
    ) == pytest.approx(certificate.exposure_delta_w)
    receiver_zero = next(
        row
        for row in certificate.block_deltas
        if (row.receiver_uav, row.subcarrier) == (0, 0)
    )
    assert receiver_zero.victim_links_before == 2
    assert receiver_zero.exposure_before_w == pytest.approx(0.6)
    moved = next(row for row in certificate.task_impacts if row.task_id == 3)
    assert moved.source_block == (1, 0)
    assert moved.target_block == (1, 1)
    assert not moved.third_party
    victim = next(row for row in certificate.task_impacts if row.task_id == 1)
    assert victim.third_party
    assert victim.affected_receiver_blocks == ((0, 0), (1, 0), (1, 1))


def test_planning_certificate_leaves_all_realized_fields_absent() -> None:
    before = _closed(_rate(1), _decision(1, 1_000_000_000.0))
    after = _closed(
        _rate(1, rate_bps=1_100_000.0),
        _decision(1, 900_000_000.0),
    )

    certificate = externality_certificate(before, after)

    (impact,) = certificate.task_impacts
    assert (
        impact.realized_rate_before_bps,
        impact.realized_rate_after_bps,
        impact.realized_sinr_before,
        impact.realized_sinr_after,
        impact.realized_sic_before,
        impact.realized_sic_after,
    ) == (None, None, None, None, None, None)
    assert impact.allocated_cpu_before_hz == pytest.approx(1_000_000_000.0)
    assert impact.allocated_cpu_after_hz == pytest.approx(900_000_000.0)
    assert impact.third_party


def test_realized_admission_and_decision_drive_allocated_cpu_fields() -> None:
    before = _closed(_rate(1), _decision(1, 1_000_000_000.0))
    after = _closed(
        _rate(1, rate_bps=1_100_000.0),
        _decision(1, 900_000_000.0),
    )
    realized_before = _realized(
        before,
        _rate(1, channel_basis=ChannelBasis.REALIZED),
        _decision(1, 800_000_000.0),
    )
    realized_after = _realized(
        after,
        _rate(1, channel_basis=ChannelBasis.REALIZED),
        _decision(1, 600_000_000.0, completed=False),
    )

    certificate = externality_certificate(
        before,
        after,
        realized_before=realized_before,
        realized_after=realized_after,
    )

    (impact,) = certificate.task_impacts
    assert impact.minimum_cpu_before_hz == pytest.approx(1_000_000_000.0)
    assert impact.minimum_cpu_after_hz == pytest.approx(900_000_000.0)
    assert impact.allocated_cpu_before_hz == pytest.approx(800_000_000.0)
    assert impact.allocated_cpu_after_hz == pytest.approx(600_000_000.0)
    assert impact.admitted_before and impact.admitted_after
    assert impact.completed_before and not impact.completed_after
    assert certificate.actual_allocated_cpu_delta_hz == pytest.approx(
        -200_000_000.0
    )
    assert certificate.actual_allocated_cpu_delta_hz == pytest.approx(
        math.fsum(
            impact.allocated_cpu_after_hz - impact.allocated_cpu_before_hz
            for impact in certificate.task_impacts
        )
    )
    assert certificate.admitted_cpu_delta_hz == pytest.approx(-100_000_000.0)
    assert certificate.admitted_cpu_delta_hz != pytest.approx(
        certificate.actual_allocated_cpu_delta_hz
    )


@pytest.mark.parametrize("missing", ("rate", "decision"))
def test_result_audit_requires_every_scheduled_task_input(missing: str) -> None:
    before = _closed(_rate(1), _decision(1, 1_000_000_000.0))
    after = _closed(_rate(1), _decision(1, 900_000_000.0))
    realized_before = _realized(
        before,
        _rate(1, channel_basis=ChannelBasis.REALIZED),
        _decision(1, 800_000_000.0),
    )
    if missing == "rate":
        realized_before = replace(realized_before, rates=())
    else:
        realized_before = replace(realized_before, decisions=())
    realized_after = _realized(
        after,
        _rate(1, channel_basis=ChannelBasis.REALIZED),
        _decision(1, 700_000_000.0),
    )

    with pytest.raises(ValueError, match="scheduled task"):
        externality_certificate(
            before,
            after,
            realized_before=realized_before,
            realized_after=realized_after,
        )


@pytest.mark.parametrize(
    "invalid_side",
    ("planning_rate", "realized_rate", "decision", "allocation_total"),
)
def test_certificate_rejects_nan_and_infinite_inputs(invalid_side: str) -> None:
    before_rate = _rate(1)
    before_decision = _decision(1, 1_000_000_000.0)
    total_cpu_hz = 1_000_000_000.0
    realized_rate = _rate(1, channel_basis=ChannelBasis.REALIZED)
    if invalid_side == "planning_rate":
        before_rate = replace(before_rate, cross_interference_w=math.nan)
    elif invalid_side == "realized_rate":
        realized_rate = replace(realized_rate, rate_bps=math.inf)
    elif invalid_side == "decision":
        before_decision = replace(before_decision, cpu_hz=math.inf)
    else:
        total_cpu_hz = math.nan

    before = _closed(
        before_rate,
        before_decision,
        total_cpu_hz=total_cpu_hz,
    )
    after = _closed(_rate(1), _decision(1, 900_000_000.0))
    realized_before = _realized(
        before,
        realized_rate,
        _decision(1, 800_000_000.0),
    )
    realized_after = _realized(
        after,
        _rate(1, channel_basis=ChannelBasis.REALIZED),
        _decision(1, 700_000_000.0),
    )

    with pytest.raises(ValueError, match="finite"):
        externality_certificate(
            before,
            after,
            realized_before=realized_before,
            realized_after=realized_after,
        )


def test_certificate_rejects_one_sided_realized_audit() -> None:
    before = _closed(_rate(1), _decision(1, 1_000_000_000.0))
    after = _closed(_rate(1), _decision(1, 900_000_000.0))
    realized_before = _realized(
        before,
        _rate(1, channel_basis=ChannelBasis.REALIZED),
        _decision(1, 800_000_000.0),
    )

    with pytest.raises(ValueError, match="both realized"):
        externality_certificate(
            before,
            after,
            realized_before=realized_before,
        )


def test_active_outcome_schema_recomputes_victim_link_exposure() -> None:
    rates = (_rate(1, cross_interference_w=0.2), _rate(2, cross_interference_w=0.3))
    outcome = SlotOutcome(
        slot_index=0,
        arrived_task_ids=(1, 2),
        completed_task_ids=(),
        missed_task_ids=(1, 2),
        decisions=(),
        rates=rates,
        total_slack_s=0.0,
        total_cross_interference_w=99.0,
    )

    assert aggregate_victim_link_exposure(rates) == pytest.approx(0.5)
    assert outcome.aggregate_victim_link_exposure_w == pytest.approx(0.5)
