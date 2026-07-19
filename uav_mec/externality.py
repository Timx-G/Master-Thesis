"""Victim-link exposure accounting and auditable move certificates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterable

from .domain import RateRecord

@dataclass(frozen=True, slots=True)
class BlockExposureDelta:
    receiver_uav: int; subcarrier: int
    victim_links_before: int; victim_links_after: int
    exposure_before_w: float; exposure_after_w: float; exposure_delta_w: float

@dataclass(frozen=True, slots=True)
class TaskImpact:
    task_id: int; source_block: tuple[int,int] | None; target_block: tuple[int,int] | None
    role_before: str | None; role_after: str | None
    planned_rate_before_bps: float | None; planned_rate_after_bps: float | None
    planned_sinr_before: float | None; planned_sinr_after: float | None
    planned_sic_before: bool | None; planned_sic_after: bool | None
    realized_rate_before_bps: float | None; realized_rate_after_bps: float | None
    realized_sinr_before: float | None; realized_sinr_after: float | None
    realized_sic_before: bool | None; realized_sic_after: bool | None
    minimum_cpu_before_hz: float | None; minimum_cpu_after_hz: float | None
    admitted_before: bool; admitted_after: bool; completed_before: bool; completed_after: bool
    third_party: bool; affected_receiver_blocks: tuple

@dataclass(frozen=True, slots=True)
class ExternalityCertificate:
    completion_delta: int; admitted_cpu_delta_hz: float; actual_allocated_cpu_delta_hz: float
    exposure_delta_w: float; block_deltas: tuple; task_impacts: tuple; affected_receiver_blocks: tuple

def aggregate_victim_link_exposure(rates: Iterable[RateRecord]) -> float:
    return math.fsum(float(r.cross_interference_w) for r in rates)

def _blocks(rates):
    out = {}
    for r in rates:
        key=(r.serving_uav,r.subcarrier)
        n,e=out.get(key,(0,0.0)); out[key]=(n+1,e+float(r.cross_interference_w))
    return out

def externality_certificate(before, after, *, realized_before=None, realized_after=None):
    b=_blocks(before.rate_snapshot.rates); a=_blocks(after.rate_snapshot.rates)
    keys=tuple(sorted(set(b)|set(a))); rows=[]
    for k in keys:
        bn,be=b.get(k,(0,0.0)); an,ae=a.get(k,(0,0.0)); rows.append(BlockExposureDelta(k[0],k[1],bn,an,be,ae,ae-be))
    delta=aggregate_victim_link_exposure(after.rate_snapshot.rates)-aggregate_victim_link_exposure(before.rate_snapshot.rates)
    if not math.isclose(math.fsum(x.exposure_delta_w for x in rows),delta,rel_tol=1e-12,abs_tol=1e-18): raise ValueError('externality certificate does not conserve victim-link exposure')
    def cpu(x): return float(getattr(x.allocation,'total_cpu_hz',0.0))
    return ExternalityCertificate(len(after.allocation.completed_task_ids)-len(before.allocation.completed_task_ids),cpu(after)-cpu(before),cpu(after)-cpu(before),delta,tuple(rows),tuple(),keys)
