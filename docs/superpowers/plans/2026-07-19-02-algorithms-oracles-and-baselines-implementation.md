# Algorithms, Oracles, and Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make externality accounting, closure-normalized IEA-CAP search, incremental evaluation, DD-MCAA, Exact Admission, Exact Schedule, and a Guo-JAGP adaptation share the semantic model established by Plan 1 while retaining distinct claims and result types.

**Architecture:** `uav_mec/externality.py` owns victim-link exposure and move certificates. `uav_mec/incremental.py` owns the dependency closure and equivalence-preserving partial recomputation. Every raw search move is rated and passed through `uav_mec/closure.py` before deduplication or comparison. DD-MCAA is a static virtual-CPU admission response; Exact Admission verifies that conditional response, while Exact Schedule enumerates complete radio states and their policy closure. `uav_mec/baselines_adapted.py` contains literature adaptations only. Immutable `MethodSpec` values make structural ablations and the joint 2-by-2 design auditable.

**Tech Stack:** Python 3.11 dataclasses and enums, NumPy, SciPy, itertools exhaustive enumeration, pytest, TOML, JSON certificates.

**Execution rule:** Run tests, static audits, and configuration parsers only. Do not run a pilot, formal experiment, shard suite, analysis, canonical publication, or manuscript result regeneration.

**Prerequisite:** Start from the clean isolated worktree after all Plan 1 commits and its semantic gate. All commands below run from `/home/link/my_workspace/.worktrees/master-thesis-ccfb`.

---

### Task 1: Conserve aggregate victim-link interference exposure

**Files:**
- Create: `uav_mec/externality.py`
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/domain.py`
- Modify: `uav_mec/scheduler.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_search.py`
- Modify: `tests/test_externality_analysis.py`

- [ ] **Step 1: Write the RED certificate-conservation test**

Add these complete local builders and test to `tests/test_search.py`:

```python
import math

import pytest

from uav_mec.closure import ClosedExecutableSchedule, RawRateSnapshot
from uav_mec.domain import (
    AllocationDecision,
    ChannelBasis,
    RateRecord,
    RealizedSlotOutcome,
    Schedule,
    ScheduleEntry,
    SicLayerCheck,
)
from uav_mec.externality import aggregate_victim_link_exposure, externality_certificate
from uav_mec.resource_allocation import AllocationPolicy, AllocationResult


def _certificate_rate(
    task_id: int,
    uav: int,
    subcarrier: int,
    cross_w: float,
    channel_basis: ChannelBasis = ChannelBasis.PLANNING,
    rate_bps: float = 1_000_000.0,
    sinr: float = 2.0,
) -> RateRecord:
    layer = SicLayerCheck(
        task_id=task_id,
        receiver_uav=uav,
        subcarrier=subcarrier,
        decode_rank=0,
        sinr=sinr,
        threshold=1.0,
        feasible=True,
    )
    return RateRecord(
        task_id=task_id,
        rate_bps=rate_bps,
        sinr=sinr,
        cross_interference_w=cross_w,
        serving_uav=uav,
        subcarrier=subcarrier,
        channel_basis=channel_basis,
        sic_layers=(layer,),
    )


def _certificate_closed(schedule: Schedule, rates: tuple) -> ClosedExecutableSchedule:
    decisions = tuple(
        AllocationDecision(
            task_id=task_id,
            alpha=0.5,
            cpu_hz=1_000_000_000.0,
            completed=True,
            delay_s=0.5,
            slack_s=0.3,
            reason="mec",
        )
        for task_id in schedule.task_ids
    )
    allocation = AllocationResult(
        decisions=decisions,
        completed_task_ids=schedule.task_ids,
        total_cpu_hz=float(len(decisions)) * 1_000_000_000.0,
    )
    rated = RawRateSnapshot(
        assignment=schedule,
        rates=rates,
        channel_basis=ChannelBasis.PLANNING,
        semantic_fingerprint="certificate-unit-v1",
    )
    return ClosedExecutableSchedule(
        schedule=schedule,
        rate_snapshot=rated,
        allocation=allocation,
        closure_rounds=1,
        allocation_policy=AllocationPolicy.DEADLINE,
    )


def _certificate_realized(
    closed: ClosedExecutableSchedule,
    rates: tuple,
) -> RealizedSlotOutcome:
    return RealizedSlotOutcome(
        plan_key=("certificate-unit-v1",),
        schedule=closed.schedule,
        decisions=closed.allocation.decisions,
        rates=rates,
        completed_task_ids=closed.allocation.completed_task_ids,
        sic_outage_task_ids=(),
    )


def test_externality_certificate_conserves_two_victim_links() -> None:
    before_schedule = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(2, 2, 0, 0, "weak"),
            ScheduleEntry(3, 3, 1, 0, "solo"),
        )
    )
    after_schedule = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(2, 2, 0, 0, "weak"),
            ScheduleEntry(3, 3, 1, 1, "solo"),
        )
    )
    before = _certificate_closed(
        before_schedule,
        (
            _certificate_rate(1, 0, 0, 0.3),
            _certificate_rate(2, 0, 0, 0.3),
            _certificate_rate(3, 1, 0, 0.2),
        ),
    )
    after = _certificate_closed(
        after_schedule,
        (
            _certificate_rate(1, 0, 0, 0.0, rate_bps=1_200_000.0, sinr=2.5),
            _certificate_rate(2, 0, 0, 0.0, rate_bps=1_200_000.0, sinr=2.5),
            _certificate_rate(3, 1, 1, 0.0),
        ),
    )
    realized_before = _certificate_realized(
        before,
        (
            _certificate_rate(1, 0, 0, 0.4, ChannelBasis.REALIZED),
            _certificate_rate(2, 0, 0, 0.4, ChannelBasis.REALIZED),
            _certificate_rate(3, 1, 0, 0.1, ChannelBasis.REALIZED),
        ),
    )
    realized_after = _certificate_realized(
        after,
        (
            _certificate_rate(1, 0, 0, 0.0, ChannelBasis.REALIZED, rate_bps=1_300_000.0, sinr=2.8),
            _certificate_rate(2, 0, 0, 0.0, ChannelBasis.REALIZED, rate_bps=1_300_000.0, sinr=2.8),
            _certificate_rate(3, 1, 1, 0.0, ChannelBasis.REALIZED),
        ),
    )
    certificate = externality_certificate(
        before,
        after,
        realized_before=realized_before,
        realized_after=realized_after,
    )
    objective_delta = aggregate_victim_link_exposure(after.rate_snapshot.rates) - aggregate_victim_link_exposure(before.rate_snapshot.rates)
    assert aggregate_victim_link_exposure(before.rate_snapshot.rates) == pytest.approx(0.8)
    assert math.fsum(row.exposure_delta_w for row in certificate.block_deltas) == pytest.approx(objective_delta)
    assert certificate.exposure_delta_w == pytest.approx(objective_delta)
    receiver_zero = next(row for row in certificate.block_deltas if (row.receiver_uav, row.subcarrier) == (0, 0))
    assert receiver_zero.victim_links_before == 2
    assert receiver_zero.exposure_before_w == pytest.approx(0.6)
    moved = next(row for row in certificate.task_impacts if row.task_id == 3)
    assert moved.source_block == (1, 0)
    assert moved.target_block == (1, 1)
    assert moved.role_before == moved.role_after == "solo"
    assert moved.planned_rate_before_bps == pytest.approx(1_000_000.0)
    assert moved.planned_sinr_before == pytest.approx(2.0)
    assert moved.planned_sic_before and moved.planned_sic_after
    assert moved.realized_rate_after_bps == pytest.approx(1_000_000.0)
    assert moved.realized_sinr_after == pytest.approx(2.0)
    assert moved.realized_sic_before and moved.realized_sic_after
    assert moved.minimum_cpu_before_hz == pytest.approx(1_000_000_000.0)
    assert moved.minimum_cpu_after_hz == pytest.approx(1_000_000_000.0)
    assert moved.admitted_before and moved.admitted_after
    assert moved.completed_before and moved.completed_after
    assert not moved.third_party
    victim = next(row for row in certificate.task_impacts if row.task_id == 1)
    assert victim.third_party
    assert victim.affected_receiver_blocks == ((0, 0), (1, 0), (1, 1))
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_search.py::test_externality_certificate_conserves_two_victim_links -q
```

Expected: import fails because `uav_mec.externality` does not exist, or the old block-power certificate undercounts the two victim links.

- [ ] **Step 3: Implement the normative metric and certificate**

Create these public types and functions in `uav_mec/externality.py`:

```python
@dataclass(frozen=True, slots=True)
class BlockExposureDelta:
    receiver_uav: int
    subcarrier: int
    victim_links_before: int
    victim_links_after: int
    exposure_before_w: float
    exposure_after_w: float
    exposure_delta_w: float


@dataclass(frozen=True, slots=True)
class TaskImpact:
    task_id: int
    source_block: tuple[int, int] | None
    target_block: tuple[int, int] | None
    role_before: str | None
    role_after: str | None
    planned_rate_before_bps: float | None
    planned_rate_after_bps: float | None
    planned_sinr_before: float | None
    planned_sinr_after: float | None
    planned_sic_before: bool | None
    planned_sic_after: bool | None
    realized_rate_before_bps: float | None
    realized_rate_after_bps: float | None
    realized_sinr_before: float | None
    realized_sinr_after: float | None
    realized_sic_before: bool | None
    realized_sic_after: bool | None
    minimum_cpu_before_hz: float | None
    minimum_cpu_after_hz: float | None
    admitted_before: bool
    admitted_after: bool
    completed_before: bool
    completed_after: bool
    third_party: bool
    affected_receiver_blocks: tuple


@dataclass(frozen=True, slots=True)
class ExternalityCertificate:
    completion_delta: int
    admitted_cpu_delta_hz: float
    actual_allocated_cpu_delta_hz: float
    exposure_delta_w: float
    block_deltas: tuple
    task_impacts: tuple
    affected_receiver_blocks: tuple


def aggregate_victim_link_exposure(rates: Iterable[RateRecord]) -> float:
    return math.fsum(record.cross_interference_w for record in rates)


def externality_certificate(
    before: ClosedExecutableSchedule,
    after: ClosedExecutableSchedule,
    *,
    realized_before: RealizedSlotOutcome | None = None,
    realized_after: RealizedSlotOutcome | None = None,
) -> ExternalityCertificate:
    before_blocks = _block_exposure(before.rate_snapshot.rates)
    after_blocks = _block_exposure(after.rate_snapshot.rates)
    keys = tuple(sorted(before_blocks.keys() | after_blocks.keys()))
    rows = tuple(_delta_row(key, before_blocks.get(key), after_blocks.get(key)) for key in keys)
    delta = aggregate_victim_link_exposure(after.rate_snapshot.rates) - aggregate_victim_link_exposure(before.rate_snapshot.rates)
    if not math.isclose(math.fsum(row.exposure_delta_w for row in rows), delta, rel_tol=1e-12, abs_tol=1e-18):
        raise ValueError("externality certificate does not conserve victim-link exposure")
    return ExternalityCertificate(
        completion_delta=len(after.allocation.completed_task_ids) - len(before.allocation.completed_task_ids),
        admitted_cpu_delta_hz=after.allocation.total_cpu_hz - before.allocation.total_cpu_hz,
        exposure_delta_w=delta,
        block_deltas=rows,
        task_impacts=_task_impacts(before, after, realized_before, realized_after, keys),
        affected_receiver_blocks=keys,
    )
```

Implement `_block_exposure` by grouping rate records on `(serving_uav, subcarrier)`, counting records, and summing each record's `cross_interference_w`; never multiply a physical block value by an assumed occupancy after grouping. `_task_impacts` joins schedule entries, planning rates, optional realized rates, and allocation decisions by stable task ID. It marks a row `third_party=True` only when its block and role did not change but at least one radio, SIC, CPU, admission, or completion field changed. Every row carries the sorted union of affected receiver blocks. When realized inputs are absent during planning search, all six realized scalar/status fields are `None`; result-audit certificates must pass both realized outcomes and populate them. `actual_allocated_cpu_delta_hz` is computed from completed/admitted task allocations after closure and must conserve exactly with the per-task CPU rows; a certificate with exposure-only or requested-CPU-only accounting is rejected. Make the evaluator objective, scheduler comparison/certificate, result summary, and externality analysis call `aggregate_victim_link_exposure`. Compatibility output may retain `total_cross_interference_w`, but active schemas and prose add `aggregate_victim_link_exposure_w`.

- [ ] **Step 4: Run GREEN**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_evaluation.py tests/test_search.py tests/test_externality_analysis.py -q
```

Expected: all selected tests pass and the conservation assertion covers completion, CPU, and exposure layers.

- [ ] **Step 5: Commit the explicit files**

```bash
git add -- uav_mec/externality.py uav_mec/evaluation.py uav_mec/domain.py uav_mec/scheduler.py tests/test_evaluation.py tests/test_search.py tests/test_externality_analysis.py
git diff --cached --check
git commit -m "fix: conserve victim-link externality certificates"
```

---

### Task 2: Search the closure-normalized neighborhood and certify termination

**Files:**
- Modify: `uav_mec/closure.py`
- Modify: `uav_mec/scheduler.py`
- Modify: `tests/test_schedule_closure.py`
- Modify: `tests/test_neighborhoods.py`
- Modify: `tests/test_search.py`
- Modify: `tests/test_mechanism_controls.py`

- [ ] **Step 1: Write the RED closure-search tests**

Add this complete builder and tests to `tests/test_search.py`:

```python
import dataclasses

import numpy as np

from uav_mec.closure import close_assignment, rate_raw_assignment
from uav_mec.domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy
from uav_mec.scheduler import SearchConfig, close_and_deduplicate_moves, optimize_schedule


def _closure_search_case() -> tuple[SlotSnapshot, Schedule, Schedule, NOMAChannel, EvaluationConfig]:
    tasks = tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=1_000_000.0,
            comp_cycles=1_800_000_000.0,
            deadline_s=0.8,
            local_cpu_hz=500_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in (1, 2, 3)
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[9e-8], [4e-8], [2e-8]]), 1)
    left = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(2, 2, 0, 0, "weak"),
        )
    )
    right = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(3, 3, 0, 0, "weak"),
        )
    )
    cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.01,
        cpu_per_uav_hz=1_100_000_000.0,
        allocation_policy=AllocationPolicy.DEADLINE,
    )
    return snapshot, left, right, NOMAChannel(ChannelConfig(), 100.0), cfg


def test_raw_moves_are_deduplicated_only_after_closure() -> None:
    snapshot, left, right, channel, cfg = _closure_search_case()
    closed = close_and_deduplicate_moves(
        snapshot,
        (left, right),
        channel,
        cfg,
        channel_basis=ChannelBasis.PLANNING,
    )
    expected = {
        close_assignment(
            snapshot,
            rate_raw_assignment(snapshot, raw, channel, cfg, channel_basis=ChannelBasis.PLANNING),
            channel,
            cfg,
        ).stable_key()
        for raw in (left, right)
    }
    assert {candidate.stable_key() for candidate in closed} == expected
    assert len(closed) == len(expected)


def test_iteration_cap_is_not_a_local_optimum_certificate() -> None:
    snapshot, left, _right, channel, cfg = _closure_search_case()
    initial = close_assignment(
        snapshot,
        rate_raw_assignment(snapshot, left, channel, cfg, channel_basis=ChannelBasis.PLANNING),
        channel,
        cfg,
    )
    result = optimize_schedule(snapshot, initial, channel, cfg, SearchConfig(max_iterations=0))
    assert result.stop_reason == "iteration_cap"
    assert not result.local_optimum_certified
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_search.py::test_raw_moves_are_deduplicated_only_after_closure tests/test_search.py::test_iteration_cap_is_not_a_local_optimum_certificate -q
```

Expected: the search compares raw schedules, lacks post-closure deduplication, or reports an iteration cap as an ordinary optimum.

- [ ] **Step 3: Implement the exact search sequence**

For every current executable state `x` and every legal raw move `m`, perform this sequence once:

```text
raw_candidate = m(x.schedule)
rated_candidate = rate_raw_assignment(raw_candidate, basis=PLANNING)
closed_candidate = close_assignment(rated_candidate, policy=method.allocation_policy)
candidate_key = (closed_candidate.stable_key(), search_semantic_fingerprint)
compare Phi(closed_candidate) with Phi(x)
```

Expose `enumerate_raw_moves(snapshot, schedule, cfg)` and `close_and_deduplicate_moves(snapshot, raw_moves, channel, cfg, *, channel_basis)` in `scheduler.py`. Preserve Replace, Position, and CrossUAV moves, including insertions into empty positions and removals. Deduplicate only by the closed key plus the Plan 1 semantic fingerprint; two raw moves that close to one state receive one objective evaluation. Use the quantized lexicographic objective `(completed_count, -total_mec_cpu_hz, -aggregate_victim_link_exposure_w)` with stable task-ID tie-breaking.

Return `stop_reason="no_improving_move"` and `local_optimum_certified=True` only after exhaustive enumeration of the configured closed neighborhood. Return `stop_reason="iteration_cap"` and `local_optimum_certified=False` whenever the guard fires. Preserve raw move count, unique closed candidate count, cache hits, accepted moves, and closure rounds in the search result.

- [ ] **Step 4: Run GREEN and neighborhood coverage**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_schedule_closure.py tests/test_neighborhoods.py tests/test_search.py tests/test_mechanism_controls.py -q
```

Expected: all selected tests pass; an iteration-cap result cannot support the local-optimum claim.

- [ ] **Step 5: Commit the explicit files**

```bash
git add -- uav_mec/closure.py uav_mec/scheduler.py tests/test_schedule_closure.py tests/test_neighborhoods.py tests/test_search.py tests/test_mechanism_controls.py
git diff --cached --check
git commit -m "refactor: search closed executable schedules"
```

---

### Task 3: Move incremental semantics into `uav_mec/incremental.py` and prove 10,000 actions

**Files:**
- Create: `uav_mec/incremental.py`
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/scheduler.py`
- Modify: `tests/test_incremental_evaluation.py`
- Modify: `tests/test_search.py`

- [ ] **Step 1: Write the RED dependency-closure tests**

Replace the existing parameterized body in `tests/test_incremental_evaluation.py` with these complete helpers and test:

```python
import numpy as np

from uav_mec.closure import close_assignment, rate_raw_assignment
from uav_mec.domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, InfeasibleScheduleError
from uav_mec.incremental import dependency_closure, evaluate_move_incremental
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy
from uav_mec.scheduler import enumerate_raw_moves


def _random_incremental_case(seed: int) -> tuple[SlotSnapshot, Schedule, NOMAChannel, EvaluationConfig]:
    rng = np.random.default_rng(seed)
    tasks = tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=float(rng.uniform(600_000.0, 1_200_000.0)),
            comp_cycles=float(rng.uniform(1.2e9, 1.8e9)),
            deadline_s=0.8,
            local_cpu_hz=400_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in range(12)
    )
    gains = rng.uniform(2e-8, 9e-8, size=(12, 2))
    snapshot = SlotSnapshot(0, tasks, gains, 2)
    entries = []
    for index, task in enumerate(tasks):
        uav = (index // 2) % 2
        subcarrier = (index // 4) % 3
        role = "strong" if index % 2 == 0 else "weak"
        entries.append(ScheduleEntry(task.task_id, task.user_id, uav, subcarrier, role))
    cfg = EvaluationConfig(
        n_subcarriers=3,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.001,
        cpu_per_uav_hz=20e9,
        allocation_policy=AllocationPolicy.DEADLINE,
    )
    return snapshot, Schedule(tuple(entries)), NOMAChannel(ChannelConfig(), 100.0), cfg


def _full_close(snapshot: SlotSnapshot, raw: Schedule, channel: NOMAChannel, cfg: EvaluationConfig):
    return close_assignment(
        snapshot,
        rate_raw_assignment(snapshot, raw, channel, cfg, channel_basis=ChannelBasis.PLANNING),
        channel,
        cfg,
    )


def _assert_closed_equal(left, right) -> None:
    assert left.stable_key() == right.stable_key()
    assert left.rate_snapshot.rates == right.rate_snapshot.rates
    assert left.allocation.decisions == right.allocation.decisions
    assert left.allocation.completed_task_ids == right.allocation.completed_task_ids


def test_ten_thousand_random_actions_match_full_closure() -> None:
    action_count = 0
    for seed in range(100):
        snapshot, initial_raw, channel, cfg = _random_incremental_case(seed)
        before = _full_close(snapshot, initial_raw, channel, cfg)
        rng = np.random.default_rng(seed + 100_000)
        for _move_index in range(100):
            moves = tuple(enumerate_raw_moves(snapshot, before.schedule, cfg))
            assert moves
            start = int(rng.integers(0, len(moves)))
            chosen = before.schedule
            full = before
            for offset in range(len(moves)):
                candidate = moves[(start + offset) % len(moves)]
                try:
                    full = _full_close(snapshot, candidate, channel, cfg)
                except InfeasibleScheduleError:
                    continue
                chosen = candidate
                break
            incremental = evaluate_move_incremental(snapshot, before, chosen, channel, cfg)
            _assert_closed_equal(incremental, full)
            before = full
            action_count += 1
    assert action_count == 10_000


def test_cross_uav_move_dependency_contains_receivers_roles_and_cpu_peers() -> None:
    snapshot, before, _channel, cfg = _random_incremental_case(301)
    candidate_entries = []
    for entry in before.entries:
        if entry.task_id == 0:
            candidate_entries.append(dataclasses.replace(entry, serving_uav=1))
        elif entry.task_id == 2:
            candidate_entries.append(dataclasses.replace(entry, serving_uav=0))
        else:
            candidate_entries.append(entry)
    candidate = Schedule(tuple(candidate_entries))
    dependencies = dependency_closure(snapshot, before, candidate, cfg)
    assert dependencies.affected_subcarriers == (0,)
    assert dependencies.receiver_blocks == ((0, 0), (1, 0))
    assert dependencies.reassigned_task_ids == (0, 2)
    assert {0, 1, 2, 3} <= set(dependencies.role_task_ids)
    assert dependencies.cpu_uavs == (0, 1)
    assert set(dependencies.cpu_task_ids) == set(range(12))
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_incremental_evaluation.py -q
```

Expected: import fails because `uav_mec.incremental` does not exist, or the 10,000-action equivalence test exposes a missing dependency.

- [ ] **Step 3: Implement the dependency closure and incremental evaluator**

Add this public record:

```python
@dataclass(frozen=True, slots=True)
class IncrementalDependencyClosure:
    affected_subcarriers: tuple
    receiver_blocks: tuple
    role_task_ids: tuple
    reassigned_task_ids: tuple
    cpu_uavs: tuple
    cpu_task_ids: tuple
```

`dependency_closure` must include all UAV receiver blocks on every source or target subcarrier, because a transmitter changes interference at every receiver using that subcarrier. It must include all tasks in groups whose membership or decode order changes, every task whose serving UAV changes, every old or new MEC server for a rate/association-affected task, and every task participating in the allocation ordering on those servers.

`evaluate_move_incremental` recomputes the dependency set, reuses only rate records outside it with the same semantic fingerprint, applies the same allocation policy, and runs the same deletion-only closure as a full evaluation. It returns a `ClosedExecutableSchedule`, never a reduced score with different semantics. Remove the duplicate incremental implementation from `evaluation.py`; thin compatibility wrappers may import the new functions but may not retain logic.

- [ ] **Step 4: Run GREEN**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_incremental_evaluation.py tests/test_search.py -q
```

Expected: all selected tests pass and the test reports exactly 10,000 checked actions.

- [ ] **Step 5: Commit the explicit files**

```bash
git add -- uav_mec/incremental.py uav_mec/evaluation.py uav_mec/scheduler.py tests/test_incremental_evaluation.py tests/test_search.py
git diff --cached --check
git commit -m "refactor: formalize incremental dependency closure"
```

---

### Task 4: Verify DD-MCAA against a distinct Exact Admission Oracle

**Files:**
- Modify: `uav_mec/resource_allocation.py`
- Modify: `uav_mec/latency.py`
- Modify: `tests/test_resource_requirements.py`
- Modify: `tests/test_resource_allocator.py`
- Modify: `tests/test_resource_policies.py`
- Create: `tests/test_exact_admission.py`

- [ ] **Step 1: Write RED analytic and exhaustive tests**

Create `tests/test_exact_admission.py`:

```python
import itertools
import math

import numpy as np
import pytest

from uav_mec.domain import Task
from uav_mec.resource_allocation import (
    ExactAdmissionResult,
    exact_admission,
    minimum_cpu_prefix_admission,
    required_resources,
)


def _admission_tasks(seed: int, count: int = 6) -> tuple:
    rng = np.random.default_rng(seed)
    return tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=float(rng.uniform(300_000.0, 900_000.0)),
            comp_cycles=float(rng.uniform(8e8, 1.8e9)),
            deadline_s=float(rng.uniform(0.6, 0.9)),
            local_cpu_hz=400_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in range(count)
    )


def _reference_subset(tasks: tuple, rates: dict[int, float], budget: float) -> tuple:
    requirements = {
        task.task_id: required_resources(task, rates[task.task_id])
        for task in tasks
    }
    feasible = tuple(task.task_id for task in tasks if requirements[task.task_id].feasible)
    candidates = []
    for width in range(len(feasible) + 1):
        for subset in itertools.combinations(feasible, width):
            cpu = math.fsum(requirements[task_id].cpu_hz for task_id in subset)
            if cpu <= budget:
                candidates.append((tuple(sorted(subset)), cpu))
    selected, _cpu = min(
        candidates,
        key=lambda row: (-len(row[0]), row[1], row[0]),
    )
    return selected


def test_dd_matches_exact_admission_on_small_random_instances() -> None:
    for seed in range(100):
        tasks = _admission_tasks(seed)
        rates = {task.task_id: 1_000_000.0 + 50_000.0 * task.task_id for task in tasks}
        budget = 4_000_000_000.0
        dd = minimum_cpu_prefix_admission(tasks, rates, budget)
        exact = exact_admission(tasks, rates, budget)
        reference = _reference_subset(tasks, rates, budget)
        assert isinstance(exact, ExactAdmissionResult)
        assert tuple(sorted(dd.completed_task_ids)) == exact.admitted_task_ids == reference
        assert dd.total_cpu_hz == pytest.approx(exact.total_cpu_hz)


def test_minimum_cpu_occurs_at_alpha_min_boundary() -> None:
    task = _admission_tasks(7, count=1)[0]
    requirement = required_resources(task, 2_000_000.0)
    next_alpha = math.nextafter(requirement.alpha, 1.0)
    next_requirement = required_resources(task, 2_000_000.0, alpha=next_alpha)
    assert next_requirement.cpu_hz >= requirement.cpu_hz
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_exact_admission.py -q
```

Expected: `ExactAdmissionResult`, the public Oracle, or the explicit boundary evaluation is absent.

- [ ] **Step 3: Implement the conditional admission Oracle and DD contract**

Add a public `ExactAdmissionResult` with `admitted_task_ids`, `rejected_task_ids`, `requirements`, `total_cpu_hz`, and `enumerated_subsets`. `exact_admission(tasks, uplink_rates_bps, cpu_budget_hz, downlink_rates_bps=None)` enumerates subsets of the same closure-before task set and frozen rates, maximizes admitted cardinality, minimizes total CPU within that cardinality, and breaks remaining ties by sorted task IDs. It does not receive or return a radio schedule.

Expose the production response as `minimum_cpu_prefix_admission`. Give `required_resources` a keyword-only `alpha: float | None = None`: `None` selects the analytic alpha-min, while an explicit validated alpha evaluates that boundary without changing admission policy. Compute `alpha_min = clip(1 - remaining_time_s * local_cpu_hz / comp_cycles, 0, 1)`. For positive alpha, compute transfer-per-alpha as `data_bits / uplink_rate_bps + result_ratio * data_bits / downlink_rate_bps`; require the independent downlink rate only for positive task rho. Compute minimum CPU at alpha-min, sort feasible items by `(cpu_hz, task_id)`, and choose the longest budget-feasible prefix. Use `math.fsum` and the existing ULP guards. Keep Deadline-First-MinCPU a separate ordering policy and never call it EDF scheduling.

- [ ] **Step 4: Run GREEN and all resource regressions**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_resource_requirements.py tests/test_resource_allocator.py tests/test_resource_policies.py tests/test_exact_admission.py -q
```

Expected: all selected tests pass, including 100 independent exact comparisons.

- [ ] **Step 5: Commit the explicit files**

```bash
git add -- uav_mec/resource_allocation.py uav_mec/latency.py tests/test_resource_requirements.py tests/test_resource_allocator.py tests/test_resource_policies.py tests/test_exact_admission.py
git diff --cached --check
git commit -m "feat: separate exact admission from dd response"
```

---

### Task 5: Build a distinct closure-complete Exact Schedule Oracle

**Files:**
- Modify: `uav_mec/exact.py`
- Modify: `uav_mec/closure.py`
- Modify: `uav_mec/run_exact_benchmark.py`
- Modify: `tests/test_exact_benchmark.py`
- Modify: `tests/test_schedule_closure.py`
- Modify: `tests/test_exact_admission.py`

- [ ] **Step 1: Write the RED result-type and certificate test**

Add the following complete case to `tests/test_exact_benchmark.py`:

```python
import numpy as np

from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.exact import ExactScheduleResult, exact_schedule
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy, ExactAdmissionResult


def _tiny_exact_case() -> tuple[SlotSnapshot, NOMAChannel, EvaluationConfig]:
    tasks = tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=700_000.0,
            comp_cycles=1_500_000_000.0,
            deadline_s=0.8,
            local_cpu_hz=400_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in (1, 2, 3)
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[9e-8], [5e-8], [2e-8]]), 1)
    cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.01,
        cpu_per_uav_hz=2_000_000_000.0,
        allocation_policy=AllocationPolicy.DEADLINE,
    )
    return snapshot, NOMAChannel(ChannelConfig(), 100.0), cfg


def test_exact_schedule_has_a_distinct_complete_certificate() -> None:
    snapshot, channel, cfg = _tiny_exact_case()
    result = exact_schedule(snapshot, channel, cfg)
    assert isinstance(result, ExactScheduleResult)
    assert not isinstance(result, ExactAdmissionResult)
    assert result.certificate.complete
    assert result.certificate.raw_states_enumerated > result.certificate.unique_closed_states
    assert result.certificate.best_bound == result.objective
    assert result.certificate.lexicographic_gap == (0, 0.0, 0.0)
    assert set(result.plan.schedule.task_ids) == set(result.plan.allocation.completed_task_ids)
    assert result.certificate.cpu_nontrivial
    assert result.certificate.schedule_nontrivial
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_exact_benchmark.py::test_exact_schedule_has_a_distinct_complete_certificate tests/test_exact_admission.py -q
```

Expected: the old generic `ExactResult` conflates domains or the scheduler scores raw states without closure.

- [ ] **Step 3: Implement exact schedule types and exhaustive closure enumeration**

Replace the generic result with:

```python
@dataclass(frozen=True, slots=True)
class ExactScheduleCertificate:
    complete: bool
    raw_states_enumerated: int
    unique_closed_states: int
    best_bound: tuple[int, float, float]
    lexicographic_gap: tuple[int, float, float]
    gap_status: str  # exact | bounded | unknown
    cpu_nontrivial: bool
    schedule_nontrivial: bool
    semantic_fingerprint: str


@dataclass(frozen=True, slots=True)
class ExactScheduleResult:
    plan: FrozenSlotPlan
    objective: tuple[int, float, float]
    certificate: ExactScheduleCertificate
```

Enumerate every legal unscheduled/position assignment for the small instance, including solo and two-user blocks. Normalize and rate each raw state with `ChannelBasis.PLANNING`, apply `close_assignment` with the configured policy, deduplicate by the closed stable key plus semantic fingerprint, and optimize the same quantized global objective as search. `complete=True` is legal only after the raw state iterator is exhausted. An exhaustive result has `best_bound == objective`, `gap_status="exact"`, and finite zero components in the three-layer gap. A timeout or state cap returns `complete=False`; use `gap_status="bounded"` only with a finite certified bound, otherwise `unknown`, and represent unknown components as `None` rather than NaN. It must not be labelled Exact in reports.

Compute `cpu_nontrivial` by proving that individually feasible tasks can exceed at least one UAV CPU budget under a legal radio state. Compute `schedule_nontrivial` by proving at least two legal closed radio states exist and the optimum is not obtained merely by serving every task. Serialize the result under an `exact_schedule` schema; never emit or parse it as `ExactAdmissionResult`.

- [ ] **Step 4: Run GREEN**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_exact_benchmark.py tests/test_schedule_closure.py tests/test_exact_admission.py -q
```

Expected: all selected tests pass and incomplete enumerations are visibly non-Exact.

- [ ] **Step 5: Commit the explicit files**

```bash
git add -- uav_mec/exact.py uav_mec/closure.py uav_mec/run_exact_benchmark.py tests/test_exact_benchmark.py tests/test_schedule_closure.py tests/test_exact_admission.py
git diff --cached --check
git commit -m "feat: certify closure-complete exact schedules"
```

---

### Task 6: Implement and audit the Guo-JAGP adapted baseline

**Files:**
- Create: `uav_mec/baselines_adapted.py`
- Modify: `uav_mec/baselines_v2.py`
- Modify: `uav_mec/experiment.py`
- Create: `tests/test_baselines_adapted.py`
- Modify: `tests/test_thesis_suites.py`
- Modify: `docs/literature_baseline_audit_2024_2026.md`
- Create: `docs/audits/2026-07-19-adapted-baseline-mapping.md`

- [ ] **Step 1: Write the RED hand-recursion and integration tests**

Create `tests/test_baselines_adapted.py`:

```python
import numpy as np
import pytest

from uav_mec.baselines_adapted import guo_jagp_adapted, guo_required_powers
from uav_mec.domain import ChannelBasis, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ChannelConfig, NOMAChannel


def test_guo_power_recursion_matches_hand_calculation() -> None:
    powers = guo_required_powers(
        channel_power_gains=(4.0, 1.0),
        qos_spectral_efficiency=(1.0, 1.0),
        noise_w=1.0,
    )
    assert powers == pytest.approx((0.5, 1.0))


def test_guo_adaptation_is_deterministic_and_returns_a_raw_schedule() -> None:
    tasks = tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=600_000.0,
            comp_cycles=1_200_000_000.0,
            deadline_s=0.8,
            local_cpu_hz=400_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in range(4)
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[9e-8, 2e-8], [6e-8, 4e-8], [3e-8, 8e-8], [2e-8, 7e-8]]),
        2,
    )
    cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.1,
        cpu_per_uav_hz=4e9,
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)
    first = guo_jagp_adapted(snapshot, channel, cfg, channel_basis=ChannelBasis.PLANNING, qos_spectral_efficiency=0.5)
    second = guo_jagp_adapted(snapshot, channel, cfg, channel_basis=ChannelBasis.PLANNING, qos_spectral_efficiency=0.5)
    assert first == second
    assert len(first.entries) <= 2 * snapshot.num_uavs * cfg.n_subcarriers
    assert len(first.task_ids) == len(set(first.task_ids))
```

The first result is hand-derived: both targets have `beta = 2^1 - 1 = 1`; the weak layer requires `p_weak = 1 * 1 / 1 = 1`; the strong layer then requires `p_strong = 1 * (1 + 1 * 1) / 4 = 0.5`. Inputs and outputs are ordered strongest to weakest.

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_baselines_adapted.py -q
```

Expected: import fails because `uav_mec.baselines_adapted` does not exist.

- [ ] **Step 3: Implement the verified recursion**

Use this complete recursion after validating equal non-empty lengths, positive finite gains, non-negative finite rates, and positive finite noise:

```python
def guo_required_powers(
    channel_power_gains: Sequence[float],
    qos_spectral_efficiency: Sequence[float],
    noise_w: float,
) -> tuple:
    gains = tuple(float(value) for value in channel_power_gains)
    targets = tuple(float(value) for value in qos_spectral_efficiency)
    if not gains or len(gains) != len(targets):
        raise ValueError("gains and targets must have the same positive length")
    if any(not math.isfinite(value) or value <= 0.0 for value in gains):
        raise ValueError("channel gains must be finite and positive")
    if any(not math.isfinite(value) or value < 0.0 for value in targets):
        raise ValueError("spectral efficiencies must be finite and non-negative")
    if not math.isfinite(noise_w) or noise_w <= 0.0:
        raise ValueError("noise_w must be finite and positive")
    powers = [0.0] * len(gains)
    accumulated_received_w = 0.0
    for index in range(len(gains) - 1, -1, -1):
        beta = math.exp2(targets[index]) - 1.0
        powers[index] = beta * (noise_w + accumulated_received_w) / gains[index]
        accumulated_received_w = math.fsum((accumulated_received_w, gains[index] * powers[index]))
    return tuple(powers)
```

- [ ] **Step 4: Implement the adaptation and mapping audit**

Freeze the adapted Guo potential as total QoS-required user transmit power, not any thesis-v2 completion/CPU/exposure objective:

```text
P_Guo(schedule) = sum over groups g of sum over users n in g of p_n(schedule)
```

Within each group, order users by `(-selected_gain, task_id)` and obtain `p_n` from `guo_required_powers`. The selected gain is the task-to-serving-UAV power gain on the requested `ChannelBasis`. If any required power exceeds `cfg.transmit_power_w`, that group and schedule have infinite potential. As in the source paper's orthogonal-group/inter-cell-coordination model, the adapted potential includes receiver noise and intra-group recursion but no thesis cross-UAV exposure, MEC CPU, deadline completion, or closure term. The common evaluator adds those thesis-v2 semantics only after this function returns its raw schedule.

Implement these helpers with the stated contracts:

| Helper | Contract |
|---|---|
| `_guo_group_power(snapshot, entries, channel, cfg, basis, qos)` | Sort one non-empty group, run the recursion, enforce the user power ceiling, and return `(total_power_w, powers_by_task_id)` |
| `guo_total_required_power(snapshot, schedule, channel, cfg, basis, qos)` | Validate at most two entries per block and return the stable `math.fsum` of group powers |
| `_place_guo_task(schedule, task, uav, subcarrier, snapshot, basis)` | Remove the task's current entry, reject a full target block, insert it, normalize roles on the selected basis, and return the candidate |
| `_guo_selected_tasks(snapshot, channel, cfg, basis, qos)` | Select at most `2 * num_uavs * n_subcarriers` offload-required tasks by `(minimum feasible solo power, task_id)` so unscheduling cannot trivially reduce the potential |

Use this deterministic best-response control flow:

```text
selected = _guo_selected_tasks(
    snapshot, channel, cfg, channel_basis, qos_spectral_efficiency
)
state = Schedule()
for task in selected:
    placements = []
    for serving_uav in range(snapshot.num_uavs):
        for subcarrier in range(cfg.n_subcarriers):
            candidate = _place_guo_task(
                state, task, serving_uav, subcarrier, snapshot, channel_basis
            )
            if candidate is None:
                continue
            potential = guo_total_required_power(
                snapshot,
                candidate,
                channel,
                cfg,
                channel_basis,
                qos_spectral_efficiency,
            )
            placements.append((potential, serving_uav, subcarrier, candidate))
    if not placements:
        raise ValueError("selected Guo task has no radio position")
    state = min(placements, key=lambda row: (row[0], row[1], row[2], task.task_id))[3]

visited = {state.stable_key()}
while True:
    accepted_in_sweep = False
    for task in selected:
        current_potential = guo_total_required_power(
            snapshot,
            state,
            channel,
            cfg,
            channel_basis,
            qos_spectral_efficiency,
        )
        responses = []
        for serving_uav in range(snapshot.num_uavs):
            for subcarrier in range(cfg.n_subcarriers):
                candidate = _place_guo_task(
                    state, task, serving_uav, subcarrier, snapshot, channel_basis
                )
                if candidate is None:
                    continue
                candidate_potential = guo_total_required_power(
                    snapshot,
                    candidate,
                    channel,
                    cfg,
                    channel_basis,
                    qos_spectral_efficiency,
                )
                responses.append(
                    (candidate_potential, serving_uav, subcarrier, candidate)
                )
        best = min(responses, key=lambda row: (row[0], row[1], row[2], task.task_id))
        if best[0] < current_potential:
            state = best[3]
            if state.stable_key() in visited:
                raise RuntimeError("strict Guo potential decrease repeated a state")
            visited.add(state.stable_key())
            accepted_in_sweep = True
    if not accepted_in_sweep:
        return state
```

This is the source paper's dynamic single-user best-response structure adapted to a two-user block cap. A unilateral move is accepted only on strict decrease of the Guo total-required-power exact potential; deterministic placement ties do not authorize an equal-potential move. The common experiment path subsequently applies canonical planning physics, configured CPU policy, closure, and frozen realized replay. Do not call the adaptation a reproduction.

The mapping audit must include one explicit row for each of: decision variables, objective/potential, association, grouping, rate/SIC, power, channel, CPU, task/deadline, mobility, and stopping rule. Each row records original-paper semantics, available source/equation, adaptation, shared evaluator boundary, and residual comparability limitation. Pin Guo et al., ICC 2019, DOI `10.1109/ICC.2019.8761779`; record Eq. 5 for the recursion, Eqs. 9--10 for `C_n = p_n + E_n` and group-power contribution, Eq. 11 for exact-potential differences, and Algorithm 1 for sequential best response. Update the closest-work audit to distinguish those verified definitions, this implementation's two-user/capacity adaptation choices, and unavailable author-code behavior.

- [ ] **Step 5: Run GREEN**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_baselines_adapted.py tests/test_thesis_suites.py -q
```

Expected: both tests pass and the hand recursion returns `(0.5, 1.0)`.

- [ ] **Step 6: Commit the explicit files**

```bash
git add -- uav_mec/baselines_adapted.py uav_mec/baselines_v2.py uav_mec/experiment.py tests/test_baselines_adapted.py tests/test_thesis_suites.py docs/literature_baseline_audit_2024_2026.md docs/audits/2026-07-19-adapted-baseline-mapping.md
git diff --cached --check
git commit -m "feat: add audited Guo JAGP adaptation"
```

---

### Task 7: Register immutable `MethodSpec` factors and the joint 2-by-2 design

**Files:**
- Modify: `uav_mec/experiment.py`
- Modify: `uav_mec/run_thesis_suite.py`
- Modify: `uav_mec/experiment_registry.py`
- Modify: `tests/test_thesis_suites.py`
- Modify: `tests/test_experiment_registry.py`
- Modify: `tests/test_mechanism_controls.py`
- Modify: `configs/thesis/experiment_registry_v2.toml`
- Modify: `configs/thesis/confirmatory_main_v2.toml`
- Modify: `configs/thesis/confirmatory_resource_fixed_v2.toml`
- Modify: `configs/thesis/confirmatory_joint_v2.toml`
- Modify: `configs/thesis/pilot_main_v2.toml`
- Modify: `configs/thesis/pilot_resource_fixed_v2.toml`
- Modify: `configs/thesis/pilot_joint_v2.toml`
- Modify: `configs/thesis/smoke.toml`

- [ ] **Step 1: Write the RED factor-isolation tests**

Add to `tests/test_mechanism_controls.py`:

```python
import dataclasses

from uav_mec.experiment import MethodSpec, thesis_method_specs
from uav_mec.experiment_registry import thesis_contrasts


def _factor_differences(left: MethodSpec, right: MethodSpec) -> set[str]:
    ignored = {"method_id", "display_name"}
    return {
        field.name
        for field in dataclasses.fields(MethodSpec)
        if field.name not in ignored and getattr(left, field.name) != getattr(right, field.name)
    }


def test_h1_and_h2_structural_ablations_change_only_registered_factors() -> None:
    specs = thesis_method_specs()
    assert _factor_differences(specs["full"], specs["intra_global"]) == {"neighborhood_scope"}
    assert _factor_differences(specs["full"], specs["cross_bilateral"]) == {"acceptance_rule"}
    assert specs["full"].initializer_ids == specs["intra_global"].initializer_ids
    assert specs["full"].initializer_ids == specs["cross_bilateral"].initializer_ids


def test_joint_methods_form_an_exact_two_by_two_factorial() -> None:
    specs = thesis_method_specs()
    keys = ("iea_dd", "iea_edf", "exchange_dd", "exchange_edf")
    joint = {key: specs[key] for key in keys}
    assert {spec.wireless_kernel for spec in joint.values()} == {"iea_cap", "exchange_stable"}
    assert {spec.allocation_policy.value for spec in joint.values()} == {"deadline", "edf_min_cpu"}
    signatures = {(spec.wireless_kernel, spec.allocation_policy.value) for spec in joint.values()}
    assert len(signatures) == 4
    assert len({spec.initializer_ids for spec in joint.values()}) == 1
    h4 = thesis_contrasts()["H4"]
    assert (h4.treatment_id, h4.control_id) == ("iea_dd", "exchange_edf")


def test_fixed_radio_resource_methods_name_their_schedule_source() -> None:
    specs = thesis_method_specs()
    assert specs["fixed_radio_dd"].fixed_radio_source == "dd_generated"
    assert specs["fixed_radio_edf"].fixed_radio_source == "dd_generated"
    assert specs["fixed_radio_edf"].display_name == "Deadline-First-MinCPU (static share)"
    assert "EDF" not in specs["fixed_radio_edf"].display_name
    assert specs["fixed_radio_dd_radio_neutral"].fixed_radio_source == "radio_neutral"
    assert specs["fixed_radio_edf_radio_neutral"].fixed_radio_source == "radio_neutral"
    assert specs["fixed_radio_dd_deadline_generated"].fixed_radio_source == "deadline_first_generated"
    assert specs["fixed_radio_edf_deadline_generated"].fixed_radio_source == "deadline_first_generated"
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
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_mechanism_controls.py tests/test_experiment_registry.py -q
```

Expected: methods are string branches without immutable factor declarations, or paired methods differ in hidden settings.

- [ ] **Step 3: Add and use `MethodSpec`**

Use this field table; each field is required and validated, with no implicit method-name behavior:

| Field | Type | Meaning |
|---|---|---|
| `method_id` | `str` | Stable machine identifier |
| `display_name` | `str` | Report label |
| `wireless_kernel` | `str` | Raw schedule generator/search kernel |
| `initializer_ids` | tuple of `str` | Ordered shared multistart initializers |
| `neighborhood_scope` | `str` | Intra, bilateral cross-UAV, or global cross-UAV |
| `acceptance_rule` | `str` | Bilateral or global lexicographic acceptance |
| `traversal_rule` | `str` | First or best improvement with frozen candidate order |
| `allocation_policy` | `AllocationPolicy` | Policy used by closure or frozen-radio admission |
| `fixed_radio_source` | optional `str` | Named source for conditional resource methods |
| `adaptation_id` | optional `str` | Literature mapping identifier |
| `physics_model_id` | `str` | Canonical radio/SIC evaluator identifier |
| `latency_model_id` | `str` | Canonical latency API identifier |
| `closure_mode` | `str` | Policy closure or frozen-rate conditional admission |
| `realized_replay` | `bool` | Whether the method receives frozen realized audit |

Define `@dataclass(frozen=True, slots=True) class MethodSpec` in `experiment.py`. `thesis_method_specs()` returns one validated dictionary used by the CLI, suite runner, registry, manifests, and tests. Delete method-name substring dispatch. Store the complete serialized spec and its SHA-256 hash in every method result.

In `experiment_registry.py`, add a frozen `ContrastSpec` with required fields `hypothesis_id`, `treatment_id`, `control_id`, and `stratum`. `thesis_contrasts()` returns the validated H1--H4 mapping consumed by tests and registry serialization; H3 and H4 use the exact IDs asserted above.

Register Local-only, Fair-OFDMA, Greedy, IA-Greedy, per-UAV ILP, Exchange-Stable, Intra-Global, Cross-Bilateral, single-start first-improvement, full IEA-CAP, Guo-JAGP adapted, the six fixed-radio resource methods, and the four joint cells. Register Exact Admission only in the Resource suite and Exact Schedule only in the Exact suite. H1 pairs differ only in `neighborhood_scope`; H2 pairs differ only in `acceptance_rule`. Every end-to-end NOMA method names `canonical_noma_v2`, `canonical_task_latency_v2`, policy closure, and frozen realized replay.

The joint IDs are exactly `iea_dd`, `iea_edf`, `exchange_dd`, and `exchange_edf`; their display names spell out that the compatibility suffix `edf` means non-preemptive Deadline-First-MinCPU static-share admission. The four cells differ only in `wireless_kernel` and `allocation_policy`. H4 is treatment `iea_dd` minus control `exchange_edf`.

The H3 primary pair is exactly `fixed_radio_dd` versus `fixed_radio_edf` on one `dd_generated` closure-before `RawRateSnapshot`. The two sensitivity strata are `radio_neutral` and `deadline_first_generated`, each with a DD and Deadline-First-MinCPU response. These six fixed-radio methods consume a frozen raw/rate snapshot and must not invoke policy closure or wireless re-evaluation while comparing admission policies.

- [ ] **Step 4: Update only the explicit registries and configurations**

Write the exact IDs and source labels above into the eight listed TOML files. The registry declares each primary contrast, treatment/control order, fixed-radio source, and factor hash. Keep `dd_generated` as the H3 primary stratum and `radio_neutral` plus `deadline_first_generated` as sensitivity strata. Do not execute any configuration in this task.

- [ ] **Step 5: Run GREEN and parse every updated configuration**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_thesis_suites.py tests/test_experiment_registry.py tests/test_mechanism_controls.py -q
/home/link/miniconda/envs/llm-ops-torch26/bin/python - <<'PY'
import tomllib
from pathlib import Path

paths = (
    Path("configs/thesis/experiment_registry_v2.toml"),
    Path("configs/thesis/confirmatory_main_v2.toml"),
    Path("configs/thesis/confirmatory_resource_fixed_v2.toml"),
    Path("configs/thesis/confirmatory_joint_v2.toml"),
    Path("configs/thesis/pilot_main_v2.toml"),
    Path("configs/thesis/pilot_resource_fixed_v2.toml"),
    Path("configs/thesis/pilot_joint_v2.toml"),
    Path("configs/thesis/smoke.toml"),
)
for path in paths:
    with path.open("rb") as handle:
        tomllib.load(handle)
print(f"parsed_configs={len(paths)}")
PY
```

Expected: tests pass and `parsed_configs=8` prints. No experiment output is created.

- [ ] **Step 6: Commit the explicit files**

```bash
git add -- uav_mec/experiment.py uav_mec/run_thesis_suite.py uav_mec/experiment_registry.py tests/test_thesis_suites.py tests/test_experiment_registry.py tests/test_mechanism_controls.py configs/thesis/experiment_registry_v2.toml configs/thesis/confirmatory_main_v2.toml configs/thesis/confirmatory_resource_fixed_v2.toml configs/thesis/confirmatory_joint_v2.toml configs/thesis/pilot_main_v2.toml configs/thesis/pilot_resource_fixed_v2.toml configs/thesis/pilot_joint_v2.toml configs/thesis/smoke.toml
git diff --cached --check
git commit -m "refactor: register auditable method factors"
```

---

### Task 8: Gate algorithm claims without running experiments

**Files:**
- Modify: `uav_mec/runtime_audit.py`
- Modify: `tests/test_runtime_audit.py`
- Modify: `tests/test_project_metadata.py`
- Modify: `README.md`

- [ ] **Step 1: Write the RED claim-gate test**

Add to `tests/test_runtime_audit.py`:

```python
from uav_mec.runtime_audit import AlgorithmAuditRecord, validate_algorithm_claim_gate


def test_claim_gate_rejects_caps_incomplete_exact_and_unverified_incremental_counts() -> None:
    invalid = AlgorithmAuditRecord(
        search_stop_reasons=("no_improving_move", "iteration_cap"),
        local_optimum_certified=(True, False),
        incremental_actions_checked=9_999,
        externality_conserved=True,
        exact_schedule_complete=False,
        exact_admission_type="ExactAdmissionResult",
        exact_schedule_type="ExactScheduleResult",
        method_spec_hashes=("a" * 64,),
        adapted_baseline_mapping_hash="b" * 64,
    )
    errors = validate_algorithm_claim_gate(invalid)
    assert "iteration_cap" in errors
    assert "incremental_actions_checked" in errors
    assert "exact_schedule_complete" in errors


def test_claim_gate_accepts_complete_static_evidence() -> None:
    valid = AlgorithmAuditRecord(
        search_stop_reasons=("no_improving_move",),
        local_optimum_certified=(True,),
        incremental_actions_checked=10_000,
        externality_conserved=True,
        exact_schedule_complete=True,
        exact_admission_type="ExactAdmissionResult",
        exact_schedule_type="ExactScheduleResult",
        method_spec_hashes=("a" * 64,),
        adapted_baseline_mapping_hash="b" * 64,
    )
    assert validate_algorithm_claim_gate(valid) == ()
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_runtime_audit.py::test_claim_gate_rejects_caps_incomplete_exact_and_unverified_incremental_counts tests/test_runtime_audit.py::test_claim_gate_accepts_complete_static_evidence -q
```

Expected: the static algorithm claim record and validator do not yet cover all required fields.

- [ ] **Step 3: Implement the static gate and documentation boundary**

Add the frozen `AlgorithmAuditRecord` with exactly the fields constructed above. `validate_algorithm_claim_gate` returns stable error codes and rejects mismatched stop-reason/certification pairs, any iteration cap used as an optimum, fewer than 10,000 incremental actions, failed exposure conservation, incomplete Exact Schedule, conflated exact result type names, an empty MethodSpec hash set, or any mapping/spec digest that is not exactly 64 lowercase hexadecimal characters.

In `README.md`, document that these are implementation-correctness gates only. They do not establish an empirical effect, exact-gap distribution, runtime claim, thesis conclusion, or submission result. Pilot and formal execution belong to later approved plans.

- [ ] **Step 4: Run GREEN and the full Plan 2 gate**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_externality_analysis.py tests/test_schedule_closure.py tests/test_neighborhoods.py tests/test_search.py tests/test_incremental_evaluation.py tests/test_resource_requirements.py tests/test_resource_allocator.py tests/test_resource_policies.py tests/test_exact_admission.py tests/test_exact_benchmark.py tests/test_baselines_adapted.py tests/test_thesis_suites.py tests/test_experiment_registry.py tests/test_mechanism_controls.py tests/test_runtime_audit.py tests/test_project_metadata.py -q
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest -q
rg -n "from \.pairing|id\(allocation_score_with_policy\)" uav_mec/closure.py uav_mec/evaluation.py uav_mec/incremental.py uav_mec/exact.py uav_mec/baselines_v2.py uav_mec/baselines_adapted.py uav_mec/experiment.py
git diff --check
```

Expected: both pytest commands pass, `rg` prints nothing, and `git diff --check` exits `0`. Do not run a configuration to obtain performance numbers.

- [ ] **Step 5: Commit the explicit files**

```bash
git add -- uav_mec/runtime_audit.py tests/test_runtime_audit.py tests/test_project_metadata.py README.md
git diff --cached --check
git commit -m "chore: gate algorithm correctness claims"
```

---

## Plan 2 completion gate

Plan 2 is complete only when all eight task commits exist, the full pytest suite is green, the 10,000-action test completed, the Guo hand calculation is frozen at `(0.5, 1.0)`, Exact Admission and Exact Schedule serialize as distinct types, all search caps remain uncertified, and the worktree is clean. Record commit SHAs and test commands in the handoff. Do not attach performance or hypothesis language to this implementation handoff.
