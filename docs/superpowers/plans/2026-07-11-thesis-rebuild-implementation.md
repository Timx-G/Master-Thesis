# Thesis Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a constraint-correct, reproducible multi-UAV NOMA-MEC simulation pipeline containing an interference-aware cross-UAV scheduler and deadline-driven offloading/CPU allocator, then regenerate the evidence used by the professional master's thesis.

**Architecture:** Preserve the current runners and results as legacy evidence while adding a single typed scientific pipeline. Immutable traces feed an explicit schedule, the schedule feeds physical evaluation and deadline-driven allocation, and canonical artifacts feed statistics, figures, and LaTeX macros. Every production behavior is introduced through a failing test and every milestone is committed independently.

**Tech Stack:** Python 3.11, NumPy, SciPy/HiGHS MILP, pytest, Matplotlib, standard-library `dataclasses`, `tomllib`, `argparse`, `csv`, `json`, `hashlib`, and Tectonic LaTeX.

---

## Scope And File Map

The implementation uses new modules for the trusted pipeline so legacy scripts remain inspectable during migration.

| Path | Responsibility |
|---|---|
| `uav_mec/domain.py` | Immutable task, trace, schedule, allocation, rate, and outcome records |
| `uav_mec/model.py` | Correct noise and Urban air-to-ground channel formulas |
| `uav_mec/resource_allocation.py` | Minimum offload/CPU derivation and maximum-cardinality allocation |
| `uav_mec/schedule.py` | Structural schedule validation, serving-UAV derivation, and stable encoding |
| `uav_mec/evaluation.py` | Schedule SINR/rate calculation, SIC validation, and slot evaluation |
| `uav_mec/trace.py` | Deterministic exogenous mobility and task traces |
| `uav_mec/simulation.py` | Unique-task episode execution shared by every method |
| `uav_mec/scheduler.py` | ILP initialization, feasibility repair, neighborhoods, and local search |
| `uav_mec/baselines_v2.py` | Controlled baselines using the same evaluator and allocator |
| `uav_mec/exact.py` | Complete small-instance schedule enumeration |
| `uav_mec/statistics.py` | Paired bootstrap, Wilcoxon, Holm correction, and effect sizes |
| `uav_mec/artifacts.py` | Atomic canonical output, manifests, and checksums |
| `uav_mec/experiment.py` | Config-driven CLI and experiment orchestration |
| `uav_mec/run_exact_benchmark.py` | Complete-enumeration optimality-gap evidence on small instances |
| `uav_mec/analyze_results.py` | Evidence validation, secondary metrics, paired tests, and thesis gates |
| `uav_mec/publish_canonical.py` | Whitelist and checksum lightweight versioned thesis evidence |
| `uav_mec/generate_figures_v2.py` | Canonical-only figure and LaTeX macro generation |
| `configs/thesis/*.toml` | Frozen experiment matrices |
| `requirements-lock.txt` | Exact clean-environment Python dependency set generated after verification |
| `tests/` | Unit, regression, integration, and end-to-end tests |
| `artifacts/canonical/` | Versioned lightweight evidence used by the paper |
| `paper/generated_results.tex` | Generated numeric macros; never hand-edited |
| `thesis/` | Professional-master thesis source, chapter files, bibliography, and build output |
| `defense/` | Defense outline, innovation comparison, and high-risk question bank |

Legacy `run_*`, DRL, and local `results/` files are not deleted. New code must not import legacy experiment runners.

---

### Task 1: Establish The Test And Metadata Baseline

**Files:**
- Create: `tests/test_project_metadata.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

- [ ] **Step 1: Write the failing metadata test**

```python
from pathlib import Path
import tomllib


def test_project_metadata_matches_current_research_topic():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["requires-python"] == ">=3.11"
    assert project["description"] == (
        "Interference-aware pairing and deadline-driven resource allocation "
        "for full-reuse multi-UAV NOMA-MEC"
    )
    assert "pytest>=8.0,<9" in project["optional-dependencies"]["dev"]
```

- [ ] **Step 2: Run the test and verify it fails for the stale description and Python range**

Run: `python -m pytest tests/test_project_metadata.py -v`

Expected: FAIL because the current description still describes trajectory design and constrained DRL.

- [ ] **Step 3: Update package metadata and test configuration**

Apply this exact `pyproject.toml` metadata/configuration change while retaining the existing runtime dependency list:

```toml
[project]
name = "uav-mec-noma-offloading"
version = "0.2.0"
description = "Interference-aware pairing and deadline-driven resource allocation for full-reuse multi-UAV NOMA-MEC"
requires-python = ">=3.11"
license = {text = "MIT"}

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<9",
    "black>=24.0,<25",
    "isort>=5.13,<6",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

Pin `requirements.txt` to the verified scientific environment ranges:

```text
numpy>=1.26,<2
scipy>=1.14,<2
torch>=2.1,<2.2
gymnasium>=1.0,<2
matplotlib>=3.8,<4
pandas>=2.0,<3
```

- [ ] **Step 4: Run the metadata test**

Run: `python -m pytest tests/test_project_metadata.py -v`

Expected: PASS, 1 test passed.

- [ ] **Step 5: Record the clean baseline**

Run: `python -m pytest -q`

Expected: PASS with no warnings from project tests.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt tests/test_project_metadata.py
git commit -m "test: establish thesis pipeline baseline"
```

---

### Task 2: Add Immutable Domain Records

**Files:**
- Create: `uav_mec/domain.py`
- Create: `tests/test_domain.py`

- [ ] **Step 1: Write failing tests for task identity and stable schedules**

```python
import numpy as np

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task


def test_task_local_delay_and_deadline_identity():
    task = Task(
        task_id=1203,
        user_id=3,
        arrival_slot=12,
        data_bits=800_000.0,
        comp_cycles=1_200_000_000.0,
        deadline_s=0.8,
        local_cpu_hz=1_000_000_000.0,
    )
    assert task.local_delay_s == 1.2
    assert task.absolute_deadline_s == 12.8
    assert task.requires_offload


def test_schedule_key_is_independent_of_input_order():
    a = ScheduleEntry(7, 7, 1, 0, "weak")
    b = ScheduleEntry(2, 2, 0, 1, "solo")
    assert Schedule((a, b)).stable_key() == Schedule((b, a)).stable_key()


def test_slot_snapshot_rejects_wrong_channel_shape():
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    try:
        SlotSnapshot(0, (task,), np.ones((2, 2)), 2)
    except ValueError as exc:
        assert "channel_gains" in str(exc)
    else:
        raise AssertionError("SlotSnapshot accepted a channel matrix with the wrong user count")


def test_slot_snapshot_owns_a_read_only_channel_copy():
    source = np.ones((1, 1))
    task = Task(1, 0, 0, 1.0, 1.0, 0.8, 1.0)
    snapshot = SlotSnapshot(0, (task,), source, 1)
    source[0, 0] = 9.0
    assert snapshot.channel_gains[0, 0] == 1.0
    try:
        snapshot.channel_gains[0, 0] = 2.0
    except ValueError:
        return
    raise AssertionError("SlotSnapshot channel matrix remained mutable")
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `python -m pytest tests/test_domain.py -v`

Expected: ERROR with `ModuleNotFoundError: No module named 'uav_mec.domain'`.

- [ ] **Step 3: Implement the domain records**

Create `uav_mec/domain.py` with these complete public records:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

NomaRole = Literal["strong", "weak", "solo"]


@dataclass(frozen=True, slots=True)
class Task:
    task_id: int
    user_id: int
    arrival_slot: int
    data_bits: float
    comp_cycles: float
    deadline_s: float
    local_cpu_hz: float

    def __post_init__(self) -> None:
        if min(self.data_bits, self.comp_cycles, self.deadline_s, self.local_cpu_hz) <= 0:
            raise ValueError("task quantities must be positive")

    @property
    def local_delay_s(self) -> float:
        return self.comp_cycles / self.local_cpu_hz

    @property
    def absolute_deadline_s(self) -> float:
        return self.arrival_slot + self.deadline_s

    @property
    def requires_offload(self) -> bool:
        return self.local_delay_s > self.deadline_s


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    task_id: int
    user_id: int
    serving_uav: int
    subcarrier: int
    noma_role: NomaRole


@dataclass(frozen=True, slots=True)
class Schedule:
    entries: tuple[ScheduleEntry, ...] = ()

    def stable_key(self) -> tuple[tuple[int, int, int, int, str], ...]:
        return tuple(sorted(
            (
                entry.serving_uav,
                entry.subcarrier,
                entry.user_id,
                entry.task_id,
                entry.noma_role,
            )
            for entry in self.entries
        ))


@dataclass(frozen=True, slots=True)
class SlotSnapshot:
    slot_index: int
    tasks: tuple[Task, ...]
    channel_gains: np.ndarray
    num_uavs: int

    def __post_init__(self) -> None:
        expected = (len(self.tasks), self.num_uavs)
        gains = np.array(self.channel_gains, dtype=float, copy=True)
        if gains.shape != expected:
            raise ValueError(f"channel_gains must have shape {expected}")
        if not np.isfinite(gains).all() or (gains < 0).any():
            raise ValueError("channel_gains must be finite and non-negative")
        gains.setflags(write=False)
        object.__setattr__(self, "channel_gains", gains)


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    task_id: int
    alpha: float
    cpu_hz: float
    completed: bool
    delay_s: float | None
    slack_s: float
    reason: str


@dataclass(frozen=True, slots=True)
class RateRecord:
    task_id: int
    rate_bps: float
    sinr: float
    sic_feasible: bool
    cross_interference_w: float
    serving_uav: int
    subcarrier: int


@dataclass(frozen=True, slots=True)
class SlotOutcome:
    slot_index: int
    arrived_task_ids: tuple[int, ...]
    completed_task_ids: tuple[int, ...]
    missed_task_ids: tuple[int, ...]
    decisions: tuple[AllocationDecision, ...]
    rates: tuple[RateRecord, ...]
    total_slack_s: float
    total_cross_interference_w: float
    runtime_s: float = 0.0

    @property
    def completed_count(self) -> int:
        return len(self.completed_task_ids)
```

- [ ] **Step 4: Run the domain tests**

Run: `python -m pytest tests/test_domain.py -v`

Expected: PASS, 4 tests passed.

- [ ] **Step 5: Commit**

```bash
git add uav_mec/domain.py tests/test_domain.py
git commit -m "feat: add immutable thesis domain records"
```

---

### Task 3: Correct Noise And Urban Path-Loss Semantics

**Files:**
- Modify: `uav_mec/model.py`
- Create: `tests/test_physics.py`

- [ ] **Step 1: Write failing numerical tests**

```python
import math

import pytest

from uav_mec.model import ChannelConfig, NOMAChannel, noise_power_dbm, noise_power_w


def test_noise_for_one_subcarrier_includes_nf_once():
    bandwidth_hz = 5e6 / 3
    assert noise_power_dbm(bandwidth_hz, -174.0, 7.0) == pytest.approx(-104.7815, abs=1e-3)
    assert 10 * math.log10(noise_power_w(bandwidth_hz, -174.0, 7.0)) + 30 == pytest.approx(
        -104.7815,
        abs=1e-3,
    )


def test_channel_uses_urban_parameters_and_db_averaged_loss():
    cfg = ChannelConfig()
    assert (cfg.environment, cfg.a_los, cfg.b_los, cfg.eta_los, cfg.eta_nlos) == (
        "urban",
        9.61,
        0.16,
        1.0,
        20.0,
    )
    assert not hasattr(cfg, "noise_power_dbm")
    channel = NOMAChannel(cfg, 100.0)
    p_los = channel._los_probability(400.0)
    distance = math.hypot(400.0, 100.0)
    fspl_db = 20 * math.log10(4 * math.pi * cfg.fc * distance / 3e8)
    expected_gain = 10 ** (-(fspl_db + p_los * 1.0 + (1 - p_los) * 20.0) / 10)
    assert channel.path_loss(400.0) == pytest.approx(expected_gain, rel=1e-12)
```

- [ ] **Step 2: Run the tests and verify they fail for missing helpers and current linear-loss averaging**

Run: `python -m pytest tests/test_physics.py -v`

Expected: FAIL during import of `noise_power_dbm`, then FAIL on path-loss value after helpers are introduced.

- [ ] **Step 3: Implement one-source noise conversion and Urban path loss**

Add these functions and update `ChannelConfig`/`NOMAChannel`:

```python
def noise_power_dbm(
    bandwidth_hz: float,
    noise_psd_dbm_hz: float = -174.0,
    noise_figure_db: float = 7.0,
) -> float:
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth_hz must be positive")
    return noise_psd_dbm_hz + noise_figure_db + 10.0 * math.log10(bandwidth_hz)


def noise_power_w(
    bandwidth_hz: float,
    noise_psd_dbm_hz: float = -174.0,
    noise_figure_db: float = 7.0,
) -> float:
    dbm = noise_power_dbm(bandwidth_hz, noise_psd_dbm_hz, noise_figure_db)
    return 10.0 ** ((dbm - 30.0) / 10.0)
```

Delete the ambiguous `noise_power_dbm` field and its per-subcarrier comments. Use this `ChannelConfig` field set, retaining unrelated frequency/power fields already present and changing the environment comment from Suburban to Urban:

```python
environment: str = "urban"
noise_psd_dbm_hz: float = -174.0
noise_figure_db: float = 7.0
a_los: float = 9.61
b_los: float = 0.16
eta_los: float = 1.0
eta_nlos: float = 20.0
```

Replace noise initialization and path-loss averaging with:

```python
self.noise_linear = noise_power_w(
    cfg.bandwidth,
    cfg.noise_psd_dbm_hz,
    cfg.noise_figure_db,
)


def noise_for_bandwidth(self, bandwidth_hz: float) -> float:
    return noise_power_w(
        bandwidth_hz,
        self.cfg.noise_psd_dbm_hz,
        self.cfg.noise_figure_db,
    )


def path_loss(self, d_horizontal: float) -> float:
    if d_horizontal < 0:
        raise ValueError("d_horizontal must be non-negative")
    d_3d = math.hypot(d_horizontal, self.H)
    fspl_db = 20.0 * math.log10(4.0 * math.pi * self.cfg.fc * d_3d / self.c)
    p_los = self._los_probability(d_horizontal)
    mean_loss_db = fspl_db + p_los * self.cfg.eta_los + (1.0 - p_los) * self.cfg.eta_nlos
    return 10.0 ** (-mean_loss_db / 10.0)
```

New trusted-pipeline code must call `noise_for_bandwidth(B/S)` directly and must not multiply a stored noise power by `1/S`.

- [ ] **Step 4: Run the physics tests**

Run: `python -m pytest tests/test_physics.py -v`

Expected: PASS, 2 tests passed.

- [ ] **Step 5: Run the existing import smoke check**

Run: `python -c "from uav_mec.model import ScenarioConfig, NOMAChannel; c=ScenarioConfig(); print(NOMAChannel(c.channel, c.uav.height).noise_for_bandwidth(c.channel.bandwidth/c.n_subcarriers))"`

Expected: prints a finite positive value close to `3.325e-14` W.

- [ ] **Step 6: Commit**

```bash
git add uav_mec/model.py tests/test_physics.py
git commit -m "fix: use one-source noise and urban path loss"
```

---

### Task 4: Derive Deadline Resource Requirements

**Files:**
- Create: `uav_mec/resource_allocation.py`
- Create: `tests/test_resource_requirements.py`

- [ ] **Step 1: Write failing analytical tests**

```python
import pytest

from uav_mec.domain import Task
from uav_mec.resource_allocation import RequirementReason, required_resources


def make_task(comp_cycles: float) -> Task:
    return Task(1, 0, 0, 1_000_000.0, comp_cycles, 0.8, 1_000_000_000.0)


def test_local_feasible_task_requires_no_radio_or_mec():
    req = required_resources(make_task(600_000_000.0), rate_bps=1.0, result_ratio=0.1)
    assert req.feasible
    assert req.local_only
    assert req.alpha == 0.0
    assert req.cpu_hz == 0.0
    assert req.reason is RequirementReason.LOCAL


def test_heavy_task_uses_minimum_offload_and_cpu():
    task = make_task(1_200_000_000.0)
    req = required_resources(task, rate_bps=4_000_000.0, result_ratio=0.1)
    expected_alpha = 1.0 - 0.8e9 / 1.2e9
    expected_cpu = expected_alpha * 1.2e9 / (0.8 - expected_alpha * 1.1e6 / 4e6)
    assert req.alpha == pytest.approx(expected_alpha)
    assert req.cpu_hz == pytest.approx(expected_cpu)
    assert req.feasible


def test_link_is_infeasible_when_transmission_alone_exceeds_deadline():
    req = required_resources(make_task(2_000_000_000.0), rate_bps=100_000.0, result_ratio=0.1)
    assert not req.feasible
    assert req.reason is RequirementReason.LINK_INFEASIBLE


def test_negative_result_ratio_is_rejected():
    with pytest.raises(ValueError, match="result_ratio"):
        required_resources(make_task(1_200_000_000.0), rate_bps=4_000_000.0, result_ratio=-0.1)
```

- [ ] **Step 2: Run the tests and verify module import failure**

Run: `python -m pytest tests/test_resource_requirements.py -v`

Expected: ERROR with `ModuleNotFoundError: No module named 'uav_mec.resource_allocation'`.

- [ ] **Step 3: Implement the analytical requirement function**

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .domain import Task


class RequirementReason(str, Enum):
    LOCAL = "local"
    MEC = "mec"
    INVALID_RATE = "invalid_rate"
    LINK_INFEASIBLE = "link_infeasible"
    CPU_BUDGET = "cpu_budget"


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    task_id: int
    feasible: bool
    local_only: bool
    alpha: float
    cpu_hz: float
    reason: RequirementReason


def required_resources(
    task: Task,
    rate_bps: float,
    result_ratio: float,
) -> ResourceRequirement:
    if not math.isfinite(result_ratio) or result_ratio < 0:
        raise ValueError("result_ratio must be finite and non-negative")
    if task.local_delay_s <= task.deadline_s:
        return ResourceRequirement(
            task.task_id,
            True,
            True,
            0.0,
            0.0,
            RequirementReason.LOCAL,
        )
    if not math.isfinite(rate_bps) or rate_bps <= 0:
        return ResourceRequirement(
            task.task_id,
            False,
            False,
            0.0,
            math.inf,
            RequirementReason.INVALID_RATE,
        )
    alpha = 1.0 - task.deadline_s * task.local_cpu_hz / task.comp_cycles
    alpha = min(max(alpha, 0.0), 1.0)
    denominator = task.deadline_s - alpha * (1.0 + result_ratio) * task.data_bits / rate_bps
    if denominator <= 0:
        return ResourceRequirement(
            task.task_id,
            False,
            False,
            alpha,
            math.inf,
            RequirementReason.LINK_INFEASIBLE,
        )
    cpu_hz = alpha * task.comp_cycles / denominator
    return ResourceRequirement(
        task.task_id,
        True,
        False,
        alpha,
        cpu_hz,
        RequirementReason.MEC,
    )
```

- [ ] **Step 4: Run the analytical tests**

Run: `python -m pytest tests/test_resource_requirements.py -v`

Expected: PASS, 4 tests passed.

- [ ] **Step 5: Add a numerical minimality test**

Append this test to `tests/test_resource_requirements.py`:

```python
def test_analytical_requirement_is_minimal_on_dense_alpha_grid():
    import numpy as np

    task = make_task(1_700_000_000.0)
    rate = 2_500_000.0
    req = required_resources(task, rate, 0.1)
    candidates = np.linspace(req.alpha, 1.0, 20_001)
    denominators = task.deadline_s - candidates * 1.1 * task.data_bits / rate
    cpu = np.where(
        denominators > 0,
        candidates * task.comp_cycles / denominators,
        np.inf,
    )
    assert req.cpu_hz == pytest.approx(float(cpu.min()), rel=1e-12)
```

- [ ] **Step 6: Run the numerical cross-check**

Run: `python -m pytest tests/test_resource_requirements.py -v`

Expected: PASS, 5 tests passed.

- [ ] **Step 7: Commit**

```bash
git add uav_mec/resource_allocation.py tests/test_resource_requirements.py
git commit -m "feat: derive deadline resource requirements"
```

---

### Task 5: Implement Maximum-Cardinality MEC Allocation

**Files:**
- Modify: `uav_mec/resource_allocation.py`
- Create: `tests/test_resource_allocator.py`

- [ ] **Step 1: Write failing allocation tests**

```python
from itertools import combinations

from uav_mec.domain import Task
from uav_mec.resource_allocation import allocate_deadline_tasks, required_resources


def tasks() -> tuple[Task, ...]:
    return (
        Task(10, 0, 0, 1e6, 1.1e9, 0.8, 1e9),
        Task(11, 1, 0, 1e6, 1.4e9, 0.8, 1e9),
        Task(12, 2, 0, 1e6, 1.9e9, 0.8, 1e9),
    )


def test_allocator_selects_smallest_cpu_costs_with_stable_ties():
    rates = {10: 4e6, 11: 4e6, 12: 4e6}
    reqs = [required_resources(task, rates[task.task_id], 0.1) for task in tasks()]
    budget = reqs[0].cpu_hz + reqs[1].cpu_hz + 1.0
    result = allocate_deadline_tasks(tasks(), rates, budget, 0.1)
    assert result.completed_task_ids == (10, 11)
    assert result.total_cpu_hz <= budget


def test_allocator_cardinality_matches_brute_force():
    rates = {10: 3e6, 11: 2e6, 12: 5e6}
    task_list = tasks()
    requirements = {
        task.task_id: required_resources(task, rates[task.task_id], 0.1)
        for task in task_list
    }
    budget = 2.0e9
    brute_best = 0
    for size in range(len(task_list) + 1):
        for subset in combinations(task_list, size):
            if sum(requirements[t.task_id].cpu_hz for t in subset) <= budget:
                brute_best = max(brute_best, size)
    result = allocate_deadline_tasks(task_list, rates, budget, 0.1)
    assert len(result.completed_task_ids) == brute_best
```

- [ ] **Step 2: Run tests and verify the missing allocator failure**

Run: `python -m pytest tests/test_resource_allocator.py -v`

Expected: FAIL importing `allocate_deadline_tasks`.

- [ ] **Step 3: Implement the allocation result and stable prefix algorithm**

Append to `uav_mec/resource_allocation.py`:

```python
from collections.abc import Mapping, Sequence

from .domain import AllocationDecision


@dataclass(frozen=True, slots=True)
class AllocationResult:
    decisions: tuple[AllocationDecision, ...]
    completed_task_ids: tuple[int, ...]
    total_cpu_hz: float


def allocate_deadline_tasks(
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    result_ratio: float,
) -> AllocationResult:
    if cpu_budget_hz < 0 or not math.isfinite(cpu_budget_hz):
        raise ValueError("cpu_budget_hz must be finite and non-negative")
    local_decisions: list[AllocationDecision] = []
    mec_requirements: list[ResourceRequirement] = []
    rejected: dict[int, ResourceRequirement] = {}
    for task in tasks:
        req = required_resources(task, rates_bps.get(task.task_id, 0.0), result_ratio)
        if req.local_only:
            local_decisions.append(AllocationDecision(
                task.task_id,
                0.0,
                0.0,
                True,
                task.local_delay_s,
                task.deadline_s - task.local_delay_s,
                req.reason.value,
            ))
        elif req.feasible and req.cpu_hz <= cpu_budget_hz:
            mec_requirements.append(req)
        else:
            rejected[task.task_id] = req

    mec_requirements.sort(key=lambda req: (req.cpu_hz, req.task_id))
    chosen: set[int] = set()
    used = 0.0
    for req in mec_requirements:
        if used + req.cpu_hz <= cpu_budget_hz + 1e-9:
            chosen.add(req.task_id)
            used += req.cpu_hz

    decisions = list(local_decisions)
    requirement_by_id = {req.task_id: req for req in mec_requirements}
    for task in tasks:
        if task.task_id in chosen:
            req = requirement_by_id[task.task_id]
            rate = rates_bps[task.task_id]
            local_delay = (1.0 - req.alpha) * task.comp_cycles / task.local_cpu_hz
            edge_delay = req.alpha * (
                (1.0 + result_ratio) * task.data_bits / rate
                + task.comp_cycles / req.cpu_hz
            )
            delay = max(local_delay, edge_delay)
            decisions.append(AllocationDecision(
                task.task_id,
                req.alpha,
                req.cpu_hz,
                True,
                delay,
                max(task.deadline_s - delay, 0.0),
                req.reason.value,
            ))
        elif task.task_id in requirement_by_id:
            req = requirement_by_id[task.task_id]
            decisions.append(AllocationDecision(
                task.task_id,
                req.alpha,
                0.0,
                False,
                None,
                0.0,
                RequirementReason.CPU_BUDGET.value,
            ))
        elif task.task_id in rejected:
            req = rejected[task.task_id]
            reason = (
                RequirementReason.CPU_BUDGET
                if req.feasible and req.cpu_hz > cpu_budget_hz
                else req.reason
            )
            decisions.append(AllocationDecision(
                task.task_id,
                req.alpha,
                0.0,
                False,
                None,
                0.0,
                reason.value,
            ))

    unique = {decision.task_id: decision for decision in decisions}
    ordered = tuple(unique[task_id] for task_id in sorted(unique))
    completed = tuple(decision.task_id for decision in ordered if decision.completed)
    return AllocationResult(ordered, completed, used)
```

- [ ] **Step 4: Run allocator tests**

Run: `python -m pytest tests/test_resource_allocator.py -v`

Expected: PASS, 2 tests passed.

- [ ] **Step 5: Run all resource-allocation tests together**

Run: `python -m pytest tests/test_resource_requirements.py tests/test_resource_allocator.py -v`

Expected: PASS, 7 tests passed.

- [ ] **Step 6: Commit**

```bash
git add uav_mec/resource_allocation.py tests/test_resource_allocator.py
git commit -m "feat: maximize deadline completions under cpu budget"
```

---

### Task 6: Validate Explicit Schedules And Derive Serving UAVs

**Files:**
- Create: `uav_mec/schedule.py`
- Create: `tests/test_schedule.py`

- [ ] **Step 1: Write failing schedule-invariant tests**

```python
import numpy as np
import pytest

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.schedule import ScheduleValidationError, normalize_roles, serving_uav_by_task, validate_schedule


def snapshot() -> SlotSnapshot:
    tasks = (
        Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.2e9, 0.8, 1e9),
        Task(3, 2, 0, 1e6, 0.5e9, 0.8, 1e9),
    )
    return SlotSnapshot(0, tasks, np.array([[1e-8, 4e-8], [3e-8, 2e-8], [2e-8, 1e-8]]), 2)


def test_duplicate_task_is_rejected():
    schedule = Schedule((
        ScheduleEntry(1, 0, 0, 0, "solo"),
        ScheduleEntry(1, 0, 1, 0, "solo"),
    ))
    with pytest.raises(ScheduleValidationError, match="duplicate task"):
        validate_schedule(schedule, snapshot(), n_subcarriers=1)


def test_local_feasible_task_must_not_consume_radio_resource():
    schedule = Schedule((ScheduleEntry(3, 2, 0, 0, "solo"),))
    with pytest.raises(ScheduleValidationError, match="local-feasible"):
        validate_schedule(schedule, snapshot(), n_subcarriers=1)


def test_roles_are_normalized_on_actual_serving_channel():
    raw = Schedule((
        ScheduleEntry(1, 0, 1, 0, "strong"),
        ScheduleEntry(2, 1, 1, 0, "weak"),
    ))
    normalized = normalize_roles(raw, snapshot())
    assert normalized.entries == (
        ScheduleEntry(1, 0, 1, 0, "strong"),
        ScheduleEntry(2, 1, 1, 0, "weak"),
    )
    assert serving_uav_by_task(normalized) == {1: 1, 2: 1}
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run: `python -m pytest tests/test_schedule.py -v`

Expected: ERROR with `ModuleNotFoundError: No module named 'uav_mec.schedule'`.

- [ ] **Step 3: Implement structural validation and role normalization**

Create `uav_mec/schedule.py`:

```python
from __future__ import annotations

from collections import defaultdict

from .domain import Schedule, ScheduleEntry, SlotSnapshot


class ScheduleValidationError(ValueError):
    pass


def serving_uav_by_task(schedule: Schedule) -> dict[int, int]:
    return {entry.task_id: entry.serving_uav for entry in schedule.entries}


def validate_schedule(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    n_subcarriers: int,
) -> None:
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    seen_tasks: set[int] = set()
    seen_users: set[int] = set()
    groups: dict[tuple[int, int], list[ScheduleEntry]] = defaultdict(list)
    for entry in schedule.entries:
        if entry.task_id in seen_tasks:
            raise ScheduleValidationError(f"duplicate task {entry.task_id}")
        if entry.user_id in seen_users:
            raise ScheduleValidationError(f"duplicate user {entry.user_id}")
        task = task_by_id.get(entry.task_id)
        if task is None or task.user_id != entry.user_id:
            raise ScheduleValidationError(f"task/user mismatch for task {entry.task_id}")
        if not task.requires_offload:
            raise ScheduleValidationError(f"local-feasible task {entry.task_id} consumes radio")
        if not 0 <= entry.serving_uav < snapshot.num_uavs:
            raise ScheduleValidationError(f"invalid UAV {entry.serving_uav}")
        if not 0 <= entry.subcarrier < n_subcarriers:
            raise ScheduleValidationError(f"invalid subcarrier {entry.subcarrier}")
        seen_tasks.add(entry.task_id)
        seen_users.add(entry.user_id)
        groups[(entry.serving_uav, entry.subcarrier)].append(entry)

    for resource_block, entries in groups.items():
        if len(entries) > 2:
            raise ScheduleValidationError(f"capacity exceeded at {resource_block}")
        roles = {entry.noma_role for entry in entries}
        expected = {"solo"} if len(entries) == 1 else {"strong", "weak"}
        if roles != expected:
            raise ScheduleValidationError(
                f"roles {sorted(roles)} invalid at {resource_block}; expected {sorted(expected)}"
            )


def normalize_roles(schedule: Schedule, snapshot: SlotSnapshot) -> Schedule:
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    groups: dict[tuple[int, int], list[ScheduleEntry]] = defaultdict(list)
    for entry in schedule.entries:
        groups[(entry.serving_uav, entry.subcarrier)].append(entry)
    normalized: list[ScheduleEntry] = []
    for (uav, subcarrier), entries in sorted(groups.items()):
        if len(entries) == 1:
            entry = entries[0]
            normalized.append(ScheduleEntry(entry.task_id, entry.user_id, uav, subcarrier, "solo"))
            continue
        ordered = sorted(
            entries,
            key=lambda entry: (
                -snapshot.channel_gains[row_by_task[entry.task_id], uav],
                entry.task_id,
            ),
        )
        strong, weak = ordered
        normalized.extend((
            ScheduleEntry(strong.task_id, strong.user_id, uav, subcarrier, "strong"),
            ScheduleEntry(weak.task_id, weak.user_id, uav, subcarrier, "weak"),
        ))
    return Schedule(tuple(normalized))
```

- [ ] **Step 4: Run schedule tests**

Run: `python -m pytest tests/test_schedule.py -v`

Expected: PASS, 3 tests passed.

- [ ] **Step 5: Add capacity and role regression cases**

Append to `tests/test_schedule.py`:

```python
@pytest.mark.parametrize(
    ("entries", "message"),
    (
        (
            (
                ScheduleEntry(1, 0, 0, 0, "strong"),
                ScheduleEntry(2, 1, 0, 0, "weak"),
                ScheduleEntry(3, 2, 0, 0, "solo"),
            ),
            "capacity exceeded",
        ),
        (
            (
                ScheduleEntry(1, 0, 0, 0, "strong"),
                ScheduleEntry(2, 1, 0, 0, "strong"),
            ),
            "roles",
        ),
    ),
)
def test_resource_block_capacity_and_roles_are_hard_constraints(entries, message):
    target = snapshot()
    if message == "capacity exceeded":
        target = SlotSnapshot(
            target.slot_index,
            target.tasks[:2] + (Task(3, 2, 0, 1e6, 1.2e9, 0.8, 1e9),),
            target.channel_gains,
            target.num_uavs,
        )
    with pytest.raises(ScheduleValidationError, match=message):
        validate_schedule(Schedule(entries), target, n_subcarriers=1)
```

- [ ] **Step 6: Run schedule tests again**

Run: `python -m pytest tests/test_schedule.py -v`

Expected: PASS, 5 tests passed.

- [ ] **Step 7: Commit**

```bash
git add uav_mec/schedule.py tests/test_schedule.py
git commit -m "feat: validate explicit serving schedules"
```

---

### Task 7: Evaluate Rates, SIC, Delays, And CPU On The Actual Schedule

**Files:**
- Create: `uav_mec/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing evaluator tests**

```python
import numpy as np
import pytest

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, InfeasibleScheduleError, evaluate_schedule
from uav_mec.model import ChannelConfig, NOMAChannel


def config(gamma: float = 1.0) -> EvaluationConfig:
    return EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=gamma,
        cpu_per_uav_hz=4e9,
        result_ratio=0.1,
    )


def test_cross_uav_user_uses_schedule_serving_channel_and_cpu_pool():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[1e-10, 1e-7]]), 2)
    schedule = Schedule((ScheduleEntry(1, 0, 1, 0, "solo"),))
    outcome = evaluate_schedule(snapshot, schedule, NOMAChannel(ChannelConfig(), 100.0), config())
    rate = outcome.rates[0]
    decision = outcome.decisions[0]
    assert rate.task_id == 1
    assert rate.rate_bps > 1e6
    assert decision.completed
    assert decision.cpu_hz <= config().cpu_per_uav_hz


def test_pair_below_hard_sic_threshold_is_rejected():
    tasks = (
        Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.4e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[1e-8], [1e-8]]), 1)
    schedule = Schedule((
        ScheduleEntry(1, 0, 0, 0, "strong"),
        ScheduleEntry(2, 1, 0, 0, "weak"),
    ))
    with pytest.raises(InfeasibleScheduleError, match="SIC"):
        evaluate_schedule(snapshot, schedule, NOMAChannel(ChannelConfig(), 100.0), config())


def test_every_arrived_task_has_exactly_one_outcome():
    tasks = (
        Task(1, 0, 0, 1e6, 0.4e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.8e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[1e-8], [1e-8]]), 1)
    outcome = evaluate_schedule(snapshot, Schedule(), NOMAChannel(ChannelConfig(), 100.0), config())
    assert outcome.arrived_task_ids == (1, 2)
    assert outcome.completed_task_ids == (1,)
    assert outcome.missed_task_ids == (2,)
    assert {decision.task_id for decision in outcome.decisions} == {1, 2}


def test_evaluation_config_rejects_nonphysical_sic_residual():
    with pytest.raises(ValueError, match="sic_residual"):
        EvaluationConfig(1, 5e6, 0.2, 1.1, 1.0, 4e9, 0.1)
```

- [ ] **Step 2: Run tests and verify module import failure**

Run: `python -m pytest tests/test_evaluation.py -v`

Expected: ERROR with `ModuleNotFoundError: No module named 'uav_mec.evaluation'`.

- [ ] **Step 3: Implement evaluation configuration and rate computation**

Create `uav_mec/evaluation.py` with these public types and helpers:

```python
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from .domain import AllocationDecision, RateRecord, Schedule, SlotOutcome, SlotSnapshot
from .model import NOMAChannel
from .resource_allocation import allocate_deadline_tasks
from .schedule import ScheduleValidationError, validate_schedule


class InfeasibleScheduleError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    n_subcarriers: int
    bandwidth_hz: float
    transmit_power_w: float
    sic_residual: float
    gamma_sic_min: float
    cpu_per_uav_hz: float
    result_ratio: float

    def __post_init__(self) -> None:
        if self.n_subcarriers <= 0:
            raise ValueError("n_subcarriers must be positive")
        positive = (self.bandwidth_hz, self.transmit_power_w, self.gamma_sic_min, self.cpu_per_uav_hz)
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("bandwidth, power, SIC threshold, and CPU budget must be finite and positive")
        if not math.isfinite(self.sic_residual) or not 0.0 <= self.sic_residual <= 1.0:
            raise ValueError("sic_residual must be finite and in [0, 1]")
        if not math.isfinite(self.result_ratio) or self.result_ratio < 0:
            raise ValueError("result_ratio must be finite and non-negative")


def compute_rate_records(
    snapshot: SlotSnapshot,
    schedule: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> tuple[RateRecord, ...]:
    validate_schedule(schedule, snapshot, cfg.n_subcarriers)
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for entry in schedule.entries:
        groups[(entry.serving_uav, entry.subcarrier)].append(entry)
    bandwidth = cfg.bandwidth_hz / cfg.n_subcarriers
    noise = channel.noise_for_bandwidth(bandwidth)
    records: list[RateRecord] = []
    for (uav, subcarrier), own_entries in sorted(groups.items()):
        cross = 0.0
        for (other_uav, other_subcarrier), other_entries in groups.items():
            if other_uav == uav or other_subcarrier != subcarrier:
                continue
            for entry in other_entries:
                cross += cfg.transmit_power_w * snapshot.channel_gains[row_by_task[entry.task_id], uav]
        by_role = {entry.noma_role: entry for entry in own_entries}
        if len(own_entries) == 1:
            entry = own_entries[0]
            received = cfg.transmit_power_w * snapshot.channel_gains[row_by_task[entry.task_id], uav]
            sinr = received / (cross + noise)
            records.append(RateRecord(
                entry.task_id,
                bandwidth * math.log2(1.0 + sinr),
                sinr,
                True,
                cross,
                uav,
                subcarrier,
            ))
            continue
        strong = by_role["strong"]
        weak = by_role["weak"]
        recv_strong = cfg.transmit_power_w * snapshot.channel_gains[row_by_task[strong.task_id], uav]
        recv_weak = cfg.transmit_power_w * snapshot.channel_gains[row_by_task[weak.task_id], uav]
        if recv_strong < recv_weak:
            raise InfeasibleScheduleError("strong/weak order does not match serving channel")
        sinr_strong = recv_strong / (recv_weak + cross + noise)
        sinr_weak = recv_weak / (cfg.sic_residual * recv_strong + cross + noise)
        pair_feasible = sinr_strong >= cfg.gamma_sic_min
        records.extend((
            RateRecord(
                strong.task_id,
                bandwidth * math.log2(1.0 + sinr_strong),
                sinr_strong,
                pair_feasible,
                cross,
                uav,
                subcarrier,
            ),
            RateRecord(
                weak.task_id,
                bandwidth * math.log2(1.0 + sinr_weak),
                sinr_weak,
                pair_feasible,
                cross,
                uav,
                subcarrier,
            ),
        ))
    return tuple(sorted(records, key=lambda record: record.task_id))
```

- [ ] **Step 4: Implement unified slot evaluation**

Append this function to `uav_mec/evaluation.py`:

```python
def evaluate_schedule(
    snapshot: SlotSnapshot,
    schedule: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> SlotOutcome:
    try:
        rates = compute_rate_records(snapshot, schedule, channel, cfg)
    except ScheduleValidationError as exc:
        raise InfeasibleScheduleError(str(exc)) from exc
    violations = [record.task_id for record in rates if not record.sic_feasible]
    if violations:
        raise InfeasibleScheduleError(f"SIC threshold violated by tasks {violations}")

    task_by_id = {task.task_id: task for task in snapshot.tasks}
    rate_by_id = {record.task_id: record.rate_bps for record in rates}
    entries_by_uav: dict[int, list[int]] = defaultdict(list)
    for entry in schedule.entries:
        entries_by_uav[entry.serving_uav].append(entry.task_id)

    decisions: dict[int, AllocationDecision] = {}
    for task in snapshot.tasks:
        if not task.requires_offload:
            decisions[task.task_id] = AllocationDecision(
                task.task_id,
                0.0,
                0.0,
                True,
                task.local_delay_s,
                task.deadline_s - task.local_delay_s,
                "local",
            )

    for uav in range(snapshot.num_uavs):
        scheduled_tasks = tuple(task_by_id[task_id] for task_id in entries_by_uav.get(uav, ()))
        allocation = allocate_deadline_tasks(
            scheduled_tasks,
            rate_by_id,
            cfg.cpu_per_uav_hz,
            cfg.result_ratio,
        )
        decisions.update({decision.task_id: decision for decision in allocation.decisions})

    for task in snapshot.tasks:
        if task.task_id not in decisions:
            decisions[task.task_id] = AllocationDecision(
                task.task_id,
                0.0,
                0.0,
                False,
                None,
                0.0,
                "unscheduled",
            )

    ordered = tuple(decisions[task_id] for task_id in sorted(decisions))
    completed = tuple(decision.task_id for decision in ordered if decision.completed)
    missed = tuple(decision.task_id for decision in ordered if not decision.completed)
    return SlotOutcome(
        slot_index=snapshot.slot_index,
        arrived_task_ids=tuple(sorted(task_by_id)),
        completed_task_ids=completed,
        missed_task_ids=missed,
        decisions=ordered,
        rates=rates,
        total_slack_s=sum(decision.slack_s for decision in ordered if decision.completed),
        total_cross_interference_w=sum(record.cross_interference_w for record in rates),
    )
```

- [ ] **Step 5: Run evaluator tests**

Run: `python -m pytest tests/test_evaluation.py -v`

Expected: PASS, 4 tests passed.

- [ ] **Step 6: Run schedule, resource, and evaluator tests together**

Run: `python -m pytest tests/test_schedule.py tests/test_resource_requirements.py tests/test_resource_allocator.py tests/test_evaluation.py -q`

Expected: PASS with 16 tests and no warnings.

- [ ] **Step 7: Commit**

```bash
git add uav_mec/evaluation.py tests/test_evaluation.py
git commit -m "feat: evaluate feasible schedules on actual serving UAVs"
```

---

### Task 8: Generate Immutable Exogenous Traces And Count Tasks Once

**Files:**
- Modify: `uav_mec/domain.py`
- Create: `uav_mec/trace.py`
- Create: `uav_mec/simulation.py`
- Create: `tests/test_trace.py`
- Create: `tests/test_simulation.py`

- [ ] **Step 1: Write failing determinism and unique-task tests**

```python
import numpy as np

from uav_mec.domain import Schedule
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ScenarioConfig
from uav_mec.simulation import run_episode
from uav_mec.trace import generate_trace


def test_trace_is_identical_for_the_same_seed():
    cfg = ScenarioConfig(num_users=6, num_slots=4, random_burst=True, user_mobility=True)
    first = generate_trace(cfg, 123)
    second = generate_trace(cfg, 123)
    assert first.seed == second.seed
    assert [slot.tasks for slot in first.slots] == [slot.tasks for slot in second.slots]
    for left, right in zip(first.slots, second.slots, strict=True):
        np.testing.assert_array_equal(left.channel_gains, right.channel_gains)


def test_episode_denominator_equals_unique_arrivals_without_retries():
    cfg = ScenarioConfig(num_users=6, num_slots=5, random_burst=True, user_mobility=True)
    trace = generate_trace(cfg, 321)
    eval_cfg = EvaluationConfig(
        cfg.n_subcarriers,
        cfg.channel.bandwidth,
        cfg.channel.p_max,
        0.02,
        1.0,
        cfg.f_per_uav,
        cfg.task.result_ratio,
    )
    result = run_episode(trace, lambda snapshot: Schedule(), cfg, eval_cfg)
    expected_ids = {task.task_id for slot in trace.slots for task in slot.tasks}
    assert result.total_arrivals == len(expected_ids)
    assert len(result.task_ids) == len(set(result.task_ids))
    assert result.total_completed + result.total_missed == result.total_arrivals
```

- [ ] **Step 2: Run tests and verify missing trace/simulation modules**

Run: `python -m pytest tests/test_trace.py tests/test_simulation.py -v`

Expected: ERROR importing `uav_mec.trace` and `uav_mec.simulation`.

- [ ] **Step 3: Add trace and episode records to the domain module**

Append to `uav_mec/domain.py`:

```python
@dataclass(frozen=True, slots=True)
class ExperimentTrace:
    seed: int
    slots: tuple[SlotSnapshot, ...]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int
    outcomes: tuple[SlotOutcome, ...]

    @property
    def task_ids(self) -> tuple[int, ...]:
        return tuple(task_id for outcome in self.outcomes for task_id in outcome.arrived_task_ids)

    @property
    def total_arrivals(self) -> int:
        return len(self.task_ids)

    @property
    def total_completed(self) -> int:
        return sum(outcome.completed_count for outcome in self.outcomes)

    @property
    def total_missed(self) -> int:
        return self.total_arrivals - self.total_completed

    @property
    def on_time_rate(self) -> float:
        return self.total_completed / max(self.total_arrivals, 1)
```

- [ ] **Step 4: Implement deterministic trace generation**

Create `uav_mec/trace.py` with one RNG owned by trace generation. Reuse the fixed cluster/UAV position constants from `model.py`, but do not instantiate `MultiUAVMECEnv`:

```python
from __future__ import annotations

import math
import numpy as np

from .domain import ExperimentTrace, SlotSnapshot, Task
from .model import (
    FIXED_CLUSTER_CENTERS_3,
    FIXED_CLUSTER_CENTERS_4,
    FIXED_CLUSTER_CENTERS_5,
    NOMAChannel,
    ScenarioConfig,
    UAV_HOVER_POSITIONS,
)


def _centers(cfg: ScenarioConfig) -> np.ndarray:
    known = {3: FIXED_CLUSTER_CENTERS_3, 4: FIXED_CLUSTER_CENTERS_4, 5: FIXED_CLUSTER_CENTERS_5}
    if cfg.num_clusters not in known:
        raise ValueError("trusted trace supports 3, 4, or 5 fixed clusters")
    return known[cfg.num_clusters].copy()


def generate_trace(cfg: ScenarioConfig, seed: int) -> ExperimentTrace:
    if cfg.num_users <= 0 or cfg.num_slots <= 0 or cfg.n_subcarriers <= 0:
        raise ValueError("trusted trace requires positive users, slots, and subcarriers")
    if cfg.num_uavs not in UAV_HOVER_POSITIONS:
        raise ValueError(f"trusted trace has no fixed layout for K={cfg.num_uavs}")
    if not cfg.fixed_cluster_centers:
        raise ValueError("trusted trace requires fixed_cluster_centers=True")
    rng = np.random.default_rng(seed)
    centers = _centers(cfg)
    cluster_ids = np.arange(cfg.num_users) % cfg.num_clusters
    rng.shuffle(cluster_ids)
    positions = np.vstack([
        np.clip(rng.normal(centers[cluster_ids[user]], cfg.cluster_std), 0.0, cfg.area_size)
        for user in range(cfg.num_users)
    ])
    uav_positions = np.asarray(UAV_HOVER_POSITIONS[cfg.num_uavs], dtype=float)
    channel = NOMAChannel(cfg.channel, cfg.uav.height)
    burst_cluster = int(rng.integers(0, cfg.num_clusters))
    next_switch = int(rng.integers(cfg.burst_period_min, cfg.burst_period_max + 1))
    slots: list[SlotSnapshot] = []
    for slot in range(cfg.num_slots):
        if slot > 0 and cfg.user_mobility:
            angles = rng.uniform(0.0, 2.0 * math.pi, cfg.num_users)
            distance = cfg.user_speed * cfg.uav.dt
            positions[:, 0] += distance * np.cos(angles)
            positions[:, 1] += distance * np.sin(angles)
            positions = np.clip(positions, 0.0, cfg.area_size)
        if cfg.random_burst and slot >= next_switch:
            burst_cluster = int(rng.integers(0, cfg.num_clusters))
            next_switch = slot + int(rng.integers(cfg.burst_period_min, cfg.burst_period_max + 1))
        tasks: list[Task] = []
        for user in range(cfg.num_users):
            probability = cfg.task.arrival_prob
            if cfg.random_burst:
                probability = cfg.burst_rate if cluster_ids[user] == burst_cluster else cfg.cool_rate
            if rng.random() >= probability:
                continue
            data_bits = float(rng.uniform(cfg.task.data_min, cfg.task.data_max))
            density = float(rng.uniform(cfg.task.comp_min, cfg.task.comp_max))
            tasks.append(Task(
                task_id=slot * cfg.num_users + user,
                user_id=user,
                arrival_slot=slot,
                data_bits=data_bits,
                comp_cycles=data_bits * density,
                deadline_s=cfg.task.t_max,
                local_cpu_hz=cfg.task.f_local,
            ))
        gains = np.zeros((len(tasks), cfg.num_uavs), dtype=float)
        for row, task in enumerate(tasks):
            for uav in range(cfg.num_uavs):
                horizontal = float(np.linalg.norm(positions[task.user_id] - uav_positions[uav]))
                gains[row, uav] = channel.path_loss(horizontal)
        slots.append(SlotSnapshot(slot, tuple(tasks), gains, cfg.num_uavs))
    return ExperimentTrace(seed, tuple(slots))
```

- [ ] **Step 5: Implement the one-path episode runner**

Create `uav_mec/simulation.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import time

from .domain import EpisodeResult, ExperimentTrace, Schedule, SlotSnapshot
from .evaluation import EvaluationConfig, evaluate_schedule
from .model import NOMAChannel, ScenarioConfig

Scheduler = Callable[[SlotSnapshot], Schedule]


def run_episode(
    trace: ExperimentTrace,
    scheduler: Scheduler,
    scenario: ScenarioConfig,
    evaluation: EvaluationConfig,
) -> EpisodeResult:
    channel = NOMAChannel(scenario.channel, scenario.uav.height)
    outcomes = []
    seen: set[int] = set()
    for snapshot in trace.slots:
        ids = {task.task_id for task in snapshot.tasks}
        if seen.intersection(ids):
            raise ValueError("trace repeats a task_id across slots")
        seen.update(ids)
        started = time.perf_counter()
        schedule = scheduler(snapshot)
        outcome = evaluate_schedule(snapshot, schedule, channel, evaluation)
        outcomes.append(replace(outcome, runtime_s=time.perf_counter() - started))
    return EpisodeResult(trace.seed, tuple(outcomes))
```

- [ ] **Step 6: Run trace and simulation tests**

Run: `python -m pytest tests/test_trace.py tests/test_simulation.py -v`

Expected: PASS, 2 tests passed.

- [ ] **Step 7: Run all trusted-core tests**

Run: `python -m pytest tests/test_domain.py tests/test_physics.py tests/test_resource_requirements.py tests/test_resource_allocator.py tests/test_schedule.py tests/test_evaluation.py tests/test_trace.py tests/test_simulation.py -q`

Expected: PASS with no duplicate task IDs, no warnings, and no failures.

- [ ] **Step 8: Commit**

```bash
git add uav_mec/domain.py uav_mec/trace.py uav_mec/simulation.py tests/test_trace.py tests/test_simulation.py
git commit -m "feat: add deterministic unique-task simulation"
```

---

### Task 9: Build A Fail-Fast Per-UAV ILP Initializer

**Files:**
- Create: `uav_mec/scheduler.py`
- Create: `tests/test_initializer.py`

- [ ] **Step 1: Write failing initializer and solver-failure tests**

```python
from types import SimpleNamespace

import numpy as np
import pytest

from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, evaluate_schedule
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.scheduler import InitializationError, build_initial_schedule


def snapshot() -> SlotSnapshot:
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(6))
    gains = np.array([
        [8e-8, 1e-8],
        [6e-8, 2e-8],
        [4e-8, 3e-8],
        [3e-8, 5e-8],
        [2e-8, 7e-8],
        [1e-8, 9e-8],
    ])
    return SlotSnapshot(0, tasks, gains, 2)


def eval_cfg() -> EvaluationConfig:
    return EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)


def test_initializer_returns_a_fully_feasible_schedule():
    channel = NOMAChannel(ChannelConfig(), 100.0)
    result = build_initial_schedule(snapshot(), channel, eval_cfg())
    assert result.solver_status == "optimal"
    outcome = evaluate_schedule(snapshot(), result.schedule, channel, eval_cfg())
    assert all(record.sic_feasible for record in outcome.rates)
    assert len({entry.task_id for entry in result.schedule.entries}) == len(result.schedule.entries)


def test_initializer_never_silently_falls_back_when_milp_fails(monkeypatch):
    monkeypatch.setattr(
        "uav_mec.scheduler.milp",
        lambda *args, **kwargs: SimpleNamespace(success=False, message="forced failure"),
    )
    with pytest.raises(InitializationError, match="forced failure"):
        build_initial_schedule(snapshot(), NOMAChannel(ChannelConfig(), 100.0), eval_cfg())
```

- [ ] **Step 2: Run tests and verify missing scheduler module**

Run: `python -m pytest tests/test_initializer.py -v`

Expected: ERROR with `ModuleNotFoundError: No module named 'uav_mec.scheduler'`.

- [ ] **Step 3: Implement candidate construction and MILP selection**

Create `uav_mec/scheduler.py` with the following initializer. Candidate benefit is computed by the trusted evaluator with the other UAV empty; the combined result is repaired in Step 4.

```python
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .domain import Schedule, ScheduleEntry, SlotSnapshot
from .evaluation import (
    EvaluationConfig,
    InfeasibleScheduleError,
    compute_rate_records,
    evaluate_schedule,
)
from .model import NOMAChannel
from .schedule import normalize_roles


class InitializationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class InitializationResult:
    schedule: Schedule
    solver_status: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    uav: int
    task_ids: tuple[int, ...]
    benefit: int


def _candidate_schedule(
    candidate: _Candidate,
    subcarrier: int,
    snapshot: SlotSnapshot,
) -> Schedule:
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    raw = Schedule(tuple(
        ScheduleEntry(task_id, task_by_id[task_id].user_id, candidate.uav, subcarrier, "solo")
        for task_id in candidate.task_ids
    ))
    return normalize_roles(raw, snapshot)


def _candidates_for_uav(
    snapshot: SlotSnapshot,
    uav: int,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> list[_Candidate]:
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    heavy = [task for task in snapshot.tasks if task.requires_offload]
    assigned = [
        task for task in heavy
        if int(np.argmax(snapshot.channel_gains[row_by_task[task.task_id]])) == uav
    ]
    local_count = sum(not task.requires_offload for task in snapshot.tasks)
    candidates: list[_Candidate] = []
    task_groups = [(task.task_id,) for task in assigned]
    task_groups.extend(tuple(task.task_id for task in pair) for pair in combinations(assigned, 2))
    for task_ids in task_groups:
        trial = _Candidate(uav, tuple(sorted(task_ids)), 0)
        schedule = _candidate_schedule(trial, 0, snapshot)
        try:
            outcome = evaluate_schedule(snapshot, schedule, channel, cfg)
        except InfeasibleScheduleError:
            continue
        benefit = max(outcome.completed_count - local_count, 0)
        candidates.append(_Candidate(uav, tuple(sorted(task_ids)), benefit))
    return candidates


def _solve_uav_candidates(candidates: list[_Candidate], n_subcarriers: int) -> list[_Candidate]:
    if not candidates:
        return []
    task_ids = sorted({task_id for candidate in candidates for task_id in candidate.task_ids})
    rows = []
    upper = []
    for task_id in task_ids:
        rows.append([float(task_id in candidate.task_ids) for candidate in candidates])
        upper.append(1.0)
    rows.append([1.0] * len(candidates))
    upper.append(float(n_subcarriers))
    objective = np.array([
        -float(candidate.benefit) - 1e-6 / (1 + sum(candidate.task_ids))
        for candidate in candidates
    ])
    result = milp(
        c=objective,
        integrality=np.ones(len(candidates)),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=LinearConstraint(np.asarray(rows), -np.inf, np.asarray(upper)),
    )
    if not result.success:
        raise InitializationError(f"per-UAV MILP failed: {result.message}")
    return [candidate for candidate, chosen in zip(candidates, result.x, strict=True) if chosen > 0.5]
```

- [ ] **Step 4: Implement deterministic SIC feasibility repair and the public initializer**

Append to `uav_mec/scheduler.py`:

```python
def _remove_block(schedule: Schedule, block: tuple[int, int]) -> Schedule:
    return Schedule(tuple(
        entry for entry in schedule.entries
        if (entry.serving_uav, entry.subcarrier) != block
    ))


def repair_sic(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> Schedule:
    current = normalize_roles(schedule, snapshot)
    while current.entries:
        records = compute_rate_records(snapshot, current, channel, cfg)
        violating_ids = {record.task_id for record in records if not record.sic_feasible}
        if not violating_ids:
            return current
        violating_blocks = sorted({
            (entry.serving_uav, entry.subcarrier)
            for entry in current.entries
            if entry.task_id in violating_ids
        })
        current = _remove_block(current, violating_blocks[-1])
    return current


def build_initial_schedule(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> InitializationResult:
    entries: list[ScheduleEntry] = []
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    for uav in range(snapshot.num_uavs):
        selected = _solve_uav_candidates(
            _candidates_for_uav(snapshot, uav, channel, cfg),
            cfg.n_subcarriers,
        )
        selected.sort(key=lambda candidate: (-candidate.benefit, candidate.task_ids))
        for subcarrier, candidate in enumerate(selected):
            entries.extend(
                ScheduleEntry(
                    task_id,
                    task_by_id[task_id].user_id,
                    uav,
                    subcarrier,
                    "solo",
                )
                for task_id in candidate.task_ids
            )
    repaired = repair_sic(Schedule(tuple(entries)), snapshot, channel, cfg)
    evaluate_schedule(snapshot, repaired, channel, cfg)
    return InitializationResult(repaired, "optimal")
```

- [ ] **Step 5: Run initializer tests**

Run: `python -m pytest tests/test_initializer.py -v`

Expected: PASS, 2 tests passed.

- [ ] **Step 6: Add a deterministic-repeat regression test**

Append to `tests/test_initializer.py`:

```python
def test_initializer_is_deterministic_across_repeated_solves():
    channel = NOMAChannel(ChannelConfig(), 100.0)
    keys = {
        build_initial_schedule(snapshot(), channel, eval_cfg()).schedule.stable_key()
        for _ in range(5)
    }
    assert len(keys) == 1
```

- [ ] **Step 7: Run initializer tests five times in one process**

Run: `python -m pytest tests/test_initializer.py -v`

Expected: PASS, 3 tests passed; the repeat regression observes one stable schedule key.

- [ ] **Step 8: Commit**

```bash
git add uav_mec/scheduler.py tests/test_initializer.py
git commit -m "feat: add fail-fast feasible ilp initialization"
```

---

### Task 10: Generate Deterministic Replace, Position, And Cross-UAV Neighbors

**Files:**
- Modify: `uav_mec/scheduler.py`
- Create: `tests/test_neighborhoods.py`

- [ ] **Step 1: Write failing neighborhood tests**

```python
import numpy as np

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.scheduler import generate_neighbors


def snapshot() -> SlotSnapshot:
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(5))
    gains = np.array([
        [9e-8, 1e-8],
        [7e-8, 2e-8],
        [6e-8, 4e-8],
        [2e-8, 8e-8],
        [1e-8, 9e-8],
    ])
    return SlotSnapshot(0, tasks, gains, 2)


def initial() -> Schedule:
    return Schedule((
        ScheduleEntry(0, 0, 0, 0, "strong"),
        ScheduleEntry(1, 1, 0, 0, "weak"),
        ScheduleEntry(3, 3, 1, 0, "strong"),
        ScheduleEntry(4, 4, 1, 0, "weak"),
    ))


def test_intra_neighborhood_has_no_cross_uav_reassignment():
    neighbors = generate_neighbors(initial(), snapshot(), n_subcarriers=2, allow_cross_uav=False)
    original_uav = {entry.task_id: entry.serving_uav for entry in initial().entries}
    for neighbor in neighbors:
        for entry in neighbor.schedule.entries:
            if entry.task_id in original_uav:
                assert entry.serving_uav == original_uav[entry.task_id]


def test_cross_neighborhood_contains_actual_service_exchange():
    neighbors = generate_neighbors(initial(), snapshot(), n_subcarriers=2, allow_cross_uav=True)
    assert any(
        neighbor.move_type == "cross"
        and any(entry.task_id == 0 and entry.serving_uav == 1 for entry in neighbor.schedule.entries)
        for neighbor in neighbors
    )


def test_neighbors_are_unique_and_stably_ordered():
    neighbors = generate_neighbors(initial(), snapshot(), n_subcarriers=2, allow_cross_uav=True)
    keys = [neighbor.schedule.stable_key() for neighbor in neighbors]
    assert len(keys) == len(set(keys))
    assert [(neighbor.move_type, neighbor.schedule.stable_key()) for neighbor in neighbors] == sorted(
        (neighbor.move_type, neighbor.schedule.stable_key()) for neighbor in neighbors
    )
```

- [ ] **Step 2: Run tests and verify missing `generate_neighbors`**

Run: `python -m pytest tests/test_neighborhoods.py -v`

Expected: FAIL importing `generate_neighbors`.

- [ ] **Step 3: Implement immutable neighbor records and helpers**

Append to `uav_mec/scheduler.py`:

```python
from collections import defaultdict
from typing import Literal

from .schedule import validate_schedule

MoveType = Literal["replace", "position", "cross"]


@dataclass(frozen=True, slots=True)
class Neighbor:
    schedule: Schedule
    move_type: MoveType
    affected_blocks: tuple[tuple[int, int], ...]


def _with_task(entry: ScheduleEntry, task_id: int, user_id: int) -> ScheduleEntry:
    return ScheduleEntry(task_id, user_id, entry.serving_uav, entry.subcarrier, entry.noma_role)


def _nearest_uav(snapshot: SlotSnapshot, task_id: int) -> int:
    row = next(row for row, task in enumerate(snapshot.tasks) if task.task_id == task_id)
    return int(np.argmax(snapshot.channel_gains[row]))


def _normalized_neighbor(
    entries: list[ScheduleEntry],
    move_type: MoveType,
    affected: set[tuple[int, int]],
    snapshot: SlotSnapshot,
    n_subcarriers: int,
) -> Neighbor:
    schedule = normalize_roles(Schedule(tuple(entries)), snapshot)
    validate_schedule(schedule, snapshot, n_subcarriers)
    return Neighbor(schedule, move_type, tuple(sorted(affected)))
```

- [ ] **Step 4: Implement all three neighborhoods with stable deduplication**

Append to `uav_mec/scheduler.py`:

```python
def generate_neighbors(
    schedule: Schedule,
    snapshot: SlotSnapshot,
    n_subcarriers: int,
    allow_cross_uav: bool,
) -> tuple[Neighbor, ...]:
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    scheduled_ids = {entry.task_id for entry in schedule.entries}
    unscheduled = [
        task for task in snapshot.tasks
        if task.requires_offload and task.task_id not in scheduled_ids
    ]
    neighbors: list[Neighbor] = []
    entries = list(schedule.entries)

    for index, old in enumerate(entries):
        for task in unscheduled:
            if _nearest_uav(snapshot, task.task_id) != old.serving_uav:
                continue
            changed = entries.copy()
            changed[index] = _with_task(old, task.task_id, task.user_id)
            neighbors.append(_normalized_neighbor(
                changed,
                "replace",
                {(old.serving_uav, old.subcarrier)},
                snapshot,
                n_subcarriers,
            ))

    for left in range(len(entries)):
        for right in range(left + 1, len(entries)):
            first, second = entries[left], entries[right]
            if first.serving_uav != second.serving_uav or first.subcarrier == second.subcarrier:
                continue
            changed = entries.copy()
            changed[left] = _with_task(first, second.task_id, second.user_id)
            changed[right] = _with_task(second, first.task_id, first.user_id)
            neighbors.append(_normalized_neighbor(
                changed,
                "position",
                {
                    (first.serving_uav, first.subcarrier),
                    (second.serving_uav, second.subcarrier),
                },
                snapshot,
                n_subcarriers,
            ))

    if allow_cross_uav:
        for left in range(len(entries)):
            for right in range(left + 1, len(entries)):
                first, second = entries[left], entries[right]
                if first.serving_uav == second.serving_uav:
                    continue
                changed = entries.copy()
                changed[left] = _with_task(first, second.task_id, second.user_id)
                changed[right] = _with_task(second, first.task_id, first.user_id)
                neighbors.append(_normalized_neighbor(
                    changed,
                    "cross",
                    {
                        (first.serving_uav, first.subcarrier),
                        (second.serving_uav, second.subcarrier),
                    },
                    snapshot,
                    n_subcarriers,
                ))

    unique: dict[tuple, Neighbor] = {}
    for neighbor in neighbors:
        unique[neighbor.schedule.stable_key()] = neighbor
    return tuple(sorted(
        unique.values(),
        key=lambda neighbor: (neighbor.move_type, neighbor.schedule.stable_key()),
    ))
```

- [ ] **Step 5: Run neighborhood tests**

Run: `python -m pytest tests/test_neighborhoods.py -v`

Expected: PASS, 3 tests passed.

- [ ] **Step 6: Commit**

```bash
git add uav_mec/scheduler.py tests/test_neighborhoods.py
git commit -m "feat: add deterministic pairing neighborhoods"
```

---

### Task 11: Implement Global And Bilateral First-Improvement Search

**Files:**
- Modify: `uav_mec/scheduler.py`
- Create: `tests/test_search.py`

- [ ] **Step 1: Write failing objective and convergence tests**

```python
import numpy as np

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.scheduler import Acceptance, SearchConfig, optimize_schedule


def test_replace_search_strictly_improves_and_converges():
    tasks = (
        Task(1, 0, 0, 2e6, 1.9e9, 0.8, 1e9),
        Task(2, 1, 0, 0.4e6, 1.1e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[2e-9], [9e-8]]), 1)
    initial = Schedule((ScheduleEntry(1, 0, 0, 0, "solo"),))
    evaluation = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)
    result = optimize_schedule(
        initial,
        snapshot,
        NOMAChannel(ChannelConfig(), 100.0),
        evaluation,
        SearchConfig(False, Acceptance.GLOBAL, 20),
    )
    assert result.outcome.completed_count >= 1
    assert any(entry.task_id == 2 for entry in result.schedule.entries)
    assert all(event.after_objective > event.before_objective for event in result.events if event.accepted)
    assert result.iterations <= 20
    assert result.converged
    assert result.termination_reason == "local_optimum"


def test_search_is_deterministic():
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(4))
    snapshot = SlotSnapshot(0, tasks, np.array([[9e-8], [7e-8], [5e-8], [3e-8]]), 1)
    initial = Schedule((
        ScheduleEntry(0, 0, 0, 0, "strong"),
        ScheduleEntry(3, 3, 0, 0, "weak"),
    ))
    evaluation = EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = SearchConfig(False, Acceptance.GLOBAL, 20)
    keys = {optimize_schedule(initial, snapshot, channel, evaluation, cfg).schedule.stable_key() for _ in range(5)}
    assert len(keys) == 1
```

- [ ] **Step 2: Run tests and verify missing search API**

Run: `python -m pytest tests/test_search.py -v`

Expected: FAIL importing `Acceptance`, `SearchConfig`, and `optimize_schedule`.

- [ ] **Step 3: Implement objectives, events, and acceptance rules**

Append to `uav_mec/scheduler.py`:

```python
from enum import Enum

from .domain import SlotOutcome


class Acceptance(str, Enum):
    GLOBAL = "global"
    BILATERAL = "bilateral"


@dataclass(frozen=True, slots=True)
class SearchConfig:
    allow_cross_uav: bool
    acceptance: Acceptance
    max_iterations: int = 50

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True, slots=True)
class SearchEvent:
    move_type: MoveType
    accepted: bool
    reason: str
    before_objective: tuple[int, float, float]
    after_objective: tuple[int, float, float]
    schedule_key: tuple


@dataclass(frozen=True, slots=True)
class SearchResult:
    schedule: Schedule
    outcome: SlotOutcome
    events: tuple[SearchEvent, ...]
    iterations: int
    converged: bool
    termination_reason: str


def network_objective(outcome: SlotOutcome) -> tuple[int, float, float]:
    return (
        outcome.completed_count,
        round(outcome.total_slack_s, 12),
        round(-outcome.total_cross_interference_w, 18),
    )


def _block_utility(
    schedule: Schedule,
    outcome: SlotOutcome,
    block: tuple[int, int],
) -> tuple[int, float]:
    task_ids = {
        entry.task_id for entry in schedule.entries
        if (entry.serving_uav, entry.subcarrier) == block
    }
    decisions = [decision for decision in outcome.decisions if decision.task_id in task_ids]
    return (
        sum(decision.completed for decision in decisions),
        round(sum(decision.slack_s for decision in decisions if decision.completed), 12),
    )


def _bilateral_accepts(
    before_schedule: Schedule,
    before: SlotOutcome,
    neighbor: Neighbor,
    after: SlotOutcome,
) -> bool:
    comparisons = [
        (
            _block_utility(before_schedule, before, block),
            _block_utility(neighbor.schedule, after, block),
        )
        for block in neighbor.affected_blocks
    ]
    return bool(comparisons) and all(new >= old for old, new in comparisons) and any(
        new > old for old, new in comparisons
    )
```

- [ ] **Step 4: Implement fail-safe first-improvement search**

Append to `uav_mec/scheduler.py`:

```python
def optimize_schedule(
    initial: Schedule,
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    evaluation: EvaluationConfig,
    search: SearchConfig,
) -> SearchResult:
    current = normalize_roles(initial, snapshot)
    current_outcome = evaluate_schedule(snapshot, current, channel, evaluation)
    events: list[SearchEvent] = []
    visited = {current.stable_key()}
    iterations = 0
    while iterations < search.max_iterations:
        iterations += 1
        improved = False
        for neighbor in generate_neighbors(
            current,
            snapshot,
            evaluation.n_subcarriers,
            search.allow_cross_uav,
        ):
            before_objective = network_objective(current_outcome)
            if neighbor.schedule.stable_key() in visited:
                events.append(SearchEvent(
                    neighbor.move_type,
                    False,
                    "visited schedule",
                    before_objective,
                    before_objective,
                    neighbor.schedule.stable_key(),
                ))
                continue
            try:
                candidate_outcome = evaluate_schedule(snapshot, neighbor.schedule, channel, evaluation)
            except InfeasibleScheduleError as exc:
                events.append(SearchEvent(
                    neighbor.move_type,
                    False,
                    str(exc),
                    before_objective,
                    before_objective,
                    neighbor.schedule.stable_key(),
                ))
                continue
            after_objective = network_objective(candidate_outcome)
            accepted = (
                after_objective > before_objective
                if search.acceptance is Acceptance.GLOBAL
                else _bilateral_accepts(current, current_outcome, neighbor, candidate_outcome)
            )
            events.append(SearchEvent(
                neighbor.move_type,
                accepted,
                "strict improvement" if accepted else "acceptance criterion not met",
                before_objective,
                after_objective,
                neighbor.schedule.stable_key(),
            ))
            if accepted:
                current = neighbor.schedule
                current_outcome = candidate_outcome
                visited.add(current.stable_key())
                improved = True
                break
        if not improved:
            break
    converged = not improved
    if not converged:
        termination_reason = "max_iterations"
    elif search.acceptance is Acceptance.GLOBAL:
        termination_reason = "local_optimum"
    else:
        termination_reason = "cycle_free_fixed_point"
    return SearchResult(
        current,
        current_outcome,
        tuple(events),
        iterations,
        converged,
        termination_reason,
    )
```

- [ ] **Step 5: Run search tests**

Run: `python -m pytest tests/test_search.py -v`

Expected: PASS, 2 tests passed.

- [ ] **Step 6: Add a direct bilateral/global criterion test**

Extend the imports in `tests/test_search.py`:

```python
from uav_mec.domain import AllocationDecision, SlotOutcome
from uav_mec.scheduler import Neighbor, _bilateral_accepts, network_objective
```

Append this test:

```python
def test_global_gain_can_fail_bilateral_acceptance():
    schedule = Schedule((
        ScheduleEntry(1, 0, 0, 0, "solo"),
        ScheduleEntry(2, 1, 0, 1, "strong"),
        ScheduleEntry(3, 2, 0, 1, "weak"),
    ))
    before = SlotOutcome(
        0,
        (1, 2, 3),
        (1,),
        (2, 3),
        (
            AllocationDecision(1, 0.4, 1e9, True, 0.7, 0.1, "complete"),
            AllocationDecision(2, 0.4, 0.0, False, None, 0.0, "cpu_budget"),
            AllocationDecision(3, 0.4, 0.0, False, None, 0.0, "cpu_budget"),
        ),
        (),
        0.1,
        2e-9,
    )
    after = SlotOutcome(
        0,
        (1, 2, 3),
        (2, 3),
        (1,),
        (
            AllocationDecision(1, 0.4, 0.0, False, None, 0.0, "cpu_budget"),
            AllocationDecision(2, 0.4, 1e9, True, 0.75, 0.05, "complete"),
            AllocationDecision(3, 0.4, 1e9, True, 0.75, 0.05, "complete"),
        ),
        (),
        0.1,
        1e-9,
    )
    neighbor = Neighbor(schedule, "position", ((0, 0), (0, 1)))
    assert network_objective(after) > network_objective(before)
    assert not _bilateral_accepts(schedule, before, neighbor, after)
```

- [ ] **Step 7: Run the complete scheduler test group**

Run: `python -m pytest tests/test_initializer.py tests/test_neighborhoods.py tests/test_search.py -q`

Expected: PASS with 9 tests and no nondeterministic failures.

- [ ] **Step 8: Commit**

```bash
git add uav_mec/scheduler.py tests/test_search.py
git commit -m "feat: add global and bilateral schedule search"
```

---

### Task 12: Add Controlled Baselines And Complete Small-Instance Enumeration

**Files:**
- Create: `uav_mec/baselines_v2.py`
- Create: `uav_mec/exact.py`
- Create: `tests/test_baselines_v2.py`
- Create: `tests/test_exact.py`

- [ ] **Step 1: Write failing baseline and exact-reference tests**

Create `tests/test_baselines_v2.py`:

```python
import numpy as np

from uav_mec.baselines_v2 import local_only_schedule
from uav_mec.domain import SlotSnapshot, Task


def test_local_only_never_schedules_radio_users():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(0, (task,), np.array([[1e-8]]), 1)
    assert local_only_schedule(snapshot).entries == ()
```

Create `tests/test_exact.py`:

```python
import numpy as np

from uav_mec.baselines_v2 import greedy_schedule
from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, evaluate_schedule
from uav_mec.exact import exact_schedule
from uav_mec.model import ChannelConfig, NOMAChannel


def test_complete_enumeration_is_never_worse_than_greedy():
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(5))
    gains = np.array([[9e-8], [7e-8], [5e-8], [3e-8], [1e-8]])
    snapshot = SlotSnapshot(0, tasks, gains, 1)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)
    greedy = evaluate_schedule(snapshot, greedy_schedule(snapshot, 2), channel, cfg)
    exact = exact_schedule(snapshot, channel, cfg, max_heavy_tasks=6)
    assert exact.outcome.completed_count >= greedy.completed_count
    assert exact.complete_search
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `python -m pytest tests/test_baselines_v2.py tests/test_exact.py -v`

Expected: ERROR importing `uav_mec.baselines_v2` and `uav_mec.exact`.

- [ ] **Step 3: Implement controlled local and greedy baselines**

Create `uav_mec/baselines_v2.py`:

```python
from __future__ import annotations

import numpy as np

from .domain import Schedule, ScheduleEntry, SlotSnapshot
from .schedule import normalize_roles


def local_only_schedule(snapshot: SlotSnapshot) -> Schedule:
    return Schedule()


def greedy_schedule(snapshot: SlotSnapshot, n_subcarriers: int) -> Schedule:
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    entries: list[ScheduleEntry] = []
    for uav in range(snapshot.num_uavs):
        candidates = [
            task for task in snapshot.tasks
            if task.requires_offload
            and int(np.argmax(snapshot.channel_gains[row_by_task[task.task_id]])) == uav
        ]
        candidates.sort(key=lambda task: (
            -snapshot.channel_gains[row_by_task[task.task_id], uav],
            task.task_id,
        ))
        selected = candidates[: 2 * n_subcarriers]
        for subcarrier in range(n_subcarriers):
            left = subcarrier
            right = len(selected) - 1 - subcarrier
            if left > right:
                break
            for index in sorted({left, right}):
                task = selected[index]
                entries.append(ScheduleEntry(task.task_id, task.user_id, uav, subcarrier, "solo"))
    return normalize_roles(Schedule(tuple(entries)), snapshot)
```

- [ ] **Step 4: Implement complete assignment enumeration for at most six heavy tasks**

Create `uav_mec/exact.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .domain import Schedule, ScheduleEntry, SlotOutcome, SlotSnapshot
from .evaluation import EvaluationConfig, InfeasibleScheduleError, evaluate_schedule
from .model import NOMAChannel
from .schedule import normalize_roles
from .scheduler import network_objective


@dataclass(frozen=True, slots=True)
class ExactResult:
    schedule: Schedule
    outcome: SlotOutcome
    evaluated_schedules: int
    complete_search: bool


def exact_schedule(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    max_heavy_tasks: int,
) -> ExactResult:
    heavy = tuple(task for task in snapshot.tasks if task.requires_offload)
    if len(heavy) > max_heavy_tasks:
        raise ValueError(f"exact search limited to {max_heavy_tasks} heavy tasks")
    resource_blocks = snapshot.num_uavs * cfg.n_subcarriers
    choices = range(-1, resource_blocks)
    best_schedule = Schedule()
    best_outcome = evaluate_schedule(snapshot, best_schedule, channel, cfg)
    evaluated = 1
    for assignment in product(choices, repeat=len(heavy)):
        counts = {block: assignment.count(block) for block in range(resource_blocks)}
        if any(count > 2 for count in counts.values()):
            continue
        entries = []
        for task, block in zip(heavy, assignment, strict=True):
            if block < 0:
                continue
            uav, subcarrier = divmod(block, cfg.n_subcarriers)
            entries.append(ScheduleEntry(task.task_id, task.user_id, uav, subcarrier, "solo"))
        schedule = normalize_roles(Schedule(tuple(entries)), snapshot)
        try:
            outcome = evaluate_schedule(snapshot, schedule, channel, cfg)
        except InfeasibleScheduleError:
            continue
        evaluated += 1
        candidate = (network_objective(outcome), schedule.stable_key())
        incumbent = (network_objective(best_outcome), best_schedule.stable_key())
        if candidate[0] > incumbent[0] or (candidate[0] == incumbent[0] and candidate[1] < incumbent[1]):
            best_schedule, best_outcome = schedule, outcome
    return ExactResult(best_schedule, best_outcome, evaluated, True)
```

- [ ] **Step 5: Run baseline and exact tests**

Run: `python -m pytest tests/test_baselines_v2.py tests/test_exact.py -v`

Expected: PASS, 2 tests passed.

- [ ] **Step 6: Add full-feasible-space assertions**

Extend `tests/test_exact.py` with the required imports and test:

```python
import numpy as np

from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, evaluate_schedule
from uav_mec.exact import exact_schedule
from uav_mec.model import ChannelConfig, NOMAChannel


def test_exact_search_enumerates_non_nearest_uav_associations():
    tasks = tuple(Task(i, i, 0, 0.2e6, 1.2e9, 0.8, 1e9) for i in range(4))
    gains = np.array([
        [1e-6, 1e-12],
        [1e-7, 1e-12],
        [9e-8, 8e-8],
        [9e-9, 8e-9],
    ])
    snapshot = SlotSnapshot(0, tasks, gains, 2)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 8e9, 0.1)
    result = exact_schedule(snapshot, channel, cfg, max_heavy_tasks=4)
    serving = {entry.task_id: entry.serving_uav for entry in result.schedule.entries}
    assert result.outcome.completed_count == 4
    assert any(serving[task.task_id] != int(np.argmax(gains[row])) for row, task in enumerate(tasks))
    assert len(serving) == len(result.schedule.entries)
    verified = evaluate_schedule(snapshot, result.schedule, channel, cfg)
    assert all(record.sic_feasible for record in verified.rates)
```

- [ ] **Step 7: Run the expanded exact-search test**

Run: `python -m pytest tests/test_exact.py -v`

Expected: PASS, 2 tests passed; the complete search uses a non-nearest association when capacity requires it.

- [ ] **Step 8: Commit**

```bash
git add uav_mec/baselines_v2.py uav_mec/exact.py tests/test_baselines_v2.py tests/test_exact.py
git commit -m "feat: add controlled baselines and exact reference"
```

---

### Task 13: Add Controlled Resource-Allocation Policies For Ablation

**Files:**
- Modify: `uav_mec/resource_allocation.py`
- Modify: `uav_mec/evaluation.py`
- Create: `tests/test_resource_policies.py`

- [ ] **Step 1: Write failing policy-isolation tests**

```python
import pytest

from uav_mec.domain import Task
from uav_mec.resource_allocation import AllocationPolicy, allocate_with_policy


def test_equal_cpu_policy_assigns_identical_cpu_to_all_scheduled_tasks():
    tasks = (
        Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.6e9, 0.8, 1e9),
    )
    result = allocate_with_policy(
        AllocationPolicy.CROSSING_EQUAL,
        tasks,
        {1: 4e6, 2: 4e6},
        4e9,
        0.1,
    )
    assert {decision.cpu_hz for decision in result.decisions} == {2e9}


def test_fixed_half_policy_really_uses_alpha_half():
    task = Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9)
    result = allocate_with_policy(
        AllocationPolicy.FIXED_HALF_EQUAL,
        (task,),
        {1: 5e6},
        4e9,
        0.1,
    )
    assert result.decisions[0].alpha == 0.5


def test_deadline_policy_matches_the_proven_allocator():
    tasks = (
        Task(1, 0, 0, 1e6, 1.2e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.8e9, 0.8, 1e9),
    )
    result = allocate_with_policy(AllocationPolicy.DEADLINE, tasks, {1: 3e6, 2: 3e6}, 2e9, 0.1)
    assert result.total_cpu_hz <= 2e9
    by_id = {decision.task_id: decision for decision in result.decisions}
    assert by_id[1].alpha == pytest.approx(1.0 / 3.0)
    assert by_id[2].alpha == pytest.approx(5.0 / 9.0)
    assert result.completed_task_ids
```

- [ ] **Step 2: Run tests and verify missing policy API**

Run: `python -m pytest tests/test_resource_policies.py -v`

Expected: FAIL importing `AllocationPolicy` and `allocate_with_policy`.

- [ ] **Step 3: Implement fixed/equal and crossing/equal allocations**

Append to `uav_mec/resource_allocation.py`:

```python
class AllocationPolicy(str, Enum):
    DEADLINE = "deadline"
    CROSSING_EQUAL = "crossing_equal"
    FIXED_HALF_EQUAL = "fixed_half_equal"


def _equal_cpu_allocation(
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    result_ratio: float,
    fixed_alpha: float | None,
) -> AllocationResult:
    if not tasks:
        return AllocationResult((), (), 0.0)
    cpu_hz = cpu_budget_hz / len(tasks)
    decisions = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        rate = rates_bps.get(task.task_id, 0.0)
        if rate <= 0:
            decisions.append(AllocationDecision(task.task_id, 0.0, 0.0, False, None, 0.0, "invalid_rate"))
            continue
        if fixed_alpha is None:
            local_full = task.comp_cycles / task.local_cpu_hz
            edge_unit = (1.0 + result_ratio) * task.data_bits / rate + task.comp_cycles / cpu_hz
            alpha = local_full / (local_full + edge_unit)
        else:
            alpha = fixed_alpha
        local_delay = (1.0 - alpha) * task.comp_cycles / task.local_cpu_hz
        edge_delay = alpha * (
            (1.0 + result_ratio) * task.data_bits / rate
            + task.comp_cycles / cpu_hz
        )
        delay = max(local_delay, edge_delay)
        completed = delay <= task.deadline_s
        decisions.append(AllocationDecision(
            task.task_id,
            alpha,
            cpu_hz,
            completed,
            delay,
            max(task.deadline_s - delay, 0.0) if completed else 0.0,
            "crossing_equal" if fixed_alpha is None else "fixed_half_equal",
        ))
    ordered = tuple(decisions)
    return AllocationResult(
        ordered,
        tuple(decision.task_id for decision in ordered if decision.completed),
        cpu_budget_hz,
    )


def allocate_with_policy(
    policy: AllocationPolicy,
    tasks: Sequence[Task],
    rates_bps: Mapping[int, float],
    cpu_budget_hz: float,
    result_ratio: float,
) -> AllocationResult:
    if policy is AllocationPolicy.DEADLINE:
        return allocate_deadline_tasks(tasks, rates_bps, cpu_budget_hz, result_ratio)
    if policy is AllocationPolicy.CROSSING_EQUAL:
        return _equal_cpu_allocation(tasks, rates_bps, cpu_budget_hz, result_ratio, None)
    if policy is AllocationPolicy.FIXED_HALF_EQUAL:
        return _equal_cpu_allocation(tasks, rates_bps, cpu_budget_hz, result_ratio, 0.5)
    raise ValueError(f"unsupported allocation policy {policy}")
```

- [ ] **Step 4: Route evaluation through the selected policy**

Add this final field to `EvaluationConfig`:

```python
allocation_policy: AllocationPolicy = AllocationPolicy.DEADLINE
```

Replace the call to `allocate_deadline_tasks` in `evaluate_schedule` with:

```python
allocation = allocate_with_policy(
    cfg.allocation_policy,
    scheduled_tasks,
    rate_by_id,
    cfg.cpu_per_uav_hz,
    cfg.result_ratio,
)
```

Import `AllocationPolicy` and `allocate_with_policy` from `resource_allocation.py` and remove the direct allocator import.

- [ ] **Step 5: Run policy and evaluator tests**

Run: `python -m pytest tests/test_resource_policies.py tests/test_evaluation.py -v`

Expected: PASS, 7 tests passed.

- [ ] **Step 6: Commit**

```bash
git add uav_mec/resource_allocation.py uav_mec/evaluation.py tests/test_resource_policies.py
git commit -m "feat: isolate resource allocation policies"
```

---

### Task 14: Implement Correct Paired Statistics

**Files:**
- Create: `uav_mec/statistics.py`
- Create: `tests/test_statistics.py`

- [ ] **Step 1: Write failing tests for Holm, bootstrap, and effect size**

```python
import numpy as np
import pytest

from uav_mec.statistics import holm_adjust, paired_bootstrap_ci, paired_comparison


def test_holm_step_down_uses_cumulative_maximum():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_paired_bootstrap_is_deterministic():
    differences = np.array([1.0, 2.0, 3.0, 4.0])
    assert paired_bootstrap_ci(differences, 10_000, 20260711) == paired_bootstrap_ci(
        differences,
        10_000,
        20260711,
    )


def test_paired_comparison_reports_positive_effect():
    proposed = np.array([80.0, 82.0, 84.0, 86.0, 88.0])
    baseline = np.array([75.0, 77.0, 78.0, 80.0, 82.0])
    result = paired_comparison(proposed, baseline, bootstrap_seed=7)
    assert result.mean_difference > 0
    assert result.rank_biserial == pytest.approx(1.0)
    assert result.ci_low > 0
```

- [ ] **Step 2: Run tests and verify module import failure**

Run: `python -m pytest tests/test_statistics.py -v`

Expected: ERROR with `ModuleNotFoundError: No module named 'uav_mec.statistics'`.

- [ ] **Step 3: Implement the statistical functions**

Create `uav_mec/statistics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class PairedComparison:
    mean_difference: float
    ci_low: float
    ci_high: float
    wilcoxon_w: float
    p_raw: float
    rank_biserial: float


def paired_bootstrap_ci(
    differences: np.ndarray,
    resamples: int = 10_000,
    seed: int = 20260711,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("differences must be a finite vector with at least two values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_comparison(
    proposed: np.ndarray,
    baseline: np.ndarray,
    bootstrap_seed: int = 20260711,
) -> PairedComparison:
    left = np.asarray(proposed, dtype=float)
    right = np.asarray(baseline, dtype=float)
    if left.shape != right.shape:
        raise ValueError("paired samples must have the same shape")
    differences = left - right
    ci_low, ci_high = paired_bootstrap_ci(differences, 10_000, bootstrap_seed)
    if np.all(differences == 0):
        w_value, p_value = 0.0, 1.0
    else:
        test = stats.wilcoxon(left, right, alternative="two-sided", zero_method="wilcox")
        w_value, p_value = float(test.statistic), float(test.pvalue)
    return PairedComparison(
        float(differences.mean()),
        ci_low,
        ci_high,
        w_value,
        p_value,
        rank_biserial(differences),
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        if not 0.0 <= p_value <= 1.0:
            raise ValueError(f"invalid p-value for {name}: {p_value}")
        running = max(running, min((total - rank) * p_value, 1.0))
        adjusted[name] = running
    return adjusted
```

- [ ] **Step 4: Run statistics tests**

Run: `python -m pytest tests/test_statistics.py -v`

Expected: PASS, 3 tests passed.

- [ ] **Step 5: Commit**

```bash
git add uav_mec/statistics.py tests/test_statistics.py
git commit -m "feat: add correct paired thesis statistics"
```

---

### Task 15: Write Atomic Canonical Artifacts And Manifests

**Files:**
- Create: `uav_mec/artifacts.py`
- Create: `tests/test_artifacts.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing artifact tests**

```python
import json
from pathlib import Path

from uav_mec.artifacts import RunArtifactWriter, sha256_file


def test_writer_creates_manifest_data_and_checksums(tmp_path: Path):
    writer = RunArtifactWriter(tmp_path / "run")
    writer.write_json("summary.json", {"method": "full", "rate": 80.0})
    writer.write_csv("tasks.csv", [{"task_id": 1, "completed": True}])
    writer.write_jsonl("events.jsonl", [])
    writer.finalize({"git_sha": "abc123", "seeds": [42]})
    assert json.loads((tmp_path / "run/summary.json").read_text())["rate"] == 80.0
    assert (tmp_path / "run/tasks.csv").exists()
    assert (tmp_path / "run/events.jsonl").read_text(encoding="utf-8") == ""
    assert (tmp_path / "run/manifest.json").exists()
    checksums = (tmp_path / "run/checksums.sha256").read_text()
    assert sha256_file(tmp_path / "run/summary.json") in checksums


def test_writer_refuses_to_overwrite_existing_run(tmp_path: Path):
    path = tmp_path / "run"
    RunArtifactWriter(path)
    try:
        RunArtifactWriter(path)
    except FileExistsError:
        return
    raise AssertionError("writer overwrote an existing run directory")
```

- [ ] **Step 2: Run tests and verify missing artifact module**

Run: `python -m pytest tests/test_artifacts.py -v`

Expected: ERROR with `ModuleNotFoundError: No module named 'uav_mec.artifacts'`.

- [ ] **Step 3: Implement atomic JSON/CSV output and checksums**

Create `uav_mec/artifacts.py`:

```python
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Iterable, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)

    def _atomic_text(self, relative: str, text: str) -> None:
        destination = self.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def write_json(self, relative: str, value: object) -> None:
        self._atomic_text(relative, json.dumps(value, indent=2, sort_keys=True) + "\n")

    def write_jsonl(self, relative: str, rows: Iterable[Mapping[str, object]]) -> None:
        text = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
        self._atomic_text(relative, text)

    def write_csv(
        self,
        relative: str,
        rows: Iterable[Mapping[str, object]],
        fieldnames: Iterable[str] | None = None,
    ) -> None:
        materialized = list(rows)
        names = list(fieldnames) if fieldnames is not None else sorted(
            {key for row in materialized for key in row}
        )
        if not names:
            raise ValueError("fieldnames are required for an empty CSV")
        destination = self.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=names)
                writer.writeheader()
                writer.writerows(materialized)
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def finalize(self, manifest: Mapping[str, object]) -> None:
        self.write_json("manifest.json", dict(manifest))
        files = sorted(
            path for path in self.output_dir.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        )
        lines = [f"{sha256_file(path)}  {path.relative_to(self.output_dir)}" for path in files]
        self._atomic_text("checksums.sha256", "\n".join(lines) + "\n")
```

- [ ] **Step 4: Add precise artifact ignore rules**

Replace the blanket `results/` rule with rules that preserve legacy local results but allow canonical evidence:

```gitignore
results/
artifacts/runs/
artifacts/exploratory/
!artifacts/canonical/
!artifacts/canonical/**
```

Keep checkpoint/model ignore rules unchanged.

- [ ] **Step 5: Run artifact tests**

Run: `python -m pytest tests/test_artifacts.py -v`

Expected: PASS, 2 tests passed.

- [ ] **Step 6: Commit**

```bash
git add uav_mec/artifacts.py tests/test_artifacts.py .gitignore
git commit -m "feat: add auditable canonical artifacts"
```

---

### Task 16: Add A Fair OFDMA Baseline Through The Shared Outcome Builder

**Files:**
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/baselines_v2.py`
- Create: `tests/test_ofdma_fair.py`

- [ ] **Step 1: Write the failing Fair OFDMA contract test**

```python
import numpy as np

from uav_mec.baselines_v2 import evaluate_fair_ofdma
from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ChannelConfig, NOMAChannel


def test_fair_ofdma_uses_shared_task_and_cpu_semantics():
    tasks = tuple(Task(i, i, 0, 1e6, 1.3e9, 0.8, 1e9) for i in range(8))
    gains = np.array([
        [9e-8, 1e-8],
        [8e-8, 2e-8],
        [7e-8, 3e-8],
        [6e-8, 4e-8],
        [4e-8, 6e-8],
        [3e-8, 7e-8],
        [2e-8, 8e-8],
        [1e-8, 9e-8],
    ])
    snapshot = SlotSnapshot(0, tasks, gains, 2)
    cfg = EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)
    outcome = evaluate_fair_ofdma(snapshot, NOMAChannel(ChannelConfig(), 100.0), cfg)
    assert outcome.arrived_task_ids == tuple(range(8))
    assert len(outcome.decisions) == 8
    assert all(record.sic_feasible for record in outcome.rates)
    assert all(record.rate_bps > 0 for record in outcome.rates)
```

- [ ] **Step 2: Run the test and verify the missing OFDMA evaluator**

Run: `python -m pytest tests/test_ofdma_fair.py -v`

Expected: FAIL importing `evaluate_fair_ofdma`.

- [ ] **Step 3: Extract the shared outcome builder from NOMA evaluation**

Add `from collections.abc import Mapping` and import `serving_uav_by_task` from `uav_mec.schedule`. Move the allocation and task-outcome portion of `evaluate_schedule` into this function in `uav_mec/evaluation.py`:

```python
def build_outcome_from_rates(
    snapshot: SlotSnapshot,
    uav_by_task: Mapping[int, int],
    rates: tuple[RateRecord, ...],
    cfg: EvaluationConfig,
) -> SlotOutcome:
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    rate_by_id = {record.task_id: record.rate_bps for record in rates}
    if len(rate_by_id) != len(rates):
        raise ValueError("duplicate rate record")
    if set(rate_by_id) != set(uav_by_task):
        raise ValueError("rate records and serving-UAV mapping must have identical task IDs")
    if any(task_id not in task_by_id for task_id in uav_by_task):
        raise ValueError("serving-UAV mapping contains an unknown task")
    if any(not 0 <= serving_uav < snapshot.num_uavs for serving_uav in uav_by_task.values()):
        raise ValueError("serving-UAV mapping contains an invalid UAV")
    entries_by_uav: dict[int, list[int]] = defaultdict(list)
    for task_id, serving_uav in uav_by_task.items():
        entries_by_uav[serving_uav].append(task_id)
    decisions: dict[int, AllocationDecision] = {}
    for task in snapshot.tasks:
        if not task.requires_offload:
            decisions[task.task_id] = AllocationDecision(
                task.task_id,
                0.0,
                0.0,
                True,
                task.local_delay_s,
                task.deadline_s - task.local_delay_s,
                "local",
            )
    for uav in range(snapshot.num_uavs):
        scheduled_tasks = tuple(task_by_id[task_id] for task_id in entries_by_uav.get(uav, ()))
        allocation = allocate_with_policy(
            cfg.allocation_policy,
            scheduled_tasks,
            rate_by_id,
            cfg.cpu_per_uav_hz,
            cfg.result_ratio,
        )
        decisions.update({decision.task_id: decision for decision in allocation.decisions})
    for task in snapshot.tasks:
        if task.task_id not in decisions:
            decisions[task.task_id] = AllocationDecision(
                task.task_id,
                0.0,
                0.0,
                False,
                None,
                0.0,
                "unscheduled",
            )
    ordered = tuple(decisions[task_id] for task_id in sorted(decisions))
    return SlotOutcome(
        snapshot.slot_index,
        tuple(sorted(task_by_id)),
        tuple(decision.task_id for decision in ordered if decision.completed),
        tuple(decision.task_id for decision in ordered if not decision.completed),
        ordered,
        rates,
        sum(decision.slack_s for decision in ordered if decision.completed),
        sum(record.cross_interference_w for record in rates),
    )
```

Reduce `evaluate_schedule` to rate/SIC validation followed by:

```python
return build_outcome_from_rates(snapshot, serving_uav_by_task(schedule), rates, cfg)
```

- [ ] **Step 4: Implement equal-service-count TDMA/OFDMA rates**

Append to `uav_mec/baselines_v2.py`:

```python
import math
from collections import defaultdict

from .domain import RateRecord, SlotOutcome
from .evaluation import EvaluationConfig, build_outcome_from_rates
from .model import NOMAChannel


def evaluate_fair_ofdma(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> SlotOutcome:
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    selected: dict[int, list] = defaultdict(list)
    for uav in range(snapshot.num_uavs):
        candidates = [
            task for task in snapshot.tasks
            if task.requires_offload
            and int(np.argmax(snapshot.channel_gains[row_by_task[task.task_id]])) == uav
        ]
        candidates.sort(key=lambda task: (
            -snapshot.channel_gains[row_by_task[task.task_id], uav],
            task.task_id,
        ))
        selected[uav] = candidates[: 2 * cfg.n_subcarriers]

    assignments: list[tuple[int, int, int, int]] = []
    for uav, tasks in selected.items():
        for index, task in enumerate(tasks):
            subcarrier = index % cfg.n_subcarriers
            phase = index // cfg.n_subcarriers
            assignments.append((task.task_id, uav, subcarrier, phase))
    phases = 2 if any(len(tasks) > cfg.n_subcarriers for tasks in selected.values()) else 1
    active_phases = {
        task_id: tuple(range(phases)) if len(selected[uav]) <= cfg.n_subcarriers else (phase,)
        for task_id, uav, _subcarrier, phase in assignments
    }
    bandwidth = cfg.bandwidth_hz / cfg.n_subcarriers
    noise = channel.noise_for_bandwidth(bandwidth)
    rates = []
    for task_id, serving_uav, subcarrier, _phase in assignments:
        row = row_by_task[task_id]
        received = cfg.transmit_power_w * snapshot.channel_gains[row, serving_uav]
        phase_sinrs = []
        phase_cross = []
        for phase in active_phases[task_id]:
            cross = 0.0
            for other_id, other_uav, other_subcarrier, _other_phase in assignments:
                if (
                    other_uav != serving_uav
                    and other_subcarrier == subcarrier
                    and phase in active_phases[other_id]
                ):
                    other_row = row_by_task[other_id]
                    cross += cfg.transmit_power_w * snapshot.channel_gains[other_row, serving_uav]
            phase_cross.append(cross)
            phase_sinrs.append(received / (cross + noise))
        rate_bps = sum(bandwidth * math.log2(1.0 + sinr) for sinr in phase_sinrs) / phases
        rates.append(RateRecord(
            task_id,
            rate_bps,
            sum(phase_sinrs) / len(phase_sinrs),
            True,
            sum(phase_cross) / phases,
            serving_uav,
            subcarrier,
        ))
    uav_by_task = {task_id: uav for task_id, uav, _subcarrier, _phase in assignments}
    return build_outcome_from_rates(
        snapshot,
        uav_by_task,
        tuple(sorted(rates, key=lambda item: item.task_id)),
        cfg,
    )
```

This baseline serves up to `2*S` users per UAV, matching NOMA admission capacity, but two users on a subcarrier transmit in separate half-slots.

- [ ] **Step 5: Run OFDMA and evaluator tests**

Run: `python -m pytest tests/test_ofdma_fair.py tests/test_evaluation.py -v`

Expected: PASS, 4 tests passed.

- [ ] **Step 6: Commit**

```bash
git add uav_mec/evaluation.py uav_mec/baselines_v2.py tests/test_ofdma_fair.py
git commit -m "feat: add fair ofdma reference baseline"
```

---

### Task 17: Add The Config-Driven Experiment CLI

**Files:**
- Modify: `uav_mec/simulation.py`
- Create: `uav_mec/experiment.py`
- Create: `configs/thesis/smoke.toml`
- Create: `tests/test_experiment_cli.py`

- [ ] **Step 1: Write the failing end-to-end CLI test**

```python
import json
from pathlib import Path

from uav_mec.experiment import run_experiment


def test_smoke_experiment_writes_complete_bundle(tmp_path: Path):
    output = tmp_path / "smoke-run"
    run_experiment(Path("configs/thesis/smoke.toml"), output)
    assert (output / "manifest.json").exists()
    assert (output / "tasks.csv").exists()
    assert (output / "slots.csv").exists()
    assert (output / "events.jsonl").exists()
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["methods"]) == {"local_only", "greedy", "full"}
    assert summary["seeds"] == [42, 43]
    assert all(
        len(values["on_time_rate_by_seed"]) == 2
        for values in summary["methods"].values()
    )
```

- [ ] **Step 2: Run the test and verify missing experiment module/config**

Run: `python -m pytest tests/test_experiment_cli.py -v`

Expected: ERROR importing `uav_mec.experiment`.

- [ ] **Step 3: Add a generic outcome-method episode runner**

Append to `uav_mec/simulation.py`:

```python
OutcomeMethod = Callable[[SlotSnapshot], SlotOutcome]


def run_outcome_episode(
    trace: ExperimentTrace,
    method: OutcomeMethod,
) -> EpisodeResult:
    outcomes = []
    seen: set[int] = set()
    for snapshot in trace.slots:
        ids = {task.task_id for task in snapshot.tasks}
        if seen.intersection(ids):
            raise ValueError("trace repeats a task_id across slots")
        seen.update(ids)
        started = time.perf_counter()
        outcome = method(snapshot)
        if outcome.slot_index != snapshot.slot_index:
            raise ValueError("method returned an outcome for the wrong slot")
        outcomes.append(replace(outcome, runtime_s=time.perf_counter() - started))
    return EpisodeResult(trace.seed, tuple(outcomes))
```

Import `SlotOutcome` from `domain.py`.

- [ ] **Step 4: Create the exact smoke configuration**

Create `configs/thesis/smoke.toml`:

```toml
run_id = "smoke"
seeds = [42, 43]
methods = ["local_only", "greedy", "full"]

[scenario]
num_users = 6
num_slots = 3
num_uavs = 2
num_clusters = 3
n_subcarriers = 2
random_burst = true
user_mobility = true

[evaluation]
gamma_sic_min = 1.0
sic_residual = 0.02
allocation_policy = "deadline"

[search]
max_iterations = 20
```

- [ ] **Step 5: Implement config loading, method construction, and serialization**

Create `uav_mec/experiment.py` with these public functions. Search always uses deadline allocation; `full_fixed_half` and `full_crossing` only change final evaluation of the frozen full schedule.

```python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

import numpy as np
import scipy

from .artifacts import RunArtifactWriter
from .baselines_v2 import evaluate_fair_ofdma, greedy_schedule, local_only_schedule
from .domain import Schedule, SlotOutcome, SlotSnapshot
from .evaluation import EvaluationConfig, evaluate_schedule
from .model import ChannelConfig, NOMAChannel, ScenarioConfig
from .resource_allocation import AllocationPolicy
from .scheduler import (
    Acceptance,
    SearchConfig,
    build_initial_schedule,
    optimize_schedule,
    repair_sic,
)
from .simulation import run_outcome_episode
from .trace import generate_trace


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    run_id: str
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    scenario: ScenarioConfig
    evaluation: EvaluationConfig
    max_iterations: int


def _policy(value: str) -> AllocationPolicy:
    return AllocationPolicy(value)


def load_definition(path: Path) -> ExperimentDefinition:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    scenario = ScenarioConfig(**raw["scenario"])
    evaluation_raw = raw["evaluation"]
    evaluation = EvaluationConfig(
        n_subcarriers=scenario.n_subcarriers,
        bandwidth_hz=scenario.channel.bandwidth,
        transmit_power_w=scenario.channel.p_max,
        sic_residual=float(evaluation_raw["sic_residual"]),
        gamma_sic_min=float(evaluation_raw["gamma_sic_min"]),
        cpu_per_uav_hz=scenario.f_per_uav,
        result_ratio=scenario.task.result_ratio,
        allocation_policy=_policy(evaluation_raw["allocation_policy"]),
    )
    return ExperimentDefinition(
        str(raw["run_id"]),
        tuple(int(seed) for seed in raw["seeds"]),
        tuple(str(method) for method in raw["methods"]),
        scenario,
        evaluation,
        int(raw["search"]["max_iterations"]),
    )


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _slot_method(
    name: str,
    scenario: ScenarioConfig,
    target_evaluation: EvaluationConfig,
    max_iterations: int,
    event_sink: list[dict],
):
    channel = NOMAChannel(scenario.channel, scenario.uav.height)
    search_evaluation = replace(target_evaluation, allocation_policy=AllocationPolicy.DEADLINE)

    def evaluate_noma(snapshot: SlotSnapshot, schedule: Schedule) -> SlotOutcome:
        return evaluate_schedule(snapshot, schedule, channel, target_evaluation)

    def method(snapshot: SlotSnapshot) -> SlotOutcome:
        if name == "local_only":
            return evaluate_noma(snapshot, local_only_schedule(snapshot))
        if name == "fair_ofdma":
            return evaluate_fair_ofdma(snapshot, channel, target_evaluation)
        if name == "greedy":
            schedule = repair_sic(
                greedy_schedule(snapshot, scenario.n_subcarriers),
                snapshot,
                channel,
                search_evaluation,
            )
            return evaluate_noma(snapshot, schedule)
        initialization = build_initial_schedule(snapshot, channel, search_evaluation)
        event_sink.append({
            "slot": snapshot.slot_index,
            "move_type": "initialization",
            "accepted": True,
            "reason": initialization.solver_status,
            "before": None,
            "after": None,
            "schedule_key": initialization.schedule.stable_key(),
        })
        if name == "ilp":
            return evaluate_noma(snapshot, initialization.schedule)
        search_map = {
            "intra_bilateral": SearchConfig(False, Acceptance.BILATERAL, max_iterations),
            "cross_bilateral": SearchConfig(True, Acceptance.BILATERAL, max_iterations),
            "intra_global": SearchConfig(False, Acceptance.GLOBAL, max_iterations),
            "full": SearchConfig(True, Acceptance.GLOBAL, max_iterations),
            "full_fixed_half": SearchConfig(True, Acceptance.GLOBAL, max_iterations),
            "full_crossing": SearchConfig(True, Acceptance.GLOBAL, max_iterations),
        }
        if name not in search_map:
            raise ValueError(f"unknown method {name}")
        result = optimize_schedule(
            initialization.schedule,
            snapshot,
            channel,
            search_evaluation,
            search_map[name],
        )
        if not result.converged:
            raise RuntimeError(
                f"search hit max_iterations={max_iterations} at slot {snapshot.slot_index}"
            )
        event_sink.extend({
            "slot": snapshot.slot_index,
            "move_type": event.move_type,
            "accepted": event.accepted,
            "reason": event.reason,
            "before": event.before_objective,
            "after": event.after_objective,
            "schedule_key": event.schedule_key,
        } for event in result.events)
        event_sink.append({
            "slot": snapshot.slot_index,
            "move_type": "termination",
            "accepted": True,
            "reason": result.termination_reason,
            "before": None,
            "after": result.iterations,
            "schedule_key": result.schedule.stable_key(),
        })
        return evaluate_noma(snapshot, result.schedule)

    return method
```

- [ ] **Step 6: Implement experiment aggregation and canonical writing**

Append to `uav_mec/experiment.py`:

```python
def run_experiment(config_path: Path, output_dir: Path) -> None:
    definition = load_definition(config_path)
    writer = RunArtifactWriter(output_dir)
    task_rows = []
    slot_rows = []
    event_rows = []
    method_summary: dict[str, list[float]] = {name: [] for name in definition.methods}
    for seed in definition.seeds:
        trace = generate_trace(definition.scenario, seed)
        task_by_id = {
            task.task_id: task
            for snapshot in trace.slots
            for task in snapshot.tasks
        }
        for name in definition.methods:
            target = definition.evaluation
            if name == "full_fixed_half":
                target = replace(target, allocation_policy=AllocationPolicy.FIXED_HALF_EQUAL)
            elif name == "full_crossing":
                target = replace(target, allocation_policy=AllocationPolicy.CROSSING_EQUAL)
            events: list[dict] = []
            episode = run_outcome_episode(
                trace,
                _slot_method(name, definition.scenario, target, definition.max_iterations, events),
            )
            method_summary[name].append(episode.on_time_rate)
            for outcome in episode.outcomes:
                decision_by_id = {decision.task_id: decision for decision in outcome.decisions}
                rate_by_id = {record.task_id: record for record in outcome.rates}
                for task_id in outcome.arrived_task_ids:
                    task = task_by_id[task_id]
                    decision = decision_by_id[task_id]
                    rate = rate_by_id.get(task_id)
                    task_rows.append({
                        "method": name,
                        "seed": seed,
                        "slot": outcome.slot_index,
                        "task_id": task_id,
                        "user_id": task.user_id,
                        "arrival_slot": task.arrival_slot,
                        "data_bits": task.data_bits,
                        "comp_cycles": task.comp_cycles,
                        "deadline_s": task.deadline_s,
                        "local_cpu_hz": task.local_cpu_hz,
                        "completed": decision.completed,
                        "delay_s": decision.delay_s,
                        "alpha": decision.alpha,
                        "cpu_hz": decision.cpu_hz,
                        "rate_bps": None if rate is None else rate.rate_bps,
                        "sinr": None if rate is None else rate.sinr,
                        "sic_feasible": None if rate is None else rate.sic_feasible,
                        "cross_interference_w": None if rate is None else rate.cross_interference_w,
                        "serving_uav": None if rate is None else rate.serving_uav,
                        "subcarrier": None if rate is None else rate.subcarrier,
                        "reason": decision.reason,
                    })
                slot_rows.append({
                    "method": name,
                    "seed": seed,
                    "slot": outcome.slot_index,
                    "arrivals": len(outcome.arrived_task_ids),
                    "completed": outcome.completed_count,
                    "slack_s": outcome.total_slack_s,
                    "cross_interference_w": outcome.total_cross_interference_w,
                    "sic_violations": sum(not record.sic_feasible for record in outcome.rates),
                    "runtime_s": outcome.runtime_s,
                })
            event_rows.extend({"method": name, "seed": seed, **event} for event in events)
    summary = {
        "run_id": definition.run_id,
        "seeds": list(definition.seeds),
        "initializer_status_counts": {
            status: sum(
                row.get("move_type") == "initialization" and row.get("reason") == status
                for row in event_rows
            )
            for status in sorted({
                str(row["reason"])
                for row in event_rows
                if row.get("move_type") == "initialization"
            })
        },
        "search_termination_counts": {
            status: sum(
                row.get("move_type") == "termination" and row.get("reason") == status
                for row in event_rows
            )
            for status in sorted({
                str(row["reason"])
                for row in event_rows
                if row.get("move_type") == "termination"
            })
        },
        "methods": {
            name: {
                "on_time_rate_by_seed": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
            for name, values in method_summary.items()
        },
    }
    writer.write_json("summary.json", summary)
    writer.write_csv(
        "tasks.csv",
        task_rows,
        fieldnames=(
            "method", "seed", "slot", "task_id", "user_id", "arrival_slot",
            "data_bits", "comp_cycles", "deadline_s", "local_cpu_hz", "completed",
            "delay_s", "alpha", "cpu_hz", "rate_bps", "sinr", "sic_feasible",
            "cross_interference_w", "serving_uav", "subcarrier", "reason",
        ),
    )
    writer.write_csv(
        "slots.csv",
        slot_rows,
        fieldnames=(
            "method", "seed", "slot", "arrivals", "completed", "slack_s",
            "cross_interference_w", "sic_violations", "runtime_s",
        ),
    )
    writer.write_jsonl("events.jsonl", event_rows)
    writer.finalize({
        "run_id": definition.run_id,
        "git_sha": _git_sha(),
        "config_path": str(config_path),
        "config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
        "seeds": list(definition.seeds),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "solver": {"name": "scipy.optimize.milp/HiGHS", "package_version": scipy.__version__},
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_experiment(args.config, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the CLI test**

Run: `python -m pytest tests/test_experiment_cli.py -v`

Expected: PASS, 1 test passed and a complete bundle exists under pytest's temporary directory.

- [ ] **Step 8: Run the command-line smoke experiment**

Run: `python -m uav_mec.experiment --config configs/thesis/smoke.toml --output "$(mktemp -d)/run"`

Expected: exit 0; the fresh temporary `run` directory contains `manifest.json`, `summary.json`, `tasks.csv`, `slots.csv`, `events.jsonl`, and `checksums.sha256`.

- [ ] **Step 9: Commit**

```bash
git add uav_mec/simulation.py uav_mec/artifacts.py uav_mec/experiment.py configs/thesis/smoke.toml tests/test_experiment_cli.py
git commit -m "feat: add reproducible experiment cli"
```

---

### Task 18: Add Mechanism Controls And Strong Controlled Baselines

**Files:**
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/baselines_v2.py`
- Modify: `uav_mec/experiment.py`
- Create: `tests/test_mechanism_controls.py`
- Create: `tests/test_ia_greedy.py`

- [ ] **Step 1: Write failing cross-interference-scale test**

```python
import numpy as np

from uav_mec.domain import Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, compute_rate_records
from uav_mec.model import ChannelConfig, NOMAChannel


def test_zero_cross_scale_removes_only_cross_uav_interference():
    tasks = (
        Task(1, 0, 0, 1e6, 1.3e9, 0.8, 1e9),
        Task(2, 1, 0, 1e6, 1.3e9, 0.8, 1e9),
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[9e-8, 8e-8], [8e-8, 9e-8]]), 2)
    schedule = Schedule((
        ScheduleEntry(1, 0, 0, 0, "solo"),
        ScheduleEntry(2, 1, 1, 0, "solo"),
    ))
    base = dict(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=1.0,
        cpu_per_uav_hz=4e9,
        result_ratio=0.1,
    )
    channel = NOMAChannel(ChannelConfig(), 100.0)
    isolated = compute_rate_records(snapshot, schedule, channel, EvaluationConfig(**base, cross_interference_scale=0.0))
    coupled = compute_rate_records(snapshot, schedule, channel, EvaluationConfig(**base, cross_interference_scale=1.0))
    assert all(record.cross_interference_w == 0.0 for record in isolated)
    assert all(left.rate_bps > right.rate_bps for left, right in zip(isolated, coupled, strict=True))
```

- [ ] **Step 2: Write failing IA-Greedy feasibility test**

```python
import numpy as np

from uav_mec.baselines_v2 import ia_greedy_schedule
from uav_mec.domain import SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, evaluate_schedule
from uav_mec.model import ChannelConfig, NOMAChannel


def test_ia_greedy_returns_a_feasible_schedule_on_shared_spectrum():
    tasks = tuple(Task(i, i, 0, 1e6, 1.4e9, 0.8, 1e9) for i in range(6))
    gains = np.array([
        [9e-8, 2e-8], [7e-8, 3e-8], [5e-8, 4e-8],
        [4e-8, 5e-8], [3e-8, 7e-8], [2e-8, 9e-8],
    ])
    snapshot = SlotSnapshot(0, tasks, gains, 2)
    channel = NOMAChannel(ChannelConfig(), 100.0)
    cfg = EvaluationConfig(2, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)
    schedule = ia_greedy_schedule(snapshot, channel, cfg)
    outcome = evaluate_schedule(snapshot, schedule, channel, cfg)
    assert all(record.sic_feasible for record in outcome.rates)
```

- [ ] **Step 3: Run tests and verify missing controls/baseline**

Run: `python -m pytest tests/test_mechanism_controls.py tests/test_ia_greedy.py -v`

Expected: FAIL because `EvaluationConfig` lacks `cross_interference_scale` and `ia_greedy_schedule` is missing.

- [ ] **Step 4: Add the controlled interference scale**

Add this final defaulted field to `EvaluationConfig`:

```python
cross_interference_scale: float = 1.0
```

Add validation in `EvaluationConfig.__post_init__`:

```python
if not 0.0 <= self.cross_interference_scale <= 1.0:
    raise ValueError("cross_interference_scale must be in [0, 1]")
```

After accumulating raw `cross` in `compute_rate_records`, and inside each active-phase loop in `evaluate_fair_ofdma` before appending `phase_cross`, apply exactly:

```python
cross *= cfg.cross_interference_scale
```

- [ ] **Step 5: Implement IA-Greedy on the trusted evaluator**

Append to `uav_mec/baselines_v2.py`:

```python
from itertools import combinations

from .evaluation import InfeasibleScheduleError, evaluate_schedule


def ia_greedy_schedule(
    snapshot: SlotSnapshot,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
) -> Schedule:
    row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}
    task_by_id = {task.task_id: task for task in snapshot.tasks}
    current = Schedule()
    used: set[int] = set()
    for subcarrier in range(cfg.n_subcarriers):
        for uav in range(snapshot.num_uavs):
            candidates = [
                task for task in snapshot.tasks
                if task.requires_offload
                and task.task_id not in used
                and int(np.argmax(snapshot.channel_gains[row_by_task[task.task_id]])) == uav
            ]
            groups = [(task.task_id,) for task in candidates]
            groups.extend(tuple(task.task_id for task in pair) for pair in combinations(candidates, 2))
            feasible = []
            for task_ids in groups:
                additions = tuple(
                    ScheduleEntry(task_id, task_by_id[task_id].user_id, uav, subcarrier, "solo")
                    for task_id in task_ids
                )
                trial = normalize_roles(Schedule(current.entries + additions), snapshot)
                try:
                    outcome = evaluate_schedule(snapshot, trial, channel, cfg)
                except InfeasibleScheduleError:
                    continue
                feasible.append((
                    -outcome.completed_count,
                    outcome.total_cross_interference_w,
                    trial.stable_key(),
                    trial,
                ))
            if feasible:
                current = min(feasible)[-1]
                used = {entry.task_id for entry in current.entries}
    return current
```

- [ ] **Step 6: Register controlled baseline names in the CLI**

Import `ia_greedy_schedule` in `uav_mec/experiment.py`. Insert this branch immediately after the `greedy` branch in `_slot_method`:

```python
        if name == "ia_greedy":
            schedule = ia_greedy_schedule(snapshot, channel, search_evaluation)
            return evaluate_noma(snapshot, schedule)
```

Replace `search_map` with the complete mapping below. `intra_bilateral` remains an explicit ablation alias of the published `exchange_stable` baseline:

```python
        search_map = {
            "exchange_stable": SearchConfig(False, Acceptance.BILATERAL, max_iterations),
            "intra_bilateral": SearchConfig(False, Acceptance.BILATERAL, max_iterations),
            "cross_bilateral": SearchConfig(True, Acceptance.BILATERAL, max_iterations),
            "intra_global": SearchConfig(False, Acceptance.GLOBAL, max_iterations),
            "full": SearchConfig(True, Acceptance.GLOBAL, max_iterations),
            "full_fixed_half": SearchConfig(True, Acceptance.GLOBAL, max_iterations),
            "full_crossing": SearchConfig(True, Acceptance.GLOBAL, max_iterations),
        }
```

Add this keyword when `load_definition` constructs `EvaluationConfig`, so frozen mechanism suites can select the negative control:

```python
        cross_interference_scale=float(evaluation_raw.get("cross_interference_scale", 1.0)),
```

Append to `tests/test_ia_greedy.py`:

```python
from uav_mec.experiment import _slot_method, load_definition
from uav_mec.trace import generate_trace


def test_registered_controlled_baselines_execute_on_shared_trace():
    definition = load_definition(Path("configs/thesis/smoke.toml"))
    snapshot = generate_trace(definition.scenario, definition.seeds[0]).slots[0]
    for name in ("ia_greedy", "exchange_stable", "intra_bilateral"):
        outcome = _slot_method(
            name,
            definition.scenario,
            definition.evaluation,
            definition.max_iterations,
            [],
        )(snapshot)
        assert outcome.arrived_task_ids == tuple(sorted(task.task_id for task in snapshot.tasks))
```

Also add `from pathlib import Path` to the test imports.

- [ ] **Step 7: Run mechanism and baseline tests**

Run: `python -m pytest tests/test_mechanism_controls.py tests/test_ia_greedy.py tests/test_experiment_cli.py -v`

Expected: PASS, 4 tests passed.

- [ ] **Step 8: Commit**

```bash
git add uav_mec/evaluation.py uav_mec/baselines_v2.py uav_mec/experiment.py tests/test_mechanism_controls.py tests/test_ia_greedy.py
git commit -m "feat: add interference controls and strong baselines"
```

---

### Task 19: Separate Estimated CSI From True Evaluation Channels

**Files:**
- Modify: `uav_mec/domain.py`
- Modify: `uav_mec/trace.py`
- Modify: `uav_mec/scheduler.py`
- Modify: `uav_mec/baselines_v2.py`
- Modify: `uav_mec/experiment.py`
- Create: `tests/test_csi_error.py`

- [ ] **Step 1: Write failing true/estimated-channel tests**

```python
import numpy as np

from uav_mec.domain import SlotSnapshot, Task
from uav_mec.model import ScenarioConfig
from uav_mec.scheduler import _nearest_uav
from uav_mec.trace import generate_trace


def test_scheduler_uses_estimated_csi_for_nearest_uav():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(
        0,
        (task,),
        np.array([[9e-8, 1e-8]]),
        2,
        estimated_channel_gains=np.array([[1e-8, 9e-8]]),
    )
    assert _nearest_uav(snapshot, 1) == 1
    assert snapshot.channel_gains[0, 0] > snapshot.channel_gains[0, 1]


def test_csi_error_is_deterministic_and_does_not_change_true_gains():
    cfg = ScenarioConfig(num_users=6, num_slots=2, random_burst=True)
    perfect = generate_trace(cfg, 99, csi_error_std=0.0)
    noisy_a = generate_trace(cfg, 99, csi_error_std=0.1)
    noisy_b = generate_trace(cfg, 99, csi_error_std=0.1)
    for clean, left, right in zip(perfect.slots, noisy_a.slots, noisy_b.slots, strict=True):
        np.testing.assert_array_equal(clean.channel_gains, left.channel_gains)
        np.testing.assert_array_equal(left.estimated_channel_gains, right.estimated_channel_gains)
```

- [ ] **Step 2: Run tests and verify the missing estimated-channel API**

Run: `python -m pytest tests/test_csi_error.py -v`

Expected: FAIL because `SlotSnapshot` does not accept `estimated_channel_gains` and `generate_trace` lacks `csi_error_std`.

- [ ] **Step 3: Extend `SlotSnapshot` without changing evaluation truth**

Add to `SlotSnapshot` after `num_uavs`:

```python
estimated_channel_gains: np.ndarray | None = None
```

Extend `__post_init__`:

```python
if self.estimated_channel_gains is not None:
    estimated = np.array(self.estimated_channel_gains, dtype=float, copy=True)
    if estimated.shape != expected:
        raise ValueError(f"estimated_channel_gains must have shape {expected}")
    if not np.isfinite(estimated).all() or (estimated < 0).any():
        raise ValueError("estimated_channel_gains must be finite and non-negative")
    estimated.setflags(write=False)
    object.__setattr__(self, "estimated_channel_gains", estimated)
```

Add this property:

```python
@property
def planning_channel_gains(self) -> np.ndarray:
    return self.channel_gains if self.estimated_channel_gains is None else self.estimated_channel_gains

def as_planning_snapshot(self) -> "SlotSnapshot":
    return SlotSnapshot(
        self.slot_index,
        self.tasks,
        self.planning_channel_gains,
        self.num_uavs,
    )
```

Replace `_nearest_uav` in `uav_mec/scheduler.py` with:

```python
def _nearest_uav(snapshot: SlotSnapshot, task_id: int) -> int:
    row = next(row for row, task in enumerate(snapshot.tasks) if task.task_id == task_id)
    return int(np.argmax(snapshot.planning_channel_gains[row]))
```

In `greedy_schedule`, `ia_greedy_schedule`, and the candidate ordering portion of `evaluate_fair_ofdma`, bind and use the planning matrix exactly as follows:

```python
planning_gains = snapshot.planning_channel_gains
row_by_task = {task.task_id: row for row, task in enumerate(snapshot.tasks)}

candidates = [
    task for task in snapshot.tasks
    if task.requires_offload
    and int(np.argmax(planning_gains[row_by_task[task.task_id]])) == uav
]
candidates.sort(key=lambda task: (
    -planning_gains[row_by_task[task.task_id], uav],
    task.task_id,
))
```

Preserve each function's existing `used` predicate where applicable. Keep the Fair OFDMA received-power and interference calculations on `snapshot.channel_gains`, because those are final true-channel rates. `evaluation.py` must continue to read only `snapshot.channel_gains`.

- [ ] **Step 4: Generate deterministic multiplicative CSI estimates**

Change the trace signature to:

```python
def generate_trace(
    cfg: ScenarioConfig,
    seed: int,
    csi_error_std: float = 0.0,
) -> ExperimentTrace:
```

Add this exact validation at the top of `generate_trace`:

```python
if csi_error_std < 0:
    raise ValueError("csi_error_std must be non-negative")
```

After true gains are built, create estimates with a separate RNG so adding CSI error does not perturb task/mobility generation:

```python
csi_rng = np.random.default_rng(seed + 7_919)
estimated = None
if csi_error_std > 0:
    relative_error = csi_rng.normal(0.0, csi_error_std, gains.shape)
    estimated = np.maximum(gains * (1.0 + relative_error), 0.0)
slots.append(SlotSnapshot(slot, tuple(tasks), gains, cfg.num_uavs, estimated))
```

Initialize `csi_rng` once before the slot loop. Remove the previous four-argument `SlotSnapshot` append.

- [ ] **Step 5: Add CSI error to experiment definitions and manifests**

Replace the definition record with:

```python
@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    run_id: str
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    scenario: ScenarioConfig
    evaluation: EvaluationConfig
    max_iterations: int
    csi_error_std: float = 0.0
```

In `load_definition`, read and validate the trace value:

```python
csi_error_std = float(raw.get("trace", {}).get("csi_error_std", 0.0))
if csi_error_std < 0:
    raise ValueError("csi_error_std must be non-negative")
```

Construct the definition with keywords so the new field cannot be shifted positionally:

```python
return ExperimentDefinition(
    run_id=str(raw["run_id"]),
    seeds=tuple(int(seed) for seed in raw["seeds"]),
    methods=tuple(str(method) for method in raw["methods"]),
    scenario=scenario,
    evaluation=evaluation,
    max_iterations=int(raw["search"]["max_iterations"]),
    csi_error_std=csi_error_std,
)
```

Task 20 will replace the direct `raw["seeds"]` expression with the already specified compact-range parser.

Replace `_slot_method` with this full version. Every planning decision sees `snapshot.as_planning_snapshot()`; only the final method return is evaluated against the original true snapshot:

```python
def _slot_method(
    name: str,
    scenario: ScenarioConfig,
    target_evaluation: EvaluationConfig,
    max_iterations: int,
    event_sink: list[dict],
):
    channel = NOMAChannel(scenario.channel, scenario.uav.height)
    search_evaluation = replace(target_evaluation, allocation_policy=AllocationPolicy.DEADLINE)

    def evaluate_noma(snapshot: SlotSnapshot, schedule: Schedule) -> SlotOutcome:
        return evaluate_schedule(snapshot, schedule, channel, target_evaluation)

    def method(snapshot: SlotSnapshot) -> SlotOutcome:
        planning = snapshot.as_planning_snapshot()
        if name == "local_only":
            return evaluate_noma(snapshot, local_only_schedule(planning))
        if name == "fair_ofdma":
            return evaluate_fair_ofdma(snapshot, channel, target_evaluation)
        if name == "greedy":
            schedule = repair_sic(
                greedy_schedule(planning, scenario.n_subcarriers),
                planning,
                channel,
                search_evaluation,
            )
            return evaluate_noma(snapshot, schedule)
        if name == "ia_greedy":
            schedule = ia_greedy_schedule(planning, channel, search_evaluation)
            return evaluate_noma(snapshot, schedule)
        initialization = build_initial_schedule(planning, channel, search_evaluation)
        event_sink.append({
            "slot": snapshot.slot_index,
            "move_type": "initialization",
            "accepted": True,
            "reason": initialization.solver_status,
            "before": None,
            "after": None,
            "schedule_key": initialization.schedule.stable_key(),
        })
        if name == "ilp":
            return evaluate_noma(snapshot, initialization.schedule)
        search_map = {
            "exchange_stable": SearchConfig(False, Acceptance.BILATERAL, max_iterations),
            "intra_bilateral": SearchConfig(False, Acceptance.BILATERAL, max_iterations),
            "cross_bilateral": SearchConfig(True, Acceptance.BILATERAL, max_iterations),
            "intra_global": SearchConfig(False, Acceptance.GLOBAL, max_iterations),
            "full": SearchConfig(True, Acceptance.GLOBAL, max_iterations),
            "full_fixed_half": SearchConfig(True, Acceptance.GLOBAL, max_iterations),
            "full_crossing": SearchConfig(True, Acceptance.GLOBAL, max_iterations),
        }
        if name not in search_map:
            raise ValueError(f"unknown method {name}")
        result = optimize_schedule(
            initialization.schedule,
            planning,
            channel,
            search_evaluation,
            search_map[name],
        )
        if not result.converged:
            raise RuntimeError(
                f"search hit max_iterations={max_iterations} at slot {snapshot.slot_index}"
            )
        event_sink.extend({
            "slot": snapshot.slot_index,
            "move_type": event.move_type,
            "accepted": event.accepted,
            "reason": event.reason,
            "before": event.before_objective,
            "after": event.after_objective,
            "schedule_key": event.schedule_key,
        } for event in result.events)
        event_sink.append({
            "slot": snapshot.slot_index,
            "move_type": "termination",
            "accepted": True,
            "reason": result.termination_reason,
            "before": None,
            "after": result.iterations,
            "schedule_key": result.schedule.stable_key(),
        })
        return evaluate_noma(snapshot, result.schedule)

    return method
```

Change experiment trace generation to:

```python
trace = generate_trace(definition.scenario, seed, definition.csi_error_std)
```

Add `"csi_error_std": definition.csi_error_std` to the Task 17 manifest; Task 20's resolved-definition manifest supersedes this scalar while retaining it.

Append this true-evaluation regression to `tests/test_csi_error.py` with these imports:

```python
from uav_mec.evaluation import EvaluationConfig
from uav_mec.experiment import _slot_method
from uav_mec.model import ChannelConfig, NOMAChannel
```

```python
def test_final_rate_uses_true_channel_after_estimated_csi_planning():
    task = Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9)
    snapshot = SlotSnapshot(
        0,
        (task,),
        np.array([[9e-8, 1e-8]]),
        2,
        estimated_channel_gains=np.array([[1e-8, 9e-8]]),
    )
    scenario = ScenarioConfig(num_users=1, num_slots=1, num_uavs=2, n_subcarriers=1)
    cfg = EvaluationConfig(1, 5e6, 0.2, 0.02, 1.0, 4e9, 0.1)
    outcome = _slot_method("greedy", scenario, cfg, 5, [])(snapshot)
    record = outcome.rates[0]
    channel = NOMAChannel(ChannelConfig(), scenario.uav.height)
    noise = channel.noise_for_bandwidth(cfg.bandwidth_hz)
    counterfactual_nearest_rate = cfg.bandwidth_hz * np.log2(1.0 + cfg.transmit_power_w * 9e-8 / noise)
    assert record.serving_uav == 1
    assert record.rate_bps < counterfactual_nearest_rate
```

- [ ] **Step 6: Run CSI and regression tests**

Run: `python -m pytest tests/test_csi_error.py tests/test_trace.py tests/test_evaluation.py tests/test_initializer.py -v`

Expected: PASS; scheduler choice changes under inverted estimated CSI while rate evaluation remains tied to true gains.

- [ ] **Step 7: Commit**

```bash
git add uav_mec/domain.py uav_mec/trace.py uav_mec/scheduler.py uav_mec/baselines_v2.py uav_mec/experiment.py tests/test_csi_error.py
git commit -m "feat: evaluate robustness with separate estimated csi"
```

---

### Task 20: Define Frozen Exploratory And Confirmatory Experiment Suites

**Files:**
- Modify: `uav_mec/experiment.py`
- Create: `uav_mec/run_thesis_suite.py`
- Create: `configs/thesis/exploratory.toml`
- Create: `configs/thesis/confirmatory_main.toml`
- Create: `configs/thesis/confirmatory_pairing.toml`
- Create: `configs/thesis/confirmatory_resource.toml`
- Create: `tests/test_thesis_suites.py`

- [ ] **Step 1: Write failing suite-definition tests**

```python
from dataclasses import replace
from pathlib import Path

from uav_mec.experiment import _slot_method, load_definition
from uav_mec.resource_allocation import AllocationPolicy
from uav_mec.run_thesis_suite import build_suite
from uav_mec.trace import generate_trace


def test_confirmatory_seed_range_is_frozen_and_unseen():
    definition = load_definition(Path("configs/thesis/confirmatory_main.toml"))
    assert definition.seeds == tuple(range(1000, 1050))


def test_pairing_ablation_changes_only_topology_and_acceptance():
    definition = load_definition(Path("configs/thesis/confirmatory_pairing.toml"))
    assert definition.methods == (
        "intra_bilateral",
        "cross_bilateral",
        "intra_global",
        "full",
    )


def test_resource_ablation_freezes_schedule_search_policy():
    definition = load_definition(Path("configs/thesis/confirmatory_resource.toml"))
    assert definition.methods == ("full_fixed_half", "full_crossing", "full")
    snapshot = generate_trace(definition.scenario, definition.seeds[0]).slots[0]
    signatures = []
    for method in definition.methods:
        target = definition.evaluation
        if method == "full_fixed_half":
            target = replace(target, allocation_policy=AllocationPolicy.FIXED_HALF_EQUAL)
        elif method == "full_crossing":
            target = replace(target, allocation_policy=AllocationPolicy.CROSSING_EQUAL)
        outcome = _slot_method(
            method,
            definition.scenario,
            target,
            definition.max_iterations,
            [],
        )(snapshot)
        signatures.append(tuple(
            (record.task_id, record.serving_uav, record.subcarrier)
            for record in outcome.rates
        ))
    assert len(set(signatures)) == 1


def test_scalability_and_robustness_cases_are_complete():
    scalability = build_suite("scalability")
    robustness = build_suite("robustness")
    assert {case.name for case in scalability} == {
        "n10", "n15", "n20", "n25", "n30", "n40", "n50",
        "k1", "k2", "k3", "k4", "k5",
    }
    assert {"nf3", "nf7", "nf10", "csi0", "csi05", "csi10", "csi20"}.issubset(
        {case.name for case in robustness}
    )
```

- [ ] **Step 2: Run tests and verify missing configs/suite module**

Run: `python -m pytest tests/test_thesis_suites.py -v`

Expected: ERROR importing `uav_mec.run_thesis_suite` or loading missing TOML files.

- [ ] **Step 3: Support compact frozen seed ranges**

In `load_definition`, replace direct seed parsing with:

```python
if "seeds" in raw:
    seeds = tuple(int(seed) for seed in raw["seeds"])
else:
    seed_start = int(raw["seed_start"])
    seed_count = int(raw["seed_count"])
    seeds = tuple(range(seed_start, seed_start + seed_count))
if not seeds or len(set(seeds)) != len(seeds):
    raise ValueError("seeds must be non-empty and unique")
```

Use `seeds` when constructing `ExperimentDefinition`.

- [ ] **Step 4: Create the exploratory and main confirmatory configs**

All four files use this scenario/evaluation/search body:

```toml
[scenario]
num_users = 20
num_slots = 100
num_uavs = 2
num_clusters = 3
n_subcarriers = 3
fixed_cluster_centers = true
random_burst = true
user_mobility = true

[evaluation]
gamma_sic_min = 1.0
sic_residual = 0.02
allocation_policy = "deadline"
cross_interference_scale = 1.0

[trace]
csi_error_std = 0.0

[search]
max_iterations = 50
```

Prefix `configs/thesis/exploratory.toml` with:

```toml
run_id = "exploratory"
seed_start = 42
seed_count = 20
methods = ["local_only", "greedy", "ia_greedy", "ilp", "exchange_stable", "intra_global", "full", "fair_ofdma"]
```

Prefix `configs/thesis/confirmatory_main.toml` with:

```toml
run_id = "confirmatory-main"
seed_start = 1000
seed_count = 50
methods = ["local_only", "greedy", "ia_greedy", "ilp", "exchange_stable", "intra_global", "full", "fair_ofdma"]
```

Prefix `configs/thesis/confirmatory_pairing.toml` with:

```toml
run_id = "confirmatory-pairing"
seed_start = 1000
seed_count = 50
methods = ["intra_bilateral", "cross_bilateral", "intra_global", "full"]
```

Prefix `configs/thesis/confirmatory_resource.toml` with:

```toml
run_id = "confirmatory-resource"
seed_start = 1000
seed_count = 50
methods = ["full_fixed_half", "full_crossing", "full"]
```

- [ ] **Step 5: Expose definition execution separately from TOML loading**

Add `from enum import Enum`. Then replace `run_experiment` with the following complete serialization and execution functions:

```python
def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def definition_to_dict(definition: ExperimentDefinition) -> dict[str, object]:
    value = _jsonable(asdict(definition))
    if not isinstance(value, dict):
        raise TypeError("serialized experiment definition must be a mapping")
    return value


def run_definition(
    definition: ExperimentDefinition,
    config_path: Path,
    output_dir: Path,
) -> None:
    writer = RunArtifactWriter(output_dir)
    task_rows: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    method_summary: dict[str, list[float]] = {name: [] for name in definition.methods}
    for seed in definition.seeds:
        trace = generate_trace(definition.scenario, seed, definition.csi_error_std)
        task_by_id = {
            task.task_id: task
            for snapshot in trace.slots
            for task in snapshot.tasks
        }
        for name in definition.methods:
            target = definition.evaluation
            if name == "full_fixed_half":
                target = replace(target, allocation_policy=AllocationPolicy.FIXED_HALF_EQUAL)
            elif name == "full_crossing":
                target = replace(target, allocation_policy=AllocationPolicy.CROSSING_EQUAL)
            events: list[dict] = []
            episode = run_outcome_episode(
                trace,
                _slot_method(name, definition.scenario, target, definition.max_iterations, events),
            )
            method_summary[name].append(episode.on_time_rate)
            for outcome in episode.outcomes:
                decision_by_id = {decision.task_id: decision for decision in outcome.decisions}
                rate_by_id = {record.task_id: record for record in outcome.rates}
                for task_id in outcome.arrived_task_ids:
                    task = task_by_id[task_id]
                    decision = decision_by_id[task_id]
                    rate = rate_by_id.get(task_id)
                    task_rows.append({
                        "method": name,
                        "seed": seed,
                        "slot": outcome.slot_index,
                        "task_id": task_id,
                        "user_id": task.user_id,
                        "arrival_slot": task.arrival_slot,
                        "data_bits": task.data_bits,
                        "comp_cycles": task.comp_cycles,
                        "deadline_s": task.deadline_s,
                        "local_cpu_hz": task.local_cpu_hz,
                        "completed": decision.completed,
                        "delay_s": decision.delay_s,
                        "alpha": decision.alpha,
                        "cpu_hz": decision.cpu_hz,
                        "rate_bps": None if rate is None else rate.rate_bps,
                        "sinr": None if rate is None else rate.sinr,
                        "sic_feasible": None if rate is None else rate.sic_feasible,
                        "cross_interference_w": None if rate is None else rate.cross_interference_w,
                        "serving_uav": None if rate is None else rate.serving_uav,
                        "subcarrier": None if rate is None else rate.subcarrier,
                        "reason": decision.reason,
                    })
                slot_rows.append({
                    "method": name,
                    "seed": seed,
                    "slot": outcome.slot_index,
                    "arrivals": len(outcome.arrived_task_ids),
                    "completed": outcome.completed_count,
                    "slack_s": outcome.total_slack_s,
                    "cross_interference_w": outcome.total_cross_interference_w,
                    "sic_violations": sum(not record.sic_feasible for record in outcome.rates),
                    "runtime_s": outcome.runtime_s,
                })
            event_rows.extend({"method": name, "seed": seed, **event} for event in events)

    summary = {
        "run_id": definition.run_id,
        "seeds": list(definition.seeds),
        "initializer_status_counts": {
            status: sum(
                row.get("move_type") == "initialization" and row.get("reason") == status
                for row in event_rows
            )
            for status in sorted({
                str(row["reason"])
                for row in event_rows
                if row.get("move_type") == "initialization"
            })
        },
        "search_termination_counts": {
            status: sum(
                row.get("move_type") == "termination" and row.get("reason") == status
                for row in event_rows
            )
            for status in sorted({
                str(row["reason"])
                for row in event_rows
                if row.get("move_type") == "termination"
            })
        },
        "methods": {
            name: {
                "on_time_rate_by_seed": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
            for name, values in method_summary.items()
        },
    }
    writer.write_json("summary.json", summary)
    writer.write_csv(
        "tasks.csv",
        task_rows,
        fieldnames=(
            "method", "seed", "slot", "task_id", "user_id", "arrival_slot",
            "data_bits", "comp_cycles", "deadline_s", "local_cpu_hz", "completed",
            "delay_s", "alpha", "cpu_hz", "rate_bps", "sinr", "sic_feasible",
            "cross_interference_w", "serving_uav", "subcarrier", "reason",
        ),
    )
    writer.write_csv(
        "slots.csv",
        slot_rows,
        fieldnames=(
            "method", "seed", "slot", "arrivals", "completed", "slack_s",
            "cross_interference_w", "sic_violations", "runtime_s",
        ),
    )
    writer.write_jsonl("events.jsonl", event_rows)
    writer.finalize({
        "run_id": definition.run_id,
        "git_sha": _git_sha(),
        "config_path": str(config_path),
        "config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
        "resolved_definition": definition_to_dict(definition),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "solver": {"name": "scipy.optimize.milp/HiGHS", "package_version": scipy.__version__},
    })


def run_experiment(config_path: Path, output_dir: Path) -> None:
    run_definition(load_definition(config_path), config_path, output_dir)
```

- [ ] **Step 6: Implement exact suite case construction**

Create `uav_mec/run_thesis_suite.py`:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

from .experiment import ExperimentDefinition, load_definition, run_definition


@dataclass(frozen=True, slots=True)
class SuiteCase:
    name: str
    definition: ExperimentDefinition


BASE_PATH = Path("configs/thesis/confirmatory_main.toml")
SWEEP_METHODS = ("greedy", "exchange_stable", "full")


def _with_scenario(base: ExperimentDefinition, name: str, **changes) -> SuiteCase:
    scenario = replace(base.scenario, **changes)
    evaluation = replace(
        base.evaluation,
        n_subcarriers=scenario.n_subcarriers,
        bandwidth_hz=scenario.channel.bandwidth,
        transmit_power_w=scenario.channel.p_max,
        cpu_per_uav_hz=scenario.f_per_uav,
        result_ratio=scenario.task.result_ratio,
    )
    return SuiteCase(name, replace(base, run_id=name, methods=SWEEP_METHODS, scenario=scenario, evaluation=evaluation))


def build_suite(name: str) -> tuple[SuiteCase, ...]:
    base = load_definition(BASE_PATH)
    if name == "mechanism":
        return tuple(
            SuiteCase(
                case_name,
                replace(
                    base,
                    run_id=case_name,
                    methods=SWEEP_METHODS,
                    evaluation=replace(base.evaluation, cross_interference_scale=scale),
                ),
            )
            for case_name, scale in (("coupling0", 0.0), ("coupling05", 0.5), ("coupling1", 1.0))
        )
    if name == "scalability":
        users = tuple(_with_scenario(base, f"n{count}", num_users=count) for count in (10, 15, 20, 25, 30, 40, 50))
        uavs = tuple(
            _with_scenario(
                base,
                f"k{count}",
                num_uavs=count,
                num_clusters=3 if count <= 3 else count,
            )
            for count in (1, 2, 3, 4, 5)
        )
        return users + uavs
    if name == "robustness":
        cases: list[SuiteCase] = []
        for nf in (3.0, 7.0, 10.0):
            channel = replace(base.scenario.channel, noise_figure_db=nf)
            cases.append(_with_scenario(base, f"nf{int(nf)}", channel=channel))
        for label, error in (("csi0", 0.0), ("csi05", 0.05), ("csi10", 0.10), ("csi20", 0.20)):
            cases.append(SuiteCase(label, replace(base, run_id=label, methods=SWEEP_METHODS, csi_error_std=error)))
        for label, residual in (("eps0", 0.0), ("eps02", 0.02), ("eps05", 0.05), ("eps10", 0.10)):
            cases.append(SuiteCase(label, replace(
                base,
                run_id=label,
                methods=SWEEP_METHODS,
                evaluation=replace(base.evaluation, sic_residual=residual),
            )))
        for label, threshold in (("sicm3", 10 ** -0.3), ("sic0", 1.0), ("sic3", 10 ** 0.3), ("sic5", 10 ** 0.5)):
            cases.append(SuiteCase(label, replace(
                base,
                run_id=label,
                methods=SWEEP_METHODS,
                evaluation=replace(base.evaluation, gamma_sic_min=threshold),
            )))
        for label, cpu in (("cpu2", 2e9), ("cpu4", 4e9), ("cpu8", 8e9)):
            cases.append(_with_scenario(base, label, f_per_uav=cpu))
        for label, deadline in (("t06", 0.6), ("t08", 0.8), ("t10", 1.0)):
            cases.append(_with_scenario(base, label, task=replace(base.scenario.task, t_max=deadline)))
        cases.extend((
            _with_scenario(base, "load_light", task=replace(
                base.scenario.task,
                data_min=0.2e6,
                data_max=1.0e6,
                comp_min=300.0,
                comp_max=600.0,
            )),
            _with_scenario(base, "load_default"),
            _with_scenario(base, "load_heavy", task=replace(
                base.scenario.task,
                data_min=0.8e6,
                data_max=3.0e6,
                comp_min=800.0,
                comp_max=1500.0,
            )),
        ))
        return tuple(cases)
    raise ValueError(f"unknown suite {name}")


def run_suite(name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for case in build_suite(name):
        run_definition(case.definition, BASE_PATH, output_dir / case.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("mechanism", "scalability", "robustness"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_suite(args.suite, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run suite-definition tests**

Run: `python -m pytest tests/test_thesis_suites.py -v`

Expected: PASS, 4 tests passed.

- [ ] **Step 8: Run a two-seed suite smoke test without changing frozen configs**

Extend the imports in `tests/test_thesis_suites.py`:

```python
import json

import pytest

from uav_mec.experiment import run_definition
from uav_mec.run_thesis_suite import BASE_PATH
```

Append this integration test:

```python
@pytest.mark.parametrize(
    ("suite_name", "value_path", "expected"),
    (
        ("mechanism", ("evaluation", "cross_interference_scale"), 0.0),
        ("scalability", ("scenario", "num_users"), 10),
        ("robustness", ("scenario", "channel", "noise_figure_db"), 3.0),
    ),
)
def test_suite_case_executes_with_resolved_manifest(
    tmp_path: Path,
    suite_name: str,
    value_path: tuple[str, ...],
    expected: object,
):
    case = build_suite(suite_name)[0]
    definition = replace(
        case.definition,
        seeds=(42, 43),
        scenario=replace(case.definition.scenario, num_slots=2),
    )
    output = tmp_path / suite_name
    run_definition(definition, BASE_PATH, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    resolved = manifest["resolved_definition"]
    assert resolved["seeds"] == [42, 43]
    assert resolved["scenario"]["num_slots"] == 2
    value = resolved
    for key in value_path:
        value = value[key]
    assert value == expected
```

- [ ] **Step 9: Run suite integration tests**

Run: `python -m pytest tests/test_thesis_suites.py -v`

Expected: PASS, 7 tests passed and all three temporary bundles contain resolved manifests.

- [ ] **Step 10: Commit**

```bash
git add uav_mec/experiment.py uav_mec/run_thesis_suite.py configs/thesis tests/test_thesis_suites.py
git commit -m "feat: freeze thesis experiment suites"
```

---

### Task 21: Produce Complete-Enumeration Optimality-Gap Evidence

**Files:**
- Create: `uav_mec/run_exact_benchmark.py`
- Create: `configs/thesis/confirmatory_exact.toml`
- Create: `tests/test_exact_benchmark.py`

- [ ] **Step 1: Write the failing exact-benchmark integration test**

```python
import csv
import json
from dataclasses import replace
from pathlib import Path

from uav_mec.experiment import load_definition
from uav_mec.run_exact_benchmark import run_exact_definition


def test_exact_benchmark_never_reports_heuristic_above_optimum(tmp_path: Path):
    base = load_definition(Path("configs/thesis/confirmatory_exact.toml"))
    definition = replace(
        base,
        seeds=(42,),
        scenario=replace(base.scenario, num_slots=2, num_users=4),
    )
    output = tmp_path / "exact"
    run_exact_definition(
        definition,
        Path("configs/thesis/confirmatory_exact.toml"),
        output,
    )
    with (output / "exact_slots.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["complete_search"] == "True" for row in rows)
    assert all(int(row["full_completed"]) <= int(row["exact_completed"]) for row in rows)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert 0.0 <= summary["optimal_slot_fraction"] <= 1.0
    assert summary["mean_completion_gap"] >= 0.0
```

- [ ] **Step 2: Run the test and verify the missing runner/config failure**

Run: `python -m pytest tests/test_exact_benchmark.py -v`

Expected: ERROR importing `uav_mec.run_exact_benchmark` or loading the missing config.

- [ ] **Step 3: Freeze the small-instance confirmatory config**

Create `configs/thesis/confirmatory_exact.toml`:

```toml
run_id = "confirmatory-exact"
seed_start = 1000
seed_count = 50
methods = ["full", "exact"]

[scenario]
num_users = 5
num_slots = 10
num_uavs = 2
num_clusters = 3
n_subcarriers = 2
fixed_cluster_centers = true
random_burst = true
user_mobility = true

[evaluation]
gamma_sic_min = 1.0
sic_residual = 0.02
allocation_policy = "deadline"
cross_interference_scale = 1.0

[trace]
csi_error_std = 0.0

[search]
max_iterations = 50
```

- [ ] **Step 4: Implement the complete-enumeration comparison runner**

Create `uav_mec/run_exact_benchmark.py`:

```python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import scipy

from .artifacts import RunArtifactWriter
from .exact import exact_schedule
from .experiment import (
    ExperimentDefinition,
    _git_sha,
    _slot_method,
    definition_to_dict,
    load_definition,
)
from .model import NOMAChannel
from .trace import generate_trace


def run_exact_definition(
    definition: ExperimentDefinition,
    config_path: Path,
    output_dir: Path,
) -> None:
    if definition.scenario.num_users > 5:
        raise ValueError("complete benchmark is frozen at no more than five users")
    writer = RunArtifactWriter(output_dir)
    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    initializer_statuses: list[str] = []
    search_terminations: list[str] = []
    for seed in definition.seeds:
        trace = generate_trace(definition.scenario, seed, definition.csi_error_std)
        full_completed = 0
        exact_completed = 0
        arrivals = 0
        for snapshot in trace.slots:
            events: list[dict] = []
            started = time.perf_counter()
            full = _slot_method(
                "full",
                definition.scenario,
                definition.evaluation,
                definition.max_iterations,
                events,
            )(snapshot)
            initializer_statuses.extend(
                str(event["reason"])
                for event in events
                if event.get("move_type") == "initialization"
            )
            search_terminations.extend(
                str(event["reason"])
                for event in events
                if event.get("move_type") == "termination"
            )
            full_runtime = time.perf_counter() - started
            channel = NOMAChannel(
                definition.scenario.channel,
                definition.scenario.uav.height,
            )
            started = time.perf_counter()
            exact = exact_schedule(
                snapshot,
                channel,
                definition.evaluation,
                max_heavy_tasks=5,
            )
            exact_runtime = time.perf_counter() - started
            if full.completed_count > exact.outcome.completed_count:
                raise AssertionError("heuristic completion count exceeds complete enumeration")
            slot_arrivals = len(full.arrived_task_ids)
            arrivals += slot_arrivals
            full_completed += full.completed_count
            exact_completed += exact.outcome.completed_count
            rows.append({
                "seed": seed,
                "slot": snapshot.slot_index,
                "arrivals": slot_arrivals,
                "full_completed": full.completed_count,
                "exact_completed": exact.outcome.completed_count,
                "completion_gap": exact.outcome.completed_count - full.completed_count,
                "full_runtime_s": full_runtime,
                "exact_runtime_s": exact_runtime,
                "evaluated_schedules": exact.evaluated_schedules,
                "complete_search": exact.complete_search,
            })
        seed_rows.append({
            "seed": seed,
            "arrivals": arrivals,
            "full_on_time_rate": full_completed / arrivals if arrivals else 0.0,
            "exact_on_time_rate": exact_completed / arrivals if arrivals else 0.0,
            "completion_gap": exact_completed - full_completed,
        })
    gaps = np.asarray([int(row["completion_gap"]) for row in rows], dtype=float)
    summary = {
        "run_id": definition.run_id,
        "seeds": list(definition.seeds),
        "initializer_status_counts": {
            status: initializer_statuses.count(status)
            for status in sorted(set(initializer_statuses))
        },
        "search_termination_counts": {
            status: search_terminations.count(status)
            for status in sorted(set(search_terminations))
        },
        "optimal_slot_fraction": float(np.mean(gaps == 0.0)),
        "mean_completion_gap": float(np.mean(gaps)),
        "max_completion_gap": int(np.max(gaps)),
        "by_seed": seed_rows,
    }
    writer.write_csv(
        "exact_slots.csv",
        rows,
        fieldnames=(
            "seed", "slot", "arrivals", "full_completed", "exact_completed",
            "completion_gap", "full_runtime_s", "exact_runtime_s",
            "evaluated_schedules", "complete_search",
        ),
    )
    writer.write_json("summary.json", summary)
    writer.finalize({
        "run_id": definition.run_id,
        "git_sha": _git_sha(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "resolved_definition": definition_to_dict(definition),
        "search_space": "all unscheduled or UAV/subcarrier assignments with block capacity two",
        "solver": {"name": "scipy.optimize.milp/HiGHS", "package_version": scipy.__version__},
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_exact_definition(load_definition(args.config), args.config, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run exact benchmark tests**

Run: `python -m pytest tests/test_exact.py tests/test_exact_benchmark.py -v`

Expected: PASS; complete enumeration is never worse than the heuristic and every search is marked complete.

- [ ] **Step 6: Commit**

```bash
git add uav_mec/run_exact_benchmark.py configs/thesis/confirmatory_exact.toml tests/test_exact_benchmark.py
git commit -m "feat: add complete-enumeration benchmark"
```

---

### Task 22: Validate Evidence And Compute Thesis Statistics

**Files:**
- Create: `uav_mec/analyze_results.py`
- Create: `tests/test_result_analysis.py`

- [ ] **Step 1: Write failing validation and comparison tests**

```python
import copy

import pytest

from uav_mec.analyze_results import Bundle, comparison_report, validate_bundle


def bundle() -> Bundle:
    methods = ("full", "intra_global")
    tasks = []
    slots = []
    values = {"full": (1.0, 1.0, 1.0, 1.0), "intra_global": (0.5, 0.5, 0.5, 0.5)}
    for seed in range(4):
        for method in methods:
            for task_id in (1, 2):
                completed = method == "full" or task_id == 1
                tasks.append({
                    "method": method, "seed": str(seed), "slot": "0", "task_id": str(task_id),
                    "user_id": str(task_id), "arrival_slot": "0", "data_bits": "1000000",
                    "comp_cycles": "1200000000", "deadline_s": "0.8", "local_cpu_hz": "1000000000",
                    "completed": str(completed), "delay_s": "0.7" if completed else "",
                    "cpu_hz": "1000000000" if completed else "0", "serving_uav": "0" if completed else "",
                    "sic_feasible": "True" if completed else "", "reason": "complete" if completed else "unscheduled",
                })
            slots.append({
                "method": method, "seed": str(seed), "slot": "0", "runtime_s": "0.01",
                "sic_violations": "0",
            })
    summary = {
        "seeds": list(range(4)),
        "initializer_status_counts": {"optimal": 8},
        "search_termination_counts": {"local_optimum": 8},
        "methods": {name: {"on_time_rate_by_seed": list(values[name])} for name in methods},
    }
    manifest = {
        "resolved_definition": {
            "evaluation": {"cpu_per_uav_hz": 4e9},
            "scenario": {"num_uavs": 1},
        }
    }
    return Bundle("pairing", None, manifest, summary, tasks, slots)


def test_duplicate_task_result_is_a_hard_validation_failure():
    value = bundle()
    duplicated = copy.deepcopy(value.tasks)
    duplicated.append(copy.deepcopy(duplicated[0]))
    with pytest.raises(ValueError, match="duplicate task outcome"):
        validate_bundle(Bundle(value.name, value.path, value.manifest, value.summary, duplicated, value.slots))


def test_positive_paired_effect_has_ci_and_holm_fields():
    report = comparison_report({"pairing": bundle().summary})
    result = report["innovation1_cross"]
    assert result["mean_difference_pp"] == pytest.approx(50.0)
    assert result["ci_low_pp"] > 0.0
    assert 0.0 <= result["p_holm"] <= 1.0
```

- [ ] **Step 2: Run tests and verify the missing analysis module**

Run: `python -m pytest tests/test_result_analysis.py -v`

Expected: ERROR with `ModuleNotFoundError: No module named 'uav_mec.analyze_results'`.

- [ ] **Step 3: Implement bundle loading, checksum verification, and hard invariants**

Create `uav_mec/analyze_results.py` with the imports, records, and validation functions below:

```python
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from .artifacts import RunArtifactWriter, sha256_file
from .statistics import holm_adjust, paired_comparison


@dataclass(frozen=True, slots=True)
class Bundle:
    name: str
    path: Path | None
    manifest: dict
    summary: dict
    tasks: list[dict[str, str]]
    slots: list[dict[str, str]]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_checksums(path: Path) -> None:
    lines = (path / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, relative = line.split("  ", 1)
        if sha256_file(path / relative) != expected:
            raise ValueError(f"checksum mismatch: {path / relative}")


def load_bundle(name: str, path: Path) -> Bundle:
    verify_checksums(path)
    return Bundle(
        name,
        path,
        json.loads((path / "manifest.json").read_text(encoding="utf-8")),
        json.loads((path / "summary.json").read_text(encoding="utf-8")),
        _read_csv(path / "tasks.csv"),
        _read_csv(path / "slots.csv"),
    )


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"true", "1"}


def validate_bundle(bundle: Bundle) -> dict[str, object]:
    methods = tuple(bundle.summary["methods"])
    seeds = tuple(int(seed) for seed in bundle.summary["seeds"])
    initializer_methods = {
        "ilp", "exchange_stable", "intra_bilateral", "cross_bilateral",
        "intra_global", "full", "full_fixed_half", "full_crossing",
    }
    statuses = bundle.summary.get("initializer_status_counts", {})
    if initializer_methods.intersection(methods) and not statuses:
        raise ValueError("missing ILP initializer status evidence")
    failed_statuses = {name: count for name, count in statuses.items() if name != "optimal" and count}
    if failed_statuses:
        raise ValueError(f"non-optimal ILP initializer status: {failed_statuses}")
    search_methods = {
        "exchange_stable", "intra_bilateral", "cross_bilateral", "intra_global",
        "full", "full_fixed_half", "full_crossing",
    }
    terminations = bundle.summary.get("search_termination_counts", {})
    if search_methods.intersection(methods) and not terminations:
        raise ValueError("missing search termination evidence")
    accepted_terminations = {"local_optimum", "cycle_free_fixed_point"}
    failed_terminations = {
        name: count for name, count in terminations.items()
        if name not in accepted_terminations and count
    }
    if failed_terminations:
        raise ValueError(f"non-converged search termination: {failed_terminations}")
    expected_seed_slots: dict[str, set[tuple[int, int]]] = {}
    for method in methods:
        expected_seed_slots[method] = {
            (int(row["seed"]), int(row["slot"]))
            for row in bundle.slots
            if row["method"] == method
        }
    if len({frozenset(value) for value in expected_seed_slots.values()}) != 1:
        raise ValueError("methods do not share identical seed/slot traces")

    seen: set[tuple[str, int, int]] = set()
    task_sets: dict[tuple[str, int], set[int]] = defaultdict(set)
    immutable: dict[tuple[int, int], tuple[str, ...]] = {}
    identity_fields = (
        "user_id", "arrival_slot", "data_bits", "comp_cycles", "deadline_s", "local_cpu_hz",
    )
    sic_violations = 0
    cpu_by_pool: dict[tuple[str, int, int, int], float] = defaultdict(float)
    for row in bundle.tasks:
        method = row["method"]
        seed = int(row["seed"])
        task_id = int(row["task_id"])
        key = (method, seed, task_id)
        if key in seen:
            raise ValueError(f"duplicate task outcome {key}")
        seen.add(key)
        task_sets[(method, seed)].add(task_id)
        identity = tuple(row[field] for field in identity_fields)
        trace_key = (seed, task_id)
        if trace_key in immutable and immutable[trace_key] != identity:
            raise ValueError(f"task input changed across methods: {trace_key}")
        immutable[trace_key] = identity
        if row.get("sic_feasible", "") and not _is_true(row["sic_feasible"]):
            sic_violations += 1
        if row.get("serving_uav", ""):
            pool = (method, seed, int(row["slot"]), int(row["serving_uav"]))
            cpu_by_pool[pool] += float(row["cpu_hz"])
    for seed in seeds:
        method_sets = [task_sets[(method, seed)] for method in methods]
        if len({frozenset(value) for value in method_sets}) != 1:
            raise ValueError(f"methods do not share identical task IDs for seed {seed}")

    sic_violations += sum(int(row["sic_violations"]) for row in bundle.slots)
    if sic_violations:
        raise ValueError(f"found {sic_violations} SIC violations")
    cpu_budget = float(bundle.manifest["resolved_definition"]["evaluation"]["cpu_per_uav_hz"])
    exceeded = {key: value for key, value in cpu_by_pool.items() if value > cpu_budget + 1e-6}
    if exceeded:
        raise ValueError(f"CPU budget exceeded: {exceeded}")

    for method in methods:
        expected = bundle.summary["methods"][method]["on_time_rate_by_seed"]
        actual = []
        for seed in seeds:
            rows = [row for row in bundle.tasks if row["method"] == method and int(row["seed"]) == seed]
            actual.append(sum(_is_true(row["completed"]) for row in rows) / len(rows) if rows else 0.0)
        if not np.allclose(actual, expected, atol=1e-12, rtol=0.0):
            raise ValueError(f"summary/task mismatch for {method}")
    return {
        "methods": list(methods),
        "seeds": list(seeds),
        "unique_tasks": len(immutable),
        "sic_violations": 0,
        "cpu_budget_violations": 0,
        "initializer_failures": 0,
        "search_nonconvergence": 0,
    }
```

- [ ] **Step 4: Implement secondary metrics and all predeclared comparisons**

Append to `uav_mec/analyze_results.py`:

```python
COMPARISONS = (
    ("innovation1_cross", "pairing", "full", "intra_global"),
    ("innovation1_acceptance", "pairing", "full", "cross_bilateral"),
    ("innovation2_deadline", "resource", "full", "full_crossing"),
    ("resource_fixed_half", "resource", "full", "full_fixed_half"),
    ("baseline_greedy", "main", "full", "greedy"),
    ("baseline_ia_greedy", "main", "full", "ia_greedy"),
    ("baseline_ilp", "main", "full", "ilp"),
    ("baseline_exchange", "main", "full", "exchange_stable"),
    ("baseline_ofdma", "main", "full", "fair_ofdma"),
)


def comparison_report(summaries: Mapping[str, dict]) -> dict[str, dict[str, float]]:
    raw: dict[str, dict[str, float]] = {}
    for name, bundle_name, proposed, baseline in COMPARISONS:
        if bundle_name not in summaries:
            continue
        methods = summaries[bundle_name]["methods"]
        if proposed not in methods or baseline not in methods:
            continue
        left = np.asarray(methods[proposed]["on_time_rate_by_seed"], dtype=float) * 100.0
        right = np.asarray(methods[baseline]["on_time_rate_by_seed"], dtype=float) * 100.0
        result = paired_comparison(left, right)
        raw[name] = {
            "proposed": proposed,
            "baseline": baseline,
            "mean_difference_pp": result.mean_difference,
            "ci_low_pp": result.ci_low,
            "ci_high_pp": result.ci_high,
            "wilcoxon_w": result.wilcoxon_w,
            "p_raw": result.p_raw,
            "rank_biserial": result.rank_biserial,
        }
    adjusted = holm_adjust({name: value["p_raw"] for name, value in raw.items()})
    for name, value in raw.items():
        value["p_holm"] = adjusted[name]
    return raw


def _jain(values: list[float]) -> float:
    vector = np.asarray(values, dtype=float)
    denominator = len(vector) * float(np.square(vector).sum())
    return float(vector.sum() ** 2 / denominator) if denominator > 0 else 0.0


def bundle_metrics(bundle: Bundle) -> dict[str, dict[str, float]]:
    methods = tuple(bundle.summary["methods"])
    seeds = tuple(int(seed) for seed in bundle.summary["seeds"])
    scenario = bundle.manifest["resolved_definition"]["scenario"]
    cpu_budget = float(bundle.manifest["resolved_definition"]["evaluation"]["cpu_per_uav_hz"])
    metrics: dict[str, dict[str, float]] = {}
    for method in methods:
        seed_metrics = []
        method_slots = [row for row in bundle.slots if row["method"] == method]
        for seed in seeds:
            rows = [row for row in bundle.tasks if row["method"] == method and int(row["seed"]) == seed]
            completed = [_is_true(row["completed"]) for row in rows]
            delays = [float(row["delay_s"]) for row in rows if _is_true(row["completed"]) and row["delay_s"]]
            by_user: dict[int, list[bool]] = defaultdict(list)
            for row, done in zip(rows, completed, strict=True):
                by_user[int(row["user_id"])].append(done)
            user_rates = [sum(values) / len(values) for values in by_user.values()]
            used_cpu = sum(float(row["cpu_hz"]) for row in rows if row["cpu_hz"])
            denominator = int(scenario["num_slots"]) * int(scenario["num_uavs"]) * cpu_budget
            seed_metrics.append({
                "on_time_rate": sum(completed) / len(completed) if completed else 0.0,
                "drop_rate": 1.0 - (sum(completed) / len(completed) if completed else 0.0),
                "mean_completed_delay_s": float(np.mean(delays)) if delays else 0.0,
                "jain_user_on_time": _jain(user_rates),
                "cpu_utilization": used_cpu / denominator,
            })
        runtimes = np.asarray([float(row["runtime_s"]) for row in method_slots], dtype=float)
        metrics[method] = {
            key: float(np.mean([row[key] for row in seed_metrics]))
            for key in seed_metrics[0]
        }
        metrics[method]["runtime_p50_s"] = float(np.quantile(runtimes, 0.50))
        metrics[method]["runtime_p95_s"] = float(np.quantile(runtimes, 0.95))
    return metrics
```

- [ ] **Step 5: Implement analysis output and non-negotiable thesis gates**

Append to `uav_mec/analyze_results.py`:

```python
def analyze(
    main_path: Path,
    pairing_path: Path,
    resource_path: Path,
    exact_path: Path,
    output_dir: Path,
    require_thesis_gates: bool = False,
) -> dict[str, object]:
    bundles = {
        name: load_bundle(name, path)
        for name, path in (
            ("main", main_path),
            ("pairing", pairing_path),
            ("resource", resource_path),
        )
    }
    validation = {name: validate_bundle(bundle) for name, bundle in bundles.items()}
    verify_checksums(exact_path)
    exact_rows = _read_csv(exact_path / "exact_slots.csv")
    if not all(_is_true(row["complete_search"]) for row in exact_rows):
        raise ValueError("exact benchmark contains an incomplete search")
    if any(int(row["full_completed"]) > int(row["exact_completed"]) for row in exact_rows):
        raise ValueError("exact benchmark is below the heuristic")
    exact_summary = json.loads((exact_path / "summary.json").read_text(encoding="utf-8"))
    comparisons = comparison_report({name: bundle.summary for name, bundle in bundles.items()})
    metrics = {name: bundle_metrics(bundle) for name, bundle in bundles.items()}
    gates = {
        "constraint_violations_zero": all(
            row["sic_violations"] == 0
            and row["cpu_budget_violations"] == 0
            and row["initializer_failures"] == 0
            and row["search_nonconvergence"] == 0
            for row in validation.values()
        ),
        "innovation1_cross_ci_positive": comparisons["innovation1_cross"]["ci_low_pp"] > 0.0,
        "innovation1_acceptance_ci_positive": comparisons["innovation1_acceptance"]["ci_low_pp"] > 0.0,
        "innovation2_deadline_ci_positive": comparisons["innovation2_deadline"]["ci_low_pp"] > 0.0,
        "full_beats_predeclared_strong_baseline": comparisons["baseline_exchange"]["ci_low_pp"] > 0.0,
        "n20_runtime_p95_below_one_second": metrics["main"]["full"]["runtime_p95_s"] < 1.0,
        "exact_search_complete": all(_is_true(row["complete_search"]) for row in exact_rows),
    }
    report = {
        "validation": validation,
        "comparisons": comparisons,
        "metrics": metrics,
        "exact": exact_summary,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    writer = RunArtifactWriter(output_dir)
    writer.write_json("statistics.json", comparisons)
    writer.write_json("metrics.json", metrics)
    writer.write_json("acceptance.json", {
        "validation": validation,
        "exact": exact_summary,
        "gates": gates,
        "all_gates_pass": report["all_gates_pass"],
    })
    writer.finalize({
        "inputs": {
            "main": str(main_path),
            "pairing": str(pairing_path),
            "resource": str(resource_path),
            "exact": str(exact_path),
        }
    })
    if require_thesis_gates and not report["all_gates_pass"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"thesis evidence gates failed: {failed}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-thesis-gates", action="store_true")
    args = parser.parse_args()
    analyze(args.main, args.pairing, args.resource, args.exact, args.output, args.require_thesis_gates)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run result-analysis and statistical tests**

Run: `python -m pytest tests/test_statistics.py tests/test_result_analysis.py -v`

Expected: PASS, 5 tests passed.

- [ ] **Step 7: Commit**

```bash
git add uav_mec/analyze_results.py tests/test_result_analysis.py
git commit -m "feat: enforce thesis evidence gates"
```

---

### Task 23: Publish Canonical Evidence And Generate Paper Assets

**Files:**
- Create: `uav_mec/publish_canonical.py`
- Create: `uav_mec/generate_figures_v2.py`
- Create: `tests/test_canonical_publication.py`
- Create: `tests/test_figure_generation_v2.py`
- Generate after confirmatory gates: `paper/generated_results.tex`
- Generate after confirmatory gates: `paper/generated_statistics.tex`
- Generate after confirmatory gates: `paper/generated_robustness.tex`

- [ ] **Step 1: Write failing canonical-publication tests**

```python
import json
from pathlib import Path

from uav_mec.artifacts import RunArtifactWriter
from uav_mec.publish_canonical import publish_canonical


def source_bundle(path: Path, run_id: str) -> None:
    writer = RunArtifactWriter(path)
    writer.write_json("summary.json", {"run_id": run_id, "seeds": [1000, 1001], "methods": {}})
    writer.finalize({"run_id": run_id, "resolved_definition": {}})


def test_publisher_keeps_only_whitelisted_lightweight_evidence(tmp_path: Path):
    sources = {}
    for name in ("main", "pairing", "resource", "exact"):
        path = tmp_path / f"source-{name}"
        source_bundle(path, name)
        sources[name] = path
    analysis = tmp_path / "source-analysis"
    writer = RunArtifactWriter(analysis)
    writer.write_json("statistics.json", {})
    writer.write_json("metrics.json", {})
    writer.write_json("acceptance.json", {"all_gates_pass": True})
    writer.finalize({"inputs": {}})
    suite_roots = {}
    for suite in ("mechanism", "scalability", "robustness"):
        case = tmp_path / f"suite-{suite}" / "case0"
        source_bundle(case, f"{suite}-case0")
        suite_roots[suite] = case.parent
    output = tmp_path / "canonical-v1"
    publish_canonical(sources, analysis, suite_roots, output)
    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index["runs"]["main"] == "runs/main"
    assert index["suites"]["mechanism"]["case0"] == "suites/mechanism/case0"
    assert (output / "analysis/statistics.json").exists()
    assert not list(output.rglob("tasks.csv"))
    assert (output / "checksums.sha256").exists()
```

- [ ] **Step 2: Run the publication test and verify the missing module**

Run: `python -m pytest tests/test_canonical_publication.py -v`

Expected: ERROR importing `uav_mec.publish_canonical`.

- [ ] **Step 3: Implement whitelist-only canonical publication**

Create `uav_mec/publish_canonical.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Mapping

from .analyze_results import verify_checksums
from .artifacts import RunArtifactWriter, sha256_file


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_json_files(
    writer: RunArtifactWriter,
    source: Path,
    destination: str,
    names: tuple[str, ...],
) -> None:
    verify_checksums(source)
    for name in names:
        writer.write_json(f"{destination}/{name}", _json(source / name))


def publish_canonical(
    runs: Mapping[str, Path],
    analysis: Path,
    suite_roots: Mapping[str, Path],
    output_dir: Path,
) -> None:
    required_runs = {"main", "pairing", "resource", "exact"}
    if set(runs) != required_runs:
        raise ValueError(f"canonical runs must be {sorted(required_runs)}")
    writer = RunArtifactWriter(output_dir)
    provenance: dict[str, str] = {}
    run_index = {}
    for name, source in sorted(runs.items()):
        destination = f"runs/{name}"
        _publish_json_files(writer, source, destination, ("summary.json", "manifest.json"))
        run_index[name] = destination
        provenance[f"run:{name}"] = sha256_file(source / "checksums.sha256")
    _publish_json_files(
        writer,
        analysis,
        "analysis",
        ("statistics.json", "metrics.json", "acceptance.json", "manifest.json"),
    )
    provenance["analysis"] = sha256_file(analysis / "checksums.sha256")
    suite_index: dict[str, dict[str, str]] = {}
    for suite_name, root in sorted(suite_roots.items()):
        cases = {}
        for case in sorted(path for path in root.iterdir() if path.is_dir()):
            destination = f"suites/{suite_name}/{case.name}"
            _publish_json_files(writer, case, destination, ("summary.json", "manifest.json"))
            cases[case.name] = destination
            provenance[f"suite:{suite_name}:{case.name}"] = sha256_file(case / "checksums.sha256")
        if not cases:
            raise ValueError(f"suite {suite_name} contains no cases")
        suite_index[suite_name] = cases
    writer.write_json("index.json", {
        "schema_version": 1,
        "runs": run_index,
        "analysis": "analysis",
        "suites": suite_index,
    })
    writer.finalize({
        "schema_version": 1,
        "source_bundle_checksums": provenance,
        "policy": "JSON summary/statistics/manifest only; raw task and slot rows excluded",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--pairing", type=Path, required=True)
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--scalability", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish_canonical(
        {"main": args.main, "pairing": args.pairing, "resource": args.resource, "exact": args.exact},
        args.analysis,
        {"mechanism": args.mechanism, "scalability": args.scalability, "robustness": args.robustness},
        args.output,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the canonical-publication test**

Run: `python -m pytest tests/test_canonical_publication.py -v`

Expected: PASS, 1 test passed.

- [ ] **Step 5: Write failing figure and macro tests**

```python
import json
from pathlib import Path

import pytest

from uav_mec.generate_figures_v2 import generate_assets, method_mean_ci


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_error_bar_is_bootstrap_ci_and_macros_match_statistics(tmp_path: Path):
    root = tmp_path / "canonical"
    index = {
        "runs": {name: f"runs/{name}" for name in ("main", "pairing", "resource", "exact")},
        "analysis": "analysis",
        "suites": {},
    }
    write_json(root / "index.json", index)
    summaries = {
        "main": {"seeds": [1, 2, 3, 4], "methods": {
            "full": {"on_time_rate_by_seed": [0.80, 0.82, 0.84, 0.86]},
            "exchange_stable": {"on_time_rate_by_seed": [0.70, 0.72, 0.74, 0.76]},
        }},
        "pairing": {"seeds": [1, 2, 3, 4], "methods": {
            "intra_bilateral": {"on_time_rate_by_seed": [0.60] * 4},
            "cross_bilateral": {"on_time_rate_by_seed": [0.65] * 4},
            "intra_global": {"on_time_rate_by_seed": [0.70] * 4},
            "full": {"on_time_rate_by_seed": [0.80] * 4},
        }},
        "resource": {"seeds": [1, 2, 3, 4], "methods": {
            "full_fixed_half": {"on_time_rate_by_seed": [0.55] * 4},
            "full_crossing": {"on_time_rate_by_seed": [0.70] * 4},
            "full": {"on_time_rate_by_seed": [0.80] * 4},
        }},
        "exact": {"optimal_slot_fraction": 0.92, "mean_completion_gap": 0.08},
    }
    for name, summary in summaries.items():
        write_json(root / f"runs/{name}/summary.json", summary)
    statistics = {
        "innovation1_cross": {"mean_difference_pp": 10.0, "ci_low_pp": 8.0, "ci_high_pp": 12.0, "p_holm": 0.01, "rank_biserial": 1.0},
        "innovation1_acceptance": {"mean_difference_pp": 15.0, "ci_low_pp": 13.0, "ci_high_pp": 17.0, "p_holm": 0.01, "rank_biserial": 1.0},
        "innovation2_deadline": {"mean_difference_pp": 10.0, "ci_low_pp": 9.0, "ci_high_pp": 11.0, "p_holm": 0.01, "rank_biserial": 1.0},
        "baseline_exchange": {"mean_difference_pp": 10.0, "ci_low_pp": 8.0, "ci_high_pp": 12.0, "p_holm": 0.01, "rank_biserial": 1.0},
    }
    write_json(root / "analysis/statistics.json", statistics)
    write_json(root / "analysis/metrics.json", {"main": {"full": {
        "on_time_rate": 0.83,
        "drop_rate": 0.17,
        "mean_completed_delay_s": 0.71,
        "jain_user_on_time": 0.96,
        "cpu_utilization": 0.72,
        "runtime_p50_s": 0.08,
        "runtime_p95_s": 0.12,
    }}})
    write_json(root / "analysis/acceptance.json", {
        "all_gates_pass": True,
        "validation": {"main": {"sic_violations": 0}},
    })
    figures = tmp_path / "figures"
    macros = tmp_path / "generated_results.tex"
    statistics_table = tmp_path / "generated_statistics.tex"
    robustness = tmp_path / "generated_robustness.tex"
    generate_assets(root, figures, macros, statistics_table, robustness)
    mean, low, high = method_mean_ci(summaries["main"], "full")
    assert mean == pytest.approx(83.0)
    assert low < mean < high
    text = macros.read_text(encoding="utf-8")
    assert "\\newcommand{\\InnovationOneCrossGainPP}{10.00}" in text
    assert "\\newcommand{\\ExactOptimalPercent}{92.00\\%}" in text
    assert "innovation1\\_cross" in statistics_table.read_text(encoding="utf-8")
    assert (figures / "main_on_time.pdf").exists()
    assert (figures / "pairing_ablation.pdf").exists()
    assert (figures / "resource_ablation.pdf").exists()
```

- [ ] **Step 6: Run the test and verify the missing generator**

Run: `python -m pytest tests/test_figure_generation_v2.py -v`

Expected: ERROR importing `uav_mec.generate_figures_v2`.

- [ ] **Step 7: Implement canonical-only figures and generated LaTeX values**

Create `uav_mec/generate_figures_v2.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .statistics import paired_bootstrap_ci


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def method_mean_ci(summary: dict, method: str) -> tuple[float, float, float]:
    values = np.asarray(summary["methods"][method]["on_time_rate_by_seed"], dtype=float) * 100.0
    low, high = paired_bootstrap_ci(values, 10_000, 20260711)
    return float(values.mean()), low, high


def _bar_figure(summary: dict, methods: tuple[str, ...], labels: tuple[str, ...], output: Path) -> None:
    values = [method_mean_ci(summary, method) for method in methods]
    means = np.asarray([value[0] for value in values])
    errors = np.asarray([
        [value[0] - value[1] for value in values],
        [value[2] - value[0] for value in values],
    ])
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    colors = ("#3B6FB6", "#E07A3F", "#3F8F68", "#8A5A9E", "#6B7280", "#C44E52", "#8172B3", "#4C9F9A")
    axis.bar(np.arange(len(methods)), means, yerr=errors, capsize=4, color=colors[:len(methods)])
    axis.set_xticks(np.arange(len(methods)), labels, rotation=18, ha="right")
    axis.set_ylabel("On-time completion rate (%)")
    axis.set_ylim(0.0, min(100.0, max(means + errors[1]) + 8.0))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        figure.savefig(
            output.with_suffix(f".{suffix}"),
            dpi=220,
            bbox_inches="tight",
            metadata=metadata,
        )
    plt.close(figure)


def _suite_figure(
    root: Path,
    cases: dict[str, str],
    case_names: tuple[str, ...],
    labels: tuple[str, ...],
    output: Path,
) -> None:
    if not cases:
        return
    summaries = [_json(root / cases[name] / "summary.json") for name in case_names]
    methods = tuple(name for name in ("greedy", "exchange_stable", "full") if name in summaries[0]["methods"])
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for method, color, marker in zip(methods, ("#3B6FB6", "#E07A3F", "#3F8F68"), ("o", "s", "^"), strict=True):
        values = [method_mean_ci(summary, method) for summary in summaries]
        means = np.asarray([value[0] for value in values])
        errors = np.asarray([
            [value[0] - value[1] for value in values],
            [value[2] - value[0] for value in values],
        ])
        axis.errorbar(labels, means, yerr=errors, label=method.replace("_", " ").title(), color=color, marker=marker, capsize=3)
    axis.set_ylabel("On-time completion rate (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        figure.savefig(
            output.with_suffix(f".{suffix}"),
            dpi=220,
            bbox_inches="tight",
            metadata=metadata,
        )
    plt.close(figure)


def _suite_table(root: Path, index: dict) -> str:
    robustness = index.get("suites", {}).get("robustness", {})
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Case & Exchange-Stable (\\%) & Proposed (\\%) \\\\",
        "\\midrule",
    ]
    for case, relative in sorted(robustness.items()):
        summary = _json(root / relative / "summary.json")
        exchange = method_mean_ci(summary, "exchange_stable")[0]
        full = method_mean_ci(summary, "full")[0]
        escaped_case = case.replace("_", "\\_")
        lines.append(f"{escaped_case} & {exchange:.2f} & {full:.2f} \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    return "\n".join(lines) + "\n"


def _statistics_table(statistics: dict, metrics: dict) -> str:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Comparison & $\\Delta$ (pp) & 95\\% CI & $p_{\\mathrm{Holm}}$ & $r_{rb}$ \\\\",
        "\\midrule",
    ]
    for name, value in statistics.items():
        escaped_name = name.replace("_", "\\_")
        p_value = float(value["p_holm"])
        p_text = "$<0.0001$" if p_value < 0.0001 else f"{p_value:.4f}"
        lines.append(
            f"{escaped_name} & {value['mean_difference_pp']:.2f} & "
            f"[{value['ci_low_pp']:.2f}, {value['ci_high_pp']:.2f}] & "
            f"{p_text} & {value['rank_biserial']:.3f} \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}", "", "\\bigskip", ""))
    lines.extend((
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & On-time (\\%) & Drop (\\%) & Delay (s) & Jain & CPU (\\%) & P95 (ms) \\\\",
        "\\midrule",
    ))
    for method, value in metrics["main"].items():
        escaped_method = method.replace("_", "\\_")
        lines.append(
            f"{escaped_method} & {100.0 * value['on_time_rate']:.2f} & "
            f"{100.0 * value['drop_rate']:.2f} & {value['mean_completed_delay_s']:.4f} & "
            f"{value['jain_user_on_time']:.4f} & {100.0 * value['cpu_utilization']:.2f} & "
            f"{1000.0 * value['runtime_p95_s']:.2f} \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    return "\n".join(lines) + "\n"


def generate_assets(
    canonical_root: Path,
    figures_dir: Path,
    macros_path: Path,
    statistics_path: Path,
    robustness_path: Path,
) -> None:
    index = _json(canonical_root / "index.json")
    summaries = {
        name: _json(canonical_root / relative / "summary.json")
        for name, relative in index["runs"].items()
    }
    analysis = canonical_root / index["analysis"]
    statistics = _json(analysis / "statistics.json")
    metrics = _json(analysis / "metrics.json")
    acceptance = _json(analysis / "acceptance.json")
    if not acceptance["all_gates_pass"]:
        raise RuntimeError("refusing to generate thesis assets from failed evidence gates")
    figures_dir.mkdir(parents=True, exist_ok=True)
    _bar_figure(
        summaries["main"],
        tuple(summaries["main"]["methods"]),
        tuple(name.replace("_", " ").title() for name in summaries["main"]["methods"]),
        figures_dir / "main_on_time",
    )
    _bar_figure(
        summaries["pairing"],
        ("intra_bilateral", "cross_bilateral", "intra_global", "full"),
        ("Intra/Bilateral", "Cross/Bilateral", "Intra/Global", "Cross/Global"),
        figures_dir / "pairing_ablation",
    )
    _bar_figure(
        summaries["resource"],
        ("full_fixed_half", "full_crossing", "full"),
        ("Fixed 0.5 / Equal", "Crossing / Equal", "Deadline / Min-CPU"),
        figures_dir / "resource_ablation",
    )
    suite_index = index.get("suites", {})
    mechanism = suite_index.get("mechanism", {})
    if mechanism:
        _suite_figure(
            canonical_root,
            mechanism,
            ("coupling0", "coupling05", "coupling1"),
            ("0", "0.5", "1"),
            figures_dir / "mechanism_interference",
        )
    scalability = suite_index.get("scalability", {})
    if scalability:
        _suite_figure(
            canonical_root,
            scalability,
            ("n10", "n15", "n20", "n25", "n30", "n40", "n50"),
            ("10", "15", "20", "25", "30", "40", "50"),
            figures_dir / "scalability_users",
        )
        _suite_figure(
            canonical_root,
            scalability,
            ("k1", "k2", "k3", "k4", "k5"),
            ("1", "2", "3", "4", "5"),
            figures_dir / "scalability_uavs",
        )
    full_mean, full_low, full_high = method_mean_ci(summaries["main"], "full")
    validation = acceptance["validation"]
    sic_count = sum(int(value["sic_violations"]) for value in validation.values())
    exact = summaries["exact"]
    macros = {
        "FullOnTimePercent": f"{full_mean:.2f}\\%",
        "FullOnTimeCILowPercent": f"{full_low:.2f}\\%",
        "FullOnTimeCIHighPercent": f"{full_high:.2f}\\%",
        "InnovationOneCrossGainPP": f"{statistics['innovation1_cross']['mean_difference_pp']:.2f}",
        "InnovationOneCrossCILowPP": f"{statistics['innovation1_cross']['ci_low_pp']:.2f}",
        "InnovationOneCrossCIHighPP": f"{statistics['innovation1_cross']['ci_high_pp']:.2f}",
        "InnovationOneAcceptanceGainPP": f"{statistics['innovation1_acceptance']['mean_difference_pp']:.2f}",
        "InnovationOneAcceptanceCILowPP": f"{statistics['innovation1_acceptance']['ci_low_pp']:.2f}",
        "InnovationOneAcceptanceCIHighPP": f"{statistics['innovation1_acceptance']['ci_high_pp']:.2f}",
        "InnovationTwoGainPP": f"{statistics['innovation2_deadline']['mean_difference_pp']:.2f}",
        "InnovationTwoCILowPP": f"{statistics['innovation2_deadline']['ci_low_pp']:.2f}",
        "InnovationTwoCIHighPP": f"{statistics['innovation2_deadline']['ci_high_pp']:.2f}",
        "StrongBaselineGainPP": f"{statistics['baseline_exchange']['mean_difference_pp']:.2f}",
        "StrongBaselineCILowPP": f"{statistics['baseline_exchange']['ci_low_pp']:.2f}",
        "StrongBaselineCIHighPP": f"{statistics['baseline_exchange']['ci_high_pp']:.2f}",
        "FullRuntimeP95Ms": f"{1000.0 * metrics['main']['full']['runtime_p95_s']:.2f}",
        "ExactOptimalPercent": f"{100.0 * exact['optimal_slot_fraction']:.2f}\\%",
        "ExactMeanCompletionGap": f"{exact['mean_completion_gap']:.3f}",
        "SICViolationCount": str(sic_count),
        "ConfirmatorySeedCount": str(len(summaries["main"]["seeds"])),
    }
    macros_path.parent.mkdir(parents=True, exist_ok=True)
    macros_path.write_text(
        "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items()),
        encoding="utf-8",
    )
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(_statistics_table(statistics, metrics), encoding="utf-8")
    robustness_path.parent.mkdir(parents=True, exist_ok=True)
    robustness_path.write_text(_suite_table(canonical_root, index), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=Path("figures/thesis"))
    parser.add_argument("--macros", type=Path, default=Path("paper/generated_results.tex"))
    parser.add_argument("--statistics-table", type=Path, default=Path("paper/generated_statistics.tex"))
    parser.add_argument("--robustness-table", type=Path, default=Path("paper/generated_robustness.tex"))
    args = parser.parse_args()
    generate_assets(args.canonical, args.figures, args.macros, args.statistics_table, args.robustness_table)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run generation tests**

Run: `python -m pytest tests/test_canonical_publication.py tests/test_figure_generation_v2.py -v`

Expected: PASS, 2 tests passed; generated bar error bars use deterministic 95% bootstrap intervals.

- [ ] **Step 9: Commit**

```bash
git add uav_mec/publish_canonical.py uav_mec/generate_figures_v2.py tests/test_canonical_publication.py tests/test_figure_generation_v2.py
git commit -m "feat: generate thesis assets from canonical evidence"
```

---

### Task 24: Run Exploratory And Frozen Confirmatory Experiments

**Files:**
- Create: `artifacts/exploratory/exploratory-v1/` (ignored raw bundle)
- Create: `artifacts/runs/confirmatory-v1/` (ignored raw bundles)
- Create: `artifacts/canonical/v1/` (tracked lightweight evidence)
- Generate: `figures/thesis/*.{pdf,png}`
- Generate: `paper/generated_results.tex`
- Generate: `paper/generated_statistics.tex`
- Generate: `paper/generated_robustness.tex`

- [ ] **Step 1: Run the complete automated test gate before spending experiment time**

Run: `python -m pytest -q`

Expected: PASS with no failures, warnings promoted by project configuration, duplicate task IDs, or nondeterministic regressions.

- [ ] **Step 2: Run the frozen exploratory seeds only**

Run: `python -m uav_mec.experiment --config configs/thesis/exploratory.toml --output artifacts/exploratory/exploratory-v1`

Expected: exit 0 and a checksum-complete bundle for seeds `42..61`. These results may diagnose correctness or runtime, but may not be quoted as confirmatory evidence.

- [ ] **Step 3: Validate exploratory invariants without inspecting confirmatory seeds**

Run: `python -c "from pathlib import Path; from uav_mec.analyze_results import load_bundle,validate_bundle; print(validate_bundle(load_bundle('exploratory',Path('artifacts/exploratory/exploratory-v1'))))"`

Expected: prints `sic_violations: 0` and `cpu_budget_violations: 0`; task and slot traces are paired across all methods.

- [ ] **Step 4: Record the exact pre-confirmatory revision**

Run: `git status --short`

Expected: no uncommitted source/config changes. Raw exploratory output is ignored. If source or config changes are required to fix a correctness failure, commit them, rerun Steps 1-3 from a new output directory, and record that new Git SHA; do not inspect seeds `1000..1049` first.

- [ ] **Step 5: Run the three independent confirmatory matrices**

Run: `python -m uav_mec.experiment --config configs/thesis/confirmatory_main.toml --output artifacts/runs/confirmatory-v1/main`

Expected: exit 0; all eight methods finish seeds `1000..1049` on identical traces.

Run: `python -m uav_mec.experiment --config configs/thesis/confirmatory_pairing.toml --output artifacts/runs/confirmatory-v1/pairing`

Expected: exit 0; the strict 2x2 pairing matrix finishes seeds `1000..1049`.

Run: `python -m uav_mec.experiment --config configs/thesis/confirmatory_resource.toml --output artifacts/runs/confirmatory-v1/resource`

Expected: exit 0; only the resource-allocation policy differs across the three methods.

- [ ] **Step 6: Run complete small-instance enumeration**

Run: `python -m uav_mec.run_exact_benchmark --config configs/thesis/confirmatory_exact.toml --output artifacts/runs/confirmatory-v1/exact`

Expected: exit 0; every row has `complete_search=True` and `full_completed <= exact_completed`.

- [ ] **Step 7: Run mechanism, scalability, and robustness suites**

Run: `python -m uav_mec.run_thesis_suite --suite mechanism --output artifacts/runs/confirmatory-v1/mechanism`

Expected: exit 0 with `coupling0`, `coupling05`, and `coupling1` bundles.

Run: `python -m uav_mec.run_thesis_suite --suite scalability --output artifacts/runs/confirmatory-v1/scalability`

Expected: exit 0 with all `N=10..50` and `K=1..5` bundles.

Run: `python -m uav_mec.run_thesis_suite --suite robustness --output artifacts/runs/confirmatory-v1/robustness`

Expected: exit 0 with every predeclared noise, CSI, SIC, CPU, deadline, and task-load case.

- [ ] **Step 8: Compute formal statistics and enforce all thesis gates**

Run: `python -m uav_mec.analyze_results --main artifacts/runs/confirmatory-v1/main --pairing artifacts/runs/confirmatory-v1/pairing --resource artifacts/runs/confirmatory-v1/resource --exact artifacts/runs/confirmatory-v1/exact --output artifacts/runs/confirmatory-v1/analysis --require-thesis-gates`

Expected: exit 0. `acceptance.json` must report all of the following as true: zero constraint violations; positive 95% CIs for cross-UAV search, global acceptance, and deadline allocation; the full method beats the predeclared strongest controlled baseline (`exchange_stable`); `N=20` P95 runtime is below one second; complete enumeration is complete. All other controlled comparisons are still reported, even when not thesis gates.

If this command exits nonzero, stop here. Preserve all failed-run artifacts, do not publish canonical evidence, do not select favorable seeds, and return to the algorithm task named by the failed gate before running a new versioned confirmatory directory.

- [ ] **Step 9: Publish only lightweight passed evidence**

Run: `python -m uav_mec.publish_canonical --main artifacts/runs/confirmatory-v1/main --pairing artifacts/runs/confirmatory-v1/pairing --resource artifacts/runs/confirmatory-v1/resource --exact artifacts/runs/confirmatory-v1/exact --analysis artifacts/runs/confirmatory-v1/analysis --mechanism artifacts/runs/confirmatory-v1/mechanism --scalability artifacts/runs/confirmatory-v1/scalability --robustness artifacts/runs/confirmatory-v1/robustness --output artifacts/canonical/v1`

Expected: exit 0. The canonical tree contains only JSON evidence, manifests, an index, and checksums; it contains no raw `tasks.csv`, `slots.csv`, checkpoints, or models.

- [ ] **Step 10: Generate every paper number and primary figure from canonical evidence**

Run: `python -m uav_mec.generate_figures_v2 --canonical artifacts/canonical/v1 --figures figures/thesis --macros paper/generated_results.tex --statistics-table paper/generated_statistics.tex --robustness-table paper/generated_robustness.tex`

Expected: exit 0; main, two ablation, mechanism, and two scalability figures exist in PDF and PNG, while both generated TeX files contain no hand-entered result values.

- [ ] **Step 11: Revalidate canonical checksums and asset tests**

Run: `python -c "from pathlib import Path; from uav_mec.analyze_results import verify_checksums; verify_checksums(Path('artifacts/canonical/v1')); print('canonical checksums: OK')"`

Expected: prints `canonical checksums: OK`.

Run: `python -m pytest tests/test_result_analysis.py tests/test_canonical_publication.py tests/test_figure_generation_v2.py -v`

Expected: PASS, 4 tests passed.

- [ ] **Step 12: Commit only canonical evidence and generated assets**

```bash
git add artifacts/canonical/v1 figures/thesis paper/generated_results.tex paper/generated_statistics.tex paper/generated_robustness.tex
git diff --cached --check
git commit -m "data: publish confirmatory thesis evidence"
```

Expected: the commit excludes `artifacts/runs/` and `artifacts/exploratory/` but includes the canonical manifest and checksums.

---

### Task 25: Rewrite The Professional Thesis, Paper, Defense Pack, And Reproduction Path

**Files:**
- Modify: `paper/main_v2.tex`
- Modify: `paper/refs_v2.bib`
- Modify: `README.md`
- Create: `requirements-lock.txt`
- Create: `thesis/main.tex`
- Create: `thesis/chapters/01-introduction.tex`
- Create: `thesis/chapters/02-related-work.tex`
- Create: `thesis/chapters/03-system-model.tex`
- Create: `thesis/chapters/04-interference-aware-pairing.tex`
- Create: `thesis/chapters/05-deadline-allocation.tex`
- Create: `thesis/chapters/06-implementation-experiments.tex`
- Create: `thesis/chapters/07-deployment-limitations.tex`
- Create: `thesis/chapters/08-conclusion.tex`
- Create: `thesis/claims-matrix.md`
- Create: `defense/outline.md`
- Create: `defense/innovation-comparison.md`
- Create: `defense/risk-questions.md`
- Create: `tests/test_active_thesis_claims.py`
- Create: `tests/test_latex_citations.py`
- Create: `Makefile`
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write failing active-claim and document-structure tests**

```python
from pathlib import Path
import re


ACTIVE_PROSE = (
    Path("README.md"),
    Path("paper/main_v2.tex"),
    *sorted(Path("thesis/chapters").glob("*.tex")),
    Path("thesis/claims-matrix.md"),
    Path("defense/outline.md"),
    Path("defense/innovation-comparison.md"),
    Path("defense/risk-questions.md"),
)


def active_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in ACTIVE_PROSE if path.exists())


def test_active_documents_remove_frozen_or_unsupported_claims():
    text = active_text()
    forbidden = (
        r"73\.4\\?%", r"\+11\.3", r"\+2\.6", r"94\.8\\?%",
        r"NP[- ]complete", r"NP 完全", r"the first to", r"首次.*提出",
    )
    for pattern in forbidden:
        assert not re.search(pattern, text, flags=re.IGNORECASE), pattern
    assert not re.search(r"TODO|TBD|FIXME|待补|占位", text, flags=re.IGNORECASE)


def test_professional_thesis_has_two_and_only_two_named_innovations():
    comparison = Path("defense/innovation-comparison.md").read_text(encoding="utf-8")
    assert comparison.count("## 创新点") == 2
    assert "干扰外部性感知的跨 UAV 联合关联与配对" in comparison
    assert "面向截止时间的卸载与 MEC 算力分配" in comparison


def test_thesis_has_eight_chapters_and_uses_generated_evidence():
    chapters = sorted(Path("thesis/chapters").glob("*.tex"))
    assert len(chapters) == 8
    text = active_text()
    for macro in (
        "\\FullOnTimePercent", "\\InnovationOneCrossGainPP",
        "\\InnovationTwoGainPP", "\\ExactOptimalPercent", "\\FullRuntimeP95Ms",
    ):
        assert macro in text
    main = Path("thesis/main.tex").read_text(encoding="utf-8")
    assert "\\input{paper/generated_results.tex}" in main
    assert "\\bibliography{paper/refs_v2}" in main


def test_reproduction_lock_is_exact_and_portable():
    lines = [
        line.strip()
        for line in Path("requirements-lock.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    assert all("==" in line for line in lines)
    assert all("file://" not in line and "-e " not in line for line in lines)
```

- [ ] **Step 2: Write the failing citation-integrity test**

```python
from pathlib import Path
import re


def test_every_active_citation_has_a_bibliography_entry():
    tex_files = [Path("paper/main_v2.tex"), *sorted(Path("thesis/chapters").glob("*.tex"))]
    cited = set()
    for path in tex_files:
        for group in re.findall(r"\\cite\{([^}]+)\}", path.read_text(encoding="utf-8")):
            cited.update(key.strip() for key in group.split(","))
    bibliography = Path("paper/refs_v2.bib").read_text(encoding="utf-8")
    defined = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
    assert cited
    assert not (cited - defined), f"undefined citation keys: {sorted(cited - defined)}"
```

- [ ] **Step 3: Run document tests and verify they fail on the legacy claims and missing thesis**

Run: `python -m pytest tests/test_active_thesis_claims.py tests/test_latex_citations.py -v`

Expected: FAIL because active prose contains frozen numbers/NP-complete claims and the eight-chapter thesis/defense files do not exist.

- [ ] **Step 4: Audit and normalize the bibliography against primary sources**

For every entry retained in `paper/refs_v2.bib`, verify title, author list, venue, year, volume/pages, and DOI against the publisher page or the paper itself. Remove uncited entries and add only primary sources actually cited by the new text. Keep these required topic groups represented:

```text
MEC survey and computation offloading
UAV communications and UAV-enabled MEC
uplink NOMA/SIC and grouped user pairing
multi-UAV interference and matching externalities
deadline-aware MEC resource allocation
paired bootstrap, Wilcoxon, Holm, and rank-biserial statistics
```

Do not add a citation merely to support a novelty superlative; the active manuscripts make no “first” claim.

- [ ] **Step 5: Replace the IEEE manuscript with the evidence-backed article**

Rewrite `paper/main_v2.tex` under the title:

```latex
\title{Interference-Externality-Aware Cross-UAV Pairing and Deadline-Driven Resource Allocation for Full-Reuse NOMA-MEC Networks}
```

The complete active article must use this section order and evidence contract:

```latex
\input{paper/generated_results.tex}

\begin{abstract}
Full-frequency-reuse multi-UAV NOMA-MEC systems couple user association and pairing through
cross-UAV interference while sharing finite edge CPU under per-task deadlines. We first develop
an interference-externality-aware scheduler whose Replace, Position, and CrossUAV neighborhoods
recompute the actual serving channel, hard SIC feasibility, cross-UAV interference, and per-UAV
MEC load under a deterministic global first-improvement rule. We then derive the minimum offload
ratio and minimum MEC frequency required by each deadline and use their ordered prefix to maximize
the number of completed tasks under each UAV CPU budget. A unified simulator counts every arriving
task once and evaluates all methods on identical immutable traces. Across \ConfirmatorySeedCount{}
unseen seeds, the full method reaches \FullOnTimePercent{} on-time completion; the isolated cross-UAV
and deadline-allocation contributions are \InnovationOneCrossGainPP{} and \InnovationTwoGainPP{}
percentage points, respectively. The final schedules contain \SICViolationCount{} SIC violations,
and the measured $N=20$ scheduling P95 is \FullRuntimeP95Ms{} ms.
\end{abstract}

\section{Introduction}
\section{Related Work}
\section{System Model and Problem Formulation}
\section{Interference-Externality-Aware Association and Pairing}
\section{Deadline-Driven Offloading and MEC Allocation}
\section{Implementation and Experimental Protocol}
\section{Results and Mechanism Analysis}
\section{Engineering Implications and Limitations}
\section{Conclusion}
```

Write finished prose for every listed section. The results section must use the generated point-estimate and CI macros for full performance, both Innovation 1 components, Innovation 2, and the strong baseline (`\FullOnTimePercent` through `\StrongBaselineCIHighPP`), plus `\ExactOptimalPercent`, `\ExactMeanCompletionGap`, `\FullRuntimeP95Ms`, and `\SICViolationCount`; include `paper/generated_statistics.tex`, `paper/generated_robustness.tex`, and the generated figures. Describe complete enumeration as a small-instance empirical reference, not a general optimality guarantee. Move PPO/MAPPO to one historical-scope sentence and do not present them as core baselines.

- [ ] **Step 6: Create a compilable eight-chapter professional-master thesis**

Create `thesis/main.tex` as the portable content manuscript below. Institution-specific cover/authorization pages can later wrap this body without changing its evidence or chapters:

```latex
\documentclass[UTF8,openany]{ctexbook}
\usepackage[a4paper,margin=2.7cm]{geometry}
\usepackage{amsmath,amssymb,amsthm,bm}
\usepackage{graphicx,booktabs,multirow}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{hyperref}
\usepackage[numbers,sort&compress]{natbib}
\newtheorem{proposition}{命题}[chapter]
\newtheorem{theorem}{定理}[chapter]
\input{paper/generated_results.tex}
\title{面向全频复用多无人机 NOMA-MEC 网络的干扰感知用户配对与截止时间资源分配研究}
\author{}
\date{2026年}
\begin{document}
\frontmatter
\maketitle
\chapter*{摘要}
本文面向全频复用多无人机分组 NOMA-MEC 网络，研究跨无人机关联与用户配对造成的干扰外部性，以及任务截止时间约束下卸载比例和边缘算力的联合分配问题。本文形成两项相互独立且可组合的算法创新，并通过固定外生轨迹、未见随机种子、完整小规模枚举和可追溯实验产物进行验证。
\chapter*{Abstract}
This professional-master thesis studies interference externalities in cross-UAV association and grouped-NOMA pairing, together with deadline-constrained offloading and MEC CPU allocation, under full frequency reuse. Two independently ablated algorithmic contributions are evaluated through immutable paired traces, unseen confirmatory seeds, complete small-instance enumeration, and checksum-audited artifacts.
\tableofcontents
\mainmatter
\input{thesis/chapters/01-introduction.tex}
\input{thesis/chapters/02-related-work.tex}
\input{thesis/chapters/03-system-model.tex}
\input{thesis/chapters/04-interference-aware-pairing.tex}
\input{thesis/chapters/05-deadline-allocation.tex}
\input{thesis/chapters/06-implementation-experiments.tex}
\input{thesis/chapters/07-deployment-limitations.tex}
\input{thesis/chapters/08-conclusion.tex}
\backmatter
\bibliographystyle{plainnat}
\bibliography{paper/refs_v2}
\end{document}
```

Write the eight chapter files with these exact chapter/section responsibilities; all prose must be complete and citation-backed:

```text
01 绪论：工程背景、问题边界、专业硕士价值、两项创新、论文结构。
02 相关工作：MEC、UAV-MEC、分组 NOMA、匹配外部性、截止时间分配；用对比表明确本文边界，不声称“首次”。
03 系统模型：唯一任务语义、显式 Schedule、真实服务 UAV、全频复用 SINR、硬 SIC、时延、CPU、词典序目标和全部约束。
04 创新点一：ILP 初始化、SIC 修复、Replace/Position/CrossUAV 邻域、全局与双边准则、有限终止证明、复杂度与信令开销、真实机制案例。
05 创新点二：alpha_min 与 f_min 完整推导、分母不可行条件、按 f_min 排序的最大基数证明、与 crossing/equal 和 fixed-half/equal 的边界对照。
06 系统实现与实验：唯一数据流、配置、50 个确认种子、2x2/资源消融、基线、公平 OFDMA、机制负控制、扩展性、鲁棒性、配对统计、完整枚举；所有结果引用生成宏或生成图表。
07 工程部署与局限：集中式 CSI 假设、协调开销、CSI 误差、能耗/轨迹/功率未联合优化、外部效度、部署步骤和后续工作。
08 总结与展望：只总结两项创新及确认性证据，明确不把轨迹、功率、DRL 或复杂性证明包装成额外创新。
```

Chapter 5 must include these exact mathematical statements rather than the legacy crossing-point optimum claim:

```latex
\alpha_i^{\min}=\max\!\left(0,1-\frac{T_i f_i^{\mathrm{loc}}}{C_i}\right),
\qquad
f_i^{\min}=\frac{\alpha_i^{\min}C_i}
{T_i-\alpha_i^{\min}(1+\rho)D_i/R_i}.
```

It must state that a non-positive denominator is infeasible, and prove by an exchange argument that selecting tasks in nondecreasing `f_i^{\min}` maximizes completed-task cardinality under a single UAV CPU budget.

- [ ] **Step 7: Create the claim matrix and defense materials**

Create `thesis/claims-matrix.md` with one row per quantitative claim and these columns:

```markdown
| Claim ID | Active claim | Generated macro/figure | Canonical source | Statistical gate | Scope limitation |
|---|---|---|---|---|---|
```

Include rows for both innovation effects, every controlled baseline comparison, exact-search gap, zero SIC violations, and `N=20` P95 runtime. No row may cite `results/` or an exploratory artifact.

Create `defense/innovation-comparison.md` with exactly two `## 创新点` headings. For each, give: engineering pain point, existing-method limitation, mathematical idea, algorithm mechanism, isolated ablation, complexity/overhead, evidence macro, and limitation. End with a table showing that Innovation 1 changes scheduling topology/acceptance while Innovation 2 changes offload/CPU allocation, so neither result is obtained by silently changing the other component.

Create `defense/outline.md` as a 20-slide defense sequence: scene and problem (1-4), model and credibility repairs (5-6), Innovation 1 (7-10), Innovation 2 (11-13), implementation and protocol (14-15), results and mechanism (16-18), deployment/limitations (19), conclusion (20). Each result slide must name its canonical figure or macro.

Create `defense/risk-questions.md` with concise evidence-backed answers to at least these 15 questions:

```text
为什么是专业硕士价值而非纯仿真拼接？
两个创新点如何彼此独立？
为什么跨 UAV 后必须改变实际服务 UAV？
为什么 SIC 阈值是系统参数而不是 Shannon 下限？
为什么旧结果无效，如何防止重复计数？
为什么不用旧 NP-complete 证明？
全局准则是否牺牲个体稳定性？
Replace、Position、CrossUAV 各解决什么局部最优？
算法为什么有限终止？
alpha_min 和 f_min 的适用条件是什么？
排序分配为何达到最大任务数？
与 Fair OFDMA、IA-Greedy、Exchange-Stable 的公平性如何保证？
为什么确认性种子不能调参？
CSI 不完美、SIC 残差和噪声变化下是否稳健？
集中式算法的协调开销与部署限制是什么？
```

- [ ] **Step 8: Rewrite the README around the trusted pipeline**

Replace the legacy result-first README with these top-level sections, in order:

```markdown
# Full-Reuse Multi-UAV NOMA-MEC Thesis Project
## Research Question
## Two Contributions
## Trusted Data Flow
## Repository Layout
## Quick Verification
## Full Confirmatory Reproduction
## Canonical Evidence And Paper Assets
## Thesis And Paper Builds
## Legacy Scope
## Known Limitations
```

`Quick Verification` must contain `python -m pytest -q`, the smoke experiment command using a fresh output directory, canonical checksum verification, asset generation, and both Tectonic build commands. `Full Confirmatory Reproduction` must reproduce the exact Task 24 commands and warn that outputs are fail-fast/non-overwriting. Mark `results/`, old figures, old PDFs, DRL runners, and checkpoints as legacy; do not delete them and do not quote them as evidence.

- [ ] **Step 9: Build an isolated dependency-lock environment**

Run: `python -m venv .venv-lock`

Expected: exit 0 and a new ignored `.venv-lock/` environment.

Run: `.venv-lock/bin/python -m pip install --upgrade pip`

Expected: exit 0.

Run: `.venv-lock/bin/python -m pip install -e ".[dev]"`

Expected: exit 0 with the project and all runtime/test dependencies installed from declared ranges.

- [ ] **Step 10: Freeze and validate exact Python dependencies**

Run: `.venv-lock/bin/python -m pip freeze --exclude-editable > requirements-lock.txt`

Expected: `requirements-lock.txt` contains only exact `name==version` entries and no local project path.

Run: `.venv-lock/bin/python -c "from pathlib import Path; lines=[x for x in Path('requirements-lock.txt').read_text().splitlines() if x and not x.startswith('#')]; assert lines and all('==' in x and 'file://' not in x for x in lines); print(len(lines), 'locked packages')"`

Expected: prints a positive locked-package count.

- [ ] **Step 11: Add one-command verification and CI**

Create `Makefile`:

```makefile
PYTHON ?= python
TECTONIC ?= tectonic
CANONICAL ?= artifacts/canonical/v1

.PHONY: test assets paper thesis verify

test:
	$(PYTHON) -m pytest -q

assets:
	$(PYTHON) -m uav_mec.generate_figures_v2 --canonical $(CANONICAL) --figures figures/thesis --macros paper/generated_results.tex --statistics-table paper/generated_statistics.tex --robustness-table paper/generated_robustness.tex

paper: assets
	mkdir -p build/paper
	$(TECTONIC) -X compile paper/main_v2.tex --outdir build/paper

thesis: assets
	mkdir -p build/thesis
	$(TECTONIC) -X compile thesis/main.tex --outdir build/thesis

verify: test paper thesis
```

Create `.github/workflows/ci.yml`:

```yaml
name: thesis-verification
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -r requirements-lock.txt
      - run: python -m pip install -e . --no-deps
      - run: python -m pytest -q
```

CI verifies code and evidence-contract tests; local `make verify` additionally compiles both manuscripts because Tectonic installation is platform-managed.

- [ ] **Step 12: Run active-claim, citation, lock, and full test gates**

Run: `python -m pytest tests/test_active_thesis_claims.py tests/test_latex_citations.py -v`

Expected: PASS, 5 tests passed; no active file contains a frozen or unsupported claim, every citation key exists, and the dependency lock is exact.

Run: `python -m pytest -q`

Expected: PASS with the complete trusted pipeline and document tests.

- [ ] **Step 13: Compile the article and degree-thesis manuscript**

Run: `tectonic -X compile paper/main_v2.tex --outdir build/paper`

Expected: exit 0 and `build/paper/main_v2.pdf`; no undefined references or citations remain in the log.

Run: `tectonic -X compile thesis/main.tex --outdir build/thesis`

Expected: exit 0 and `build/thesis/main.pdf`; all eight chapters, generated values, figures, algorithms, and bibliography render.

- [ ] **Step 14: Verify from a second clean environment**

Run: `python -m venv /tmp/uav-mec-thesis-clean-v1`

Expected: exit 0; the path must not already exist.

Run: `/tmp/uav-mec-thesis-clean-v1/bin/python -m pip install -r requirements-lock.txt`

Expected: exit 0 with only locked packages.

Run: `/tmp/uav-mec-thesis-clean-v1/bin/python -m pip install -e . --no-deps`

Expected: exit 0.

Run: `/tmp/uav-mec-thesis-clean-v1/bin/python -m pytest -q`

Expected: PASS for the complete suite from the clean interpreter.

Run: `/tmp/uav-mec-thesis-clean-v1/bin/python -m uav_mec.generate_figures_v2 --canonical artifacts/canonical/v1 --figures figures/thesis --macros paper/generated_results.tex --statistics-table paper/generated_statistics.tex --robustness-table paper/generated_robustness.tex`

Expected: exit 0 without reading `results/`, checkpoints, or exploratory artifacts.

- [ ] **Step 15: Run the documented reproduction target**

Run: `make verify`

Expected: tests pass, canonical assets regenerate byte-for-byte in content, and both PDFs compile without reading legacy `results/`.

- [ ] **Step 16: Audit the final diff and commit the defense-ready source**

Run: `rg -n '73\.4|11\.3|2\.6|94\.8|NP[- ]complete|NP 完全|the first to|首次.*提出|TODO|TBD|FIXME|待补|占位' README.md paper/main_v2.tex thesis defense`

Expected: no matches.

Run: `git diff --check`

Expected: no whitespace errors.

```bash
git add README.md requirements-lock.txt Makefile .github/workflows/ci.yml paper/main_v2.tex paper/refs_v2.bib thesis defense tests/test_active_thesis_claims.py tests/test_latex_citations.py
git commit -m "docs: rebuild professional thesis and defense package"
```

---

## Final Acceptance Checklist

- [ ] `python -m pytest -q` passes from the committed revision.
- [ ] All final schedules have zero structural, SIC, duplicate-task, and CPU-budget violations.
- [ ] Same-seed traces and results are deterministic; methods use identical exogenous tasks and channels.
- [ ] Innovation 1 cross-UAV and acceptance effects each have paired 95% CI above zero.
- [ ] Innovation 2 deadline-allocation effect has paired 95% CI above zero.
- [ ] The full method beats the predeclared strongest controlled baseline; every other controlled and Fair OFDMA comparison is reported without hiding unfavorable results.
- [ ] Complete enumeration is truly complete for the frozen small-instance search space and the heuristic never exceeds it.
- [ ] `N=20` P95 scheduling runtime is below one second.
- [ ] Canonical checksums pass and no active manuscript reads legacy result files.
- [ ] Every quantitative paper/thesis claim resolves to a generated macro, generated table, or canonical figure.
- [ ] Both Tectonic builds succeed and the defense pack names exactly two innovations.

If any evidence checklist item fails, the implementation is not defense-ready. Preserve the evidence, report the failed gate, and return to the responsible algorithm or experiment task; never repair a claim by choosing seeds or manually editing a generated number.
