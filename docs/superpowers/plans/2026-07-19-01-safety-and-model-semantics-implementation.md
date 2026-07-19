# Safety and Model Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the current research state and establish one unit-safe, policy-explicit model for task identity, release/deadline time, task-level result ratio, channel provenance, per-layer SIC, admission closure, canonical latency, and frozen realized evaluation.

**Architecture:** `Task` owns job identity and all task-specific timing/traffic data. `uav_mec/latency.py` is the only active latency formula. Every radio evaluation requires a `ChannelBasis`; planning uses estimated gains and realized replay uses true gains without changing the frozen association, radio position, decode order, offload ratio, or CPU reservation. `uav_mec/closure.py` is the single raw-assignment-to-executable-plan boundary. Cache keys include every semantic input. The old delay helpers in `pairing.py` and `model.py` remain clearly isolated legacy code and are unreachable from the thesis-v2 pipeline.

**Tech Stack:** Python 3.11 dataclasses and enums, NumPy, pytest, TOML, Git worktrees, SHA-256.

**Execution rule:** Task 1 runs the approved baseline builds and minimal smoke checks. Pilot and formal runs remain forbidden throughout this plan. After Task 1, run only tests, static validation, and document/configuration builds; do not launch another experiment or publish evidence.

---

### Task 1: Protect generated evidence, checkpoint source, and enter an isolated worktree

**Files:**
- Modify: `.gitignore`
- Create: `docs/audits/2026-07-19-partial-pilot-provenance.md`
- Create: `docs/audits/2026-07-19-research-checkpoint-allowlist.txt`
- Test: repository-level shell checks below

- [ ] **Step 1: Run RED safety check**

```bash
cd /home/link/my_workspace/Master-Thesis
test ! -e docs/audits/2026-07-19-partial-pilot-provenance.md
git check-ignore -q artifacts/pilot/power-v2-20260717/main/shard_index.json
```

Expected: the first command succeeds and the second exits `1`; the current repository has no committed provenance record and does not yet protect pilot outputs.

- [ ] **Step 2: Prove no pilot process is active and capture immutable evidence**

```bash
cd /home/link/my_workspace/Master-Thesis
! tmux has-session -t thesis-power-pilot
find artifacts/pilot/power-v2-20260717/main/shards -mindepth 1 -maxdepth 1 -type d | wc -l
find artifacts/pilot/power-v2-20260717/externality/shards -mindepth 1 -maxdepth 1 -type d | wc -l
sha256sum artifacts/pilot/power-v2-20260717/main/shard_index.json
sha256sum artifacts/pilot/power-v2-20260717/externality/shard_index.json
git rev-parse HEAD
```

Expected at plan authoring time: Main has `100` shard directories and Externality has `48`. If either count changed, stop and classify the new state before modifying source. Record the two printed hashes and the printed commit SHA in the audit.

- [ ] **Step 3: Add the safety metadata**

Add only these ignore rules; do not add a global PDF or Office-file rule:

```gitignore
artifacts/pilot/
artifacts/smoke/
artifacts/preregistration/
artifacts/formal/
artifacts/runs/v2-confirmatory/
artifacts/calibration/
artifacts/analysis/
artifacts/canonical/v2/
build/
```

The audit must contain this exact normative block followed by the five command outputs from Step 2:

```text
classification: exploratory_incomplete
confirmatory_use: forbidden
power_lock_use: forbidden
canonical_v2_use: forbidden
resume_or_pool_as_same_run: forbidden
reason: Main completed, Externality partial, Resource/Joint absent, and no
        start-time source snapshot proves a common final implementation state.
```

Write one repository-relative source path per line in the allowlist. The list may contain only files visible in `git status --short`; it must exclude `artifacts/`, `deliverables/defense_2026/`, `figures/`, `paper/generated_robustness.tex`, `paper/generated_statistics.tex`, binaries, logs, and caches.

- [ ] **Step 4: Run GREEN safety checks**

```bash
cd /home/link/my_workspace/Master-Thesis
git check-ignore artifacts/pilot/power-v2-20260717/main/shard_index.json
git check-ignore artifacts/smoke/check/summary.json
git check-ignore artifacts/preregistration/freeze.json
git check-ignore artifacts/formal/example/manifest.json
git check-ignore artifacts/runs/v2-confirmatory/example/manifest.json
git check-ignore artifacts/calibration/example/summary.json
git check-ignore artifacts/analysis/example/metrics.json
git check-ignore artifacts/canonical/v2/full/MANIFEST.json
/home/link/miniconda/envs/llm-ops-torch26/bin/python - <<'PY'
from pathlib import Path

root = Path.cwd().resolve()
allowlist = Path("docs/audits/2026-07-19-research-checkpoint-allowlist.txt")
paths = allowlist.read_text(encoding="utf-8").splitlines()
forbidden_roots = {"artifacts", "deliverables", "figures", "build", ".cache", "__pycache__"}
forbidden_names = {"generated_robustness.tex", "generated_statistics.tex"}
forbidden_suffixes = {
    ".aux", ".docx", ".jpeg", ".jpg", ".log", ".out", ".pdf", ".png",
    ".pptx", ".pyc", ".synctex.gz", ".toc",
}
maximum_source_bytes = 5 * 1024 * 1024
assert paths
assert paths == sorted(set(paths))
for raw in paths:
    path = Path(raw)
    assert raw == path.as_posix()
    assert not path.is_absolute() and ".." not in path.parts
    assert path.parts[0] not in forbidden_roots
    assert not any(part in {".cache", "__pycache__"} for part in path.parts)
    assert path.name not in forbidden_names
    suffix = ".synctex.gz" if path.name.lower().endswith(".synctex.gz") else path.suffix.lower()
    assert suffix not in forbidden_suffixes
    resolved = root / path
    assert resolved.is_file(), raw
    assert resolved.stat().st_size <= maximum_source_bytes, raw
print(f"validated_allowlist={len(paths)}")
PY
```

Expected: all eight ignored paths print, `validated_allowlist` is positive, and all commands exit `0`. `artifacts/canonical/v1/` remains tracked and is not covered by the new canonical-v2 rule.

- [ ] **Step 5: Commit the explicit files**

```bash
cd /home/link/my_workspace/Master-Thesis
git add -- .gitignore docs/audits/2026-07-19-partial-pilot-provenance.md docs/audits/2026-07-19-research-checkpoint-allowlist.txt
git diff --cached --name-only
git commit -m "docs: freeze incomplete pilot provenance"
```

Expected staged names: exactly the three paths above.

- [ ] **Step 6: Checkpoint only the explicit allowlist**

```bash
cd /home/link/my_workspace/Master-Thesis
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest -q
while IFS= read -r checkpoint_path; do
  git add -- "$checkpoint_path"
done < docs/audits/2026-07-19-research-checkpoint-allowlist.txt
/home/link/miniconda/envs/llm-ops-torch26/bin/python - <<'PY'
import subprocess
from pathlib import Path

allowed = set(Path("docs/audits/2026-07-19-research-checkpoint-allowlist.txt").read_text(encoding="utf-8").splitlines())
staged = set(subprocess.check_output(["git", "diff", "--cached", "--name-only"], text=True).splitlines())
assert staged
assert staged <= allowed, sorted(staged - allowed)
print(f"staged_allowlisted={len(staged)}")
PY
git diff --cached --check
git commit -m "chore: checkpoint thesis v2 research source"
```

Expected: the machine check exits `0`; no generated or experimental output is committed.

- [ ] **Step 7: Create and verify the isolated worktree**

```bash
cd /home/link/my_workspace/Master-Thesis
checkpoint_sha=$(git rev-parse HEAD)
[[ "$checkpoint_sha" =~ ^[0-9a-f]{40}$ ]]
! git rev-parse --verify --quiet refs/tags/thesis-v2-research-checkpoint-20260719
git tag -a thesis-v2-research-checkpoint-20260719 "$checkpoint_sha" -m "Thesis-v2 selective research checkpoint 2026-07-19"
test "$(git rev-parse thesis-v2-research-checkpoint-20260719^{commit})" = "$checkpoint_sha"
mkdir -p /home/link/my_workspace/.worktrees
git worktree add /home/link/my_workspace/.worktrees/master-thesis-ccfb -b feature/ustc-ccfb-v2 "$checkpoint_sha"
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
test "$(git rev-parse HEAD)" = "$checkpoint_sha"
test -z "$(git status --porcelain)"
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest -q
/home/link/miniconda/envs/llm-ops-torch26/bin/python -c "from pathlib import Path; from uav_mec.analyze_results import verify_checksums; verify_checksums(Path('artifacts/canonical/v1')); print('canonical-v1 checksums: OK')"
mkdir -p build/worktree-baseline-thesis build/worktree-baseline-paper
tectonic -X compile thesis/main.tex --outdir build/worktree-baseline-thesis --reruns 2
tectonic -X compile paper/main_v2.tex --outdir build/worktree-baseline-paper --reruns 2
make smoke PYTHON=/home/link/miniconda/envs/llm-ops-torch26/bin/python RUN_TAG=worktree-baseline
make exact-smoke PYTHON=/home/link/miniconda/envs/llm-ops-torch26/bin/python RUN_TAG=worktree-baseline
test -z "$(git status --short --untracked-files=all)"
```

Expected: the annotated tag resolves byte-for-byte to the recorded checkpoint SHA; the checkpoint suite and canonical-v1 checksum verification pass; both manuscripts compile without regenerating tracked assets; both minimal smoke bundles finalize with valid checksums; and the final status is clean because build/smoke outputs are ignored. The smoke output is baseline diagnostics only and must not enter a pilot, formal analysis, or canonical bundle. Leave all excluded files untouched in the original checkout. If a baseline command fails, diagnose it before Task 2; do not normalize an unexplained existing failure.

---

### Task 2: Make task identity, release time, slot duration, deadline, and rho unit-safe

**Files:**
- Modify: `uav_mec/domain.py`
- Modify: `uav_mec/trace.py`
- Modify: `uav_mec/model.py`
- Modify: `tests/test_domain.py`
- Modify: `tests/test_trace.py`

- [ ] **Step 1: Write the RED domain tests**

Add these imports and tests to `tests/test_domain.py`:

```python
import numpy as np
import pytest

from uav_mec.domain import SlotSnapshot, Task, TaskIdentity


def _timed_task(*, task_id: int = 31, user_id: int = 7, job_index: int = 2) -> Task:
    return Task(
        task_id=task_id,
        user_id=user_id,
        arrival_slot=3,
        data_bits=1_000_000.0,
        comp_cycles=1_400_000_000.0,
        deadline_s=0.20,
        local_cpu_hz=1_000_000_000.0,
        job_index=job_index,
        slot_duration_s=0.25,
        result_ratio=0.15,
    )


def test_task_identity_and_times_are_in_seconds() -> None:
    task = _timed_task()
    assert task.identity == TaskIdentity(user_id=7, job_index=2)
    assert task.release_time_s == pytest.approx(0.75)
    assert task.remaining_time_s == pytest.approx(0.20)
    assert task.absolute_deadline_s == pytest.approx(0.95)
    assert task.result_ratio == pytest.approx(0.15)
    assert task.absolute_deadline_s != task.arrival_slot + task.deadline_s


@pytest.mark.parametrize("field,value", [("slot_duration_s", 0.0), ("result_ratio", -0.1)])
def test_task_rejects_invalid_slot_duration_or_result_ratio(field: str, value: float) -> None:
    values = {
        "task_id": 31,
        "user_id": 7,
        "arrival_slot": 3,
        "data_bits": 1_000_000.0,
        "comp_cycles": 1_400_000_000.0,
        "deadline_s": 0.20,
        "local_cpu_hz": 1_000_000_000.0,
        "job_index": 2,
        "slot_duration_s": 0.25,
        "result_ratio": 0.15,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        Task(**values)


def test_snapshot_rejects_duplicate_job_identity() -> None:
    first = _timed_task(task_id=31)
    second = _timed_task(task_id=32)
    with pytest.raises(ValueError, match="identity"):
        SlotSnapshot(3, (first, second), np.ones((2, 1)), 1)
```

Add these imports and this parameterized trace test to `tests/test_trace.py`:

```python
from uav_mec.model import ScenarioConfig, TaskConfig, UAVConfig
from uav_mec.trace import generate_trace
import pytest


@pytest.mark.parametrize("arrival_mode", ("periodic", "bernoulli"))
def test_every_arrival_mode_materializes_stable_job_identity_and_task_semantics(
    arrival_mode: str,
) -> None:
    scenario = ScenarioConfig(
        num_users=4,
        num_slots=4,
        uav=UAVConfig(dt=0.25),
        task=TaskConfig(
            arrival_mode=arrival_mode,
            arrival_prob=1.0,
            period_choices=(1,),
            period_jitter_slots=0,
            t_min=0.20,
            t_max=0.20,
            result_ratio=0.0,
        ),
    )
    trace = generate_trace(scenario, seed=17)
    tasks = tuple(task for slot in trace.slots for task in slot.tasks)
    assert tasks
    assert all(task.release_time_s == task.arrival_slot * 0.25 for task in tasks)
    assert all(0.0 < task.remaining_time_s <= task.slot_duration_s for task in tasks)
    assert all(task.result_ratio == 0.0 for task in tasks)
    assert len({task.identity for task in tasks}) == len(tasks)
    assert all(task.task_id == (task.user_id << 32) | task.job_index for task in tasks)
    for user_id in range(scenario.num_users):
        user_jobs = [task.job_index for task in tasks if task.user_id == user_id]
        assert user_jobs == list(range(len(user_jobs)))
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_domain.py tests/test_trace.py -q
```

Expected: collection or assertions fail because `TaskIdentity`, `slot_duration_s`, `result_ratio`, `release_time_s`, and unit-safe absolute deadlines are absent.

- [ ] **Step 3: Implement the domain contract and trace materialization**

Add this value object and fields in `uav_mec/domain.py`:

```python
@dataclass(frozen=True, order=True, slots=True)
class TaskIdentity:
    user_id: int
    job_index: int


@dataclass(frozen=True, slots=True)
class Task:
    task_id: int
    user_id: int
    arrival_slot: int
    data_bits: float
    comp_cycles: float
    deadline_s: float
    local_cpu_hz: float
    job_index: int = 0
    slot_duration_s: float = 1.0
    result_ratio: float = 0.0

    @property
    def identity(self) -> TaskIdentity:
        return TaskIdentity(self.user_id, self.job_index)

    @property
    def release_time_s(self) -> float:
        return self.arrival_slot * self.slot_duration_s

    @property
    def remaining_time_s(self) -> float:
        return self.deadline_s

    @property
    def absolute_deadline_s(self) -> float:
        return self.release_time_s + self.remaining_time_s
```

Retain the existing positive finite checks and add integer checks for `task_id`, `user_id`, and `arrival_slot`; add positive finite validation for `slot_duration_s` and non-negative finite validation for `result_ratio`. Do not impose `deadline_s <= slot_duration_s` in the general `Task` value object because numerical/unit tests and future queued extensions may represent a longer deadline. Instead, `generate_trace` must reject every trusted no-queue scenario whose generated deadline exceeds `ScenarioConfig.uav.dt`; the registered main family therefore enforces `0 < remaining_time_s <= slot_duration_s`. In `SlotSnapshot.__post_init__`, reject duplicate `task_id` values and duplicate `TaskIdentity` values. In `trace.py`, maintain one next-job counter per user for every arrival mode. On each release, assign that counter to `job_index`, increment it exactly once, and set `task_id = (user_id << 32) | job_index`. Copy `ScenarioConfig.uav.dt` and `ScenarioConfig.task.result_ratio` into every released `Task`. Update the older Bernoulli-ID assertion in `tests/test_trace.py` to the common mapping. Keep `TaskConfig.result_ratio` as a task-generation default, not an evaluator parameter. Strengthen `TaskConfig.__post_init__` in `model.py` to validate that default.

- [ ] **Step 4: Run GREEN**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_domain.py tests/test_trace.py tests/test_simulation.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the explicit files**

```bash
git add -- uav_mec/domain.py uav_mec/trace.py uav_mec/model.py tests/test_domain.py tests/test_trace.py
git diff --cached --check
git commit -m "refactor: make task timing and rho explicit"
```

---

### Task 3: Establish `uav_mec/latency.py` as the sole active delay formula

**Files:**
- Create: `uav_mec/latency.py`
- Modify: `uav_mec/resource_allocation.py`
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/experiment.py`
- Modify: `uav_mec/run_thesis_suite.py`
- Create: `tests/test_latency.py`
- Modify: `tests/test_csi_error.py`
- Modify: `tests/test_exact.py`
- Modify: `tests/test_resource_requirements.py`
- Modify: `tests/test_resource_allocator.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_resource_policies.py`
- Modify: `tests/test_ia_greedy.py`
- Modify: `tests/test_incremental_evaluation.py`
- Modify: `tests/test_initializer.py`
- Modify: `tests/test_mechanism_controls.py`
- Modify: `tests/test_ofdma_fair.py`
- Modify: `tests/test_search.py`
- Modify: `tests/test_simulation.py`

- [ ] **Step 1: Write the RED canonical-latency tests**

Create `tests/test_latency.py` as a self-contained test module:

```python
import pytest

from uav_mec.domain import Task
from uav_mec.latency import branch_delays_s, completion_delay_s


def _task(rho: float) -> Task:
    return Task(
        task_id=1,
        user_id=0,
        arrival_slot=2,
        data_bits=800.0,
        comp_cycles=1_200.0,
        deadline_s=0.8,
        local_cpu_hz=2_000.0,
        slot_duration_s=1.0,
        result_ratio=rho,
    )


def test_canonical_delay_uses_distinct_uplink_and_downlink_rates() -> None:
    delays = branch_delays_s(
        _task(0.25),
        alpha=0.5,
        uplink_rate_bps=400.0,
        mec_cpu_hz=600.0,
        downlink_rate_bps=100.0,
    )
    assert delays.local_s == pytest.approx(0.3)
    assert delays.uplink_s == pytest.approx(1.0)
    assert delays.mec_s == pytest.approx(1.0)
    assert delays.downlink_s == pytest.approx(1.0)
    assert delays.edge_s == pytest.approx(3.0)
    assert completion_delay_s(
        _task(0.25), 0.5, 400.0, 600.0, 100.0
    ) == pytest.approx(3.0)


def test_zero_rho_needs_no_downlink_rate() -> None:
    assert completion_delay_s(_task(0.0), 0.5, 400.0, 600.0, None) == pytest.approx(2.0)


def test_positive_rho_requires_an_independent_downlink_rate() -> None:
    with pytest.raises(ValueError, match="downlink_rate_bps"):
        completion_delay_s(_task(0.25), 0.5, 400.0, 600.0, None)
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_latency.py -q
```

Expected: import fails because `uav_mec.latency` does not exist.

- [ ] **Step 3: Implement the canonical API**

Create this public shape in `uav_mec/latency.py`:

```python
from __future__ import annotations

import math
from dataclasses import dataclass

from .domain import Task


@dataclass(frozen=True, slots=True)
class BranchDelays:
    local_s: float
    uplink_s: float
    mec_s: float
    downlink_s: float

    @property
    def edge_s(self) -> float:
        return math.fsum((self.uplink_s, self.mec_s, self.downlink_s))

    @property
    def completion_s(self) -> float:
        return max(self.local_s, self.edge_s)


def branch_delays_s(
    task: Task,
    alpha: float,
    uplink_rate_bps: float,
    mec_cpu_hz: float,
    downlink_rate_bps: float | None,
) -> BranchDelays:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if alpha == 0.0:
        return BranchDelays(task.local_delay_s, 0.0, 0.0, 0.0)
    if not math.isfinite(uplink_rate_bps) or uplink_rate_bps <= 0.0:
        raise ValueError("uplink_rate_bps must be finite and positive")
    if not math.isfinite(mec_cpu_hz) or mec_cpu_hz <= 0.0:
        raise ValueError("mec_cpu_hz must be finite and positive")
    if task.result_ratio > 0.0 and (
        downlink_rate_bps is None
        or not math.isfinite(downlink_rate_bps)
        or downlink_rate_bps <= 0.0
    ):
        raise ValueError("downlink_rate_bps is required for positive result_ratio")
    downlink_s = 0.0 if task.result_ratio == 0.0 else alpha * task.result_ratio * task.data_bits / float(downlink_rate_bps)
    return BranchDelays(
        local_s=(1.0 - alpha) * task.comp_cycles / task.local_cpu_hz,
        uplink_s=alpha * task.data_bits / uplink_rate_bps,
        mec_s=alpha * task.comp_cycles / mec_cpu_hz,
        downlink_s=downlink_s,
    )


def completion_delay_s(
    task: Task,
    alpha: float,
    uplink_rate_bps: float,
    mec_cpu_hz: float,
    downlink_rate_bps: float | None = None,
) -> float:
    return branch_delays_s(
        task, alpha, uplink_rate_bps, mec_cpu_hz, downlink_rate_bps
    ).completion_s
```

Preserve the existing overflow-safe arithmetic when moving the formula from `resource_allocation.py`; the code above fixes the public contract and the four terms, not permission to regress numerical guards.

- [ ] **Step 4: Route resource admission and evaluation through the canonical API**

Remove `result_ratio` from `EvaluationConfig`, `completion_delay_s`, `required_resources`, `screening_required_resources`, `allocate_deadline_tasks`, `allocate_with_policy`, and `allocation_score_with_policy`. Those functions obtain rho from each `Task`. Add `downlink_rates_bps: Mapping[int, float] | None = None` only where a collection is evaluated. For the registered main, resource, joint, exact, and pilot configurations rho is zero, so no downlink map is synthesized. `experiment.py` must stop copying scenario rho into `EvaluationConfig`.

Update every listed test module so task constructors carry `result_ratio`, `EvaluationConfig` construction no longer has a global ratio position, resource calls no longer pass a global ratio, and positive-rho tests pass a genuinely separate downlink rate. In `run_thesis_suite.py`, stop replacing a nonexistent evaluation-level ratio. Delete assertions that treat `(1 + rho) * data / uplink_rate` as the edge transfer term.

- [ ] **Step 5: Run GREEN and the active-call-site audit**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_latency.py tests/test_csi_error.py tests/test_exact.py tests/test_resource_requirements.py tests/test_resource_allocator.py tests/test_resource_policies.py tests/test_evaluation.py tests/test_ia_greedy.py tests/test_incremental_evaluation.py tests/test_initializer.py tests/test_mechanism_controls.py tests/test_ofdma_fair.py tests/test_search.py tests/test_simulation.py -q
rg -n "cfg\.result_ratio|result_ratio=.*allocate_|required_resources\([^)]*result_ratio" uav_mec/evaluation.py uav_mec/resource_allocation.py uav_mec/experiment.py uav_mec/run_thesis_suite.py
```

Expected: pytest passes and `rg` prints nothing.

- [ ] **Step 6: Commit the explicit files**

```bash
git add -- uav_mec/latency.py uav_mec/resource_allocation.py uav_mec/evaluation.py uav_mec/experiment.py uav_mec/run_thesis_suite.py tests/test_latency.py tests/test_csi_error.py tests/test_exact.py tests/test_resource_requirements.py tests/test_resource_allocator.py tests/test_evaluation.py tests/test_resource_policies.py tests/test_ia_greedy.py tests/test_incremental_evaluation.py tests/test_initializer.py tests/test_mechanism_controls.py tests/test_ofdma_fair.py tests/test_search.py tests/test_simulation.py
git diff --cached --check
git commit -m "refactor: centralize task-level latency semantics"
```

---

### Task 4: Require explicit `ChannelBasis` and record every SIC layer

**Files:**
- Modify: `uav_mec/domain.py`
- Modify: `uav_mec/schedule.py`
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/baselines_v2.py`
- Modify: `uav_mec/exact.py`
- Modify: `uav_mec/experiment.py`
- Modify: `uav_mec/scheduler.py`
- Modify: `uav_mec/simulation.py`
- Modify: `tests/test_domain.py`
- Modify: `tests/test_schedule.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_csi_error.py`
- Modify: `tests/test_exact.py`
- Modify: `tests/test_ia_greedy.py`
- Modify: `tests/test_incremental_evaluation.py`
- Modify: `tests/test_initializer.py`
- Modify: `tests/test_mechanism_controls.py`
- Modify: `tests/test_neighborhoods.py`
- Modify: `tests/test_resource_policies.py`
- Modify: `tests/test_search.py`

- [ ] **Step 1: Write the RED channel-provenance and layer tests**

Add this complete helper and tests to `tests/test_evaluation.py`:

```python
import numpy as np
import pytest

from uav_mec.domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig, compute_rate_records
from uav_mec.model import ChannelConfig, NOMAChannel


def _basis_case() -> tuple[SlotSnapshot, Schedule, NOMAChannel, EvaluationConfig]:
    tasks = (
        Task(1, 0, 0, 1e6, 1.4e9, 0.8, 1e9, slot_duration_s=1.0, result_ratio=0.0),
        Task(2, 1, 0, 1e6, 1.4e9, 0.8, 1e9, slot_duration_s=1.0, result_ratio=0.0),
    )
    snapshot = SlotSnapshot(
        0,
        tasks,
        np.array([[1e-9], [8e-9]]),
        1,
        estimated_channel_gains=np.array([[9e-9], [2e-9]]),
    )
    schedule = Schedule(
        (
            ScheduleEntry(1, 0, 0, 0, "strong"),
            ScheduleEntry(2, 1, 0, 0, "weak"),
        )
    )
    evaluation = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.01,
        cpu_per_uav_hz=4e9,
    )
    return snapshot, schedule, NOMAChannel(ChannelConfig(), 100.0), evaluation


def test_channel_basis_changes_gains_without_erasing_provenance() -> None:
    snapshot, schedule, channel, evaluation = _basis_case()
    planning = compute_rate_records(
        snapshot, schedule, channel, evaluation, channel_basis=ChannelBasis.PLANNING
    )
    realized = compute_rate_records(
        snapshot,
        schedule,
        channel,
        evaluation,
        channel_basis=ChannelBasis.REALIZED,
    )
    assert {record.channel_basis for record in planning} == {ChannelBasis.PLANNING}
    assert {record.channel_basis for record in realized} == {ChannelBasis.REALIZED}
    assert tuple(record.task_id for record in realized) == tuple(
        record.task_id for record in planning
    )
    assert tuple(
        (layer.task_id, layer.decode_rank)
        for record in realized
        for layer in record.sic_layers
    ) == tuple(
        (layer.task_id, layer.decode_rank)
        for record in planning
        for layer in record.sic_layers
    )


def test_two_user_block_checks_both_decode_layers() -> None:
    snapshot, schedule, channel, evaluation = _basis_case()
    records = compute_rate_records(
        snapshot, schedule, channel, evaluation, channel_basis=ChannelBasis.PLANNING
    )
    layers = tuple(layer for record in records for layer in record.sic_layers)
    assert {(layer.decode_rank, layer.task_id) for layer in layers} == {(0, 1), (1, 2)}
    assert all(layer.sinr >= 0.0 for layer in layers)
    assert all(layer.threshold == pytest.approx(evaluation.gamma_sic_min) for layer in layers)
    assert all(record.sic_feasible == all(layer.feasible for layer in record.sic_layers) for record in records)
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_evaluation.py::test_channel_basis_changes_gains_without_erasing_provenance tests/test_evaluation.py::test_two_user_block_checks_both_decode_layers -q
```

Expected: `ChannelBasis`, `channel_basis`, and `sic_layers` are absent.

- [ ] **Step 3: Add the explicit types and basis selection**

Add these domain types:

```python
class ChannelBasis(str, Enum):
    PLANNING = "planning"
    REALIZED = "realized"


@dataclass(frozen=True, slots=True)
class SicLayerCheck:
    task_id: int
    receiver_uav: int
    subcarrier: int
    decode_rank: int
    sinr: float
    threshold: float
    feasible: bool
```

Add `channel_basis: ChannelBasis` and an immutable `sic_layers: tuple` containing only `SicLayerCheck` values to `RateRecord`; implement `sic_feasible` as `all(layer.feasible for layer in sic_layers)`. Add `SlotSnapshot.channel_gains_for(channel_basis)` and remove `as_planning_snapshot`, which destroys provenance. Require a keyword-only `channel_basis` in `normalize_roles`, `normalize_roles_for_blocks`, `build_evaluation_context`, `compute_rate_records`, `evaluate_schedule`, and their internal helpers. Never default it. The two normalization functions accept only `ChannelBasis.PLANNING`; passing `REALIZED` raises `ValueError` because replay must preserve the planned order.

When constructing a planning schedule, sort each `(receiver_uav, subcarrier)` group by `(-received_power_w, task_id)` using planning gains. When evaluating an already normalized schedule, preserve its entry order and stored roles. `ChannelBasis.REALIZED` changes only the gains used below; it never normalizes, sorts, swaps roles, or changes a decode rank. For each frozen decode rank `q`, calculate exactly:

```python
denominator = math.fsum((
    noise_w,
    cross_w,
    math.fsum(received_w[q + 1 :]),
    cfg.sic_residual * math.fsum(received_w[:q]),
))
sinr = received_w[q] / denominator
```

Create one `SicLayerCheck` for every scheduled layer, including a solo layer. Planning evaluation raises on any failed layer. Realized replay records failure but does not mutate the plan. Add a source guard proving the realized path contains neither `normalize_roles` nor received-power sorting. Update every explicitly listed formal module and test call site to pass `ChannelBasis.PLANNING` or `ChannelBasis.REALIZED`; remove every use of a copied planning snapshot.

- [ ] **Step 4: Run GREEN**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_domain.py tests/test_schedule.py tests/test_evaluation.py tests/test_csi_error.py tests/test_exact.py tests/test_ia_greedy.py tests/test_incremental_evaluation.py tests/test_initializer.py tests/test_mechanism_controls.py tests/test_neighborhoods.py tests/test_resource_policies.py tests/test_search.py -q
```

Expected: all selected tests pass and every evaluation call states its basis.

- [ ] **Step 5: Commit the explicit files**

```bash
git add -- uav_mec/domain.py uav_mec/schedule.py uav_mec/evaluation.py uav_mec/baselines_v2.py uav_mec/exact.py uav_mec/experiment.py uav_mec/scheduler.py uav_mec/simulation.py tests/test_domain.py tests/test_schedule.py tests/test_evaluation.py tests/test_csi_error.py tests/test_exact.py tests/test_ia_greedy.py tests/test_incremental_evaluation.py tests/test_initializer.py tests/test_mechanism_controls.py tests/test_neighborhoods.py tests/test_resource_policies.py tests/test_search.py
git diff --cached --check
git commit -m "refactor: make channel basis and sic layers explicit"
```

---

### Task 5: Centralize finite admission closure in `uav_mec/closure.py`

**Files:**
- Create: `uav_mec/closure.py`
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/experiment.py`
- Modify: `uav_mec/scheduler.py`
- Create: `tests/test_schedule_closure.py`
- Modify: `tests/test_experiment_cli.py`
- Modify: `tests/test_search.py`

- [ ] **Step 1: Write the RED closure tests**

Create `tests/test_schedule_closure.py` with a local case builder:

```python
import numpy as np

from uav_mec.closure import ClosedExecutableSchedule, close_assignment, rate_raw_assignment
from uav_mec.domain import ChannelBasis, Schedule, ScheduleEntry, SlotSnapshot, Task
from uav_mec.evaluation import EvaluationConfig
from uav_mec.model import ChannelConfig, NOMAChannel
from uav_mec.resource_allocation import AllocationPolicy


def _overloaded_case() -> tuple[SlotSnapshot, Schedule, NOMAChannel, EvaluationConfig]:
    tasks = tuple(
        Task(
            task_id=index,
            user_id=index,
            arrival_slot=0,
            data_bits=1_000_000.0,
            comp_cycles=1_600_000_000.0,
            deadline_s=0.8,
            local_cpu_hz=500_000_000.0,
            slot_duration_s=1.0,
            result_ratio=0.0,
        )
        for index in (1, 2)
    )
    snapshot = SlotSnapshot(0, tasks, np.array([[9e-8], [3e-8]]), 1)
    raw = Schedule(
        (
            ScheduleEntry(1, 1, 0, 0, "strong"),
            ScheduleEntry(2, 2, 0, 0, "weak"),
        )
    )
    cfg = EvaluationConfig(
        n_subcarriers=1,
        bandwidth_hz=5e6,
        transmit_power_w=0.2,
        sic_residual=0.02,
        gamma_sic_min=0.01,
        cpu_per_uav_hz=1_200_000_000.0,
        allocation_policy=AllocationPolicy.DEADLINE,
    )
    return snapshot, raw, NOMAChannel(ChannelConfig(), 100.0), cfg


def test_raw_snapshot_is_distinct_from_closed_executable_schedule() -> None:
    snapshot, raw, channel, cfg = _overloaded_case()
    rated = rate_raw_assignment(
        snapshot, raw, channel, cfg, channel_basis=ChannelBasis.PLANNING
    )
    closed = close_assignment(snapshot, rated, channel, cfg)
    assert isinstance(closed, ClosedExecutableSchedule)
    assert set(closed.schedule.task_ids) == set(closed.allocation.completed_task_ids)
    assert set(closed.schedule.task_ids) < set(rated.assignment.task_ids)
    assert closed.rate_snapshot.assignment == closed.schedule


def test_closure_is_idempotent_and_deletion_only() -> None:
    snapshot, raw, channel, cfg = _overloaded_case()
    first = close_assignment(
        snapshot,
        rate_raw_assignment(snapshot, raw, channel, cfg, channel_basis=ChannelBasis.PLANNING),
        channel,
        cfg,
    )
    second = close_assignment(snapshot, first.rate_snapshot, channel, cfg)
    assert second.stable_key() == first.stable_key()
    assert first.closure_rounds <= len(raw.entries)
    assert set(first.schedule.task_ids) <= set(raw.task_ids)
```

Add this property to `Schedule`:

```python
@property
def task_ids(self) -> tuple:
    return tuple(sorted(entry.task_id for entry in self.entries))
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_schedule_closure.py -q
```

Expected: import fails because `uav_mec.closure` does not exist.

- [ ] **Step 3: Implement the typed closure boundary**

Use these public dataclasses and signatures:

```python
@dataclass(frozen=True, slots=True)
class RawRateSnapshot:
    assignment: Schedule
    rates: tuple
    channel_basis: ChannelBasis
    semantic_fingerprint: str

    def stable_key(self) -> tuple:
        return (self.assignment.stable_key(), self.channel_basis.value, self.semantic_fingerprint)


@dataclass(frozen=True, slots=True)
class ClosedExecutableSchedule:
    schedule: Schedule
    rate_snapshot: RawRateSnapshot
    allocation: AllocationResult
    closure_rounds: int
    allocation_policy: AllocationPolicy

    def stable_key(self) -> tuple:
        return (
            self.schedule.stable_key(),
            self.rate_snapshot.channel_basis.value,
            self.rate_snapshot.semantic_fingerprint,
            self.allocation_policy.value,
        )


def rate_raw_assignment(
    snapshot: SlotSnapshot,
    assignment: Schedule,
    channel: NOMAChannel,
    cfg: EvaluationConfig,
    *,
    channel_basis: ChannelBasis,
) -> RawRateSnapshot:
    normalized = normalize_roles(assignment, snapshot, channel_basis=channel_basis)
    rates = compute_rate_records(
        snapshot, normalized, channel, cfg, channel_basis=channel_basis
    )
    return RawRateSnapshot(normalized, rates, channel_basis, semantic_fingerprint(snapshot, channel, cfg, channel_basis))
```

Implement `semantic_fingerprint` in `uav_mec/evaluation.py` in this task, using the complete canonical byte-stream field list in Task 7. Import it into `closure.py`; do not use a temporary constant, object identity, or process-local hash while closure is being introduced.

`close_assignment` accepts only a `RawRateSnapshot`, applies the selected allocation policy independently per serving UAV, deletes every radio entry not admitted, re-normalizes roles and re-rates on the same `ChannelBasis`, then repeats. It never inserts. It returns only when scheduled IDs equal admitted offload IDs. Reject a fingerprint mismatch. Raise `ClosureConvergenceError` if rounds exceed the original entry count; that exception denotes an implementation defect, not an ordinary infeasible candidate.

Replace local closure-like filtering in `evaluation.py` and `scheduler.py` with this API. Make the end-to-end planning path in `experiment.py` close every raw baseline/search assignment before freezing it. Do not create a parallel module for the same state transition.

- [ ] **Step 4: Run GREEN and finite-state regression**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_schedule_closure.py tests/test_evaluation.py tests/test_experiment_cli.py tests/test_search.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the explicit files**

```bash
git add -- uav_mec/closure.py uav_mec/evaluation.py uav_mec/experiment.py uav_mec/scheduler.py tests/test_schedule_closure.py tests/test_experiment_cli.py tests/test_search.py
git diff --cached --check
git commit -m "feat: close raw radio assignments into executable schedules"
```

---

### Task 6: Freeze the planned schedule and replay realized channels without re-optimization

**Files:**
- Modify: `uav_mec/closure.py`
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/experiment.py`
- Modify: `uav_mec/domain.py`
- Modify: `tests/test_schedule_closure.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_experiment_cli.py`
- Modify: `tests/test_csi_error.py`
- Modify: `tests/test_resource_policies.py`

- [ ] **Step 1: Write the RED frozen-plan test**

Add this test to `tests/test_evaluation.py`; it uses the `_basis_case` helper from Task 4 and the public closure API from Task 5:

```python
import inspect

from uav_mec.closure import close_assignment, freeze_plan, rate_raw_assignment
from uav_mec.evaluation import evaluate_frozen_plan


def test_realized_replay_never_reallocates_or_recloses() -> None:
    snapshot, schedule, channel, evaluation = _basis_case()
    rated = rate_raw_assignment(
        snapshot, schedule, channel, evaluation, channel_basis=ChannelBasis.PLANNING
    )
    plan = freeze_plan(close_assignment(snapshot, rated, channel, evaluation))
    planned_decisions = tuple(
        (decision.task_id, decision.alpha, decision.cpu_hz)
        for decision in plan.allocation.decisions
    )
    replay_source = inspect.getsource(evaluate_frozen_plan)
    assert "allocate_with_policy" not in replay_source
    assert "close_assignment" not in replay_source
    realized = evaluate_frozen_plan(snapshot, plan, channel, evaluation)
    assert realized.plan_key == plan.stable_key()
    assert realized.schedule == plan.schedule
    assert tuple(
        (decision.task_id, decision.alpha, decision.cpu_hz)
        for decision in realized.decisions
    ) == planned_decisions
    assert {record.channel_basis for record in realized.rates} == {ChannelBasis.REALIZED}
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_evaluation.py::test_realized_replay_never_reallocates_or_recloses -q
```

Expected: `freeze_plan` and `evaluate_frozen_plan` are absent, or the old realized evaluator calls allocation again.

- [ ] **Step 3: Add immutable planning and realized result types**

Add these records to `uav_mec/domain.py`:

```python
@dataclass(frozen=True, slots=True)
class FrozenSlotPlan:
    schedule: Schedule
    planning_rates: tuple
    allocation: AllocationResult
    semantic_fingerprint: str

    def stable_key(self) -> tuple:
        decisions = tuple(
            (row.task_id, row.alpha.hex(), row.cpu_hz.hex(), row.completed)
            for row in self.allocation.decisions
        )
        return (self.schedule.stable_key(), decisions, self.semantic_fingerprint)


@dataclass(frozen=True, slots=True)
class RealizedSlotOutcome:
    plan_key: tuple
    schedule: Schedule
    decisions: tuple
    rates: tuple
    completed_task_ids: tuple
    sic_outage_task_ids: tuple
```

`freeze_plan` accepts a planning-basis `ClosedExecutableSchedule` only. `evaluate_frozen_plan` recomputes rates and all layer checks with `ChannelBasis.REALIZED`, then recomputes delay through `uav_mec.latency.completion_delay_s` using each frozen `alpha` and `cpu_hz`. It must preserve association, subcarrier, decode rank, and CPU decisions. A realized SIC failure becomes an outage and miss; it is not repaired, reclosed, or reallocated. Replace the current `evaluate_realized_schedule` behavior and make `experiment.py` persist the frozen plan key and realized outage IDs.

- [ ] **Step 4: Run GREEN**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_schedule_closure.py tests/test_evaluation.py tests/test_experiment_cli.py tests/test_csi_error.py tests/test_resource_policies.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the explicit files**

```bash
git add -- uav_mec/closure.py uav_mec/evaluation.py uav_mec/experiment.py uav_mec/domain.py tests/test_schedule_closure.py tests/test_evaluation.py tests/test_experiment_cli.py tests/test_csi_error.py tests/test_resource_policies.py
git diff --cached --check
git commit -m "fix: replay realized channels against frozen plans"
```

---

### Task 7: Make cache semantics complete and isolate every legacy delay path

**Files:**
- Modify: `uav_mec/evaluation.py`
- Modify: `uav_mec/closure.py`
- Modify: `uav_mec/pairing.py`
- Modify: `uav_mec/model.py`
- Modify: `tests/test_latency.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_project_metadata.py`
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `uav_mec/code_CLAUDE.md`
- Modify: `Makefile`
- Modify: `configs/thesis/confirmatory_main_v2.toml`
- Modify: `configs/thesis/confirmatory_resource_fixed_v2.toml`
- Modify: `configs/thesis/confirmatory_resource_strong.toml`
- Modify: `configs/thesis/confirmatory_joint_v2.toml`
- Modify: `configs/thesis/confirmatory_exact.toml`
- Modify: `configs/thesis/pilot_main_v2.toml`
- Modify: `configs/thesis/pilot_externality_v2.toml`
- Modify: `configs/thesis/pilot_resource_fixed_v2.toml`
- Modify: `configs/thesis/pilot_joint_v2.toml`
- Modify: `configs/thesis/externality_stress.toml`
- Modify: `configs/thesis/smoke.toml`
- Modify: `configs/thesis/smoke_exact.toml`

- [ ] **Step 1: Write RED cache-key tests**

Add to `tests/test_evaluation.py`:

```python
import dataclasses

from uav_mec.evaluation import semantic_fingerprint
from uav_mec.resource_allocation import AllocationPolicy


def test_semantic_cache_key_separates_basis_policy_gain_and_task_rho() -> None:
    snapshot, _schedule, channel, evaluation = _basis_case()
    planning = semantic_fingerprint(snapshot, channel, evaluation, ChannelBasis.PLANNING)
    realized = semantic_fingerprint(snapshot, channel, evaluation, ChannelBasis.REALIZED)
    changed_policy = dataclasses.replace(evaluation, allocation_policy=AllocationPolicy.MIN_SLACK_MIN_CPU)
    policy_key = semantic_fingerprint(snapshot, channel, changed_policy, ChannelBasis.PLANNING)
    changed_task = dataclasses.replace(snapshot.tasks[0], result_ratio=0.2)
    rho_snapshot = SlotSnapshot(
        snapshot.slot_index,
        (changed_task, snapshot.tasks[1]),
        snapshot.channel_gains,
        snapshot.num_uavs,
        snapshot.estimated_channel_gains,
    )
    rho_key = semantic_fingerprint(
        rho_snapshot,
        channel,
        evaluation,
        ChannelBasis.PLANNING,
        downlink_rates_bps={1: 500_000.0, 2: 1_000_000.0},
    )
    downlink_a = semantic_fingerprint(
        rho_snapshot,
        channel,
        evaluation,
        ChannelBasis.PLANNING,
        downlink_rates_bps={1: 1_000_000.0, 2: 1_000_000.0},
    )
    downlink_b = semantic_fingerprint(
        rho_snapshot,
        channel,
        evaluation,
        ChannelBasis.PLANNING,
        downlink_rates_bps={1: 2_000_000.0, 2: 1_000_000.0},
    )
    assert len({planning, realized, policy_key, rho_key, downlink_a, downlink_b}) == 6
```

Add to `tests/test_latency.py`:

```python
import ast
from pathlib import Path

import numpy as np
import pytest

from uav_mec.model import ChannelConfig, NOMAChannel, offload_delay
from uav_mec.pairing import evaluate_pairing, kkt_alpha


def test_formal_pipeline_does_not_call_legacy_delay_helpers() -> None:
    formal_paths = (
        Path("uav_mec/closure.py"),
        Path("uav_mec/evaluation.py"),
        Path("uav_mec/resource_allocation.py"),
        Path("uav_mec/scheduler.py"),
        Path("uav_mec/exact.py"),
        Path("uav_mec/baselines_v2.py"),
        Path("uav_mec/experiment.py"),
    )
    forbidden_calls = {"kkt_alpha", "evaluate_pairing", "offload_delay"}
    for path in formal_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        imported_pairing = any(
            isinstance(node, ast.ImportFrom)
            and (node.module or "").endswith("pairing")
            or isinstance(node, ast.Import)
            and any(alias.name.endswith("pairing") for alias in node.names)
            for node in ast.walk(tree)
        )
        assert not imported_pairing, path
        assert not called.intersection(forbidden_calls), (path, called)


def test_legacy_delay_entry_points_emit_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="legacy-v1 delay semantics"):
        kkt_alpha(1_000.0, 2_000.0, 5_000.0, 10_000.0, 4_000.0)
    with pytest.warns(DeprecationWarning, match="legacy-v1 delay semantics"):
        offload_delay(1_000.0, 2_000.0, 5_000.0, 10_000.0)
    with pytest.warns(DeprecationWarning, match="legacy-v1 delay semantics"):
        result = evaluate_pairing(
            {},
            np.empty((0, 1)),
            NOMAChannel(ChannelConfig(), 100.0),
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=int),
            1,
            4e9,
            1e9,
            0.8,
        )
    assert result[:2] == (0, 0)
```

- [ ] **Step 2: Run RED**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_evaluation.py::test_semantic_cache_key_separates_basis_policy_gain_and_task_rho tests/test_latency.py::test_formal_pipeline_does_not_call_legacy_delay_helpers -q
```

Expected: the fingerprint is absent or omits one semantic dimension; legacy active calls are reported if any remain.

- [ ] **Step 3: Implement a complete semantic fingerprint**

Audit and complete the Task 5 `semantic_fingerprint` canonical byte stream. It contains, in order: schema version, slot index, ordered task identity and every task field including `slot_duration_s` and `result_ratio`, selected `ChannelBasis`, selected gain matrix dtype/shape/bytes, channel bandwidth/noise parameters, transmit power, SIC residual, SIC threshold, cross-interference scale, subcarrier count, per-UAV CPU budgets, allocation policy, the ordered independent downlink-rate items or an explicit no-downlink sentinel, and canonical-latency schema version. Use SHA-256 and no process identity. Rate caches may use a radio-only projection; every allocation, closure, and search cache key uses the full fingerprint including downlink data. Add `downlink_rate_items` to both allocation-cache signatures and remove `id(allocation_score_with_policy)` from cache keys. Add a test that identical semantic inputs in a fresh object produce the same key.

- [ ] **Step 4: Mark the old formulas as legacy and document the boundary**

In `pairing.py`, label `kkt_alpha` and `evaluate_pairing` as legacy-v1 APIs in module and function docstrings. In `model.py`, label `offload_delay` as legacy-v1. Preserve them only for the named legacy scripts; do not silently redirect their array-based arguments into the task API. Add a `DeprecationWarning` on each call with the message `legacy-v1 delay semantics; forbidden in thesis-v2 formal pipeline`.

Add this exact policy to root `CLAUDE.md`:

```markdown
### Canonical thesis-v2 semantics

- Formal code must use `uav_mec.latency`, explicit `ChannelBasis`, and `uav_mec.closure`.
- A realized evaluation replays a `FrozenSlotPlan`; it never reallocates CPU or changes radio decisions.
- `uav_mec.pairing.kkt_alpha`, `uav_mec.pairing.evaluate_pairing`, and `uav_mec.model.offload_delay` are legacy-v1 only and must not be imported or called by the thesis-v2 pipeline.
- Pilot and confirmatory jobs are forbidden until the semantic gate in the approved implementation plan is green.
```

Replace the outdated active-semantics mappings in both root `CLAUDE.md` and `uav_mec/code_CLAUDE.md` so thesis-v2 task/delay, radio/SIC, closure, and scheduling rows point respectively to `domain.py` plus `latency.py`, `evaluation.py`, `closure.py`, and `scheduler.py`. Move `pairing.py`, `baselines_pairing.py`, `run_main_experiment.py`, and `model.offload_delay` under a clearly labelled legacy-v1 section. In `README.md`, state that canonical-v1 verifies only its historical snapshot and cannot validate thesis-v2; list the semantic gate before any v2 run instructions.

Add this non-experimental target to `Makefile` and include `semantic-gate` in its explicit `.PHONY` declaration:

```make
semantic-gate:
	$(PYTHON) -m pytest tests/test_latency.py tests/test_domain.py tests/test_trace.py tests/test_schedule.py tests/test_schedule_closure.py tests/test_resource_requirements.py tests/test_resource_allocator.py tests/test_resource_policies.py tests/test_evaluation.py tests/test_search.py tests/test_project_metadata.py -q
```

- [ ] **Step 5: Freeze main-family rho and slot-duration configuration**

In each of the twelve explicitly listed TOML files, add `result_ratio = 0.0` under `[scenario.task]` and add `[scenario.uav]` with `dt = 1.0`. Assert each deadline bound is positive and no larger than `dt`. Do not add a downlink rate to a rho-zero configuration. Positive-rho robustness configurations are out of scope until they can name an independent downlink-rate source.

- [ ] **Step 6: Run GREEN and the semantic gate**

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_latency.py tests/test_domain.py tests/test_trace.py tests/test_schedule.py tests/test_schedule_closure.py tests/test_resource_requirements.py tests/test_resource_allocator.py tests/test_resource_policies.py tests/test_evaluation.py tests/test_search.py tests/test_experiment_cli.py tests/test_project_metadata.py -q
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest -q
rg -n "as_planning_snapshot|cfg\.result_ratio" uav_mec tests
git diff --check
```

Expected: both pytest commands pass, `rg` prints nothing, and `git diff --check` exits `0`. Do not run an experiment after the gate.

- [ ] **Step 7: Commit the explicit files**

```bash
git add -- uav_mec/evaluation.py uav_mec/closure.py uav_mec/pairing.py uav_mec/model.py tests/test_latency.py tests/test_evaluation.py tests/test_project_metadata.py CLAUDE.md README.md uav_mec/code_CLAUDE.md Makefile configs/thesis/confirmatory_main_v2.toml configs/thesis/confirmatory_resource_fixed_v2.toml configs/thesis/confirmatory_resource_strong.toml configs/thesis/confirmatory_joint_v2.toml configs/thesis/confirmatory_exact.toml configs/thesis/pilot_main_v2.toml configs/thesis/pilot_externality_v2.toml configs/thesis/pilot_resource_fixed_v2.toml configs/thesis/pilot_joint_v2.toml configs/thesis/externality_stress.toml configs/thesis/smoke.toml configs/thesis/smoke_exact.toml
git diff --cached --check
git commit -m "chore: gate thesis v2 semantic provenance"
```

---

## Plan 1 completion gate

Plan 1 is complete only when every commit required by the seven tasks exists, the original checkout still contains every excluded user/generated file, the isolated worktree is clean, the full pytest suite is green, and no experiment beyond the named Task 1 baseline smoke checks has been launched. Record every commit SHA in the implementation handoff; do not interpret test success as formal evidence.
