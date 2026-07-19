# Confirmatory Execution and Canonical-v2 Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one final power decision and immutable confirmatory lock, execute all eight registered evidence families without optional stopping or seed leakage, analyze H1--H4 exactly as preregistered, and publish a verified canonical-v2 full/light archive even when performance results are negative.

**Architecture:** This plan contains protocol execution, not algorithm development. It consumes the tested runner, registry, seed ledger, lock, analyzer and publisher delivered by Plans 1--3. Development seeds are used for the last capacity checks, pilot seeds are used once for power freezing, and formal seeds are opened in the fixed family order. Every resumable run is bound to one lock and one immutable shard plan; canonical-v2 is derived only after integrity and correctness audits pass.

**Tech Stack:** Python 3.11, pytest, immutable JSON/JSONL/CSV shards, SHA-256 manifests, SciPy paired inference, Git, GNU Make.

---

## Fixed paths and protocol constants

All commands run from:

```text
/home/link/my_workspace/.worktrees/master-thesis-ccfb
```

Use these exact paths throughout the run:

```text
registry:       configs/thesis/experiment_registry_v2.toml
pilot root:     artifacts/preregistration/v2/power-pilot/final
power decision: artifacts/preregistration/v2/power-pilot/final/decision/power_decision.json
trace lock:     artifacts/preregistration/v2/trace_lock.json
runtime budget: artifacts/preregistration/v2/runtime_budget.json
lock root:      artifacts/preregistration/v2/confirmatory-lock
formal root:    artifacts/runs/v2-confirmatory
analysis root:  artifacts/analysis/v2-confirmatory
canonical root: artifacts/canonical/v2
seed ledger:    artifacts/preregistration/v2/seed-access.jsonl
```

The formal family order is immutable:

```text
resource -> exact -> externality -> main -> joint
-> scaling -> robustness -> trace_driven
```

The runner may resume missing shards but may never append seeds, replace a locked seed, change a case, or accept CLI overrides for sample size, slots, methods or scenarios.

### Task 1: Pass the pre-open readiness gate

**Files:**
- Read: `configs/thesis/experiment_registry_v2.toml`
- Read: `artifacts/preregistration/v2/seed-access.jsonl`
- Create generated: `artifacts/preregistration/v2/readiness-report.json`
- Create generated: `artifacts/preregistration/v2/readiness-report.md`

- [ ] **Step 1: Verify branch, cleanliness and exact source commit**

Run:

```bash
cd /home/link/my_workspace/.worktrees/master-thesis-ccfb
test "$(git branch --show-current)" = "feature/ustc-ccfb-v2"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
source_sha=$(git rev-parse HEAD)
[[ "$source_sha" =~ ^[0-9a-f]{40}$ ]]
printf '%s\n' "$source_sha"
```

Expected: all checks exit zero and one 40-character SHA is printed. Generated artifacts must already be ignored; an untracked source/config/test file is a failure, not something to hide with `git add` or `git stash`.

- [ ] **Step 2: Run the complete software and protocol test suite freshly**

```bash
mkdir -p artifacts/preregistration/v2
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest -q \
  --junitxml=artifacts/preregistration/v2/test-report.junit.xml
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.correctness_bundle build \
  --test-report artifacts/preregistration/v2/test-report.junit.xml \
  --repo-root . \
  --registry configs/thesis/experiment_registry_v2.toml \
  --output artifacts/preregistration/v2/correctness/v2
make evidence-test
```

Expected: every command exits zero. The JUnit report and correctness bundle are checksum-closed and contain the fresh pass count, duration, source SHA, 10,000-action equivalence, three-layer externality conservation, DD/Exact agreement, MethodSpec, and closest-work audit IDs. The correctness bundle opens no pilot or formal seed.

- [ ] **Step 3: Audit seed access before opening the final pilot**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.seed_access audit \
  --registry configs/thesis/experiment_registry_v2.toml \
  --ledger artifacts/preregistration/v2/seed-access.jsonl \
  --phase before-final-pilot \
  --output artifacts/preregistration/v2/seed-audit-before-pilot.json
```

Expected: development accesses are confined to `0--999`; no seed in `1000--1099` or any formal pool has a `trace_run`, `algorithm_run`, or `analysis` access. Merely parsing registry integers is recorded separately as `definition_read` and does not count as opening a seed.

- [ ] **Step 4: Generate the machine-readable readiness report**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.provenance readiness \
  --registry configs/thesis/experiment_registry_v2.toml \
  --seed-audit artifacts/preregistration/v2/seed-audit-before-pilot.json \
  --test-report artifacts/preregistration/v2/test-report.junit.xml \
  --correctness artifacts/preregistration/v2/correctness/v2 \
  --output-json artifacts/preregistration/v2/readiness-report.json \
  --output-markdown artifacts/preregistration/v2/readiness-report.md
```

Expected: `ready_for_pilot=true` and `ready_for_formal=false`. The latter must remain false until the power decision, trace lock, runtime budget and confirmatory lock exist.

### Task 2: Freeze GeoLife inputs and calibrate only with development seeds

**Files:**
- Use external input: `/home/link/my_workspace/.research-inputs/geolife/Geolife Trajectories 1.3.zip`
- Create generated: `artifacts/preregistration/v2/trace_lock.json`
- Create generated: `artifacts/calibration/v2/exact-nontriviality/`
- Create generated: `artifacts/calibration/v2/runtime-capacity/`

- [ ] **Step 1: Acquire the official archive without adding it to Git**

Open the Microsoft Research GeoLife download page recorded in the approved specification, select the official `Geolife Trajectories 1.3` archive, and save it at the exact external-input path above. Do not use a test fixture or a third-party mirror. Run:

```bash
test -f '/home/link/my_workspace/.research-inputs/geolife/Geolife Trajectories 1.3.zip'
sha256sum '/home/link/my_workspace/.research-inputs/geolife/Geolife Trajectories 1.3.zip'
```

Expected: a nonempty archive and one SHA-256 line. Record the official landing-page URL, resolved download URL, retrieval date, displayed dataset version, licence text hash and archive hash in the trace-lock input manifest. If licence acceptance or an interactive download prevents acquisition, stop before any formal seed is opened and request the archive from the user; never substitute synthetic data while retaining the `trace_driven` label.

- [ ] **Step 2: Freeze participant-disjoint trace scenarios**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.geolife_lock build \
  --archive '/home/link/my_workspace/.research-inputs/geolife/Geolife Trajectories 1.3.zip' \
  --protocol configs/thesis/geolife_protocol_v2.toml \
  --registry configs/thesis/experiment_registry_v2.toml \
  --output artifacts/preregistration/v2/trace_lock.json
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.geolife_lock verify \
  --lock artifacts/preregistration/v2/trace_lock.json \
  --archive '/home/link/my_workspace/.research-inputs/geolife/Geolife Trajectories 1.3.zip'
```

Expected: archive and licence hashes match; train, calibration and final participant sets are pairwise disjoint; each final composite scenario uses distinct participants; locked seeds are exactly `29000--29029`; seed changes cannot alter participant/trajectory binding.

- [ ] **Step 3: Calibrate Exact nontriviality using development seeds only**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_development_calibration \
  --registry configs/thesis/experiment_registry_v2.toml \
  --case exact_nontriviality \
  --seeds 600:649 \
  --output artifacts/calibration/v2/exact-nontriviality
```

Expected: every accessed seed lies in `0--999`; complete enumeration is verified; both `cpu_nontrivial_fraction` and `schedule_nontrivial_fraction` are reported. Adjusting the frozen Exact scenario is allowed only here and must be followed by tests, a source/config commit, and a complete restart of Tasks 1--2. No formal Exact seed may be inspected during calibration.

- [ ] **Step 4: Measure capacity with development seeds only**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_development_calibration \
  --registry configs/thesis/experiment_registry_v2.toml \
  --case runtime_capacity \
  --seeds 700:719 \
  --workers 2 \
  --trace-lock artifacts/preregistration/v2/trace_lock.json \
  --output artifacts/calibration/v2/runtime-capacity
```

Expected: descriptive per-family CPU time, wall time, P50/P95 slot time, peak RSS, disk bytes, event bytes, candidates and cache hits. This is a budgeting input, not a performance claim.

- [ ] **Step 5: Re-run readiness after any calibration change**

If no source/config/analysis file changed, continue. If any such file changed, the current attempt is terminal: commit the intended paths, preserve every existing readiness/trace/calibration byte, and stop for a protocol amendment that assigns a new exact attempt root before rerunning Task 1. Do not delete, overwrite, or ambiguously reuse the fixed paths above. No formal seed has been opened at this point, so candidate formal pools remain eligible only if the amended seed audit proves they are still unseen.

### Task 3: Run the final H1--H4 pilot and freeze power

**Files:**
- Create generated: `artifacts/preregistration/v2/power-pilot/final/`
- Create generated: `artifacts/preregistration/v2/power-pilot/final/decision/power_decision.json`

- [ ] **Step 1: Prove the final pilot output root is unused**

```bash
test ! -e artifacts/preregistration/v2/power-pilot/final
```

Expected: exit zero on a new attempt. An existing directory may be resumed only if its shard plan has the same source, registry, configuration, analysis and environment hashes and `MANIFEST.json` records `incomplete`; an identical complete bundle is verified and reused without rerunning. Any other existing directory is preserved in place and causes a stop for a protocol amendment naming a new exact attempt root; it is never overwritten or silently relabelled as final.

- [ ] **Step 2: Execute all four pilot families on seeds 1000--1099**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_power_pilot_suite \
  --registry configs/thesis/experiment_registry_v2.toml \
  --output artifacts/preregistration/v2/power-pilot/final \
  --workers 2 \
  --shard-size 1 \
  --simulations 10000 \
  --analysis-seed 20260719
```

Expected: Main, Externality unique H2 primary grid, Resource and Joint each contain exactly 100 completed seed shards; all four use identical final source/metric/analysis semantics; only run ID, pilot pool, log level and the preregistered formal method superset differ from formal definitions. The suite merges the four bundles and atomically writes the sole decision at `decision/power_decision.json`. The frozen exploratory `artifacts/pilot/power-v2-20260717` is neither resumed nor pooled.

- [ ] **Step 3: Verify and merge the pilot without selecting results**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_power_pilot_suite \
  --registry configs/thesis/experiment_registry_v2.toml \
  --output artifacts/preregistration/v2/power-pilot/final \
  --workers 2 \
  --shard-size 1 \
  --simulations 10000 \
  --analysis-seed 20260719 \
  --resume
decision_hash_1=$(sha256sum artifacts/preregistration/v2/power-pilot/final/decision/power_decision.json | cut -d' ' -f1)
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_power_pilot_suite \
  --registry configs/thesis/experiment_registry_v2.toml \
  --output artifacts/preregistration/v2/power-pilot/final \
  --workers 2 \
  --shard-size 1 \
  --simulations 10000 \
  --analysis-seed 20260719 \
  --resume
decision_hash_2=$(sha256sum artifacts/preregistration/v2/power-pilot/final/decision/power_decision.json | cut -d' ' -f1)
test "$decision_hash_1" = "$decision_hash_2"
```

Expected: both resume calls perform zero new work when complete, validate the immutable shard plans and merged inputs, and leave the decision byte-identical. Duplicate seeds, missing shards, metadata drift or payload conflicts are hard failures.

- [ ] **Step 4: Inspect the suite-owned common sample-size decision without recomputing it**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("artifacts/preregistration/v2/power-pilot/final/decision/power_decision.json")
decision = json.loads(path.read_text(encoding="utf-8"))
assert decision["candidate_sizes"] == [50, 100, 200]
assert decision["simulations"] == 10000
assert decision["analysis_seed"] == 20260719
assert decision["multiplicity_family"] == ["H1", "H2", "H3", "H4"]
assert decision["evaluated_n"] == 200
assert set(decision["power_table_by_n"]) == {"50", "100", "200"}
assert all(
    set(row) == {"H1", "H2", "H3", "H4"}
    for row in decision["power_table_by_n"].values()
)
assert decision["selected_n"] in {None, 50, 100, 200}
assert decision["evidence_insufficient"] is (decision["selected_n"] is None)
print(decision["selected_n"], decision["evidence_insufficient"])
PY
```

Expected: the estimand is seed-level paired mean completion-rate difference; each H1--H4 vector uses an independently spawned resampling stream; the test is two-sided paired t; each replicate applies one four-member Holm family. `selected_n` is the smallest common value in `{50,100,200}` with at least 80% zero-rejection power for all four MDEs. If no candidate qualifies, the file records `selected_n=null` and `evidence_insufficient=true`; it must not invent `N>200` or claim 80% power.

- [ ] **Step 5: Preserve the audited pilot bundle as the only power input**

Expected: `MANIFEST.json` and `SHA256SUMS` cover the pilot shards, four merges, resampling metadata, four stream IDs, decision and deterministic rerun hash. No operator invokes `uav_mec.run_power_pilot` directly; that module is library logic called inside the suite so execution and power cannot use different source hashes.

### Task 4: Freeze the runtime budget and confirmatory lock

**Files:**
- Create generated: `artifacts/preregistration/v2/runtime_budget.json`
- Create generated: `artifacts/preregistration/v2/source-snapshot/`
- Create generated: `artifacts/preregistration/v2/environment.json`
- Create generated: `artifacts/preregistration/v2/confirmatory-lock/confirmatory_lock.json`
- Create generated: `artifacts/preregistration/v2/confirmatory-lock/MANIFEST.json`
- Create generated: `artifacts/preregistration/v2/confirmatory-lock/SHA256SUMS`

- [ ] **Step 1: Build a complete budget before formal access**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.runtime_budget build \
  --registry configs/thesis/experiment_registry_v2.toml \
  --power-decision artifacts/preregistration/v2/power-pilot/final/decision/power_decision.json \
  --runtime-audit artifacts/calibration/v2/runtime-capacity/runtime_audit.json \
  --execution-policy configs/thesis/execution_policy_v2.json \
  --retry-fraction 0.10 \
  --output artifacts/preregistration/v2/runtime_budget.json
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.runtime_budget verify \
  --budget artifacts/preregistration/v2/runtime_budget.json
```

Expected: every family and case has measured-stratum-derived CPU hours, wall time, peak memory, disk, event-log cap, concurrency and retry allowance. Exact uses concurrency `1`; every other family uses concurrency `2`; shard size is `1` throughout. The verifier binds the registry, power decision, runtime audit, and `configs/thesis/execution_policy_v2.json` hashes. `primary_sample_size` equals `selected_n` when available and otherwise the audited `evaluated_n=200`, with `power_guarantee=false`. If the measured plan is not affordable, stop and obtain approval for a revised preregistered protocol; do not silently reduce a case, grid, seed prefix, or slot count.

- [ ] **Step 2: Re-audit candidate-pool blindness**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.seed_access audit \
  --registry configs/thesis/experiment_registry_v2.toml \
  --ledger artifacts/preregistration/v2/seed-access.jsonl \
  --phase before-formal-lock \
  --output artifacts/preregistration/v2/seed-audit-before-lock.json
```

Expected: pilot accesses are exactly `1000--1099`; formal pools have no trace, run or analysis access. Any formal access invalidates the full family pool and requires a new disjoint pool to be approved before continuing.

- [ ] **Step 3: Capture and verify the exact source and environment**

```bash
test ! -e artifacts/preregistration/v2/source-snapshot
test ! -e artifacts/preregistration/v2/environment.json
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.provenance capture \
  --repo-root . \
  --source-output artifacts/preregistration/v2/source-snapshot \
  --environment-output artifacts/preregistration/v2/environment.json
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.provenance verify \
  --source-snapshot artifacts/preregistration/v2/source-snapshot \
  --environment artifacts/preregistration/v2/environment.json
```

Expected: the committed research tree, any complete binary patch, fixed interpreter, OS/architecture, and dependency versions are checksum-closed. Formal execution requires a clean research-source state; if capture reports a patch, stop, commit the intended allowlisted source change, rerun all tests, and restart Tasks 1--4 rather than locking a transient implementation state.

- [ ] **Step 4: Generate the immutable lock**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.confirmatory_lock build \
  --registry configs/thesis/experiment_registry_v2.toml \
  --power-decision artifacts/preregistration/v2/power-pilot/final/decision/power_decision.json \
  --runtime-budget artifacts/preregistration/v2/runtime_budget.json \
  --trace-lock artifacts/preregistration/v2/trace_lock.json \
  --seed-ledger artifacts/preregistration/v2/seed-access.jsonl \
  --source-snapshot artifacts/preregistration/v2/source-snapshot \
  --environment artifacts/preregistration/v2/environment.json \
  --correctness artifacts/preregistration/v2/correctness/v2 \
  --output artifacts/preregistration/v2/confirmatory-lock
```

Expected: the lock records a clean source commit, source snapshot hash, registry/config/analysis/environment hashes, paired estimand, t-test/Holm family, MDEs, common `selected_n`, exact per-family seed lists, fixed family order, runtime plan and allowed commands. If `selected_n=null`, freeze the conservative `N=200` prefixes with `power_guarantee=false` and the exact reason; do not state that 80% was achieved.

- [ ] **Step 5: Verify the lock twice and make it read-only**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.confirmatory_lock verify \
  --lock artifacts/preregistration/v2/confirmatory-lock
lock_hash_1=$(sha256sum artifacts/preregistration/v2/confirmatory-lock/confirmatory_lock.json | cut -d' ' -f1)
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.confirmatory_lock verify \
  --lock artifacts/preregistration/v2/confirmatory-lock
lock_hash_2=$(sha256sum artifacts/preregistration/v2/confirmatory-lock/confirmatory_lock.json | cut -d' ' -f1)
test "$lock_hash_1" = "$lock_hash_2"
test -f artifacts/preregistration/v2/confirmatory-lock/MANIFEST.json
test -f artifacts/preregistration/v2/confirmatory-lock/SHA256SUMS
chmod -R a-w artifacts/preregistration/v2/confirmatory-lock
```

Expected: both verifications pass and hashes match. Do not commit generated preregistration data into the source branch; canonical full later archives it.

### Task 5: Execute the eight formal families in the frozen order

**Files:**
- Create generated: `artifacts/runs/v2-confirmatory/resource/`
- Create generated: `artifacts/runs/v2-confirmatory/exact/`
- Create generated: `artifacts/runs/v2-confirmatory/externality/`
- Create generated: `artifacts/runs/v2-confirmatory/main/`
- Create generated: `artifacts/runs/v2-confirmatory/joint/`
- Create generated: `artifacts/runs/v2-confirmatory/scaling/`
- Create generated: `artifacts/runs/v2-confirmatory/robustness/`
- Create generated: `artifacts/runs/v2-confirmatory/trace_driven/`

- [ ] **Step 0: Verify the append-only ledger boundary before opening formal seeds**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.seed_access audit \
  --registry configs/thesis/experiment_registry_v2.toml \
  --ledger artifacts/preregistration/v2/seed-access.jsonl \
  --phase before-formal-run \
  --lock artifacts/preregistration/v2/confirmatory-lock \
  --output artifacts/preregistration/v2/seed-audit-before-formal-run.json
```

Expected: the live ledger begins with the byte-identical prefix copied into the lock; no post-prefix record exists yet; every later record will require a formal permit bound to this lock, family, and case. Formal and analysis roots do not contain a mutable ledger copy.

- [ ] **Step 1: Start Resource and verify it before continuing**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory \
  --lock artifacts/preregistration/v2/confirmatory-lock \
  --family resource \
  --root artifacts/runs/v2-confirmatory
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory \
  --lock artifacts/preregistration/v2/confirmatory-lock \
  --family resource \
  --root artifacts/runs/v2-confirmatory \
  --verify-only
```

Expected: fixed-radio policies share byte-identical raw snapshots; DD and Exact Admission agree on completion cardinality and same-cardinality minimum CPU for all applicable instances; primary and both generator sensitivities are retained. Any Oracle disagreement stops the protocol before Exact.

- [ ] **Step 2: Run and verify Exact**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family exact --root artifacts/runs/v2-confirmatory
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family exact --root artifacts/runs/v2-confirmatory --verify-only
```

Expected: complete enumeration or a verifiable zero-gap certificate is distinguished from timeout; completion, CPU and exposure gaps are separate; CPU and scheduling nontrivial fractions are separate. Falling below 30% closes only the corresponding performance claim and does not justify changing the locked scenario.

- [ ] **Step 3: Run and verify Externality**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family externality --root artifacts/runs/v2-confirmatory
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family externality --root artifacts/runs/v2-confirmatory --verify-only
```

Expected: only the unique high-overlap/high-load/high-coupling grid contributes H2; all other grids remain descriptive; accepted certificates preserve task/block aggregate identities and contain reproducible third-party impacts.

- [ ] **Step 4: Run and verify Main**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family main --root artifacts/runs/v2-confirmatory
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family main --root artifacts/runs/v2-confirmatory --verify-only
```

Expected: all methods share exogenous traces; every accepted move strictly improves closed-state `Phi`; `iteration_cap` cases are retained and never counted as local optima; the closest-work adaptation is present with its mapping manifest.

- [ ] **Step 5: Run and verify Joint**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family joint --root artifacts/runs/v2-confirmatory
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family joint --root artifacts/runs/v2-confirmatory --verify-only
```

Expected: all four cells `iea_dd`, `exchange_dd`, `iea_edf`, `exchange_edf` exist for every locked seed. H4 is the preregistered `iea_dd - exchange_edf` diagonal endpoint; the four conditional effects and difference-in-differences interaction are retained as secondary estimands.

- [ ] **Step 6: Run and verify Scaling**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family scaling --root artifacts/runs/v2-confirmatory
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family scaling --root artifacts/runs/v2-confirmatory --verify-only
```

Expected: every preregistered load and equal-per-UAV scaling cell is present, with reference hardware, candidates, cache hit rate, memory, disk and CPU-time provenance.

- [ ] **Step 7: Run and verify Robustness**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family robustness --root artifacts/runs/v2-confirmatory
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family robustness --root artifacts/runs/v2-confirmatory --verify-only
```

Expected: every frozen one-axis/small-factor case is present. Realized CSI outage is retained as robustness data, while planning SIC, duplicate assignment, capacity and CPU violations remain hard integrity failures.

- [ ] **Step 8: Run and verify Trace-driven**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family trace_driven --root artifacts/runs/v2-confirmatory
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --lock artifacts/preregistration/v2/confirmatory-lock --family trace_driven --root artifacts/runs/v2-confirmatory --verify-only
```

Expected: source participant bindings match `trace_lock.json`; seeds control only orthogonal exogenous quantities; intervals, if emitted, use participant-clustered bootstrap; prose metadata labels GeoLife as a mobility supplement rather than UAV-MEC workload data.

- [ ] **Step 9: Verify the complete formal root and frozen order**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory \
  --lock artifacts/preregistration/v2/confirmatory-lock \
  --root artifacts/runs/v2-confirmatory \
  --verify-all
```

Expected: all locked seeds and only locked seeds exist; all schemas/checksums/hash bindings match; no seed conflict or missing shard exists; family start/finalization order matches the lock.

If a process or machine stops, rerun the same family command with `--resume`; the runner executes only missing valid shards. Never alter the lock, append a seed, skip an unfavorable shard, or merge data from a different run root.

### Task 6: Analyze H1--H4 and separate integrity from performance claims

**Files:**
- Create generated: `artifacts/analysis/v2-confirmatory/`

- [ ] **Step 1: Run the registry-driven analyzer**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.analyze_confirmatory \
  build \
  --lock artifacts/preregistration/v2/confirmatory-lock \
  --runs-root artifacts/runs/v2-confirmatory \
  --output artifacts/analysis/v2-confirmatory
```

Expected: exactly H1--H4 paired t p-values enter the primary Holm family. Each hypothesis also contains a 10,000-repeat paired bootstrap mean CI, Wilcoxon sensitivity, matched rank-biserial effect, MDE and episode-level completed-task translation. Joint includes four conditional contrasts plus the separate interaction CI. Exact contains gaps without pseudo-significance tests.

- [ ] **Step 2: Audit integrity and correctness gates**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.analyze_confirmatory \
  verify \
  --lock artifacts/preregistration/v2/confirmatory-lock \
  --runs-root artifacts/runs/v2-confirmatory \
  --analysis artifacts/analysis/v2-confirmatory
```

Expected: `integrity_gates` and `correctness_gates` are structurally separate from `claim_gates.H1` through `claim_gates.H4`. A negative or uncertain H1--H4 result must not turn an otherwise valid archive into an integrity failure.

- [ ] **Step 3: Freeze claim wording states without manual editing**

Verify that each H1--H4 state is exactly one of:

```text
meaningful_supported
positive_mde_observed_but_uncertain
positive_but_small
insufficient
negative
```

The analyzer derives the state from the point estimate, bootstrap CI, MDE and Holm-adjusted paired-t result. Do not revise the state because a different baseline or subgroup looks better.

- [ ] **Step 4: Reproduce the analysis byte-for-byte**

```bash
test ! -e artifacts/analysis/v2-confirmatory.first
mv artifacts/analysis/v2-confirmatory artifacts/analysis/v2-confirmatory.first
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.analyze_confirmatory \
  build \
  --lock artifacts/preregistration/v2/confirmatory-lock \
  --runs-root artifacts/runs/v2-confirmatory \
  --output artifacts/analysis/v2-confirmatory
diff -qr artifacts/analysis/v2-confirmatory.first artifacts/analysis/v2-confirmatory
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.analyze_confirmatory \
  verify \
  --lock artifacts/preregistration/v2/confirmatory-lock \
  --runs-root artifacts/runs/v2-confirmatory \
  --analysis artifacts/analysis/v2-confirmatory
```

Expected: no differences. Keep both directories until canonical publication succeeds; afterward the first copy remains an ignored reproducibility audit or is moved to a dated audit archive, never silently overwritten.

### Task 7: Publish and verify canonical-v2 full/light

**Files:**
- Create generated: `artifacts/canonical/v2/MANIFEST.json`
- Create generated: `artifacts/canonical/v2/SHA256SUMS`
- Create generated: `artifacts/canonical/v2/full/`
- Create generated: `artifacts/canonical/v2/light/`

- [ ] **Step 1: Prove the canonical destination is unused**

```bash
test ! -e artifacts/canonical/v2
```

Expected: exit zero. If the path already exists and complete verification passes, reuse the sealed bundle without publishing again. If it exists but verification fails, preserve it and stop for a new exact canonical attempt path; never partially reuse or overwrite it as an official bundle.

- [ ] **Step 2: Publish full from immutable inputs and derive light**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.canonical_v2 publish \
  --lock artifacts/preregistration/v2/confirmatory-lock \
  --runs-root artifacts/runs/v2-confirmatory \
  --analysis artifacts/analysis/v2-confirmatory \
  --output artifacts/canonical/v2
```

Expected: full contains configs, source snapshot, environment, traces, raw shard bytes, summaries, event certificates, statistics, figures, LaTeX staging, claim matrix, acceptance and QA. Light is deterministically derived from full and records a full path/hash for every file. The publisher refuses any integrity gate failure but preserves negative performance claim states.

- [ ] **Step 3: Verify canonical closure and source reproducibility**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.canonical_v2 verify \
  --root artifacts/canonical/v2
(
  cd artifacts/canonical/v2
  sha256sum --check SHA256SUMS
)
```

Expected: every file is root-checksummed; every claim evidence path resolves; every light file resolves to a full hash; all locked formal seeds are present; source is a clean snapshot or includes a complete patch/source byte archive. No path points to lifecycle-dependent pilot, smoke, run or canonical-v1 data.

- [ ] **Step 4: Run the publication-source guard before manuscript writing**

```bash
make verify-canonical-v2 \
  CANONICAL_V2=artifacts/canonical/v2
```

Expected: the publisher has already generated bilingual figures, tables and LaTeX macros inside its temporary tree before sealing `full` and deriving `light`; their manifests record claim IDs and source hashes. Verification is read-only. No command generates or edits a publication asset after `MANIFEST.json` and `SHA256SUMS` exist, and no v1/pilot/smoke/raw-run numerical source is accepted.

### Task 8: Record the execution handoff without overstating results

**Files:**
- Read: `artifacts/canonical/v2/full/analysis/acceptance.json`
- Read: `artifacts/canonical/v2/full/summaries/`
- Read: `artifacts/canonical/v2/MANIFEST.json`

- [ ] **Step 1: Record observed outcomes and limitations**

Report to the user the actual selected N, seed counts, wall/CPU time, peak memory, disk usage, retries, iteration-cap rate, Exact nontriviality fractions, integrity/correctness status and each H1--H4 claim state by reading the sealed canonical files. Do not create, edit, or append any file under `artifacts/canonical/v2` after publication. Do not translate an uncertain or negative state into a positive claim.

- [ ] **Step 2: Enumerate protocol deviations**

Confirm that the sealed summaries enumerate every retry, interrupted shard and environment variation. A deterministic retry of a missing shard under the same lock is not excluded; a seed/config/source change is a protocol violation and prevents canonical publication until resolved under an explicitly documented replacement protocol.

- [ ] **Step 3: Hand canonical-v2/light to Plan 5**

Plan 5 may now write results, discussion, abstracts, conclusions, highlights and cover letter. It must consume generated macros/figures and claim states rather than copying numbers by hand.

## Stop and rollback rules

- A planning duplicate, block-capacity, planned SIC or CPU violation stops the active family immediately. Use `systematic-debugging`. Any code/config/analysis change invalidates the lock; it does not authorize reuse of the fixed pilot, lock, formal, analysis, or canonical paths.
- A realized CSI outage is retained as robustness evidence and does not invalidate planning integrity.
- A DD/Exact Admission mismatch stops before further Resource claims; it cannot be waived as noise.
- An incremental/full evaluation mismatch stops all search-based formal work.
- An `iteration_cap` result remains in the data and closes the local-optimum claim for that instance; it is not relabelled.
- H1--H4 failure never authorizes tuning, seed deletion, subgroup substitution or optional stopping. Valid negative evidence is still canonicalized.
- Before any formal access, a semantics-preserving implementation fix still requires an approved new attempt root, a fresh final pilot, budget, source snapshot and lock. After any formal seed has been read, a source/config/analysis change additionally contaminates every accessed family candidate pool: stop, preserve the entire attempt, amend the approved specification and registry with fully disjoint replacement ranges, and obtain approval before running again. Unchanged-process interruption is the only condition that may use `--resume` under the same lock. Reducing the registered neighborhood while keeping the local-optimum claim is forbidden.
- If canonical integrity cannot pass, no result-dependent manuscript text is final. The professional-master thesis may still be completed with verified correctness/mechanism evidence and honestly scoped claims after a replacement protocol is approved.
