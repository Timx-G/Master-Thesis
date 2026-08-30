# Thesis Quality and Evidence Upgrade Implementation Plan

> **For agentic workers:** Execute each stage against immutable traces, preserve `artifacts/canonical/v1/`, and do not publish `canonical/v2` until every declared gate passes.

**Goal:** Strengthen the professional-master thesis with stronger resource-allocation baselines, externality stress experiments, mechanism evidence, recent-literature positioning, and a fully traceable second evidence release.

**Architecture:** Extend the existing configuration-driven experiment runner rather than creating a second simulator. All new methods must pass through the same realized-schedule evaluator, use the same seed-level traces, and emit the existing atomic CSV/JSON/SHA-256 evidence bundle. New exploratory outputs live outside `canonical/v1`; only a frozen, audited bundle may be promoted to `canonical/v2`.

**Tech Stack:** Python 3.11, NumPy/SciPy, dataclasses/TOML, pytest, paired bootstrap/Wilcoxon-Holm analysis, LaTeX, PptxGenJS/OOXML.

---

## 1. Research-quality diagnosis

The current evidence already establishes implementation correctness, strict two-factor ablation, paired statistical significance, small-instance complete enumeration, and zero recorded SIC/CPU-budget violations. Four gaps prevent a stronger thesis claim:

1. `Fixed 0.5/Equal` and `Crossing/Equal` are controlled ablations, not state-of-the-art literature algorithms.
2. The formal `K=2, N=20` case is close to the completion ceiling, so innovation-one effects are statistically stable but small.
3. Existing UAV-count scaling keeps user count fixed, which reduces per-UAV pressure and cannot test scalability under comparable load.
4. The paper reports aggregate completion and interference but lacks direct mechanism diagnostics for third-party externality effects.
5. The eight thesis chapter files are presently a compact technical manuscript rather than a full professional-master dissertation: derivations, algorithm pseudocode, parameter provenance, literature comparison, implementation architecture, negative results, and chapter summaries need substantial expansion. Page or word targets must follow the university template, but expansion must add evidence and reasoning rather than filler.

The thesis must therefore distinguish three evidence types: controlled ablation, optimization/oracle evidence, and literature/engineering comparison.

## 2. Frozen experiment additions

### 2.1 Resource-allocation evidence

Add policies that reuse the full method's wireless schedule:

- `EDF-MinCPU`: derive each task's minimum feasible CPU, then admit by earliest absolute/relative deadline with stable task-ID ties.
- `MinSlack-MinCPU`: derive minimum requirements, then admit the currently least-slack task first.
- `Exact-Oracle`: enumerate feasible per-UAV task subsets only for declared small instances, maximizing completion count and then minimizing CPU cost.
- Keep `Crossing/Equal` and `Fixed 0.5/Equal` as controlled lower baselines, explicitly labeled as such.

Required comparisons:

- DD-MCAA versus EDF-MinCPU and MinSlack-MinCPU on identical schedules and 50 paired seeds.
- DD-MCAA completion count versus Exact-Oracle on every declared small resource instance.
- CPU utilization, admitted-task count, missed-deadline reason, and runtime in addition to completion rate.

### 2.2 Externality stress evidence

Predeclare the following cases before viewing results:

- Equal-load scaling: `(K,N)=(2,20),(3,30),(4,40),(5,50)`.
- Default and heavy task loads, with a combined heavy-task/high-arrival/high-overlap case.
- Default and stronger cross-receiver coupling.
- Default and higher user/UAV coverage overlap where supported by the model.
- CSI-error and heterogeneous CPU cases retained as robustness rather than innovation claims.

Mechanism outputs should include accepted move type, before/after completion, before/after cross interference, number of affected non-participating tasks, and cases where bilateral and global acceptance disagree. If a metric cannot be derived without changing algorithm semantics, record the limitation instead of approximating it.

### 2.3 Literature comparison

Use a three-level rule:

1. Faithfully reproduced published algorithm, only if source/code and all required variables are available.
2. Clearly named `*-adapted` baseline when trajectory/power variables must be frozen to match this thesis.
3. Comparison-table-only treatment when objectives and decision variables are not commensurate.

No adapted method may be called a faithful SOTA reproduction. The closest 2024--2026 multi-UAV NOMA-MEC papers must be compared by scenario, objective, decisions, constraints, and metric before any implementation choice.

## 3. Predeclared acceptance gates

The v2 evidence release is publishable only if all applicable gates pass:

1. All automated tests and evidence-contract tests pass.
2. Every final schedule has zero task-uniqueness, SIC, and CPU-budget violations.
3. DD-MCAA matches Exact-Oracle completion cardinality on all declared small resource instances.
4. DD-MCAA is not worse than EDF-MinCPU or MinSlack-MinCPU in mean paired completion; all effect sizes and confidence intervals are reported even when small.
5. Equal-load stress experiments complete without search non-convergence, and runtime P50/P95 is reported for every scale.
6. Innovation-one claims are limited to cases supported by confidence intervals and mechanism logs; no universal dominance or monotonic-interference claim is allowed.
7. Literature comparisons disclose every adaptation and use the same trace, radio model, CPU budget, task denominator, and final evaluator.
8. Manuscript, figures, reports, and PPT read all final numbers from the published evidence bundle rather than manual transcription.
9. Every thesis chapter contains a clear research question, assumptions or inputs, method/evidence, boundary discussion, and chapter summary where appropriate; no unrun result or unsupported SOTA claim appears merely to increase length.

## 4. Execution stages

### Stage A: Code and unit tests

- Add allocation-policy tests before policy implementations.
- Add suite-construction tests for equal-load scaling and stress cases.
- Add evidence-schema tests for new mechanism metrics.
- Run focused tests, then the complete pytest suite.

### Stage B: Exploratory dry run

- Run 2--5 seeds and shortened slots only to detect crashes, saturation, and unbounded runtime.
- Do not quote these results in the thesis.
- Freeze final seed range, slots, parameters, comparisons, and gates after the dry run.

### Stage C: Confirmatory execution

- Use unseen seeds `2000--2049` for strong resource comparisons and `3000--3049` for externality stress so development and confirmation remain separated and the two evidence families are independently reproducible.
- Run resource strong-baseline, resource exact-oracle, equal-load stress, and mechanism suites into distinct non-overwriting directories.
- Generate checksums immediately when each bundle closes.

### Stage D: Analysis and publication

- Validate every bundle and compute same-seed paired statistics with Holm correction.
- Publish a versioned canonical bundle only after gate evaluation.
- Generate figures, LaTeX macros, tables, and a machine-readable claim matrix from canonical evidence.

### Stage E: Thesis and defense materials

- Reframe the two contributions as algorithm/mechanism innovations supported by a unified model, not as a first-ever scenario.
- Rename the current DD-MCAA page to a controlled ablation and add a separate strong-baseline/oracle result when data exist.
- Expand related work, formal assumptions, complexity, mechanism analysis, external validity, and negative-result discussion.
- Keep future or failed comparisons visible as limitations; do not silently remove adverse cases.

## 5. Deliverables

- New tested allocation policies and experiment methods.
- Frozen v2 experiment configurations and run manifest.
- Strong-baseline and exact-oracle result bundles.
- Equal-load externality stress and mechanism result bundles.
- Updated statistical analysis, figures, generated macros, and claim matrix.
- Revised article, thesis chapters, midterm report, and defense deck with consistent evidence boundaries.
- Reproduction commands and final QA report.
