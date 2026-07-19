# Evidence Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable evidence pipeline that freezes the approved registry and seed policy, performs the preregistered paired inference and power analysis, binds every formal run to an immutable confirmatory lock, and publishes a complete `canonical/v2/full` archive plus a deterministic `canonical/v2/light` derivative.

**Architecture:** The registry is the sole declaration of evidence families, candidate seed pools, primary hypotheses, case grids, and execution order. A mandatory seed-access recorder guards every trace generation, run, and analysis; pilot results feed a paired-t/Holm power decision, runtime and GeoLife locks are frozen, and those inputs are sealed into one confirmatory lock before any formal seed is opened. Family-specific runners produce immutable generic shards, a registry-driven analyzer separates integrity, correctness, performance, and optional claims, and the canonical publisher copies the authoritative raw bytes into `full` before deriving `light` by hash.

**Tech Stack:** Python 3.11 at `/home/link/miniconda/envs/llm-ops-torch26/bin/python`, dataclasses, TOML/JSON/CSV/JSONL, NumPy, SciPy, pytest, SHA-256, Git source snapshots, Make.

---

## Non-negotiable execution rules

- Use `/home/link/miniconda/envs/llm-ops-torch26/bin/python` in every Python and pytest command in this plan.
- During implementation and unit tests, generate traces only with development seeds `0` through `999`. Reading or asserting an integer in a config is not a seed access; generating a trace, running an algorithm, or analyzing result rows is an access and must enter the ledger.
- Do not run pilot or formal experiments while implementing this plan. Pilot and formal commands in Task 12 are operator handoff commands to execute only after the implementation commit is clean and verified.
- Do not reuse or append the stopped partial pilot. Its Main 100-shard and Externality 48-shard material remains exploratory and cannot be merged with the new final pilot.
- Never pass a seed, seed count, slot count, method list, or scenario override to the formal runner. The verified confirmatory lock is the only source of those values.
- Do not stage generated content under `artifacts/pilot`, `artifacts/smoke`, `artifacts/formal`, `artifacts/preregistration`, or `artifacts/canonical`.
- Every commit command below uses an explicit file allowlist. Before committing, run `git diff --cached --name-only` and compare it to that task's allowlist.
- Preserve v1 entry points for historical reproducibility. New v2 code lives in focused modules; compatibility wrappers may re-export public v2 APIs but may not silently change v1 bundle semantics.

## Approved protocol constants

| Purpose or family | Candidate seeds | Slots | Sample-size rule |
|---|---:|---:|---|
| Development | `0..999` | test-specific | never formal evidence |
| Power pilot | `1000..1099` | `100` | exactly 100 per H1--H4 family |
| Resource | `22000..22199` | `100` | common power-selected prefix |
| Externality | `23000..23199` | `100` | common power-selected prefix for the unique H2 primary grid; fixed 30 for each of the other eight grids |
| Main | `24000..24199` | `100` | common power-selected prefix |
| Robustness | `25000..25029` | `50` | fixed 30 |
| Exact | `26000..26049` | `20` | fixed 50 |
| Joint | `27000..27199` | `100` | common power-selected prefix |
| Scaling | `28000..28029` | `50` | fixed 30 |
| Trace-driven | `29000..29029` | `50` | fixed 30 |

The common power-selected size is the smallest value in `(50, 100, 200)` for which every H1--H4 Holm-adjusted rejection probability is at least `0.80` under the registered MDE. If no candidate passes, `selected_n` is null and no positive H1--H4 formal claim is authorized. A separately authorized conservative run may use 200 only when the lock records `power_guarantee=false` and the manuscripts state that no verified 80 percent guarantee exists.

The four primary hypothesis rows are exact:

| ID | Family / case | Treatment minus control | MDE |
|---|---|---|---:|
| H1 | `main/main_k3_n45` | `full - intra_global` | `0.5 pp` |
| H2 | `externality/overlap_high_coupling1_load12` | `full - cross_bilateral` | `0.5 pp` |
| H3 | `resource/dd_generated` | `fixed_radio_dd - fixed_radio_edf` | `2.0 pp` |
| H4 | `joint/joint_k3_n45` | `iea_dd - exchange_edf` | `2.0 pp` |

Registry case IDs, roles, and sample policies are also frozen; suite builders must consume these rows instead of retaining a second hard-coded list:

| Family | Cases | Role and sample policy |
|---|---|---|
| Resource | `dd_generated`; `radio_neutral`; `deadline_first_generated` | first is primary, others supporting; each uses the common selected N |
| Exact | `exact_schedule` | correctness; fixed 50, with mechanism claim gates reported separately |
| Externality | `overlap_low_coupling0_load08`; `overlap_low_coupling0_load12`; `overlap_low_coupling1_load08`; `overlap_low_coupling1_load12`; `overlap_high_coupling0_load08`; `overlap_high_coupling0_load12`; `overlap_high_coupling1_load08`; `overlap_high_coupling1_load12`; `overlap_medium_coupling05_load10` | `overlap_high_coupling1_load12` is primary and uses selected N; each other case is supporting and uses exactly 30 |
| Main | `main_k3_n45` | primary; selected N |
| Joint | `joint_k3_n45` | primary; selected N |
| Scaling | `load_k3_n30`; `load_k3_n45`; `load_k3_n60`; `equal_k2_n30`; `equal_k3_n45`; `equal_k4_n60`; `equal_k5_n75` | supporting; fixed 30 per case; the first three are K=3 load cells and the latter four are equal-per-UAV cells |
| Robustness | `csi0`; `csi05`; `csi10`; `csi20`; `sic_residual0`; `sic_residual02`; `sic_residual05`; `sic_residual10`; `sic_threshold_m3`; `sic_threshold0`; `sic_threshold3`; `sic_threshold5`; `cpu_homogeneous`; `cpu_heterogeneous`; `deadline_light`; `deadline_default`; `deadline_heavy`; `arrival_periodic`; `arrival_jitter`; `arrival_burst`; `pathloss_al_hourani`; `pathloss_3gpp_uma_av_audited` | supporting; fixed 30 per case; each case records its one-axis factor and locked parameter table |
| Trace-driven | `geolife_final` | supporting external-validity case; fixed 30 |

Externality’s primary case additionally freezes `K=3`, `N=45`, `S=3`, `cluster_layout="pairwise_overlap"`, `cluster_std=5.0`, `period_choices=[2]`, and `cross_interference_scale=1.0`. Case seed sets may overlap within one family because cases are paired on that family’s candidate prefix; candidate pools remain disjoint across families.

## File responsibility map

### Create

- `uav_mec/seed_access.py` — seed purposes, permits, append-only hash-chained ledger, and pre-lock access audit.
- `uav_mec/resource_snapshots.py` — immutable raw radio/rate snapshots and fixed-rate resource-policy evaluation.
- `uav_mec/run_resource_experiment.py` — one-search-per-generator Resource/H3 runner.
- `uav_mec/runtime_budget.py` — registry-complete CPU, wall-clock, memory, disk, log, concurrency, and retry budget.
- `uav_mec/geolife_lock.py` — participant-disjoint GeoLife protocol and trace-lock generation/validation.
- `uav_mec/run_trace_experiment.py` — locked trace-driven descriptive runner.
- `uav_mec/provenance.py` — deterministic source, patch, config, analysis, and environment snapshots.
- `uav_mec/correctness_bundle.py` — checksum-closed software-test and algorithm-correctness evidence assembled before the final pilot.
- `uav_mec/confirmatory_lock.py` — immutable confirmatory-lock builder and verifier.
- `uav_mec/run_confirmatory.py` — ordered formal family dispatcher bound to the lock.
- `uav_mec/claim_gates.py` — machine-readable H1--H4, Exact, mechanism, efficiency, and real-time claim classification.
- `uav_mec/analyze_confirmatory.py` — registry-driven v2 validation, paired inference, Joint estimands, and acceptance production.
- `uav_mec/canonical_v2.py` — authoritative full archive, deterministic light derivation, and closure verification.
- `configs/thesis/confirmatory_trace_v2.toml` — trace-driven formal experiment definition.
- `configs/thesis/geolife_protocol_v2.toml` — dataset source, license, modes, resampling, placement, and split policy.
- `configs/thesis/confirmatory_scaling_v2.toml` — K=3 load and equal-per-UAV scaling cells.
- `configs/thesis/confirmatory_robustness_v2.toml` — one-axis CSI, SIC, CPU/deadline, arrivals, CPU heterogeneity, and path-loss cells.
- `configs/thesis/execution_policy_v2.json` — measured-machine concurrency and per-case shard policy hashed into the runtime budget.
- `tests/conftest.py` — the fully defined `development_seed_access` fixture used by every trace-generating unit test.
- `tests/test_seed_access.py`
- `tests/test_resource_snapshots.py`
- `tests/test_runtime_budget.py`
- `tests/test_geolife_lock.py`
- `tests/test_provenance.py`
- `tests/test_correctness_bundle.py`
- `tests/test_confirmatory_lock.py`
- `tests/test_confirmatory_analysis.py`
- `tests/test_claim_gates.py`
- `tests/test_canonical_v2.py`

### Modify

- `configs/thesis/experiment_registry_v2.toml`
- `configs/thesis/confirmatory_resource_fixed_v2.toml`
- `configs/thesis/externality_stress.toml`
- `configs/thesis/confirmatory_main_v2.toml`
- `configs/thesis/confirmatory_exact.toml`
- `configs/thesis/confirmatory_joint_v2.toml`
- `uav_mec/experiment_registry.py`
- `uav_mec/trace.py`
- `uav_mec/experiment.py`
- `uav_mec/evaluation.py`
- `uav_mec/externality_analysis.py`
- `uav_mec/statistics.py`
- `uav_mec/power_analysis.py`
- `uav_mec/run_power_pilot.py`
- `uav_mec/run_power_pilot_suite.py`
- `uav_mec/exact.py`
- `uav_mec/run_exact_benchmark.py`
- `uav_mec/run_shards.py`
- `uav_mec/runtime_audit.py`
- `uav_mec/geolife.py`
- `uav_mec/artifacts.py`
- `uav_mec/analyze_results.py`
- `uav_mec/publish_canonical.py`
- `uav_mec/run_thesis_suite.py`
- `tests/test_experiment_registry.py`
- `tests/test_trace.py`
- `tests/test_simulation.py`
- `tests/test_csi_error.py`
- `tests/test_experiment_cli.py`
- `tests/test_ia_greedy.py`
- `tests/test_resource_policies.py`
- `tests/test_thesis_suites.py`
- `tests/test_externality_analysis.py`
- `tests/test_capacity_analysis.py`
- `tests/test_statistics.py`
- `tests/test_power_analysis.py`
- `tests/test_power_pilot.py`
- `tests/test_exact.py`
- `tests/test_exact_benchmark.py`
- `tests/test_run_shards.py`
- `tests/test_runtime_audit.py`
- `tests/test_geolife.py`
- `tests/test_artifacts.py`
- `tests/test_result_analysis.py`
- `tests/test_canonical_publication.py`
- `Makefile`

### Runtime outputs, never staged in source commits

```text
artifacts/preregistration/v2/
  seed-access.jsonl
  correctness/v2/
    MANIFEST.json
    SHA256SUMS
  power-pilot/final/
    MANIFEST.json
    SHA256SUMS
    decision/power_decision.json
  runtime_budget.json
  trace_lock.json
  confirmatory-lock/
    confirmatory_lock.json
    MANIFEST.json
    SHA256SUMS

artifacts/runs/v2-confirmatory/
  MANIFEST.json
  SHA256SUMS
  resource/
  exact/
  externality/
  main/
  joint/
  scaling/
  robustness/
  trace_driven/

artifacts/analysis/v2-confirmatory/
  MANIFEST.json
  SHA256SUMS
  statistics.json
  metrics.json
  acceptance.json
  statistical_claims.csv

artifacts/canonical/v2/
  MANIFEST.json
  SHA256SUMS
  full/
    MANIFEST.json
    SHA256SUMS
    analysis/statistical_claims.csv
    analysis/claim_matrix.csv
  light/
    MANIFEST.json
    SHA256SUMS
    statistical_claims.csv
    claim_matrix.csv
```

Every new v2 directory bundle, including source snapshots, calibration bundles, shard roots, case merges, formal root, analysis root, and canonical root, uses the exact closure filenames `MANIFEST.json` and `SHA256SUMS`. `SHA256SUMS` covers every regular file beneath that bundle except itself, with POSIX-relative paths sorted bytewise. `SUCCESS.json` remains the shard completion marker but never replaces either closure file. Lowercase `manifest.json` and `checksums.sha256` remain v1-only and are rejected inside a v2 root.

## Public API contracts

Implement the following declarations exactly. These are complete declaration tables, not partial code; collection fields described as “immutable sequence” are stored as tuples in the implementation.

### `uav_mec/experiment_registry.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| frozen slotted dataclass `RegisteredCase` | `case_id: str`; `role: str`; `fixed_sample_size: int or None` | `role` is one of `primary`, `supporting`, or `correctness`; a non-null fixed size is positive. |
| frozen slotted dataclass `ExperimentFamily` | `name: str`; `config_path: Path`; `seed_start: int`; `seed_count: int`; `num_slots: int`; `methods: immutable sequence of str`; `sample_size_policy: str`; `formal_order: int`; `cases: immutable sequence of RegisteredCase` | Names and case IDs are unique, seeds are a contiguous closed-open range, `num_slots` is positive, and formal orders are unique. |
| frozen slotted dataclass `PrimaryHypothesis` | `hypothesis_id: str`; `family: str`; `case_id: str or None`; `treatment: str`; `control: str`; `outcome: str`; `estimand: str`; `primary_test: str`; `sensitivity_test: str`; `multiplicity_family: str`; `mde_pp: float` | The family and optional case resolve in the same registry; treatment and control are registered methods; `mde_pp` is positive. |
| frozen slotted dataclass `ExperimentRegistry` | `development_seeds: immutable sequence of int`; `pilot_seeds: immutable sequence of int`; `candidate_sample_sizes: immutable sequence of int`; `descriptive_externality_n: int`; `families: immutable sequence of ExperimentFamily`; `hypotheses: immutable sequence of PrimaryHypothesis` | Seed pools are pairwise disjoint; candidate sizes are exactly `50`, `100`, and `200`; `descriptive_externality_n` is exactly `30`; exactly four primary hypotheses share one multiplicity-family identifier. |
| `load_registry` | `(path: Path) -> ExperimentRegistry` | Parse TOML, resolve config paths relative to the repository root, enforce every registry invariant, and raise `ValueError` naming the violated field. |
| `registered_family` | `(registry: ExperimentRegistry, name: str) -> ExperimentFamily` | Return the unique family or raise `KeyError(name)`. |
| `validate_registered_definition` | `(registry: ExperimentRegistry, family_name: str, definition: ExperimentDefinition) -> None` | Compare config path-derived definition values, seed interval, slot count, methods, case definitions, and sample-size policy; raise `ValueError` with the first mismatched field. |

### `uav_mec/seed_access.py` and `uav_mec/trace.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| string enum `SeedPurpose` | `DEVELOPMENT="development"`; `PILOT="pilot"`; `FORMAL="formal"` | Serialized values are the lowercase strings shown. |
| frozen slotted dataclass `SeedPermit` | `purpose: SeedPurpose`; `allowed_seeds: immutable sequence of int`; `family: str or None`; `case_id: str or None`; `registry_sha256: str`; `lock_sha256: str or None` | A formal permit requires family, case ID, and lock digest; development and pilot permits require none of those three fields. |
| slotted dataclass `SeedAccessRecorder` | `ledger_path: Path`; `permit: SeedPermit` | The constructor does not create a record until an access is requested. |
| `SeedAccessRecorder.record` | `(self, operation: str, seeds: Sequence[int], metadata: Mapping[str, object]) -> None` | Reject any seed outside the permit before work starts; append one canonical-JSON line with timestamp, purpose, family, case ID, sorted unique seeds, operation, metadata, registry/lock digests, previous-record digest, and current digest; flush and `fsync` before returning. |
| `development_permit` | `(registry_path: Path) -> SeedPermit` | Load only the development pool and bind its registry digest. |
| `pilot_permit` | `(registry_path: Path) -> SeedPermit` | Load only the pilot pool and bind its registry digest. |
| `formal_permit` | `(lock_path: Path, family_name: str, case_id: str) -> SeedPermit` | Verify the confirmatory lock, resolve the exact locked case, then expose only that case’s seeds and both registry and lock digests. |
| `verify_seed_ledger` | `(path: Path) -> immutable sequence of JSON-object mappings` | Recompute the hash chain and canonical encoding; reject malformed lines, missing keys, duplicate seed entries within a record, or broken predecessor hashes with `ValueError` naming the line. |
| `audit_candidate_pool_unseen` | `(registry: ExperimentRegistry, ledger_records: Sequence[Mapping[str, object]]) -> None` | Raise `ValueError` if any formal candidate seed appears in a development or pilot record; formal records are allowed only when bound to a valid lock digest. |
| `audit_seed_access` | `(registry_path: Path, ledger_path: Path, phase: str, output_path: Path, keyword-only lock_path: Path or None = None) -> dict[str, object]` | Accept phase `before-final-pilot`, `before-formal-lock`, or `before-formal-run`; verify the ledger, summarize accesses by purpose/family/case/pool, enforce the phase boundary, and atomically write canonical JSON without recording a new seed access. For `before-formal-run`, require the lock, verify the exact copied prelock prefix byte length/head digest, then validate every post-prefix record against its family/case permit and lock digest before producing output. |
| `generate_trace` | `(cfg: ScenarioConfig, seed: int, csi_error_std: float, keyword-only seed_access: SeedAccessRecorder) -> ExperimentTrace` | Call `record("generate_trace", [seed], metadata)` before initializing any RNG; metadata contains the scenario fingerprint and CSI error. Reject missing recorder by Python signature and disallowed seeds through the recorder. |

### `uav_mec/statistics.py` and `uav_mec/power_analysis.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| frozen slotted dataclass `PairedComparison` | `mean_difference: float`; `ci_low: float`; `ci_high: float`; `t_statistic: float`; `p_t_raw: float`; `wilcoxon_w: float`; `p_wilcoxon_raw: float`; `rank_biserial: float` | All quantities use paired treatment-minus-control differences; p-values are two-sided. |
| `paired_comparison` | `(proposed: numpy.ndarray, baseline: numpy.ndarray, bootstrap_resamples: int = 10_000, bootstrap_seed: int = 20260719) -> PairedComparison` | Require equal one-dimensional finite arrays with at least two pairs and a positive resample count; use a paired t-test as primary, Wilcoxon signed-rank as sensitivity, deterministic paired bootstrap CI, and rank-biserial effect size; raise `ValueError` for shape, length, finiteness, or resample-count violations. Formal analysis uses exactly the shown defaults. |
| frozen slotted dataclass `PowerDecision` | `candidate_sizes: immutable sequence of int`; `selected_n: int or None`; `evaluated_n: int`; `power_table_by_n: mapping from int to mapping from hypothesis ID to float`; `simulations: int`; `analysis_seed: int`; `target_power: float`; `primary_test: str`; `alternative: str`; `multiplicity_family: immutable sequence of str`; `resampling_stream_ids: dict[str, int]`; `evidence_insufficient: bool` | Candidates and table keys are exactly `50`, `100`, `200` in ascending order; each row has exactly H1--H4 and `evaluated_n` is always `200`. `selected_n` is the first row where every value reaches target; `analysis_seed` is `20260719` for the final pilot; `evidence_insufficient` is true exactly when `selected_n` is null. `primary_test` is `paired_t`, `alternative` is `two-sided`, and the multiplicity family is the four registered IDs in order. Canonical JSON encodes integer candidate keys as decimal strings. |
| `select_sample_size` | `(pilot_differences: Mapping[str, Sequence[float]], mde_by_hypothesis: Mapping[str, float], keyword-only candidates: Sequence[int] = (50, 100, 200), simulations: int = 10_000, target_power: float = 0.8, alpha: float = 0.05, seed: int = 20260719) -> PowerDecision` | Validate identical four-key mappings and positive finite MDEs; simulate paired differences from centered empirical residuals plus the registered MDE, apply a two-sided paired t-test to each hypothesis, apply Holm across all four raw p-values per replicate, and select the first candidate for which every hypothesis reaches target power. Independent deterministic streams are derived from the analysis seed and hypothesis ID. |

### `uav_mec/resource_snapshots.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| string enum `RadioGenerator` | `DD_GENERATED="dd_generated"`; `RADIO_NEUTRAL="radio_neutral"`; `DEADLINE_FIRST_GENERATED="deadline_first_generated"` | The serialized generator value is part of the snapshot ID. |
| frozen slotted dataclass `RawRadioSnapshot` | `snapshot_id: str`; `seed: int`; `slot: int`; `generator: RadioGenerator`; `schedule: Schedule`; `planned_rates: immutable sequence of RateRecord`; `realized_rates: immutable sequence of RateRecord` | The ID is the SHA-256 of canonical raw schedule/rate content plus seed, slot, and generator; no allocation outcome is stored. |
| `build_radio_neutral_schedule` | `(snapshot: SlotSnapshot, channel: NOMAChannel, evaluation: EvaluationConfig) -> Schedule` | Construct a deterministic wireless-only IA schedule by descending planned-rate score and stable task ID, enforcing association, channel, SIC, and power feasibility while ignoring task CPU demand and every allocation policy. |
| `build_deadline_first_schedule` | `(snapshot: SlotSnapshot, channel: NOMAChannel, evaluation: EvaluationConfig) -> Schedule` | Construct a deterministic schedule by ascending absolute deadline, then descending planned rate, then stable task ID, enforcing the same wireless constraints and ignoring allocation policy. |
| `freeze_radio_snapshot` | `(seed: int, snapshot: SlotSnapshot, generator: RadioGenerator, schedule: Schedule, channel: NOMAChannel, evaluation: EvaluationConfig) -> RawRadioSnapshot` | Compute planned and realized rate records exactly once, deep-freeze all arrays/records, derive the content ID, and reject a schedule that is not wirelessly feasible for the slot. |
| `evaluate_frozen_resource_snapshot` | `(snapshot: SlotSnapshot, raw: RawRadioSnapshot, policy: AllocationPolicy, evaluation: EvaluationConfig) -> SlotOutcome` | Reuse the raw schedule and rates byte-for-byte, vary only CPU/resource allocation, and reject any policy path that attempts to regenerate or mutate the radio state. |

### `uav_mec/resource_allocation.py` and `uav_mec/exact.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| frozen slotted dataclass `ExactAdmissionResult` | `admitted_task_ids: immutable sequence of int`; `rejected_task_ids: immutable sequence of int`; `requirements: mapping from int to ResourceRequirement`; `total_cpu_hz: float`; `enumerated_subsets: int` | This is the fixed-radio CPU-subset oracle from Plan 2. It never receives or returns a radio schedule and is never serialized as Exact Schedule. |
| `exact_admission` | `(tasks: Sequence[Task], uplink_rates_bps: Mapping[int, float], cpu_budget_hz: float, downlink_rates_bps: Mapping[int, float] or None = None) -> ExactAdmissionResult` | Enumerate subsets of the same preclosure task set and frozen rates; maximize admitted cardinality, minimize total CPU within that cardinality, then break ties by sorted task IDs. |
| frozen slotted dataclass `ExactScheduleCertificate` | `complete: bool`; `raw_states_enumerated: int`; `unique_closed_states: int`; `best_bound: tuple of int, float, float`; `lexicographic_gap: tuple of int, float, float`; `cpu_nontrivial: bool`; `schedule_nontrivial: bool`; `semantic_fingerprint: str` | This is the Plan 2 certificate. `complete=true` is legal only after exhaustive raw-state enumeration; a complete result has `best_bound == objective` and zero three-layer gap. CPU and scheduling nontriviality are independent booleans. |
| frozen slotted dataclass `ExactScheduleResult` | `plan: FrozenSlotPlan`; `objective: tuple of int, float, float`; `certificate: ExactScheduleCertificate` | Serialize under schema `exact_schedule`; never emit, parse, or type-check it as `ExactAdmissionResult`. The legacy names `ExactResult` and `ExactNontriviality` must not exist. |

### `uav_mec/artifacts.py` and `uav_mec/run_shards.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| `RunArtifactWriter.__init__` | `(self, output_dir: Path, keyword-only bundle_version: int = 1) -> None` | Version 1 preserves lowercase historical closure names. Version 2 reserves uppercase `MANIFEST.json` and `SHA256SUMS`; any other version raises `ValueError` before directory creation. Every new module in this plan passes `bundle_version=2` explicitly. |
| `RunArtifactWriter.finalize` | `(self, manifest: Mapping[str, object]) -> None` | Write the version-appropriate manifest, compute sorted checksums over every regular file except the checksum file, write the checksum file last, flush directory metadata, and forbid further writes. |
| frozen slotted dataclass `ArtifactSpec` | `relative_path: str`; `kind: str`; `merge_strategy: str`; `primary_key: immutable sequence of str` | `kind` is `json`, `jsonl`, `csv`, `parquet`, `directory`, or `opaque`; strategy is `rows`, `require_equal`, or `by_shard`. `rows` is limited to tabular kinds and requires a primary key; the other strategies require an empty key. |
| frozen slotted dataclass `ShardSpec` | `name: str`; `seeds: immutable sequence of int` | Names are zero-padded `shard-NNNN`; seeds are sorted, unique, and form one contiguous slice of the locked case sequence. |
| frozen slotted dataclass `ShardPlan` | `schema_version: int`; `family: str`; `case_id: str`; `runner_kind: str`; `config_sha256: str`; `registry_sha256: str`; `lock_sha256: str or None`; `source_sha256: str`; `environment_sha256: str`; `expected_artifacts: immutable sequence of ArtifactSpec`; `shards: immutable sequence of ShardSpec` | Digests and the whole plan are immutable after creation; the family/case pair resolves uniquely; expected paths are unique and relative. |
| frozen slotted dataclass `ShardRunReport` | `completed_shards: int`; `reused_shards: int`; `merged_path: Path or None` | Counts cover every shard in the plan exactly once. |
| `create_shard_plan` | `(definition: ExperimentDefinition, config_path: Path, output_dir: Path, keyword-only family: str, case_id: str, runner_kind: str, shard_size: int, registry_sha256: str, lock_sha256: str or None, source_sha256: str, environment_sha256: str, expected_artifacts: Sequence[ArtifactSpec]) -> ShardPlan` | Validate positive shard size, plan all seeds of exactly one registered/locked case once in order, write canonical `shard_index.json`, and refuse to overwrite a non-identical plan. |
| `run_one_shard` | `(plan_path: Path, shard_name: str) -> Path` | Verify plan/config/source/environment/lock digests, dispatch the registered runner kind, require every artifact, write `SUCCESS.json` after payloads, finalize `MANIFEST.json` and `SHA256SUMS` last, then atomically rename the temporary shard directory. |
| `resume_missing_shards` | `(plan_path: Path, keyword-only workers: int) -> ShardRunReport` | Reuse only shards whose success manifest and all artifact hashes verify; quarantine an invalid shard under `corrupt/`, execute missing shards with bounded workers, and never merge implicitly. |
| `merge_shards` | `(plan_path: Path) -> Path` | Require all shards valid; for `rows`, concatenate and stable-sort by the declared primary key while rejecting duplicates; for `require_equal`, require byte/tree-hash equality and copy once; for `by_shard`, copy under stable shard-name subpaths; finalize the merged v2 closure with `SHA256SUMS` last, then atomically publish `merged/`. |

### `uav_mec/runtime_budget.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| frozen slotted dataclass `CaseRuntimeBudget` | `case_id: str`; `role: str`; `sample_size: int`; `shard_size: int`; `cpu_hours: float`; `wall_clock_hours: float`; `peak_memory_bytes: int`; `disk_bytes: int`; `event_log_bytes: int` | Every value is derived for one registered case; sample and shard sizes are positive; estimate fields are nonnegative. |
| frozen slotted dataclass `FamilyRuntimeBudget` | `family: str`; `concurrency: int`; `retry_fraction: float`; `cases: immutable sequence of CaseRuntimeBudget`; `cpu_hours: float`; `wall_clock_hours: float`; `peak_memory_bytes: int`; `disk_bytes: int`; `event_log_bytes: int` | Cases match registry order exactly, family totals equal case totals plus the declared retry reserve, concurrency is positive, and retry fraction is in closed interval `[0, 1]`. |
| frozen slotted dataclass `RuntimeBudget` | `schema_version: int`; `registry_sha256: str`; `power_decision_sha256: str`; `audit_sha256: str`; `execution_policy_sha256: str`; `primary_sample_size: int`; `power_guarantee: bool`; `families: immutable sequence of FamilyRuntimeBudget`; `total_cpu_hours: float`; `total_disk_bytes: int` | Contains every registry family exactly once in formal order; totals equal family sums; concurrency and shard size come only from the hashed execution-policy input. `primary_sample_size` is `selected_n` when non-null, otherwise `evaluated_n=200`; `power_guarantee` is false exactly in the fallback case. |
| `build_runtime_budget` | `(registry_path: Path, power_decision_path: Path, runtime_audit_path: Path, execution_policy_path: Path, output_path: Path, keyword-only retry_fraction: float) -> RuntimeBudget` | Verify all four inputs, project every formal case including retry reserve, fail on a missing/extra policy family or uncovered runtime-audit stratum, and atomically write canonical JSON. A null selection does not block budgeting: use audited `evaluated_n=200` for the four primary families and set `power_guarantee=false`. No preset wall-clock acceptance threshold is applied. |
| `load_and_verify_runtime_budget` | `(path: Path) -> RuntimeBudget` | Recompute input and content digests, family coverage, and totals; raise `ValueError` naming the failed invariant. |

### `uav_mec/run_development_calibration.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| string enum `DevelopmentCalibrationCase` | `EXACT_NONTRIVIALITY="exact_nontriviality"`; `RUNTIME_CAPACITY="runtime_capacity"` | No other calibration case is accepted. |
| `parse_development_seed_range` | `(value: str, registry: ExperimentRegistry) -> immutable sequence of int` | Parse an inclusive `START:END` pair, require `START <= END`, and reject any integer outside the registry development pool. |
| `run_development_calibration` | `(registry_path: Path, case: DevelopmentCalibrationCase, seeds: Sequence[int], output_dir: Path, keyword-only workers: int, trace_lock_path: Path or None, seed_access: SeedAccessRecorder) -> Path` | Require a development permit and positive workers; record access before dispatch. Exact produces independent CPU/schedule nontriviality counts and fractions. Runtime produces one measured audit row per registered family/case; Trace-driven uses only a verified GeoLife `calibration` split and therefore requires a trace lock. Atomically return the checksum-valid bundle path. |

### `uav_mec/geolife_lock.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| frozen slotted dataclass `LockedTraceSegment` | `scenario_id: str`; `participant_id: str`; `split: str`; `mode: str`; `source_path: str`; `source_start: str`; `source_end: str`; `slot_seconds: float`; `translation_m: pair of float`; `rotation_deg: float`; `group_id: str` | `split` is exactly `train`, `calibration`, or `final`; participant IDs never cross splits; group IDs identify participant-level bootstrap clusters. |
| frozen slotted dataclass `TraceLock` | `schema_version: int`; `dataset_sha256: str`; `dataset_version: str`; `source_url: str`; `retrieved_on: str`; `license_text: str`; `license_sha256: str`; `protocol_sha256: str`; `registry_sha256: str`; `seeds: immutable sequence of int`; `segments: immutable sequence of LockedTraceSegment` | Dataset digest binds the original archive bytes; source/retrieval/license/protocol/registry metadata is explicit; formal seeds match the `trace_driven` family; segments are ordered by split, participant, and source time. |
| `build_trace_lock` | `(archive_path: Path, protocol_path: Path, registry_path: Path, output_path: Path) -> TraceLock` | Validate archive digest and protocol, derive deterministic participant-disjoint splits, bind transform parameters and timestamps, record dataset version/retrieval/license metadata and registry/protocol hashes, and refuse overwrite of non-identical output. |
| `load_and_verify_trace_lock` | `(path: Path) -> TraceLock` | Recompute content digest, verify participant disjointness, monotonic segment times, allowed modes, formal seed coverage, and required license/source metadata. |

### `uav_mec/provenance.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| frozen slotted dataclass `SourceSnapshot` | `git_sha: str`; `clean: bool`; `tree_sha256: str`; `patch_sha256: str or None`; `archive_path: Path`; `patch_path: Path or None` | A clean tree has no patch fields; a dirty tree includes a binary patch and binds both archive and patch bytes. |
| frozen slotted dataclass `EnvironmentSnapshot` | `python_version: str`; `platform: str`; `architecture: str`; `dependency_versions: dict[str, str]`; `sha256: str` | Dependencies are the imported versions required by the package and are sorted by distribution name before hashing. |
| `capture_source_snapshot` | `(repo_root: Path, output_dir: Path) -> SourceSnapshot` | Create a deterministic committed-tree archive, optionally capture the complete working-tree patch, hash both, write `source_snapshot.json` as the final payload, then finalize the v2 closure; refuse non-identical overwrite. |
| `capture_environment_snapshot` | `(output_path: Path) -> EnvironmentSnapshot` | Capture the fixed interpreter, OS, architecture, and dependency versions; atomically write canonical JSON with its self-digest. |
| `build_readiness_report` | `(registry_path: Path, seed_audit_path: Path, test_report_path: Path, correctness_dir: Path, output_json_path: Path, output_markdown_path: Path) -> dict[str, object]` | Verify the audit’s registry digest and phase, JUnit pass/failure counts, source hash, and correctness bundle closure; derive `ready_for_pilot` and `ready_for_formal` only from explicit machine-readable prerequisites, write byte-stable JSON and Markdown atomically as one transaction, and never execute a test or experiment. Before-final-pilot reports always leave formal readiness false. |

### `uav_mec/confirmatory_lock.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| frozen slotted dataclass `LockedCase` | `case_id: str`; `role: str`; `seeds: immutable sequence of int` | The case resolves in its family; seeds are sorted, unique, inside that family’s candidate pool, and have the exact selected or fixed count. |
| frozen slotted dataclass `LockedFamily` | `family: str`; `config_path: str`; `config_sha256: str`; `num_slots: int`; `methods: immutable sequence of str`; `cases: immutable sequence of LockedCase` | Values equal the validated registry/config definition and preserve formal family/case order. Externality contains one power-selected primary case and eight descriptive cases of exactly 30 seeds each. |
| frozen slotted dataclass `ConfirmatoryLock` | `schema_version: int`; `source_git_sha: str`; `source_snapshot_sha256: str`; `registry_sha256: str`; `analysis_code_sha256: str`; `power_decision_sha256: str`; `pilot_bundle_sha256: str`; `power_table_by_n: mapping`; `runtime_budget_sha256: str`; `trace_lock_sha256: str`; `seed_ledger_relpath: str`; `seed_ledger_prefix_bytes: int`; `seed_ledger_prefix_sha256: str`; `seed_ledger_head_sha256: str`; `correctness_bundle_sha256: str`; `primary_test: str`; `alternative: str`; `sensitivity_test: str`; `multiplicity_family: immutable sequence of str`; `mde_by_hypothesis: mapping`; `estimand: str`; `bootstrap_resamples: int`; `bootstrap_seed: int`; `families: immutable sequence of LockedFamily`; `allowed_command_templates: immutable sequence of str`; `environment_sha256: str`; `power_guarantee: bool` | The tests are paired t (`primary_test=paired_t`, `alternative=two_sided`) and Wilcoxon sensitivity, multiplicity contains the four ordered IDs, bootstrap count is `10000`, bootstrap seed is `20260719`, every family/case is locked, the frozen prelock ledger prefix is copied into the lock, correctness evidence is checksum-closed, and command templates name only guarded entry points. |
| `build_confirmatory_lock` | `(registry_path: Path, power_decision_path: Path, runtime_budget_path: Path, trace_lock_path: Path, seed_ledger_path: Path, source_snapshot_dir: Path, environment_path: Path, correctness_dir: Path, output_dir: Path) -> ConfirmatoryLock` | Verify all dependency digests and unseen candidate pools. Copy the exact prelock ledger prefix and checksum-closed correctness bundle into the lock. If `selected_n` exists, require matching budget N and true power guarantee; otherwise require `evaluated_n=200`, `evidence_insufficient=true`, matching budget N 200, and the fixed approved-protocol fallback authorization recorded by source-of-truth commit `adda1539d480044df728366a878dcdc0e069e5c4`; set false guarantee. Hash analyzer source, write all lock assets under a temporary directory, and atomically publish. |
| `load_and_verify_confirmatory_lock` | `(path: Path) -> ConfirmatoryLock` | Verify the lock self-digest and every referenced immutable asset; raise before returning if any byte differs. |
| `locked_case_definition` | `(lock: ConfirmatoryLock, family_name: str, case_id: str) -> ExperimentDefinition` | Load the locked config, verify its digest and case fields against the exact locked family/case, then return a definition restricted to that case’s seeds; unknown family or case raises `KeyError`. |

### `uav_mec/claim_gates.py`, `uav_mec/analyze_confirmatory.py`, and `uav_mec/canonical_v2.py`

| Declaration | Fields or exact signature | Required behavior and errors |
|---|---|---|
| string enum `ClaimStatus` | `MEANINGFUL_SUPPORTED="meaningful_supported"`; `POSITIVE_MDE_OBSERVED_BUT_UNCERTAIN="positive_mde_observed_but_uncertain"`; `POSITIVE_BUT_SMALL="positive_but_small"`; `INSUFFICIENT="insufficient"`; `NEGATIVE="negative"` | Serialized values are the lowercase strings shown. |
| `classify_primary_claim` | `(keyword-only estimate_pp: float, ci_low_pp: float, ci_high_pp: float, p_holm: float, mde_pp: float, alpha: float = 0.05) -> ClaimStatus` | Apply the ordered gate in Task 9; reject non-finite inputs, invalid CI order, p-values outside `[0, 1]`, or nonpositive MDE. |
| `analyze_confirmatory` | `(lock_path: Path, runs_root: Path, output_dir: Path, keyword-only seed_access_by_case: Mapping[tuple[str, str], SeedAccessRecorder]) -> dict[str, object]` | Resolve the registry, every locked family/case root, and correctness evidence through the verified lock and formal root; record every case’s accessed seeds through its exact permit; compute registry-driven paired inference and four-way Holm; keep correctness/supporting/descriptive evidence outside multiplicity; atomically publish `statistics.json`, `metrics.json`, `acceptance.json`, the nineteen-column `statistical_claims.csv`, and the v2 closure. Never emit `claim_matrix.csv`. |
| `verify_confirmatory_analysis` | `(lock_path: Path, runs_root: Path, analysis_root: Path) -> None` | Read-only verification of analysis manifest/checksums, lock/run lineage, case coverage, four-member Holm inputs, the exact nineteen-field `statistical_claims.csv`, derived acceptance rows, absence of `claim_matrix.csv`, and absence of undeclared files; do not append seed access because result values are not recomputed. |
| pure callable contract `PublicationAssetGenerator` | `(analysis_root: Path, staging_root: Path) -> Mapping[str, str]` | Implemented by Plan 5 as `generate_bilingual_publication_assets`; read verified `statistics.json`, `metrics.json`, `acceptance.json`, and nineteen-column `statistical_claims.csv`; write only regular files beneath an initially absent staging root, including exactly one root twelve-column `claim_matrix.csv`; return every generated POSIX-relative path mapped to its SHA-256; produce byte-identical outputs for identical analysis bytes. |
| `publish_canonical_v2` | `(lock_path: Path, runs_root: Path, analysis_root: Path, output_dir: Path, keyword-only asset_generator: PublicationAssetGenerator = generate_bilingual_publication_assets) -> None` | Verify lock, formal runs, and analysis; reject an analyzer-produced `claim_matrix.csv`; invoke the generator inside the publisher’s temporary tree; verify its returned path/hash map and exact twelve-column matrix; include `statistical_claims.csv`, the sole generated `claim_matrix.csv`, generated tables, bilingual figures, and LaTeX staging before computing full/light manifests and root `SHA256SUMS`; atomically publish only after every matrix artifact path resolves and closure verification passes. |
| `verify_canonical_v2` | `(root: Path) -> None` | Verify full and light manifests, all file hashes, the light allowlist relation, registry/lock digest agreement, and the absence of undeclared files. |

## Command-line contracts shared with formal orchestration

These are the only accepted v2 command surfaces. Every parser rejects unknown options; normal `--help` is the only option that may omit required arguments. Paths are resolved before mutation, output paths must be beneath their documented artifact root, and a non-resumable command refuses an existing destination.

| Module and subcommand | Exact required and optional arguments | Contract |
|---|---|---|
| `uav_mec.seed_access audit` | required `--registry PATH --ledger PATH --phase PHASE --output PATH`; `--lock DIR` required only for `before-formal-run` | Phase is exactly `before-final-pilot`, `before-formal-lock`, or `before-formal-run`; this read-only audit writes no ledger record, verifies the lock's prelock prefix and every post-prefix record when a lock is required, and rejects a boundary violation before output creation. |
| `uav_mec.run_power_pilot_suite` | required `--registry PATH --output DIR`; optional `--workers INTEGER --shard-size INTEGER --simulations INTEGER --analysis-seed INTEGER --resume` | Defaults are workers `2`, shard size `1`, simulations `10000`, and analysis seed `20260719`; numeric values are validated positive before output creation and only the registered pilot pool is allowed. The suite runs and merges all four registered pilot cases, then atomically writes its sole decision at `OUTPUT/decision/power_decision.json`; there is no second `run_power_pilot` operator step. |
| `uav_mec.runtime_budget build` | required `--registry PATH --power-decision PATH --runtime-audit PATH --execution-policy PATH --retry-fraction FLOAT --output PATH` | The execution-policy JSON has schema version `1` and exactly eight records with keys `family`, `concurrency`, and `shard_size`; IDs are the exact family IDs below and both integers are positive. There are no command-line worker or shard overrides after lock construction. |
| `uav_mec.runtime_budget verify` | required `--budget PATH` | Calls `load_and_verify_runtime_budget` and exits nonzero on drift. |
| `uav_mec.run_development_calibration` | required `--registry PATH --case ID --seeds START:END --output DIR`; optional `--workers INTEGER --trace-lock PATH` | Case ID is exactly `exact_nontriviality` or `runtime_capacity`; the inclusive range must remain inside development seeds; workers default to one and must be positive. `runtime_capacity` requires `--trace-lock` and uses only calibration participants; `exact_nontriviality` rejects that option. The CLI creates its own development recorder under the new output bundle. |
| `uav_mec.geolife_lock build` | required `--archive PATH --protocol PATH --registry PATH --output PATH` | Builds the trace lock without running a policy or opening a formal seed. |
| `uav_mec.geolife_lock verify` | required `--lock PATH --archive PATH` | Verifies internal closure and recomputes the dataset digest from the supplied regular, non-symlink archive. |
| `uav_mec.provenance capture` | required `--repo-root PATH --source-output DIR --environment-output PATH` | Captures source and environment without executing an experiment. |
| `uav_mec.provenance verify` | required `--source-snapshot DIR --environment PATH` | Recomputes both snapshot closures. |
| `uav_mec.provenance readiness` | required `--registry PATH --seed-audit PATH --test-report FILE --correctness DIR --output-json PATH --output-markdown PATH` | Verifies the pre-open audit, fresh JUnit report and correctness bundle, then writes deterministic JSON/Markdown readiness views without running tests or opening seeds. |
| `uav_mec.correctness_bundle build` | required `--test-report FILE --repo-root DIR --registry PATH --output DIR` | Read-only validation of the fixed test IDs, source/config hashes and static algorithm/certificate records; writes one checksum-closed correctness bundle without opening a pilot or formal seed. |
| `uav_mec.confirmatory_lock build` | required `--registry PATH --power-decision PATH --runtime-budget PATH --trace-lock PATH --seed-ledger PATH --source-snapshot DIR --environment PATH --correctness DIR --output DIR` | Atomically creates one immutable lock directory containing the prelock ledger prefix and correctness bundle. |
| `uav_mec.confirmatory_lock verify` | required `--lock DIR` | Recomputes the lock and every embedded/referenced digest. |
| `uav_mec.run_confirmatory` family mode | required `--lock DIR --family ID --root FORMAL_ROOT`; at most one of `--resume` and `--verify-only` | Runs or verifies one registered family. Shard size and worker concurrency are read from the lock-bound runtime budget. The parser rejects `--workers`, `--shard-size`, `--output`, `--all`, and every seed/slot/method/scenario override. |
| `uav_mec.run_confirmatory` aggregate verification mode | required `--lock DIR --root FORMAL_ROOT --verify-all` | Verifies all eight families in registered order without running a missing shard; rejects `--family`, `--resume`, and `--verify-only`. |
| `uav_mec.analyze_confirmatory build` | required `--lock DIR --runs-root FORMAL_ROOT --output DIR` | Resolves registry and correctness inputs from the lock/formal closure and constructs analysis; no registry, correctness-root, bootstrap, or seed override is accepted. |
| `uav_mec.analyze_confirmatory verify` | required `--lock DIR --runs-root FORMAL_ROOT --analysis DIR` | Performs read-only closure and derivation verification; replaces any undefined separate claim-gate verification CLI. |
| `uav_mec.canonical_v2 publish` | required `--lock DIR --runs-root FORMAL_ROOT --analysis DIR --output DIR` | Calls the installed pure bilingual asset generator internally; rejects registry, rendered-assets, and family-root overrides. |
| `uav_mec.canonical_v2 verify` | required `--root DIR` | Verifies root closure without generating or changing an asset. |

The only family IDs accepted anywhere are `resource`, `exact`, `externality`, `main`, `joint`, `scaling`, `robustness`, and `trace_driven`.

## Task 1: Freeze the v2 registry and enforce seed-access recording

**Files:**

- Create: `uav_mec/seed_access.py`
- Create: `tests/conftest.py`
- Create: `tests/test_seed_access.py`
- Create: `configs/thesis/confirmatory_trace_v2.toml`
- Modify: `uav_mec/experiment_registry.py:13-141`
- Modify: `uav_mec/trace.py:59-79`
- Modify: `configs/thesis/experiment_registry_v2.toml:1-78`
- Modify: `configs/thesis/confirmatory_resource_fixed_v2.toml:1-47`
- Modify: `configs/thesis/externality_stress.toml:1-41`
- Modify: `configs/thesis/confirmatory_main_v2.toml:1-53`
- Modify: `configs/thesis/confirmatory_exact.toml:1-26`
- Modify: `configs/thesis/confirmatory_joint_v2.toml:1-45`
- Modify: `uav_mec/experiment.py:677-772`
- Modify: `uav_mec/run_exact_benchmark.py:49-54`
- Modify: `uav_mec/externality_analysis.py:190-205`
- Modify: every trace-generating test listed in the file responsibility map
- Test: `tests/test_experiment_registry.py`
- Test: `tests/test_seed_access.py`
- Test: `tests/test_thesis_suites.py`

- [ ] **Step 1: Define the development recorder fixture before any test uses it**

Create `tests/conftest.py` with this complete fixture contract:

```python
from pathlib import Path

import pytest

from uav_mec.seed_access import SeedAccessRecorder, development_permit


@pytest.fixture
def development_seed_access(tmp_path: Path) -> SeedAccessRecorder:
    return SeedAccessRecorder(
        ledger_path=tmp_path / "seed-access.jsonl",
        permit=development_permit(Path("configs/thesis/experiment_registry_v2.toml")),
    )
```

Update every direct `generate_trace()` test call to accept this fixture and pass `seed_access=development_seed_access`. Replace actual generation seeds `1000`, `1001`, `2000`, `2001`, `3000`, `4000`, and formal-config first seeds with deterministic development values `100`, `101`, `200`, `201`, `300`, `400`, and `42` respectively. Do not change assertions that merely inspect configured formal integers.

- [ ] **Step 2: Write registry and seed-policy failing tests**

Add these exact test names:

```text
test_v2_registry_freezes_approved_candidate_pools_and_all_eight_families
test_v2_registry_freezes_externality_descriptive_n_at_thirty
test_v2_registry_freezes_every_case_id_role_and_sample_policy
test_registry_declares_disjoint_development_pilot_and_formal_pools
test_registry_links_each_family_to_one_matching_config
test_registry_primary_hypotheses_freeze_test_estimand_and_holm_family
test_registry_rejects_config_seed_method_or_slot_drift
test_generate_trace_requires_verified_access_outside_development_pool
test_seed_ledger_records_trace_generation_with_hash_chain
test_seed_ledger_rejects_tampering
test_candidate_pool_audit_rejects_a_prelock_formal_access
test_seed_access_audit_cli_writes_checksum_bound_pre_pilot_report
test_seed_access_audit_rejects_unknown_phase_or_boundary_violation_before_output
test_tests_generate_only_development_seeds
```

The registry test must assert the exact protocol and case tables above, family order `resource, exact, externality, main, joint, scaling, robustness, trace_driven`, `descriptive_externality_n=30`, H1--H4 case/treatment/control/MDE membership, two-sided paired t as primary, two-sided Wilcoxon as sensitivity, and `primary_h1_h4` as the sole primary multiplicity family. The H2 semantic test asserts `K=3`, `N=45`, `S=3`, pairwise overlap, 5-m cluster spread, period choice 2, and coupling 1.0. The seed test creates its ledger under `tmp_path`; it never writes the repository ledger.

- [ ] **Step 3: Run the RED tests**

Run:

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_experiment_registry.py \
  tests/test_seed_access.py \
  tests/test_thesis_suites.py -q
```

Expected: FAIL because the registry still contains the 12000-series pools and seven families, `generate_trace` has no recorder argument, and `uav_mec.seed_access` does not exist.

- [ ] **Step 4: Implement the registry and ledger contracts**

Implement the public contracts above. The ledger must:

1. validate every seed against the permit before writing;
2. append one canonical JSON object per operation under an exclusive file lock;
3. store `sequence`, UTC timestamp, operation, purpose, family, case ID, seeds, registry hash, lock hash, metadata, previous record hash, and record hash;
4. flush and `fsync` before releasing the lock;
5. verify sequence and the complete previous-hash chain when read;
6. reject an empty operation, duplicate seed in one record, non-integer seed, or metadata containing non-finite JSON values.

Change `generate_trace()` to require the keyword-only recorder and record before constructing any RNG. Pass the recorder through `run_definition`, `run_exact_definition`, and externality analysis. A config load or integer assertion remains read-only and does not record.

Implement `seed_access audit` exactly as declared in the shared CLI table. `before-final-pilot` requires every algorithm/trace/analysis access to lie in development seeds and proves no pilot or formal candidate seed has been opened. `before-formal-lock` permits complete pilot access but no formal candidate access. `before-formal-run` requires formal accesses, if any, to carry a verified lock plus exact family/case permit. All phases include registry/ledger/head digests and pool counts; parser or boundary failure leaves no output.

Update the TOML files to the approved pools. `experiment_registry_v2.toml` must set `descriptive_externality_n=30`. `externality_stress.toml` must freeze the primary H2 base at `cluster_std=5.0`, `period_choices=[2]`, `cross_interference_scale=1.0`. Add the `trace_driven` family and config. Do not edit legacy v1 configs; registry validation must prove they are unreachable from canonical v2.

- [ ] **Step 5: Run the GREEN tests and the affected trace tests**

Run:

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_experiment_registry.py \
  tests/test_seed_access.py \
  tests/test_trace.py \
  tests/test_simulation.py \
  tests/test_csi_error.py \
  tests/test_experiment_cli.py \
  tests/test_ia_greedy.py \
  tests/test_resource_policies.py \
  tests/test_thesis_suites.py \
  tests/test_externality_analysis.py \
  tests/test_capacity_analysis.py -q
```

Expected: PASS, and every generated trace in these tests uses a seed no greater than 999.

- [ ] **Step 6: Commit with an explicit allowlist**

```bash
git add \
  configs/thesis/experiment_registry_v2.toml \
  configs/thesis/confirmatory_resource_fixed_v2.toml \
  configs/thesis/externality_stress.toml \
  configs/thesis/confirmatory_main_v2.toml \
  configs/thesis/confirmatory_exact.toml \
  configs/thesis/confirmatory_joint_v2.toml \
  configs/thesis/confirmatory_trace_v2.toml \
  uav_mec/experiment_registry.py \
  uav_mec/seed_access.py \
  uav_mec/trace.py \
  uav_mec/experiment.py \
  uav_mec/run_exact_benchmark.py \
  uav_mec/externality_analysis.py \
  tests/conftest.py \
  tests/test_experiment_registry.py \
  tests/test_seed_access.py \
  tests/test_trace.py \
  tests/test_simulation.py \
  tests/test_csi_error.py \
  tests/test_experiment_cli.py \
  tests/test_ia_greedy.py \
  tests/test_resource_policies.py \
  tests/test_thesis_suites.py \
  tests/test_externality_analysis.py \
  tests/test_capacity_analysis.py
git diff --cached --name-only
git commit -m "feat: freeze v2 registry and seed access policy"
```

## Task 2: Align paired inference and four-hypothesis Holm power

**Files:**

- Modify: `uav_mec/statistics.py:12-137`
- Modify: `uav_mec/power_analysis.py:13-132`
- Modify: `uav_mec/run_power_pilot.py:17-146`
- Modify: `uav_mec/run_power_pilot_suite.py:13-53`
- Modify: `tests/test_statistics.py`
- Modify: `tests/test_power_analysis.py`
- Modify: `tests/test_power_pilot.py`

- [ ] **Step 1: Write the failing inference tests**

Add these exact tests:

```text
test_paired_comparison_reports_paired_t_primary_and_wilcoxon_sensitivity
test_paired_comparison_all_zero_differences_returns_unit_primary_and_sensitivity_p
test_power_simulation_uses_two_sided_paired_t_with_four_member_holm
test_power_simulation_records_distinct_reproducible_resampling_stream_ids
test_power_simulation_selects_the_smallest_common_candidate
test_power_simulation_reports_no_selection_when_one_hypothesis_misses_target
test_power_decision_bundle_records_test_estimand_mde_and_input_hashes
test_power_decision_records_candidates_analysis_seed_full_power_table_and_insufficient_flag
test_power_pilot_rejects_semantic_drift_between_pilot_and_formal_primary_configs
test_power_suite_cli_freezes_candidates_simulations_analysis_seed_and_pilot_pool
test_power_suite_cli_rejects_nonpositive_workers_shards_or_simulations_before_output
```

Use synthetic difference vectors in percentage points. The test that distinguishes the t test from Wilcoxon must use a finite vector for which SciPy returns different p-values, compute the expected two-sided t p-value with `scipy.stats.ttest_1samp(differences, 0.0)`, and compare it to `p_t_raw`. Do not inspect pilot or formal artifacts.

- [ ] **Step 2: Run the RED tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_statistics.py \
  tests/test_power_analysis.py \
  tests/test_power_pilot.py -q
```

Expected: FAIL because `PairedComparison` has no t-test fields, power uses one-sided Wilcoxon, and power output has no resampling stream or semantic-hash metadata.

- [ ] **Step 3: Implement the registered primary and sensitivity tests**

Keep `paired_bootstrap_ci`, `rank_biserial`, and `holm_adjust`. Change `paired_comparison` so:

- differences are `proposed - baseline`;
- the 95 percent interval is the existing deterministic 10,000-resample paired mean bootstrap;
- primary `p_t_raw` is a two-sided paired-t p-value;
- sensitivity `p_wilcoxon_raw` is two-sided signed-rank;
- all-zero differences return zero statistics where defined, unit p-values, and zero effect size;
- constant nonzero differences return a deterministic infinite signed t statistic and zero primary p-value without serializing NaN.

In `select_sample_size`, center each pilot vector, spawn one deterministic RNG stream per sorted hypothesis ID from a master `SeedSequence`, resample each family independently, shift by its registered MDE, compute two-sided one-sample t p-values, apply one four-member Holm decision, and count rejection per hypothesis. Store candidate sizes, analysis seed, child stream identifiers, and the complete three-candidate-by-four-hypothesis power table in the decision. Evaluate all three candidates for auditability, choose the first common passing row, and always set `evaluated_n=200`. When none passes, set `selected_n=null` and `evidence_insufficient=true`; otherwise set the flag false.

In `run_power_pilot.py`, hash and record the registry, all four pilot bundle checksums, the four primary config files, `statistics.py`, `power_analysis.py`, and the analysis seed. Record `outcome=seed_episode_on_time_completion_rate`, `estimand=mean_paired_difference_pp`, `primary_test=paired_t`, `alternative=two_sided`, `sensitivity_test=two_sided_wilcoxon`, and `multiplicity_family=[H1,H2,H3,H4]`.

Implement `run_power_pilot_suite` with the exact shared CLI contract. Freeze candidates to `50 100 200`; the parser provides no candidate, MDE, test, alternative, multiplicity, formal seed, or formal config override. Defaults are two workers, shard size one, 10,000 simulations, and analysis seed 20260719. Validate all numeric options and the absent/new output-root rule before creating a pilot permit or output parent; `--resume` first verifies the immutable plan and every completed shard. After all four primary case merges verify, call the `run_power_pilot.py` library logic internally and atomically publish `output/decision/power_decision.json` plus its manifest. The module `uav_mec.run_power_pilot` is not a separate operator CLI, so a pilot cannot be merged under one source hash and powered later under another.

- [ ] **Step 4: Run GREEN tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_statistics.py \
  tests/test_power_analysis.py \
  tests/test_power_pilot.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit with an explicit allowlist**

```bash
git add \
  uav_mec/statistics.py \
  uav_mec/power_analysis.py \
  uav_mec/run_power_pilot.py \
  uav_mec/run_power_pilot_suite.py \
  tests/test_statistics.py \
  tests/test_power_analysis.py \
  tests/test_power_pilot.py
git diff --cached --name-only
git commit -m "feat: align paired inference and Holm power"
```

## Task 3: Materialize three-source Resource raw snapshots

**Depends on:** the core schedule work must already expose the approved raw assignment and closed executable schedule semantics. Do not create a second closure implementation in this task.

**Files:**

- Create: `uav_mec/resource_snapshots.py`
- Create: `uav_mec/run_resource_experiment.py`
- Create: `tests/test_resource_snapshots.py`
- Modify: `uav_mec/evaluation.py:245-507,886-970`
- Modify: `uav_mec/experiment.py:52-76,301-647,677-894`
- Modify: `configs/thesis/confirmatory_resource_fixed_v2.toml`
- Modify: `tests/test_resource_policies.py`
- Modify: `tests/test_thesis_suites.py`

- [ ] **Step 1: Write the failing snapshot tests**

Add these exact tests:

```text
test_resource_policies_share_one_byte_identical_raw_assignment_and_rate_snapshot
test_rejected_resource_task_remains_in_raw_snapshot_without_rate_recomputation
test_exact_admission_receives_the_preclosure_candidate_set
test_resource_snapshot_id_changes_when_assignment_or_rate_bytes_change
test_resource_runner_emits_dd_neutral_and_deadline_first_generators
test_resource_runner_searches_each_generator_once_per_seed_slot
test_resource_runner_emits_fixed_half_and_crossing_alpha_min_for_each_generator
test_resource_summary_reports_policy_effect_by_generator
test_resource_manifest_hashes_raw_snapshot_artifact
```

Build the first three tests from a manually constructed `SlotSnapshot`, `Schedule`, and deterministic `RateRecord` tuple. Monkeypatch `compute_rate_records` with a counter in the runner test and assert exactly one planned-rate computation and one realized-rate computation per generator, independent of the number of resource policies. The preclosure test must schedule one task that Exact rejects for CPU and assert that its task ID remains present in the raw rate tuple supplied to all policies.

- [ ] **Step 2: Run the RED tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_resource_snapshots.py \
  tests/test_resource_policies.py \
  tests/test_thesis_suites.py -q
```

Expected: FAIL because the snapshot module and runner do not exist, fixed-radio methods repeat search/evaluation, and there is no raw snapshot artifact.

- [ ] **Step 3: Implement snapshot serialization and fixed-rate evaluation**

`snapshot_id` must be the SHA-256 of canonical JSON containing seed, slot, generator, sorted schedule entries, and sorted planned and realized rate records. Serialization must reject non-finite values and must not include policy output.

`evaluate_frozen_resource_snapshot` must:

1. select the planned or realized frozen rate tuple according to the explicit evaluation phase;
2. group tasks by the serving UAV in the raw assignment;
3. call `allocate_with_policy` on the unchanged task/rate set for each UAV;
4. preserve rejected tasks as misses without deleting them from the snapshot or recomputing any wireless value;
5. return a standard `SlotOutcome` carrying the frozen rates and the selected policy's CPU decisions.

The Resource runner must generate these sources once per seed-slot:

- `dd_generated`: full radio search with DD response, used for H3 primary;
- `radio_neutral`: the registered CPU-neutral wireless generator, whose score cannot inspect a resource-policy admission set;
- `deadline_first_generated`: the same full wireless search with Deadline-First response.

Apply the complete six-policy resource comparison to every source: Fixed-Half, Crossing/AlphaMin, Deadline-First-MinCPU, MinSlack-MinCPU, DD-MCAA, and Exact Admission. The registered compatibility name `EDF-MinCPU` may appear only as a display alias for Deadline-First-MinCPU. Write `raw_radio_snapshots.jsonl` once and include `snapshot_id` and `generator` in task/slot result rows. The summary must preserve H3 primary separately from the two sensitivity sources. The three source strata and six response policies are resolved through the Plan 2 `frozen_resource_method_specs()` table; no policy branch may trigger another radio search.

- [ ] **Step 4: Run GREEN tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_resource_snapshots.py \
  tests/test_resource_policies.py \
  tests/test_thesis_suites.py \
  tests/test_resource_allocator.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit with an explicit allowlist**

```bash
git add \
  configs/thesis/confirmatory_resource_fixed_v2.toml \
  uav_mec/resource_snapshots.py \
  uav_mec/run_resource_experiment.py \
  uav_mec/evaluation.py \
  uav_mec/experiment.py \
  tests/test_resource_snapshots.py \
  tests/test_resource_policies.py \
  tests/test_thesis_suites.py
git diff --cached --name-only
git commit -m "feat: materialize fixed-radio resource snapshots"
```

## Task 4: Separate Exact CPU and scheduling nontriviality

**Depends on:** the Exact oracle and heuristic must already use the same approved closed-schedule feasible region and quantized lexicographic objective.

**Files:**

- Modify: `uav_mec/exact.py:19-103`
- Modify: `uav_mec/run_exact_benchmark.py:27-254`
- Modify: `configs/thesis/confirmatory_exact.toml`
- Modify: `tests/test_exact.py`
- Modify: `tests/test_exact_benchmark.py`

- [ ] **Step 1: Write failing independent-nontriviality tests**

Add these exact tests:

```text
test_exact_classifies_cpu_and_schedule_nontriviality_independently
test_cpu_only_pressure_is_not_schedule_nontrivial
test_radio_capacity_pressure_is_schedule_nontrivial_without_cpu_pressure
test_multiple_wireless_states_with_distinct_secondary_objectives_are_schedule_nontrivial
test_exact_benchmark_writes_two_nontriviality_flags_and_reasons
test_exact_summary_reports_cpu_and_schedule_nontrivial_fractions
test_exact_claim_gate_requires_thirty_percent_for_each_claim_type
test_exact_rows_keep_all_three_conditional_gap_layers
test_exact_schedule_uses_plan2_result_and_certificate_types
test_exact_schedule_never_serializes_as_exact_admission
test_legacy_exact_result_and_nontriviality_types_are_absent
```

Use deterministic development-seed fixtures or directly constructed snapshots. The CPU-only case must have at least two legal wireless states, all tasks individually radio feasible, one UAV whose summed minimum CPU exceeds its budget, and no radio capacity/SIC reason preventing full wireless service. The capacity case must have more offload candidates than legal radio positions while CPU is ample. The secondary-objective case must have at least two full-service feasible states with different CPU or aggregate victim-link interference objective layers.

- [ ] **Step 2: Run the RED tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_exact.py \
  tests/test_exact_benchmark.py -q
```

Expected: FAIL because the benchmark does not yet expose the two Plan 2 certificate booleans independently, the CSV has one conflated `nontrivial` field, or a legacy generic Exact result name remains.

- [ ] **Step 3: Collect nontriviality evidence during exhaustive enumeration**

For every structurally legal assignment:

1. normalize NOMA roles and evaluate wireless SIC/rates without allowing CPU rejection to erase the raw state;
2. count the state as wireless feasible only when all scheduled layers pass planned SIC;
3. record whether a full-service wireless state exists;
4. calculate each scheduled task's minimum CPU from the frozen rates and detect per-UAV budget overload;
5. collect distinct quantized secondary objective pairs for full-service states;
6. retain complete enumeration as the only way to set `ExactScheduleCertificate.complete=true`.

Classify CPU nontrivial when a wireless-feasible task group is individually feasible yet its summed minimum CPU exceeds at least one serving-UAV budget. Classify scheduling nontrivial when at least two wireless-feasible states exist and either no full-service state exists because of capacity/SIC, or multiple full-service states have distinct secondary objective values whose ordering affects the lexicographic optimum. A CPU-only rejection does not satisfy scheduling nontriviality.

Do not introduce another Exact result or nested nontriviality type. Return the Plan 2 `ExactScheduleResult` and populate its `ExactScheduleCertificate`; write these exact benchmark CSV columns as derived audit evidence:

```text
cpu_nontrivial
schedule_nontrivial
cpu_overloaded_uav_ids
unique_closed_states
full_service_closed_state_exists
nontriviality_reason
certificate_complete
semantic_fingerprint
completion_gap
cpu_gap_hz
interference_gap_w
```

The summary computes fractions over all formal seed-slot instances. It records the 30 percent thresholds but does not make them canonical publication blockers; they only gate the DD-optimality and IEA-near-optimality performance interpretations.

- [ ] **Step 4: Run GREEN tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_exact.py \
  tests/test_exact_benchmark.py -q
```

Expected: PASS. This command uses only direct synthetic snapshots and development seeds.

- [ ] **Step 5: Commit with an explicit allowlist**

```bash
git add \
  configs/thesis/confirmatory_exact.toml \
  uav_mec/exact.py \
  uav_mec/run_exact_benchmark.py \
  tests/test_exact.py \
  tests/test_exact_benchmark.py
git diff --cached --name-only
git commit -m "feat: audit exact instance nontriviality"
```

## Task 5: Generalize immutable shard planning, resume, and merge

**Files:**

- Modify: `uav_mec/run_shards.py:30-306`
- Modify: `uav_mec/artifacts.py:15-200`
- Modify: `tests/test_run_shards.py`
- Modify: `tests/test_artifacts.py`

- [ ] **Step 1: Write failing generic-shard tests**

Add these exact tests:

```text
test_shard_plan_binds_family_case_registry_lock_source_environment_and_artifact_schema
test_run_one_shard_refuses_a_name_not_in_the_frozen_plan
test_resume_runs_only_missing_shards
test_merge_refuses_missing_shard_without_creating_merged_bundle
test_merge_rejects_duplicate_seed_with_conflicting_valid_payloads
test_merge_rejects_schema_drift_across_valid_shards
test_merge_is_byte_identical_across_shard_completion_order
test_generic_merge_preserves_standard_exact_resource_and_trace_artifacts
test_resume_rejects_config_source_environment_or_lock_hash_drift
test_artifact_writer_stream_copies_bytes_without_json_reserialization
test_v2_artifact_writer_uses_uppercase_manifest_and_sha256sums_closure_names
```

All tests use seeds `71` and `72` under explicit synthetic family `main` and case `default`. Build valid fake shard bundles with `RunArtifactWriter`; never call an experiment runner. For the conflict test, create two individually checksummed shard bundles that both claim the same family/case/seed key but contain different row bytes and assert merge refusal. For completion-order determinism, create equivalent roots with opposite directory creation order and compare SHA-256 of every merged file.

- [ ] **Step 2: Run the RED tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_run_shards.py \
  tests/test_artifacts.py -q
```

Expected: FAIL because there is no public plan, run-one, or merge-only API; the current merge hardcodes three files and does not bind registry, lock, or environment.

- [ ] **Step 3: Implement the immutable plan and generic artifact merge**

`create_shard_plan` writes `shard_index.json` atomically only into an empty case root. The canonical plan JSON contains all `ShardPlan` fields, including exact `family` and `case_id`, and sorted shard specs. A second call may reuse the root only when the complete plan is byte-identical. Formal layout is `FORMAL_ROOT/family/case_id/`; a single-case family still uses its explicit registered case ID.

Each finalized shard manifest must repeat its plan digest, family, case ID, exact seed tuple, runner kind, and a sorted `artifacts` mapping whose values repeat the declared kind, merge strategy, and primary-key sequence. Kinds and strategies match `ArtifactSpec` exactly. `_validate_shard` verifies checksums, plan digest, family/case/seed uniqueness, hash bindings, and artifact declarations before reuse.

`merge_shards` must:

1. validate every planned shard and refuse if any is absent;
2. prove that family and case IDs match and the union of shard seed sets equals the locked case seed set exactly once;
3. reject duplicate rows by each artifact's declared primary key, including conflicting rows whose shard checksums are individually valid;
4. require identical CSV headers and artifact type declarations;
5. sort shards by their plan order and rows by declared primary key;
6. merge all declared artifacts, including `tasks.csv`, `slots.csv`, `events.jsonl`, `raw_radio_snapshots.jsonl`, `exact_slots.csv`, trace provenance JSONL, and summaries;
7. write to a temporary sibling directory and rename only after final checksum validation;
8. if a valid merged bundle already exists, verify its source shard checksums and return it without mutation.

Add `bundle_version` and the exact uppercase/lowercase finalization behavior from the public contract, then extend `RunArtifactWriter` with these exact methods:

| Method | Contract |
|---|---|
| `copy_file(self, relative: str, source: Path) -> None` | Stream one regular file without reserialization, reject symlinks and unsafe relative paths, and register the copied bytes in final checksums. |
| `copy_tree(self, destination: str, source: Path) -> None` | Walk regular files in POSIX-relative-path order, call `copy_file` for each destination, reject special files and symlinks, and preserve empty directories only when they are declared artifacts. |

- [ ] **Step 4: Preserve the existing convenience runner**

Keep `run_sharded_experiment(config_path, output_dir, shard_size=1)` as a development/pilot compatibility wrapper. It resolves the config’s explicit case ID, constructs a plan with `lock_sha256=None`, resumes missing shards, merges, and returns `ShardRunReport`. Formal execution calls the lower-level APIs with a non-null lock hash and exact case ID in Task 8.

- [ ] **Step 5: Run GREEN tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_run_shards.py \
  tests/test_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit with an explicit allowlist**

```bash
git add \
  uav_mec/run_shards.py \
  uav_mec/artifacts.py \
  tests/test_run_shards.py \
  tests/test_artifacts.py
git diff --cached --name-only
git commit -m "feat: generalize immutable shard resume and merge"
```

## Task 6: Freeze a registry-complete runtime budget and expose guarded development calibration

**Depends on:** Task 4 for the Plan 2 Exact Schedule certificate. The real Trace-driven runtime calibration is intentionally deferred until Task 7 provides a verified GeoLife lock; Task 6 unit tests inject checksum-valid calibration-runner doubles and never inspect a formal trace participant.

**Files:**

- Create: `configs/thesis/execution_policy_v2.json`
- Create: `uav_mec/runtime_budget.py`
- Create: `uav_mec/run_development_calibration.py`
- Create: `tests/test_runtime_budget.py`
- Create: `tests/test_development_calibration.py`
- Modify: `uav_mec/runtime_audit.py:11-71`
- Modify: `uav_mec/experiment.py:686-893`
- Modify: `tests/test_runtime_audit.py`
- Modify: `tests/test_thesis_suites.py`

- [ ] **Step 1: Write failing runtime audit and budget tests**

Add these exact tests:

```text
test_runtime_audit_is_descriptive_without_fixed_realtime_gate
test_runtime_audit_reports_reference_hardware_p50_p95_candidates_cache_memory_and_cpu_time
test_runtime_budget_covers_every_registry_family
test_runtime_budget_covers_every_registered_case_exactly_once
test_runtime_budget_includes_cpu_wall_memory_disk_log_concurrency_and_retry_fields
test_runtime_budget_binds_positive_shard_size_and_execution_policy_hash_per_family
test_runtime_budget_uses_power_selected_n_only_for_h1_h4_families
test_runtime_budget_uses_evaluated_n_two_hundred_when_selected_n_is_null
test_runtime_budget_marks_power_guarantee_false_for_null_selection_fallback
test_runtime_budget_uses_thirty_seeds_for_each_descriptive_externality_grid
test_runtime_budget_uses_fixed_counts_for_exact_scaling_robustness_and_trace_driven
test_runtime_budget_rejects_missing_family_audit
test_runtime_budget_rejects_missing_extra_or_duplicate_execution_policy_family
test_runtime_budget_rejects_nonpositive_concurrency_shard_size_or_retry_outside_zero_one
test_runtime_budget_checksum_changes_when_power_or_audit_input_changes
test_execution_policy_freezes_exact_one_other_two_and_all_shard_sizes_one
test_development_calibration_accepts_only_exact_nontriviality_and_runtime_capacity
test_development_calibration_rejects_any_seed_outside_zero_through_999_before_output
test_exact_development_calibration_reports_both_nontriviality_counts_and_fractions
test_runtime_development_calibration_emits_one_audit_row_per_registry_family_case
test_runtime_development_calibration_requires_trace_lock_for_trace_driven
test_development_calibration_records_access_before_dispatch
test_development_calibration_cli_validates_range_workers_case_and_trace_lock
```

Use a synthetic audit JSON under `tmp_path` with one measured row per registered runner/case, a synthetic power decision with `selected_n=50`, and a synthetic schema-version-1 execution-policy JSON containing all eight exact family IDs with positive concurrency and shard sizes. Separately load the repository execution-policy file and assert the exact frozen values below. No timing benchmark runs in the test and no test probes current available memory.

- [ ] **Step 2: Run the RED tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_runtime_audit.py \
  tests/test_runtime_budget.py \
  tests/test_development_calibration.py -q
```

Expected: FAIL because runtime audit still exposes fixed K3/K5 pass thresholds and the budget/calibration modules do not exist.

- [ ] **Step 3: Make runtime audit descriptive**

Remove `RUNTIME_LIMITS_S` and `runtime_gate_pass`. Preserve the five-seed, twenty-slot development audit shape as a calibrated measurement protocol, not a formal pass condition. The report must include:

```text
reference_hardware
python_environment_sha256
p50_runtime_s
p95_runtime_s
mean_runtime_s
candidate_count_p50
candidate_count_p95
candidate_count_total
cache_hit_rate
peak_memory_bytes
log_bytes
measured_cpu_seconds
```

Join termination events to slot rows so formal evidence records candidate evaluations and cache hits per slot. Do not assert an efficiency multiple. A later optional real-time claim is eligible only when measured P95 is no greater than the registered slot duration and deployment assumptions are separately true.

- [ ] **Step 4: Implement and validate the budget calculation**

`build_runtime_budget` reads and hashes the registry, power decision, audit, and execution-policy JSON. The policy contains exactly one object for each exact family ID, in registry order, with only `family`, `concurrency`, and `shard_size`; both numeric values are strictly positive integers. For every registered family/case the builder calculates expected slot-method evaluations, mean-runtime CPU hours, P95-based wall estimate at declared concurrency, peak concurrent memory, raw/summary/event disk, and retry reserve. It refuses a missing or duplicate audit entry, any missing/extra/duplicate policy family, a negative size, a non-finite value, or a retry fraction outside the closed interval from zero to one. It writes canonical JSON atomically and refuses overwrite.

Create `configs/thesis/execution_policy_v2.json` with these complete source-controlled bytes (formatting may be normalized once, but key/value content and array order are exact):

```json
{
  "schema_version": 1,
  "families": [
    {"family": "resource", "concurrency": 2, "shard_size": 1},
    {"family": "exact", "concurrency": 1, "shard_size": 1},
    {"family": "externality", "concurrency": 2, "shard_size": 1},
    {"family": "main", "concurrency": 2, "shard_size": 1},
    {"family": "joint", "concurrency": 2, "shard_size": 1},
    {"family": "scaling", "concurrency": 2, "shard_size": 1},
    {"family": "robustness", "concurrency": 2, "shard_size": 1},
    {"family": "trace_driven", "concurrency": 2, "shard_size": 1}
  ]
}
```

This policy is the conservative choice for the measured 5.4-GiB machine: Exact is serial, all other families use two workers, and every case shards one seed at a time. A different machine policy requires a reviewed source change, new runtime-budget hash, new confirmatory lock, and no formal CLI override.

Define `primary_sample_size = selected_n if selected_n is not None else evaluated_n`. Require a null-selection decision to have `evaluated_n=200` and `evidence_insufficient=true`; budget all four primary families at 200 and set `power_guarantee=false` rather than failing budget construction. A selected decision sets `power_guarantee=true`. For Resource, include three raw generators and four allocation policies without charging four wireless searches per generator. For Externality, create nine `CaseRuntimeBudget` entries: the unique H2 primary uses `primary_sample_size` and each descriptive grid uses exactly 30. Main and Joint also use `primary_sample_size`. Exact, Scaling, Robustness, and Trace-driven use their fixed registered counts. Validate every registry family/case pair exactly once against audit rows and reject a family-only aggregate that hides a missing case. Store concurrency and per-case shard size so the formal runner reads them only from the verified lock-bound budget; the formal CLI exposes neither override. No default worker count is hard-coded, because safe concurrency depends on the measured execution policy and available memory.

Implement the `build` and `verify` parsers exactly as declared in “Command-line contracts shared with formal orchestration.” Parser tests call `main()` with explicit argument arrays and assert that a missing execution policy, unknown family, or unsupported override exits with code 2 before any output path is created.

- [ ] **Step 5: Implement the real development-calibration entry point**

Implement the public enum, seed-range parser, function, and CLI contracts above. The CLI creates `development_permit(registry_path)` and `SeedAccessRecorder(output_temp/seed-access.jsonl, permit)` itself; callers cannot supply a purpose or permit on the command line. It validates the inclusive seed range, case, worker count, required/forbidden trace-lock combination, absent destination, and resolved safe path before creating `output_temp`.

For `exact_nontriviality`, dispatch only the registered Exact case with the requested development seeds and return `exact_nontriviality.json` containing `seed_count`, `slot_count`, `complete_certificate_count`, `cpu_nontrivial_count`, `cpu_nontrivial_fraction`, `schedule_nontrivial_count`, and `schedule_nontrivial_fraction`. Every denominator is the same seed-slot count, and an incomplete certificate is an integrity error rather than a false flag.

For `runtime_capacity`, traverse every registered family/case pair in registry order, use the case’s registered method set and at most 20 slots per development seed, and write `runtime_audit.json` with the complete descriptive fields from Step 3 plus family, case ID, runner kind, seed count, and slot count. Trace-driven loads only locked `calibration` segments; train and final are rejected for this command. Unit tests monkeypatch the eight internal runner adapters with call-recording deterministic doubles, assert full family/case coverage, and prove `record("development_calibration", seeds, metadata)` occurs before the first adapter call. Task 7 adds the real GeoLife integration test.

Both cases publish through `RunArtifactWriter`; the manifest binds registry, source, environment, requested case, seed list, trace-lock digest when present, and the ledger head. A failed adapter leaves no final output directory.

- [ ] **Step 6: Run GREEN tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_runtime_audit.py \
  tests/test_runtime_budget.py \
  tests/test_development_calibration.py \
  tests/test_thesis_suites.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit with an explicit allowlist**

```bash
git add \
  configs/thesis/execution_policy_v2.json \
  uav_mec/runtime_audit.py \
  uav_mec/runtime_budget.py \
  uav_mec/run_development_calibration.py \
  uav_mec/experiment.py \
  tests/test_runtime_audit.py \
  tests/test_runtime_budget.py \
  tests/test_development_calibration.py \
  tests/test_thesis_suites.py
git diff --cached --name-only
git commit -m "feat: freeze runtime budgets and development calibration"
```

## Task 7: Build a participant-disjoint GeoLife trace lock

**Files:**

- Create: `configs/thesis/geolife_protocol_v2.toml`
- Create: `uav_mec/geolife_lock.py`
- Create: `uav_mec/run_trace_experiment.py`
- Create: `tests/test_geolife_lock.py`
- Modify: `uav_mec/geolife.py:24-49,142-263`
- Modify: `tests/test_geolife.py`
- Modify: `tests/test_development_calibration.py`
- Modify: `uav_mec/statistics.py`
- Modify: `tests/test_statistics.py`

- [ ] **Step 1: Write failing lock and participant-cluster tests**

Add these exact tests:

```text
test_geolife_segment_preserves_source_start_and_end_time
test_trace_lock_rejects_participant_overlap_across_train_calibration_and_final
test_trace_lock_rejects_seed_pool_other_than_29000_through_29029
test_trace_lock_records_dataset_source_license_segments_resampling_and_groups
test_trace_lock_is_deterministic_for_the_same_archive_and_protocol
test_trace_lock_changes_when_archive_or_protocol_bytes_change
test_exogenous_seed_does_not_change_locked_trace_participants_or_positions
test_trace_runner_refuses_an_unlocked_participant_or_segment
test_participant_clustered_bootstrap_resamples_source_participants
test_geolife_cli_build_and_verify_match_shared_contract
test_geolife_cli_rejects_symlink_archive_before_creating_output
test_runtime_development_calibration_uses_only_geolife_calibration_split
```

Reuse the synthetic ZIP construction pattern already defined in `tests/test_geolife.py`, extended to at least three participants and two transport modes. The split-overlap test must place one participant in both calibration and final. The exogenous-seed test must build two episodes from one locked final scenario with different development seeds and assert identical participant IDs and position arrays while allowing orthogonal task/channel draws to differ. CLI tests call `main()` with explicit argument arrays and use only synthetic archive bytes under `tmp_path`.

- [ ] **Step 2: Run the RED tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_geolife.py \
  tests/test_geolife_lock.py \
  tests/test_development_calibration.py \
  tests/test_statistics.py -q
```

Expected: FAIL because GeoLife segments do not preserve absolute source times, no trace lock or runner exists, and statistics lacks participant-clustered bootstrap.

- [ ] **Step 3: Extend segment provenance and implement the lock**

Add `source_start` and `source_end` ISO-8601 UTC strings to `GeoLifeSegment`; preserve them through resampling and placement. `build_trace_lock` must verify the archive SHA-256, source/license fields, accepted modes, split declarations, segment existence, resampling and placement bounds, scenario-ID uniqueness, group definitions, and the exact registered trace seed pool. A participant may contribute multiple segments inside one split but cannot appear in another split.

The protocol TOML contains fixed source URL, license text, accepted `walk` and `bike` modes, slot seconds, minimum segment samples, area bounds, exact `train`/`calibration`/`final` split lists, deterministic placement parameters, and final scenario groups. Dataset bytes remain outside Git. Train may support model-free preprocessing, calibration is the only split development calibration may inspect, and confirmatory Trace-driven runs read only final.

`run_trace_experiment` reads only locked final segments. Its formal seed controls task profiles, CSI error, and other exogenous quantities but never participant selection, split, source segment, resampling, translation, or rotation. Every emitted row includes `scenario_id`, `participant_id`, `source_path`, `source_start`, `source_end`, and `trace_lock_sha256`. Wire the Task 6 runtime calibration adapter to the same verified lock loader but require `split="calibration"`; the integration test supplies a lock containing all three splits and asserts that train/final scenario IDs never reach the adapter.

Add the exact statistics API `participant_clustered_bootstrap_ci(differences: numpy.ndarray, participant_ids: Sequence[str], resamples: int = 10_000, seed: int = 20260719) -> pair of float`.

It resamples unique participants with replacement, carries all observations belonging to each selected participant, computes the paired mean, and rejects empty IDs, shape mismatch, fewer than two participants, non-finite values, or invalid resample counts.

Implement `geolife_lock build` and `geolife_lock verify` exactly as declared in the shared CLI table. Resolve the archive with `Path.resolve(strict=True)`, require a regular non-symlink file, stream its digest in bounded chunks, and complete all validation before creating the output parent or temporary file.

- [ ] **Step 4: Run GREEN tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_geolife.py \
  tests/test_geolife_lock.py \
  tests/test_development_calibration.py \
  tests/test_statistics.py -q
```

Expected: PASS using only synthetic ZIP bytes and development seeds.

- [ ] **Step 5: Commit with an explicit allowlist**

```bash
git add \
  configs/thesis/geolife_protocol_v2.toml \
  uav_mec/geolife.py \
  uav_mec/geolife_lock.py \
  uav_mec/run_trace_experiment.py \
  uav_mec/statistics.py \
  tests/test_geolife.py \
  tests/test_geolife_lock.py \
  tests/test_development_calibration.py \
  tests/test_statistics.py
git diff --cached --name-only
git commit -m "feat: lock participant-disjoint GeoLife traces"
```

## Task 8: Capture provenance, build the confirmatory lock, and guard formal dispatch

**Depends on:** Tasks 1, 2, 5, 6, and 7. The final lock also hashes the Resource and Exact implementation from Tasks 3 and 4, so build a real lock only after every implementation task is complete and the worktree state is frozen.

**Files:**

- Create: `uav_mec/provenance.py`
- Create: `uav_mec/correctness_bundle.py`
- Create: `uav_mec/confirmatory_lock.py`
- Create: `uav_mec/run_confirmatory.py`
- Create: `tests/test_provenance.py`
- Create: `tests/test_confirmatory_lock.py`
- Modify: `uav_mec/run_shards.py`

- [ ] **Step 0: Assemble the checksum-closed correctness bundle before the final pilot**

Create `uav_mec/correctness_bundle.py` and `tests/test_correctness_bundle.py`. The bundle consumes a fresh fixed-interpreter JUnit XML report, verifies the Plan 2 `AlgorithmAuditRecord`, the 10,000-action incremental-equivalence test, externality three-layer conservation test, DD/Exact Admission agreement test, MethodSpec factor test, and closest-work adaptation mapping test by exact test IDs and source hashes. Its public builder is `build_correctness_bundle(test_report_path, repo_root, registry_path, output_dir)`; it writes only regular files under `artifacts/preregistration/v2/correctness/v2/`, with `MANIFEST.json` and `SHA256SUMS` written last, and records the source commit and registry/config hashes. Tests reject stale source hashes, missing required test IDs, extra files, NaN values, and an exposure-only certificate that lacks completion or allocated-CPU conservation. The confirmatory-lock builder copies this bundle into its immutable lock and stores `correctness_bundle_sha256`; the analyzer reads it through the lock and never invents a `runs_root/correctness/` directory. The bundle test also proves that a ledger prefix plus valid post-prefix permits passes audit, while a changed prefix byte, missing chain link, wrong lock digest, undeclared family/case, or seed outside the permit fails before any report is written.

Run:

    /home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest tests/test_correctness_bundle.py -q

- [ ] **Step 1: Write failing provenance and lock tests**

Add these exact tests:

```text
test_source_snapshot_archives_all_research_source_bytes_deterministically
test_dirty_source_snapshot_archives_a_complete_binary_patch
test_dirty_source_snapshot_refuses_a_missing_patch
test_environment_snapshot_hashes_only_registered_stable_fields
test_provenance_cli_capture_and_verify_match_shared_contract
test_provenance_readiness_cli_writes_byte_stable_json_and_markdown
test_pre_pilot_readiness_never_reports_ready_for_formal
test_readiness_rejects_seed_audit_registry_or_phase_drift_before_outputs
test_confirmatory_lock_contains_all_required_protocol_and_provenance_hashes
test_confirmatory_lock_uses_power_selected_prefixes_for_h1_h4
test_confirmatory_lock_uses_n_two_hundred_and_false_power_guarantee_for_null_selection
test_confirmatory_lock_uses_fixed_prefixes_for_other_families
test_confirmatory_lock_contains_one_locked_case_per_registered_case
test_confirmatory_lock_freezes_externality_primary_selected_n_and_eight_descriptive_n_thirty
test_confirmatory_lock_rejects_a_prelock_access_to_any_candidate_formal_pool
test_confirmatory_lock_rejects_missing_or_stale_runtime_budget
test_confirmatory_lock_rejects_missing_or_stale_trace_lock
test_confirmatory_lock_invalidates_on_registry_config_analysis_source_or_environment_drift
test_formal_runner_rejects_seed_slot_method_and_scenario_overrides
test_formal_runner_enforces_resource_exact_externality_main_joint_scaling_robustness_trace_driven_order
test_formal_runner_dispatches_each_family_to_its_registered_runner_kind
test_formal_runner_records_access_before_opening_a_formal_seed
test_formal_permit_exposes_only_the_exact_locked_case_seed_set
test_confirmatory_lock_cli_build_and_verify_match_shared_contract
test_formal_cli_accepts_exact_family_resume_and_verify_only_modes
test_formal_cli_accepts_root_verify_all_without_dispatching_work
test_formal_cli_rejects_workers_shard_size_output_all_and_protocol_overrides
test_formal_runner_reads_workers_and_shard_size_only_from_locked_runtime_budget
```

The source tests must initialize a temporary Git repository under `tmp_path`, commit two small allowlisted files, and use a deterministic binary file change for the dirty-patch case. The formal-runner tests must monkeypatch runner dispatch with call-recording functions and must not execute any trace or formal seed.

- [ ] **Step 2: Run the RED tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_provenance.py \
  tests/test_confirmatory_lock.py \
  tests/test_run_shards.py -q
```

Expected: FAIL because provenance, lock, and formal dispatcher modules do not exist.

- [ ] **Step 3: Implement deterministic provenance capture**

`capture_source_snapshot` must archive the exact bytes of these roots and files in sorted relative-path order:

```text
uav_mec
tests
configs/thesis
Makefile
pyproject.toml
requirements.txt
requirements-lock.txt
```

Normalize archive owner, group, mode, and modification time so unchanged bytes produce the same `tree_sha256`. Reject symlinks, unsafe paths, sockets, and files outside the allowlist. Record Git commit and clean state. When dirty, write a complete binary patch covering tracked changes and include all allowlisted untracked source bytes in the archive; compute and store `patch_sha256`. A dirty snapshot with no patch bytes is invalid.

`capture_environment_snapshot` records Python version, platform, architecture, and exact NumPy, SciPy, Matplotlib, and pytest versions. It excludes hostname, process ID, current time, output path, and worker count so equivalent shard workers share one stable environment hash.

Implement `provenance capture` and `provenance verify` exactly as declared in the shared CLI table. The capture command resolves the repository and output paths, rejects an output inside a source allowlist root, validates both destinations before creating either, and removes its temporary sibling if either capture fails. The verify command is read-only.

Implement `provenance readiness` as a report renderer over a verified seed-audit JSON, a fresh JUnit report, and a checksum-closed correctness bundle, not as an experiment or a substitute for the test commands. It verifies the registry digest, accepted audit schema and phase, source hash, pass/failure counts, and bundle closure, renders the same sorted facts to JSON and Markdown, and publishes the two outputs transactionally through temporary siblings. For `before-final-pilot`, `ready_for_pilot` is true only when all software/correctness checks pass and the audit proves all candidate pools unseen, while `ready_for_formal` is unconditionally false because power, budget, trace, source, environment, and confirmatory locks do not yet exist. A mismatch or one pre-existing output leaves both destinations unchanged.

- [ ] **Step 4: Implement confirmatory-lock construction and verification**

The lock builder must verify, in this order:

1. registry schema and all registered config semantics;
2. complete pilot bundle hashes and exact pilot seed set `1000..1099`;
3. paired-t/Holm power metadata, exact candidate set, 10,000 simulations, analysis seed 20260719, the complete three-by-four power table, selected N, and the logically equivalent evidence-insufficient flag;
4. runtime budget registry/power/execution-policy hashes, positive per-family concurrency and shard size, and complete family coverage;
5. trace lock registry seed set, dataset/protocol hashes, and participant split validity;
6. source snapshot and dirty-patch completeness;
7. environment snapshot hash;
8. seed-ledger hash chain and absence of candidate formal access before the lock boundary; copy the exact prefix bytes into `seed-access-prelock.jsonl`, record its byte length/head, and store the normalized live-ledger relative path. Later formal/analysis records must append after this prefix and carry the current lock/family/case permit.

Then derive one `LockedCase` per registered case. Use `selected_n` for Resource, H2 primary Externality, Main, and Joint when present; when it is null, use exactly the first 200 family candidate seeds, require the decision/budget fallback invariants, and freeze `power_guarantee=false`. Use registry-fixed counts for Exact, Scaling, Robustness, and Trace-driven and exactly 30 seeds for each of the eight descriptive Externality grids. Preserve family and case order. Reject missing, extra, duplicate, or role-mismatched cases even when a family-level union of seeds would look valid. Store the four-member Holm family, MDEs, estimand, bootstrap count 10,000, bootstrap seed 20260719, pilot input hashes, simulation count, power analysis seed, complete power table, source commit/snapshot, config hashes, analysis-code hash, runtime budget, trace lock, environment, ledger head, and normalized allowed command templates.

Write the lock with `RunArtifactWriter(output_dir, bundle_version=2)` into a new directory, never as an overwritable standalone file. Verification must recalculate `MANIFEST.json`, `SHA256SUMS`, every referenced hash, and reject extra, missing, reordered, or duplicate family/hypothesis entries.

Implement `confirmatory_lock build` and `confirmatory_lock verify` exactly as declared in the shared CLI table. Parse and fully validate every input before creating the output directory. The build parser has no source, seed, sample-size, family, worker, shard-size, or analysis override beyond the named immutable input paths; the verify parser accepts only `--lock DIR`.

- [ ] **Step 5: Implement ordered formal dispatch without override arguments**

Implement only the two parser modes declared in the shared CLI table:

```text
run_confirmatory --lock DIR --family ID --root FORMAL_ROOT
run_confirmatory --lock DIR --family ID --root FORMAL_ROOT --resume
run_confirmatory --lock DIR --family ID --root FORMAL_ROOT --verify-only
run_confirmatory --lock DIR --root FORMAL_ROOT --verify-all
```

`--resume` and `--verify-only` are mutually exclusive. Aggregate `--verify-all` is mutually exclusive with family mode and never creates, resumes, or dispatches a shard. Reject `--workers`, `--shard-size`, `--output`, `--all`, and any seed, count, slot, method, case, scenario, or config option at parse time. For family execution, read positive concurrency from the family budget and shard size from each case budget in the lock-bound runtime budget. This avoids an unsafe machine-independent worker default and makes the runtime policy part of the confirmatory boundary.

Before constructing a formal permit, verify the current source, config, analysis code, environment, runtime budget, execution policy, trace lock, and lock bundle. Write one atomic `formal_run_state.json` directly under `FORMAL_ROOT`; each case writes beneath `FORMAL_ROOT/family/case_id`. Refuse to start a family until all predecessor families and all their cases are checksum-complete and integrity-validated. Performance failure does not stop later families; a publication-blocking integrity failure does.

Dispatch mapping is exact:

```text
resource -> run_resource_experiment
exact -> run_exact_definition
externality -> registered suite runner, primary plus eight descriptive grids
main -> run_definition
joint -> registered suite runner
scaling -> registered suite runner
robustness -> registered suite runner
trace_driven -> run_trace_experiment
```

The family ID is passed only to this fixed dispatch table, never evaluated as a module or path. The dispatcher resolves the live ledger from `lock.seed_ledger_relpath`, verifies the lock-copied prefix before constructing `formal_permit(lock, family, case_id)`, and writes each plan beneath `FORMAL_ROOT/family/case_id`. Every runner receives a case-scoped recorder writing to that one append-only ledger and records the run operation before the first call to `generate_trace` or locked trace construction. Shard plans receive non-null registry, lock, source, and environment hashes. `--verify-only` and `--verify-all` validate all case/shard/merge manifests plus state ordering but do not record seed access because they never read result rows or generate traces. The formal root never contains a second mutable ledger copy.

- [ ] **Step 6: Run GREEN tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_provenance.py \
  tests/test_confirmatory_lock.py \
  tests/test_run_shards.py -q
```

Expected: PASS without opening a pilot or formal seed.

- [ ] **Step 7: Commit with an explicit allowlist**

```bash
git add \
  uav_mec/provenance.py \
  uav_mec/confirmatory_lock.py \
  uav_mec/run_confirmatory.py \
  uav_mec/run_shards.py \
  tests/test_provenance.py \
  tests/test_confirmatory_lock.py \
  tests/test_run_shards.py
git diff --cached --name-only
git commit -m "feat: bind formal runs to confirmatory locks"
```

## Task 9: Produce registry-driven statistics and separated claim gates

**Depends on:** Tasks 1 through 8 plus the core correctness evidence for incremental/full equivalence, externality-certificate conservation, closed-schedule integrity, DD/Exact Admission agreement, and explicit termination reasons.

**Files:**

- Create: `uav_mec/claim_gates.py`
- Create: `uav_mec/analyze_confirmatory.py`
- Create: `tests/test_claim_gates.py`
- Create: `tests/test_confirmatory_analysis.py`
- Modify: `uav_mec/analyze_results.py:117-419,504-628`
- Modify: `tests/test_result_analysis.py`

- [ ] **Step 1: Write failing claim-classification tests**

Add these exact tests to `tests/test_claim_gates.py`:

```text
test_claim_is_meaningful_supported_only_when_ci_low_exceeds_mde_and_holm_passes
test_claim_reports_observed_mde_but_uncertain_when_estimate_reaches_mde_and_ci_low_does_not
test_claim_reports_positive_but_small_when_stable_effect_is_below_mde
test_claim_is_insufficient_when_ci_contains_zero_or_holm_fails
test_claim_is_negative_when_point_estimate_is_negative
test_realtime_claim_requires_measured_p95_within_slot_and_deployment_assumptions
test_efficiency_claim_reports_measurement_without_a_fixed_speedup_threshold
```

Classification precedence is exact: a negative point estimate is `negative`; otherwise CI containing zero or failed Holm is `insufficient`; otherwise CI lower bound above MDE is `meaningful_supported`; otherwise point estimate at least MDE is `positive_mde_observed_but_uncertain`; otherwise it is `positive_but_small`.

- [ ] **Step 2: Write failing analyzer tests**

Add these exact tests to `tests/test_confirmatory_analysis.py`:

```text
test_analyzer_holm_family_is_exactly_h1_h2_h3_h4_paired_t_tests
test_analyzer_reports_wilcoxon_as_sensitivity_without_using_it_for_primary_gates
test_analyzer_rejects_self_consistent_bundle_missing_a_locked_seed
test_analyzer_rejects_missing_extra_or_role_mismatched_locked_case
test_analyzer_validates_externality_primary_and_all_eight_descriptive_case_counts
test_analyzer_rejects_extra_seed_config_source_environment_or_lock_hash
test_analyzer_translates_each_mde_to_mean_added_tasks_per_episode
test_analyzer_reports_all_arrivals_requires_offload_and_local_feasible_estimands
test_analyzer_reports_joint_four_simple_effects_and_interaction_interval
test_analyzer_reports_h3_by_all_three_radio_generators
test_h3_generalization_requires_direction_stability_across_generators
test_analyzer_validates_dd_exact_cardinality_and_tie_cpu_agreement
test_analyzer_preserves_realized_csi_outage_as_robustness_not_planning_violation
test_analyzer_marks_iteration_cap_as_local_optimum_claim_failure
test_analyzer_marks_exact_nontriviality_failures_as_claim_failures_not_archive_failures
test_acceptance_separates_integrity_correctness_primary_and_optional_claims
test_negative_hypothesis_result_remains_in_acceptance_when_integrity_passes
test_analyzer_build_cli_accepts_only_lock_runs_root_and_output
test_analyzer_verify_cli_is_read_only_and_accepts_only_lock_runs_root_analysis
test_analyzer_cli_rejects_registry_correctness_and_family_root_overrides
test_analyzer_writes_nineteen_column_statistical_claims_not_claim_matrix
test_analyzer_verify_rejects_claim_matrix_alias_or_missing_statistical_claims
```

Define two ordinary helper functions in `tests/test_confirmatory_analysis.py`: `_write_synthetic_lock(root: Path, case_seed_labels: Mapping[tuple[str, str], Sequence[int]]) -> Path` writes a checksum-valid lock whose family/case records use only caller-supplied synthetic labels; `_write_synthetic_runs_root(root: Path, lock_path: Path) -> Path` writes one checksum-valid minimal root per locked `family/case_id` plus `correctness/` and `formal_run_state.json`. Every test calls these helpers explicitly, so no hidden or undefined pytest fixture supplies a formal seed. Do not load the repository formal config through an experiment runner. Include separate `planned_sic_violations` and `realized_sic_outages` fields so the CSI test proves the distinction.

- [ ] **Step 3: Run the RED tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_claim_gates.py \
  tests/test_confirmatory_analysis.py \
  tests/test_result_analysis.py -q
```

Expected: FAIL because no claim-gate or v2 analyzer module exists and the v1 analyzer uses hard-coded comparisons plus one mixed gate set.

- [ ] **Step 4: Implement registry-driven family loading and paired statistics**

Load the registry, correctness bundle, and case definitions only through the verified lock. Under `runs_root`, require `formal_run_state.json` and one `family/case_id` directory for every `LockedCase`; the mutable append-only ledger remains at the lock's declared `seed_ledger_relpath` and is verified for prefix/permit closure before analysis reads any values. For every case, verify role, checksums, lock hash, exact locked seeds, config/source/environment hashes, method set, slot coverage, shared trace identity, unique task IDs, planned SIC, block capacity, CPU budgets, and required artifact schemas. Construct one case-scoped recorder per locked case and record analysis access before reading its outcome values. Reject a missing/extra case or an aggregate family directory even when its union of seeds matches.

For H1--H4, obtain treatment/control and optional case ID solely from the registry. Compute seed-episode paired completion-rate differences in percentage points, deterministic 10,000-resample bootstrap CI, paired t primary p-value, Wilcoxon sensitivity p-value, and rank-biserial effect. Apply `holm_adjust` to exactly four `p_t_raw` values. Auxiliary comparisons use separately named families and cannot change these four adjusted values.

Compute `mean_added_tasks_per_episode = mde_pp * mean_arrivals_per_episode / 100`. Also report requires-offload conditional completion and pure-local completion proportion without changing the primary denominator.

For Joint, report these paired seed vectors and intervals exactly:

```text
wireless_under_dd = iea_dd - exchange_dd
wireless_under_deadline_first = iea_edf - exchange_edf
resource_under_iea = iea_dd - iea_edf
resource_under_exchange = exchange_dd - exchange_edf
interaction = wireless_under_dd - wireless_under_deadline_first
```

H4 remains `iea_dd - exchange_edf`; interaction is secondary and receives no H4 MDE.

- [ ] **Step 5: Implement the acceptance schema and gate evidence paths**

Write `statistics.json`, `metrics.json`, `acceptance.json`, and `statistical_claims.csv`. The analyzer must not write any file named `claim_matrix.csv`; that name is reserved for the sole twelve-column thesis--journal matrix generated by Plan 5 during canonical publication. `acceptance.json` has these top-level sections:

```text
schema_version
integrity_gates
correctness_gates
claim_gates
optional_claim_gates
all_integrity_gates_pass
```

Publication-blocking integrity gates include complete tests/audit input, planned hard constraints, shared traces, incremental/full equivalence, DD/Exact agreement under assumptions, complete shard/config/source/environment provenance, automated analysis lineage, and absence of manual values. Realized CSI outage is a robustness metric. Exact 30 percent fractions, iteration-cap local-optimum eligibility, effect magnitude, efficiency, and real-time wording are claim gates, not archive blockers.

Each `statistical_claims.csv` row contains these exact nineteen fields:

```text
claim_id
category
family
case_id
treatment
control
estimand
mde_pp
estimate_pp
ci_low_pp
ci_high_pp
p_t_raw
p_holm
p_wilcoxon_raw
rank_biserial
mean_added_tasks_per_episode
status
wording_level
evidence_paths
```

Evidence paths are canonical-root-relative strings that the publisher can validate later; multiple paths are encoded as a canonical JSON array inside the CSV cell.

Keep the old `analyze()` entry point in `analyze_results.py` for v1 fixtures, but move shared checksum/bundle validation helpers there or re-export them without routing v2 through the old `COMPARISONS` constant.

Implement exactly two analyzer subcommands: `analyze_confirmatory build --lock DIR --runs-root FORMAL_ROOT --output DIR` and `analyze_confirmatory verify --lock DIR --runs-root FORMAL_ROOT --analysis DIR`. Build rejects `--registry`, `--formal-root`, `--correctness-root`, family mappings, bootstrap count/seed, and seed/protocol overrides; formal bootstrap values are the locked 10,000 and 20260719. It resolves all inputs, creates one case-scoped formal analysis recorder per lock case, and calls `analyze_confirmatory`. Parser rejection and all integrity validation occur before the output parent is created. Verify calls `verify_confirmatory_analysis`, opens outputs read-only, does not append the seed ledger, and replaces any separate `claim_gates verify` command.

- [ ] **Step 6: Run GREEN tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_claim_gates.py \
  tests/test_confirmatory_analysis.py \
  tests/test_result_analysis.py \
  tests/test_statistics.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit with an explicit allowlist**

```bash
git add \
  uav_mec/claim_gates.py \
  uav_mec/analyze_confirmatory.py \
  uav_mec/analyze_results.py \
  tests/test_claim_gates.py \
  tests/test_confirmatory_analysis.py \
  tests/test_result_analysis.py
git diff --cached --name-only
git commit -m "feat: add registry driven claim gates"
```

## Task 10: Publish canonical v2 full and derive light by hash

**Depends on:** Task 9 acceptance/statistics output and the publication plan’s `uav_mec.publication_assets.generate_bilingual_publication_assets` pure generator landing first. Task 9 supplies `statistical_claims.csv`; Plan 5’s generator alone supplies the shared twelve-column `claim_matrix.csv`. There is no independently supplied rendered-assets or claim-matrix directory: the publisher invokes the generator before hashing or publishing either tree. The publisher may archive negative H1--H4 results; it may not archive failed publication-blocking integrity evidence as canonical v2.

**Files:**

- Create: `uav_mec/canonical_v2.py`
- Create: `tests/test_canonical_v2.py`
- Modify: `uav_mec/artifacts.py`
- Modify: `uav_mec/publish_canonical.py:164-283`
- Modify: `tests/test_artifacts.py`
- Modify: `tests/test_canonical_publication.py`

- [ ] **Step 1: Write failing canonical full/light tests**

Add these exact tests:

```text
test_v2_publisher_archives_every_locked_case_raw_shard_byte_inside_full
test_v2_publisher_archives_configs_source_environment_trace_summaries_events_and_analysis
test_v2_publisher_derives_light_only_from_full
test_light_provenance_resolves_every_file_to_a_full_sha256
test_statistical_claims_evidence_paths_exist_and_are_covered_by_root_checksums
test_v2_publisher_accepts_only_generator_claim_matrix_with_exact_twelve_columns
test_v2_publisher_validates_every_claim_matrix_artifact_path_against_root_closure
test_v2_publisher_rejects_analyzer_claim_matrix_or_second_generated_claim_matrix
test_root_uses_manifest_json_and_sha256sums_names
test_v2_verifier_rejects_unlisted_missing_tampered_or_symlinked_file
test_v2_publisher_rejects_failed_publication_blocking_integrity_gate
test_v2_publisher_keeps_negative_claims_when_integrity_passes
test_v2_publisher_requires_dirty_patch_or_complete_source_snapshot
test_v2_publisher_refuses_external_content_address_references_without_local_bytes
test_v2_publication_is_byte_deterministic_for_identical_inputs
test_v2_publisher_invokes_pure_bilingual_asset_generator_before_manifesting
test_v2_publisher_hashes_generated_tables_figures_and_latex_into_full_and_light
test_v2_publisher_rejects_generator_missing_extra_or_digest_mismatched_output
test_v2_publisher_writes_publication_assets_manifest_before_root_checksums
test_v2_cli_accepts_only_lock_runs_root_analysis_and_output
test_v2_cli_rejects_registry_rendered_assets_and_family_root_overrides
test_v1_publisher_remains_compatible
```

Construct tiny full inputs under `tmp_path` with one raw shard per locked family/case, a source archive, environment JSON, trace lock, accepted-event certificate, and analysis files including the nineteen-column `statistical_claims.csv`. Define `_fake_asset_generator(analysis_root: Path, staging_root: Path) -> Mapping[str, str]` in `tests/test_canonical_v2.py`; it writes fixed bilingual table, figure, and LaTeX bytes plus one root `claim_matrix.csv` with the exact twelve-column schema below, then returns every exact relative-path/SHA-256 pair. Negative tests define separate named fake generators for a missing path, undeclared extra file, wrong digest, wrong claim header, unsafe or missing artifact path, and a second claim-matrix path. The negative-claim input sets H1 status to `negative` while all integrity gates pass. The determinism test publishes twice to different output paths and compares every relative path plus SHA-256, excluding no files.

- [ ] **Step 2: Run the RED tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_canonical_v2.py \
  tests/test_canonical_publication.py \
  tests/test_artifacts.py -q
```

Expected: FAIL because current v2 publication deliberately excludes raw rows, has no full/light structure, uses lowercase run-bundle root names, and requires one undifferentiated pass flag.

- [ ] **Step 3: Implement the authoritative full archive**

Create a new output directory and copy bytes without JSON reserialization into:

```text
full/configs
full/source_snapshot
full/environment
full/traces
full/raw_shards
full/summaries
full/event_certificates
full/analysis/statistics
full/analysis/tables
full/analysis/figures
full/analysis/latex
full/analysis/PUBLICATION_ASSETS.json
full/analysis/statistical_claims.csv
full/analysis/claim_matrix.csv
full/analysis/acceptance.json
full/analysis/qa_report.md
```

All locked family/case roots and every planned shard must be present and checksum-valid. Copy the actual raw bytes and their source checksums; a path or content-address digest without local bytes is insufficient. Copy the exact configs named by the lock, source archive and patch when applicable, environment snapshot, trace lock, merged summaries, event certificates, registry-driven statistics, `statistical_claims.csv`, acceptance, and QA report. Reject `analysis_root/claim_matrix.csv`: the analyzer is never a producer of the publication matrix.

After upstream verification and before either manifest exists, create a temporary `asset-staging` directory beside the temporary canonical tree and call `asset_generator(analysis_root, asset_staging)`. The default callable is imported from `uav_mec.publication_assets.generate_bilingual_publication_assets`. Require its return mapping to contain every regular staging file exactly once; normalize each key as a safe POSIX-relative path, reject symlinks/directories/special files and reserved canonical names, and recompute every reported SHA-256. Require exactly one returned path named `claim_matrix.csv` and forbid any other basename or normalized path that denotes a claim matrix. Copy verified generated `tables/`, `figures/`, and `latex/` bytes into `full/analysis/` without reserialization, and copy the sole root staging `claim_matrix.csv` to `full/analysis/claim_matrix.csv`. Write `full/analysis/PUBLICATION_ASSETS.json` with generator module/name, generator source SHA-256, analysis manifest SHA-256, direct hashes for `publication/figure_manifest.toml`, `publication/claim_bindings.toml`, `uav_mec/publication_assets.py`, and `uav_mec/publication_claims.py`, and the sorted returned path/hash map. Recompute each direct source hash from the locked source snapshot and reject drift. Remove the staging directory only after those bytes and this manifest are in the temporary full tree. A generator failure leaves no output root.

Before asset generation, validate `all_integrity_gates_pass=true`, but do not require positive claim statuses, Exact nontriviality thresholds, efficiency, or real-time eligibility. Resolve every canonical-JSON-array `evidence_paths` cell in `statistical_claims.csv` against the future canonical root and require each target to be a regular local file.

After copying generated assets, validate `full/analysis/claim_matrix.csv` with the exact header `claim_id,claim_text,scope,equation,theorem,test,experiment_family,artifact_path,figure_or_table,thesis_section,paper_section,gate_status`. Reject reordered, missing, or extra fields, blank/duplicate claim IDs, a blank required field, an unknown gate status, and an absolute or parent-traversing `artifact_path`. After full and light are constructed, require every `artifact_path` to resolve to a regular file inside the temporary canonical root; after root `SHA256SUMS` is written, verify every resolved path is covered by it. This twelve-column file is the only artifact called `claim_matrix.csv` in full or light and is the single source shared by both manuscripts.

- [ ] **Step 4: Derive light deterministically from full**

Select only these full-backed classes:

```text
configs
summaries
statistics
tables
figures
latex
PUBLICATION_ASSETS.json
statistical_claims.csv
claim_matrix.csv
acceptance.json
qa_report.md
```

For every light file, write `light/provenance.json` with its light path, source full path, full SHA-256, and light SHA-256. No light writer may read `artifacts/runs`, pilot, smoke, legacy, or canonical-v1 paths.

Generated tables, bilingual figures, and LaTeX staging follow the same full-to-light copy-and-provenance rule as statistics. No publication asset may be created or changed after these copies. Finalize `full/MANIFEST.json` plus `full/SHA256SUMS`, then `light/MANIFEST.json` plus `light/SHA256SUMS`. Finally write root `MANIFEST.json`, then root `SHA256SUMS` covering every regular file except the root checksum file itself. Sort all checksum lines by POSIX relative path. `verify_canonical_v2` rejects an extra file, missing file, digest mismatch, symlink, unsafe path, broken light provenance, unresolved claim evidence, inconsistent lock/registry hash, or any generated-asset path/hash missing from the asset manifest and all applicable closures.

Implement exactly `canonical_v2 publish --lock DIR --runs-root FORMAL_ROOT --analysis DIR --output DIR` and `canonical_v2 verify --root DIR`. The publish CLI calls the default generator internally and offers no generator/staging/rendered-assets override; tests inject a fake only through the public function. Reject registry and per-family root overrides because both resolve through the lock and runs root. Keep `publish_canonical()` as the v1 function in `publish_canonical.py`; do not make the old CLI silently publish v2.

- [ ] **Step 5: Run GREEN tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_canonical_v2.py \
  tests/test_canonical_publication.py \
  tests/test_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit with an explicit allowlist**

```bash
git add \
  uav_mec/canonical_v2.py \
  uav_mec/artifacts.py \
  uav_mec/publish_canonical.py \
  tests/test_canonical_v2.py \
  tests/test_canonical_publication.py \
  tests/test_artifacts.py
git diff --cached --name-only
git commit -m "feat: publish canonical v2 full and light"
```

## Task 11: Add guarded Makefile entry points

**Files:**

- Modify: `Makefile:1-53`
- Modify: `tests/test_project_metadata.py`

- [ ] **Step 1: Write failing Makefile contract tests**

Add these exact tests:

```text
test_makefile_uses_fixed_evidence_python_for_v2_targets
test_makefile_exposes_evidence_test_and_development_smoke_targets
test_makefile_exposes_power_budget_trace_lock_confirmatory_lock_targets
test_makefile_exposes_ordered_formal_analysis_canonical_and_verify_targets
test_formal_make_target_has_no_seed_slot_method_or_scenario_override_variables
test_formal_make_target_has_no_worker_shard_size_output_or_all_override
test_runtime_budget_make_target_requires_an_execution_policy_file
test_canonical_make_target_points_to_v2_full_light_publisher
test_canonical_make_target_has_no_registry_or_rendered_assets_override
test_v2_make_targets_resolve_dynamic_paths_and_reject_prefix_escape_before_mutation
test_v2_make_targets_reject_symlink_inputs_and_existing_nonresumable_outputs
```

Parse Makefile text only. Assert the exact interpreter `/home/link/miniconda/envs/llm-ops-torch26/bin/python`, required target names, and absence of seed/slot override flags in the formal target recipe.

- [ ] **Step 2: Run the RED test**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_project_metadata.py -q
```

Expected: FAIL because the v2 targets do not exist and Makefile defaults to canonical-v1.

- [ ] **Step 3: Add exact guarded targets**

Define:

```make
EVIDENCE_PYTHON := /home/link/miniconda/envs/llm-ops-torch26/bin/python
REGISTRY_V2 := configs/thesis/experiment_registry_v2.toml
PREREG_ROOT ?= artifacts/preregistration/v2
PILOT_ROOT ?= artifacts/preregistration/v2/power-pilot/final
PILOT_WORKERS ?= 2
FORMAL_ROOT ?= artifacts/runs/v2-confirmatory
LOCK_ROOT ?= artifacts/preregistration/v2/confirmatory-lock
ANALYSIS_ROOT ?= artifacts/analysis/v2-confirmatory
CANONICAL_V2 ?= artifacts/canonical/v2
EXECUTION_POLICY ?= configs/thesis/execution_policy_v2.json
GEOLIFE_ARCHIVE ?=
RETRY_FRACTION ?=
```

Add these phony targets:

```text
evidence-test
evidence-smoke
power-pilot-v2
runtime-budget-v2
trace-lock-v2
source-snapshot-v2
confirmatory-lock-v2
formal-v2
confirmatory-v2
analyze-confirmatory-v2
verify-analysis-v2
canonical-v2
verify-canonical-v2
```

`evidence-smoke` uses `configs/thesis/smoke.toml`, development permit, a two-seed shard root, and no formal config. Every target that needs a runtime path checks that the variable is nonempty, resolves it with the fixed interpreter, proves it lies beneath the expected artifact prefix, and refuses an existing destination unless the underlying command is explicitly resumable. The source-controlled defaults are the exact roots in the runtime-output map; Plan 4 validates that each nonresumable root is absent before use. A caller-supplied override must pass the same resolved-prefix and symlink checks and may not be synthesized from an unvalidated shell substitution inside a recipe.

`power-pilot-v2` validates `PILOT_WORKERS` as a positive integer and invokes the suite with `--workers "$(PILOT_WORKERS)"`; its source-controlled default is two and it has no formal seed or hypothesis override. `runtime-budget-v2` requires `EXECUTION_POLICY` to resolve to an existing regular non-symlink JSON file and passes it as `runtime_budget build --execution-policy`; it does not invent concurrency from the current machine. `formal-v2` validates `FAMILY` against `resource exact externality main joint scaling robustness trace_driven` and invokes exactly `run_confirmatory --lock "$(LOCK_ROOT)" --family "$(FAMILY)" --root "$(FORMAL_ROOT)" --resume`. It exposes no seed, slot, method, scenario, worker, shard-size, or family-output variable. `confirmatory-v2` is read-only aggregate verification and invokes exactly `run_confirmatory --lock "$(LOCK_ROOT)" --root "$(FORMAL_ROOT)" --verify-all`; there is no `--all` execution mode.

`analyze-confirmatory-v2` invokes exactly `analyze_confirmatory build --lock "$(LOCK_ROOT)" --runs-root "$(FORMAL_ROOT)" --output "$(ANALYSIS_ROOT)"`. Add `verify-analysis-v2` with exactly `analyze_confirmatory verify --lock "$(LOCK_ROOT)" --runs-root "$(FORMAL_ROOT)" --analysis "$(ANALYSIS_ROOT)"`. `canonical-v2` invokes exactly `canonical_v2 publish --lock "$(LOCK_ROOT)" --runs-root "$(FORMAL_ROOT)" --analysis "$(ANALYSIS_ROOT)" --output "$(CANONICAL_V2)"`; the bilingual generator is internal and no rendered-assets or registry variable is accepted. `verify-canonical-v2` passes only `--root "$(CANONICAL_V2)"`.

Change the writing-facing default canonical input to `artifacts/canonical/v2/light`; retain a separately named v1 variable for historical rebuilds.

- [ ] **Step 4: Run GREEN metadata tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_project_metadata.py -q
```

Expected: PASS.

- [ ] **Step 5: Exercise only non-experiment help and test gates**

Run:

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.canonical_v2 --help
make evidence-test
```

Expected: both help commands exit zero and `make evidence-test` runs the test suite. Do not invoke `evidence-smoke`, pilot, trace-lock, or formal targets in this implementation step.

- [ ] **Step 6: Commit with an explicit allowlist**

```bash
git add Makefile tests/test_project_metadata.py
git diff --cached --name-only
git commit -m "build: add guarded evidence workflows"
```

## Task 12: Verify the implementation and transfer execution authority to Plan 4

This task is implementation-only. It must not build a real GeoLife lock, open pilot/formal seeds, calibrate on the real dataset, create a confirmatory lock, run a family, analyze formal rows, or publish canonical output. The sole operator protocol for those state-changing actions is `docs/superpowers/plans/2026-07-19-04-confirmatory-execution-implementation.md`; do not copy its campaign commands into this plan.

- [ ] **Step 1: Run the complete GREEN suite**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest -q
```

Expected: PASS with no generated trace outside development seeds. Record the exact pass count and duration in the implementation handoff; do not hard-code a historical count.

- [ ] **Step 2: Run the focused evidence GREEN suite**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_experiment_registry.py \
  tests/test_seed_access.py \
  tests/test_statistics.py \
  tests/test_power_analysis.py \
  tests/test_power_pilot.py \
  tests/test_resource_snapshots.py \
  tests/test_exact.py \
  tests/test_exact_benchmark.py \
  tests/test_run_shards.py \
  tests/test_runtime_audit.py \
  tests/test_runtime_budget.py \
  tests/test_development_calibration.py \
  tests/test_geolife.py \
  tests/test_geolife_lock.py \
  tests/test_provenance.py \
  tests/test_confirmatory_lock.py \
  tests/test_claim_gates.py \
  tests/test_confirmatory_analysis.py \
  tests/test_canonical_v2.py \
  tests/test_project_metadata.py -q
```

Expected: PASS using only direct synthetic inputs, fake runner adapters, synthetic ZIPs, and development seeds.

- [ ] **Step 3: Smoke every shared CLI without state changes**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.seed_access audit --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_power_pilot_suite --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_development_calibration --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.runtime_budget build --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.runtime_budget verify --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.geolife_lock build --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.geolife_lock verify --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.provenance capture --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.provenance verify --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.provenance readiness --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.confirmatory_lock build --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.confirmatory_lock verify --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.run_confirmatory --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.analyze_confirmatory build --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.analyze_confirmatory verify --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.canonical_v2 publish --help
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m uav_mec.canonical_v2 verify --help
```

Expected: every command exits zero after rendering help and creates no file.

- [ ] **Step 4: Re-run parser-drift and path-safety smoke tests**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python -m pytest \
  tests/test_seed_access.py \
  tests/test_power_pilot.py \
  tests/test_development_calibration.py \
  tests/test_runtime_budget.py \
  tests/test_geolife_lock.py \
  tests/test_provenance.py \
  tests/test_confirmatory_lock.py \
  tests/test_confirmatory_analysis.py \
  tests/test_canonical_v2.py \
  tests/test_project_metadata.py \
  -k 'cli or makefile or make_target or path or override or readiness' -q
```

Expected: PASS, including rejection of legacy `--output` formal roots, formal worker/shard overrides, `--all`, analyzer bootstrap overrides, external rendered assets, prefix escapes, symlinks, and pre-existing nonresumable destinations.

- [ ] **Step 5: Exercise the safe Make gate only**

```bash
make evidence-test
```

Expected: PASS. Do not invoke `evidence-smoke`, development calibration, pilot, trace-lock, runtime-budget, source-snapshot, confirmatory-lock, formal, analysis-build, or canonical-publish targets in this task.

- [ ] **Step 6: Verify the source handoff is clean**

```bash
git diff --check
git status --short
git diff --cached --name-only
```

Expected: no whitespace errors, no generated artifact staged, and no uncommitted implementation file. Task 12 performs no edit and therefore has no `git add` command; any path shown here must be resolved with the explicit allowlist of the task that owns it.

- [ ] **Step 7: Record the interface-only handoff to Plan 4**

The handoff records the implementation commit SHA, test count/duration, fixed interpreter, registry hash, execution-policy hash, all CLI help commands above, and the single authority statement: Plan 4 owns readiness audit, GeoLife acquisition/lock, development calibration execution, final pilot, budget/source/environment/confirmatory lock creation, all formal family runs, analysis build/verify, and canonical publish/verify. No pilot or formal command is executed or repeated here.

## Final plan self-review checklist

- [ ] Every approved seed pool, family/case ID, role, H1--H4 comparison, MDE, H2 primary grid, sample-size rule, and formal execution order has exactly one registry source.
- [ ] Every actual trace generation, run, and result analysis requires a recorder; config integer inspection does not record.
- [ ] Primary inference is paired t, Wilcoxon is sensitivity, and exactly four primary p-values enter Holm.
- [ ] Power resamples the four pilot vectors independently and freezes a common candidate size.
- [ ] Resource compares policies on immutable preclosure snapshots from DD, neutral, and Deadline-First generators.
- [ ] Exact CPU and scheduling nontriviality remain separate fields on the Plan 2 `ExactScheduleCertificate`; Exact Admission remains a distinct fixed-radio result type; both 30 percent claim gates are explicit.
- [ ] Generic shards support standard, Resource, Exact, and Trace schemas with deterministic conflict-rejecting merge.
- [ ] Runtime is descriptive and the preregistration budget covers every family/case, CPU, wall, memory, disk, logs, source-controlled execution-policy concurrency/shards, and retry reserve.
- [ ] GeoLife train, calibration, and final participants are split-disjoint; calibration reads only calibration and formal seeds cannot alter final trace membership or placement.
- [ ] The confirmatory lock contains source/patch, registry, config, analysis, pilot, power, budget, trace, environment, command, and ledger hashes.
- [ ] Acceptance separates publication-blocking integrity from correctness, H1--H4, Exact/mechanism, efficiency, and real-time claims.
- [ ] Analyzer build is lock-driven, analyzer verify is read-only, and neither accepts bootstrap or registry overrides.
- [ ] Analyzer detail is named only `statistical_claims.csv`; canonical publication invokes the Plan 5 pure bilingual asset generator before closure and accepts exactly one twelve-column `claim_matrix.csv` shared by both manuscripts; both tables’ evidence paths resolve, full contains all raw/generated bytes locally, light derives only from full, and negative results remain publishable when integrity passes.
- [ ] Every v2 bundle uses `MANIFEST.json` and `SHA256SUMS`; lowercase closure names occur only in v1 compatibility code.
- [ ] Every test command uses the fixed interpreter, every implementation test stays in development seeds, and every commit stages an explicit path list.
