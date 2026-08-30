# 面向时限任务的多无人机移动边缘计算干扰感知调度与卸载资源分配方法研究

## Research Question

This professional-master project studies slot-level scheduling in a full-frequency-reuse multi-UAV grouped-NOMA MEC network. The controller must maximize on-time task completions while preserving unique-task semantics, actual serving-UAV channels, hard SIC feasibility, and per-UAV CPU budgets.

## Two Contributions

The overall method is the **Externality-Aware and Deadline-Driven Joint Scheduling and Resource Allocation algorithm (EAD-JSRA)**:

1. **Interference-Externality-Aware Cross-UAV Association and Pairing algorithm (IEA-CAP).** A feasible per-UAV MILP initializer is refined through Replace, Position, and CrossUAV neighborhoods under deterministic global first improvement.
2. **Deadline-Driven Minimum-Cost Admission and Allocation algorithm (DD-MCAA).** Minimum offload ratios and CPU requirements are derived from each task deadline; a sorted feasible prefix maximizes admitted-task cardinality per UAV.

Confirmatory EAD-JSRA reaches `\FullOnTimePercent`; generated values live in `paper/generated_results.tex`. IEA-CAP provides a small but statistically stable improvement, while DD-MCAA has a larger engineering effect.

## Trusted Data Flow

```text
TOML definition -> immutable trace -> scheduler/baseline -> hard evaluator
-> atomic raw bundle + SHA-256 -> invariant/statistical analysis
-> passed evidence gates -> canonical JSON -> figures and LaTeX
```

Every method receives identical task IDs and channels for a seed. Raw confirmatory outputs are ignored; only checksum-audited lightweight evidence under `artifacts/canonical/v1/` is tracked.

## Repository Layout

- `uav_mec/`: trusted domain, trace, scheduler, evaluator, allocation, analysis, and publication code.
- `configs/thesis/`: frozen smoke, exploratory, confirmatory, ablation, and exact definitions.
- `artifacts/canonical/v1/`: tracked summaries, manifests, statistics, acceptance gates, and checksums.
- `figures/thesis/`: figures generated exclusively from canonical evidence.
- `paper/`: active IEEE-style article and generated result tables.
- `thesis/`: eight-chapter professional-master content manuscript and claim matrix.
- `defense/`: 20-slide outline, two-contribution comparison, and risk questions.
- `deliverables/defense_2026/`: proposal report, midterm report, and midterm defense deck.
- `tests/`: behavioral, invariant, statistical, evidence-contract, and document tests.

## Quick Verification

```bash
python -m pytest -q

make smoke
make exact-smoke
make sharded-smoke

python -m uav_mec.experiment \
  --config configs/thesis/smoke.toml \
  --output /tmp/uav-mec-smoke-$(date +%s)

python -c "from pathlib import Path; from uav_mec.analyze_results import verify_checksums; verify_checksums(Path('artifacts/canonical/v1')); print('canonical checksums: OK')"

python -m uav_mec.generate_figures_v2 \
  --canonical artifacts/canonical/v1 \
  --figures figures/thesis \
  --macros paper/generated_results.tex \
  --statistics-table paper/generated_statistics.tex \
  --robustness-table paper/generated_robustness.tex

tectonic -X compile paper/main_v2.tex --outdir build/paper --reruns 2
tectonic -X compile thesis/main.tex --outdir build/thesis --reruns 2
```

## EAD-JSRA Confirmatory Reproduction

Outputs are fail-fast and non-overwriting. Use a new versioned directory when rerunning.

```bash
python -m uav_mec.experiment --config configs/thesis/exploratory.toml --output artifacts/exploratory/exploratory-v2
python -m uav_mec.experiment --config configs/thesis/confirmatory_main.toml --output artifacts/runs/confirmatory-v2/main
python -m uav_mec.experiment --config configs/thesis/confirmatory_pairing.toml --output artifacts/runs/confirmatory-v2/pairing
python -m uav_mec.experiment --config configs/thesis/confirmatory_resource.toml --output artifacts/runs/confirmatory-v2/resource
python -m uav_mec.run_exact_benchmark --config configs/thesis/confirmatory_exact.toml --output artifacts/runs/confirmatory-v2/exact
python -m uav_mec.run_thesis_suite --suite mechanism --output artifacts/runs/confirmatory-v2/mechanism
python -m uav_mec.run_thesis_suite --suite scalability --output artifacts/runs/confirmatory-v2/scalability
python -m uav_mec.run_thesis_suite --suite robustness --output artifacts/runs/confirmatory-v2/robustness
python -m uav_mec.experiment --config configs/thesis/confirmatory_main_v2.toml --output artifacts/runs/confirmatory-v2/main-reality-calibrated
python -m uav_mec.run_thesis_suite --suite capacity --output artifacts/runs/confirmatory-v2/capacity --workers 1
python -m uav_mec.analyze_results --main artifacts/runs/confirmatory-v2/main --pairing artifacts/runs/confirmatory-v2/pairing --resource artifacts/runs/confirmatory-v2/resource --exact artifacts/runs/confirmatory-v2/exact --output artifacts/runs/confirmatory-v2/analysis --require-thesis-gates
```

Before a full capacity run, execute the same frozen cases with bounded seeds and slots. The
resolved manifests record the shortened protocol, while the source configuration remains fixed at
50 seeds and 100 slots.

```bash
python -m uav_mec.run_thesis_suite \
  --suite capacity \
  --output artifacts/exploratory/capacity-dry-run \
  --workers 1 \
  --seed-count 2 \
  --development-seed-start 42 \
  --num-slots 3
```

The reality-calibrated main case uses `K=3`, `N=45`, three subcarriers, and two-user NOMA.
The capacity suite scans `N={30,45,60}` at `K=3` and preserves 15 registered users per UAV for
`(K,N)={(2,30),(3,45),(4,60),(5,75)}`. Its checksummed analysis is written to
`analysis/capacity_metrics.json` inside the suite output.

The registered v2 protocols use three deterministic multi-start worker processes inside each
slot. Keep suite-level workers at one for runtime audits so outer case parallelism does not distort
the reported online P50/P95. Long confirmatory definitions can be resumed by seed with
`python -m uav_mec.run_shards --config <config> --output <new-dir> --shard-size 1`.

Publish only after all gates pass, using `python -m uav_mec.publish_canonical --help` and a new canonical version.

## Canonical Evidence And Paper Assets

- Gate decision: `artifacts/canonical/v1/analysis/acceptance.json`
- Paired inference: `artifacts/canonical/v1/analysis/statistics.json`
- Engineering metrics: `artifacts/canonical/v1/analysis/metrics.json`
- Evidence map: `artifacts/canonical/v1/index.json`
- Claim provenance: `thesis/claims-matrix.md`

No active manuscript reads legacy `results/`, checkpoints, or exploratory artifacts.

## Thesis And Paper Builds

```bash
make paper
make thesis
make verify
```

Institution-specific cover, authorization, and formatting pages may wrap `thesis/main.tex` without changing its evidence contract.

## Legacy Scope

`results/`, old root-level figures, old PDFs, PPO/MAPPO runners, agents, and checkpoints are retained only for research history. They are not trusted thesis evidence and must not be quoted as active results.

## Known Limitations

- Centralized CSI and inter-UAV coordination are assumed.
- UAV trajectory, power, battery, and long-horizon queues are outside the active optimization.
- Monte Carlo simulation provides the primary evidence; no field-deployment claim is made.
- Innovation 1 is statistically reliable but small in absolute magnitude, is not monotone with interference coupling, and is not strictly dominant at every UAV count.
