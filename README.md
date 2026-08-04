# Full-Reuse Multi-UAV NOMA-MEC Thesis Project

## Research Question

This professional-master project studies slot-level scheduling in a full-frequency-reuse multi-UAV grouped-NOMA MEC network. The controller must maximize on-time task completions while preserving unique-task semantics, actual serving-UAV channels, hard SIC feasibility, and per-UAV CPU budgets.

## Two Contributions

1. **Interference-externality-aware cross-UAV association and pairing.** A feasible per-UAV MILP initializer is refined through Replace, Position, and CrossUAV neighborhoods under deterministic global first improvement.
2. **Deadline-driven offloading and MEC allocation.** Minimum offload ratios and CPU requirements are derived from each task deadline; a sorted feasible prefix maximizes admitted-task cardinality per UAV.

The confirmatory full method reaches `\FullOnTimePercent`; generated values live in `paper/generated_results.tex`. Innovation 1 is a small but statistically stable improvement. Innovation 2 has a larger engineering effect.

## Trusted Data Flow

```text
TOML definition -> immutable trace -> scheduler/baseline -> hard evaluator
-> atomic raw bundle + SHA-256 -> invariant/statistical analysis
-> passed evidence gates -> canonical JSON -> figures and LaTeX
```

Every method receives identical task IDs and channels for a seed. Raw confirmatory outputs are ignored. The tracked canonical-v1 bundle verifies only its historical snapshot and cannot validate thesis-v2.

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

## Thesis-v2 Semantic Gate

Run this non-experimental gate before any v2 smoke, pilot, confirmatory, or publication command:

```bash
make semantic-gate PYTHON=/home/link/miniconda/envs/llm-ops-torch26/bin/python
```

A canonical-v1 checksum check cannot validate thesis-v2 semantics.

## Algorithm Claim Gate

The static algorithm checks are implementation-correctness gates only. They do
not establish an empirical effect, exact-gap distribution, runtime claim,
thesis conclusion, or submission result. Pilot and formal execution belong to later approved plans.

## Quick Verification

```bash
python -m pytest -q

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

## Full Confirmatory Reproduction

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
python -m uav_mec.analyze_results --main artifacts/runs/confirmatory-v2/main --pairing artifacts/runs/confirmatory-v2/pairing --resource artifacts/runs/confirmatory-v2/resource --exact artifacts/runs/confirmatory-v2/exact --output artifacts/runs/confirmatory-v2/analysis --require-thesis-gates
```

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
