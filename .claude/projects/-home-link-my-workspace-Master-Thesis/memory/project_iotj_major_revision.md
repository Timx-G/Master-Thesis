---
name: IoTJ Major Revision Status (2026-04-13 Full Pass Completed)
description: Full record of 2026-04-13 5-reviewer comprehensive fix session; all P1/P2/P3 issues addressed
type: project
---

Paper: "Interference-Aware User Pairing and Task Offloading for Grouped NOMA Multi-UAV MEC Networks"
Target: IEEE IoTJ (CCF-B)
Full revision completed: 2026-04-13
Paper: /home/link/my_workspace/Master-Thesis/paper/main_v2.tex (PDF compiled clean)

**Why:** 5 virtual reviewers all gave Major Revision. P1 code bugs invalidated statistical tests; P2 weakened NOMA claims.

## CODE FIXES (uav_mec/)

### PRNG Leakage (M3.2) ✅
- `multi_uav_env.py`: Added `self.strategy_rng = np.random.default_rng(_seed + 999983)` in reset()
- `run_main_experiment.py`: `strategy_random` now uses `env.strategy_rng` (NOT env.rng)
- Impact: Random baseline runs on same mobility/traffic as all other baselines

### ExchStable cross_interf=0 Bug (M3.3) ✅
- `baselines_pairing.py` `exchange_stable_pairing()`: Added `cross_interf_for_sub(sub_idx, k)` helper
- ExchStable result: 67.7% → **69.2%** (no longer equal to ILP-Optimal 67.7%)

### Orthogonal-Band OFDMA Baseline (M5.3) ✅
- `baselines_pairing.py`: Added `ofdma_ortho_rates()` (each UAV gets B/K bandwidth, zero cross-UAV interference)
- `run_main_experiment.py`: Added `strategy_ofdma()`, `strategy_ofdma_ortho()`, `_evaluate_oma()` helper
- `run_episode()` now accepts strategy returning 5-tuple for pre-computed results

### ILP+BlindSwap Ablation Restored ✅
- `pairing.py` `swap_optimize()`: Added `blind_cross_uav=True` parameter
- `run_main_experiment.py`: Added `strategy_ilp_blind_swap`

## UPDATED RESULTS (python -m uav_mec.run_main_experiment)

| Method | Result | vs Ours |
|--------|--------|---------|
| Ours(ILP+Swap) | **73.2±1.9%** | — |
| BG+Swap | 72.5±2.0% | -0.7*** |
| Greedy+Swap | 71.6±2.2% | -1.6*** |
| WorstCase+Swap | 71.5±2.5% | -1.8*** |
| IntraSwapOnly | 71.1±2.1% | -2.1*** |
| ILP+BlindSwap | 70.5±2.0% | -2.7*** |
| ExchangeStable | **69.2±1.9%** | -4.1*** |
| ILP-Optimal | 67.7±2.2% | -5.6*** |
| BenefitGreedy | 67.0±2.2% | -6.2*** |
| OFDMA(OrthoBank) | **65.5±2.6%** | -7.7*** |
| OFDMA(FullReuse) | **64.1±1.8%** | -9.1*** |
| Random | **63.8±2.1%** | -9.5*** |
| Greedy | 62.1±2.3% | -11.1*** |

## PAPER CHANGES (main_v2.tex — all compiled clean)

1. Abstract: "12 baselines, 4.1pp, 7.7pp OrthoOFDMA / 9.1pp FullReuse" (was: 10 baselines, 5.5pp, 23.6pp)
2. Lemma 1: "**Conditionally** Optimal Offloading Ratio" (was: "Optimal")
3. Table I: Rebuilt with Jain column; PPO/MAPPO removed; correct new numbers throughout
4. ExchStable description: Removed false "67.7%=ILP" claim; new accurate text
5. NOMA vs OFDMA: Two reference points (FullReuse +9.1pp, OrthoBank +7.7pp)
6. DRL section: `\label{sec:results_drl}`, strong disclaimer about DRL+Greedy hybrid
7. Remark 2: Removed invalid empirical-vs-theoretical approximation comparison
8. Coordination overhead paragraph: Added inter-UAV CSI sharing latency (~0.38ms)
9. +2.1pp discussion: Added justification for cross-UAV swap value at scale
10. Path loss clarification: 5dB UAV-level vs 10.9dB pair-level are different dimensions
11. SIC + block fading: Note on SIC ordering stability under large-scale fading
12. Conclusion: All stats updated; Jain fairness finding added
13. Scalability/K tables: N=20 and K=2 ExchStable rows updated to 69.2
14. Overhead/Energy tables: All affected numbers updated
