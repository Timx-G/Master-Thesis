# Quantitative Claims Matrix

| Claim ID | Active claim | Generated macro/figure | Canonical source | Statistical gate | Scope limitation |
|---|---|---|---|---|---|
| C-MAIN | Full on-time rate is 98.11% | `\FullOnTimePercent`, `main_on_time.pdf` | `runs/main/summary.json` | Descriptive bootstrap CI | N=20, K=2 frozen scenario |
| C-I1-X | CrossUAV neighborhood adds 0.05 pp | `\InnovationOneCrossGainPP` | `analysis/statistics.json:innovation1_cross` | 95% CI lower bound above zero | Small absolute effect in a saturated regime |
| C-I1-A | Global acceptance adds 0.12 pp | `\InnovationOneAcceptanceGainPP` | `analysis/statistics.json:innovation1_acceptance` | 95% CI lower bound above zero | Centralized network utility is required |
| C-I2 | Deadline/min-CPU adds 8.69 pp | `\InnovationTwoGainPP`, `resource_ablation.pdf` | `analysis/statistics.json:innovation2_deadline` | 95% CI lower bound above zero | Unit task value and per-UAV hard CPU budget |
| C-FIXED | Deadline/min-CPU adds 21.97 pp over fixed-half/equal | generated statistics table | `analysis/statistics.json:resource_fixed_half` | Paired CI and Holm p-value | Fixed alpha is an intentionally weak control |
| C-GREEDY | Full exceeds greedy by 3.24 pp | generated statistics table | `analysis/statistics.json:baseline_greedy` | Paired CI and Holm p-value | Same immutable tasks and channels |
| C-IA | Full exceeds IA-greedy by 0.49 pp | generated statistics table | `analysis/statistics.json:baseline_ia_greedy` | Paired CI and Holm p-value | Same evaluator and resource policy |
| C-ILP | Full exceeds per-UAV ILP by 1.30 pp | generated statistics table | `analysis/statistics.json:baseline_ilp` | Paired CI and Holm p-value | Isolates search refinement after initialization |
| C-EX | Full exceeds exchange-stable by 0.19 pp | `\StrongBaselineGainPP` | `analysis/statistics.json:baseline_exchange` | Predeclared strong-baseline gate | Small absolute effect; no universal dominance claim |
| C-OFDMA | Full exceeds fair OFDMA by 0.58 pp | generated statistics table | `analysis/statistics.json:baseline_ofdma` | Paired CI and Holm p-value | OFDMA uses fair rotating service, not a static subset |
| C-EXACT | Heuristic matches complete small enumeration in 100% of slots | `\ExactOptimalPercent`, `\ExactMeanCompletionGap` | `runs/exact/summary.json` | Complete-search gate | At most five users; not a general guarantee |
| C-SIC | Final SIC violation count is zero | `\SICViolationCount` | `analysis/acceptance.json` | Constraint-zero gate | Evaluated channel/SIC model only |
| C-RUNTIME | N=20 scheduling P95 is 40.93 ms | `\FullRuntimePNinetyFiveMs` | `analysis/metrics.json` | P95 below one-second gate | Python implementation on the experiment host |

All paths are relative to `artifacts/canonical/v1/`. Exploratory and legacy result directories are not evidence sources.
