# Code Architecture — UAV NOMA MEC Paper

> 本文件用于辅助 agent 在阅读 `paper/main_v2.tex` 某一章节后，**快速定位对应代码**。
> 项目根目录：`/home/link/my_workspace/Master-Thesis/`

---

## 一、论文章节 → 代码文件 速查表

| 论文章节 | 对应代码文件 | 关键函数 / 类 |
|----------|-------------|--------------|
| **§III-A 网络模型** (UAV 位置、信道、用户移动) | `uav_mec/model.py` | `ScenarioConfig`, `NOMAChannel`, `UAVModel` |
| **§III-B NOMA 信道模型** (SINR公式 eq.3–5, 交叉干扰 eq.6) | `uav_mec/model.py` + `uav_mec/pairing.py` | `NOMAChannel.path_loss()`, `compute_rates_for_pairing()` |
| **§III-B SIC 可行性约束** (eq.4) | `uav_mec/baselines_pairing.py` | `ilp_optimal_pairing()` 内 `sic_gamma_min` 过滤逻辑 (line ~102–115) |
| **§III-C 任务模型与卸载** (时延公式 eq.7) | `uav_mec/pairing.py` | `evaluate_pairing()` 内 delay 计算 (line ~166–172) |
| **§III-C 最优卸载比** (eq.8, Lemma 1) | `uav_mec/pairing.py` | `kkt_alpha()` (line 27–55) |
| **§III-D 问题公式化** (eq.9 目标函数) | `uav_mec/run_main_experiment.py` | `run_episode()` → `evaluate_pairing()` 返回 `deadline_rate` |
| **§IV-A Stage 1: ILP 初始化** | `uav_mec/baselines_pairing.py` | `ilp_optimal_pairing()` (line 69–190) |
| **§IV-B Stage 2: Cross-UAV Swap** (Swap 1/2/3, Prop.2) | `uav_mec/pairing.py` | `swap_optimize()` (line 316–484) |
| **§IV-B Swap 1 Replace** | `uav_mec/pairing.py` | `swap_optimize()` line ~358–397 |
| **§IV-B Swap 2 Position** | `uav_mec/pairing.py` | `swap_optimize()` line ~399–435 |
| **§IV-B Swap 3 CrossUAV** | `uav_mec/pairing.py` | `swap_optimize()` line ~437–482 |
| **§IV-C Stage 3: 闭式卸载比** | `uav_mec/pairing.py` | `kkt_alpha()` |
| **§V 主实验 (Table V, 15 baselines)** | `uav_mec/run_main_experiment.py` | `main()`, `methods` dict (line ~448–465) |
| **§V 2×2 消融 (Table VI)** | `uav_mec/run_main_experiment.py` | `strategy_intra_swap_only()`, `strategy_exchange_stable_cross_uav()` |
| **§V DRL 对比 (Fig.11)** | `uav_mec/run_drl_convergence_and_mappo.py` | `main()` + `agents/ppo.py`, `agents/mappo.py` |
| **§V PPO zero-shot 评估** | `uav_mec/run_ppo_fair.py` | train seeds 100–321, test seeds 42–61 |
| **§V N 扩展性 (Table VII)** | `uav_mec/run_main_experiment.py` + `uav_mec/run_supplementary_experiments.py` | N=10–30 canonical; N=40–50 supplementary run |
| **§V K 扩展性 (Table VIII)** | `uav_mec/run_k_scalability.py` | `main()`, total MEC = 8 GHz fixed |
| **§V NF 敏感性 (Table IX)** | `uav_mec/run_nf_sic_sensitivity.py` | `run_episode_nf()`, NF=5/7/9 dB |
| **§V SIC γ_min 敏感性 (Table X)** | `uav_mec/run_nf_sic_sensitivity.py` | `strategy_ours_sic()` + `ilp_optimal_pairing(..., sic_gamma_min=gmin)` |
| **§V CSI 误差敏感性 (Table XI)** | `uav_mec/run_supplementary_experiments.py` | imperfect CSI sweep |
| **§V 小规模穷举最优 (§V-C)** | `uav_mec/run_exhaustive_and_positions.py` | N=8, S=2, exhaustive search |
| **§V 运行时统计** | `uav_mec/run_main_experiment.py` | `main()` 内 `elapsed` 计时 |
| **Appendix A NP 完备性证明** | 仅在论文中，无对应代码 | — |

---

## 二、核心文件职责

### `uav_mec/model.py`
物理层模型，**不含算法逻辑**。
- `ScenarioConfig` — 所有实验参数（N, K, S, t_max, f_per_uav 等）
- `NOMAChannel` — 路径损耗 `path_loss(d)`、噪声功率
- `TaskGenerator` — 随机突发任务生成
- `UAVModel` — 无人机悬停位置（`UAV_HOVER_POSITIONS`）

### `uav_mec/pairing.py`
**核心算法文件**（Stage 2 + Stage 3）。
- `kkt_alpha()` — 最优卸载比（crossing-point 公式，非标准 KKT）
- `compute_rates_for_pairing()` — 给定配对计算所有用户 SINR/速率，含交叉干扰
- `evaluate_pairing()` — 给定配对返回 (met, total, rates, alphas, served)
- `greedy_pairing()` — Greedy 基线
- `benefit_greedy_pairing()` — BenefitGreedy 基线
- `swap_optimize()` — Stage 2 全部三种 swap，`cross_uav=False` 开关控制 CrossUAV

### `uav_mec/baselines_pairing.py`
**Stage 1（ILP 初始化）+ 所有其他基线**。
- `ilp_optimal_pairing(..., sic_gamma_min=1.0)` — per-UAV ILP，SIC 阈值可配置
- `exchange_stable_pairing()` — ExchangeStable（双边准则）
- `exchange_stable_cross_uav_pairing()` — ExStable+CrossUAV
- `worst_case_pairing()` — WorstCase+Swap 初始化
- `ia_greedy_pairing()` — IA-Greedy（干扰感知贪心，无 swap）
- `ofdma_rates()` / `ofdma_ortho_rates()` / `ofdma_fair_rates()` — OFDMA 基线

### `uav_mec/run_main_experiment.py`
**主实验入口**（20 seeds × 16 方法）。
- `run_episode(cfg, seed, strategy_fn)` — 单 episode 运行逻辑（100 slot）
- `strategy_ours()` — Ours = ILP 初始化 + swap_optimize
- `methods` dict (line ~448) — 所有 16 方法注册表
- `main()` — Wilcoxon + Holm-Bonferroni 统计检验

### `uav_mec/multi_uav_env.py`
仿真环境（任务队列、用户移动、突发模式）。由 `run_episode()` 调用，一般不需要直接修改。

---

## 三、实验结果 → JSON 文件 映射

| 论文数据来源 | JSON 文件 | 说明 |
|-------------|-----------|------|
| Table V（主结果）| `results/phase2_main_experiment.json` | **权威来源**，20 seeds × 16 methods |
| Table VII（N 扩展）| `results/phase2_supplementary.json` / `results/supplementary_experiments.json` | N=10–50 |
| Table VIII（K 扩展）| `results/k_scalability_full.json` | K=1–5 |
| Table IX/X（NF/SIC 敏感性）| `results/nf_sic_sensitivity.json` | NF=5/7/9 dB, γ_min=0/3/5 dB |
| Fig. 11（DRL 收敛）| `results/drl_convergence_mappo.json` | PPO+MAPPO 2000 episode |
| PPO zero-shot（§V-D）| `results/ppo_fair_comparison.json` | train 100–321 / test 42–61 |
| 穷举最优（§V-C）| `results/phase2_table2_full.json` / `results/reviewer_extra_experiments.json` | N=8, S=2 |

> **数字权威优先级：JSON > 代码 > 论文**。如发现不一致，以 JSON 为准修改论文。

---

## 四、图表 → 生成脚本

所有图表由 `uav_mec/generate_figures_v2.py` 生成，输出至 `figures/`。

| 论文图 | 文件名 | 生成函数 |
|--------|--------|----------|
| Fig.1 系统模型 | `fig1_system_model.pdf` | 手工绘制（不由脚本生成）|
| Fig.2 算法流程 | `fig2_algorithm_flowchart.pdf` | 手工绘制 |
| Fig.3 主对比 | `fig3_main_comparison.pdf` | `generate_figures_v2.py` |
| Fig.4 消融 2×2 | `fig4_ablation.pdf` | `generate_figures_v2.py` |
| Fig.5 N 扩展性 | `fig5_scalability.pdf` | `generate_figures_v2.py` |
| Fig.6 t_max 敏感性 | `fig6_tmax_sensitivity.pdf` | `generate_figures_v2.py` |
| Fig.8 时延 CDF | `fig8_delay_cdf.pdf` | `generate_figures_v2.py` |
| Fig.9 Swap 可视化 | `fig9_pairing_visualization.pdf` | `generate_figures_v2.py` |
| Fig.11 DRL 收敛 | `fig_drl_convergence_comparison.pdf` | `run_drl_convergence_and_mappo.py` |

---

## 五、关键设计决策（影响代码-论文一致性）

1. **`kkt_alpha` 非标准 KKT**：函数名含 "kkt" 但注释明确说明这是 `max(τ_loc, τ_off)` 的 crossing-point 解，不是标准 KKT（因为 max 函数不可微）。论文 Lemma 1 已正确标注此限定。

2. **ILP 在 Stage 1 忽略交叉干扰**：`ilp_optimal_pairing()` 的 benefit 估计使用 `sinr_s = recv_s / (recv_w + noise_sub)`（无 `I_cross`）。交叉干扰的影响由 Stage 2 swap 纠正。

3. **SIC 阈值只在 Stage 1 过滤**：`sic_gamma_min` 参数控制 ILP 初始化时的候选对过滤。Swap（Stage 2）不含显式 SIC 过滤，靠全 SINR 模型的低速率隐式排除不可行配对。

4. **Greedy < Random 的物理原因**：`greedy_pairing()` 按信道增益排序 → 高功率用户集中在相同子载波 → 跨 UAV 干扰加剧。在干扰受限场景（$I_\text{cross} \gg \sigma^2$）下效果反不如随机。

5. **strategy_random 使用独立 PRNG**：`env.strategy_rng`（非 `env.rng`），保证不同方法的信道/任务随机流完全相同，比较公平。
