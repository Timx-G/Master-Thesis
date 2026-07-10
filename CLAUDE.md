# CLAUDE.md — Multi-UAV Grouped NOMA MEC: User Pairing Optimization

> **Last updated**: 2026-04-15
> **Author**: Mingxi Guan
> **Target**: IEEE Internet of Things Journal (CCF-B)

---

## 0. Environment Setup (IMPORTANT)

### Python 环境

**只有一个环境有所有依赖，始终用这个：**

```bash
/home/link/miniconda/envs/llm-ops-torch26/bin/python

# 或激活：
conda activate llm-ops-torch26

# 验证：
python -c "import numpy, scipy, torch, gymnasium, matplotlib; print('All OK')"
```

| Package | Version |
|---------|---------|
| Python | 3.11.13 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| torch | 2.1.0+cu121 |
| gymnasium | 1.2.3 |
| matplotlib | 3.10.8 |

**其他环境缺包：** `llm-ops` 缺 scipy/gymnasium/matplotlib；`base` 缺全部。

### 代理问题

系统配置了 `http_proxy=http://172.25.96.1:7897`，可能不通。安装包时绕过：

```bash
env -u http_proxy -u https_proxy pip install <package>
```

### LaTeX 编译

```bash
cd ~/my_workspace/Master-Thesis/paper
env -u http_proxy -u https_proxy /home/link/.local/bin/tectonic -o . main_v2.tex
# 输出：paper/main_v2.pdf
```

---

## 1. 项目概述

**论文标题**：*Interference-Aware User Pairing and Task Offloading for Grouped NOMA Multi-UAV MEC Networks*

**核心方法**：Cross-UAV Swap + per-UAV ILP 初始化 + 闭式 crossing-point 最优卸载比（非标准 KKT，见下）

**核心结论**：Ours = **73.4%** deadline 满足率，**+11.3 pp** vs Greedy（$p < 0.001$，$d = 6.20$，20 seeds，15 基线）

### 研究演化（历史路径，勿重复）

| Phase | 方向 | 结论 | 状态 |
|-------|------|------|------|
| A | Single-UAV DC-PPO 轨迹 | Fixed-λ 优于 DRL | Archived |
| B | Multi-UAV DRL 轨迹 | Hover 优于 DRL（6轮验证） | Archived |
| C | DRL 分组（6维动作） | PPO ≈ Greedy（0pp 差距） | Archived |
| **D（当前）** | **ILP 初始化 + Cross-UAV Swap** | **+11.3pp，d=6.20** | **Active** |

---

## 2. 代码架构

```
Master-Thesis/
├── CLAUDE.md                        # 本文件（项目总览）
├── uav_mec/
│   ├── CLAUDE.md                    # 论文章节 → 代码文件速查（给 agent 用）
│   ├── model.py                     # 物理层：信道、UAV、任务生成器、ScenarioConfig
│   ├── multi_uav_env.py             # 仿真环境（任务队列、用户移动、突发模式）
│   ├── pairing.py                   # 核心算法：kkt_alpha / swap_optimize / evaluate_pairing
│   ├── baselines_pairing.py         # 全部基线：ILP / ExStable / OFDMA / IA-Greedy 等
│   ├── run_main_experiment.py       # 主实验（20 seeds × 16 methods，含统计检验）
│   ├── run_nf_sic_sensitivity.py    # NF / γ_SIC_min 敏感性实验
│   ├── run_k_scalability.py         # K 扩展性实验
│   ├── run_supplementary_experiments.py  # N 扩展、CSI误差、t_max、子载波敏感性
│   ├── run_exhaustive_and_positions.py   # 小规模穷举最优 + UAV 位置鲁棒性
│   ├── run_drl_convergence_and_mappo.py  # DRL 训练收敛曲线
│   ├── run_ppo_fair.py              # PPO zero-shot 评估（train/test split）
│   ├── run_round2_experiments.py    # PPO 第二轮规范评估
│   ├── generate_figures_v2.py       # 所有图表生成（Fig.3–9，Fig.11）
│   └── agents/
│       ├── ppo.py                   # PPO agent（论文 DRL 基线）
│       └── mappo.py                 # MAPPO agent（论文 DRL 基线）
├── paper/
│   ├── main_v2.tex                  # 当前论文（IEEE IoTJ 格式）
│   ├── refs_v2.bib                  # 参考文献（~42 条）
│   └── main_v2.pdf                  # 编译输出
├── figures/                         # 生成的图（PDF + PNG）
├── results/                         # 实验数据（JSON，见 §5）
├── docs/                            # 研究笔记、导师反馈记录
└── archive/
    ├── phase_a_scripts/             # Phase A：DC-PPO 实验
    ├── phase_b_drl_scripts/         # Phase B：DRL 轨迹/分组
    └── old_scripts/                 # 清理的冗余脚本（非活跃）
```

---

## 3. 核心参数（锁定，勿随意更改）

| 参数 | 值 | 来源 |
|------|----|------|
| 区域 | 800×800 m | — |
| UAV 高度 | 100 m | DJI M300 |
| UAV 数量 K | 2（扩展 K=1–5） | — |
| 用户数 N | 20 | — |
| 子载波数 S | 3（×1.67 MHz） | Narrowband IoT |
| NOMA 对大小 | 2 | 工程实践 |
| 载波频率 | 2 GHz | LTE Band 1 |
| 带宽 | 5 MHz | LTE |
| 噪声 PSD | −174 dBm/Hz | 热噪声 |
| 接收机 NF | 7 dB（默认） | UAV 接收机 |
| 噪声/子载波 | −104.8 dBm | NF=7 dB 下 |
| 发射功率 | 0.2 W (23 dBm) | 3GPP Class 3 |
| SIC 残差 ε | 0.02 | 文献 |
| MEC 算力/UAV | 4 GHz（K=2 时） | 嵌入式 MEC 服务器 |
| 本地 CPU | 1 GHz | IoT MCU |
| 截止时间 t_max | 0.8 s | 软实时校准值 |
| 到达概率 | 0.8（burst cluster） | 密集 IoT |
| 用户速度 | 2 m/s | ITU 行人 |
| 结果数据比 ρ | 0.1 | 文献 |

---

## 4. 核心实验结果（权威数字，以 JSON 为准）

### 主实验（K=2, N=20, 20 seeds）

> 源文件：`results/phase2_main_experiment.json`

| 方法 | Deadline% ±95%CI | vs Ours |
|------|-----------------|---------|
| **Ours (ILP+Swap)** | **73.4 ±1.0** | — |
| BG+Swap | 72.4 ±1.0 | −1.0*** |
| IA-Greedy+Swap | 71.9 ±1.0 | −1.4*** |
| Greedy+Swap | 71.6 ±1.1 | −1.8*** |
| WorstCase+Swap | 71.2 ±1.2 | −2.1*** |
| IntraSwapOnly | 70.8 ±1.0 | −2.6*** |
| ILP+BlindSwap | 70.4 ±0.9 | −3.0*** |
| ExStable+CrossUAV | 69.6 ±0.8 | −3.8*** |
| ExchangeStable | 69.1 ±0.9 | −4.2*** |
| IA-Greedy | 68.5 ±0.9 | −4.9*** |
| Per-UAV ILP | 67.7 ±1.1 | −5.7*** |
| BenefitGreedy | 67.0 ±1.1 | −6.4*** |
| OFDMA(OrthoBank) | 65.5 ±1.2 | −7.9*** |
| OFDMA(FullReuse) | 64.1 ±0.9 | −9.3*** |
| Random | 63.7 ±1.0 | −9.7*** |
| Greedy | 62.1 ±1.1 | −11.3*** |

### 扩展性

- **N 扩展**（K=2）：gap vs Greedy = +5.9 pp (N=10) → +19.4 pp (N=50)，单调递增
- **K 扩展**（总算力固定 8 GHz）：K=1→73.4%→76.8%（K=4），K=3 时 71.5%（中间低谷）
- **弛豫上界**：93.2%；Ours 达到上界的 78.8%
- **小规模穷举**（N=8, S=2）：Ours = 94.8% of exact optimum

### 敏感性结论

| 扰动 | Ours 变化 | Gap vs Greedy 变化 |
|------|-----------|-------------------|
| NF: 5→9 dB | ≤0.1 pp | 稳定 +11.3 pp |
| γ_SIC,min: 0→5 dB | ≤0.3 pp (73.2–73.5%) | 11.1–11.4 pp |
| SIC 残差 ε: 0→0.10 | 75.5→71.4% | +11.7→+12.3 pp（gap 反而扩大）|
| CSI 误差 σ_e: 0→0.20 | 73.4→72.5% | +11.3→+10.3 pp |

---

## 5. 实验结果 JSON 映射

> **数字权威优先级：JSON > 代码运行输出 > 论文文本**

| JSON 文件 | 对应论文内容 |
|-----------|-------------|
| `results/phase2_main_experiment.json` | **Table V 主结果**（16方法×20seeds，含统计） |
| `results/nf_sic_sensitivity.json` | Table IX (NF) + Table X (γ_SIC_min) |
| `results/k_scalability_full.json` | Table VIII (K扩展) |
| `results/phase2_supplementary.json` | Table VII (N扩展，N=10–30) |
| `results/supplementary_experiments.json` | N=40–50, CSI误差, t_max, 子载波 |
| `results/drl_convergence_mappo.json` | Fig.11 DRL 收敛曲线 |
| `results/ppo_fair_comparison.json` | §V-D PPO zero-shot 评估 |
| `results/phase2_table2_full.json` | §V-C 穷举最优（N=8） |

---

## 6. 论文章节 → 代码文件速查

详细映射见 `uav_mec/CLAUDE.md`。快速索引：

| 论文内容 | 代码文件 | 关键函数 |
|----------|---------|---------|
| SINR 公式 (eq.3–6) | `pairing.py` | `compute_rates_for_pairing()` |
| 最优卸载比 (eq.8, Lemma 1) | `pairing.py` | `kkt_alpha()` |
| SIC 可行性过滤 | `baselines_pairing.py` | `ilp_optimal_pairing(..., sic_gamma_min)` |
| Stage 1 ILP | `baselines_pairing.py` | `ilp_optimal_pairing()` |
| Stage 2 Swap | `pairing.py` | `swap_optimize()` |
| 全部基线实现 | `baselines_pairing.py` | 见文件顶部 docstring |
| 主实验 + 统计检验 | `run_main_experiment.py` | `main()` |
| DRL 基线 | `run_drl_convergence_and_mappo.py` | `agents/ppo.py`, `agents/mappo.py` |

---

## 7. 关键设计陷阱（勿犯）

1. **`kkt_alpha` 不是标准 KKT**：`τ = max(τ_loc, τ_off)` 不可微，用的是 crossing-point 条件。论文 Lemma 1 已注明，但函数名保留历史命名，**不要在论文中写"standard KKT"**。

2. **ILP 初始化忽略交叉干扰**：`ilp_optimal_pairing()` 的 benefit 估计用 SINR = recv_s / (recv_w + noise)，**无 I_cross**。交叉干扰由 Stage 2 Swap 纠正。这不是 bug，是设计。

3. **SIC 阈值只过滤 Stage 1**：`sic_gamma_min` 参数仅控制 ILP 候选对的过滤，Swap（Stage 2）没有显式 SIC 过滤——靠低 SINR → 低速率 → 低 deadline count 隐式排除。

4. **Greedy < Random**：Greedy 按信道增益排序，使高功率用户聚集于同一子载波，在干扰受限场景下反而放大交叉干扰。这是已验证的物理现象，**不是代码 bug**。

5. **strategy_random 用独立 PRNG**：`env.strategy_rng` 而非 `env.rng`，确保所有方法的信道/任务随机流完全相同。Random 的随机性只影响配对选择，不影响环境。

6. **UAV 轨迹无价值**（已验证 6 轮）：在 100 m LoS MEC 场景，轨迹优化对 deadline rate 的边际贡献可忽略。**不要重新引入轨迹优化**。

7. **DRL 对离散配对无效**（已验证）：PPO/MAPPO 的连续动作空间无法直接表示离散 per-user 配对。**不要用 DRL 做配对**（可作为对比基线）。

---

## 8. 常用命令

```bash
# 激活环境
conda activate llm-ops-torch26
PY=/home/link/miniconda/envs/llm-ops-torch26/bin/python

# 主实验（~30s）
$PY -m uav_mec.run_main_experiment

# NF / γ_SIC_min 敏感性（~60s）
$PY -m uav_mec.run_nf_sic_sensitivity

# K 扩展性
$PY -m uav_mec.run_k_scalability

# 生成所有图表
$PY -m uav_mec.generate_figures_v2

# 编译论文
cd paper && env -u http_proxy -u https_proxy ~/.local/bin/tectonic -o . main_v2.tex

# 单次 episode 快速验证
$PY -c "
from uav_mec.run_main_experiment import run_episode, strategy_ours
from uav_mec.model import ScenarioConfig
cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True, user_mobility=True)
r = run_episode(cfg, 42, strategy_ours)
print(f'Deadline: {r[\"deadline_rate\"]:.1f}%')  # 期望 ~73%
"
```

---

## 9. 论文审稿状态（2026-04-15）

**已修复的高风险问题**（hostile reviewer preemption）：
- `second layer of SIC feasibility verification` → 已改为准确描述
- `Complete Holm Correction` → 已改为 `Holm Correction Summary`（表格为压缩版）
- `critical enabler` / `central role` / `central source` → 已软化
- `sustained effectiveness` → 已改为 `consistent empirical performance`
- `representational bottleneck` (DRL caption) → 已软化为 "consistent with continuous action space..."
- CI 上界 74.4 → 74.3（修正计算）
- Random% 正文 63.8 → 63.7（与表格对齐）
- `<0.1% violation rate`（无代码支撑）→ 已删除
- `γ_SIC_min` 敏感性表：代码 bug 已修（`sic_gamma_min` 参数接通），已重跑，三行数据为真实扫描结果

**仍需作者决策**：
- BG+Swap 的 Holm 乘数 ×1 逻辑需作者核对代码输出日志
