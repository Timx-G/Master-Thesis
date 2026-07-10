# 多无人机分组 NOMA MEC 干扰感知用户配对与任务卸载

> **论文标题**：Interference-Aware User Pairing and Task Offloading for Grouped NOMA Multi-UAV MEC Networks
> **目标期刊**：IEEE Internet of Things Journal（CCF-B 类）
> **作者**：Mingxi Guan，中国科学技术大学
> **核心结果**：73.4% 截止时间满足率，比 Greedy 基线高 +11.3pp（20 个随机种子，$p < 0.001$）

---

## 一、论文背景与问题

### 1.1 研究场景

- $K$ 架无人机（UAV）在 800m×800m 区域以固定高度（100m）悬停
- 每架 UAV 携带 MEC 服务器（算力 4 GHz），为地面 $N=20$ 个 IoT 用户提供服务
- 频谱资源：5 MHz 带宽，$S=3$ 个子载波，**全频率复用**（频率复用因子=1）
- 每个子载波上两个用户组成一个分组 NOMA 对，通过 SIC 解码

### 1.2 核心挑战：全频率复用下的跨 UAV 干扰耦合

当所有 UAV 共用所有子载波时，**UAV-$k$ 上一对用户的配对决策会改变 UAV-$k'$ 上所有用户受到的干扰**，导致所有配对决策全局耦合。

这与单小区 NOMA 用户配对有本质区别：
- **单小区**：配对决策相互独立
- **多 UAV 全频复用**：配对决策全局耦合 → **NP 完全**（通过独立集归约证明，见附录 A）

### 1.3 为什么已有方法不够用？

| 方法类别 | 代表工作 | 局限性 |
|---------|---------|--------|
| 单 UAV NOMA 配对 | Ali et al., Altin et al. | 不处理跨 UAV 干扰 |
| 多 UAV + 正交频谱 | Li et al. (2023) | 每 UAV 独立子带，无跨 UAV 干扰 |
| 交换稳定匹配 | Di et al., Altin (TES) | 双边准则在干扰外部性下效果有限 |
| DRL | PPO, MAPPO | 连续动作空间无法直接表示离散配对 |

---

## 二、核心创新点

### 创新点 1：全频复用多 UAV 分组 NOMA MEC 框架

首篇同时处理以下三者的论文：
1. 多 UAV 全频率复用 NOMA
2. 跨 UAV 干扰耦合的用户配对优化
3. MEC 任务卸载截止时间目标

### 创新点 2：干扰感知跨 UAV 交换算法（核心贡献）

提出三阶段算法：

```
阶段 1: 每 UAV 的 ILP 初始化
  → 枚举所有 2 用户候选对
  → 估算每对的截止时间收益 b(i,j)
  → 求解 ILP：选取最多 S 个不重叠的对，最大化 Σb(i,j)

阶段 2: 跨 UAV 交换精化（主要性能来源）
  → 当 N(π) 中存在改进交换时循环：
      对 {Replace, Position, CrossUAV} 中的每种交换类型：
          若全局截止时间满足数严格增加 → 接受，重启扫描

阶段 3: 闭式最优卸载比
  → 对每个被服务用户 i：
      通过分段线性交叉点条件（非标准 KKT）计算 α_i*
```

**关键洞察**：在最近 UAV 分配下，将用户移至更远 UAV 会被双边准则拒绝（20 个种子 0/660 接受），但全局准则允许这种改变并获得 +2.6pp 收益。

### 创新点 3：闭式卸载比（非标准 KKT）

任务时延 $\tau_i = \max(\tau_{\text{loc}}, \tau_{\text{off}})$ 是分段线性非可微函数，令 $\tau_{\text{loc}}(\alpha^*) = \tau_{\text{off}}(\alpha^*)$ 得到唯一最小化点：

$$\alpha_i^* = \frac{C_i/f_{\text{local}}}{C_i/f_{\text{local}} + (1+\rho)D_i/R_i + C_i/f_{\text{alloc},i}}$$

### 创新点 4：NP 完全性证明

通过独立集（IS）归约证明离线配对选择核心为 NP 完全。

---

## 三、算法设计与证明

### 3.1 三阶段算法概览

| 阶段 | 名称 | 复杂度 | 性能贡献 |
|------|------|--------|---------|
| Stage 1 | Per-UAV ILP 初始化 | $O(N^2/K^2)$ | +5.6pp vs Greedy |
| Stage 2 | 跨 UAV 交换精化 | $O(T \cdot |\mathcal{N}(\pi)| \cdot KS)$ | +5.7pp （主要来源）|
| Stage 3 | 闭式卸载比 | $O(N)$ | — |

总实测运行时间：14ms（$N=20$，Python 仿真），远低于 1s 时隙预算。

### 3.2 关键定理

- **命题 1**：离线配对选择核心为 NP 完全（IS 归约）
- **命题 2**：交换过程在有限步内收敛至 $\mathcal{N}(\pi)$-局部最优（每次接受严格增加目标值）
- **引理 1**：闭式交叉点条件对固定 $R_i$ 和 $f_{\text{alloc},i}$ 唯一最小化 $\tau_i$

### 3.3 设计陷阱（已验证，勿重复）

1. **UAV 轨迹优化无价值**：100m LoS MEC 场景下，轨迹优化边际贡献可忽略（已验证 6 轮）
2. **20 用户全频 NOMA 不切实**：SIC 链路崩溃，须分组
3. **DRL 无法超越贪心配对**：PPO/MAPPO/TD3 全部测试，连续动作空间无法直接索引离散配对
4. **初始化方法影响极小**：交换存在时，BG 初始化 vs Greedy 初始化只差 0.7pp

---

## 四、实验结果

### 4.1 主实验结果（Table V）

| 方法 | 截止时间率（%，±95%CI）| 与 Ours 差距（pp）|
|------|----------------------|------------------|
| **Ours（ILP+Swap）** | **73.4±1.0** | — |
| BG+Swap | 72.4±1.0 | −1.0*** |
| IA-Greedy+Swap | 71.9±1.0 | −1.4*** |
| ExchangeStable | 69.1±0.9 | −4.2*** |
| Per-UAV ILP | 67.7±1.1 | −5.7*** |
| OFDMA(OrthoBank) | 65.5±1.2 | −7.9*** |
| OFDMA(FullReuse) | 64.1±0.9 | −9.3*** |
| **Greedy（主要基线）** | 62.1±1.1 | **−11.3**** |

所有 15 个差距 $p_{\text{corr}} < 0.001$（Wilcoxon 符号秩检验，Holm-Bonferroni 校正）

### 4.2 扩展性

- **N 扩展**：差距单调递增，+5.7pp（N=10）→ +16.1pp（N=30）
- **K 扩展**：K=1~5 全程差距 +7.9~+24.3pp
- **参数鲁棒性**：NF/SIC/CSI 扰动下差距始终 ≥10pp

### 4.3 2×2 消融实验（Table VI）

|  | 双边准则 | 全局准则 |
|--|---------|---------|
| **仅 Intra-UAV** | ExchangeStable: 69.1% | IntraSwapOnly: 70.8% |
| **+ Cross-UAV** | ExStable+X-UAV: 69.6% | **Ours: 73.4%** |

两个组件缺一不可：拓扑单独贡献 +0.4pp（双边），在全局准则下贡献 +2.6pp。

---

## 五、代码结构

```
Master-Thesis/
├── README.md                          # 本文件
├── CLAUDE.md                          # 项目总览（agent 使用）
├── requirements.txt                   # Python 依赖
├── run_mappo_convergence.py           # MAPPO 训练并保存历史
│
├── uav_mec/                           # 主 Python 包
│   ├── CLAUDE.md                      # 论文章节→代码速查（agent 使用）
│   ├── model.py                       # 物理层：信道、UAV、任务生成器
│   ├── multi_uav_env.py               # 仿真环境（任务队列、用户移动）
│   ├── pairing.py                     # 核心算法：kkt_alpha / swap_optimize
│   ├── baselines_pairing.py           # 全部基线：ILP / ExStable / OFDMA 等
│   ├── run_main_experiment.py         # 主实验（20 seeds × 16 方法）
│   ├── run_nf_sic_sensitivity.py      # NF / γ_SIC_min 敏感性实验
│   ├── run_k_scalability.py           # K 扩展性实验
│   ├── run_supplementary_experiments.py  # N 扩展、CSI 误差等
│   ├── run_exhaustive_and_positions.py   # 小规模穷举最优
│   ├── run_drl_convergence_and_mappo.py  # DRL 训练（PPO + MAPPO）
│   ├── run_ppo_fair.py                # PPO 零样本评估
│   ├── run_round2_experiments.py      # PPO 规范评估
│   ├── generate_figures_v2.py         # 图 3/5/6/7/DRL 生成脚本
│   └── generate_fig1_fig9.py          # 图 1（系统模型）/ 图 4（Swap 可视化）
│   └── agents/
│       ├── ppo.py                     # PPO agent
│       └── mappo.py                   # MAPPO agent
│
├── paper/
│   ├── main_v2.tex                    # 论文 LaTeX 源文件
│   ├── refs_v2.bib                    # 参考文献（~42 条）
│   └── main_v2.pdf                    # 编译输出
│
├── figures/                           # 生成的图表（PDF + PNG）
├── results/                           # 实验数据（JSON）
├── docs/                              # 研究笔记
└── archive/                           # 历史实验存档
```

---

## 六、所有图表：生成脚本与对应数据

| 论文图编号 | 文件名 | 生成函数 | 数据来源 |
|-----------|--------|---------|---------|
| Fig. 1 | `fig1_system_model.pdf` | `generate_fig1_fig9.py::fig1_system_model()` | 无（纯绘制） |
| Fig. 2 | `fig3_main_comparison.pdf` | `generate_figures_v2.py::fig3_main_comparison()` | `phase2_main_experiment.json` |
| Fig. 3 | `fig7_timeseries.pdf` | `generate_figures_v2.py::fig7_timeseries()` | 实时运行仿真 |
| Fig. 4 | `fig9_pairing_visualization.pdf` | `generate_fig1_fig9.py::fig9_pairing_visualization()` | 无（纯绘制） |
| Fig. 5 | `fig_drl_convergence_comparison.pdf` | `generate_figures_v2.py::fig_drl_convergence()` | `ppo_convergence_2k.json` + `mappo_convergence_history.json` |
| Fig. 6 | `fig5_scalability.pdf` | `generate_figures_v2.py::fig5_scalability()` | `n_scalability_3methods.json` |
| Fig. 7 | `fig6_tmax_sensitivity.pdf` | `generate_figures_v2.py::fig6_tmax()` | `tmax_sensitivity_3methods.json` |

### 一键重绘所有图表

```bash
conda activate llm-ops-torch26
cd ~/my_workspace/Master-Thesis

# 重绘图 1 和图 4（系统模型 + Swap 可视化）
python -m uav_mec.generate_fig1_fig9

# 重绘图 2/3/5/6/7（主比较 + 时序 + DRL + 扩展性 + t_max）
python -m uav_mec.generate_figures_v2

# 编译论文
cd paper && env -u http_proxy -u https_proxy ~/.local/bin/tectonic -o . main_v2.tex
```

---

## 七、实验数据文件

| JSON 文件 | 对应论文内容 | 如何生成 |
|-----------|------------|---------|
| `phase2_main_experiment.json` | **Table V 主结果**（16 方法×20 seeds）| `python -m uav_mec.run_main_experiment` |
| `nf_sic_sensitivity.json` | Table IX（NF）+ Table X（γ_SIC_min）| `python -m uav_mec.run_nf_sic_sensitivity` |
| `k_scalability_full.json` | Table VIII（K 扩展）| `python -m uav_mec.run_k_scalability` |
| `n_scalability_3methods.json` | Fig. 6 N 扩展性（3 条线）| 手动运行（见下方） |
| `tmax_sensitivity_3methods.json` | Fig. 7 t_max 敏感性（3 条线）| 手动运行（见下方） |
| `ppo_convergence_2k.json` | Fig. 5 PPO 训练曲线 | `python -m uav_mec.run_ppo_fair` |
| `mappo_convergence_history.json` | Fig. 5 MAPPO 训练曲线 | `python run_mappo_convergence.py` |
| `phase2_supplementary.json` | CSI 误差 / 子载波 / t_max 等 | `python -m uav_mec.run_supplementary_experiments` |
| `phase2_table2_full.json` | 小规模穷举最优（§V-C）| `python -m uav_mec.run_exhaustive_and_positions` |

**数字权威优先级：JSON > 代码运行输出 > 论文文本。如发现不一致，以 JSON 为准修改论文。**

---

## 八、完整运行指南

### 8.1 环境配置

```bash
# 唯一可用的完整环境
conda activate llm-ops-torch26

# 验证
python -c "import numpy, scipy, torch, gymnasium, matplotlib; print('All OK')"
```

| 包 | 版本 |
|----|------|
| Python | 3.11.13 |
| numpy | 1.26.4 |
| scipy | 1.17.1 |
| torch | 2.1.0+cu121 |
| gymnasium | 1.2.3 |
| matplotlib | 3.10.8 |

**注意**：系统代理 `http_proxy=http://172.25.96.1:7897` 可能不通。安装包时绕过：
```bash
env -u http_proxy -u https_proxy pip install <package>
```

### 8.2 快速验证（单次 episode）

```bash
python -c "
from uav_mec.run_main_experiment import run_episode, strategy_ours
from uav_mec.model import ScenarioConfig
cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True, user_mobility=True)
r = run_episode(cfg, 42, strategy_ours)
print(f'截止时间率: {r[\"deadline_rate\"]:.1f}%')  # 期望 ~73%
"
```

### 8.3 主实验（约 30 秒）

```bash
python -m uav_mec.run_main_experiment
# 输出：results/phase2_main_experiment.json
# 打印：16 种方法的截止时间率 + Holm-Bonferroni 统计检验
```

### 8.4 敏感性实验

```bash
# NF 和 SIC γ_min 敏感性（约 60 秒）
python -m uav_mec.run_nf_sic_sensitivity

# K 扩展性
python -m uav_mec.run_k_scalability

# N 扩展性（3 种方法，N=10-30，约 5 分钟）
python -c "
import json, numpy as np
from pathlib import Path
from uav_mec.model import ScenarioConfig
from uav_mec.run_main_experiment import run_episode, strategy_ours, strategy_greedy
from uav_mec.baselines_pairing import exchange_stable_pairing
# ... (见 run_supplementary_experiments.py 中的 run_n_scalability)
"

# t_max 敏感性（3 种方法，约 5 分钟）
# 见 run_supplementary_experiments.py
```

### 8.5 DRL 训练（较慢）

```bash
# PPO 2000 episodes 训练 + 零样本测试（约 3 分钟）
python -m uav_mec.run_ppo_fair

# MAPPO 2000 episodes 训练 + 保存历史（约 7 分钟）
python run_mappo_convergence.py
```

### 8.6 编译论文

```bash
cd ~/my_workspace/Master-Thesis/paper
env -u http_proxy -u https_proxy ~/.local/bin/tectonic -o . main_v2.tex
# 输出：main_v2.pdf
```

---

## 九、论文章节与代码对照

| 论文章节 | 对应代码 | 关键函数 |
|---------|---------|---------|
| §III-A 网络模型 | `model.py` | `ScenarioConfig`, `NOMAChannel` |
| §III-B NOMA 信道（SINR 公式）| `pairing.py` | `compute_rates_for_pairing()` |
| §III-B SIC 可行性过滤 | `baselines_pairing.py` | `ilp_optimal_pairing(..., sic_gamma_min)` |
| §III-C 任务模型 + 卸载 | `pairing.py` | `evaluate_pairing()` |
| §III-C 最优卸载比（引理 1）| `pairing.py` | `kkt_alpha()` |
| §IV-A Stage 1 ILP 初始化 | `baselines_pairing.py` | `ilp_optimal_pairing()` |
| §IV-B Stage 2 跨 UAV 交换（核心）| `pairing.py` | `swap_optimize()` |
| §IV-C Stage 3 闭式卸载比 | `pairing.py` | `kkt_alpha()` |
| §V 主实验（Table V）| `run_main_experiment.py` | `main()`, `methods` dict |
| §V-B 2×2 消融 | `run_main_experiment.py` | `strategy_intra_swap_only()`, `strategy_exchange_stable_cross_uav()` |
| §V-D DRL 对比 | `run_drl_convergence_and_mappo.py` | `train_ppo_extended()`, `train_mappo()` |
| §V-E N 扩展性 | `run_supplementary_experiments.py` | `run_n_scalability()` |
| §V-E K 扩展性 | `run_k_scalability.py` | `main()` |
| §V-F 鲁棒性 | `run_nf_sic_sensitivity.py` | `main()` |
| 附录 A（NP 完全性证明）| 仅在论文中，无对应代码 | — |
| 附录 B（Holm 表）| `run_main_experiment.py` | 统计检验输出 |
| 附录 C（N 扩展数值）| `run_supplementary_experiments.py` | `run_n_scalability()` |

---

## 十、关键参数（锁定，勿随意更改）

| 参数 | 值 | 来源 |
|------|----|------|
| 区域 | 800×800 m | — |
| UAV 高度 $H$ | 100 m | DJI M300 |
| UAV 数量 $K$ | 2（扩展 K=1–5）| — |
| 用户数 $N$ | 20 | — |
| 子载波数 $S$ | 3（×1.67 MHz）| Narrowband IoT |
| NOMA 对大小 | 2 | 工程实践 |
| 带宽 $B$ | 5 MHz | LTE |
| 接收机噪声系数 NF | 7 dB（默认）| UAV 接收机 |
| SIC 残差 $\varepsilon$ | 0.02 | 文献 |
| MEC 算力 / UAV | 4 GHz | 嵌入式 MEC |
| 本地 CPU | 1 GHz | IoT MCU |
| 截止时间 $t_{\max}$ | 0.8 s | 软实时校准值 |
| 发射功率 | 0.2 W（23 dBm）| 3GPP Class 3 |
| 到达概率 | 0.8 | 密集 IoT |
| 用户速度 | 2 m/s | ITU 行人 |

---

## 十一、设计陷阱与历史教训

### 已证明无效的方向（勿重复）

| Phase | 尝试方向 | 结论 | 状态 |
|-------|---------|------|------|
| A | 单 UAV DC-PPO 轨迹优化 | 固定 λ 优于 DRL | 存档 |
| B | 多 UAV DRL 轨迹 | 悬停优于 DRL（验证 6 轮）| 存档 |
| C | DRL 分组（6 维动作）| PPO ≈ Greedy（0pp 差距）| 存档 |
| **D（当前）**| **ILP 初始化 + 跨 UAV 交换** | **+11.3pp，$d=6.20$** | **Active** |

### 关键设计注意事项

1. **`kkt_alpha` 命名误导**：函数名含 "kkt" 但 $\max(\tau_{\text{loc}}, \tau_{\text{off}})$ 不可微，实为交叉点条件，不是标准 KKT。**论文中不要写 "standard KKT"。**

2. **ILP Stage 1 忽略跨 UAV 干扰**：`ilp_optimal_pairing()` 的收益估计使用 SINR 无 $I_{\text{cross}}$。这是设计决策，不是 bug。交叉干扰由 Stage 2 交换修正。

3. **SIC 阈值只过滤 Stage 1**：`sic_gamma_min` 参数只控制 ILP 初始化时的候选对过滤，Stage 2 交换无显式 SIC 过滤（靠低 SINR → 低速率 → 低截止满足数隐式排除）。

4. **`strategy_random` 使用独立 PRNG**：`env.strategy_rng`（非 `env.rng`），确保不同方法的信道/任务随机流完全相同，比较公平。

5. **Greedy < Random**：Greedy 按信道增益排序 → 高功率用户集中在相同子载波 → 干扰受限场景下反而加剧交叉干扰。这是真实物理现象，不是代码 bug。

---

## 十二、投稿状态与已修复的关键问题

| 类别 | 已修复的问题 |
|------|------------|
| 代码-论文一致性 | SIC `γ_SIC_min` 敏感性表：`sic_gamma_min` 参数已接通 `ilp_optimal_pairing`；重跑获得真实扫描结果 |
| 统计表述 | "complete Holm correction" → "Holm Correction Summary"；WorstCase+Swap 的 Holm 乘数已从 ×15 修正为 ×1 |
| 因果过度声明 | DRL 图注中 "structural representational bottleneck" 软化；Conclusion 中 "central source" 等措辞已软化 |
| 数字一致性 | CI 上界 74.4 → 74.3；Random% 正文与表格已对齐 |
| 未支撑数字 | "<0.1% violation rate"（无代码统计）已删除 |
| MAPPO 数据 | MAPPO 训练曲线已从头跑出并保存（`mappo_convergence_history.json`）；不再用横线代替训练曲线 |

---

## 十三、如何向导师讲述本论文



> 在多无人机全频复用分组 NOMA MEC 网络中，用户配对决策全局耦合，是 NP 完全问题。我们提出了一个干扰感知的跨 UAV 交换算法，用全局优先改进准则代替双边准则，实现了比 Greedy 高 11.3pp 的截止时间满足率。

### 讲给导师的逻辑线

1. **问题为什么难？** → 全频复用让所有配对决策全局耦合（NP 完全，附录 A 证明），已有单小区方法无法直接迁移。

2. **关键洞察是什么？** → 双边准则（ExchangeStable）在干扰外部性下失效：把用户移到更远 UAV 可能被双边拒绝（实验：0/660 次接受），但对全局干扰结构有益。

3. **我们如何解决？** → 三阶段：ILP 初始化（快速找到好起点）→ 跨 UAV 交换+全局准则（修正干扰耦合）→ 闭式卸载比（Stage 3 与配对解耦，独立最优）。

4. **为什么这有效？** → 2×2 消融：跨 UAV 拓扑在双边准则下只贡献 +0.4pp，在全局准则下贡献 +2.6pp，两者缺一不可。

5. **实验证据有多强？** → 20 种子、15 基线、$p_{\text{corr}} < 0.001$、效应量 $d=6.20$。N 扩展：差距单调递增（越密集越有价值）。

6. **与 DRL 的关系？** → DRL 是架构参考，不是主要竞争者。PPO 平台在 ~60%（训练全程），MAPPO 因多智能体非平稳性从 ~61% 退化至 ~46%，均远低于 Ours 73.4%。

---

## 十四、常用命令速查

```bash
PY=/home/link/miniconda/envs/llm-ops-torch26/bin/python

# 激活环境
conda activate llm-ops-torch26

# 主实验（~30s）
$PY -m uav_mec.run_main_experiment

# 重绘所有图
$PY -m uav_mec.generate_fig1_fig9
$PY -m uav_mec.generate_figures_v2

# 编译论文
cd paper && env -u http_proxy -u https_proxy ~/.local/bin/tectonic -o . main_v2.tex

# 单次验证
$PY -c "
from uav_mec.run_main_experiment import run_episode, strategy_ours
from uav_mec.model import ScenarioConfig
cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True, user_mobility=True)
r = run_episode(cfg, 42, strategy_ours)
print(f'截止时间率: {r[\"deadline_rate\"]:.1f}%')
"
```

---

*最后更新：2026-04-16*
*项目路径：`/home/link/my_workspace/Master-Thesis/`*
