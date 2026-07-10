# Multi-UAV Grouped NOMA MEC: Phase框架 v3

> **日期**: 2026-04-13
> **方向**: Hungarian + Cross-UAV Swap + KKT 三层混合优化
> **目标**: IEEE IoTJ (CCF-B), 21天完成

---

## 总览

```
Phase 0 (2-3天)          Phase 1 (5-7天)          Phase 2 (5-7天)          Phase 3 (5-7天)
参数标定+可行性    →    算法实现+单元测试    →    全实验+图表     →     论文+投稿

Gate: gap>=5pp          Gate: gap>=4pp(10ep)     Gate: p<0.05           Gate: 导师approve
Kill: gap<3pp           Kill: gap<2pp            Kill: p>0.1
```

---

## Phase 0: 参数标定与可行性验证 (2-3天)

### 目标
确认问题设置给算法留有足够优化空间

### 任务
1. 当前参数(t_max=0.7s, N=20, n_sub=3, arrival=0.8)下Greedy baseline跑100 episodes
2. 单slot穷举所有合法配对，确认 optimal vs greedy gap
3. 如果gap不足，调整t_max或N

### Gate Check
| 条件 | 标准 |
|------|------|
| PASS | Greedy deadline% 在 60-85% **且** 单slot穷举gap >= 5pp |
| KILL | Greedy > 90%(太简单) 或 < 50%(参数坏) 或 gap < 3pp(无空间) |

### 产出
- `results/phase0_calibration.json`: 参数配置+baseline数据
- 确定的最终参数集

---

## Phase 1: 核心算法实现与单元测试 (5-7天)

### 目标
实现Hungarian + Cross-UAV Swap + KKT三层优化，通过单元测试

### 新增代码

| 文件 | 内容 | 预估行数 |
|------|------|---------|
| `uav_mec/pairing.py` | Hungarian匹配 + Cross-UAV Swap + KKT alpha | ~400行 |
| `uav_mec/baselines_pairing.py` | 6个baseline实现 | ~200行 |
| `uav_mec/run_main_experiment.py` | 20-seed主实验脚本 | ~150行 |
| `uav_mec/run_ablations.py` | 消融实验脚本 | ~100行 |

### 重构代码

| 文件 | 改动 |
|------|------|
| `multi_uav_env.py` | 提取PairingStrategy接口，支持策略注入 |

### 算法细节

```python
# pairing.py 核心接口

def hungarian_pairing(h_gains, n_sub, users_per_sub):
    """构造cost矩阵 → scipy.optimize.linear_sum_assignment → 配对结果"""

def cross_uav_swap(pairing, h_gains, rates_fn, max_iter=50):
    """三种swap操作:
    - Replace: served ↔ unserved
    - Position: 交换子载波位置
    - Cross-UAV: UAV0用户 ↔ UAV1用户
    贪心接受改进的swap，直到无改进"""

def kkt_alpha(D, C, R, f_alloc, f_local):
    """闭式解: α* = (C/f_local) / (C/f_local + D/R + C/f_alloc)"""
```

### 理论产出（并行写LaTeX）

| # | 理论结果 | 内容 |
|---|---------|------|
| Proposition 1 | NP-hardness | 跨UAV干扰耦合使配对问题NP-hard |
| Lemma 1 | KKT最优性 | α*使local delay = offload delay是单用户最优 |
| Proposition 2 | Swap单调性 | 每次swap不降低系统deadline rate |
| 复杂度 | O(M³) per slot | Hungarian O(M³) + Swap O(M²) + KKT O(N) |

### Gate Check
| 条件 | 标准 |
|------|------|
| PASS | Hungarian+Swap vs Greedy >= 4pp (10 episodes, 3 seeds) |
| KILL | gap < 2pp → 算法贡献不成立 |

### 产出
- `pairing.py` 通过单元测试
- 重构后的 `multi_uav_env.py`
- LaTeX Section II (System Model) + Section III (Algorithm) 初稿

---

## Phase 2: 全实验与图表生成 (5-7天)

### 实验清单

| # | 实验 | 方法数 | Seeds | 验证贡献 |
|---|------|--------|-------|---------|
| Exp1 | **主对比** | 7 (Ours + 6 baselines) | 20 | C1+C2全景 |
| Exp2 | 消融-Swap | 2 (Ours vs No-Swap) | 20 | C3 |
| Exp3 | 消融-KKT | 2 (Ours vs Fixed-α) | 20 | C2(KKT) |
| Exp4 | 消融-Hungarian | 2 (Ours vs Greedy+Swap) | 20 | C2(Hungarian) |
| Exp5 | Scalability | N={10,15,20,25,30} | 10 | 可扩展性 |
| Exp6 | 子载波敏感性 | n_sub={2,3,4,5} | 10 | 参数影响 |
| Exp7 | Burst时序 | 逐slot记录 | 3 | C4可视化 |
| Exp8 | DRL对比 | PPO/MAPPO vs Ours | 10 | Discussion |

### 6个Baseline

| # | Baseline | 对标贡献 | 预期差距 |
|---|----------|---------|---------|
| 1 | **Greedy Pairing** (信道排序top-K配对) | C2主对比 | Ours > 5pp+ |
| 2 | **Random Pairing** (随机配对) | 下界 | Ours > 15pp+ |
| 3 | **Hungarian-Only** (无Swap) | C3(Swap必要性) | Ours > 3pp+ |
| 4 | **Equal Alpha** (α=0.5固定) | C2(KKT价值) | Ours > 2pp+ |
| 5 | **Single-UAV** (1架8GHz) | C1(多UAV必要性) | Ours > 10pp+ |
| 6 | **OFDMA** (纯正交无NOMA) | C1(NOMA增益) | Ours > 3pp+ |

### 4个消融

| 消融 | 去掉什么 | 预期退化 | 证明什么 |
|------|---------|---------|---------|
| Ablation-Swap | 去Cross-UAV Swap | >= 3pp | C3: Swap处理干扰耦合 |
| Ablation-KKT | α固定为0.5 | >= 2pp | KKT闭式解价值 |
| Ablation-Hungarian | 用Greedy替代Hungarian | >= 2pp | Hungarian初始解质量 |
| Ablation-Interference | 去跨UAV干扰建模 | >= 3pp | C1: 干扰建模必要性 |

### 指标体系

**主指标** (决定论文成败):
| 指标 | 计算方式 | 越高/低越好 |
|------|---------|------------|
| Deadline Met Rate (%) | met_tasks / total_tasks × 100 | 越高越好 |
| Average Delay (s) | mean(all task delays) | 越低越好 |

**辅助指标** (丰富论文):
| 指标 | 计算方式 | 用途 |
|------|---------|------|
| Jain Fairness Index | (Σqi)² / (N·Σqi²) | 跨用户QoS公平性 |
| Cross-UAV Interference (dB) | mean(I_inter) per slot | 干扰管理效果 |
| Computation Overhead (ms/slot) | 算法wallclock time | 实时性证明 |

**统计方法**:
- 20 seeds, 固定seed集: {42,123,456,789,1024,...}
- Wilcoxon signed-rank test (paired, non-parametric)
- Cohen's d (effect size): d >= 0.8 = large effect
- 95% Bootstrap CI
- Bonferroni correction for multiple comparisons

### Figure清单 (7张)

| # | 内容 | 类型 | 支撑 |
|---|------|------|------|
| Fig.1 | 系统模型(2UAV+3cluster+grouped NOMA+干扰) | 系统图 | C1 |
| Fig.2 | 算法流程(Hungarian→Swap→KKT三层) | 流程图 | C2 |
| Fig.3 | 主对比柱状图(7方法×deadline%+delay) | 柱状图 | 全部 |
| Fig.4 | 消融柱状图(4 ablations) | 柱状图 | C2+C3 |
| Fig.5 | Scalability折线(N=10~30) | 折线图 | 可扩展性 |
| Fig.6 | Burst时序图(逐slot deadline rate) | 时序图 | C4 |
| Fig.7 | 配对可视化(用户-子载波-UAV分配矩阵) | 热力图 | C2 |

### Table清单 (3张)

| # | 内容 | 支撑 |
|---|------|------|
| Table I | 系统参数 | Section II |
| Table II | 主实验结果(7方法×5指标, mean±std, p, d) | Section IV |
| Table III | 计算开销对比(ms/slot) | Section IV |

### Gate Check
| 条件 | 标准 |
|------|------|
| PASS | Exp1 Wilcoxon p < 0.05, Cohen's d >= 0.8, 至少2个消融退化 >= 3pp |
| DELAY | p 在 0.05-0.1 之间 → 增加到30 seeds |
| KILL | p > 0.1 (极不可能，如果Phase 1 gate已过) |

### 产出
- `results/exp_main_*.csv`, `results/exp_ablation_*.csv`
- `figures/fig1-7.pdf`
- LaTeX Section IV (Results) 初稿

---

## Phase 3: 论文完成与投稿 (5-7天)

### 论文结构

| Section | 页数 | 内容 | 来源 |
|---------|------|------|------|
| I. Introduction | 1.5 | 问题动机+贡献 | Phase 3写 |
| II. System Model | 2.0 | Grouped NOMA + 干扰 + CMDP | Phase 1写 |
| III. Algorithm | 2.0 | Hungarian + Swap + KKT + 理论 | Phase 1写 |
| IV. Results | 3.0 | 全部实验+图表 | Phase 2写 |
| V. Related Work | 1.0 | 25+篇引用 | Phase 3写 |
| VI. Conclusion | 0.3 | 总结+future work | Phase 3写 |
| References | 1.2 | ~30篇 | 累积 |
| **Total** | **~11页** | | |

### Gate Check
| 条件 | 标准 |
|------|------|
| PASS | 导师approve投稿 |
| DELAY | 导师要求补实验 → 返回Phase 2 |

### 产出
- `paper/main_v2.tex` + `paper/main_v2.pdf`
- 投稿就绪

---

## 代码清理计划

### 归档（移入archive/分支）
- Phase A全部实验脚本: `exp*.py`, `run_dcppo*.py`, `full_experiment*.py`
- Phase B轨迹方向脚本: `train_grouped_noma.py` (DRL版)
- 旧baseline: `baselines_multi.py` (DRL-action版)

### 保留不动
- `model.py`: 物理模型 (核心资产)
- `agents/ppo.py`, `agents/mappo.py`: DRL agents (作为baseline)
- `paper/main.tex`: Phase A论文 (参考)

### 新增
| 文件 | 用途 |
|------|------|
| `uav_mec/pairing.py` | **核心**: Hungarian + Swap + KKT |
| `uav_mec/baselines_pairing.py` | 6个baseline |
| `uav_mec/run_main_experiment.py` | 20-seed主实验 |
| `uav_mec/run_ablations.py` | 消融实验 |
| `uav_mec/generate_figures_v2.py` | 图表生成 |

### 重构
| 文件 | 改动 |
|------|------|
| `multi_uav_env.py` | 提取PairingStrategy接口，支持策略注入 |

---

## 风险矩阵

| 风险 | 概率 | 影响 | 缓解 | 何时检测 |
|------|------|------|------|---------|
| Hungarian vs Greedy gap < 3pp | 中 | Kill | 调参数(t_max/N/n_sub) | Phase 0 Day 1 |
| Cross-UAV Swap退化 < 1pp | 中 | C3不成立 | 增强干扰模型或放弃C3 | Phase 1 |
| 20-seed统计不显著 | 低 | 延迟 | 增到30 seeds | Phase 2 |
| 导师大改方向 | 低 | 高影响 | Phase 1后即汇报 | Phase 1结束 |

---

## Plan B (如果Phase 0或1 Kill)

转"Multi-UAV NOMA-MEC Comprehensive Benchmark"综述实验型论文:
- 用已有全部数据(DRL/Greedy/Random/穷举)
- 贡献变为系统建模+全面实验对比
- 投Computer Networks (CCF-B) 或 Ad Hoc Networks (CCF-C)
- 时间: 3-4周
