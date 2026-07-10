# Multi-UAV MEC-NOMA 论文执行计划 v2（终版）

> **基于三博士辩论收敛 + 五博士创新点评估 + 数字标准**
> **日期**: 2026-04-12
> **状态**: 批准后严格执行，每Phase存档不可回退

---

## 核心判断（五博士结论）

**创新模式**: 旧算法(Independent PPO, 2017) + 新场景(Multi-UAV shared-spectrum NOMA) — 符合CCF-B wireless标准范式

**录用概率条件**:
| 条件 | 录用概率 |
|------|---------|
| C2闭式解 + 非平稳+10pp+ + NOMA>OFDMA + 20-seed | **75-85%** |
| 以上任一不满足 | 50-65% |
| C2仍是启发式 或 NOMA<OFDMA | <40% |

**数字决定命运。**

---

## 论文数字标准（五博士共识）

### 必须达到的硬指标

| 对比 | 最低安全线 | 理想区间 | Kill线 |
|------|-----------|---------|--------|
| Ours vs BCD-greedy | +10pp, p<0.05 | +15pp+ | <+8pp |
| Ours vs MAPPO | +8pp, p<0.05 | +12pp+ | <+5pp |
| Ours vs Single-UAV | +20pp | +25pp+ | <+15pp |
| Ours vs BCD-static (平稳) | 不为负即可 | +3-5pp | 负值=论文崩溃 |
| **Ours vs BCD-static (非平稳)** | **+8pp, p<0.05** | **+10-15pp** | **<+3pp** |
| NOMA vs OFDMA | NOMA胜+3pp | +5pp+ | OFDMA胜=前提崩溃 |

### 绝对值区间

| 指标 | 可发表 | 危险 | 必死 |
|------|--------|------|------|
| Ours deadline% | 70-85% | 60-70% | <60% |
| Ours delay (s) | <0.85 | 0.85-1.0 | >1.0 |
| Jain's fairness | >0.8 | 0.6-0.8 | <0.6 |

---

## 执行原则

1. **先固地基再盖楼**：环境基础设施必须先稳定，再做算法工作
2. **环境改动与算法改动解耦**：不在不稳定环境上调算法
3. **旧数据全部作废**：Phase 2的78.1%/76.6%在P0完成后不再引用
4. **相对门槛为主**：gate check用DRL vs BCD-static相对差距
5. **三阶段统计**：关键实验走3-seed→7-seed→20-seed
6. **每个Phase存档**：git tag + 数字快照写入archive，不可回退

---

## 存档机制

每个Phase完成后执行：

```bash
# 1. 创建git tag
git tag -a v3.X-phaseYY -m "Phase YY completed: [简述]"

# 2. 写入数字快照
# 文件: results/archive_phaseYY.json
{
    "phase": "0A",
    "date": "2026-04-XX",
    "git_tag": "v3.0-phase0a",
    "numbers": {
        "R_ppo_stationary_3seed": [x, y, z],
        "R_bcd_static_stationary_3seed": [a, b, c],
        "R_random_3seed": [d, e, f],
        ...
    },
    "gate_results": {
        "G1_ppo_converges": true/false,
        "G2_ppo_gt_bcd": true/false,
        ...
    },
    "decision": "PASS/FAIL/KILL",
    "notes": "..."
}

# 3. 冻结声明
# Phase 0A完成后: 禁止修改cluster生成 + KKT optimizer核心逻辑
# Phase 0B完成后: 禁止修改multi_uav_env.py
# Phase 1完成后:  禁止修改任何算法代码
```

---

## Phase依赖图

```
Phase 0A: 地基层（串行，~2天）
  T7(固定cluster) → T1(KKT optimizer)
  │
  ├── 存档: archive_phase0a.json + git tag v3.0-phase0a
  │
Phase 0B: 环境真实化（~3天）
  ├── T2(用户移动) → 单独3-seed验证
  ├── T3(非平稳burst) → 单独3-seed验证
  ├── T2+T3组合 → 3-seed验证 ← 论文C4成败关键
  ├── T4(OFDMA) → 验证NOMA>OFDMA ← 论文C1前提
  ├── T5(BCD代码) → 建立可复现baseline
  └── T8(SINR logging)
  │
  ├── 存档: archive_phase0b.json + git tag v3.1-phase0b
  │
Phase 1: 算法+全量实验（~5天）
  T6(Het-MAPPO) + 3→7→20-seed统计验证
  │
  ├── 存档: archive_phase1.json + git tag v3.2-phase1
  │
Phase 2: 扩展层（~2天）
  T9(K=3) + [T10 IC-CPPO bonus]
  │
  └── 存档: archive_phase2.json + git tag v3.3-phase2
```

---

## Phase 0A: 地基层

### T7: 固定Cluster Centers

**目标**: 消除跨seed的拓扑差异

**实现**:
```python
FIXED_CLUSTER_CENTERS = np.array([
    [200.0, 200.0],   # Cluster 0: 左下
    [600.0, 200.0],   # Cluster 1: 右下
    [400.0, 600.0],   # Cluster 2: 上中
])
# 三角形布局，最小间距 ~283m > 碰撞距离30m
# 用户cluster归属: 均匀分配 ~7人/cluster
# 用户位置: cluster中心 + N(0, 50m)
```

**Gate Check**:
- [ ] 20个不同seed下cluster centers完全一致
- [ ] 用户分布可视化: 3团明显分离
- [ ] env.reset()前后cluster不变

**时间**: 0.5天

---

### T1: KKT Closed-Form Optimizer

**目标**: `_compute_per_user_actions()` 从启发式 → 严格闭式解

**DRL-Optimizer接口**（辩论收敛）:
- DRL输出5维不变: `[dx, dy, cw0, cw1, alpha_base]`
- `alpha_base` ∈ [0.05, 0.95]: scaling factor × KKT最优α
- `cluster_weights`: softmax后给KKT的资源优先级

**卸载比α闭式解**:
```
令 τ_local = τ_offload:
  (1-α)C/f_local = αD/R + αC/f_alloc

解出:
  α_kkt = (C/f_local) / (C/f_local + D/R + C/f_alloc)
        = f_alloc·R / (f_alloc·R + f_local·(D·f_alloc/C + R))

简化（令 r=R, fl=f_local, fa=f_alloc, d=D, c=C）:
  α_kkt = (fl · r) / (fl · r + fa · d/r · fl + c/fa · fl · ... )

最终:
  α_final = clip(alpha_base × α_kkt, 0, 1)
```

**功率分配闭式解**:
```
SIC解码顺序: 按 h_i·p_i 降序
水注法KKT条件:
  p_i* = min(p_max, max(0, μ - σ²/h_{i,k}))
  μ由 Σp_i ≤ N·p_max 确定（二分查找μ）
```

**Gate Check（数字标准嵌入）**:

| # | 检查项 | PASS | FAIL | KILL |
|---|--------|------|------|------|
| G0a.1 | KKT互补松弛残差 | < 1e-6 | 1e-6 ~ 1e-3 (可调) | > 1e-3 |
| G0a.2 | 闭式解 vs scipy数值解 | 偏差 < 1% | 1-5% (检查推导) | > 5% |
| G0a.3 | 1-seed 100ep reward | 无NaN无发散 | 偶发NaN (<1%) | 频繁NaN |
| G0a.4 | reward scale对比旧环境 | 同一量级 | 变化2-5x (需归一化) | 变化>10x |
| G0a.5 | 3-seed PPO收敛 | 前500ep有上升趋势 | 500ep flat (延长到5000) | 5000ep无上升 → **KILL K1** |

**预期数字（Phase 0A输出，替代旧数据）**:

| 指标 | 预期范围 | 说明 |
|------|---------|------|
| R_ppo_new (3-seed, 平稳) | 65-85% | KKT可能更好或更差，范围宽 |
| R_random_new | 50-65% | 环境难度参考 |
| R_ppo_new vs R_random_new | > +10pp | DRL有基本价值 |
| reward scale | 记录min/max/mean/std | 为Phase 0B和Phase 1提供参考 |

**时间**: 2天

---

### Phase 0A 存档

```json
// results/archive_phase0a.json
{
    "phase": "0A",
    "git_tag": "v3.0-phase0a",
    "environment": "fixed_cluster + kkt_optimizer, stationary, no_mobility",
    "numbers": {
        "R_ppo_stationary_3seed": {"seeds": [42,123,456], "values": [], "mean": 0, "std": 0},
        "R_random_3seed": {"seeds": [42,123,456], "values": [], "mean": 0, "std": 0},
        "reward_stats": {"min": 0, "max": 0, "mean": 0, "std": 0},
        "kkt_residual": 0,
        "kkt_vs_scipy_error": 0
    },
    "gate_results": {
        "G0a.1_kkt_residual": false,
        "G0a.2_kkt_vs_scipy": false,
        "G0a.3_no_nan": false,
        "G0a.4_reward_scale": false,
        "G0a.5_ppo_converges": false
    },
    "decision": "PENDING"
}
```

**冻结**: cluster生成逻辑 + KKT optimizer核心公式不可修改

---

## Phase 0B: 环境真实化

### T2: 用户移动 (2m/s随机游走)

**Gate Check（带数字标准）**:

| # | 检查项 | PASS | FAIL |
|---|--------|------|------|
| G0b.t2.1 | 100slot后用户偏移 | ~200m ± 50m | >400m或<50m |
| G0b.t2.2 | 边界处理 | 不越界 | 出现负坐标或>800 |
| G0b.t2.3 | PPO(仅移动) vs PPO(静止) | 性能下降 < 10pp | 下降 > 15pp (移动太难) |
| G0b.t2.4 | PPO(仅移动) 绝对值 | > 60% | < 55% → 考虑降低速度到1m/s |

**预期**: 移动使问题稍难，PPO性能下降3-8pp

**时间**: 0.5天

---

### T3: 非平稳Burst

**实现**: 随机周期U[8,15] + 随机cluster选择

**Gate Check（带数字标准）**:

| # | 检查项 | PASS | FAIL | KILL |
|---|--------|------|------|------|
| G0b.t3.1 | Burst序列FFT | 无20-slot周期峰 | 残留周期性 | — |
| G0b.t3.2 | PPO(仅burst) vs PPO(静止) | 下降 < 10pp | 下降 > 15pp | — |
| G0b.t3.3 | BCD(仅burst) vs BCD(静止) | 下降 > PPO下降 | BCD下降 ≤ PPO下降 | — |
| G0b.t3.4 | **PPO vs BCD (仅burst环境)** | **gap > +3pp** | gap 0-3pp (C4风险) | **gap < 0 → KILL** |

**预期**: BCD下降8-15pp，PPO下降3-8pp → gap从+1.5pp拉大到+5-10pp

**时间**: 0.5天

---

### T2+T3组合验证（**论文C4成败关键**）

**Gate Check（五博士数字标准直接嵌入）**:

| # | 检查项 | PASS(可发表) | CAUTION(需调叙事) | KILL(C4崩溃) |
|---|--------|-------------|-------------------|-------------|
| G0b.combo.1 | PPO(动态) 绝对值 | 70-85% | 60-70% | <60% |
| G0b.combo.2 | BCD-static(动态) 绝对值 | 记录 | — | — |
| **G0b.combo.3** | **PPO vs BCD-static (动态)** | **≥ +8pp** | +5-7pp | **< +3pp → 触发辩论** |
| G0b.combo.4 | PPO(动态) vs PPO(平稳) 下降 | < 8pp | 8-12pp | >15pp (环境太难) |
| G0b.combo.5 | BCD(动态) vs BCD(平稳) 下降 | > 10pp | 5-10pp | <5pp (burst无效果) |

**决策树**:
```
G0b.combo.3 ≥ +10pp → 🟢 C4杀手锏确认，进入Phase 1
G0b.combo.3 = +5~10pp → 🟡 C4可用但需加强narrative，进入Phase 1
G0b.combo.3 = +3~5pp → 🟠 触发辩论：是否修改环境参数(加大burst差异/加快移动)
G0b.combo.3 < +3pp → 🔴 KILL: C4不可行，重新评估论文方向
```

**时间**: 0.5天

---

### T4: OFDMA Baseline（**论文C1前提**）

**Gate Check（五博士数字标准直接嵌入）**:

| # | 检查项 | PASS | CAUTION | KILL |
|---|--------|------|---------|------|
| G0b.t4.1 | OFDMA无用户间干扰 | SINR公式正确 | — | — |
| **G0b.t4.2** | **NOMA vs OFDMA (3-seed PPO)** | **NOMA > OFDMA + 3pp** | NOMA > OFDMA 0-3pp | **OFDMA ≥ NOMA → 前提崩溃 KILL** |
| G0b.t4.3 | NOMA vs OFDMA (BCD-static) | NOMA > OFDMA | — | OFDMA > NOMA |

**预期**: NOMA共频5MHz > OFDMA频分2×2.5MHz → +5-8pp吞吐量优势

**时间**: 0.5天

---

### T5: BCD-static/greedy代码

**Gate Check**:

| # | 检查项 | PASS | FAIL |
|---|--------|------|------|
| G0b.t5.1 | BCD-static收敛 | 目标函数单调不增 | 震荡 |
| G0b.t5.2 | BCD 3-seed稳定 | std < 3% | std > 5% |
| G0b.t5.3 | BCD-greedy << BCD-static | BCD-greedy至少低8pp | 差距 < 5pp |

**时间**: 1天

---

### T8: SINR Logging

**实现**: info dict添加per-user SINR/I_inter/separation + per-episode raw scores

**Gate Check**: SINR ∈ [-10dB, 30dB]，可导出CSV

**时间**: 0.5天

---

### Phase 0B 存档

```json
// results/archive_phase0b.json
{
    "phase": "0B",
    "git_tag": "v3.1-phase0b",
    "environment": "fixed_cluster + kkt + mobility_2mps + random_burst + ofdma_support",
    "numbers": {
        "stationary_env": {
            "R_ppo_3seed": {"values": [], "mean": 0, "std": 0},
            "R_bcd_static_3seed": {"values": [], "mean": 0, "std": 0},
            "R_bcd_greedy_3seed": {"values": [], "mean": 0, "std": 0},
            "R_random_3seed": {"values": [], "mean": 0, "std": 0}
        },
        "mobility_only": {
            "R_ppo_3seed": {"values": [], "mean": 0, "std": 0},
            "delta_vs_stationary": 0
        },
        "burst_only": {
            "R_ppo_3seed": {"values": [], "mean": 0, "std": 0},
            "R_bcd_3seed": {"values": [], "mean": 0, "std": 0},
            "ppo_vs_bcd_gap": 0
        },
        "dynamic_full": {
            "R_ppo_3seed": {"values": [], "mean": 0, "std": 0},
            "R_bcd_static_3seed": {"values": [], "mean": 0, "std": 0},
            "R_bcd_greedy_3seed": {"values": [], "mean": 0, "std": 0},
            "ppo_vs_bcd_static_gap": 0,
            "ppo_vs_bcd_greedy_gap": 0,
            "bcd_static_degradation": 0,
            "ppo_degradation": 0
        },
        "noma_vs_ofdma": {
            "R_noma_ppo_3seed": {"values": [], "mean": 0, "std": 0},
            "R_ofdma_ppo_3seed": {"values": [], "mean": 0, "std": 0},
            "noma_advantage": 0
        }
    },
    "gate_results": {
        "G0b_combo3_ppo_vs_bcd_dynamic": 0,
        "G0b_combo3_verdict": "PASS/CAUTION/KILL",
        "G0b_t4_noma_vs_ofdma": 0,
        "G0b_t4_verdict": "PASS/KILL"
    },
    "decision": "PENDING",
    "frozen_files": ["uav_mec/multi_uav_env.py", "uav_mec/model.py"]
}
```

**冻结**: multi_uav_env.py + model.py 环境核心代码不可修改

---

## Phase 1: 算法层 + 全量实验

### 前置条件: Phase 0A + 0B 全部PASS

### T6: Het-MAPPO

**架构**: 独立Actor + 共享Critic（不共享参数）

**Gate Check**:

| # | 检查项 | PASS | 预期 |
|---|--------|------|------|
| G1.t6.1 | Het-MAPPO收敛 | 500ep内learning curve上升 | — |
| G1.t6.2 | Het-MAPPO vs MAPPO(参数共享) | Het-MAPPO > MAPPO | Het-MAPPO介于Ind-PPO和MAPPO之间 |
| G1.t6.3 | Het-MAPPO vs Ind-PPO | 记录差距（不要求Het胜） | Ind-PPO可能仍然更好 |

**时间**: 1.5天

---

### 全量实验：三阶段统计验证

#### Stage 1: 3-seed Gate（~3h）

所有7个算法 × 动态环境，确认数字在合理范围内。

**数字标准（五博士硬指标直接适用）**:

| 对比 | 3-seed PASS | 3-seed CAUTION | 3-seed KILL |
|------|------------|----------------|-------------|
| Ours vs BCD-greedy | mean gap > +10pp | gap +5-10pp | gap < +5pp |
| Ours vs MAPPO | mean gap > +8pp | gap +3-8pp | gap < +3pp |
| Ours vs BCD-static (动态) | mean gap > +5pp | gap +2-5pp | gap < 0 |
| NOMA vs OFDMA | NOMA mean > OFDMA + 3pp | gap 0-3pp | OFDMA > NOMA |

#### Stage 2: 7-seed Intermediate（仅Table V，~3h）

**目的**: 非平稳gap的power估计

| 检查项 | PASS → 继续20-seed | CAUTION → 调narrative | KILL |
|--------|--------------------|-----------------------|------|
| PPO vs BCD-static (动态) gap | **≥ +8pp** | +5-7pp (降低C4权重) | < +3pp |
| bootstrap pilot CI下界 | > 0 | 包含0但mean > +5pp | CI完全包含0且mean < +3pp |
| PPO std | < 5% | 5-8% (考虑增seed到30) | > 10% (训练不稳定) |

#### Stage 3: 20-seed Full Run（~60-80h CPU）

| Table | 配置数 | Seeds | 预计时间 |
|-------|--------|-------|---------|
| Table III: 主结果 (7算法×5指标) | 7 | 20 | ~7h |
| Table IV: 协调机制 (Ind/Het/MAPPO) | 3 | 20 | ~7h |
| Table V: 平稳vs非平稳 | 2env × 4algo | 20 | ~18h |
| Table VI: NOMA vs OFDMA | 2 × 2algo | 20 | ~9h |
| Table VII: 消融 | 8变体 | 10-20 | ~10h |
| Figures | 收敛+轨迹+SINR | 20+1 | ~7h |

**消融seed分配（PhD#3建议）**:
- Hybrid vs Pure DRL: 20-seed（核心C2）
- K=1/2/3: 10-seed（差异大）
- ±Inter-UAV interference: 10-seed（开关性差异）
- ±Dynamic burst: 复用Table V数据

---

### Phase 1 最终数字验证（五博士标准）

**20-seed IQM + bootstrap 95% CI 必须达到**:

| 对比 | IQM gap | CI下界 | 判定 |
|------|---------|--------|------|
| **Ours vs BCD-greedy** | ≥ +10pp | > 0 | 🟢可发表 / 🔴<+8pp=危险 |
| **Ours vs MAPPO** | ≥ +8pp | > 0 | 🟢可发表 / 🔴<+5pp=危险 |
| **Ours vs BCD-static (非平稳)** | **≥ +8pp** | **> 0** | 🟢**C4成立** / 🔴<+3pp=C4崩溃 |
| Ours vs BCD-static (平稳) | ≥ 0 | — | 不为负即可 |
| Ours vs Single-UAV | ≥ +20pp | > 0 | 🟢多UAV必要性 |
| NOMA vs OFDMA | ≥ +3pp | > 0 | 🟢C1前提 / 🔴OFDMA胜=KILL |

**绝对值**:

| 指标 | 20-seed IQM | 🟢可发表 | 🟡危险 | 🔴必死 |
|------|-------------|---------|--------|--------|
| Ours deadline% (非平稳) | — | 70-85% | 60-70% | <60% |
| Ours avg delay | — | <0.85s | 0.85-1.0s | >1.0s |
| Ours Jain's fairness | — | >0.8 | 0.6-0.8 | <0.6 |
| UAV spatial separation | — | 200-400m | 100-200m | <100m (没分离) |

**统计方法**:
- IQM + bootstrap 95% CI（rliable库, n_bootstrap=50000）
- Checkpoint: **last-checkpoint统一标准**
- per-episode raw scores保存
- Table III: Bonferroni校正（α=0.05/6≈0.008）
- 消融(10-seed): Welch's t-test + Cohen's d

---

### Phase 1 存档

```json
// results/archive_phase1.json
{
    "phase": "1",
    "git_tag": "v3.2-phase1",
    "stat_method": "IQM + bootstrap 95% CI, 20-seed, last-checkpoint",
    "table_III_main_results": {
        "Ind-PPO":    {"deadline_iqm": 0, "ci": [0,0], "delay": 0, "fairness": 0, "energy": 0, "separation": 0},
        "Het-MAPPO":  {"deadline_iqm": 0, "ci": [0,0]},
        "MAPPO":      {"deadline_iqm": 0, "ci": [0,0]},
        "BCD-static": {"deadline_iqm": 0, "ci": [0,0]},
        "BCD-greedy": {"deadline_iqm": 0, "ci": [0,0]},
        "Single-UAV": {"deadline_iqm": 0, "ci": [0,0]},
        "HandCraft":  {"deadline_iqm": 0, "ci": [0,0]}
    },
    "table_V_stationary_vs_dynamic": {
        "stationary": {"ppo": 0, "bcd_static": 0, "gap": 0},
        "dynamic":    {"ppo": 0, "bcd_static": 0, "gap": 0},
        "bcd_degradation": 0,
        "ppo_degradation": 0
    },
    "table_VI_noma_vs_ofdma": {
        "noma_ppo": 0, "ofdma_ppo": 0, "advantage": 0
    },
    "paper_claim_validation": {
        "C1_inter_uav_interference_matters": false,
        "C2_hybrid_better_than_pure": false,
        "C3_independent_better_than_mappo": false,
        "C4_drl_adapts_better_than_bcd": false,
        "all_claims_supported": false
    },
    "decision": "PENDING",
    "frozen_files": ["全部算法代码"]
}
```

---

## Phase 2: 扩展层

### T9: K=3 (3-seed)

**Gate Check**:
| # | 检查项 | PASS | 说明 |
|---|--------|------|------|
| G2.t9.1 | K=3训练收敛 | learning curve上升 | 不崩溃即可 |
| G2.t9.2 | K=3 vs K=2 | K=3 ≥ K=2 或下降 < 5pp | 更多UAV应该更好或至少持平 |
| G2.t9.3 | K=3 > K=1 | gap > +15pp | 多UAV价值 |

**时间**: 1天

### T10: IC-CPPO (Bonus)

**3-seed gate**: adaptive λ不卡死 + IC-CPPO ≥ Ind-PPO → 加入C5
**不通过 → 放弃C5，论文按C1-C4四贡献提交（不影响75-85%概率）**

**时间**: 2天

---

### Phase 2 存档

```json
// results/archive_phase2.json
{
    "phase": "2",
    "git_tag": "v3.3-phase2",
    "k3_results": {
        "K1_3seed": {"values": [], "mean": 0},
        "K2_3seed": {"values": [], "mean": 0},
        "K3_3seed": {"values": [], "mean": 0}
    },
    "ic_cppo_gate": {
        "attempted": false,
        "passed": false,
        "values": []
    },
    "final_contribution_list": ["C1", "C2", "C3", "C4"],
    "decision": "PENDING"
}
```

---

## Kill条件汇总（红线）

| # | 条件 | 触发时机 | 后果 | 恢复方案 |
|---|------|---------|------|---------|
| **K1** | KKT后PPO 5000ep无上升趋势 | Phase 0A | 重新设计DRL-Optimizer接口 | 改为α_base直接输出，不乘KKT |
| **K2** | 非平稳PPO vs BCD-static < +3pp (3-seed) | Phase 0B | C4不可行 | 加大burst差异/加快移动 |
| **K3** | OFDMA ≥ NOMA (3-seed) | Phase 0B | C1前提崩溃 | 检查OFDMA实现是否正确 |
| **K4** | 20-seed PPO劣于BCD-static (任何环境) | Phase 1 | 论文方向崩溃 | 全面辩论 |
| **K5** | 20-seed Ours deadline% < 60% | Phase 1 | 绝对性能不足 | 调参/调环境 |
| **K6** | 论文4个claim任一不被数据支撑 | Phase 1后 | 降级投稿 | 重新组织narrative |

---

## 总时间线

```
Day 1-2:   Phase 0A (T7+T1) → 存档 archive_phase0a.json
Day 3-5:   Phase 0B (T2→T3→组合→T4/T5/T8) → 存档 archive_phase0b.json
Day 6-7:   Phase 1前半 (T6 Het-MAPPO + 3-seed gate + 7-seed Table V)
Day 8-11:  Phase 1后半 (20-seed全量实验) → 存档 archive_phase1.json
Day 12:    Phase 2 (K=3 + IC-CPPO gate) → 存档 archive_phase2.json
Day 13:    rliable后处理 + 全部图表生成
Day 14-23: 论文写作 (III→IV→VI→I→II→V→VII)
Day 24-25: 内部评审 + 修改

总计: 25个工作日（5周）
```

---

## 论文写作顺序（在Phase 1数据完成后开始）

```
Day 14-15: Section III (System Model) — 最核心，奠定C1
Day 16-17: Section IV (Framework) — C2+C3
Day 18-19: Section VI (Results) — 填入20-seed数据，制作图表
Day 20:    Section V (Setup) — 参数表+baseline描述
Day 21-22: Section I+II (Intro+Related Work) — 最后写，因为需要知道实验结论
Day 23:    Section VII (Conclusion) + Abstract
Day 24-25: 内部评审 + 一轮修改
```
