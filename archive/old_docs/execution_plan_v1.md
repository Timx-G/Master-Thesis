# Multi-UAV MEC-NOMA 论文执行计划 v1

> **基于三博士辩论收敛**
> **日期**: 2026-04-12
> **状态**: 待批准后严格执行

---

## 执行原则

1. **先固地基再盖楼**：环境基础设施必须先稳定，再做算法工作
2. **环境改动与算法改动解耦**：不在不稳定环境上调算法
3. **旧数据全部作废**：Phase 2的78.1%/76.6%在P0完成后不再引用
4. **相对门槛**：gate check用DRL vs BCD-static的相对差距，不用绝对数字
5. **三阶段统计**：关键实验走3-seed→7-seed→20-seed
6. **每个Phase有git tag**：完成后打tag冻结代码，不可回退修改

---

## Phase依赖图

```
Phase 0A: 地基层（串行）
  T7(固定cluster) → T1(KKT optimizer)
  │
  ├── gate check: 1-seed sanity + 3-seed baseline建立
  │
Phase 0B: 环境真实化（T1之后，内部可并行）
  ├── T2(用户移动2m/s)     ← 单独验证
  ├── T3(非平稳burst)      ← 单独验证
  ├── T4(OFDMA baseline)   ← 独立模块
  ├── T5(BCD代码实现)      ← 依赖T1
  └── T8(SINR logging)     ← 依赖T1
  │
  ├── gate check: T2/T3分别验证 → 组合验证 → 3-seed新baseline
  │
Phase 1: 算法层（依赖0A+0B全部通过）
  T6(Het-MAPPO) + Independent PPO在新环境下的3-seed→7-seed→20-seed
  │
Phase 2: 扩展层
  T9(K=3, 3-seed) + [T10 IC-CPPO bonus]
```

---

## Phase 0A: 地基层

### T7: 固定Cluster Centers

**目标**: 消除跨seed的拓扑差异，使实验可复现

**实现方案**:
```python
# 替换 multi_uav_env.py 中 reset() 的 cluster 生成逻辑
# 旧: centers = self.rng.uniform(...)
# 新: 使用固定的3个cluster中心

FIXED_CLUSTER_CENTERS = np.array([
    [200.0, 200.0],   # Cluster 0: 左下
    [600.0, 200.0],   # Cluster 1: 右下
    [400.0, 600.0],   # Cluster 2: 上中
])
# 三角形布局，最小间距 ~283m，确保UAV需要移动才能服务不同cluster
```

**用户分配**: 保持cluster内高斯随机（std=50m），但cluster归属按均匀分配（每cluster约6-7人）

**Gate Check**:
- [ ] 20个不同seed下cluster centers完全一致（`np.allclose`）
- [ ] 用户总体分布可视化合理（3团明显分离）
- [ ] env.reset()前后cluster不变

**预计时间**: 0.5天

---

### T1: KKT Closed-Form Optimizer

**目标**: 将 `_compute_per_user_actions()` 从启发式替换为严格闭式解

**接口定义**（辩论收敛，代码实现前必须确认）:

DRL仍输出5维: `[dx, dy, cw0, cw1, alpha_base]`

Optimizer接收:
- `alpha_base` (DRL输出): 总体offloading倾向scaling factor ∈ [0.05, 0.95]
- `cluster_weights` (DRL输出): 资源分配优先级
- `h_{i,k}` (物理量): 用户i到UAV k的信道增益
- `f_alloc` (物理量): 分配给用户i的计算资源

**卸载比闭式解**:
```
# 令 ∂τ/∂α = 0，解出最优卸载比
# τ_i = max((1-α)C/f_local, αD/R + αC/f_alloc)
# 最优解: 令两项相等
#   (1-α*)C/f_local = α*D/R + α*C/f_alloc
#   α* = f_local·R / (f_local·R + f_alloc·D_ratio + f_local·D/C·f_alloc/R)
# 简化: α* = f_local·R / (f_local·R + C·f_local/(C/f_alloc) + D·f_local/R)
# 最终:
α_kkt = (f_local * R_i) / (f_local * R_i + f_alloc_i * (D_i / R_i * f_local + C_i / f_alloc_i * f_local))

# DRL的调节:
α_final = clip(alpha_base * α_kkt, 0, 1)
```

**功率分配闭式解**:
```
# 水注法: 给定SIC解码顺序(按 h_i·p_i 降序)
# KKT条件: p_i* = max(0, μ_k - σ²/h_{i,k})
# μ_k 由总功率约束 Σp_i ≤ N·p_max 决定
```

**Gate Check**:
- [ ] KKT互补松弛条件残差 < 1e-6（单用户验证）
- [ ] 单用户case: 闭式解 vs scipy.optimize数值解，偏差 < 1%
- [ ] 替换后1-seed 100ep: reward无NaN、无发散
- [ ] 打印reward统计（min/max/mean/std）对比旧环境
- [ ] 3-seed Independent PPO: 能收敛（learning curve有单调上升趋势）

**预计时间**: 2天

**Phase 0A整体Gate**:
- [ ] T7 + T1 都通过各自gate
- [ ] 3-seed建立新baseline数字：记录 R_ppo_new, R_random_new
- [ ] `git tag v3.0-phase0a`

---

## Phase 0B: 环境真实化

### 串行验证顺序: T2单独 → T3单独 → T2+T3组合 → T4/T5/T8并行

### T2: 用户移动模型 (2m/s随机游走)

**实现方案**:
```python
# 在 step() 中，每个时隙更新用户位置
# 随机游走: 每slot随机方向，步长=v_user*dt
for i in range(N):
    angle = self.rng.uniform(0, 2*np.pi)
    dx = self.cfg.user_speed * self.cfg.dt * np.cos(angle)
    dy = self.cfg.user_speed * self.cfg.dt * np.sin(angle)
    self.user_positions[i] += [dx, dy]
    # 边界反弹
    self.user_positions[i] = np.clip(self.user_positions[i], 0, self.cfg.area_size)
```

**Gate Check**:
- [ ] 用户轨迹可视化: 100slot后用户位置偏移约200m(=2m/s×100s)，合理
- [ ] 边界处理正确（不越界）
- [ ] 3-seed PPO（仅加移动，不加burst）: 记录 delta_mobility = R_ppo_mobility - R_ppo_new

**预计时间**: 0.5天

---

### T3: 非平稳Burst机制

**实现方案**:
```python
# 替换确定性轮转 burst_idx = (slot // 20) % 3
# 改为随机周期+随机选择
if self.slot >= self.next_burst_switch:
    self.burst_cluster = self.rng.integers(0, self.cfg.num_clusters)
    # 下次切换时间: U[8, 15] slots
    self.next_burst_switch = self.slot + self.rng.integers(
        self.cfg.burst_period_min, self.cfg.burst_period_max + 1)
```

新增config参数:
- `burst_period_min`: 8
- `burst_period_max`: 15
- `dynamic_burst_random`: True (新flag，与旧的`dynamic_burst`区分)

**Gate Check**:
- [ ] Burst序列FFT无明显周期峰（vs旧实现的20-slot峰）
- [ ] Burst间隔分布可视化: 确认U[8,15]
- [ ] 3-seed PPO（仅加burst，不加移动）: 记录 delta_burst = R_ppo_burst - R_ppo_new
- [ ] 3-seed BCD-static（仅加burst）: 记录 delta_burst_bcd
- [ ] **关键验证**: delta_burst_bcd > delta_burst_ppo（BCD受burst影响更大）

**预计时间**: 0.5天

---

### T2+T3组合验证

**Gate Check**:
- [ ] 3-seed PPO（移动+burst同时开启）: 记录 R_ppo_dynamic
- [ ] 3-seed BCD-static（移动+burst同时开启）: 记录 R_bcd_dynamic
- [ ] **核心gate G_dynamic**: R_ppo_dynamic - R_bcd_dynamic > 5pp
- [ ] 如果 < 5pp → 触发辩论（三博士重新评估C4是否可行）
- [ ] 如果 > 10pp → C4论文杀手锏确认

**预计时间**: 0.5天（跑实验）

---

### T4: OFDMA Baseline (可与T2/T3并行)

**实现方案**:
在 `model.py` 中新增 `compute_multi_uav_ofdma_rates()`:
```python
def compute_multi_uav_ofdma_rates(self, uav_positions, user_positions,
                                   user_assignment, powers, alphas, K):
    """OFDMA: 每UAV分得 B/K 带宽，每UAV内再按用户数均分，无多用户干扰"""
    bw_per_uav = self.bandwidth / K
    for k in range(K):
        users_k = [i for i in range(N) if user_assignment[i] == k]
        bw_per_user = bw_per_uav / max(len(users_k), 1)
        for i in users_k:
            snr = powers[i] * h[i,k] / self.noise_power
            rates[i] = bw_per_user * np.log2(1 + snr)
    return rates
```

**Gate Check**:
- [ ] OFDMA模式下SINR公式无用户间干扰
- [ ] OFDMA rate < NOMA rate（在同等功率下，验证NOMA的频谱效率优势）

**预计时间**: 0.5天

---

### T5: BCD-static/greedy代码实现

**实现方案**: 新建 `uav_mec/baselines_multi.py`

```python
class BCDStatic:
    """Block Coordinate Descent for Multi-UAV:
    交替优化 UAV位置(固定资源) 和 资源分配(固定位置)，迭代至收敛"""

    def optimize_one_slot(self, env_state):
        # Block 1: 固定(α,p)，优化UAV位置（梯度下降或网格搜索）
        # Block 2: 固定UAV位置，优化(α,p)（调用KKT闭式解）
        # 迭代直到目标函数变化 < epsilon

class BCDGreedy:
    """贪心BCD: UAV总飞向当前负载最重的cluster"""
```

**Gate Check**:
- [ ] BCD-static在静态场景下收敛（目标函数单调不增）
- [ ] 3-seed可复现，结果稳定（std < 2%）

**预计时间**: 1天

---

### T8: SINR Logging

**实现方案**: 在 `step()` 的 `info` dict 中添加:
```python
info["sinr_db"] = 10 * np.log10(sinr_linear + 1e-20)  # per-user SINR
info["i_inter"] = i_inter_per_user     # per-user inter-UAV interference
info["uav_separation"] = np.linalg.norm(uav0_pos - uav1_pos)
info["per_episode_deadline_rates"] = [...]  # for rliable
```

**Gate Check**:
- [ ] SINR值在合理范围（-10dB ~ 30dB）
- [ ] 可导出为CSV用于Fig.5 CDF

**预计时间**: 0.5天

---

### Phase 0B整体Gate:
- [ ] T2, T3, T4, T5, T8 各自gate通过
- [ ] T2+T3组合gate通过（G_dynamic > 5pp）
- [ ] `git tag v3.1-phase0b`
- [ ] **冻结环境代码**，后续Phase禁止修改multi_uav_env.py

---

## Phase 1: 算法层 + 全量实验

### 前置条件: Phase 0A + 0B 全部通过

### T6: Het-MAPPO实现

**架构设计** (PhD#2建议采纳):
- 每UAV独立Actor（不共享参数）
- 共享Critic: 输入 = concat([embed_uav0, embed_uav1, global_features])
- 每UAV的embedding: 局部obs → MLP(256,128) → 64维embedding
- Critic warmup: 前200 episodes只更新critic

**初始化策略**: 用Phase 0B训好的Independent PPO actor作为Het-MAPPO的actor初始化

**Plan B**: 如果Het-MAPPO 2周内不收敛，退回到parameter-sharing MAPPO + heterogeneous observation

**Gate Check**:
- [ ] 简化环境(固定信道)下Het-MAPPO能收敛
- [ ] 完整环境3-seed: Het-MAPPO vs Independent PPO，记录差距

**预计时间**: 1.5天

---

### 全量实验时间线

```
3-seed gate (所有算法):           ~3h
  └─ 确认所有算法能跑通，数字合理

7-seed intermediate (Table V only): ~3h
  └─ G_dynamic gap > 8pp → 继续
  └─ G_dynamic gap 5-8pp → 可接受但调整narrative
  └─ G_dynamic gap < 5pp → 触发辩论，重新评估C4

20-seed full run:                   ~60-80h CPU
  └─ Table III: 7算法 × 5指标      ~7h
  └─ Table IV: 3 MARL对比           ~7h
  └─ Table V: 平稳 vs 非平稳        ~18h
  └─ Table VI: NOMA vs OFDMA         ~9h
  └─ Table VII: 消融                 ~10h
  └─ Figures: 收敛+轨迹              ~7h

rliable后处理:                      ~1h
  └─ IQM + bootstrap CI + performance profile
```

### 统计方法 (PhD#3建议采纳):
- 20-seed: IQM + bootstrap 95% CI (rliable库)
- 10-seed消融: Welch's t-test + Cohen's d
- Checkpoint策略: **last-checkpoint**（统一标准，不做best-selection）
- 保存per-episode raw scores供rliable使用
- Bonferroni校正: Table III的6个pairwise comparisons

**Phase 1 Gate**:
- [ ] 20-seed数据完整
- [ ] Table V: DRL vs BCD-static在非平稳环境gap > 8pp (bootstrap CI下界 > 0)
- [ ] Table III: Independent PPO在所有主指标上不劣于任何baseline
- [ ] `git tag v3.2-phase1`

---

## Phase 2: 扩展层

### T9: K=3扩展 (3-seed)

**实现**: 扩展multi_uav_env.py支持K=3，可能需要:
- 新增3号UAV相关obs维度
- 3-cluster对3-UAV的对称布局
- 碰撞约束扩展为3对

**Gate Check**:
- [ ] K=3训练不崩溃
- [ ] K=3性能 > K=2（更多UAV应该更好）或性能下降 < 15%（可在论文中讨论）

**预计时间**: 1天

### T10: IC-CPPO (Bonus, 仅在时间充裕时做)

**3-seed gate check**:
- [ ] adaptive λ_d收敛且不卡在λ_max
- [ ] IC-CPPO > Independent PPO（有正增益）
- [ ] 如果3-seed不通过 → 放弃C5，论文按C1-C4四贡献发

**预计时间**: 2天

---

## 总时间线

```
Day 1-2:   Phase 0A (T7固定cluster + T1 KKT optimizer)
Day 3-4:   Phase 0B (T2移动 + T3 burst + T4 OFDMA + T5 BCD + T8 SINR)
Day 5:     Phase 0B组合验证 + 3-seed gate check全部算法
Day 6-7:   Phase 1 (T6 Het-MAPPO + 7-seed intermediate for Table V)
Day 8-10:  20-seed全量实验 (可后台并行跑)
Day 11:    Phase 2 (K=3 + IC-CPPO gate)
Day 12:    rliable后处理 + 图表生成
Day 13-22: 论文写作 (III→IV→VI→I→II→V→VII)
Day 23-25: 内部评审 + 修改
```

**总计: ~25个工作日（5周）到投稿就绪**

---

## Kill条件（任何一条触发则停下辩论）

| # | 条件 | 触发时机 | 后果 |
|---|------|---------|------|
| K1 | KKT optimizer后PPO完全不收敛（5000ep仍无上升趋势） | Phase 0A | 重新设计DRL-Optimizer接口 |
| K2 | 非平稳环境下DRL vs BCD-static gap < 3pp (7-seed) | Phase 1 | 重新评估C4，可能需要改环境设计 |
| K3 | 20-seed实验中Independent PPO劣于BCD-static | Phase 1 | 触发全面辩论，重新评估论文方向 |
| K4 | 论文写作时发现实验数据无法支撑任何一个核心贡献 | Phase 2+ | 降级投稿目标或重新设计实验 |
