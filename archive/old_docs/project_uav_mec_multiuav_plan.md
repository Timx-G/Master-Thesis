---
name: Multi-UAV论文完整路线图
description: Multi-UAV MEC-NOMA论文，Independent PPO+Hybrid Optimizer，共享频段inter-UAV干扰，Phase进度追踪
type: project
originSessionId: 87f06c6d-5c5f-4975-bc1a-cc2a6fc9d27d
---
# Multi-UAV MEC 论文路线图（2026-04-11更新）

## 当前状态：Phase 2 通过，进入 Phase 3

## 关键决策记录

### MAPPO→Independent PPO转型（2026-04-11）
- MAPPO 3-seed: [69.5%, 60.6%, 69.0%] mean=66.4% ← 参数共享致命缺陷
- Independent PPO 3-seed: [81.1%, 73.4%, 79.6%] mean=78.1% ← 稳定且碾压
- MAPPO参数共享导致两UAV对相同观测产生相同动作→都飞向burst cluster→干扰爆炸
- Independent PPO通过不同网络权重自然分化出互补角色
- nearest-UAV归属规则本身提供隐式协调：一架飞近cluster→抢走用户→另一架自动服务其他区域

### 论文定位（确认）
**Title**: "Interference-Aware Hybrid DRL-Optimization for Cooperative Multi-UAV Trajectory Design and NOMA Task Offloading in MEC Networks"

**贡献**:
- C1: Multi-UAV MEC-NOMA with shared frequency + inter-UAV interference formulation (新问题)
- C2: Hybrid架构 — DRL(5维轨迹决策) + Optimizer(per-user解析解) (实用且有效)
- C3: Dual-constraint Lagrangian (delay + interference/fairness) with physical threshold
- C4: 实验证明DRL >> BCD (+14.8pp), 发现隐式协调 > 显式MARL

**CCF-B可行性依据**（论文搜索验证）:
- IEEE IoTJ 2024: MATD3(2018算法) + multi-UAV MEC = 发表
- Expert Systems 2025: MDQN(2015算法) + UAV-NOMA-MEC = 发表
- Computer Networks 2024: MADDPG(2017算法) + UAV trajectory = 发表
- 规律：wireless/MEC的CCF-B不要求算法创新，要求问题建模+实验

## 技术架构

### 环境 V3 (800m, Multi-UAV)
- 800m × 800m, 3 clusters, N=20 users
- K=2 UAVs, f_per_uav=4GHz, **共享5MHz频段**
- 动态burst: 每20slot轮流, burst=0.9, cool=0.2
- 用户归属: nearest-UAV
- Inter-UAV干扰: SIC只消本UAV内干扰，跨UAV不可消除

### 算法: Independent PPO + Hybrid Optimizer
- Per-UAV观测: 9维 [自身xy, 队友xy, 3集群负载, 电池, 时间]
- Per-UAV动作: 5维 [dx, dy, cluster_weight×3]
- Optimizer: per-user α(channel-based) + p(water-filling)
- 各UAV独立训练，无通信，无共享critic

### Baselines
1. Single-UAV BCD (~48%)
2. Multi-UAV BCD-static (76.6%)
3. Multi-UAV BCD-greedy (63.3%)
4. Hand-crafted巡逻 (72%)
5. MAPPO-参数共享 (66.4% — 失败对照)
6. Independent PPO + Hybrid (**78.1% — ours**)

## Phase进度

### Phase 1: 环境搭建 ✅ 通过
- multi_uav_env.py + compute_multi_uav_rates()
- Bug修复: _assign_users()同位置全归UAV0
- G0: 干扰-12.9pp ✅ | G1: 2UAV>1UAV +8.1pp ✅ | G2: greedy-13.3pp ✅

### Phase 2: DRL训练 ✅ 通过
- MAPPO失败(参数共享) → Independent PPO成功
- **3-seed验证**: [81.1%, 73.4%, 79.6%] mean=78.1% std=3.3%
- vs BCD-greedy: **+14.8pp** ✅✅✅
- vs BCD-static: +1.5pp ✅
- vs Hand-craft: +6.1pp ✅

### Phase 3: 约束+全量 ← 下一步
- 加IC-CPPO: delay constraint(per-UAV λ_d) + interference constraint(shared λ_I)
- 20 seeds全量实验
- 消融: Hybrid vs Pure, Independent vs MAPPO, 2UAV vs 1UAV

### Phase 4: 论文重写
### Phase 5: 内部评审+投稿

## 核心洞察
1. Inter-UAV干扰是optimizer结构性无法解决的（叠加性，非单用户可调）
2. 共享频段+两UAV靠近=SINR崩溃→空间分离=trajectory必须→DRL价值
3. 隐式协调(interference-aware reward + nearest-UAV)> 显式MARL(MAPPO)
4. Hybrid DRL+Optimizer = 各司其职（DRL做轨迹，optimizer做per-user分配）
