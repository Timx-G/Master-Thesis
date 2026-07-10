# 理论设计：DRL动态NOMA分组+卸载联合优化

> **状态**: 待审查。审查通过后才允许写代码。
> **日期**: 2026-04-12

---

## 1. 场景概述

- 800m × 800m 区域，K=2 UAV 固定悬停（不移动）
- N=20 地面用户，3个固定cluster，**用户以2m/s随机游走**
- 5MHz OFDMA分成 S=5 子载波，每子载波放2用户NOMA pair
- 两UAV共享全部5子载波（inter-UAV干扰存在）
- 非平稳burst: 随机周期U[8,15]切换burst cluster
- **DRL决策**: 每slot为每UAV选择哪些用户服务、如何配对、每用户卸载比α

## 2. 为什么DRL有价值

| 决策 | 为什么不能用静态/贪心 |
|------|---------------------|
| 用户分组 | 用户移动→信道时变→最优分组每slot变化。贪心(按瞬时信道排序)是myopic的，不考虑task queue状态 |
| 卸载比α | KKT闭式解假设单slot最优，不考虑queue中等待的任务。DRL可以学习跨slot的α调度策略 |
| 联合优化 | 分组和α耦合（选不同用户→不同SINR→不同最优α），贪心解耦不如联合优化 |

**DRL的核心优势**: sequential decision making。在时变信道+task queue动态下，当前slot的分组决策影响未来slot的queue状态。贪心只看当前slot，DRL看trajectory。

## 3. MDP定义

### 3.1 State Space (per agent, 即per UAV)

每个UAV观测到：

| 维度 | 含义 | 范围 | 说明 |
|------|------|------|------|
| 2 | own_xy / L | [0,1] | 固定不变（悬停），但仍提供位置信息 |
| 2 | other_xy / L | [0,1] | 另一UAV位置 |
| 3 | cluster_loads (归一化) | [0,1] | 每cluster的active任务比例 |
| 1 | battery / budget | [0,1] | 电池剩余（悬停也耗电） |
| 1 | time_frac | [0,1] | 剩余时间比例 |

**Total per agent: 9维，joint: 18维**（与当前一致，无需改obs space）

注意：用户级信息（谁有任务、谁的信道好）**没有在obs中**。DRL只能通过cluster_loads间接感知。这是一个设计选择：

- **Option A**: obs只有9维（当前方案）→ DRL只能学cluster级策略，per-user细节靠启发式
- **Option B**: obs加入per-user信道状态（20×1 = +20维）→ DRL能学per-user策略，但obs变大

**建议先用Option A**。如果gap不够再升级到Option B。

### 3.2 Action Space

**这是最关键的设计决策。**

#### 方案A: 连续优先级分数 + alpha_base（推荐）

每UAV输出:
```
action_k = [priority_score_0, ..., priority_score_9,  alpha_base]
           |<------- 10维 user优先级 ------->|        |1维|
```
- `priority_score_i` ∈ [-1, 1]: 对本UAV归属的第i个用户的服务优先级
  - 按分数排序，取top-S*2=10个（实际可能不到10个active）
  - 配对：分数最高和最低配对（最大化SIC增益差）
- `alpha_base` ∈ [-1, 1] → mapped to [0.05, 0.95]: 全局卸载倾向

**Per-agent action dim: 11**
**Joint action dim: 22**

**问题**: 每UAV最多10个归属用户，但不同slot归属用户可能不同（因为nearest-UAV assignment随用户移动变化）。Priority score是对**固定位置的slot内归属用户**排序，不是对全局20用户。

**解决方案**:
- DRL输出10维priority score（对应本UAV当前slot的最多10个归属用户）
- 如果归属用户<10个，多余的score被忽略
- 归属用户>10个不会发生（20用户/2UAV=10）

#### 方案B: 直接输出per-user alpha（更简单）

每UAV输出:
```
action_k = [alpha_0, alpha_1, ..., alpha_9]
           |<------- 10维 per-user α ------->|
```
- `alpha_i` ∈ [0, 1]: 直接作为第i个归属用户的卸载比
- alpha=0 → 用户不offload（等效于"不选入分组"）
- alpha>0 → 用户offload，按alpha大小排序进入subcarrier pair

**Per-agent action dim: 10**
**Joint action dim: 20**

**优势**: 更直接，alpha本身就是分组+offload的联合决策（alpha=0即不分组）
**劣势**: 10维连续，PPO可能探索效率低

#### 方案C: 紧凑版——3维（最小可行）

每UAV输出:
```
action_k = [cluster_0_weight, cluster_1_weight, alpha_base]
```
- cluster weights决定per-cluster的服务优先级（哪个cluster的用户先服务）
- 按cluster weight × channel gain排序选top-10
- alpha_base同前

**Per-agent action dim: 3**
**Joint action dim: 6**（与当前完全一致！）

**优势**: 动作空间最小，PPO最容易收敛
**劣势**: per-user粒度不够，可能不如方案A/B

### 3.3 推荐: 先方案C，不行再升级方案A

理由：
1. 方案C与当前代码几乎相同（action dim=6），改动最小
2. 先验证"cluster级分组策略"是否足够
3. 如果gap不够，再升级到方案A的per-user优先级

### 3.4 Transition: 环境step()逻辑

```
每个slot:
  1. 用户移动 (2m/s随机游走) → 信道变化
  2. 新任务到达 (burst pattern)
  3. 用户→UAV归属 (nearest-UAV)
  4. DRL输出action → 解析为分组策略
  5. 分组:
     - 每UAV的归属用户按 (cluster_weight[cid] × channel_gain) 排序
     - Top-10选入served set
     - Served users配对: 最强配最弱 (maximize SIC gain gap)
  6. Rate计算: compute_grouped_noma_rates (per-pair SINR + inter-UAV干扰)
  7. Alpha: alpha_base × KKT_alpha (per served user)
  8. Delay计算: served用户走offload path, unserved走local-only
  9. Task queue更新: 满足deadline的任务清除, 超时的累积age
  10. Reward计算
```

### 3.5 Reward

```
reward = (met_tasks / total_tasks) - 0.01 * e_hover / 128.0
```
- met_tasks: 本slot满足deadline的任务数
- total_tasks: 本slot有active任务的用户数
- e_hover: 悬停能耗（固定值，不影响优化方向，但保持energy-aware建模）
- 不需要collision penalty（UAV不移动）

### 3.6 Episode终止

- T=100 slots（不变）
- 电池耗尽终止（不太可能，悬停100s消耗~8000J/9000J预算）

## 4. Baselines

| # | Baseline | 分组策略 | α策略 | 说明 |
|---|----------|---------|-------|------|
| 1 | **Greedy** | 每slot按信道排序top-10 + 强弱配对 | KKT α | 最强启发式 |
| 2 | **Random** | 每slot随机选10用户 + 随机配对 | KKT α | 下界 |
| 3 | **Round-Robin** | 每slot轮流选不同10用户 | KKT α | 公平性baseline |
| 4 | **Fixed-Best** | 贪心分组 | 固定α=0.95 | 上一轮发现的"simple best" |
| 5 | **Hungarian** | 匈牙利算法最优匹配（per-slot穷举） | KKT α | 计算上界（可能慢） |

## 5. 关键验证指标

**Gate check (4月26日)**:
```
DRL(动态分组+α) vs Greedy(启发式分组+KKT α) ≥ +5pp
在用户移动(2m/s) + 非平稳burst的动态环境中
```

**为什么这次可能work**:
- 贪心每slot独立决策（myopic），不考虑queue中等待的任务
- 用户移动使当前slot的分组影响未来slot的queue状态（序贯决策）
- burst切换时，贪心需要多个slot适应（重新排序），DRL可以预判

## 6. 代码改动清单

| 文件 | 改动 | 量级 |
|------|------|------|
| `multi_uav_env.py` | action解析从轨迹改为分组；去掉UAV移动；分组逻辑 | 中 |
| `model.py` | compute_grouped_noma_rates接受外部pairing参数 | 小 |
| `baselines_multi.py` | 从轨迹baseline改为分组baseline | 中 |
| `agents/ppo.py` | 不变（改obs/act dim即可） | 0 |

## 7. 理论-代码对应表（审查清单）

| 理论公式/概念 | 代码位置 | 验证方法 |
|--------------|---------|---------|
| SINR_{i,k} = p_i·h_{i,k} / (pair干扰 + cross-UAV干扰 + noise) | model.py:compute_grouped_noma_rates | 单pair手算对比 |
| α* = C/f_local / (C/f_local + D/R + C/f_alloc) | env._compute_alphas | bisection数值解对比 |
| 分组：cluster_weight × channel_gain排序 | env.step() 分组逻辑 | 手工构造case验证 |
| Reward = met/total | env.step() reward | 手动计数对比 |
| 用户移动: 2m/s随机游走 | env.step() mobility | 100slot后偏移~200m |
| Burst: 随机周期U[8,15] | env.step() burst | FFT无周期峰 |
| f_alloc = f_per_uav / n_served | env._compute_alphas & delay | 两处一致（BUG-C修复） |
