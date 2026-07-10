# Multi-UAV MEC-NOMA 论文大纲（终版 v2）

> **基于五博士评审 + 工程师反馈修正**
> **日期**: 2026-04-12
> **目标**: IEEE IoTJ (CCF-B), ~13页双栏

---

## 逻辑主线（修正后）

```
C1(干扰建模 + 动态非平稳环境)
    ├──→ C2(Hybrid分层: DRL轨迹 + KKT闭式解资源)
    │        └──→ C3(实验发现: 隐式协调 > 显式MARL)
    │
    └──→ C4(实验发现: DRL在动态环境中远超BCD)
         ← 直接从C1的"动态性"分支出来

[可选bonus] C5: IC-CPPO dual constraint + λ_max (gate check通过再加)
```

**修正要点**:
- 非平稳性嵌入C1的问题建模（System Model里就定义动态burst+用户移动）
- C4不是"我们发明了适应非平稳的方法"，而是"在我们建模的真实动态场景下，DRL远超BCD"
- C3→C4之间不再有因果关系，两者平行从C1分支出来
- Plan B变Plan A：直接按C1-C4四贡献规划，IC-CPPO作为bonus

---

## Title

"Cooperative Multi-UAV Trajectory Design for NOMA-MEC Networks:
A Hybrid DRL-Optimization Approach with Inter-UAV Interference Management"

---

## 贡献清单（4个核心 + 1个可选）

| # | 贡献 | 性质 | 核心证据 |
|---|------|------|---------|
| C1 | Multi-UAV共频NOMA干扰建模 + 动态非平稳环境定义 | 问题建模 | SINR公式 + I_inter物理分析 |
| C2 | Hybrid DRL(5维轨迹) + KKT闭式解(per-user α,p) | 系统架构 | 可分离结构证明 + Proposition |
| C3 | Independent PPO > MAPPO: 隐式协调 > 显式MARL | 实验发现 | Table IV + Fig.6轨迹可视化 |
| C4 | 动态环境下DRL适应性远超BCD(+10pp+) | 实验发现 | Table V + Fig.7时序图 |
| [C5] | IC-CPPO dual constraint + λ_max物理推导 | 理论(bonus) | 仅gate check通过时加入 |

---

## 页数预算

| 节 | 页数 | 图表预算 |
|----|------|---------|
| Abstract + Keywords | 0.3 | — |
| I. Introduction | 1.5 | — |
| II. Related Work | 1.5 | Table I (文献对比矩阵) |
| III. System Model & Problem Formulation | 2.0 | Fig.1 (系统图) |
| IV. Proposed Framework | 1.8 | Algorithm 1 |
| V. Experimental Setup | 0.8 | Table II (参数表) |
| VI. Results & Analysis | 3.5 | Fig.2-8, Table III-VII |
| VII. Conclusion | 0.3 | — |
| References (~55篇) | 1.3 | — |
| **Total** | **~13.0** | **8 fig + 7 table** |

---

## Section I: Introduction (1.5页)

**核心任务**: 建立"为什么这个问题值得研究"的论证链

### 内容要点

**Para 1-2: 背景** (0.3页)
- IoT应用爆发 → MEC必要 → 固定基站不足(灾害/临时场景)
- UAV-MEC的天然优势: 机动性 + LoS信道 + 按需部署
- 引用经典survey定调

**Para 3-4: 从单UAV到多UAV + NOMA** (0.4页)
- 单UAV在800m级区域覆盖不足 → 多UAV协作是必然
- NOMA提升频谱效率，但多数multi-UAV工作假设正交频段
- **核心声称**: 当K架UAV共享同一频段时，跨UAV干扰I_inter不可SIC消除，使轨迹设计从"覆盖优化"质变为"干扰管理"

**Para 5: 同时，真实环境是动态非平稳的** (0.15页)
- 任务负载burst式变化、用户位置移动
- 传统BCD求解单slot子问题，无法处理时序决策的长期影响
- → 动态性是问题建模的内禀特征，不是附加贡献

**Para 6: 现有方法的三重不足** (0.2页)
- (a) BCD/SCA: 静态优化，无法适应非平稳(→C4)
- (b) 端到端MARL(MAPPO): 参数共享→策略同质化→两UAV同飞→干扰爆炸(→C3)
- (c) 42维端到端DRL: 维度诅咒，但per-user分配其实可闭式精确求解(→C2)

**Para 7: 本文方案与贡献清单** (0.25页)
- 列出C1-C4（C5仅在实现时提及）
- 一句话总结每个贡献

**Para 8: 论文组织** (0.1页)

### 防御策略
| 审稿人可能问 | 在哪防御 |
|-------------|---------|
| "为什么不频分？" | Para 3: 5MHz单载波极限，频分=每UAV仅2.5MHz，后续Table VI有NOMA vs OFDMA实验 |
| "Multi-UAV MEC不新" | Para 4: 新在shared-spectrum inter-UAV interference，配合Table I gap矩阵 |
| "PPO不是新算法" | Para 6(c): 创新在问题分解洞察和实验发现，不在PPO本身 |

---

## Section II: Related Work (1.5页)

**核心任务**: 为C1-C4各建立文献空白

### II-A. UAV-Assisted Mobile Edge Computing (0.4页)
- 单UAV → multi-UAV的演进 [1][2][3][6][8][11][12][15]
- 强调大多数multi-UAV MEC假设正交频段或忽略跨UAV干扰
- **落脚**: 共享频谱下的跨UAV干扰建模是空白 → **C1**

### II-B. NOMA-Enabled UAV-MEC (0.3页)
- UAV+NOMA+MEC组合 [21][22][24][27]
- 现有工作集中在单UAV NOMA 或 multi-UAV但无跨UAV干扰
- [21] Liu 2025是最接近的竞争者——需明确说明差异（shared-spectrum + DRL + 动态环境）
- **落脚**: multi-UAV + NOMA + shared spectrum交叉点几乎无人探索 → **C1**

### II-C. DRL and Multi-Agent RL for UAV Networks (0.4页)
- DRL用于UAV轨迹/资源 [9][10][49][50]
- MARL方法: MAPPO, MADDPG, CTDE范式
- Hybrid/分层DRL兴起 [51][52][61][62]
- 参数共享局限性 [41][42][43][44]
- **落脚**: (1) 端到端DRL在高维失败→需分层 → **C2**; (2) 参数共享在异构角色下适得其反 → **C3**

### II-D. Constrained DRL for Wireless (0.3页)
- CMDP/Safe RL综述 [39], CPO [Achiam], RCPO [Tessler]
- 无线场景应用 [28][29][30][31]
- 理论进展 [32][33][34]
- **落脚**: 约束DRL在无线中活跃，但λ上界缺乏物理解释 → **[C5 if applicable]**
- 注：若C5不启用，本节缩短为0.15页，仅作为背景

### 关键物件: Table I

| 文献 | Multi-UAV | NOMA | Shared-Spectrum | Inter-UAV Interference | DRL | Dynamic Env |
|------|:---------:|:----:|:---------------:|:----------------------:|:---:|:-----------:|
| [1] Ma 2024 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| [21] Liu 2025 | ✓ | ✓ | ? | ✗ | ✗ | ✗ |
| [9] Zhou 2024 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| ... | | | | | | |
| **Ours** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

---

## Section III: System Model & Problem Formulation (2页)

**核心任务**: 严格建模，建立C1的问题贡献

### III-A. Network Model (0.4页)
- 区域: 800m × 800m
- UAV: K=2 (扩展实验K=3), 固定高度100m, v_max=20m/s
- 用户: N=20, 3个**固定**cluster中心 (消除跨seed问题差异)
- **动态非平稳建模**（嵌入C1，非独立贡献）:
  - 任务burst: 每周期随机选择burst cluster，burst率0.9，其他0.2
  - 用户移动: 2m/s随机游走（**需要新增代码实现**）
- UAV运动学: q_k[t+1] = q_k[t] + v_k[t]·Δt, 碰撞约束 ||q_i-q_j|| ≥ 30m
- 电池约束: B_k[0]=9000J, 含悬停/飞行/计算功耗

### III-B. NOMA Communication with Inter-UAV Interference (0.6页) ← **C1物理核心**
- 信道模型: 3GPP TR 36.777 UAV LoS概率 + 路径损耗
- 上行NOMA + SIC:
  ```
  SINR_{i,k} = p_i · h_{i,k} / (I_intra + I_inter + σ²)

  I_intra = Σ_{j∈S_k, π(j)>π(i)} ε · p_j · h_{j,k}    (imperfect SIC, ε=0.02)
  I_inter = Σ_{k'≠k} Σ_{j∈S_{k'}} p_j · h_{j,k}         (跨UAV，不可SIC消除)
  ```
- 上行速率: R_i = B · log₂(1 + SINR_i)

- **Remark 1** (物理核心洞察):
  I_inter与UAV间距直接相关。两UAV距离d, 用户密度ρ时:
  I_inter ∝ ρ · P_avg / d^α
  → 两UAV靠近同一cluster → I_inter暴增 → 所有用户SINR崩溃
  → **空间分离是干扰管理的必要条件**
  → **轨迹设计 = 干扰管理**

### III-C. Task Model & Delay Analysis (0.4页)
- 任务: (D_i, C_i, t_max_i), 部分卸载 α_i ∈ [0,1]
- 时延:
  ```
  τ_i = max(τ_local, τ_offload)
      = max((1-α_i)·C_i / f_local,  α_i·D_i / R_i + α_i·C_i / f_alloc)
  ```
- MEC算力分配: f_alloc按信道质量加权

### III-D. CMDP Formulation (0.6页)
- 状态空间: 每UAV 9维 [自身xy(2), 队友xy(2), 3集群负载(3), 电池(1), 时间(1)]
- 动作空间: 每UAV 5维 [dx, dy, cluster_weight_0, cluster_weight_1, α_base]
- 奖励: throughput-based (deadline满足的任务数)
- 约束: E[avg_delay] ≤ τ*, 电池 ≥ 0, 碰撞回避

- **NP-hardness** (一句话继承):
  "Multi-UAV联合问题包含Phase A的单UAV单slot子问题作为特例（令K=1, T=1），
  后者已被证明NP-hard（via MaxFS归约[我们的先前工作]），故本问题亦NP-hard。"

### 防御策略
| 审稿人可能问 | 防御 |
|-------------|------|
| "SINR模型太简化" | 引用3GPP TR 36.777; imperfect SIC ε=0.02 |
| "为什么K=2？" | 最小非平凡case，K=3在消融实验中 |
| "固定高度100m？" | 引用DJI M300操作规范，3D扩展列为future work |
| "用户移动2m/s依据？" | IoT设备/行人速度，引用ITU标准 |

---

## Section IV: Proposed Hybrid DRL-Optimization Framework (1.8页)

**核心任务**: 描述求解方法，建立C2+C3

### IV-A. Problem Decomposition & Architecture Overview (0.4页)
- **可分离结构揭示**: 给定UAV位置q_k和用户归属S_k，per-user (α_i, p_i) 问题是单slot凸优化（可闭式精确求解）
- 上层 = 时序非凸（哪里飞、何时飞）→ DRL
- 下层 = 单slot凸（分多少、功率多大）→ KKT闭式解
- 42维端到端 vs 5维+闭式解 → DRL动作空间降低88%

- **Architecture图**: Fig.1（系统模型 + Hybrid框架，一张合体图）

### IV-B. Per-User KKT Closed-Form Optimizer (0.5页) ← **C2核心**

**卸载比闭式解**:
- 对τ_i关于α_i求导，令 ∂τ/∂α = 0:
  ```
  α_i* = f_local · R_i / (f_local · R_i + f_alloc · (D_i·f_local/C_i + 1))
  ```
  当R_i很大 → α*→1 (全部卸载); R_i→0 → α*→0 (本地计算)

**功率分配闭式解**:
- 给定SIC解码顺序（按接收功率降序），水注功率分配 + SIC约束的KKT条件
- 给出闭式表达式

**用户归属**: nearest-UAV rule（最近分配）
- 简单且与DRL解耦
- 提供隐式协调信号（见IV-D）

- **Proposition 1**: 在给定UAV位置和归属下，(α*, p*) 是per-user子问题的全局最优解。
  证明: 验证KKT条件 + 约束资格（Slater条件满足）

### IV-C. Independent PPO for Multi-UAV Trajectory (0.5页)
- 每UAV一个独立PPO(Actor-Critic)，**无参数共享、无通信、无shared critic**
- 网络: 2层MLP(256, 128), tanh激活
- 训练: GAE(λ=0.95), clip ratio 0.2, lr=3e-4, batch 2048
- 观测: 9维; 动作: 5维

- **为什么不用CTDE？** (新增防御段，1段)
  (1) Shared critic输入包含对方action，但action空间随user assignment变化 → non-stationarity
  (2) Independent训练更适合部署场景（UAV间可能无稳定通信链路）
  (3) 实验中Het-MAPPO作为CTDE代表验证

- **Algorithm 1**: Independent PPO + Hybrid Optimizer训练流程伪代码

### IV-D. Implicit Coordination via Environment Structure (0.4页) ← **C3理论动机**

**MAPPO失败机制** (物理解释，不做纳什均衡证明):
- 参数共享 + 相似观测 → 相同动作 → 两UAV同飞burst cluster
- 共频干扰I_inter暴增 → 所有用户SINR崩溃 → reward骤降

**Independent PPO的隐式协调** (物理直觉 + 实验观察):
- 不同随机初始化 → 自然权重分化 → 互补轨迹策略
- nearest-UAV归属规则的机制: UAV_0飞近cluster A → 抢走A的用户 → UAV_1的reward自动依赖服务B和C → 无需通信即分工
- **实验证据**: Fig.6 (轨迹可视化) + Fig.4 (UAV间距演化) 将在VI-C节展示

### [Bonus] IV-E. IC-CPPO Dual Constraint (0.4页) ← 仅gate check通过时写入
- Per-UAV delay λ_d + shared interference λ_I
- λ_max物理推导（适配multi-UAV）
- 投影更新

---

## Section V: Experimental Setup (0.8页)

### V-A. Simulation Parameters (0.25页)

**Table II: 系统参数**

| 参数 | 值 | 依据 |
|------|-----|------|
| 区域 | 800×800 m | 城市应急覆盖 |
| K (UAVs) | 2 (扩展3) | 最小非平凡 |
| N (用户) | 20 (3 clusters) | NOMA压力 |
| 高度 | 100 m | DJI M300 |
| v_max | 20 m/s | — |
| f_per_UAV | 4 GHz | Jetson Orin |
| f_local | 1 GHz | IoT终端 |
| 带宽 | 5 MHz NOMA | LTE单载波 |
| p_max | 0.2 W (23dBm) | 3GPP Class 3 |
| 任务数据 | U[0.4, 2.0] Mbit | 压缩图像 |
| 计算密度 | U[500, 1000] cyc/bit | — |
| Deadline | 1.0 s | 严格IoT |
| 电池 | 9000 J/UAV | 含悬停~8000J |
| SIC imperfect ε | 0.02 | [引用] |
| burst周期 | 随机U[8,15] slots | 非平稳 |
| 用户移动 | 2 m/s随机游走 | ITU行人 |
| Cluster中心 | 固定3点 | 可复现 |

### V-B. Baseline Algorithms (0.3页)

| # | Baseline | 类型 | 在论文中的角色 |
|---|----------|------|--------------|
| 1 | **BCD-static** | 传统优化 | 静态最优参照；预期在平稳环境中competitive |
| 2 | **BCD-greedy** | 传统优化 | 贪心失败case (+14.8pp优势) |
| 3 | **MAPPO** (参数共享) | MARL | C3的negative对照 |
| 4 | **Het-MAPPO** (不共享参数+shared critic) | CTDE | C3的CTDE代表，防御"为什么不用CTDE" |
| 5 | **OFDMA-频分** | 多接入对照 | 防御"为什么不频分"；每UAV 2.5MHz正交 |
| 6 | **Single-UAV** | 消融 | 证明多UAV必要性 |
| 7 | **HandCraft** | 启发式 | 预设巡逻路线baseline |

**注**: CPO/MACPO根据实现难度和时间决定是否加入。Het-MAPPO已足够作为CTDE代表。

### V-C. Evaluation Protocol (0.25页)
- **20 seeds**, IQM (Interquartile Mean) + bootstrap 95% CI
- 训练: 2000 episodes; 每50 episode评估10 episodes
- 统计检验: Mann-Whitney U (unpaired), Bonferroni校正
- **6个指标**:
  1. Deadline satisfaction rate (%) — 主指标
  2. Average delay (s)
  3. Jain's fairness index (跨cluster)
  4. Energy efficiency (tasks/Joule)
  5. Average UAV spatial separation (m) — 量化隐式协调
  6. [若C5] Constraint violation rate

---

## Section VI: Results & Analysis (3.5页)

### VI-A. Main Performance Comparison (0.7页)

**对应**: C1+C2全景性能验证

**Table III**: 7算法 × 5指标, 20-seed IQM + 95% CI

**Fig.2**: 主结果柱状图 + bootstrap CI error bar

**Fig.3**: 训练收敛曲线 (DRL算法, shaded area = seed variance)

**预期claim**:
- Independent PPO + Hybrid >> BCD-greedy (+15pp), >> MAPPO (+12pp)
- vs BCD-static: 平稳环境差距可能较小(此处不强调)，在VI-D中展示动态环境优势

### VI-B. Inter-UAV Interference Analysis (0.5页)

**对应**: C1核心验证——干扰建模的价值

**Fig.4**: UAV空间分离距离 vs 时间步
- Independent PPO: 两UAV保持200-400m分离
- MAPPO: 两UAV频繁靠近(<50m)

**Fig.5**: 用户SINR CDF对比
- 条件1: 有inter-UAV interference (完整模型)
- 条件2: I_inter=0 (关闭跨UAV干扰)
- 展示I_inter对低SINR尾部的显著影响

**消融**: 关闭I_inter后，DRL不再学会空间分离（两UAV都飞向burst cluster）

### VI-C. Implicit vs Explicit Coordination (0.6页)

**对应**: C3核心验证

**Table IV**: 协调机制对比
| 方法 | Deadline% | UAV分离距离 | 说明 |
|------|----------|------------|------|
| Independent PPO | ~78% | ~300m | 自然分工 |
| Het-MAPPO (CTDE) | ~72%? | ~200m? | shared critic但不共享参数 |
| MAPPO (参数共享) | ~66% | ~80m | 同质化→同飞 |

**Fig.6**: 轨迹可视化 (2×1子图)
- 左: Independent PPO — 两条互补轨迹
- 右: MAPPO — 两条高度重叠轨迹

**分析**: 为什么nearest-UAV + random init诱导分工（物理直觉解释，非博弈论证明）

### VI-D. DRL Adaptability in Dynamic Environments (0.6页) ← **全文杀手锏**

**对应**: C4核心验证 (从C1的动态性分支出来)

**实验设计**:
- 平稳环境: 固定burst pattern, 用户不移动
- 非平稳环境: 随机burst切换 + 用户2m/s移动

**Table V**: 平稳 vs 非平稳环境对比
| 算法 | 平稳 Deadline% | 非平稳 Deadline% | 性能下降 |
|------|--------------|-----------------|---------|
| Independent PPO | ~78% | ~73% | -5pp |
| BCD-static | ~77% | ~63% | **-14pp** |
| BCD-greedy | ~63% | ~55% | -8pp |

**预期核心发现**: BCD在非平稳环境性能骤降(-14pp)，DRL仅温和下降(-5pp)，差距从+1pp拉大到**+10pp以上**

**Fig.7**: 时序deadline%曲线
- X轴: 时间步; Y轴: 滑动窗口deadline%
- burst切换时刻用竖线标注
- BCD: 每次切换后性能骤降，恢复慢
- DRL: 平滑适应，切换后快速调整

**物理解释**: BCD每个slot基于当前信息独立求解，无法预判burst切换；DRL通过cluster_loads观测隐式学习切换模式

### VI-E. NOMA vs OFDMA (0.4页)

**对应**: 防御"为什么不频分" + C1支撑

**Table VI**: NOMA共频 vs OFDMA频分
| 接入方式 | Deadline% | 吞吐量 | 是否有跨UAV干扰 |
|---------|----------|--------|----------------|
| NOMA 5MHz共享 | ~78% | 高 | 是（但Hybrid管理） |
| OFDMA 2×2.5MHz | ~72%? | 较低 | 无 |

**预期发现**: 尽管NOMA引入跨UAV干扰，但通过空间分离策略+水注功率控制，NOMA在吞吐量/deadline上仍优于OFDMA → 证明共频+干扰管理 > 频分

**NOMA干扰饱和**: Greedy全功率 << Random offload的物理解释（所有用户p_max → 干扰淹没信号）

### VI-F. Ablation Studies (0.7页)

**Table VII**: 消融汇总

| 消融维度 | 变体A | 变体B | 差异 | 验证 |
|---------|-------|-------|------|------|
| 架构 | Hybrid (5维+闭式) | Pure DRL (42维端到端) | +Xpp | C2必要性 |
| UAV数量 | K=2 | K=1 / K=3 | +Xpp / +Xpp | 可扩展性 |
| 干扰模型 | I_inter ON | I_inter OFF | -Xpp | C1必要性 |
| 环境动态性 | 非平稳 | 平稳 | 与VI-D交叉 | C4 |

**Fig.8**: K=1,2,3的Deadline%对比柱状图

**注**: K=3实验至少3-seed初步结果，证明方法可扩展即可

---

## Section VII: Conclusion (0.3页)

1. 总结C1-C4（一句话一个）
2. 核心数字: deadline%, vs BCD优势, vs MAPPO优势
3. 最重要的两个发现:
   - 隐式协调 > 显式MARL（在有nearest-UAV结构的环境中）
   - DRL在非平稳环境中展现本质优势（vs BCD +10pp+）
4. Future work: 3D轨迹、K>3、不完美CSI、real UAV testbed

---

## 图表清单（精简版: 8 fig + 7 table）

### Figures

| # | 内容 | 类型 | 支撑 |
|---|------|------|------|
| Fig.1 | 系统模型(800m区域+3cluster+2UAV+干扰) + Hybrid框架图 | 系统图 | C1+C2 |
| Fig.2 | 主结果柱状图 + bootstrap CI | 柱状图 | 全部 |
| Fig.3 | 训练收敛曲线 (shaded=seed variance) | 折线图 | C2+C3 |
| Fig.4 | UAV空间分离距离 vs 时间: Independent vs MAPPO | 折线图 | C1+C3 |
| Fig.5 | SINR CDF: 有/无inter-UAV干扰 | CDF图 | C1 |
| Fig.6 | 轨迹可视化: Independent PPO vs MAPPO (2×1) | 轨迹图 | C3 |
| Fig.7 | 非平稳时序图: burst切换时DRL vs BCD | 时序图 | **C4** |
| Fig.8 | K=1,2,3 扩展性柱状图 | 柱状图 | C1 |

### Tables

| # | 内容 | 支撑 |
|---|------|------|
| Table I | 文献对比矩阵 (Multi-UAV/NOMA/Shared/DRL/Dynamic) | Related Work |
| Table II | 系统参数 | III |
| Table III | 主结果: 7算法 × 5指标 (IQM+CI) | 全部 |
| Table IV | 协调机制: Independent vs Het-MAPPO vs MAPPO | C3 |
| Table V | 平稳 vs 非平稳: DRL vs BCD | **C4** |
| Table VI | NOMA vs OFDMA | C1 |
| Table VII | 消融汇总: 架构/K值/干扰/动态性 | 全部 |

---

## 审稿人风险防御速查表

| # | 审稿人可能问 | 在哪防御 | 如何防 |
|---|-------------|---------|--------|
| 1 | "为什么不频分？" | I + Table VI | NOMA vs OFDMA实验证明共频+管理 > 频分 |
| 2 | "K=2太少" | VI-F + Fig.8 | K=3扩展实验 |
| 3 | "PPO不是新算法" | I + IV-A | 创新在问题分解洞察+实验发现，不在PPO |
| 4 | "vs BCD差距小" | VI-D + Fig.7 | 非平稳环境差距+10pp+ |
| 5 | "Independent>MAPPO因为调参" | VI-C + Table IV | Het-MAPPO消融 + 物理机制解释 |
| 6 | "Optimizer是启发式？" | IV-B | KKT闭式解 + Proposition 1 |
| 7 | "统计不严谨" | V-C | 20-seed IQM + bootstrap CI |
| 8 | "为什么不用CTDE？" | IV-C | Het-MAPPO作为CTDE代表 + non-stationarity论证 |
| 9 | "SINR模型简化" | III-B | 3GPP TR 36.777 + imperfect SIC |
| 10 | "NP-hard证明？" | III-D | 继承单UAV子问题的MaxFS归约 |

---

## 代码实现待办（必须在写作前完成）

| # | 任务 | 优先级 | 预计工时 | 说明 |
|---|------|--------|---------|------|
| 1 | 重写 `_compute_per_user_actions()` 为KKT闭式解 | **P0** | 2天 | 当前是启发式，C2的根基 |
| 2 | 实现用户移动 (2m/s随机游走) | **P0** | 0.5天 | III-A声称了但代码没有 |
| 3 | 实现非平稳burst (随机切换U[8,15]) | **P0** | 0.5天 | VI-D杀手锏的前提 |
| 4 | 实现OFDMA-频分baseline | **P0** | 0.5天 | Table VI防御 |
| 5 | 实现BCD-static/greedy为正式代码 | **P0** | 1天 | 20-seed需代码跑出，不能手算 |
| 6 | 实现Het-MAPPO (不共享参数+shared critic) | **P1** | 1.5天 | Table IV的CTDE代表 |
| 7 | 固定cluster中心为deterministic布局 | **P1** | 0.5天 | 消除跨seed问题差异 |
| 8 | 添加SINR logging到环境info dict | **P1** | 0.5天 | Fig.5 CDF数据来源 |
| 9 | K=3环境扩展 | **P2** | 1天 | Fig.8扩展性 |
| 10 | [Bonus] IC-CPPO实现 + gate check | **P3** | 2天 | C5，通过再加 |

**总计P0**: ~4.5天
**总计P0+P1**: ~7天
**总计全部**: ~10天
**论文写作**: ~10天
**总到投稿**: ~4周

---

## 执行顺序建议

```
Week 1: P0代码修复 (KKT optimizer, 用户移动, 非平稳burst, OFDMA, BCD)
         └─ 每完成一项立即跑3-seed gate check验证数字是否合理

Week 2: P1代码 (Het-MAPPO, 固定cluster, SINR log) + 20-seed全量实验开始跑
         └─ IC-CPPO gate check (bonus)

Week 3: 实验收尾 + K=3扩展(3-seed) + 论文写作 III-IV节

Week 4: 论文写作 I-II, V-VII + 图表制作 + 内部评审
```
