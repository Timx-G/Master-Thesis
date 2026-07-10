# 文献搜索提示词：以创新点为中心的CCF-B级文献调研

## 任务说明

你是一名学术文献搜索助手。我正在撰写一篇目标期刊为 **IEEE IoTJ / TVT (CCF-B, IF≈10)** 的硕士论文。

**论文标题**: "Interference-Aware Hybrid DRL-Optimization for Cooperative Multi-UAV Trajectory Design and NOMA Task Offloading in MEC Networks"

**核心场景**: K=2架UAV共享5MHz频段、800m×800m区域、N=20用户分布在3个集群中，使用NOMA上行链路进行MEC计算卸载。UAV之间存在跨UAV干扰（共享频谱无法SIC消除），需要通过轨迹设计实现空间隔离。

我需要你围绕本文的 **5个核心创新点** 搜索文献。每个创新点需要 **10-20篇** 高质量参考文献，用于：
1. 证明该方向是活跃研究热点（有人做）
2. 证明现有工作的不足（但没人做到我们这步）
3. 为本文的技术选择提供理论依据

---

## 严格质量要求

### 期刊/会议等级（必须满足之一）
- **CCF-A**: IEEE JSAC, IEEE TMC, IEEE TPDS, INFOCOM, MobiCom, NeurIPS, ICML, ICLR
- **CCF-B**: IEEE IoTJ, IEEE TVT, IEEE TWC, IEEE TCOM, IEEE TSC, ICC, GLOBECOM, AAAI, IJCAI
- **CCF-C（仅补充）**: IEEE WCL, IEEE CL, Computer Networks, Computer Communications
- **顶级综述例外**: IEEE Communications Surveys & Tutorials, Proceedings of the IEEE

### 时间范围
- **核心文献**: 2023-2026 年（至少占 70%）
- **经典奠基**: 2020-2022 年（最多 30%，仅限高引用 >50次 或奠基性工作）

### 每篇论文必须输出
```
[编号]. 标题
- 作者: ...
- 期刊/会议: XXX, 年份, CCF等级
- DOI/arXiv: ...
- 所属创新点: [I1/I2/I3/I4/I5]
- 一句话摘要: ...
- 核心技术: ...
- 本文与之的差异/推进: [我们的工作在哪个维度超越或补充了这篇]
```

---

## 已引用文献（勿重复搜索）

以下文献已在本文参考列表中，搜索时请跳过，但可以找它们的**后续工作/改进版本/同组新成果**：

Mao2017(MEC survey), Shi2016(Edge vision), Mozaffari2019(UAV tutorial), Zeng2019(UAV 5G), Ding2017/2018(NOMA), Liu2017(NOMA 5G), Schulman2017(PPO), Fujimoto2018(TD3), Haarnoja2018(SAC), Tessler2019(RCPO), Achiam2017(CPO), Stooke2020(PID-Lagrangian), Zhang2020(FOCOPS), Yang2022(CUP), Agarwal2021(PG theory), Altman1999(CMDP), Wang2023(multi-UAV MEC DRL), Hu2023(UAV NOMA MEC SCA), Li2022(NOMA UAV DRL), Zhang2023(UAV NOMA IoT), Chen2024(multi-UAV MEC scheduling), Zhan2020, Cao2021, Xie2020, Luo2008

---

## 创新点 I1: Multi-UAV MEC-NOMA联合优化 + 跨UAV共享频谱干扰建模

### 我们的创新
K架UAV共享同一5MHz频段为NOMA用户服务。用户i由UAV_k服务时：
```
SINR_i = h_{i,k}·p_i / (I_intra + I_inter + σ²)
I_intra: 本UAV内NOMA干扰（SIC可部分消除）
I_inter: 跨UAV干扰 = Σ_{j∈UAV_{k'}, k'≠k} p_j·h_{j,k}（SIC不可消除）
```
**现有文献的缺口**: 大多数multi-UAV MEC工作假设正交频段（每UAV独立频段）或忽略跨UAV干扰。本文是少数建模shared-spectrum inter-UAV NOMA interference的工作。

### 搜索方向（需10-20篇）

**方向A: Multi-UAV MEC联合优化（6-8篇）**
搜索词:
- "multi-UAV mobile edge computing task offloading" 2023 2024 2025 2026 IEEE
- "cooperative UAV MEC trajectory optimization resource allocation"
- "multi-UAV assisted edge computing deep reinforcement learning"
- "joint trajectory design multi-UAV computation offloading"

要找的论文特征: 2+架UAV联合优化轨迹+卸载，但**大概率假设正交频段**——这正是我们的对比点。

**方向B: UAV-NOMA-MEC三者结合（4-6篇）**
搜索词:
- "UAV NOMA mobile edge computing" 2023 2024 2025 IEEE
- "NOMA assisted UAV MEC offloading power allocation"
- "non-orthogonal multiple access UAV trajectory edge computing"
- "UAV NOMA uplink SIC task offloading optimization"

要找的论文特征: UAV+NOMA+MEC，但**大概率是单UAV**——本文扩展到multi-UAV + shared spectrum。

**方向C: Multi-UAV共享频谱/干扰管理（3-5篇）**
搜索词:
- "multi-UAV shared spectrum interference management" 2023 2024 2025
- "inter-UAV interference multi-cell NOMA"
- "UAV co-channel interference trajectory design"
- "multi-UAV frequency reuse interference coordination"
- "cooperative multi-UAV NOMA cross-cell interference"

要找的论文特征: 多UAV共频干扰建模。这方面文献较少，找到即为高价值引用。

### 创新点论证逻辑
找到的文献应能支撑: "Multi-UAV MEC有人做(方向A)、UAV-NOMA-MEC有人做(方向B)、但**Multi-UAV + NOMA + shared spectrum干扰**几乎没人做(方向C的空白) → 本文填补该空白"

---

## 创新点 I2: Hybrid DRL(轨迹) + Analytical Optimizer(用户级分配) 分层架构

### 我们的创新
将联合优化问题分解为两层:
- **上层DRL**: 每架UAV独立PPO，5维动作（dx, dy, cluster_weight×3），决定去哪飞、优先服务哪个集群
- **下层Optimizer**: 给定UAV位置后，用channel-quality-based解析解计算per-user卸载比α_i和water-filling功率p_i

**为什么这样做**: 42维端到端DRL需要>5M样本（维度诅咒），但per-user分配是凸问题，optimizer瞬间精确求解。DRL只负责optimizer无法解决的时序决策（多步轨迹规划+集群间协调）。

### 搜索方向（需10-20篇）

**方向A: Hybrid DRL + 传统优化 在无线/MEC领域（6-8篇）**
搜索词:
- "hybrid deep reinforcement learning optimization mobile edge computing" 2023 2024 2025
- "DRL convex optimization joint UAV resource allocation"
- "reinforcement learning combined optimization wireless network"
- "learning-based trajectory optimization analytical resource allocation"
- "two-level optimization DRL wireless communication"

要找的论文特征: DRL做高层决策 + 凸优化/解析解做低层分配。越接近我们的"DRL轨迹+optimizer分配"架构越好。

**方向B: 端到端DRL在高维动作空间的困难（3-5篇）**
搜索词:
- "deep reinforcement learning high dimensional action space challenge" 2023 2024
- "curse of dimensionality continuous action DRL wireless"
- "scalability deep reinforcement learning resource allocation"
- "action space decomposition reinforcement learning"

要找的论文特征: 论证端到端DRL在>20维连续动作空间的困难——为我们选择Hybrid架构提供理论依据。

**方向C: 凸优化/Water-filling在NOMA功率分配（3-4篇）**
搜索词:
- "NOMA power allocation water-filling optimization"
- "optimal offloading ratio analytical solution MEC"
- "convex optimization NOMA uplink power control"

要找的论文特征: 证明per-user功率/卸载比分配可以用解析方法高效求解——为我们的下层optimizer提供理论基础。

### 创新点论证逻辑
"端到端DRL在高维空间失败(方向B) → 有人提出Hybrid架构(方向A) → 但没人把Hybrid用在multi-UAV+NOMA+inter-UAV干扰场景 → per-user分配有成熟解析解(方向C) → 我们的两层分解是自然且有根据的"

---

## 创新点 I3: 约束DRL + 物理推导的Lagrangian上界 λ_max

### 我们的创新
标准Lagrangian方法（RCPO）中λ_d无上界，当λ_d > λ_crit = r̄/c̄ 时，增广奖励r'=r-λ·c变为负数，策略梯度崩溃（Lagrangian saturation）。

本文推导:
```
λ_crit = r̄/c̄ ≈ Np_a / (c̄_comp + c̄_tx) = 10/1.2 ≈ 8.3
λ_max = (2/3)·λ_crit = 5.0
```

保证E[r'] ≥ r̄(1-ρ) > 0，训练方差降低47%。

### 搜索方向（需10-15篇）

**方向A: 约束RL/Safe RL方法论（5-7篇）**
搜索词:
- "constrained reinforcement learning CMDP" 2023 2024 2025 NeurIPS ICML ICLR AAAI
- "safe reinforcement learning Lagrangian" 2023 2024 2025
- "constrained policy optimization deep reinforcement learning"
- "primal-dual reinforcement learning constraint satisfaction"
- "Lagrangian multiplier stability convergence reinforcement learning"

要找的论文特征: CMDP求解新方法（CPO后续、FOCOPS后续、CUP后续、新的primal-dual方法），特别关注**Lagrangian乘子的稳定性/收敛性分析**。

**方向B: 约束DRL应用于无线通信/MEC（5-7篇）**
搜索词:
- "constrained deep reinforcement learning wireless communication" 2023 2024 2025
- "CMDP mobile edge computing task offloading"
- "safe reinforcement learning UAV trajectory"
- "Lagrangian relaxation DRL resource allocation delay constraint"
- "constrained PPO edge computing deadline"

要找的论文特征: 将约束DRL(RCPO/CPO/Lagrangian PPO)应用于无线场景解决QoS/延迟约束——但**很可能没有物理推导λ_max**，这是我们的差异点。

**方向C: Lagrangian乘子上界/稳定性的理论分析（2-3篇，可能很少）**
搜索词:
- "Lagrangian multiplier upper bound constrained optimization"
- "dual variable stability primal-dual reinforcement learning"
- "reward-cost ratio analysis constrained MDP"
- "Lagrangian saturation policy gradient"

要找的论文特征: 这个方向文献极少。任何分析λ乘子上界、reward-cost ratio、augmented reward符号问题的论文都是gold。找不到也说明**这是我们的理论贡献空白点**。

### 创新点论证逻辑
"约束RL方法很多(方向A) → 有人在无线场景用但不分析λ稳定性(方向B) → λ上界的理论分析几乎空白(方向C) → 本文填补该理论空白 + 首次给出物理推导的λ_max"

---

## 创新点 I4: Independent Learning > Centralized MARL（隐式协调 > 显式协调）

### 我们的创新
实验发现Independent PPO（每UAV独立网络，无通信）以78.1%显著超越MAPPO（参数共享+集中critic）的66.4%。

原因: MAPPO参数共享导致两架UAV对相似观测产生相同动作→都飞向burst cluster→干扰爆炸。Independent PPO通过不同随机初始化自然分化出互补角色，nearest-UAV归属规则提供隐式协调信号。

### 搜索方向（需10-15篇）

**方向A: Multi-Agent RL用于multi-UAV/无线网络（5-7篇）**
搜索词:
- "multi-agent reinforcement learning multi-UAV" 2023 2024 2025 IEEE
- "MAPPO MADDPG multi-UAV trajectory optimization"
- "CTDE centralized training decentralized execution UAV"
- "multi-agent deep reinforcement learning wireless resource allocation"
- "cooperative multi-agent UAV MEC"

要找的论文特征: 用MAPPO/MADDPG/MATD3/QMIX做multi-UAV协调。重点看它们是否遇到**参数共享导致策略同质化**的问题。

**方向B: Independent Learning vs Centralized MARL对比（3-5篇）**
搜索词:
- "independent learning vs centralized multi-agent" 2023 2024 2025
- "independent PPO multi-agent decentralized"
- "parameter sharing failure multi-agent reinforcement learning"
- "emergent coordination independent reinforcement learning"
- "decentralized multi-agent learning without communication"

要找的论文特征: Independent learning在某些场景优于CTDE/MAPPO的实验证据。特别关注**环境结构提供隐式协调**的分析。

**方向C: Multi-UAV协调机制对比研究（3-4篇）**
搜索词:
- "multi-UAV coordination mechanism comparison"
- "implicit coordination multi-robot cooperative"
- "centralized vs decentralized multi-UAV planning"
- "emergent specialization multi-agent reinforcement learning"

要找的论文特征: 比较不同协调机制（集中式/分布式/独立/通信）在UAV或机器人场景的效果。

### 创新点论证逻辑
"MARL是multi-UAV的主流方案(方向A) → 但CTDE/参数共享有理论缺陷(方向B找到的证据) → 本文实验证实Independent > MAPPO + 给出物理解释(nearest-UAV隐式协调) → 这一发现挑战conventional wisdom"

---

## 创新点 I5: NOMA干扰饱和效应 + 智能功率控制的必要性

### 我们的创新
Greedy（全用户最大功率全量卸载）仅达15.4%，远低于Random(63.0%)。物理解释：N=20用户同时以p_max=0.2W发射时，聚合干扰 I≈19×g×0.2=3.8g W，SINR趋近于零，NOMA速率崩溃。

**核心洞察**: 在dense NOMA网络中，"少即是多"——智能控制谁发射、发射多少功率，比让所有人全力发射效果好得多。

### 搜索方向（需10-15篇）

**方向A: NOMA功率控制与干扰管理（5-7篇）**
搜索词:
- "NOMA power control interference management" 2023 2024 2025 IEEE
- "NOMA uplink power allocation SIC optimization"
- "non-orthogonal multiple access interference mitigation"
- "NOMA user pairing power allocation strategy"
- "dense NOMA network interference analysis"

要找的论文特征: 分析NOMA场景中功率控制的重要性，特别是**不恰当功率分配导致性能恶化**的分析。

**方向B: NOMA近远效应/干扰饱和现象（3-5篇）**
搜索词:
- "NOMA near-far effect analysis"
- "NOMA interference saturation dense users"
- "NOMA SIC failure overloaded"
- "NOMA performance degradation high user density"
- "NOMA scalability limitation many users"

要找的论文特征: 分析当用户数增加时NOMA性能退化的论文。我们的"greedy anomaly"需要文献支撑：在密集用户场景下，贪心全功率策略会导致干扰崩溃。

**方向C: NOMA vs OMA对比 + NOMA在UAV/MEC中的优势（3-4篇）**
搜索词:
- "NOMA OMA comparison UAV communication" 2023 2024 2025
- "NOMA advantage mobile edge computing offloading"
- "NOMA spectral efficiency UAV multi-user"

要找的论文特征: 证明NOMA在UAV-MEC场景相比OMA的频谱效率优势（我们的实验：NOMA 82.5% vs OMA-TDMA 9.0%），作为选用NOMA的依据。

### 创新点论证逻辑
"NOMA有明确的频谱效率优势(方向C) → 但功率控制至关重要(方向A) → 不当功率分配在密集场景导致性能崩溃(方向B) → 本文首次在UAV-MEC-NOMA场景量化这一效应并命名为'NOMA interference saturation effect' → 证明智能功率控制(DRL)不可或缺"

---

## 最终输出要求

### 数量要求
| 创新点 | 最少篇数 | 最多篇数 |
|--------|---------|---------|
| I1: Multi-UAV NOMA MEC + 跨UAV干扰 | 10 | 20 |
| I2: Hybrid DRL + Optimizer分层架构 | 10 | 20 |
| I3: 约束DRL + λ_max物理上界 | 10 | 15 |
| I4: Independent > MAPPO隐式协调 | 10 | 15 |
| I5: NOMA干扰饱和效应 | 10 | 15 |
| **总计** | **50** | **85** |

注意：一篇论文可能同时属于多个创新点（例如一篇"multi-UAV NOMA MEC with constrained DRL"同时属于I1+I3），这种情况标注所有相关创新点，但只计一次。最终去重后确保不少于50篇唯一论文。

### 分组输出
按创新点分组输出。每个创新点输出后，紧跟一段 **"空白分析"**：
> "在该创新点方向上，现有文献的共同缺陷/未覆盖之处是：______。本文的推进在于：______。"

### 最终综合分析
所有创新点输出后，提供：

1. **创新点强度评估表**:

| 创新点 | 支撑文献数 | 文献空白程度 | CCF-B充分性评估 |
|--------|-----------|------------|---------------|
| I1 | X篇 | 高/中/低 | 充分/需加强/不足 |
| ... | ... | ... | ... |

2. **论文定位声明**: 本文在现有文献图景中的确切位置（一段话）

3. **风险点预警**: 哪个创新点的文献支撑最弱，可能被审稿人质疑，建议如何补强

4. **推荐的Related Work结构**: 基于搜索结果，建议Related Work分几个小节、每节引哪些核心文献

---

## 搜索策略建议

1. **Google Scholar** 为主，结合 **IEEE Xplore**, **Semantic Scholar**, **DBLP**
2. 对每个搜索词，至少翻阅前3页结果
3. 找到高引用论文后，检查其**"被引用"列表**（forward citation）找到最新后续工作
4. 找到高相关论文后，检查其**参考文献列表**（backward citation）找到被遗漏的经典
5. 特别关注 IEEE IoTJ, IEEE TVT, IEEE TWC 2024-2026 年的论文（目标期刊同行）
6. 对于方向C（跨UAV干扰、λ上界理论、Independent vs MAPPO）这些可能文献稀少的方向，**找不到也是重要信息**——说明这是空白点，请明确报告
