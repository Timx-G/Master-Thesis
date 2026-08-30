# 2024--2026 年多 UAV NOMA-MEC 文献基线可比性审计

审计日期：2026-07-13

## 1. 审计目的与结论

本审计回答两个问题：

1. 近两年最接近本项目的算法，能否在现有仿真框架中被忠实复现并作为同问题口径的先进基线；
2. 在不能忠实复现时，论文应如何组织比较证据，避免把自定义适配算法冒充为 SOTA。

结论如下。

- 本轮不应新增一个声称为“原论文复现”或“SOTA”的文献算法。六篇近邻工作均联合优化了本项目主动冻结或排除的变量，例如 UAV 轨迹、用户发射功率、动态带宽、跨 UAV 任务迁移、长期能耗或任务优先级效用。直接冻结这些变量会移除原算法的主要动作维度，直接放开这些变量又会使 EAD-JSRA 处于不公平的决策空间。
- 当前最可靠的证据链是“受控基线 + 同口径强机制基线 + 小规模精确最优”。现有 `fixed-half/equal` 和 `crossing/equal` 只能称为受控资源策略；`IA-greedy`、`per-UAV ILP`、`exchange-stable` 和公平 OFDMA 属于同一模型下的强机制基线；完整枚举是小规模最优性参照。
- `exchange-stable` 与联盟博弈/交换匹配文献的思想一致，但它不是 Lyu 等人算法的逐式复现，必须写成“exchange-stable（联盟/匹配机制基线）”，不能写成“复现 Lyu et al.”。
- 后续若必须增加文献适配基线，优先选择 Lin 等人 2025 年的 `POS_MADDPG_NOMA`，因为其目标也包含成功卸载任务数；但必须先取得完整算法参数或作者代码，预注册变量冻结规则，并命名为 `POS-MADDPG-NOMA-static-adapted`。在此之前不得报告其为已复现 SOTA。

## 2. 本项目的比较口径

正式确认性实验的比较对象不是一般的“多 UAV MEC”，而是以下受限问题：

- UAV 在给定位置悬停；UAV 轨迹、飞行能耗和电池不进入当前优化；
- 地面用户移动，但每个到达任务只在其到达时隙评价一次，不存在跨时隙队列重试；
- 每个任务最多由一个实际服务 UAV、一个子载波位置服务，部分卸载仅发生在本地 CPU 与该服务 UAV 之间；
- UAV 之间全频复用；每个 UAV/子载波最多承载两个上行 NOMA 用户；采用固定用户发射功率、硬 SIC 约束和显式跨接收端同频干扰；
- 计算资源是逐 UAV 的硬 CPU 预算；核心指标是唯一到达任务的按时完成率；
- 全网目标按 `(按时完成数, 总截止裕量, -跨 UAV 干扰)` 作字典序比较；
- 正式主场景为 `K=2, N=20, S=3`，所有算法共享不可变任务与信道轨迹。

以上口径可由 `configs/thesis/confirmatory_main.toml`、`uav_mec/evaluation.py`、`uav_mec/resource_allocation.py`、`uav_mec/scheduler.py` 和 `uav_mec/simulation.py` 复核。

## 3. 六篇近邻文献审计

| 文献 | 场景与频谱 | 主要决策变量 | 优化目标与任务模型 | 原算法 | 在当前框架中忠实复现？ |
|---|---|---|---|---|---|
| Chen et al., IEEE TWC 2026, DOI [10.1109/TWC.2025.3593884](https://doi.org/10.1109/TWC.2025.3593884) | 多 UAV 协作 NOMA-MEC；公开摘要未充分说明是否采用与本文相同的跨 UAV 全频复用干扰口径 | UAV 三维轨迹、移动用户关联、卸载比例和资源分配 | 最大化长期平均系统效用；显式考虑任务优先级和 QoS，不是单位价值硬截止完成基数 | 拉格朗日对偶解耦 + improved soft actor-critic（ISAC） | **否。** 冻结轨迹等变量后会删除 ISAC 的主要动作；其效用、状态转移、训练过程和本文单时隙字典序目标不同；公开页面未提供可直接接入本框架的代码 |
| Wu et al., Expert Systems with Applications 2025, DOI [10.1016/j.eswa.2025.126567](https://doi.org/10.1016/j.eswa.2025.126567) | 三维动态多 UAV MEC-NOMA；移动 SMD；论文讨论动态解码次序及簇间干扰 | 解析卸载策略、K-means 用户聚类、UAV 三维轨迹和功率分配 | 最小化系统总时延，并满足能耗约束；属于长期移动控制问题 | 数学推导 + K-means + cooperative MATD3 | **否。** 当前项目不优化轨迹或功率，且完成基数目标与总时延目标不同；只保留聚类或卸载部分不能称为 MATD3 原算法复现 |
| Lin et al., IET Communications 2025, DOI [10.1049/cmu2.70063](https://doi.org/10.1049/cmu2.70063) | 多 UAV 应急 MEC；传感器采用 NOMA；动态频谱/载波接入，不是固定 `K x S` 全频复用调度对象 | 服务 UAV/卸载目标、NOMA 载波、用户发射功率、带宽/频谱接入和 UAV 轨迹 | 在时延、能耗和传输速率约束下最大化成功卸载任务数；部分卸载、长期 MDP | POS_MADDPG_NOMA，Gumbel-Softmax 处理离散/连续混合动作 | **本轮否，未来最优先适配。** 目标最接近，但三类核心动作在当前模型中被固定；官方页面声明数据按需提供，未给出可直接复用的公开代码。取得代码后只能以 `static-adapted` 名义进行预注册适配 |
| Wang and Luo, Electronics 2024, DOI [10.3390/electronics13214240](https://doi.org/10.3390/electronics13214240) | 多 UAV NOMA-MEC；UAV 之间使用 FDMA 避免干扰，UAV 内部使用 NOMA；任务数据可同时分配到多个 UAV | 发射功率、计算资源、任务数据分配和 UAV 轨迹 | 在因果/时序与任务完成约束下最小化系统加权能耗；多时隙连续任务处理 | 交替优化 + successive convex approximation（SCA） | **否，属于结构性不可比。** 其跨 UAV 正交频谱恰好消除了本文创新点一所研究的外部性，而且“一任务多 UAV 分割”违反本文唯一服务 UAV 语义 |
| Lyu et al., IEEE IoTJ 2024, DOI [10.1109/JIOT.2023.3348164](https://doi.org/10.1109/JIOT.2023.3348164) | 海洋 IoT 的 NOMA-UAV 应急通信；用户在载波联盟中竞争，未建立本文的多接收端全频复用外部性问题 | IoT 设备发射功率、UAV 计算资源、任务卸载和载波分配 | 最小化设备计算开销，即任务完成时间与设备能耗的加权和，不是按时完成任务基数 | 拟凸/凸资源优化 + 联盟博弈任务卸载，收敛到 Nash-stable 解 | **否。** 当前 `exchange-stable` 可作为同类机制基线，但目标、功率变量、资源子问题和跨 UAV 干扰语义不同，因此不能标注为 Lyu 算法复现 |
| Liu et al., Sensors 2025, DOI [10.3390/s25164965](https://doi.org/10.3390/s25164965) | 两 UAV 协作 MEC-NOMA；NOMA 用于区域内 IoT--UAV 上行，另建 UAV--UAV 任务传输；未显式处理本文的跨接收端全频复用外部性 | 本地/边缘 UAV/云 UAV 任务比例、UAV 间任务迁移和 UAV 轨迹 | 在能耗约束下最小化总系统时延；跨时隙轨迹与任务迁移 | TD3 联合卸载与轨迹优化 | **否。** 其“协作”是 UAV 间任务迁移，而本文 CrossUAV 是改变地面用户的实际服务接收机；决策和时延链均不同，官方页面仅声明数据可按请求提供 |

## 4. 逐项不可比性判据

文献算法只有同时满足以下条件，才可被称为“同问题口径的直接比较”：

1. **相同可控变量**：均只能决定服务 UAV、子载波、NOMA 角色、部分卸载比例和 MEC CPU；
2. **相同物理层**：均使用相同全频复用、跨 UAV 同频干扰、接收顺序、SIC 残差和带宽口径；
3. **相同任务语义**：一项任务只计数一次，且由同一个服务 UAV 接收和计算；
4. **相同目标**：主目标均为单位价值任务的按时完成数量，而不是总时延、能耗或加权效用；
5. **相同信息与预算**：均获得同一 CSI、任务到达和逐 UAV CPU 预算；
6. **相同时间口径**：均为单时隙决策，或均处理队列和长期轨迹；
7. **可复现实现**：算法步骤、超参数、停止条件和随机训练预算可审计。

六篇近邻文献均至少违反第 1、2、4 或 6 项，因此不能把论文中的原始结果数字搬入本项目，也不能仅根据算法名称做柱状图比较。

## 5. 论文应采用的比较层级

### 5.1 层级 A：受控资源策略

- `Fixed-half/Equal`：固定卸载比例 0.5，CPU 平均分配；
- `Crossing/Equal`：在平均 CPU 下平衡本地与边缘分支时延。

用途：隔离“截止时间反推最小卸载比例与最小 CPU”这一机制。它们是消融/朴素控制组，不是先进算法。

### 5.2 层级 B：同口径强机制基线

- `IA-greedy`：候选组经过同一个全网物理层与 DD-MCAA 评价；
- `per-UAV ILP`：检验局部 MILP 初始化与后续联合搜索的差异；
- `exchange-stable`：采用局部交换稳定准则，代表联盟/匹配类机制；
- `fair OFDMA`：在相同服务机会口径下检验 NOMA 与正交接入的差异。

这些方法可在同一不可变轨迹、同一硬 SIC 和同一 CPU 预算下比较，因而比强行移植异构文献算法更有内部效度。

### 5.3 层级 C：精确参照

- 对至多五个需卸载任务完整枚举所有“未服务/UAV/子载波”分配；
- 使用同一硬评价器筛除容量、SIC、唯一任务和 CPU 违规；
- 报告启发式与全局最优的完成数量差、达到最优的时隙比例和运行时间。

精确参照不能证明一般规模全局最优，但能够证明算法不是只击败了人为弱基线。

## 6. 后续文献适配的执行规范

若论文终稿仍要求加入一项近两年文献算法，按以下顺序执行：

1. 向 Lin 等人申请代码、训练参数、网络结构和原始场景配置；
2. 在运行确认性种子前冻结适配协议，明确删除了哪些轨迹、功率和带宽动作；
3. 将适配版本命名为 `POS-MADDPG-NOMA-static-adapted`，不得省略 `adapted`；
4. 所有方法使用相同任务、真实/估计 CSI、功率、悬停位置、带宽、CPU 和随机种子；
5. 为 DRL 基线固定训练交互次数、独立训练种子、模型选择规则和推理预算；
6. 同时报告按时完成率、约束违规率、训练成本和在线推理时间；
7. 把“原论文变量被冻结后可能削弱其优势”列为效度威胁；
8. 适配实验单列为补充证据，不替代当前机制消融与小规模精确参照。

若无法取得足够复现信息，则停止适配，不根据摘要自行补全网络、奖励或超参数。

## 7. 可直接用于论文的严谨措辞

### 推荐措辞

> 由于近年多 UAV NOMA-MEC 方法通常联合优化 UAV 轨迹、用户功率、动态带宽或跨 UAV 任务迁移，其决策空间和优化目标与本文的固定悬停、全频复用、硬截止完成基数问题并不一致。本文因此不直接搬用异构场景下的文献数值，而是在相同任务与信道轨迹、相同 SIC 和 CPU 硬约束下，与干扰感知贪心、逐 UAV ILP、交换稳定匹配和公平 OFDMA 等同口径强基线进行配对比较，并在小规模实例上通过完整枚举提供精确最优参照。

> Fixed-half/Equal 与 Crossing/Equal 用于隔离 DD-MCAA 的资源需求推导和准入机制，属于受控消融策略，不作为现有先进方法的替代。

> Exchange-stable 基线代表联盟博弈和交换匹配类局部稳定机制，但并非对某一篇文献算法的逐式复现。

### 禁止措辞

- “本文算法全面优于当前 SOTA”；
- “本文复现并击败了 ISAC、MATD3 或 POS_MADDPG_NOMA”；
- “Crossing/Equal 是文献中的最先进算法”；
- “近年研究均未考虑多 UAV NOMA-MEC”；
- “小规模完整枚举证明本文算法在一般规模上全局最优”。

## 8. 审计引用的原始/官方页面

1. P. Chen et al., “QoS-Oriented Task Offloading in NOMA-Based Multi-UAV Cooperative MEC Systems,” *IEEE Transactions on Wireless Communications*, vol. 25, pp. 1965--1980, 2026. DOI: [10.1109/TWC.2025.3593884](https://doi.org/10.1109/TWC.2025.3593884).
2. G. Wu, G. Chen, and X. Gu, “Multi-agent learning based UAV-assisted MEC-NOMA for cellular offloading in Internet of Things,” *Expert Systems with Applications*, vol. 270, art. 126567, 2025. DOI: [10.1016/j.eswa.2025.126567](https://doi.org/10.1016/j.eswa.2025.126567).
3. Y. Lin, Z. Xu, J. Li, J. Wang, and C. Li, “Deep Reinforcement Learning-Based Intelligent Resource Management in Multi-UAVs-Assisted MEC Emergency Communication System,” *IET Communications*, vol. 19, art. e70063, 2025. DOI: [10.1049/cmu2.70063](https://doi.org/10.1049/cmu2.70063).
4. S. Wang and Z. Luo, “An Energy-Efficient Scheme Design for NOMA-Based UAV-Assisted MEC Systems,” *Electronics*, vol. 13, no. 21, art. 4240, 2024. DOI: [10.3390/electronics13214240](https://doi.org/10.3390/electronics13214240).
5. T. Lyu, H. Xu, F. Liu, M. Li, L. Li, and Z. Han, “Computing Offloading and Resource Allocation of NOMA-Based UAV Emergency Communication in Marine Internet of Things,” *IEEE Internet of Things Journal*, vol. 11, no. 9, pp. 15571--15586, 2024. DOI: [10.1109/JIOT.2023.3348164](https://doi.org/10.1109/JIOT.2023.3348164).
6. J. Liu et al., “Multi-UAV-Assisted Task Offloading and Trajectory Optimization for Edge Computing via NOMA,” *Sensors*, vol. 25, no. 16, art. 4965, 2025. DOI: [10.3390/s25164965](https://doi.org/10.3390/s25164965).

本审计只判断模型与算法是否可公平比较，不以期刊层级替代技术可比性判断。
