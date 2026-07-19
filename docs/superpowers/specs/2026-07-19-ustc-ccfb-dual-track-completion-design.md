# 中科大专业硕士论文与 CCF-B 双轨完善设计

**状态：** 已逐节获得用户批准并完成三路独立复核，待用户书面确认<br>
**版本：** 1.0<br>
**日期：** 2026-07-19<br>
**项目：** Master-Thesis<br>
**研究主题：** 多无人机全频复用 NOMA-MEC 中周期性时限任务的干扰外部性感知调度与资源分配<br>
**学位类型：** 中国科学技术大学信息科学技术学院专业硕士<br>
**论文目标：** 中科大学位论文定稿 + 以 *Computer Networks* 为现实目标的 CCF-B 英文投稿稿件

## 1. 决策摘要

本设计采用用户批准的“方案 B：双轨 CCF-B 升级”。项目保留现有 EAD-JSRA 研究主线，但将贡献边界、实验协议、论文模板和证据链统一到同一个可审计的 v2 版本。

最终交付物包括：

1. 一份符合中国科学技术大学 2026 年模板的专业硕士学位论文；
2. 一篇按 *Computer Networks* 官方投稿要求组织的英文研究论文；
3. 一套通过校验和、约束检查和统计审计的 `artifacts/canonical/v2/` 证据包；
4. 论文图表、LaTeX 数值宏、claim matrix、复现指南和投稿辅助材料；
5. 代码、配置、测试、实验和论文之间的一致性审计报告。

本设计不承诺论文必然被 CCF-B 期刊录用。其目标是将研究方案、实现、实验、理论与写作提升到可诚实投稿和接受同行评审的水平。任何未通过证据门禁的性能主张都必须删除或降级，不允许通过选择性报告补救。

## 2. 已批准事项与约束

用户在 2026-07-19 逐项批准：

1. 采用方案 B 的中科大学位论文与 CCF-B 英文稿双轨路线；
2. 采用本设计规定的研究架构和贡献边界；
3. 采用预注册式实验与统计协议；
4. 采用官方模板、单一证据源和双论文交付架构；
5. 采用本文规定的执行顺序、失败回退和完成标准；
6. 在规格与实施计划审阅完成后，允许建立选择性安全检查点和 sibling Git worktree。

执行期间还必须遵守：

- 不覆盖或丢弃用户已有改动；
- 不中断正在运行的功效预实验；
- 不把答辩材料、pilot 产物或生成 PDF 混入研究代码检查点；
- 不伪造实验结果、文献基线、学校查重结果或期刊接受前景；
- 不替用户提交论文、上传外部系统或对外发布仓库，除非用户另行明确授权。

## 3. 当前基线与主要缺口

### 3.1 已具备基础

当前项目已经具备研究生论文所需的主要工程基础：

- EAD-JSRA、IEA-CAP 和 DD-MCAA 的 v1/v2 实现基础；
- 周期任务、多起点最佳改进、外部性证书、增量评价器、资源策略、Exact Oracle、分片运行、功效分析和 GeoLife 解析等 v2 代码；
- 403 项自动化测试已经在当前环境中通过；
- canonical-v1 具备校验和、零 SIC/CPU 违规和自动图表生成链；
- v2 K=3、N=45 探索性功效预实验已产生 Main 100/100 个种子和 Externality 48/100 个种子的分片，主场景不再接近 100% 完成率天花板；该不完整运行不能用于最终样本量冻结。

### 3.2 不能直接作为最终证据的内容

- canonical-v1 对应旧提交和旧算法语义，不能证明当前 dirty v2 实现；
- v1 主实验接近饱和，IEA-CAP 的实际收益过小；
- v1 Exact K=2、N=5 场景为零 gap，但资源竞争不足，只能作为正确性 sanity check；
- 当前中文论文约 8,806 个汉字，且使用通用 `ctexbook`；
- 当前活动英文稿使用 `IEEEtran`，与 *Computer Networks* 目标不一致；
- 当前生成的 LaTeX 数值宏仍来自 canonical-v1；
- `main_v2_pre_round1.tex` 包含旧模型、旧数字和不可直接比较的 DRL 基线，只能保留为历史稿；
- 代码和论文在 K/N、搜索策略、邻域、符号、任务身份、结果数字和理论表述上存在版本不一致。

### 3.3 Git 与计算现场

设计批准时：

- 分支为 `feature/core-algorithm`，领先远端 48 个提交；
- 有 65 个 tracked 文件修改，约 7,613 行新增、922 行删除；
- 另有多项未跟踪研究文件、约 299 MiB pilot 产物和无关答辩材料；
- `thesis-power-pilot` 曾按 Main、Externality、Resource、Joint 顺序启动；书面规格复核时会话已不存在，现场只有 Main 100 个和 Externality 48 个种子分片，Resource/Joint 未生成，且没有证明四族正常结束的完成状态。

因此，该运行按“未证实正常结束的部分探索运行”冻结，不能补齐后冒充同构正式 pilot；任何实施还必须先完成来源/终止状态清点、选择性检查点和隔离，不能盲目提交全部工作树。

## 4. 目标论文定位

### 4.1 中文学位论文题目

> 多无人机边缘计算的干扰外部性感知调度与时限卸载

该题目保持在学校模板建议的 25 个汉字以内，同时覆盖主要研究对象和方法。

### 4.2 英文题目

> Externality-Aware Scheduling and Deadline-Constrained Offloading for Multi-UAV Edge Computing

最终英文题目可在不改变研究边界的前提下，根据 *Computer Networks* 编辑要求做语言层面的精简。

### 4.3 论文核心问题

在悬停多 UAV、跨 UAV 全频复用、UAV 内上行 NOMA 和有限 MEC CPU 的系统中，一个任务的服务 UAV、子载波或配对变化会同时影响多个接收端的干扰、SIC、速率和资源准入。本文研究：

1. 如何显式识别并审计跨接收端的第三方干扰外部性；
2. 如何在有限联合邻域中寻找严格改善的关联与配对调度；
3. 如何从硬截止时间推导最小卸载比例和最小 MEC CPU；
4. 如何在固定无线调度下实现最大按时完成基数的 CPU 准入；
5. 两个模块联合后是否产生可辨识、具有实际意义且可复现的收益。

## 5. 系统模型与研究边界

### 5.1 网络模型

- 系统包含 `K` 架固定悬停 UAV，每架 UAV 配备一台 MEC 服务器；
- `N` 个地面终端在二维区域内移动；
- 所有 UAV 复用 `S` 个上行子载波；
- 每个 `(UAV, subcarrier)` 资源块最多服务两个 NOMA 用户；
- 每个任务最多关联一个实际服务 UAV 和一个子载波；
- NOMA 角色由目标 UAV 上的接收功率排序规范化得到，不作为可任意设置的独立动作；
- 用户发射功率在主问题中固定；
- UAV 轨迹、飞行能耗、联合功率控制、跨时隙队列和重传不属于主问题决策变量。

显式调度对象为：

```text
x_i = (served, serving_uav, subcarrier, noma_role)
```

其中 `noma_role` 是派生字段。任务身份由 `(user_id, job_index)` 唯一确定，不能继续依赖只适用于单作业的 `slot * N + user` 假设。

### 5.2 信道、全频复用和 SIC

主信道采用 Al-Hourani Urban LoS/NLoS 概率加权大尺度信道增益，不叠加未建模的瞬时小尺度衰落。鲁棒性实验可使用经条款、适用高度和参数来源审计的 3GPP aerial 模型。正文必须区分“真实大尺度信道增益” `h` 和“估计大尺度信道增益” `h_hat`，不将其称为完整瞬时 CSI。

对任务 `i` 在接收 UAV `u_i`、子载波 `s_i` 上的跨 UAV 干扰定义为：

```text
I_i^cross = sum_{j: u_j != u_i, s_j = s_i} p_j h_{j,u_i}
```

令 `G_(u,s)` 为接收 UAV `u`、子载波 `s` 上的至多两个用户，并按实际接收功率非增序排列为 `pi(1),pi(2)`。稳定任务 ID 用于接收功率相等时的确定性 tie-break。第 `q` 个解码用户的 SINR 必须与实现统一为：

```text
gamma_pi(q) = p_pi(q) h_pi(q),u /
  (sigma_s^2 + I_u,s^cross
   + sum_{l>q} p_pi(l) h_pi(l),u
   + epsilon_SIC sum_{l<q} p_pi(l) h_pi(l),u)
```

其中 `sigma_s^2` 由噪声功率谱密度、子载波带宽和接收机噪声系数共同计算。单用户块使用同一公式，但两个组内求和为空。速率为 `R_i^ul=B_s log2(1+gamma_i)`；每一需要解码的层均执行冻结的 SIC 门限检查。

SIC 只消除同一接收 UAV、同一子载波 NOMA 组内按解码顺序允许消除的信号。其他 UAV 的同频发射始终作为跨接收端干扰。每次候选动作必须在实际接收 UAV 上重新计算接收功率、解码顺序、残余 SIC、SINR、SIC 可行性和速率。

调度规划使用 `h_hat`，实际结果复核使用 `h`。无 CSI 误差的主场景二者相同；CSI 误差鲁棒性场景中，估计信道上的计划约束违规为实现错误，而真实信道上的 realized SIC outage 是必须保留的结果指标，不属于代码硬约束违规。

### 5.3 周期性时限任务

每个用户生成周期作业 `J_{i,r}`，具有释放时刻 `a_{i,r}`、数据量 `L_i`、计算密度 `c_i`、绝对截止时刻 `d_{i,r}`、本地 CPU `f_i^loc` 和结果比例 `rho_i`。时隙长度为 `Delta`，剩余时限为：

```text
tau_i = d_{i,r} - a_{i,r}
```

主场景冻结 `0 < tau_i <= Delta`，使“释放时隙内完成”与绝对截止时间定义一致。用户级任务属性在种子初始化时冻结，作业按周期和初相位释放，不设置跨时隙等待队列。扩展模式包括周期抖动、随机突发和 GeoLife 移动轨迹，但不能把 GeoLife 称为真实 UAV-MEC 业务数据集。

MEC CPU 采用**静态并行虚拟份额模型**：同一时隙中，每个已准入任务获得持续、不可抢占迁移的固定虚拟 CPU 速率 `f_i^mec`，每架 UAV 始终满足 `sum_i f_i^mec <= F_u`。该模型表示容器/虚拟机 CPU 份额预留，不是单处理器 EDF 可抢占调度。DD-MCAA 的最长前缀定理只对这一模型成立；论文必须明确其保守性，不能外推到可随时间复用 CPU 的真实 EDF demand-bound 模型。

### 5.4 时延和完成判定

卸载比例为 `alpha_i`，本地与边缘分支并行。上行速率和下行结果返回速率分别记为 `R_i^ul`、`R_i^dl`：

```text
T_i^loc  = (1-alpha_i) L_i c_i / f_i^loc
T_i^edge = alpha_i L_i / R_i^ul
           + alpha_i L_i c_i / f_i^mec
           + rho_i alpha_i L_i / R_i^dl
```

任务按时完成当且仅当：

```text
max(T_i^loc, T_i^edge) <= tau_i
```

所有算法、Oracle、日志和论文统计必须调用同一个规范时延函数。正式主实验令 `rho_i=0`，明确采用“结果数据相对较小、忽略返回时延”的上行卸载模型。返回时延鲁棒性实验使用独立、固定且有参数来源的 `R_i^dl`；禁止继续用 `(1+rho_i)L_i/R_i^ul` 把上行 NOMA 速率冒充下行速率。任何时延语义变化都会使此前 pilot 对正式功效冻结失效。

### 5.5 无线发送、CPU 准入与规范化闭包

纯本地可按时完成的任务不进入无线调度。系统显式区分两个对象：

- `raw_radio_assignment/rate_snapshot r`：闭包前的无线位置、NOMA 角色、计划 SINR 和冻结速率；它可以包含最终无法获得 CPU 的候选任务；
- `closed_executable_schedule x=cl(r;p)`：资源策略 `p` 准入并删除未准入任务、完成必要重算后的可执行调度；其中每个保留的卸载任务均获 CPU 准入。

对需要卸载的端到端候选调度，策略相关评价器执行：

1. 规范化 NOMA 角色并计算上行速率；
2. 指定资源策略在每架 UAV 上选择 CPU 准入集合；完整 IEA-CAP 使用 DD-MCAA；
3. 同时移除已获无线位置但未获 CPU 准入的任务；
4. 重新计算受影响速率、角色和 CPU 需求，直到所有保留的无线任务均获准入。

该闭包只删除任务、不自动重新加入任务，最多删除当前候选中的全部任务，因此有限终止。被移除任务可在后续 Insert/Replace 邻域中重新竞争。主实验和 Joint 的每个方法都用其明示的资源策略形成各自闭包；IEA-CAP、端到端基线、增量评价器和 Exact Schedule Oracle 必须共享上述对象和闭包语义，避免“被拒任务是否继续产生干扰”的歧义。

Resource/H3 与 Exact Admission Oracle 是有意隔离 CPU 响应的条件实验：它们接收同一个 `r`，在闭包前任务集合和冻结速率上分别求准入集合；比较期间不先用 DD 删除任务，也不因某策略拒绝任务而重算无线速率。未准入任务不计为该策略完成，但冻结快照下的保守干扰保持相同。该实验只支持“给定无线分配与速率快照”的资源结论，不能冒充完整端到端可执行结果。

完整方法的规范邻域定义为：对原始合法动作 `m` 先形成 raw assignment，再比较 `N_cl(x)={cl(m(x);DD-MCAA)}` 中的闭包状态。因此“联合邻域局部最优”只相对于这个闭包规范化邻域成立。

### 5.6 全网目标

定义量化后的全网字典序目标：

```text
Phi(x) = (C(x), -F_min(x), -E_cross(x))
```

- `C(x)`：唯一到达任务中的按时完成数；
- `F_min(x)`：已准入卸载任务的最低 MEC CPU 总需求；
- `E_cross(x)`：每条已调度受害链路在其实际接收端经历的跨 UAV 同频干扰暴露之和。

同一 NOMA 块的两个任务会分别记录相同接收端上的跨 UAV 干扰，因此 `E_cross` 是 aggregate victim-link interference exposure，不是物理空间总辐射功率；兼容日志字段可以继续名为 `I_cross`，论文必须使用准确术语。比较优先级固定为完成数、CPU、干扰暴露。量化精度、稳定求和顺序和 tie-break 必须冻结，以保证串并行和不同硬件上的一致判定。

主指标 `C(x)` 使用全部唯一到达任务作为分母；同时预注册并报告 `requires_offload` 条件完成率和纯本地可完成比例。H1--H4 不得在看到 pilot 或正式结果后切换分母。

### 5.7 完整优化问题

对每个时隙，定义二元无线位置变量 `y_(i,u,s)`、CPU 准入变量 `a_i`、计划完成变量 `z_i`，以及连续变量 `alpha_i` 和逐服务 UAV 的 CPU 份额 `f_(i,u)`。令 `I_off` 为不能纯本地按时完成、必须卸载的任务集合。规范可行域至少包含：

```text
y_(i,u,s), a_i, z_i in {0,1}
0 <= f_(i,u),  i in I_off
sum_(u,s) y_(i,u,s) <= 1                       # 唯一服务位置
sum_i y_(i,u,s) <= 2                           # NOMA 块容量
a_i = sum_(u,s) y_(i,u,s),  i in I_off          # 闭包状态只保留获准入发送者
z_i = a_i,  i in I_off                          # 保留卸载任务必须计划按时完成
a_i alpha_i^min <= alpha_i <= a_i               # 未准入时 alpha_i=0
0 <= f_(i,u) <= F_u sum_s y_(i,u,s)             # CPU 只能位于服务 UAV
sum_i f_(i,u) <= F_u                            # 静态 CPU 份额
sum_u f_(i,u) >= a_i f_i^min(alpha_i,y,h_hat)   # 准入任务满足最低 CPU
a_i=1 => max(T_i^loc,T_i^edge) <= tau_i         # 计划完成约束
y-layer active => gamma_i(y,h_hat) >= gamma_min # 每个需解码层的计划 SIC/SINR
```

纯本地可完成任务的 `z_i` 由本地分支确定，且强制 `y=a=f=0`。时延公式中的 `f_i^mec=sum_u f_(i,u)`。上述等式描述最终 `closed_executable_schedule`；闭包前 raw assignment 允许 `sum y=1,a=0`，但不能作为 Exact Schedule Oracle 的最终解。`f_i^min` 只对已准入卸载任务定义，未准入任务对 CPU 目标贡献为零。NOMA 顺序、SINR、速率和最低 CPU 都由 `(y,h_hat)` 派生，形成离散—连续耦合的字典序组合优化问题：

```text
lexicographically maximize (sum_i z_i, -sum_(i,u) f_(i,u), -E_cross(y))
```

计划 SIC 门限施加于冻结解码顺序中每个实际调度层在其解码步骤的 `gamma_i`；除非模型、实现和实验另行统一定义，不得暗中加入仅基于接收功率差的第二套门限。实现阶段以单一规范函数冻结此语义并用属性测试覆盖 strong、weak 和 solo 三种情况。

Exact Schedule Oracle 对同一个规范可行域求全局最优；IEA-CAP 是该可行域上的联合邻域局部搜索。规划输出用真实 `h` 再执行 realized-outcome 复核，但不能用真实信道反向修改同一正式调度。

旧稿中的 Independent-Set NP-hardness 证明在独立复核通过前不得复用。最终论文要么给出对上述规范问题成立的严格约化，要么只陈述其混合离散耦合性质并用 Exact 小实例刻画，不以未经验证的 NP-hardness 作为贡献。

## 6. 贡献架构与主张边界

### 6.1 主要算法贡献：IEA-CAP

IEA-CAP 是干扰外部性感知的跨 UAV 关联与 NOMA 配对方法。既有研究已经涉及 externality-aware association、potential game、NOMA 分组和多 AP/Multi-UAV MEC，因此“外部性感知”“有限改进收敛”和“外部性证书”本身均不作为新颖性主张。本文唯一安全的核心差异定位为：**在全频复用上行 NOMA-MEC 中，对候选关联/配对动作在实际接收 UAV 上重算跨接收端第三方干扰，并把该变化与解析时限 CPU 响应纳入同一次全网候选评价。**

具体设计包括：

1. 用统一调度对象表示服务归属、子载波和规范化 NOMA 角色；
2. 联合考虑 Replace/Insert/Drop、Position/IntraRelocate 和 CrossSwap/CrossRelocate 邻域；
3. 对每个候选在所有受影响同频接收端重新评价；
4. 在嵌入 DD-MCAA 资源响应后使用全网字典序目标接受动作；
5. 生成参与者、第三方影响和全网干扰变化的可复核外部性证书；
6. 使用多个确定性起点和最佳改进，减少单起点首次改进的路径依赖；
7. 以增量精确评价降低完整重算成本，并通过随机动作等价测试保证语义一致。

外部性证书至少记录参与任务、第三方任务、源/目标块、全部受影响同频接收块、角色变化、计划/真实 SINR、SIC、速率、最低 CPU、准入和完成状态。证书必须满足聚合一致性：任务级完成数/CPU delta 与全网目标 delta 相等，块级干扰暴露 delta 的稳定求和与 `E_cross(x')-E_cross(x)` 相等。

论文至少证明：

- 合法邻域动作保持任务唯一性、块容量和服务归属一致性；
- 每个接受动作严格改善规范化 `Phi`；
- 在有限状态空间和严格改善下有限终止；
- 只有以“无改善邻居”结束时，才能声称联合邻域局部最优；
- 完整依赖集合下，增量评价与全网完整重算等价。

增量依赖闭包必须形式化包含：源/目标子载波上的所有 UAV 接收块、角色改变的组内任务、服务 UAV 改变的任务、包含任一速率/归属受影响任务的 MEC 服务器，以及 CPU 排序变化波及的同服务器任务。10,000 个随机动作等价测试是实现证据，不能替代依赖闭包证明。

令无线位置数 `P=2KS`、需卸载任务数 `A<=N`、起点数 `M`、实际接受动作数 `R`，并令 `C_eval(A,K,S)` 为一次候选的依赖重算、策略响应和闭包总成本。完整 Replace 类候选为 `O(PN)`，Position/CrossUAV 候选为 `O(P^2)`；实现相关保守界写为：

```text
O(M (R+1) (PN + P^2) C_eval(A,K,S))
```

实现阶段必须从冻结的数据结构和闭包算法单独推导 `C_eval`，不能把 DD 的排序成本冒充整个 IEA-CAP 成本。在“批量删除未准入任务后，剩余任务 SINR/速率不降、`f_i^min` 不升”的单调性证明和属性测试均通过时，一次闭包只需一次 DD 排序和一次删除后复核；否则按最多 `A` 轮删除给出保守闭包界。`R` 是实例相关量；有限状态空间可能很大，不把上述表达式包装成一般多项式最坏步数保证。

`max_iterations` 只是安全保护。正式运行触及上限时必须记录 `stop_reason=iteration_cap`，该实例不能声称局部最优；若正式主结果中存在此类终止，局部最优主张门禁失败。

### 6.2 理论支撑贡献：DD-MCAA

本地分支的截止约束给出最小卸载比例：

```text
alpha_i^min = clip(1 - tau_i f_i^loc / (L_i c_i), 0, 1)
```

令：

```text
b_i = L_i/R_i^ul + rho_i L_i/R_i^dl
```

在 `tau_i - alpha_i b_i > 0` 时，边缘分支需要的最低 CPU 为：

```text
f_i^min(alpha_i) = alpha_i L_i c_i
                   / (tau_i - alpha_i b_i)
```

主实验 `rho_i=0`，因此 `b_i=L_i/R_i^ul`。在可行区间内该函数关于 `alpha_i` 严格递增，因此最低 CPU 出现在 `alpha_i=alpha_i^min`。对每架 UAV，将可行任务按 `(f_i^min, task_id)` 稳定升序排列并选取预算内最长前缀。

论文至少证明：

- `alpha_i^min` 的正确性；
- `f_i^min` 的正确性和边界条件；
- 单位任务价值、独立 CPU 消耗和逐 UAV 独立预算下，最长前缀获得最大完成基数；
- 在最大基数集合中，该前缀具有最低 CPU 总需求；
- 在固定无线分配、冻结速率和逐 UAV 独立预算下，单次 DD-MCAA 资源响应的总排序复杂度为 `O(A log A)`，其中 `A` 为需卸载候选任务数。

该 `O(A log A)` 不包含无线速率/SIC 重算、闭包、候选枚举或 IEA-CAP 搜索。DD-MCAA 的定理严格依赖第 5.3 节的静态并行虚拟 CPU 份额模型。它不适用于随时间复用单处理器的 EDF 可抢占调度，也不声称解决异质任务价值、多服务器联合拆分、任务依赖、缓存耦合或一般无线—计算联合全局最优问题。鉴于 deadline admission 已有接近研究，本模块定位为有严格适用条件的解析资源响应贡献，不使用“首次提出一般时限准入”的措辞。

### 6.3 联合方法：EAD-JSRA

EAD-JSRA 是 IEA-CAP 与 DD-MCAA 的闭环组合：无线候选决定接收端、干扰、SIC 和速率，DD-MCAA 返回最低卸载比例、CPU 需求和准入结果，规范化闭包移除未准入无线任务，IEA-CAP 再用全网目标决定是否接受闭包后的候选。

EAD-JSRA 不包装成第三项独立算法创新；其价值由无线机制、资源模块和 2×2 联合实验共同证明。

### 6.4 明确禁止的表述

论文不得声称：

- 首次研究多 UAV NOMA-MEC；
- 首次研究跨 UAV 用户关联或 NOMA 配对；
- 局部最优等于全局最优；
- 固定功率、固定悬停问题已经联合优化功率或轨迹；
- 单时隙模型已经解决长期队列和跨时隙截止；
- GeoLife 是真实 UAV-MEC 任务数据；
- 统计显著自动等于工程上有意义；
- canonical-v1 可以验证 dirty v2 代码。

## 7. 基线、精确参照和可比性

### 7.1 主实验基线

- `Local-only`；
- `Fair-OFDMA`；
- 局部信道/速率 `Greedy`；
- 使用统一物理评价的 `IA-Greedy`；
- `per-UAV ILP`；
- `Exchange-Stable`；
- `Intra-Global`；
- `Cross-Bilateral`；
- 单起点首次改进消融；
- 完整 IEA-CAP。

结构消融必须共享初始解、任务轨迹、信道轨迹、候选穷尽规则和资源响应，仅改变被研究因素。

`Fair-OFDMA` 必须保持相同总频谱、用户功率上限、UAV CPU 和任务轨迹；`per-UAV ILP` 只能称为逐 UAV 局部最优初始化；所有基线使用同一个规范时延函数和真实结果复核。若算法计算预算不同，应报告实际运行时间和候选数，而不是通过削弱完整方法实现表面等预算。

### 7.2 资源基线

在同一个冻结无线调度上比较：

- `Fixed-Half`；
- `Crossing/AlphaMin`；
- `Deadline-First-MinCPU Admission`（代码兼容名可暂保留 `EDF-MinCPU`）；
- `MinSlack-MinCPU`；
- `DD-MCAA`；
- `Exact Admission Oracle`。

该控制组只是按最早截止时间排列的静态份额准入，不是 EDF 可抢占 CPU 调度，论文不得简称为“EDF 调度”。H3 只允许使用固定无线调度比较，不能让不同资源策略反向改变无线搜索后再声称纯资源贡献。

当前 fixed-radio 调度由 DD 响应参与生成，会使 H3 只能解释为“条件于 DD 生成调度的资源策略效应”。正式设计增加至少一个策略中立无线生成器和一个 Deadline-First 生成调度的对称敏感性；只有方向跨调度来源稳定时，才能将 DD 优势推广到一般固定无线调度。

这里的 `Exact Admission Oracle` 只在同一个闭包前任务集合与冻结速率快照上穷举 CPU 准入子集，用于验证 DD-MCAA 的最大基数/同基数最低 CPU 定理；它不得接收 DD 闭包后的幸存集合，也不得与下节联合枚举无线状态的 `Exact Schedule Oracle` 混名。

### 7.3 Exact Schedule Oracle

Exact Schedule 小实例在第 5.7 节的完整闭包可行域上联合选择无线位置与 CPU，按照相同字典序目标求全局最优。只有完整状态枚举结束，或求解器返回可核验的零 optimality gap/bound certificate 时，实例才能标记为 Exact。超时实例只报告 incumbent、best bound 和剩余 gap，不计入“达到全局最优比例”。逐时隙依次报告完成数 gap、完成数相同时的最低 CPU gap、前两层相同时的干扰暴露 gap。

非平凡性分成两个独立门禁：

- **CPU 非平凡：** 存在一组任务均可单独无线可行、存在合法无线安排，且至少一架 UAV 上这些任务的 `sum f_i^min > F_u`；
- **调度非平凡：** 存在至少两个可行无线状态，且无线容量/SIC/外部性使完整字典序最优不是“服务全部任务”的直接结果。

支持 DD 最优性性能证据时，至少 30% 正式 Exact 实例必须 CPU 非平凡；支持 IEA 接近最优证据时，至少 30% 必须调度非平凡。否则对应 Exact 结果只能作为正确性测试。

### 7.4 文献适配基线

在标注任何 `adapted` 文献基线前，必须建立决策变量、目标、约束、信道、功率、移动性和任务模型映射表。closest-work 审计至少覆盖：

- Guo et al., uplink multi-cell NOMA association/grouping and externality potential game, ICC 2019；
- Xu et al., multi-AP NOMA-MEC association/resource/power with interference and computation externalities, *IEEE Access* 2020；
- Katwe et al., multi-UAV NOMA dynamic clustering/interference, *IEEE TWC* 2022；
- Uwizeyimana et al., multi-UAV NOMA joint association/pairing, *IEEE Access* 2024；
- Chen et al., QoS-oriented NOMA multi-UAV cooperative MEC, *IEEE TWC* 2026；
- Lyu et al., deadline-sensitive MEC admission, *IEEE TCOM* 2018。

至少实现一个经上述映射审计、能够在当前统一物理层下公平运行的 closest-work adaptation。Exact Oracle 和自定义强基线不能替代文献可比性；源码或定义不足时可以如实声明局部无法忠实复现，但不得因此跳过全部 adapted baseline，也不得仅复用论文名称包装自定义启发式。

英文稿是否达到“CCF-B submission-ready”必须把 closest-work 可比性作为独立审查项。

## 8. 实验与统计协议

### 8.1 证据族

正式证据包括：

1. Main；
2. Externality；
3. Resource；
4. Exact；
5. Joint；
6. Scaling；
7. Robustness；
8. Trace-driven。

### 8.2 主假设

| 假设 | 实验族 | Treatment | Control | 最小有意义效应 |
|---|---|---|---|---:|
| H1 | Main | `full` | `intra_global` | 0.5 pp |
| H2 | Externality | `full` | `cross_bilateral` | 0.5 pp |
| H3 | Resource | `fixed_radio_dd` | `fixed_radio_edf` | 2.0 pp |
| H4 | Joint | `iea_dd` | `exchange_edf` | 2.0 pp |

H4 是两个因素同时改变的**端到端完整栈效应**，不是 2×2 交互。联合实验还必须单独报告：

```text
wireless effect under DD = iea_dd - exchange_dd
wireless effect under Deadline-First = iea_edf - exchange_edf
resource effect under IEA = iea_dd - iea_edf
resource effect under Exchange = exchange_dd - exchange_edf
interaction = (iea_dd - exchange_dd)
              - (iea_edf - exchange_edf)
```

interaction 作为预注册次要 estimand 报告配对区间，不套用 H4 的 2 pp MDE，也不能用 H4 对角差替代。

H2 的唯一主格冻结为 `K=3,N=45,S=3`、`pairwise_overlap`、`cluster_std=5 m`、`period_choices=[2]`、`cross_interference_scale=1.0` 的高重叠高负载场景。其他 overlap/load/coupling 格点均为次要机制分析，不能替换 H2。

### 8.3 种子池

原注册表的 Resource 正式种子 `12000` 已被开发测试实际生成并评价，因此 12000--18199 整组废弃，以避免不同实验族使用不同“未见”标准。新的候选池为：

| 实验族 | 正式种子 | 时隙 |
|---|---:|---:|
| Resource | 22000--22199 | 100 |
| Externality | 23000--23199 | 100 |
| Main | 24000--24199 | 100 |
| Robustness | 25000--25029 | 50 |
| Exact | 26000--26049 | 20 |
| Joint | 27000--27199 | 100 |
| Scaling | 28000--28029 | 50 |
| Trace-driven | 29000--29029 | 50 |

开发种子为 0--999；功效 pilot 为 1000--1099。旧规格中的 2000--18000 系列和名称为 confirmatory 但仍使用 pilot 种子的遗留配置不得进入 canonical-v2。

候选池在实施阶段通过源码、测试、产物和访问日志扫描后才写入 `confirmatory_lock.json` 成为权威。所有测试实际生成轨迹时只能使用 0--999；仅断言配置中的整数不算污染，但任何轨迹生成、算法运行或结果分析都写入 seed-access ledger。若新池已被读取，必须在不查看替代池结果的前提下整体轮换。

### 8.4 功效冻结

设计批准时正在运行的 pilot 来自后续将被修改的模型、算法和统计代码，因此只作为探索性效应、方差和运行时间证据，不能自动冻结最终样本量。最终功效 pilot 必须使用 1000--1099，且与正式运行在场景、任务、评价、搜索、指标和分析语义上完全一致；只允许 run ID、种子池、日志级别和正式方法超集这些预先列明的差异。H2 pilot 必须与上节唯一主格一致。

最终功效模拟的 estimand 是种子级平均完成率配对差，主检验与正式分析统一为双侧 paired t-test，并且只对 H1--H4 四个主检验 p 值执行同一个 Holm family。因为四个正式实验族使用互斥种子池，四个 pilot 向量在每次功效模拟中独立重采样；不能用相同重采样索引制造正式设计中不存在的跨假设相关结构。样本量只能从 `{50,100,200}` 选择，并取使四个假设在真实效应等于各自 MDE 时“拒绝零均值差”的 Holm 校正功效均不低于 80% 的最小公共 `N`。

该 80% 只表示拒绝 `H0: delta=0` 的功效，不称为“完整 claim-gate 功效”；点估计达到 MDE 和置信区间位置另行解释。若最终代码、配置、指标或分析哈希改变，power lock 立即失效，必须重跑 pilot；无法重跑时可以保守使用 `N=200`，但必须声明没有经验证的 80% 功效保证。

功效完成后生成不可人工编辑的 `confirmatory_lock.json`，至少包含：

- source commit 和 clean/patch snapshot hash；
- registry、每个主格配置和分析代码哈希；
- outcome/均值差 estimand、双侧 paired t-test、Holm family 和 MDE；
- `selected_n` 与每个实验族的确切种子前缀；
- pilot 输入哈希、模拟次数、分析随机种子和功效结果；
- 正式 runner 允许的命令和环境摘要。

所有正式 runner 必须读取并验证 lock；仅生成 `power_decision.json` 而不约束正式种子数量不算冻结。

正式实验开始后禁止：

- 追加样本或提前停止；
- 修改 MDE；
- 根据方法输赢调整场景；
- 删除不利种子；
- 将辅助分析替换为失败的主假设。

若所需样本量超过 200，相应主张标记为证据不足。

### 8.5 推断方法

- 独立统计单位：一个随机种子的完整 episode；
- 公共随机数：所有方法共享位置、任务、真实/估计信道和外生随机量；
- H1--H4 主 estimand：全部到达任务的种子级 on-time completion-rate 配对均值差；
- 关键次要 estimand：条件于 `requires_offload` 的完成率差，以及纯本地可完成比例；
- 主区间：种子级 10,000 次配对 bootstrap 95% CI；
- 主检验：均值 estimand 对齐的双侧 paired t-test；
- 敏感性检验：双侧 Wilcoxon signed-rank，仅检验秩位置稳健性，不用于 H1--H4 主门禁；
- 多重比较：H1--H4 的四个 paired t-test p 值共同 Holm 校正；
- 效应量：matched rank-biserial correlation；
- Exact：逐实例 gap 和相等率，不做伪显著性检验；
- 图表同时显示种子分布、点估计和置信区间，不只报告柱状均值。

正式 analyzer 必须由 registry 驱动并精确生成 H1--H4；当前 v1 `COMPARISONS` 列表和“对所有基线一起 Holm”均不符合本设计。辅助基线在各自明确命名的 family 内校正，不能改变主 family。每个 MDE 还要换算为“每个 episode 平均新增完成任务数”，给出工程量级解释。

### 8.6 鲁棒性和外部效度

核心鲁棒性轴包括：

- K3 下 N=30/45/60 的负载；
- `(K,N)=(2,30),(3,45),(4,60),(5,75)` 的等人均扩展；
- CSI 大尺度增益相对误差；
- SIC 门限与残差；
- CPU 与 deadline 压力；
- periodic、jitter、burst 到达；
- 同质和异质 CPU；
- Al-Hourani 与经审计的 3GPP aerial 模型；
- GeoLife 步行/骑行片段。

鲁棒性采用预先定义的单轴或小型因子设计，不进行不可承受的全笛卡尔积。Externality 只有 H2 主格使用功效冻结后的 `N`；其余八个格点预先冻结为较小描述性样本，不纳入 H1--H4，也不作逐格显著性筛选。

GeoLife 建立独立 `trace_lock.json`，记录数据集哈希、许可/来源、源参与者 ID、交通方式、片段时间、重采样参数、复合场景 ID、29000--29029 候选随机种子和分组。用于规则/参数开发的参与者与最终描述性场景参与者互斥；同一源参与者不得跨 train/calibration/final split。随机种子只控制与已冻结轨迹正交的外生量，不能改变参与者划分。GeoLife 只作描述性配对补充；若给出区间，必须按源参与者聚类。

打开正式种子前必须生成计算预算，包含每族 CPU 小时、预计墙钟、峰值内存、磁盘、事件日志上限、并发数和失败重跑余量。若预算不可承受，只能在正式种子打开前缩小预注册的次要描述性网格，不能事后删格。

## 9. 证据门禁和负面结果

### 9.1 完整性硬门禁

以下任一失败都禁止发布 canonical-v2：

1. 全量自动化测试失败；
2. 正式调度在规划信道上出现任务重复、块容量、计划 SIC 或 CPU 硬约束违规；
3. 方法之间任务或外生轨迹不一致；
4. 增量评价与完整重算存在语义差异；
5. DD-MCAA 与 Exact Admission Oracle 在适用条件下不一致；
6. 正式分片哈希、配置、种子、源码或环境信息不完整；
7. 分析图表不是从 canonical 输入自动生成；
8. 存在手工改数、未运行数字或伪造基线。

CSI 误差场景在真实信道上产生的 realized SIC outage/掉期不是完整性失败，而是必须保留的鲁棒性结果。`all_integrity_gates_pass` 不得把真实环境不确定性与实现违规混为一谈。

### 9.2 证据发布与性能主张分离

canonical-v2 是有效实验事实的不可变证据包。H1--H4 出现负面结果时，只要完整性硬门禁通过，仍应发布 canonical-v2，并在 `acceptance.json` 中记录主张失败。

性能失败不阻止真实证据归档；它只阻止对应论文主张。这一分离避免因结果不利而拒绝发布数据。

### 9.3 主张门禁

- 95% bootstrap 均值差 CI 下界大于 0，且 H1--H4 的 paired t-test 经 Holm 校正后显著，方可写“稳定正向优势”；
- 点估计达到相应 MDE 但 CI 下界未超过 MDE，只能写“观察到的平均幅度达到预设阈值，但不确定区间仍包含较小效应”；
- 只有 95% CI 下界同时超过 MDE，才可写“证据支持效应至少达到预设的实际意义阈值”；
- 低于 MDE 但 CI 下界大于 0，只能写“统计上可辨识但实际效应较小”；
- CI 包含 0 时，必须写证据不足或未观察到稳定优势；
- 负效应必须保留并讨论。

### 9.4 正确性和机制门禁

- 100% 接受动作严格改善规范化 `Phi`；
- 正式终止解只有在 `stop_reason=no_improving_move` 时才能记为局部最优；
- 外部性压力场景存在逐事件可复核的第三方影响；
- DD 与 Oracle 的完成基数和同基数最低 CPU 100% 一致；
- 至少 10,000 个随机合法动作的增量评价逐字段等价；
- 外部性证书的任务级和块级 delta 满足聚合守恒；
- Exact 仅对有穷尽/零 gap 证书的实例按三层字典序报告最优 gap，不用第一层相等替代完整最优；
- CPU/调度非平凡性、iteration-cap、外部性事件和三层 gap 均进入机器可读 gate schema。

### 9.5 运行时间表述

必须报告参考硬件、P50/P95 单槽运行时间、候选数、缓存命中率、峰值内存和实验总 CPU 时间。不预设无法证明的 `<1 s` 门禁；只有实测 P95 不超过调度时隙且部署假设成立时，才能使用“实时”或“在线时限内”表述。

增量评价的速度提升作为实测结果报告，不预先强制 3 倍。若没有实质加速，不能把效率提升列为贡献。

## 10. Canonical-v2 证据架构

```text
artifacts/canonical/v2/
  MANIFEST.json
  SHA256SUMS
  full/
    configs/
    source_snapshot/
    environment/
    traces/
    raw_shards/
    summaries/
    event_certificates/
    analysis/
      statistics/
      figures/
      latex/
      claim_matrix.csv
      acceptance.json
      qa_report.md
  light/
    configs/
    summaries/
    statistics/
    figures/
    latex/
    claim_matrix.csv
    acceptance.json
    qa_report.md
```

`full/` 是本地完整、自包含、不可变的权威审计归档；所有 raw shard 字节、权威统计、图表、LaTeX 宏、claim matrix、acceptance 和 QA 报告都必须实际位于 full 内并被根 `SHA256SUMS` 覆盖，不允许只引用生命周期不受控的外部 CAS。`light/` 是从 full 确定性导出的论文构建和未来获授权后可分发子集；light 中每个摘要和图表都能沿哈希链解析到 full，不能以“publisher 未复制 task/slot 行”为由丢失原始证据。

数据流固定为：

```text
frozen source + configs
  -> immutable seed shards
  -> checksum and constraint audit
  -> paired statistical analysis
  -> canonical-v2/full
  -> canonical-v2/light
  -> figures/tables/LaTeX macros
  -> USTC thesis and Computer Networks manuscript
```

`acceptance.json` 必须由 registry 驱动的 v2 analyzer/publisher 生成，并区分：

- integrity gates；
- correctness gates；
- H1--H4 claim gates；
- optional efficiency/real-time claims。

两份论文禁止直接读取 `artifacts/runs/`、pilot、smoke、legacy results 或 canonical-v1 的数值。

`claim_matrix.csv` 中每条证据路径必须可解析、文件存在且被 `SHA256SUMS` 覆盖。正式运行必须来自 clean commit，或同时归档完整二进制 patch/source snapshot；只记录 `dirty=true` 而不保存变更内容不满足复现要求。

## 11. 中科大学位论文设计

### 11.1 模板与前后置内容

采用中国科学技术大学 2026 年官方 `ustcthesis`：

```latex
\documentclass[
  degree=master,
  degree-type=professional,
  language=chinese
]{ustcthesis}
```

2026-07-17 发布的官方 Word 撰写规范晚于 2026-07-02 发布的 LaTeX 包。引入时必须同时冻结两者的 URL、发布日期、版本和 SHA-256，并建立逐项差异矩阵；发生冲突时以研究生院更新更晚的规范和学院书面要求为准，不能用“能够编译”替代格式合规。快速检查可使用 Tectonic；最终定稿必须从干净环境在 TeX Live 2021 或更新版本中按官方说明使用 XeLaTeX/`latexmk` 编译。

必须包含：

- 中英文题名和完整学籍信息；
- 原创性与使用授权声明；
- 如实填写的生成式 AI 工具使用声明；
- 中英文摘要和相互对应的关键词；
- 目录、插图清单、附表清单；
- 符号与缩略语说明；
- 参考文献、附录、致谢和在读期间科研成果。

最终提交检查还包括：双面打印和书脊、正文第 1 章从右页开始、原创性/授权/AI 声明签字与日期、纸质版和电子版内容一致、提交日期不晚于学位授予日期。`degree-type=professional` 必须与学籍系统和学院确认；不得仅凭项目描述猜成 `engineering`。实践导师按较新的 2026-07-17 Word 规范和学院要求处理，专业学位无实践导师时不得虚构。

正文目标为 35,000--45,000 个汉字，且机器检查不少于学校一般要求的 30,000 字。中文摘要约 600 字，关键词 3--8 个。参考文献使用 GB/T 7714--2025 顺序编码制。

作者姓名、学号、导师、实践导师、院系、学科专业、研究方向、提交日期和成果由用户提供，禁止猜测。

### 11.2 章节结构和篇幅目标

| 章节 | 内容 | 建议正文汉字 |
|---|---|---:|
| 第 1 章 | 绪论 | 4,000 |
| 第 2 章 | 理论基础与相关工作 | 6,000 |
| 第 3 章 | 系统模型与问题定义 | 5,000 |
| 第 4 章 | IEA-CAP | 6,000 |
| 第 5 章 | DD-MCAA 与 EAD-JSRA | 4,500 |
| 第 6 章 | 系统实现与实验方法 | 5,000 |
| 第 7 章 | 实验结果与机制分析 | 7,000 |
| 第 8 章 | 工程部署与研究局限 | 3,000 |
| 第 9 章 | 总结与展望 | 1,500 |

篇幅是规划目标，不通过重复背景、重复结果或大段代码凑字数。

### 11.3 附录

- 命题和定理完整证明；
- Exact Oracle 定义；
- 参数、来源和配置表；
- 统计协议和补充结果；
- 复现命令、环境与校验和说明；
- 必要的外部性证书案例。

## 12. Computer Networks 英文稿设计

活动稿迁移到 Elsevier 官方 LaTeX 模板，不继续以 `IEEEtran` 作为投稿稿。

正文结构：

1. Introduction；
2. Related Work；
3. System Model and Problem Formulation；
4. IEA-CAP；
5. DD-MCAA and Joint EAD-JSRA；
6. Experimental Methodology；
7. Results and Mechanism Analysis；
8. Discussion and Limitations；
9. Conclusion；
10. Appendices/Supplementary Material。

投稿包包括：

- 不超过 250 词的独立摘要；
- 1--7 个关键词；
- 3--5 条 highlights，每条不超过 85 个字符；
- CRediT 作者贡献；
- Funding、Competing Interests 和 Data Availability；
- 生成式 AI 使用声明；
- glossary；
- 每位作者不超过 100 词的简介和独立证件式照片；
- 独立 competing-interest 声明文件；
- 可编辑 LaTeX 源文件、独立图片和带说明的补充材料；
- 补充证明、配置、附加结果和复现材料；
- 必要时使用 TikZ/Matplotlib 制作的确定性矢量 graphical abstract。

根据期刊现行政策，生成式 AI 不用于创建或修改投稿图片。所有系统图、流程图、数据图和 graphical abstract 必须由代码、矢量绘图或人工布局生成。

投稿前增加 scope gate：英文稿必须以跨层网络调度、离散网络算法、资源管理和可复现系统机制为主线，不能写成只讨论 NOMA/SIC 的纯物理层论文，以降低 *Computer Networks* desk-reject 风险。AI 声明置于参考文献之前。

Data Availability 采用期刊允许的真实选项：获用户明确外部上传授权后，将轻量证据、代码版本和必要数据归档到可引用存储库；未获授权前，只能在投稿包中说明拟共享范围或不能共享的理由，不能把本地 `canonical-v2` 称为公开数据存储库，也不能声称已经完全满足在线提交步骤。

## 13. 双论文单一事实源

两份论文共享：

- 同一个模型符号表；
- 同一个经过 DOI、年份、卷期和引用对应性审计的 `.bib` 数据库；
- 同一个 canonical-v2；
- 同一套统计脚本、绘图数据和结果宏；
- 同一个 `claim_matrix.csv`。

学位论文使用 `ustcthesis` 的 GB/T 7714--2025 样式，英文稿使用期刊要求的数字顺序样式；二者可以使用不同 `.bst/.bbx`，但不得维护互相冲突的文献元数据。

两份稿件不直接复用同一个带文字的图片文件：同一数据和绘图代码分别生成中文标注的 thesis target 与英文标注的 journal target。每张图保存输入数据哈希、脚本、生成命令和人工复核记录。

建立 thesis--journal 重叠与版权台账：学位论文已经公开时在 cover letter 中如实披露；期刊先发表时不直接复用出版商排版后的图文资产，源数据和作者自制矢量图按许可处理。

`claim_matrix.csv` 至少包含：

```text
claim_id, claim_text, scope, equation, theorem,
test, experiment_family, artifact_path, figure_or_table,
thesis_section, paper_section, gate_status
```

当前 `paper/generated_results.tex`、`generated_statistics.tex` 和 `generated_robustness.tex` 在 canonical-v2 发布前不得作为 v2 结论来源。

## 14. AI 使用、作者责任和图像政策

中科大学位论文和英文投稿稿均需如实披露 Codex/生成式 AI 的使用。披露范围应覆盖实际发生的项目审计、代码与测试辅助、文献组织、数据分析检查、结构重写和语言修改。

必须满足：

- AI 不列为作者；
- 用户逐段审核文字、公式、引用和结论；
- 所有文献 DOI 和元数据由程序或人工复核；
- 实验结果由真实运行产物生成；
- 用户对研究设计、数据真实性、原创性和最终提交承担作者责任；
- 不写与实际使用不符的“未使用 AI”声明；
- 不使用生成式图像进入 *Computer Networks* 投稿稿。

中科大当前声明模板还要求作者确认核心观点、研究设计、实验方案和论证过程由本人独立完成。鉴于 Codex 已深度参与方案审计和设计，学生必须在签署前独立复核、重新推导并能在答辩中解释全部核心模型、公式、算法、实验和结论，保留人工决策/修改记录，并与导师确认实际 AI 使用范围是否符合学校细则。若无法真实满足声明文字，必须停止签署并向导师或学位办公室取得书面指导；本项目不能替学生作出虚假独立完成声明。

## 15. 执行、安全隔离与提交策略

### 15.1 规格与计划门禁

1. 本设计文档单独提交；
2. 完成独立设计审查并修正；
3. 用户审阅书面规格；
4. 编制逐文件、逐测试和逐实验实施计划；
5. 用户审阅实施计划后才开始算法和论文实现。

### 15.2 选择性安全检查点

实施开始时，仅检查点化与研究有关的：

- `uav_mec/`；
- `tests/`；
- `configs/thesis/`；
- `thesis/`；
- `paper/` 中的源码和文献；
- `docs/` 中的研究规格、计划和审计；
- 必要的 Makefile/README 变更。

明确排除：

- `artifacts/pilot/` 和 `artifacts/smoke/`；
- `deliverables/defense_2026/` 的用户材料；
- 生成 PDF、日志、缓存和大体积图像；
- 与本研究升级无关的文件。

检查点前保存文件清单、`git status`、diff stat 和哈希。不得使用 destructive reset/checkout 清理用户工作树。

当前原工作树中的研究代码、正式配置和分析脚本在部分 pilot 的来源与终止状态清点完成、选择性检查点建立前保持字节级不变；规格文档是此阶段唯一允许提交的文件。现有 pilot 必须尽可能保存实际启动命令、解析后的配置、分片清单、日志和可重建源码状态的提交/补丁哈希。若进程、分片或来源状态不能完整还原，该批结果只能标记为探索性，不得进入 canonical-v2 或确认性检验，也不得与以后补跑的分片拼接后称为同一次冻结 pilot。

每个检查点和后续提交都使用显式文件 allowlist。禁止使用 `git add .`、`git add -A`、`git stash`、切换原工作树分支、`git clean`、`git reset` 或其他会隐藏、移动、覆盖用户材料或正在写入产物的操作。提交前必须检查 cached 文件清单，并断言以下路径或类型的暂存数量为零：pilot/smoke 产物、答辩材料、PDF、PPTX、DOCX、日志、缓存和未列入计划的大文件。当前 `.gitignore` 尚不足以保护所有 pilot/smoke 路径，实施计划必须先增加针对性忽略/校验规则，不能把“未被忽略”等同于“允许提交”。

仓库内已有的说明文件（包括可能过时的 `CLAUDE.md`）不能直接作为现行事实源；实施阶段必须逐项与代码、测试、配置和本规格核对并显式更正。

### 15.3 隔离 worktree

从显式记录的选择性检查点提交 SHA 创建命名分支和 sibling worktree：

```text
/home/link/my_workspace/.worktrees/master-thesis-ccfb
```

原工作树继续保存 pilot 和用户已有材料，并在迁移后对研究实现保持只读；所有后续研究代码、配置、论文和实验基础设施变更只在新 worktree 进行。新 worktree 创建后先执行：

- 依赖和环境检查；
- 全量测试；
- canonical-v1 校验和验证；
- 当前论文和英文稿的基线编译；
- 最小 smoke 实验。

基线失败时先诊断，不把已有失败带入实施。

## 16. 实施阶段

### Phase 0：设计与计划

- 完成本规格、独立审查、用户审阅和实施计划。

### Phase 1：安全与语义基线

- 冻结已停止的部分探索性 pilot，清点其命令、配置、分片、日志、来源状态和可判定的终止原因，不续跑、不拼接为正式证据；
- 使用显式 allowlist 建立选择性检查点，再从记录的 SHA 创建 worktree；
- 建立任务、信道、SIC、调度、时延和日志的唯一语义；
- 为现有模型错误和论文—代码不一致编写失败测试。

### Phase 2：正确性与算法闭环

- DD 解析边界和 Oracle 属性测试；
- IEA 联合邻域、终止原因和局部最优复核；
- 增量评价随机等价测试；
- 外部性证书；
- 固定无线资源实验和 2×2 组合；
- 强基线与 closest-work 映射审计。

### Phase 3：实验基础设施

- 机器可读 registry；
- 分片、resume、merge 和冲突拒绝；
- runtime/capacity 审计；
- GeoLife 和鲁棒性场景；
- canonical-v2 publisher 和 claim gates；
- 图表与 LaTeX 宏自动生成。

### Phase 4：探索性校准与功效冻结

- 当前 pilot 只用于探索性诊断；最终算法、正式配置和分析器完成后，按正式目标配置重新执行一轮同构 pilot；
- 只用开发/pilot 种子校准非平凡性、负载和计算可行性；
- 冻结样本量、配置、代码提交和正式执行预算，并生成不可变的 `confirmatory_lock.json`；正式 runner 必须读取该锁中的样本量、种子和哈希，禁止人工另行覆盖。

### Phase 5：确认性实验

顺序为：

```text
Resource -> Exact -> Externality -> Main -> Joint
-> Scaling -> Robustness -> Trace-driven
```

使用分片和断点续跑。任何硬约束违规立即停止相关实验族并按系统调试流程定位。

### Phase 6：论文与投稿稿

实验运行期间可以并行撰写背景、模型、算法、证明和方法；摘要、结果、讨论和结论只能在 canonical-v2 后完成。

### Phase 7：最终验证

- 全量测试和属性测试；
- canonical 校验和与可复现重跑；
- 中科大模板和 Elsevier 模板编译；
- 汉字数、摘要、关键词、引用、图表和占位符检查；
- 公式—代码—实验—结论一致性审计；
- 独立代码审查和完成前证据复核。

## 17. 失败回退规则

### 17.1 IEA-CAP 效果不足

- 先检查实现、天花板、邻域、增量等价和终止原因；
- 正式种子已打开后不得重新调参；
- H1/H2 未达 MDE 时，保留机制和正确性贡献，删除普遍性能领先主张；
- 不删除不利场景或只保留显著子组。

### 17.2 DD-MCAA 效果不足或 Oracle 不一致

- Oracle 不一致属于实现/假设错误，立即停止；
- 修复前不得运行正式 Resource；
- H3 未达 MDE 但 Oracle 一致时，将 DD 定位为精确、低复杂度资源响应，不声称显著完成率优势。

### 17.3 联合端到端效应或交互为负

- 保留完整 2×2 结果；
- 分析无线速率改善是否导致 CPU 压力集中；
- H4 的主检验是端到端对角比较，不得将其误称为交互效应；无线因素×资源因素的差分中差分作为预注册次要交互量单独报告；
- H4 失败或次要交互为负时，不得只与更弱基线比较后声称联合优势或协同增益。

### 17.4 Exact 过于简单

- 只使用开发种子调整任务压力和容量；
- 正式种子必须保持未打开；
- 若无法获得非平凡实例，只保留正确性结论。

### 17.5 运行时间过高

- 允许精确缓存、安全剪枝、数据结构优化、分片和并行；
- 不允许缩减正式邻域后继续沿用完整邻域局部最优主张；
- 不满足实时条件时删除实时表述。

### 17.6 CCF-B 主张门禁失败

- 仍完成真实、规范的专业硕士论文；
- 英文稿保留为诚实的研究稿或缩小主张后再评估目标期刊；
- 不以虚假数字或过度包装维持 B 类表述。

## 18. 完成定义

### 18.1 可自动完成的研究交付

研究交付只有在以下条件全部满足时才称为完成：

1. 全量测试、属性测试、约束审计和校验和通过；
2. canonical-v2/full 包含源码、配置、环境、种子、原始/汇总数据、统计、事件日志、图表和 claim matrix，canonical-v2/light 通过闭包校验；
3. 负面结果与正面结果使用相同证据标准；
4. 中文论文正文不少于 30,000 字，官方模板组成完整并通过最终 XeLaTeX 编译；
5. 英文研究稿通过 *Computer Networks* scope gate，至少完成一项经公平适配的 closest-work 强基线，并符合当前结构要求，投稿辅助文件齐全；
6. 所有实验结果、统计量以及由它们导出的正文数值均由 canonical-v2 自动生成，不存在手填结果；
7. 所有公式、命题、定理、代码、实验、图表和结论相互一致；
8. 未解析引用、缺图、未定义符号、占位结果数字和不真实 AI 声明为零；
9. 独立代码审查、论文一致性审查和完成前验证通过。

### 18.2 必须由学生、导师或行政流程完成的事项

下列事项不能由自动化代理代替，也不能在未发生时声称完成：

1. 学生提供并确认个人、导师、专业、成果、基金和作者贡献信息；
2. 学生独立复核、重新推导并能够答辩解释核心观点、模型、算法、实验和结论；
3. 导师审阅与签字、原创性/AI 使用声明签署、纸质与电子版一致性确认；
4. 学院查重、预审、盲审、答辩、学位系统填报和最终归档；
5. 获得用户明确授权后的外部数据/代码归档、期刊系统上传及作者最终提交确认。

因此，“研究交付完成”不等于“学校授予学位”或“期刊投稿/录用完成”。

“达到 CCF-B 标准”在本项目中指具备清晰创新边界、充分理论与实验、强可比性、可复现证据和规范英文稿件，而不是对录用结果作保证。

## 19. 主要权威来源

- 中国科学技术大学研究生院，学位授予/论文模板：<https://gradschool.ustc.edu.cn/column/190>
- 中国科学技术大学研究生院，硕士学位申请材料：<https://gradschool.ustc.edu.cn/article/232>
- CCF 第七版推荐目录发布页：<https://www.ccf.org.cn/Academic_Evaluation/By_category/>
- CCF 网络与信息安全分类：<https://www.ccf.org.cn/Academic_Evaluation/CN/>
- *Computer Networks* Guide for Authors：<https://www.sciencedirect.com/journal/computer-networks/publish/guide-for-authors>
- Al-Hourani et al., urban air-to-ground path-loss model：<https://doi.org/10.1109/GLOCOM.2014.7037248>
- Al-Hourani et al., optimal low-altitude-platform altitude：<https://doi.org/10.1109/LWC.2014.2342736>
- 3GPP TR 36.777：<https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3231>
- Microsoft GeoLife：<https://www.microsoft.com/en-us/research/project/geolife-building-social-networks-using-human-location-history/downloads/>
- Guo et al., uplink multi-cell NOMA association/grouping and externality potential game：<https://doi.org/10.1109/ICC.2019.8761779>
- Xu et al., multi-AP NOMA-MEC association/resource/power with computation and interference externalities：<https://doi.org/10.1109/ACCESS.2020.2984798>
- Uwizeyimana et al., multi-UAV NOMA association/pairing：<https://doi.org/10.1109/ACCESS.2024.3417320>
- Katwe et al., multi-UAV NOMA clustering/interference：<https://doi.org/10.1109/TWC.2021.3113640>
- Chen et al., QoS-oriented NOMA multi-UAV cooperative MEC：<https://doi.org/10.1109/TWC.2025.3593884>
- Lyu et al., deadline-sensitive admission：<https://doi.org/10.1109/TCOMM.2018.2799937>

## 20. 待实施阶段补充的信息

以下信息不影响设计和计划，可在论文模板实施时一次性向用户收集：

- 学生姓名与学号；
- 导师姓名、职称及与学籍系统一致的顺序；
- 是否有实践导师；
- 院系、学科专业和研究方向的官方名称；
- 预计提交日期；
- 中图分类号和 UDC 分类号；
- 在读期间论文、专利、软件著作权、项目和获奖信息；
- 基金项目与经费声明；
- 英文稿作者顺序、通信作者和 CRediT 贡献。

这些字段在获得真实信息前使用显式、可机器检查的占位符，不允许猜测或虚构。
