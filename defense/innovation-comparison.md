# Two-Contribution Defense Comparison

## 创新点一：干扰外部性感知的跨 UAV 联合关联与配对

**工程痛点**：全频复用时，一个用户改变服务 UAV 或子载波会同时改变自身服务信道、其他 UAV 的同频干扰、SIC 角色和 MEC 负载。

**已有方法局限**：最近 UAV 关联和双边 exchange-stable 准则只观察局部参与块，无法完整评价第三方干扰外部性。

**数学思想**：使用全网词典序目标 `(按时完成数，总裕量，负跨干扰)`；主目标严格优先，后两项只做确定性平局处理。

**算法机制**：per-UAV MILP 可行初始化；SIC 修复；Replace、Position、CrossUAV 三类邻域；每个候选在实际服务接收机上重新评价；全局 first-improvement 接受并重启扫描。

**隔离消融**：严格 2×2 的 intra/cross 与 bilateral/global 矩阵。CrossUAV 独立贡献 `\InnovationOneCrossGainPP` pp；全局接受独立贡献 `\InnovationOneAcceptanceGainPP` pp，两项 95% CI 下界均为正。

**复杂度与开销**：候选数随到达任务数近似二次增长，每个候选执行完整网络评价；需要集中式 CSI 和 UAV 间调度协调。N=20 P95 为 `\FullRuntimePNinetyFiveMs` ms。

**限制**：绝对增益较小；干扰耦合变化下增益不单调；K=1 时不占优，K=5 时与 exchange-stable 持平。

## 创新点二：面向截止时间的卸载与 MEC 算力分配

**工程痛点**：均分 CPU 或固定卸载比例可能把算力分给无法按时完成的任务，降低单位时隙完成任务数。

**已有方法局限**：crossing/equal 优化单任务时延交点但不优化硬预算下的准入基数；fixed-half/equal 甚至不适配任务异质性。

**数学思想**：先由本地截止约束求 `alpha_min`，再由边缘截止约束求 `f_min`；非正传输余量直接判链路不可行。

**算法机制**：本地可行任务零 MEC 成本完成；其余可行任务按 `(f_min, task_id)` 排序，选择满足每 UAV CPU 硬预算的最长前缀。

**隔离消融**：排程搜索保持不变，只切换资源策略。相对 crossing/equal 的贡献为 `\InnovationTwoGainPP` pp，95% CI 为 [`\InnovationTwoCILowPP`, `\InnovationTwoCIHighPP`]。

**复杂度与开销**：需求推导 O(M)，排序 O(M log M)；无需迭代连续优化。交换论证保证单位价值、单预算下的最大任务数。

**限制**：不覆盖任务权重、长期队列、跨 UAV CPU 迁移或能耗约束。

| Dimension | Innovation 1 | Innovation 2 |
|---|---|---|
| Changed object | Serving UAV, subcarrier, NOMA partner/role | Offload ratio, MEC CPU admission |
| Acceptance basis | Network schedule objective | Per-UAV CPU-prefix feasibility |
| Controlled ablation | Topology × acceptance | Same schedule, three resource policies |
| Primary evidence | Pairing 2×2 and mechanism curves | Resource ablation and CPU/deadline robustness |
| Independence | Uses the same deadline allocator in all four cells | Uses the same full scheduler in all three policies |

Neither contribution is obtained by silently changing the other component.
