# 05 算法设计：Lyapunov在线优化

本文档详细介绍基于Lyapunov优化的在线卸载决策算法。

---

## 1. Lyapunov优化框架概述

### 1.1 为什么选择Lyapunov优化？

| 方法 | 优点 | 缺点 | 适用性 |
|------|------|------|--------|
| **穷举法** | 最优解 | 指数复杂度 | 不可行 |
| **贪心算法** | 简单高效 | 无长期性能保证 | 基线对比 |
| **DRL** | 端到端学习 | 训练困难，可解释性差 | 复杂场景 |
| **Lyapunov优化** | 有理论保证，在线决策 | 需要设计虚拟队列 | **本文采用** |

### 1.2 核心思想

**三步走策略**：

1. **建立虚拟队列**：将长期约束转化为队列稳定性问题
2. **定义Lyapunov函数**：队列状态的"能量"度量
3. **最小化漂移加惩罚**：每时隙的在线决策准则

```
┌─────────────────────────────────────────────────────────────┐
│                 Lyapunov优化框架                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  长期优化目标                虚拟队列                        │
│  ┌─────────────┐           ┌─────────────┐                 │
│  │ min 长期平均 │    →     │ 队列稳定性  │                 │
│  │ 代价        │  转化为   │ 约束        │                 │
│  └─────────────┘           └─────────────┘                 │
│         ↓                         ↓                         │
│  ┌─────────────────────────────────────────────┐           │
│  │        Lyapunov函数 L(Q)                    │           │
│  │        （队列状态的二次函数）                  │           │
│  └─────────────────────────────────────────────┘           │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────┐           │
│  │   每时隙：min [Δ(Q) + V·Cost]               │           │
│  │   漂移      即时代价                         │           │
│  └─────────────────────────────────────────────┘           │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────┐           │
│  │        在线决策算法                          │           │
│  │        （无需知道未来信息）                   │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 虚拟队列设计

### 2.1 三个虚拟队列

根据优化目标和约束，我们设计三个虚拟队列：

#### 队列1：AoI队列 $Q_i^{AoI}(k)$

**目的**：跟踪AoI与目标值的偏差

**更新规则**：
$$Q_i^{AoI}(k+1) = \max\{0, Q_i^{AoI}(k) + A_i(k) - A_{target}\}$$

其中：
- $A_i(k)$：车辆 $i$ 在时隙 $k$ 的实际AoI
- $A_{target}$：目标AoI（如200ms）

**直观理解**：
- 如果AoI长期高于目标，队列积压增加
- 队列积压大 → 算法更倾向于减少AoI

#### 队列2：能耗队列 $Q_i^{E}(k)$

**目的**：跟踪能耗与预算的偏差

**更新规则**：
$$Q_i^{E}(k+1) = \max\{0, Q_i^{E}(k) + E_i(k) - \bar{E}_{budget}\}$$

其中：
- $E_i(k)$：时隙 $k$ 的实际能耗
- $\bar{E}_{budget}$：每时隙平均能量预算

#### 队列3：截止时间违反队列 $Q_i^{D}(k)$

**目的**：跟踪截止时间违反情况

**更新规则**：
$$Q_i^{D}(k+1) = \max\{0, Q_i^{D}(k) + \mathbb{1}[T_i(k) > T_i^{max}] - \epsilon\}$$

其中：
- $\mathbb{1}[T_i(k) > T_i^{max}]$：超时指示（0或1）
- $\epsilon$：容忍的违反率（如0.05）

### 2.2 队列含义图解

```
队列积压随时间的演化示例：

Q_i^{AoI}
   ↑
   │      ╱╲
   │     ╱  ╲     ╱╲
   │    ╱    ╲   ╱  ╲
   │   ╱      ╲ ╱    ╲
   │  ╱        ╲      ╲
   │ ╱                  ╲
   └──────────────────────→ k
     AoI持续高于目标    AoI得到控制
     队列积压增加       队列积压减少

含义：队列积压 = 约束被违反的程度
     积压越大，算法越"紧张"地满足约束
```

---

## 3. Lyapunov函数与漂移

### 3.1 Lyapunov函数定义

定义Lyapunov函数为所有虚拟队列的二次和：

$$L(\mathbf{Q}(k)) = \frac{1}{2} \sum_{i=1}^{N} \left[ (Q_i^{AoI}(k))^2 + (Q_i^{E}(k))^2 + (Q_i^{D}(k))^2 \right]$$

**性质**：
- $L(\mathbf{Q}) \geq 0$
- $L(\mathbf{Q}) = 0$ 当且仅当所有队列为空
- $L(\mathbf{Q})$ 衡量队列的"总能量"

### 3.2 条件Lyapunov漂移

**定义**：

$$\Delta(\mathbf{Q}(k)) = \mathbb{E}[L(\mathbf{Q}(k+1)) - L(\mathbf{Q}(k)) | \mathbf{Q}(k)]$$

**含义**：队列状态在一个时隙内的期望变化

### 3.3 漂移上界推导

**引理（漂移上界）**：

$$\Delta(\mathbf{Q}(k)) \leq B + \sum_{i=1}^{N} \mathbb{E}\left[ Q_i^{AoI}(k)(A_i(k) - A_{target}) + Q_i^{E}(k)(E_i(k) - \bar{E}_{budget}) + Q_i^{D}(k)(\mathbb{1}[T_i(k) > T_i^{max}] - \epsilon) \right]$$

其中常数 $B$ 为：

$$B = \frac{1}{2} \sum_{i=1}^{N} \left[ (A_{max} - A_{target})^2 + (E_{max} - \bar{E}_{budget})^2 + (1 - \epsilon)^2 \right]$$

**证明**：

对于单个队列 $Q(k)$，有：
$$Q(k+1)^2 = [\max\{0, Q(k) + a(k) - b\}]^2 \leq [Q(k) + a(k) - b]^2$$
$$= Q(k)^2 + (a(k)-b)^2 + 2Q(k)(a(k)-b)$$

对所有队列求和并取条件期望，即得引理。

---

## 4. 漂移加惩罚算法

### 4.1 漂移加惩罚函数

**定义**：

$$\Delta(\mathbf{Q}(k)) + V \cdot \mathbb{E}[Cost(k) | \mathbf{Q}(k)]$$

其中：
- $V > 0$：控制参数
- $Cost(k) = \sum_i [w_1 A_i(k) + w_2 E_i(k)]$：即时代价

### 4.2 在线决策准则

**定理（最优决策准则）**：

每个时隙，选择使以下表达式最小的决策 $\mathbf{x}(k), \mathbf{a}(k)$：

$$\sum_{i=1}^{N} \left[ V \cdot Cost_i(k) + Q_i^{AoI}(k) \cdot A_i(k) + Q_i^{E}(k) \cdot E_i(k) + Q_i^{D}(k) \cdot \mathbb{1}[T_i(k) > T_i^{max}] \right]$$

### 4.3 单车决策的分解

对于每辆车 $i$，独立最小化：

$$\min_{x_i, a_i} \underbrace{V \cdot Cost_i}_{即时代价权重} + \underbrace{Q_i^{AoI} \cdot A_i}_{AoI约束压力} + \underbrace{Q_i^{E} \cdot E_i}_{能耗约束压力} + \underbrace{Q_i^{D} \cdot \mathbb{1}[T_i > T_i^{max}]}_{截止时间约束压力}$$

**直观理解**：
- 队列积压大 → 对应约束的权重增加
- 自动平衡即时性能和长期约束

---

## 5. 算法实现

### 5.1 算法伪代码

```
算法 ETPO-Lyapunov：基于Lyapunov的事件触发周期任务卸载
════════════════════════════════════════════════════════════════

输入：
  - 车辆集合 N，RSU集合 M
  - 控制参数 V
  - 目标参数：A_target, E_budget, ε

初始化：
  - 所有虚拟队列 Q_i^{AoI}(0) = Q_i^{E}(0) = Q_i^{D}(0) = 0

════════════════════════════════════════════════════════════════

FOR 每个时隙 k = 1, 2, 3, ...:

  1. 【任务生成检测】
     FOR each vehicle i:
       检查事件触发条件
       IF 触发: 生成任务 τ_i(k)

  2. 【预测】（见04文档）
     FOR each vehicle i:
       预测未来K个时隙的任务负载

  3. 【卸载决策】
     FOR each vehicle i with task:

       3.1 计算本地执行的Lyapunov代价：
           Cost_local = V·(w1·A_local + w2·E_local)
                      + Q_i^{AoI}·A_local
                      + Q_i^{E}·E_local
                      + Q_i^{D}·𝟙[T_local > T_max]

       3.2 FOR each available RSU m:
           计算卸载到RSU m的代价：
           Cost_offload[m] = V·(w1·A_offload + w2·E_offload)
                           + Q_i^{AoI}·A_offload
                           + Q_i^{E}·E_offload
                           + Q_i^{D}·𝟙[T_offload > T_max]

       3.3 选择最优决策：
           IF Cost_local ≤ min(Cost_offload):
               x_i(k) = 0 (本地)
           ELSE:
               x_i(k) = 1 (卸载)
               a_i(k) = argmin Cost_offload

  4. 【资源分配】（见06文档）
     FOR each RSU m:
       为卸载到该RSU的任务分配计算和通信资源

  5. 【任务执行】
     执行所有任务，记录实际延迟和能耗

  6. 【队列更新】
     FOR each vehicle i:
       Q_i^{AoI}(k+1) = max{0, Q_i^{AoI}(k) + A_i(k) - A_target}
       Q_i^{E}(k+1) = max{0, Q_i^{E}(k) + E_i(k) - E_budget}
       Q_i^{D}(k+1) = max{0, Q_i^{D}(k) + 𝟙[T_i > T_max] - ε}

════════════════════════════════════════════════════════════════
```

### 5.2 Python实现

```python
"""
Lyapunov在线卸载优化
文件：algorithms/lyapunov_optimizer.py
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Task:
    """任务数据类"""
    id: str
    vehicle_id: int
    data_size: float      # bits
    compute_load: float   # CPU cycles
    deadline: float       # seconds
    result_size: float    # bits


@dataclass
class Decision:
    """卸载决策"""
    vehicle_id: int
    offload: bool
    rsu_id: Optional[int]
    expected_delay: float
    expected_energy: float


@dataclass
class Result:
    """执行结果"""
    vehicle_id: int
    completed: bool
    delay: float
    energy: float
    aoi: float


class LyapunovOptimizer:
    """
    基于Lyapunov优化的在线卸载决策器
    """

    def __init__(self,
                 num_vehicles: int,
                 V: float = 100.0,
                 A_target: float = 0.2,
                 E_budget_per_slot: float = 0.1,
                 deadline_tolerance: float = 0.05,
                 w_aoi: float = 0.4,
                 w_energy: float = 0.3,
                 w_deadline: float = 0.3):
        """
        初始化Lyapunov优化器

        Args:
            num_vehicles: 车辆数量
            V: 控制参数，越大越关注即时代价
            A_target: 目标AoI (秒)
            E_budget_per_slot: 每时隙能量预算 (焦耳)
            deadline_tolerance: 截止时间违反容忍率
            w_aoi: AoI权重
            w_energy: 能耗权重
            w_deadline: 截止时间权重
        """
        self.N = num_vehicles
        self.V = V
        self.A_target = A_target
        self.E_budget = E_budget_per_slot
        self.epsilon = deadline_tolerance

        # 权重
        self.w_aoi = w_aoi
        self.w_energy = w_energy
        self.w_deadline = w_deadline

        # 初始化虚拟队列
        self.Q_aoi = np.zeros(num_vehicles)
        self.Q_energy = np.zeros(num_vehicles)
        self.Q_deadline = np.zeros(num_vehicles)

    def make_decisions(self,
                       vehicles: List,
                       rsus: List,
                       tasks: Dict[int, Task],
                       channel_states: Dict) -> List[Decision]:
        """
        为当前时隙做出所有车辆的卸载决策

        Args:
            vehicles: 车辆列表
            rsus: RSU列表
            tasks: 当前时隙的任务 {vehicle_id: Task}
            channel_states: 信道状态信息

        Returns:
            decisions: 决策列表
        """
        decisions = []

        for i, vehicle in enumerate(vehicles):
            # 检查是否有任务
            if i not in tasks:
                decisions.append(Decision(
                    vehicle_id=i,
                    offload=False,
                    rsu_id=None,
                    expected_delay=0,
                    expected_energy=0
                ))
                continue

            task = tasks[i]

            # 计算本地执行的代价
            local_metrics = self._calc_local_metrics(vehicle, task)
            local_cost = self._calc_lyapunov_cost(i, local_metrics)

            # 计算卸载到各RSU的代价
            best_rsu_id = None
            best_offload_cost = float('inf')
            best_offload_metrics = None

            available_rsus = self._get_available_rsus(vehicle, rsus)

            for rsu in available_rsus:
                offload_metrics = self._calc_offload_metrics(
                    vehicle, task, rsu, channel_states
                )
                offload_cost = self._calc_lyapunov_cost(i, offload_metrics)

                if offload_cost < best_offload_cost:
                    best_offload_cost = offload_cost
                    best_rsu_id = rsu.id
                    best_offload_metrics = offload_metrics

            # 选择最优决策
            if best_rsu_id is None or local_cost <= best_offload_cost:
                # 本地执行
                decisions.append(Decision(
                    vehicle_id=i,
                    offload=False,
                    rsu_id=None,
                    expected_delay=local_metrics['delay'],
                    expected_energy=local_metrics['energy']
                ))
            else:
                # 卸载执行
                decisions.append(Decision(
                    vehicle_id=i,
                    offload=True,
                    rsu_id=best_rsu_id,
                    expected_delay=best_offload_metrics['delay'],
                    expected_energy=best_offload_metrics['energy']
                ))

        return decisions

    def _calc_local_metrics(self, vehicle, task: Task) -> Dict[str, float]:
        """计算本地执行的各项指标"""
        # 延迟
        delay = task.compute_load / vehicle.cpu_freq

        # 能耗
        energy = vehicle.kappa * (vehicle.cpu_freq ** 2) * task.compute_load

        # AoI（简化：等于延迟）
        aoi = delay

        # 是否超时
        deadline_violated = 1.0 if delay > task.deadline else 0.0

        return {
            'delay': delay,
            'energy': energy,
            'aoi': aoi,
            'deadline_violated': deadline_violated
        }

    def _calc_offload_metrics(self, vehicle, task: Task, rsu,
                               channel_states: Dict) -> Dict[str, float]:
        """计算卸载执行的各项指标"""
        # 获取信道状态
        channel = channel_states.get((vehicle.id, rsu.id), {})
        tx_rate = channel.get('rate', 10e6)  # 默认10 Mbps

        # 上传延迟
        delay_tx = task.data_size / tx_rate

        # 执行延迟（假设获得1/10的RSU资源）
        allocated_freq = rsu.max_freq / 10
        delay_exec = task.compute_load / allocated_freq

        # 下载延迟
        delay_down = task.result_size / tx_rate

        # 总延迟
        delay = delay_tx + delay_exec + delay_down

        # 能耗
        energy = (vehicle.tx_power * delay_tx +
                  vehicle.idle_power * (delay_exec + delay_down))

        # AoI
        aoi = delay

        # 是否超时
        deadline_violated = 1.0 if delay > task.deadline else 0.0

        return {
            'delay': delay,
            'energy': energy,
            'aoi': aoi,
            'deadline_violated': deadline_violated
        }

    def _calc_lyapunov_cost(self, i: int, metrics: Dict[str, float]) -> float:
        """
        计算Lyapunov代价

        Args:
            i: 车辆索引
            metrics: 性能指标

        Returns:
            cost: Lyapunov代价
        """
        # 即时代价
        instant_cost = (self.w_aoi * metrics['aoi'] +
                        self.w_energy * metrics['energy'])

        # Lyapunov代价 = V*即时代价 + 队列*约束偏差
        cost = (self.V * instant_cost +
                self.Q_aoi[i] * metrics['aoi'] +
                self.Q_energy[i] * metrics['energy'] +
                self.Q_deadline[i] * metrics['deadline_violated'])

        return cost

    def _get_available_rsus(self, vehicle, rsus: List) -> List:
        """获取车辆可用的RSU列表"""
        available = []
        for rsu in rsus:
            distance = np.linalg.norm(vehicle.position - rsu.position)
            if distance <= rsu.coverage_radius:
                available.append(rsu)
        return available

    def update_queues(self,
                      results: List[Result],
                      current_aois: np.ndarray):
        """
        更新虚拟队列

        Args:
            results: 执行结果列表
            current_aois: 当前各车辆的AoI
        """
        for result in results:
            i = result.vehicle_id

            # 更新AoI队列
            self.Q_aoi[i] = max(0,
                self.Q_aoi[i] + result.aoi - self.A_target)

            # 更新能耗队列
            self.Q_energy[i] = max(0,
                self.Q_energy[i] + result.energy - self.E_budget)

            # 更新截止时间队列
            violated = 1.0 if not result.completed else 0.0
            self.Q_deadline[i] = max(0,
                self.Q_deadline[i] + violated - self.epsilon)

    def get_queue_status(self) -> Dict[str, np.ndarray]:
        """获取队列状态（用于监控）"""
        return {
            'Q_aoi': self.Q_aoi.copy(),
            'Q_energy': self.Q_energy.copy(),
            'Q_deadline': self.Q_deadline.copy(),
            'avg_Q_aoi': np.mean(self.Q_aoi),
            'avg_Q_energy': np.mean(self.Q_energy),
            'avg_Q_deadline': np.mean(self.Q_deadline)
        }

    def reset(self):
        """重置队列状态"""
        self.Q_aoi = np.zeros(self.N)
        self.Q_energy = np.zeros(self.N)
        self.Q_deadline = np.zeros(self.N)


class AdaptiveLyapunovOptimizer(LyapunovOptimizer):
    """
    自适应Lyapunov优化器

    根据系统状态动态调整参数V
    """

    def __init__(self, *args,
                 V_min: float = 10,
                 V_max: float = 1000,
                 adaptation_rate: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.V_min = V_min
        self.V_max = V_max
        self.adaptation_rate = adaptation_rate
        self.V_history = [self.V]

    def adapt_V(self):
        """
        自适应调整V

        策略：
        - 队列积压大 → 减小V，更关注约束
        - 队列积压小 → 增大V，更关注即时代价
        """
        # 计算平均队列积压
        avg_queue = (np.mean(self.Q_aoi) + np.mean(self.Q_energy) +
                     np.mean(self.Q_deadline)) / 3

        # 目标队列积压（根据经验设定）
        target_queue = 1.0

        # 调整V
        if avg_queue > target_queue:
            # 队列积压大，减小V
            self.V = max(self.V_min,
                        self.V * (1 - self.adaptation_rate))
        else:
            # 队列积压小，增大V
            self.V = min(self.V_max,
                        self.V * (1 + self.adaptation_rate))

        self.V_history.append(self.V)


# ============================================================
# 测试代码
# ============================================================

def test_lyapunov_optimizer():
    """测试Lyapunov优化器"""
    print("=" * 60)
    print("测试Lyapunov优化器")
    print("=" * 60)

    # 创建优化器
    optimizer = LyapunovOptimizer(
        num_vehicles=5,
        V=100,
        A_target=0.2,
        E_budget_per_slot=0.1
    )

    print("\n初始队列状态：")
    status = optimizer.get_queue_status()
    print(f"  Q_aoi: {status['Q_aoi']}")
    print(f"  Q_energy: {status['Q_energy']}")
    print(f"  Q_deadline: {status['Q_deadline']}")

    # 模拟几个时隙的队列更新
    print("\n模拟5个时隙的队列更新：")

    for slot in range(5):
        # 模拟结果
        results = [
            Result(
                vehicle_id=i,
                completed=np.random.random() > 0.1,
                delay=np.random.uniform(0.05, 0.3),
                energy=np.random.uniform(0.05, 0.2),
                aoi=np.random.uniform(0.1, 0.4)
            )
            for i in range(5)
        ]

        # 更新队列
        optimizer.update_queues(results, np.zeros(5))

        status = optimizer.get_queue_status()
        print(f"  时隙{slot}: avg_Q_aoi={status['avg_Q_aoi']:.3f}, "
              f"avg_Q_energy={status['avg_Q_energy']:.3f}, "
              f"avg_Q_deadline={status['avg_Q_deadline']:.3f}")

    print("\n" + "=" * 60)
    print("测试完成！")


if __name__ == "__main__":
    test_lyapunov_optimizer()
```

---

## 6. 理论分析

### 6.1 性能界定理

**定理1（Lyapunov性能界）**：

设最优长期平均代价为 $Cost^*$，则ETPO-Lyapunov算法的性能满足：

$$\lim_{K \to \infty} \frac{1}{K} \sum_{k=1}^{K} \mathbb{E}[Cost(k)] \leq Cost^* + \frac{B}{V}$$

其中 $B$ 是常数：

$$B = \frac{N}{2} \left[ (A_{max})^2 + (E_{max})^2 + 1 \right]$$

**证明思路**：

1. 对漂移加惩罚不等式两边求和：
   $$\sum_{k=0}^{K-1} \Delta(\mathbf{Q}(k)) + V \sum_{k=0}^{K-1} \mathbb{E}[Cost(k)] \leq KB + V \cdot K \cdot Cost^*$$

2. 利用 $L(\mathbf{Q}(K)) \geq 0$ 和 $L(\mathbf{Q}(0)) = 0$：
   $$V \sum_{k=0}^{K-1} \mathbb{E}[Cost(k)] \leq KB + V \cdot K \cdot Cost^*$$

3. 除以 $VK$ 并取极限即得。

### 6.2 队列稳定性

**定理2（队列稳定性）**：

所有虚拟队列是均值速率稳定的：

$$\lim_{K \to \infty} \frac{\mathbb{E}[Q_i(K)]}{K} = 0, \quad \forall i$$

**含义**：长期约束渐近满足。

### 6.3 V的选择与权衡

**定理3（O(1/V, V)权衡）**：

- 代价偏离最优：$O(1/V)$
- 平均队列长度：$O(V)$

**推荐**：$V = O(\sqrt{K})$ 可以在代价和延迟之间取得平衡。

| V值范围 | 代价性能 | 约束满足 | 推荐场景 |
|---------|----------|----------|----------|
| 1-10 | 较差 | 好 | 严格约束场景 |
| 10-100 | 中等 | 中等 | 一般场景 |
| 100-1000 | 好 | 较差 | 优化代价为主 |

### 6.4 算法复杂度

**单时隙决策复杂度**：

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 单车辆决策 | $O(M)$ | 遍历可用RSU |
| 所有车辆决策 | $O(N' \cdot M)$ | $N'$为有任务车辆数 |
| 队列更新 | $O(N')$ | 线性更新 |

**空间复杂度**：$O(N)$（存储3个虚拟队列）

**详细分析**：请参见 [附录E：算法复杂度分析](./07.2_附录E_算法复杂度分析.md)

---

## 7. 关键要点总结

### 7.1 本章核心贡献

1. **虚拟队列设计**：将AoI、能耗、截止时间约束转化为队列
2. **漂移加惩罚算法**：实现在线最优决策
3. **理论保证**：证明了性能界和队列稳定性

### 7.2 与论文写作的对应

本章内容对应论文的：
- **Section V-D: Lyapunov-based Online Offloading**

### 7.3 实验验证要点

1. 长期平均代价随V的变化
2. 队列稳定性验证
3. 与基线算法的性能对比
4. 约束满足情况

### 7.4 下一步

Lyapunov优化给出卸载决策后，下一章介绍**资源分配与服务迁移**。

---

**上一篇**：[04_算法设计_任务负载预测.md](./04_算法设计_任务负载预测.md)

**下一篇**：[06_算法设计_资源分配与服务迁移.md](./06_算法设计_资源分配与服务迁移.md)
