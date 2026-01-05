"""
Lyapunov在线优化算法
文件: simulation/algorithms/lyapunov.py

功能说明:
- 实现基于Lyapunov优化的在线卸载决策
- 虚拟队列设计：AoI队列、能耗队列、截止时间队列
- Drift-Plus-Penalty框架实现

算法来源: docs/05_算法设计_Lyapunov优化.md
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class OffloadDecision(Enum):
    """卸载决策枚举"""

    LOCAL = 0  # 本地计算
    OFFLOAD = 1  # 卸载到RSU


@dataclass
class LyapunovConfig:
    """Lyapunov优化器配置"""

    # 控制参数V (权衡队列稳定性和目标优化)
    # V越大：越关注目标优化，队列可能较长
    # V越小：越关注队列稳定，目标优化可能较差
    V: float = 1000.0  # 来自parameters.yaml

    # 虚拟队列上限
    queue_max: float = 100.0  # 来自parameters.yaml

    # 目标权重
    weight_aoi: float = 0.5  # AoI权重
    weight_energy: float = 0.1  # 能耗权重
    weight_deadline: float = 0.4  # 截止时间权重

    # 约束阈值
    aoi_target: float = 0.05  # AoI目标阈值 (s)
    energy_budget: float = 0.1  # 能耗预算 (J/任务)
    deadline_target: float = 0.95  # 截止时间满足率目标

    # 时隙参数
    T_check: float = 0.1  # 检查周期 (s)

    # 调试选项
    verbose: bool = False


@dataclass
class VirtualQueue:
    """
    虚拟队列数据结构

    虚拟队列用于将长期约束转化为队列稳定性问题
    来源: docs/05_算法设计_Lyapunov优化.md §2.2
    """

    name: str
    value: float = 0.0  # 当前队列值
    max_value: float = 100.0  # 队列上限

    # 统计信息
    sum_arrivals: float = 0.0
    sum_departures: float = 0.0
    history: List[float] = field(default_factory=list)

    def update(self, arrival: float, departure: float):
        """
        更新虚拟队列

        Args:
            arrival: 到达量 (违反约束的部分)
            departure: 离开量 (满足约束的部分)

        公式: Q(k+1) = max{Q(k) + arrival - departure, 0}
        """
        self.value = max(0.0, min(self.value + arrival - departure, self.max_value))

        self.sum_arrivals += arrival
        self.sum_departures += departure
        self.history.append(self.value)

    def get_average(self) -> float:
        """获取队列平均值"""
        if not self.history:
            return 0.0
        return np.mean(self.history)

    def reset(self):
        """重置队列"""
        self.value = 0.0
        self.sum_arrivals = 0.0
        self.sum_departures = 0.0
        self.history.clear()


@dataclass
class VehicleState:
    """车辆Lyapunov状态"""

    vehicle_id: int

    # 虚拟队列
    Q_aoi: VirtualQueue = field(default_factory=lambda: VirtualQueue("AoI"))
    Q_energy: VirtualQueue = field(default_factory=lambda: VirtualQueue("Energy"))
    Q_deadline: VirtualQueue = field(default_factory=lambda: VirtualQueue("Deadline"))

    # 当前AoI
    current_aoi: float = 0.0

    # 累计统计
    total_tasks: int = 0
    local_tasks: int = 0
    offload_tasks: int = 0

    def reset_queues(self, max_value: float = 100.0):
        """重置所有虚拟队列"""
        self.Q_aoi = VirtualQueue("AoI", max_value=max_value)
        self.Q_energy = VirtualQueue("Energy", max_value=max_value)
        self.Q_deadline = VirtualQueue("Deadline", max_value=max_value)


class LyapunovOptimizer:
    """
    Lyapunov在线优化器

    功能:
    - 基于Drift-Plus-Penalty框架做在线卸载决策
    - 通过虚拟队列将长期约束转化为逐时隙优化
    - 提供理论性能保证

    核心思想:
    - Lyapunov函数: L(k) = (1/2) Σ Q_i(k)²
    - Drift: ΔL(k) = E[L(k+1) - L(k) | Q(k)]
    - Drift-Plus-Penalty: ΔL(k) + V·f₀(k)

    算法来源: docs/05_算法设计_Lyapunov优化.md §3
    """

    def __init__(self, config: Optional[LyapunovConfig] = None):
        """
        初始化Lyapunov优化器

        Args:
            config: 配置参数
        """
        self.config = config or LyapunovConfig()

        # 车辆状态表: {vehicle_id: VehicleState}
        self._vehicle_states: Dict[int, VehicleState] = {}

        # 全局统计
        self.stats = {
            "total_decisions": 0,
            "local_decisions": 0,
            "offload_decisions": 0,
            "total_drift": 0.0,
            "total_penalty": 0.0,
        }

    def initialize_vehicle(self, vehicle_id: int):
        """
        初始化车辆状态

        Args:
            vehicle_id: 车辆ID
        """
        if vehicle_id not in self._vehicle_states:
            state = VehicleState(vehicle_id=vehicle_id)
            state.reset_queues(max_value=self.config.queue_max)
            self._vehicle_states[vehicle_id] = state

    def get_vehicle_state(self, vehicle_id: int) -> VehicleState:
        """获取车辆状态"""
        self.initialize_vehicle(vehicle_id)
        return self._vehicle_states[vehicle_id]

    def compute_local_cost(
        self, task_compute_cycles: float, local_cpu_freq: float, kappa: float = 1.0e-27
    ) -> Tuple[float, float, float]:
        """
        计算本地处理的代价

        Args:
            task_compute_cycles: 任务计算需求 (cycles)
            local_cpu_freq: 本地CPU频率 (Hz)
            kappa: 能耗系数

        Returns:
            (delay, energy, deadline_prob): 时延、能耗、截止时间满足概率

        公式:
        - T_local = C / f_local
        - E_local = κ·f_local²·C

        来源: docs/01_系统模型.md §4.1
        """
        # 计算时延
        delay = task_compute_cycles / local_cpu_freq

        # 计算能耗
        energy = kappa * (local_cpu_freq**2) * task_compute_cycles

        # 截止时间满足概率 (简化：基于时延估计)
        # 假设deadline为80ms
        deadline = 0.08
        deadline_prob = 1.0 if delay <= deadline else 0.0

        return delay, energy, deadline_prob

    def compute_offload_cost(
        self,
        task_data_size: float,
        task_compute_cycles: float,
        task_result_size: float,
        channel_rate: float,
        rsu_cpu_freq: float,
        tx_power: float = 0.2,
        idle_power: float = 0.1,
    ) -> Tuple[float, float, float]:
        """
        计算卸载处理的代价

        Args:
            task_data_size: 任务数据大小 (bits)
            task_compute_cycles: 任务计算需求 (cycles)
            task_result_size: 结果数据大小 (bits)
            channel_rate: 信道传输速率 (bps)
            rsu_cpu_freq: RSU分配的CPU频率 (Hz)
            tx_power: 发射功率 (W)
            idle_power: 空闲功率 (W)

        Returns:
            (delay, energy, deadline_prob): 时延、能耗、截止时间满足概率

        公式:
        - T_up = D / R
        - T_exe = C / f_rsu
        - T_down = D_result / R
        - E_offload = P_tx·T_up + P_idle·(T_exe + T_down)

        来源: docs/01_系统模型.md §4.2
        """
        # 避免除零
        if channel_rate <= 0:
            return float("inf"), float("inf"), 0.0
        if rsu_cpu_freq <= 0:
            return float("inf"), float("inf"), 0.0

        # 上传时延
        t_up = task_data_size / channel_rate

        # 计算时延
        t_exe = task_compute_cycles / rsu_cpu_freq

        # 下载时延
        t_down = task_result_size / channel_rate

        # 总时延
        delay = t_up + t_exe + t_down

        # 能耗
        energy = tx_power * t_up + idle_power * (t_exe + t_down)

        # 截止时间满足概率
        deadline = 0.08
        deadline_prob = 1.0 if delay <= deadline else 0.0

        return delay, energy, deadline_prob

    def compute_drift_plus_penalty(
        self,
        vehicle_id: int,
        local_cost: Tuple[float, float, float],
        offload_cost: Tuple[float, float, float],
        current_aoi: float,
    ) -> Tuple[float, float]:
        """
        计算Drift-Plus-Penalty值

        Args:
            vehicle_id: 车辆ID
            local_cost: 本地处理代价 (delay, energy, deadline_prob)
            offload_cost: 卸载处理代价 (delay, energy, deadline_prob)
            current_aoi: 当前AoI

        Returns:
            (dpp_local, dpp_offload): 本地和卸载的DPP值

        算法:
        最小化 Drift + V·Penalty

        Drift ≈ Σ Q_i(k)·(arrival_i - departure_i)
        Penalty = 加权目标函数

        来源: docs/05_算法设计_Lyapunov优化.md §3.2
        """
        state = self.get_vehicle_state(vehicle_id)
        V = self.config.V

        # 解包代价
        delay_local, energy_local, deadline_prob_local = local_cost
        delay_offload, energy_offload, deadline_prob_offload = offload_cost

        # ========== 计算本地决策的DPP ==========
        # AoI更新: 如果本地处理，新AoI = delay_local (任务完成后更新)
        aoi_arrival_local = max(0, delay_local - self.config.aoi_target)
        aoi_departure_local = self.config.aoi_target

        # 能耗约束
        energy_arrival_local = max(0, energy_local - self.config.energy_budget)
        energy_departure_local = self.config.energy_budget

        # 截止时间约束
        deadline_arrival_local = max(0, self.config.deadline_target - deadline_prob_local)
        deadline_departure_local = deadline_prob_local

        # Drift项 (本地)
        drift_local = (
            state.Q_aoi.value * (aoi_arrival_local - aoi_departure_local)
            + state.Q_energy.value * (energy_arrival_local - energy_departure_local)
            + state.Q_deadline.value * (deadline_arrival_local - deadline_departure_local)
        )

        # Penalty项 (本地): 加权目标
        penalty_local = (
            self.config.weight_aoi * delay_local
            + self.config.weight_energy * energy_local
            + self.config.weight_deadline * (1 - deadline_prob_local)
        )

        dpp_local = drift_local + V * penalty_local

        # ========== 计算卸载决策的DPP ==========
        aoi_arrival_offload = max(0, delay_offload - self.config.aoi_target)
        aoi_departure_offload = self.config.aoi_target

        energy_arrival_offload = max(0, energy_offload - self.config.energy_budget)
        energy_departure_offload = self.config.energy_budget

        deadline_arrival_offload = max(0, self.config.deadline_target - deadline_prob_offload)
        deadline_departure_offload = deadline_prob_offload

        # Drift项 (卸载)
        drift_offload = (
            state.Q_aoi.value * (aoi_arrival_offload - aoi_departure_offload)
            + state.Q_energy.value * (energy_arrival_offload - energy_departure_offload)
            + state.Q_deadline.value * (deadline_arrival_offload - deadline_departure_offload)
        )

        # Penalty项 (卸载)
        penalty_offload = (
            self.config.weight_aoi * delay_offload
            + self.config.weight_energy * energy_offload
            + self.config.weight_deadline * (1 - deadline_prob_offload)
        )

        dpp_offload = drift_offload + V * penalty_offload

        return dpp_local, dpp_offload

    def make_decision(
        self,
        vehicle_id: int,
        task_data_size: float,
        task_compute_cycles: float,
        task_result_size: float,
        local_cpu_freq: float,
        channel_rate: float,
        rsu_cpu_freq: float,
        current_aoi: float,
        kappa: float = 1.0e-27,
    ) -> Tuple[OffloadDecision, dict]:
        """
        做出卸载决策

        Args:
            vehicle_id: 车辆ID
            task_data_size: 任务数据大小 (bits)
            task_compute_cycles: 任务计算需求 (cycles)
            task_result_size: 结果数据大小 (bits)
            local_cpu_freq: 本地CPU频率 (Hz)
            channel_rate: 信道传输速率 (bps)
            rsu_cpu_freq: RSU分配的CPU频率 (Hz)
            current_aoi: 当前AoI
            kappa: 能耗系数

        Returns:
            (decision, info): 决策和详细信息

        算法:
        1. 计算本地和卸载的代价
        2. 计算两种选择的DPP值
        3. 选择DPP值较小的决策

        来源: docs/05_算法设计_Lyapunov优化.md 算法3
        """
        self.initialize_vehicle(vehicle_id)

        # 计算本地代价
        local_cost = self.compute_local_cost(task_compute_cycles, local_cpu_freq, kappa)

        # 计算卸载代价
        offload_cost = self.compute_offload_cost(
            task_data_size, task_compute_cycles, task_result_size, channel_rate, rsu_cpu_freq
        )

        # 计算DPP
        dpp_local, dpp_offload = self.compute_drift_plus_penalty(
            vehicle_id, local_cost, offload_cost, current_aoi
        )

        # 做出决策
        if dpp_local <= dpp_offload:
            decision = OffloadDecision.LOCAL
            chosen_cost = local_cost
        else:
            decision = OffloadDecision.OFFLOAD
            chosen_cost = offload_cost

        # 更新统计
        state = self.get_vehicle_state(vehicle_id)
        state.total_tasks += 1

        if decision == OffloadDecision.LOCAL:
            state.local_tasks += 1
            self.stats["local_decisions"] += 1
        else:
            state.offload_tasks += 1
            self.stats["offload_decisions"] += 1

        self.stats["total_decisions"] += 1

        # 返回详细信息
        info = {
            "decision": decision.name,
            "dpp_local": dpp_local,
            "dpp_offload": dpp_offload,
            "local_cost": {
                "delay": local_cost[0],
                "energy": local_cost[1],
                "deadline_prob": local_cost[2],
            },
            "offload_cost": {
                "delay": offload_cost[0],
                "energy": offload_cost[1],
                "deadline_prob": offload_cost[2],
            },
            "queue_states": {
                "Q_aoi": state.Q_aoi.value,
                "Q_energy": state.Q_energy.value,
                "Q_deadline": state.Q_deadline.value,
            },
        }

        if self.config.verbose:
            print(f"[Lyapunov] 车辆 {vehicle_id}: {decision.name}")
            print(f"  DPP: local={dpp_local:.4f}, offload={dpp_offload:.4f}")
            print(f"  时延: local={local_cost[0]*1000:.2f}ms, offload={offload_cost[0]*1000:.2f}ms")

        return decision, info

    def update_queues(
        self, vehicle_id: int, actual_delay: float, actual_energy: float, deadline_met: bool
    ):
        """
        更新虚拟队列（任务完成后调用）

        Args:
            vehicle_id: 车辆ID
            actual_delay: 实际时延 (s)
            actual_energy: 实际能耗 (J)
            deadline_met: 是否满足截止时间
        """
        state = self.get_vehicle_state(vehicle_id)

        # AoI队列更新
        aoi_arrival = max(0, actual_delay - self.config.aoi_target)
        aoi_departure = self.config.aoi_target
        state.Q_aoi.update(aoi_arrival, aoi_departure)

        # 能耗队列更新
        energy_arrival = max(0, actual_energy - self.config.energy_budget)
        energy_departure = self.config.energy_budget
        state.Q_energy.update(energy_arrival, energy_departure)

        # 截止时间队列更新
        deadline_arrival = 0.0 if deadline_met else self.config.deadline_target
        deadline_departure = 1.0 if deadline_met else 0.0
        state.Q_deadline.update(deadline_arrival, deadline_departure)

        # 更新当前AoI
        state.current_aoi = actual_delay

    def get_lyapunov_function(self) -> float:
        """
        计算当前Lyapunov函数值

        Returns:
            L: Lyapunov函数值 L = (1/2) Σ_i Σ_q Q_q,i²
        """
        L = 0.0
        for state in self._vehicle_states.values():
            L += 0.5 * (state.Q_aoi.value**2 + state.Q_energy.value**2 + state.Q_deadline.value**2)
        return L

    def get_queue_statistics(self, vehicle_id: int) -> dict:
        """获取车辆的队列统计"""
        state = self.get_vehicle_state(vehicle_id)
        return {
            "Q_aoi": {
                "current": state.Q_aoi.value,
                "average": state.Q_aoi.get_average(),
                "arrivals": state.Q_aoi.sum_arrivals,
                "departures": state.Q_aoi.sum_departures,
            },
            "Q_energy": {
                "current": state.Q_energy.value,
                "average": state.Q_energy.get_average(),
                "arrivals": state.Q_energy.sum_arrivals,
                "departures": state.Q_energy.sum_departures,
            },
            "Q_deadline": {
                "current": state.Q_deadline.value,
                "average": state.Q_deadline.get_average(),
                "arrivals": state.Q_deadline.sum_arrivals,
                "departures": state.Q_deadline.sum_departures,
            },
        }

    def get_global_statistics(self) -> dict:
        """获取全局统计"""
        result = self.stats.copy()
        result["num_vehicles"] = len(self._vehicle_states)
        result["lyapunov_function"] = self.get_lyapunov_function()

        if self.stats["total_decisions"] > 0:
            result["offload_ratio"] = (
                self.stats["offload_decisions"] / self.stats["total_decisions"]
            )
        else:
            result["offload_ratio"] = 0.0

        return result

    def reset(self):
        """重置所有状态"""
        self._vehicle_states.clear()
        self.stats = {
            "total_decisions": 0,
            "local_decisions": 0,
            "offload_decisions": 0,
            "total_drift": 0.0,
            "total_penalty": 0.0,
        }


# ============================================================
# 辅助函数
# ============================================================


def analyze_v_tradeoff(V_values: List[float], n_tasks: int = 100) -> List[dict]:
    """
    分析V参数的权衡效果

    Args:
        V_values: 要测试的V值列表
        n_tasks: 每个V值的任务数

    Returns:
        results: 分析结果列表
    """
    results = []

    for V in V_values:
        config = LyapunovConfig(V=V, verbose=False)
        optimizer = LyapunovOptimizer(config)

        total_delay = 0.0
        total_energy = 0.0
        deadline_met_count = 0

        for i in range(n_tasks):
            # 模拟任务参数
            task_data = np.random.uniform(3e6, 5e6)
            task_compute = np.random.uniform(8e7, 1.2e8)
            task_result = task_data * 0.1

            # 模拟信道和资源
            channel_rate = np.random.uniform(5e6, 15e6)
            rsu_freq = np.random.uniform(2e9, 5e9)

            decision, info = optimizer.make_decision(
                vehicle_id=0,
                task_data_size=task_data,
                task_compute_cycles=task_compute,
                task_result_size=task_result,
                local_cpu_freq=2e9,
                channel_rate=channel_rate,
                rsu_cpu_freq=rsu_freq,
                current_aoi=0.05,
            )

            # 记录结果
            if decision == OffloadDecision.LOCAL:
                delay = info["local_cost"]["delay"]
                energy = info["local_cost"]["energy"]
            else:
                delay = info["offload_cost"]["delay"]
                energy = info["offload_cost"]["energy"]

            total_delay += delay
            total_energy += energy
            if delay <= 0.08:
                deadline_met_count += 1

            # 更新队列
            optimizer.update_queues(0, delay, energy, delay <= 0.08)

        stats = optimizer.get_global_statistics()
        results.append(
            {
                "V": V,
                "avg_delay_ms": total_delay / n_tasks * 1000,
                "avg_energy_mJ": total_energy / n_tasks * 1000,
                "deadline_rate": deadline_met_count / n_tasks,
                "offload_ratio": stats["offload_ratio"],
                "lyapunov_function": stats["lyapunov_function"],
            }
        )

    return results


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 Lyapunov 在线优化算法")
    print("=" * 70)

    # 测试1: 基本决策
    print("\n【测试1】基本卸载决策")
    config = LyapunovConfig(V=1000, verbose=True)
    optimizer = LyapunovOptimizer(config)

    # 模拟任务
    task_data = 4e6  # 4Mbits = 0.5MB
    task_compute = 1e8  # 100 MCycles
    task_result = 4e5  # 0.4Mbits

    decision, info = optimizer.make_decision(
        vehicle_id=1,
        task_data_size=task_data,
        task_compute_cycles=task_compute,
        task_result_size=task_result,
        local_cpu_freq=2e9,
        channel_rate=10e6,
        rsu_cpu_freq=5e9,
        current_aoi=0.05,
    )

    print(f"\n决策结果: {decision.name}")
    print(f"本地时延: {info['local_cost']['delay']*1000:.2f}ms")
    print(f"卸载时延: {info['offload_cost']['delay']*1000:.2f}ms")

    # 测试2: 队列更新
    print("\n【测试2】虚拟队列更新")
    optimizer.update_queues(vehicle_id=1, actual_delay=0.06, actual_energy=0.05, deadline_met=True)

    queue_stats = optimizer.get_queue_statistics(1)
    print(f"AoI队列: {queue_stats['Q_aoi']['current']:.4f}")
    print(f"能耗队列: {queue_stats['Q_energy']['current']:.4f}")
    print(f"截止时间队列: {queue_stats['Q_deadline']['current']:.4f}")

    # 测试3: 多任务决策
    print("\n【测试3】多任务决策序列")
    optimizer_multi = LyapunovOptimizer(LyapunovConfig(V=1000, verbose=False))

    decisions_history = []
    for i in range(20):
        # 随机任务参数
        task_data = np.random.uniform(3e6, 5e6)
        task_compute = np.random.uniform(8e7, 1.2e8)
        channel_rate = np.random.uniform(5e6, 15e6)

        decision, _ = optimizer_multi.make_decision(
            vehicle_id=0,
            task_data_size=task_data,
            task_compute_cycles=task_compute,
            task_result_size=task_data * 0.1,
            local_cpu_freq=2e9,
            channel_rate=channel_rate,
            rsu_cpu_freq=5e9,
            current_aoi=0.05,
        )
        decisions_history.append(decision.name[0])  # L or O

        # 模拟更新
        optimizer_multi.update_queues(0, 0.05, 0.04, True)

    print(f"决策序列: {''.join(decisions_history)}")
    print(f"(L=Local, O=Offload)")

    stats = optimizer_multi.get_global_statistics()
    print(f"卸载比例: {stats['offload_ratio']:.2%}")

    # 测试4: Lyapunov函数
    print("\n【测试4】Lyapunov函数值")
    L = optimizer_multi.get_lyapunov_function()
    print(f"L(k) = {L:.4f}")

    # 测试5: V参数权衡分析
    print("\n【测试5】V参数权衡分析")
    V_values = [100, 500, 1000, 2000, 5000]
    results = analyze_v_tradeoff(V_values, n_tasks=50)

    print(
        f"{'V':>6} | {'Delay(ms)':>10} | {'Energy(mJ)':>10} | {'Deadline%':>10} | {'Offload%':>10}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['V']:>6} | {r['avg_delay_ms']:>10.2f} | {r['avg_energy_mJ']:>10.4f} | "
            f"{r['deadline_rate']*100:>10.1f} | {r['offload_ratio']*100:>10.1f}"
        )

    # 测试6: 全局统计
    print("\n【测试6】全局统计")
    stats = optimizer_multi.get_global_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Lyapunov测试完成!")
    print("=" * 70)


# 运行测试: python3 -m simulation.algorithms.lyapunov
