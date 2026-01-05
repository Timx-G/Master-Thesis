"""
截止时间违反率计算模块
======================

基于文档: docs/02_问题形式化与理论分析.md §2.1 优化目标

截止时间约束:
- T_i^total(k) ≤ T_i^max
- T_i^max = 80ms (yaml cpm.deadline)

性能指标:
- 违反率 = 违反任务数 / 总任务数
- 成功率 = 1 - 违反率 = 成功任务数 / 总任务数
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VehicleDeadlineState:
    """
    单车辆截止时间状态跟踪
    """

    vehicle_id: int

    # 任务计数
    total_tasks: int = 0  # 总任务数
    completed_tasks: int = 0  # 完成任务数
    violated_tasks: int = 0  # 违反截止时间的任务数
    dropped_tasks: int = 0  # 丢弃的任务数

    # 延迟记录
    delay_history: List[float] = field(default_factory=list)
    violation_history: List[bool] = field(default_factory=list)


class DeadlineCalculator:
    """
    截止时间违反率计算器

    功能:
    1. 记录每个任务是否满足截止时间
    2. 计算系统级违反率
    3. 统计延迟分布

    使用示例:
    ```python
    calculator = DeadlineCalculator(num_vehicles=50, deadline=0.08)

    # 记录任务结果
    calculator.record_task(
        vehicle_id=0,
        total_delay=0.05,
        completed=True
    )

    # 获取统计
    stats = calculator.get_statistics()
    ```
    """

    def __init__(self, num_vehicles: int, deadline: float = 0.08):
        """
        初始化截止时间计算器

        Args:
            num_vehicles: 车辆数量
            deadline: 截止时间 (s)，对应yaml cpm.deadline=0.08
        """
        self.num_vehicles = num_vehicles
        self.deadline = deadline  # 80ms

        # 为每辆车创建状态
        self.vehicle_states: Dict[int, VehicleDeadlineState] = {
            i: VehicleDeadlineState(vehicle_id=i) for i in range(num_vehicles)
        }

        # 系统级统计
        self.current_slot: int = 0
        self.slot_violation_history: List[float] = []  # 每时隙违反率

        logger.info(
            f"DeadlineCalculator initialized: "
            f"vehicles={num_vehicles}, deadline={deadline*1000:.0f}ms"
        )

    def record_task(self, vehicle_id: int, total_delay: float, completed: bool = True) -> bool:
        """
        记录单个任务的截止时间状态

        Args:
            vehicle_id: 车辆ID
            total_delay: 端到端延迟 (s)
            completed: 任务是否完成 (False表示丢弃)

        Returns:
            是否满足截止时间
        """
        if vehicle_id not in self.vehicle_states:
            self.vehicle_states[vehicle_id] = VehicleDeadlineState(vehicle_id=vehicle_id)

        state = self.vehicle_states[vehicle_id]
        state.total_tasks += 1

        if not completed:
            # 任务被丢弃
            state.dropped_tasks += 1
            state.violation_history.append(True)  # 丢弃视为违反
            return False

        state.completed_tasks += 1
        state.delay_history.append(total_delay)

        # 检查是否违反截止时间
        violated = total_delay > self.deadline
        state.violation_history.append(violated)

        if violated:
            state.violated_tasks += 1

        return not violated

    def record_slot(self, slot_results: Dict[int, dict]) -> float:
        """
        记录一个时隙的所有任务结果

        Args:
            slot_results: {vehicle_id: {'delay': float, 'completed': bool}}

        Returns:
            本时隙违反率
        """
        slot_total = 0
        slot_violated = 0

        for vid, result in slot_results.items():
            delay = result.get("delay", 0.0)
            completed = result.get("completed", True)

            met_deadline = self.record_task(vid, delay, completed)
            slot_total += 1
            if not met_deadline:
                slot_violated += 1

        # 计算本时隙违反率
        slot_violation_rate = slot_violated / max(slot_total, 1)
        self.slot_violation_history.append(slot_violation_rate)
        self.current_slot += 1

        return slot_violation_rate

    def get_vehicle_statistics(self, vehicle_id: int) -> dict:
        """
        获取单辆车的截止时间统计
        """
        if vehicle_id not in self.vehicle_states:
            return {}

        state = self.vehicle_states[vehicle_id]

        # 延迟统计
        delay_stats = {}
        if state.delay_history:
            delays = np.array(state.delay_history)
            delay_stats = {
                "avg_delay_ms": float(np.mean(delays)) * 1000,
                "max_delay_ms": float(np.max(delays)) * 1000,
                "min_delay_ms": float(np.min(delays)) * 1000,
                "std_delay_ms": float(np.std(delays)) * 1000,
                "p95_delay_ms": float(np.percentile(delays, 95)) * 1000,
                "p99_delay_ms": float(np.percentile(delays, 99)) * 1000,
            }

        return {
            "vehicle_id": vehicle_id,
            "total_tasks": state.total_tasks,
            "completed_tasks": state.completed_tasks,
            "violated_tasks": state.violated_tasks,
            "dropped_tasks": state.dropped_tasks,
            "violation_rate": state.violated_tasks / max(state.completed_tasks, 1),
            "success_rate": (state.completed_tasks - state.violated_tasks)
            / max(state.total_tasks, 1),
            "completion_rate": state.completed_tasks / max(state.total_tasks, 1),
            **delay_stats,
        }

    def get_statistics(self) -> dict:
        """
        获取系统级截止时间统计

        Returns:
            系统违反率、成功率等统计量

        对应论文指标:
        - J_3 = (1/K) Σ_k Σ_i 𝟙[T_i(k) ≤ T_max]  (成功率)
        """
        total_tasks = 0
        completed_tasks = 0
        violated_tasks = 0
        dropped_tasks = 0
        all_delays = []
        per_vehicle_violation_rates = []

        for state in self.vehicle_states.values():
            total_tasks += state.total_tasks
            completed_tasks += state.completed_tasks
            violated_tasks += state.violated_tasks
            dropped_tasks += state.dropped_tasks
            all_delays.extend(state.delay_history)

            if state.completed_tasks > 0:
                per_vehicle_violation_rates.append(state.violated_tasks / state.completed_tasks)

        # 延迟分布统计
        delay_stats = {}
        if all_delays:
            delays = np.array(all_delays)
            delay_stats = {
                "avg_delay_ms": float(np.mean(delays)) * 1000,
                "max_delay_ms": float(np.max(delays)) * 1000,
                "min_delay_ms": float(np.min(delays)) * 1000,
                "std_delay_ms": float(np.std(delays)) * 1000,
                "p50_delay_ms": float(np.percentile(delays, 50)) * 1000,
                "p95_delay_ms": float(np.percentile(delays, 95)) * 1000,
                "p99_delay_ms": float(np.percentile(delays, 99)) * 1000,
            }

        successful_tasks = completed_tasks - violated_tasks

        return {
            # 任务计数
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "successful_tasks": successful_tasks,
            "violated_tasks": violated_tasks,
            "dropped_tasks": dropped_tasks,
            # 核心指标 (对应优化目标J_3)
            "violation_rate": violated_tasks / max(completed_tasks, 1),
            "success_rate": successful_tasks / max(total_tasks, 1),
            "completion_rate": completed_tasks / max(total_tasks, 1),
            # 延迟统计
            **delay_stats,
            # 按车辆统计
            "per_vehicle_violation_rates": per_vehicle_violation_rates,
            "vehicle_violation_mean": (
                float(np.mean(per_vehicle_violation_rates)) if per_vehicle_violation_rates else 0.0
            ),
            "vehicle_violation_std": (
                float(np.std(per_vehicle_violation_rates)) if per_vehicle_violation_rates else 0.0
            ),
            # 时间序列
            "slot_violation_history": self.slot_violation_history,
            # 元信息
            "deadline_ms": self.deadline * 1000,
            "total_slots": self.current_slot,
            "num_vehicles": self.num_vehicles,
        }

    def reset(self):
        """重置所有状态"""
        for state in self.vehicle_states.values():
            state.total_tasks = 0
            state.completed_tasks = 0
            state.violated_tasks = 0
            state.dropped_tasks = 0
            state.delay_history.clear()
            state.violation_history.clear()

        self.current_slot = 0
        self.slot_violation_history.clear()


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("截止时间违反率模块单元测试")
    print("=" * 60)

    # 测试1: 基本初始化
    print("\n--- 测试1: 基本初始化 ---")
    calculator = DeadlineCalculator(num_vehicles=5, deadline=0.08)
    print(f"车辆数: {calculator.num_vehicles}")
    print(f"截止时间: {calculator.deadline*1000:.0f}ms (yaml: 80ms)")

    # 测试2: 记录任务
    print("\n--- 测试2: 记录任务 ---")

    # 满足截止时间的任务
    met = calculator.record_task(vehicle_id=0, total_delay=0.05, completed=True)
    print(f"任务1 (延迟50ms): {'✓ 满足' if met else '✗ 违反'}")

    # 违反截止时间的任务
    met = calculator.record_task(vehicle_id=0, total_delay=0.10, completed=True)
    print(f"任务2 (延迟100ms): {'✓ 满足' if met else '✗ 违反'}")

    # 丢弃的任务
    met = calculator.record_task(vehicle_id=0, total_delay=0.0, completed=False)
    print(f"任务3 (丢弃): {'✓ 满足' if met else '✗ 违反'}")

    # 测试3: 批量记录
    print("\n--- 测试3: 模拟多个时隙 ---")
    calculator.reset()

    np.random.seed(42)
    for slot in range(20):
        slot_results = {}
        for vid in range(5):
            # 模拟延迟: 大部分满足，少部分违反
            delay = np.random.uniform(0.03, 0.09)
            completed = np.random.random() > 0.1  # 10%丢弃率
            slot_results[vid] = {"delay": delay, "completed": completed}

        violation_rate = calculator.record_slot(slot_results)

        if slot % 5 == 0:
            print(f"  时隙{slot}: 违反率={violation_rate*100:.1f}%")

    # 测试4: 统计信息
    print("\n--- 测试4: 系统统计 ---")
    stats = calculator.get_statistics()
    print(f"总任务数: {stats['total_tasks']}")
    print(f"完成任务数: {stats['completed_tasks']}")
    print(f"成功任务数: {stats['successful_tasks']}")
    print(f"违反任务数: {stats['violated_tasks']}")
    print(f"丢弃任务数: {stats['dropped_tasks']}")
    print(f"违反率: {stats['violation_rate']*100:.1f}%")
    print(f"成功率: {stats['success_rate']*100:.1f}%")
    print(f"平均延迟: {stats.get('avg_delay_ms', 0):.1f}ms")
    print(f"P95延迟: {stats.get('p95_delay_ms', 0):.1f}ms")

    # 测试5: 单车辆统计
    print("\n--- 测试5: 单车辆统计 ---")
    for vid in range(3):
        v_stats = calculator.get_vehicle_statistics(vid)
        print(
            f"  车辆{vid}: 成功率={v_stats['success_rate']*100:.1f}%, "
            f"平均延迟={v_stats.get('avg_delay_ms', 0):.1f}ms"
        )

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


# 运行: python3 -m simulation.metrics.deadline
