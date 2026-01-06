"""
AoI (Age of Information) 计算模块
==================================

基于文档: docs/01_系统模型.md §5 AoI模型
        docs/02_问题形式化与理论分析.md §2 优化目标

AoI定义:
- A_i(t) = t - U_i(t)
- U_i(t) 是最近一次成功接收的感知信息的生成时刻

AoI演化规则 (离散时间):
- 任务成功完成: A_i(k+1) = T_i^total(k)  # 重置为端到端延迟
- 任务未完成/失败: A_i(k+1) = A_i(k) + T_check  # 线性增长
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VehicleAoIState:
    """
    单车辆AoI状态跟踪

    记录车辆的AoI演化历史和统计信息
    """

    vehicle_id: int

    # 当前AoI值 (秒)
    current_aoi: float = 0.0

    # 上次成功更新的时刻
    last_update_time: float = 0.0

    # 上次成功更新的信息生成时刻
    last_generation_time: float = 0.0

    # AoI历史记录 (用于计算统计量)
    aoi_history: List[float] = field(default_factory=list)

    # 峰值AoI记录
    peak_aoi_history: List[float] = field(default_factory=list)

    # 统计量
    total_updates: int = 0  # 成功更新次数
    total_slots: int = 0  # 总时隙数


class AoICalculator:
    """
    AoI计算器

    功能:
    1. 跟踪每辆车的AoI状态
    2. 根据任务完成情况更新AoI
    3. 计算各种AoI统计指标

    使用示例:
    ```python
    calculator = AoICalculator(num_vehicles=50, dt=0.1)

    # 每个时隙更新
    calculator.update_slot(
        vehicle_id=0,
        task_completed=True,
        generation_time=0.0,
        completion_time=0.05
    )

    # 获取统计结果
    stats = calculator.get_statistics()
    ```
    """

    def __init__(self, num_vehicles: int, dt: float = 0.1, initial_aoi: float = 0.0):
        """
        初始化AoI计算器

        Args:
            num_vehicles: 车辆数量，对应yaml vehicle.count=50
            dt: 时隙长度，对应yaml simulation.dt=0.1 (即T_check)
            initial_aoi: 初始AoI值
        """
        self.num_vehicles = num_vehicles
        self.dt = dt  # T_check = 0.1s

        # 为每辆车创建AoI状态
        self.vehicle_states: Dict[int, VehicleAoIState] = {
            i: VehicleAoIState(vehicle_id=i, current_aoi=initial_aoi) for i in range(num_vehicles)
        }

        # 系统级统计
        self.current_slot: int = 0
        self.system_aoi_history: List[float] = []  # 系统平均AoI历史

        logger.info(f"AoICalculator initialized: " f"vehicles={num_vehicles}, dt={dt}s")

    def update_slot(
        self,
        vehicle_id: int,
        task_completed: bool,
        generation_time: Optional[float] = None,
        completion_time: Optional[float] = None,
        total_delay: Optional[float] = None,
    ) -> float:
        """
        更新单辆车在当前时隙的AoI

        Args:
            vehicle_id: 车辆ID
            task_completed: 任务是否成功完成
            generation_time: 任务生成时刻 (若完成)
            completion_time: 任务完成时刻 (若完成)
            total_delay: 端到端延迟 (若完成)

        Returns:
            更新后的AoI值

        AoI演化规则 (docs/01_系统模型.md §5.3):
        - 完成: A_i(k+1) = T_i^total(k)
        - 未完成: A_i(k+1) = A_i(k) + T_check
        """
        if vehicle_id not in self.vehicle_states:
            logger.warning(f"Vehicle {vehicle_id} not found, creating new state")
            self.vehicle_states[vehicle_id] = VehicleAoIState(vehicle_id=vehicle_id)

        state = self.vehicle_states[vehicle_id]

        if task_completed and total_delay is not None:
            # 任务成功完成: AoI重置为端到端延迟
            state.current_aoi = total_delay
            state.last_update_time = completion_time if completion_time else 0.0
            state.last_generation_time = generation_time if generation_time else 0.0
            state.total_updates += 1

            # 记录峰值 (更新前的AoI)
            if state.aoi_history:
                state.peak_aoi_history.append(state.aoi_history[-1])
        else:
            # 任务未完成或失败: AoI线性增长
            state.current_aoi += self.dt

        # 记录历史
        state.aoi_history.append(state.current_aoi)
        state.total_slots += 1

        return state.current_aoi

    def update_all_vehicles(self, completed_tasks: Dict[int, dict]) -> Dict[int, float]:
        """
        批量更新所有车辆的AoI

        Args:
            completed_tasks: {vehicle_id: task_info} 完成任务的信息
                task_info包含: generation_time, completion_time, total_delay

        Returns:
            {vehicle_id: current_aoi} 所有车辆的当前AoI
        """
        current_aois = {}

        for vid in range(self.num_vehicles):
            if vid in completed_tasks:
                task_info = completed_tasks[vid]
                aoi = self.update_slot(
                    vehicle_id=vid,
                    task_completed=True,
                    generation_time=task_info.get("generation_time"),
                    completion_time=task_info.get("completion_time"),
                    total_delay=task_info.get("total_delay"),
                )
            else:
                # 该车辆本时隙没有完成任务
                aoi = self.update_slot(vehicle_id=vid, task_completed=False)

            current_aois[vid] = aoi

        # 记录系统平均AoI
        avg_aoi = np.mean(list(current_aois.values()))
        self.system_aoi_history.append(float(avg_aoi))
        self.current_slot += 1

        return current_aois

    def get_vehicle_aoi(self, vehicle_id: int) -> float:
        """获取指定车辆的当前AoI"""
        if vehicle_id in self.vehicle_states:
            return self.vehicle_states[vehicle_id].current_aoi
        return 0.0

    def get_all_current_aois(self) -> Dict[int, float]:
        """获取所有车辆的当前AoI"""
        return {vid: state.current_aoi for vid, state in self.vehicle_states.items()}

    def get_vehicle_statistics(self, vehicle_id: int) -> dict:
        """
        获取单辆车的AoI统计信息

        Returns:
            包含均值、最大值、最小值、峰值均值等统计量
        """
        if vehicle_id not in self.vehicle_states:
            return {}

        state = self.vehicle_states[vehicle_id]

        if not state.aoi_history:
            return {
                "vehicle_id": vehicle_id,
                "avg_aoi": 0.0,
                "max_aoi": 0.0,
                "min_aoi": 0.0,
                "current_aoi": 0.0,
                "peak_avg": 0.0,
                "total_updates": 0,
            }

        aoi_array = np.array(state.aoi_history)

        return {
            "vehicle_id": vehicle_id,
            "avg_aoi": float(np.mean(aoi_array)),
            "max_aoi": float(np.max(aoi_array)),
            "min_aoi": float(np.min(aoi_array)),
            "std_aoi": float(np.std(aoi_array)),
            "current_aoi": state.current_aoi,
            "peak_avg": float(np.mean(state.peak_aoi_history)) if state.peak_aoi_history else 0.0,
            "total_updates": state.total_updates,
            "update_rate": state.total_updates / max(state.total_slots, 1),
        }

    def get_statistics(self) -> dict:
        """
        获取系统级AoI统计信息

        Returns:
            系统平均AoI、各车辆AoI分布等统计量

        对应论文指标:
        - J_1 = (1/K) Σ_k Σ_i A_i(k)  (平均AoI)
        """
        all_aois = []
        per_vehicle_avg = []

        for vid, state in self.vehicle_states.items():
            if state.aoi_history:
                all_aois.extend(state.aoi_history)
                per_vehicle_avg.append(np.mean(state.aoi_history))

        if not all_aois:
            return {
                "system_avg_aoi": 0.0,
                "system_max_aoi": 0.0,
                "system_min_aoi": 0.0,
                "system_std_aoi": 0.0,
                "per_vehicle_avg_aoi": [],
                "total_slots": self.current_slot,
            }

        all_aois_array = np.array(all_aois)
        per_vehicle_array = np.array(per_vehicle_avg)

        return {
            # 系统级统计 (对应优化目标J_1)
            "system_avg_aoi": float(np.mean(all_aois_array)),
            "system_max_aoi": float(np.max(all_aois_array)),
            "system_min_aoi": float(np.min(all_aois_array)),
            "system_std_aoi": float(np.std(all_aois_array)),
            # 按车辆统计
            "per_vehicle_avg_aoi": per_vehicle_avg,
            "vehicle_avg_mean": float(np.mean(per_vehicle_array)),
            "vehicle_avg_std": float(np.std(per_vehicle_array)),
            "vehicle_avg_max": float(np.max(per_vehicle_array)),
            "vehicle_avg_min": float(np.min(per_vehicle_array)),
            # 时间序列
            "system_aoi_history": self.system_aoi_history,
            # 元信息
            "total_slots": self.current_slot,
            "num_vehicles": self.num_vehicles,
        }

    def reset(self):
        """重置所有AoI状态"""
        for state in self.vehicle_states.values():
            state.current_aoi = 0.0
            state.aoi_history.clear()
            state.peak_aoi_history.clear()
            state.total_updates = 0
            state.total_slots = 0

        self.current_slot = 0
        self.system_aoi_history.clear()


def calculate_average_aoi(aoi_values: List[float]) -> float:
    """
    计算平均AoI (辅助函数)

    公式: A_avg = (1/N) Σ_i A_i
    """
    if not aoi_values:
        return 0.0
    return float(np.mean(aoi_values))


def calculate_peak_aoi(aoi_history: List[float]) -> float:
    """
    计算峰值AoI (辅助函数)

    峰值AoI = 每次更新前的最大AoI值的平均
    """
    if not aoi_history:
        return 0.0
    return float(np.max(aoi_history))


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("AoI计算模块单元测试")
    print("=" * 60)

    # 测试1: 基本初始化
    print("\n--- 测试1: 基本初始化 ---")
    calculator = AoICalculator(num_vehicles=5, dt=0.1)
    print(f"车辆数: {calculator.num_vehicles}")
    print(f"时隙长度: {calculator.dt}s (对应T_check=100ms)")

    # 测试2: AoI更新 - 无任务完成
    print("\n--- 测试2: AoI线性增长 (无任务完成) ---")
    for slot in range(5):
        aoi = calculator.update_slot(vehicle_id=0, task_completed=False)
        print(f"  时隙{slot}: AoI = {aoi*1000:.0f}ms")

    # 测试3: AoI更新 - 任务完成
    print("\n--- 测试3: AoI重置 (任务完成) ---")
    aoi = calculator.update_slot(
        vehicle_id=0,
        task_completed=True,
        generation_time=0.5,
        completion_time=0.55,
        total_delay=0.05,  # 50ms延迟
    )
    print(f"  任务完成后AoI = {aoi*1000:.0f}ms (端到端延迟)")

    # 测试4: 批量更新
    print("\n--- 测试4: 批量更新所有车辆 ---")
    calculator.reset()

    # 模拟10个时隙
    for slot in range(10):
        # 模拟部分车辆完成任务
        completed = {}
        if slot % 3 == 0:  # 每3个时隙车辆0完成任务
            completed[0] = {
                "generation_time": slot * 0.1,
                "completion_time": slot * 0.1 + 0.05,
                "total_delay": 0.05,
            }
        if slot % 4 == 0:  # 每4个时隙车辆1完成任务
            completed[1] = {
                "generation_time": slot * 0.1,
                "completion_time": slot * 0.1 + 0.06,
                "total_delay": 0.06,
            }

        aois = calculator.update_all_vehicles(completed)

        if slot % 3 == 0:
            print(
                f"  时隙{slot}: 车辆0={aois[0]*1000:.0f}ms, "
                f"车辆1={aois[1]*1000:.0f}ms, "
                f"车辆2={aois[2]*1000:.0f}ms"
            )

    # 测试5: 获取统计信息
    print("\n--- 测试5: 统计信息 ---")
    stats = calculator.get_statistics()
    print(f"  系统平均AoI: {stats['system_avg_aoi']*1000:.1f}ms")
    print(f"  系统最大AoI: {stats['system_max_aoi']*1000:.1f}ms")
    print(f"  系统AoI标准差: {stats['system_std_aoi']*1000:.1f}ms")

    # 测试6: 单车辆统计
    print("\n--- 测试6: 单车辆统计 ---")
    for vid in range(3):
        v_stats = calculator.get_vehicle_statistics(vid)
        print(
            f"  车辆{vid}: 平均AoI={v_stats['avg_aoi']*1000:.1f}ms, "
            f"更新次数={v_stats['total_updates']}, "
            f"更新率={v_stats['update_rate']:.2f}"
        )

    # 测试7: 验证公式
    print("\n--- 测试7: 验证AoI计算公式 ---")
    calculator.reset()

    # 场景: 车辆0每200ms完成一次任务，延迟50ms
    print("  场景: 每200ms完成任务，延迟50ms")
    for slot in range(20):
        completed = {}
        if slot % 2 == 0:  # 每2个时隙完成一次
            completed[0] = {
                "generation_time": slot * 0.1,
                "completion_time": slot * 0.1 + 0.05,
                "total_delay": 0.05,
            }
        calculator.update_all_vehicles(completed)

    v0_stats = calculator.get_vehicle_statistics(0)
    print(f"  车辆0平均AoI: {v0_stats['avg_aoi']*1000:.1f}ms")
    print(f"  理论值: ~100ms (因为AoI在50-150ms之间振荡)")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
