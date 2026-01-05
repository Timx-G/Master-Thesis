"""
RSU (路边单元) 类定义
文件: simulation/core/rsu.py

参数来源: config/parameters.yaml 的 rsu 和 communication 部分
参考文档: docs/01_系统模型.md §1
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np


@dataclass
class RSU:
    """
    RSU数据类

    属性严格对应 parameters.yaml 中的 rsu 和 communication 部分
    """

    # ==================== 基础标识 ====================
    id: int  # RSU唯一ID
    position: np.ndarray  # 位置 [x, y] (m)

    # ==================== 覆盖范围 ====================
    # 参数来源: parameters.yaml -> network.rsu_coverage_radius = 400.0
    coverage_radius: float = 400.0  # 覆盖半径 (m)

    # ==================== 计算资源 ====================
    # 参数来源: parameters.yaml -> rsu.cpu_capacity = 20.0e9
    cpu_capacity: float = 20.0e9  # 总计算能力 (Hz) = 20 GHz
    cpu_available: float = field(init=False)  # 可用计算能力 (Hz)

    # ==================== 通信资源 ====================
    # 参数来源: parameters.yaml -> communication.bandwidth = 20.0e6
    bandwidth_total: float = 20.0e6  # 总带宽 (Hz) = 20 MHz
    bandwidth_available: float = field(init=False)  # 可用带宽 (Hz)

    # ==================== 回程网络 ====================
    # 参数来源: parameters.yaml -> rsu.backhaul_bandwidth = 1.0e9
    backhaul_bandwidth: float = 1.0e9  # 回程带宽 (bps) = 1 Gbps

    # ==================== 覆盖车辆 ====================
    covered_vehicle_ids: Set[int] = field(default_factory=set)  # 覆盖的车辆ID集合

    # ==================== 任务队列 ====================
    task_queue: List = field(default_factory=list)  # 待处理任务队列
    processing_tasks: Dict[int, dict] = field(
        default_factory=dict
    )  # 正在处理的任务 {task_id: info}

    # ==================== 相位管理 ====================
    phase_assignments: Dict[int, float] = field(default_factory=dict)  # {vehicle_id: phase_offset}

    # ==================== 统计信息 ====================
    total_tasks_received: int = 0  # 接收任务总数
    total_tasks_completed: int = 0  # 完成任务总数
    total_tasks_dropped: int = 0  # 丢弃任务数（资源不足）
    total_energy_consumed: float = 0.0  # 总能耗 (J)

    def __post_init__(self):
        """
        初始化后处理

        逻辑:
        1. 初始化可用资源为总容量
        """
        self.cpu_available = self.cpu_capacity
        self.bandwidth_available = self.bandwidth_total

    def is_vehicle_covered(self, vehicle_position: np.ndarray) -> bool:
        """
        判断车辆是否在覆盖范围内

        Args:
            vehicle_position: 车辆位置 [x, y]

        Returns:
            covered: 是否被覆盖

        判断条件: ||vehicle_pos - rsu_pos|| <= coverage_radius
        来源: docs/01_系统模型.md §1.2
        """
        distance = np.linalg.norm(vehicle_position - self.position)
        return distance <= self.coverage_radius

    def update_coverage(self, vehicles: List) -> Set[int]:
        """
        更新覆盖车辆列表

        Args:
            vehicles: 车辆对象列表

        Returns:
            covered_ids: 当前覆盖的车辆ID集合

        逻辑:
        1. 清空旧的覆盖列表
        2. 遍历所有车辆，判断是否在覆盖范围内
        3. 返回新的覆盖列表
        """
        self.covered_vehicle_ids.clear()

        for vehicle in vehicles:
            if self.is_vehicle_covered(vehicle.position):
                self.covered_vehicle_ids.add(vehicle.id)

        return self.covered_vehicle_ids.copy()

    def allocate_cpu(self, required_freq: float) -> bool:
        """
        分配CPU资源

        Args:
            required_freq: 需求频率 (Hz)

        Returns:
            success: 是否分配成功

        约束: Σ f_i^edge <= F_max
        来源: docs/01_系统模型.md §4.3
        """
        if self.cpu_available >= required_freq:
            self.cpu_available -= required_freq
            return True
        return False

    def release_cpu(self, freq: float):
        """
        释放CPU资源

        Args:
            freq: 释放的频率 (Hz)

        逻辑:
        1. 增加可用CPU
        2. 限制不超过总容量（防止错误累积）
        """
        self.cpu_available = min(self.cpu_available + freq, self.cpu_capacity)

    def allocate_bandwidth(self, required_bw: float) -> bool:
        """
        分配带宽资源

        Args:
            required_bw: 需求带宽 (Hz)

        Returns:
            success: 是否分配成功
        """
        if self.bandwidth_available >= required_bw:
            self.bandwidth_available -= required_bw
            return True
        return False

    def release_bandwidth(self, bw: float):
        """
        释放带宽资源

        Args:
            bw: 释放的带宽 (Hz)
        """
        self.bandwidth_available = min(self.bandwidth_available + bw, self.bandwidth_total)

    def get_cpu_utilization(self) -> float:
        """
        获取CPU利用率

        Returns:
            utilization: 利用率 [0, 1]
        """
        return 1.0 - (self.cpu_available / self.cpu_capacity)

    def get_bandwidth_utilization(self) -> float:
        """
        获取带宽利用率

        Returns:
            utilization: 利用率 [0, 1]
        """
        return 1.0 - (self.bandwidth_available / self.bandwidth_total)

    def get_queue_length(self) -> int:
        """
        获取任务队列长度

        Returns:
            length: 队列中的任务数
        """
        return len(self.task_queue)

    def get_num_processing_tasks(self) -> int:
        """
        获取正在处理的任务数

        Returns:
            num: 正在处理的任务数
        """
        return len(self.processing_tasks)

    def get_statistics(self) -> dict:
        """
        获取RSU统计信息

        Returns:
            stats: 统计字典
        """
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "coverage_radius": self.coverage_radius,
            "num_covered_vehicles": len(self.covered_vehicle_ids),
            "cpu_utilization": self.get_cpu_utilization(),
            "bandwidth_utilization": self.get_bandwidth_utilization(),
            "queue_length": self.get_queue_length(),
            "num_processing": self.get_num_processing_tasks(),
            "total_received": self.total_tasks_received,
            "total_completed": self.total_tasks_completed,
            "total_dropped": self.total_tasks_dropped,
            "success_rate": (
                self.total_tasks_completed / self.total_tasks_received
                if self.total_tasks_received > 0
                else 0.0
            ),
        }

    def reset_resources(self):
        """
        重置资源到初始状态（用于每个时隙开始）
        """
        self.cpu_available = self.cpu_capacity
        self.bandwidth_available = self.bandwidth_total

    def __repr__(self) -> str:
        cpu_util = self.get_cpu_utilization() * 100
        return (
            f"RSU(id={self.id}, pos=({self.position[0]:.0f}, {self.position[1]:.0f}), "
            f"coverage={len(self.covered_vehicle_ids)} vehicles, "
            f"CPU={cpu_util:.1f}%)"
        )


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    from simulation.core.vehicle import Vehicle

    print("=" * 70)
    print("测试 RSU 类")
    print("=" * 70)

    # 从parameters.yaml加载参数（模拟）
    coverage_radius = 400.0
    cpu_capacity = 20.0e9
    bandwidth = 20.0e6

    # 创建RSU
    rsu = RSU(
        id=1,
        position=np.array([500.0, 0.0]),
        coverage_radius=coverage_radius,
        cpu_capacity=cpu_capacity,
        bandwidth_total=bandwidth,
    )

    print(f"\n【1】初始状态:")
    print(f"   {rsu}")
    stats = rsu.get_statistics()
    print(f"   CPU容量: {stats['cpu_utilization']*100:.1f}%使用")
    print(f"   带宽容量: {stats['bandwidth_utilization']*100:.1f}%使用")

    # 创建测试车辆
    print(f"\n【2】创建测试车辆并更新覆盖:")
    vehicles = []
    for i in range(10):
        pos_x = 500 + (i - 5) * 50  # RSU周围分布
        v = Vehicle(id=i, position=np.array([pos_x, 0.0]), speed=25.0)
        vehicles.append(v)
        distance = np.linalg.norm(v.position - rsu.position)
        covered = "✓" if distance <= coverage_radius else "✗"
        print(f"   车辆{i}: pos={pos_x:.0f}m, 距离={distance:.0f}m {covered}")

    covered = rsu.update_coverage(vehicles)
    print(f"   覆盖车辆ID: {sorted(covered)}")

    # 测试资源分配
    print(f"\n【3】测试资源分配:")
    cpu_req = 5.0e9  # 5 GHz
    bw_req = 5.0e6  # 5 MHz

    success_cpu = rsu.allocate_cpu(cpu_req)
    success_bw = rsu.allocate_bandwidth(bw_req)
    print(f"   分配5GHz CPU: {'成功' if success_cpu else '失败'}")
    print(f"   分配5MHz 带宽: {'成功' if success_bw else '失败'}")
    print(f"   CPU利用率: {rsu.get_cpu_utilization()*100:.1f}%")
    print(f"   带宽利用率: {rsu.get_bandwidth_utilization()*100:.1f}%")

    # 释放资源
    print(f"\n【4】释放资源:")
    rsu.release_cpu(cpu_req)
    rsu.release_bandwidth(bw_req)
    print(f"   释放后CPU利用率: {rsu.get_cpu_utilization()*100:.1f}%")
    print(f"   释放后带宽利用率: {rsu.get_bandwidth_utilization()*100:.1f}%")

    # 测试统计信息
    print(f"\n【5】统计信息:")
    stats = rsu.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
