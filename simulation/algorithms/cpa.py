"""
集中式相位分配算法 (Centralized Phase Allocation, CPA)
文件: simulation/algorithms/cpa.py

功能说明:
- 实现基于RSU覆盖的集中式相位分配
- 均匀分配相位偏移，避免任务触发冲突
- 支持动态车辆加入/离开

算法来源: docs/03_算法设计_相位对齐调度.md §2
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CPAConfig:
    """CPA算法配置参数"""

    # 时隙参数
    T_check: float = 0.1  # 检查周期 (s)，来自parameters.yaml

    # 相位分配策略
    strategy: str = "uniform"  # 分配策略: "uniform", "random"

    # 调试选项
    verbose: bool = False


class CentralizedPhaseAllocator:
    """
    集中式相位分配器

    功能:
    - RSU为其覆盖范围内的车辆集中分配相位偏移
    - 均匀分配策略: φ_i = (i-1) * T_check / n
    - 支持车辆动态加入/离开时的相位重分配

    算法来源: docs/03_算法设计_相位对齐调度.md §2.2
    """

    def __init__(self, config: Optional[CPAConfig] = None):
        """
        初始化CPA分配器

        Args:
            config: CPA配置参数
        """
        self.config = config or CPAConfig()

        # 每个RSU的相位分配表: {rsu_id: {vehicle_id: phase_offset}}
        self._phase_tables: Dict[int, Dict[int, float]] = {}

        # 统计信息
        self.stats = {
            "total_allocations": 0,
            "total_reallocations": 0,
            "total_releases": 0,
        }

    def initialize_rsu(self, rsu_id: int):
        """
        初始化RSU的相位分配表

        Args:
            rsu_id: RSU标识符
        """
        if rsu_id not in self._phase_tables:
            self._phase_tables[rsu_id] = {}

    def allocate_phases(self, rsu_id: int, vehicle_ids: List[int]) -> Dict[int, float]:
        """
        为RSU覆盖的车辆分配相位偏移

        Args:
            rsu_id: RSU标识符
            vehicle_ids: 覆盖范围内的车辆ID列表

        Returns:
            phase_assignments: {vehicle_id: phase_offset} 相位分配结果

        算法:
        - 均匀分配: φ_i = (i-1) * T_check / n
        - 确保相位偏移在 [0, T_check) 范围内

        来源: docs/03_算法设计_相位对齐调度.md 公式(3)
        """
        self.initialize_rsu(rsu_id)

        n = len(vehicle_ids)
        if n == 0:
            return {}

        T_check = self.config.T_check
        phase_assignments = {}

        if self.config.strategy == "uniform":
            # 均匀分配策略
            # φ_i = (i-1) * T_check / n, i = 1, 2, ..., n
            for i, vid in enumerate(sorted(vehicle_ids)):
                phase = (i * T_check) / n
                phase_assignments[vid] = phase

        elif self.config.strategy == "random":
            # 随机分配策略（用于对比实验）
            for vid in vehicle_ids:
                phase = np.random.uniform(0, T_check)
                phase_assignments[vid] = phase

        # 更新内部相位表
        self._phase_tables[rsu_id] = phase_assignments.copy()

        # 更新统计
        self.stats["total_allocations"] += n

        if self.config.verbose:
            print(f"[CPA] RSU {rsu_id}: 分配 {n} 辆车的相位")
            for vid, phase in phase_assignments.items():
                print(f"      车辆 {vid}: φ = {phase*1000:.2f} ms")

        return phase_assignments

    def update_phases_on_vehicle_join(
        self, rsu_id: int, new_vehicle_id: int, existing_vehicle_ids: List[int]
    ) -> Dict[int, float]:
        """
        车辆加入时更新相位分配

        Args:
            rsu_id: RSU标识符
            new_vehicle_id: 新加入的车辆ID
            existing_vehicle_ids: 现有车辆ID列表

        Returns:
            updated_phases: 更新后的相位分配

        策略: 重新均匀分配所有车辆的相位

        注意: 这会导致所有车辆的相位变化，实际系统中可能需要渐进式调整
        """
        all_vehicle_ids = list(set(existing_vehicle_ids) | {new_vehicle_id})

        self.stats["total_reallocations"] += 1

        return self.allocate_phases(rsu_id, all_vehicle_ids)

    def update_phases_on_vehicle_leave(
        self, rsu_id: int, leaving_vehicle_id: int, remaining_vehicle_ids: List[int]
    ) -> Dict[int, float]:
        """
        车辆离开时更新相位分配

        Args:
            rsu_id: RSU标识符
            leaving_vehicle_id: 离开的车辆ID
            remaining_vehicle_ids: 剩余车辆ID列表

        Returns:
            updated_phases: 更新后的相位分配

        策略: 重新均匀分配剩余车辆的相位
        """
        # 确保移除离开的车辆
        remaining = [vid for vid in remaining_vehicle_ids if vid != leaving_vehicle_id]

        self.stats["total_releases"] += 1
        self.stats["total_reallocations"] += 1

        return self.allocate_phases(rsu_id, remaining)

    def get_vehicle_phase(self, rsu_id: int, vehicle_id: int) -> Optional[float]:
        """
        获取车辆的相位偏移

        Args:
            rsu_id: RSU标识符
            vehicle_id: 车辆标识符

        Returns:
            phase: 相位偏移，若未分配则返回None
        """
        if rsu_id not in self._phase_tables:
            return None
        return self._phase_tables[rsu_id].get(vehicle_id)

    def get_rsu_phases(self, rsu_id: int) -> Dict[int, float]:
        """
        获取RSU的所有相位分配

        Args:
            rsu_id: RSU标识符

        Returns:
            phases: {vehicle_id: phase_offset}
        """
        return self._phase_tables.get(rsu_id, {}).copy()

    def should_trigger(self, vehicle_id: int, rsu_id: int, current_time: float) -> bool:
        """
        判断车辆是否应该在当前时刻触发CPM任务

        Args:
            vehicle_id: 车辆标识符
            rsu_id: RSU标识符
            current_time: 当前时间 (s)

        Returns:
            should_trigger: 是否应该触发

        逻辑:
        - 获取车辆相位 φ_i
        - 计算时隙内的时间偏移: t_offset = current_time mod T_check
        - 如果 |t_offset - φ_i| < ε (容差)，则触发

        来源: docs/03_算法设计_相位对齐调度.md §2.2
        """
        phase = self.get_vehicle_phase(rsu_id, vehicle_id)
        if phase is None:
            return False

        T_check = self.config.T_check

        # 计算当前时隙内的偏移
        t_offset = current_time % T_check

        # 容差: 允许微小的时间误差
        epsilon = T_check * 0.01  # 1%的容差

        # 检查是否在相位时刻附近
        diff = abs(t_offset - phase)

        # 处理周期边界情况
        diff = min(diff, T_check - diff)

        return diff < epsilon

    def calculate_collision_probability(
        self, rsu_id: int, collision_window: float = 0.005
    ) -> float:
        """
        计算相位冲突概率

        Args:
            rsu_id: RSU标识符
            collision_window: 冲突窗口 (s)，默认5ms

        Returns:
            probability: 冲突概率 [0, 1]

        定义: 如果两辆车的相位差 < collision_window，则认为可能冲突
        """
        phases = self.get_rsu_phases(rsu_id)
        if len(phases) < 2:
            return 0.0

        phase_values = sorted(phases.values())
        n = len(phase_values)

        collision_count = 0
        total_pairs = n * (n - 1) // 2

        for i in range(n):
            for j in range(i + 1, n):
                diff = phase_values[j] - phase_values[i]
                # 考虑周期边界
                diff = min(diff, self.config.T_check - diff)

                if diff < collision_window:
                    collision_count += 1

        return collision_count / total_pairs if total_pairs > 0 else 0.0

    def get_statistics(self) -> dict:
        """
        获取CPA统计信息

        Returns:
            stats: 统计字典
        """
        result = self.stats.copy()
        result["num_rsus"] = len(self._phase_tables)
        result["total_vehicles"] = sum(len(phases) for phases in self._phase_tables.values())
        return result

    def reset(self):
        """重置所有状态"""
        self._phase_tables.clear()
        self.stats = {
            "total_allocations": 0,
            "total_reallocations": 0,
            "total_releases": 0,
        }


# ============================================================
# 辅助函数
# ============================================================


def compute_optimal_phase_spacing(n_vehicles: int, T_check: float) -> float:
    """
    计算最优相位间隔

    Args:
        n_vehicles: 车辆数量
        T_check: 检查周期 (s)

    Returns:
        spacing: 最优相位间隔 (s)

    公式: Δφ = T_check / n
    """
    if n_vehicles <= 0:
        return T_check
    return T_check / n_vehicles


def verify_phase_allocation(
    phases: Dict[int, float], T_check: float, min_spacing: float = 0.001
) -> Tuple[bool, List[str]]:
    """
    验证相位分配的有效性

    Args:
        phases: {vehicle_id: phase_offset}
        T_check: 检查周期 (s)
        min_spacing: 最小相位间隔 (s)

    Returns:
        valid: 是否有效
        issues: 问题列表
    """
    issues = []

    # 检查相位范围
    for vid, phase in phases.items():
        if phase < 0 or phase >= T_check:
            issues.append(f"车辆 {vid} 的相位 {phase} 超出范围 [0, {T_check})")

    # 检查相位间隔
    phase_list = sorted(phases.values())
    for i in range(len(phase_list) - 1):
        spacing = phase_list[i + 1] - phase_list[i]
        if spacing < min_spacing:
            issues.append(f"相位间隔过小: {spacing*1000:.3f}ms < {min_spacing*1000:.3f}ms")

    return len(issues) == 0, issues


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 CPA (集中式相位分配) 算法")
    print("=" * 70)

    # 测试1: 基本相位分配
    print("\n【测试1】基本相位分配")
    config = CPAConfig(T_check=0.1, strategy="uniform", verbose=True)
    cpa = CentralizedPhaseAllocator(config)

    # 为RSU 0分配5辆车的相位
    vehicle_ids = [1, 2, 3, 4, 5]
    phases = cpa.allocate_phases(rsu_id=0, vehicle_ids=vehicle_ids)

    print(f"\n分配结果:")
    for vid, phase in sorted(phases.items()):
        print(f"  车辆 {vid}: φ = {phase*1000:.2f} ms")

    # 验证相位间隔
    expected_spacing = 0.1 / 5  # 20ms
    print(f"\n预期间隔: {expected_spacing*1000:.2f} ms")

    # 测试2: 车辆加入
    print("\n【测试2】车辆加入")
    new_phases = cpa.update_phases_on_vehicle_join(
        rsu_id=0, new_vehicle_id=6, existing_vehicle_ids=vehicle_ids
    )
    print(f"加入车辆6后，新间隔: {0.1/6*1000:.2f} ms")
    for vid, phase in sorted(new_phases.items()):
        print(f"  车辆 {vid}: φ = {phase*1000:.2f} ms")

    # 测试3: 车辆离开
    print("\n【测试3】车辆离开")
    remaining_phases = cpa.update_phases_on_vehicle_leave(
        rsu_id=0, leaving_vehicle_id=3, remaining_vehicle_ids=[1, 2, 3, 4, 5, 6]
    )
    print(f"车辆3离开后，新间隔: {0.1/5*1000:.2f} ms")
    for vid, phase in sorted(remaining_phases.items()):
        print(f"  车辆 {vid}: φ = {phase*1000:.2f} ms")

    # 测试4: 触发判断
    print("\n【测试4】触发判断")
    test_times = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    for t in test_times:
        triggered = []
        for vid in [1, 2, 4, 5, 6]:
            if cpa.should_trigger(vid, rsu_id=0, current_time=t):
                triggered.append(vid)
        print(f"  t = {t*1000:.0f}ms: 触发车辆 = {triggered}")

    # 测试5: 冲突概率计算
    print("\n【测试5】冲突概率")
    collision_prob = cpa.calculate_collision_probability(rsu_id=0, collision_window=0.005)
    print(f"  冲突窗口5ms时的冲突概率: {collision_prob:.2%}")

    # 测试6: 验证相位分配
    print("\n【测试6】验证相位分配")
    current_phases = cpa.get_rsu_phases(rsu_id=0)
    valid, issues = verify_phase_allocation(current_phases, T_check=0.1)
    print(f"  相位分配有效: {valid}")
    if issues:
        for issue in issues:
            print(f"    问题: {issue}")

    # 测试7: 统计信息
    print("\n【测试7】统计信息")
    stats = cpa.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("CPA测试完成!")
    print("=" * 70)


# 运行测试: python3 -m simulation.algorithms.cpa
