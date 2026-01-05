"""
分布式相位调整算法 (Distributed Phase Adjustment, DPA)
文件: simulation/algorithms/dpa.py

功能说明:
- 实现车辆自主的分布式相位调整
- 2-竞争性算法：在最大间隙中点插入
- 支持车辆切换RSU时的相位迁移

算法来源: docs/03_算法设计_相位对齐调度.md §3
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class DPAConfig:
    """DPA算法配置参数"""

    # 时隙参数
    T_check: float = 0.1  # 检查周期 (s)

    # 间隙阈值
    min_gap_threshold: float = 0.005  # 最小间隙阈值 (s)，默认5ms

    # 相位迁移策略
    migration_strategy: str = "reinsert"  # "reinsert" 或 "preserve"

    # 调试选项
    verbose: bool = False


@dataclass
class PhaseSlot:
    """相位槽位信息"""

    vehicle_id: int
    phase: float  # 相位偏移 (s)
    timestamp: float = 0.0  # 分配时间戳


class DistributedPhaseAdjuster:
    """
    分布式相位调整器

    功能:
    - 车辆自主选择相位（无需集中协调）
    - 2-竞争性算法：在最大间隙中点插入新相位
    - 支持RSU切换时的相位迁移

    算法原理:
    - 新车辆查询当前RSU的已分配相位
    - 找到最大间隙 [φ_j, φ_{j+1}]
    - 在中点 (φ_j + φ_{j+1})/2 处插入

    竞争比证明: docs/03_算法设计_相位对齐调度.md §3.3
    """

    def __init__(self, config: Optional[DPAConfig] = None):
        """
        初始化DPA调整器

        Args:
            config: DPA配置参数
        """
        self.config = config or DPAConfig()

        # 每个RSU的相位槽位列表: {rsu_id: [PhaseSlot, ...]}
        self._phase_slots: Dict[int, List[PhaseSlot]] = {}

        # 统计信息
        self.stats = {
            "total_insertions": 0,
            "total_migrations": 0,
            "total_removals": 0,
            "average_gap_at_insertion": 0.0,
            "insertion_gaps": [],  # 记录每次插入时的间隙大小
        }

    def initialize_rsu(self, rsu_id: int):
        """
        初始化RSU的相位槽位表

        Args:
            rsu_id: RSU标识符
        """
        if rsu_id not in self._phase_slots:
            self._phase_slots[rsu_id] = []

    def _find_max_gap(self, rsu_id: int) -> Tuple[float, float, float]:
        """
        找到最大相位间隙

        Args:
            rsu_id: RSU标识符

        Returns:
            (gap_start, gap_end, gap_size): 间隙起点、终点、大小

        算法:
        1. 将所有相位排序
        2. 计算相邻相位的间隙
        3. 考虑环绕边界 (最后一个到第一个的间隙)
        4. 返回最大间隙

        来源: docs/03_算法设计_相位对齐调度.md 算法2
        """
        self.initialize_rsu(rsu_id)
        slots = self._phase_slots[rsu_id]
        T_check = self.config.T_check

        # 空RSU：整个周期都是间隙
        if len(slots) == 0:
            return (0.0, T_check, T_check)

        # 单个车辆：间隙为整个周期
        if len(slots) == 1:
            phase = slots[0].phase
            # 间隙从 phase 到 phase + T_check (环绕)
            return (phase, phase, T_check)

        # 排序相位
        phases = sorted([slot.phase for slot in slots])

        # 计算所有间隙
        max_gap = 0.0
        max_start = 0.0
        max_end = 0.0

        # 相邻间隙
        for i in range(len(phases) - 1):
            gap = phases[i + 1] - phases[i]
            if gap > max_gap:
                max_gap = gap
                max_start = phases[i]
                max_end = phases[i + 1]

        # 环绕间隙 (最后一个到第一个)
        wrap_gap = (T_check - phases[-1]) + phases[0]
        if wrap_gap > max_gap:
            max_gap = wrap_gap
            max_start = phases[-1]
            max_end = phases[0]

        return (max_start, max_end, max_gap)

    def insert_vehicle(self, rsu_id: int, vehicle_id: int, current_time: float = 0.0) -> float:
        """
        插入新车辆的相位（2-竞争性算法）

        Args:
            rsu_id: RSU标识符
            vehicle_id: 车辆标识符
            current_time: 当前时间

        Returns:
            assigned_phase: 分配的相位偏移

        算法（2-竞争性插入）:
        1. 找到最大间隙 [φ_j, φ_{j+1}]
        2. 在中点插入: φ_new = (φ_j + φ_{j+1}) / 2
        3. 处理环绕情况

        来源: docs/03_算法设计_相位对齐调度.md 算法2
        """
        self.initialize_rsu(rsu_id)
        T_check = self.config.T_check

        # 检查车辆是否已存在
        for slot in self._phase_slots[rsu_id]:
            if slot.vehicle_id == vehicle_id:
                if self.config.verbose:
                    print(
                        f"[DPA] 车辆 {vehicle_id} 已在RSU {rsu_id}，相位 = {slot.phase*1000:.2f}ms"
                    )
                return slot.phase

        # 找最大间隙
        gap_start, gap_end, gap_size = self._find_max_gap(rsu_id)

        # 在中点插入
        if gap_end >= gap_start:
            # 正常间隙
            new_phase = (gap_start + gap_end) / 2
        else:
            # 环绕间隙: gap_start -> T_check -> 0 -> gap_end
            # 中点 = gap_start + gap_size/2，然后对 T_check 取模
            new_phase = (gap_start + gap_size / 2) % T_check

        # 创建新槽位
        new_slot = PhaseSlot(vehicle_id=vehicle_id, phase=new_phase, timestamp=current_time)
        self._phase_slots[rsu_id].append(new_slot)

        # 更新统计
        self.stats["total_insertions"] += 1
        self.stats["insertion_gaps"].append(gap_size)
        self.stats["average_gap_at_insertion"] = np.mean(self.stats["insertion_gaps"])

        if self.config.verbose:
            print(f"[DPA] 车辆 {vehicle_id} 插入RSU {rsu_id}")
            print(
                f"      最大间隙: [{gap_start*1000:.2f}, {gap_end*1000:.2f}]ms (大小={gap_size*1000:.2f}ms)"
            )
            print(f"      分配相位: {new_phase*1000:.2f}ms")

        return new_phase

    def remove_vehicle(self, rsu_id: int, vehicle_id: int) -> bool:
        """
        移除车辆的相位分配

        Args:
            rsu_id: RSU标识符
            vehicle_id: 车辆标识符

        Returns:
            success: 是否成功移除
        """
        if rsu_id not in self._phase_slots:
            return False

        initial_len = len(self._phase_slots[rsu_id])
        self._phase_slots[rsu_id] = [
            slot for slot in self._phase_slots[rsu_id] if slot.vehicle_id != vehicle_id
        ]

        removed = len(self._phase_slots[rsu_id]) < initial_len

        if removed:
            self.stats["total_removals"] += 1
            if self.config.verbose:
                print(f"[DPA] 车辆 {vehicle_id} 从RSU {rsu_id} 移除")

        return removed

    def migrate_vehicle(
        self, vehicle_id: int, from_rsu_id: int, to_rsu_id: int, current_time: float = 0.0
    ) -> Optional[float]:
        """
        车辆RSU切换时的相位迁移

        Args:
            vehicle_id: 车辆标识符
            from_rsu_id: 源RSU ID
            to_rsu_id: 目标RSU ID
            current_time: 当前时间

        Returns:
            new_phase: 新分配的相位，迁移失败返回None

        策略:
        - "reinsert": 在新RSU重新插入（使用2-竞争性算法）
        - "preserve": 尝试保留原相位（如果不冲突）

        来源: docs/03_算法设计_相位对齐调度.md §3.4
        """
        # 获取旧相位
        old_phase = self.get_vehicle_phase(from_rsu_id, vehicle_id)

        # 从旧RSU移除
        self.remove_vehicle(from_rsu_id, vehicle_id)

        if self.config.migration_strategy == "preserve" and old_phase is not None:
            # 尝试保留原相位
            if self._can_preserve_phase(to_rsu_id, old_phase):
                self.initialize_rsu(to_rsu_id)
                new_slot = PhaseSlot(vehicle_id=vehicle_id, phase=old_phase, timestamp=current_time)
                self._phase_slots[to_rsu_id].append(new_slot)
                self.stats["total_migrations"] += 1

                if self.config.verbose:
                    print(
                        f"[DPA] 车辆 {vehicle_id} 迁移 RSU {from_rsu_id} -> {to_rsu_id}，保留相位 {old_phase*1000:.2f}ms"
                    )

                return old_phase

        # 重新插入策略
        new_phase = self.insert_vehicle(to_rsu_id, vehicle_id, current_time)
        self.stats["total_migrations"] += 1

        if self.config.verbose:
            print(
                f"[DPA] 车辆 {vehicle_id} 迁移 RSU {from_rsu_id} -> {to_rsu_id}，新相位 {new_phase*1000:.2f}ms"
            )

        return new_phase

    def _can_preserve_phase(self, rsu_id: int, phase: float, min_distance: float = None) -> bool:
        """
        检查相位是否可以保留（不与现有相位冲突）

        Args:
            rsu_id: RSU标识符
            phase: 要检查的相位
            min_distance: 最小距离阈值

        Returns:
            can_preserve: 是否可以保留
        """
        if min_distance is None:
            min_distance = self.config.min_gap_threshold

        self.initialize_rsu(rsu_id)
        T_check = self.config.T_check

        for slot in self._phase_slots[rsu_id]:
            diff = abs(slot.phase - phase)
            # 考虑环绕
            diff = min(diff, T_check - diff)
            if diff < min_distance:
                return False

        return True

    def get_vehicle_phase(self, rsu_id: int, vehicle_id: int) -> Optional[float]:
        """
        获取车辆的相位偏移

        Args:
            rsu_id: RSU标识符
            vehicle_id: 车辆标识符

        Returns:
            phase: 相位偏移，若未分配则返回None
        """
        if rsu_id not in self._phase_slots:
            return None

        for slot in self._phase_slots[rsu_id]:
            if slot.vehicle_id == vehicle_id:
                return slot.phase

        return None

    def get_rsu_phases(self, rsu_id: int) -> Dict[int, float]:
        """
        获取RSU的所有相位分配

        Args:
            rsu_id: RSU标识符

        Returns:
            phases: {vehicle_id: phase_offset}
        """
        if rsu_id not in self._phase_slots:
            return {}

        return {slot.vehicle_id: slot.phase for slot in self._phase_slots[rsu_id]}

    def get_all_gaps(self, rsu_id: int) -> List[Tuple[float, float, float]]:
        """
        获取所有相位间隙

        Args:
            rsu_id: RSU标识符

        Returns:
            gaps: [(gap_start, gap_end, gap_size), ...]
        """
        self.initialize_rsu(rsu_id)
        slots = self._phase_slots[rsu_id]
        T_check = self.config.T_check

        if len(slots) == 0:
            return [(0.0, T_check, T_check)]

        if len(slots) == 1:
            phase = slots[0].phase
            return [(phase, phase, T_check)]

        phases = sorted([slot.phase for slot in slots])
        gaps = []

        # 相邻间隙
        for i in range(len(phases) - 1):
            gap_start = phases[i]
            gap_end = phases[i + 1]
            gap_size = gap_end - gap_start
            gaps.append((gap_start, gap_end, gap_size))

        # 环绕间隙
        wrap_start = phases[-1]
        wrap_end = phases[0]
        wrap_size = (T_check - wrap_start) + wrap_end
        gaps.append((wrap_start, wrap_end, wrap_size))

        return gaps

    def calculate_min_gap(self, rsu_id: int) -> float:
        """
        计算最小相位间隙

        Args:
            rsu_id: RSU标识符

        Returns:
            min_gap: 最小间隙 (s)
        """
        gaps = self.get_all_gaps(rsu_id)
        if not gaps:
            return self.config.T_check
        return min(gap[2] for gap in gaps)

    def calculate_gap_variance(self, rsu_id: int) -> float:
        """
        计算间隙方差（衡量均匀性）

        Args:
            rsu_id: RSU标识符

        Returns:
            variance: 间隙方差

        说明: 方差越小，相位分布越均匀
        """
        gaps = self.get_all_gaps(rsu_id)
        if len(gaps) < 2:
            return 0.0

        gap_sizes = [gap[2] for gap in gaps]
        return np.var(gap_sizes)

    def get_competitive_ratio(self, rsu_id: int) -> float:
        """
        计算当前分配的竞争比

        Args:
            rsu_id: RSU标识符

        Returns:
            ratio: 竞争比 (最优间隙 / 最小间隙)

        理论上限: 2-竞争性
        来源: docs/03_算法设计_相位对齐调度.md §3.3
        """
        self.initialize_rsu(rsu_id)
        n = len(self._phase_slots[rsu_id])

        if n <= 1:
            return 1.0

        # 最优间隙 (均匀分配)
        optimal_gap = self.config.T_check / n

        # 实际最小间隙
        actual_min_gap = self.calculate_min_gap(rsu_id)

        if actual_min_gap <= 0:
            return float("inf")

        return optimal_gap / actual_min_gap

    def should_trigger(self, vehicle_id: int, rsu_id: int, current_time: float) -> bool:
        """
        判断车辆是否应该在当前时刻触发CPM任务

        Args:
            vehicle_id: 车辆标识符
            rsu_id: RSU标识符
            current_time: 当前时间 (s)

        Returns:
            should_trigger: 是否应该触发
        """
        phase = self.get_vehicle_phase(rsu_id, vehicle_id)
        if phase is None:
            return False

        T_check = self.config.T_check

        # 计算当前时隙内的偏移
        t_offset = current_time % T_check

        # 容差
        epsilon = T_check * 0.01

        diff = abs(t_offset - phase)
        diff = min(diff, T_check - diff)

        return diff < epsilon

    def get_statistics(self) -> dict:
        """
        获取DPA统计信息

        Returns:
            stats: 统计字典
        """
        result = self.stats.copy()
        result["num_rsus"] = len(self._phase_slots)
        result["total_vehicles"] = sum(len(slots) for slots in self._phase_slots.values())

        # 移除内部列表，只保留统计值
        if "insertion_gaps" in result:
            del result["insertion_gaps"]

        return result

    def reset(self):
        """重置所有状态"""
        self._phase_slots.clear()
        self.stats = {
            "total_insertions": 0,
            "total_migrations": 0,
            "total_removals": 0,
            "average_gap_at_insertion": 0.0,
            "insertion_gaps": [],
        }


# ============================================================
# 辅助函数: 竞争比分析
# ============================================================


def analyze_competitive_ratio(n_vehicles: int, n_trials: int = 100) -> dict:
    """
    分析DPA算法的竞争比

    Args:
        n_vehicles: 车辆数量
        n_trials: 试验次数

    Returns:
        analysis: 分析结果
    """
    config = DPAConfig(T_check=0.1, verbose=False)

    ratios = []
    min_gaps = []

    for _ in range(n_trials):
        dpa = DistributedPhaseAdjuster(config)

        # 随机顺序插入车辆
        vehicle_order = np.random.permutation(n_vehicles)
        for vid in vehicle_order:
            dpa.insert_vehicle(rsu_id=0, vehicle_id=int(vid))

        ratio = dpa.get_competitive_ratio(rsu_id=0)
        min_gap = dpa.calculate_min_gap(rsu_id=0)

        ratios.append(ratio)
        min_gaps.append(min_gap)

    return {
        "n_vehicles": n_vehicles,
        "n_trials": n_trials,
        "mean_ratio": np.mean(ratios),
        "max_ratio": np.max(ratios),
        "min_ratio": np.min(ratios),
        "std_ratio": np.std(ratios),
        "mean_min_gap_ms": np.mean(min_gaps) * 1000,
        "optimal_gap_ms": (0.1 / n_vehicles) * 1000,
    }


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 DPA (分布式相位调整) 算法")
    print("=" * 70)

    # 测试1: 基本插入
    print("\n【测试1】基本插入（2-竞争性算法）")
    config = DPAConfig(T_check=0.1, verbose=True)
    dpa = DistributedPhaseAdjuster(config)

    # 依次插入5辆车
    for vid in range(1, 6):
        phase = dpa.insert_vehicle(rsu_id=0, vehicle_id=vid)

    print(f"\n最终相位分配:")
    phases = dpa.get_rsu_phases(rsu_id=0)
    for vid, phase in sorted(phases.items(), key=lambda x: x[1]):
        print(f"  车辆 {vid}: φ = {phase*1000:.2f}ms")

    # 测试2: 间隙分析
    print("\n【测试2】间隙分析")
    gaps = dpa.get_all_gaps(rsu_id=0)
    print("所有间隙:")
    for start, end, size in gaps:
        print(f"  [{start*1000:.2f}, {end*1000:.2f}]ms (大小={size*1000:.2f}ms)")

    min_gap = dpa.calculate_min_gap(rsu_id=0)
    variance = dpa.calculate_gap_variance(rsu_id=0)
    print(f"最小间隙: {min_gap*1000:.2f}ms")
    print(f"间隙方差: {variance*1e6:.4f}ms²")

    # 测试3: 竞争比
    print("\n【测试3】竞争比分析")
    ratio = dpa.get_competitive_ratio(rsu_id=0)
    optimal_gap = 0.1 / 5  # 均匀分配的间隙
    print(f"最优间隙 (均匀): {optimal_gap*1000:.2f}ms")
    print(f"实际最小间隙: {min_gap*1000:.2f}ms")
    print(f"竞争比: {ratio:.3f} (理论上限=2)")

    # 测试4: 车辆移除
    print("\n【测试4】车辆移除")
    dpa.remove_vehicle(rsu_id=0, vehicle_id=3)
    phases_after = dpa.get_rsu_phases(rsu_id=0)
    print(f"移除车辆3后:")
    for vid, phase in sorted(phases_after.items(), key=lambda x: x[1]):
        print(f"  车辆 {vid}: φ = {phase*1000:.2f}ms")

    # 测试5: RSU迁移
    print("\n【测试5】RSU迁移")
    # 先在RSU 1添加一些车辆
    for vid in [10, 11, 12]:
        dpa.insert_vehicle(rsu_id=1, vehicle_id=vid)

    print(f"迁移前 RSU 1 相位:")
    for vid, phase in dpa.get_rsu_phases(rsu_id=1).items():
        print(f"  车辆 {vid}: φ = {phase*1000:.2f}ms")

    # 迁移车辆1从RSU 0到RSU 1
    new_phase = dpa.migrate_vehicle(vehicle_id=1, from_rsu_id=0, to_rsu_id=1)

    print(f"\n迁移后 RSU 1 相位:")
    for vid, phase in sorted(dpa.get_rsu_phases(rsu_id=1).items(), key=lambda x: x[1]):
        print(f"  车辆 {vid}: φ = {phase*1000:.2f}ms")

    # 测试6: 触发判断
    print("\n【测试6】触发判断")
    test_times = [0.0, 0.025, 0.05, 0.075]
    for t in test_times:
        triggered = []
        for vid in dpa.get_rsu_phases(rsu_id=1).keys():
            if dpa.should_trigger(vid, rsu_id=1, current_time=t):
                triggered.append(vid)
        print(f"  t = {t*1000:.0f}ms: 触发 = {triggered}")

    # 测试7: 大规模竞争比分析
    print("\n【测试7】竞争比统计分析")
    for n in [5, 10, 20]:
        analysis = analyze_competitive_ratio(n_vehicles=n, n_trials=50)
        print(
            f"  {n}辆车: 平均竞争比={analysis['mean_ratio']:.3f}, "
            f"最大={analysis['max_ratio']:.3f}, "
            f"平均最小间隙={analysis['mean_min_gap_ms']:.2f}ms "
            f"(最优={analysis['optimal_gap_ms']:.2f}ms)"
        )

    # 测试8: 统计信息
    print("\n【测试8】统计信息")
    stats = dpa.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("DPA测试完成!")
    print("=" * 70)


# 运行测试: python3 -m simulation.algorithms.dpa
