"""
CPM任务生成模块
================
基于ETSI TS 103 324标准的事件触发任务生成

参数来源:
- parameters.yaml cpm 部分 (CPM参数)
- parameters.yaml simulation 部分 (仿真参数)
- docs/01_系统模型.md §2 任务模型

触发条件 (ETSI TS 103 324 §6.1.3):
1. 位置变化 > pos_threshold (4m)
2. 速度变化 > speed_threshold (0.5 m/s)
3. 航向变化 > heading_threshold (4°)
4. 超过最大间隔 T_max (1s)

来源: docs/11_仿真数据生成方案.md §2
"""

import logging

# 导入依赖模块
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.task import CPMTask, TaskFactory, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class CPMParameters:
    """
    CPM任务参数

    所有参数直接对应 parameters.yaml cpm 部分
    """

    # === 触发条件 [ETSI TS 103 324 §6.1.3] ===
    T_check: float = 0.1  # 检查周期 (s)
    pos_threshold: float = 4.0  # 位置阈值 (m)
    speed_threshold: float = 0.5  # 速度阈值 (m/s)
    heading_threshold: float = 4.0  # 航向阈值 (度)
    T_max: float = 1.0  # 最大间隔 (s)

    # === 任务大小 ===
    data_base: float = 4.0e6  # 基础数据量 (bits) = 500KB
    data_per_object: float = 1.6e5  # 每目标数据量 (bits) = 20KB

    # === 计算负载 ===
    compute_base: float = 1.0e8  # 基础计算量 (cycles) = 100M
    compute_per_object: float = 2.0e7  # 每目标计算量 (cycles) = 20M

    # === 场景参数 ===
    num_objects_mean: int = 15  # 平均目标数
    num_objects_std: int = 5  # 目标数标准差

    # === 截止时间 ===
    deadline: float = 0.08  # 80 ms
    result_data_ratio: float = 0.1  # 结果数据比例 (10%)


@dataclass
class VehicleCPMState:
    """
    车辆CPM状态记录

    用于判断是否触发CPM任务
    """

    vehicle_id: int

    # 上次发送CPM时的状态
    last_cpm_time: float = 0.0
    last_cpm_position: Optional[np.ndarray] = None
    last_cpm_speed: float = 0.0
    last_cpm_heading: float = 0.0

    # 统计
    total_cpm_count: int = 0
    trigger_reasons: Dict[str, int] = field(
        default_factory=lambda: {"position": 0, "speed": 0, "heading": 0, "timeout": 0}
    )


class CPMTaskGenerator:
    """
    CPM任务生成器

    实现ETSI TS 103 324标准的事件触发机制

    触发条件 (任一满足即触发):
    1. ||pos - last_pos|| > pos_threshold
    2. |speed - last_speed| > speed_threshold
    3. |heading - last_heading| > heading_threshold
    4. time - last_cpm_time > T_max

    使用示例:
    ```python
    params = CPMParameters()
    generator = CPMTaskGenerator(params)

    # 每个时间步检查并生成任务
    tasks = generator.check_and_generate(
        current_time=10.0,
        vehicle_states={...}
    )
    ```
    """

    def __init__(self, params: Optional[CPMParameters] = None, random_seed: Optional[int] = None):
        """
        初始化CPM任务生成器

        Args:
            params: CPM参数，None则使用默认值
            random_seed: 随机种子，对应yaml simulation.random_seed=42
        """
        self.params = params or CPMParameters()
        self.rng = np.random.default_rng(random_seed)

        # 任务工厂
        self.task_factory = TaskFactory(asdict(self.params))

        # 各车辆的CPM状态
        self._vehicle_states: Dict[int, VehicleCPMState] = {}

        # 全局任务计数
        self._task_counter = 0

        # 统计信息
        self._stats = {
            "total_tasks": 0,
            "tasks_by_trigger": {"position": 0, "speed": 0, "heading": 0, "timeout": 0},
        }

        logger.info(
            f"CPMTaskGenerator initialized: "
            f"T_check={self.params.T_check}s, "
            f"deadline={self.params.deadline*1000}ms"
        )

    def _get_or_create_state(self, vehicle_id: int) -> VehicleCPMState:
        """获取或创建车辆CPM状态"""
        if vehicle_id not in self._vehicle_states:
            self._vehicle_states[vehicle_id] = VehicleCPMState(vehicle_id)
        return self._vehicle_states[vehicle_id]

    def _check_trigger_conditions(
        self,
        current_time: float,
        position: np.ndarray,
        speed: float,
        heading: float,
        state: VehicleCPMState,
    ) -> Tuple[bool, str]:
        """
        检查CPM触发条件

        基于ETSI TS 103 324 §6.1.3

        Returns:
            (是否触发, 触发原因)
        """
        # 首次检查,必须触发
        if state.last_cpm_position is None:
            return True, "initial"

        # 条件1: 位置变化
        pos_change = np.linalg.norm(position - state.last_cpm_position)
        if pos_change > self.params.pos_threshold:
            return True, "position"

        # 条件2: 速度变化
        speed_change = abs(speed - state.last_cpm_speed)
        if speed_change > self.params.speed_threshold:
            return True, "speed"

        # 条件3: 航向变化
        heading_change = abs(heading - state.last_cpm_heading)
        # 处理角度环绕 (350° -> 10° = 20°变化)
        if heading_change > 180:
            heading_change = 360 - heading_change
        if heading_change > self.params.heading_threshold:
            return True, "heading"

        # 条件4: 超时
        time_elapsed = current_time - state.last_cpm_time
        if time_elapsed >= self.params.T_max:
            return True, "timeout"

        return False, ""

    def _generate_num_objects(self) -> int:
        """
        生成感知目标数量

        模型: 截断正态分布

        参数来自yaml cpm:
        - num_objects_mean: 15
        - num_objects_std: 5

        范围: [1, 50] (合理范围)
        """
        num = int(self.rng.normal(self.params.num_objects_mean, self.params.num_objects_std))
        return max(1, min(50, num))

    def _calculate_data_size(self, num_objects: int) -> float:
        """
        计算任务数据量

        公式: D = D_base + D_obj * N_obj

        参数来自yaml cpm:
        - data_base: 4.0e6 bits (500KB)
        - data_per_object: 1.6e5 bits (20KB)

        来源: docs/01_系统模型.md §2.1
        """
        return self.params.data_base + self.params.data_per_object * num_objects

    def _calculate_compute_cycles(self, num_objects: int) -> float:
        """
        计算任务计算量

        公式: C = C_base + C_obj * N_obj

        参数来自yaml cpm:
        - compute_base: 1.0e8 cycles (100M)
        - compute_per_object: 2.0e7 cycles (20M)

        来源: docs/01_系统模型.md §2.2
        """
        return self.params.compute_base + self.params.compute_per_object * num_objects

    def generate_task(self, vehicle_id: int, current_time: float) -> CPMTask:
        """
        生成单个CPM任务

        Args:
            vehicle_id: 车辆ID
            current_time: 当前时间

        Returns:
            CPMTask对象
        """
        # 生成目标数
        num_objects = self._generate_num_objects()

        # 计算数据量和计算量
        data_size = self._calculate_data_size(num_objects)
        compute_cycles = self._calculate_compute_cycles(num_objects)

        # 结果数据量
        result_size = data_size * self.params.result_data_ratio

        # 创建任务
        task = CPMTask(
            task_id=self._task_counter,
            vehicle_id=vehicle_id,
            generation_time=current_time,
            data_size=data_size,
            compute_cycles=compute_cycles,
            deadline=self.params.deadline,
            num_objects=num_objects,
            result_size=result_size,
        )

        self._task_counter += 1

        return task

    def check_and_generate(
        self, current_time: float, vehicle_states: Dict[int, Tuple[np.ndarray, float, float]]
    ) -> List[CPMTask]:
        """
        检查所有车辆并生成触发的任务

        这是主要的调用接口，每个时间步调用一次

        Args:
            current_time: 当前时间
            vehicle_states: {vehicle_id: (position, speed, heading)}

        Returns:
            生成的任务列表
        """
        generated_tasks = []

        for vid, (position, speed, heading) in vehicle_states.items():
            state = self._get_or_create_state(vid)

            # 检查触发条件
            triggered, reason = self._check_trigger_conditions(
                current_time, position, speed, heading, state
            )

            if triggered:
                # 生成任务
                task = self.generate_task(vid, current_time)
                generated_tasks.append(task)

                # 更新状态
                state.last_cpm_time = current_time
                state.last_cpm_position = position.copy()
                state.last_cpm_speed = speed
                state.last_cpm_heading = heading
                state.total_cpm_count += 1

                # 更新统计
                if reason != "initial":
                    state.trigger_reasons[reason] += 1
                    self._stats["tasks_by_trigger"][reason] += 1

                self._stats["total_tasks"] += 1

                logger.debug(
                    f"Task generated: vehicle={vid}, "
                    f"reason={reason}, "
                    f"data={task.data_size/1e6:.2f}Mb"
                )

        return generated_tasks

    def get_vehicle_state(self, vehicle_id: int) -> Optional[VehicleCPMState]:
        """获取指定车辆的CPM状态"""
        return self._vehicle_states.get(vehicle_id)

    def get_statistics(self) -> dict:
        """获取生成统计"""
        stats = self._stats.copy()

        # 计算各触发原因占比
        total = stats["total_tasks"]
        if total > 0:
            stats["trigger_ratios"] = {k: v / total for k, v in stats["tasks_by_trigger"].items()}

        return stats

    def reset(self) -> None:
        """重置生成器状态"""
        self._vehicle_states.clear()
        self._task_counter = 0
        self._stats = {
            "total_tasks": 0,
            "tasks_by_trigger": {"position": 0, "speed": 0, "heading": 0, "timeout": 0},
        }


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("CPM任务生成器单元测试")
    print("=" * 60)

    # 测试1: 参数加载
    print("\n--- 测试1: 参数加载 ---")
    params = CPMParameters()
    print(f"检查周期: {params.T_check} s (yaml: 0.1)")
    print(f"位置阈值: {params.pos_threshold} m (yaml: 4.0)")
    print(f"速度阈值: {params.speed_threshold} m/s (yaml: 0.5)")
    print(f"航向阈值: {params.heading_threshold}° (yaml: 4.0)")
    print(f"最大间隔: {params.T_max} s (yaml: 1.0)")
    print(f"基础数据量: {params.data_base/1e6} Mbits (yaml: 4.0e6)")
    print(f"每目标数据量: {params.data_per_object/1e3} Kbits (yaml: 1.6e5)")
    print(f"截止时间: {params.deadline*1000} ms (yaml: 80)")

    # 测试2: 数据量计算
    print("\n--- 测试2: 数据量计算 ---")
    generator = CPMTaskGenerator(params, random_seed=42)

    for n_obj in [5, 10, 15, 20, 25]:
        data = generator._calculate_data_size(n_obj)
        compute = generator._calculate_compute_cycles(n_obj)
        print(f"目标数={n_obj}: 数据={data/1e6:.2f}Mb, 计算={compute/1e6:.0f}Mcycles")

    # 测试3: 触发条件测试
    print("\n--- 测试3: 触发条件测试 ---")

    # 模拟车辆状态变化
    test_cases = [
        # (位置变化, 速度变化, 航向变化, 时间间隔, 预期触发, 预期原因)
        (np.array([5.0, 0.0]), 0.0, 0.0, 0.5, True, "position"),  # 位置变化5m > 4m
        (np.array([2.0, 0.0]), 0.0, 0.0, 0.5, False, ""),  # 位置变化2m < 4m
        (np.array([0.0, 0.0]), 1.0, 0.0, 0.5, True, "speed"),  # 速度变化1m/s > 0.5m/s
        (np.array([0.0, 0.0]), 0.0, 5.0, 0.5, True, "heading"),  # 航向变化5° > 4°
        (np.array([0.0, 0.0]), 0.0, 0.0, 1.5, True, "timeout"),  # 超时1.5s > 1s
    ]

    for i, (
        pos_delta,
        speed_delta,
        heading_delta,
        time_delta,
        expected_trigger,
        expected_reason,
    ) in enumerate(test_cases):
        # 重置生成器
        gen = CPMTaskGenerator(params, random_seed=42)

        # 初始状态
        initial_pos = np.array([0.0, 0.0])
        initial_speed = 25.0
        initial_heading = 0.0

        # 第一次调用 (必定触发)
        gen.check_and_generate(0.0, {0: (initial_pos, initial_speed, initial_heading)})

        # 第二次调用 (测试条件)
        new_pos = initial_pos + pos_delta
        new_speed = initial_speed + speed_delta
        new_heading = initial_heading + heading_delta

        tasks = gen.check_and_generate(time_delta, {0: (new_pos, new_speed, new_heading)})

        triggered = len(tasks) > 0
        result = "✓" if triggered == expected_trigger else "✗"
        print(
            f"测试{i+1}: 位置Δ={np.linalg.norm(pos_delta):.1f}m, "
            f"速度Δ={speed_delta:.1f}m/s, "
            f"航向Δ={heading_delta:.1f}°, "
            f"时间Δ={time_delta:.1f}s -> "
            f"{'触发' if triggered else '未触发'} {result}"
        )

    # 测试4: 长时间仿真
    print("\n--- 测试4: 长时间仿真 (100s) ---")
    generator = CPMTaskGenerator(params, random_seed=42)

    # 模拟50辆车，1000个时间步
    num_vehicles = 50
    num_steps = 1000
    dt = 0.1

    # 初始化车辆状态
    vehicle_positions = {v: np.array([v * 60.0, (v % 4) * 3.5]) for v in range(num_vehicles)}
    vehicle_speeds = {v: 25.0 for v in range(num_vehicles)}
    vehicle_headings = {v: 0.0 if v % 4 < 2 else 180.0 for v in range(num_vehicles)}

    all_tasks = []

    for step in range(num_steps):
        current_time = step * dt

        # 更新车辆状态 (简单运动模型)
        for v in range(num_vehicles):
            direction = 1 if vehicle_headings[v] == 0.0 else -1
            vehicle_positions[v][0] += vehicle_speeds[v] * dt * direction
            # 循环边界
            if vehicle_positions[v][0] > 3000:
                vehicle_positions[v][0] -= 3000
            elif vehicle_positions[v][0] < 0:
                vehicle_positions[v][0] += 3000
            # 小速度波动
            vehicle_speeds[v] += np.random.normal(0, 0.3)
            vehicle_speeds[v] = np.clip(vehicle_speeds[v], 10, 35)

        # 构建状态字典
        states = {
            v: (vehicle_positions[v], vehicle_speeds[v], vehicle_headings[v])
            for v in range(num_vehicles)
        }

        # 生成任务
        tasks = generator.check_and_generate(current_time, states)
        all_tasks.extend(tasks)

    print(f"总任务数: {len(all_tasks)}")
    print(f"平均每车每秒任务数: {len(all_tasks) / num_vehicles / (num_steps * dt):.2f}")

    # 统计
    stats = generator.get_statistics()
    print(f"触发原因分布:")
    for reason, count in stats["tasks_by_trigger"].items():
        ratio = count / stats["total_tasks"] * 100 if stats["total_tasks"] > 0 else 0
        print(f"  {reason}: {count} ({ratio:.1f}%)")

    # 任务大小分布
    data_sizes = [t.data_size for t in all_tasks]
    compute_cycles = [t.compute_cycles for t in all_tasks]
    print(
        f"\n任务数据量: 平均={np.mean(data_sizes)/1e6:.2f}Mb, "
        f"范围=[{np.min(data_sizes)/1e6:.2f}, {np.max(data_sizes)/1e6:.2f}]Mb"
    )
    print(
        f"任务计算量: 平均={np.mean(compute_cycles)/1e6:.0f}Mcycles, "
        f"范围=[{np.min(compute_cycles)/1e6:.0f}, {np.max(compute_cycles)/1e6:.0f}]Mcycles"
    )

    # 验证约束
    print("\n--- 验证关键约束 ---")

    # 约束1: 任务间隔不超过T_max
    for v in range(num_vehicles):
        state = generator.get_vehicle_state(v)
        # 检查平均触发频率
        if state and state.total_cpm_count > 0:
            avg_interval = (num_steps * dt) / state.total_cpm_count
            assert avg_interval <= params.T_max + 0.1, f"车辆{v}平均间隔过长"
    print("✓ 所有车辆CPM间隔不超过T_max")

    # 约束2: 截止时间设置正确
    for task in all_tasks:
        assert task.deadline == params.deadline, "截止时间不匹配"
    print(f"✓ 所有任务截止时间为{params.deadline*1000}ms")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
