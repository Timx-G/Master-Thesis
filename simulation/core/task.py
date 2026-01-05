"""
CPM任务类定义

功能说明：
- 定义了CPM协作感知任务的数据结构
- 包含任务大小、计算需求、截止时间等属性
- 跟踪任务的生命周期状态

"""

from dataclasses import dataclass, field  # 2. 自动生成类的工具
from enum import Enum  # 4. 枚举类型
from typing import Optional  # 3. 类型提示

import numpy as np

from simulation.core import vehicle  # 1. 引入数学库


class TaskStatus(Enum):
    """任务状态枚举"""

    GENERATED = "generated"  # 任务已生成,未处理
    QUEUED = "queued"  # 任务已排队,等待处理
    TRANSMITTING = "transmitting"  # 任务正在传输
    COMPUTING = "computing"  # 任务正在计算
    COMPLETED = "completed"  # 任务已完成
    DROPPED = "dropped"  # 任务被丢弃
    FAILED = "failed"  # 任务处理失败


@dataclass
class CPMTask:
    """
    CPM 协作感知任务类
    任务模型基于 ETSI TS 103 324 标准
    参考文档：docs/01_系统模型.md $2.3
    """

    # ==================== 基础属性 ====================
    task_id: int  # 任务唯一标识符
    vehicle_id: int  # 任务所属车辆标识符
    generation_time: float  # 任务生成时间 (s)

    # ==================== 任务需求 ====================
    # 数据量模型: D_i(k) = D_base + η * N_obj
    data_size: float  # 任务数据大小 (bits)
    compute_cycles: float  # 任务计算需求 (CPU cycles)
    deadline: float  # 任务截止时间 (s)

    # 感知目标信息
    num_objects: int = 0  # 检测(感知)到的目标数量

    # 结果数据大小 (通常远小于输入)
    result_size: float = 0.0  # 结果数据大小 (bits)

    # ==================== 决策变量 ====================
    offload_decision: bool = False  # 是否选择卸载任务
    target_rsu_id: Optional[int] = None  # 目标RSU标识符 (若卸载)
    allocated_cpu_freq: float = 0.0  # 分配的CPU频率 (Hz)
    allocated_bandwidth: float = 0.0  # 分配的带宽 (Hz)

    # ==================== 时延组成 ====================
    # 参考文档： docs/01_系统模型.md $4.2.1
    transmission_delay: float = 0.0  # 传输时延 (s)
    queue_delay: float = 0.0  # 排队时延 (s)
    computation_delay: float = 0.0  # 计算时延 (s)
    download_delay: float = 0.0  # 结果下载时延 (s)

    # ==================== 能耗组成 ====================
    energy_consumption: float = 0.0  # 任务总能耗 (J)

    # ==================== 状态跟踪 ====================
    status: TaskStatus = TaskStatus.GENERATED  # 任务当前状态
    completion_time: Optional[float] = None  # 任务完成时刻 (s)

    # 时间戳记录
    queued_time: Optional[float] = None  # 任务入队时刻 (s)
    start_transmission_time: Optional[float] = None  # 传输开始时刻 (s)
    start_computation_time: Optional[float] = None  # 计算开始时刻 (s)

    def __post_init__(self):
        """
        初始化后处理: 确保数据合理性

        逻辑：
        1. 如果未指定结果大小，默认为输入的10%
        2. 验证参数合理性
        """

        if self.result_size == 0.0:
            self.result_size = 0.1 * self.data_size  # 默认结果大小为输入的10%

        # 参数合理性检查
        assert self.data_size > 0, "数据大小必须大于0"
        assert self.compute_cycles > 0, "计算需求必须大于0"
        assert self.deadline > 0, "截止时间必须大于0"

    def get_total_delay(self) -> float:
        """
        计算任务的总时延

            Returns:
                total_delay: 总延迟(s)

            逻辑：
            - 本地计算： 仅计算延迟
            - 卸载计算： 传输延迟 + 排队延迟 + 计算延迟 + 下载延迟
        """
        if not self.offload_decision:
            # 本地处理： 仅计算时间
            return self.computation_delay
        else:
            # 卸载处理： 传输 + 排队 + 计算 + 下载
            return (
                self.transmission_delay
                + self.queue_delay
                + self.computation_delay
                + self.download_delay
            )

    def is_deadline_violated(self) -> bool:
        """
        判断任务是否违反截止时间

            Returns:
                violated: True = 违反截止时间, False = 未违反

            逻辑：
            - 如果任务未完成，无法判断(返回False)
            - 如果已完成，检查(完成时间 - 截止时间) 是否 > 截止时间
        """
        if self.completion_time is None:
            return False  # 任务未完成，无法判断

        actual_delay = self.completion_time - self.generation_time
        return actual_delay > self.deadline

    def get_age_of_information(self, current_time: float) -> float:
        """
        计算任务的信息年龄 (AoI)

            Args:
                current_time: 当前时间 (s)

            Returns:
                aoi: 信息年龄 (s)

            逻辑：
            - 如果任务已完成： AoI = 当前时间 - 生成时间
            - 如果任务未完成： AoI = 当前时间 - 生成时间(信息持续老化)

        参考： docs/01_系统模型.md $5.1
        """
        return current_time - self.generation_time

    def mark_as_queued(self, current_time: float):
        """
        标记任务为排队状态

            Args:
                current_time: 当前时间 (s)

            逻辑：
            - 更新任务状态为 QUEUED
            - 记录入队时间戳
        """
        self.status = TaskStatus.QUEUED
        self.queued_time = current_time

    def mark_as_transmitting(self, current_time: float):
        """
        标记任务为传输中状态

            Args:
                current_time: 当前时间 (s)

            逻辑：
            - 更新任务状态为 TRANSMITTING
            - 记录传输开始时间戳
        """
        self.status = TaskStatus.TRANSMITTING
        self.start_transmission_time = current_time

    def mark_as_computing(self, current_time: float):
        """
        标记任务为计算中状态

            Args:
                current_time: 当前时间 (s)

            逻辑：
            - 更新任务状态为 COMPUTING
            - 记录计算开始时间戳
        """
        self.status = TaskStatus.COMPUTING
        self.start_computation_time = current_time

    def mark_as_completed(self, current_time: float):
        """
        标记任务为已完成状态

            Args:
                current_time: 当前时间 (s)

            逻辑：
            - 更新任务状态为 COMPLETED
            - 记录完成时间戳
        """
        self.status = TaskStatus.COMPLETED
        self.completion_time = current_time

    def mark_as_dropped(self):
        """
        标记任务为已丢弃状态

            逻辑：
            - 更新任务状态为 DROPPED
        """
        self.status = TaskStatus.DROPPED

    def mark_as_failed(self):
        """
        标记任务为处理失败状态

            逻辑：
            - 更新任务状态为 FAILED
        """
        self.status = TaskStatus.FAILED

    def get_summary(self) -> dict:
        """
        获取任务摘要信息

            Returns:
                summary: 任务摘要字典

            逻辑：
            - 返回任务的关键属性和状态信息
        """
        return {
            "task_id": self.task_id,
            "vehicle_id": self.vehicle_id,
            "generation_time": self.generation_time,
            "data_size_mb": self.data_size / (8 * 1024 * 1024),  # 转换为MB
            "compute_mcycles": self.compute_cycles / 1e6,  # 转换为MCycles
            "deadline_ms": self.deadline * 1000,  # 转换为ms
            "num_objects": self.num_objects,
            "offload_decision": "Offload" if self.offload_decision else "Local",
            "target_rsu": self.target_rsu_id,
            "status": self.status.value,
            "total_delay_ms": self.get_total_delay() * 1000,  # 转换为ms
            "deadline_violated": self.is_deadline_violated(),
            "energy_j": self.energy_consumption,
        }

    def __repr__(self) -> str:
        """任务对象的字符串表示"""
        return (
            f"CPMTask(task_id={self.task_id}, vehicle_id={self.vehicle_id}, "
            f"data={self.data_size/8*1024:.1f}KB,"
            f"compute={self.compute_cycles/1e6:.1f}MCycles, "
            f"deadline={self.deadline*1000:.0f}ms, "
            f"status={self.status.value})"
        )


# ============================================================
# 任务工厂类
# ============================================================


class TaskFactory:
    """
    任务工厂类

    功能：
    - 根据车辆状态(配置参数)生成CPM任务实例
    - 自动计算任务大小和计算需求
    """

    def __init__(self, config: dict):
        """
        初始化任务工厂

        Args:
            config: 配置参数字典(parameters.yaml的cpm部分)
        """
        self.data_base = config.get("data_base", 4.0e6)  # 基础数据大小 (bits)
        self.data_per_object = config.get("data_per_object", 1.6e5)  # 每目标数据大小 (bits)
        self.compute_base = config.get("compute_base", 1.0e8)  # 基础计算需求 (CPU cycles)
        self.compute_per_object = config.get(
            "compute_per_object", 2.0e7
        )  # 每目标计算需求 (CPU cycles)
        self.num_objects_mean = config.get("num_objects_mean", 15)  # 目标数量均值
        self.num_objects_std = config.get("num_objects_std", 5)  # 目标数量标准差
        self.deadline = config.get("deadline", 0.08)  # 任务截止时间 (s)
        self.result_data_ratio = config.get("result_data_ratio", 0.1)  # 结果数据大小比例

        self._task_id_counter = 0  # 任务ID计数器

    def create_task(self, vehicle_id: int, generation_time: float) -> CPMTask:
        """ "
        创建一个CPM任务

        Args:
            vehicle_id: 任务所属车辆ID
            generation_time: 任务生成时间 (s)

        Returns:
            task: 生成的CPM任务实例

        逻辑：
        1. 随机生成检测到的目标数量 (正态分布)
        2. 计算任务数据大小和计算需求
        3. 创建并返回CPMTask实例
        """
        # 生成目标数量（正态分布，限制在合理范围）
        num_objects = int(np.random.normal(self.num_objects_mean, self.num_objects_std))
        num_objects = max(0, min(num_objects, 50))  # 确保非负且不超过50个目标

        # 计算数据量： D = D_base + η * N_obj
        data_size = self.data_base + self.data_per_object * num_objects

        # 计算需求： C = C_base + ζ * N_obj
        compute_cycles = self.compute_base + self.compute_per_object * num_objects

        # 结果数据量
        result_size = self.result_data_ratio * data_size

        # 创建任务
        task = CPMTask(
            task_id=self._task_id_counter,
            vehicle_id=vehicle_id,
            generation_time=generation_time,
            data_size=data_size,
            compute_cycles=compute_cycles,
            deadline=self.deadline,
            num_objects=num_objects,
            result_size=result_size,
        )

        self._task_id_counter += 1  # 任务ID自增
        return task


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 CPMTask 类")
    print("=" * 70)

    # 测试1: 创建任务
    print("\n【测试1】创建任务")
    task = CPMTask(
        task_id=1,
        vehicle_id=10,
        generation_time=0.0,
        data_size=4.0e6,  # 0.5 MB
        compute_cycles=1.2e8,  # 120 MCycles
        deadline=0.08,  # 80 ms
        num_objects=10,
    )
    print(f"任务创建成功: {task}")
    print(f"任务摘要: {task.get_summary()}")

    # 测试2: 状态转换
    print("\n【测试2】状态转换")
    print(f"初始状态: {task.status.value}")

    task.mark_as_queued(0.01)
    if task.queued_time is not None:
        print(f"入队后: {task.status.value}, 入队时刻={task.queued_time*1000:.1f}ms")
    else:
        print(f"入队后: {task.status.value}, 入队时刻=None")

    task.mark_as_transmitting(0.02)
    if task.start_transmission_time is not None:
        print(f"传输中: {task.status.value}, 传输开始={task.start_transmission_time*1000:.1f}ms")
    else:
        print(f"传输中: {task.status.value}, 传输开始=None")

    task.mark_as_computing(0.03)
    if task.start_computation_time is not None:
        print(f"计算中: {task.status.value}, 计算开始={task.start_computation_time*1000:.1f}ms")
    else:
        print(f"计算中: {task.status.value}, 计算开始=None")

    # 模拟延迟
    task.transmission_delay = 0.01
    task.queue_delay = 0.005
    task.computation_delay = 0.02
    task.download_delay = 0.001

    task.mark_as_completed(0.066)
    if task.completion_time is not None:
        print(f"已完成: {task.status.value}, 完成时刻={task.completion_time*1000:.1f}ms")
    else:
        print(f"已完成: {task.status.value}, 完成时刻=None")
    print(f"总延迟: {task.get_total_delay()*1000:.2f}ms")
    print(f"是否违反截止时间: {task.is_deadline_violated()}")

    # 测试3: AoI计算
    print("\n【测试3】AoI计算")
    current_time = 0.1
    aoi = task.get_age_of_information(current_time)
    print(f"当前时刻={current_time*1000:.0f}ms, AoI={aoi*1000:.0f}ms")

    # 测试4: 任务工厂
    print("\n【测试4】任务工厂")
    config = {
        "data_base": 4.0e6,
        "data_per_object": 1.6e5,
        "compute_base": 1.0e8,
        "compute_per_object": 2.0e7,
        "num_objects_mean": 15,
        "num_objects_std": 5,
        "deadline": 0.08,
    }

    factory = TaskFactory(config)

    print("生成10个随机任务:")
    for i in range(10):
        task = factory.create_task(vehicle_id=i, generation_time=i * 0.1)
        print(f"  {task}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


# 运行测试：python3 -m simulation.core.task
