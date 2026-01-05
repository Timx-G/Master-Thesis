"""
车辆类定义
文件： simulation/core/vehicle.py

参数来源： config/parameters.yaml的vehicle 和 network 部分
参考文档： docs/01_系统模型.md $1
"""

from dataclasses import dataclass, field
from typing import Optional
from xml.dom.minicompat import NodeList

import numpy as np


@dataclass
class Vehicle:
    """
    车辆数据类

    ps: 属性应和parameters.yaml的vehicle对应
    """

    # ==================== 基础标识 ====================
    id: int  # 车辆ID

    # ==================== 运动状态 ====================
    # 参数来源： parameters.yaml 的 network.lane_width -> = 3.5m
    position: np.ndarray  # 车辆位置，二维坐标 (x, y)，单位：m
    speed: float  # 车辆速度，单位：m/s, 范围[10,35]
    heading: float = 0.0  # 车辆航向角，单位：弧度，范围[0, 2π]
    lane: int = 0  # 车辆所在车道编号，范围[0, num_lanes-1]

    # ==================== 计算资源 ====================
    # 参数来源： parameters.yaml -> vehicle.local_cpu_freq = 2.0e9
    local_cpu_freq: float = 2.0e9  # 车辆本地CPU频率，单位：Hz

    # ==================== CPM生成状态 ====================
    # 用于ETSI TS 103 324 触发条件判断
    last_cpm_time: float = 0.0  # 上次生成CPM的时间，单位：s
    last_cpm_position: Optional[np.ndarray] = None  # 上次生成CPM时的位置，二维坐标 (x, y)，单位：m
    last_cpm_speed: float = 0.0  # 上次生成CPM时的速度，单位：m/s
    last_cpm_heading: float = 0.0  # 上次生成CPM时的航向角，单位：弧度

    # ==================== 网络连接 =======================
    connected_rsu_id: Optional[int] = None  # 当前连接的RSU ID，None表示未连接
    phase_offset: float = 0.0  # 当前连接的RSU的相位偏移，单位：s 范围: [0, T_check)

    # ==================== 性能统计 =======================
    total_tasks_generated: int = 0  # 生成的任务总数
    total_tasks_completed: int = 0  # 完成的任务总数
    total_energy_consumed: float = 0.0  # 总能耗，单位：J
    current_aoi: float = 0.0  # 当前平均AoI，单位：s

    # ==================== 内部状态 =======================
    velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0])
    )  # 车辆速度向量，二维 (vx, vy)，单位：m/s

    def __post_init__(self):
        """
        初始化后处理

        逻辑：
        1. 如果未设置上次CPM位置，初始化当前位置
        2. 根据速度和航向计算速度向量
        """
        if self.last_cpm_position is None:
            self.last_cpm_position = self.position.copy()
            self.last_cpm_speed = self.speed
            self.last_cpm_heading = self.heading

        # 计算速度向量 [vx, vy]
        self.velocity = np.array(
            [self.speed * np.cos(self.heading), self.speed * np.sin(self.heading)]
        )

    def update_position(self, dt: float, road_length: float):
        """
        更新车辆位置（简化的匀速直线运动模型

        Args:
            dt (float): 时间步长，单位：s; 来自parameters.yaml
            road_length (float): 道路长度，单位：m，用于位置环绕 parameters.yaml

        逻辑：
        1. 根据速度向量和时间步长计算位移
        2. 更新x坐标(沿道路方向)
        3. 处理循环边界(到达终点后重置到起点)
        """
        displacement = self.speed * dt
        self.position[0] += displacement

        # 循环道路模型
        if self.position[0] > road_length:
            self.position[0] -= road_length
        elif self.position[0] < 0:
            self.position[0] += road_length

    def update_speed(
        self, dt: float, speed_std: float = 0.5, speed_min: float = 10.0, speed_max: float = 35.0
    ):
        """
        更新速度（添加随机波动 模拟 真实驾驶）

        Args:
            dt (float): 时间步长，单位：s;
            speed_std (float): 速度波动标准差，单位：m/s
            speed_min (float): 最小速度，单位：m/s
            speed_max (float): 最大速度，单位：m/s

        逻辑：
            1. 添加高斯白噪声(均值0, 标准差与dt成比例)
            2. 限制速度在[min, max]范围内
        """
        noise = np.random.normal(0, speed_std * np.sqrt(dt))
        self.speed += noise
        self.speed = np.clip(self.speed, speed_min, speed_max)

        # 更新速度向量
        self.velocity[0] = self.speed * np.cos(self.heading)
        self.velocity[1] = self.speed * np.sin(self.heading)

    def compute_local_time(self, compute_cycles: float) -> float:
        """
        计算本地处理时间

        Args:
            compute_cycles (float): 任务计算需求，单位：CPU cycles

        Returns:
            time: 计算时间(s)

        公式： T_local = C / f_local
        来源： docs/01_系统模型.md $4.1.1
        """
        return compute_cycles / self.local_cpu_freq

    def compute_local_energy(self, compute_cycles: float, kappa: float = 1.0e-27) -> float:
        """
        计算本地处理能耗

        Args:
            compute_cycles (float): 任务计算需求，单位：CPU cycles
            kappa (float): 车辆处理器能耗系数，单位：Joule/(cycle·Hz²)

        Returns:
            energy: 能耗(J)

        公式： E_local = K * f_local^2 * C
        来源： docs/01_系统模型.md $4.1.2 (CMOS 功耗模型)
        """
        return kappa * (self.local_cpu_freq**2) * compute_cycles

    def get_distance_to_rsu(self, rsu_position: np.ndarray) -> float:
        """
        计算与RSU的欧几里得距离

        Args:
            rsu_position: RSU位置 [x, y]

        Returns:
            distance: 距离 (m)
        """
        return float(np.linalg.norm(self.position - rsu_position))

    def is_in_rsu_coverage(self, rsu_position: np.ndarray, coverage_radius: float) -> bool:
        """
        判断是否在RSU覆盖范围内

        Args:
            rsu_position: RSU位置[x,y]
            coverage_radius: RSU覆盖半径(m) ; 来自parameters.yaml

        Returns:
            covered: 是否在覆盖范围内
        """
        distance = self.get_distance_to_rsu(rsu_position)
        return distance <= coverage_radius

    def update_cpm_state(self, current_time: float):
        """
        更新CPM生成状态（在生成新任务后调用）

        Args:
            current_time (float): 当前时间，单位：s
        """
        self.last_cpm_time = current_time
        self.last_cpm_position = self.position.copy()
        self.last_cpm_speed = self.speed
        self.last_cpm_heading = self.heading

    def get_state_dict(self) -> dict:
        """
        获取车辆状态（用于日志记录）

        Returns:
            state: 状态字典
        """
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "speed": self.speed,
            "heading": self.heading,
            "lane": self.lane,
            "connected_rsu": self.connected_rsu_id,
            "phase": self.phase_offset,
            "aoi": self.current_aoi,
            "total_tasks": self.total_tasks_generated,
            "completed_tasks": self.total_tasks_completed,
            "energy": self.total_energy_consumed,
        }

    def __repr__(self) -> str:
        return (
            f"Vehicle(id={self.id}, pos=({self.position[0]:.1f}, {self.position[1]:.1f}),)"
            f"speed={self.speed:.2f}m/s, lane={self.lane},rsu={self.connected_rsu_id})"
        )


# ==================== 单元测试 =======================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 Vehicle 类")
    print("=" * 70)

    # 从parameters.yaml加载参数（模拟）
    road_length = 3000.0
    speed_min = 10.0
    speed_max = 35.0

    # 创建车辆
    v = Vehicle(id=1, position=np.array([100.0, 7.0]), speed=25.0, lane=2)  # lane=2, lane_width=3.5

    print(f"\n【1】初始状态:")
    print(f"   {v}")
    print(f"   速度向量: {v.velocity}")

    # 测试运动更新
    print(f"\n【2】模拟10秒运动 (dt=0.1s):")
    for i in range(10):
        v.update_position(dt=0.1, road_length=road_length)
        v.update_speed(dt=0.1, speed_min=speed_min, speed_max=speed_max)
        if i % 2 == 0:  # 每0.2秒打印一次
            print(f"   t={i*0.1:.1f}s: pos={v.position[0]:.1f}m, speed={v.speed:.2f}m/s")

    # 测试计算时间和能耗
    print(f"\n【3】本地计算测试:")
    compute_cycles = 1.0e8  # 100 MCycles
    time = v.compute_local_time(compute_cycles)
    energy = v.compute_local_energy(compute_cycles)
    print(f"   计算需求: {compute_cycles/1e6:.0f} MCycles")
    print(f"   计算时间: {time*1000:.2f} ms")
    print(f"   能耗: {energy:.6f} J")

    # 测试RSU覆盖判断
    print(f"\n【4】RSU覆盖测试:")
    rsu_pos = np.array([500.0, 0.0])
    coverage_radius = 400.0
    distance = v.get_distance_to_rsu(rsu_pos)
    is_covered = v.is_in_rsu_coverage(rsu_pos, coverage_radius)
    print(f"   RSU位置: {rsu_pos}")
    print(f"   车辆位置: {v.position}")
    print(f"   距离: {distance:.1f}m")
    print(f"   覆盖半径: {coverage_radius}m")
    print(f"   是否覆盖: {'是' if is_covered else '否'}")

    # 测试状态字典
    print(f"\n【5】状态字典:")
    state = v.get_state_dict()
    for key, value in state.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
