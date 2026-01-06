"""
车辆轨迹生成模块
================
生成符合道路约束的车辆运动轨迹

参数来源:
- parameters.yaml network 部分 (道路参数)
- parameters.yaml vehicle 部分 (车辆参数)
- parameters.yaml simulation 部分 (仿真参数)

轨迹模型:
- 直线道路模型 (双向4车道)
- 速度服从截断正态分布
- 位置循环 (到达道路终点后从起点重新进入)

来源: docs/11_仿真数据生成方案.md §1
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Direction(Enum):
    """车辆行驶方向"""

    EASTBOUND = 1  # 东向 (x增大)
    WESTBOUND = -1  # 西向 (x减小)


@dataclass
class TrajectoryParameters:
    """
    轨迹生成参数

    所有参数直接对应 parameters.yaml
    """

    # === 来自 parameters.yaml network ===
    road_length: float = 3000.0  # [3GPP TR 37.885] 道路长度 (m)
    num_lanes: int = 4  # 车道数 (双向各2车道)
    lane_width: float = 3.5  # [ISO标准] 车道宽度 (m)

    # === 来自 parameters.yaml vehicle ===
    vehicle_count: int = 50  # 车辆数量
    speed_mean: float = 25.0  # 平均速度 (m/s) = 90 km/h
    speed_std: float = 5.0  # 速度标准差 (m/s)
    speed_min: float = 10.0  # 最小速度 (m/s) = 36 km/h
    speed_max: float = 35.0  # 最大速度 (m/s) = 126 km/h

    # === 来自 parameters.yaml simulation ===
    dt: float = 0.1  # 时间步长 (s) = T_check
    duration: float = 100.0  # 仿真时长 (s)
    random_seed: int = 42  # 随机种子

    @property
    def num_steps(self) -> int:
        """总仿真步数"""
        return int(self.duration / self.dt)

    @property
    def road_width(self) -> float:
        """道路总宽度"""
        return self.num_lanes * self.lane_width

    @property
    def lane_centers(self) -> List[float]:
        """
        各车道中心线y坐标

        布局 (4车道):
        - 车道0: y = 1.75  (东向外侧)
        - 车道1: y = 5.25  (东向内侧)
        - 车道2: y = 8.75  (西向内侧)
        - 车道3: y = 12.25 (西向外侧)
        """
        return [(i + 0.5) * self.lane_width for i in range(self.num_lanes)]

    def get_lane_direction(self, lane_id: int) -> Direction:
        """
        获取车道行驶方向

        车道0,1: 东向 (x增大)
        车道2,3: 西向 (x减小)
        """
        if lane_id < self.num_lanes // 2:
            return Direction.EASTBOUND
        return Direction.WESTBOUND


@dataclass
class VehicleTrajectory:
    """
    单个车辆的完整轨迹

    存储整个仿真期间的位置和速度序列
    """

    vehicle_id: int
    lane_id: int
    direction: Direction

    # 轨迹数据 (num_steps x 维度)
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    speeds: np.ndarray = field(default_factory=lambda: np.array([]))
    headings: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_state_at(self, step: int) -> Tuple[np.ndarray, float, float]:
        """
        获取指定时间步的状态

        Returns:
            (position, speed, heading)
        """
        return (self.positions[step], self.speeds[step], self.headings[step])

    def get_position_at(self, step: int) -> np.ndarray:
        """获取指定时间步的位置"""
        return self.positions[step]

    def get_speed_at(self, step: int) -> float:
        """获取指定时间步的速度"""
        return self.speeds[step]


class TrajectoryGenerator:
    """
    轨迹生成器

    生成所有车辆在整个仿真期间的轨迹

    特点:
    1. 车辆均匀分布在各车道
    2. 速度服从截断正态分布
    3. 到达道路终点后循环
    4. 支持预生成和实时生成两种模式

    使用示例:
    ```python
    params = TrajectoryParameters()
    generator = TrajectoryGenerator(params)

    # 预生成所有轨迹
    trajectories = generator.generate_all_trajectories()

    # 获取某时刻所有车辆位置
    positions = generator.get_all_positions_at(step=100)
    ```
    """

    def __init__(
        self, params: Optional[TrajectoryParameters] = None, random_seed: Optional[int] = None
    ):
        """
        初始化轨迹生成器

        Args:
            params: 轨迹参数，None则使用默认值
            random_seed: 随机种子，覆盖params中的设置
        """
        self.params = params or TrajectoryParameters()

        seed = random_seed if random_seed is not None else self.params.random_seed
        self.rng = np.random.default_rng(seed)

        # 存储生成的轨迹
        self.trajectories: Dict[int, VehicleTrajectory] = {}

        # 是否已生成
        self._generated = False

        logger.info(
            f"TrajectoryGenerator initialized: "
            f"{self.params.vehicle_count} vehicles, "
            f"{self.params.road_length}m road, "
            f"{self.params.num_lanes} lanes"
        )

    def _generate_initial_speed(self) -> float:
        """
        生成初始速度 (截断正态分布)

        公式: v ~ TruncatedNormal(μ, σ, v_min, v_max)

        参数来自 yaml vehicle 部分:
        - speed_mean: 25.0 m/s
        - speed_std: 5.0 m/s
        - speed_min: 10.0 m/s
        - speed_max: 35.0 m/s

        来源: docs/11_仿真数据生成方案.md §1.2
        """
        while True:
            speed = self.rng.normal(self.params.speed_mean, self.params.speed_std)
            if self.params.speed_min <= speed <= self.params.speed_max:
                return speed

    def _generate_initial_position(self, lane_id: int, direction: Direction) -> np.ndarray:
        """
        生成初始位置

        - x: 在道路范围内均匀分布
        - y: 车道中心线

        Args:
            lane_id: 车道ID
            direction: 行驶方向

        Returns:
            初始位置 [x, y]
        """
        x = self.rng.uniform(0, self.params.road_length)
        y = self.params.lane_centers[lane_id]

        return np.array([x, y])

    def _update_position(
        self, position: np.ndarray, speed: float, direction: Direction, dt: float
    ) -> np.ndarray:
        """
        更新位置 (考虑循环边界)

        公式: x(t+dt) = x(t) + v * dt * direction

        边界处理:
        - 东向: x > road_length 时，x = x - road_length
        - 西向: x < 0 时，x = x + road_length

        来源: docs/11_仿真数据生成方案.md §1.3
        """
        new_position = position.copy()

        # 更新x坐标
        new_position[0] += speed * dt * direction.value

        # 循环边界处理
        if new_position[0] > self.params.road_length:
            new_position[0] -= self.params.road_length
        elif new_position[0] < 0:
            new_position[0] += self.params.road_length

        return new_position

    def _update_speed(self, current_speed: float) -> float:
        """
        更新速度 (简单随机游走模型)

        模型: v(t+dt) = clip(v(t) + noise, v_min, v_max)

        noise ~ N(0, 0.5) m/s (小幅波动)

        来源: docs/11_仿真数据生成方案.md §1.2
        """
        noise = self.rng.normal(0, 0.5)
        new_speed = current_speed + noise

        # 限制在有效范围
        new_speed = np.clip(new_speed, self.params.speed_min, self.params.speed_max)

        return new_speed

    def _direction_to_heading(self, direction: Direction) -> float:
        """
        方向转航向角

        东向: 0度
        西向: 180度
        """
        if direction == Direction.EASTBOUND:
            return 0.0
        return 180.0

    def generate_single_trajectory(self, vehicle_id: int) -> VehicleTrajectory:
        """
        生成单个车辆的完整轨迹

        Args:
            vehicle_id: 车辆ID

        Returns:
            VehicleTrajectory对象
        """
        num_steps = self.params.num_steps

        # 分配车道 (均匀分布)
        lane_id = vehicle_id % self.params.num_lanes
        direction = self.params.get_lane_direction(lane_id)

        # 初始化数组
        positions = np.zeros((num_steps, 2))
        speeds = np.zeros(num_steps)
        headings = np.zeros(num_steps)
        timestamps = np.arange(num_steps) * self.params.dt

        # 生成初始状态
        positions[0] = self._generate_initial_position(lane_id, direction)
        speeds[0] = self._generate_initial_speed()
        headings[0] = self._direction_to_heading(direction)

        # 逐步生成轨迹
        for t in range(1, num_steps):
            speeds[t] = self._update_speed(speeds[t - 1])
            positions[t] = self._update_position(
                positions[t - 1], speeds[t], direction, self.params.dt
            )
            headings[t] = self._direction_to_heading(direction)

        trajectory = VehicleTrajectory(
            vehicle_id=vehicle_id,
            lane_id=lane_id,
            direction=direction,
            positions=positions,
            speeds=speeds,
            headings=headings,
            timestamps=timestamps,
        )

        return trajectory

    def generate_all_trajectories(self) -> Dict[int, VehicleTrajectory]:
        """
        生成所有车辆的轨迹

        Returns:
            {vehicle_id: VehicleTrajectory}
        """
        logger.info(f"Generating trajectories for {self.params.vehicle_count} vehicles...")

        self.trajectories.clear()

        for vid in range(self.params.vehicle_count):
            self.trajectories[vid] = self.generate_single_trajectory(vid)

        self._generated = True

        logger.info(f"Generated {len(self.trajectories)} trajectories")

        return self.trajectories

    def get_all_positions_at(self, step: int) -> Dict[int, np.ndarray]:
        """
        获取某时间步所有车辆位置

        Args:
            step: 时间步索引

        Returns:
            {vehicle_id: position}
        """
        if not self._generated:
            self.generate_all_trajectories()

        return {vid: traj.get_position_at(step) for vid, traj in self.trajectories.items()}

    def get_all_speeds_at(self, step: int) -> Dict[int, float]:
        """
        获取某时间步所有车辆速度

        Args:
            step: 时间步索引

        Returns:
            {vehicle_id: speed}
        """
        if not self._generated:
            self.generate_all_trajectories()

        return {vid: traj.get_speed_at(step) for vid, traj in self.trajectories.items()}

    def get_vehicles_in_range(self, center: np.ndarray, radius: float, step: int) -> List[int]:
        """
        获取指定范围内的车辆ID

        用于RSU覆盖范围查询

        Args:
            center: 中心点位置
            radius: 半径，对应yaml network.rsu_coverage_radius=400.0
            step: 时间步索引

        Returns:
            范围内的车辆ID列表
        """
        if not self._generated:
            self.generate_all_trajectories()

        in_range = []
        for vid, traj in self.trajectories.items():
            pos = traj.get_position_at(step)
            distance = np.linalg.norm(pos - center)
            if distance <= radius:
                in_range.append(vid)

        return in_range

    def get_trajectory(self, vehicle_id: int) -> Optional[VehicleTrajectory]:
        """获取指定车辆的轨迹"""
        return self.trajectories.get(vehicle_id)

    def get_statistics(self) -> dict:
        """
        获取轨迹统计信息

        Returns:
            统计字典
        """
        if not self._generated:
            return {}

        all_speeds = []
        for traj in self.trajectories.values():
            all_speeds.extend(traj.speeds)

        all_speeds = np.array(all_speeds)

        # 各车道车辆数
        lane_counts = {}
        for traj in self.trajectories.values():
            lane_id = traj.lane_id
            lane_counts[lane_id] = lane_counts.get(lane_id, 0) + 1

        return {
            "vehicle_count": len(self.trajectories),
            "num_steps": self.params.num_steps,
            "speed_mean": float(np.mean(all_speeds)),
            "speed_std": float(np.std(all_speeds)),
            "speed_min": float(np.min(all_speeds)),
            "speed_max": float(np.max(all_speeds)),
            "lane_distribution": lane_counts,
        }


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("轨迹生成器单元测试")
    print("=" * 60)

    # 测试1: 参数加载
    print("\n--- 测试1: 参数加载 ---")
    params = TrajectoryParameters()
    print(f"道路长度: {params.road_length} m (yaml: 3000.0)")
    print(f"车道数: {params.num_lanes} (yaml: 4)")
    print(f"车道宽度: {params.lane_width} m (yaml: 3.5)")
    print(f"车辆数: {params.vehicle_count} (yaml: 50)")
    print(f"平均速度: {params.speed_mean} m/s (yaml: 25.0)")
    print(f"速度范围: [{params.speed_min}, {params.speed_max}] m/s")
    print(f"时间步长: {params.dt} s (yaml: 0.1)")
    print(f"仿真时长: {params.duration} s (yaml: 100.0)")
    print(f"总步数: {params.num_steps} (yaml: 1000)")

    # 测试2: 车道中心线
    print("\n--- 测试2: 车道布局 ---")
    for i, y in enumerate(params.lane_centers):
        direction = params.get_lane_direction(i)
        print(f"车道{i}: y={y:.2f}m, 方向={direction.name}")

    # 测试3: 生成轨迹
    print("\n--- 测试3: 轨迹生成 ---")
    generator = TrajectoryGenerator(params, random_seed=42)
    trajectories = generator.generate_all_trajectories()
    print(f"生成轨迹数: {len(trajectories)}")

    # 测试4: 轨迹统计
    print("\n--- 测试4: 轨迹统计 ---")
    stats = generator.get_statistics()
    print(f"车辆数: {stats['vehicle_count']}")
    print(f"时间步数: {stats['num_steps']}")
    print(f"实际速度均值: {stats['speed_mean']:.2f} m/s (目标: 25.0)")
    print(f"实际速度标准差: {stats['speed_std']:.2f} m/s (目标: 5.0)")
    print(f"速度范围: [{stats['speed_min']:.2f}, {stats['speed_max']:.2f}] m/s")
    print(f"车道分布: {stats['lane_distribution']}")

    # 测试5: 单个轨迹检查
    print("\n--- 测试5: 单个轨迹检查 ---")
    traj = generator.get_trajectory(0)
    if traj is not None:
        print(f"车辆0:")
        print(f"  车道: {traj.lane_id}")
        print(f"  方向: {traj.direction.name}")
        print(f"  初始位置: {traj.positions[0]}")
        print(f"  初始速度: {traj.speeds[0]:.2f} m/s")
        print(f"  末位置: {traj.positions[-1]}")
        print(f"  末速度: {traj.speeds[-1]:.2f} m/s")
    else:
        print("车辆0轨迹未找到")

    # 测试6: 某时刻所有车辆位置
    print("\n--- 测试6: 时刻快照 ---")
    positions = generator.get_all_positions_at(step=500)  # t=50s
    print(f"t=50s时车辆位置 (前5辆):")
    for vid in range(min(5, len(positions))):
        pos = positions[vid]
        print(f"  车辆{vid}: x={pos[0]:.1f}m, y={pos[1]:.1f}m")

    # 测试7: 范围查询
    print("\n--- 测试7: RSU覆盖范围查询 ---")
    rsu_position = np.array([600.0, 7.0])  # RSU位置
    coverage_radius = 400.0  # yaml network.rsu_coverage_radius

    in_range = generator.get_vehicles_in_range(rsu_position, coverage_radius, step=500)
    print(f"RSU位置: {rsu_position}")
    print(f"覆盖半径: {coverage_radius} m")
    print(f"覆盖车辆数: {len(in_range)}")
    print(f"覆盖车辆ID: {in_range[:10]}...")  # 显示前10个

    # 验证约束
    print("\n--- 验证关键约束 ---")

    # 约束1: 速度在有效范围内
    for traj in trajectories.values():
        assert np.all(traj.speeds >= params.speed_min), "速度低于最小值"
        assert np.all(traj.speeds <= params.speed_max), "速度高于最大值"
    print("✓ 所有速度在有效范围内")

    # 约束2: 位置在道路范围内
    for traj in trajectories.values():
        assert np.all(traj.positions[:, 0] >= 0), "x坐标为负"
        assert np.all(traj.positions[:, 0] <= params.road_length), "x超出道路"
    print("✓ 所有位置在道路范围内")

    # 约束3: 车辆均匀分布在各车道
    lane_counts = stats["lane_distribution"]
    expected_per_lane = params.vehicle_count // params.num_lanes
    for lane_id, count in lane_counts.items():
        assert abs(count - expected_per_lane) <= 2, f"车道{lane_id}分布不均"
    print("✓ 车辆均匀分布在各车道")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
