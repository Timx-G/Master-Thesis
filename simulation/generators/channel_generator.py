"""
信道状态生成模块
================
批量生成和管理V2I信道状态

参数来源:
- parameters.yaml communication 部分
- parameters.yaml network 部分

功能:
1. 为所有车辆-RSU对生成信道状态
2. 支持时间相关的信道演化
3. 提供信道状态查询接口

来源: docs/11_仿真数据生成方案.md §3
"""

import logging

# 导入依赖模块
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.channel import ChannelManager, ChannelParameters, ChannelState, V2IChannel

logger = logging.getLogger(__name__)


@dataclass
class ChannelGeneratorConfig:
    """
    信道生成器配置

    整合 parameters.yaml 中的相关参数
    """

    # === 来自 parameters.yaml communication ===
    carrier_freq: float = 5.9e9
    bandwidth: float = 20.0e6
    tx_power_dbm: float = 23.0
    noise_psd_dbm_hz: float = -174.0
    path_loss_exponent: float = 3.5
    shadowing_std: float = 4.0
    rician_factor_db: float = 6.0

    # === 来自 parameters.yaml network ===
    rsu_coverage_radius: float = 400.0
    rsu_count: int = 5
    rsu_spacing: float = 600.0
    road_length: float = 3000.0

    # === 来自 parameters.yaml simulation ===
    random_seed: int = 42

    # === 信道演化参数 ===
    shadowing_correlation_distance: float = 50.0  # 阴影衰落相关距离 (m)
    shadowing_correlation_time: float = 1.0  # 阴影衰落相关时间 (s)
    update_small_scale: bool = False  # 是否更新小尺度衰落


class ChannelStateGenerator:
    """
    信道状态生成器

    为仿真生成和管理所有V2I链路的信道状态

    特点:
    1. 基于3GPP TR 37.885信道模型
    2. 支持空间相关的阴影衰落
    3. 批量生成和查询

    使用示例:
    ```python
    config = ChannelGeneratorConfig()
    generator = ChannelStateGenerator(config)

    # 生成RSU位置
    rsu_positions = generator.generate_rsu_positions()

    # 更新所有信道状态
    states = generator.update_all_channels(
        vehicle_positions={...},
        current_time=10.0
    )
    ```
    """

    def __init__(
        self, config: Optional[ChannelGeneratorConfig] = None, random_seed: Optional[int] = None
    ):
        """
        初始化信道状态生成器

        Args:
            config: 配置参数
            random_seed: 随机种子
        """
        self.config = config or ChannelGeneratorConfig()

        seed = random_seed if random_seed is not None else self.config.random_seed
        self.rng = np.random.default_rng(seed)

        # 创建信道参数
        channel_params = ChannelParameters(
            carrier_freq=self.config.carrier_freq,
            bandwidth=self.config.bandwidth,
            tx_power_dbm=self.config.tx_power_dbm,
            noise_psd_dbm_hz=self.config.noise_psd_dbm_hz,
            path_loss_exponent=self.config.path_loss_exponent,
            shadowing_std=self.config.shadowing_std,
            rician_factor_db=self.config.rician_factor_db,
        )

        # 创建信道管理器
        self.channel_manager = ChannelManager(channel_params, seed)

        # RSU位置
        self._rsu_positions: Dict[int, np.ndarray] = {}

        # 缓存的阴影衰落值 (用于空间相关性)
        # key: (vehicle_id, rsu_id), value: (last_position, last_shadowing)
        self._shadowing_cache: Dict[Tuple[int, int], Tuple[np.ndarray, float]] = {}

        # 统计信息
        self._stats = {
            "total_updates": 0,
            "active_links": 0,
            "avg_rate_mbps": 0.0,
            "coverage_ratio": 0.0,
        }

        logger.info(
            f"ChannelStateGenerator initialized: "
            f"{self.config.rsu_count} RSUs, "
            f"coverage={self.config.rsu_coverage_radius}m"
        )

    def generate_rsu_positions(self) -> Dict[int, np.ndarray]:
        """
        生成RSU位置

        布局: 沿道路中线均匀分布

        参数:
        - rsu_count: 5 (yaml network.rsu_count)
        - rsu_spacing: 600m (yaml network.rsu_spacing)
        - 第一个RSU位于 x = rsu_spacing / 2 = 300m

        Returns:
            {rsu_id: position}
        """
        self._rsu_positions.clear()

        # 道路中线y坐标
        road_center_y = (4 * self.config.rsu_coverage_radius) / 2  # 约7m
        road_center_y = 7.0  # 简化为固定值

        for i in range(self.config.rsu_count):
            x = self.config.rsu_spacing / 2 + i * self.config.rsu_spacing
            # 确保不超出道路
            if x > self.config.road_length:
                x = self.config.road_length - 50

            self._rsu_positions[i] = np.array([x, road_center_y])

        logger.info(f"Generated {len(self._rsu_positions)} RSU positions")

        return self._rsu_positions.copy()

    def get_rsu_positions(self) -> Dict[int, np.ndarray]:
        """获取RSU位置"""
        if not self._rsu_positions:
            return self.generate_rsu_positions()
        return self._rsu_positions.copy()

    def _generate_correlated_shadowing(
        self, vehicle_id: int, rsu_id: int, current_position: np.ndarray
    ) -> float:
        """
        生成空间相关的阴影衰落

        模型: 指数相关模型

        ρ(Δd) = exp(-Δd / d_corr)

        X_new = ρ * X_old + sqrt(1-ρ²) * N(0, σ²)

        参数:
        - shadowing_std: 4.0 dB (yaml communication.shadowing_std)
        - shadowing_correlation_distance: 50m
        """
        cache_key = (vehicle_id, rsu_id)

        if cache_key in self._shadowing_cache:
            last_pos, last_shadowing = self._shadowing_cache[cache_key]

            # 计算位移
            displacement = np.linalg.norm(current_position - last_pos)

            # 相关系数
            rho = np.exp(-displacement / self.config.shadowing_correlation_distance)

            # 生成新的阴影衰落
            innovation = self.rng.normal(0, self.config.shadowing_std)
            new_shadowing = rho * last_shadowing + np.sqrt(1 - rho**2) * innovation
        else:
            # 首次生成
            new_shadowing = self.rng.normal(0, self.config.shadowing_std)

        # 更新缓存
        self._shadowing_cache[cache_key] = (current_position.copy(), new_shadowing)

        return new_shadowing

    def update_all_channels(
        self, vehicle_positions: Dict[int, np.ndarray], current_time: float
    ) -> List[ChannelState]:
        """
        更新所有车辆-RSU链路的信道状态

        Args:
            vehicle_positions: {vehicle_id: position}
            current_time: 当前时间

        Returns:
            更新的ChannelState列表
        """
        if not self._rsu_positions:
            self.generate_rsu_positions()

        updated_states = []
        vehicles_covered = set()

        for vid, v_pos in vehicle_positions.items():
            for rid, r_pos in self._rsu_positions.items():
                # 计算距离
                distance = np.linalg.norm(v_pos - r_pos)

                # 仅处理覆盖范围内的链路
                if distance <= self.config.rsu_coverage_radius:
                    vehicles_covered.add(vid)

                    # 更新信道状态
                    state = self.channel_manager.update_channel_state(
                        vid, rid, float(distance), current_time
                    )
                    updated_states.append(state)

        # 更新统计
        self._stats["total_updates"] += 1
        self._stats["active_links"] = len(updated_states)
        self._stats["coverage_ratio"] = (
            len(vehicles_covered) / len(vehicle_positions) if vehicle_positions else 0
        )

        if updated_states:
            self._stats["avg_rate_mbps"] = np.mean(
                [s.transmission_rate / 1e6 for s in updated_states]
            )

        return updated_states

    def get_channel_state(self, vehicle_id: int, rsu_id: int) -> Optional[ChannelState]:
        """获取指定链路的信道状态"""
        return self.channel_manager.get_channel_state(vehicle_id, rsu_id)

    def get_best_rsu(self, vehicle_id: int) -> Optional[Tuple[int, ChannelState]]:
        """
        获取车辆的最佳RSU

        选择标准: 最高传输速率
        """
        return self.channel_manager.get_best_rsu_for_vehicle(vehicle_id)

    def get_vehicles_in_coverage(
        self, rsu_id: int, vehicle_positions: Dict[int, np.ndarray]
    ) -> List[int]:
        """
        获取RSU覆盖范围内的车辆

        Args:
            rsu_id: RSU ID
            vehicle_positions: {vehicle_id: position}

        Returns:
            车辆ID列表
        """
        if rsu_id not in self._rsu_positions:
            return []

        rsu_pos = self._rsu_positions[rsu_id]
        in_coverage = []

        for vid, v_pos in vehicle_positions.items():
            distance = np.linalg.norm(v_pos - rsu_pos)
            if distance <= self.config.rsu_coverage_radius:
                in_coverage.append(vid)

        return in_coverage

    def get_transmission_time(self, vehicle_id: int, rsu_id: int, data_size: float) -> float:
        """
        计算传输时间

        Args:
            vehicle_id: 车辆ID
            rsu_id: RSU ID
            data_size: 数据大小 (bits)

        Returns:
            传输时间 (s)，无链路返回inf
        """
        state = self.get_channel_state(vehicle_id, rsu_id)
        if state is None or state.transmission_rate <= 0:
            return float("inf")

        return data_size / state.transmission_rate

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return self._stats.copy()

    def reset(self) -> None:
        """重置生成器状态"""
        self._shadowing_cache.clear()
        self.channel_manager.clear()
        self._stats = {
            "total_updates": 0,
            "active_links": 0,
            "avg_rate_mbps": 0.0,
            "coverage_ratio": 0.0,
        }


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("信道状态生成器单元测试")
    print("=" * 60)

    # 测试1: 配置加载
    print("\n--- 测试1: 配置加载 ---")
    config = ChannelGeneratorConfig()
    print(f"载波频率: {config.carrier_freq/1e9} GHz (yaml: 5.9)")
    print(f"带宽: {config.bandwidth/1e6} MHz (yaml: 20.0)")
    print(f"RSU数量: {config.rsu_count} (yaml: 5)")
    print(f"RSU间距: {config.rsu_spacing} m (yaml: 600.0)")
    print(f"覆盖半径: {config.rsu_coverage_radius} m (yaml: 400.0)")

    # 测试2: RSU位置生成
    print("\n--- 测试2: RSU位置生成 ---")
    generator = ChannelStateGenerator(config, random_seed=42)
    rsu_positions = generator.generate_rsu_positions()

    for rid, pos in rsu_positions.items():
        print(f"RSU {rid}: x={pos[0]:.0f}m, y={pos[1]:.1f}m")

    # 测试3: 信道状态更新
    print("\n--- 测试3: 信道状态更新 ---")

    # 模拟车辆位置
    vehicle_positions = {
        0: np.array([100.0, 1.75]),  # 靠近RSU0
        1: np.array([350.0, 5.25]),  # RSU0覆盖边缘
        2: np.array([600.0, 8.75]),  # 靠近RSU1
        3: np.array([1500.0, 12.25]),  # RSU2附近
        4: np.array([2800.0, 1.75]),  # RSU4附近
    }

    states = generator.update_all_channels(vehicle_positions, current_time=0.0)

    print(f"更新链路数: {len(states)}")
    print(f"\n{'车辆':<6} {'RSU':<6} {'距离(m)':<10} {'SNR(dB)':<10} {'速率(Mbps)':<12}")
    print("-" * 50)

    for state in states[:10]:  # 显示前10个
        print(
            f"{state.vehicle_id:<6} {state.rsu_id:<6} "
            f"{state.distance:<10.1f} {state.snr_db:<10.1f} "
            f"{state.transmission_rate/1e6:<12.2f}"
        )

    # 测试4: 最佳RSU选择
    print("\n--- 测试4: 最佳RSU选择 ---")
    for vid in vehicle_positions.keys():
        best = generator.get_best_rsu(vid)
        if best:
            rid, state = best
            print(
                f"车辆{vid}: 最佳RSU={rid}, "
                f"距离={state.distance:.0f}m, "
                f"速率={state.transmission_rate/1e6:.2f}Mbps"
            )
        else:
            print(f"车辆{vid}: 无覆盖")

    # 测试5: RSU覆盖查询
    print("\n--- 测试5: RSU覆盖查询 ---")
    for rid in rsu_positions.keys():
        covered = generator.get_vehicles_in_coverage(rid, vehicle_positions)
        print(f"RSU {rid} 覆盖车辆: {covered}")

    # 测试6: 传输时间计算
    print("\n--- 测试6: 传输时间计算 ---")
    data_size = 4.0e6  # 500KB = 4Mbits (yaml cpm.data_base)

    for vid in vehicle_positions.keys():
        best = generator.get_best_rsu(vid)
        if best:
            rid, _ = best
            tx_time = generator.get_transmission_time(vid, rid, data_size)
            print(f"车辆{vid} -> RSU{rid}: 传输500KB需要 {tx_time*1000:.2f}ms")

    # 测试7: 大规模仿真
    print("\n--- 测试7: 大规模仿真 (50车辆, 1000步) ---")

    # 重置
    generator.reset()
    generator.generate_rsu_positions()

    # 生成50辆车的随机位置
    num_vehicles = 50
    num_steps = 1000

    total_links = 0
    total_rate = 0.0
    coverage_sum = 0.0

    for step in range(num_steps):
        # 随机位置 (简化)
        positions = {
            v: np.array([np.random.uniform(0, 3000), np.random.choice([1.75, 5.25, 8.75, 12.25])])
            for v in range(num_vehicles)
        }

        states = generator.update_all_channels(positions, current_time=step * 0.1)

        total_links += len(states)
        if states:
            total_rate += np.mean([s.transmission_rate for s in states])

        stats = generator.get_statistics()
        coverage_sum += stats["coverage_ratio"]

    print(f"平均活动链路数: {total_links / num_steps:.1f}")
    print(f"平均传输速率: {total_rate / num_steps / 1e6:.2f} Mbps")
    print(f"平均覆盖率: {coverage_sum / num_steps * 100:.1f}%")

    # 验证约束
    print("\n--- 验证关键约束 ---")

    # 约束1: RSU间距正确
    for i in range(len(rsu_positions) - 1):
        spacing = rsu_positions[i + 1][0] - rsu_positions[i][0]
        assert abs(spacing - config.rsu_spacing) < 1, f"RSU间距不正确"
    print(f"✓ RSU间距为{config.rsu_spacing}m")

    # 约束2: 覆盖重叠 (相邻RSU覆盖有交集)
    for i in range(len(rsu_positions) - 1):
        gap = rsu_positions[i + 1][0] - rsu_positions[i][0]
        overlap = 2 * config.rsu_coverage_radius - gap
        assert overlap > 0, "RSU覆盖无重叠"
    print(f"✓ RSU覆盖有重叠 (重叠区域约{overlap:.0f}m)")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
