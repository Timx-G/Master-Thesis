"""
V2I信道模型模块
================
基于 3GPP TR 37.885 Urban NLoS 信道模型

参数来源:
- parameters.yaml communication 部分
- docs/01_系统模型.md §3 通信模型

信道模型组成:
1. 大尺度衰落: 路径损耗 + 阴影衰落
2. 小尺度衰落: Rician衰落（可选）
3. Shannon传输速率计算
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChannelParameters:
    """
    信道参数配置类

    所有参数直接对应 parameters.yaml communication 部分
    """

    # === 来自 parameters.yaml communication ===
    carrier_freq: float = 5.9e9  # [FCC/ETSI] 5.9 GHz ITS频段
    bandwidth: float = 20.0e6  # [3GPP TS 36.213] 20 MHz
    tx_power_dbm: float = 23.0  # [FCC Part 95] 200 mW
    noise_psd_dbm_hz: float = -174.0  # [物理常数] 热噪声功率谱密度
    path_loss_exponent: float = 3.5  # [3GPP TR 37.885] Urban NLoS
    shadowing_std: float = 4.0  # [3GPP TR 37.885] 阴影衰落标准差 (dB)
    rician_factor_db: float = 6.0  # [3GPP TR 37.885] K因子

    # === 派生参数 ===
    @property
    def tx_power_watts(self) -> float:
        """发射功率转换: dBm -> Watts"""
        return 10 ** ((self.tx_power_dbm - 30) / 10)

    @property
    def noise_power_watts(self) -> float:
        """
        噪声功率计算

        公式: N = N0 * B
        - N0: 噪声功率谱密度 (W/Hz)
        - B: 带宽 (Hz)

        来源: docs/01_系统模型.md §3.2
        """
        noise_psd_watts_hz = 10 ** ((self.noise_psd_dbm_hz - 30) / 10)
        return noise_psd_watts_hz * self.bandwidth

    @property
    def rician_factor_linear(self) -> float:
        """Rician K因子转换: dB -> 线性"""
        return 10 ** (self.rician_factor_db / 10)


@dataclass
class ChannelState:
    """
    信道状态（单个车辆-RSU链路）

    记录某一时刻的信道条件
    """

    vehicle_id: int
    rsu_id: int
    distance: float  # 车辆到RSU距离 (m)
    path_loss_db: float  # 路径损耗 (dB)
    shadowing_db: float  # 阴影衰落 (dB)
    channel_gain_db: float  # 总信道增益 (dB)
    snr_db: float  # 信噪比 (dB)
    transmission_rate: float  # 传输速率 (bps)
    timestamp: float = 0.0  # 时间戳


class V2IChannel:
    """
    V2I信道模型实现

    基于 3GPP TR 37.885 Urban NLoS 模型

    主要功能:
    1. 路径损耗计算
    2. 阴影衰落生成
    3. 信道增益计算
    4. Shannon传输速率计算

    使用示例:
    ```python
    params = ChannelParameters()  # 使用yaml默认值
    channel = V2IChannel(params)

    # 计算100米距离的传输速率
    rate = channel.calculate_transmission_rate(distance=100.0)
    ```
    """

    # === 3GPP TR 37.885 路径损耗模型常数 ===
    # Urban NLoS: PL = A*log10(d) + B + C*log10(fc/1e9)
    PL_CONSTANT_A: float = 36.7  # 距离系数
    PL_CONSTANT_B: float = 22.7  # 常数项
    PL_CONSTANT_C: float = 26.0  # 频率系数

    # 最小参考距离 (避免log(0))
    MIN_DISTANCE: float = 1.0  # 1米

    def __init__(
        self, params: Optional[ChannelParameters] = None, random_seed: Optional[int] = None
    ):
        """
        初始化V2I信道

        Args:
            params: 信道参数，None则使用默认值(对应yaml配置)
            random_seed: 随机种子，对应yaml simulation.random_seed
        """
        self.params = params or ChannelParameters()
        self.rng = np.random.default_rng(random_seed)

        # 预计算常用值
        self._precompute_constants()

        logger.info(
            f"V2IChannel initialized: "
            f"fc={self.params.carrier_freq/1e9:.1f}GHz, "
            f"B={self.params.bandwidth/1e6:.0f}MHz, "
            f"Pt={self.params.tx_power_dbm}dBm"
        )

    def _precompute_constants(self) -> None:
        """预计算频率相关的路径损耗常数"""
        # 频率项: C * log10(fc / 1e9)
        fc_ghz = self.params.carrier_freq / 1e9
        self._pl_freq_term = self.PL_CONSTANT_C * np.log10(fc_ghz)

        # 完整路径损耗公式的常数部分: B + C*log10(fc)
        self._pl_constant_part = self.PL_CONSTANT_B + self._pl_freq_term

    def calculate_path_loss(self, distance: float) -> float:
        """
        计算路径损耗

        公式 (3GPP TR 37.885 Urban NLoS):
            PL(d) = 36.7*log10(d) + 22.7 + 26*log10(fc/1GHz)

        Args:
            distance: 车辆到RSU的距离 (m)

        Returns:
            路径损耗 (dB)，正值表示损耗

        来源: docs/01_系统模型.md §3.1
        """
        # 确保最小距离
        d = max(distance, self.MIN_DISTANCE)

        # PL = A*log10(d) + B + C*log10(fc)
        path_loss_db = self.PL_CONSTANT_A * np.log10(d) + self._pl_constant_part

        return path_loss_db

    def generate_shadowing(self) -> float:
        """
        生成阴影衰落

        模型: 对数正态分布，标准差 σ_s

        公式: X_s ~ N(0, σ_s^2)

        参数: shadowing_std = 4.0 dB [3GPP TR 37.885]

        Returns:
            阴影衰落值 (dB)

        来源: docs/01_系统模型.md §3.1
        """
        return self.rng.normal(0, self.params.shadowing_std)

    def generate_rician_fading(self) -> float:
        """
        生成Rician小尺度衰落（可选）

        模型: Rician分布，K因子决定LoS分量强度

        参数: rician_factor_db = 6.0 dB [3GPP TR 37.885]

        Returns:
            小尺度衰落功率增益 (线性值)

        来源: docs/01_系统模型.md §3.1
        注意: 基础仿真可忽略小尺度衰落，专注大尺度衰落
        """
        K = self.params.rician_factor_linear

        # Rician分布生成
        # H = sqrt(K/(K+1)) + sqrt(1/(K+1)) * CN(0,1)
        sigma = np.sqrt(1 / (2 * (K + 1)))
        mean = np.sqrt(K / (K + 1))

        # 复高斯分量
        h_real = self.rng.normal(mean, sigma)
        h_imag = self.rng.normal(0, sigma)

        # 功率增益
        power_gain = h_real**2 + h_imag**2

        return power_gain

    def calculate_channel_gain(
        self, distance: float, include_shadowing: bool = True, include_small_scale: bool = False
    ) -> Tuple[float, float, float]:
        """
        计算总信道增益

        公式: h = -PL - X_s (dB)

        Args:
            distance: 距离 (m)
            include_shadowing: 是否包含阴影衰落
            include_small_scale: 是否包含小尺度衰落

        Returns:
            (信道增益dB, 路径损耗dB, 阴影衰落dB)

        来源: docs/01_系统模型.md §3.1
        """
        # 路径损耗
        path_loss_db = self.calculate_path_loss(distance)

        # 阴影衰落
        shadowing_db = self.generate_shadowing() if include_shadowing else 0.0

        # 信道增益 (dB) = -路径损耗 - 阴影衰落
        channel_gain_db = -path_loss_db - shadowing_db

        # 小尺度衰落（可选）
        if include_small_scale:
            small_scale_gain = self.generate_rician_fading()
            channel_gain_db += 10 * np.log10(small_scale_gain)

        return channel_gain_db, path_loss_db, shadowing_db

    def calculate_snr(self, distance: float, include_shadowing: bool = True) -> Tuple[float, float]:
        """
        计算信噪比

        公式: SNR = Pt * |h|^2 / N
              SNR_dB = Pt_dBm - 30 + h_dB - N_dBm

        Args:
            distance: 距离 (m)
            include_shadowing: 是否包含阴影衰落

        Returns:
            (SNR线性值, SNR_dB)

        来源: docs/01_系统模型.md §3.2
        """
        # 信道增益
        channel_gain_db, _, _ = self.calculate_channel_gain(
            distance, include_shadowing, include_small_scale=False
        )

        # 噪声功率 (dBm)
        noise_power_dbm = 10 * np.log10(self.params.noise_power_watts) + 30

        # SNR (dB) = Pt (dBm) + h (dB) - N (dBm)
        snr_db = self.params.tx_power_dbm + channel_gain_db - noise_power_dbm

        # 转换为线性值
        snr_linear = 10 ** (snr_db / 10)

        return snr_linear, snr_db

    def calculate_transmission_rate(
        self,
        distance: float,
        allocated_bandwidth: Optional[float] = None,
        include_shadowing: bool = True,
    ) -> float:
        """
        计算Shannon传输速率

        公式: R = B * log2(1 + SNR)

        Args:
            distance: 距离 (m)
            allocated_bandwidth: 分配带宽 (Hz)，None则使用总带宽
            include_shadowing: 是否包含阴影衰落

        Returns:
            传输速率 (bps)

        来源: docs/01_系统模型.md §3.2
        """
        # 使用分配带宽或总带宽
        bandwidth = allocated_bandwidth or self.params.bandwidth

        # 计算SNR
        snr_linear, _ = self.calculate_snr(distance, include_shadowing)

        # Shannon公式
        # R = B * log2(1 + SNR)
        rate = bandwidth * np.log2(1 + snr_linear)

        return rate

    def calculate_transmission_time(
        self,
        distance: float,
        data_size: float,
        allocated_bandwidth: Optional[float] = None,
        include_shadowing: bool = True,
    ) -> float:
        """
        计算传输时间

        公式: T_tx = D / R

        Args:
            distance: 距离 (m)
            data_size: 数据大小 (bits)
            allocated_bandwidth: 分配带宽 (Hz)
            include_shadowing: 是否包含阴影衰落

        Returns:
            传输时间 (s)

        来源: docs/01_系统模型.md §3.3
        """
        rate = self.calculate_transmission_rate(distance, allocated_bandwidth, include_shadowing)

        # 避免除以零
        if rate <= 0:
            return float("inf")

        return data_size / rate

    def get_channel_state(
        self,
        vehicle_id: int,
        rsu_id: int,
        distance: float,
        timestamp: float = 0.0,
        allocated_bandwidth: Optional[float] = None,
    ) -> ChannelState:
        """
        获取完整信道状态

        Args:
            vehicle_id: 车辆ID
            rsu_id: RSU ID
            distance: 距离 (m)
            timestamp: 时间戳
            allocated_bandwidth: 分配带宽

        Returns:
            ChannelState对象
        """
        # 计算各组件
        channel_gain_db, path_loss_db, shadowing_db = self.calculate_channel_gain(
            distance, include_shadowing=True
        )

        snr_linear, snr_db = self.calculate_snr(distance, include_shadowing=True)

        rate = self.calculate_transmission_rate(
            distance, allocated_bandwidth, include_shadowing=True
        )

        return ChannelState(
            vehicle_id=vehicle_id,
            rsu_id=rsu_id,
            distance=distance,
            path_loss_db=path_loss_db,
            shadowing_db=shadowing_db,
            channel_gain_db=channel_gain_db,
            snr_db=snr_db,
            transmission_rate=rate,
            timestamp=timestamp,
        )


class ChannelManager:
    """
    信道管理器

    管理多车辆-RSU链路的信道状态

    功能:
    1. 维护所有活动链路的信道状态
    2. 批量更新信道状态
    3. 提供信道状态查询接口
    """

    def __init__(
        self, params: Optional[ChannelParameters] = None, random_seed: Optional[int] = None
    ):
        """
        初始化信道管理器

        Args:
            params: 信道参数
            random_seed: 随机种子，对应yaml simulation.random_seed=42
        """
        self.channel = V2IChannel(params, random_seed)
        self.params = self.channel.params

        # 存储各链路的信道状态
        # key: (vehicle_id, rsu_id), value: ChannelState
        self._channel_states: dict[Tuple[int, int], ChannelState] = {}

        # 统计信息
        self._stats = {"total_updates": 0, "avg_snr_db": 0.0, "avg_rate_mbps": 0.0}

    def update_channel_state(
        self,
        vehicle_id: int,
        rsu_id: int,
        distance: float,
        timestamp: float,
        allocated_bandwidth: Optional[float] = None,
    ) -> ChannelState:
        """
        更新单个链路的信道状态

        Args:
            vehicle_id: 车辆ID
            rsu_id: RSU ID
            distance: 距离
            timestamp: 时间戳
            allocated_bandwidth: 分配带宽

        Returns:
            更新后的ChannelState
        """
        state = self.channel.get_channel_state(
            vehicle_id, rsu_id, distance, timestamp, allocated_bandwidth
        )

        self._channel_states[(vehicle_id, rsu_id)] = state
        self._stats["total_updates"] += 1

        return state

    def batch_update(
        self,
        vehicle_positions: dict[int, np.ndarray],
        rsu_positions: dict[int, np.ndarray],
        coverage_radius: float,
        timestamp: float,
    ) -> list[ChannelState]:
        """
        批量更新所有覆盖范围内链路的信道状态

        Args:
            vehicle_positions: {vehicle_id: position}
            rsu_positions: {rsu_id: position}
            coverage_radius: RSU覆盖半径，对应yaml network.rsu_coverage_radius=400.0
            timestamp: 当前时间戳

        Returns:
            更新的ChannelState列表
        """
        updated_states = []

        for vid, v_pos in vehicle_positions.items():
            for rid, r_pos in rsu_positions.items():
                # 计算距离
                distance = np.linalg.norm(v_pos - r_pos)

                # 仅更新覆盖范围内的链路
                if distance <= coverage_radius:
                    state = self.update_channel_state(vid, rid, float(distance), timestamp)
                    updated_states.append(state)

        # 更新统计
        if updated_states:
            self._stats["avg_snr_db"] = np.mean([s.snr_db for s in updated_states])
            self._stats["avg_rate_mbps"] = np.mean(
                [s.transmission_rate / 1e6 for s in updated_states]
            )

        return updated_states

    def get_channel_state(self, vehicle_id: int, rsu_id: int) -> Optional[ChannelState]:
        """获取指定链路的信道状态"""
        return self._channel_states.get((vehicle_id, rsu_id))

    def get_best_rsu_for_vehicle(self, vehicle_id: int) -> Optional[Tuple[int, ChannelState]]:
        """
        获取某车辆的最佳RSU（最高传输速率）

        Args:
            vehicle_id: 车辆ID

        Returns:
            (rsu_id, ChannelState) 或 None
        """
        best_rsu = None
        best_rate = 0.0

        for (vid, rid), state in self._channel_states.items():
            if vid == vehicle_id and state.transmission_rate > best_rate:
                best_rate = state.transmission_rate
                best_rsu = (rid, state)

        return best_rsu

    def get_statistics(self) -> dict:
        """获取信道统计信息"""
        return self._stats.copy()

    def clear(self) -> None:
        """清空所有信道状态"""
        self._channel_states.clear()


# ==================== 单元测试 ====================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("V2I信道模型单元测试")
    print("=" * 60)

    # 测试1: 参数加载
    print("\n--- 测试1: 参数加载 ---")
    params = ChannelParameters()
    print(f"载波频率: {params.carrier_freq/1e9} GHz (yaml: 5.9)")
    print(f"带宽: {params.bandwidth/1e6} MHz (yaml: 20.0)")
    print(f"发射功率: {params.tx_power_dbm} dBm (yaml: 23.0)")
    print(f"噪声PSD: {params.noise_psd_dbm_hz} dBm/Hz (yaml: -174.0)")
    print(f"路径损耗指数: {params.path_loss_exponent} (yaml: 3.5)")
    print(f"阴影衰落std: {params.shadowing_std} dB (yaml: 4.0)")
    print(f"发射功率(W): {params.tx_power_watts:.4f}")
    print(f"噪声功率(W): {params.noise_power_watts:.2e}")

    # 测试2: 路径损耗计算
    print("\n--- 测试2: 路径损耗计算 (3GPP TR 37.885) ---")
    channel = V2IChannel(params, random_seed=42)

    distances = [10, 50, 100, 200, 400]
    print(f"{'距离(m)':<12} {'路径损耗(dB)':<15}")
    print("-" * 30)
    for d in distances:
        pl = channel.calculate_path_loss(d)
        print(f"{d:<12} {pl:<15.2f}")

    # 测试3: SNR计算
    print("\n--- 测试3: SNR计算 ---")
    print(f"{'距离(m)':<12} {'SNR(dB)':<15}")
    print("-" * 30)
    for d in distances:
        _, snr_db = channel.calculate_snr(d, include_shadowing=False)
        print(f"{d:<12} {snr_db:<15.2f}")

    # 测试4: 传输速率计算
    print("\n--- 测试4: Shannon传输速率 ---")
    print(f"{'距离(m)':<12} {'速率(Mbps)':<15} {'传输500KB(ms)':<15}")
    print("-" * 45)

    data_size = 4.0e6  # 500 KB = 4 Mbits (对应yaml cpm.data_base)

    for d in distances:
        rate = channel.calculate_transmission_rate(d, include_shadowing=False)
        tx_time = channel.calculate_transmission_time(d, data_size, include_shadowing=False)
        print(f"{d:<12} {rate/1e6:<15.2f} {tx_time*1000:<15.2f}")

    # 测试5: 完整信道状态
    print("\n--- 测试5: 完整信道状态 ---")
    state = channel.get_channel_state(vehicle_id=0, rsu_id=0, distance=100.0, timestamp=0.0)
    print(f"车辆ID: {state.vehicle_id}")
    print(f"RSU ID: {state.rsu_id}")
    print(f"距离: {state.distance} m")
    print(f"路径损耗: {state.path_loss_db:.2f} dB")
    print(f"阴影衰落: {state.shadowing_db:.2f} dB")
    print(f"信道增益: {state.channel_gain_db:.2f} dB")
    print(f"SNR: {state.snr_db:.2f} dB")
    print(f"传输速率: {state.transmission_rate/1e6:.2f} Mbps")

    # 测试6: 信道管理器
    print("\n--- 测试6: 信道管理器批量更新 ---")
    manager = ChannelManager(params, random_seed=42)

    # 模拟车辆和RSU位置
    vehicle_positions = {
        0: np.array([100.0, 0.0]),
        1: np.array([200.0, 3.5]),
        2: np.array([350.0, 7.0]),
    }
    rsu_positions = {0: np.array([0.0, 0.0]), 1: np.array([600.0, 0.0])}

    states = manager.batch_update(
        vehicle_positions,
        rsu_positions,
        coverage_radius=400.0,  # yaml network.rsu_coverage_radius
        timestamp=0.0,
    )

    print(f"更新链路数: {len(states)}")
    stats = manager.get_statistics()
    print(f"平均SNR: {stats['avg_snr_db']:.2f} dB")
    print(f"平均速率: {stats['avg_rate_mbps']:.2f} Mbps")

    # 测试7: 获取最佳RSU
    print("\n--- 测试7: 车辆最佳RSU选择 ---")
    for vid in vehicle_positions.keys():
        best = manager.get_best_rsu_for_vehicle(vid)
        if best:
            rid, state = best
            print(f"车辆{vid}: 最佳RSU={rid}, 速率={state.transmission_rate/1e6:.2f}Mbps")
        else:
            print(f"车辆{vid}: 无可用RSU")

    # 验证关键约束
    print("\n--- 验证关键约束 ---")

    # 约束1: 100m距离应能在80ms内传输500KB
    rate_100m = channel.calculate_transmission_rate(100, include_shadowing=False)
    tx_time_100m = 4.0e6 / rate_100m
    deadline = 0.08  # yaml cpm.deadline

    print(f"100m处传输500KB: {tx_time_100m*1000:.2f}ms (截止时间: {deadline*1000}ms)")
    assert tx_time_100m < deadline, "100m处应能满足80ms截止时间"
    print("✓ 100m距离满足截止时间约束")

    # 约束2: 覆盖边缘(400m)应有合理速率
    rate_400m = channel.calculate_transmission_rate(400, include_shadowing=False)
    print(f"400m边缘速率: {rate_400m/1e6:.2f} Mbps")
    assert rate_400m > 1e6, "覆盖边缘应有>1Mbps速率"
    print("✓ 覆盖边缘有合理速率")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
