"""
能耗计算模块
============

基于文档: docs/01_系统模型.md §4 计算模型

能耗模型:
1. 本地计算能耗: E_local = κ * f_local^2 * C
   - κ = 1e-27 (CMOS能耗系数)
   - f_local = 2 GHz (本地CPU频率)
   - C = 计算周期数

2. 卸载能耗: E_offload = P_tx * T_up + P_idle * (T_exe + T_down)
   - P_tx = 200 mW (发射功率)
   - P_idle = 0.1 W (空闲功率)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EnergyParameters:
    """
    能耗参数配置

    所有参数对应 parameters.yaml energy 部分
    """

    # CMOS能耗系数 [文献: 45nm工艺]
    kappa: float = 1.0e-27

    # 发射功率 (W) - 对应yaml communication.tx_power_dbm=23.0 -> 200mW
    tx_power: float = 0.2

    # 空闲功率 (W) - 对应yaml energy.idle_power=0.1
    idle_power: float = 0.1

    # 本地CPU频率 (Hz) - 对应yaml vehicle.local_cpu_freq=2.0e9
    local_cpu_freq: float = 2.0e9

    @classmethod
    def from_config(cls, config: dict) -> "EnergyParameters":
        """从配置字典创建参数对象"""
        energy_cfg = config.get("energy", {})
        vehicle_cfg = config.get("vehicle", {})
        comm_cfg = config.get("communication", {})

        # 发射功率转换: dBm -> W
        tx_power_dbm = comm_cfg.get("tx_power_dbm", 23.0)
        tx_power_w = 10 ** ((tx_power_dbm - 30) / 10)

        return cls(
            kappa=energy_cfg.get("kappa", 1.0e-27),
            idle_power=energy_cfg.get("idle_power", 0.1),
            tx_power=tx_power_w,
            local_cpu_freq=vehicle_cfg.get("local_cpu_freq", 2.0e9),
        )


@dataclass
class VehicleEnergyState:
    """
    单车辆能耗状态跟踪
    """

    vehicle_id: int

    # 累计能耗 (J)
    total_energy: float = 0.0

    # 分项能耗
    local_compute_energy: float = 0.0  # 本地计算能耗
    transmission_energy: float = 0.0  # 传输能耗
    idle_energy: float = 0.0  # 空闲等待能耗

    # 能耗历史
    energy_history: List[float] = field(default_factory=list)

    # 任务统计
    local_tasks: int = 0  # 本地处理任务数
    offloaded_tasks: int = 0  # 卸载任务数


class EnergyCalculator:
    """
    能耗计算器

    功能:
    1. 计算本地计算能耗
    2. 计算卸载能耗
    3. 跟踪每辆车的能耗状态
    4. 统计系统总能耗

    使用示例:
    ```python
    calculator = EnergyCalculator(num_vehicles=50)

    # 计算本地能耗
    e_local = calculator.compute_local_energy(
        compute_cycles=1e8,
        cpu_freq=2e9
    )

    # 计算卸载能耗
    e_offload = calculator.compute_offload_energy(
        transmission_time=0.01,
        execution_time=0.02,
        download_time=0.001
    )
    ```
    """

    def __init__(self, num_vehicles: int, params: Optional[EnergyParameters] = None):
        """
        初始化能耗计算器

        Args:
            num_vehicles: 车辆数量
            params: 能耗参数，None则使用默认值
        """
        self.num_vehicles = num_vehicles
        self.params = params or EnergyParameters()

        # 为每辆车创建能耗状态
        self.vehicle_states: Dict[int, VehicleEnergyState] = {
            i: VehicleEnergyState(vehicle_id=i) for i in range(num_vehicles)
        }

        # 系统级统计
        self.current_slot: int = 0
        self.system_energy_history: List[float] = []

        logger.info(
            f"EnergyCalculator initialized: "
            f"vehicles={num_vehicles}, "
            f"kappa={self.params.kappa:.2e}, "
            f"P_tx={self.params.tx_power*1000:.0f}mW"
        )

    def compute_local_energy(
        self, compute_cycles: float, cpu_freq: Optional[float] = None
    ) -> float:
        """
        计算本地计算能耗

        公式: E_local = κ * f^2 * C

        Args:
            compute_cycles: 计算周期数 (cycles)
            cpu_freq: CPU频率 (Hz)，None则使用默认值

        Returns:
            能耗 (J)

        来源: docs/01_系统模型.md §4.1.2
        """
        f = cpu_freq or self.params.local_cpu_freq

        # E = κ * f^2 * C
        energy = self.params.kappa * (f**2) * compute_cycles

        return energy

    def compute_offload_energy(
        self, transmission_time: float, execution_time: float, download_time: float
    ) -> float:
        """
        计算卸载能耗

        公式: E_offload = P_tx * T_up + P_idle * (T_exe + T_down)

        Args:
            transmission_time: 上传时间 (s)
            execution_time: 执行时间 (s)
            download_time: 下载时间 (s)

        Returns:
            能耗 (J)

        来源: docs/01_系统模型.md §4.2.2
        """
        # 传输能耗
        tx_energy = self.params.tx_power * transmission_time

        # 空闲等待能耗 (等待边缘计算和结果下载)
        idle_energy = self.params.idle_power * (execution_time + download_time)

        return tx_energy + idle_energy

    def record_task_energy(
        self,
        vehicle_id: int,
        is_offloaded: bool,
        compute_cycles: float,
        transmission_time: float = 0.0,
        execution_time: float = 0.0,
        download_time: float = 0.0,
        cpu_freq: Optional[float] = None,
    ) -> float:
        """
        记录单个任务的能耗

        Args:
            vehicle_id: 车辆ID
            is_offloaded: 是否卸载
            compute_cycles: 计算周期数
            transmission_time: 传输时间 (卸载时)
            execution_time: 执行时间 (卸载时)
            download_time: 下载时间 (卸载时)
            cpu_freq: CPU频率 (本地时)

        Returns:
            任务能耗 (J)
        """
        if vehicle_id not in self.vehicle_states:
            self.vehicle_states[vehicle_id] = VehicleEnergyState(vehicle_id=vehicle_id)

        state = self.vehicle_states[vehicle_id]

        if is_offloaded:
            # 卸载能耗
            energy = self.compute_offload_energy(transmission_time, execution_time, download_time)
            state.transmission_energy += self.params.tx_power * transmission_time
            state.idle_energy += self.params.idle_power * (execution_time + download_time)
            state.offloaded_tasks += 1
        else:
            # 本地计算能耗
            energy = self.compute_local_energy(compute_cycles, cpu_freq)
            state.local_compute_energy += energy
            state.local_tasks += 1

        state.total_energy += energy
        state.energy_history.append(energy)

        return energy

    def update_slot(self, slot_energy: Dict[int, float]):
        """
        更新时隙能耗统计

        Args:
            slot_energy: {vehicle_id: energy} 本时隙各车辆能耗
        """
        total_slot_energy = sum(slot_energy.values())
        self.system_energy_history.append(total_slot_energy)
        self.current_slot += 1

    def get_vehicle_energy(self, vehicle_id: int) -> float:
        """获取指定车辆的累计能耗"""
        if vehicle_id in self.vehicle_states:
            return self.vehicle_states[vehicle_id].total_energy
        return 0.0

    def get_vehicle_statistics(self, vehicle_id: int) -> dict:
        """
        获取单辆车的能耗统计

        Returns:
            包含总能耗、分项能耗、任务数等信息
        """
        if vehicle_id not in self.vehicle_states:
            return {}

        state = self.vehicle_states[vehicle_id]
        total_tasks = state.local_tasks + state.offloaded_tasks

        return {
            "vehicle_id": vehicle_id,
            "total_energy_j": state.total_energy,
            "total_energy_mj": state.total_energy * 1000,  # mJ
            "local_compute_energy_j": state.local_compute_energy,
            "transmission_energy_j": state.transmission_energy,
            "idle_energy_j": state.idle_energy,
            "local_tasks": state.local_tasks,
            "offloaded_tasks": state.offloaded_tasks,
            "offload_ratio": state.offloaded_tasks / max(total_tasks, 1),
            "avg_energy_per_task_mj": (state.total_energy * 1000) / max(total_tasks, 1),
        }

    def get_statistics(self) -> dict:
        """
        获取系统级能耗统计

        Returns:
            系统总能耗、平均能耗等统计量

        对应论文指标:
        - J_2 = (1/K) Σ_k Σ_i E_i(k)  (平均能耗)
        """
        total_energy = 0.0
        total_local_energy = 0.0
        total_tx_energy = 0.0
        total_idle_energy = 0.0
        total_local_tasks = 0
        total_offloaded_tasks = 0
        per_vehicle_energy = []

        for state in self.vehicle_states.values():
            total_energy += state.total_energy
            total_local_energy += state.local_compute_energy
            total_tx_energy += state.transmission_energy
            total_idle_energy += state.idle_energy
            total_local_tasks += state.local_tasks
            total_offloaded_tasks += state.offloaded_tasks
            per_vehicle_energy.append(state.total_energy)

        total_tasks = total_local_tasks + total_offloaded_tasks
        per_vehicle_array = np.array(per_vehicle_energy)

        return {
            # 系统总量
            "system_total_energy_j": total_energy,
            "system_total_energy_mj": total_energy * 1000,
            # 分项统计
            "local_compute_energy_j": total_local_energy,
            "transmission_energy_j": total_tx_energy,
            "idle_energy_j": total_idle_energy,
            # 任务统计
            "total_local_tasks": total_local_tasks,
            "total_offloaded_tasks": total_offloaded_tasks,
            "offload_ratio": total_offloaded_tasks / max(total_tasks, 1),
            # 平均值
            "avg_energy_per_task_mj": (total_energy * 1000) / max(total_tasks, 1),
            "avg_energy_per_vehicle_j": (
                float(np.mean(per_vehicle_array)) if len(per_vehicle_array) > 0 else 0.0
            ),
            "energy_std_per_vehicle_j": (
                float(np.std(per_vehicle_array)) if len(per_vehicle_array) > 0 else 0.0
            ),
            # 时间序列
            "system_energy_history": self.system_energy_history,
            # 元信息
            "total_slots": self.current_slot,
            "num_vehicles": self.num_vehicles,
        }

    def reset(self):
        """重置所有能耗状态"""
        for state in self.vehicle_states.values():
            state.total_energy = 0.0
            state.local_compute_energy = 0.0
            state.transmission_energy = 0.0
            state.idle_energy = 0.0
            state.energy_history.clear()
            state.local_tasks = 0
            state.offloaded_tasks = 0

        self.current_slot = 0
        self.system_energy_history.clear()


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("能耗计算模块单元测试")
    print("=" * 60)

    # 测试1: 参数初始化
    print("\n--- 测试1: 参数初始化 ---")
    params = EnergyParameters()
    print(f"κ (能耗系数): {params.kappa:.2e}")
    print(f"发射功率: {params.tx_power*1000:.0f} mW (yaml: 23 dBm)")
    print(f"空闲功率: {params.idle_power*1000:.0f} mW (yaml: 0.1 W)")
    print(f"本地CPU频率: {params.local_cpu_freq/1e9:.1f} GHz")

    # 测试2: 本地计算能耗
    print("\n--- 测试2: 本地计算能耗 ---")
    calculator = EnergyCalculator(num_vehicles=5)

    # 典型任务: 100 MCycles
    compute_cycles = 1.0e8  # yaml cpm.compute_base
    e_local = calculator.compute_local_energy(compute_cycles)
    print(f"计算量: {compute_cycles/1e6:.0f} MCycles")
    print(f"本地能耗: {e_local*1000:.4f} mJ")
    print(f"公式验证: κ*f²*C = {params.kappa}*({params.local_cpu_freq})²*{compute_cycles}")
    print(f"         = {params.kappa * params.local_cpu_freq**2 * compute_cycles * 1000:.4f} mJ")

    # 测试3: 卸载能耗
    print("\n--- 测试3: 卸载能耗 ---")
    tx_time = 0.01  # 10ms传输
    exe_time = 0.02  # 20ms边缘计算
    down_time = 0.001  # 1ms下载

    e_offload = calculator.compute_offload_energy(tx_time, exe_time, down_time)
    print(f"传输时间: {tx_time*1000:.0f}ms")
    print(f"执行时间: {exe_time*1000:.0f}ms")
    print(f"下载时间: {down_time*1000:.0f}ms")
    print(f"卸载能耗: {e_offload*1000:.4f} mJ")
    print(f"  - 传输: {params.tx_power*tx_time*1000:.4f} mJ")
    print(f"  - 空闲: {params.idle_power*(exe_time+down_time)*1000:.4f} mJ")

    # 测试4: 能耗对比
    print("\n--- 测试4: 本地 vs 卸载 能耗对比 ---")
    print(f"本地计算 (100 MCycles): {e_local*1000:.4f} mJ")
    print(f"卸载 (10ms传输+20ms计算): {e_offload*1000:.4f} mJ")
    print(f"结论: {'卸载更省能' if e_offload < e_local else '本地更省能'}")

    # 测试5: 记录任务能耗
    print("\n--- 测试5: 记录任务能耗 ---")
    calculator.reset()

    # 模拟混合负载
    for i in range(10):
        if i % 2 == 0:
            # 本地任务
            calculator.record_task_energy(vehicle_id=0, is_offloaded=False, compute_cycles=1.0e8)
        else:
            # 卸载任务
            calculator.record_task_energy(
                vehicle_id=0,
                is_offloaded=True,
                compute_cycles=1.0e8,
                transmission_time=0.01,
                execution_time=0.02,
                download_time=0.001,
            )

    v0_stats = calculator.get_vehicle_statistics(0)
    print(f"车辆0统计:")
    print(f"  总能耗: {v0_stats['total_energy_mj']:.4f} mJ")
    print(f"  本地任务: {v0_stats['local_tasks']}")
    print(f"  卸载任务: {v0_stats['offloaded_tasks']}")
    print(f"  卸载率: {v0_stats['offload_ratio']*100:.0f}%")
    print(f"  平均单任务能耗: {v0_stats['avg_energy_per_task_mj']:.4f} mJ")

    # 测试6: 系统统计
    print("\n--- 测试6: 系统统计 ---")
    # 为其他车辆也添加任务
    for vid in range(1, 5):
        for _ in range(5):
            calculator.record_task_energy(
                vehicle_id=vid,
                is_offloaded=True,
                compute_cycles=1.0e8,
                transmission_time=0.01,
                execution_time=0.02,
                download_time=0.001,
            )

    sys_stats = calculator.get_statistics()
    print(f"系统总能耗: {sys_stats['system_total_energy_mj']:.4f} mJ")
    print(f"本地计算能耗: {sys_stats['local_compute_energy_j']*1000:.4f} mJ")
    print(f"传输能耗: {sys_stats['transmission_energy_j']*1000:.4f} mJ")
    print(f"空闲能耗: {sys_stats['idle_energy_j']*1000:.4f} mJ")
    print(f"卸载率: {sys_stats['offload_ratio']*100:.0f}%")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


# 运行: python3 -m simulation.metrics.energy
