"""
基线算法6: 无相位对齐 (No Phase Alignment)
文件: simulation/baselines/no_phase_align.py

功能说明:
- 使用Lyapunov优化但不进行相位对齐
- 用于验证相位对齐的收益

来源: docs/07_仿真实验设计.md §3.1
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NoPhaseAlignConfig:
    """无相位对齐配置"""

    # Lyapunov参数
    V: float = 1000.0

    # 权重
    weight_delay: float = 0.5
    weight_energy: float = 0.5

    # 能耗参数
    kappa: float = 1.0e-27
    tx_power: float = 0.2
    idle_power: float = 0.1

    # 调试选项
    verbose: bool = False


class NoPhaseAlignBaseline:
    """
    无相位对齐基线算法

    特点:
    - 使用简化的卸载决策（类似Lyapunov）
    - 但任务触发时间完全由ETSI条件决定，无相位调度
    - 用于对比PAPO中相位对齐的收益
    """

    def __init__(self, config: Optional[NoPhaseAlignConfig] = None):
        self.config = config or NoPhaseAlignConfig()
        self.name = "NoPhaseAlign"

        # 虚拟队列（简化版）
        self._queue_delay: Dict[int, float] = {}
        self._queue_energy: Dict[int, float] = {}

        self.stats = {
            "total_tasks": 0,
            "local_tasks": 0,
            "offload_tasks": 0,
            "deadline_violations": 0,
            "total_delay": 0.0,
            "total_energy": 0.0,
            "collision_count": 0,  # 任务冲突计数
        }

    def _get_queue(self, vehicle_id: int) -> Tuple[float, float]:
        """获取车辆队列状态"""
        if vehicle_id not in self._queue_delay:
            self._queue_delay[vehicle_id] = 0.0
            self._queue_energy[vehicle_id] = 0.0
        return self._queue_delay[vehicle_id], self._queue_energy[vehicle_id]

    def _update_queue(self, vehicle_id: int, delay: float, energy: float):
        """更新虚拟队列"""
        q_d, q_e = self._get_queue(vehicle_id)

        # 简单队列更新
        target_delay = 0.05  # 目标时延
        target_energy = 0.05  # 目标能耗

        self._queue_delay[vehicle_id] = max(0, q_d + delay - target_delay)
        self._queue_energy[vehicle_id] = max(0, q_e + energy - target_energy)

    def make_decision(
        self,
        vehicle_id: int,
        task_compute_cycles: float,
        task_data_size: float,
        task_result_size: float,
        task_deadline: float,
        local_cpu_freq: float,
        channel_rate: float,
        rsu_cpu_freq: float,
        rsu_available: bool = True,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        做出卸载决策
        """
        cfg = self.config

        # 计算本地代价
        delay_local = task_compute_cycles / local_cpu_freq
        energy_local = cfg.kappa * (local_cpu_freq**2) * task_compute_cycles

        # 计算卸载代价
        if rsu_available and channel_rate > 0:
            t_up = task_data_size / channel_rate
            t_exe = task_compute_cycles / rsu_cpu_freq
            t_down = task_result_size / channel_rate
            delay_offload = t_up + t_exe + t_down
            energy_offload = cfg.tx_power * t_up + cfg.idle_power * (t_exe + t_down)
        else:
            delay_offload = float("inf")
            energy_offload = float("inf")

        # 获取队列状态
        q_d, q_e = self._get_queue(vehicle_id)

        # 计算DPP (简化版)
        V = cfg.V

        cost_local = (
            q_d * delay_local
            + q_e * energy_local
            + V * (cfg.weight_delay * delay_local + cfg.weight_energy * energy_local)
        )

        cost_offload = (
            q_d * delay_offload
            + q_e * energy_offload
            + V * (cfg.weight_delay * delay_offload + cfg.weight_energy * energy_offload)
        )

        # 决策
        if cost_local <= cost_offload:
            offload = False
            delay, energy = delay_local, energy_local
            decision = "LOCAL"
        else:
            offload = True
            delay, energy = delay_offload, energy_offload
            decision = "OFFLOAD"

        # 更新队列
        self._update_queue(vehicle_id, delay, energy)

        # 检查截止时间
        deadline_met = delay <= task_deadline

        # 更新统计
        self.stats["total_tasks"] += 1
        self.stats["total_delay"] += delay
        self.stats["total_energy"] += energy

        if offload:
            self.stats["offload_tasks"] += 1
        else:
            self.stats["local_tasks"] += 1

        if not deadline_met:
            self.stats["deadline_violations"] += 1

        info = {
            "decision": decision,
            "delay": delay,
            "energy": energy,
            "deadline_met": deadline_met,
            "queue_delay": self._queue_delay[vehicle_id],
            "queue_energy": self._queue_energy[vehicle_id],
        }

        return offload, info

    def record_collision(self, num_collisions: int = 1):
        """记录任务冲突"""
        self.stats["collision_count"] += num_collisions

    def get_statistics(self) -> dict:
        stats = self.stats.copy()
        if stats["total_tasks"] > 0:
            stats["avg_delay"] = stats["total_delay"] / stats["total_tasks"]
            stats["avg_energy"] = stats["total_energy"] / stats["total_tasks"]
            stats["offload_ratio"] = stats["offload_tasks"] / stats["total_tasks"]
            stats["deadline_rate"] = 1 - stats["deadline_violations"] / stats["total_tasks"]
        else:
            stats["avg_delay"] = 0.0
            stats["avg_energy"] = 0.0
            stats["offload_ratio"] = 0.0
            stats["deadline_rate"] = 0.0
        return stats

    def reset(self):
        self._queue_delay.clear()
        self._queue_energy.clear()
        self.stats = {
            "total_tasks": 0,
            "local_tasks": 0,
            "offload_tasks": 0,
            "deadline_violations": 0,
            "total_delay": 0.0,
            "total_energy": 0.0,
            "collision_count": 0,
        }


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 NoPhaseAlign 基线算法")
    print("=" * 60)

    baseline = NoPhaseAlignBaseline()

    # 模拟多车辆任务
    print("\n模拟10辆车各处理5个任务:")
    for vid in range(10):
        for _ in range(5):
            baseline.make_decision(
                vehicle_id=vid,
                task_compute_cycles=1.0e8,
                task_data_size=4e6,
                task_result_size=4e5,
                task_deadline=0.08,
                local_cpu_freq=2e9,
                channel_rate=np.random.uniform(5e6, 15e6),
                rsu_cpu_freq=5e9,
            )

    stats = baseline.get_statistics()
    print("\n统计信息:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)


# 运行测试: python3 -m simulation.baselines.no_phase_align
