"""
基线算法3: 贪心算法 (Greedy)
文件: simulation/baselines/greedy.py

功能说明:
- 基于即时代价贪心选择卸载决策
- 不考虑长期约束和队列状态

来源: docs/07_仿真实验设计.md §3.1
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np


class GreedyMetric(Enum):
    """贪心优化指标"""

    DELAY = "delay"  # 最小化时延
    ENERGY = "energy"  # 最小化能耗
    WEIGHTED = "weighted"  # 加权组合


@dataclass
class GreedyConfig:
    """贪心算法配置"""

    # 优化目标
    metric: GreedyMetric = GreedyMetric.DELAY

    # 权重 (用于WEIGHTED模式)
    weight_delay: float = 0.5
    weight_energy: float = 0.5

    # 能耗参数
    kappa: float = 1.0e-27
    tx_power: float = 0.2
    idle_power: float = 0.1

    # 调试选项
    verbose: bool = False


class GreedyBaseline:
    """
    贪心基线算法

    特点:
    - 每个任务独立决策，选择即时代价最小的选项
    - 不考虑队列状态和长期约束
    - 多种优化目标：最小时延、最小能耗、加权组合

    适用场景: 作为简单启发式的参考
    """

    def __init__(self, config: Optional[GreedyConfig] = None):
        """
        初始化

        Args:
            config: 算法配置
        """
        self.config = config or GreedyConfig()
        self.name = f"Greedy_{self.config.metric.value}"

        # 统计
        self.stats = {
            "total_tasks": 0,
            "local_tasks": 0,
            "offload_tasks": 0,
            "deadline_violations": 0,
            "total_delay": 0.0,
            "total_energy": 0.0,
        }

    def _compute_local_cost(
        self, task_compute_cycles: float, local_cpu_freq: float
    ) -> Tuple[float, float]:
        """计算本地处理代价"""
        delay = task_compute_cycles / local_cpu_freq
        energy = self.config.kappa * (local_cpu_freq**2) * task_compute_cycles
        return delay, energy

    def _compute_offload_cost(
        self,
        task_data_size: float,
        task_compute_cycles: float,
        task_result_size: float,
        channel_rate: float,
        rsu_cpu_freq: float,
    ) -> Tuple[float, float]:
        """计算卸载处理代价"""
        if channel_rate <= 0 or rsu_cpu_freq <= 0:
            return float("inf"), float("inf")

        t_up = task_data_size / channel_rate
        t_exe = task_compute_cycles / rsu_cpu_freq
        t_down = task_result_size / channel_rate
        delay = t_up + t_exe + t_down

        energy = self.config.tx_power * t_up + self.config.idle_power * (t_exe + t_down)

        return delay, energy

    def _compute_cost(self, delay: float, energy: float) -> float:
        """根据配置计算综合代价"""
        metric = self.config.metric

        if metric == GreedyMetric.DELAY:
            return delay
        elif metric == GreedyMetric.ENERGY:
            return energy
        else:  # WEIGHTED
            # 归一化处理
            return self.config.weight_delay * delay + self.config.weight_energy * energy

    def make_decision(
        self,
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
        做出卸载决策（贪心选择）

        Args:
            task_compute_cycles: 任务计算需求
            task_data_size: 任务数据大小
            task_result_size: 结果数据大小
            task_deadline: 截止时间
            local_cpu_freq: 本地CPU频率
            channel_rate: 信道速率
            rsu_cpu_freq: RSU CPU频率
            rsu_available: RSU是否可用

        Returns:
            (offload_decision, info): 决策和详细信息
        """
        # 计算本地代价
        delay_local, energy_local = self._compute_local_cost(task_compute_cycles, local_cpu_freq)
        cost_local = self._compute_cost(delay_local, energy_local)

        # 计算卸载代价
        if rsu_available and channel_rate > 0:
            delay_offload, energy_offload = self._compute_offload_cost(
                task_data_size, task_compute_cycles, task_result_size, channel_rate, rsu_cpu_freq
            )
            cost_offload = self._compute_cost(delay_offload, energy_offload)
        else:
            delay_offload, energy_offload = float("inf"), float("inf")
            cost_offload = float("inf")

        # 贪心选择
        if cost_local <= cost_offload:
            offload = False
            delay, energy = delay_local, energy_local
            decision = "LOCAL"
        else:
            offload = True
            delay, energy = delay_offload, energy_offload
            decision = "OFFLOAD"

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
            "cost_local": cost_local,
            "cost_offload": cost_offload,
        }

        if self.config.verbose:
            print(
                f"[Greedy] {decision}: delay={delay*1000:.2f}ms, "
                f"cost_local={cost_local:.4f}, cost_offload={cost_offload:.4f}"
            )

        return offload, info

    def get_statistics(self) -> dict:
        """获取统计信息"""
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
        """重置统计"""
        self.stats = {
            "total_tasks": 0,
            "local_tasks": 0,
            "offload_tasks": 0,
            "deadline_violations": 0,
            "total_delay": 0.0,
            "total_energy": 0.0,
        }


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 Greedy 基线算法")
    print("=" * 60)

    # 测试不同优化目标
    for metric in GreedyMetric:
        print(f"\n【{metric.value}模式】")
        config = GreedyConfig(metric=metric, verbose=True)
        baseline = GreedyBaseline(config)

        for i in range(3):
            # 变化信道条件
            channel_rate = 5e6 + i * 5e6  # 5, 10, 15 Mbps

            offload, info = baseline.make_decision(
                task_compute_cycles=1.0e8,
                task_data_size=4e6,
                task_result_size=4e5,
                task_deadline=0.08,
                local_cpu_freq=2e9,
                channel_rate=channel_rate,
                rsu_cpu_freq=5e9,
            )

        stats = baseline.get_statistics()
        print(f"  卸载比例: {stats['offload_ratio']:.2%}")

    print("\n" + "=" * 60)


# 运行测试: python3 -m simulation.baselines.greedy
