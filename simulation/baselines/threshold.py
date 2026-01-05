"""
基线算法5: 阈值策略 (Threshold-based)
文件: simulation/baselines/threshold.py

功能说明:
- 基于信道质量或任务大小的阈值决策
- 简单但常用的启发式方法
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ThresholdType(Enum):
    """阈值类型"""

    CHANNEL_RATE = "channel_rate"  # 信道速率阈值
    TASK_SIZE = "task_size"  # 任务大小阈值
    COMPUTE_CYCLES = "compute"  # 计算量阈值


@dataclass
class ThresholdConfig:
    """阈值策略配置"""

    # 阈值类型
    threshold_type: ThresholdType = ThresholdType.CHANNEL_RATE

    # 阈值
    channel_threshold: float = 8e6  # 信道速率阈值 (bps)
    task_size_threshold: float = 5e6  # 任务大小阈值 (bits)
    compute_threshold: float = 1.2e8  # 计算量阈值 (cycles)

    # 能耗参数
    kappa: float = 1.0e-27
    tx_power: float = 0.2
    idle_power: float = 0.1

    # 调试选项
    verbose: bool = False


class ThresholdBaseline:
    """
    阈值策略基线算法

    特点:
    - 信道模式: 信道好(>阈值)时卸载
    - 任务大小模式: 任务大(>阈值)时卸载（利用RSU算力）
    - 计算量模式: 计算重(>阈值)时卸载
    """

    def __init__(self, config: Optional[ThresholdConfig] = None):
        self.config = config or ThresholdConfig()
        self.name = f"Threshold_{self.config.threshold_type.value}"

        self.stats = {
            "total_tasks": 0,
            "local_tasks": 0,
            "offload_tasks": 0,
            "deadline_violations": 0,
            "total_delay": 0.0,
            "total_energy": 0.0,
        }

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
        做出卸载决策（基于阈值）
        """
        cfg = self.config

        # 根据阈值类型决策
        if not rsu_available:
            offload = False
        elif cfg.threshold_type == ThresholdType.CHANNEL_RATE:
            offload = channel_rate >= cfg.channel_threshold
        elif cfg.threshold_type == ThresholdType.TASK_SIZE:
            offload = task_data_size >= cfg.task_size_threshold
        else:  # COMPUTE_CYCLES
            offload = task_compute_cycles >= cfg.compute_threshold

        # 计算代价
        if offload and channel_rate > 0:
            t_up = task_data_size / channel_rate
            t_exe = task_compute_cycles / rsu_cpu_freq
            t_down = task_result_size / channel_rate
            delay = t_up + t_exe + t_down
            energy = cfg.tx_power * t_up + cfg.idle_power * (t_exe + t_down)
            decision = "OFFLOAD"
        else:
            offload = False
            delay = task_compute_cycles / local_cpu_freq
            energy = cfg.kappa * (local_cpu_freq**2) * task_compute_cycles
            decision = "LOCAL"

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
        }

        return offload, info

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
    print("测试 Threshold 基线算法")
    print("=" * 60)

    for threshold_type in ThresholdType:
        print(f"\n【{threshold_type.value}模式】")
        config = ThresholdConfig(threshold_type=threshold_type)
        baseline = ThresholdBaseline(config)

        # 测试变化条件
        for rate in [5e6, 10e6, 15e6]:
            baseline.make_decision(
                task_compute_cycles=1.0e8,
                task_data_size=4e6,
                task_result_size=4e5,
                task_deadline=0.08,
                local_cpu_freq=2e9,
                channel_rate=rate,
                rsu_cpu_freq=5e9,
            )

        stats = baseline.get_statistics()
        print(f"  卸载比例: {stats['offload_ratio']:.2%}")

    print("\n" + "=" * 60)


# 运行测试: python3 -m simulation.baselines.threshold
