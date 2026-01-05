"""
基线算法1: 全本地计算 (Local Only)
文件: simulation/baselines/local_only.py

功能说明:
- 所有任务在本地车辆计算
- 不进行任务卸载
- 用于对比卸载方案的收益

来源: docs/07_仿真实验设计.md §3.1
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LocalOnlyConfig:
    """全本地算法配置"""

    # 能耗参数
    kappa: float = 1.0e-27  # 能耗系数

    # 调试选项
    verbose: bool = False


class LocalOnlyBaseline:
    """
    全本地计算基线算法

    特点:
    - 所有任务本地处理，不卸载
    - 计算时延 T = C / f_local
    - 能耗 E = κ · f² · C

    适用场景: 作为纯本地方案的下界参考
    """

    def __init__(self, config: Optional[LocalOnlyConfig] = None):
        """
        初始化

        Args:
            config: 算法配置
        """
        self.config = config or LocalOnlyConfig()
        self.name = "LocalOnly"

        # 统计
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "deadline_violations": 0,
            "total_delay": 0.0,
            "total_energy": 0.0,
        }

    def make_decision(
        self,
        task_compute_cycles: float,
        task_data_size: float,
        task_deadline: float,
        local_cpu_freq: float,
        **kwargs,  # 忽略其他参数
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        做出卸载决策（始终本地）

        Args:
            task_compute_cycles: 任务计算需求 (cycles)
            task_data_size: 任务数据大小 (bits)
            task_deadline: 任务截止时间 (s)
            local_cpu_freq: 本地CPU频率 (Hz)

        Returns:
            (offload_decision, info): 卸载决策和详细信息
        """
        # 始终本地计算
        offload = False

        # 计算时延和能耗
        delay = task_compute_cycles / local_cpu_freq
        energy = self.config.kappa * (local_cpu_freq**2) * task_compute_cycles

        # 检查截止时间
        deadline_met = delay <= task_deadline

        # 更新统计
        self.stats["total_tasks"] += 1
        self.stats["total_delay"] += delay
        self.stats["total_energy"] += energy

        if deadline_met:
            self.stats["completed_tasks"] += 1
        else:
            self.stats["deadline_violations"] += 1

        info = {
            "decision": "LOCAL",
            "delay": delay,
            "energy": energy,
            "deadline_met": deadline_met,
        }

        if self.config.verbose:
            print(
                f"[LocalOnly] 任务: delay={delay*1000:.2f}ms, "
                f"energy={energy*1e6:.2f}μJ, deadline_met={deadline_met}"
            )

        return offload, info

    def process_task(
        self, task_compute_cycles: float, local_cpu_freq: float
    ) -> Tuple[float, float]:
        """
        处理任务（计算时延和能耗）

        Args:
            task_compute_cycles: 计算需求
            local_cpu_freq: CPU频率

        Returns:
            (delay, energy): 时延和能耗
        """
        delay = task_compute_cycles / local_cpu_freq
        energy = self.config.kappa * (local_cpu_freq**2) * task_compute_cycles
        return delay, energy

    def get_statistics(self) -> dict:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats["total_tasks"] > 0:
            stats["avg_delay"] = stats["total_delay"] / stats["total_tasks"]
            stats["avg_energy"] = stats["total_energy"] / stats["total_tasks"]
            stats["deadline_rate"] = stats["completed_tasks"] / stats["total_tasks"]
        else:
            stats["avg_delay"] = 0.0
            stats["avg_energy"] = 0.0
            stats["deadline_rate"] = 0.0
        return stats

    def reset(self):
        """重置统计"""
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "deadline_violations": 0,
            "total_delay": 0.0,
            "total_energy": 0.0,
        }


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 LocalOnly 基线算法")
    print("=" * 60)

    baseline = LocalOnlyBaseline(LocalOnlyConfig(verbose=True))

    # 测试多个任务
    tasks = [
        {"compute": 1.0e8, "data": 4e6, "deadline": 0.08},
        {"compute": 1.5e8, "data": 5e6, "deadline": 0.08},
        {"compute": 0.8e8, "data": 3e6, "deadline": 0.08},
    ]

    print("\n处理任务:")
    for i, task in enumerate(tasks):
        offload, info = baseline.make_decision(
            task_compute_cycles=task["compute"],
            task_data_size=task["data"],
            task_deadline=task["deadline"],
            local_cpu_freq=2e9,
        )
        print(f"  任务{i}: {info['decision']}, {info['delay']*1000:.2f}ms")

    print("\n统计信息:")
    for k, v in baseline.get_statistics().items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)


# 运行测试: python3 -m simulation.baselines.local_only
