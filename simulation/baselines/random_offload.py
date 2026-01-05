"""
基线算法4: 随机卸载 (Random Offload)
文件: simulation/baselines/random_offload.py

功能说明:
- 以固定概率随机决定是否卸载
- 用于作为下界参考

来源: docs/07_仿真实验设计.md §3.1
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class RandomOffloadConfig:
    """随机卸载配置"""

    # 卸载概率
    offload_prob: float = 0.5

    # 能耗参数
    kappa: float = 1.0e-27
    tx_power: float = 0.2
    idle_power: float = 0.1

    # 随机种子
    seed: Optional[int] = None

    # 调试选项
    verbose: bool = False


class RandomOffloadBaseline:
    """
    随机卸载基线算法

    特点:
    - 以固定概率 p 卸载，1-p 本地
    - 无任何智能决策
    - 作为下界参考
    """

    def __init__(self, config: Optional[RandomOffloadConfig] = None):
        """
        初始化

        Args:
            config: 算法配置
        """
        self.config = config or RandomOffloadConfig()
        self.name = f"Random_p{self.config.offload_prob:.1f}"

        if self.config.seed is not None:
            self._rng = np.random.RandomState(self.config.seed)
        else:
            self._rng = np.random.RandomState()

        # 统计
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
        做出卸载决策（随机选择）

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
        # 随机决策
        if rsu_available and channel_rate > 0:
            offload = self._rng.random() < self.config.offload_prob
        else:
            offload = False  # RSU不可用时本地处理

        # 计算代价
        if offload:
            t_up = task_data_size / channel_rate
            t_exe = task_compute_cycles / rsu_cpu_freq
            t_down = task_result_size / channel_rate
            delay = t_up + t_exe + t_down
            energy = self.config.tx_power * t_up + self.config.idle_power * (t_exe + t_down)
            decision = "OFFLOAD"
        else:
            delay = task_compute_cycles / local_cpu_freq
            energy = self.config.kappa * (local_cpu_freq**2) * task_compute_cycles
            decision = "LOCAL"

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
        }

        if self.config.verbose:
            print(f"[Random] {decision}: delay={delay*1000:.2f}ms")

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
    print("测试 RandomOffload 基线算法")
    print("=" * 60)

    # 测试不同卸载概率
    for prob in [0.3, 0.5, 0.7]:
        print(f"\n【卸载概率 p={prob}】")
        config = RandomOffloadConfig(offload_prob=prob, seed=42)
        baseline = RandomOffloadBaseline(config)

        for i in range(100):
            baseline.make_decision(
                task_compute_cycles=1.0e8,
                task_data_size=4e6,
                task_result_size=4e5,
                task_deadline=0.08,
                local_cpu_freq=2e9,
                channel_rate=10e6,
                rsu_cpu_freq=5e9,
            )

        stats = baseline.get_statistics()
        print(f"  实际卸载比例: {stats['offload_ratio']:.2%}")
        print(f"  平均时延: {stats['avg_delay']*1000:.2f}ms")
        print(f"  截止时间满足率: {stats['deadline_rate']:.2%}")

    print("\n" + "=" * 60)


# 运行测试: python3 -m simulation.baselines.random_offload
