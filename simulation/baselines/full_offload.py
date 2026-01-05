"""
基线算法2: 全卸载 (Full Offload)
文件: simulation/baselines/full_offload.py

功能说明:
- 所有任务卸载到RSU计算
- 用于对比本地方案的能耗优势

来源: docs/07_仿真实验设计.md §3.1
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FullOffloadConfig:
    """全卸载算法配置"""

    # 通信参数
    tx_power: float = 0.2  # 发射功率 (W)
    idle_power: float = 0.1  # 空闲功率 (W)

    # 默认资源分配
    default_rsu_freq: float = 5.0e9  # 默认分配CPU频率 (Hz)

    # 调试选项
    verbose: bool = False


class FullOffloadBaseline:
    """
    全卸载基线算法

    特点:
    - 所有任务卸载到RSU
    - 时延 = 上传 + 计算 + 下载
    - 能耗 = 发射能耗 + 空闲能耗

    适用场景: 作为纯卸载方案的参考
    """

    def __init__(self, config: Optional[FullOffloadConfig] = None):
        """
        初始化

        Args:
            config: 算法配置
        """
        self.config = config or FullOffloadConfig()
        self.name = "FullOffload"

        # 统计
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "deadline_violations": 0,
            "total_delay": 0.0,
            "total_energy": 0.0,
            "failed_offloads": 0,  # 无法卸载的任务
        }

    def make_decision(
        self,
        task_compute_cycles: float,
        task_data_size: float,
        task_result_size: float,
        task_deadline: float,
        channel_rate: float,
        rsu_cpu_freq: Optional[float] = None,
        rsu_available: bool = True,
        **kwargs,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        做出卸载决策（始终卸载）

        Args:
            task_compute_cycles: 任务计算需求 (cycles)
            task_data_size: 任务数据大小 (bits)
            task_result_size: 结果数据大小 (bits)
            task_deadline: 任务截止时间 (s)
            channel_rate: 信道速率 (bps)
            rsu_cpu_freq: RSU分配的CPU频率
            rsu_available: RSU是否可用

        Returns:
            (offload_decision, info): 卸载决策和详细信息
        """
        # 检查RSU可用性
        if not rsu_available or channel_rate <= 0:
            self.stats["total_tasks"] += 1
            self.stats["failed_offloads"] += 1
            return False, {"decision": "LOCAL_FALLBACK", "reason": "RSU unavailable"}

        # 使用默认或指定的RSU频率
        rsu_freq = rsu_cpu_freq or self.config.default_rsu_freq

        # 计算各项时延
        t_up = task_data_size / channel_rate
        t_exe = task_compute_cycles / rsu_freq
        t_down = task_result_size / channel_rate
        total_delay = t_up + t_exe + t_down

        # 计算能耗
        energy = self.config.tx_power * t_up + self.config.idle_power * (t_exe + t_down)

        # 检查截止时间
        deadline_met = total_delay <= task_deadline

        # 更新统计
        self.stats["total_tasks"] += 1
        self.stats["total_delay"] += total_delay
        self.stats["total_energy"] += energy

        if deadline_met:
            self.stats["completed_tasks"] += 1
        else:
            self.stats["deadline_violations"] += 1

        info = {
            "decision": "OFFLOAD",
            "delay": total_delay,
            "t_up": t_up,
            "t_exe": t_exe,
            "t_down": t_down,
            "energy": energy,
            "deadline_met": deadline_met,
        }

        if self.config.verbose:
            print(
                f"[FullOffload] 任务: delay={total_delay*1000:.2f}ms "
                f"(up={t_up*1000:.2f}, exe={t_exe*1000:.2f}, down={t_down*1000:.2f}), "
                f"energy={energy*1e3:.2f}mJ"
            )

        return True, info

    def get_statistics(self) -> dict:
        """获取统计信息"""
        stats = self.stats.copy()
        effective_tasks = stats["total_tasks"] - stats["failed_offloads"]

        if effective_tasks > 0:
            stats["avg_delay"] = stats["total_delay"] / effective_tasks
            stats["avg_energy"] = stats["total_energy"] / effective_tasks
            stats["deadline_rate"] = stats["completed_tasks"] / effective_tasks
        else:
            stats["avg_delay"] = 0.0
            stats["avg_energy"] = 0.0
            stats["deadline_rate"] = 0.0

        stats["offload_success_rate"] = (
            (stats["total_tasks"] - stats["failed_offloads"]) / stats["total_tasks"]
            if stats["total_tasks"] > 0
            else 0.0
        )

        return stats

    def reset(self):
        """重置统计"""
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "deadline_violations": 0,
            "total_delay": 0.0,
            "total_energy": 0.0,
            "failed_offloads": 0,
        }


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 FullOffload 基线算法")
    print("=" * 60)

    baseline = FullOffloadBaseline(FullOffloadConfig(verbose=True))

    # 测试多个任务
    print("\n处理任务:")
    for i in range(5):
        offload, info = baseline.make_decision(
            task_compute_cycles=1.0e8,
            task_data_size=4e6,
            task_result_size=4e5,
            task_deadline=0.08,
            channel_rate=10e6,
            rsu_cpu_freq=5e9,
        )
        print(f"  任务{i}: {info['decision']}, {info['delay']*1000:.2f}ms")

    print("\n统计信息:")
    for k, v in baseline.get_statistics().items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)


# 运行测试: python3 -m simulation.baselines.full_offload
