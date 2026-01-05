"""
基线算法模块
文件: simulation/baselines/__init__.py

包含6个基线算法用于对比实验
"""

from simulation.baselines.full_offload import FullOffloadBaseline, FullOffloadConfig
from simulation.baselines.greedy import GreedyBaseline, GreedyConfig, GreedyMetric
from simulation.baselines.local_only import LocalOnlyBaseline, LocalOnlyConfig
from simulation.baselines.no_phase_align import NoPhaseAlignBaseline, NoPhaseAlignConfig
from simulation.baselines.random_offload import RandomOffloadBaseline, RandomOffloadConfig
from simulation.baselines.threshold import ThresholdBaseline, ThresholdConfig, ThresholdType

__all__ = [
    # Local Only
    "LocalOnlyConfig",
    "LocalOnlyBaseline",
    # Full Offload
    "FullOffloadConfig",
    "FullOffloadBaseline",
    # Greedy
    "GreedyMetric",
    "GreedyConfig",
    "GreedyBaseline",
    # Random
    "RandomOffloadConfig",
    "RandomOffloadBaseline",
    # Threshold
    "ThresholdType",
    "ThresholdConfig",
    "ThresholdBaseline",
    # No Phase Align
    "NoPhaseAlignConfig",
    "NoPhaseAlignBaseline",
]


def get_all_baselines() -> dict:
    """
    获取所有基线算法实例

    Returns:
        baselines: {name: instance}
    """
    return {
        "LocalOnly": LocalOnlyBaseline(),
        "FullOffload": FullOffloadBaseline(),
        "Greedy_delay": GreedyBaseline(GreedyConfig(metric=GreedyMetric.DELAY)),
        "Greedy_energy": GreedyBaseline(GreedyConfig(metric=GreedyMetric.ENERGY)),
        "Random_0.5": RandomOffloadBaseline(RandomOffloadConfig(offload_prob=0.5)),
        "Threshold_channel": ThresholdBaseline(
            ThresholdConfig(threshold_type=ThresholdType.CHANNEL_RATE)
        ),
        "NoPhaseAlign": NoPhaseAlignBaseline(),
    }


# 模块入口
if __name__ == "__main__":
    print("=" * 60)
    print("基线算法模块")
    print("=" * 60)

    baselines = get_all_baselines()
    print(f"\n共 {len(baselines)} 个基线算法:")
    for name, obj in baselines.items():
        print(f"  - {name}: {type(obj).__name__}")

    print("\n" + "=" * 60)
