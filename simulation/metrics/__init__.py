"""
metrics模块 - 性能指标计算
==========================

包含:
- aoi: AoI (Age of Information) 计算
- energy: 能耗计算
- deadline: 截止时间违反率计算
- fairness: 公平性指标计算 (Jain指数等)
"""

from simulation.metrics.aoi import (
    AoICalculator,
    VehicleAoIState,
    calculate_average_aoi,
    calculate_peak_aoi,
)
from simulation.metrics.deadline import DeadlineCalculator, VehicleDeadlineState
from simulation.metrics.energy import EnergyCalculator, EnergyParameters, VehicleEnergyState
from simulation.metrics.fairness import (
    FairnessCalculator,
    FairnessMetrics,
    analyze_fairness_by_speed,
    calculate_coefficient_of_variation,
    calculate_gini_coefficient,
    calculate_jain_fairness_index,
    calculate_max_min_ratio,
    calculate_weighted_jain_fairness,
)

__all__ = [
    # AoI
    "AoICalculator",
    "VehicleAoIState",
    "calculate_average_aoi",
    "calculate_peak_aoi",
    # Energy
    "EnergyCalculator",
    "EnergyParameters",
    "VehicleEnergyState",
    # Deadline
    "DeadlineCalculator",
    "VehicleDeadlineState",
    # Fairness
    "FairnessCalculator",
    "FairnessMetrics",
    "calculate_jain_fairness_index",
    "calculate_max_min_ratio",
    "calculate_coefficient_of_variation",
    "calculate_gini_coefficient",
    "calculate_weighted_jain_fairness",
    "analyze_fairness_by_speed",
]
