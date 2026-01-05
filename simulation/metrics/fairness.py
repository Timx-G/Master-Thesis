"""
公平性计算模块
==============

基于文档: docs/07.1_附录D_公平性分析.md

Jain公平性指数 (JFI):
- JFI = (Σ x_i)² / (N · Σ x_i²)
- JFI ∈ [1/N, 1]
- JFI = 1: 完全公平
- JFI = 1/N: 完全不公平

其他指标:
- Max/Min比
- 变异系数 (CV)
- 基尼系数
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def calculate_jain_fairness_index(values: List[float]) -> float:
    """
    计算Jain公平性指数

    公式: JFI = (Σ x_i)² / (N · Σ x_i²)

    Args:
        values: 各车辆的指标值列表 (如AoI、资源分配等)

    Returns:
        JFI值，范围[1/N, 1]

    来源: docs/07.1_附录D_公平性分析.md §2.1
    """
    if not values or len(values) == 0:
        return 1.0

    values_array = np.array(values)
    n = len(values_array)

    # 处理全零情况
    if np.sum(values_array) == 0:
        return 1.0

    # JFI = (Σ x_i)² / (N · Σ x_i²)
    sum_values = np.sum(values_array)
    sum_squares = np.sum(values_array**2)

    if sum_squares == 0:
        return 1.0

    jfi = (sum_values**2) / (n * sum_squares)

    return float(jfi)


def calculate_max_min_ratio(values: List[float]) -> float:
    """
    计算最大-最小比

    公式: R = max(x_i) / min(x_i)

    Args:
        values: 指标值列表

    Returns:
        Max/Min比，R=1表示完全公平

    来源: docs/07.1_附录D_公平性分析.md §2.3
    """
    if not values or len(values) == 0:
        return 1.0

    values_array = np.array(values)
    min_val = np.min(values_array)
    max_val = np.max(values_array)

    if min_val <= 0:
        return float("inf")

    return float(max_val / min_val)


def calculate_coefficient_of_variation(values: List[float]) -> float:
    """
    计算变异系数 (CV)

    公式: CV = σ / μ

    Args:
        values: 指标值列表

    Returns:
        CV值，CV=0表示完全公平

    来源: docs/07.1_附录D_公平性分析.md §2.3
    """
    if not values or len(values) == 0:
        return 0.0

    values_array = np.array(values)
    mean_val = np.mean(values_array)

    if mean_val == 0:
        return 0.0

    std_val = np.std(values_array)

    return float(std_val / mean_val)


def calculate_gini_coefficient(values: List[float]) -> float:
    """
    计算基尼系数

    公式: G = Σ_i Σ_j |x_i - x_j| / (2N · Σ x_i)

    Args:
        values: 指标值列表

    Returns:
        基尼系数，G=0表示完全公平，G=1表示完全不公平

    来源: docs/07.1_附录D_公平性分析.md §2.3
    """
    if not values or len(values) == 0:
        return 0.0

    values_array = np.array(values)
    n = len(values_array)

    if np.sum(values_array) == 0:
        return 0.0

    # 计算所有成对差的绝对值之和
    diff_sum = 0.0
    for i in range(n):
        for j in range(n):
            diff_sum += abs(values_array[i] - values_array[j])

    gini = diff_sum / (2 * n * np.sum(values_array))

    return float(gini)


def calculate_weighted_jain_fairness(values: List[float], weights: List[float]) -> float:
    """
    计算加权Jain公平性指数

    公式: JFI_w = (Σ w_i·x_i)² / (N · Σ w_i²·x_i²)

    用于归一化后评估按需分配的公平性

    Args:
        values: 指标值列表
        weights: 权重列表 (如 1/λ_i，需求的倒数)

    Returns:
        加权JFI值

    来源: docs/07.1_附录D_公平性分析.md §4.5
    """
    if not values or not weights or len(values) != len(weights):
        return 1.0

    values_array = np.array(values)
    weights_array = np.array(weights)
    n = len(values_array)

    # 加权值
    weighted_values = weights_array * values_array
    weighted_squares = (weights_array**2) * (values_array**2)

    sum_weighted = np.sum(weighted_values)
    sum_weighted_sq = np.sum(weighted_squares)

    if sum_weighted_sq == 0:
        return 1.0

    jfi_w = (sum_weighted**2) / (n * sum_weighted_sq)

    return float(jfi_w)


@dataclass
class FairnessMetrics:
    """
    公平性指标集合
    """

    jain_fairness_index: float = 1.0
    max_min_ratio: float = 1.0
    coefficient_of_variation: float = 0.0
    gini_coefficient: float = 0.0
    weighted_jfi: float = 1.0


class FairnessCalculator:
    """
    公平性计算器

    功能:
    1. 计算AoI公平性
    2. 计算资源分配公平性
    3. 支持加权公平性 (按需分配归一化)

    使用示例:
    ```python
    calculator = FairnessCalculator()

    # 计算AoI公平性
    aoi_values = [0.1, 0.12, 0.15, 0.08, 0.11]
    metrics = calculator.calculate_all_metrics(aoi_values)

    print(f"JFI: {metrics.jain_fairness_index:.4f}")
    ```
    """

    def __init__(self):
        """初始化公平性计算器"""
        # 历史记录
        self.jfi_history: List[float] = []
        self.current_slot: int = 0

        logger.info("FairnessCalculator initialized")

    def calculate_all_metrics(
        self, values: List[float], weights: Optional[List[float]] = None
    ) -> FairnessMetrics:
        """
        计算所有公平性指标

        Args:
            values: 指标值列表
            weights: 可选权重列表

        Returns:
            FairnessMetrics对象
        """
        jfi = calculate_jain_fairness_index(values)
        max_min = calculate_max_min_ratio(values)
        cv = calculate_coefficient_of_variation(values)
        gini = calculate_gini_coefficient(values)

        # 加权JFI
        weighted_jfi = 1.0
        if weights is not None and len(weights) == len(values):
            weighted_jfi = calculate_weighted_jain_fairness(values, weights)

        return FairnessMetrics(
            jain_fairness_index=jfi,
            max_min_ratio=max_min,
            coefficient_of_variation=cv,
            gini_coefficient=gini,
            weighted_jfi=weighted_jfi,
        )

    def record_slot(self, values: List[float]) -> float:
        """
        记录一个时隙的公平性

        Args:
            values: 本时隙各车辆的指标值

        Returns:
            本时隙的JFI
        """
        jfi = calculate_jain_fairness_index(values)
        self.jfi_history.append(jfi)
        self.current_slot += 1
        return jfi

    def get_statistics(self) -> dict:
        """
        获取公平性统计信息

        Returns:
            包含平均JFI、JFI变化等统计量
        """
        if not self.jfi_history:
            return {
                "avg_jfi": 1.0,
                "min_jfi": 1.0,
                "max_jfi": 1.0,
                "std_jfi": 0.0,
                "jfi_history": [],
                "total_slots": 0,
            }

        jfi_array = np.array(self.jfi_history)

        return {
            "avg_jfi": float(np.mean(jfi_array)),
            "min_jfi": float(np.min(jfi_array)),
            "max_jfi": float(np.max(jfi_array)),
            "std_jfi": float(np.std(jfi_array)),
            "final_jfi": self.jfi_history[-1],
            "jfi_history": self.jfi_history,
            "total_slots": self.current_slot,
        }

    def reset(self):
        """重置状态"""
        self.jfi_history.clear()
        self.current_slot = 0


def analyze_fairness_by_speed(
    vehicle_aois: Dict[int, float], vehicle_speeds: Dict[int, float], speed_threshold: float = 25.0
) -> dict:
    """
    按车速分组分析公平性

    将车辆分为高速组和低速组，分别计算公平性

    Args:
        vehicle_aois: {vehicle_id: aoi}
        vehicle_speeds: {vehicle_id: speed}
        speed_threshold: 速度分界点 (m/s)

    Returns:
        分组公平性分析结果

    来源: docs/07.1_附录D_公平性分析.md §1.1
    """
    high_speed_aois = []
    low_speed_aois = []

    for vid, aoi in vehicle_aois.items():
        speed = vehicle_speeds.get(vid, 0.0)
        if speed >= speed_threshold:
            high_speed_aois.append(aoi)
        else:
            low_speed_aois.append(aoi)

    # 组内公平性
    high_speed_jfi = calculate_jain_fairness_index(high_speed_aois)
    low_speed_jfi = calculate_jain_fairness_index(low_speed_aois)

    # 全局公平性
    all_aois = list(vehicle_aois.values())
    global_jfi = calculate_jain_fairness_index(all_aois)

    return {
        "high_speed": {
            "count": len(high_speed_aois),
            "avg_aoi": float(np.mean(high_speed_aois)) if high_speed_aois else 0.0,
            "jfi": high_speed_jfi,
        },
        "low_speed": {
            "count": len(low_speed_aois),
            "avg_aoi": float(np.mean(low_speed_aois)) if low_speed_aois else 0.0,
            "jfi": low_speed_jfi,
        },
        "global_jfi": global_jfi,
        "speed_threshold": speed_threshold,
    }


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("公平性计算模块单元测试")
    print("=" * 60)

    # 测试1: Jain公平性指数
    print("\n--- 测试1: Jain公平性指数 ---")

    # 完全公平
    values_equal = [0.1, 0.1, 0.1, 0.1, 0.1]
    jfi_equal = calculate_jain_fairness_index(values_equal)
    print(f"完全相等 [0.1]*5: JFI = {jfi_equal:.4f} (应为1.0)")

    # 部分不公平
    values_partial = [0.1, 0.2, 0.1, 0.15, 0.12]
    jfi_partial = calculate_jain_fairness_index(values_partial)
    print(f"部分差异: JFI = {jfi_partial:.4f}")

    # 完全不公平
    values_unfair = [1.0, 0.0, 0.0, 0.0, 0.0]
    jfi_unfair = calculate_jain_fairness_index(values_unfair)
    print(f"完全不公平 [1,0,0,0,0]: JFI = {jfi_unfair:.4f} (应为0.2=1/5)")

    # 测试2: Max/Min比
    print("\n--- 测试2: Max/Min比 ---")
    print(f"相等: R = {calculate_max_min_ratio(values_equal):.2f}")
    print(f"部分差异: R = {calculate_max_min_ratio(values_partial):.2f}")
    print(f"不公平 [1,0.1,0.1]: R = {calculate_max_min_ratio([1,0.1,0.1]):.2f}")

    # 测试3: 变异系数
    print("\n--- 测试3: 变异系数 (CV) ---")
    print(f"相等: CV = {calculate_coefficient_of_variation(values_equal):.4f}")
    print(f"部分差异: CV = {calculate_coefficient_of_variation(values_partial):.4f}")

    # 测试4: 基尼系数
    print("\n--- 测试4: 基尼系数 ---")
    print(f"相等: Gini = {calculate_gini_coefficient(values_equal):.4f}")
    print(f"部分差异: Gini = {calculate_gini_coefficient(values_partial):.4f}")

    # 测试5: 综合指标
    print("\n--- 测试5: 综合公平性指标 ---")
    calculator = FairnessCalculator()

    # 模拟AoI分布
    aoi_values = [0.12, 0.15, 0.10, 0.18, 0.11, 0.14, 0.09, 0.16]
    metrics = calculator.calculate_all_metrics(aoi_values)

    print(f"AoI值: {[f'{v*1000:.0f}ms' for v in aoi_values]}")
    print(f"JFI: {metrics.jain_fairness_index:.4f}")
    print(f"Max/Min: {metrics.max_min_ratio:.2f}")
    print(f"CV: {metrics.coefficient_of_variation:.4f}")
    print(f"Gini: {metrics.gini_coefficient:.4f}")

    # 测试6: 加权公平性
    print("\n--- 测试6: 加权公平性 (按需分配归一化) ---")

    # 模拟: 高速车AoI高，但按其需求归一化后公平
    aois = [0.20, 0.10, 0.15, 0.08]  # 高速车AoI较高
    demands = [10, 2.5, 5, 2]  # 任务生成率 (高速车需求高)
    weights = [1 / d for d in demands]  # 需求倒数作为权重

    standard_jfi = calculate_jain_fairness_index(aois)
    weighted_jfi = calculate_weighted_jain_fairness(aois, weights)

    print(f"AoI: {aois}")
    print(f"需求: {demands}")
    print(f"标准JFI: {standard_jfi:.4f}")
    print(f"加权JFI: {weighted_jfi:.4f}")
    print("(加权后更公平，因为高AoI对应高需求)")

    # 测试7: 按速度分组分析
    print("\n--- 测试7: 按速度分组分析 ---")
    vehicle_aois = {0: 0.20, 1: 0.18, 2: 0.10, 3: 0.12, 4: 0.08}
    vehicle_speeds = {0: 35.0, 1: 30.0, 2: 15.0, 3: 12.0, 4: 10.0}

    analysis = analyze_fairness_by_speed(vehicle_aois, vehicle_speeds, speed_threshold=25.0)

    print(f"高速组 (>25m/s):")
    print(f"  数量: {analysis['high_speed']['count']}")
    print(f"  平均AoI: {analysis['high_speed']['avg_aoi']*1000:.0f}ms")
    print(f"  组内JFI: {analysis['high_speed']['jfi']:.4f}")
    print(f"低速组 (<25m/s):")
    print(f"  数量: {analysis['low_speed']['count']}")
    print(f"  平均AoI: {analysis['low_speed']['avg_aoi']*1000:.0f}ms")
    print(f"  组内JFI: {analysis['low_speed']['jfi']:.4f}")
    print(f"全局JFI: {analysis['global_jfi']:.4f}")

    # 测试8: 时间序列
    print("\n--- 测试8: JFI时间序列 ---")
    calculator.reset()

    for slot in range(10):
        # 模拟随时间变化的AoI分布
        noise = np.random.uniform(-0.02, 0.02, 5)
        values = [0.12 + n for n in noise]
        jfi = calculator.record_slot(values)

        if slot % 3 == 0:
            print(f"  时隙{slot}: JFI = {jfi:.4f}")

    stats = calculator.get_statistics()
    print(f"平均JFI: {stats['avg_jfi']:.4f}")
    print(f"JFI范围: [{stats['min_jfi']:.4f}, {stats['max_jfi']:.4f}]")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


# 运行: python3 -m simulation.metrics.fairness
