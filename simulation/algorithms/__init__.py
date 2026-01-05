"""
算法模块
文件: simulation/algorithms/__init__.py

包含:
- CPA: 集中式相位分配算法
- DPA: 分布式相位调整算法
- Lyapunov: 在线优化算法
- ResourceAllocator: 资源分配算法
- Predictor: 任务预测算法
"""

# ============================================================
# 相位分配算法
# ============================================================

from simulation.algorithms.cpa import (
    CentralizedPhaseAllocator,
    CPAConfig,
    compute_optimal_phase_spacing,
    verify_phase_allocation,
)
from simulation.algorithms.dpa import (
    DistributedPhaseAdjuster,
    DPAConfig,
    PhaseSlot,
    analyze_competitive_ratio,
)
from simulation.algorithms.lyapunov import (
    LyapunovConfig,
    LyapunovOptimizer,
    OffloadDecision,
    VehicleState,
    VirtualQueue,
    analyze_v_tradeoff,
)
from simulation.algorithms.predictor import (
    AggregateLoadPredictor,
    ETSITriggerConfig,
    HistoricalLoadPredictor,
    KinematicPredictor,
    LoadPredictorConfig,
    VehicleKinematicState,
)
from simulation.algorithms.resource_alloc import (
    AllocationStrategy,
    ResourceAllocator,
    ResourceAllocConfig,
    ResourceRequest,
    compare_allocation_strategies,
    compute_kkt_optimal_allocation,
    verify_kkt_conditions,
)

# ============================================================
# Lyapunov优化算法
# ============================================================


# ============================================================
# 资源分配算法
# ============================================================


# ============================================================
# 任务预测算法
# ============================================================


# ============================================================
# 导出列表
# ============================================================

__all__ = [
    # CPA
    "CPAConfig",
    "CentralizedPhaseAllocator",
    "compute_optimal_phase_spacing",
    "verify_phase_allocation",
    # DPA
    "DPAConfig",
    "PhaseSlot",
    "DistributedPhaseAdjuster",
    "analyze_competitive_ratio",
    # Lyapunov
    "LyapunovConfig",
    "OffloadDecision",
    "VirtualQueue",
    "VehicleState",
    "LyapunovOptimizer",
    "analyze_v_tradeoff",
    # Resource Allocation
    "AllocationStrategy",
    "ResourceRequest",
    "ResourceAllocConfig",
    "ResourceAllocator",
    "compute_kkt_optimal_allocation",
    "verify_kkt_conditions",
    "compare_allocation_strategies",
    # Predictor
    "ETSITriggerConfig",
    "VehicleKinematicState",
    "KinematicPredictor",
    "LoadPredictorConfig",
    "HistoricalLoadPredictor",
    "AggregateLoadPredictor",
]


# ============================================================
# 版本信息
# ============================================================

__version__ = "0.1.0"
__author__ = "V2X-MEC Research Team"


# ============================================================
# 快速使用示例
# ============================================================


def get_default_papo_components():
    """
    获取PAPO框架的默认组件配置

    Returns:
        dict: 包含所有算法组件的字典

    使用示例:
        components = get_default_papo_components()
        cpa = components["phase_allocator"]
        lyapunov = components["optimizer"]
    """
    return {
        "phase_allocator": CentralizedPhaseAllocator(CPAConfig()),
        "phase_adjuster": DistributedPhaseAdjuster(DPAConfig()),
        "optimizer": LyapunovOptimizer(LyapunovConfig()),
        "resource_allocator": ResourceAllocator(ResourceAllocConfig()),
        "predictor": AggregateLoadPredictor(),
    }


# ============================================================
# 模块测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 algorithms 模块导入")
    print("=" * 70)

    # 测试所有导入
    print("\n【测试1】模块导入检查")

    imports_ok = True
    for name in __all__:
        try:
            obj = eval(name)
            print(f"  ✓ {name}")
        except NameError:
            print(f"  ✗ {name} - 导入失败")
            imports_ok = False

    print(f"\n导入状态: {'全部成功' if imports_ok else '存在失败'}")

    # 测试快速创建
    print("\n【测试2】快速创建PAPO组件")
    components = get_default_papo_components()

    for name, obj in components.items():
        print(f"  {name}: {type(obj).__name__}")

    # 测试基本功能
    print("\n【测试3】基本功能验证")

    # CPA
    cpa = components["phase_allocator"]
    phases = cpa.allocate_phases(rsu_id=0, vehicle_ids=[1, 2, 3])
    print(f"  CPA分配: {len(phases)} 辆车")

    # DPA
    dpa = components["phase_adjuster"]
    phase = dpa.insert_vehicle(rsu_id=0, vehicle_id=1)
    print(f"  DPA插入: 相位={phase*1000:.2f}ms")

    # Lyapunov
    lyapunov = components["optimizer"]
    decision, _ = lyapunov.make_decision(
        vehicle_id=0,
        task_data_size=4e6,
        task_compute_cycles=1e8,
        task_result_size=4e5,
        local_cpu_freq=2e9,
        channel_rate=10e6,
        rsu_cpu_freq=5e9,
        current_aoi=0.05,
    )
    print(f"  Lyapunov决策: {decision.name}")

    # Resource Allocator
    allocator = components["resource_allocator"]
    requests = [
        ResourceRequest(task_id=i, vehicle_id=i, compute_cycles=1e8, data_size=4e6, deadline=0.08)
        for i in range(3)
    ]
    cpu_allocs = allocator.allocate_cpu(requests, total_cpu=20e9)
    print(f"  资源分配: {len(cpu_allocs)} 个任务")

    # Predictor
    import numpy as np

    predictor = components["predictor"]
    predictor.update_vehicle(0, np.array([0.0, 0.0]), 20.0, 0.0)
    predictor.update_rsu_coverage(0, [0])
    pred, unc = predictor.predict_rsu_load(0, current_time=0.0)
    print(f"  负载预测: {pred:.2f} ± {unc:.2f}")

    print("\n" + "=" * 70)
    print(f"algorithms 模块 v{__version__} 测试完成!")
    print("=" * 70)


# 运行测试: python3 -m simulation.algorithms
