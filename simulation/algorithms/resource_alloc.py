"""
资源分配算法
文件: simulation/algorithms/resource_alloc.py

功能说明:
- 实现RSU计算资源分配（基于KKT条件的最优解）
- 实现带宽资源分配
- 支持多种分配策略（加权公平、按需分配等）

算法来源: docs/06_算法设计_资源分配与服务迁移.md §2
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class AllocationStrategy(Enum):
    """资源分配策略枚举"""

    EQUAL = "equal"  # 均匀分配
    PROPORTIONAL = "proportional"  # 按需求比例分配
    WEIGHTED_FAIR = "weighted_fair"  # 加权公平分配 (KKT最优)
    PRIORITY = "priority"  # 优先级分配


@dataclass
class ResourceRequest:
    """资源请求数据结构"""

    task_id: int
    vehicle_id: int

    # 任务需求
    compute_cycles: float  # 计算需求 (cycles)
    data_size: float  # 数据大小 (bits)
    deadline: float  # 截止时间 (s)

    # 权重 (用于加权分配)
    weight: float = 1.0

    # 优先级 (用于优先级分配)
    priority: int = 0

    # 分配结果
    allocated_cpu: float = 0.0  # 分配的CPU频率 (Hz)
    allocated_bw: float = 0.0  # 分配的带宽 (Hz)


@dataclass
class ResourceAllocConfig:
    """资源分配配置"""

    # 分配策略
    cpu_strategy: AllocationStrategy = AllocationStrategy.WEIGHTED_FAIR
    bw_strategy: AllocationStrategy = AllocationStrategy.PROPORTIONAL

    # 资源约束
    min_cpu_per_task: float = 0.5e9  # 最小CPU分配 (Hz)
    min_bw_per_task: float = 1.0e6  # 最小带宽分配 (Hz)

    # 公平性参数
    max_cpu_ratio: float = 0.5  # 单任务最大占用比例
    max_bw_ratio: float = 0.5

    # 调试选项
    verbose: bool = False


class ResourceAllocator:
    """
    RSU资源分配器

    功能:
    - 为卸载任务分配RSU的计算和通信资源
    - 支持多种分配策略
    - 基于KKT条件的最优解

    核心优化问题:
    min Σ_j w_j · C_j / f_j   (加权完成时间最小化)
    s.t. Σ_j f_j ≤ F_max     (总资源约束)
         f_j ≥ 0             (非负约束)

    KKT最优解: f_j* = sqrt(w_j·C_j) / Σ_i sqrt(w_i·C_i) · F_max

    来源: docs/06_算法设计_资源分配与服务迁移.md §2.3
    """

    def __init__(self, config: Optional[ResourceAllocConfig] = None):
        """
        初始化资源分配器

        Args:
            config: 配置参数
        """
        self.config = config or ResourceAllocConfig()

        # 统计信息
        self.stats = {
            "total_allocations": 0,
            "total_cpu_allocated": 0.0,
            "total_bw_allocated": 0.0,
            "rejected_requests": 0,
        }

    def allocate_cpu(self, requests: List[ResourceRequest], total_cpu: float) -> Dict[int, float]:
        """
        分配CPU资源

        Args:
            requests: 资源请求列表
            total_cpu: 可用总CPU (Hz)

        Returns:
            allocations: {task_id: allocated_cpu}

        根据配置的策略进行分配
        """
        if not requests:
            return {}

        strategy = self.config.cpu_strategy

        if strategy == AllocationStrategy.EQUAL:
            return self._allocate_equal(requests, total_cpu, "cpu")
        elif strategy == AllocationStrategy.PROPORTIONAL:
            return self._allocate_proportional(requests, total_cpu, "cpu")
        elif strategy == AllocationStrategy.WEIGHTED_FAIR:
            return self._allocate_weighted_fair(requests, total_cpu)
        elif strategy == AllocationStrategy.PRIORITY:
            return self._allocate_priority(requests, total_cpu, "cpu")
        else:
            return self._allocate_equal(requests, total_cpu, "cpu")

    def allocate_bandwidth(
        self, requests: List[ResourceRequest], total_bw: float
    ) -> Dict[int, float]:
        """
        分配带宽资源

        Args:
            requests: 资源请求列表
            total_bw: 可用总带宽 (Hz)

        Returns:
            allocations: {task_id: allocated_bw}
        """
        if not requests:
            return {}

        strategy = self.config.bw_strategy

        if strategy == AllocationStrategy.EQUAL:
            return self._allocate_equal(requests, total_bw, "bw")
        elif strategy == AllocationStrategy.PROPORTIONAL:
            return self._allocate_proportional(requests, total_bw, "bw")
        elif strategy == AllocationStrategy.WEIGHTED_FAIR:
            return self._allocate_bw_weighted(requests, total_bw)
        elif strategy == AllocationStrategy.PRIORITY:
            return self._allocate_priority(requests, total_bw, "bw")
        else:
            return self._allocate_equal(requests, total_bw, "bw")

    def _allocate_equal(
        self, requests: List[ResourceRequest], total_resource: float, resource_type: str
    ) -> Dict[int, float]:
        """
        均匀分配策略

        每个任务获得相等的资源份额
        f_j = F_max / n
        """
        n = len(requests)
        per_task = total_resource / n

        # 应用最小和最大限制
        if resource_type == "cpu":
            per_task = max(per_task, self.config.min_cpu_per_task)
            per_task = min(per_task, total_resource * self.config.max_cpu_ratio)
        else:
            per_task = max(per_task, self.config.min_bw_per_task)
            per_task = min(per_task, total_resource * self.config.max_bw_ratio)

        allocations = {}
        for req in requests:
            allocations[req.task_id] = per_task
            self.stats["total_allocations"] += 1

            if resource_type == "cpu":
                self.stats["total_cpu_allocated"] += per_task
            else:
                self.stats["total_bw_allocated"] += per_task

        if self.config.verbose:
            print(f"[ResourceAlloc] 均匀分配 {resource_type}: {per_task/1e9:.2f}GHz/任务")

        return allocations

    def _allocate_proportional(
        self, requests: List[ResourceRequest], total_resource: float, resource_type: str
    ) -> Dict[int, float]:
        """
        按需求比例分配策略

        f_j = (demand_j / Σ demand_i) · F_max
        """
        # 计算总需求
        if resource_type == "cpu":
            demands = {req.task_id: req.compute_cycles for req in requests}
        else:
            demands = {req.task_id: req.data_size for req in requests}

        total_demand = sum(demands.values())
        if total_demand <= 0:
            return self._allocate_equal(requests, total_resource, resource_type)

        allocations = {}
        for req in requests:
            ratio = demands[req.task_id] / total_demand
            alloc = ratio * total_resource

            # 应用限制
            if resource_type == "cpu":
                alloc = max(alloc, self.config.min_cpu_per_task)
                alloc = min(alloc, total_resource * self.config.max_cpu_ratio)
            else:
                alloc = max(alloc, self.config.min_bw_per_task)
                alloc = min(alloc, total_resource * self.config.max_bw_ratio)

            allocations[req.task_id] = alloc
            self.stats["total_allocations"] += 1

            if resource_type == "cpu":
                self.stats["total_cpu_allocated"] += alloc
            else:
                self.stats["total_bw_allocated"] += alloc

        if self.config.verbose:
            print(f"[ResourceAlloc] 比例分配 {resource_type}: 总需求={total_demand:.2e}")

        return allocations

    def _allocate_weighted_fair(
        self, requests: List[ResourceRequest], total_cpu: float
    ) -> Dict[int, float]:
        """
        加权公平分配策略 (KKT最优解)

        优化问题:
        min Σ_j w_j · C_j / f_j
        s.t. Σ_j f_j ≤ F_max

        KKT最优解:
        f_j* = sqrt(w_j · C_j) / Σ_i sqrt(w_i · C_i) · F_max

        来源: docs/06_算法设计_资源分配与服务迁移.md §2.3 定理1
        """
        # 计算 sqrt(w_j * C_j) for each task
        sqrt_wc = {}
        for req in requests:
            sqrt_wc[req.task_id] = np.sqrt(req.weight * req.compute_cycles)

        # 计算归一化因子
        total_sqrt_wc = sum(sqrt_wc.values())

        if total_sqrt_wc <= 0:
            return self._allocate_equal(requests, total_cpu, "cpu")

        allocations = {}
        for req in requests:
            # KKT最优分配
            ratio = sqrt_wc[req.task_id] / total_sqrt_wc
            alloc = ratio * total_cpu

            # 应用约束
            alloc = max(alloc, self.config.min_cpu_per_task)
            alloc = min(alloc, total_cpu * self.config.max_cpu_ratio)

            allocations[req.task_id] = alloc
            req.allocated_cpu = alloc

            self.stats["total_allocations"] += 1
            self.stats["total_cpu_allocated"] += alloc

        if self.config.verbose:
            print(f"[ResourceAlloc] KKT最优分配:")
            for tid, alloc in allocations.items():
                print(f"  任务{tid}: {alloc/1e9:.2f}GHz ({alloc/total_cpu*100:.1f}%)")

        return allocations

    def _allocate_bw_weighted(
        self, requests: List[ResourceRequest], total_bw: float
    ) -> Dict[int, float]:
        """
        带宽加权分配（按数据量和权重）

        f_j* = sqrt(w_j · D_j) / Σ_i sqrt(w_i · D_i) · B_max
        """
        sqrt_wd = {}
        for req in requests:
            sqrt_wd[req.task_id] = np.sqrt(req.weight * req.data_size)

        total_sqrt_wd = sum(sqrt_wd.values())

        if total_sqrt_wd <= 0:
            return self._allocate_equal(requests, total_bw, "bw")

        allocations = {}
        for req in requests:
            ratio = sqrt_wd[req.task_id] / total_sqrt_wd
            alloc = ratio * total_bw

            alloc = max(alloc, self.config.min_bw_per_task)
            alloc = min(alloc, total_bw * self.config.max_bw_ratio)

            allocations[req.task_id] = alloc
            req.allocated_bw = alloc

            self.stats["total_allocations"] += 1
            self.stats["total_bw_allocated"] += alloc

        return allocations

    def _allocate_priority(
        self, requests: List[ResourceRequest], total_resource: float, resource_type: str
    ) -> Dict[int, float]:
        """
        优先级分配策略

        高优先级任务优先获得资源，直到满足需求或资源耗尽
        """
        # 按优先级排序（高优先级在前）
        sorted_requests = sorted(requests, key=lambda x: -x.priority)

        allocations = {}
        remaining = total_resource
        n_remaining = len(requests)

        for req in sorted_requests:
            if n_remaining <= 0:
                break

            # 基础分配 + 优先级加成
            base_alloc = remaining / n_remaining
            priority_bonus = 1.0 + req.priority * 0.1  # 每级优先级+10%
            alloc = base_alloc * priority_bonus

            # 约束
            if resource_type == "cpu":
                alloc = max(alloc, self.config.min_cpu_per_task)
                alloc = min(alloc, remaining, total_resource * self.config.max_cpu_ratio)
            else:
                alloc = max(alloc, self.config.min_bw_per_task)
                alloc = min(alloc, remaining, total_resource * self.config.max_bw_ratio)

            allocations[req.task_id] = alloc
            remaining -= alloc
            n_remaining -= 1

            self.stats["total_allocations"] += 1
            if resource_type == "cpu":
                self.stats["total_cpu_allocated"] += alloc
            else:
                self.stats["total_bw_allocated"] += alloc

        return allocations

    def allocate_joint(
        self, requests: List[ResourceRequest], total_cpu: float, total_bw: float
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        联合分配CPU和带宽

        Args:
            requests: 资源请求列表
            total_cpu: 可用总CPU (Hz)
            total_bw: 可用总带宽 (Hz)

        Returns:
            (cpu_allocs, bw_allocs): CPU和带宽分配结果
        """
        cpu_allocs = self.allocate_cpu(requests, total_cpu)
        bw_allocs = self.allocate_bandwidth(requests, total_bw)

        # 更新请求对象
        for req in requests:
            if req.task_id in cpu_allocs:
                req.allocated_cpu = cpu_allocs[req.task_id]
            if req.task_id in bw_allocs:
                req.allocated_bw = bw_allocs[req.task_id]

        return cpu_allocs, bw_allocs

    def compute_completion_time(self, request: ResourceRequest, channel_rate: float) -> float:
        """
        计算任务完成时间

        Args:
            request: 资源请求（已分配资源）
            channel_rate: 信道速率 (bps)

        Returns:
            completion_time: 完成时间 (s)

        公式: T = T_up + T_exe + T_down
              = D/R + C/f + D_result/R
        """
        if request.allocated_cpu <= 0 or request.allocated_bw <= 0:
            return float("inf")

        # 上传时延 (使用分配的带宽计算有效速率)
        # 简化：假设速率与带宽成正比
        effective_rate = channel_rate * (request.allocated_bw / 20e6)  # 归一化到20MHz

        t_up = request.data_size / effective_rate
        t_exe = request.compute_cycles / request.allocated_cpu
        t_down = request.data_size * 0.1 / effective_rate  # 结果约为输入的10%

        return t_up + t_exe + t_down

    def verify_deadline_feasibility(
        self,
        requests: List[ResourceRequest],
        total_cpu: float,
        total_bw: float,
        channel_rate: float,
    ) -> Tuple[bool, List[int]]:
        """
        验证截止时间可行性

        Args:
            requests: 资源请求列表
            total_cpu: 可用总CPU
            total_bw: 可用总带宽
            channel_rate: 信道速率

        Returns:
            (feasible, violated_tasks): 是否可行，违反截止时间的任务列表
        """
        # 先进行分配
        self.allocate_joint(requests, total_cpu, total_bw)

        violated = []
        for req in requests:
            completion_time = self.compute_completion_time(req, channel_rate)
            if completion_time > req.deadline:
                violated.append(req.task_id)

        return len(violated) == 0, violated

    def compute_weighted_completion_time(
        self, requests: List[ResourceRequest], channel_rate: float
    ) -> float:
        """
        计算加权完成时间（目标函数）

        Args:
            requests: 资源请求列表（已分配）
            channel_rate: 信道速率

        Returns:
            wct: 加权完成时间 Σ w_j · T_j
        """
        wct = 0.0
        for req in requests:
            t_j = self.compute_completion_time(req, channel_rate)
            wct += req.weight * t_j
        return wct

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return self.stats.copy()

    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            "total_allocations": 0,
            "total_cpu_allocated": 0.0,
            "total_bw_allocated": 0.0,
            "rejected_requests": 0,
        }


# ============================================================
# 辅助函数: 最优性分析
# ============================================================


def compute_kkt_optimal_allocation(
    weights: List[float], compute_cycles: List[float], total_cpu: float
) -> List[float]:
    """
    计算KKT最优分配（闭式解）

    Args:
        weights: 任务权重列表
        compute_cycles: 计算需求列表
        total_cpu: 总CPU资源

    Returns:
        allocations: 最优分配列表

    公式: f_j* = sqrt(w_j · C_j) / Σ sqrt(w_i · C_i) · F_max
    """
    n = len(weights)
    if n == 0:
        return []

    sqrt_wc = [np.sqrt(w * c) for w, c in zip(weights, compute_cycles)]
    total_sqrt_wc = sum(sqrt_wc)

    if total_sqrt_wc <= 0:
        return [total_cpu / n] * n

    return [(s / total_sqrt_wc) * total_cpu for s in sqrt_wc]


def verify_kkt_conditions(
    allocations: List[float],
    weights: List[float],
    compute_cycles: List[float],
    total_cpu: float,
    tolerance: float = 1e-6,
) -> Tuple[bool, dict]:
    """
    验证KKT条件是否满足

    Args:
        allocations: 分配结果
        weights: 权重
        compute_cycles: 计算需求
        total_cpu: 总资源
        tolerance: 容差

    Returns:
        (satisfied, details): 是否满足，详细信息
    """
    n = len(allocations)

    # 条件1: 资源约束 Σf_j ≤ F_max
    total_alloc = sum(allocations)
    constraint_satisfied = total_alloc <= total_cpu + tolerance

    # 条件2: 非负约束 f_j ≥ 0
    non_negative = all(f >= -tolerance for f in allocations)

    # 条件3: 梯度条件 (对于内点解)
    # ∂L/∂f_j = -w_j·C_j/f_j² + λ = 0
    # => w_j·C_j/f_j² = λ (常数)

    if all(f > tolerance for f in allocations):
        ratios = [w * c / (f**2) for w, c, f in zip(weights, compute_cycles, allocations)]
        gradient_consistent = max(ratios) - min(ratios) < tolerance * max(ratios)
    else:
        gradient_consistent = True  # 边界情况

    details = {
        "total_allocation": total_alloc,
        "constraint_slack": total_cpu - total_alloc,
        "constraint_satisfied": constraint_satisfied,
        "non_negative": non_negative,
        "gradient_consistent": gradient_consistent,
    }

    satisfied = constraint_satisfied and non_negative and gradient_consistent
    return satisfied, details


def compare_allocation_strategies(
    requests: List[ResourceRequest], total_cpu: float, channel_rate: float = 10e6
) -> dict:
    """
    比较不同分配策略的性能

    Args:
        requests: 资源请求列表
        total_cpu: 总CPU资源
        channel_rate: 信道速率

    Returns:
        comparison: 比较结果
    """
    strategies = [
        AllocationStrategy.EQUAL,
        AllocationStrategy.PROPORTIONAL,
        AllocationStrategy.WEIGHTED_FAIR,
    ]

    results = {}

    for strategy in strategies:
        # 创建请求副本
        requests_copy = [
            ResourceRequest(
                task_id=r.task_id,
                vehicle_id=r.vehicle_id,
                compute_cycles=r.compute_cycles,
                data_size=r.data_size,
                deadline=r.deadline,
                weight=r.weight,
            )
            for r in requests
        ]

        config = ResourceAllocConfig(cpu_strategy=strategy, verbose=False)
        allocator = ResourceAllocator(config)

        # 分配
        cpu_allocs = allocator.allocate_cpu(requests_copy, total_cpu)

        # 更新请求
        for req in requests_copy:
            req.allocated_cpu = cpu_allocs.get(req.task_id, 0)
            req.allocated_bw = 5e6  # 固定带宽

        # 计算目标
        wct = allocator.compute_weighted_completion_time(requests_copy, channel_rate)

        # 计算公平性 (Jain's Index)
        allocs = list(cpu_allocs.values())
        jain = (sum(allocs) ** 2) / (len(allocs) * sum(a**2 for a in allocs))

        results[strategy.value] = {
            "weighted_completion_time": wct,
            "jain_fairness": jain,
            "allocations": cpu_allocs,
        }

    return results


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 资源分配算法")
    print("=" * 70)

    # 创建测试请求
    requests = [
        ResourceRequest(
            task_id=0, vehicle_id=0, compute_cycles=1.0e8, data_size=4e6, deadline=0.08, weight=1.0
        ),
        ResourceRequest(
            task_id=1, vehicle_id=1, compute_cycles=1.5e8, data_size=5e6, deadline=0.08, weight=1.5
        ),
        ResourceRequest(
            task_id=2, vehicle_id=2, compute_cycles=0.8e8, data_size=3e6, deadline=0.08, weight=0.8
        ),
        ResourceRequest(
            task_id=3,
            vehicle_id=3,
            compute_cycles=1.2e8,
            data_size=4.5e6,
            deadline=0.08,
            weight=1.2,
        ),
    ]

    total_cpu = 20e9  # 20 GHz
    total_bw = 20e6  # 20 MHz

    # 测试1: KKT最优分配
    print("\n【测试1】KKT最优分配 (Weighted Fair)")
    config = ResourceAllocConfig(cpu_strategy=AllocationStrategy.WEIGHTED_FAIR, verbose=True)
    allocator = ResourceAllocator(config)

    cpu_allocs = allocator.allocate_cpu(requests, total_cpu)

    print(f"\n分配结果:")
    for tid, alloc in sorted(cpu_allocs.items()):
        req = requests[tid]
        print(
            f"  任务{tid}: {alloc/1e9:.2f}GHz (权重={req.weight}, 需求={req.compute_cycles/1e6:.0f}MC)"
        )

    # 测试2: 验证KKT条件
    print("\n【测试2】KKT条件验证")
    weights = [r.weight for r in requests]
    cycles = [r.compute_cycles for r in requests]
    allocs_list = [cpu_allocs[r.task_id] for r in requests]

    satisfied, details = verify_kkt_conditions(allocs_list, weights, cycles, total_cpu)
    print(f"KKT条件满足: {satisfied}")
    for key, value in details.items():
        print(f"  {key}: {value}")

    # 测试3: 均匀分配对比
    print("\n【测试3】均匀分配")
    config_equal = ResourceAllocConfig(cpu_strategy=AllocationStrategy.EQUAL)
    allocator_equal = ResourceAllocator(config_equal)

    cpu_allocs_equal = allocator_equal.allocate_cpu(requests, total_cpu)
    print(f"均匀分配: 每任务 {list(cpu_allocs_equal.values())[0]/1e9:.2f}GHz")

    # 测试4: 比例分配
    print("\n【测试4】比例分配")
    config_prop = ResourceAllocConfig(cpu_strategy=AllocationStrategy.PROPORTIONAL)
    allocator_prop = ResourceAllocator(config_prop)

    # 重置请求
    for r in requests:
        r.allocated_cpu = 0

    cpu_allocs_prop = allocator_prop.allocate_cpu(requests, total_cpu)
    print("比例分配结果:")
    for tid, alloc in sorted(cpu_allocs_prop.items()):
        print(f"  任务{tid}: {alloc/1e9:.2f}GHz")

    # 测试5: 联合分配
    print("\n【测试5】联合CPU+带宽分配")
    config_joint = ResourceAllocConfig(
        cpu_strategy=AllocationStrategy.WEIGHTED_FAIR, bw_strategy=AllocationStrategy.PROPORTIONAL
    )
    allocator_joint = ResourceAllocator(config_joint)

    # 重置
    for r in requests:
        r.allocated_cpu = 0
        r.allocated_bw = 0

    cpu_allocs_j, bw_allocs_j = allocator_joint.allocate_joint(requests, total_cpu, total_bw)

    print("联合分配结果:")
    for req in requests:
        print(
            f"  任务{req.task_id}: CPU={req.allocated_cpu/1e9:.2f}GHz, "
            f"BW={req.allocated_bw/1e6:.2f}MHz"
        )

    # 测试6: 完成时间计算
    print("\n【测试6】完成时间计算")
    channel_rate = 10e6  # 10 Mbps

    for req in requests:
        t = allocator_joint.compute_completion_time(req, channel_rate)
        deadline_ok = "✓" if t <= req.deadline else "✗"
        print(
            f"  任务{req.task_id}: T={t*1000:.2f}ms (deadline={req.deadline*1000:.0f}ms) {deadline_ok}"
        )

    # 测试7: 策略对比
    print("\n【测试7】分配策略对比")

    # 重新创建请求
    requests_compare = [
        ResourceRequest(
            task_id=i,
            vehicle_id=i,
            compute_cycles=r.compute_cycles,
            data_size=r.data_size,
            deadline=r.deadline,
            weight=r.weight,
        )
        for i, r in enumerate(requests)
    ]

    comparison = compare_allocation_strategies(requests_compare, total_cpu)

    print(f"{'策略':<15} | {'加权完成时间(ms)':<18} | {'Jain公平性':<12}")
    print("-" * 50)
    for strategy, result in comparison.items():
        wct_ms = result["weighted_completion_time"] * 1000
        jain = result["jain_fairness"]
        print(f"{strategy:<15} | {wct_ms:<18.2f} | {jain:<12.4f}")

    # 测试8: 截止时间可行性
    print("\n【测试8】截止时间可行性验证")

    requests_feas = [
        ResourceRequest(
            task_id=i,
            vehicle_id=i,
            compute_cycles=r.compute_cycles,
            data_size=r.data_size,
            deadline=r.deadline,
            weight=r.weight,
        )
        for i, r in enumerate(requests)
    ]

    allocator_feas = ResourceAllocator(ResourceAllocConfig())
    feasible, violated = allocator_feas.verify_deadline_feasibility(
        requests_feas, total_cpu, total_bw, channel_rate
    )

    print(f"可行性: {feasible}")
    if violated:
        print(f"违反截止时间的任务: {violated}")

    # 测试9: 统计信息
    print("\n【测试9】统计信息")
    stats = allocator_joint.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("资源分配测试完成!")
    print("=" * 70)


# 运行测试: python3 -m simulation.algorithms.resource_alloc
