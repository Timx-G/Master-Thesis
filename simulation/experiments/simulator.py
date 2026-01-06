"""
仿真引擎
文件: simulation/experiments/simulator.py

功能说明:
- 整合所有模块的仿真主循环
- 管理仿真状态和时间推进
- 收集和汇总性能指标

来源: docs/07_仿真实验设计.md §3.2
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 算法
from simulation.algorithms.cpa import CentralizedPhaseAllocator, CPAConfig
from simulation.algorithms.dpa import DistributedPhaseAdjuster, DPAConfig
from simulation.algorithms.lyapunov import LyapunovConfig, LyapunovOptimizer, OffloadDecision
from simulation.algorithms.resource_alloc import (
    ResourceAllocator,
    ResourceAllocConfig,
    ResourceRequest,
)
from simulation.core.rsu import RSU
from simulation.core.task import CPMTask, TaskFactory, TaskStatus

# 核心模块
from simulation.core.vehicle import Vehicle

# 生成器 - 使用实际类名
from simulation.generators.task_generator import CPMParameters, CPMTaskGenerator

# 指标 - 注意构造函数参数
from simulation.metrics.aoi import AoICalculator
from simulation.metrics.deadline import DeadlineCalculator
from simulation.metrics.energy import EnergyCalculator
from simulation.metrics.fairness import FairnessCalculator, calculate_jain_fairness_index


class PhaseAllocMethod(Enum):
    """相位分配方法"""

    CPA = "cpa"  # 集中式
    DPA = "dpa"  # 分布式
    NONE = "none"  # 无相位对齐


@dataclass
class SimulatorConfig:
    """仿真器配置"""

    # 场景参数
    road_length: float = 3000.0
    num_lanes: int = 4
    lane_width: float = 3.5
    num_vehicles: int = 50
    num_rsus: int = 5
    rsu_coverage: float = 400.0
    rsu_spacing: float = 600.0

    # 时间参数
    duration: float = 100.0  # 仿真时长 (s)
    dt: float = 0.1  # 时间步长 (s)
    warmup_time: float = 10.0  # 预热时间 (s)

    # 车辆参数
    speed_mean: float = 25.0
    speed_std: float = 5.0
    speed_min: float = 10.0
    speed_max: float = 35.0
    local_cpu_freq: float = 2.0e9

    # RSU参数
    rsu_cpu_capacity: float = 20.0e9
    rsu_bandwidth: float = 20.0e6

    # CPM任务参数
    cpm_data_base: float = 4.0e6
    cpm_data_per_object: float = 1.6e5
    cpm_compute_base: float = 1.0e8
    cpm_compute_per_object: float = 2.0e7
    cpm_num_objects_mean: int = 15
    cpm_num_objects_std: int = 5
    cpm_deadline: float = 0.08  # 80ms

    # 算法选择
    phase_method: PhaseAllocMethod = PhaseAllocMethod.CPA
    use_lyapunov: bool = True

    # Lyapunov参数
    lyapunov_V: float = 1000.0

    # 随机种子
    random_seed: int = 42

    # 调试
    verbose: bool = False
    log_interval: int = 100  # 每多少时隙打印一次

    @classmethod
    def from_yaml(cls, path: str = "simulation/config/parameters.yaml") -> "SimulatorConfig":
        """从YAML文件加载配置"""
        import yaml

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠  YAML配置文件未找到: {path}, 使用默认配置")
            return cls()

        # 字段映射：YAML -> SimulatorConfig
        field_map = {
            # network
            "road_length": "road_length",
            "num_lanes": "num_lanes",
            "lane_width": "lane_width",
            "rsu_count": "num_rsus",
            "rsu_coverage_radius": "rsu_coverage",
            "rsu_spacing": "rsu_spacing",
            # vehicle
            "count": "num_vehicles",
            "speed_mean": "speed_mean",
            "speed_std": "speed_std",
            "speed_min": "speed_min",
            "speed_max": "speed_max",
            "local_cpu_freq": "local_cpu_freq",
            # rsu
            "cpu_capacity": "rsu_cpu_capacity",
            "backhaul_bandwidth": "rsu_bandwidth",
            # cpm
            "data_base": "cpm_data_base",
            "data_per_object": "cpm_data_per_object",
            "compute_base": "cpm_compute_base",
            "compute_per_object": "cpm_compute_per_object",
            "num_objects_mean": "cpm_num_objects_mean",
            "num_objects_std": "cpm_num_objects_std",
            "deadline": "cpm_deadline",
            # simulation
            "duration": "duration",
            "dt": "dt",
            # lyapunov
            "V": "lyapunov_V",
        }

        # 扁平化并映射
        params = {}
        for section, values in config.items():
            if isinstance(values, dict):
                for yaml_key, value in values.items():
                    config_key = field_map.get(yaml_key)
                    if config_key and hasattr(cls, config_key):
                        params[config_key] = value

        return cls(**params)


@dataclass
class SlotResult:
    """单时隙结果"""

    slot: int
    time: float
    num_tasks_generated: int = 0
    num_tasks_completed: int = 0
    num_tasks_dropped: int = 0
    num_offloaded: int = 0
    num_local: int = 0
    avg_delay: float = 0.0
    avg_energy: float = 0.0
    avg_aoi: float = 0.0


class Simulator:
    """
    V2X-MEC仿真器

    功能:
    - 初始化仿真环境（车辆、RSU、信道）
    - 运行时隙级仿真循环
    - 收集和汇总性能指标
    """

    def __init__(self, config: SimulatorConfig):
        """
        初始化仿真器

        Args:
            config: 仿真配置
        """
        self.config = config
        np.random.seed(config.random_seed)

        # 时间状态
        self.current_slot = 0
        self.current_time = 0.0
        self.total_slots = int(config.duration / config.dt)
        self.warmup_slots = int(config.warmup_time / config.dt)

        # 初始化实体
        self.vehicles: List[Vehicle] = []
        self.rsus: List[RSU] = []
        self._init_entities()

        # 初始化任务生成器 - 使用配置中的参数
        cpm_params = CPMParameters(
            T_check=config.dt,
            pos_threshold=4.0,
            speed_threshold=0.5,
            heading_threshold=4.0,
            T_max=1.0,
            data_base=config.cpm_data_base,
            data_per_object=config.cpm_data_per_object,
            compute_base=config.cpm_compute_base,
            compute_per_object=config.cpm_compute_per_object,
            num_objects_mean=config.cpm_num_objects_mean,
            num_objects_std=config.cpm_num_objects_std,
            deadline=config.cpm_deadline,
        )
        self.task_generator = CPMTaskGenerator(params=cpm_params, random_seed=config.random_seed)

        # 初始化算法模块
        self._init_algorithms()

        # 初始化指标收集器 - 使用配置中的参数
        self.aoi_calculator = AoICalculator(num_vehicles=config.num_vehicles, dt=config.dt)
        self.energy_calculator = EnergyCalculator(num_vehicles=config.num_vehicles)
        self.deadline_calculator = DeadlineCalculator(
            num_vehicles=config.num_vehicles, deadline=config.cpm_deadline
        )

        # 结果记录
        self.slot_results: List[SlotResult] = []

        # 统计
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "dropped_tasks": 0,
            "offloaded_tasks": 0,
            "local_tasks": 0,
        }

    def _init_entities(self):
        """初始化车辆和RSU"""
        cfg = self.config

        # 初始化RSU
        for i in range(cfg.num_rsus):
            x = (i + 0.5) * cfg.rsu_spacing
            y = cfg.lane_width * cfg.num_lanes / 2

            rsu = RSU(
                id=i,
                position=np.array([x, y]),
                coverage_radius=cfg.rsu_coverage,
                cpu_capacity=cfg.rsu_cpu_capacity,
                bandwidth_total=cfg.rsu_bandwidth,
            )
            self.rsus.append(rsu)

        # 初始化车辆
        for i in range(cfg.num_vehicles):
            x = np.random.uniform(0, cfg.road_length)
            lane = np.random.randint(0, cfg.num_lanes)
            y = (lane + 0.5) * cfg.lane_width
            speed = np.random.uniform(cfg.speed_min, cfg.speed_max)

            vehicle = Vehicle(
                id=i,
                position=np.array([x, y]),
                speed=speed,
                lane=lane,
                local_cpu_freq=cfg.local_cpu_freq,
            )
            self.vehicles.append(vehicle)

    def _init_algorithms(self):
        """初始化算法模块"""
        cfg = self.config

        # 相位分配
        if cfg.phase_method == PhaseAllocMethod.CPA:
            self.phase_allocator = CentralizedPhaseAllocator(CPAConfig(T_check=cfg.dt))
        elif cfg.phase_method == PhaseAllocMethod.DPA:
            self.phase_allocator = DistributedPhaseAdjuster(DPAConfig(T_check=cfg.dt))
        else:
            self.phase_allocator = None

        # Lyapunov优化器
        if cfg.use_lyapunov:
            self.lyapunov = LyapunovOptimizer(LyapunovConfig(V=cfg.lyapunov_V, T_check=cfg.dt))
        else:
            self.lyapunov = None

        # 资源分配
        self.resource_allocator = ResourceAllocator(ResourceAllocConfig())

        # 初始化相位分配
        self._allocate_initial_phases()

    def _allocate_initial_phases(self):
        """初始化相位分配"""
        if self.phase_allocator is None:
            return

        # 更新RSU覆盖
        for rsu in self.rsus:
            covered_ids = []
            for v in self.vehicles:
                if rsu.is_vehicle_covered(v.position):
                    covered_ids.append(v.id)
                    v.connected_rsu_id = rsu.id

            rsu.covered_vehicle_ids = set(covered_ids)

            # 分配相位
            if isinstance(self.phase_allocator, CentralizedPhaseAllocator):
                phases = self.phase_allocator.allocate_phases(rsu.id, covered_ids)
            else:
                for vid in covered_ids:
                    self.phase_allocator.insert_vehicle(rsu.id, vid)
                phases = self.phase_allocator.get_rsu_phases(rsu.id)

            # 更新车辆相位
            for vid, phase in phases.items():
                for v in self.vehicles:
                    if v.id == vid:
                        v.phase_offset = phase
                        break

    def step(self) -> SlotResult:
        """
        执行一个时隙的仿真

        Returns:
            result: 本时隙结果
        """
        slot = self.current_slot
        current_time = self.current_time
        cfg = self.config

        result = SlotResult(slot=slot, time=current_time)

        # 1. 更新车辆位置
        for v in self.vehicles:
            v.update_position(cfg.dt, cfg.road_length)
            v.update_speed(cfg.dt, cfg.speed_std, cfg.speed_min, cfg.speed_max)

        # 2. 更新RSU覆盖和相位
        self._update_coverage_and_phases()

        # 3. 生成任务 - 使用CPMTaskGenerator的接口
        tasks = self._generate_tasks(current_time)
        result.num_tasks_generated = len(tasks)
        self.stats["total_tasks"] += len(tasks)

        # 4. 处理任务（卸载决策 + 资源分配）
        completed, dropped, offloaded, local = self._process_tasks(tasks, current_time)
        result.num_tasks_completed = completed
        result.num_tasks_dropped = dropped
        result.num_offloaded = offloaded
        result.num_local = local

        self.stats["completed_tasks"] += completed
        self.stats["dropped_tasks"] += dropped
        self.stats["offloaded_tasks"] += offloaded
        self.stats["local_tasks"] += local

        # 5. 更新AoI - 使用正确的方法
        for v in self.vehicles:
            # 检查该车辆是否有任务完成
            vehicle_task = None
            for t in tasks:
                if t.vehicle_id == v.id:
                    vehicle_task = t
                    break

            if vehicle_task and vehicle_task.status == TaskStatus.COMPLETED:
                self.aoi_calculator.update_slot(
                    vehicle_id=v.id,
                    task_completed=True,
                    generation_time=vehicle_task.generation_time,
                    completion_time=vehicle_task.completion_time,
                    total_delay=vehicle_task.get_total_delay(),
                )
            else:
                self.aoi_calculator.update_slot(vehicle_id=v.id, task_completed=False)

        # 6. 计算本时隙指标
        if slot >= self.warmup_slots:
            delays = [t.get_total_delay() for t in tasks if t.status == TaskStatus.COMPLETED]
            energies = [t.energy_consumption for t in tasks if t.status == TaskStatus.COMPLETED]

            result.avg_delay = float(np.mean(delays)) if delays else 0.0
            result.avg_energy = float(np.mean(energies)) if energies else 0.0

            # 系统平均AoI
            aoi_stats = self.aoi_calculator.get_statistics()
            result.avg_aoi = aoi_stats.get("system_avg_aoi", 0.0)

        # 记录结果
        self.slot_results.append(result)

        # 推进时间
        self.current_slot += 1
        self.current_time += cfg.dt

        # 打印进度
        if cfg.verbose and slot % cfg.log_interval == 0:
            print(
                f"[Slot {slot}/{self.total_slots}] "
                f"tasks={result.num_tasks_generated}, "
                f"completed={result.num_tasks_completed}, "
                f"AoI={result.avg_aoi*1000:.1f}ms"
            )

        return result

    def _update_coverage_and_phases(self):
        """更新RSU覆盖和相位分配"""
        for rsu in self.rsus:
            old_covered = rsu.covered_vehicle_ids.copy()
            new_covered = rsu.update_coverage(self.vehicles)

            # 处理车辆进入/离开
            entered = new_covered - old_covered
            left = old_covered - new_covered

            # 更新相位
            if self.phase_allocator is not None:
                for vid in entered:
                    if isinstance(self.phase_allocator, CentralizedPhaseAllocator):
                        phases = self.phase_allocator.update_phases_on_vehicle_join(
                            rsu.id, vid, list(old_covered)
                        )
                    else:
                        self.phase_allocator.insert_vehicle(rsu.id, vid, self.current_time)
                        phases = self.phase_allocator.get_rsu_phases(rsu.id)

                    # 更新车辆
                    for v in self.vehicles:
                        if v.id in phases:
                            v.phase_offset = phases[v.id]
                            v.connected_rsu_id = rsu.id

                for vid in left:
                    if isinstance(self.phase_allocator, CentralizedPhaseAllocator):
                        self.phase_allocator.update_phases_on_vehicle_leave(
                            rsu.id, vid, list(new_covered)
                        )
                    else:
                        self.phase_allocator.remove_vehicle(rsu.id, vid)

    def _generate_tasks(self, current_time: float) -> List[CPMTask]:
        """生成本时隙的任务"""
        # 构建CPMTaskGenerator需要的状态字典
        vehicle_states = {}
        for v in self.vehicles:
            # (position, speed, heading) 格式
            vehicle_states[v.id] = (v.position, v.speed, v.heading)

        # 使用CPMTaskGenerator的check_and_generate方法
        tasks = self.task_generator.check_and_generate(current_time, vehicle_states)

        return tasks

    def _process_tasks(
        self, tasks: List[CPMTask], current_time: float
    ) -> Tuple[int, int, int, int]:
        """
        处理任务

        Returns:
            (completed, dropped, offloaded, local)
        """
        completed = 0
        dropped = 0
        offloaded = 0
        local = 0

        for task in tasks:
            vehicle = self._get_vehicle(task.vehicle_id)
            if vehicle is None:
                continue

            rsu = self._get_connected_rsu(vehicle)

            # 做出卸载决策
            if self.lyapunov is not None and rsu is not None:
                # 估算信道速率
                channel_rate = self._estimate_channel_rate(vehicle, rsu)

                decision, info = self.lyapunov.make_decision(
                    vehicle_id=vehicle.id,
                    task_data_size=task.data_size,
                    task_compute_cycles=task.compute_cycles,
                    task_result_size=task.result_size,
                    local_cpu_freq=vehicle.local_cpu_freq,
                    channel_rate=channel_rate,
                    rsu_cpu_freq=5e9,  # 简化：固定分配
                    current_aoi=vehicle.current_aoi,
                )

                task.offload_decision = decision == OffloadDecision.OFFLOAD
            else:
                # 无Lyapunov时默认本地
                task.offload_decision = False

            # 计算时延和能耗
            if task.offload_decision and rsu is not None:
                self._process_offload_task(task, vehicle, rsu)
                offloaded += 1
            else:
                self._process_local_task(task, vehicle)
                local += 1

            # 检查截止时间
            total_delay = task.get_total_delay()
            if total_delay <= task.deadline:
                task.mark_as_completed(current_time + total_delay)
                completed += 1

                # 记录指标
                self.deadline_calculator.record_task(
                    vehicle_id=task.vehicle_id, total_delay=total_delay, completed=True
                )
            else:
                task.mark_as_dropped()
                dropped += 1
                self.deadline_calculator.record_task(
                    vehicle_id=task.vehicle_id, total_delay=total_delay, completed=False
                )

        return completed, dropped, offloaded, local

    def _process_local_task(self, task: CPMTask, vehicle: Vehicle):
        """处理本地任务"""
        task.computation_delay = task.compute_cycles / vehicle.local_cpu_freq
        task.energy_consumption = (
            self.energy_calculator.params.kappa * (vehicle.local_cpu_freq**2) * task.compute_cycles
        )

    def _process_offload_task(self, task: CPMTask, vehicle: Vehicle, rsu: RSU):
        """处理卸载任务"""
        # 估算信道速率
        channel_rate = self._estimate_channel_rate(vehicle, rsu)

        # 简化资源分配
        allocated_freq = 5e9  # 固定分配5GHz

        # 计算时延
        task.transmission_delay = task.data_size / channel_rate
        task.computation_delay = task.compute_cycles / allocated_freq
        task.download_delay = task.result_size / channel_rate

        # 计算能耗
        task.energy_consumption = (
            self.energy_calculator.params.tx_power * task.transmission_delay
            + self.energy_calculator.params.idle_power
            * (task.computation_delay + task.download_delay)
        )

        task.target_rsu_id = rsu.id
        task.allocated_cpu_freq = allocated_freq

    def _estimate_channel_rate(self, vehicle: Vehicle, rsu: RSU) -> float:
        """估算信道速率（简化模型）"""
        distance = vehicle.get_distance_to_rsu(rsu.position)

        # 简化路径损耗模型
        if distance < 10:
            distance = 10

        path_loss_db = 128.1 + 37.6 * np.log10(distance / 1000)
        path_loss = 10 ** (-path_loss_db / 10)

        # SNR
        tx_power = 0.2  # W
        noise_power = 1e-13  # W
        snr = tx_power * path_loss / noise_power

        # Shannon容量
        bandwidth = 10e6  # 分配10MHz
        rate = bandwidth * np.log2(1 + snr)

        return max(rate, 1e6)  # 最小1Mbps

    def _get_vehicle(self, vehicle_id: int) -> Optional[Vehicle]:
        """获取车辆对象"""
        for v in self.vehicles:
            if v.id == vehicle_id:
                return v
        return None

    def _get_connected_rsu(self, vehicle: Vehicle) -> Optional[RSU]:
        """获取车辆连接的RSU"""
        if vehicle.connected_rsu_id is None:
            return None
        for rsu in self.rsus:
            if rsu.id == vehicle.connected_rsu_id:
                return rsu
        return None

    def run(self, progress_callback=None) -> dict:
        """
        运行完整仿真

        Args:
            progress_callback: 进度回调函数

        Returns:
            results: 仿真结果
        """
        start_time = time.time()

        while self.current_slot < self.total_slots:
            self.step()

            if progress_callback:
                progress = self.current_slot / self.total_slots
                progress_callback(progress)

        elapsed = time.time() - start_time

        # 汇总结果
        results = self.get_results()
        results["elapsed_time"] = elapsed

        return results

    def get_results(self) -> dict:
        """
        获取仿真结果

        Returns:
            results: 结果字典
        """
        # 过滤预热期
        valid_results = [r for r in self.slot_results if r.slot >= self.warmup_slots]

        if not valid_results:
            return {"error": "No valid results after warmup"}

        # 汇总指标
        avg_delay = np.mean([r.avg_delay for r in valid_results if r.avg_delay > 0])
        avg_energy = np.mean([r.avg_energy for r in valid_results if r.avg_energy > 0])
        avg_aoi = np.mean([r.avg_aoi for r in valid_results])

        # 任务统计
        total_tasks = self.stats["total_tasks"]
        completed = self.stats["completed_tasks"]

        # 公平性 - 使用正确的方法
        vehicle_aois = []
        for vid in range(self.config.num_vehicles):
            if vid in self.aoi_calculator.vehicle_states:
                state = self.aoi_calculator.vehicle_states[vid]
                if state.aoi_history:
                    vehicle_aois.append(np.mean(state.aoi_history))

        if vehicle_aois:
            jain_fairness = calculate_jain_fairness_index(vehicle_aois)
        else:
            jain_fairness = 1.0

        return {
            "config": {
                "num_vehicles": self.config.num_vehicles,
                "num_rsus": self.config.num_rsus,
                "duration": self.config.duration,
                "phase_method": self.config.phase_method.value,
            },
            "performance": {
                "avg_delay_ms": float(avg_delay * 1000) if not np.isnan(avg_delay) else 0,
                "avg_energy_mJ": float(avg_energy * 1000) if not np.isnan(avg_energy) else 0,
                "avg_aoi_ms": float(avg_aoi * 1000) if not np.isnan(avg_aoi) else 0,
                "deadline_rate": completed / total_tasks if total_tasks > 0 else 0,
                "jain_fairness": jain_fairness,
            },
            "statistics": self.stats.copy(),
            "offload_ratio": (
                self.stats["offloaded_tasks"] / total_tasks if total_tasks > 0 else 0
            ),
        }

    def reset(self):
        """重置仿真器"""
        self.current_slot = 0
        self.current_time = 0.0
        self.slot_results.clear()
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "dropped_tasks": 0,
            "offloaded_tasks": 0,
            "local_tasks": 0,
        }

        # 重新初始化
        np.random.seed(self.config.random_seed)
        self.vehicles.clear()
        self.rsus.clear()
        self._init_entities()
        self._allocate_initial_phases()
        self.task_generator.reset()


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 Simulator 仿真引擎")
    print("=" * 70)

    # 创建配置
    config = SimulatorConfig(
        num_vehicles=20,
        num_rsus=3,
        duration=10.0,  # 短时间测试
        dt=0.1,
        warmup_time=1.0,
        verbose=True,
        log_interval=10,
    )

    print(f"\n配置: {config.num_vehicles}车, {config.num_rsus}个RSU, {config.duration}秒")

    # 创建仿真器
    sim = Simulator(config)

    # 运行仿真
    print("\n开始仿真...")
    results = sim.run()

    # 打印结果
    print("\n" + "=" * 70)
    print("仿真结果")
    print("=" * 70)

    print("\n【性能指标】")
    perf = results["performance"]
    print(f"  平均时延: {perf['avg_delay_ms']:.2f} ms")
    print(f"  平均能耗: {perf['avg_energy_mJ']:.4f} mJ")
    print(f"  平均AoI: {perf['avg_aoi_ms']:.2f} ms")
    print(f"  截止时间满足率: {perf['deadline_rate']:.2%}")
    print(f"  Jain公平性: {perf['jain_fairness']:.4f}")

    print("\n【统计信息】")
    stats = results["statistics"]
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  完成任务: {stats['completed_tasks']}")
    print(f"  丢弃任务: {stats['dropped_tasks']}")
    print(f"  卸载任务: {stats['offloaded_tasks']}")
    print(f"  本地任务: {stats['local_tasks']}")
    print(f"  卸载比例: {results['offload_ratio']:.2%}")

    print(f"\n运行时间: {results['elapsed_time']:.2f}秒")

    print("\n" + "=" * 70)


# 运行测试: python3 -m simulation.experiments.simulator
