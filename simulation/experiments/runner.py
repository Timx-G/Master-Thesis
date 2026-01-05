"""
实验运行器
文件: simulation/experiments/runner.py

功能说明:
- 批量运行实验
- 对比不同算法
- 参数敏感性分析
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from simulation.baselines import get_all_baselines
from simulation.experiments.simulator import PhaseAllocMethod, Simulator, SimulatorConfig


@dataclass
class ExperimentConfig:
    """实验配置"""

    name: str = "default"
    description: str = ""

    # 基础仿真配置
    num_vehicles: int = 50
    num_rsus: int = 5
    duration: float = 100.0
    warmup_time: float = 10.0

    # 重复次数
    num_runs: int = 5

    # 随机种子基础值
    base_seed: int = 42

    # 输出目录
    output_dir: str = "./results"

    # 是否保存详细结果
    save_details: bool = True


class ExperimentRunner:
    """
    实验运行器

    功能:
    - 运行单个实验或批量实验
    - 算法对比实验
    - 参数敏感性实验
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        初始化

        Args:
            config: 实验配置
        """
        self.config = config or ExperimentConfig()
        self.results: List[dict] = []

    def run_single(self, sim_config: SimulatorConfig, run_id: int = 0) -> dict:
        """
        运行单次仿真

        Args:
            sim_config: 仿真配置
            run_id: 运行ID

        Returns:
            result: 仿真结果
        """
        # 设置种子
        sim_config.random_seed = self.config.base_seed + run_id

        # 创建仿真器
        sim = Simulator(sim_config)

        # 运行
        result = sim.run()
        result["run_id"] = run_id

        return result

    def run_multiple(
        self,
        sim_config: SimulatorConfig,
        num_runs: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        多次运行并统计

        Args:
            sim_config: 仿真配置
            num_runs: 运行次数
            progress_callback: 进度回调

        Returns:
            summary: 汇总结果
        """
        num_runs = num_runs or self.config.num_runs

        all_results = []

        for i in range(num_runs):
            result = self.run_single(sim_config, run_id=i)
            all_results.append(result)

            if progress_callback:
                progress_callback(i + 1, num_runs)

        # 汇总统计
        summary = self._summarize_results(all_results)
        summary["num_runs"] = num_runs
        summary["all_results"] = all_results

        return summary

    def _summarize_results(self, results: List[dict]) -> dict:
        """汇总多次运行结果"""
        if not results:
            return {}

        # 提取性能指标
        delays = [r["performance"]["avg_delay_ms"] for r in results]
        energies = [r["performance"]["avg_energy_mJ"] for r in results]
        aois = [r["performance"]["avg_aoi_ms"] for r in results]
        deadline_rates = [r["performance"]["deadline_rate"] for r in results]
        fairness = [r["performance"]["jain_fairness"] for r in results]

        return {
            "performance": {
                "avg_delay_ms": {
                    "mean": np.mean(delays),
                    "std": np.std(delays),
                    "min": np.min(delays),
                    "max": np.max(delays),
                },
                "avg_energy_mJ": {
                    "mean": np.mean(energies),
                    "std": np.std(energies),
                },
                "avg_aoi_ms": {
                    "mean": np.mean(aois),
                    "std": np.std(aois),
                },
                "deadline_rate": {
                    "mean": np.mean(deadline_rates),
                    "std": np.std(deadline_rates),
                },
                "jain_fairness": {
                    "mean": np.mean(fairness),
                    "std": np.std(fairness),
                },
            }
        }

    def run_algorithm_comparison(
        self, algorithms: Optional[List[str]] = None, verbose: bool = True
    ) -> dict:
        """
        运行算法对比实验

        Args:
            algorithms: 要对比的算法列表
            verbose: 是否打印进度

        Returns:
            comparison: 对比结果
        """
        if algorithms is None:
            algorithms = ["PAPO_CPA", "PAPO_DPA", "NoPhaseAlign", "LocalOnly", "FullOffload"]

        comparison = {}

        for alg_name in algorithms:
            if verbose:
                print(f"\n运行算法: {alg_name}")

            # 创建对应配置
            sim_config = self._create_config_for_algorithm(alg_name)

            # 运行
            def progress(i, n):
                if verbose:
                    print(f"  运行 {i}/{n}...")

            result = self.run_multiple(sim_config, progress_callback=progress)
            comparison[alg_name] = result

            if verbose:
                perf = result["performance"]
                print(
                    f"  结果: AoI={perf['avg_aoi_ms']['mean']:.2f}ms, "
                    f"Delay={perf['avg_delay_ms']['mean']:.2f}ms"
                )

        return comparison

    def _create_config_for_algorithm(self, alg_name: str) -> SimulatorConfig:
        """为算法创建配置"""
        cfg = self.config

        base_config = SimulatorConfig(
            num_vehicles=cfg.num_vehicles,
            num_rsus=cfg.num_rsus,
            duration=cfg.duration,
            warmup_time=cfg.warmup_time,
        )

        if alg_name == "PAPO_CPA":
            base_config.phase_method = PhaseAllocMethod.CPA
            base_config.use_lyapunov = True
        elif alg_name == "PAPO_DPA":
            base_config.phase_method = PhaseAllocMethod.DPA
            base_config.use_lyapunov = True
        elif alg_name == "NoPhaseAlign":
            base_config.phase_method = PhaseAllocMethod.NONE
            base_config.use_lyapunov = True
        elif alg_name == "LocalOnly":
            base_config.phase_method = PhaseAllocMethod.NONE
            base_config.use_lyapunov = False
        elif alg_name == "FullOffload":
            base_config.phase_method = PhaseAllocMethod.CPA
            base_config.use_lyapunov = False

        return base_config

    def run_parameter_sensitivity(
        self, param_name: str, param_values: List[Any], verbose: bool = True
    ) -> dict:
        """
        参数敏感性分析

        Args:
            param_name: 参数名称
            param_values: 参数值列表
            verbose: 是否打印进度

        Returns:
            sensitivity: 敏感性分析结果
        """
        sensitivity = {"param_name": param_name, "results": []}

        for value in param_values:
            if verbose:
                print(f"\n{param_name} = {value}")

            # 创建配置
            sim_config = SimulatorConfig(
                num_vehicles=self.config.num_vehicles,
                num_rsus=self.config.num_rsus,
                duration=self.config.duration,
            )

            # 设置参数
            if hasattr(sim_config, param_name):
                setattr(sim_config, param_name, value)

            # 运行
            result = self.run_multiple(sim_config, num_runs=3)
            result["param_value"] = value

            sensitivity["results"].append(result)

            if verbose:
                perf = result["performance"]
                print(f"  AoI={perf['avg_aoi_ms']['mean']:.2f}ms")

        return sensitivity

    def save_results(self, results: dict, filename: str):
        """保存结果到文件"""
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)

        # 转换为可序列化格式
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(filepath, "w") as f:
            json.dump(convert(results), f, indent=2)

        print(f"结果已保存到: {filepath}")


def print_comparison_table(comparison: dict):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("算法对比结果")
    print("=" * 80)

    header = f"{'算法':<15} | {'AoI(ms)':<12} | {'Delay(ms)':<12} | {'Energy(mJ)':<12} | {'Deadline%':<10}"
    print(header)
    print("-" * 80)

    for alg_name, result in comparison.items():
        perf = result["performance"]
        aoi = f"{perf['avg_aoi_ms']['mean']:.2f}±{perf['avg_aoi_ms']['std']:.2f}"
        delay = f"{perf['avg_delay_ms']['mean']:.2f}±{perf['avg_delay_ms']['std']:.2f}"
        energy = f"{perf['avg_energy_mJ']['mean']:.4f}"
        deadline = f"{perf['deadline_rate']['mean']*100:.1f}%"

        print(f"{alg_name:<15} | {aoi:<12} | {delay:<12} | {energy:<12} | {deadline:<10}")

    print("=" * 80)


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 ExperimentRunner")
    print("=" * 70)

    # 创建实验配置（简短测试）
    exp_config = ExperimentConfig(
        name="quick_test",
        num_vehicles=20,
        num_rsus=3,
        duration=10.0,  # 短时间
        num_runs=2,  # 少量重复
    )

    runner = ExperimentRunner(exp_config)

    # 测试单次运行
    print("\n【测试1】单次运行")
    sim_config = SimulatorConfig(
        num_vehicles=20,
        num_rsus=3,
        duration=10.0,
        warmup_time=1.0,
    )
    result = runner.run_single(sim_config)
    print(f"  完成，AoI={result['performance']['avg_aoi_ms']:.2f}ms")

    # 测试多次运行
    print("\n【测试2】多次运行统计")
    summary = runner.run_multiple(sim_config, num_runs=2)
    perf = summary["performance"]
    print(f"  AoI: {perf['avg_aoi_ms']['mean']:.2f} ± {perf['avg_aoi_ms']['std']:.2f} ms")

    # 测试算法对比（简化）
    print("\n【测试3】算法对比（简化）")
    comparison = runner.run_algorithm_comparison(
        algorithms=["PAPO_CPA", "NoPhaseAlign"], verbose=True
    )
    print_comparison_table(comparison)

    print("\n" + "=" * 70)
    print("ExperimentRunner 测试完成!")
    print("=" * 70)


# 运行测试: python3 -m simulation.experiments.runner
