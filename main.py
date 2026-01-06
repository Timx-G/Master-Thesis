#!/usr/bin/env python3
"""
V2X-MEC CPM任务卸载仿真平台
主程序入口

文件: main.py

使用方法:
    python3 main.py                    # 运行默认仿真
    python3 main.py --mode quick       # 快速测试
    python3 main.py --mode compare     # 算法对比
    python3 main.py --mode sensitivity # 参数敏感性
    python3 main.py --help             # 帮助信息
"""

import argparse
import sys
import time

import numpy as np

from simulation.experiments.runner import ExperimentConfig, ExperimentRunner, print_comparison_table

# 仿真模块
from simulation.experiments.simulator import PhaseAllocMethod, Simulator, SimulatorConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="V2X-MEC CPM任务卸载仿真平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 main.py                        # 默认仿真
  python3 main.py --mode quick           # 快速测试 (10秒)
  python3 main.py --mode full            # 完整仿真 (100秒)
  python3 main.py --mode compare         # 算法对比实验
  python3 main.py -n 100 -r 5 -d 50      # 自定义参数
        """,
    )

    # 模式选择
    parser.add_argument(
        "--mode",
        "-m",
        choices=["quick", "full", "compare", "sensitivity"],
        default="quick",
        help="运行模式 (default: quick)",
    )

    # 仿真参数
    parser.add_argument("--num-vehicles", "-n", type=int, default=50, help="车辆数量 (default: 50)")

    parser.add_argument("--num-rsus", "-r", type=int, default=5, help="RSU数量 (default: 5)")

    parser.add_argument(
        "--duration", "-d", type=float, default=None, help="仿真时长(秒)，默认根据模式自动设置"
    )

    parser.add_argument("--num-runs", type=int, default=5, help="重复运行次数 (default: 5)")

    # 算法选择
    parser.add_argument(
        "--phase-method",
        choices=["cpa", "dpa", "none"],
        default="cpa",
        help="相位分配方法 (default: cpa)",
    )

    # 输出选项
    parser.add_argument(
        "--output", "-o", type=str, default="./results", help="输出目录 (default: ./results)"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    parser.add_argument("--seed", type=int, default=42, help="随机种子 (default: 42)")

    return parser.parse_args()


def run_quick_test(args):
    """快速测试模式"""
    print("\n" + "=" * 60)
    print("快速测试模式")
    print("=" * 60)

    duration = args.duration or 10.0

    # 从YAML加载基础配置
    config = SimulatorConfig.from_yaml()

    # 覆盖命令行参数
    config.num_vehicles = args.num_vehicles
    config.num_rsus = args.num_rsus
    config.duration = duration
    config.warmup_time = min(2.0, duration * 0.1)
    config.phase_method = PhaseAllocMethod(args.phase_method)
    config.random_seed = args.seed
    config.verbose = args.verbose

    print(f"\n配置: {config.num_vehicles}辆车, {config.num_rsus}个RSU, {duration}秒")
    print(f"相位方法: {config.phase_method.value}")
    print(f"Deadline: {config.cpm_deadline * 1000:.0f} ms")

    sim = Simulator(config)

    print("\n开始仿真...")
    start = time.time()
    results = sim.run()
    elapsed = time.time() - start

    _print_results(results, elapsed)


def run_full_simulation(args):
    """完整仿真模式"""
    print("\n" + "=" * 60)
    print("完整仿真模式")
    print("=" * 60)

    duration = args.duration or 100.0

    exp_config = ExperimentConfig(
        name="full_simulation",
        num_vehicles=args.num_vehicles,
        num_rsus=args.num_rsus,
        duration=duration,
        warmup_time=10.0,
        num_runs=args.num_runs,
        base_seed=args.seed,
        output_dir=args.output,
    )

    print(f"\n配置: {exp_config.num_vehicles}辆车, {exp_config.num_rsus}个RSU")
    print(f"时长: {duration}秒, 重复: {exp_config.num_runs}次")

    runner = ExperimentRunner(exp_config)

    sim_config = SimulatorConfig(
        num_vehicles=args.num_vehicles,
        num_rsus=args.num_rsus,
        duration=duration,
        warmup_time=10.0,
        phase_method=PhaseAllocMethod(args.phase_method),
        verbose=args.verbose,
    )

    print("\n开始仿真...")

    def progress(i, n):
        print(f"  运行 {i}/{n} 完成")

    start = time.time()
    summary = runner.run_multiple(sim_config, progress_callback=progress)
    elapsed = time.time() - start

    _print_summary(summary, elapsed)

    # 保存结果
    runner.save_results(summary, "full_simulation.json")


def run_algorithm_comparison(args):
    """算法对比模式"""
    print("\n" + "=" * 60)
    print("算法对比实验")
    print("=" * 60)

    duration = args.duration or 50.0

    exp_config = ExperimentConfig(
        name="algorithm_comparison",
        num_vehicles=args.num_vehicles,
        num_rsus=args.num_rsus,
        duration=duration,
        warmup_time=5.0,
        num_runs=args.num_runs,
        base_seed=args.seed,
        output_dir=args.output,
    )

    runner = ExperimentRunner(exp_config)

    algorithms = [
        "PAPO_CPA",
        "PAPO_DPA",
        "NoPhaseAlign",
        "LocalOnly",
    ]

    print(f"\n对比算法: {algorithms}")
    print(f"配置: {exp_config.num_vehicles}车, {duration}秒, {exp_config.num_runs}次重复")

    start = time.time()
    comparison = runner.run_algorithm_comparison(algorithms, verbose=True)
    elapsed = time.time() - start

    print_comparison_table(comparison)
    print(f"\n总用时: {elapsed:.1f}秒")

    # 保存结果
    runner.save_results(comparison, "algorithm_comparison.json")


def run_sensitivity_analysis(args):
    """参数敏感性分析"""
    print("\n" + "=" * 60)
    print("参数敏感性分析")
    print("=" * 60)

    duration = args.duration or 30.0

    exp_config = ExperimentConfig(
        name="sensitivity",
        num_vehicles=args.num_vehicles,
        num_rsus=args.num_rsus,
        duration=duration,
        warmup_time=5.0,
        num_runs=3,
        base_seed=args.seed,
        output_dir=args.output,
    )

    runner = ExperimentRunner(exp_config)

    # 车辆数量敏感性
    print("\n【车辆数量敏感性】")
    vehicle_counts = [20, 30, 50, 70, 100]
    sensitivity = runner.run_parameter_sensitivity("num_vehicles", vehicle_counts, verbose=True)

    print("\n结果:")
    print(f"{'车辆数':<10} | {'AoI(ms)':<15} | {'Deadline%':<12}")
    print("-" * 45)
    for result in sensitivity["results"]:
        n = result["param_value"]
        aoi = result["performance"]["avg_aoi_ms"]["mean"]
        dr = result["performance"]["deadline_rate"]["mean"]
        print(f"{n:<10} | {aoi:<15.2f} | {dr*100:<12.1f}")

    runner.save_results(sensitivity, "sensitivity_vehicles.json")


def _print_results(results: dict, elapsed: float):
    """打印单次运行结果"""
    print("\n" + "=" * 60)
    print("仿真结果")
    print("=" * 60)

    perf = results["performance"]
    stats = results["statistics"]

    print("\n【性能指标】")
    print(f"  平均AoI:        {perf['avg_aoi_ms']:.2f} ms")
    print(f"  平均时延:       {perf['avg_delay_ms']:.2f} ms")
    print(f"  平均能耗:       {perf['avg_energy_mJ']:.4f} mJ")
    print(f"  截止时间满足率: {perf['deadline_rate']:.2%}")
    print(f"  Jain公平性:     {perf['jain_fairness']:.4f}")

    print("\n【任务统计】")
    print(f"  总任务数:   {stats['total_tasks']}")
    print(f"  完成任务:   {stats['completed_tasks']}")
    print(f"  丢弃任务:   {stats['dropped_tasks']}")
    print(f"  卸载比例:   {results['offload_ratio']:.2%}")

    print(f"\n运行时间: {elapsed:.2f}秒")


def _print_summary(summary: dict, elapsed: float):
    """打印多次运行汇总"""
    print("\n" + "=" * 60)
    print("仿真结果汇总")
    print("=" * 60)

    perf = summary["performance"]

    print(f"\n运行次数: {summary['num_runs']}")

    print("\n【性能指标 (mean ± std)】")
    print(
        f"  平均AoI:        {perf['avg_aoi_ms']['mean']:.2f} ± {perf['avg_aoi_ms']['std']:.2f} ms"
    )
    print(
        f"  平均时延:       {perf['avg_delay_ms']['mean']:.2f} ± {perf['avg_delay_ms']['std']:.2f} ms"
    )
    print(
        f"  截止时间满足率: {perf['deadline_rate']['mean']:.2%} ± {perf['deadline_rate']['std']:.2%}"
    )
    print(
        f"  Jain公平性:     {perf['jain_fairness']['mean']:.4f} ± {perf['jain_fairness']['std']:.4f}"
    )

    print(f"\n总用时: {elapsed:.1f}秒")


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("V2X-MEC CPM任务卸载仿真平台")
    print("=" * 60)

    np.random.seed(args.seed)

    if args.mode == "quick":
        run_quick_test(args)
    elif args.mode == "full":
        run_full_simulation(args)
    elif args.mode == "compare":
        run_algorithm_comparison(args)
    elif args.mode == "sensitivity":
        run_sensitivity_analysis(args)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
