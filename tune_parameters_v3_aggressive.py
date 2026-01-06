#!/usr/bin/env python3
"""
参数调优脚本 V3 - 激进策略
目标: 达到 deadline_rate > 80%

策略: 基于V2发现，重点测试deadline和RSU数量
"""

import time
from simulation.experiments.simulator import Simulator, SimulatorConfig, PhaseAllocMethod


def run_test(config: SimulatorConfig) -> dict:
    """运行测试并返回完整结果"""
    sim = Simulator(config)
    results = sim.run()

    if "performance" in results:
        return {
            "deadline_rate": results["performance"]["deadline_rate"] * 100.0,
            "avg_aoi_ms": results["performance"]["avg_aoi_ms"],
            "avg_energy_mJ": results["performance"]["avg_energy_mJ"],
            "offload_ratio": results.get("offload_ratio", 0) * 100.0,
        }
    else:
        return {
            "deadline_rate": 0.0,
            "avg_aoi_ms": 0.0,
            "avg_energy_mJ": 0.0,
            "offload_ratio": 0.0,
        }


def main():
    print("=" * 60)
    print("V2X-MEC 参数调优工具 V3 (激进策略)")
    print("目标: deadline_rate > 80%")
    print("=" * 60)

    results = []

    # 测试1: 继续放宽deadline
    print("\n" + "=" * 60)
    print("第1组: 继续放宽deadline (200ms → 500ms)")
    print("=" * 60)

    for deadline in [0.22, 0.25, 0.30, 0.35, 0.40, 0.50]:
        config = SimulatorConfig(
            num_vehicles=50,
            num_rsus=5,
            duration=20.0,
            warmup_time=2.0,
            phase_method=PhaseAllocMethod.CPA,
            random_seed=42,
            verbose=False,
            cpm_deadline=deadline,
        )

        print(f"\n测试: Deadline = {deadline*1000:.0f} ms")
        start = time.time()
        result = run_test(config)
        elapsed = time.time() - start

        print(f"  deadline_rate: {result['deadline_rate']:.2f}%")
        print(f"  avg_aoi: {result['avg_aoi_ms']:.2f} ms")
        print(f"  耗时: {elapsed:.1f}s")

        results.append(("Deadline " + str(int(deadline * 1000)) + "ms", result["deadline_rate"], config))

        if result["deadline_rate"] >= 80.0:
            print(f"\n🎉 达到目标! deadline_rate = {result['deadline_rate']:.2f}%")
            break

    # 如果还未达标，测试2: 增加RSU数量
    if max(r[1] for r in results) < 80.0:
        print("\n" + "=" * 60)
        print("第2组: 增加RSU数量 (5 → 10)")
        print("=" * 60)

        for num_rsus in [7, 10, 15]:
            config = SimulatorConfig(
                num_vehicles=50,
                num_rsus=num_rsus,
                duration=20.0,
                warmup_time=2.0,
                phase_method=PhaseAllocMethod.CPA,
                random_seed=42,
                verbose=False,
                cpm_deadline=0.20,  # 使用200ms
            )

            print(f"\n测试: RSU数量 = {num_rsus}")
            start = time.time()
            result = run_test(config)
            elapsed = time.time() - start

            print(f"  deadline_rate: {result['deadline_rate']:.2f}%")
            print(f"  avg_aoi: {result['avg_aoi_ms']:.2f} ms")
            print(f"  耗时: {elapsed:.1f}s")

            results.append((f"RSU{num_rsus} + deadline200ms", result["deadline_rate"], config))

            if result["deadline_rate"] >= 80.0:
                print(f"\n🎉 达到目标! deadline_rate = {result['deadline_rate']:.2f}%")
                break

    # 如果还未达标，测试3: 综合优化
    if max(r[1] for r in results) < 80.0:
        print("\n" + "=" * 60)
        print("第3组: 综合优化")
        print("=" * 60)

        combinations = [
            ("RSU7 + deadline250ms + compute50M", {
                "num_rsus": 7,
                "cpm_deadline": 0.25,
                "cpm_compute_base": 0.5e8,
            }),
            ("RSU10 + deadline200ms + compute30M", {
                "num_rsus": 10,
                "cpm_deadline": 0.20,
                "cpm_compute_base": 0.3e8,
            }),
            ("RSU10 + deadline300ms", {
                "num_rsus": 10,
                "cpm_deadline": 0.30,
            }),
        ]

        for name, params in combinations:
            config = SimulatorConfig(
                num_vehicles=50,
                duration=20.0,
                warmup_time=2.0,
                phase_method=PhaseAllocMethod.CPA,
                random_seed=42,
                verbose=False,
                **params
            )

            print(f"\n测试: {name}")
            start = time.time()
            result = run_test(config)
            elapsed = time.time() - start

            print(f"  deadline_rate: {result['deadline_rate']:.2f}%")
            print(f"  avg_aoi: {result['avg_aoi_ms']:.2f} ms")
            print(f"  offload_ratio: {result['offload_ratio']:.2f}%")
            print(f"  耗时: {elapsed:.1f}s")

            results.append((name, result["deadline_rate"], config))

            if result["deadline_rate"] >= 80.0:
                print(f"\n🎉 达到目标! deadline_rate = {result['deadline_rate']:.2f}%")
                break

    # 输出汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"{'配置':<45} {'Deadline Rate':>12}")
    print("-" * 60)

    for name, rate, _ in results:
        status = "✓" if rate >= 80.0 else " "
        print(f"{status} {name:<43} {rate:>10.2f}%")

    # 找出最佳配置
    best_name, best_rate, best_config = max(results, key=lambda x: x[1])

    print("\n" + "=" * 60)
    print(f"最佳配置: {best_name}")
    print(f"Deadline Rate: {best_rate:.2f}%")
    print("=" * 60)

    if best_rate >= 80.0:
        print("\n✓ 已达到目标!")
        print("\n建议的参数配置:")
        print(f"  num_rsus: {best_config.num_rsus}")
        print(f"  cpm_deadline: {best_config.cpm_deadline}  # {best_config.cpm_deadline*1000:.0f} ms")
        print(f"  cpm_compute_base: {best_config.cpm_compute_base}  # {best_config.cpm_compute_base/1e6:.0f} MCycles")
        print("\n下一步:")
        print("  1. 更新 simulation/config/parameters.yaml")
        print("  2. 运行完整实验验证")
        print("  3. 开始DRL基线实现")
    else:
        print(f"\n⚠ 仍未达到目标 (当前最佳: {best_rate:.2f}%)")
        print("\n可能原因分析:")
        print("  1. 通信信道是主要瓶颈（而非计算）")
        print("  2. 需要分析算法逻辑（可能资源分配次优）")
        print("  3. 可能需要调整通信参数（带宽、发射功率）")


if __name__ == "__main__":
    import sys
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
