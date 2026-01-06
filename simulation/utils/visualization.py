"""
可视化模块
==========

提供仿真结果的可视化功能

功能:
1. AoI时间序列图
2. 能耗对比图
3. 截止时间分布图
4. 公平性分析图
5. 算法对比图
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入matplotlib，不可用时提供降级方案
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    MATPLOTLIB_AVAILABLE = True

    # 设置中文字体和样式
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, visualization disabled")


# 颜色方案
COLORS = {
    "primary": "#2196F3",  # 蓝色
    "secondary": "#4CAF50",  # 绿色
    "accent": "#FF9800",  # 橙色
    "error": "#F44336",  # 红色
    "warning": "#FFC107",  # 黄色
    "info": "#00BCD4",  # 青色
    "purple": "#9C27B0",  # 紫色
    "grey": "#9E9E9E",  # 灰色
}

# 算法颜色映射
ALGORITHM_COLORS = {
    "PAPO": "#2196F3",
    "Greedy": "#4CAF50",
    "Random": "#FF9800",
    "LocalOnly": "#F44336",
    "FullOffload": "#9C27B0",
    "RoundRobin": "#00BCD4",
    "NoPhase": "#795548",
    "DQN": "#E91E63",
    "PPO": "#3F51B5",
}


def check_matplotlib():
    """检查matplotlib是否可用"""
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib is not installed. Run: pip install matplotlib")
        return False
    return True


class SimulationVisualizer:
    """
    仿真结果可视化器

    使用示例:
    ```python
    viz = SimulationVisualizer(output_dir="results/figures")

    # 绘制AoI时间序列
    viz.plot_aoi_timeseries(aoi_history, title="AoI Evolution")

    # 绘制算法对比
    viz.plot_algorithm_comparison(results_dict, metric='avg_aoi')
    ```
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化可视化器

        Args:
            output_dir: 图片输出目录
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"SimulationVisualizer initialized, output: {self.output_dir}")

    def _save_figure(self, fig, filename: str, tight: bool = True):
        """保存图片"""
        if tight:
            fig.tight_layout()

        filepath = self.output_dir / filename
        fig.savefig(filepath, bbox_inches="tight")
        logger.info(f"Figure saved: {filepath}")
        plt.close(fig)

    def plot_aoi_timeseries(
        self,
        aoi_history: List[float],
        title: str = "System Average AoI",
        filename: str = "aoi_timeseries.png",
        dt: float = 0.1,
        show_target: Optional[float] = None,
    ):
        """
        绘制AoI时间序列图

        Args:
            aoi_history: 系统平均AoI历史 (秒)
            title: 图标题
            filename: 输出文件名
            dt: 时隙长度 (秒)
            show_target: 显示目标AoI线
        """
        if not check_matplotlib():
            return

        fig, ax = plt.subplots(figsize=(12, 5))

        # 时间轴 (秒)
        time_axis = np.arange(len(aoi_history)) * dt

        # AoI转换为毫秒
        aoi_ms = np.array(aoi_history) * 1000

        ax.plot(time_axis, aoi_ms, color=COLORS["primary"], linewidth=1.5, label="Average AoI")

        # 显示目标线
        if show_target:
            ax.axhline(
                y=show_target * 1000,
                color=COLORS["error"],
                linestyle="--",
                linewidth=1.5,
                label=f"Target ({show_target*1000:.0f}ms)",
            )

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("AoI (ms)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="upper right")
        ax.set_xlim(0, time_axis[-1] if len(time_axis) > 0 else 1)
        ax.set_ylim(0, None)

        self._save_figure(fig, filename)

    def plot_aoi_per_vehicle(
        self,
        per_vehicle_aoi: List[float],
        title: str = "AoI Distribution by Vehicle",
        filename: str = "aoi_per_vehicle.png",
    ):
        """
        绘制各车辆AoI分布

        Args:
            per_vehicle_aoi: 各车辆的平均AoI (秒)
        """
        if not check_matplotlib():
            return

        fig, ax = plt.subplots(figsize=(12, 5))

        vehicle_ids = range(len(per_vehicle_aoi))
        aoi_ms = np.array(per_vehicle_aoi) * 1000

        bars = ax.bar(vehicle_ids, aoi_ms, color=COLORS["primary"], alpha=0.7)

        # 平均线
        avg_aoi = np.mean(aoi_ms)
        ax.axhline(
            y=float(avg_aoi),
            color=COLORS["error"],
            linestyle="--",
            linewidth=2,
            label=f"Average: {avg_aoi:.1f}ms",
        )

        ax.set_xlabel("Vehicle ID", fontsize=12)
        ax.set_ylabel("Average AoI (ms)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        self._save_figure(fig, filename)

    def plot_energy_breakdown(
        self,
        local_energy: float,
        tx_energy: float,
        idle_energy: float,
        title: str = "Energy Consumption Breakdown",
        filename: str = "energy_breakdown.png",
    ):
        """
        绘制能耗分解饼图

        Args:
            local_energy: 本地计算能耗 (J)
            tx_energy: 传输能耗 (J)
            idle_energy: 空闲能耗 (J)
        """
        if not check_matplotlib():
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        sizes = [local_energy, tx_energy, idle_energy]
        labels = ["Local Compute", "Transmission", "Idle"]
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]
        explode = (0.02, 0.02, 0.02)

        # 过滤零值
        non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0]
        if non_zero:
            sizes, labels, colors, explode = zip(*non_zero)

        wedges, texts, *autotexts_list = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            shadow=False,
            startangle=90,
        )

        autotexts = autotexts_list[0] if autotexts_list else []

        ax.set_title(title, fontsize=14)
        ax.axis("equal")

        self._save_figure(fig, filename)

    def plot_delay_distribution(
        self,
        delays: List[float],
        deadline: float = 0.08,
        title: str = "Delay Distribution",
        filename: str = "delay_distribution.png",
    ):
        """
        绘制延迟分布直方图

        Args:
            delays: 延迟列表 (秒)
            deadline: 截止时间 (秒)
        """
        if not check_matplotlib():
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        delays_ms = np.array(delays) * 1000
        deadline_ms = deadline * 1000

        # 直方图
        n, bins, patches = ax.hist(
            delays_ms, bins=30, color=COLORS["primary"], alpha=0.7, edgecolor="white"
        )

        # 标记超时部分
        for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):  # type: ignore
            if left_edge >= deadline_ms:
                patch.set_facecolor(COLORS["error"])

        # 截止时间线
        ax.axvline(
            x=deadline_ms,
            color=COLORS["error"],
            linestyle="--",
            linewidth=2,
            label=f"Deadline ({deadline_ms:.0f}ms)",
        )

        # 统计信息
        violation_rate = np.sum(delays_ms > deadline_ms) / len(delays_ms) * 100
        avg_delay = np.mean(delays_ms)

        textstr = f"Avg: {avg_delay:.1f}ms\nViolation: {violation_rate:.1f}%"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.95,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=props,
        )

        ax.set_xlabel("Delay (ms)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()

        self._save_figure(fig, filename)

    def plot_algorithm_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: Optional[List[str]] = None,
        title: str = "Algorithm Comparison",
        filename: str = "algorithm_comparison.png",
    ):
        """
        绘制算法对比柱状图

        Args:
            results: {algorithm_name: {metric: value}}
            metrics: 要对比的指标列表
        """
        if not check_matplotlib():
            return

        if metrics is None:
            metrics = ["avg_aoi_ms", "energy_mj", "success_rate", "jfi"]

        algorithms = list(results.keys())
        n_algs = len(algorithms)
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [results[alg].get(metric, 0) for alg in algorithms]
            colors = [ALGORITHM_COLORS.get(alg, COLORS["grey"]) for alg in algorithms]

            bars = ax.bar(range(n_algs), values, color=colors, alpha=0.8)

            ax.set_xticks(range(n_algs))
            ax.set_xticklabels(algorithms, rotation=45, ha="right")
            ax.set_ylabel(metric)
            ax.set_title(metric.replace("_", " ").title())

            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(
                    f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig.suptitle(title, fontsize=14)
        self._save_figure(fig, filename)

    def plot_fairness_analysis(
        self,
        vehicle_aois: List[float],
        vehicle_speeds: Optional[List[float]] = None,
        title: str = "Fairness Analysis",
        filename: str = "fairness_analysis.png",
    ):
        """
        绘制公平性分析图

        Args:
            vehicle_aois: 各车辆平均AoI
            vehicle_speeds: 各车辆速度 (可选)
        """
        if not check_matplotlib():
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 左图: AoI分布箱线图
        ax1 = axes[0]
        aoi_ms = np.array(vehicle_aois) * 1000

        bp = ax1.boxplot([aoi_ms], patch_artist=True)
        bp["boxes"][0].set_facecolor(COLORS["primary"])
        bp["boxes"][0].set_alpha(0.7)

        ax1.scatter(np.ones(len(aoi_ms)), aoi_ms, alpha=0.5, color=COLORS["secondary"], s=30)

        ax1.set_ylabel("AoI (ms)", fontsize=12)
        ax1.set_title("AoI Distribution", fontsize=12)
        ax1.set_xticklabels(["All Vehicles"])

        # 右图: 如果有速度信息，绘制AoI vs 速度
        ax2 = axes[1]
        if vehicle_speeds is not None:
            speeds_kmh = np.array(vehicle_speeds) * 3.6  # m/s -> km/h

            scatter = ax2.scatter(speeds_kmh, aoi_ms, c=aoi_ms, cmap="RdYlGn_r", alpha=0.7, s=50)

            # 拟合线
            if len(speeds_kmh) > 2:
                z = np.polyfit(speeds_kmh, aoi_ms, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(speeds_kmh), max(speeds_kmh), 100)
                ax2.plot(x_line, p(x_line), "--", color=COLORS["error"], linewidth=2, label="Trend")

            ax2.set_xlabel("Speed (km/h)", fontsize=12)
            ax2.set_ylabel("AoI (ms)", fontsize=12)
            ax2.set_title("AoI vs Vehicle Speed", fontsize=12)
            ax2.legend()

            plt.colorbar(scatter, ax=ax2, label="AoI (ms)")
        else:
            # 没有速度信息，绘制CDF
            sorted_aoi = np.sort(aoi_ms)
            cdf = np.arange(1, len(sorted_aoi) + 1) / len(sorted_aoi)

            ax2.plot(sorted_aoi, cdf, color=COLORS["primary"], linewidth=2)
            ax2.set_xlabel("AoI (ms)", fontsize=12)
            ax2.set_ylabel("CDF", fontsize=12)
            ax2.set_title("AoI CDF", fontsize=12)

        fig.suptitle(title, fontsize=14)
        self._save_figure(fig, filename)

    def plot_convergence(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Convergence",
        filename: str = "convergence.png",
    ):
        """
        绘制收敛曲线 (用于DRL训练)

        Args:
            history: {'loss': [...], 'reward': [...], ...}
        """
        if not check_matplotlib():
            return

        n_plots = len(history)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))

        if n_plots == 1:
            axes = [axes]

        colors_list = list(COLORS.values())

        for idx, (key, values) in enumerate(history.items()):
            ax = axes[idx]
            ax.plot(values, color=colors_list[idx % len(colors_list)], linewidth=1)
            ax.set_xlabel("Episode", fontsize=10)
            ax.set_ylabel(key, fontsize=10)
            ax.set_title(key.replace("_", " ").title(), fontsize=11)

        fig.suptitle(title, fontsize=14)
        self._save_figure(fig, filename)


def create_summary_report(
    output_dir: str, aoi_stats: dict, energy_stats: dict, deadline_stats: dict, fairness_stats: dict
):
    """
    生成综合报告图

    Args:
        output_dir: 输出目录
        aoi_stats: AoI统计
        energy_stats: 能耗统计
        deadline_stats: 截止时间统计
        fairness_stats: 公平性统计
    """
    if not check_matplotlib():
        return

    viz = SimulationVisualizer(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. AoI时间序列
    ax1 = axes[0, 0]
    if "system_aoi_history" in aoi_stats:
        history = aoi_stats["system_aoi_history"]
        time_axis = np.arange(len(history)) * 0.1
        ax1.plot(time_axis, np.array(history) * 1000, color=COLORS["primary"])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("AoI (ms)")
    ax1.set_title("System Average AoI")

    # 2. 能耗分解
    ax2 = axes[0, 1]
    local_e = energy_stats.get("local_compute_energy_j", 0) * 1000
    tx_e = energy_stats.get("transmission_energy_j", 0) * 1000
    idle_e = energy_stats.get("idle_energy_j", 0) * 1000

    sizes = [local_e, tx_e, idle_e]
    labels = ["Local", "Tx", "Idle"]
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sizes, labels, colors = zip(*non_zero)
        ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
    ax2.set_title("Energy Breakdown")

    # 3. 关键指标
    ax3 = axes[1, 0]
    metrics_names = ["Avg AoI\n(ms)", "Success\nRate (%)", "JFI", "Energy\n(mJ)"]
    metrics_values = [
        aoi_stats.get("system_avg_aoi", 0) * 1000,
        deadline_stats.get("success_rate", 0) * 100,
        fairness_stats.get("avg_jfi", 0),
        energy_stats.get("system_total_energy_mj", 0),
    ]

    bars = ax3.bar(
        metrics_names,
        metrics_values,
        color=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["purple"]],
    )
    ax3.set_title("Key Metrics Summary")

    for bar, val in zip(bars, metrics_values):
        ax3.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    # 4. 延迟分布
    ax4 = axes[1, 1]
    avg_delay = deadline_stats.get("avg_delay_ms", 0)
    p95_delay = deadline_stats.get("p95_delay_ms", 0)
    deadline = deadline_stats.get("deadline_ms", 80)

    x = ["Average", "P95", "Deadline"]
    y = [avg_delay, p95_delay, deadline]
    colors = [COLORS["primary"], COLORS["accent"], COLORS["error"]]
    ax4.bar(x, y, color=colors)
    ax4.set_ylabel("Delay (ms)")
    ax4.set_title("Delay Statistics")

    fig.suptitle("Simulation Summary Report", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    filepath = Path(output_dir) / "summary_report.png"
    fig.savefig(filepath, bbox_inches="tight")
    logger.info(f"Summary report saved: {filepath}")
    plt.close(fig)


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("可视化模块单元测试")
    print("=" * 60)

    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed, skipping tests")
        print("Install with: pip install matplotlib")
        exit(0)

    # 创建测试数据
    np.random.seed(42)

    # 测试1: AoI时间序列
    print("\n--- 测试1: AoI时间序列 ---")
    viz = SimulationVisualizer("test_figures")

    # 模拟AoI历史
    aoi_history = []
    aoi = 0.05
    for _ in range(500):
        if np.random.random() < 0.3:
            aoi = 0.05 + np.random.uniform(0, 0.02)
        else:
            aoi += 0.1
        aoi_history.append(aoi)

    viz.plot_aoi_timeseries(aoi_history, show_target=0.2)
    print("  AoI时间序列图已生成")

    # 测试2: 各车辆AoI
    print("\n--- 测试2: 各车辆AoI分布 ---")
    per_vehicle_aoi = np.random.uniform(0.08, 0.18, 50).tolist()
    viz.plot_aoi_per_vehicle(per_vehicle_aoi)
    print("  各车辆AoI图已生成")

    # 测试3: 能耗分解
    print("\n--- 测试3: 能耗分解 ---")
    viz.plot_energy_breakdown(local_energy=0.02, tx_energy=0.015, idle_energy=0.008)
    print("  能耗分解图已生成")

    # 测试4: 延迟分布
    print("\n--- 测试4: 延迟分布 ---")
    delays = np.random.gamma(2, 0.02, 500).tolist()
    viz.plot_delay_distribution(delays, deadline=0.08)
    print("  延迟分布图已生成")

    # 测试5: 算法对比
    print("\n--- 测试5: 算法对比 ---")
    results = {
        "PAPO": {"avg_aoi_ms": 95, "energy_mj": 35, "success_rate": 0.96, "jfi": 0.88},
        "Greedy": {"avg_aoi_ms": 120, "energy_mj": 42, "success_rate": 0.89, "jfi": 0.75},
        "Random": {"avg_aoi_ms": 180, "energy_mj": 55, "success_rate": 0.72, "jfi": 0.82},
        "LocalOnly": {"avg_aoi_ms": 150, "energy_mj": 28, "success_rate": 0.85, "jfi": 0.90},
    }
    viz.plot_algorithm_comparison(results)
    print("  算法对比图已生成")

    # 测试6: 公平性分析
    print("\n--- 测试6: 公平性分析 ---")
    vehicle_speeds = np.random.uniform(10, 35, 50).tolist()
    viz.plot_fairness_analysis(per_vehicle_aoi, vehicle_speeds)
    print("  公平性分析图已生成")

    # 测试7: 综合报告
    print("\n--- 测试7: 综合报告 ---")
    aoi_stats = {"system_avg_aoi": 0.12, "system_aoi_history": aoi_history}
    energy_stats = {
        "local_compute_energy_j": 0.02,
        "transmission_energy_j": 0.015,
        "idle_energy_j": 0.008,
        "system_total_energy_mj": 43,
    }
    deadline_stats = {
        "avg_delay_ms": 55,
        "p95_delay_ms": 75,
        "deadline_ms": 80,
        "success_rate": 0.94,
    }
    fairness_stats = {"avg_jfi": 0.87}

    create_summary_report("test_figures", aoi_stats, energy_stats, deadline_stats, fairness_stats)
    print("  综合报告已生成")

    print(f"\n所有图片已保存到: test_figures/")
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


# 运行: python3 -m simulation.utils.visualization
