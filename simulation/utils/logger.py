"""
日志模块
========

提供统一的日志配置和格式化

功能:
1. 控制台彩色输出
2. 文件日志记录
3. 按模块分级
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# ANSI颜色代码
class Colors:
    """终端颜色代码"""

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"


class ColoredFormatter(logging.Formatter):
    """
    彩色日志格式化器

    不同级别使用不同颜色
    """

    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.RED,
    }

    def __init__(
        self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_colors: bool = True
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        # 保存原始levelname
        orig_levelname = record.levelname

        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"

        result = super().format(record)

        # 恢复
        record.levelname = orig_levelname

        return result


def setup_logger(
    name: str = "simulation",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    use_colors: bool = True,
) -> logging.Logger:
    """
    配置日志器

    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志文件目录，None则使用默认
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        use_colors: 是否使用彩色输出

    Returns:
        配置好的Logger对象

    使用示例:
    ```python
    logger = setup_logger("simulation", level=logging.DEBUG)
    logger.info("仿真开始")
    logger.debug("详细信息")
    ```
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 日志格式
    console_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file_fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    date_fmt = "%H:%M:%S"

    # 控制台Handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # 检测是否支持颜色
        supports_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        formatter = ColoredFormatter(
            fmt=console_fmt, datefmt=date_fmt, use_colors=use_colors and supports_color
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件Handler
    if file_output:
        log_path: Path
        if log_dir is None:
            # 默认日志目录
            log_path = Path(__file__).parent.parent.parent / "logs"
        else:
            log_path = Path(log_dir)

        log_path.mkdir(parents=True, exist_ok=True)

        # 按日期命名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)

        file_formatter = logging.Formatter(fmt=file_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"日志文件: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取子日志器

    Args:
        name: 模块名称

    Returns:
        Logger对象

    使用示例:
    ```python
    # 在各模块中使用
    logger = get_logger(__name__)
    logger.info("模块初始化")
    ```
    """
    return logging.getLogger(name)


class SimulationLogger:
    """
    仿真专用日志器

    提供仿真相关的便捷日志方法
    """

    def __init__(self, name: str = "simulation"):
        self.logger = get_logger(name)
        self._slot = 0

    def set_slot(self, slot: int):
        """设置当前时隙"""
        self._slot = slot

    def slot_info(self, message: str):
        """带时隙前缀的INFO日志"""
        self.logger.info(f"[Slot {self._slot:04d}] {message}")

    def slot_debug(self, message: str):
        """带时隙前缀的DEBUG日志"""
        self.logger.debug(f"[Slot {self._slot:04d}] {message}")

    def slot_warning(self, message: str):
        """带时隙前缀的WARNING日志"""
        self.logger.warning(f"[Slot {self._slot:04d}] {message}")

    def slot_error(self, message: str):
        """带时隙前缀的ERROR日志"""
        self.logger.error(f"[Slot {self._slot:04d}] {message}")

    def task_event(self, vehicle_id: int, event: str, details: str = ""):
        """记录任务事件"""
        msg = f"Vehicle {vehicle_id}: {event}"
        if details:
            msg += f" | {details}"
        self.slot_debug(msg)

    def metrics_summary(self, metrics: dict):
        """记录指标摘要"""
        self.logger.info("=" * 50)
        self.logger.info("仿真指标摘要")
        self.logger.info("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("日志模块单元测试")
    print("=" * 60)

    # 测试1: 基本日志
    print("\n--- 测试1: 基本日志配置 ---")
    logger = setup_logger(
        "test_logger", level=logging.DEBUG, console_output=True, file_output=False  # 测试时不写文件
    )

    logger.debug("这是DEBUG级别")
    logger.info("这是INFO级别")
    logger.warning("这是WARNING级别")
    logger.error("这是ERROR级别")

    # 测试2: 子日志器
    print("\n--- 测试2: 子日志器 ---")
    child_logger = get_logger("test_logger.module1")
    child_logger.info("来自子模块的日志")

    # 测试3: 仿真日志器
    print("\n--- 测试3: 仿真专用日志器 ---")
    sim_logger = SimulationLogger("test_simulation")

    # 设置日志级别
    logging.getLogger("test_simulation").setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColoredFormatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
    )
    logging.getLogger("test_simulation").addHandler(handler)

    sim_logger.set_slot(0)
    sim_logger.slot_info("仿真开始")

    sim_logger.set_slot(10)
    sim_logger.task_event(0, "任务生成", "data=500KB")
    sim_logger.task_event(0, "任务完成", "delay=50ms")

    sim_logger.set_slot(100)
    sim_logger.slot_info("仿真结束")

    # 测试4: 指标摘要
    print("\n--- 测试4: 指标摘要 ---")
    metrics = {
        "avg_aoi_ms": 120.5,
        "total_energy_mj": 45.2,
        "success_rate": 0.95,
        "jain_fairness": 0.87,
    }
    sim_logger.metrics_summary(metrics)

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


# 运行: python3 -m simulation.utils.logger
