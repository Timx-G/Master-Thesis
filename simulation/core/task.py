"""
CPM任务类定义

功能说明：
- 定义了CPM协作感知任务的数据结构
- 包含任务大小、计算需求、截止时间等属性
- 跟踪任务的生命周期状态

"""

from dataclasses import dataclass, field  # 2. 自动生成类的工具
from enum import Enum  # 4. 枚举类型
from typing import Optional  # 3. 类型提示

import numpy as np  # 1. 引入数学库


class TaskStatus(Enum):
    """任务状态枚举"""

    GENERATED = "generated"  # 任务已生成,未处理
    QUEUED = "queued"  # 任务已排队,等待处理
    TRANSMITTING = "transmitting"  # 任务正在传输
    COMPUTING = "computing"  # 任务正在计算
    COMPULETED = "completed"  # 任务已完成
    DROPPED = "dropped"  # 任务被丢弃
    FAILED = "failed"  # 任务处理失败


if __name__ == "__main__":
    print("=" * 70)
    print("This is a module for defining CPM task data structures.")
    print("=" * 70)
