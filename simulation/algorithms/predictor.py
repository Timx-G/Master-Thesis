"""
任务预测算法
文件: simulation/algorithms/predictor.py

功能说明:
- 基于运动学的CPM触发预测 (KinematicPredictor)
- 基于历史数据的任务负载预测
- 可选的神经网络预测器 (GRU-based)

算法来源: docs/04_算法设计_任务负载预测.md
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ETSITriggerConfig:
    """ETSI TS 103 324 触发条件配置"""

    # 位置变化阈值 (m)
    pos_threshold: float = 4.0

    # 速度变化阈值 (m/s)
    speed_threshold: float = 0.5

    # 航向变化阈值 (degrees)
    heading_threshold: float = 4.0

    # 最大触发间隔 (s)
    T_max: float = 1.0

    # 检查周期 (s)
    T_check: float = 0.1


@dataclass
class VehicleKinematicState:
    """车辆运动学状态"""

    vehicle_id: int

    # 当前状态
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    speed: float = 0.0
    heading: float = 0.0  # radians

    # 上次CPM触发时的状态
    last_cpm_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    last_cpm_speed: float = 0.0
    last_cpm_heading: float = 0.0
    last_cpm_time: float = 0.0

    # 预测相关
    predicted_next_trigger: float = 0.0
    trigger_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))


class KinematicPredictor:
    """
    基于运动学的CPM触发预测器

    功能:
    - 根据ETSI TS 103 324触发条件预测下次CPM触发时间
    - 使用车辆运动学状态（位置、速度、航向）
    - 支持多车辆预测

    预测原理:
    - 位置触发: T_pos = pos_threshold / speed
    - 速度触发: T_speed = speed_threshold / |acceleration|
    - 航向触发: T_heading = heading_threshold / |yaw_rate|
    - 最大间隔: T_max
    - 预测值: T_pred = min(T_pos, T_speed, T_heading, T_max)

    来源: docs/04_算法设计_任务负载预测.md §2
    """

    def __init__(self, config: Optional[ETSITriggerConfig] = None):
        """
        初始化预测器

        Args:
            config: ETSI触发条件配置
        """
        self.config = config or ETSITriggerConfig()

        # 车辆状态表: {vehicle_id: VehicleKinematicState}
        self._vehicle_states: Dict[int, VehicleKinematicState] = {}

        # 预测统计
        self.stats = {
            "total_predictions": 0,
            "total_actual_triggers": 0,
            "prediction_errors": [],  # 预测误差列表
            "mae": 0.0,  # 平均绝对误差
            "rmse": 0.0,  # 均方根误差
        }

    def initialize_vehicle(
        self,
        vehicle_id: int,
        position: np.ndarray,
        speed: float,
        heading: float,
        current_time: float = 0.0,
    ):
        """
        初始化车辆状态

        Args:
            vehicle_id: 车辆ID
            position: 初始位置 [x, y]
            speed: 初始速度 (m/s)
            heading: 初始航向 (rad)
            current_time: 当前时间
        """
        state = VehicleKinematicState(
            vehicle_id=vehicle_id,
            position=position.copy(),
            speed=speed,
            heading=heading,
            last_cpm_position=position.copy(),
            last_cpm_speed=speed,
            last_cpm_heading=heading,
            last_cpm_time=current_time,
        )
        self._vehicle_states[vehicle_id] = state

    def update_vehicle_state(
        self, vehicle_id: int, position: np.ndarray, speed: float, heading: float
    ):
        """
        更新车辆当前状态

        Args:
            vehicle_id: 车辆ID
            position: 当前位置
            speed: 当前速度
            heading: 当前航向
        """
        if vehicle_id not in self._vehicle_states:
            self.initialize_vehicle(vehicle_id, position, speed, heading)
            return

        state = self._vehicle_states[vehicle_id]
        state.position = position.copy()
        state.speed = speed
        state.heading = heading

    def predict_next_trigger(
        self, vehicle_id: int, current_time: float, acceleration: float = 0.0, yaw_rate: float = 0.0
    ) -> Tuple[float, str]:
        """
        预测下次CPM触发时间

        Args:
            vehicle_id: 车辆ID
            current_time: 当前时间
            acceleration: 加速度 (m/s²)，可选
            yaw_rate: 航向变化率 (rad/s)，可选

        Returns:
            (predicted_time, trigger_reason): 预测触发时间和原因

        算法:
        1. 计算距上次触发的状态变化
        2. 预测各条件的触发时间
        3. 返回最小值

        来源: docs/04_算法设计_任务负载预测.md 算法1
        """
        if vehicle_id not in self._vehicle_states:
            return current_time + self.config.T_check, "unknown"

        state = self._vehicle_states[vehicle_id]
        cfg = self.config

        # 计算当前状态变化量
        pos_change = np.linalg.norm(state.position - state.last_cpm_position)
        speed_change = abs(state.speed - state.last_cpm_speed)
        heading_change = abs(state.heading - state.last_cpm_heading)
        # 处理航向角跨越 2π 的情况
        heading_change = min(heading_change, 2 * np.pi - heading_change)
        heading_change_deg = np.degrees(heading_change)

        time_since_last = current_time - state.last_cpm_time

        # 计算各条件剩余时间
        candidates = []

        # 位置条件
        if state.speed > 0.1:  # 避免除零
            remaining_pos = max(0, cfg.pos_threshold - pos_change)
            t_pos = remaining_pos / state.speed
            candidates.append((current_time + t_pos, "position"))

        # 速度条件
        if abs(acceleration) > 0.01:
            remaining_speed = max(0, cfg.speed_threshold - speed_change)
            t_speed = remaining_speed / abs(acceleration)
            candidates.append((current_time + t_speed, "speed"))

        # 航向条件
        if abs(yaw_rate) > 0.001:
            remaining_heading = max(0, cfg.heading_threshold - heading_change_deg)
            t_heading = remaining_heading / abs(np.degrees(yaw_rate))
            candidates.append((current_time + t_heading, "heading"))

        # 最大间隔条件
        remaining_max = max(0, cfg.T_max - time_since_last)
        candidates.append((current_time + remaining_max, "max_interval"))

        # 选择最早触发
        if not candidates:
            return current_time + cfg.T_check, "default"

        predicted_time, reason = min(candidates, key=lambda x: x[0])

        # 更新预测
        state.predicted_next_trigger = predicted_time
        self.stats["total_predictions"] += 1

        return predicted_time, reason

    def check_trigger_condition(self, vehicle_id: int, current_time: float) -> Tuple[bool, str]:
        """
        检查是否满足CPM触发条件

        Args:
            vehicle_id: 车辆ID
            current_time: 当前时间

        Returns:
            (should_trigger, reason): 是否触发和原因

        来源: ETSI TS 103 324 §6.1.3
        """
        if vehicle_id not in self._vehicle_states:
            return False, "unknown_vehicle"

        state = self._vehicle_states[vehicle_id]
        cfg = self.config

        # 条件1: 位置变化
        pos_change = np.linalg.norm(state.position - state.last_cpm_position)
        if pos_change >= cfg.pos_threshold:
            return True, "position"

        # 条件2: 速度变化
        speed_change = abs(state.speed - state.last_cpm_speed)
        if speed_change >= cfg.speed_threshold:
            return True, "speed"

        # 条件3: 航向变化
        heading_change = abs(state.heading - state.last_cpm_heading)
        heading_change = min(heading_change, 2 * np.pi - heading_change)
        if np.degrees(heading_change) >= cfg.heading_threshold:
            return True, "heading"

        # 条件4: 最大间隔
        time_since_last = current_time - state.last_cpm_time
        if time_since_last >= cfg.T_max:
            return True, "max_interval"

        return False, "none"

    def record_trigger(
        self, vehicle_id: int, current_time: float, predicted_time: Optional[float] = None
    ):
        """
        记录实际触发（更新状态和统计）

        Args:
            vehicle_id: 车辆ID
            current_time: 实际触发时间
            predicted_time: 之前预测的时间（用于计算误差）
        """
        if vehicle_id not in self._vehicle_states:
            return

        state = self._vehicle_states[vehicle_id]

        # 更新last_cpm状态
        state.last_cpm_position = state.position.copy()
        state.last_cpm_speed = state.speed
        state.last_cpm_heading = state.heading
        state.last_cpm_time = current_time

        # 记录触发历史
        state.trigger_history.append(current_time)

        # 更新统计
        self.stats["total_actual_triggers"] += 1

        if predicted_time is not None:
            error = abs(current_time - predicted_time)
            self.stats["prediction_errors"].append(error)

            # 更新MAE和RMSE
            errors = self.stats["prediction_errors"]
            self.stats["mae"] = np.mean(errors)
            self.stats["rmse"] = np.sqrt(np.mean([e**2 for e in errors]))

    def get_trigger_frequency(self, vehicle_id: int) -> float:
        """
        获取车辆的平均触发频率

        Args:
            vehicle_id: 车辆ID

        Returns:
            frequency: 触发频率 (Hz)
        """
        if vehicle_id not in self._vehicle_states:
            return 0.0

        state = self._vehicle_states[vehicle_id]
        history = list(state.trigger_history)

        if len(history) < 2:
            return 0.0

        # 计算平均间隔
        intervals = [history[i + 1] - history[i] for i in range(len(history) - 1)]
        avg_interval = np.mean(intervals)

        if avg_interval <= 0:
            return 0.0

        return float(1.0 / avg_interval)

    def predict_load_next_slot(
        self, vehicle_ids: List[int], current_time: float, slot_duration: float = 0.1
    ) -> Dict[int, float]:
        """
        预测下一个时隙的任务负载

        Args:
            vehicle_ids: 车辆ID列表
            current_time: 当前时间
            slot_duration: 时隙长度

        Returns:
            load_predictions: {vehicle_id: expected_tasks}

        简化模型: 如果预测触发时间在下一时隙内，则期望任务数为1
        """
        predictions = {}
        slot_end = current_time + slot_duration

        for vid in vehicle_ids:
            pred_time, _ = self.predict_next_trigger(vid, current_time)

            if pred_time <= slot_end:
                predictions[vid] = 1.0
            else:
                # 部分触发概率
                remaining = pred_time - current_time
                if remaining > 0:
                    prob = min(1.0, slot_duration / remaining)
                    predictions[vid] = prob
                else:
                    predictions[vid] = 1.0

        return predictions

    def get_statistics(self) -> dict:
        """获取预测统计"""
        result = {
            "total_predictions": self.stats["total_predictions"],
            "total_actual_triggers": self.stats["total_actual_triggers"],
            "mae": self.stats["mae"],
            "rmse": self.stats["rmse"],
            "num_vehicles": len(self._vehicle_states),
        }
        return result

    def reset(self):
        """重置预测器"""
        self._vehicle_states.clear()
        self.stats = {
            "total_predictions": 0,
            "total_actual_triggers": 0,
            "prediction_errors": [],
            "mae": 0.0,
            "rmse": 0.0,
        }


# ============================================================
# 基于历史的负载预测器
# ============================================================


@dataclass
class LoadPredictorConfig:
    """负载预测器配置"""

    # 历史窗口大小
    history_window: int = 10

    # 预测方法: "average", "ema", "linear"
    method: str = "ema"

    # EMA衰减因子
    ema_alpha: float = 0.3

    # 异常检测阈值 (标准差倍数)
    anomaly_threshold: float = 3.0


class HistoricalLoadPredictor:
    """
    基于历史数据的负载预测器

    功能:
    - 记录历史负载数据
    - 使用多种方法预测未来负载
    - 支持RSU级别和车辆级别预测

    来源: docs/04_算法设计_任务负载预测.md §3
    """

    def __init__(self, config: Optional[LoadPredictorConfig] = None):
        """
        初始化预测器

        Args:
            config: 配置参数
        """
        self.config = config or LoadPredictorConfig()

        # RSU负载历史: {rsu_id: deque of load values}
        self._rsu_history: Dict[int, Deque[float]] = {}

        # 车辆任务历史: {vehicle_id: deque of task counts}
        self._vehicle_history: Dict[int, Deque[int]] = {}

        # EMA状态
        self._ema_state: Dict[str, float] = {}

        # 统计
        self.stats = {
            "total_predictions": 0,
            "prediction_errors": [],
        }

    def record_rsu_load(self, rsu_id: int, load: float):
        """
        记录RSU负载

        Args:
            rsu_id: RSU ID
            load: 当前负载值（任务数或资源利用率）
        """
        if rsu_id not in self._rsu_history:
            self._rsu_history[rsu_id] = deque(maxlen=self.config.history_window)

        self._rsu_history[rsu_id].append(load)

        # 更新EMA
        key = f"rsu_{rsu_id}"
        if key in self._ema_state:
            alpha = self.config.ema_alpha
            self._ema_state[key] = alpha * load + (1 - alpha) * self._ema_state[key]
        else:
            self._ema_state[key] = load

    def record_vehicle_tasks(self, vehicle_id: int, task_count: int):
        """
        记录车辆任务数

        Args:
            vehicle_id: 车辆ID
            task_count: 当前时隙的任务数
        """
        if vehicle_id not in self._vehicle_history:
            self._vehicle_history[vehicle_id] = deque(maxlen=self.config.history_window)

        self._vehicle_history[vehicle_id].append(task_count)

        # 更新EMA
        key = f"vehicle_{vehicle_id}"
        if key in self._ema_state:
            alpha = self.config.ema_alpha
            self._ema_state[key] = alpha * task_count + (1 - alpha) * self._ema_state[key]
        else:
            self._ema_state[key] = float(task_count)

    def predict_rsu_load(self, rsu_id: int) -> float:
        """
        预测RSU下一时隙负载

        Args:
            rsu_id: RSU ID

        Returns:
            predicted_load: 预测负载
        """
        self.stats["total_predictions"] += 1

        method = self.config.method

        if method == "average":
            return self._predict_average(f"rsu_{rsu_id}", self._rsu_history.get(rsu_id))
        elif method == "ema":
            return self._predict_ema(f"rsu_{rsu_id}")
        elif method == "linear":
            return self._predict_linear(self._rsu_history.get(rsu_id))
        else:
            return self._predict_average(f"rsu_{rsu_id}", self._rsu_history.get(rsu_id))

    def predict_vehicle_tasks(self, vehicle_id: int) -> float:
        """
        预测车辆下一时隙任务数

        Args:
            vehicle_id: 车辆ID

        Returns:
            predicted_tasks: 预测任务数
        """
        self.stats["total_predictions"] += 1

        method = self.config.method

        if method == "average":
            return self._predict_average(
                f"vehicle_{vehicle_id}", self._vehicle_history.get(vehicle_id)
            )
        elif method == "ema":
            return self._predict_ema(f"vehicle_{vehicle_id}")
        elif method == "linear":
            return self._predict_linear(self._vehicle_history.get(vehicle_id))
        else:
            return self._predict_average(
                f"vehicle_{vehicle_id}", self._vehicle_history.get(vehicle_id)
            )

    def _predict_average(self, key: str, history: Optional[Deque]) -> float:
        """简单平均预测"""
        if history is None or len(history) == 0:
            return 0.0
        return float(np.mean(list(history)))

    def _predict_ema(self, key: str) -> float:
        """指数移动平均预测"""
        return self._ema_state.get(key, 0.0)

    def _predict_linear(self, history: Optional[Deque]) -> float:
        """线性回归预测"""
        if history is None or len(history) < 2:
            return self._predict_average("", history)

        values = list(history)
        n = len(values)
        x = np.arange(n)

        # 简单线性回归
        x_mean = np.mean(x)
        y_mean = np.mean(values)

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return float(y_mean)

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # 预测下一个点
        prediction = slope * n + intercept

        return float(max(0, prediction))  # 非负

    def predict_total_load(self, rsu_ids: List[int]) -> float:
        """
        预测所有RSU的总负载

        Args:
            rsu_ids: RSU ID列表

        Returns:
            total_load: 预测总负载
        """
        return sum(self.predict_rsu_load(rid) for rid in rsu_ids)

    def detect_anomaly(self, rsu_id: int, current_load: float) -> bool:
        """
        检测负载异常

        Args:
            rsu_id: RSU ID
            current_load: 当前负载

        Returns:
            is_anomaly: 是否异常
        """
        history = self._rsu_history.get(rsu_id)
        if history is None or len(history) < 5:
            return False

        values = list(history)
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return False

        z_score = abs(current_load - mean) / std
        return bool(z_score > self.config.anomaly_threshold)

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "total_predictions": self.stats["total_predictions"],
            "num_rsus_tracked": len(self._rsu_history),
            "num_vehicles_tracked": len(self._vehicle_history),
        }

    def reset(self):
        """重置预测器"""
        self._rsu_history.clear()
        self._vehicle_history.clear()
        self._ema_state.clear()
        self.stats = {"total_predictions": 0, "prediction_errors": []}


# ============================================================
# 聚合负载预测器 (综合预测)
# ============================================================


class AggregateLoadPredictor:
    """
    聚合负载预测器

    功能:
    - 结合运动学预测和历史预测
    - 提供RSU级别的任务到达预测
    - 支持预测不确定性估计

    来源: docs/04_算法设计_任务负载预测.md §4
    """

    def __init__(
        self,
        kinematic_config: Optional[ETSITriggerConfig] = None,
        load_config: Optional[LoadPredictorConfig] = None,
        kinematic_weight: float = 0.6,
    ):
        """
        初始化聚合预测器

        Args:
            kinematic_config: 运动学预测器配置
            load_config: 历史负载预测器配置
            kinematic_weight: 运动学预测权重 (0-1)
        """
        self.kinematic_predictor = KinematicPredictor(kinematic_config)
        self.historical_predictor = HistoricalLoadPredictor(load_config)
        self.kinematic_weight = kinematic_weight

        # RSU-车辆映射: {rsu_id: set of vehicle_ids}
        self._rsu_coverage: Dict[int, set] = {}

    def update_rsu_coverage(self, rsu_id: int, vehicle_ids: List[int]):
        """
        更新RSU覆盖的车辆列表

        Args:
            rsu_id: RSU ID
            vehicle_ids: 覆盖的车辆ID列表
        """
        self._rsu_coverage[rsu_id] = set(vehicle_ids)

    def update_vehicle(self, vehicle_id: int, position: np.ndarray, speed: float, heading: float):
        """
        更新车辆状态

        Args:
            vehicle_id: 车辆ID
            position: 位置
            speed: 速度
            heading: 航向
        """
        self.kinematic_predictor.update_vehicle_state(vehicle_id, position, speed, heading)

    def record_trigger(self, vehicle_id: int, current_time: float):
        """
        记录任务触发

        Args:
            vehicle_id: 车辆ID
            current_time: 触发时间
        """
        self.kinematic_predictor.record_trigger(vehicle_id, current_time)
        self.historical_predictor.record_vehicle_tasks(vehicle_id, 1)

    def record_rsu_load(self, rsu_id: int, task_count: int):
        """
        记录RSU负载

        Args:
            rsu_id: RSU ID
            task_count: 任务数
        """
        self.historical_predictor.record_rsu_load(rsu_id, float(task_count))

    def predict_rsu_load(
        self, rsu_id: int, current_time: float, slot_duration: float = 0.1
    ) -> Tuple[float, float]:
        """
        预测RSU下一时隙的任务负载

        Args:
            rsu_id: RSU ID
            current_time: 当前时间
            slot_duration: 时隙长度

        Returns:
            (predicted_load, uncertainty): 预测负载和不确定性
        """
        # 运动学预测
        vehicle_ids = list(self._rsu_coverage.get(rsu_id, set()))
        kinematic_pred = self.kinematic_predictor.predict_load_next_slot(
            vehicle_ids, current_time, slot_duration
        )
        kinematic_total = sum(kinematic_pred.values())

        # 历史预测
        historical_pred = self.historical_predictor.predict_rsu_load(rsu_id)

        # 加权组合
        w = self.kinematic_weight
        combined_pred = w * kinematic_total + (1 - w) * historical_pred

        # 不确定性估计（简化：两种预测的差异）
        uncertainty = abs(kinematic_total - historical_pred)

        return combined_pred, uncertainty

    def predict_system_load(
        self, current_time: float, slot_duration: float = 0.1
    ) -> Dict[int, Tuple[float, float]]:
        """
        预测所有RSU的负载

        Args:
            current_time: 当前时间
            slot_duration: 时隙长度

        Returns:
            predictions: {rsu_id: (predicted_load, uncertainty)}
        """
        predictions = {}
        for rsu_id in self._rsu_coverage.keys():
            predictions[rsu_id] = self.predict_rsu_load(rsu_id, current_time, slot_duration)
        return predictions

    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "kinematic_stats": self.kinematic_predictor.get_statistics(),
            "historical_stats": self.historical_predictor.get_statistics(),
            "num_rsus": len(self._rsu_coverage),
            "kinematic_weight": self.kinematic_weight,
        }

    def reset(self):
        """重置预测器"""
        self.kinematic_predictor.reset()
        self.historical_predictor.reset()
        self._rsu_coverage.clear()


# ============================================================
# 单元测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("测试 任务预测算法")
    print("=" * 70)

    # 测试1: 运动学预测器
    print("\n【测试1】运动学CPM触发预测")
    config = ETSITriggerConfig(
        pos_threshold=4.0, speed_threshold=0.5, heading_threshold=4.0, T_max=1.0, T_check=0.1
    )
    kinematic = KinematicPredictor(config)

    # 初始化车辆
    kinematic.initialize_vehicle(
        vehicle_id=1,
        position=np.array([0.0, 0.0]),
        speed=20.0,  # 20 m/s = 72 km/h
        heading=0.0,
        current_time=0.0,
    )

    print(f"车辆1: 速度=20m/s, 位置阈值=4m")
    pred_time, reason = kinematic.predict_next_trigger(1, current_time=0.0)
    print(f"预测触发时间: {pred_time*1000:.0f}ms (原因: {reason})")
    print(f"理论值: {4.0/20.0*1000:.0f}ms (位置变化4m)")

    # 模拟车辆移动
    print("\n模拟车辆移动:")
    for t in [0.1, 0.15, 0.2]:
        # 更新位置
        new_pos = np.array([20.0 * t, 0.0])
        kinematic.update_vehicle_state(1, new_pos, 20.0, 0.0)

        # 检查触发
        should_trigger, trig_reason = kinematic.check_trigger_condition(1, t)
        print(f"  t={t*1000:.0f}ms: pos={new_pos[0]:.1f}m, 触发={should_trigger} ({trig_reason})")

        if should_trigger:
            kinematic.record_trigger(1, t, pred_time)
            break

    # 测试2: 触发频率
    print("\n【测试2】触发频率估计")
    # 记录多次触发
    for i in range(5):
        kinematic.record_trigger(1, i * 0.2)

    freq = kinematic.get_trigger_frequency(1)
    print(f"车辆1触发频率: {freq:.1f} Hz (预期: 5 Hz)")

    # 测试3: 历史负载预测
    print("\n【测试3】历史负载预测")
    load_config = LoadPredictorConfig(history_window=10, method="ema", ema_alpha=0.3)
    historical = HistoricalLoadPredictor(load_config)

    # 记录历史负载
    loads = [3, 4, 5, 4, 6, 5, 7, 6, 5, 8]
    for load in loads:
        historical.record_rsu_load(rsu_id=0, load=load)

    # 预测
    pred_load = historical.predict_rsu_load(rsu_id=0)
    print(f"历史负载: {loads}")
    print(f"EMA预测下一时隙负载: {pred_load:.2f}")
    print(f"简单平均: {np.mean(loads):.2f}")

    # 测试不同方法
    print("\n不同预测方法比较:")
    for method in ["average", "ema", "linear"]:
        historical.config.method = method
        pred = historical.predict_rsu_load(rsu_id=0)
        print(f"  {method}: {pred:.2f}")

    # 测试4: 异常检测
    print("\n【测试4】负载异常检测")
    normal_load = 5.0
    abnormal_load = 15.0

    is_anomaly_normal = historical.detect_anomaly(0, normal_load)
    is_anomaly_abnormal = historical.detect_anomaly(0, abnormal_load)

    print(f"负载={normal_load}: 异常={is_anomaly_normal}")
    print(f"负载={abnormal_load}: 异常={is_anomaly_abnormal}")

    # 测试5: 聚合预测器
    print("\n【测试5】聚合负载预测")
    aggregate = AggregateLoadPredictor(
        kinematic_config=config, load_config=load_config, kinematic_weight=0.6
    )

    # 设置RSU覆盖
    aggregate.update_rsu_coverage(rsu_id=0, vehicle_ids=[1, 2, 3])

    # 初始化车辆
    for vid in [1, 2, 3]:
        aggregate.update_vehicle(
            vid, position=np.array([vid * 10.0, 0.0]), speed=20.0 + vid * 2, heading=0.0
        )

    # 记录历史
    for _ in range(5):
        aggregate.record_rsu_load(0, np.random.randint(3, 6))

    # 预测
    pred, uncertainty = aggregate.predict_rsu_load(0, current_time=0.0)
    print(f"RSU 0 预测负载: {pred:.2f} ± {uncertainty:.2f}")

    # 测试6: 系统级预测
    print("\n【测试6】系统级预测")
    aggregate.update_rsu_coverage(rsu_id=1, vehicle_ids=[4, 5])
    for vid in [4, 5]:
        aggregate.update_vehicle(vid, np.array([100.0 + vid, 0.0]), 25.0, 0.0)
    for _ in range(5):
        aggregate.record_rsu_load(1, np.random.randint(2, 4))

    system_pred = aggregate.predict_system_load(current_time=0.0)
    print("系统负载预测:")
    for rsu_id, (pred, unc) in system_pred.items():
        print(f"  RSU {rsu_id}: {pred:.2f} ± {unc:.2f}")

    # 测试7: 统计信息
    print("\n【测试7】统计信息")
    print("运动学预测器:")
    for k, v in kinematic.get_statistics().items():
        print(f"  {k}: {v}")

    print("\n历史负载预测器:")
    for k, v in historical.get_statistics().items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("任务预测测试完成!")
    print("=" * 70)


# 运行测试: python3 -m simulation.algorithms.predictor
