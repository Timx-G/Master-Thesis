"""
Multi-UAV baselines for dynamic NOMA grouping (no trajectory).

Actions: [cluster_w0, cluster_w1, alpha_base] × 2 UAVs = 6 dim
  cluster_w0/w1: priority weights → softmax → user selection ordering
  alpha_base: global offloading aggressiveness [-1,1] → [0.05, 0.95]
"""
from __future__ import annotations
import numpy as np
from .model import ScenarioConfig


class GreedyGrouping:
    """Greedy: equal cluster weights (pure channel-quality ordering) + high alpha."""

    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg

    def get_action(self, env):
        action = np.zeros(env.action_dim, dtype=np.float32)
        # Equal cluster weights (0,0) → softmax([0,0,0]) = [1/3, 1/3, 1/3]
        # alpha_base = 1.0 → mapped to 0.95 (aggressive offload)
        for k in range(self.cfg.num_uavs):
            action[k * 3 + 0] = 0.0   # cluster_w0 = neutral
            action[k * 3 + 1] = 0.0   # cluster_w1 = neutral
            action[k * 3 + 2] = 1.0   # alpha_base → 0.95
        return action


class RandomGrouping:
    """Random: random cluster weights + random alpha each slot."""

    def __init__(self, cfg: ScenarioConfig, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(42)

    def get_action(self, env):
        return self.rng.uniform(-1, 1, env.action_dim).astype(np.float32)


class RoundRobinGrouping:
    """Round-Robin: cycle through cluster priorities each burst period."""

    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg
        self.slot = 0
        self.cycle_len = 15  # change priority every ~15 slots

    def get_action(self, env):
        action = np.zeros(env.action_dim, dtype=np.float32)
        phase = (self.slot // self.cycle_len) % 3

        for k in range(self.cfg.num_uavs):
            # Prioritize one cluster at a time
            if phase == 0:
                action[k * 3 + 0] = 1.0   # cluster 0 high
                action[k * 3 + 1] = -1.0  # cluster 1 low
            elif phase == 1:
                action[k * 3 + 0] = -1.0
                action[k * 3 + 1] = 1.0   # cluster 1 high
            else:
                action[k * 3 + 0] = -1.0
                action[k * 3 + 1] = -1.0  # cluster 2 high (reference=0 > -1)
            action[k * 3 + 2] = 1.0   # alpha_base → 0.95
        self.slot += 1
        return action


class FixedBestGrouping:
    """Fixed best: equal weights + alpha=0.95. The "Hover+alpha=0.95" baseline."""

    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg

    def get_action(self, env):
        action = np.zeros(env.action_dim, dtype=np.float32)
        for k in range(self.cfg.num_uavs):
            action[k * 3 + 2] = 1.0   # alpha_base → 0.95
        return action
