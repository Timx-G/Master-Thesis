"""
Baseline agents for UAV-MEC NOMA offloading comparison.

All agents implement: select_action(obs) -> np.ndarray of shape (action_dim,) in [-1, 1].

Baselines:
  1. LocalOnly     — all tasks computed locally (offload_ratio = 0)
  2. FullOffload   — all tasks fully offloaded (offload_ratio = 1)
  3. RandomAgent   — uniform random actions
  4. GreedyDelay   — heuristic: fly toward nearest user, offload heavy tasks
  5. RoundRobin    — cycle through users, offload in round-robin fashion
"""

from __future__ import annotations

import numpy as np


class BaseAgent:
    """Base class — all baselines share the same interface."""

    def __init__(self, obs_dim: int, act_dim: int, num_users: int, seed: int = 0):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_users = num_users
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def _make_action(self, dx: float, dy: float,
                     offload: np.ndarray, power: np.ndarray) -> np.ndarray:
        """Build action vector in [-1, 1] space."""
        # dx, dy already in [-1, 1]
        # offload in [0, 1] -> map to [-1, 1]
        off_mapped = offload * 2.0 - 1.0
        # power in [0, 1] -> map to [-1, 1]
        pow_mapped = power * 2.0 - 1.0
        return np.concatenate([[dx, dy], off_mapped, pow_mapped]).astype(np.float32)


class LocalOnly(BaseAgent):
    """All local computation: offload_ratio=0, UAV hovers in center."""

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        offload = np.zeros(self.num_users)
        power = np.zeros(self.num_users)
        return self._make_action(0.0, 0.0, offload, power)


class FullOffload(BaseAgent):
    """Fully offload all tasks, max power, UAV hovers."""

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        offload = np.ones(self.num_users)
        power = np.ones(self.num_users)
        return self._make_action(0.0, 0.0, offload, power)


class RandomAgent(BaseAgent):
    """Uniform random actions each step."""

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, self.act_dim).astype(np.float32)


class GreedyDelay(BaseAgent):
    """Heuristic: fly toward closest user with task, offload heavy tasks.

    Strategy:
      - Move toward user with highest data size (needs offloading most)
      - Offload ratio proportional to task size
      - Power allocation proportional to distance (farther = more power)
    """

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        N = self.num_users
        # Parse obs: [uav_x, uav_y, dist_1..N, has_task_1..N, data_1..N, comp_1..N, ch_1..N, time]
        dists = obs[2: 2 + N]
        has_task = obs[2 + N: 2 + 2 * N]
        data_norm = obs[2 + 2 * N: 2 + 3 * N]

        # Move toward user with largest data and active task
        weighted = data_norm * has_task
        if weighted.sum() > 0:
            # Move toward weighted centroid (closer to heavy tasks)
            # Using distances as proxy (move toward nearest heavy-task user)
            target_idx = np.argmax(weighted / (dists + 0.01))
            # Simple: move in direction that reduces distance to target
            # Since we only have distances, not directions, use a heuristic
            dx = -0.5 * np.sign(dists[target_idx] - 0.3)
            dy = -0.5 * np.sign(dists[target_idx] - 0.3)
        else:
            dx, dy = 0.0, 0.0

        # Offload heavy tasks, keep light tasks local
        offload = np.clip(data_norm * has_task * 2.0, 0.0, 1.0)

        # Power proportional to distance (farther users need more power)
        power = np.clip(dists * has_task * 2.0, 0.0, 1.0)

        return self._make_action(float(dx), float(dy), offload, power)


class RoundRobin(BaseAgent):
    """Round-robin: cycle through users, offload one user fully per step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_user = 0

    def select_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        N = self.num_users
        has_task = obs[2 + N: 2 + 2 * N]

        # Find next user with task
        offload = np.zeros(N)
        power = np.zeros(N)
        for _ in range(N):
            if has_task[self.current_user] > 0.5:
                offload[self.current_user] = 1.0
                power[self.current_user] = 1.0
                self.current_user = (self.current_user + 1) % N
                break
            self.current_user = (self.current_user + 1) % N

        return self._make_action(0.0, 0.0, offload, power)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BASELINE_REGISTRY = {
    "local": LocalOnly,
    "full_offload": FullOffload,
    "random": RandomAgent,
    "greedy": GreedyDelay,
    "round_robin": RoundRobin,
}


def make_baseline(name: str, obs_dim: int, act_dim: int,
                  num_users: int, seed: int = 0) -> BaseAgent:
    cls = BASELINE_REGISTRY[name]
    return cls(obs_dim, act_dim, num_users, seed=seed)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("baselines.py self-test")
    print("=" * 60)

    obs_dim, act_dim, N = 53, 22, 10
    obs = np.random.randn(obs_dim).astype(np.float32)

    for name, cls in BASELINE_REGISTRY.items():
        agent = cls(obs_dim, act_dim, N, seed=42)
        action = agent.select_action(obs)
        assert action.shape == (act_dim,), f"{name}: bad shape {action.shape}"
        assert np.all(action >= -1.0) and np.all(action <= 1.0), f"{name}: out of range"
        print(f"  {name:15s}  action_mean={action.mean():.3f}  action_std={action.std():.3f}")

    # Factory test
    for name in BASELINE_REGISTRY:
        a = make_baseline(name, obs_dim, act_dim, N)
        a.select_action(obs)

    print("\nbaselines.py: ALL TESTS PASSED")
