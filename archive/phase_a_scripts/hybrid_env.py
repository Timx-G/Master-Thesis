"""
Hybrid Environment Wrapper: 5-dim DRL action + closed-form per-user optimizer.

DRL decides (5-dim):
  [0] dx: UAV movement x
  [1] dy: UAV movement y
  [2] cluster_ratio_0: resource share for cluster 0 vs others
  [3] cluster_ratio_1: resource share for cluster 1 vs others
  [4] alpha_base: global offloading aggressiveness

Per-user optimizer computes (40-dim):
  α_i = alpha_base * sigmoid(channel_quality_i)  (better channel → more offload)
  p_i = proportional to α_i * D_i  (more data to send → more power)
"""
from __future__ import annotations
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .env import UAVMECEnv
from .model import ScenarioConfig


class HybridUAVMECEnv(gym.Wrapper):
    """Wraps UAVMECEnv: DRL outputs 5-dim, optimizer fills per-user actions."""

    def __init__(self, cfg: ScenarioConfig = None, **kwargs):
        self.cfg = cfg or ScenarioConfig()
        self.inner_env = UAVMECEnv(cfg=self.cfg, **kwargs)
        super().__init__(self.inner_env)

        N = self.cfg.num_users
        # DRL action: 5-dim
        self.drl_action_dim = 5
        self.action_space = spaces.Box(-1, 1, shape=(self.drl_action_dim,), dtype=np.float32)
        # Obs same as inner env
        self.observation_space = self.inner_env.observation_space
        self.obs_dim = self.inner_env.obs_dim
        self.action_dim = self.drl_action_dim

        # Precompute cluster assignments (set in reset)
        self._cluster_ids = None

    def reset(self, **kwargs):
        obs, info = self.inner_env.reset(**kwargs)
        self._assign_clusters()
        return obs, info

    def _assign_clusters(self):
        """Assign each user to nearest cluster center (K-means style)."""
        N = self.cfg.num_users
        nc = self.cfg.num_clusters
        pos = self.inner_env.user_positions  # (N, 2)
        # Simple K-means: find cluster centers from user positions
        from scipy.cluster.vq import kmeans2
        try:
            centers, labels = kmeans2(pos, nc, minit='points')
        except Exception:
            labels = np.random.randint(0, nc, N)
        self._cluster_ids = labels

    def _compute_per_user_actions(self, drl_action):
        """Convert 5-dim DRL action to N-dim (α_i, p_i) using channel-aware optimizer."""
        N = self.cfg.num_users
        nc = self.cfg.num_clusters

        dx, dy = drl_action[0], drl_action[1]
        # Cluster resource ratios via softmax
        raw_ratios = np.array([drl_action[2], drl_action[3],
                               0.0])[:nc]  # pad if nc < 3
        exp_r = np.exp(raw_ratios - raw_ratios.max())
        cluster_weights = exp_r / exp_r.sum()

        # Global offloading aggressiveness: map [-1,1] to [0.05, 0.95]
        alpha_base = 0.05 + 0.9 * (drl_action[4] + 1) / 2

        # Get channel gains for each user
        uav_pos = self.inner_env.uav.position
        user_pos = self.inner_env.user_positions
        h = np.zeros(N)
        for i in range(N):
            d_h = np.linalg.norm(uav_pos[:2] - user_pos[i])
            h[i] = self.inner_env.channel.path_loss(d_h)

        # Normalize channel gains to [0, 1]
        h_log = np.log10(h + 1e-20)
        h_min, h_max = h_log.min(), h_log.max()
        if h_max - h_min > 1e-8:
            h_norm = (h_log - h_min) / (h_max - h_min)
        else:
            h_norm = np.ones(N) * 0.5

        # Per-user α: better channel → more offload, weighted by cluster priority
        alphas = np.zeros(N)
        powers = np.zeros(N)
        for i in range(N):
            cid = self._cluster_ids[i] if self._cluster_ids is not None else 0
            cw = cluster_weights[min(cid, nc-1)]
            # α_i = alpha_base * channel_quality * cluster_weight
            alphas[i] = np.clip(alpha_base * (0.3 + 0.7 * h_norm[i]) * (0.5 + cw), 0, 1)
            # Power proportional to α and data size (if task active)
            if self.inner_env.task_queue_active[i]:
                powers[i] = np.clip(alphas[i] * 0.8 + 0.1, 0.1, 1.0)
            else:
                powers[i] = 0.1

        # Convert to [-1, 1] action space of inner env
        full_action = np.zeros(2 + 2*N, dtype=np.float32)
        full_action[0] = dx
        full_action[1] = dy
        full_action[2:2+N] = alphas * 2 - 1      # [0,1] → [-1,1]
        full_action[2+N:2+2*N] = powers * 2 - 1   # [0,1] → [-1,1]
        return full_action

    def step(self, drl_action):
        drl_action = np.asarray(drl_action, dtype=np.float32).clip(-1, 1)
        full_action = self._compute_per_user_actions(drl_action)
        return self.inner_env.step(full_action)
