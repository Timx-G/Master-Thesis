"""
TD3 (Twin Delayed DDPG) baseline for UAV-MEC offloading.

Fujimoto et al. "Addressing Function Approximation Error in Actor-Critic Methods" (ICML 2018)

Key features over DDPG:
  - Twin Q-networks to reduce overestimation
  - Delayed policy updates (every d steps)
  - Target policy smoothing (noise on target actions)

Used to demonstrate that SAC-Lagrangian's constraint handling provides
advantage beyond just using a better base RL algorithm.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TD3Agent:
    """Twin Delayed DDPG for continuous action spaces."""

    def __init__(self, obs_dim: int, act_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 policy_delay: int = 2, noise_std: float = 0.2,
                 target_noise: float = 0.2, noise_clip: float = 0.5,
                 hidden: int = 256, buffer_size: int = 200_000,
                 batch_size: int = 256, warmup_steps: int = 2000,
                 seed: int = 0, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.rng = np.random.default_rng(seed)

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # Twin Critics
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(self.device)
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Init weights
        for net in [self.actor, self.q1, self.q2]:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.zeros_(m.bias)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        # Observation normalization (Welford)
        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_var = np.ones(obs_dim, dtype=np.float64)
        self.obs_count = 1e-4

        # Replay buffer
        self.buf_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.buf_act = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.buf_rew = np.zeros(buffer_size, dtype=np.float32)
        self.buf_next = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.buf_done = np.zeros(buffer_size, dtype=np.float32)
        self.buf_ptr = 0
        self.buf_size = 0
        self.buf_max = buffer_size

        self.total_steps = 0
        self._update_count = 0

    def _update_obs_stats(self, obs: np.ndarray):
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += (delta * delta2 - self.obs_var) / self.obs_count

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        return ((obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)).astype(np.float32)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self._update_obs_stats(obs)
        obs_n = self._norm_obs(obs)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(self.device)
            action = self.actor(obs_t).cpu().numpy().flatten()
        if np.any(np.isnan(action)):
            action = np.zeros(self.act_dim, dtype=np.float32)
        if not deterministic:
            noise = self.rng.normal(0, self.noise_std, self.act_dim).astype(np.float32)
            action = (action + noise).clip(-1.0, 1.0)
        return action

    def store(self, obs, act, rew, next_obs, done):
        obs_n = self._norm_obs(obs)
        next_obs_n = self._norm_obs(next_obs)
        self.buf_obs[self.buf_ptr] = obs_n
        self.buf_act[self.buf_ptr] = act
        self.buf_rew[self.buf_ptr] = rew
        self.buf_next[self.buf_ptr] = next_obs_n
        self.buf_done[self.buf_ptr] = float(done)
        self.buf_ptr = (self.buf_ptr + 1) % self.buf_max
        self.buf_size = min(self.buf_size + 1, self.buf_max)
        self.total_steps += 1

    def update(self) -> dict:
        if self.buf_size < self.warmup_steps:
            return {}

        idx = self.rng.integers(0, self.buf_size, self.batch_size)
        obs = torch.FloatTensor(self.buf_obs[idx]).to(self.device)
        act = torch.FloatTensor(self.buf_act[idx]).to(self.device)
        rew = torch.FloatTensor(self.buf_rew[idx]).unsqueeze(1).to(self.device)
        nxt = torch.FloatTensor(self.buf_next[idx]).to(self.device)
        done = torch.FloatTensor(self.buf_done[idx]).unsqueeze(1).to(self.device)

        # ---- Critic update ----
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(act) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_act = (self.actor_target(nxt) + noise).clamp(-1.0, 1.0)

            q1_next = self.q1_target(torch.cat([nxt, next_act], -1))
            q2_next = self.q2_target(torch.cat([nxt, next_act], -1))
            q_next = torch.min(q1_next, q2_next)
            q_target = rew + (1.0 - done) * self.gamma * q_next
            q_target = q_target.clamp(-2000.0, 0.0)

        sa = torch.cat([obs, act], -1)
        q1_val = self.q1(sa)
        q2_val = self.q2(sa)
        critic_loss = F.mse_loss(q1_val, q_target) + F.mse_loss(q2_val, q_target)

        if not torch.isnan(critic_loss):
            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
            self.critic_opt.step()

        # ---- Delayed policy update ----
        self._update_count += 1
        actor_loss_val = 0.0
        if self._update_count % self.policy_delay == 0:
            actor_loss = -self.q1(torch.cat([obs, self.actor(obs)], -1)).mean()
            if not torch.isnan(actor_loss):
                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_opt.step()
                actor_loss_val = actor_loss.item()

            # Soft update targets
            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "obs_mean": self.obs_mean,
            "obs_var": self.obs_var,
            "obs_count": self.obs_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target = copy.deepcopy(self.actor)
        self.q1.load_state_dict(ckpt["q1"])
        self.q1_target = copy.deepcopy(self.q1)
        self.q2.load_state_dict(ckpt["q2"])
        self.q2_target = copy.deepcopy(self.q2)
        self.obs_mean = ckpt["obs_mean"]
        self.obs_var = ckpt["obs_var"]
        self.obs_count = ckpt["obs_count"]


if __name__ == "__main__":
    print("=" * 60)
    print("td3.py self-test")
    print("=" * 60)
    agent = TD3Agent(53, 22)

    obs = np.random.randn(53).astype(np.float32)
    act = agent.select_action(obs)
    assert act.shape == (22,), f"Bad shape: {act.shape}"
    assert np.all(np.abs(act) <= 1.0), "Action out of [-1,1]"

    act_det = agent.select_action(obs, deterministic=True)
    assert act_det.shape == (22,)

    # Fill buffer
    for i in range(2500):
        o = np.random.randn(53).astype(np.float32)
        a = np.random.uniform(-1, 1, 22).astype(np.float32)
        r = np.random.randn()
        o2 = np.random.randn(53).astype(np.float32)
        agent.store(o, a, r, o2, i == 2499)

    for _ in range(5):
        info = agent.update()

    assert "critic_loss" in info
    print(f"  critic_loss={info['critic_loss']:.4f}")
    print(f"  actor_loss={info['actor_loss']:.4f}")

    # Test save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        agent.save(f.name)
        agent2 = TD3Agent(53, 22)
        agent2.load(f.name)
        os.unlink(f.name)

    a1 = agent.select_action(obs, deterministic=True)
    a2 = agent2.select_action(obs, deterministic=True)
    assert np.allclose(a1, a2, atol=1e-5), "Save/load mismatch"

    print("\ntd3.py: ALL TESTS PASSED")
