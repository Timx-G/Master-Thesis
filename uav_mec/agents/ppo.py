"""
PPO baseline for UAV-MEC offloading (without Lagrangian constraints).

Used to demonstrate that SAC-Lagrangian outperforms unconstrained PPO.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOAgent:
    """Proximal Policy Optimization (clip variant) for continuous actions.

    Simplified and stabilised for the UAV-MEC environment.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 lr: float = 1e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, epochs: int = 4, hidden: int = 128,
                 mini_batch: int = 64,
                 seed: int = 0, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.mini_batch = mini_batch
        self.rng = np.random.default_rng(seed)

        # Shared trunk + heads
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        ).to(self.device)
        self.mu_head = nn.Linear(hidden, act_dim).to(self.device)
        self.v_head = nn.Linear(hidden, 1).to(self.device)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim, device=self.device))

        # Init weights small
        for m in [self.net, self.mu_head, self.v_head]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.orthogonal_(p, gain=0.5)

        all_params = (
            list(self.net.parameters()) +
            list(self.mu_head.parameters()) +
            list(self.v_head.parameters()) +
            [self.log_std]
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr, eps=1e-5)

        # Running observation normalizer
        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_var = np.ones(obs_dim, dtype=np.float64)
        self.obs_count = 1e-4

        self._reset_buffer()
        self._cached_log_prob = 0.0
        self._cached_val = 0.0

    def _reset_buffer(self):
        self.buf_obs = []
        self.buf_act = []
        self.buf_rew = []
        self.buf_done = []
        self.buf_logp = []
        self.buf_val = []

    def _update_obs_stats(self, obs: np.ndarray):
        """Welford online mean/var."""
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += (delta * delta2 - self.obs_var) / self.obs_count

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(self.obs_var, 1e-6)) + 1e-8
        return np.clip((obs - self.obs_mean) / std, -10.0, 10.0).astype(np.float32)

    def _forward(self, obs_t: torch.Tensor):
        h = self.net(obs_t)
        mu = self.mu_head(h)
        val = self.v_head(h).squeeze(-1)
        std = self.log_std.exp().clamp(min=0.05, max=1.0).expand_as(mu)
        return mu, std, val

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self._update_obs_stats(obs)
        obs_n = self._norm_obs(obs)

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(self.device)
            mu, std, val = self._forward(obs_t)
            if torch.isnan(mu).any():
                return np.zeros(self.act_dim, dtype=np.float32)
            dist = Normal(mu, std)
            if deterministic:
                action = mu
            else:
                action = dist.sample()
            action = action.clamp(-1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)

        self._cached_log_prob = log_prob.item()
        self._cached_val = val.item()
        return action.cpu().numpy().flatten()

    def store(self, obs, act, rew, done):
        self.buf_obs.append(self._norm_obs(obs))
        self.buf_act.append(act.copy())
        self.buf_rew.append(float(rew))
        self.buf_done.append(float(done))
        self.buf_logp.append(self._cached_log_prob)
        self.buf_val.append(self._cached_val)

    def update(self) -> dict:
        T = len(self.buf_obs)
        if T < 4:
            self._reset_buffer()
            return {}

        obs = torch.FloatTensor(np.array(self.buf_obs)).to(self.device)
        act = torch.FloatTensor(np.array(self.buf_act)).to(self.device)
        old_logp = torch.FloatTensor(self.buf_logp).to(self.device)
        vals = torch.FloatTensor(self.buf_val).to(self.device)

        # GAE
        advs = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            nv = vals[t + 1].item() if t + 1 < T else 0.0
            mask = 1.0 - self.buf_done[t]
            delta = self.buf_rew[t] + self.gamma * nv * mask - vals[t].item()
            gae = delta + self.gamma * self.lam * mask * gae
            gae = max(-500.0, min(500.0, gae))  # prevent float32 overflow
            advs[t] = gae
        rets = (advs + vals).detach()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        total_loss = 0.0
        n_up = 0
        for _ in range(self.epochs):
            idx = torch.randperm(T, device=self.device)
            for s in range(0, T, self.mini_batch):
                mb = idx[s: s + self.mini_batch]
                if len(mb) < 2:
                    continue
                mu, std, v = self._forward(obs[mb])
                if torch.isnan(mu).any():
                    continue
                dist = Normal(mu, std)
                new_logp = dist.log_prob(act[mb]).sum(-1)
                ratio = (new_logp - old_logp[mb]).exp().clamp(0.0, 10.0)

                s1 = ratio * advs[mb]
                s2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advs[mb]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = (v - rets[mb]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + 0.5 * v_loss - 0.01 * ent

                if torch.isnan(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                has_nan_grad = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in self.optimizer.param_groups[0]["params"]
                )
                if has_nan_grad:
                    self.optimizer.zero_grad()
                    continue
                nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 0.3)
                self.optimizer.step()
                total_loss += loss.item()
                n_up += 1

        self._reset_buffer()
        return {"ppo_loss": total_loss / max(n_up, 1)}

    def save(self, path: str):
        torch.save({
            "net": self.net.state_dict(),
            "mu_head": self.mu_head.state_dict(),
            "v_head": self.v_head.state_dict(),
            "log_std": self.log_std.data,
            "obs_mean": self.obs_mean,
            "obs_var": self.obs_var,
            "obs_count": self.obs_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.mu_head.load_state_dict(ckpt["mu_head"])
        self.v_head.load_state_dict(ckpt["v_head"])
        self.log_std.data = ckpt["log_std"]
        self.obs_mean = ckpt["obs_mean"]
        self.obs_var = ckpt["obs_var"]
        self.obs_count = ckpt["obs_count"]


if __name__ == "__main__":
    print("=" * 60)
    print("ppo.py self-test")
    print("=" * 60)
    agent = PPOAgent(53, 22)
    obs = np.random.randn(53).astype(np.float32)
    act = agent.select_action(obs)
    assert act.shape == (22,)
    agent.store(obs, act, -1.0, False)

    for i in range(300):
        o = np.random.randn(53).astype(np.float32)
        a = agent.select_action(o)
        agent.store(o, a, np.random.randn(), i == 299)

    info = agent.update()
    print(f"  PPO loss: {info.get('ppo_loss', 'N/A')}")
    print("\nppo.py: ALL TESTS PASSED")
