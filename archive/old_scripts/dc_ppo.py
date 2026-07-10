"""
DC-PPO: Lagrangian-Constrained PPO for UAV-MEC NOMA Offloading.

Core mechanism: PPO + Lagrangian dual multiplier for delay constraint.
  - Reward augmented: r' = r - λ_d * cost_delay
  - λ_d updated via projected gradient ascent each episode
  - No Safety Layer (causes IS ratio bias in on-policy PPO)
  - No phased scheduling (PPO on-policy cannot recover from early bad habits)

Reference:
  - Tessler et al. 2019, "Reward Constrained Policy Optimization"
  - Schulman et al. 2017, "Proximal Policy Optimization Algorithms"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class DCPPOConfig:
    """DC-PPO hyperparameters."""
    # PPO base
    lr: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    hidden: int = 128
    mini_batch: int = 64

    # Lagrangian constraint
    delay_budget: float = 0.5        # soft constraint target (per-task avg delay, s)
    lambda_init: float = 0.1
    lambda_max: float = 5.0          # cap in sweet-spot [4,6]: r'=9-5×1.2=3>0, no collapse
    lr_lambda: float = 3e-2          # fast: reaches cap ~240ep, then stable training


class DCPPOAgent:
    """Lagrangian-Constrained PPO agent.

    Augments PPO reward with Lagrangian penalty:
        r' = r - λ_d * cost_delay
    where λ_d is updated via projected gradient ascent after each episode.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 cfg: Optional[DCPPOConfig] = None,
                 seed: int = 0, device: str = "cpu"):
        self.cfg = cfg or DCPPOConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)

        # ---- PPO networks (identical to PPOAgent) ----
        self.net = nn.Sequential(
            nn.Linear(obs_dim, self.cfg.hidden), nn.Tanh(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden), nn.Tanh(),
        ).to(self.device)
        self.mu_head = nn.Linear(self.cfg.hidden, act_dim).to(self.device)
        self.v_head = nn.Linear(self.cfg.hidden, 1).to(self.device)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim, device=self.device))

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
        self.optimizer = torch.optim.Adam(all_params, lr=self.cfg.lr, eps=1e-5)

        # ---- Lagrangian dual variable ----
        self._lambda_d = self.cfg.lambda_init
        self._episode_costs = []
        self._episode_count = 0

        # ---- Observation normalization (Welford online algorithm) ----
        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_var = np.ones(obs_dim, dtype=np.float64)
        self.obs_M2 = np.zeros(obs_dim, dtype=np.float64)   # sum of squared deviations
        self.obs_count = 1e-4

        # ---- PPO rollout buffer ----
        self._reset_buffer()
        self._cached_log_prob = 0.0
        self._cached_val = 0.0

    @property
    def lambda_d(self) -> float:
        return self._lambda_d

    # ---- Observation normalization ----

    def _update_obs_stats(self, obs: np.ndarray):
        # Standard Welford online algorithm: track M2 (sum of squared deviations),
        # then derive variance as M2 / count. Avoids the per-step division that
        # accumulated floating-point error in the previous obs_var running-average form.
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean   # updated mean used here (Welford step)
        self.obs_M2 += delta * delta2
        self.obs_var = self.obs_M2 / max(self.obs_count, 1)

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(self.obs_var, 1e-6)) + 1e-8
        return np.clip((obs - self.obs_mean) / std, -10.0, 10.0).astype(np.float32)

    # ---- PPO forward ----

    def _forward(self, obs_t: torch.Tensor):
        h = self.net(obs_t)
        mu = self.mu_head(h)
        val = self.v_head(h).squeeze(-1)
        std = self.log_std.exp().clamp(min=0.05, max=1.0).expand_as(mu)
        return mu, std, val

    # ---- Action selection (no Safety Layer — pure PPO) ----

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

    # ---- Store with Lagrangian-augmented reward ----

    def store(self, obs, act, rew, done, cost_delay: float = 0.0):
        obs_n = self._norm_obs(obs)

        # Lagrangian reward augmentation: r' = r - λ_d * cost_delay
        modified_rew = rew - self._lambda_d * cost_delay

        self.buf_obs.append(obs_n)
        self.buf_act.append(act.copy())
        self.buf_rew.append(float(modified_rew))
        self.buf_done.append(float(done))
        self.buf_logp.append(self._cached_log_prob)
        self.buf_val.append(self._cached_val)

        self._episode_costs.append(cost_delay)

    # ---- End-of-episode update ----

    def _reset_buffer(self):
        self.buf_obs = []
        self.buf_act = []
        self.buf_rew = []
        self.buf_done = []
        self.buf_logp = []
        self.buf_val = []

    def update(self) -> dict:
        self._episode_count += 1

        # Lagrangian λ update (projected gradient ascent)
        if self._episode_costs:
            avg_cost = np.mean(self._episode_costs)
            self._lambda_d += self.cfg.lr_lambda * (avg_cost - self.cfg.delay_budget)
            self._lambda_d = max(0.0, min(self._lambda_d, self.cfg.lambda_max))
            self._episode_costs = []

        # PPO update
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
            delta = self.buf_rew[t] + self.cfg.gamma * nv * mask - vals[t].item()
            gae = delta + self.cfg.gamma * self.cfg.lam * mask * gae
            gae = max(-500.0, min(500.0, gae))
            advs[t] = gae
        rets = (advs + vals).detach()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        total_loss = 0.0
        n_up = 0
        for _ in range(self.cfg.epochs):
            idx = torch.randperm(T, device=self.device)
            for s in range(0, T, self.cfg.mini_batch):
                mb = idx[s: s + self.cfg.mini_batch]
                if len(mb) < 2:
                    continue
                mu, std, v = self._forward(obs[mb])
                if torch.isnan(mu).any():
                    continue
                dist = Normal(mu, std)
                new_logp = dist.log_prob(act[mb]).sum(-1)
                ratio = (new_logp - old_logp[mb]).exp().clamp(0.0, 10.0)

                s1 = ratio * advs[mb]
                s2 = ratio.clamp(1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advs[mb]
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
        return {"ppo_loss": total_loss / max(n_up, 1),
                "lambda_d": self._lambda_d,
                "episode": self._episode_count}

    # ---- Save / Load ----

    def save(self, path: str):
        torch.save({
            "net": self.net.state_dict(),
            "mu_head": self.mu_head.state_dict(),
            "v_head": self.v_head.state_dict(),
            "log_std": self.log_std.data,
            "obs_mean": self.obs_mean,
            "obs_var": self.obs_var,
            "obs_count": self.obs_count,
            "lambda_d": self._lambda_d,
            "episode_count": self._episode_count,
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
        self._lambda_d = ckpt["lambda_d"]
        self._episode_count = ckpt["episode_count"]


if __name__ == "__main__":
    print("=" * 60)
    print("dc_ppo.py v2 self-test")
    print("=" * 60)
    agent = DCPPOAgent(53, 22)
    obs = np.random.randn(53).astype(np.float32)
    act = agent.select_action(obs)
    assert act.shape == (22,)
    agent.store(obs, act, -1.0, False, cost_delay=0.5)

    for i in range(300):
        o = np.random.randn(53).astype(np.float32)
        a = agent.select_action(o)
        agent.store(o, a, np.random.randn(), i == 299, cost_delay=np.random.rand())

    info = agent.update()
    print(f"  PPO loss: {info.get('ppo_loss', 'N/A'):.4f}")
    print(f"  Lambda_d: {info.get('lambda_d', 'N/A'):.4f}")
    print(f"  Episode: {info.get('episode', 'N/A')}")
    print("\ndc_ppo.py v2: ALL TESTS PASSED")
