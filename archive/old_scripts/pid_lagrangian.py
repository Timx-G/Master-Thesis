"""
PID-Lagrangian PPO agent (Stooke et al. 2020).

Replaces the simple projected gradient ascent λ update with a
PID controller, reducing multiplier oscillation:
    e_t   = avg_delay_ep - delay_budget
    λ += Kp*e_t + Ki*integral(e) + Kd*(e_t - e_{t-1})
    λ = clip(λ, 0, lambda_max)

Everything else is identical to PPOAgent.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class PIDLagrangianConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    hidden: int = 128
    mini_batch: int = 64
    delay_budget: float = 0.5
    lambda_init: float = 0.1
    lambda_max: float = 5.0
    Kp: float = 3e-2       # proportional
    Ki: float = 1e-3       # integral
    Kd: float = 3e-3       # derivative


class PIDLagrangianAgent:
    """PPO + PID controller for Lagrangian multiplier (Stooke 2020)."""

    def __init__(self, obs_dim: int, act_dim: int,
                 cfg: Optional[PIDLagrangianConfig] = None,
                 seed: int = 0, device: str = "cpu"):
        self.cfg = cfg or PIDLagrangianConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)

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

        all_params = (list(self.net.parameters()) +
                      list(self.mu_head.parameters()) +
                      list(self.v_head.parameters()) + [self.log_std])
        self.optimizer = torch.optim.Adam(all_params, lr=self.cfg.lr, eps=1e-5)

        self._lambda_d = self.cfg.lambda_init
        self._integral = 0.0
        self._prev_error = 0.0
        self._episode_costs: list[float] = []

        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_var = np.ones(obs_dim, dtype=np.float64)
        self.obs_count = 1e-4
        self._reset_buffer()
        self._cached_log_prob = 0.0
        self._cached_val = 0.0

    @property
    def lambda_d(self) -> float:
        return self._lambda_d

    def _update_obs_stats(self, obs):
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        self.obs_var += (delta * (obs - self.obs_mean) - self.obs_var) / self.obs_count

    def _norm_obs(self, obs):
        std = np.sqrt(np.maximum(self.obs_var, 1e-6)) + 1e-8
        return np.clip((obs - self.obs_mean) / std, -10.0, 10.0).astype(np.float32)

    def _forward(self, obs_t):
        h = self.net(obs_t)
        mu = self.mu_head(h)
        val = self.v_head(h).squeeze(-1)
        std = self.log_std.exp().clamp(min=0.05, max=1.0).expand_as(mu)
        return mu, std, val

    def select_action(self, obs, deterministic=False):
        self._update_obs_stats(obs)
        obs_n = self._norm_obs(obs)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(self.device)
            mu, std, val = self._forward(obs_t)
            if torch.isnan(mu).any():
                return np.zeros(self.act_dim, dtype=np.float32)
            dist = Normal(mu, std)
            action = mu if deterministic else dist.sample()
            action = action.clamp(-1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)
        self._cached_log_prob = log_prob.item()
        self._cached_val = val.item()
        return action.cpu().numpy().flatten()

    def store(self, obs, act, rew, done, cost_delay: float = 0.0):
        obs_n = self._norm_obs(obs)
        modified_rew = rew - self._lambda_d * cost_delay
        self.buf_obs.append(obs_n)
        self.buf_act.append(act.copy())
        self.buf_rew.append(float(modified_rew))
        self.buf_done.append(float(done))
        self.buf_logp.append(self._cached_log_prob)
        self.buf_val.append(self._cached_val)
        self._episode_costs.append(cost_delay)

    def _reset_buffer(self):
        self.buf_obs = []; self.buf_act = []; self.buf_rew = []
        self.buf_done = []; self.buf_logp = []; self.buf_val = []

    def update(self) -> dict:
        # PID multiplier update
        if self._episode_costs:
            avg_cost = np.mean(self._episode_costs)
            e = avg_cost - self.cfg.delay_budget
            self._integral += e
            d_term = e - self._prev_error
            delta = self.cfg.Kp * e + self.cfg.Ki * self._integral + self.cfg.Kd * d_term
            self._lambda_d = float(np.clip(self._lambda_d + delta,
                                           0.0, self.cfg.lambda_max))
            self._prev_error = e
            self._episode_costs = []

        T = len(self.buf_obs)
        if T < 4:
            self._reset_buffer()
            return {}

        obs = torch.FloatTensor(np.array(self.buf_obs)).to(self.device)
        act = torch.FloatTensor(np.array(self.buf_act)).to(self.device)
        old_logp = torch.FloatTensor(self.buf_logp).to(self.device)
        vals = torch.FloatTensor(self.buf_val).to(self.device)

        advs = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            nv = vals[t+1].item() if t+1 < T else 0.0
            mask = 1.0 - self.buf_done[t]
            delta = self.buf_rew[t] + self.cfg.gamma * nv * mask - vals[t].item()
            gae = delta + self.cfg.gamma * self.cfg.lam * mask * gae
            advs[t] = gae
        rets = (advs + vals).detach()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

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
                s2 = ratio.clamp(1-self.cfg.clip_eps, 1+self.cfg.clip_eps) * advs[mb]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = (v - rets[mb]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + 0.5 * v_loss - 0.01 * ent
                if torch.isnan(loss):
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                              for p in self.optimizer.param_groups[0]["params"])
                if has_nan:
                    self.optimizer.zero_grad()
                    continue
                nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 0.3)
                self.optimizer.step()

        self._reset_buffer()
        return {"lambda_d": self._lambda_d}
