"""
DC-PPO v2: Dual-Constrained PPO with Warm-Start Lambda.

Key changes from v1:
  1. TWO constraints: delay (λ_d) + fairness (λ_f) → "Dual-Constrained" is real
  2. Warm-start: λ_d_init=5.0, λ_f_init=1.0 (no warm-up poison)
  3. Per-constraint learning rates: delay fast, fairness slow
  4. GAE overflow guard
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class DCPPOv2Config:
    lr: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95          # GAE lambda
    clip_eps: float = 0.2
    epochs: int = 4
    hidden: int = 128
    mini_batch: int = 64
    # Delay constraint
    delay_budget: float = 0.5
    lambda_d_init: float = 5.0   # warm-start! (not 0.1)
    lambda_d_max: float = 10.0
    lr_lambda_d: float = 3e-2
    # Fairness constraint
    fairness_target: float = 0.7
    lambda_f_init: float = 1.0
    lambda_f_max: float = 5.0
    lr_lambda_f: float = 1e-2    # slower than delay


class DCPPOv2Agent:
    """Dual-Constrained PPO: delay + fairness, warm-started lambdas."""

    def __init__(self, obs_dim: int, act_dim: int,
                 cfg: Optional[DCPPOv2Config] = None,
                 seed: int = 0, device: str = "cpu"):
        self.cfg = cfg or DCPPOv2Config()
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

        # Dual variables (warm-started!)
        self._lambda_d = self.cfg.lambda_d_init
        self._lambda_f = self.cfg.lambda_f_init
        self._ep_delay_costs = []
        self._ep_fairness_costs = []

        # Obs normalization (Welford online algorithm)
        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_var = np.ones(obs_dim, dtype=np.float64)
        self.obs_M2 = np.zeros(obs_dim, dtype=np.float64)   # sum of squared deviations
        self.obs_count = 1e-4

        self._reset_buffer()
        self._cached_log_prob = 0.0
        self._cached_val = 0.0

    @property
    def lambda_d(self): return self._lambda_d
    @property
    def lambda_f(self): return self._lambda_f

    def _update_obs_stats(self, obs):
        # Standard Welford online algorithm: accumulate M2, derive var as M2/count.
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean   # updated mean (Welford step)
        self.obs_M2 += delta * delta2
        self.obs_var = self.obs_M2 / max(self.obs_count, 1)

    def _norm_obs(self, obs):
        std = np.sqrt(np.maximum(self.obs_var, 1e-6)) + 1e-8
        return np.clip((obs - self.obs_mean) / std, -10, 10).astype(np.float32)

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
            action = action.clamp(-1, 1)
            log_prob = dist.log_prob(action).sum(-1)
        self._cached_log_prob = log_prob.item()
        self._cached_val = val.item()
        return action.cpu().numpy().flatten()

    def store(self, obs, act, rew, done, cost_delay=0.0, cost_fairness=0.0):
        obs_n = self._norm_obs(obs)
        # Dual-penalized reward
        mod_rew = rew - self._lambda_d * cost_delay - self._lambda_f * cost_fairness
        mod_rew = np.clip(mod_rew, -20.0, 20.0)

        self.buf_obs.append(obs_n)
        self.buf_act.append(act.copy())
        self.buf_rew.append(float(mod_rew))
        self.buf_done.append(float(done))
        self.buf_logp.append(self._cached_log_prob)
        self.buf_val.append(self._cached_val)
        self._ep_delay_costs.append(cost_delay)
        self._ep_fairness_costs.append(cost_fairness)

    def _reset_buffer(self):
        self.buf_obs, self.buf_act, self.buf_rew = [], [], []
        self.buf_done, self.buf_logp, self.buf_val = [], [], []

    def update(self):
        # Dual variable updates (projected gradient ascent)
        if self._ep_delay_costs:
            avg_d = np.mean(self._ep_delay_costs)
            self._lambda_d += self.cfg.lr_lambda_d * (avg_d - self.cfg.delay_budget)
            self._lambda_d = max(0, min(self._lambda_d, self.cfg.lambda_d_max))
            self._ep_delay_costs = []

        if self._ep_fairness_costs:
            avg_f = np.mean(self._ep_fairness_costs)
            self._lambda_f += self.cfg.lr_lambda_f * avg_f  # cost>0 when fairness violated
            self._lambda_f = max(0, min(self._lambda_f, self.cfg.lambda_f_max))
            self._ep_fairness_costs = []

        # PPO update
        T = len(self.buf_obs)
        if T < 4:
            self._reset_buffer()
            return {}

        obs = torch.FloatTensor(np.array(self.buf_obs)).to(self.device)
        act = torch.FloatTensor(np.array(self.buf_act)).to(self.device)
        old_logp = torch.FloatTensor(self.buf_logp).to(self.device)
        vals = torch.FloatTensor(self.buf_val).to(self.device)

        # GAE with overflow guard
        advs = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            nv = vals[t+1].item() if t+1 < T else 0.0
            mask = 1.0 - self.buf_done[t]
            delta = self.buf_rew[t] + self.cfg.gamma * nv * mask - vals[t].item()
            gae = delta + self.cfg.gamma * self.cfg.lam * mask * gae
            gae = max(-500.0, min(500.0, gae))
            advs[t] = gae
        rets = (advs + vals).detach()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        total_loss, n_up = 0.0, 0
        for _ in range(self.cfg.epochs):
            idx = torch.randperm(T, device=self.device)
            for s in range(0, T, self.cfg.mini_batch):
                mb = idx[s:s+self.cfg.mini_batch]
                if len(mb) < 2: continue
                mu, std, v = self._forward(obs[mb])
                if torch.isnan(mu).any(): continue
                dist = Normal(mu, std)
                new_logp = dist.log_prob(act[mb]).sum(-1)
                ratio = (new_logp - old_logp[mb]).exp().clamp(0, 10)
                s1 = ratio * advs[mb]
                s2 = ratio.clamp(1-self.cfg.clip_eps, 1+self.cfg.clip_eps) * advs[mb]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = (v - rets[mb]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + 0.5*v_loss - 0.01*ent
                if torch.isnan(loss): continue
                self.optimizer.zero_grad()
                loss.backward()
                if any(p.grad is not None and torch.isnan(p.grad).any()
                       for p in self.optimizer.param_groups[0]["params"]):
                    self.optimizer.zero_grad(); continue
                nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 0.3)
                self.optimizer.step()
                total_loss += loss.item(); n_up += 1

        self._reset_buffer()
        return {"loss": total_loss/max(n_up,1),
                "lambda_d": self._lambda_d, "lambda_f": self._lambda_f}

    def save(self, path):
        torch.save({"net": self.net.state_dict(),
                     "mu_head": self.mu_head.state_dict(),
                     "v_head": self.v_head.state_dict(),
                     "log_std": self.log_std.data,
                     "obs_mean": self.obs_mean, "obs_var": self.obs_var,
                     "obs_count": self.obs_count,
                     "lambda_d": self._lambda_d, "lambda_f": self._lambda_f}, path)
