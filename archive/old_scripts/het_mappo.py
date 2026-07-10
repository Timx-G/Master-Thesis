"""
Het-MAPPO: Heterogeneous Multi-Agent PPO for cooperative Multi-UAV MEC.

Key difference from MAPPO: each UAV has its OWN actor network (no parameter sharing).
Shared centralized critic is retained for CTDE.

This is the CTDE representative baseline for validating C3
(Independent PPO > explicit MARL coordination).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class HetMAPPOAgent:
    """Het-MAPPO: independent actors + shared critic."""

    def __init__(self, agent_obs_dim: int, agent_act_dim: int,
                 global_obs_dim: int, num_agents: int = 2,
                 lr_actor: float = 3e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, epochs: int = 4,
                 mini_batch: int = 64, hidden: int = 128,
                 seed: int = 0, device: str = "cpu"):
        self.num_agents = num_agents
        self.agent_obs_dim = agent_obs_dim
        self.agent_act_dim = agent_act_dim
        self.global_obs_dim = global_obs_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.mini_batch = mini_batch

        torch.manual_seed(seed)

        # Independent actors — one per UAV (NO parameter sharing)
        self.actors = []
        self.mu_heads = []
        self.log_stds = []
        self.actor_opts = []

        for k in range(num_agents):
            # Different seed per actor for weight diversity
            torch.manual_seed(seed + k * 1000)
            actor = nn.Sequential(
                nn.Linear(agent_obs_dim, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
            ).to(self.device)
            mu_head = nn.Linear(hidden, agent_act_dim).to(self.device)
            log_std = nn.Parameter(
                -0.5 * torch.ones(agent_act_dim, device=self.device))

            for m in [actor, mu_head]:
                for p in m.parameters():
                    if p.dim() > 1:
                        nn.init.orthogonal_(p, gain=0.5)

            opt = torch.optim.Adam(
                list(actor.parameters()) +
                list(mu_head.parameters()) + [log_std],
                lr=lr_actor, eps=1e-5)

            self.actors.append(actor)
            self.mu_heads.append(mu_head)
            self.log_stds.append(log_std)
            self.actor_opts.append(opt)

        # Shared centralized critic
        torch.manual_seed(seed)
        self.critic = nn.Sequential(
            nn.Linear(global_obs_dim, hidden * 2), nn.Tanh(),
            nn.Linear(hidden * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        ).to(self.device)
        for p in self.critic.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=0.5)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic, eps=1e-5)

        # Obs normalization (per-agent, shared stats for simplicity)
        self.obs_mean = np.zeros(agent_obs_dim, dtype=np.float64)
        self.obs_var = np.ones(agent_obs_dim, dtype=np.float64)
        self.obs_count = 1e-4

        self._reset_buffer()

    def _update_obs_stats(self, obs):
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        self.obs_var += (delta * (obs - self.obs_mean) - self.obs_var) / self.obs_count

    def _norm_obs(self, obs):
        std = np.sqrt(np.maximum(self.obs_var, 1e-6)) + 1e-8
        return np.clip((obs - self.obs_mean) / std, -10, 10).astype(np.float32)

    def select_actions(self, global_obs: np.ndarray,
                       deterministic: bool = False) -> np.ndarray:
        K = self.num_agents
        actions = []
        log_probs = []

        for k in range(K):
            obs_k = global_obs[k * self.agent_obs_dim:(k+1) * self.agent_obs_dim]
            self._update_obs_stats(obs_k)
            obs_n = self._norm_obs(obs_k)

            with torch.no_grad():
                obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(self.device)
                h = self.actors[k](obs_t)
                mu = self.mu_heads[k](h)
                if torch.isnan(mu).any():
                    actions.append(np.zeros(self.agent_act_dim, np.float32))
                    log_probs.append(0.0)
                    continue
                std = self.log_stds[k].exp().clamp(0.05, 1.0).expand_as(mu)
                dist = Normal(mu, std)
                act = mu if deterministic else dist.sample()
                act = act.clamp(-1, 1)
                lp = dist.log_prob(act).sum(-1)

            actions.append(act.cpu().numpy().flatten())
            log_probs.append(lp.item())

        # Centralized value
        with torch.no_grad():
            global_n = np.concatenate([
                self._norm_obs(
                    global_obs[k*self.agent_obs_dim:(k+1)*self.agent_obs_dim])
                for k in range(K)])
            global_t = torch.FloatTensor(global_n).unsqueeze(0).to(self.device)
            val = self.critic(global_t).item()

        self._cached_log_probs = log_probs
        self._cached_val = val
        return np.concatenate(actions)

    def store(self, global_obs, joint_action, reward, done):
        K = self.num_agents
        agent_obs = []
        for k in range(K):
            obs_k = global_obs[k*self.agent_obs_dim:(k+1)*self.agent_obs_dim]
            agent_obs.append(self._norm_obs(obs_k))

        self.buf_agent_obs.append(agent_obs)
        self.buf_actions.append(
            [joint_action[k*self.agent_act_dim:(k+1)*self.agent_act_dim].copy()
             for k in range(K)])
        self.buf_rewards.append(float(reward))
        self.buf_dones.append(float(done))
        self.buf_log_probs.append(list(self._cached_log_probs))
        self.buf_values.append(self._cached_val)
        self.buf_global_obs.append(np.concatenate(agent_obs).astype(np.float32))

    def _reset_buffer(self):
        self.buf_agent_obs = []
        self.buf_actions = []
        self.buf_rewards = []
        self.buf_dones = []
        self.buf_log_probs = []
        self.buf_values = []
        self.buf_global_obs = []
        self._cached_log_probs = [0.0] * self.num_agents
        self._cached_val = 0.0

    def update(self):
        T = len(self.buf_rewards)
        if T < 4:
            self._reset_buffer()
            return {}

        K = self.num_agents
        vals = torch.FloatTensor(self.buf_values).to(self.device)

        # GAE
        advs = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            nv = vals[t+1].item() if t+1 < T else 0.0
            mask = 1.0 - self.buf_dones[t]
            delta = self.buf_rewards[t] + self.gamma * nv * mask - vals[t].item()
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            gae = max(-500, min(500, gae))
            advs[t] = gae
        rets = (advs + vals).detach()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        global_obs_t = torch.FloatTensor(
            np.array(self.buf_global_obs)).to(self.device)

        total_loss = 0.0
        n_updates = 0

        for _ in range(self.epochs):
            idx = torch.randperm(T, device=self.device)
            for s in range(0, T, self.mini_batch):
                mb = idx[s:s+self.mini_batch]
                if len(mb) < 2:
                    continue

                # Critic update
                v_pred = self.critic(global_obs_t[mb]).squeeze(-1)
                v_loss = (v_pred - rets[mb]).pow(2).mean()
                self.critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()

                # Per-agent actor update (INDEPENDENT networks)
                for k in range(K):
                    obs_k = torch.FloatTensor(np.array(
                        [self.buf_agent_obs[i][k] for i in mb.cpu().numpy()]
                    )).to(self.device)
                    act_k = torch.FloatTensor(np.array(
                        [self.buf_actions[i][k] for i in mb.cpu().numpy()]
                    )).to(self.device)
                    old_lp = torch.FloatTensor(
                        [self.buf_log_probs[i][k] for i in mb.cpu().numpy()]
                    ).to(self.device)

                    h = self.actors[k](obs_k)
                    mu = self.mu_heads[k](h)
                    if torch.isnan(mu).any():
                        continue
                    std = self.log_stds[k].exp().clamp(0.05, 1.0).expand_as(mu)
                    dist = Normal(mu, std)
                    new_lp = dist.log_prob(act_k).sum(-1)
                    ratio = (new_lp - old_lp).exp().clamp(0, 10)

                    s1 = ratio * advs[mb]
                    s2 = ratio.clamp(1 - self.clip_eps,
                                     1 + self.clip_eps) * advs[mb]
                    pi_loss = -torch.min(s1, s2).mean()
                    ent = dist.entropy().sum(-1).mean()
                    loss = pi_loss - 0.02 * ent

                    if torch.isnan(loss):
                        continue

                    self.actor_opts[k].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.actors[k].parameters()) +
                        list(self.mu_heads[k].parameters()) +
                        [self.log_stds[k]], 0.5)
                    self.actor_opts[k].step()

                total_loss += v_loss.item()
                n_updates += 1

        self._reset_buffer()
        return {"loss": total_loss / max(n_updates, 1)}
