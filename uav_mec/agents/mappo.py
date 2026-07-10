"""
MAPPO: Multi-Agent PPO for cooperative Multi-UAV MEC.

Architecture: CTDE (Centralized Training, Decentralized Execution)
  - Shared Critic: global state → V(s) (centralized)
  - Independent Actors: local obs → action (decentralized, parameter sharing)
  - Per-agent: 9-dim obs, 5-dim action

For UAV-MEC with shared frequency band, MAPPO learns:
  Skill 1: Spatial separation (avoid inter-UAV interference)
  Skill 2: Dynamic burst tracking (anticipatory trajectory planning)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class MAPPOConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    mini_batch: int = 64
    hidden: int = 64           # smaller network for 5-dim action
    entropy_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class MAPPOAgent:
    """Multi-Agent PPO with shared critic and parameter-shared actors."""

    def __init__(self, agent_obs_dim: int, agent_act_dim: int,
                 global_obs_dim: int, num_agents: int = 2,
                 cfg: Optional[MAPPOConfig] = None,
                 seed: int = 0, device: str = "cpu"):
        self.cfg = cfg or MAPPOConfig()
        self.num_agents = num_agents
        self.agent_obs_dim = agent_obs_dim
        self.agent_act_dim = agent_act_dim
        self.global_obs_dim = global_obs_dim
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)
        H = self.cfg.hidden

        # Shared Actor (parameter sharing: same network for all agents)
        self.actor = nn.Sequential(
            nn.Linear(agent_obs_dim, H), nn.Tanh(),
            nn.Linear(H, H), nn.Tanh(),
        ).to(self.device)
        self.mu_head = nn.Linear(H, agent_act_dim).to(self.device)
        self.log_std = nn.Parameter(
            -0.5 * torch.ones(agent_act_dim, device=self.device))

        # Centralized Critic (takes global observation)
        self.critic = nn.Sequential(
            nn.Linear(global_obs_dim, H * 2), nn.Tanh(),
            nn.Linear(H * 2, H), nn.Tanh(),
            nn.Linear(H, 1),
        ).to(self.device)

        # Init
        for m in [self.actor, self.mu_head, self.critic]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.orthogonal_(p, gain=0.5)

        self.actor_opt = torch.optim.Adam(
            list(self.actor.parameters()) +
            list(self.mu_head.parameters()) + [self.log_std],
            lr=self.cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.lr_critic)

        # Obs normalization (per-agent)
        self.obs_mean = np.zeros(agent_obs_dim, dtype=np.float64)
        self.obs_var = np.ones(agent_obs_dim, dtype=np.float64)
        self.obs_count = 1e-4

        # Rollout buffers (per episode, all agents concatenated)
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
        """Select actions for ALL agents given global observation.

        Args:
            global_obs: concatenated obs for all agents (K * agent_obs_dim)
        Returns:
            joint_action: concatenated actions (K * agent_act_dim)
        """
        K = self.num_agents
        actions = []
        log_probs = []

        for k in range(K):
            obs_k = global_obs[k * self.agent_obs_dim:(k+1) * self.agent_obs_dim]
            self._update_obs_stats(obs_k)
            obs_n = self._norm_obs(obs_k)

            with torch.no_grad():
                obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(self.device)
                h = self.actor(obs_t)
                mu = self.mu_head(h)
                if torch.isnan(mu).any():
                    actions.append(np.zeros(self.agent_act_dim, np.float32))
                    log_probs.append(0.0)
                    continue
                std = self.log_std.exp().clamp(0.05, 1.0).expand_as(mu)
                dist = Normal(mu, std)
                act = mu if deterministic else dist.sample()
                act = act.clamp(-1, 1)
                lp = dist.log_prob(act).sum(-1)

            actions.append(act.cpu().numpy().flatten())
            log_probs.append(lp.item())

        # Centralized value
        with torch.no_grad():
            global_t = torch.FloatTensor(
                np.concatenate([self._norm_obs(
                    global_obs[k*self.agent_obs_dim:(k+1)*self.agent_obs_dim])
                    for k in range(K)])).unsqueeze(0).to(self.device)
            val = self.critic(global_t).item()

        self._cached_log_probs = log_probs
        self._cached_val = val
        return np.concatenate(actions)

    def store(self, global_obs, joint_action, reward, done):
        """Store transition for all agents."""
        K = self.num_agents
        # Per-agent normalized obs
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

        # Global obs (normalized, concatenated)
        self.buf_global_obs.append(
            np.concatenate(agent_obs).astype(np.float32))

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

        # GAE (shared reward, centralized value)
        advs = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            nv = vals[t+1].item() if t+1 < T else 0.0
            mask = 1.0 - self.buf_dones[t]
            delta = self.buf_rewards[t] + self.cfg.gamma * nv * mask - vals[t].item()
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            gae = max(-500, min(500, gae))
            advs[t] = gae
        rets = (advs + vals).detach()
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Flatten per-agent data
        global_obs_t = torch.FloatTensor(
            np.array(self.buf_global_obs)).to(self.device)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        n_updates = 0

        for _ in range(self.cfg.epochs):
            idx = torch.randperm(T, device=self.device)
            for s in range(0, T, self.cfg.mini_batch):
                mb = idx[s:s+self.cfg.mini_batch]
                if len(mb) < 2:
                    continue

                # Critic update
                v_pred = self.critic(global_obs_t[mb]).squeeze(-1)
                v_loss = (v_pred - rets[mb]).pow(2).mean()

                self.critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),
                                         self.cfg.max_grad_norm)
                self.critic_opt.step()

                # Actor update (per agent, parameter shared)
                pi_loss_total = torch.tensor(0.0, device=self.device)
                ent_total = torch.tensor(0.0, device=self.device)

                for k in range(K):
                    obs_k = torch.FloatTensor(np.array(
                        [self.buf_agent_obs[i][k] for i in mb.cpu().numpy()]
                    )).to(self.device)
                    act_k = torch.FloatTensor(np.array(
                        [self.buf_actions[i][k] for i in mb.cpu().numpy()]
                    )).to(self.device)
                    old_lp_k = torch.FloatTensor(
                        [self.buf_log_probs[i][k] for i in mb.cpu().numpy()]
                    ).to(self.device)

                    h = self.actor(obs_k)
                    mu = self.mu_head(h)
                    if torch.isnan(mu).any():
                        continue
                    std = self.log_std.exp().clamp(0.05, 1.0).expand_as(mu)
                    dist = Normal(mu, std)
                    new_lp = dist.log_prob(act_k).sum(-1)
                    ratio = (new_lp - old_lp_k).exp().clamp(0, 10)

                    s1 = ratio * advs[mb]
                    s2 = ratio.clamp(1-self.cfg.clip_eps,
                                     1+self.cfg.clip_eps) * advs[mb]
                    pi_loss_total += -torch.min(s1, s2).mean()
                    ent_total += dist.entropy().sum(-1).mean()

                loss = (pi_loss_total / K
                        - self.cfg.entropy_coef * ent_total / K)

                if torch.isnan(loss):
                    continue

                self.actor_opt.zero_grad()
                loss.backward()
                has_nan = any(
                    p.grad is not None and torch.isnan(p.grad).any()
                    for p in self.actor_opt.param_groups[0]["params"])
                if has_nan:
                    self.actor_opt.zero_grad()
                    continue
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) +
                    list(self.mu_head.parameters()) + [self.log_std],
                    self.cfg.max_grad_norm)
                self.actor_opt.step()

                total_actor_loss += loss.item()
                total_critic_loss += v_loss.item()
                n_updates += 1

        self._reset_buffer()
        return {"actor_loss": total_actor_loss / max(n_updates, 1),
                "critic_loss": total_critic_loss / max(n_updates, 1)}
