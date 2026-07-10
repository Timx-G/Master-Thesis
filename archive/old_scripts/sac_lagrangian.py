"""
SAC with Lagrangian Dual Constraint Handling.

Core algorithm for the paper:
  "Joint UAV Trajectory Design and NOMA-Based Task Offloading in MEC:
   A Constrained Deep Reinforcement Learning Approach"

Key design:
  - Soft Actor-Critic (Haarnoja et al. 2018) for continuous action space
  - Lagrangian dual variables (lambda_e, lambda_d) for energy and delay constraints
    (Tessler et al. 2019 ICLR: "Reward Constrained Policy Optimization")
  - Automatic entropy tuning (alpha)
  - Twin Q-networks to reduce overestimation
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class SACLagrangianConfig:
    """SAC + Lagrangian hyperparameters."""
    # Network
    hidden_dim: int = 256
    num_layers: int = 2
    # Training
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4          # entropy temperature
    lr_lambda: float = 5e-3         # Lagrangian multiplier learning rate
    gamma: float = 0.99             # discount
    tau: float = 0.005              # target network soft update
    batch_size: int = 256
    buffer_size: int = 200_000
    warmup_steps: int = 2000        # random actions before training
    # Constraints (per-user per-slot budgets)
    # Only delay constraint is meaningful; user energy is negligible (~0.4J).
    # UAV flight energy is handled in the reward, not as a Lagrangian constraint.
    delay_budget: float = 0.5              # per-user average delay budget (s)
    lambda_init: float = 0.1               # initial Lagrangian multiplier
    lambda_max: float = 20.0               # clamp max (projection gradient bound)
    lambda_update_interval: int = 5        # update lambda every N gradient steps
    # Entropy
    alpha_max: float = 0.5                 # clamp entropy temperature
    target_entropy: Optional[float] = None  # auto: -dim(A)


# ---------------------------------------------------------------------------
# Neural networks
# ---------------------------------------------------------------------------

def build_mlp(in_dim: int, out_dim: int, hidden: int, num_layers: int,
              output_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers = []
    prev = in_dim
    for _ in range(num_layers):
        layers.extend([nn.Linear(prev, hidden), nn.ReLU()])
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        layers.append(output_activation)
    net = nn.Sequential(*layers)
    # Orthogonal init for stability
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
    return net


class GaussianActor(nn.Module):
    """Squashed Gaussian policy: outputs tanh-bounded actions in [-1, 1]."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, num_layers: int):
        super().__init__()
        self.trunk = build_mlp(obs_dim, hidden, hidden, num_layers - 1)
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        """Sample action + compute log_prob (with tanh squashing correction)."""
        mu, log_std = self.forward(obs)
        # NaN guard: if network produces NaN, return zeros
        if torch.any(torch.isnan(mu)):
            act_dim = mu.shape[-1]
            action = torch.zeros_like(mu)
            log_prob = torch.zeros(mu.shape[0], 1, device=mu.device)
            return action, log_prob
        std = log_std.exp()
        dist = Normal(mu, std)
        # Reparameterisation trick
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        # Log prob with tanh correction
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor):
        mu, _ = self.forward(obs)
        if torch.any(torch.isnan(mu)):
            return torch.zeros_like(mu)
        return torch.tanh(mu)


class TwinQCritic(nn.Module):
    """Twin Q-networks for reducing overestimation."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, num_layers: int):
        super().__init__()
        self.q1 = build_mlp(obs_dim + act_dim, 1, hidden, num_layers)
        self.q2 = build_mlp(obs_dim + act_dim, 1, hidden, num_layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        sa = torch.cat([obs, act], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, obs: torch.Tensor, act: torch.Tensor):
        sa = torch.cat([obs, act], dim=-1)
        return self.q1(sa)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
        # Cost signals for Lagrangian constraints
        self.cost_energy = np.zeros(max_size, dtype=np.float32)
        self.cost_delay = np.zeros(max_size, dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done, cost_energy=0.0, cost_delay=0.0):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.cost_energy[self.ptr] = cost_energy
        self.cost_delay[self.ptr] = cost_delay
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, batch_size)
        return {
            "obs": torch.FloatTensor(self.obs[idx]).to(device),
            "act": torch.FloatTensor(self.act[idx]).to(device),
            "rew": torch.FloatTensor(self.rew[idx]).to(device).unsqueeze(-1),
            "next_obs": torch.FloatTensor(self.next_obs[idx]).to(device),
            "done": torch.FloatTensor(self.done[idx]).to(device).unsqueeze(-1),
            "cost_energy": torch.FloatTensor(self.cost_energy[idx]).to(device).unsqueeze(-1),
            "cost_delay": torch.FloatTensor(self.cost_delay[idx]).to(device).unsqueeze(-1),
        }


# ---------------------------------------------------------------------------
# SAC-Lagrangian Agent
# ---------------------------------------------------------------------------

class SACLagrangianAgent:
    """SAC with Lagrangian dual constraint handling.

    The Lagrangian augmented reward is:
        r_aug = r - lambda_e * c_e - lambda_d * c_d
    where c_e = per-step energy cost, c_d = per-step delay cost.
    lambda_e, lambda_d are dual variables updated via gradient ascent.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 cfg: Optional[SACLagrangianConfig] = None,
                 device: str = "cpu"):
        self.cfg = cfg or SACLagrangianConfig()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # -- Networks --
        self.actor = GaussianActor(
            obs_dim, act_dim, self.cfg.hidden_dim, self.cfg.num_layers
        ).to(self.device)

        self.critic = TwinQCritic(
            obs_dim, act_dim, self.cfg.hidden_dim, self.cfg.num_layers
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # -- Optimisers --
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        # -- Entropy temperature (log-space for stability) --
        self.log_alpha = torch.tensor(
            np.log(0.2), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
        self.target_entropy = (
            self.cfg.target_entropy if self.cfg.target_entropy is not None
            else -float(act_dim)  # standard SAC: -dim(A)
        )

        # -- Lagrangian dual variable (delay constraint only) --
        # Use simple projected gradient ascent instead of Adam for stability
        self._lambda_d = self.cfg.lambda_init

        # -- Observation normalization (Welford online) --
        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_var = np.ones(obs_dim, dtype=np.float64)
        self.obs_count = 1e-4

        # -- Replay buffer --
        self.buffer = ReplayBuffer(obs_dim, act_dim, self.cfg.buffer_size)
        self.total_steps = 0
        self._update_count = 0  # for lambda update interval

        # -- Reward normalization (running stats) --
        self._rew_mean = 0.0
        self._rew_var = 1.0
        self._rew_count = 1e-4

        # -- Training logs --
        self.train_log = {
            "critic_loss": [], "actor_loss": [], "alpha": [],
            "lambda_e": [], "lambda_d": [],
        }

    # -- Properties --
    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    @property
    def lambda_e(self) -> float:
        return 0.0  # energy constraint removed (user energy negligible)

    @property
    def lambda_d(self) -> float:
        return self._lambda_d

    # -- Observation normalization --
    def _update_obs_stats(self, obs: np.ndarray):
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += (delta * delta2 - self.obs_var) / self.obs_count

    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        return ((obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)).astype(np.float32)

    # -- Action selection --
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self._update_obs_stats(obs)
        obs_n = self._norm_obs(obs)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(self.device)
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
            result = action.cpu().numpy().flatten()
            # NaN guard: fall back to zero action
            if np.any(np.isnan(result)):
                result = np.zeros(self.act_dim, dtype=np.float32)
            return result

    # -- Store transition --
    def store(self, obs, act, rew, next_obs, done, cost_energy=0.0, cost_delay=0.0):
        # Store normalised observations
        obs_n = self._norm_obs(obs)
        next_obs_n = self._norm_obs(next_obs)
        self.buffer.add(obs_n, act, rew, next_obs_n, done, cost_energy, cost_delay)
        self.total_steps += 1

    # -- Update --
    def update(self) -> dict:
        """One gradient step on all components."""
        if self.buffer.size < self.cfg.warmup_steps:
            return {}

        batch = self.buffer.sample(self.cfg.batch_size, self.device)
        obs, act, rew, next_obs, done = (
            batch["obs"], batch["act"], batch["rew"], batch["next_obs"], batch["done"]
        )
        cost_e, cost_d = batch["cost_energy"], batch["cost_delay"]

        # Lagrangian augmented reward (delay constraint only)
        lam_d = self._lambda_d
        rew_aug = rew - lam_d * cost_d

        alpha = self.log_alpha.exp().detach()

        # ---- Critic update ----
        with torch.no_grad():
            next_act, next_log_prob = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_act)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_prob
            q_target = rew_aug + (1.0 - done) * self.cfg.gamma * q_next
            # Clamp to prevent value explosion
            q_target = q_target.clamp(-2000.0, 50000.0)

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # Skip update if loss is NaN
        if not torch.isnan(critic_loss):
            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_opt.step()
        else:
            return {"critic_loss": float("nan"), "actor_loss": 0.0,
                    "alpha": self.alpha, "lambda_e": self.lambda_e, "lambda_d": self.lambda_d}

        # ---- Actor update ----
        new_act, log_prob = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_act)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_new).mean()

        if not torch.isnan(actor_loss):
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

        # ---- Alpha (entropy temperature) update ----
        alpha_loss = -(self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        # Clamp alpha to [alpha_min, alpha_max]
        with torch.no_grad():
            self.log_alpha.data.clamp_(min=np.log(0.01), max=np.log(self.cfg.alpha_max))

        # ---- Lagrangian multiplier update (slower timescale) ----
        # ---- Lagrangian multiplier update (projected gradient ascent, delay only) ----
        self._update_count += 1
        if self._update_count % self.cfg.lambda_update_interval == 0:
            # Dual ascent: lambda_d += lr * (cost_d - budget), then project to [0, lambda_max]
            cost_d_mean = cost_d.mean().item()
            self._lambda_d += self.cfg.lr_lambda * (cost_d_mean - self.cfg.delay_budget)
            self._lambda_d = max(0.0, min(self._lambda_d, self.cfg.lambda_max))

        # ---- Target network soft update ----
        for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_targ.data.mul_(1.0 - self.cfg.tau)
            p_targ.data.add_(self.cfg.tau * p.data)

        # Log
        info = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "lambda_e": self.lambda_e,
            "lambda_d": self.lambda_d,
        }
        for k, v in info.items():
            self.train_log[k].append(v)
        return info

    # -- Save / Load --
    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.data,
            "lambda_d": self._lambda_d,
            "total_steps": self.total_steps,
            "obs_mean": self.obs_mean,
            "obs_var": self.obs_var,
            "obs_count": self.obs_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self._lambda_d = ckpt.get("lambda_d", self.cfg.lambda_init)
        self.total_steps = ckpt["total_steps"]
        self.obs_mean = ckpt["obs_mean"]
        self.obs_var = ckpt["obs_var"]
        self.obs_count = ckpt["obs_count"]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("sac_lagrangian.py self-test")
    print("=" * 60)

    obs_dim, act_dim = 53, 22
    agent = SACLagrangianAgent(obs_dim, act_dim)

    # Test action selection
    obs = np.random.randn(obs_dim).astype(np.float32)
    act = agent.select_action(obs)
    assert act.shape == (act_dim,), f"Bad action shape: {act.shape}"
    assert np.all(np.abs(act) <= 1.0), "Action out of [-1,1]"

    act_det = agent.select_action(obs, deterministic=True)
    assert act_det.shape == (act_dim,)

    # Fill buffer and test update
    for i in range(2500):
        o = np.random.randn(obs_dim).astype(np.float32)
        a = np.random.uniform(-1, 1, act_dim).astype(np.float32)
        r = np.random.randn()
        o2 = np.random.randn(obs_dim).astype(np.float32)
        d = i == 2499
        agent.store(o, a, r, o2, d, cost_energy=abs(r) * 0.5, cost_delay=abs(r) * 0.1)

    # Run a few updates
    for _ in range(5):
        info = agent.update()

    assert "critic_loss" in info, "Update returned no info"
    print(f"  critic_loss={info['critic_loss']:.4f}")
    print(f"  actor_loss={info['actor_loss']:.4f}")
    print(f"  alpha={info['alpha']:.4f}")
    print(f"  lambda_e={info['lambda_e']:.4f}")
    print(f"  lambda_d={info['lambda_d']:.4f}")

    # Test save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        agent.save(f.name)
        agent2 = SACLagrangianAgent(obs_dim, act_dim)
        agent2.load(f.name)
        os.unlink(f.name)

    a1 = agent.select_action(obs, deterministic=True)
    a2 = agent2.select_action(obs, deterministic=True)
    assert np.allclose(a1, a2, atol=1e-5), "Save/load mismatch"

    print("\nsac_lagrangian.py: ALL TESTS PASSED")
