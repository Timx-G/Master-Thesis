"""
DQN baseline for UAV-MEC offloading.

Discretises the continuous action space into a manageable set of actions.
Used as a comparison to show SAC-Lagrangian's advantage in continuous control.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNAgent:
    """DQN with discretised action space.

    Discretisation:
      - UAV movement: 5 choices (stay, up, down, left, right)
      - Offloading: 3 levels per user (0, 0.5, 1.0) — but combinatorial explosion
        → simplified: all-local, all-offload, or proportional
      - Power: fixed at max for offloading users
    Total: 5 (movement) × 3 (offload strategy) = 15 discrete actions
    """

    def __init__(self, obs_dim: int, act_dim: int, num_users: int,
                 lr: float = 1e-3, gamma: float = 0.99, eps_start: float = 1.0,
                 eps_end: float = 0.05, eps_decay: int = 5000,
                 buffer_size: int = 100_000, batch_size: int = 128,
                 target_update: int = 100, hidden: int = 256,
                 seed: int = 0, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_users = num_users
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.rng = np.random.default_rng(seed)

        # Epsilon schedule
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0

        # Discrete actions
        self._build_action_table()
        n_actions = len(self.action_table)

        # Networks
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        ).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Buffer
        self.buffer_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.buffer_act = np.zeros(buffer_size, dtype=np.int64)
        self.buffer_rew = np.zeros(buffer_size, dtype=np.float32)
        self.buffer_next = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.buffer_done = np.zeros(buffer_size, dtype=np.float32)
        self.buf_ptr, self.buf_size, self.buf_max = 0, 0, buffer_size

    def _build_action_table(self):
        """Pre-compute discrete action vectors."""
        N = self.num_users
        moves = [(0, 0), (0.8, 0), (-0.8, 0), (0, 0.8), (0, -0.8)]
        offload_modes = [
            np.zeros(N),                   # all local
            np.ones(N),                     # all offload
            np.full(N, 0.5),               # half offload
        ]
        self.action_table = []
        for dx, dy in moves:
            for off in offload_modes:
                power = off.copy()  # power matches offload
                act = np.concatenate([[dx, dy], off * 2 - 1, power * 2 - 1]).astype(np.float32)
                self.action_table.append(act)

    def _epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-self.steps / self.eps_decay)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic and self.rng.random() < self._epsilon():
            idx = self.rng.integers(len(self.action_table))
        else:
            with torch.no_grad():
                q = self.q_net(torch.FloatTensor(obs).unsqueeze(0).to(self.device))
                idx = q.argmax(dim=-1).item()
        self.steps += 1
        return self.action_table[idx]

    def store(self, obs, act_vec, rew, next_obs, done):
        # Find closest discrete action
        diffs = [np.linalg.norm(act_vec - a) for a in self.action_table]
        act_idx = int(np.argmin(diffs))
        self.buffer_obs[self.buf_ptr] = obs
        self.buffer_act[self.buf_ptr] = act_idx
        self.buffer_rew[self.buf_ptr] = rew
        self.buffer_next[self.buf_ptr] = next_obs
        self.buffer_done[self.buf_ptr] = float(done)
        self.buf_ptr = (self.buf_ptr + 1) % self.buf_max
        self.buf_size = min(self.buf_size + 1, self.buf_max)

    def update(self) -> dict:
        if self.buf_size < self.batch_size:
            return {}

        idx = self.rng.integers(0, self.buf_size, self.batch_size)
        obs = torch.FloatTensor(self.buffer_obs[idx]).to(self.device)
        act = torch.LongTensor(self.buffer_act[idx]).to(self.device)
        rew = torch.FloatTensor(self.buffer_rew[idx]).to(self.device)
        nxt = torch.FloatTensor(self.buffer_next[idx]).to(self.device)
        done = torch.FloatTensor(self.buffer_done[idx]).to(self.device)

        q_val = self.q_net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(nxt).max(dim=1)[0]
            target = rew + (1 - done) * self.gamma * q_next

        loss = F.mse_loss(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"dqn_loss": loss.item(), "epsilon": self._epsilon()}

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.q_net.state_dict())


if __name__ == "__main__":
    print("=" * 60)
    print("dqn.py self-test")
    print("=" * 60)
    agent = DQNAgent(53, 22, 10)
    obs = np.random.randn(53).astype(np.float32)
    act = agent.select_action(obs)
    assert act.shape == (22,)
    print(f"  Discrete actions: {len(agent.action_table)}")
    print(f"  Action shape: {act.shape}")

    for i in range(300):
        o = np.random.randn(53).astype(np.float32)
        a = agent.select_action(o)
        r = np.random.randn()
        o2 = np.random.randn(53).astype(np.float32)
        agent.store(o, a, r, o2, i == 299)
        agent.update()

    print("\ndqn.py: ALL TESTS PASSED")
