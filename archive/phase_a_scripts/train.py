"""
Unified training script for all agents.

Usage:
    python3 -m uav_mec.train --algo sac_lagrangian --episodes 500
    python3 -m uav_mec.train --algo dqn --episodes 300
    python3 -m uav_mec.train --algo ppo --episodes 300
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .env import UAVMECEnv
from .model import ScenarioConfig
from .agents.sac_lagrangian import SACLagrangianAgent, SACLagrangianConfig
from .agents.dqn import DQNAgent
from .agents.ppo import PPOAgent
from .agents.td3 import TD3Agent
from .agents.baselines import make_baseline


def evaluate(env: UAVMECEnv, agent, num_episodes: int = 5, seed_base: int = 9999) -> dict:
    """Evaluate agent over multiple episodes, return mean metrics."""
    rewards, delays, energies, violations, tasks_all = [], [], [], [], []
    all_slot_delays = []   # per-slot avg delay for p95 computation
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        ep_reward = 0.0
        done = False
        while not done:
            if hasattr(agent, "select_action"):
                act = agent.select_action(obs, deterministic=True)
            else:
                act = agent.select_action(obs)
            obs, rew, terminated, truncated, info = env.step(act)
            ep_reward += rew
            done = terminated or truncated
            # collect slot-level avg delay for p95
            n_slot = info.get("slot_tasks", 0)
            if n_slot > 0:
                all_slot_delays.append(info["slot_delay"] / n_slot)

        rewards.append(ep_reward)
        delays.append(info["episode_total_delay"])
        energies.append(info["episode_total_energy"])
        violations.append(info["episode_deadline_violations"])
        tasks_all.append(info["episode_tasks_served"])

    avg_tasks = np.mean(tasks_all)
    safe_tasks = np.maximum(avg_tasks, 1)
    deadline_rate = 1.0 - np.mean(violations) / safe_tasks
    p95_delay = float(np.percentile(all_slot_delays, 95)) if all_slot_delays else 0.0
    return {
        "reward": np.mean(rewards),
        "reward_std": np.std(rewards),
        # episode-level totals
        "avg_delay": np.mean(delays),
        "avg_energy": np.mean(energies),
        "avg_violations": np.mean(violations),
        "avg_tasks": avg_tasks,
        # per-task normalized (for paper Table)
        "delay_per_task": np.mean(delays) / safe_tasks,      # s/task
        "energy_per_task": np.mean(energies) / safe_tasks,   # J/task
        "violation_rate": np.mean(violations) / safe_tasks,  # = 1 - deadline_rate
        "p95_delay": p95_delay,                              # p95 slot-avg delay (s)
        # primary metric
        "deadline_rate": deadline_rate,
    }


def train_sac_lagrangian(env: UAVMECEnv, episodes: int, eval_interval: int = 20,
                         save_dir: str = "checkpoints") -> list[dict]:
    """Train SAC-Lagrangian agent."""
    cfg = SACLagrangianConfig()
    agent = SACLagrangianAgent(env.obs_dim, env.action_dim, cfg)
    history = []

    os.makedirs(save_dir, exist_ok=True)
    best_reward = -float("inf")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        done = False

        while not done:
            if agent.total_steps < cfg.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Cost signals for Lagrangian (per-user per-slot)
            # NOTE: energy cost = USER energy only (local + transmit), NOT fly energy.
            # Fly energy is already in the reward; including it here would double-penalize.
            n_tasks = max(info.get("slot_tasks", 1), 1)
            cost_energy = info.get("slot_energy", 0) / n_tasks
            cost_delay = info.get("slot_delay", 0) / n_tasks

            agent.store(obs, action, reward, next_obs, done,
                        cost_energy=cost_energy, cost_delay=cost_delay)
            agent.update()

            obs = next_obs
            ep_reward += reward

        # Evaluation
        if ep % eval_interval == 0 or ep == 1:
            metrics = evaluate(env, agent)
            metrics["episode"] = ep
            metrics["lambda_e"] = agent.lambda_e
            metrics["lambda_d"] = agent.lambda_d
            metrics["alpha"] = agent.alpha
            history.append(metrics)

            print(f"  [EP {ep:4d}] R={metrics['reward']:.1f}  "
                  f"delay={metrics['avg_delay']:.1f}  energy={metrics['avg_energy']:.0f}  "
                  f"viol={metrics['avg_violations']:.1f}  "
                  f"λe={agent.lambda_e:.3f} λd={agent.lambda_d:.3f} α={agent.alpha:.3f}")

            if metrics["reward"] > best_reward:
                best_reward = metrics["reward"]
                agent.save(os.path.join(save_dir, "sac_lagrangian_best.pt"))

    agent.save(os.path.join(save_dir, "sac_lagrangian_final.pt"))
    return history


def train_dqn(env: UAVMECEnv, episodes: int, eval_interval: int = 20,
              save_dir: str = "checkpoints") -> list[dict]:
    """Train DQN agent."""
    agent = DQNAgent(env.obs_dim, env.action_dim, env.cfg.num_users)
    history = []
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
            ep_reward += reward

        if ep % eval_interval == 0 or ep == 1:
            metrics = evaluate(env, agent)
            metrics["episode"] = ep
            history.append(metrics)
            print(f"  [DQN EP {ep:4d}] R={metrics['reward']:.1f}  "
                  f"delay={metrics['avg_delay']:.1f}  viol={metrics['avg_violations']:.1f}")

    agent.save(os.path.join(save_dir, "dqn_final.pt"))
    return history


def train_ppo(env: UAVMECEnv, episodes: int, eval_interval: int = 20,
              save_dir: str = "checkpoints") -> list[dict]:
    """Train PPO agent."""
    agent = PPOAgent(env.obs_dim, env.action_dim)
    history = []
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, done)
            obs = next_obs
            ep_reward += reward

        agent.update()

        if ep % eval_interval == 0 or ep == 1:
            metrics = evaluate(env, agent)
            metrics["episode"] = ep
            history.append(metrics)
            print(f"  [PPO EP {ep:4d}] R={metrics['reward']:.1f}  "
                  f"delay={metrics['avg_delay']:.1f}  viol={metrics['avg_violations']:.1f}")

    agent.save(os.path.join(save_dir, "ppo_final.pt"))
    return history


def train_td3(env: UAVMECEnv, episodes: int, eval_interval: int = 20,
              save_dir: str = "checkpoints") -> list[dict]:
    """Train TD3 agent."""
    agent = TD3Agent(env.obs_dim, env.action_dim)
    history = []
    os.makedirs(save_dir, exist_ok=True)
    best_reward = -float("inf")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        done = False

        while not done:
            if agent.total_steps < agent.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
            ep_reward += reward

        if ep % eval_interval == 0 or ep == 1:
            metrics = evaluate(env, agent)
            metrics["episode"] = ep
            history.append(metrics)
            print(f"  [TD3 EP {ep:4d}] R={metrics['reward']:.1f}  "
                  f"delay={metrics['avg_delay']:.1f}  viol={metrics['avg_violations']:.1f}")

            if metrics["reward"] > best_reward:
                best_reward = metrics["reward"]
                agent.save(os.path.join(save_dir, "td3_best.pt"))

    agent.save(os.path.join(save_dir, "td3_final.pt"))
    return history


def eval_baseline(env: UAVMECEnv, name: str, num_episodes: int = 20) -> dict:
    """Evaluate a heuristic baseline."""
    agent = make_baseline(name, env.obs_dim, env.action_dim, env.cfg.num_users)
    metrics = evaluate(env, agent, num_episodes=num_episodes)
    print(f"  [{name:15s}] R={metrics['reward']:.1f}  "
          f"delay={metrics['avg_delay']:.1f}  energy={metrics['avg_energy']:.0f}  "
          f"viol={metrics['avg_violations']:.1f}  "
          f"deadline_rate={metrics['deadline_rate']*100:.1f}%")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UAV-MEC Training")
    parser.add_argument("--algo", type=str, default="sac_lagrangian",
                        choices=["sac_lagrangian", "dqn", "ppo", "td3", "baselines", "all"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--num-users", type=int, default=10)
    parser.add_argument("--num-slots", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    cfg = ScenarioConfig(num_users=args.num_users, num_slots=args.num_slots, seed=42)
    env = UAVMECEnv(cfg=cfg)

    print("=" * 60)
    print(f"  UAV-MEC Training: {args.algo}")
    print(f"  Users={args.num_users}, Slots={args.num_slots}, Episodes={args.episodes}")
    print("=" * 60)

    t0 = time.time()

    if args.algo == "sac_lagrangian":
        train_sac_lagrangian(env, args.episodes, args.eval_interval, args.save_dir)
    elif args.algo == "dqn":
        train_dqn(env, args.episodes, args.eval_interval, args.save_dir)
    elif args.algo == "ppo":
        train_ppo(env, args.episodes, args.eval_interval, args.save_dir)
    elif args.algo == "td3":
        train_td3(env, args.episodes, args.eval_interval, args.save_dir)
    elif args.algo == "baselines":
        for name in ["local", "full_offload", "random", "greedy", "round_robin"]:
            eval_baseline(env, name)
    elif args.algo == "all":
        print("\n--- Heuristic Baselines ---")
        for name in ["local", "full_offload", "random", "greedy", "round_robin"]:
            eval_baseline(env, name)
        print("\n--- DQN ---")
        train_dqn(env, args.episodes, args.eval_interval, args.save_dir)
        print("\n--- PPO ---")
        train_ppo(env, args.episodes, args.eval_interval, args.save_dir)
        print("\n--- TD3 ---")
        train_td3(env, args.episodes, args.eval_interval, args.save_dir)
        print("\n--- SAC-Lagrangian (Ours) ---")
        train_sac_lagrangian(env, args.episodes, args.eval_interval, args.save_dir)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
