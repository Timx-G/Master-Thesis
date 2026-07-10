"""
Complete experiment runner for the paper (v2 — post-9-expert review).

Experiments:
  Exp1: Baseline comparison (9 algorithms, 5 seeds, 1000ep)
  Exp2: Ablation study (4 SAC variants, 5 seeds, 1000ep)
  Exp3: Scalability (varying N, 5 seeds, 1000ep)
  Exp4: Convergence curves (4 DRL algos, 5 seeds, 1000ep)
  Exp5: Lagrangian multiplier dynamics (3 seeds, 1000ep)
  Exp6: Constraint sensitivity (3 budget levels, 5 seeds, 1000ep)  [NEW]
  Exp7: NOMA vs OMA comparison (5 seeds, 1000ep)  [NEW]
  Exp8: UAV trajectory visualization  [NEW]

Unified config: all DRL training uses 1000 episodes, 5 seeds, 50 eval episodes.
"""

from __future__ import annotations

import os
import sys
import time
import json
import copy

import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)

from uav_mec.env import UAVMECEnv
from uav_mec.model import ScenarioConfig, ChannelConfig
from uav_mec.agents.sac_lagrangian import SACLagrangianAgent, SACLagrangianConfig
from uav_mec.agents.dqn import DQNAgent
from uav_mec.agents.ppo import PPOAgent
from uav_mec.agents.td3 import TD3Agent
from uav_mec.agents.baselines import make_baseline
from uav_mec.train import evaluate, train_sac_lagrangian, train_dqn, train_ppo, train_td3

RESULTS_DIR = os.path.join(_ROOT, "results")
FIGURES_DIR = os.path.join(_ROOT, "figures")
CKPT_DIR = os.path.join(_ROOT, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ====== Unified experiment constants ======
TRAIN_EP = 1000
EVAL_EP = 50
SEEDS = [42, 123, 456, 789, 1024]


# ====================================================================
# Exp1: Baseline Comparison
# ====================================================================

def exp1_baseline_comparison():
    print("=" * 70)
    print("  EXP1: Baseline Comparison (9 algorithms, 5 seeds)")
    print("=" * 70)

    rows = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        cfg = ScenarioConfig(num_users=10, num_slots=100, seed=seed)
        env = UAVMECEnv(cfg=cfg)

        for name in ["local", "full_offload", "random", "greedy", "round_robin"]:
            agent = make_baseline(name, env.obs_dim, env.action_dim, 10, seed=seed)
            m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
            m.update({"algorithm": name, "seed": seed, "num_users": 10})
            rows.append(m)
            print(f"  {name:15s}: R={m['reward']:.1f}  delay={m['avg_delay']:.1f}  "
                  f"energy={m['avg_energy']:.0f}  viol={m['avg_violations']:.1f}")

        for algo_name, train_fn, AgentCls, load_path in [
            ("DQN", train_dqn, DQNAgent, "dqn_final.pt"),
            ("PPO", train_ppo, PPOAgent, "ppo_final.pt"),
            ("TD3", train_td3, TD3Agent, "td3_final.pt"),
        ]:
            env_t = UAVMECEnv(cfg=cfg)
            train_fn(env_t, TRAIN_EP, eval_interval=TRAIN_EP, save_dir=CKPT_DIR)
            if algo_name == "DQN":
                agent = AgentCls(env.obs_dim, env.action_dim, 10, seed=seed)
            else:
                agent = AgentCls(env.obs_dim, env.action_dim, seed=seed)
            agent.load(os.path.join(CKPT_DIR, load_path))
            m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
            m.update({"algorithm": algo_name, "seed": seed, "num_users": 10})
            rows.append(m)
            print(f"  {algo_name:15s}: R={m['reward']:.1f}  delay={m['avg_delay']:.1f}  "
                  f"energy={m['avg_energy']:.0f}  viol={m['avg_violations']:.1f}")

        # SAC-Lagrangian
        env_sac = UAVMECEnv(cfg=cfg)
        train_sac_lagrangian(env_sac, TRAIN_EP, eval_interval=TRAIN_EP, save_dir=CKPT_DIR)
        sac_agent = SACLagrangianAgent(env.obs_dim, env.action_dim)
        sac_agent.load(os.path.join(CKPT_DIR, "sac_lagrangian_best.pt"))
        m = evaluate(env, sac_agent, num_episodes=EVAL_EP, seed_base=seed * 100)
        m.update({"algorithm": "SAC-Lagrangian", "seed": seed, "num_users": 10})
        rows.append(m)
        print(f"  {'SAC-Lagrangian':15s}: R={m['reward']:.1f}  delay={m['avg_delay']:.1f}  "
              f"energy={m['avg_energy']:.0f}  viol={m['avg_violations']:.1f}")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp1_baseline.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return df


# ====================================================================
# Exp2: Ablation Study (unified 1000ep, 5 seeds)
# ====================================================================

def exp2_ablation():
    print("\n" + "=" * 70)
    print("  EXP2: Ablation Study (5 seeds, 1000ep)")
    print("=" * 70)

    configs = {
        "Full (Ours)": SACLagrangianConfig(),
        "w/o Lagrangian": SACLagrangianConfig(lambda_init=0.0, lr_lambda=0.0, lambda_max=0.0),
        "Fixed Lambda": SACLagrangianConfig(lr_lambda=0.0, lambda_init=2.0),
        "w/o Entropy Tuning": SACLagrangianConfig(lr_alpha=0.0),
    }

    rows = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        for name, sac_cfg in configs.items():
            cfg = ScenarioConfig(num_users=10, num_slots=100, seed=seed)
            env = UAVMECEnv(cfg=cfg)
            agent = SACLagrangianAgent(env.obs_dim, env.action_dim, sac_cfg)

            obs, _ = env.reset(seed=seed)
            for ep in range(1, TRAIN_EP + 1):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                while not done:
                    if agent.total_steps < sac_cfg.warmup_steps:
                        action = env.action_space.sample()
                    else:
                        action = agent.select_action(obs)
                    next_obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc
                    n_t = max(info.get("slot_tasks", 1), 1)
                    agent.store(obs, action, reward, next_obs, done,
                                cost_energy=(info.get("slot_energy", 0) + info.get("slot_fly_energy", 0)) / n_t,
                                cost_delay=info.get("slot_delay", 0) / n_t)
                    agent.update()
                    obs = next_obs

            m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
            m.update({"config": name, "seed": seed})
            rows.append(m)
            print(f"  {name:25s}: R={m['reward']:.1f}  viol={m['avg_violations']:.1f}  "
                  f"deadline_rate={m['deadline_rate']*100:.1f}%")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp2_ablation.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return df


# ====================================================================
# Exp3: Scalability (varying N, 5 seeds)
# ====================================================================

def exp3_scalability():
    user_counts = [5, 10, 15, 20]

    print("\n" + "=" * 70)
    print("  EXP3: Scalability (varying N, 5 seeds)")
    print("=" * 70)

    rows = []
    for N in user_counts:
        for seed in SEEDS:
            print(f"\n--- N={N}, Seed={seed} ---")
            cfg = ScenarioConfig(num_users=N, num_slots=100, seed=seed)
            env = UAVMECEnv(cfg=cfg)

            for bname in ["local", "full_offload", "greedy", "random"]:
                agent = make_baseline(bname, env.obs_dim, env.action_dim, N, seed=seed)
                m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
                m.update({"algorithm": bname, "num_users": N, "seed": seed})
                rows.append(m)

            t0 = time.time()
            env_sac = UAVMECEnv(cfg=cfg)
            train_sac_lagrangian(env_sac, TRAIN_EP, eval_interval=TRAIN_EP, save_dir=CKPT_DIR)
            train_time = time.time() - t0

            sac_agent = SACLagrangianAgent(env.obs_dim, env.action_dim)
            sac_agent.load(os.path.join(CKPT_DIR, "sac_lagrangian_best.pt"))
            m = evaluate(env, sac_agent, num_episodes=EVAL_EP, seed_base=seed * 100)
            m.update({"algorithm": "SAC-Lagrangian", "num_users": N, "seed": seed,
                       "train_time": train_time})
            rows.append(m)
            print(f"  SAC-Lagrangian: R={m['reward']:.1f}  viol={m['avg_violations']:.1f}  "
                  f"time={train_time:.0f}s")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp3_scalability.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return df


# ====================================================================
# Exp4: Convergence Curves (5 seeds)
# ====================================================================

def exp4_convergence():
    print("\n" + "=" * 70)
    print("  EXP4: Convergence Curves (5 seeds)")
    print("=" * 70)

    all_results = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        cfg = ScenarioConfig(num_users=10, num_slots=100, seed=seed)

        for algo_name, train_fn in [
            ("SAC-Lagrangian", train_sac_lagrangian),
            ("DQN", train_dqn),
            ("PPO", train_ppo),
            ("TD3", train_td3),
        ]:
            env = UAVMECEnv(cfg=cfg)
            hist = train_fn(env, TRAIN_EP, eval_interval=5, save_dir=CKPT_DIR)
            key = f"{algo_name}_seed{seed}"
            all_results[key] = hist

    path = os.path.join(RESULTS_DIR, "exp4_convergence.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: {path}")
    return all_results


# ====================================================================
# Exp5: Lagrangian Multiplier Dynamics (3 seeds)
# ====================================================================

def exp5_lagrangian_dynamics():
    print("\n" + "=" * 70)
    print("  EXP5: Lagrangian Multiplier Dynamics (3 seeds)")
    print("=" * 70)

    all_rows = []
    for seed in SEEDS[:3]:
        cfg = ScenarioConfig(num_users=10, num_slots=100, seed=seed)
        env = UAVMECEnv(cfg=cfg)
        hist = train_sac_lagrangian(env, TRAIN_EP, eval_interval=2, save_dir=CKPT_DIR)
        for h in hist:
            h["seed"] = seed
            all_rows.append(h)

    df = pd.DataFrame(all_rows)
    path = os.path.join(RESULTS_DIR, "exp5_lagrangian.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return df


# ====================================================================
# Exp6: Constraint Sensitivity (3 budget levels, 5 seeds) [NEW]
# ====================================================================

def exp6_constraint_sensitivity():
    print("\n" + "=" * 70)
    print("  EXP6: Constraint Sensitivity (3 budget levels)")
    print("=" * 70)

    budget_levels = {
        "Loose":  SACLagrangianConfig(delay_budget=0.6),
        "Medium": SACLagrangianConfig(delay_budget=0.4),
        "Tight":  SACLagrangianConfig(delay_budget=0.3),
    }

    rows = []
    for level_name, sac_cfg in budget_levels.items():
        for seed in SEEDS:
            print(f"\n--- {level_name}, Seed={seed} ---")
            cfg = ScenarioConfig(num_users=10, num_slots=100, seed=seed)

            # SAC-Lagrangian (Full)
            env = UAVMECEnv(cfg=cfg)
            agent = SACLagrangianAgent(env.obs_dim, env.action_dim, sac_cfg)
            obs, _ = env.reset(seed=seed)
            for ep in range(1, TRAIN_EP + 1):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                while not done:
                    if agent.total_steps < sac_cfg.warmup_steps:
                        action = env.action_space.sample()
                    else:
                        action = agent.select_action(obs)
                    next_obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc
                    n_t = max(info.get("slot_tasks", 1), 1)
                    agent.store(obs, action, reward, next_obs, done,
                                cost_energy=(info.get("slot_energy", 0) + info.get("slot_fly_energy", 0)) / n_t,
                                cost_delay=info.get("slot_delay", 0) / n_t)
                    agent.update()
                    obs = next_obs

            m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
            m.update({"method": "SAC-Lagrangian", "budget_level": level_name, "seed": seed,
                       "lambda_e": agent.lambda_e, "lambda_d": agent.lambda_d})
            rows.append(m)

            # w/o Lagrangian baseline (same env, no constraint)
            no_lag_cfg = SACLagrangianConfig(lambda_init=0.0, lr_lambda=0.0, lambda_max=0.0)
            env2 = UAVMECEnv(cfg=cfg)
            agent2 = SACLagrangianAgent(env2.obs_dim, env2.action_dim, no_lag_cfg)
            obs, _ = env2.reset(seed=seed)
            for ep in range(1, TRAIN_EP + 1):
                obs, _ = env2.reset(seed=seed + ep)
                done = False
                while not done:
                    if agent2.total_steps < no_lag_cfg.warmup_steps:
                        action = env2.action_space.sample()
                    else:
                        action = agent2.select_action(obs)
                    next_obs, reward, term, trunc, info = env2.step(action)
                    done = term or trunc
                    n_t = max(info.get("slot_tasks", 1), 1)
                    agent2.store(obs, action, reward, next_obs, done,
                                 cost_energy=(info.get("slot_energy", 0) + info.get("slot_fly_energy", 0)) / n_t,
                                 cost_delay=info.get("slot_delay", 0) / n_t)
                    agent2.update()
                    obs = next_obs

            m2 = evaluate(env2, agent2, num_episodes=EVAL_EP, seed_base=seed * 100)
            m2.update({"method": "w/o Lagrangian", "budget_level": level_name, "seed": seed,
                        "lambda_e": 0.0, "lambda_d": 0.0})
            rows.append(m2)

            print(f"  Full: R={m['reward']:.1f}  viol={m['avg_violations']:.1f}  "
                  f"λe={m['lambda_e']:.3f} λd={m['lambda_d']:.3f}")
            print(f"  NoLag: R={m2['reward']:.1f}  viol={m2['avg_violations']:.1f}")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp6_constraint_sensitivity.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return df


# ====================================================================
# Exp7: NOMA vs OMA (TDMA) [NEW]
# ====================================================================

def exp7_noma_vs_oma():
    """Compare NOMA vs TDMA (OMA) under SAC-Lagrangian."""
    print("\n" + "=" * 70)
    print("  EXP7: NOMA vs OMA (TDMA)")
    print("=" * 70)

    rows = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # NOMA (default)
        cfg_noma = ScenarioConfig(num_users=10, num_slots=100, seed=seed)
        env_noma = UAVMECEnv(cfg=cfg_noma)
        train_sac_lagrangian(env_noma, TRAIN_EP, eval_interval=TRAIN_EP, save_dir=CKPT_DIR)
        sac_noma = SACLagrangianAgent(env_noma.obs_dim, env_noma.action_dim)
        sac_noma.load(os.path.join(CKPT_DIR, "sac_lagrangian_best.pt"))
        m = evaluate(env_noma, sac_noma, num_episodes=EVAL_EP, seed_base=seed * 100)
        m.update({"access": "NOMA", "seed": seed})
        rows.append(m)
        print(f"  NOMA: R={m['reward']:.1f}  viol={m['avg_violations']:.1f}")

        # OMA (TDMA) — simulate by reducing bandwidth per user
        # In TDMA, each user gets B/N bandwidth exclusively
        oma_channel = ChannelConfig(bandwidth=5e6 / 10)  # 500 KHz per user
        cfg_oma = ScenarioConfig(num_users=10, num_slots=100, seed=seed)
        cfg_oma.channel = oma_channel
        env_oma = UAVMECEnv(cfg=cfg_oma)
        train_sac_lagrangian(env_oma, TRAIN_EP, eval_interval=TRAIN_EP, save_dir=CKPT_DIR)
        sac_oma = SACLagrangianAgent(env_oma.obs_dim, env_oma.action_dim)
        sac_oma.load(os.path.join(CKPT_DIR, "sac_lagrangian_best.pt"))
        m = evaluate(env_oma, sac_oma, num_episodes=EVAL_EP, seed_base=seed * 100)
        m.update({"access": "OMA-TDMA", "seed": seed})
        rows.append(m)
        print(f"  OMA:  R={m['reward']:.1f}  viol={m['avg_violations']:.1f}")

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "exp7_noma_oma.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return df


# ====================================================================
# Exp8: UAV Trajectory Visualization [NEW]
# ====================================================================

def exp8_trajectory_visualization():
    """Record and plot UAV trajectories for different algorithms."""
    print("\n" + "=" * 70)
    print("  EXP8: UAV Trajectory Visualization")
    print("=" * 70)

    cfg = ScenarioConfig(num_users=10, num_slots=100, seed=42)

    trajectories = {}

    # SAC-Lagrangian
    env = UAVMECEnv(cfg=cfg)
    train_sac_lagrangian(env, TRAIN_EP, eval_interval=TRAIN_EP, save_dir=CKPT_DIR)
    sac_agent = SACLagrangianAgent(env.obs_dim, env.action_dim)
    sac_agent.load(os.path.join(CKPT_DIR, "sac_lagrangian_best.pt"))

    for algo_name, agent in [("SAC-Lagrangian", sac_agent),
                              ("Random", make_baseline("random", env.obs_dim, env.action_dim, 10)),
                              ("Greedy", make_baseline("greedy", env.obs_dim, env.action_dim, 10))]:
        obs, info = env.reset(seed=42)
        traj = [info["uav_pos"][:2].tolist()]
        done = False
        while not done:
            if hasattr(agent, "select_action"):
                act = agent.select_action(obs, deterministic=True) if algo_name != "Random" else agent.select_action(obs)
            else:
                act = agent.select_action(obs)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
            traj.append(info["uav_pos"][:2].tolist())
        trajectories[algo_name] = traj

    # Save user positions
    env.reset(seed=42)
    trajectories["user_positions"] = env.user_positions.tolist()

    path = os.path.join(RESULTS_DIR, "exp8_trajectories.json")
    with open(path, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"Saved: {path}")

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 7))
        users = np.array(trajectories["user_positions"])
        ax.scatter(users[:, 0], users[:, 1], c="black", marker="^", s=80, label="Users", zorder=5)

        colors = {"SAC-Lagrangian": "#e63946", "Random": "#999999", "Greedy": "#457B9D"}
        for name, traj in trajectories.items():
            if name == "user_positions":
                continue
            t = np.array(traj)
            ax.plot(t[:, 0], t[:, 1], color=colors.get(name, "#333"), linewidth=2,
                    label=name, alpha=0.8)
            ax.plot(t[0, 0], t[0, 1], "o", color=colors.get(name), markersize=8)
            ax.plot(t[-1, 0], t[-1, 1], "s", color=colors.get(name), markersize=8)

        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("UAV Trajectory Comparison")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

        path = os.path.join(FIGURES_DIR, "fig5_trajectory.pdf")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved: {path}")
    except ImportError:
        print("matplotlib not available, skipping figure")

    return trajectories


# ====================================================================
# Generate figures
# ====================================================================

def generate_figures():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    plt.rcParams.update({"font.size": 11, "figure.figsize": (8, 5), "figure.dpi": 300})

    # --- Fig 1: Baseline bar chart ---
    csv = os.path.join(RESULTS_DIR, "exp1_baseline.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        metrics = [
            ("avg_delay", "Average Delay (s)", False),
            ("avg_energy", "Average Energy (J)", False),
            ("deadline_rate", "Deadline Success Rate", True),
        ]
        for ax, (col, ylabel, _) in zip(axes, metrics):
            grouped = df.groupby("algorithm")[col]
            means = grouped.mean()
            stds = grouped.std()
            order = ["local", "full_offload", "random", "round_robin",
                     "greedy", "DQN", "PPO", "TD3", "SAC-Lagrangian"]
            order = [o for o in order if o in means.index]
            means, stds = means[order], stds[order]
            colors = ["#ccc"] * (len(order) - 1) + ["#e63946"]
            ax.bar(range(len(order)), means, yerr=stds, capsize=3,
                   color=colors, edgecolor="#333", linewidth=0.8)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("Exp1: Baseline Comparison", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "fig1_baseline.pdf"), bbox_inches="tight")
        plt.close()
        print("Saved: fig1_baseline.pdf")

    # --- Fig 2: Convergence (mean±std ribbon) ---
    json_path = os.path.join(RESULTS_DIR, "exp4_convergence.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)

        fig, ax = plt.subplots(figsize=(8, 5))
        algo_colors = {"SAC-Lagrangian": "#e63946", "DQN": "#457B9D",
                       "PPO": "#2A9D8F", "TD3": "#E9C46A"}
        for algo in ["SAC-Lagrangian", "DQN", "PPO", "TD3"]:
            seed_curves = {}
            for key, hist in data.items():
                if key.startswith(algo):
                    for h in hist:
                        ep = h["episode"]
                        seed_curves.setdefault(ep, []).append(h["reward"])
            if not seed_curves:
                continue
            eps = sorted(seed_curves.keys())
            means = [np.mean(seed_curves[e]) for e in eps]
            stds = [np.std(seed_curves[e]) for e in eps]
            means, stds = np.array(means), np.array(stds)
            c = algo_colors.get(algo, "#999")
            ax.plot(eps, means, label=algo, color=c, linewidth=2)
            ax.fill_between(eps, means - stds, means + stds, alpha=0.15, color=c)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Evaluation Reward")
        ax.set_title("Convergence Comparison (mean ± std)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "fig2_convergence.pdf"), bbox_inches="tight")
        plt.close()
        print("Saved: fig2_convergence.pdf")

    # --- Fig 3: Lagrangian dynamics ---
    csv = os.path.join(RESULTS_DIR, "exp5_lagrangian.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for seed in df["seed"].unique():
            sub = df[df["seed"] == seed]
            ax1.plot(sub["episode"], sub["lambda_e"], alpha=0.5, label=f"λe (seed={seed})")
            ax1.plot(sub["episode"], sub["lambda_d"], alpha=0.5, linestyle="--",
                     label=f"λd (seed={seed})")
            ax2.plot(sub["episode"], sub["reward"], alpha=0.5, label=f"seed={seed}")
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Lagrangian Multiplier")
        ax1.set_title("Dual Variable Evolution"); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Reward")
        ax2.set_title("Training Reward"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "fig3_lagrangian.pdf"), bbox_inches="tight")
        plt.close()
        print("Saved: fig3_lagrangian.pdf")

    # --- Fig 4: Scalability ---
    csv = os.path.join(RESULTS_DIR, "exp3_scalability.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for algo in df["algorithm"].unique():
            sub = df[df["algorithm"] == algo].groupby("num_users").agg(
                delay_mean=("avg_delay", "mean"), delay_std=("avg_delay", "std"),
                dr_mean=("deadline_rate", "mean"), dr_std=("deadline_rate", "std")
            ).reset_index()
            style = "-o" if algo == "SAC-Lagrangian" else "--s"
            lw = 2.5 if algo == "SAC-Lagrangian" else 1.5
            ax1.errorbar(sub["num_users"], sub["delay_mean"], yerr=sub["delay_std"],
                         fmt=style, label=algo, linewidth=lw, capsize=3)
            ax2.errorbar(sub["num_users"], sub["dr_mean"], yerr=sub["dr_std"],
                         fmt=style, label=algo, linewidth=lw, capsize=3)
        ax1.set_xlabel("N"); ax1.set_ylabel("Average Delay (s)")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
        ax2.set_xlabel("N"); ax2.set_ylabel("Deadline Success Rate")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        fig.suptitle("Scalability", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "fig4_scalability.pdf"), bbox_inches="tight")
        plt.close()
        print("Saved: fig4_scalability.pdf")

    # --- Fig 6: Constraint sensitivity ---
    csv = os.path.join(RESULTS_DIR, "exp6_constraint_sensitivity.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        levels = ["Loose", "Medium", "Tight"]
        for method in ["SAC-Lagrangian", "w/o Lagrangian"]:
            sub = df[df["method"] == method]
            means_r = [sub[sub["budget_level"] == l]["reward"].mean() for l in levels]
            stds_r = [sub[sub["budget_level"] == l]["reward"].std() for l in levels]
            means_v = [sub[sub["budget_level"] == l]["avg_violations"].mean() for l in levels]
            stds_v = [sub[sub["budget_level"] == l]["avg_violations"].std() for l in levels]
            style = "-o" if method == "SAC-Lagrangian" else "--s"
            c = "#e63946" if method == "SAC-Lagrangian" else "#457B9D"
            ax1.errorbar(levels, means_r, yerr=stds_r, fmt=style, color=c,
                         label=method, linewidth=2, capsize=5)
            ax2.errorbar(levels, means_v, yerr=stds_v, fmt=style, color=c,
                         label=method, linewidth=2, capsize=5)
        ax1.set_ylabel("Reward"); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.set_ylabel("Deadline Violations"); ax2.legend(); ax2.grid(alpha=0.3)
        fig.suptitle("Constraint Budget Sensitivity", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES_DIR, "fig6_constraint_sensitivity.pdf"), bbox_inches="tight")
        plt.close()
        print("Saved: fig6_constraint_sensitivity.pdf")


# ====================================================================
# Print summary
# ====================================================================

def print_summary():
    csv = os.path.join(RESULTS_DIR, "exp1_baseline.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        print("\n" + "=" * 70)
        print("  TABLE 1: Baseline Comparison (mean ± std, 5 seeds)")
        print("=" * 70)
        print(f"{'Algorithm':<18s} {'Reward':>14s} {'Delay(s)':>14s} "
              f"{'Energy(J)':>14s} {'Deadline%':>14s}")
        print("-" * 76)
        for algo in ["local", "full_offload", "random", "round_robin", "greedy",
                      "DQN", "PPO", "TD3", "SAC-Lagrangian"]:
            sub = df[df["algorithm"] == algo]
            if len(sub) == 0:
                continue
            print(f"{algo:<18s} "
                  f"{sub['reward'].mean():>7.1f}±{sub['reward'].std():>5.1f} "
                  f"{sub['avg_delay'].mean():>7.1f}±{sub['avg_delay'].std():>5.1f} "
                  f"{sub['avg_energy'].mean():>7.0f}±{sub['avg_energy'].std():>4.0f} "
                  f"{sub['deadline_rate'].mean()*100:>6.1f}±{sub['deadline_rate'].std()*100:>4.1f}")


# ====================================================================
# Main
# ====================================================================

def main():
    overall_start = time.time()

    exp1_baseline_comparison()
    exp2_ablation()
    exp3_scalability()
    exp4_convergence()
    exp5_lagrangian_dynamics()
    exp6_constraint_sensitivity()
    exp7_noma_vs_oma()
    exp8_trajectory_visualization()

    generate_figures()
    print_summary()

    total = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"  TOTAL: {total:.0f}s ({total/60:.1f} min)")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Figures: {FIGURES_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
