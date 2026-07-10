"""
Full Experiment: DC-PPO Paper Main Results  [v2: hidden=256, entropy=0.02, 1500ep, best-ckpt]

Scene : N=20 users, area=200x200m, f_uav=8GHz, t_max=1.0s, T=100 slots
Seeds : [42, 123, 456, 789, 1024]  (5 seeds)
Epochs: 1500 episodes (DC-PPO/PPO/PPO-Lag), 2000 episodes (TD3)

Algorithms
----------
1. DC-PPO         (Ours)   PPO + Lagrangian(lambda_max=5, lr=3e-2)
2. PPO                     unconstrained baseline
3. PPO-Lagrangian          PPO + Lagrangian(lambda_max=20, ablation — shows instability)
4. TD3                     off-policy unconstrained baseline
5. Greedy                  heuristic baseline
6. Random                  lower bound

Primary metric : deadline_rate  (tasks meeting t_max / total tasks served)
Secondary      : avg_delay, lambda_d trajectory (for convergence figure)

Run:
    cd ~/my_workspace/my_project/V2X_MEC_Offloading_Thesis
    python3 -m uav_mec.full_experiment_dcppo 2>&1 | tee results/full_exp_log.txt
"""

from __future__ import annotations

import copy
import os
import sys
import json
import time

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)

from uav_mec.env import UAVMECEnv
from uav_mec.model import ScenarioConfig
from uav_mec.agents.dc_ppo import DCPPOAgent, DCPPOConfig
from uav_mec.agents.ppo import PPOAgent
from uav_mec.agents.td3 import TD3Agent
from uav_mec.agents.baselines import make_baseline
from uav_mec.train import evaluate

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_USERS     = 20
TRAIN_EP      = 1500      # PPO / DC-PPO / PPO-Lagrangian
TD3_TRAIN_EP  = 2000      # TD3 needs more steps for 42-dim off-policy
EVAL_EP       = 50
SEEDS     = [42, 123, 456, 789, 1024]
EVAL_INTERVAL = 100          # print progress every N episodes

RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Training helpers ───────────────────────────────────────────────────────────

def _make_env(seed):
    cfg = ScenarioConfig(num_users=NUM_USERS, num_slots=100, seed=seed)
    return UAVMECEnv(cfg=cfg)


def train_dc_ppo(seed: int, lambda_max: float = 5.0,
                 lr_lambda: float = 3e-2, label: str = "DC-PPO") -> tuple:
    """Train DC-PPO (or PPO-Lagrangian variant). Returns (best_agent, lambda_trajectory)."""
    env = _make_env(seed)
    cfg = DCPPOConfig(lambda_max=lambda_max, lr_lambda=lr_lambda)
    agent = DCPPOAgent(env.obs_dim, env.action_dim, cfg, seed=seed)
    lam_hist = []

    best_deadline = 0.0
    best_state = None

    for ep in range(1, TRAIN_EP + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            n_t = max(info.get("slot_tasks", 1), 1)
            cost = info.get("slot_delay", 0) / n_t
            agent.store(obs, action, reward, done, cost_delay=cost)
            obs = next_obs
        agent.update()
        lam_hist.append(agent.lambda_d)

        if ep % EVAL_INTERVAL == 0:
            m = evaluate(env, agent, num_episodes=10, seed_base=seed * 100 + ep)
            dr = m['deadline_rate']
            print(f"    [{label} seed={seed} EP{ep:4d}] "
                  f"deadline={dr*100:.1f}%  λd={agent.lambda_d:.3f}")
            if dr > best_deadline:
                best_deadline = dr
                best_state = {
                    "net": copy.deepcopy(agent.net.state_dict()),
                    "mu_head": copy.deepcopy(agent.mu_head.state_dict()),
                    "v_head": copy.deepcopy(agent.v_head.state_dict()),
                    "log_std": agent.log_std.data.clone(),
                    "obs_mean": agent.obs_mean.copy(),
                    "obs_var": agent.obs_var.copy(),
                    "obs_count": agent.obs_count,
                }

    # Restore best checkpoint
    if best_state is not None:
        agent.net.load_state_dict(best_state["net"])
        agent.mu_head.load_state_dict(best_state["mu_head"])
        agent.v_head.load_state_dict(best_state["v_head"])
        agent.log_std.data = best_state["log_std"]
        agent.obs_mean = best_state["obs_mean"]
        agent.obs_var = best_state["obs_var"]
        agent.obs_count = best_state["obs_count"]
        print(f"    [{label} seed={seed}] Restored best checkpoint: {best_deadline*100:.1f}%")

    return agent, lam_hist


def train_ppo(seed: int) -> PPOAgent:
    """Train vanilla PPO (no Lagrangian). Returns best checkpoint."""
    env = _make_env(seed)
    agent = PPOAgent(env.obs_dim, env.action_dim, seed=seed)

    best_deadline = 0.0
    best_state = None

    for ep in range(1, TRAIN_EP + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            agent.store(obs, action, reward, done)
            obs = next_obs
        agent.update()

        if ep % EVAL_INTERVAL == 0:
            m = evaluate(env, agent, num_episodes=10, seed_base=seed * 100 + ep)
            dr = m['deadline_rate']
            print(f"    [PPO     seed={seed} EP{ep:4d}] deadline={dr*100:.1f}%")
            if dr > best_deadline:
                best_deadline = dr
                best_state = {
                    "net": copy.deepcopy(agent.net.state_dict()),
                    "mu_head": copy.deepcopy(agent.mu_head.state_dict()),
                    "v_head": copy.deepcopy(agent.v_head.state_dict()),
                    "log_std": agent.log_std.data.clone(),
                    "obs_mean": agent.obs_mean.copy(),
                    "obs_var": agent.obs_var.copy(),
                    "obs_count": agent.obs_count,
                }

    if best_state is not None:
        agent.net.load_state_dict(best_state["net"])
        agent.mu_head.load_state_dict(best_state["mu_head"])
        agent.v_head.load_state_dict(best_state["v_head"])
        agent.log_std.data = best_state["log_std"]
        agent.obs_mean = best_state["obs_mean"]
        agent.obs_var = best_state["obs_var"]
        agent.obs_count = best_state["obs_count"]
        print(f"    [PPO     seed={seed}] Restored best checkpoint: {best_deadline*100:.1f}%")

    return agent


def train_td3(seed: int) -> TD3Agent:
    """Train TD3 with 2000 episodes. Returns best checkpoint."""
    env = _make_env(seed)
    agent = TD3Agent(env.obs_dim, env.action_dim, seed=seed)

    best_deadline = 0.0
    best_state = None

    for ep in range(1, TD3_TRAIN_EP + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            if agent.total_steps < agent.warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            agent.store(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs

        if ep % EVAL_INTERVAL == 0:
            m = evaluate(env, agent, num_episodes=10, seed_base=seed * 100 + ep)
            dr = m['deadline_rate']
            print(f"    [TD3     seed={seed} EP{ep:4d}] deadline={dr*100:.1f}%")
            if dr > best_deadline:
                best_deadline = dr
                best_state = {
                    "actor": copy.deepcopy(agent.actor.state_dict()),
                    "obs_mean": agent.obs_mean.copy() if hasattr(agent, "obs_mean") else None,
                }

    if best_state is not None:
        agent.actor.load_state_dict(best_state["actor"])
        if best_state["obs_mean"] is not None and hasattr(agent, "obs_mean"):
            agent.obs_mean = best_state["obs_mean"]
        print(f"    [TD3     seed={seed}] Restored best checkpoint: {best_deadline*100:.1f}%")

    return agent


def eval_heuristic(name: str, seed: int) -> dict:
    env = _make_env(seed)
    agent = make_baseline(name, env.obs_dim, env.action_dim, NUM_USERS, seed=seed)
    m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
    return m


# ── Main experiment ────────────────────────────────────────────────────────────

def run():
    t_total = time.time()

    all_rows = []
    lam_trajectories = {}   # algo -> {seed -> list of lambda_d per episode}

    print("=" * 72)
    print(f"  FULL EXPERIMENT v2: DC-PPO  N={NUM_USERS}  PPO/DC-PPO={TRAIN_EP}ep  TD3={TD3_TRAIN_EP}ep  {len(SEEDS)} seeds")
    print("=" * 72)

    # ── Heuristics (seed-independent, fast) ───────────────────────────────────
    print("\n[1/6] Heuristics: Greedy + Random")
    for name in ["greedy", "random"]:
        for seed in SEEDS:
            m = eval_heuristic(name, seed)
            m.update({"algorithm": name, "seed": seed, "lambda_d_final": 0.0})
            all_rows.append(m)
        means = [r["deadline_rate"] for r in all_rows if r["algorithm"] == name]
        print(f"  {name:6s}: deadline={np.mean(means)*100:.1f}%±{np.std(means)*100:.1f}%")

    # ── PPO ───────────────────────────────────────────────────────────────────
    print("\n[2/6] PPO (unconstrained baseline)")
    for seed in SEEDS:
        print(f"  seed={seed}")
        t0 = time.time()
        agent = train_ppo(seed)
        env = _make_env(seed)
        m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
        m.update({"algorithm": "PPO", "seed": seed, "lambda_d_final": 0.0})
        all_rows.append(m)
        print(f"  → deadline={m['deadline_rate']*100:.1f}%  [{time.time()-t0:.0f}s]")
    ppo_mean = np.mean([r["deadline_rate"] for r in all_rows if r["algorithm"] == "PPO"])
    print(f"  PPO mean={ppo_mean*100:.1f}%")

    # ── TD3 ───────────────────────────────────────────────────────────────────
    print("\n[3/6] TD3 (off-policy baseline)")
    for seed in SEEDS:
        print(f"  seed={seed}")
        t0 = time.time()
        agent = train_td3(seed)
        env = _make_env(seed)
        m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
        m.update({"algorithm": "TD3", "seed": seed, "lambda_d_final": 0.0})
        all_rows.append(m)
        print(f"  → deadline={m['deadline_rate']*100:.1f}%  [{time.time()-t0:.0f}s]")
    td3_mean = np.mean([r["deadline_rate"] for r in all_rows if r["algorithm"] == "TD3"])
    print(f"  TD3 mean={td3_mean*100:.1f}%")

    # ── PPO-Lagrangian (ablation: lambda_max=20, shows instability) ───────────
    print("\n[4/6] PPO-Lagrangian (ablation: lambda_max=20)")
    lam_trajectories["PPO-Lagrangian"] = {}
    for seed in SEEDS:
        print(f"  seed={seed}")
        t0 = time.time()
        agent, lam_hist = train_dc_ppo(seed, lambda_max=20.0, lr_lambda=3e-2,
                                       label="PPO-Lag")
        env = _make_env(seed)
        m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
        m.update({"algorithm": "PPO-Lagrangian", "seed": seed,
                  "lambda_d_final": agent.lambda_d})
        all_rows.append(m)
        lam_trajectories["PPO-Lagrangian"][str(seed)] = lam_hist
        print(f"  → deadline={m['deadline_rate']*100:.1f}%  λd={agent.lambda_d:.3f}  "
              f"[{time.time()-t0:.0f}s]")
    plag_mean = np.mean([r["deadline_rate"] for r in all_rows
                         if r["algorithm"] == "PPO-Lagrangian"])
    print(f"  PPO-Lagrangian mean={plag_mean*100:.1f}%")

    # ── DC-PPO (Ours) ─────────────────────────────────────────────────────────
    print("\n[5/6] DC-PPO (Ours: lambda_max=5)")
    lam_trajectories["DC-PPO"] = {}
    for seed in SEEDS:
        print(f"  seed={seed}")
        t0 = time.time()
        agent, lam_hist = train_dc_ppo(seed, lambda_max=5.0, lr_lambda=3e-2,
                                       label="DC-PPO")
        env = _make_env(seed)
        m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
        m.update({"algorithm": "DC-PPO", "seed": seed,
                  "lambda_d_final": agent.lambda_d})
        all_rows.append(m)
        lam_trajectories["DC-PPO"][str(seed)] = lam_hist
        print(f"  → deadline={m['deadline_rate']*100:.1f}%  λd={agent.lambda_d:.3f}  "
              f"[{time.time()-t0:.0f}s]")
    dc_mean = np.mean([r["deadline_rate"] for r in all_rows if r["algorithm"] == "DC-PPO"])
    dc_std  = np.std([r["deadline_rate"] for r in all_rows if r["algorithm"] == "DC-PPO"])
    print(f"  DC-PPO mean={dc_mean*100:.1f}%±{dc_std*100:.1f}%")

    # ── Save raw results ───────────────────────────────────────────────────────
    print("\n[6/6] Saving results")
    csv_path = os.path.join(RESULTS_DIR, "exp1_dcppo_baseline.csv")
    import csv as csvlib
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csvlib.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
    print(f"  CSV: {csv_path}")

    lam_path = os.path.join(RESULTS_DIR, "lambda_trajectories.json")
    with open(lam_path, "w") as f:
        json.dump(lam_trajectories, f, indent=2, default=float)
    print(f"  Lambda trajectories: {lam_path}")

    # ── Final summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print(f"  TABLE: N={NUM_USERS}, t_max=1.0s, f_uav=8GHz, {TRAIN_EP}ep, {len(SEEDS)} seeds")
    print("=" * 88)
    print(f"  {'Algorithm':<18s} {'Deadline%':>12s}  {'Delay(s/task)':>14s}  "
          f"{'Energy(J/task)':>15s}  {'ViolRate':>9s}  {'λd':>6s}")
    print("  " + "-" * 82)
    order = ["DC-PPO", "PPO-Lagrangian", "PPO", "TD3", "greedy", "random"]
    for algo in order:
        rows = [r for r in all_rows if r["algorithm"] == algo]
        if not rows:
            continue
        dr   = np.mean([r["deadline_rate"]   for r in rows])
        dr_s = np.std( [r["deadline_rate"]   for r in rows])
        dpt  = np.mean([r.get("delay_per_task",  r["avg_delay"])   for r in rows])
        ept  = np.mean([r.get("energy_per_task", r["avg_energy"])  for r in rows])
        vr   = np.mean([r.get("violation_rate",  1-r["deadline_rate"]) for r in rows])
        ld   = np.mean([r["lambda_d_final"]  for r in rows])
        marker = "  ◀ OURS" if algo == "DC-PPO" else ""
        print(f"  {algo:<18s} {dr*100:>6.1f}±{dr_s*100:>4.1f}%  "
              f"{dpt:>13.4f}s  {ept:>14.4f}J  "
              f"{vr*100:>8.1f}%  {ld:>6.3f}{marker}")

    margin = dc_mean - ppo_mean
    print(f"\n  DC-PPO vs PPO margin: {margin*100:+.1f}pp")
    gate1 = dc_mean >= 0.78
    gate3 = margin >= 0.05
    print(f"  G1(DC-PPO≥78%): {'PASS ✅' if gate1 else 'FAIL ❌'}  ({dc_mean*100:.1f}%)")
    print(f"  G3(margin≥5pp): {'PASS ✅' if gate3 else 'FAIL ❌'}  ({margin*100:.1f}pp)")
    print(f"\n  Total time: {(time.time()-t_total)/60:.1f} min")
    print("=" * 88)


if __name__ == "__main__":
    run()
