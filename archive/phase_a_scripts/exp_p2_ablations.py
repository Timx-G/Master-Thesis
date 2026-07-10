"""
P2 experiments:
  P2a. Fixed-UAV ablation: UAV at center (100,100), optimize only power+offload
  P2b. Reward-shaping baseline: PPO with r = tasks - α·violations (no Lagrangian)

Run:
    python3 -u -m uav_mec.exp_p2_ablations 2>&1 | tee results/p2_log.txt
"""
from __future__ import annotations
import copy, os, sys, time, csv
import numpy as np
import torch, torch.nn as nn
from torch.distributions import Normal

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)

from uav_mec.env import UAVMECEnv
from uav_mec.model import ScenarioConfig
from uav_mec.agents.ppo import PPOAgent
from uav_mec.agents.dc_ppo import DCPPOAgent, DCPPOConfig
from uav_mec.train import evaluate

NUM_USERS = 20
TRAIN_EP  = 1500
EVAL_EP   = 50
SEEDS     = [42, 123, 456]
EVAL_INT  = 300
RESULTS   = os.path.join(_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


# ── Fixed-UAV environment wrapper ─────────────────────────────────────────────

class FixedUAVEnv(UAVMECEnv):
    """UAV fixed at area center; action ignores dx/dy displacement."""

    def step(self, action):
        # Force UAV to stay at center by zeroing displacement
        action = np.array(action, dtype=np.float32)
        action[0] = 0.0   # dx
        action[1] = 0.0   # dy
        return super().step(action)


def _make_env_fixed(seed):
    cfg = ScenarioConfig(num_users=NUM_USERS, num_slots=100, seed=seed)
    env = FixedUAVEnv(cfg=cfg)
    return env


def _make_env_normal(seed):
    cfg = ScenarioConfig(num_users=NUM_USERS, num_slots=100, seed=seed)
    return UAVMECEnv(cfg=cfg)


# ── Reward-shaping PPO: r' = r - alpha * violation_count ─────────────────────

class RewardShapingPPO(PPOAgent):
    """PPO with direct violation penalty in reward (no Lagrangian)."""

    def __init__(self, obs_dim, act_dim, alpha=2.0, seed=0):
        super().__init__(obs_dim, act_dim, seed=seed)
        self.alpha = alpha   # violation penalty weight

    def store_shaped(self, obs, act, rew, done, violations):
        """Store with shaped reward: r' = r - alpha * violations."""
        shaped_rew = rew - self.alpha * violations
        self.store(obs, act, shaped_rew, done)


# ── Training helpers ──────────────────────────────────────────────────────────

def _best_state_ppo(agent):
    return {
        "net":      copy.deepcopy(agent.net.state_dict()),
        "mu_head":  copy.deepcopy(agent.mu_head.state_dict()),
        "v_head":   copy.deepcopy(agent.v_head.state_dict()),
        "log_std":  agent.log_std.data.clone(),
        "obs_mean": agent.obs_mean.copy(),
        "obs_var":  agent.obs_var.copy(),
        "obs_count":agent.obs_count,
    }


def _restore_ppo(agent, s):
    agent.net.load_state_dict(s["net"])
    agent.mu_head.load_state_dict(s["mu_head"])
    agent.v_head.load_state_dict(s["v_head"])
    agent.log_std.data = s["log_std"]
    agent.obs_mean  = s["obs_mean"].copy()
    agent.obs_var   = s["obs_var"].copy()
    agent.obs_count = s["obs_count"]


def train_fixed_uav(seed):
    """Train DC-PPO with UAV fixed at center."""
    env = _make_env_fixed(seed)
    cfg = DCPPOConfig(lambda_max=5.0, lr_lambda=3e-2)
    agent = DCPPOAgent(env.obs_dim, env.action_dim, cfg, seed=seed)
    best_dr, best_s = 0.0, None

    for ep in range(1, TRAIN_EP + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            nobs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            n_t = max(info.get("slot_tasks", 1), 1)
            agent.store(obs, action, rew, done,
                        cost_delay=info.get("slot_delay", 0) / n_t)
            obs = nobs
        agent.update()
        if ep % EVAL_INT == 0:
            m = evaluate(env, agent, num_episodes=10, seed_base=seed*100+ep)
            dr = m["deadline_rate"]
            print(f"    [Fixed-UAV seed={seed} EP{ep:4d}] {dr*100:.1f}%")
            if dr > best_dr:
                best_dr = dr
                best_s = {"net": copy.deepcopy(agent.net.state_dict()),
                          "mu_head": copy.deepcopy(agent.mu_head.state_dict()),
                          "v_head": copy.deepcopy(agent.v_head.state_dict()),
                          "log_std": agent.log_std.data.clone(),
                          "obs_mean": agent.obs_mean.copy(),
                          "obs_var": agent.obs_var.copy(),
                          "obs_count": agent.obs_count}
    if best_s:
        agent.net.load_state_dict(best_s["net"])
        agent.mu_head.load_state_dict(best_s["mu_head"])
        agent.v_head.load_state_dict(best_s["v_head"])
        agent.log_std.data = best_s["log_std"]
        agent.obs_mean = best_s["obs_mean"]
        agent.obs_var  = best_s["obs_var"]
        agent.obs_count = best_s["obs_count"]
        print(f"    [Fixed-UAV seed={seed}] Restored best: {best_dr*100:.1f}%")
    env2 = _make_env_fixed(seed)
    m = evaluate(env2, agent, num_episodes=EVAL_EP, seed_base=seed*100)
    return m["deadline_rate"]


def train_reward_shaping(seed, alpha=2.0):
    """Train reward-shaping PPO."""
    env = _make_env_normal(seed)
    agent = RewardShapingPPO(env.obs_dim, env.action_dim, alpha=alpha, seed=seed)
    best_dr, best_s = 0.0, None

    for ep in range(1, TRAIN_EP + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            nobs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            # per-slot violation count as proxy penalty
            n_t = max(info.get("slot_tasks", 1), 1)
            violations = info.get("slot_violations", 0) / n_t
            agent.store_shaped(obs, action, rew, done, violations)
            obs = nobs
        agent.update()
        if ep % EVAL_INT == 0:
            m = evaluate(env, agent, num_episodes=10, seed_base=seed*100+ep)
            dr = m["deadline_rate"]
            print(f"    [RewShape α={alpha} seed={seed} EP{ep:4d}] {dr*100:.1f}%")
            if dr > best_dr:
                best_dr = dr
                best_s = _best_state_ppo(agent)
    if best_s:
        _restore_ppo(agent, best_s)
        print(f"    [RewShape seed={seed}] Restored best: {best_dr*100:.1f}%")
    env2 = _make_env_normal(seed)
    m = evaluate(env2, agent, num_episodes=EVAL_EP, seed_base=seed*100)
    return m["deadline_rate"]


def run():
    t0 = time.time()
    print("=" * 60)
    print("  P2 Ablations: Fixed-UAV + Reward-Shaping")
    print("=" * 60)
    rows = []

    # P2a: Fixed-UAV ablation
    print("\n[P2a] Fixed-UAV ablation (UAV at center, N=20)")
    drs_fixed = []
    for seed in SEEDS:
        t1 = time.time()
        dr = train_fixed_uav(seed)
        drs_fixed.append(dr)
        rows.append({"algo": "Fixed-UAV", "seed": seed, "dr": dr})
        print(f"  Fixed-UAV seed={seed}: {dr*100:.1f}%  [{time.time()-t1:.0f}s]")
    print(f"  Fixed-UAV: {np.mean(drs_fixed)*100:.1f}±{np.std(drs_fixed)*100:.1f}%")
    print(f"  DC-PPO (mobile) main table seeds[42,123,456]: ~82.4%")
    print(f"  Trajectory gain estimate: {82.4 - np.mean(drs_fixed)*100:+.1f}pp")

    # P2b: Reward-shaping baseline
    print("\n[P2b] Reward-shaping baseline (α=2.0)")
    drs_rs = []
    for seed in SEEDS:
        t1 = time.time()
        dr = train_reward_shaping(seed, alpha=2.0)
        drs_rs.append(dr)
        rows.append({"algo": "RewardShaping", "seed": seed, "dr": dr})
        print(f"  RewShape seed={seed}: {dr*100:.1f}%  [{time.time()-t1:.0f}s]")
    print(f"  Reward-Shaping: {np.mean(drs_rs)*100:.1f}±{np.std(drs_rs)*100:.1f}%")
    print(f"  DC-PPO vs Reward-Shaping: {82.4 - np.mean(drs_rs)*100:+.1f}pp")

    # Summary
    print("\n" + "="*60)
    print("  P2 ABLATION SUMMARY")
    print("="*60)
    print(f"  DC-PPO (mobile, Lagrangian):  ~82.4%  [main table]")
    print(f"  Fixed-UAV (Lagrangian only):  {np.mean(drs_fixed)*100:.1f}%  "
          f"[trajectory gain: {82.4-np.mean(drs_fixed)*100:+.1f}pp]")
    print(f"  Reward-Shaping (no Lagrangian): {np.mean(drs_rs)*100:.1f}%  "
          f"[Lagrangian gain: {82.4-np.mean(drs_rs)*100:+.1f}pp]")
    print(f"  PPO (baseline):               ~75.2%  [main table]")

    with open(f"{RESULTS}/p2_ablations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algo","seed","dr"])
        w.writeheader(); w.writerows(rows)
    print(f"\n  Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    run()
