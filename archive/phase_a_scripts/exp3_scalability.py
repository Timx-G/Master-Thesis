"""
Exp3: Scalability analysis — N ∈ {10, 15, 20, 25} users.

DC-PPO (lambda_max=5) vs PPO vs Random, 3 seeds × 1500ep each.
Primary metric: deadline_rate.

Run:
    cd ~/my_workspace/my_project/V2X_MEC_Offloading_Thesis
    python3 -u -m uav_mec.exp3_scalability 2>&1 | tee results/exp3_log.txt
"""

from __future__ import annotations
import copy, os, sys, time
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)

from uav_mec.env import UAVMECEnv
from uav_mec.model import ScenarioConfig
from uav_mec.agents.dc_ppo import DCPPOAgent, DCPPOConfig
from uav_mec.agents.ppo import PPOAgent
from uav_mec.agents.baselines import make_baseline
from uav_mec.train import evaluate

TRAIN_EP      = 1500
EVAL_EP       = 30
SEEDS         = [42, 123, 456]
EVAL_INTERVAL = 300
N_VALUES      = [10, 15, 20, 25]
RESULTS_DIR   = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _make_env(n_users, seed):
    cfg = ScenarioConfig(num_users=n_users, num_slots=100, seed=seed)
    return UAVMECEnv(cfg=cfg)


def _best_state_dc(agent):
    return {
        "net":       copy.deepcopy(agent.net.state_dict()),
        "mu_head":   copy.deepcopy(agent.mu_head.state_dict()),
        "v_head":    copy.deepcopy(agent.v_head.state_dict()),
        "log_std":   agent.log_std.data.clone(),
        "obs_mean":  agent.obs_mean.copy(),
        "obs_var":   agent.obs_var.copy(),
        "obs_count": agent.obs_count,
    }


def _restore_state_dc(agent, s):
    agent.net.load_state_dict(s["net"])
    agent.mu_head.load_state_dict(s["mu_head"])
    agent.v_head.load_state_dict(s["v_head"])
    agent.log_std.data = s["log_std"]
    agent.obs_mean  = s["obs_mean"]
    agent.obs_var   = s["obs_var"]
    agent.obs_count = s["obs_count"]


def train_dc_ppo(n_users, seed):
    env = _make_env(n_users, seed)
    cfg = DCPPOConfig(lambda_max=5.0, lr_lambda=3e-2)
    agent = DCPPOAgent(env.obs_dim, env.action_dim, cfg, seed=seed)
    best_dr, best_s = 0.0, None

    for ep in range(1, TRAIN_EP + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            n_t = max(info.get("slot_tasks", 1), 1)
            agent.store(obs, action, reward, done,
                        cost_delay=info.get("slot_delay", 0) / n_t)
            obs = next_obs
        agent.update()

        if ep % EVAL_INTERVAL == 0:
            m = evaluate(env, agent, num_episodes=10, seed_base=seed * 100 + ep)
            dr = m["deadline_rate"]
            print(f"    [DC-PPO N={n_users} seed={seed} EP{ep:4d}] "
                  f"deadline={dr*100:.1f}%  λd={agent.lambda_d:.3f}")
            if dr > best_dr:
                best_dr = dr
                best_s = _best_state_dc(agent)

    if best_s:
        _restore_state_dc(agent, best_s)
    env2 = _make_env(n_users, seed)
    m = evaluate(env2, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
    return m["deadline_rate"]


def train_ppo(n_users, seed):
    env = _make_env(n_users, seed)
    agent = PPOAgent(env.obs_dim, env.action_dim, seed=seed)
    best_dr, best_s = 0.0, None

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
            dr = m["deadline_rate"]
            print(f"    [PPO     N={n_users} seed={seed} EP{ep:4d}] "
                  f"deadline={dr*100:.1f}%")
            if dr > best_dr:
                best_dr = dr
                best_s = {
                    "net":      copy.deepcopy(agent.net.state_dict()),
                    "mu_head":  copy.deepcopy(agent.mu_head.state_dict()),
                    "v_head":   copy.deepcopy(agent.v_head.state_dict()),
                    "log_std":  agent.log_std.data.clone(),
                    "obs_mean": agent.obs_mean.copy(),
                    "obs_var":  agent.obs_var.copy(),
                    "obs_count":agent.obs_count,
                }

    if best_s:
        agent.net.load_state_dict(best_s["net"])
        agent.mu_head.load_state_dict(best_s["mu_head"])
        agent.v_head.load_state_dict(best_s["v_head"])
        agent.log_std.data = best_s["log_std"]
        agent.obs_mean  = best_s["obs_mean"]
        agent.obs_var   = best_s["obs_var"]
        agent.obs_count = best_s["obs_count"]

    env2 = _make_env(n_users, seed)
    m = evaluate(env2, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
    return m["deadline_rate"]


def eval_random(n_users, seed):
    env = _make_env(n_users, seed)
    agent = make_baseline("random", env.obs_dim, env.action_dim, n_users, seed=seed)
    m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
    return m["deadline_rate"]


def run():
    t0 = time.time()
    print("=" * 65)
    print(f"  Exp3: Scalability  N={N_VALUES}  {TRAIN_EP}ep  {len(SEEDS)} seeds")
    print("=" * 65)

    rows = []   # {N, algo, seed, deadline_rate}
    summary = {}  # (N, algo) -> (mean, std)

    for N in N_VALUES:
        print(f"\n  ===== N = {N} =====")

        # Random (fast)
        rdr = [eval_random(N, s) for s in SEEDS]
        for i, s in enumerate(SEEDS):
            rows.append({"N": N, "algo": "Random", "seed": s, "deadline_rate": rdr[i]})
        summary[(N, "Random")] = (np.mean(rdr), np.std(rdr))
        print(f"  Random  N={N}: {np.mean(rdr)*100:.1f}±{np.std(rdr)*100:.1f}%")

        # PPO
        pdr = []
        for s in SEEDS:
            t1 = time.time()
            dr = train_ppo(N, s)
            pdr.append(dr)
            rows.append({"N": N, "algo": "PPO", "seed": s, "deadline_rate": dr})
            print(f"  PPO    N={N} seed={s} → {dr*100:.1f}%  [{time.time()-t1:.0f}s]")
        summary[(N, "PPO")] = (np.mean(pdr), np.std(pdr))
        print(f"  PPO    N={N}: {np.mean(pdr)*100:.1f}±{np.std(pdr)*100:.1f}%")

        # DC-PPO
        ddr = []
        for s in SEEDS:
            t1 = time.time()
            dr = train_dc_ppo(N, s)
            ddr.append(dr)
            rows.append({"N": N, "algo": "DC-PPO", "seed": s, "deadline_rate": dr})
            print(f"  DC-PPO N={N} seed={s} → {dr*100:.1f}%  [{time.time()-t1:.0f}s]")
        summary[(N, "DC-PPO")] = (np.mean(ddr), np.std(ddr))
        print(f"  DC-PPO N={N}: {np.mean(ddr)*100:.1f}±{np.std(ddr)*100:.1f}%")
        print(f"  DC-PPO vs PPO margin: {(np.mean(ddr)-np.mean(pdr))*100:+.1f}pp")

    # Save CSV
    import csv as csvlib
    csv_path = os.path.join(RESULTS_DIR, "exp3_scalability.csv")
    with open(csv_path, "w", newline="") as f:
        w = csvlib.DictWriter(f, fieldnames=["N", "algo", "seed", "deadline_rate"])
        w.writeheader(); w.writerows(rows)
    print(f"\n  CSV: {csv_path}")

    # Final table
    print("\n" + "=" * 65)
    print(f"  Exp3 SCALABILITY TABLE  t_max=1.0s  f_uav=8GHz  {TRAIN_EP}ep  {len(SEEDS)}seeds")
    print("=" * 65)
    print(f"  {'N':>4}  {'DC-PPO':>12}  {'PPO':>12}  {'Random':>12}  {'margin':>8}")
    print("  " + "-" * 58)
    for N in N_VALUES:
        dc = summary[(N, "DC-PPO")]
        pp = summary[(N, "PPO")]
        rr = summary[(N, "Random")]
        margin = dc[0] - pp[0]
        marker = "  *" if N == 20 else ""
        print(f"  {N:>4}  {dc[0]*100:>5.1f}±{dc[1]*100:.1f}%  "
              f"{pp[0]*100:>5.1f}±{pp[1]*100:.1f}%  "
              f"{rr[0]*100:>5.1f}±{rr[1]*100:.1f}%  "
              f"{margin*100:>+6.1f}pp{marker}")

    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    run()
