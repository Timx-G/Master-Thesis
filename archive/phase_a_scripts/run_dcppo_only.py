"""
DC-PPO only runner — 使用v5已有的PPO/TD3/PPO-Lag/heuristic结果，
只重跑DC-PPO部分，最终合并输出完整Table。
"""
from __future__ import annotations
import copy, os, sys, json, time
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)

from uav_mec.env import UAVMECEnv
from uav_mec.model import ScenarioConfig
from uav_mec.agents.dc_ppo import DCPPOAgent, DCPPOConfig
from uav_mec.train import evaluate

NUM_USERS    = 20
TRAIN_EP     = 1500
EVAL_EP      = 50
SEEDS        = [42, 123, 456, 789, 1024]
EVAL_INTERVAL= 100
RESULTS_DIR  = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def _make_env(seed):
    cfg = ScenarioConfig(num_users=NUM_USERS, num_slots=100, seed=seed)
    return UAVMECEnv(cfg=cfg)

def train_dc_ppo(seed: int) -> tuple:
    env = _make_env(seed)
    cfg = DCPPOConfig(lambda_max=5.0, lr_lambda=3e-2)
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
            print(f"    [DC-PPO seed={seed} EP{ep:4d}] deadline={dr*100:.1f}%  λd={agent.lambda_d:.3f}")
            if dr > best_deadline:
                best_deadline = dr
                best_state = {
                    "net":      copy.deepcopy(agent.net.state_dict()),
                    "mu_head":  copy.deepcopy(agent.mu_head.state_dict()),
                    "v_head":   copy.deepcopy(agent.v_head.state_dict()),
                    "log_std":  agent.log_std.data.clone(),
                    "obs_mean": agent.obs_mean.copy(),
                    "obs_var":  agent.obs_var.copy(),
                    "obs_count":agent.obs_count,
                }

    if best_state is not None:
        agent.net.load_state_dict(best_state["net"])
        agent.mu_head.load_state_dict(best_state["mu_head"])
        agent.v_head.load_state_dict(best_state["v_head"])
        agent.log_std.data = best_state["log_std"]
        agent.obs_mean = best_state["obs_mean"]
        agent.obs_var  = best_state["obs_var"]
        agent.obs_count= best_state["obs_count"]
        print(f"    [DC-PPO seed={seed}] Restored best checkpoint: {best_deadline*100:.1f}%")

    return agent, lam_hist

def run():
    t0 = time.time()
    print("=" * 60)
    print(f"  DC-PPO ONLY  N={NUM_USERS}  {TRAIN_EP}ep  {len(SEEDS)} seeds")
    print("=" * 60)

    dc_rows = []
    lam_traj = {}

    for seed in SEEDS:
        print(f"\n  seed={seed}")
        agent, lam_hist = train_dc_ppo(seed)
        env = _make_env(seed)
        m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
        m.update({"algorithm": "DC-PPO", "seed": seed, "lambda_d_final": agent.lambda_d})
        dc_rows.append(m)
        lam_traj[str(seed)] = lam_hist
        print(f"  → deadline={m['deadline_rate']*100:.1f}%  λd={agent.lambda_d:.3f}  [{time.time()-t0:.0f}s]")

    dc_mean = np.mean([r["deadline_rate"] for r in dc_rows])
    dc_std  = np.std( [r["deadline_rate"] for r in dc_rows])
    print(f"\n  DC-PPO mean={dc_mean*100:.1f}%±{dc_std*100:.1f}%")

    # 保存DC-PPO结果
    import csv as csvlib
    csv_path = os.path.join(RESULTS_DIR, "dcppo_only_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csvlib.DictWriter(f, fieldnames=list(dc_rows[0].keys()))
        w.writeheader(); w.writerows(dc_rows)
    print(f"  CSV: {csv_path}")

    lam_path = os.path.join(RESULTS_DIR, "dcppo_lambda_traj.json")
    with open(lam_path, "w") as f:
        json.dump(lam_traj, f, indent=2, default=float)

    # 已知v5结果（从log读取）
    known = {
        "greedy":         (15.4, 0.4),
        "random":         (63.0, 0.6),
        "PPO":            (73.8, None),
        "TD3":            (60.4, None),
        "PPO-Lagrangian": (80.5, None),
    }

    print("\n" + "=" * 60)
    print(f"  FINAL TABLE  N={NUM_USERS}  t_max=1.0s  f_uav=8GHz")
    print("=" * 60)
    print(f"  {'Algorithm':<18s} {'Deadline%':>12s}")
    print("  " + "-" * 35)
    order = ["DC-PPO", "PPO-Lagrangian", "PPO", "TD3", "greedy", "random"]
    for algo in order:
        if algo == "DC-PPO":
            print(f"  {algo:<18s} {dc_mean*100:>6.1f}±{dc_std*100:>4.1f}%  ◀ OURS")
        elif algo in known:
            m_, s_ = known[algo]
            s_str = f"±{s_:.1f}%" if s_ else "      "
            print(f"  {algo:<18s} {m_:>6.1f}{s_str}")

    # 验证门
    ppo_mean = known["PPO"][0] / 100
    margin = dc_mean - ppo_mean
    print(f"\n  DC-PPO vs PPO: {margin*100:+.1f}pp")
    print(f"  G1(≥78%): {'PASS ✅' if dc_mean>=0.78 else 'FAIL ❌'}  G3(≥5pp): {'PASS ✅' if margin>=0.05 else 'FAIL ❌'}")
    print(f"\n  耗时: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    run()
