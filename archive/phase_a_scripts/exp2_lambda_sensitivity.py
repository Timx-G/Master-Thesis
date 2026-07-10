"""
Exp2: lambda_max sensitivity analysis.

Sweep lambda_max ∈ {1, 2, 5, 10, 20} with 3 seeds × 1500ep each.
Produces results/exp2_lambda_sensitivity.csv + console Table.

Run:
    cd ~/my_workspace/my_project/V2X_MEC_Offloading_Thesis
    python3 -u -m uav_mec.exp2_lambda_sensitivity 2>&1 | tee results/exp2_log.txt
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

NUM_USERS     = 20
TRAIN_EP      = 1500
EVAL_EP       = 30
SEEDS         = [42, 123, 456]
EVAL_INTERVAL = 300
LAMBDA_VALUES = [1.0, 2.0, 5.0, 10.0, 20.0]
RESULTS_DIR   = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _make_env(seed):
    cfg = ScenarioConfig(num_users=NUM_USERS, num_slots=100, seed=seed)
    return UAVMECEnv(cfg=cfg)


def train_one(seed: int, lambda_max: float) -> tuple[float, float]:
    """Train DC-PPO with given lambda_max. Returns (final_deadline_rate, lambda_d_final)."""
    env = _make_env(seed)
    cfg = DCPPOConfig(lambda_max=lambda_max, lr_lambda=3e-2)
    agent = DCPPOAgent(env.obs_dim, env.action_dim, cfg, seed=seed)

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

        if ep % EVAL_INTERVAL == 0:
            m = evaluate(env, agent, num_episodes=10, seed_base=seed * 100 + ep)
            dr = m['deadline_rate']
            print(f"    [λmax={lambda_max:.0f} seed={seed} EP{ep:4d}] "
                  f"deadline={dr*100:.1f}%  λd={agent.lambda_d:.3f}")
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

    env2 = _make_env(seed)
    m = evaluate(env2, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
    return m["deadline_rate"], agent.lambda_d


def run():
    t0 = time.time()
    print("=" * 60)
    print(f"  Exp2: lambda_max sensitivity  N={NUM_USERS}  {TRAIN_EP}ep  {len(SEEDS)} seeds")
    print(f"  Lambda values: {LAMBDA_VALUES}")
    print("=" * 60)

    rows = []
    summary = {}   # lambda_max -> (mean_dr, std_dr, mean_ld)

    for lmax in LAMBDA_VALUES:
        print(f"\n  === lambda_max = {lmax:.0f} ===")
        drs, lds = [], []
        for seed in SEEDS:
            t1 = time.time()
            dr, ld = train_one(seed, lmax)
            drs.append(dr)
            lds.append(ld)
            rows.append({"lambda_max": lmax, "seed": seed,
                         "deadline_rate": dr, "lambda_d_final": ld})
            print(f"  seed={seed} → {dr*100:.1f}%  λd={ld:.3f}  [{time.time()-t1:.0f}s]")

        mean_dr = np.mean(drs)
        std_dr  = np.std(drs)
        mean_ld = np.mean(lds)
        summary[lmax] = (mean_dr, std_dr, mean_ld)
        print(f"  λmax={lmax:.0f}:  {mean_dr*100:.1f}±{std_dr*100:.1f}%  λd_avg={mean_ld:.3f}")

    # Save CSV
    import csv as csvlib
    csv_path = os.path.join(RESULTS_DIR, "exp2_lambda_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csvlib.DictWriter(f, fieldnames=["lambda_max", "seed", "deadline_rate", "lambda_d_final"])
        w.writeheader(); w.writerows(rows)
    print(f"\n  CSV: {csv_path}")

    # Final table
    print("\n" + "=" * 55)
    print(f"  Exp2 RESULT TABLE  N={NUM_USERS}  t_max=1.0s  f_uav=8GHz")
    print("=" * 55)
    print(f"  {'λ_max':>6s}  {'Deadline%':>12s}  {'λd_final':>9s}")
    print("  " + "-" * 35)
    for lmax in LAMBDA_VALUES:
        mean_dr, std_dr, mean_ld = summary[lmax]
        marker = "  ◀ OUR CHOICE" if lmax == 5.0 else ""
        print(f"  {lmax:>6.0f}  {mean_dr*100:>6.1f}±{std_dr*100:>4.1f}%  {mean_ld:>9.3f}{marker}")

    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print("=" * 55)


if __name__ == "__main__":
    run()
