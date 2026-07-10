"""
Supplementary experiments for reviewer responses:
  A. PID-Lagrangian baseline (Stooke 2020) vs DC-PPO
  B. OFDMA vs NOMA fairness comparison
  C. Imperfect SIC robustness (ε ∈ {0.05, 0.10})
  D. Exp3 scalability with 5 seeds (consistent with main table)

Run:
    cd ~/my_workspace/my_project/V2X_MEC_Offloading_Thesis
    python3 -u -m uav_mec.exp_supplementary 2>&1 | tee results/supp_exp_log.txt
"""

from __future__ import annotations
import copy, os, sys, json, time, csv
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)

from uav_mec.env import UAVMECEnv
from uav_mec.model import ScenarioConfig, NOMAChannel
from uav_mec.agents.dc_ppo import DCPPOAgent, DCPPOConfig
from uav_mec.agents.pid_lagrangian import PIDLagrangianAgent, PIDLagrangianConfig
from uav_mec.agents.ppo import PPOAgent
from uav_mec.agents.baselines import make_baseline
from uav_mec.train import evaluate

NUM_USERS   = 20
TRAIN_EP    = 1500
EVAL_EP     = 50
SEEDS       = [42, 123, 456, 789, 1024]
SEEDS3      = [42, 123, 456]
EVAL_INT    = 300
RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_env(n_users=NUM_USERS, seed=42, sic_eps=0.0):
    cfg = ScenarioConfig(num_users=n_users, num_slots=100, seed=seed)
    env = UAVMECEnv(cfg=cfg)
    if sic_eps > 0:
        env._sic_eps = sic_eps          # monkey-patch for imperfect SIC
    return env


def _best_dc(agent):
    return {k: copy.deepcopy(v) if hasattr(v, 'state_dict') else
            (v.data.clone() if hasattr(v, 'data') else
             (v.copy() if hasattr(v, 'copy') else v))
            for k, v in {
                "net": agent.net, "mu_head": agent.mu_head,
                "v_head": agent.v_head, "log_std": agent.log_std,
                "obs_mean": agent.obs_mean, "obs_var": agent.obs_var,
                "obs_count": agent.obs_count,
            }.items()}


def _restore_dc(agent, s):
    agent.net.load_state_dict(s["net"].state_dict())
    agent.mu_head.load_state_dict(s["mu_head"].state_dict())
    agent.v_head.load_state_dict(s["v_head"].state_dict())
    agent.log_std.data = s["log_std"].data.clone()
    agent.obs_mean = s["obs_mean"].copy()
    agent.obs_var  = s["obs_var"].copy()
    agent.obs_count= s["obs_count"]


def train_agent_generic(AgentClass, AgentConfig, seed, n_users=NUM_USERS,
                        label="Agent", sic_eps=0.0):
    """Generic training loop for DC-PPO / PID-Lagrangian."""
    env = _make_env(n_users, seed, sic_eps)
    cfg = AgentConfig()
    agent = AgentClass(env.obs_dim, env.action_dim, cfg, seed=seed)
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

        if ep % EVAL_INT == 0:
            m = evaluate(env, agent, num_episodes=10, seed_base=seed*100+ep)
            dr = m["deadline_rate"]
            print(f"    [{label} seed={seed} EP{ep:4d}] "
                  f"deadline={dr*100:.1f}%  λd={agent.lambda_d:.3f}")
            if dr > best_dr:
                best_dr = dr
                # simple deep-copy of state dicts
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
        print(f"    [{label} seed={seed}] Restored best: {best_dr*100:.1f}%")

    env2 = _make_env(n_users, seed, sic_eps)
    m = evaluate(env2, agent, num_episodes=EVAL_EP, seed_base=seed*100)
    return m["deadline_rate"], agent.lambda_d


def eval_ofdma(seed):
    """OFDMA baseline: each active user gets B/N_active, no inter-user interference."""
    env = _make_env(seed=seed)
    agent = make_baseline("ofdma", env.obs_dim, env.action_dim, NUM_USERS, seed=seed)
    m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed*100)
    return m["deadline_rate"]


# ── Sub-experiments ──────────────────────────────────────────────────────────

def run_A_pid_lagrangian():
    """A: PID-Lagrangian vs DC-PPO (5 seeds)."""
    print("\n" + "="*60)
    print("  Exp-A: PID-Lagrangian baseline (Stooke 2020)")
    print("="*60)
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        dr, ld = train_agent_generic(
            PIDLagrangianAgent, PIDLagrangianConfig,
            seed, label="PID-Lag")
        rows.append({"seed": seed, "deadline_rate": dr, "lambda_d_final": ld})
        print(f"  seed={seed} → {dr*100:.1f}%  λd={ld:.3f}  [{time.time()-t0:.0f}s]")

    mean_dr = np.mean([r["deadline_rate"] for r in rows])
    std_dr  = np.std([r["deadline_rate"] for r in rows])
    print(f"  PID-Lagrangian: {mean_dr*100:.1f}±{std_dr*100:.1f}%")

    # Compare vs DC-PPO (from main table)
    dc_ppo_mean = 82.47
    print(f"  DC-PPO (from main table): {dc_ppo_mean:.1f}%")
    print(f"  DC-PPO vs PID-Lag margin: {dc_ppo_mean - mean_dr*100:+.1f}pp")

    with open(os.path.join(RESULTS_DIR, "expA_pid_lagrangian.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed","deadline_rate","lambda_d_final"])
        w.writeheader(); w.writerows(rows)
    return mean_dr, std_dr


def run_B_ofdma():
    """B: OFDMA vs NOMA comparison.
    Uses full-offload DC-PPO from main experiment vs OFDMA random.
    """
    print("\n" + "="*60)
    print("  Exp-B: OFDMA baseline (fair OMA comparison)")
    print("="*60)

    # OFDMA: modify env to compute rates without NOMA interference
    # We inject an OFDMA rate override into the baselines
    rows = []
    for seed in SEEDS:
        env = _make_env(seed=seed)
        # OFDMA agent: random policy but with OFDMA rates
        agent = make_baseline("ofdma_noma", env.obs_dim, env.action_dim,
                               NUM_USERS, seed=seed)
        m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed*100)
        rows.append({"seed": seed, "algo": "OFDMA-Random",
                     "deadline_rate": m["deadline_rate"]})
        print(f"  OFDMA-Random seed={seed}: {m['deadline_rate']*100:.1f}%")

    mean_ofdma = np.mean([r["deadline_rate"] for r in rows]) * 100
    print(f"  OFDMA-Random mean: {mean_ofdma:.1f}%")
    print(f"  DC-PPO(NOMA) 82.5% vs OFDMA-Random {mean_ofdma:.1f}%")

    with open(os.path.join(RESULTS_DIR, "expB_ofdma.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed","algo","deadline_rate"])
        w.writeheader(); w.writerows(rows)

    # OFDMA DC-PPO: train DC-PPO with OFDMA rates (requires env modification)
    # This is the fair comparison: same RL algorithm, different channel access
    print("\n  Training DC-PPO with OFDMA rates (3 seeds for speed)...")
    rows_dc_ofdma = []
    for seed in SEEDS3:
        t0 = time.time()
        env = _make_env(seed=seed)
        env._use_ofdma = True   # monkey-patch flag
        cfg = DCPPOConfig(lambda_max=5.0, lr_lambda=3e-2)
        agent = DCPPOAgent(env.obs_dim, env.action_dim, cfg, seed=seed)
        best_dr, best_s = 0.0, None
        for ep in range(1, 1001):   # 1000ep for OFDMA (faster convergence expected)
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                n_t = max(info.get("slot_tasks", 1), 1)
                agent.store(obs, action, reward, done,
                            cost_delay=info.get("slot_delay", 0)/n_t)
                obs = next_obs
            agent.update()
            if ep % 200 == 0:
                m = evaluate(env, agent, num_episodes=10, seed_base=seed*100+ep)
                dr = m["deadline_rate"]
                print(f"    [DC-PPO OFDMA seed={seed} EP{ep}] {dr*100:.1f}%")
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
        env2 = _make_env(seed=seed); env2._use_ofdma = True
        m = evaluate(env2, agent, num_episodes=30, seed_base=seed*100)
        dr = m["deadline_rate"]
        rows_dc_ofdma.append({"seed": seed, "algo": "DC-PPO-OFDMA",
                               "deadline_rate": dr})
        print(f"  DC-PPO OFDMA seed={seed} → {dr*100:.1f}%  [{time.time()-t0:.0f}s]")

    mean_dc_ofdma = np.mean([r["deadline_rate"] for r in rows_dc_ofdma])*100
    print(f"\n  DC-PPO NOMA (main): 82.5%  DC-PPO OFDMA: {mean_dc_ofdma:.1f}%")
    print(f"  NOMA gain over OFDMA: {82.5 - mean_dc_ofdma:+.1f}pp")

    for r in rows_dc_ofdma:
        rows.append(r)
    with open(os.path.join(RESULTS_DIR, "expB_ofdma.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed","algo","deadline_rate"])
        w.writeheader(); w.writerows(rows)
    return mean_ofdma, mean_dc_ofdma


def run_C_imperfect_sic():
    """C: DC-PPO robustness under imperfect SIC (ε=0.05, 0.10)."""
    print("\n" + "="*60)
    print("  Exp-C: Imperfect SIC robustness")
    print("="*60)
    # Note: imperfect SIC is evaluated at inference time on a pre-trained agent.
    # We re-evaluate the main experiment best policies under imperfect SIC.
    # Since we don't have saved checkpoints, we retrain with 3 seeds for speed.

    rows = []
    for eps in [0.0, 0.05, 0.10]:
        drs = []
        for seed in SEEDS3:
            t0 = time.time()
            # Train DC-PPO under imperfect SIC env
            env = _make_env(seed=seed, sic_eps=eps)
            cfg = DCPPOConfig(lambda_max=5.0, lr_lambda=3e-2)
            agent = DCPPOAgent(env.obs_dim, env.action_dim, cfg, seed=seed)
            best_dr, best_s = 0.0, None
            for ep in range(1, 1001):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                while not done:
                    action = agent.select_action(obs)
                    next_obs, reward, term, trunc, info = env.step(action)
                    done = term or trunc
                    n_t = max(info.get("slot_tasks", 1), 1)
                    agent.store(obs, action, reward, done,
                                cost_delay=info.get("slot_delay", 0)/n_t)
                    obs = next_obs
                agent.update()
                if ep % 500 == 0:
                    m = evaluate(env, agent, num_episodes=10, seed_base=seed*100+ep)
                    dr = m["deadline_rate"]
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
            env2 = _make_env(seed=seed, sic_eps=eps)
            m = evaluate(env2, agent, num_episodes=30, seed_base=seed*100)
            dr = m["deadline_rate"]
            drs.append(dr)
            rows.append({"sic_eps": eps, "seed": seed, "deadline_rate": dr})
            print(f"    [ε={eps} seed={seed}] {dr*100:.1f}%  [{time.time()-t0:.0f}s]")
        mean_dr = np.mean(drs)
        print(f"  SIC ε={eps}: mean={mean_dr*100:.1f}±{np.std(drs)*100:.1f}%")

    with open(os.path.join(RESULTS_DIR, "expC_imperfect_sic.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sic_eps","seed","deadline_rate"])
        w.writeheader(); w.writerows(rows)
    return rows


def run_D_scalability_5seeds():
    """D: Scalability with 5 seeds (consistent with main table)."""
    print("\n" + "="*60)
    print("  Exp-D: Scalability N=10,15,20,25 with 5 seeds")
    print("="*60)
    from uav_mec.exp3_scalability import train_dc_ppo as _train_dc, train_ppo as _train_ppo

    rows = []
    summary = {}
    for N in [10, 15, 20, 25]:
        print(f"\n  === N={N} ===")
        drs_dc, drs_pp = [], []
        for seed in SEEDS:
            dr_dc = _train_dc(N, seed)
            drs_dc.append(dr_dc)
            rows.append({"N": N, "algo": "DC-PPO", "seed": seed, "deadline_rate": dr_dc})
            print(f"  DC-PPO N={N} seed={seed} → {dr_dc*100:.1f}%")

            dr_pp = _train_ppo(N, seed)
            drs_pp.append(dr_pp)
            rows.append({"N": N, "algo": "PPO", "seed": seed, "deadline_rate": dr_pp})
            print(f"  PPO    N={N} seed={seed} → {dr_pp*100:.1f}%")

        summary[(N, "DC-PPO")] = (np.mean(drs_dc), np.std(drs_dc))
        summary[(N, "PPO")]    = (np.mean(drs_pp), np.std(drs_pp))
        margin = np.mean(drs_dc) - np.mean(drs_pp)
        print(f"  N={N}: DC-PPO={np.mean(drs_dc)*100:.1f}%, PPO={np.mean(drs_pp)*100:.1f}%, margin={margin*100:+.1f}pp")

    print("\n  === Scalability Table (5 seeds) ===")
    print(f"  {'N':>4}  {'DC-PPO':>12}  {'PPO':>12}  {'margin':>8}")
    for N in [10, 15, 20, 25]:
        dc = summary[(N, "DC-PPO")]
        pp = summary[(N, "PPO")]
        print(f"  {N:>4}  {dc[0]*100:>5.1f}±{dc[1]*100:.1f}%  "
              f"{pp[0]*100:>5.1f}±{pp[1]*100:.1f}%  "
              f"{(dc[0]-pp[0])*100:>+6.1f}pp")

    with open(os.path.join(RESULTS_DIR, "expD_scalability_5seeds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N","algo","seed","deadline_rate"])
        w.writeheader(); w.writerows(rows)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    t_all = time.time()
    print("=" * 65)
    print("  SUPPLEMENTARY EXPERIMENTS for reviewer responses")
    print("=" * 65)

    # Run in order of priority
    run_A_pid_lagrangian()
    run_B_ofdma()
    run_C_imperfect_sic()
    run_D_scalability_5seeds()

    print(f"\n  Total time: {(time.time()-t_all)/60:.1f} min")
    print("=" * 65)


if __name__ == "__main__":
    run()
