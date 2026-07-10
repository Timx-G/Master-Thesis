"""
Supplementary experiments v2 (clean implementation):
  A. PID-Lagrangian quick test (3 seeds × 1000ep)
  B. OFDMA vs NOMA: DC-PPO with OFDMA rates (3 seeds × 1000ep)
  C. Imperfect SIC: DC-PPO under ε=0.0/0.05/0.10 (3 seeds × 1000ep each)

Run:
    python3 -u -m uav_mec.exp_supp2 2>&1 | tee results/supp2_log.txt
"""
from __future__ import annotations
import copy, os, sys, time, csv
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)

from uav_mec.env import UAVMECEnv
from uav_mec.model import ScenarioConfig
from uav_mec.agents.dc_ppo import DCPPOAgent, DCPPOConfig
from uav_mec.agents.pid_lagrangian import PIDLagrangianAgent, PIDLagrangianConfig
from uav_mec.train import evaluate

NUM_USERS = 20
TRAIN_EP  = 1000       # quick test: 1000ep
EVAL_EP   = 30
SEEDS     = [42, 123, 456]
EVAL_INT  = 200
RESULTS   = os.path.join(_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def _make_env(seed, use_ofdma=False, sic_eps=0.0):
    cfg = ScenarioConfig(num_users=NUM_USERS, num_slots=100, seed=seed)
    return UAVMECEnv(cfg=cfg, use_ofdma=use_ofdma, sic_eps=sic_eps)


def _train(AgentCls, AgentCfgCls, seed, label, use_ofdma=False, sic_eps=0.0):
    env = _make_env(seed, use_ofdma, sic_eps)
    cfg = AgentCfgCls()
    agent = AgentCls(env.obs_dim, env.action_dim, cfg, seed=seed)
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
            lam = agent.lambda_d
            print(f"    [{label} seed={seed} EP{ep:4d}] "
                  f"deadline={dr*100:.1f}%  λd={lam:.3f}")
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
        agent.obs_mean = best_s["obs_mean"].copy()
        agent.obs_var  = best_s["obs_var"].copy()
        agent.obs_count= best_s["obs_count"]
        print(f"    [{label} seed={seed}] Restored best: {best_dr*100:.1f}%")

    env2 = _make_env(seed, use_ofdma, sic_eps)
    m = evaluate(env2, agent, num_episodes=EVAL_EP, seed_base=seed*100)
    return m["deadline_rate"], agent.lambda_d


# ──────────────────────────────────────────────────────────────────────────────

def run_A():
    print("\n" + "="*60)
    print("  Exp-A: PID-Lagrangian (Stooke 2020) quick test")
    print("="*60)
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        dr, ld = _train(PIDLagrangianAgent, PIDLagrangianConfig,
                        seed, "PID-Lag")
        rows.append({"algo":"PID-Lag","seed":seed,"dr":dr,"ld":ld})
        print(f"  PID-Lag seed={seed}: {dr*100:.1f}%  λd={ld:.3f}  [{time.time()-t0:.0f}s]")
    mean_dr = np.mean([r["dr"] for r in rows])
    std_dr  = np.std([r["dr"] for r in rows])
    print(f"  PID-Lag: {mean_dr*100:.1f}±{std_dr*100:.1f}%")
    print(f"  DC-PPO (main, same 3 seeds {SEEDS}): "
          f"[80.59,85.64,80.95] → mean≈82.4%")
    print(f"  DC-PPO vs PID-Lag: {82.4 - mean_dr*100:+.1f}pp")
    with open(f"{RESULTS}/expA_pid_lag.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["algo","seed","dr","ld"])
        w.writeheader(); w.writerows(rows)
    return mean_dr


def run_B():
    print("\n" + "="*60)
    print("  Exp-B: DC-PPO with OFDMA rates (fair OMA comparison)")
    print("="*60)
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        dr_noma, ld = _train(DCPPOAgent, DCPPOConfig,
                             seed, "DC-PPO-NOMA", use_ofdma=False)
        dr_ofdma, _ = _train(DCPPOAgent, DCPPOConfig,
                              seed, "DC-PPO-OFDMA", use_ofdma=True)
        rows.append({"algo":"DC-PPO-NOMA", "seed":seed,"dr":dr_noma})
        rows.append({"algo":"DC-PPO-OFDMA","seed":seed,"dr":dr_ofdma})
        print(f"  seed={seed}: NOMA={dr_noma*100:.1f}%  OFDMA={dr_ofdma*100:.1f}%  "
              f"NOMA gain={(dr_noma-dr_ofdma)*100:+.1f}pp  [{time.time()-t0:.0f}s]")
    noma  = np.mean([r["dr"] for r in rows if r["algo"]=="DC-PPO-NOMA"])*100
    ofdma = np.mean([r["dr"] for r in rows if r["algo"]=="DC-PPO-OFDMA"])*100
    print(f"  DC-PPO NOMA: {noma:.1f}%  OFDMA: {ofdma:.1f}%  gain={noma-ofdma:+.1f}pp")
    with open(f"{RESULTS}/expB_ofdma.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["algo","seed","dr"])
        w.writeheader(); w.writerows(rows)
    return noma, ofdma


def run_C():
    print("\n" + "="*60)
    print("  Exp-C: Imperfect SIC robustness (ε=0.0 / 0.05 / 0.10)")
    print("="*60)
    rows = []
    for eps in [0.0, 0.05, 0.10]:
        drs = []
        for seed in SEEDS:
            t0 = time.time()
            dr, _ = _train(DCPPOAgent, DCPPOConfig,
                           seed, f"DC-PPO ε={eps}", sic_eps=eps)
            drs.append(dr)
            rows.append({"sic_eps":eps,"seed":seed,"dr":dr})
            print(f"  ε={eps} seed={seed}: {dr*100:.1f}%  [{time.time()-t0:.0f}s]")
        print(f"  SIC ε={eps}: {np.mean(drs)*100:.1f}±{np.std(drs)*100:.1f}%")
    with open(f"{RESULTS}/expC_sic.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["sic_eps","seed","dr"])
        w.writeheader(); w.writerows(rows)
    return rows


def run():
    t0 = time.time()
    print("="*65)
    print("  SUPPLEMENTARY EXPERIMENTS v2")
    print("="*65)
    run_A()
    run_B()
    run_C()
    print(f"\n  Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    run()
