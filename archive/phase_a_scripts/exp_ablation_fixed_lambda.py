"""
CRITICAL ABLATION: Fixed-λ=5 PPO vs DC-PPO

If Fixed-λ PPO ≈ DC-PPO → DC-PPO's adaptive mechanism adds ZERO value.
If DC-PPO > Fixed-λ PPO → the adaptation phase matters.

This is the make-or-break experiment for the paper.

Run:
    python3 -u -m uav_mec.exp_ablation_fixed_lambda 2>&1 | tee results/ablation_fixed_lambda.txt
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
from uav_mec.train import evaluate

NUM_USERS = 20
TRAIN_EP  = 1500
EVAL_EP   = 50
SEEDS     = [42, 123, 456, 789, 1024]
EVAL_INT  = 300
RESULTS   = os.path.join(_ROOT, "results")


def _make_env(seed):
    cfg = ScenarioConfig(num_users=NUM_USERS, num_slots=100, seed=seed)
    return UAVMECEnv(cfg=cfg)


def _save_best(agent):
    return {
        "net":      copy.deepcopy(agent.net.state_dict()),
        "mu_head":  copy.deepcopy(agent.mu_head.state_dict()),
        "v_head":   copy.deepcopy(agent.v_head.state_dict()),
        "log_std":  agent.log_std.data.clone(),
        "obs_mean": agent.obs_mean.copy(),
        "obs_var":  agent.obs_var.copy(),
        "obs_count":agent.obs_count,
    }


def _restore(agent, s):
    agent.net.load_state_dict(s["net"])
    agent.mu_head.load_state_dict(s["mu_head"])
    agent.v_head.load_state_dict(s["v_head"])
    agent.log_std.data = s["log_std"]
    agent.obs_mean = s["obs_mean"]
    agent.obs_var  = s["obs_var"]
    agent.obs_count= s["obs_count"]


class FixedLambdaPPO(PPOAgent):
    """PPO with fixed λ penalty: r' = r - λ_fixed * cost_delay.
    No Lagrangian update — λ is constant throughout training."""

    def __init__(self, obs_dim, act_dim, lambda_fixed=5.0, seed=0):
        super().__init__(obs_dim, act_dim, seed=seed)
        self.lambda_fixed = lambda_fixed

    @property
    def lambda_d(self):
        return self.lambda_fixed

    def store_with_penalty(self, obs, act, rew, done, cost_delay):
        modified_rew = rew - self.lambda_fixed * cost_delay
        self.store(obs, act, modified_rew, done)


def train_fixed_lambda(seed, lam=5.0):
    env = _make_env(seed)
    agent = FixedLambdaPPO(env.obs_dim, env.action_dim,
                           lambda_fixed=lam, seed=seed)
    best_dr, best_s = 0.0, None

    for ep in range(1, TRAIN_EP + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            nobs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            n_t = max(info.get("slot_tasks", 1), 1)
            agent.store_with_penalty(obs, action, rew, done,
                                     cost_delay=info.get("slot_delay", 0) / n_t)
            obs = nobs
        agent.update()

        if ep % EVAL_INT == 0:
            m = evaluate(env, agent, num_episodes=10, seed_base=seed*100+ep)
            dr = m["deadline_rate"]
            print(f"    [Fixed-λ={lam} seed={seed} EP{ep:4d}] {dr*100:.1f}%")
            if dr > best_dr:
                best_dr = dr
                best_s = _save_best(agent)

    if best_s:
        _restore(agent, best_s)
        print(f"    [Fixed-λ={lam} seed={seed}] Restored best: {best_dr*100:.1f}%")

    env2 = _make_env(seed)
    m = evaluate(env2, agent, num_episodes=EVAL_EP, seed_base=seed*100)
    return m["deadline_rate"]


def train_dc_ppo(seed):
    env = _make_env(seed)
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
            print(f"    [DC-PPO   seed={seed} EP{ep:4d}] {dr*100:.1f}%  λd={agent.lambda_d:.3f}")
            if dr > best_dr:
                best_dr = dr
                best_s = _save_best(agent)

    if best_s:
        _restore(agent, best_s)
        print(f"    [DC-PPO   seed={seed}] Restored best: {best_dr*100:.1f}%")

    env2 = _make_env(seed)
    m = evaluate(env2, agent, num_episodes=EVAL_EP, seed_base=seed*100)
    return m["deadline_rate"]


def run():
    t0 = time.time()
    print("=" * 65)
    print("  CRITICAL ABLATION: Fixed-λ=5 PPO vs DC-PPO")
    print("  If equal → DC-PPO adaptive mechanism = zero value")
    print("  If DC-PPO wins → adaptation matters")
    print("=" * 65)

    dc_drs, fixed_drs = [], []

    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")

        t1 = time.time()
        dr_dc = train_dc_ppo(seed)
        dc_drs.append(dr_dc)
        print(f"  DC-PPO:    {dr_dc*100:.1f}%  [{time.time()-t1:.0f}s]")

        t1 = time.time()
        dr_fixed = train_fixed_lambda(seed, lam=5.0)
        fixed_drs.append(dr_fixed)
        print(f"  Fixed-λ=5: {dr_fixed*100:.1f}%  [{time.time()-t1:.0f}s]")

        print(f"  Diff: {(dr_dc - dr_fixed)*100:+.1f}pp")

    dc_drs = np.array(dc_drs) * 100
    fixed_drs = np.array(fixed_drs) * 100

    print("\n" + "=" * 65)
    print("  ABLATION RESULT")
    print("=" * 65)
    print(f"  DC-PPO (adaptive λ):  {dc_drs.mean():.1f} ± {dc_drs.std():.1f}%")
    print(f"  Fixed-λ=5 PPO:        {fixed_drs.mean():.1f} ± {fixed_drs.std():.1f}%")
    print(f"  Difference:           {dc_drs.mean() - fixed_drs.mean():+.1f}pp")
    print()

    # Per-seed comparison
    print("  Per-seed:")
    for i, seed in enumerate(SEEDS):
        diff = dc_drs[i] - fixed_drs[i]
        winner = "DC-PPO" if diff > 0 else "Fixed" if diff < 0 else "TIE"
        print(f"    seed={seed}: DC-PPO={dc_drs[i]:.1f}%  Fixed={fixed_drs[i]:.1f}%  "
              f"diff={diff:+.1f}pp  [{winner}]")

    # Wilcoxon test
    from scipy import stats
    diffs = dc_drs - fixed_drs
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    try:
        stat, p = stats.wilcoxon(dc_drs, fixed_drs, alternative="two-sided")
        print(f"\n  Wilcoxon two-sided: stat={stat:.1f}, p={p:.4f}")
    except Exception as e:
        print(f"\n  Wilcoxon failed: {e}")
    print(f"  DC-PPO wins: {n_pos}/5, Fixed wins: {n_neg}/5")

    # VERDICT
    margin = dc_drs.mean() - fixed_drs.mean()
    print("\n" + "=" * 65)
    if abs(margin) < 2.0:
        print("  *** VERDICT: DC-PPO ≈ Fixed-λ PPO (|margin| < 2pp) ***")
        print("  *** DC-PPO's adaptive λ mechanism adds NEGLIGIBLE value ***")
        print("  *** Paper needs FUNDAMENTAL algorithm redesign ***")
    elif margin >= 2.0:
        print("  *** VERDICT: DC-PPO > Fixed-λ PPO (margin ≥ 2pp) ***")
        print("  *** Adaptive λ mechanism has measurable benefit ***")
        print("  *** Paper can be strengthened, not abandoned ***")
    else:
        print("  *** VERDICT: Fixed-λ PPO > DC-PPO ***")
        print("  *** DC-PPO is WORSE than simple fixed penalty ***")
        print("  *** Fundamental rethink required ***")
    print("=" * 65)
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    run()
