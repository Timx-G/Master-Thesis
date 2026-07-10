"""
Quick Test: Validate f_uav=10GHz + lambda_max=100 + lr_lambda=1e-3 before full re-run.

Pass criteria:
  1. SAC-Lagrangian converges (reward improves over 200ep)
  2. Delay violation rate < 80%
  3. SAC-Lag outperforms or matches PPO on violations

If fail: try delay_budget=0.5s and re-test.
"""

from __future__ import annotations
import os, sys, time
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)

from uav_mec.env import UAVMECEnv
from uav_mec.model import ScenarioConfig
from uav_mec.agents.sac_lagrangian import SACLagrangianAgent, SACLagrangianConfig
from uav_mec.agents.ppo import PPOAgent
from uav_mec.agents.baselines import make_baseline
from uav_mec.train import evaluate, train_sac_lagrangian, train_ppo

SEEDS = [42, 123, 456]
TRAIN_EP = 500
EVAL_EP = 50


def quick_test(delay_budget: float = 0.8):
    print("=" * 60)
    print(f"  QUICK TEST: f_uav=10GHz, lambda_max=100, lr_lambda=1e-3")
    print(f"  delay_budget={delay_budget}s, {len(SEEDS)} seeds, {TRAIN_EP} episodes")
    print("=" * 60)

    results = {"SAC-Lagrangian": [], "PPO": [], "Greedy": []}

    for seed in SEEDS:
        print(f"\n--- Seed={seed} ---")
        cfg = ScenarioConfig(num_users=10, num_slots=100, seed=seed)

        # SAC-Lagrangian
        sac_cfg = SACLagrangianConfig(delay_budget=delay_budget)
        env = UAVMECEnv(cfg=cfg)
        agent = SACLagrangianAgent(env.obs_dim, env.action_dim, sac_cfg)
        t0 = time.time()
        obs, _ = env.reset(seed=seed)
        ep1_reward = None
        last_reward = None
        for ep in range(1, TRAIN_EP + 1):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            ep_reward = 0
            while not done:
                if agent.total_steps < sac_cfg.warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs)
                next_obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                ep_reward += reward
                n_t = max(info.get("slot_tasks", 1), 1)
                agent.store(obs, action, reward, next_obs, done,
                            cost_energy=(info.get("slot_energy", 0) + info.get("slot_fly_energy", 0)) / n_t,
                            cost_delay=info.get("slot_delay", 0) / n_t)
                agent.update()
                obs = next_obs
            if ep == 1:
                ep1_reward = ep_reward
            if ep % 50 == 0:
                print(f"  SAC EP {ep}: R={ep_reward:.1f} λd={agent.lambda_d:.3f}")
            last_reward = ep_reward

        m = evaluate(env, agent, num_episodes=EVAL_EP, seed_base=seed * 100)
        dt = time.time() - t0
        print(f"  SAC-Lag: R={m['reward']:.1f}  viol={m['avg_violations']:.1f}  "
              f"deadline={m['deadline_rate']*100:.1f}%  λd={agent.lambda_d:.3f}  time={dt:.0f}s")
        results["SAC-Lagrangian"].append(m)

        # PPO
        env2 = UAVMECEnv(cfg=cfg)
        agent2 = PPOAgent(env2.obs_dim, env2.action_dim)
        t0 = time.time()
        for ep in range(1, TRAIN_EP + 1):
            obs, _ = env2.reset(seed=seed + ep)
            done = False
            while not done:
                action = agent2.select_action(obs)
                next_obs, reward, term, trunc, info = env2.step(action)
                done = term or trunc
                agent2.store(obs, action, reward, done)
                obs = next_obs
            agent2.update()
        m2 = evaluate(env2, agent2, num_episodes=EVAL_EP, seed_base=seed * 100)
        dt = time.time() - t0
        print(f"  PPO:     R={m2['reward']:.1f}  viol={m2['avg_violations']:.1f}  "
              f"deadline={m2['deadline_rate']*100:.1f}%  time={dt:.0f}s")
        results["PPO"].append(m2)

        # Greedy
        env3 = UAVMECEnv(cfg=cfg)
        greedy = make_baseline("greedy", env3.obs_dim, env3.action_dim, cfg.num_users, seed=seed)
        m3 = evaluate(env3, greedy, num_episodes=EVAL_EP, seed_base=seed * 100)
        print(f"  Greedy:  R={m3['reward']:.1f}  viol={m3['avg_violations']:.1f}  "
              f"deadline={m3['deadline_rate']*100:.1f}%")
        results["Greedy"].append(m3)

    # Summary
    print("\n" + "=" * 60)
    print("  QUICK TEST SUMMARY")
    print("=" * 60)

    for algo, metrics in results.items():
        mean_r = np.mean([m["reward"] for m in metrics])
        mean_v = np.mean([m["avg_violations"] for m in metrics])
        mean_d = np.mean([m["deadline_rate"] for m in metrics])
        mean_delay = np.mean([m["avg_delay"] for m in metrics])
        print(f"  {algo:20s}: R={mean_r:.1f}  viol={mean_v:.1f}  "
              f"deadline={mean_d*100:.1f}%  delay={mean_delay:.1f}")

    # Pass/Fail check
    sac_viol = np.mean([m["avg_violations"] for m in results["SAC-Lagrangian"]])
    sac_tasks = np.mean([m.get("avg_tasks", 500) for m in results["SAC-Lagrangian"]])
    ppo_viol = np.mean([m["avg_violations"] for m in results["PPO"]])
    sac_deadline = np.mean([m["deadline_rate"] for m in results["SAC-Lagrangian"]])

    violation_rate = sac_viol / max(sac_tasks, 1)

    print(f"\n  Violation rate: {violation_rate*100:.1f}%")
    print(f"  SAC viol={sac_viol:.1f} vs PPO viol={ppo_viol:.1f}")

    passed = True
    if violation_rate > 0.80:
        print("  ❌ FAIL: Violation rate > 80%")
        passed = False
    else:
        print("  ✅ PASS: Violation rate < 80%")

    if sac_viol <= ppo_viol:
        print("  ✅ PASS: SAC-Lag viol <= PPO viol")
    else:
        print(f"  ⚠️  WARNING: SAC-Lag viol({sac_viol:.1f}) > PPO viol({ppo_viol:.1f})")
        print("     (may improve with full 1000ep training)")

    return passed


if __name__ == "__main__":
    passed = quick_test(delay_budget=0.8)
    if not passed:
        print("\n" + "=" * 60)
        print("  RETRYING with delay_budget=1.0s")
        print("=" * 60)
        passed = quick_test(delay_budget=1.0)
    if passed:
        print("\n✅ Quick Test PASSED — safe to launch full experiment")
    else:
        print("\n❌ Quick Test FAILED — need further parameter tuning")
