"""
DC-PPO Quick Test: t_max=2.0s 参数验证

三条门禁（全部通过才能进全量实验）：
  1. DC-PPO deadline rate >= 88%  (N=10)
  2. lambda_d 最终收敛在 5~18 之间（不饱和，不为零）
  3. DC-PPO 比 PPO 高 >= 5pp deadline rate

不通过 -> 立即叫停，三博士介入
"""

from __future__ import annotations
import os, sys, time
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

SEEDS      = [42, 123, 456]
TRAIN_EP   = 200
EVAL_EP    = 30
NUM_USERS  = 10

# 门禁阈值
GATE_DEADLINE_MIN   = 0.88   # DC-PPO deadline >= 88%
GATE_LAMBDA_LOW     = 5.0    # λd 下界（不能为0，说明没起效）
GATE_LAMBDA_HIGH    = 18.0   # λd 上界（不能饱和到20）
GATE_MARGIN_MIN     = 0.05   # DC-PPO - PPO >= 5pp


def train_dc_ppo(env, seed, episodes):
    cfg = DCPPOConfig()
    agent = DCPPOAgent(env.obs_dim, env.action_dim, cfg, seed=seed)
    lambda_history = []

    for ep in range(1, episodes + 1):
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
        info_ep = agent.update()
        lambda_history.append(agent.lambda_d)

        if ep % 50 == 0:
            print(f"    DC-PPO EP {ep:3d}: λd={agent.lambda_d:.3f}")

    return agent, lambda_history


def train_ppo_baseline(env, seed, episodes):
    agent = PPOAgent(env.obs_dim, env.action_dim, seed=seed)
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            agent.store(obs, action, reward, done)
            obs = next_obs
        agent.update()
    return agent


def run():
    print("=" * 65)
    print("  DC-PPO QUICK TEST — t_max=2.0s 门禁验证")
    print(f"  N={NUM_USERS}, {TRAIN_EP}ep, {len(SEEDS)} seeds")
    print(f"  门禁: deadline>={GATE_DEADLINE_MIN*100:.0f}%  "
          f"λd∈[{GATE_LAMBDA_LOW},{GATE_LAMBDA_HIGH}]  margin>={GATE_MARGIN_MIN*100:.0f}pp")
    print("=" * 65)

    dc_results, ppo_results = [], []
    lambda_finals = []

    for seed in SEEDS:
        print(f"\n=== Seed={seed} ===")
        cfg = ScenarioConfig(num_users=NUM_USERS, num_slots=100, seed=seed)

        # DC-PPO
        env = UAVMECEnv(cfg=cfg)
        t0 = time.time()
        dc_agent, lam_hist = train_dc_ppo(env, seed, TRAIN_EP)
        m_dc = evaluate(env, dc_agent, num_episodes=EVAL_EP, seed_base=seed * 100)
        dt = time.time() - t0
        lambda_finals.append(dc_agent.lambda_d)
        dc_results.append(m_dc)
        print(f"  DC-PPO: deadline={m_dc['deadline_rate']*100:.1f}%  "
              f"viol={m_dc['avg_violations']:.1f}  λd={dc_agent.lambda_d:.3f}  [{dt:.0f}s]")

        # PPO baseline
        env2 = UAVMECEnv(cfg=cfg)
        t0 = time.time()
        ppo_agent = train_ppo_baseline(env2, seed, TRAIN_EP)
        m_ppo = evaluate(env2, ppo_agent, num_episodes=EVAL_EP, seed_base=seed * 100)
        dt = time.time() - t0
        ppo_results.append(m_ppo)
        print(f"  PPO:    deadline={m_ppo['deadline_rate']*100:.1f}%  "
              f"viol={m_ppo['avg_violations']:.1f}  [{dt:.0f}s]")

    # ---- Summary ----
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    dc_deadline  = np.mean([m["deadline_rate"] for m in dc_results])
    ppo_deadline = np.mean([m["deadline_rate"] for m in ppo_results])
    dc_viol      = np.mean([m["avg_violations"] for m in dc_results])
    ppo_viol     = np.mean([m["avg_violations"] for m in ppo_results])
    lam_mean     = np.mean(lambda_finals)
    margin       = dc_deadline - ppo_deadline

    print(f"  DC-PPO : deadline={dc_deadline*100:.1f}%  viol={dc_viol:.1f}  λd_mean={lam_mean:.3f}")
    print(f"  PPO    : deadline={ppo_deadline*100:.1f}%  viol={ppo_viol:.1f}")
    print(f"  Margin : {margin*100:+.1f}pp")

    # ---- 门禁检查 ----
    print("\n" + "=" * 65)
    print("  门禁检查")
    print("=" * 65)

    gate1 = dc_deadline >= GATE_DEADLINE_MIN
    gate2 = GATE_LAMBDA_LOW <= lam_mean <= GATE_LAMBDA_HIGH
    gate3 = margin >= GATE_MARGIN_MIN

    print(f"  [{'PASS' if gate1 else 'FAIL'}] 门禁1: DC-PPO deadline {dc_deadline*100:.1f}% >= {GATE_DEADLINE_MIN*100:.0f}%")
    print(f"  [{'PASS' if gate2 else 'FAIL'}] 门禁2: λd={lam_mean:.3f} ∈ [{GATE_LAMBDA_LOW}, {GATE_LAMBDA_HIGH}]")
    print(f"  [{'PASS' if gate3 else 'FAIL'}] 门禁3: margin={margin*100:+.1f}pp >= {GATE_MARGIN_MIN*100:.0f}pp")

    all_pass = gate1 and gate2 and gate3
    print()
    if all_pass:
        print("  ✅ 全部通过 → 可以启动全量实验")
    else:
        print("  ❌ 门禁未通过 → 叫停，三博士介入")

    return all_pass, {
        "dc_deadline": dc_deadline, "ppo_deadline": ppo_deadline,
        "margin": margin, "lambda_mean": lam_mean,
        "gate1": gate1, "gate2": gate2, "gate3": gate3,
    }


if __name__ == "__main__":
    passed, summary = run()
    sys.exit(0 if passed else 1)
