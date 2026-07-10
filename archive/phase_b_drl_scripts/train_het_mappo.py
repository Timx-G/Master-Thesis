"""
1-seed Het-MAPPO + MAPPO + Independent PPO comparison.
Gate: Independent PPO > Het-MAPPO → C3 confirmed.
"""
import time, json
import numpy as np
from pathlib import Path
from .model import ScenarioConfig, TaskConfig
from .multi_uav_env import MultiUAVMECEnv
from .agents.ppo import PPOAgent
from .agents.mappo import MAPPOAgent
from .agents.het_mappo import HetMAPPOAgent


def evaluate(cfg, agent, agent_type, n_ep=10, seeds=None):
    deadlines = []
    for ep in range(n_ep):
        seed = seeds[ep] if seeds else 9999 + ep
        env = MultiUAVMECEnv(cfg)
        obs, _ = env.reset(seed=seed)
        for s in range(cfg.num_slots):
            if agent_type == "ppo":
                action = agent.select_action(obs, deterministic=True)
            else:
                action = agent.select_actions(obs, deterministic=True)
            obs, r, done, _, info = env.step(action)
            if done:
                break
        t = info["episode_tasks_served"]
        v = info["episode_deadline_violations"]
        deadlines.append((t - v) / max(t, 1) * 100)
    return np.mean(deadlines), np.std(deadlines)


def train_ppo(seed, cfg, n_ep=2000, eval_interval=200):
    env = MultiUAVMECEnv(cfg)
    agent = PPOAgent(env.obs_dim, env.action_dim, lr=3e-4, gamma=0.99,
                     hidden=256, seed=seed)
    eval_seeds = list(range(5000, 5010))
    curve = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed + ep * 1000)
        for s in range(cfg.num_slots):
            action = agent.select_action(obs)
            next_obs, r, done, _, _ = env.step(action)
            agent.store(obs, action, r, done)
            obs = next_obs
            if done:
                break
        agent.update()
        if (ep + 1) % eval_interval == 0:
            m, s = evaluate(cfg, agent, "ppo", 10, eval_seeds)
            curve.append({"ep": ep+1, "dl": round(m, 2)})
            print(f"    PPO EP{ep+1:4d}: {m:.1f}%")
    fm, fs = evaluate(cfg, agent, "ppo", 20, list(range(8000, 8020)))
    return fm, fs, curve


def train_mappo(seed, cfg, agent_cls, label, n_ep=2000, eval_interval=200):
    env = MultiUAVMECEnv(cfg)
    agent = agent_cls(
        agent_obs_dim=env.agent_obs_dim,
        agent_act_dim=env.agent_action_dim,
        global_obs_dim=env.obs_dim,
        num_agents=cfg.num_uavs,
        seed=seed)
    eval_seeds = list(range(5000, 5010))
    curve = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed + ep * 1000)
        for s in range(cfg.num_slots):
            action = agent.select_actions(obs)
            next_obs, r, done, _, _ = env.step(action)
            agent.store(obs, action, r, done)
            obs = next_obs
            if done:
                break
        agent.update()
        if (ep + 1) % eval_interval == 0:
            m, s = evaluate(cfg, agent, "mappo", 10, eval_seeds)
            curve.append({"ep": ep+1, "dl": round(m, 2)})
            print(f"    {label} EP{ep+1:4d}: {m:.1f}%")
    fm, fs = evaluate(cfg, agent, "mappo", 20, list(range(8000, 8020)))
    return fm, fs, curve


def main():
    print("=" * 60)
    print("C3 Validation: Independent PPO vs Het-MAPPO vs MAPPO")
    print("=" * 60)
    cfg = ScenarioConfig(
        fixed_cluster_centers=True, random_burst=True, user_mobility=True,
        task=TaskConfig(t_max=0.7))
    seed = 42

    print(f"\n--- Independent PPO (seed={seed}) ---")
    ppo_m, ppo_s, ppo_curve = train_ppo(seed, cfg)
    print(f"  FINAL: {ppo_m:.1f}% ± {ppo_s:.1f}%")

    print(f"\n--- Het-MAPPO (seed={seed}) ---")
    het_m, het_s, het_curve = train_mappo(seed, cfg, HetMAPPOAgent, "Het-MAPPO")
    print(f"  FINAL: {het_m:.1f}% ± {het_s:.1f}%")

    print(f"\n--- MAPPO param-sharing (seed={seed}) ---")
    mappo_m, mappo_s, mappo_curve = train_mappo(seed, cfg, MAPPOAgent, "MAPPO")
    print(f"  FINAL: {mappo_m:.1f}% ± {mappo_s:.1f}%")

    print("\n" + "=" * 60)
    print("C3 Result")
    print("=" * 60)
    print(f"  Independent PPO: {ppo_m:.1f}%")
    print(f"  Het-MAPPO:       {het_m:.1f}%")
    print(f"  MAPPO:           {mappo_m:.1f}%")
    print(f"  PPO vs Het-MAPPO: {ppo_m - het_m:+.1f}pp")
    print(f"  PPO vs MAPPO:     {ppo_m - mappo_m:+.1f}pp")
    c3 = ppo_m > het_m
    print(f"  C3 (Independent > Het-MAPPO): {'CONFIRMED' if c3 else 'REJECTED'}")

    archive = {
        "seed": seed, "date": time.strftime("%Y-%m-%d"),
        "ppo": {"final": round(ppo_m, 2), "curve": ppo_curve},
        "het_mappo": {"final": round(het_m, 2), "curve": het_curve},
        "mappo": {"final": round(mappo_m, 2), "curve": mappo_curve},
        "c3_confirmed": bool(c3),
    }
    out = Path(__file__).parent.parent / "results" / "archive_c3_validation.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(archive, f, indent=2)
    print(f"  Archive: {out}")


if __name__ == "__main__":
    main()
