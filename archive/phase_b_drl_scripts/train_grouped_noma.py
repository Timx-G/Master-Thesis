"""
3-seed PPO training for Grouped NOMA Multi-UAV environment.

Gate check: DRL(dynamic) vs BCD-static(center hover) > 5pp in non-stationary burst.

Usage:
    python -m uav_mec.train_grouped_noma
"""
import time
import json
import numpy as np
from pathlib import Path

from .model import ScenarioConfig, TaskConfig
from .multi_uav_env import MultiUAVMECEnv
from .agents.ppo import PPOAgent
from .baselines_multi import GreedyGrouping, RandomGrouping, FixedBestGrouping


def evaluate(env_cfg, agent, n_episodes=10, seeds=None, deterministic=True):
    """Evaluate agent over n_episodes, return mean deadline rate."""
    deadlines = []
    for ep in range(n_episodes):
        seed = seeds[ep] if seeds is not None else 9999 + ep
        env = MultiUAVMECEnv(env_cfg)
        obs, _ = env.reset(seed=seed)
        for s in range(env_cfg.num_slots):
            action = agent.select_action(obs, deterministic=deterministic)
            obs, r, done, _, info = env.step(action)
            if done:
                break
        t = info["episode_tasks_served"]
        v = info["episode_deadline_violations"]
        deadlines.append((t - v) / max(t, 1) * 100)
    return np.mean(deadlines), np.std(deadlines)


def evaluate_baseline(env_cfg, baseline_cls, n_episodes=10, seeds=None, **kwargs):
    """Evaluate a heuristic baseline."""
    deadlines = []
    for ep in range(n_episodes):
        seed = seeds[ep] if seeds is not None else 9999 + ep
        env = MultiUAVMECEnv(env_cfg)
        agent = baseline_cls(env_cfg, **kwargs) if kwargs else baseline_cls(env_cfg)
        obs, _ = env.reset(seed=seed)
        for s in range(env_cfg.num_slots):
            action = agent.get_action(env)
            obs, r, done, _, info = env.step(action)
            if done:
                break
        t = info["episode_tasks_served"]
        v = info["episode_deadline_violations"]
        deadlines.append((t - v) / max(t, 1) * 100)
    return np.mean(deadlines), np.std(deadlines)


def train_one_seed(seed, cfg, n_episodes=2000, eval_interval=100):
    """Train PPO for one seed, return training curve and final performance."""
    env = MultiUAVMECEnv(cfg)
    obs, _ = env.reset(seed=seed)

    agent = PPOAgent(
        obs_dim=env.obs_dim,
        act_dim=env.action_dim,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=4,
        hidden=256,
        mini_batch=64,
        seed=seed,
    )

    eval_seeds = list(range(5000, 5010))
    curve = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep * 1000)
        ep_reward = 0.0

        for step in range(cfg.num_slots):
            action = agent.select_action(obs)
            next_obs, r, done, _, info = env.step(action)
            agent.store(obs, action, r, done)
            obs = next_obs
            ep_reward += r
            if done:
                break

        agent.update()

        # Periodic evaluation
        if (ep + 1) % eval_interval == 0:
            dl_mean, dl_std = evaluate(cfg, agent, n_episodes=10, seeds=eval_seeds)
            curve.append({
                "episode": ep + 1,
                "deadline_mean": round(dl_mean, 2),
                "deadline_std": round(dl_std, 2),
                "train_reward": round(ep_reward, 2),
            })
            print(f"  Seed {seed} | EP {ep+1:4d} | "
                  f"eval={dl_mean:.1f}% ± {dl_std:.1f}% | "
                  f"train_r={ep_reward:.1f}")

    # Final evaluation with more episodes
    final_mean, final_std = evaluate(cfg, agent, n_episodes=20, seeds=list(range(8000, 8020)))
    return agent, curve, final_mean, final_std


def main():
    print("=" * 70)
    print("Dynamic NOMA Grouping: 3-seed PPO Training")
    print("=" * 70)

    cfg = ScenarioConfig(
        fixed_cluster_centers=True,
        random_burst=True,
        user_mobility=True,
    )
    print(f"Config: t_max={cfg.task.t_max}s, n_sub={cfg.n_subcarriers}, "
          f"arrival_prob={cfg.task.arrival_prob}, BW={cfg.channel.bandwidth/1e6}MHz")
    print(f"Max served/UAV: {cfg.n_subcarriers * cfg.users_per_sub}")
    print()

    # Baselines
    eval_seeds = list(range(8000, 8020))
    print("--- Baselines (20-episode eval) ---")

    greedy_m, greedy_s = evaluate_baseline(
        cfg, GreedyGrouping, n_episodes=20, seeds=eval_seeds)
    print(f"  Greedy (channel-sort + alpha=0.95): {greedy_m:.1f}% ± {greedy_s:.1f}%")

    fixed_m, fixed_s = evaluate_baseline(
        cfg, FixedBestGrouping, n_episodes=20, seeds=eval_seeds)
    print(f"  FixedBest (equal weights + alpha=0.95): {fixed_m:.1f}% ± {fixed_s:.1f}%")

    random_m, random_s = evaluate_baseline(
        cfg, RandomGrouping, n_episodes=20, seeds=eval_seeds,
        rng=np.random.default_rng(42))
    print(f"  Random: {random_m:.1f}% ± {random_s:.1f}%")
    print()

    # Train 3 seeds
    seeds = [42, 123, 456]
    n_episodes = 2000
    all_results = {}

    start = time.time()
    for seed in seeds:
        print(f"--- Training seed={seed}, {n_episodes} episodes ---")
        agent, curve, final_m, final_s = train_one_seed(
            seed, cfg, n_episodes=n_episodes, eval_interval=200)
        all_results[seed] = {
            "curve": curve,
            "final_mean": round(final_m, 2),
            "final_std": round(final_s, 2),
        }
        print(f"  FINAL: {final_m:.1f}% ± {final_s:.1f}%")
        print()

    elapsed = time.time() - start

    # Summary
    print("=" * 70)
    print("3-Seed Summary")
    print("=" * 70)
    finals = [all_results[s]["final_mean"] for s in seeds]
    ppo_mean = np.mean(finals)
    ppo_std = np.std(finals)
    gap_greedy = ppo_mean - greedy_m
    gap_random = ppo_mean - random_m

    print(f"  PPO (3-seed):  {ppo_mean:.1f}% ± {ppo_std:.1f}%  seeds={finals}")
    print(f"  Greedy:        {greedy_m:.1f}%")
    print(f"  FixedBest:     {fixed_m:.1f}%")
    print(f"  Random:        {random_m:.1f}%")
    print(f"  PPO vs Greedy: {gap_greedy:+.1f}pp")
    print(f"  PPO vs Random: {gap_random:+.1f}pp")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print()

    # Gate check: DRL vs Greedy ≥ 5pp
    gate_pass = gap_greedy >= 5.0
    print(f"  GATE: PPO vs Greedy >= 5pp? {gap_greedy:+.1f}pp -> {'PASS' if gate_pass else 'FAIL'}")

    # Save
    archive = {
        "phase": "dynamic_grouping_drl",
        "date": time.strftime("%Y-%m-%d"),
        "config": {"t_max": cfg.task.t_max, "n_sub": cfg.n_subcarriers,
                    "arrival_prob": cfg.task.arrival_prob, "user_mobility": True,
                    "random_burst": True},
        "baselines": {"greedy": float(greedy_m), "fixed_best": float(fixed_m),
                      "random": float(random_m)},
        "ppo_3seed_mean": float(ppo_mean),
        "ppo_seeds": {int(s): float(all_results[s]["final_mean"]) for s in seeds},
        "gap_vs_greedy": float(gap_greedy),
        "gate_pass": bool(gate_pass),
        "elapsed_seconds": round(elapsed, 1),
    }
    out_path = Path(__file__).parent.parent / "results" / "archive_dynamic_grouping_drl.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(archive, f, indent=2)
    print(f"  Archive: {out_path}")


if __name__ == "__main__":
    main()
