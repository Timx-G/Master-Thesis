"""
K-scalability experiment: K=2,3,4,5 UAVs with nc=K clusters each.
Runs Ours (BG+Swap) vs Greedy, 20 seeds.

Usage: python -m uav_mec.run_k_scalability
"""
import time, json, math
import numpy as np
from pathlib import Path

from .model import ScenarioConfig, UAVModel, NOMAChannel, UAV_HOVER_POSITIONS
from .multi_uav_env import MultiUAVMECEnv
from .pairing import (greedy_pairing, benefit_greedy_pairing, swap_optimize,
                       evaluate_pairing)
from .baselines_pairing import exchange_stable_pairing, ilp_optimal_pairing


def run_episode_k(cfg, seed, strategy_fn):
    """Run one episode for general K UAVs."""
    env = MultiUAVMECEnv(cfg)
    env.reset(seed=seed)
    ch = env.channel
    K = cfg.num_uavs
    uav_pos = [uav.position for uav in env.uavs]
    N = cfg.num_users

    total_met = 0
    total_tasks = 0

    for slot in range(cfg.num_slots):
        if cfg.user_mobility:
            for i in range(N):
                angle = env.rng.uniform(0, 2 * math.pi)
                step_d = cfg.user_speed * cfg.uav.dt
                env.user_positions[i, 0] += step_d * math.cos(angle)
                env.user_positions[i, 1] += step_d * math.sin(angle)
            env.user_positions = np.clip(env.user_positions, 0, cfg.area_size)

        # Nearest-UAV assignment (general K)
        assignment = np.zeros(N, dtype=int)
        for i in range(N):
            dists = [np.linalg.norm(np.array(uav_pos[k][:2]) - env.user_positions[i])
                     for k in range(K)]
            assignment[i] = int(np.argmin(dists))

        # Channel gains (general K)
        h_gains = np.zeros((N, K))
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            for k in range(K):
                h_gains[i, k] = ch.path_loss(
                    np.linalg.norm(np.array(uav_pos[k][:2]) - env.user_positions[i]))

        # Run strategy
        pairs = strategy_fn(h_gains, env.task_queue_active, assignment, env, ch)

        # Evaluate
        met, total, rates, alphas, served = evaluate_pairing(
            pairs, h_gains, ch, env.task_queue_data, env.task_queue_comp,
            env.task_queue_active, assignment, cfg.n_subcarriers,
            cfg.f_per_uav, cfg.task.f_local, cfg.task.t_max)

        total_met += met
        total_tasks += total

        # Task queue update
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            if served[i] and rates[i] > 1.0:
                alpha = alphas[i]
                n_off = max(sum(1 for j in range(N) if served[j]
                               and assignment[j] == assignment[i]
                               and rates[j] > 1.0), 1)
                f_alloc = max(cfg.f_per_uav / n_off, 1e3)
                t_loc = (1 - alpha) * env.task_queue_comp[i] / cfg.task.f_local
                t_off = (alpha * env.task_queue_data[i] / rates[i] +
                         alpha * env.task_queue_comp[i] / f_alloc +
                         cfg.task.result_ratio * alpha * env.task_queue_data[i] / rates[i]
                         ) if alpha > 0 else 0
                delay = max(t_loc, t_off)
            else:
                delay = env.task_queue_comp[i] / cfg.task.f_local
            if delay <= cfg.task.t_max:
                env.task_queue_active[i] = False
                env.task_queue_age[i] = 0
            else:
                env.task_queue_age[i] += 1
                if env.task_queue_age[i] >= cfg.max_queue_age:
                    env.task_queue_active[i] = False
                    env.task_queue_age[i] = 0

        env.slot += 1
        if env.slot < cfg.num_slots:
            new_tasks = env.task_gen.generate(N)
            nc = cfg.num_clusters
            if cfg.random_burst:
                if env.slot >= env.next_burst_switch:
                    env.burst_cluster = env.rng.integers(0, nc)
                    env.next_burst_switch = env.slot + env.rng.integers(
                        cfg.burst_period_min, cfg.burst_period_max + 1)
                for i in range(N):
                    cid = env.user_cluster_ids[i]
                    prob = (cfg.burst_rate if cid == env.burst_cluster
                            else cfg.cool_rate)
                    new_tasks["has_task"][i] = env.rng.random() < prob
                    if not new_tasks["has_task"][i]:
                        new_tasks["data_bits"][i] = 0
                        new_tasks["comp_cycles"][i] = 0
            for i in range(N):
                if new_tasks["has_task"][i] and not env.task_queue_active[i]:
                    env.task_queue_data[i] = new_tasks["data_bits"][i]
                    env.task_queue_comp[i] = new_tasks["comp_cycles"][i]
                    env.task_queue_age[i] = 0
                    env.task_queue_active[i] = True

    return total_met / max(total_tasks, 1) * 100


def main():
    print("=" * 60)
    print("K-Scalability Experiment: K=2,3,4,5 (nc=K)")
    print("=" * 60)

    seeds = list(range(42, 42 + 20))
    results = {}

    for K in [1, 2, 3, 4, 5]:
        nc = max(K, 3)  # K=1,2 use 3 clusters; K>=3 use K clusters
        cfg = ScenarioConfig(
            num_uavs=K,
            num_clusters=nc,
            fixed_cluster_centers=True,
            random_burst=True,
            user_mobility=True,
            f_per_uav=8e9 / K,  # total MEC compute = 8GHz, split across K UAVs
        )
        print(f"\n  K={K}, nc={nc}, f_per_uav={cfg.f_per_uav/1e9:.1f}GHz")

        def make_strategy_ours(cfg_):
            def fn(h, active, assign, env, ch):
                hp = ilp_optimal_pairing(h, active, assign, cfg_.n_subcarriers,
                                            channel=ch,
                                            task_data=env.task_queue_data,
                                            task_comp=env.task_queue_comp,
                                            f_per_uav=cfg_.f_per_uav,
                                            f_local=cfg_.task.f_local,
                                            t_max=cfg_.task.t_max)
                pairs, _, _ = swap_optimize(hp, h, ch, active, assign,
                                             env.task_queue_data, env.task_queue_comp,
                                             cfg_.n_subcarriers, cfg_.f_per_uav,
                                             cfg_.task.f_local, cfg_.task.t_max)
                return pairs
            return fn

        def make_strategy_greedy(cfg_):
            def fn(h, active, assign, env, ch):
                return greedy_pairing(h, active, assign, cfg_.n_subcarriers)
            return fn

        def make_strategy_es(cfg_):
            def fn(h, active, assign, env, ch):
                return exchange_stable_pairing(h, active, assign, cfg_.n_subcarriers,
                                               channel=ch,
                                               task_data=env.task_queue_data,
                                               task_comp=env.task_queue_comp,
                                               f_per_uav=cfg_.f_per_uav,
                                               f_local=cfg_.task.f_local,
                                               t_max=cfg_.task.t_max)
            return fn

        ours_dl, greedy_dl, es_dl = [], [], []
        for seed in seeds:
            ours_dl.append(run_episode_k(cfg, seed, make_strategy_ours(cfg)))
            greedy_dl.append(run_episode_k(cfg, seed, make_strategy_greedy(cfg)))
            es_dl.append(run_episode_k(cfg, seed, make_strategy_es(cfg)))

        gap_g = np.mean(ours_dl) - np.mean(greedy_dl)
        gap_es = np.mean(ours_dl) - np.mean(es_dl)
        print(f"    Ours={np.mean(ours_dl):.1f}±{np.std(ours_dl):.1f}%  "
              f"Greedy={np.mean(greedy_dl):.1f}%  "
              f"ES={np.mean(es_dl):.1f}%  "
              f"gap_G={gap_g:+.1f}pp  gap_ES={gap_es:+.1f}pp")

        results[str(K)] = {
            "ours_mean": round(np.mean(ours_dl), 2),
            "ours_std": round(np.std(ours_dl), 2),
            "greedy_mean": round(np.mean(greedy_dl), 2),
            "es_mean": round(np.mean(es_dl), 2),
            "gap_greedy": round(gap_g, 2),
            "gap_es": round(gap_es, 2),
        }

    # Save
    out = Path(__file__).parent.parent / "results" / "k_scalability_full.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
