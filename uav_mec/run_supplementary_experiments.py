"""
Supplementary experiments for reviewer responses:
  1. N-scalability: N=15, N=25 (fill gaps in scalability table)
  2. Imperfect CSI: sigma_e in {0.0, 0.01, 0.05, 0.10}
  3. SIC residual factor: epsilon in {0.0, 0.01, 0.02, 0.05, 0.10}
  4. Swap iteration distribution (histogram data)
  5. Energy consumption comparison

Usage: python -m uav_mec.run_supplementary_experiments
"""
import time, json, math, copy
import numpy as np
from pathlib import Path
from scipy import stats
from dataclasses import replace

from .model import ScenarioConfig, TaskConfig, NOMAChannel
from .multi_uav_env import MultiUAVMECEnv
from .pairing import (greedy_pairing, benefit_greedy_pairing, swap_optimize,
                       evaluate_pairing, compute_rates_for_pairing, kkt_alpha)
from .baselines_pairing import (random_pairing, ilp_optimal_pairing,
                                 exchange_stable_pairing)
from .run_main_experiment import run_episode, strategy_ours, strategy_greedy


# ====================================================================
# Experiment 1: N-scalability (fill N=15, N=25 gaps)
# ====================================================================

def run_n_scalability():
    """Run N=10,15,20,25,30 with Ours vs Greedy, 20 seeds each."""
    print("\n" + "=" * 60)
    print("Experiment 1: N-Scalability")
    print("=" * 60)

    seeds = list(range(42, 42 + 20))
    results = {}

    for N in [10, 15, 20, 25, 30]:
        # Create config with modified N
        cfg = ScenarioConfig(
            num_users=N,
            fixed_cluster_centers=True,
            random_burst=True,
            user_mobility=True,
        )
        # Adjust n_subcarriers for different N
        # Keep S=3 for all N to isolate the effect of N
        print(f"\n  N={N}, S={cfg.n_subcarriers}, seeds={len(seeds)}")

        ours_dl = []
        greedy_dl = []
        exchange_dl = []

        for seed in seeds:
            m_ours = run_episode(cfg, seed, strategy_ours)
            m_greedy = run_episode(cfg, seed, strategy_greedy)

            def strategy_es(h, active, assign, env, ch):
                return exchange_stable_pairing(
                    h, active, assign, cfg.n_subcarriers, channel=ch,
                    task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                    f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                    t_max=cfg.task.t_max)

            m_es = run_episode(cfg, seed, strategy_es)

            ours_dl.append(m_ours["deadline_rate"])
            greedy_dl.append(m_greedy["deadline_rate"])
            exchange_dl.append(m_es["deadline_rate"])

        gap_g = np.mean(ours_dl) - np.mean(greedy_dl)
        gap_es = np.mean(ours_dl) - np.mean(exchange_dl)
        print(f"    Ours={np.mean(ours_dl):.1f}±{np.std(ours_dl):.1f}%  "
              f"Greedy={np.mean(greedy_dl):.1f}%  "
              f"ExStable={np.mean(exchange_dl):.1f}%  "
              f"gap_G={gap_g:+.1f}pp  gap_ES={gap_es:+.1f}pp")

        results[N] = {
            "ours_mean": round(np.mean(ours_dl), 2),
            "ours_std": round(np.std(ours_dl), 2),
            "greedy_mean": round(np.mean(greedy_dl), 2),
            "greedy_std": round(np.std(greedy_dl), 2),
            "exchange_mean": round(np.mean(exchange_dl), 2),
            "exchange_std": round(np.std(exchange_dl), 2),
            "gap_greedy": round(gap_g, 2),
            "gap_exchange": round(gap_es, 2),
            "ours_seeds": [round(x, 2) for x in ours_dl],
            "greedy_seeds": [round(x, 2) for x in greedy_dl],
        }

    return results


# ====================================================================
# Experiment 2: Imperfect CSI
# ====================================================================

def run_episode_with_csi_error(cfg, seed, strategy_fn, sigma_e):
    """Run episode with CSI estimation error added to channel gains."""
    env = MultiUAVMECEnv(cfg)
    env.reset(seed=seed)
    ch = env.channel
    uav_pos = [uav.position for uav in env.uavs]
    N = cfg.num_users
    rng = np.random.default_rng(seed + 10000)  # separate rng for CSI error

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

        assignment = np.zeros(N, dtype=int)
        for i in range(N):
            d0 = np.linalg.norm(np.array(uav_pos[0][:2]) - env.user_positions[i])
            d1 = np.linalg.norm(np.array(uav_pos[1][:2]) - env.user_positions[i])
            assignment[i] = 0 if d0 <= d1 else 1

        # True channel gains
        h_true = np.zeros((N, 2))
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            for k in range(2):
                h_true[i, k] = ch.path_loss(
                    np.linalg.norm(np.array(uav_pos[k][:2]) - env.user_positions[i]))

        # Estimated channel gains (with error)
        if sigma_e > 0:
            noise = rng.normal(0, sigma_e, size=h_true.shape) * h_true
            h_est = np.maximum(h_true + noise, 1e-15)
        else:
            h_est = h_true.copy()

        # Pairing uses estimated CSI
        pairs = strategy_fn(h_est, env.task_queue_active, assignment, env, ch)

        # Evaluation uses TRUE CSI (actual performance)
        met, total, rates, alphas, served = evaluate_pairing(
            pairs, h_true, ch, env.task_queue_data, env.task_queue_comp,
            env.task_queue_active, assignment, cfg.n_subcarriers,
            cfg.f_per_uav, cfg.task.f_local, cfg.task.t_max)

        total_met += met
        total_tasks += total

        # Task queue update (simplified)
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


def run_csi_experiment():
    """Sweep CSI error sigma_e."""
    print("\n" + "=" * 60)
    print("Experiment 2: Imperfect CSI Robustness")
    print("=" * 60)

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                          user_mobility=True)
    seeds = list(range(42, 42 + 20))
    results = {}

    for sigma_e in [0.0, 0.01, 0.05, 0.10, 0.20]:
        ours_dl = []
        greedy_dl = []
        for seed in seeds:
            ours_dl.append(run_episode_with_csi_error(cfg, seed, strategy_ours, sigma_e))
            greedy_dl.append(run_episode_with_csi_error(cfg, seed, strategy_greedy, sigma_e))

        gap = np.mean(ours_dl) - np.mean(greedy_dl)
        print(f"  σ_e={sigma_e:.2f}: Ours={np.mean(ours_dl):.1f}±{np.std(ours_dl):.1f}%  "
              f"Greedy={np.mean(greedy_dl):.1f}%  gap={gap:+.1f}pp")

        results[str(sigma_e)] = {
            "ours_mean": round(np.mean(ours_dl), 2),
            "ours_std": round(np.std(ours_dl), 2),
            "greedy_mean": round(np.mean(greedy_dl), 2),
            "gap": round(gap, 2),
        }

    return results


# ====================================================================
# Experiment 3: SIC residual factor sensitivity
# ====================================================================

def _run_episode_sic(cfg, seed, strategy_fn, sic_eps):
    """Run independent episode with custom sic_eps."""
    env = MultiUAVMECEnv(cfg)
    env.reset(seed=seed)
    ch = env.channel
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

        assignment = np.zeros(N, dtype=int)
        for i in range(N):
            d0 = np.linalg.norm(np.array(uav_pos[0][:2]) - env.user_positions[i])
            d1 = np.linalg.norm(np.array(uav_pos[1][:2]) - env.user_positions[i])
            assignment[i] = 0 if d0 <= d1 else 1

        h_gains = np.zeros((N, 2))
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            for k in range(2):
                h_gains[i, k] = ch.path_loss(
                    np.linalg.norm(np.array(uav_pos[k][:2]) - env.user_positions[i]))

        pairs = strategy_fn(h_gains, env.task_queue_active, assignment, env, ch,
                            sic_eps)

        met, total, rates, alphas, served = evaluate_pairing(
            pairs, h_gains, ch, env.task_queue_data, env.task_queue_comp,
            env.task_queue_active, assignment, cfg.n_subcarriers,
            cfg.f_per_uav, cfg.task.f_local, cfg.task.t_max, sic_eps=sic_eps)

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
                t_off = ((1+cfg.task.result_ratio) * alpha * env.task_queue_data[i] / rates[i] +
                         alpha * env.task_queue_comp[i] / f_alloc
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


def run_sic_experiment():
    """Sweep SIC residual factor epsilon. Each method runs independently."""
    print("\n" + "=" * 60)
    print("Experiment 3: SIC Residual Factor Sensitivity")
    print("=" * 60)

    seeds = list(range(42, 42 + 20))
    results = {}

    def sic_strategy_ours(h, active, assign, env, ch, eps):
        cfg = env.cfg
        hp = ilp_optimal_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                                 task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                                 f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                                 t_max=cfg.task.t_max, sic_eps=eps)
        pairs, _, _ = swap_optimize(hp, h, ch, active, assign,
                                     env.task_queue_data, env.task_queue_comp,
                                     cfg.n_subcarriers, cfg.f_per_uav,
                                     cfg.task.f_local, cfg.task.t_max, sic_eps=eps)
        return pairs

    def sic_strategy_greedy(h, active, assign, env, ch, eps):
        return greedy_pairing(h, active, assign, env.cfg.n_subcarriers)

    for eps in [0.0, 0.01, 0.02, 0.05, 0.10]:
        cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                              user_mobility=True)
        ours_dl = []
        greedy_dl = []

        for seed in seeds:
            # Run Ours and Greedy as INDEPENDENT episodes
            ours_dl.append(_run_episode_sic(cfg, seed, sic_strategy_ours, eps))
            greedy_dl.append(_run_episode_sic(cfg, seed, sic_strategy_greedy, eps))

        gap = np.mean(ours_dl) - np.mean(greedy_dl)
        print(f"  ε={eps:.2f}: Ours={np.mean(ours_dl):.1f}±{np.std(ours_dl):.1f}%  "
              f"Greedy={np.mean(greedy_dl):.1f}%  gap={gap:+.1f}pp")

        results[str(eps)] = {
            "ours_mean": round(np.mean(ours_dl), 2),
            "ours_std": round(np.std(ours_dl), 2),
            "greedy_mean": round(np.mean(greedy_dl), 2),
            "gap": round(gap, 2),
        }

    return results


# ====================================================================
# Experiment 4: Swap iteration distribution
# ====================================================================

def run_swap_iterations():
    """Collect swap iteration counts across all seeds and slots."""
    print("\n" + "=" * 60)
    print("Experiment 4: Swap Iteration Distribution")
    print("=" * 60)

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                          user_mobility=True)
    seeds = list(range(42, 42 + 20))
    all_iters = []

    for seed in seeds:
        env = MultiUAVMECEnv(cfg)
        env.reset(seed=seed)
        ch = env.channel
        uav_pos = [uav.position for uav in env.uavs]
        N = cfg.num_users

        for slot in range(cfg.num_slots):
            if cfg.user_mobility:
                for i in range(N):
                    angle = env.rng.uniform(0, 2 * math.pi)
                    step_d = cfg.user_speed * cfg.uav.dt
                    env.user_positions[i, 0] += step_d * math.cos(angle)
                    env.user_positions[i, 1] += step_d * math.sin(angle)
                env.user_positions = np.clip(env.user_positions, 0, cfg.area_size)

            assignment = np.zeros(N, dtype=int)
            for i in range(N):
                d0 = np.linalg.norm(np.array(uav_pos[0][:2]) - env.user_positions[i])
                d1 = np.linalg.norm(np.array(uav_pos[1][:2]) - env.user_positions[i])
                assignment[i] = 0 if d0 <= d1 else 1

            h_gains = np.zeros((N, 2))
            for i in range(N):
                if not env.task_queue_active[i]:
                    continue
                for k in range(2):
                    h_gains[i, k] = ch.path_loss(
                        np.linalg.norm(np.array(uav_pos[k][:2]) - env.user_positions[i]))

            hp = ilp_optimal_pairing(h_gains, env.task_queue_active, assignment,
                                     cfg.n_subcarriers, channel=ch,
                                     task_data=env.task_queue_data,
                                     task_comp=env.task_queue_comp,
                                     f_per_uav=cfg.f_per_uav,
                                        f_local=cfg.task.f_local,
                                        t_max=cfg.task.t_max)
            pairs_sw, _, n_iters = swap_optimize(hp, h_gains, ch, env.task_queue_active,
                                           assignment, env.task_queue_data,
                                           env.task_queue_comp, cfg.n_subcarriers,
                                           cfg.f_per_uav, cfg.task.f_local,
                                           cfg.task.t_max)
            all_iters.append(n_iters)

            # Task queue update (correct: check actual delay)
            _, _, rates_sw, alphas_sw, served_sw = evaluate_pairing(
                pairs_sw, h_gains, ch, env.task_queue_data, env.task_queue_comp,
                env.task_queue_active, assignment, cfg.n_subcarriers,
                cfg.f_per_uav, cfg.task.f_local, cfg.task.t_max)
            for i in range(N):
                if not env.task_queue_active[i]:
                    continue
                if served_sw[i] and rates_sw[i] > 1.0:
                    alpha = alphas_sw[i]
                    n_off = max(sum(1 for j in range(N) if served_sw[j]
                                   and assignment[j] == assignment[i]
                                   and rates_sw[j] > 1.0), 1)
                    f_alloc = max(cfg.f_per_uav / n_off, 1e3)
                    t_loc = (1 - alpha) * env.task_queue_comp[i] / cfg.task.f_local
                    t_off = (alpha * env.task_queue_data[i] / rates_sw[i] +
                             alpha * env.task_queue_comp[i] / f_alloc +
                             cfg.task.result_ratio * alpha * env.task_queue_data[i] / rates_sw[i]
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

    iters = np.array(all_iters)
    print(f"  Total samples: {len(iters)}")
    print(f"  Mean: {iters.mean():.2f}, Median: {np.median(iters):.1f}, "
          f"Max: {iters.max()}, Min: {iters.min()}")
    hist, edges = np.histogram(iters, bins=range(0, max(iters) + 2))
    print(f"  Distribution: {dict(zip(range(len(hist)), hist.tolist()))}")

    return {
        "mean": round(float(iters.mean()), 2),
        "median": round(float(np.median(iters)), 1),
        "max": int(iters.max()),
        "min": int(iters.min()),
        "std": round(float(iters.std()), 2),
        "histogram": {str(i): int(c) for i, c in enumerate(hist) if c > 0},
        "n_samples": len(iters),
    }


# ====================================================================
# Experiment 5: Subcarrier Sensitivity (S=2,3,4,5)
# ====================================================================

def run_subcarrier_sensitivity():
    """Sweep number of subcarriers S."""
    print("\n" + "=" * 60)
    print("Experiment 5: Subcarrier Sensitivity")
    print("=" * 60)

    seeds = list(range(42, 42 + 20))  # 20 seeds for consistency
    results = {}

    for S in [2, 3, 4, 5]:
        cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                              user_mobility=True, n_subcarriers=S)
        ours_dl = []
        greedy_dl = []
        for seed in seeds:
            m_o = run_episode(cfg, seed, strategy_ours)
            m_g = run_episode(cfg, seed, strategy_greedy)
            ours_dl.append(m_o["deadline_rate"])
            greedy_dl.append(m_g["deadline_rate"])

        gap = np.mean(ours_dl) - np.mean(greedy_dl)
        print(f"  S={S}: Ours={np.mean(ours_dl):.1f}±{np.std(ours_dl):.1f}%  "
              f"Greedy={np.mean(greedy_dl):.1f}±{np.std(greedy_dl):.1f}%  gap={gap:+.1f}pp")

        results[str(S)] = {
            "ours_mean": round(np.mean(ours_dl), 2),
            "ours_std": round(np.std(ours_dl), 2),
            "greedy_mean": round(np.mean(greedy_dl), 2),
            "greedy_std": round(np.std(greedy_dl), 2),
            "gap": round(gap, 2),
        }

    return results


# ====================================================================
# Experiment 6: Per-Slot Upper Bound (per-subcarrier exhaustive)
# ====================================================================

def run_upper_bound():
    """Compute per-slot upper bound by exhaustive per-subcarrier enumeration.

    Upper bound: for each subcarrier independently, enumerate ALL possible
    4-user configurations (2 per UAV) and pick the best. This relaxes the
    user-disjointness constraint across subcarriers, so it is a valid upper bound.
    """
    print("\n" + "=" * 60)
    print("Experiment 6: Upper Bound Analysis")
    print("=" * 60)

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                          user_mobility=True)
    seeds = list(range(42, 42 + 20))

    ub_rates = []
    ours_rates = []

    for seed in seeds:
        env = MultiUAVMECEnv(cfg)
        env.reset(seed=seed)
        ch = env.channel
        uav_pos = [uav.position for uav in env.uavs]
        N = cfg.num_users

        total_met_ub = 0
        total_met_ours = 0
        total_tasks = 0

        for slot in range(cfg.num_slots):
            if cfg.user_mobility:
                for i in range(N):
                    angle = env.rng.uniform(0, 2 * math.pi)
                    step_d = cfg.user_speed * cfg.uav.dt
                    env.user_positions[i, 0] += step_d * math.cos(angle)
                    env.user_positions[i, 1] += step_d * math.sin(angle)
                env.user_positions = np.clip(env.user_positions, 0, cfg.area_size)

            assignment = np.zeros(N, dtype=int)
            for i in range(N):
                d0 = np.linalg.norm(np.array(uav_pos[0][:2]) - env.user_positions[i])
                d1 = np.linalg.norm(np.array(uav_pos[1][:2]) - env.user_positions[i])
                assignment[i] = 0 if d0 <= d1 else 1

            h_gains = np.zeros((N, 2))
            for i in range(N):
                if not env.task_queue_active[i]:
                    continue
                for k in range(2):
                    h_gains[i, k] = ch.path_loss(
                        np.linalg.norm(np.array(uav_pos[k][:2]) - env.user_positions[i]))

            active_users = [i for i in range(N) if env.task_queue_active[i]]
            total_tasks += len(active_users)

            # --- Upper bound: per-subcarrier independent maximum ---
            users_k0 = [i for i in active_users if assignment[i] == 0]
            users_k1 = [i for i in active_users if assignment[i] == 1]

            bw_sub = ch.cfg.bandwidth / cfg.n_subcarriers
            noise_sub = ch.noise_linear * (bw_sub / ch.cfg.bandwidth)
            p_max = ch.cfg.p_max
            sic_eps = 0.02
            f_local = cfg.task.f_local
            result_ratio = cfg.task.result_ratio

            # Estimate f_alloc for upper bound (optimistic: fewer offloaders)
            n_off_est = min(len(active_users), cfg.n_subcarriers * 2)
            f_alloc_ub = cfg.f_per_uav / max(n_off_est, 1)

            def count_met_config(pair0, pair1):
                """Count deadline-met users for one subcarrier config."""
                met = 0
                # Cross-UAV interference
                cross_at_0 = 0.0
                cross_at_1 = 0.0
                for u in pair1:
                    if u is not None:
                        cross_at_0 += p_max * h_gains[u, 0]
                for u in pair0:
                    if u is not None:
                        cross_at_1 += p_max * h_gains[u, 1]

                for k, pair in [(0, pair0), (1, pair1)]:
                    cross = cross_at_0 if k == 0 else cross_at_1
                    if pair[0] is not None and pair[1] is not None:
                        i_s, i_w = pair
                        if h_gains[i_s, k] < h_gains[i_w, k]:
                            i_s, i_w = i_w, i_s
                        recv_s = p_max * h_gains[i_s, k]
                        recv_w = p_max * h_gains[i_w, k]
                        sinr_s = recv_s / (recv_w + cross + noise_sub)
                        sinr_w = recv_w / (sic_eps * recv_s + cross + noise_sub)
                        R_s = bw_sub * math.log2(1 + max(sinr_s, 1e-10))
                        R_w = bw_sub * math.log2(1 + max(sinr_w, 1e-10))
                        for ui, Ri in [(i_s, R_s), (i_w, R_w)]:
                            alpha = kkt_alpha(env.task_queue_data[ui],
                                              env.task_queue_comp[ui],
                                              Ri, f_alloc_ub, f_local)
                            t_loc = (1-alpha)*env.task_queue_comp[ui]/f_local
                            t_off = ((1+result_ratio)*alpha*env.task_queue_data[ui]/Ri +
                                     alpha*env.task_queue_comp[ui]/f_alloc_ub) if alpha > 0 else 0
                            if max(t_loc, t_off) <= cfg.task.t_max:
                                met += 1
                    elif pair[0] is not None:
                        i_s = pair[0]
                        recv = p_max * h_gains[i_s, k]
                        sinr = recv / (cross + noise_sub)
                        R = bw_sub * math.log2(1 + max(sinr, 1e-10))
                        alpha = kkt_alpha(env.task_queue_data[i_s],
                                          env.task_queue_comp[i_s],
                                          R, f_alloc_ub, f_local)
                        t_loc = (1-alpha)*env.task_queue_comp[i_s]/f_local
                        t_off = ((1+result_ratio)*alpha*env.task_queue_data[i_s]/R +
                                 alpha*env.task_queue_comp[i_s]/f_alloc_ub) if alpha > 0 else 0
                        if max(t_loc, t_off) <= cfg.task.t_max:
                            met += 1
                return met

            # For each subcarrier, find max deadline-met independently
            # This relaxes user-disjointness across subcarriers -> valid upper bound
            from itertools import combinations
            pairs0_candidates = [(i, j) for i, j in combinations(users_k0, 2)] + [(i, None) for i in users_k0]
            pairs1_candidates = [(i, j) for i, j in combinations(users_k1, 2)] + [(i, None) for i in users_k1]
            if not pairs0_candidates:
                pairs0_candidates = [(None, None)]
            if not pairs1_candidates:
                pairs1_candidates = [(None, None)]

            ub_met_per_sub = 0
            for _ in range(cfg.n_subcarriers):
                best_sub = 0
                for p0 in pairs0_candidates:
                    for p1 in pairs1_candidates:
                        m = count_met_config(p0, p1)
                        if m > best_sub:
                            best_sub = m
                best_sub = min(best_sub, 4)
                ub_met_per_sub += best_sub

            # Also count local-only users that meet deadline
            all_served_ub = min(cfg.n_subcarriers * 4, len(active_users))
            local_only_ub = len(active_users) - all_served_ub
            local_met_ub = 0
            # Sort unserved by local delay ascending
            local_delays = [(i, env.task_queue_comp[i] / f_local) for i in active_users]
            local_delays.sort(key=lambda x: x[1])
            for _, ld in local_delays:
                if ld <= cfg.task.t_max:
                    local_met_ub += 1
            # Upper bound = per-sub max + all local-only that meet deadline
            # But we need to be careful: local_met already includes served users
            # Simple upper bound: served_met_ub + local_only_met
            served_count = min(cfg.n_subcarriers * 4, len(active_users))
            unserved_users = active_users[served_count:] if served_count < len(active_users) else []
            local_unserved_met = sum(1 for i in active_users
                                     if env.task_queue_comp[i] / f_local <= cfg.task.t_max)
            # Total UB = min(ub_per_sub + local_unserved_met, total_active)
            total_ub_met = min(ub_met_per_sub + max(local_unserved_met - served_count, 0),
                               len(active_users))
            total_met_ub += total_ub_met

            # --- Ours ---
            hp = ilp_optimal_pairing(h_gains, env.task_queue_active, assignment,
                                     cfg.n_subcarriers, channel=ch,
                                     task_data=env.task_queue_data,
                                     task_comp=env.task_queue_comp,
                                     f_per_uav=cfg.f_per_uav,
                                     f_local=f_local, t_max=cfg.task.t_max)
            pairs_o, _, _ = swap_optimize(hp, h_gains, ch, env.task_queue_active,
                                           assignment, env.task_queue_data,
                                           env.task_queue_comp, cfg.n_subcarriers,
                                           cfg.f_per_uav, f_local, cfg.task.t_max)
            met_o, _, rates_o, alphas_o, served_o = evaluate_pairing(
                pairs_o, h_gains, ch, env.task_queue_data, env.task_queue_comp,
                env.task_queue_active, assignment, cfg.n_subcarriers,
                cfg.f_per_uav, f_local, cfg.task.t_max)
            total_met_ours += met_o

            # Task queue update
            for i in range(N):
                if not env.task_queue_active[i]:
                    continue
                if served_o[i] and rates_o[i] > 1.0:
                    alpha = alphas_o[i]
                    n_off = max(sum(1 for j in range(N) if served_o[j]
                                   and assignment[j] == assignment[i]
                                   and rates_o[j] > 1.0), 1)
                    f_alloc = max(cfg.f_per_uav / n_off, 1e3)
                    t_loc = (1 - alpha) * env.task_queue_comp[i] / f_local
                    t_off = ((1+result_ratio) * alpha * env.task_queue_data[i] / rates_o[i] +
                             alpha * env.task_queue_comp[i] / f_alloc
                             ) if alpha > 0 else 0
                    delay = max(t_loc, t_off)
                else:
                    delay = env.task_queue_comp[i] / f_local
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

        ub_rate = total_met_ub / max(total_tasks, 1) * 100
        ours_rate = total_met_ours / max(total_tasks, 1) * 100
        ub_rates.append(ub_rate)
        ours_rates.append(ours_rate)

    ub_mean = np.mean(ub_rates)
    ours_mean = np.mean(ours_rates)
    ratio = ours_mean / ub_mean * 100 if ub_mean > 0 else 0
    print(f"  Upper Bound: {ub_mean:.1f}±{np.std(ub_rates):.1f}%")
    print(f"  Ours:        {ours_mean:.1f}±{np.std(ours_rates):.1f}%")
    print(f"  Ratio:       {ratio:.1f}% of upper bound")

    return {
        "ub_mean": round(ub_mean, 2),
        "ub_std": round(float(np.std(ub_rates)), 2),
        "ours_mean": round(ours_mean, 2),
        "ours_std": round(float(np.std(ours_rates)), 2),
        "ratio_pct": round(ratio, 1),
        "ub_seeds": [round(x, 2) for x in ub_rates],
        "ours_seeds": [round(x, 2) for x in ours_rates],
    }


# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 60)
    print("Supplementary Experiments for Reviewer Responses")
    print("=" * 60)

    all_results = {}

    # 1. N-scalability
    all_results["n_scalability"] = run_n_scalability()

    # 2. CSI robustness
    all_results["csi_robustness"] = run_csi_experiment()

    # 3. SIC sensitivity
    all_results["sic_sensitivity"] = run_sic_experiment()

    # 4. Swap iterations
    all_results["swap_iterations"] = run_swap_iterations()

    # 5. Subcarrier sensitivity
    all_results["subcarrier_sensitivity"] = run_subcarrier_sensitivity()

    # 6. Upper bound
    all_results["upper_bound"] = run_upper_bound()

    # Save
    out = Path(__file__).parent.parent / "results" / "supplementary_experiments.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to: {out}")


if __name__ == "__main__":
    main()
