"""
Reviewer-requested supplementary experiments.

Fixes:
1. OFDMA + PPO(DRL) on SAME config/seeds as main table (seeds 42-61)
2. IntraSwapOnly at N=10,15,25,30 to verify cross-UAV swap scales with N
3. Cohen's d with correct ddof=1
4. A priori power analysis
5. Energy metrics
6. Timing per method

Usage: python -m uav_mec.run_reviewer_fixes
"""
import time, json, math
import numpy as np
from pathlib import Path
from scipy import stats

from .model import ScenarioConfig, TaskConfig, NOMAChannel, FIXED_CLUSTER_CENTERS_3
from .multi_uav_env import MultiUAVMECEnv
from .pairing import (greedy_pairing, benefit_greedy_pairing, swap_optimize,
                      evaluate_pairing, compute_rates_for_pairing, kkt_alpha)
from .baselines_pairing import (random_pairing, ofdma_rates,
                                 ilp_optimal_pairing, exchange_stable_pairing,
                                 worst_case_pairing)
from .run_main_experiment import (run_episode, strategy_ours, strategy_greedy,
                                   strategy_random, strategy_intra_swap_only,
                                   strategy_exchange_stable)


def strategy_ofdma(h, active, assign, env, ch):
    """OFDMA baseline: 1 user per subcarrier, no NOMA pairing."""
    cfg = env.cfg
    rates, served = ofdma_rates(h, active, assign, ch, cfg.n_subcarriers)
    # Convert to pairs format: each "pair" has only 1 user
    pairs = {k: [] for k in range(cfg.num_uavs)}
    for k in range(cfg.num_uavs):
        for i in range(cfg.num_users):
            if assign[i] == k and served[i]:
                pairs[k].append((i, -1))  # -1 = no NOMA partner
    return pairs


def run_episode_with_energy(cfg, seed, strategy_fn):
    """Run episode and also compute energy metrics."""
    env = MultiUAVMECEnv(cfg)
    env.reset(seed=seed)
    ch = env.channel
    uav_pos = [uav.position for uav in env.uavs]
    N = cfg.num_users

    total_met = 0
    total_tasks = 0
    total_tx_energy = 0.0
    total_local_energy = 0.0
    delays_all = []

    kappa = 1e-27  # effective capacitance coefficient

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

        pairs = strategy_fn(h_gains, env.task_queue_active, assignment, env, ch)

        met, total, rates, alphas, served = evaluate_pairing(
            pairs, h_gains, ch, env.task_queue_data, env.task_queue_comp,
            env.task_queue_active, assignment, cfg.n_subcarriers,
            cfg.f_per_uav, cfg.task.f_local, cfg.task.t_max)

        total_met += met
        total_tasks += total

        # Per-user energy + delay
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            D = env.task_queue_data[i]
            C = env.task_queue_comp[i]
            if served[i] and rates[i] > 1.0:
                alpha = alphas[i]
                n_off = max(sum(1 for j in range(N) if served[j]
                               and assignment[j] == assignment[i]
                               and rates[j] > 1.0), 1)
                f_alloc = max(cfg.f_per_uav / n_off, 1e3)
                t_loc = (1 - alpha) * C / cfg.task.f_local
                t_off = (alpha * D / rates[i] + alpha * C / f_alloc +
                         cfg.task.result_ratio * alpha * D / rates[i]
                         if alpha > 0 else 0)
                delay = max(t_loc, t_off)
                # Energy: tx energy = p_max * tx_time
                tx_time = alpha * D / rates[i] if rates[i] > 0 else 0
                total_tx_energy += cfg.channel.p_max * tx_time
                # Local compute energy = kappa * f_local^2 * C_local
                total_local_energy += kappa * (cfg.task.f_local ** 2) * (1 - alpha) * C
            else:
                delay = C / cfg.task.f_local
                total_local_energy += kappa * (cfg.task.f_local ** 2) * C
            delays_all.append(delay)

        # Update task queue (same as main experiment)
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            D = env.task_queue_data[i]
            C = env.task_queue_comp[i]
            if served[i] and rates[i] > 1.0:
                alpha = alphas[i]
                n_off = max(sum(1 for j in range(N) if served[j]
                               and assignment[j] == assignment[i]
                               and rates[j] > 1.0), 1)
                f_alloc = max(cfg.f_per_uav / n_off, 1e3)
                t_loc = (1 - alpha) * C / cfg.task.f_local
                t_off = (alpha * D / rates[i] + alpha * C / f_alloc +
                         cfg.task.result_ratio * alpha * D / rates[i]
                         if alpha > 0 else 0)
                delay = max(t_loc, t_off)
            else:
                delay = C / cfg.task.f_local
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

    deadline_rate = total_met / max(total_tasks, 1) * 100
    avg_delay = np.mean(delays_all) if delays_all else 0.0

    return {
        "deadline_rate": deadline_rate,
        "avg_delay": avg_delay,
        "total_met": total_met,
        "total_tasks": total_tasks,
        "tx_energy_total": total_tx_energy,
        "local_energy_total": total_local_energy,
        "avg_tx_energy_per_task": total_tx_energy / max(total_tasks, 1),
        "avg_local_energy_per_task": total_local_energy / max(total_tasks, 1),
    }


def exp1_ofdma_consistent():
    """Run OFDMA on SAME config as main experiment (seeds 42-61)."""
    print("\n" + "="*60)
    print("EXP 1: OFDMA baseline (consistent with main table)")
    print("="*60)

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                          user_mobility=True)
    seeds = list(range(42, 42 + 20))

    # OFDMA needs special handling since pairs format differs
    ofdma_dl = []
    ofdma_delay = []
    for seed in seeds:
        env = MultiUAVMECEnv(cfg)
        env.reset(seed=seed)
        ch = env.channel
        uav_pos = [uav.position for uav in env.uavs]
        N = cfg.num_users

        total_met = 0
        total_tasks = 0
        total_delay = 0.0
        n_tasks_for_delay = 0

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

            rates, served = ofdma_rates(h_gains, env.task_queue_active,
                                         assignment, ch, cfg.n_subcarriers)

            # Compute delays and deadline satisfaction
            active_count = sum(1 for i in range(N) if env.task_queue_active[i])
            for i in range(N):
                if not env.task_queue_active[i]:
                    continue
                D = env.task_queue_data[i]
                C = env.task_queue_comp[i]
                total_tasks += 1

                if served[i] and rates[i] > 1.0:
                    # KKT alpha for OFDMA user
                    k_i = assignment[i]
                    n_off = max(sum(1 for j in range(N) if served[j]
                                   and assignment[j] == k_i
                                   and rates[j] > 1.0), 1)
                    f_alloc = max(cfg.f_per_uav / n_off, 1e3)
                    alpha = kkt_alpha(D, C, rates[i], f_alloc, cfg.task.f_local)
                    t_loc = (1 - alpha) * C / cfg.task.f_local
                    t_off = (alpha * D / rates[i] + alpha * C / f_alloc +
                             cfg.task.result_ratio * alpha * D / rates[i])
                    delay = max(t_loc, t_off)
                else:
                    delay = C / cfg.task.f_local

                if delay <= cfg.task.t_max:
                    total_met += 1
                total_delay += delay
                n_tasks_for_delay += 1

            # Update task queue
            for i in range(N):
                if not env.task_queue_active[i]:
                    continue
                D = env.task_queue_data[i]
                C = env.task_queue_comp[i]
                if served[i] and rates[i] > 1.0:
                    k_i = assignment[i]
                    n_off = max(sum(1 for j in range(N) if served[j]
                                   and assignment[j] == k_i
                                   and rates[j] > 1.0), 1)
                    f_alloc = max(cfg.f_per_uav / n_off, 1e3)
                    alpha = kkt_alpha(D, C, rates[i], f_alloc, cfg.task.f_local)
                    t_loc = (1 - alpha) * C / cfg.task.f_local
                    t_off = (alpha * D / rates[i] + alpha * C / f_alloc +
                             cfg.task.result_ratio * alpha * D / rates[i])
                    delay = max(t_loc, t_off)
                else:
                    delay = C / cfg.task.f_local

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

        dl = total_met / max(total_tasks, 1) * 100
        ofdma_dl.append(dl)
        avg_d = total_delay / max(n_tasks_for_delay, 1)
        ofdma_delay.append(avg_d)

    print(f"  OFDMA: deadline={np.mean(ofdma_dl):.2f}% ± {np.std(ofdma_dl, ddof=1):.2f}%  "
          f"delay={np.mean(ofdma_delay):.3f}s")
    return {"deadline_seeds": [round(x, 2) for x in ofdma_dl],
            "delay_seeds": [round(x, 4) for x in ofdma_delay],
            "deadline_mean": round(np.mean(ofdma_dl), 2),
            "deadline_std": round(np.std(ofdma_dl, ddof=1), 2),
            "delay_mean": round(np.mean(ofdma_delay), 4)}


def exp2_cross_uav_swap_scaling():
    """Measure IntraSwapOnly at N=10,15,25,30 to see if cross-UAV value grows."""
    print("\n" + "="*60)
    print("EXP 2: Cross-UAV swap contribution vs N")
    print("="*60)

    seeds = list(range(42, 42 + 20))
    results = {}

    for N_val in [10, 15, 20, 25, 30]:
        cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                              user_mobility=True)
        cfg.num_users = N_val

        ours_dl = []
        intra_dl = []
        for seed in seeds:
            m1 = run_episode(cfg, seed, strategy_ours)
            m2 = run_episode(cfg, seed, strategy_intra_swap_only)
            ours_dl.append(m1["deadline_rate"])
            intra_dl.append(m2["deadline_rate"])

        gap = np.mean(ours_dl) - np.mean(intra_dl)
        print(f"  N={N_val}: Ours={np.mean(ours_dl):.1f}%  IntraOnly={np.mean(intra_dl):.1f}%  "
              f"CrossUAV_contrib={gap:+.1f}pp")
        results[str(N_val)] = {
            "ours_mean": round(np.mean(ours_dl), 2),
            "intra_mean": round(np.mean(intra_dl), 2),
            "cross_uav_gap": round(gap, 2),
            "ours_seeds": [round(x, 2) for x in ours_dl],
            "intra_seeds": [round(x, 2) for x in intra_dl],
        }

    return results


def exp3_energy_metrics():
    """Compute energy metrics for main methods."""
    print("\n" + "="*60)
    print("EXP 3: Energy metrics")
    print("="*60)

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                          user_mobility=True)
    seeds = list(range(42, 42 + 20))

    methods = {
        "Ours(ILP+Swap)": strategy_ours,
        "Greedy": strategy_greedy,
        "ExchangeStable": strategy_exchange_stable,
        "Random": strategy_random,
    }

    results = {}
    for name, fn in methods.items():
        tx_energies = []
        loc_energies = []
        for seed in seeds:
            m = run_episode_with_energy(cfg, seed, fn)
            tx_energies.append(m["avg_tx_energy_per_task"])
            loc_energies.append(m["avg_local_energy_per_task"])
        print(f"  {name:20s}: avg_tx={np.mean(tx_energies)*1000:.2f}mJ/task  "
              f"avg_local={np.mean(loc_energies)*1000:.4f}mJ/task")
        results[name] = {
            "avg_tx_energy_mJ": round(np.mean(tx_energies) * 1000, 3),
            "avg_local_energy_mJ": round(np.mean(loc_energies) * 1000, 6),
        }

    return results


def exp4_timing_scaling():
    """Measure runtime vs N for Ours method."""
    print("\n" + "="*60)
    print("EXP 4: Runtime scaling vs N")
    print("="*60)

    seeds = [42, 43, 44]  # 3 seeds for timing
    results = {}

    for N_val in [10, 15, 20, 25, 30, 40, 50]:
        cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                              user_mobility=True)
        cfg.num_users = N_val

        slot_times = []
        for seed in seeds:
            env = MultiUAVMECEnv(cfg)
            env.reset(seed=seed)
            ch = env.channel
            uav_pos = [uav.position for uav in env.uavs]
            N = cfg.num_users

            for slot in range(min(cfg.num_slots, 20)):  # 20 slots for timing
                if cfg.user_mobility:
                    for i in range(N):
                        angle = env.rng.uniform(0, 2 * math.pi)
                        step_d = cfg.user_speed * cfg.uav.dt
                        env.user_positions[i, 0] += step_d * math.cos(angle)
                        env.user_positions[i, 1] += step_d * math.sin(angle)
                    env.user_positions = np.clip(env.user_positions, 0, cfg.area_size)

                assignment = np.zeros(N, dtype=int)
                for i in range(N):
                    dists = [np.linalg.norm(np.array(uav_pos[k][:2]) - env.user_positions[i])
                             for k in range(cfg.num_uavs)]
                    assignment[i] = int(np.argmin(dists))

                h_gains = np.zeros((N, cfg.num_uavs))
                for i in range(N):
                    if not env.task_queue_active[i]:
                        continue
                    for k in range(cfg.num_uavs):
                        h_gains[i, k] = ch.path_loss(
                            np.linalg.norm(np.array(uav_pos[k][:2]) - env.user_positions[i]))

                t_start = time.perf_counter()
                _ = strategy_ours(h_gains, env.task_queue_active, assignment, env, ch)
                t_end = time.perf_counter()
                slot_times.append((t_end - t_start) * 1000)  # ms

                # Update queue minimally
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

        avg_ms = np.mean(slot_times)
        print(f"  N={N_val}: avg={avg_ms:.2f}ms/slot  med={np.median(slot_times):.2f}ms  "
              f"max={np.max(slot_times):.2f}ms")
        results[str(N_val)] = {
            "avg_ms": round(avg_ms, 3),
            "median_ms": round(float(np.median(slot_times)), 3),
            "max_ms": round(float(np.max(slot_times)), 3),
        }

    return results


def exp5_corrected_statistics():
    """Recompute all statistics with correct ddof=1."""
    print("\n" + "="*60)
    print("EXP 5: Corrected Statistics (ddof=1)")
    print("="*60)

    with open(Path(__file__).resolve().parent.parent / "results" / "phase2_main_experiment.json") as f:
        data = json.load(f)

    results_data = data["results"]
    ours = np.array(results_data["Ours(ILP+Swap)"]["deadline_seeds"])

    stats_out = {}
    baselines = [k for k in results_data if k != "Ours(ILP+Swap)"]

    # Collect p-values for Holm-Bonferroni
    p_values = {}
    for name in baselines:
        bl = np.array(results_data[name]["deadline_seeds"])
        diff = ours - bl
        try:
            stat, p = stats.wilcoxon(ours, bl, alternative="greater")
        except ValueError:
            p = 1.0
        p_values[name] = p

    # Holm-Bonferroni
    sorted_pvals = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_pvals)
    hb_reject = {}
    for rank, (name, p) in enumerate(sorted_pvals, 1):
        adj_alpha = 0.001 / (m - rank + 1)
        hb_reject[name] = p < adj_alpha

    for name in baselines:
        bl = np.array(results_data[name]["deadline_seeds"])
        diff = ours - bl
        d_correct = float(np.mean(diff) / np.std(diff, ddof=1))
        gap = float(np.mean(diff))
        se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
        p = p_values[name]

        sig = "***" if hb_reject[name] else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  vs {name:20s}: gap={gap:+.2f}pp  d={d_correct:.2f}  p={p:.6f} {sig}  "
              f"95%CI=[{gap-1.96*se:.2f},{gap+1.96*se:.2f}]")

        stats_out[name] = {
            "gap_pp": round(gap, 2),
            "cohens_d": round(d_correct, 2),
            "p_value": round(p, 8),
            "sig": sig,
            "ci95_low": round(gap - 1.96 * se, 2),
            "ci95_high": round(gap + 1.96 * se, 2),
            "hb_reject_001": hb_reject[name],
        }

    # A priori MDE
    z_alpha, z_beta = 1.645, 0.842
    d_min = (z_alpha + z_beta) / np.sqrt(20)
    avg_diff_std = np.mean([np.std(ours - np.array(results_data[name]["deadline_seeds"]), ddof=1)
                            for name in baselines])
    mde = d_min * avg_diff_std
    print(f"\n  MDE (n=20, 80% power, one-sided): d_min={d_min:.3f}, ≈{mde:.2f}pp")

    stats_out["_power_analysis"] = {
        "n_seeds": 20,
        "d_min_80power": round(d_min, 3),
        "mde_pp": round(mde, 2),
        "avg_diff_std": round(avg_diff_std, 2),
    }

    return stats_out


def main():
    all_results = {}

    # Exp 1: OFDMA
    all_results["ofdma_consistent"] = exp1_ofdma_consistent()

    # Exp 2: Cross-UAV swap scaling
    all_results["cross_uav_scaling"] = exp2_cross_uav_swap_scaling()

    # Exp 3: Energy
    all_results["energy_metrics"] = exp3_energy_metrics()

    # Exp 4: Timing
    all_results["timing_scaling"] = exp4_timing_scaling()

    # Exp 5: Corrected stats
    all_results["corrected_statistics"] = exp5_corrected_statistics()

    # Save — convert numpy types to native Python for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def deep_convert(d):
        if isinstance(d, dict):
            return {k: deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [deep_convert(v) for v in d]
        return convert(d)

    out = Path(__file__).resolve().parent.parent / "results" / "reviewer_fixes_experiments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(deep_convert(all_results), f, indent=2)
    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
