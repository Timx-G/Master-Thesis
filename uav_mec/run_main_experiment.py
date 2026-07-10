"""
Phase 2: Main experiment — 7 methods × 20 seeds × 5 metrics.

Methods:
  1. Ours (ILP+Swap+delay-equalization crossing-point alpha)
  2. Greedy Pairing
  3. Random Pairing
  4. BenefitGreedy-Only (no Swap) — ablation
  5. Equal Alpha (Greedy + fixed α=0.5) — ablation
  6. Single-UAV (1 UAV, 8GHz)
  7. OFDMA (no NOMA, 1 user/subcarrier)

Usage: python -m uav_mec.run_main_experiment
"""
import time, json, math
import numpy as np
from pathlib import Path
from scipy import stats

from .model import ScenarioConfig, TaskConfig, NOMAChannel, FIXED_CLUSTER_CENTERS_3
from .multi_uav_env import MultiUAVMECEnv
from .pairing import (greedy_pairing, benefit_greedy_pairing, swap_optimize,
                       evaluate_pairing, compute_rates_for_pairing, kkt_alpha)
from .baselines_pairing import (random_pairing, ofdma_rates, ofdma_ortho_rates,
                                 ofdma_fair_rates,
                                 ilp_optimal_pairing, hungarian_pairing,
                                 exchange_stable_pairing,
                                 exchange_stable_cross_uav_pairing,
                                 worst_case_pairing,
                                 ia_greedy_pairing)


def run_episode(cfg, seed, strategy_fn):
    """Run one 100-slot episode with a given pairing strategy.

    strategy_fn(h_gains, active_mask, assignment, env, ch) -> pairs_per_uav

    Returns dict of metrics.
    """
    env = MultiUAVMECEnv(cfg)
    env.reset(seed=seed)
    ch = env.channel
    uav_pos = [uav.position for uav in env.uavs]
    N = cfg.num_users

    total_met = 0
    total_tasks = 0
    total_delay = 0.0
    slot_deadlines = []
    delays_all = []
    energy_all = []      # per-task transmission energy (Joules)
    aoi_list = []        # mean AoI per slot (slots), averaged over active users

    for slot in range(cfg.num_slots):
        # User mobility
        if cfg.user_mobility:
            for i in range(N):
                angle = env.rng.uniform(0, 2 * math.pi)
                step_d = cfg.user_speed * cfg.uav.dt
                env.user_positions[i, 0] += step_d * math.cos(angle)
                env.user_positions[i, 1] += step_d * math.sin(angle)
            env.user_positions = np.clip(env.user_positions, 0, cfg.area_size)

        # Assignment
        assignment = np.zeros(N, dtype=int)
        for i in range(N):
            d0 = np.linalg.norm(np.array(uav_pos[0][:2]) - env.user_positions[i])
            d1 = np.linalg.norm(np.array(uav_pos[1][:2]) - env.user_positions[i])
            assignment[i] = 0 if d0 <= d1 else 1

        # Channel gains
        h_gains = np.zeros((N, 2))
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            for k in range(2):
                h_gains[i, k] = ch.path_loss(
                    np.linalg.norm(np.array(uav_pos[k][:2]) - env.user_positions[i]))

        # Strategy-specific pairing.
        # Strategy may return either:
        #   (a) pairs_per_uav dict  -> evaluated via evaluate_pairing
        #   (b) pre-computed 5-tuple (met, total, rates, alphas, served)
        #       used by OFDMA baselines whose rate model bypasses evaluate_pairing
        _result = strategy_fn(h_gains, env.task_queue_active, assignment, env, ch)

        if isinstance(_result, tuple):
            # Pre-computed result (e.g., OFDMA baselines)
            met, total, rates, alphas, served = _result
        else:
            pairs = _result
            met, total, rates, alphas, served = evaluate_pairing(
                pairs, h_gains, ch, env.task_queue_data, env.task_queue_comp,
                env.task_queue_active, assignment, cfg.n_subcarriers,
                cfg.f_per_uav, cfg.task.f_local, cfg.task.t_max)

        total_met += met
        total_tasks += total
        if total > 0:
            slot_deadlines.append(met / total * 100)

        # --- Energy metric ---
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            if served[i] and rates[i] > 1.0 and alphas[i] > 0:
                p_max = 0.2  # W (3GPP Class 3)
                tx_energy = p_max * alphas[i] * env.task_queue_data[i] / rates[i]
                energy_all.append(tx_energy)
            else:
                energy_all.append(0.0)

        # --- AoI metric: mean age of active tasks this slot ---
        active_idxs = [i for i in range(N) if env.task_queue_active[i]]
        if active_idxs:
            slot_aoi = float(np.mean([env.task_queue_age[i] + 1
                                       for i in active_idxs]))
            aoi_list.append(slot_aoi)

        # Compute per-user delays for metrics
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            if served[i] and rates[i] > 1.0:
                alpha = alphas[i]
                D = env.task_queue_data[i]
                C = env.task_queue_comp[i]
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
                delay = env.task_queue_comp[i] / cfg.task.f_local
            delays_all.append(delay)
            total_delay += delay

        # Update task queue
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            if served[i] and rates[i] > 1.0:
                alpha = alphas[i]
                D = env.task_queue_data[i]
                C = env.task_queue_comp[i]
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
                delay = env.task_queue_comp[i] / cfg.task.f_local

            if delay <= cfg.task.t_max:
                env.task_queue_active[i] = False
                env.task_queue_age[i] = 0
            else:
                env.task_queue_age[i] += 1
                if env.task_queue_age[i] >= cfg.max_queue_age:
                    env.task_queue_active[i] = False
                    env.task_queue_age[i] = 0

        # New tasks
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
    jain = 0.0
    if slot_deadlines:
        q = np.array(slot_deadlines) / 100.0
        jain = (q.sum() ** 2) / (len(q) * (q ** 2).sum() + 1e-10)

    return {
        "deadline_rate": deadline_rate,
        "avg_delay": avg_delay,
        "jain_fairness": jain,
        "total_met": total_met,
        "total_tasks": total_tasks,
        "avg_energy": float(np.mean(energy_all)) if energy_all else 0.0,
        "avg_aoi":    float(np.mean(aoi_list))   if aoi_list   else 0.0,
    }


# ===== Strategy functions =====

def strategy_ours(h, active, assign, env, ch):
    """Ours: Per-UAV ILP initialization + Cross-UAV Swap (interference-aware)."""
    cfg = env.cfg
    hp = ilp_optimal_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                             task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                             f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                             t_max=cfg.task.t_max)
    # Position Swap and Cross-UAV Swap now include strong/weak reordering (fixed 2026-04-14)
    pairs, _, _ = swap_optimize(hp, h, ch, active, assign,
                                 env.task_queue_data, env.task_queue_comp,
                                 cfg.n_subcarriers, cfg.f_per_uav,
                                 cfg.task.f_local, cfg.task.t_max)
    return pairs

def strategy_bg_swap(h, active, assign, env, ch):
    """BenefitGreedy initialization + Swap (ablation baseline)."""
    cfg = env.cfg
    hp = benefit_greedy_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                            task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                            f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                            t_max=cfg.task.t_max)
    pairs, _, _ = swap_optimize(hp, h, ch, active, assign,
                                 env.task_queue_data, env.task_queue_comp,
                                 cfg.n_subcarriers, cfg.f_per_uav,
                                 cfg.task.f_local, cfg.task.t_max)
    return pairs

def strategy_greedy(h, active, assign, env, ch):
    return greedy_pairing(h, active, assign, env.cfg.n_subcarriers)

def strategy_random(h, active, assign, env, ch):
    # Use env.strategy_rng (NOT env.rng) to avoid polluting the environment's
    # PRNG stream (mobility + task arrivals) which must be identical across all baselines.
    return random_pairing(h, active, assign, env.cfg.n_subcarriers,
                          rng=env.strategy_rng)

def strategy_benefit_only(h, active, assign, env, ch):
    cfg = env.cfg
    return benefit_greedy_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                              task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                              f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                              t_max=cfg.task.t_max)

def strategy_greedy_swap(h, active, assign, env, ch):
    cfg = env.cfg
    gp = greedy_pairing(h, active, assign, cfg.n_subcarriers)
    pairs, _, _ = swap_optimize(gp, h, ch, active, assign,
                                 env.task_queue_data, env.task_queue_comp,
                                 cfg.n_subcarriers, cfg.f_per_uav,
                                 cfg.task.f_local, cfg.task.t_max)
    return pairs

def strategy_ilp_optimal(h, active, assign, env, ch):
    cfg = env.cfg
    return ilp_optimal_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                               task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                               f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                               t_max=cfg.task.t_max)

def strategy_ilp_swap(h, active, assign, env, ch):
    cfg = env.cfg
    hp = ilp_optimal_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                             task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                             f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                             t_max=cfg.task.t_max)
    pairs, _, _ = swap_optimize(hp, h, ch, active, assign,
                                 env.task_queue_data, env.task_queue_comp,
                                 cfg.n_subcarriers, cfg.f_per_uav,
                                 cfg.task.f_local, cfg.task.t_max)
    return pairs

def strategy_intra_swap_only(h, active, assign, env, ch):
    cfg = env.cfg
    hp = benefit_greedy_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                            task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                            f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                            t_max=cfg.task.t_max)
    pairs, _, _ = swap_optimize(hp, h, ch, active, assign,
                                 env.task_queue_data, env.task_queue_comp,
                                 cfg.n_subcarriers, cfg.f_per_uav,
                                 cfg.task.f_local, cfg.task.t_max,
                                 cross_uav=False)
    return pairs

def strategy_worst_swap(h, active, assign, env, ch):
    cfg = env.cfg
    wp = worst_case_pairing(h, active, assign, cfg.n_subcarriers)
    pairs, _, _ = swap_optimize(wp, h, ch, active, assign,
                                 env.task_queue_data, env.task_queue_comp,
                                 cfg.n_subcarriers, cfg.f_per_uav,
                                 cfg.task.f_local, cfg.task.t_max)
    return pairs

def strategy_ilp_blind_swap(h, active, assign, env, ch):
    """ILP init + interference-blind swap (evaluates moves ignoring cross-UAV interference).

    Ablation: reveals how much of the swap's gain comes from interference-aware
    move evaluation vs. the swap structure itself.
    """
    cfg = env.cfg
    hp = ilp_optimal_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                             task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                             f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                             t_max=cfg.task.t_max)
    pairs, _, _ = swap_optimize(hp, h, ch, active, assign,
                                 env.task_queue_data, env.task_queue_comp,
                                 cfg.n_subcarriers, cfg.f_per_uav,
                                 cfg.task.f_local, cfg.task.t_max,
                                 blind_cross_uav=True)
    return pairs


def strategy_ia_greedy(h, active, assign, env, ch):
    """IA-Greedy: Interference-Aware Greedy (no swap) — Reviewer 3 required baseline."""
    cfg = env.cfg
    return ia_greedy_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                             task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                             f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                             t_max=cfg.task.t_max)


def strategy_ia_greedy_swap(h, active, assign, env, ch):
    """IA-Greedy + Swap: Interference-aware greedy init + global first-improvement swap.

    2x2 ablation cell: {IA-Greedy init} x {Global objective}.
    Reveals the isolated contribution of interference-aware initialization
    vs. the swap's global objective.
    """
    cfg = env.cfg
    hp = ia_greedy_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                           task_data=env.task_queue_data, task_comp=env.task_queue_comp,
                           f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                           t_max=cfg.task.t_max)
    pairs, _, _ = swap_optimize(hp, h, ch, active, assign,
                                 env.task_queue_data, env.task_queue_comp,
                                 cfg.n_subcarriers, cfg.f_per_uav,
                                 cfg.task.f_local, cfg.task.t_max)
    return pairs


def strategy_exchange_stable(h, active, assign, env, ch):
    cfg = env.cfg
    return exchange_stable_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                                   task_data=env.task_queue_data,
                                   task_comp=env.task_queue_comp,
                                   f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                                   t_max=cfg.task.t_max)


def strategy_exchange_stable_cross_uav(h, active, assign, env, ch):
    """ExStable+CrossUAV: exchange-stable matching extended with cross-UAV swap moves.

    Key ablation: tests whether ExStable's gap vs Ours is due to
    (a) lack of cross-UAV capability, or (b) conservative bilateral criterion.
    """
    cfg = env.cfg
    return exchange_stable_cross_uav_pairing(
        h, active, assign, cfg.n_subcarriers, channel=ch,
        task_data=env.task_queue_data, task_comp=env.task_queue_comp,
        f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
        t_max=cfg.task.t_max)


def _evaluate_oma(rates, served, env, h, assign, cfg):
    """Evaluate metrics for OMA/OFDMA baselines given pre-computed rates.

    Returns (met, total, rates, alphas, served) matching evaluate_pairing output.
    """
    N = len(env.task_queue_active)
    alphas = np.zeros(N)
    met = 0
    total = 0
    for k in range(cfg.num_uavs):
        served_k = [i for i in range(N)
                    if assign[i] == k and env.task_queue_active[i]
                    and served[i] and rates[i] > 1.0]
        unserved_k = [i for i in range(N)
                      if assign[i] == k and env.task_queue_active[i] and not served[i]]
        n_off = len(served_k)
        f_alloc = max(cfg.f_per_uav / max(n_off, 1), 1e3)
        for i in served_k:
            total += 1
            alpha = kkt_alpha(env.task_queue_data[i], env.task_queue_comp[i],
                              rates[i], f_alloc, cfg.task.f_local)
            alphas[i] = alpha
            t_loc = (1 - alpha) * env.task_queue_comp[i] / cfg.task.f_local
            t_off = (alpha * env.task_queue_data[i] / rates[i]
                     + alpha * env.task_queue_comp[i] / f_alloc
                     + cfg.task.result_ratio * alpha * env.task_queue_data[i] / rates[i]
                     ) if alpha > 0 else 0.0
            delay = max(t_loc, t_off)
            if delay <= cfg.task.t_max:
                met += 1
        for i in unserved_k:
            total += 1
            delay = env.task_queue_comp[i] / cfg.task.f_local
            if delay <= cfg.task.t_max:
                met += 1
    return met, total, rates, alphas, served


def strategy_ofdma(h, active, assign, env, ch):
    """Full-spectrum OFDMA: one user per subcarrier, cross-UAV interference present."""
    rates, served = ofdma_rates(h, active, assign, ch, env.cfg.n_subcarriers)
    return _evaluate_oma(rates, served, env, h, assign, env.cfg)


def strategy_ofdma_ortho(h, active, assign, env, ch):
    """Orthogonal-band OFDMA: each UAV occupies a non-overlapping sub-band (no cross-UAV interference)."""
    rates, served = ofdma_ortho_rates(h, active, assign, ch, env.cfg.n_subcarriers)
    return _evaluate_oma(rates, served, env, h, assign, env.cfg)


def strategy_ofdma_fair(h, active, assign, env, ch):
    """Fair OFDMA: TDMA within subcarrier, same user count as NOMA (12 users total)."""
    rates, served = ofdma_fair_rates(h, active, assign, ch, env.cfg.n_subcarriers)
    return _evaluate_oma(rates, served, env, h, assign, env.cfg)


# ===== Main =====

def main():
    print("=" * 70)
    print("Phase 2: Main Experiment — 20 seeds")
    print("=" * 70)

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                          user_mobility=True)
    print(f"Config: n_sub={cfg.n_subcarriers}, prob={cfg.task.arrival_prob}, "
          f"t_max={cfg.task.t_max}")

    seeds = list(range(42, 42 + 20))
    methods = {
        "Ours(ILP+Swap)":    strategy_ours,
        "BG+Swap":           strategy_bg_swap,
        "Greedy+Swap":       strategy_greedy_swap,
        "ILP-Optimal":       strategy_ilp_optimal,
        "ILP+BlindSwap":     strategy_ilp_blind_swap,
        "IntraSwapOnly":     strategy_intra_swap_only,
        "WorstCase+Swap":    strategy_worst_swap,
        "ExchangeStable":    strategy_exchange_stable,
        "ExStable+CrossUAV": strategy_exchange_stable_cross_uav,
        "BenefitGreedy":     strategy_benefit_only,
        "IA-Greedy":         strategy_ia_greedy,          # R3: new baseline
        "IA-Greedy+Swap":    strategy_ia_greedy_swap,     # R4: 2x2 ablation cell
        "Greedy":            strategy_greedy,
        "Random":            strategy_random,
        "OFDMA(FullReuse)":  strategy_ofdma,
        "OFDMA(OrthoBank)":  strategy_ofdma_ortho,
    }

    results = {name: {"deadline": [], "delay": [], "jain": [], "energy": [], "aoi": []}
               for name in methods}

    start = time.time()
    for name, fn in methods.items():
        t0 = time.time()
        for seed in seeds:
            m = run_episode(cfg, seed, fn)
            results[name]["deadline"].append(m["deadline_rate"])
            results[name]["delay"].append(m["avg_delay"])
            results[name]["jain"].append(m["jain_fairness"])
            results[name]["energy"].append(m["avg_energy"])
            results[name]["aoi"].append(m["avg_aoi"])
        elapsed = time.time() - t0
        dl = results[name]["deadline"]
        jn = results[name]["jain"]
        en = results[name]["energy"]
        ao = results[name]["aoi"]
        print(f"  {name:22s}: deadline={np.mean(dl):.1f}±{np.std(dl):.1f}%  "
              f"delay={np.mean(results[name]['delay']):.3f}s  "
              f"energy={np.mean(en):.4f}J  "
              f"aoi={np.mean(ao):.2f}slots  "
              f"jain={np.mean(jn):.4f}  ({elapsed:.1f}s)")

    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Statistical tests: Ours vs each baseline (Holm-Bonferroni corrected)
    print("\n--- Statistical Tests (Ours vs baselines, Holm-Bonferroni corrected) ---")
    ours_dl = results["Ours(ILP+Swap)"]["deadline"]
    baseline_names = [n for n in methods if n != "Ours(ILP+Swap)"]
    n_comparisons = len(baseline_names)

    # Collect raw p-values and statistics
    raw_stats = {}
    for name in baseline_names:
        bl_dl = results[name]["deadline"]
        diff = np.array(ours_dl) - np.array(bl_dl)
        gap = np.mean(diff)
        try:
            W, p_raw = stats.wilcoxon(ours_dl, bl_dl, alternative="greater")
        except ValueError:
            W, p_raw = 0.0, 1.0
        d_paired = np.mean(diff) / (np.std(diff) + 1e-10)
        # Rank-biserial: r = 1 - 2*T / (n*(n+1)/2) where T = min(W+, W-)
        # scipy alternative="greater" returns W+ (sum of positive ranks)
        n = len(ours_dl)
        w_pos = W
        w_neg = n * (n + 1) / 2 - w_pos
        T_stat = min(w_pos, w_neg)
        r_rb = 1.0 - 2.0 * T_stat / (n * (n + 1) / 2)
        raw_stats[name] = {"gap": gap, "W": W, "p_raw": p_raw, "d": d_paired, "r": r_rb}

    # Holm-Bonferroni correction: sort by raw p ascending.
    # For TIED p-values, all methods in the tie group receive the SAME multiplier
    # = (m - first_rank_in_group) to be conservative (Holm step-down with ties).
    sorted_names = sorted(baseline_names, key=lambda n: raw_stats[n]["p_raw"])
    i = 0
    while i < n_comparisons:
        j = i
        tie_p = raw_stats[sorted_names[i]]["p_raw"]
        while j < n_comparisons and raw_stats[sorted_names[j]]["p_raw"] == tie_p:
            j += 1
        # All in [i, j) are tied: use multiplier for rank i (most conservative)
        holm_mult = n_comparisons - i
        for k in range(i, j):
            p_corr = min(tie_p * holm_mult, 1.0)
            raw_stats[sorted_names[k]]["p_corr"] = p_corr
            raw_stats[sorted_names[k]]["holm_mult"] = holm_mult
        i = j

    print(f"  {'Method':<22s}  {'Gap':>6s}  {'W':>8s}  {'p_raw':>8s}  {'Holm*':>5s}  "
          f"{'p_corr':>8s}  {'sig':>4s}  {'d_paired':>9s}  {'r_rb':>5s}")
    for name in baseline_names:
        s = raw_stats[name]
        pc = s["p_corr"]
        sig = "***" if pc < 0.001 else "**" if pc < 0.01 else "*" if pc < 0.05 else "ns"
        print(f"  {name:<22s}  {s['gap']:>+6.2f}pp  {s['W']:>8.1f}  {s['p_raw']:>8.5f}  "
              f"x{s['holm_mult']:<4d}  {pc:>8.5f}  {sig:>4s}  "
              f"{s['d']:>9.2f}  {s['r']:>5.2f}")

    # Save
    archive = {
        "phase": "2_main_experiment",
        "date": time.strftime("%Y-%m-%d"),
        "config": {"n_sub": cfg.n_subcarriers, "arrival_prob": cfg.task.arrival_prob,
                    "t_max": cfg.task.t_max, "N": cfg.num_users, "K": cfg.num_uavs,
                    "n_seeds": len(seeds)},
        "results": {
            name: {
                "deadline_mean": round(np.mean(v["deadline"]), 2),
                "deadline_std": round(np.std(v["deadline"]), 2),
                "delay_mean": round(np.mean(v["delay"]), 4),
                "jain_mean": round(np.mean(v["jain"]), 4),
                "energy_mean": round(np.mean(v["energy"]), 6),
                "energy_std": round(np.std(v["energy"]), 4),
                "aoi_mean": round(np.mean(v["aoi"]), 4),
                "aoi_std": round(np.std(v["aoi"]), 4),
                "deadline_seeds": [round(x, 4) for x in v["deadline"]],  # full precision for stats
                "energy_seeds": [round(x, 4) for x in v["energy"]],
                "aoi_seeds": [round(x, 4) for x in v["aoi"]],
            }
            for name, v in results.items()
        },
        "elapsed_seconds": round(total_time, 1),
    }
    out = Path(__file__).parent.parent / "results" / "phase2_main_experiment.json"
    with open(out, "w") as f:
        json.dump(archive, f, indent=2)
    print(f"\nArchive: {out}")


if __name__ == "__main__":
    main()
