"""
Reviewer fixes:
  EXP A: Exhaustive search for small N to measure true optimality gap.
  EXP B: Multiple UAV position configurations to verify robustness.
"""
import time, json, math, itertools
import numpy as np
from pathlib import Path
from scipy import stats

from .model import ScenarioConfig
from .multi_uav_env import MultiUAVMECEnv
from .pairing import evaluate_pairing, kkt_alpha
from .run_main_experiment import (run_episode, strategy_ours, strategy_greedy,
                                   strategy_exchange_stable)


# ─────────────────────────────────────────────────────────────────────────────
# EXP A: Exhaustive search (small N)
# ─────────────────────────────────────────────────────────────────────────────

def exhaustive_pairing(h_gains, active_mask, assignment, n_sub, channel,
                       task_data, task_comp, f_per_uav, f_local, t_max):
    """Enumerate ALL feasible pairings and return the globally optimal one.
    Handles solo (user, None) pairs when a UAV has fewer than 2*n_sub users.
    """
    N = h_gains.shape[0]
    K = 2

    users = {k: [i for i in range(N) if assignment[i] == k and active_mask[i]]
             for k in range(K)}

    def gen_pairings(user_list, n_slots):
        """All ways to pick n_slots pairs (2-user or 1-user) from user_list."""
        if n_slots == 0:
            yield []
            return
        if len(user_list) == 0:
            return
        # Try 2-user pair
        if len(user_list) >= 2:
            for pair in itertools.combinations(user_list, 2):
                remaining = [u for u in user_list if u not in pair]
                for rest in gen_pairings(remaining, n_slots - 1):
                    yield [pair] + rest
        # Try solo (user, None)
        for u in user_list:
            remaining = [x for x in user_list if x != u]
            for rest in gen_pairings(remaining, n_slots - 1):
                yield [(u, None)] + rest

    best_met = -1
    best_pairs = None

    uav0_options = list(gen_pairings(users[0], min(n_sub, len(users[0]))))
    uav1_options = list(gen_pairings(users[1], min(n_sub, len(users[1]))))

    if not uav0_options:
        uav0_options = [[] ]
    if not uav1_options:
        uav1_options = [[]]

    for p0 in uav0_options:
        for p1 in uav1_options:
            pairs = {0: p0, 1: p1}
            met, total, _, _, _ = evaluate_pairing(
                pairs, h_gains, channel, task_data, task_comp,
                active_mask, assignment, n_sub, f_per_uav, f_local, t_max)
            if met > best_met:
                best_met = met
                best_pairs = pairs

    return best_pairs, best_met


def strategy_exhaustive(h, active, assign, env, ch):
    cfg = env.cfg
    pairs, _ = exhaustive_pairing(
        h, active, assign, cfg.n_subcarriers, ch,
        env.task_queue_data, env.task_queue_comp,
        cfg.f_per_uav, cfg.task.f_local, cfg.task.t_max)
    return pairs if pairs is not None else {k: [] for k in range(cfg.num_uavs)}


def exp_a_exhaustive():
    """Compare Ours vs true optimum on N=6 (feasible to enumerate)."""
    print("\n" + "=" * 60)
    print("EXP A: Exhaustive search (N=6, S=2)")
    print("=" * 60)

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                          user_mobility=True)
    cfg.num_users = 8
    cfg.n_subcarriers = 2
    seeds = list(range(42, 42 + 20))

    ours_dl, exh_dl, greedy_dl = [], [], []
    for seed in seeds:
        r_ours = run_episode(cfg, seed, strategy_ours)
        r_exh  = run_episode(cfg, seed, strategy_exhaustive)
        r_gr   = run_episode(cfg, seed, strategy_greedy)
        ours_dl.append(r_ours["deadline_rate"])
        exh_dl.append(r_exh["deadline_rate"])
        greedy_dl.append(r_gr["deadline_rate"])

    gap_ours_exh = np.mean(ours_dl) - np.mean(exh_dl)
    gap_gr_exh   = np.mean(greedy_dl) - np.mean(exh_dl)

    print(f"  Exhaustive (optimal): {np.mean(exh_dl):.1f}% ± {np.std(exh_dl,ddof=1):.1f}%")
    print(f"  Ours(ILP+Swap):       {np.mean(ours_dl):.1f}% ± {np.std(ours_dl,ddof=1):.1f}%")
    print(f"  Greedy:               {np.mean(greedy_dl):.1f}% ± {np.std(greedy_dl,ddof=1):.1f}%")
    print(f"  Ours vs Optimal:      {gap_ours_exh:+.1f}pp")
    print(f"  Greedy vs Optimal:    {gap_gr_exh:+.1f}pp")
    pct_of_opt = np.mean(ours_dl) / np.mean(exh_dl) * 100
    print(f"  Ours achieves {pct_of_opt:.1f}% of optimal")

    return {
        "N": 8, "S": 2, "n_seeds": 20,
        "exhaustive_mean": round(float(np.mean(exh_dl)), 2),
        "exhaustive_std":  round(float(np.std(exh_dl, ddof=1)), 2),
        "ours_mean":       round(float(np.mean(ours_dl)), 2),
        "ours_std":        round(float(np.std(ours_dl, ddof=1)), 2),
        "greedy_mean":     round(float(np.mean(greedy_dl)), 2),
        "gap_ours_vs_opt": round(float(gap_ours_exh), 2),
        "gap_greedy_vs_opt": round(float(gap_gr_exh), 2),
        "ours_pct_of_optimal": round(float(pct_of_opt), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXP B: Multiple UAV position configurations
# ─────────────────────────────────────────────────────────────────────────────

UAV_POSITION_CONFIGS = {
    "default":     [(267, 333), (533, 267)],   # Optimised spread (paper default)
    "symmetric":   [(200, 400), (600, 400)],   # Symmetric horizontal
    "diagonal":    [(150, 150), (650, 650)],   # Corner-to-corner
    "center":      [(300, 400), (500, 400)],   # Close together
}


def run_with_uav_positions(cfg, seed, strategy_fn, positions):
    """Run an episode with custom UAV hover positions."""
    env = MultiUAVMECEnv(cfg)
    env.reset(seed=seed)

    # Override UAV positions
    for k, (px, py) in enumerate(positions):
        env.uavs[k].x = float(px)
        env.uavs[k].y = float(py)

    ch = env.channel
    uav_pos = [(uav.position[0], uav.position[1], uav.position[2]) for uav in env.uavs]
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
            d0 = math.hypot(uav_pos[0][0] - env.user_positions[i, 0],
                            uav_pos[0][1] - env.user_positions[i, 1])
            d1 = math.hypot(uav_pos[1][0] - env.user_positions[i, 0],
                            uav_pos[1][1] - env.user_positions[i, 1])
            assignment[i] = 0 if d0 <= d1 else 1

        h_gains = np.zeros((N, 2))
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            for k in range(2):
                d_h = math.hypot(uav_pos[k][0] - env.user_positions[i, 0],
                                 uav_pos[k][1] - env.user_positions[i, 1])
                h_gains[i, k] = ch.path_loss(
                    math.sqrt(d_h**2 + uav_pos[k][2]**2))

        pairs = strategy_fn(h_gains, env.task_queue_active, assignment, env, ch)

        met, total, rates, alphas, served = evaluate_pairing(
            pairs, h_gains, ch, env.task_queue_data, env.task_queue_comp,
            env.task_queue_active, assignment, cfg.n_subcarriers,
            cfg.f_per_uav, cfg.task.f_local, cfg.task.t_max)
        total_met += met
        total_tasks += total

        # Update queue
        for i in range(N):
            if not env.task_queue_active[i]:
                continue
            D, C = env.task_queue_data[i], env.task_queue_comp[i]
            if served[i] and rates[i] > 1.0:
                alpha = alphas[i]
                n_off = max(sum(1 for j in range(N)
                                if served[j] and assignment[j] == assignment[i]
                                and rates[j] > 1.0), 1)
                f_alloc = max(cfg.f_per_uav / n_off, 1e3)
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

    return total_met / max(total_tasks, 1) * 100


def exp_b_uav_positions():
    """Test Ours vs Greedy under multiple UAV position configurations."""
    print("\n" + "=" * 60)
    print("EXP B: UAV position robustness (20 seeds each)")
    print("=" * 60)

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                          user_mobility=True)
    seeds = list(range(42, 42 + 20))
    results = {}

    for config_name, positions in UAV_POSITION_CONFIGS.items():
        ours_dl, greedy_dl = [], []
        for seed in seeds:
            o = run_with_uav_positions(cfg, seed, strategy_ours, positions)
            g = run_with_uav_positions(cfg, seed, strategy_greedy, positions)
            ours_dl.append(o)
            greedy_dl.append(g)

        gap = np.mean(ours_dl) - np.mean(greedy_dl)
        print(f"  {config_name:12s}: Ours={np.mean(ours_dl):.1f}%  "
              f"Greedy={np.mean(greedy_dl):.1f}%  gap={gap:+.1f}pp")
        results[config_name] = {
            "positions": positions,
            "ours_mean":   round(float(np.mean(ours_dl)), 2),
            "ours_std":    round(float(np.std(ours_dl, ddof=1)), 2),
            "greedy_mean": round(float(np.mean(greedy_dl)), 2),
            "greedy_std":  round(float(np.std(greedy_dl, ddof=1)), 2),
            "gap_pp":      round(float(gap), 2),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    out_path = Path(__file__).resolve().parent.parent / "results"
    out_path.mkdir(exist_ok=True)

    result_a = exp_a_exhaustive()
    result_b = exp_b_uav_positions()

    out = out_path / "reviewer_extra_experiments.json"
    with open(out, "w") as f:
        json.dump({"exhaustive_search": result_a,
                   "uav_position_robustness": result_b}, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
