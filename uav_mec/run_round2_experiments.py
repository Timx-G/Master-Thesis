"""
Round-2 revision experiments.

Exp A: PPO fair comparison — each of 20 seeds trains an independent PPO
       (2000 episodes) and is evaluated on that same seed's env.

Exp B: InterferenceBlind+Swap ablation — swap accepts moves based on
       interference-UNAWARE rate estimates (no I_cross), but final
       evaluation uses true full-interference model.
       Isolates: how much of the gain comes from explicitly modeling
       cross-UAV interference during swap decisions?

Usage:
    python -m uav_mec.run_round2_experiments          # both Exp A + B
    python -m uav_mec.run_round2_experiments --ppo    # PPO only
    python -m uav_mec.run_round2_experiments --ablation  # ablation only
"""
import argparse, time, json, math
import numpy as np
from pathlib import Path
from scipy import stats

from .model import ScenarioConfig
from .multi_uav_env import MultiUAVMECEnv
from .pairing import (greedy_pairing, swap_optimize, evaluate_pairing,
                       compute_rates_for_pairing, kkt_alpha,
                       benefit_greedy_pairing)
from .baselines_pairing import ilp_optimal_pairing, exchange_stable_pairing
from .run_main_experiment import (run_episode, strategy_ours, strategy_greedy,
                                   strategy_ilp_optimal)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def rank_biserial_r(x, y):
    diff = np.asarray(x) - np.asarray(y)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    T_plus = float(np.sum(ranks[diff > 0]))
    return 2 * T_plus / (n * (n + 1) / 2) - 1

def paired_stats(a, b, label=""):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    gap  = float(np.mean(diff))
    d    = float(np.mean(diff) / (np.std(diff, ddof=0) + 1e-10))
    r    = rank_biserial_r(a, b)
    try:
        _, p = stats.wilcoxon(a, b, alternative='greater')
    except ValueError:
        p = 1.0
    se   = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci   = [round(gap - 1.96*se, 3), round(gap + 1.96*se, 3)]
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    if label:
        print(f"  {label:28s}: gap={gap:+.2f}pp  r={r:.3f}  d={d:.2f}"
              f"  p={p:.2e}{sig}  CI={ci}")
    return {"gap": round(gap,3), "r": round(r,3), "d": round(d,3),
            "p": float(p), "ci95": ci}


# ─────────────────────────────────────────────────────────────────────────────
# Exp A: PPO fair 20-seed comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_ppo_fair_20seed():
    """Train an independent PPO for each of 20 seeds; evaluate on same seed.

    Framework: the environment seed controls user positions, task arrivals,
    and mobility. Each PPO is trained for 2000 episodes starting from that
    seed's initial layout, then evaluated for one episode on the same seed.
    This makes each PPO "specialized" to one scenario — a conservative
    comparison (our algorithm has no training cost at all).
    """
    from .agents.ppo import PPOAgent

    print("\n" + "="*65)
    print("Exp A: PPO 20-seed fair comparison (2000 ep/seed)")
    print("="*65)

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                          user_mobility=True)
    seeds = list(range(42, 62))
    N, K = cfg.num_users, cfg.num_uavs

    ppo_deadline = []
    t_total = time.time()

    for seed in seeds:
        t0 = time.time()
        env = MultiUAVMECEnv(cfg)

        # ── Train ────────────────────────────────────────────────────────
        obs, _ = env.reset(seed=seed)
        agent = PPOAgent(obs_dim=env.obs_dim, act_dim=env.action_dim,
                         lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2,
                         epochs=4, hidden=128, mini_batch=64, seed=seed)

        best_deadline = 0.0
        best_state    = None

        import copy as _copy
        for ep in range(2000):
            obs, _ = env.reset(seed=seed + ep * 1000)
            done = False
            slot_rates = []
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                agent.store(obs, action, reward, terminated or truncated)
                obs = next_obs
                done = terminated or truncated
                slot_rates.append(info.get("deadline_rate", 0.0))
            agent.update()

            dl = float(np.mean(slot_rates)) * 100 if slot_rates else 0.0
            if dl > best_deadline:
                best_deadline = dl
                best_state = _copy.deepcopy({
                    "net": agent.net.state_dict(),
                    "mu":  agent.mu_head.state_dict(),
                    "log_std": agent.log_std.data.clone(),
                })

        # ── Evaluate: restore best policy, run on eval seed ──────────────
        if best_state:
            agent.net.load_state_dict(best_state["net"])
            agent.mu_head.load_state_dict(best_state["mu"])
            agent.log_std.data.copy_(best_state["log_std"])

        obs, _ = env.reset(seed=seed)
        done = False
        slot_rates_eval = []
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            slot_rates_eval.append(info.get("deadline_rate", 0.0))

        dl_eval = float(np.mean(slot_rates_eval)) * 100 if slot_rates_eval else 0.0
        ppo_deadline.append(dl_eval)
        elapsed = time.time() - t0
        print(f"  seed={seed}: best_train={best_deadline:.1f}%  "
              f"eval={dl_eval:.1f}%  ({elapsed:.0f}s)")

    mean_dl = float(np.mean(ppo_deadline))
    std_dl  = float(np.std(ppo_deadline, ddof=1))
    print(f"\n  PPO 20-seed fair: {mean_dl:.1f}±{std_dl:.1f}%")
    print(f"  Total: {(time.time()-t_total)/60:.1f} min")
    return ppo_deadline, mean_dl, std_dl


# ─────────────────────────────────────────────────────────────────────────────
# Exp B: InterferenceBlind+Swap ablation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rates_blind(pairs_per_uav, h_gains, channel, n_sub, sic_eps=0.02):
    """Rate computation WITHOUT cross-UAV interference (interference-blind)."""
    import math as _math
    N = h_gains.shape[0]
    K = h_gains.shape[1]
    rates = np.zeros(N)
    p_max   = channel.cfg.p_max
    bw_sub  = channel.cfg.bandwidth / n_sub
    noise_sub = channel.noise_linear * (bw_sub / channel.cfg.bandwidth)

    for sub_idx in range(n_sub):
        sub_pairs = {}
        for k in range(K):
            if sub_idx < len(pairs_per_uav.get(k, [])):
                sub_pairs[k] = pairs_per_uav[k][sub_idx]

        for k, (i_strong, i_weak) in sub_pairs.items():
            # Zero cross-UAV interference (blind)
            cross_interf = 0.0
            if i_weak is not None:
                recv_s = p_max * h_gains[i_strong, k]
                recv_w = p_max * h_gains[i_weak,   k]
                sinr_s = recv_s / (recv_w + noise_sub)
                sinr_w = recv_w / (sic_eps * recv_s + noise_sub)
                rates[i_strong] = bw_sub * _math.log2(1 + max(sinr_s, 1e-10))
                rates[i_weak]   = bw_sub * _math.log2(1 + max(sinr_w, 1e-10))
            elif i_strong is not None:
                recv = p_max * h_gains[i_strong, k]
                sinr = recv / noise_sub
                rates[i_strong] = bw_sub * _math.log2(1 + max(sinr, 1e-10))
    return rates


def _evaluate_blind(pairs_per_uav, h_gains, channel, task_data, task_comp,
                    active_mask, assignment, n_sub, f_per_uav, f_local,
                    t_max, sic_eps=0.02):
    """Evaluate using interference-BLIND rates (for swap candidate selection)."""
    from .pairing import kkt_alpha as _kkt
    N  = len(active_mask)
    K  = h_gains.shape[1]
    rates = _compute_rates_blind(pairs_per_uav, h_gains, channel, n_sub, sic_eps)

    served = np.zeros(N, dtype=bool)
    for k in range(K):
        for pair in pairs_per_uav.get(k, []):
            if pair[0] is not None: served[pair[0]] = True
            if len(pair) > 1 and pair[1] is not None: served[pair[1]] = True

    met = 0
    alphas = np.zeros(N)
    for k in range(K):
        sk = [i for i in range(N)
               if assignment[i]==k and active_mask[i] and served[i]
               and rates[i] > 1.0]
        f_alloc = max(f_per_uav / max(len(sk), 1), 1e3)
        for i in sk:
            a = _kkt(task_data[i], task_comp[i], rates[i],
                     f_alloc, f_local)
            alphas[i] = a
            t_loc = (1-a)*task_comp[i]/f_local
            t_off = (a*task_data[i]/rates[i] + a*task_comp[i]/f_alloc
                     + 0.1*a*task_data[i]/rates[i])
            if max(t_loc, t_off) <= t_max:
                met += 1
        for i in range(N):
            if assignment[i]==k and active_mask[i] and not served[i]:
                if task_comp[i]/f_local <= t_max:
                    met += 1
    return met


def swap_optimize_blind(initial_pairs, h_gains, channel, active_mask,
                        assignment, task_data, task_comp, n_sub, f_per_uav,
                        f_local, t_max, sic_eps=0.02, max_iter=50):
    """Swap refinement using interference-BLIND evaluation for candidate selection.

    Accepts a swap if the interference-blind deadline count improves.
    Final performance is still measured with true (full-interference) rates.
    """
    N = len(active_mask)
    K = h_gains.shape[1]
    pairs = {k: list(initial_pairs.get(k, [])) for k in range(K)}

    def eval_blind():
        return _evaluate_blind(pairs, h_gains, channel, task_data, task_comp,
                                active_mask, assignment, n_sub, f_per_uav,
                                f_local, t_max, sic_eps)

    best = eval_blind()

    for _ in range(max_iter):
        improved = False

        # --- Replace ---
        for k in range(K):
            served_set = set()
            for p in pairs[k]:
                if p[0] is not None: served_set.add(p[0])
                if len(p)>1 and p[1] is not None: served_set.add(p[1])
            unserved = [i for i in range(N)
                        if assignment[i]==k and active_mask[i]
                        and i not in served_set]
            for p_idx, pair in enumerate(pairs[k]):
                for pos in [0, 1]:
                    old_u = pair[pos] if pos < len(pair) else None
                    if old_u is None: continue
                    for new_u in unserved:
                        new_p = list(pair)
                        if pos < len(new_p): new_p[pos] = new_u
                        else: new_p.append(new_u)
                        if (len(new_p)>1 and new_p[0] is not None
                                and new_p[1] is not None
                                and h_gains[new_p[0],k] < h_gains[new_p[1],k]):
                            new_p[0], new_p[1] = new_p[1], new_p[0]
                        old_pv = pairs[k][p_idx]
                        pairs[k][p_idx] = tuple(new_p)
                        val = eval_blind()
                        if val > best:
                            best = val
                            improved = True
                            unserved = [x for x in unserved if x != new_u]
                            unserved.append(old_u)
                            break
                        else:
                            pairs[k][p_idx] = old_pv
                    if improved: break
                if improved: break

        # --- Position ---
        if not improved:
            for k in range(K):
                for p1 in range(len(pairs[k])):
                    for p2 in range(p1+1, len(pairs[k])):
                        for pos1 in [0,1]:
                            for pos2 in [0,1]:
                                u1 = pairs[k][p1][pos1] if pos1<len(pairs[k][p1]) else None
                                u2 = pairs[k][p2][pos2] if pos2<len(pairs[k][p2]) else None
                                if u1 is None or u2 is None: continue
                                op1, op2 = pairs[k][p1], pairs[k][p2]
                                np1, np2 = list(op1), list(op2)
                                if pos1<len(np1): np1[pos1] = u2
                                if pos2<len(np2): np2[pos2] = u1
                                pairs[k][p1], pairs[k][p2] = tuple(np1), tuple(np2)
                                val = eval_blind()
                                if val > best:
                                    best = val; improved = True; break
                                else:
                                    pairs[k][p1], pairs[k][p2] = op1, op2
                            if improved: break
                        if improved: break
                    if improved: break

        # --- Cross-UAV ---
        if not improved and K >= 2:
            for k1 in range(K):
                for k2 in range(k1+1, K):
                    for p1 in range(len(pairs[k1])):
                        for p2 in range(len(pairs[k2])):
                            for pos1 in [0,1]:
                                for pos2 in [0,1]:
                                    u1 = pairs[k1][p1][pos1] if pos1<len(pairs[k1][p1]) else None
                                    u2 = pairs[k2][p2][pos2] if pos2<len(pairs[k2][p2]) else None
                                    if u1 is None or u2 is None: continue
                                    op1, op2 = pairs[k1][p1], pairs[k2][p2]
                                    np1, np2 = list(op1), list(op2)
                                    if pos1<len(np1): np1[pos1] = u2
                                    if pos2<len(np2): np2[pos2] = u1
                                    pairs[k1][p1], pairs[k2][p2] = tuple(np1), tuple(np2)
                                    val = eval_blind()
                                    if val > best:
                                        best = val; improved = True; break
                                    else:
                                        pairs[k1][p1], pairs[k2][p2] = op1, op2
                                if improved: break
                            if improved: break
                        if improved: break
                    if improved: break
        if not improved:
            break

    return pairs


def strategy_interference_blind_swap(h, active, assign, env, ch):
    """ILP init + swap that ignores cross-UAV interference in move evaluation."""
    cfg = env.cfg
    hp = ilp_optimal_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                              task_data=env.task_queue_data,
                              task_comp=env.task_queue_comp,
                              f_per_uav=cfg.f_per_uav,
                              f_local=cfg.task.f_local, t_max=cfg.task.t_max)
    pairs = swap_optimize_blind(hp, h, ch, active, assign,
                                 env.task_queue_data, env.task_queue_comp,
                                 cfg.n_subcarriers, cfg.f_per_uav,
                                 cfg.task.f_local, cfg.task.t_max)
    return pairs


def run_ablation_interference_blind(seeds=None):
    """Compare three methods to isolate interference-awareness contribution:

    1. Per-UAV ILP (no swap)        — initialization only
    2. ILP + blind-swap             — swap without interference awareness
    3. Ours (ILP + aware-swap)      — swap WITH interference awareness

    Gap (2)→(3) = marginal value of modeling cross-UAV interference in swap.
    """
    if seeds is None:
        seeds = list(range(42, 62))

    print("\n" + "="*65)
    print("Exp B: Interference-blind-swap ablation (20 seeds)")
    print("="*65)

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                          user_mobility=True)
    t0 = time.time()

    ilp_dl, blind_dl, aware_dl, greedy_dl = [], [], [], []

    for seed in seeds:
        m_ilp    = run_episode(cfg, seed, strategy_ilp_optimal)
        m_blind  = run_episode(cfg, seed, strategy_interference_blind_swap)
        m_aware  = run_episode(cfg, seed, strategy_ours)
        m_greedy = run_episode(cfg, seed, strategy_greedy)
        ilp_dl.append(m_ilp["deadline_rate"])
        blind_dl.append(m_blind["deadline_rate"])
        aware_dl.append(m_aware["deadline_rate"])
        greedy_dl.append(m_greedy["deadline_rate"])

    elapsed = time.time() - t0

    print(f"\n  Results (N=20, 20 seeds):")
    print(f"  Greedy (baseline)    : {np.mean(greedy_dl):.2f}±{np.std(greedy_dl,ddof=1):.2f}%")
    print(f"  Per-UAV ILP          : {np.mean(ilp_dl):.2f}±{np.std(ilp_dl,ddof=1):.2f}%")
    print(f"  ILP+Blind-Swap       : {np.mean(blind_dl):.2f}±{np.std(blind_dl,ddof=1):.2f}%")
    print(f"  Ours (ILP+Aware-Swap): {np.mean(aware_dl):.2f}±{np.std(aware_dl,ddof=1):.2f}%")
    print()
    print("  Gap decomposition (pp):")
    print(f"  Greedy → Per-UAV ILP       : {np.mean(ilp_dl)-np.mean(greedy_dl):+.2f}  [better init]")
    print(f"  Per-UAV ILP → Blind-Swap   : {np.mean(blind_dl)-np.mean(ilp_dl):+.2f}  [swap, no interf. awareness]")
    print(f"  Blind-Swap → Aware-Swap    : {np.mean(aware_dl)-np.mean(blind_dl):+.2f}  [interf. awareness in swap]")
    print(f"  Greedy → Ours (total)      : {np.mean(aware_dl)-np.mean(greedy_dl):+.2f}")
    print()
    print("  Statistical tests:")
    st_blind_aware = paired_stats(aware_dl, blind_dl, "Ours vs Blind-Swap")
    st_ilp_blind   = paired_stats(blind_dl, ilp_dl,   "Blind-Swap vs ILP")
    st_greedy_ilp  = paired_stats(ilp_dl,  greedy_dl, "ILP vs Greedy")
    print(f"\n  Elapsed: {elapsed:.0f}s")

    return {
        "greedy":      {"mean": round(float(np.mean(greedy_dl)),3),
                        "std":  round(float(np.std(greedy_dl,ddof=1)),3),
                        "seeds": [round(x,3) for x in greedy_dl]},
        "ilp_only":    {"mean": round(float(np.mean(ilp_dl)),3),
                        "std":  round(float(np.std(ilp_dl,ddof=1)),3),
                        "seeds": [round(x,3) for x in ilp_dl]},
        "blind_swap":  {"mean": round(float(np.mean(blind_dl)),3),
                        "std":  round(float(np.std(blind_dl,ddof=1)),3),
                        "seeds": [round(x,3) for x in blind_dl]},
        "aware_swap":  {"mean": round(float(np.mean(aware_dl)),3),
                        "std":  round(float(np.std(aware_dl,ddof=1)),3),
                        "seeds": [round(x,3) for x in aware_dl]},
        "stats": {
            "ours_vs_blind": st_blind_aware,
            "blind_vs_ilp":  st_ilp_blind,
            "ilp_vs_greedy": st_greedy_ilp,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo",      action="store_true")
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    run_all = not (args.ppo or args.ablation)
    output  = {}
    out_path = Path(__file__).parent.parent / "results" / "round2_experiments.json"

    if args.ablation or run_all:
        output["ablation"] = run_ablation_interference_blind()

    if args.ppo or run_all:
        try:
            ppo_seeds, ppo_mean, ppo_std = run_ppo_fair_20seed()
            output["ppo_fair"] = {
                "seeds": [round(x,3) for x in ppo_seeds],
                "mean": round(ppo_mean,3), "std": round(ppo_std,3),
            }
        except Exception as e:
            print(f"  PPO training failed: {e}")
            print("  Using fallback: PPO excluded from paired test (documented in paper).")
            output["ppo_fair"] = {"error": str(e), "fallback": "excluded from paired test"}

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
    return output


if __name__ == "__main__":
    main()
