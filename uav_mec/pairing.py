"""
NOMA User Pairing Strategies for Multi-UAV MEC.

Three-layer optimization:
  Layer 1: User selection + subcarrier-pair assignment (Per-UAV ILP or Greedy)
  Layer 2: Cross-UAV swap refinement (Replace/Position/Cross swap)
  Layer 3: Per-user delay-equalization offloading ratio α*
            (sets τ_local = τ_offload, i.e., the crossing-point optimality condition
             for min τ = max(τ_loc, τ_off); note: standard KKT requires differentiability,
             but max(·,·) is non-smooth; this is the correct convex crossing-point formula)

All strategies produce the same output format:
  pairs_per_uav = {k: [(i_strong, i_weak), ...]} — one pair per subcarrier per UAV
  served = np.ndarray (N,) bool — which users are served
"""
from __future__ import annotations
import math
import numpy as np
# Note: scipy.linear_sum_assignment is NOT used.
# The "benefit_greedy" function below is a greedy heuristic, not optimal matching.


# =========================================================================
# Layer 3: Delay-equalization (crossing-point optimality) offloading ratio
# =========================================================================

def kkt_alpha(D: float, C: float, R: float, f_alloc: float,
              f_local: float, alpha_scale: float = 0.95,
              result_ratio: float = 0.1) -> float:
    """Delay-equalization optimal offloading ratio: set τ_local = τ_offload.

    Since τ = max(τ_loc, τ_off), and τ_loc is decreasing while τ_off is
    increasing in α, the minimum is at the unique crossing point:
    τ_loc(α*) = τ_off(α*), giving:

    α* = (C/f_local) / (C/f_local + (1+ρ)D/R + C/f_alloc)

    Note: this is NOT a standard KKT condition (max(·,·) is non-differentiable);
    it is the crossing-point optimality condition for convex-concave piece-wise
    functions. The formula is exact for the given piecewise-linear structure.

    Args:
        D: data bits, C: computation cycles, R: rate (bps),
        f_alloc: allocated MEC compute (cycles/s),
        f_local: local CPU freq, alpha_scale: upper bound scaling,
        result_ratio: result-to-input ratio ρ (default 0.1)
    Returns:
        alpha in [0, alpha_scale]
    """
    if D < 1e-3 or C < 1e-3 or R < 1.0:
        return 0.0
    t_local_full = C / f_local
    t_off_unit = (1 + result_ratio) * D / R + C / f_alloc
    alpha = t_local_full / (t_local_full + t_off_unit)
    return min(max(alpha, 0.01), alpha_scale)


# =========================================================================
# Rate computation for a given pairing
# =========================================================================

def compute_rates_for_pairing(pairs_per_uav, h_gains, channel, n_sub,
                               sic_eps=0.02):
    """Compute NOMA rates for a specific pairing assignment.

    Args:
        pairs_per_uav: {k: [(i_strong, i_weak), ...]} — pairs per UAV
        h_gains: ndarray (N, K) — channel gains from user i to UAV k
        channel: NOMAChannel instance
        n_sub: number of subcarriers
        sic_eps: imperfect SIC factor
    Returns:
        rates: ndarray (N,) — per-user rate in bps
    """
    N = h_gains.shape[0]
    K = h_gains.shape[1]
    rates = np.zeros(N)
    p_max = channel.cfg.p_max
    bw_sub = channel.cfg.bandwidth / n_sub
    noise_sub = channel.noise_linear * (bw_sub / channel.cfg.bandwidth)

    for sub_idx in range(n_sub):
        # Collect pairs on this subcarrier from all UAVs
        sub_pairs = {}
        for k in range(K):
            if sub_idx < len(pairs_per_uav.get(k, [])):
                sub_pairs[k] = pairs_per_uav[k][sub_idx]

        for k in sub_pairs:
            i_strong, i_weak = sub_pairs[k]

            # Cross-UAV interference on this subcarrier
            cross_interf = 0.0
            for k2 in sub_pairs:
                if k2 == k:
                    continue
                i_s2, i_w2 = sub_pairs[k2]
                if i_s2 is not None:
                    cross_interf += p_max * h_gains[i_s2, k]
                if i_w2 is not None:
                    cross_interf += p_max * h_gains[i_w2, k]

            if i_weak is not None:
                recv_s = p_max * h_gains[i_strong, k]
                recv_w = p_max * h_gains[i_weak, k]
                sinr_s = recv_s / (recv_w + cross_interf + noise_sub)
                sinr_w = recv_w / (sic_eps * recv_s + cross_interf + noise_sub)
                rates[i_strong] = bw_sub * math.log2(1 + max(sinr_s, 1e-10))
                rates[i_weak] = bw_sub * math.log2(1 + max(sinr_w, 1e-10))
            elif i_strong is not None:
                recv = p_max * h_gains[i_strong, k]
                sinr = recv / (cross_interf + noise_sub)
                rates[i_strong] = bw_sub * math.log2(1 + max(sinr, 1e-10))

    return rates


# =========================================================================
# Evaluate a pairing: compute total deadline-met tasks
# =========================================================================

def evaluate_pairing(pairs_per_uav, h_gains, channel, task_data, task_comp,
                      active_mask, assignment, n_sub, f_per_uav, f_local,
                      t_max, sic_eps=0.02, alpha_scale=0.95, result_ratio=0.1):
    """Evaluate a pairing: returns (met_count, total_count, rates, alphas).

    This is the unified evaluation function used by all strategies.
    """
    N = len(active_mask)
    K = h_gains.shape[1]

    # Compute rates
    rates = compute_rates_for_pairing(pairs_per_uav, h_gains, channel, n_sub,
                                       sic_eps)

    # Build served mask
    served = np.zeros(N, dtype=bool)
    for k in range(K):
        for pair in pairs_per_uav.get(k, []):
            if pair[0] is not None:
                served[pair[0]] = True
            if pair[1] is not None:
                served[pair[1]] = True

    # Compute alphas + delays
    met = 0
    total = 0
    alphas = np.zeros(N)

    for k in range(K):
        served_k = [i for i in range(N)
                     if assignment[i] == k and active_mask[i] and served[i]
                     and rates[i] > 1.0]
        unserved_k = [i for i in range(N)
                       if assignment[i] == k and active_mask[i] and not served[i]]

        n_off = len(served_k)
        f_alloc = f_per_uav / max(n_off, 1)
        f_alloc = max(f_alloc, 1e3)

        for i in served_k:
            total += 1
            alpha = kkt_alpha(task_data[i], task_comp[i], rates[i],
                              f_alloc, f_local, alpha_scale)
            alphas[i] = alpha
            t_local = (1 - alpha) * task_comp[i] / f_local
            D_off = alpha * task_data[i]
            C_off = alpha * task_comp[i]
            t_off = (D_off / rates[i] + C_off / f_alloc +
                    result_ratio * D_off / rates[i]) if D_off > 0 else 0.0
            delay = max(t_local, t_off)
            if delay <= t_max:
                met += 1

        for i in unserved_k:
            total += 1
            delay = task_comp[i] / f_local
            if delay <= t_max:
                met += 1

    return met, total, rates, alphas, served


# =========================================================================
# Layer 1a: Greedy Pairing
# =========================================================================

def greedy_pairing(h_gains, active_mask, assignment, n_sub, users_per_sub=2):
    """Greedy: per-UAV, sort by channel gain, pair strongest with weakest.

    Returns pairs_per_uav: {k: [(i_strong, i_weak), ...]}
    """
    N = h_gains.shape[0]
    K = h_gains.shape[1]
    max_served = n_sub * users_per_sub
    pairs_per_uav = {}

    for k in range(K):
        candidates = [i for i in range(N)
                       if assignment[i] == k and active_mask[i]]
        candidates.sort(key=lambda i: h_gains[i, k], reverse=True)
        selected = candidates[:max_served]

        n = len(selected)
        pairs = []
        for p in range(n // 2):
            pairs.append((selected[p], selected[n - 1 - p]))
        if n % 2 == 1:
            pairs.append((selected[n // 2], None))

        pairs_per_uav[k] = pairs[:n_sub]

    return pairs_per_uav


# =========================================================================
# Layer 1b: Benefit-Aware Greedy Pairing
# =========================================================================

def benefit_greedy_pairing(h_gains, active_mask, assignment, n_sub,
                       users_per_sub=2, channel=None, task_data=None,
                       task_comp=None, f_per_uav=4e9, f_local=1e9,
                       t_max=0.8, sic_eps=0.02):
    """Benefit-aware greedy pairing: select pairs by deadline benefit.

    For each UAV:
      - Enumerate all possible 2-user pairs from candidates
      - Estimate each pair's deadline benefit (how many users meet deadline)
      - Greedily select top non-overlapping pairs by benefit
    Note: This is a greedy heuristic, NOT optimal matching.
    """
    N = h_gains.shape[0]
    K = h_gains.shape[1]
    max_pairs = n_sub  # one pair per subcarrier
    pairs_per_uav = {}

    for k in range(K):
        candidates = [i for i in range(N)
                       if assignment[i] == k and active_mask[i]]
        if len(candidates) < 2:
            if candidates:
                pairs_per_uav[k] = [(candidates[0], None)]
            else:
                pairs_per_uav[k] = []
            continue

        # Generate all possible 2-user pairs
        from itertools import combinations
        all_pairs = list(combinations(candidates, 2))

        # For each pair, estimate benefit: how many of the 2 users meet deadline?
        # Use a simplified rate estimate (ignoring cross-UAV interference for speed)
        bw_sub = channel.cfg.bandwidth / n_sub if channel else 1e6
        noise_sub = (channel.noise_linear * bw_sub / channel.cfg.bandwidth
                     if channel else 1e-14)
        p_max = channel.cfg.p_max if channel else 0.2

        pair_benefits = []
        for (i, j) in all_pairs:
            # Ensure strong/weak ordering
            if h_gains[i, k] >= h_gains[j, k]:
                i_s, i_w = i, j
            else:
                i_s, i_w = j, i

            recv_s = p_max * h_gains[i_s, k]
            recv_w = p_max * h_gains[i_w, k]
            sinr_s = recv_s / (recv_w + noise_sub)
            sinr_w = recv_w / (sic_eps * recv_s + noise_sub)
            R_s = bw_sub * math.log2(1 + max(sinr_s, 1e-10))
            R_w = bw_sub * math.log2(1 + max(sinr_w, 1e-10))

            # How many meet deadline?
            n_pair_served = 2
            f_alloc = f_per_uav / max(min(len(candidates), n_sub * 2), 1)
            benefit = 0
            for user_i, R_i in [(i_s, R_s), (i_w, R_w)]:
                if task_data is not None and task_comp is not None:
                    alpha = kkt_alpha(task_data[user_i], task_comp[user_i],
                                      R_i, f_alloc, f_local)
                    D = task_data[user_i]
                    C = task_comp[user_i]
                    t_loc = (1 - alpha) * C / f_local
                    t_off = ((1 + 0.1) * alpha * D / R_i +
                             alpha * C / f_alloc) if alpha > 0 else 0
                    delay = max(t_loc, t_off)
                    if delay <= t_max:
                        benefit += 1
                else:
                    benefit += R_i  # fallback: maximize rate sum

            pair_benefits.append(((i_s, i_w), benefit))

        # Select top pairs greedily by benefit, ensuring no user overlap
        pair_benefits.sort(key=lambda x: x[1], reverse=True)
        selected_pairs = []
        used_users = set()
        for (pair, ben) in pair_benefits:
            if pair[0] in used_users or pair[1] in used_users:
                continue
            selected_pairs.append(pair)
            used_users.add(pair[0])
            used_users.add(pair[1])
            if len(selected_pairs) >= max_pairs:
                break

        pairs_per_uav[k] = selected_pairs

    return pairs_per_uav


# =========================================================================
# Layer 2: Cross-UAV Swap Refinement
# =========================================================================

def swap_optimize(initial_pairs, h_gains, channel, active_mask, assignment,
                   task_data, task_comp, n_sub, f_per_uav, f_local, t_max,
                   sic_eps=0.02, alpha_scale=0.95, max_iter=50,
                   cross_uav=True, blind_cross_uav=False):
    """Iterative swap refinement on an initial pairing.

    Three swap operations:
      1. Replace: swap a served user with an unserved user
      2. Position: swap two served users between subcarriers/pairs
      3. Cross-UAV: swap a user between UAV0 and UAV1 pairs

    Greedy: accept any swap that improves total deadline-met count.
    Stop when no improving swap found or max_iter reached.
    """
    N = len(active_mask)
    K = h_gains.shape[1]
    pairs = {k: list(initial_pairs.get(k, [])) for k in range(K)}

    # For blind_cross_uav mode: evaluate swap moves without cross-UAV interference.
    # We zero h_gains[i, k] for all k != assignment[i] so that compute_rates_for_pairing
    # produces zero cross-interference terms when checking candidate moves.
    if blind_cross_uav:
        h_gains_eval = h_gains.copy()
        for i in range(N):
            for k in range(K):
                if k != assignment[i]:
                    h_gains_eval[i, k] = 0.0
    else:
        h_gains_eval = h_gains

    def eval_current():
        met, total, _, _, _ = evaluate_pairing(
            pairs, h_gains_eval, channel, task_data, task_comp,
            active_mask, assignment, n_sub, f_per_uav, f_local,
            t_max, sic_eps, alpha_scale)
        return met

    best_met = eval_current()

    for iteration in range(max_iter):
        improved = False

        # --- Swap Type 1: Replace (served ↔ unserved) ---
        for k in range(K):
            served_set = set()
            for pair in pairs[k]:
                if pair[0] is not None: served_set.add(pair[0])
                if pair[1] is not None: served_set.add(pair[1])

            unserved = [i for i in range(N)
                        if assignment[i] == k and active_mask[i]
                        and i not in served_set]

            for p_idx, pair in enumerate(pairs[k]):
                for pos in [0, 1]:
                    old_user = pair[pos]
                    if old_user is None:
                        continue
                    for new_user in unserved:
                        # Try replacing old_user with new_user
                        new_pair = list(pair)
                        new_pair[pos] = new_user
                        # Maintain strong/weak ordering
                        if (new_pair[0] is not None and new_pair[1] is not None
                                and h_gains[new_pair[0], k] < h_gains[new_pair[1], k]):
                            new_pair[0], new_pair[1] = new_pair[1], new_pair[0]
                        old_pairs_k = pairs[k][p_idx]
                        pairs[k][p_idx] = tuple(new_pair)
                        new_met = eval_current()
                        if new_met > best_met:
                            best_met = new_met
                            improved = True
                            # Update unserved list
                            unserved.remove(new_user)
                            unserved.append(old_user)
                            break  # accept first improvement
                        else:
                            pairs[k][p_idx] = old_pairs_k
                    if improved:
                        break
                if improved:
                    break

        # --- Swap Type 2: Position (swap within same UAV) ---
        if not improved:
            for k in range(K):
                for p1 in range(len(pairs[k])):
                    for p2 in range(p1 + 1, len(pairs[k])):
                        # Try swapping one user between pairs
                        for pos1 in [0, 1]:
                            for pos2 in [0, 1]:
                                if pairs[k][p1][pos1] is None or pairs[k][p2][pos2] is None:
                                    continue
                                old_p1, old_p2 = pairs[k][p1], pairs[k][p2]
                                new_p1 = list(old_p1)
                                new_p2 = list(old_p2)
                                new_p1[pos1], new_p2[pos2] = new_p2[pos2], new_p1[pos1]
                                # Maintain strong/weak ordering (fix: was missing in Position Swap)
                                if (new_p1[0] is not None and new_p1[1] is not None
                                        and h_gains[new_p1[0], k] < h_gains[new_p1[1], k]):
                                    new_p1[0], new_p1[1] = new_p1[1], new_p1[0]
                                if (new_p2[0] is not None and new_p2[1] is not None
                                        and h_gains[new_p2[0], k] < h_gains[new_p2[1], k]):
                                    new_p2[0], new_p2[1] = new_p2[1], new_p2[0]
                                pairs[k][p1] = tuple(new_p1)
                                pairs[k][p2] = tuple(new_p2)
                                new_met = eval_current()
                                if new_met > best_met:
                                    best_met = new_met
                                    improved = True
                                    break
                                else:
                                    pairs[k][p1] = old_p1
                                    pairs[k][p2] = old_p2
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        # --- Swap Type 3: Cross-UAV (swap between UAVs) ---
        if not improved and K >= 2 and cross_uav:
            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    for p1 in range(len(pairs[k1])):
                        for p2 in range(len(pairs[k2])):
                            for pos1 in [0, 1]:
                                for pos2 in [0, 1]:
                                    u1 = pairs[k1][p1][pos1]
                                    u2 = pairs[k2][p2][pos2]
                                    if u1 is None or u2 is None:
                                        continue
                                    old_p1 = pairs[k1][p1]
                                    old_p2 = pairs[k2][p2]
                                    new_p1 = list(old_p1)
                                    new_p2 = list(old_p2)
                                    new_p1[pos1] = u2
                                    new_p2[pos2] = u1
                                    # Reorder by channel gain for EACH UAV's serving channel
                                    if (new_p1[0] is not None and new_p1[1] is not None
                                            and h_gains[new_p1[0], k1] < h_gains[new_p1[1], k1]):
                                        new_p1[0], new_p1[1] = new_p1[1], new_p1[0]
                                    if (new_p2[0] is not None and new_p2[1] is not None
                                            and h_gains[new_p2[0], k2] < h_gains[new_p2[1], k2]):
                                        new_p2[0], new_p2[1] = new_p2[1], new_p2[0]
                                    pairs[k1][p1] = tuple(new_p1)
                                    pairs[k2][p2] = tuple(new_p2)
                                    new_met = eval_current()
                                    if new_met > best_met:
                                        best_met = new_met
                                        improved = True
                                        break
                                    else:
                                        pairs[k1][p1] = old_p1
                                        pairs[k2][p2] = old_p2
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        if not improved:
            break

    return pairs, best_met, iteration + 1
