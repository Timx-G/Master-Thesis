"""
Baselines for NOMA pairing comparison (Phase 2).

All baselines use the same evaluate_pairing() interface from pairing.py.
Each returns pairs_per_uav: {k: [(i_strong, i_weak), ...]}
"""
from __future__ import annotations
import math
from itertools import combinations
import numpy as np
from .pairing import (greedy_pairing, benefit_greedy_pairing, kkt_alpha,
                       compute_rates_for_pairing, evaluate_pairing)


def random_pairing(h_gains, active_mask, assignment, n_sub,
                    users_per_sub=2, rng=None):
    """Baseline 1: Random user selection + random pairing."""
    if rng is None:
        rng = np.random.default_rng()
    N = h_gains.shape[0]
    K = h_gains.shape[1]
    max_served = n_sub * users_per_sub
    pairs_per_uav = {}

    for k in range(K):
        candidates = [i for i in range(N)
                       if assignment[i] == k and active_mask[i]]
        rng.shuffle(candidates)
        selected = candidates[:max_served]
        pairs = []
        for j in range(0, len(selected), 2):
            if j + 1 < len(selected):
                # Order by channel gain for valid SIC
                i_a, i_b = selected[j], selected[j + 1]
                if h_gains[i_a, k] >= h_gains[i_b, k]:
                    pairs.append((i_a, i_b))
                else:
                    pairs.append((i_b, i_a))
            else:
                pairs.append((selected[j], None))
        pairs_per_uav[k] = pairs[:n_sub]

    return pairs_per_uav


def equal_alpha_pairing(h_gains, active_mask, assignment, n_sub,
                         users_per_sub=2):
    """Baseline 4: Greedy pairing + fixed alpha=0.5 (no KKT optimization).

    Returns the same pairing as greedy, but evaluate_pairing should be called
    with alpha_scale=0.5 to simulate fixed alpha.
    """
    return greedy_pairing(h_gains, active_mask, assignment, n_sub, users_per_sub)


def single_uav_pairing(h_gains_single, active_mask, n_sub, users_per_sub=2):
    """Baseline 5: Single UAV (all users assigned to UAV 0, 8GHz MEC).

    h_gains_single: (N,1) — channel to the single UAV.
    All users assigned to UAV 0.
    """
    N = h_gains_single.shape[0]
    assignment = np.zeros(N, dtype=int)
    # Use greedy pairing for single UAV
    return greedy_pairing(h_gains_single, active_mask, assignment, n_sub,
                           users_per_sub)


def ilp_optimal_pairing(h_gains, active_mask, assignment, n_sub,
                        channel=None, task_data=None, task_comp=None,
                        f_per_uav=4e9, f_local=1e9, t_max=0.8, sic_eps=0.02,
                        sic_gamma_min=1.0):
    """ILP-optimal per-UAV pairing via scipy.optimize.milp.

    For each UAV, enumerate all 2-user pairs, compute deadline benefit,
    then solve a binary ILP to select up to n_sub non-overlapping pairs
    maximizing total benefit. This is provably optimal per-UAV (but does
    not account for cross-UAV interference at the initialization stage).
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    N = h_gains.shape[0]
    K = h_gains.shape[1]
    pairs_per_uav = {}

    for k in range(K):
        candidates = [i for i in range(N)
                      if assignment[i] == k and active_mask[i]]
        if len(candidates) < 2:
            pairs_per_uav[k] = [(candidates[0], None)] if candidates else []
            continue

        all_pairs = list(combinations(candidates, 2))

        # Estimate benefit for each pair (ignoring cross-UAV interference)
        bw_sub = channel.cfg.bandwidth / n_sub if channel else 1e6
        noise_sub = (channel.noise_linear * bw_sub / channel.cfg.bandwidth
                     if channel else 1e-14)
        p_max = channel.cfg.p_max if channel else 0.2
        n_off_est = min(len(candidates), n_sub * 2)
        f_alloc = f_per_uav / max(n_off_est, 1)

        sic_feasibility_min = sic_gamma_min  # passed from caller; default = 1.0 (0 dB)
        pair_benefits = []
        pair_tuples = []
        for (i, j) in all_pairs:
            if h_gains[i, k] >= h_gains[j, k]:
                i_s, i_w = i, j
            else:
                i_s, i_w = j, i
            recv_s = p_max * h_gains[i_s, k]
            recv_w = p_max * h_gains[i_w, k]
            sinr_s = recv_s / (recv_w + noise_sub)
            # Hard SIC feasibility filter: reject pairs where strong user SINR < γ_SIC_min
            if sinr_s < sic_feasibility_min:
                continue  # skip this pair — SIC not feasible
            sinr_w = recv_w / (sic_eps * recv_s + noise_sub)
            R_s = bw_sub * math.log2(1 + max(sinr_s, 1e-10))
            R_w = bw_sub * math.log2(1 + max(sinr_w, 1e-10))

            benefit = 0
            for user_i, R_i in [(i_s, R_s), (i_w, R_w)]:
                if task_data is not None and task_comp is not None:
                    alpha = kkt_alpha(task_data[user_i], task_comp[user_i],
                                      R_i, f_alloc, f_local)
                    t_loc = (1 - alpha) * task_comp[user_i] / f_local
                    t_off = ((1 + 0.1) * alpha * task_data[user_i] / R_i +
                             alpha * task_comp[user_i] / f_alloc) if alpha > 0 else 0
                    delay = max(t_loc, t_off)
                    if delay <= t_max:
                        benefit += 1
                else:
                    benefit += R_i
            pair_benefits.append(benefit)
            pair_tuples.append((i_s, i_w))

        n_pairs = len(pair_tuples)
        if n_pairs == 0:
            pairs_per_uav[k] = []
            continue

        # Solve ILP: maximize sum(benefit[p] * x[p])
        # subject to: each user in at most 1 pair, at most n_sub pairs
        c = np.array([-b for b in pair_benefits], dtype=float)  # negate for min

        # User-disjointness constraints: for each candidate user,
        # sum of x[p] for pairs containing that user <= 1
        cand_set = set(candidates)
        A_rows = []
        for u in candidates:
            row = np.zeros(n_pairs)
            for p_idx, (i_s, i_w) in enumerate(pair_tuples):
                if i_s == u or i_w == u:
                    row[p_idx] = 1.0
            A_rows.append(row)
        # Total pairs <= n_sub
        A_rows.append(np.ones(n_pairs))
        A = np.array(A_rows)
        b_ub = np.ones(len(A_rows))
        b_ub[-1] = n_sub

        constraints = LinearConstraint(A, ub=b_ub)
        integrality = np.ones(n_pairs)  # all binary
        bounds = Bounds(lb=0, ub=1)

        result = milp(c, constraints=constraints, integrality=integrality,
                       bounds=bounds)

        if result.success:
            x = result.x
            selected = [pair_tuples[p] for p in range(n_pairs)
                        if x[p] > 0.5]
        else:
            # Fallback to greedy if ILP fails
            sorted_pairs = sorted(zip(pair_tuples, pair_benefits),
                                  key=lambda x: x[1], reverse=True)
            selected = []
            used = set()
            for (pair, ben) in sorted_pairs:
                if pair[0] in used or pair[1] in used:
                    continue
                selected.append(pair)
                used.add(pair[0])
                used.add(pair[1])
                if len(selected) >= n_sub:
                    break

        pairs_per_uav[k] = selected[:n_sub]

    return pairs_per_uav


def worst_case_pairing(h_gains, active_mask, assignment, n_sub,
                       users_per_sub=2):
    """Adversarial pairing: pair users with SIMILAR channel gains (worst SIC).

    This is the worst case for NOMA: similar channel gains means poor SIC
    decoding diversity. Used to test swap robustness to bad initialization.
    """
    N = h_gains.shape[0]
    K = h_gains.shape[1]
    pairs_per_uav = {}

    for k in range(K):
        candidates = [i for i in range(N)
                      if assignment[i] == k and active_mask[i]]
        candidates.sort(key=lambda i: h_gains[i, k], reverse=True)
        selected = candidates[:n_sub * users_per_sub]

        # Pair adjacent users (similar channel gains = worst for SIC)
        pairs = []
        for j in range(0, len(selected) - 1, 2):
            i_a, i_b = selected[j], selected[j + 1]
            if h_gains[i_a, k] >= h_gains[i_b, k]:
                pairs.append((i_a, i_b))
            else:
                pairs.append((i_b, i_a))
        pairs_per_uav[k] = pairs[:n_sub]

    return pairs_per_uav


def exchange_stable_pairing(h_gains, active_mask, assignment, n_sub,
                            channel=None, task_data=None, task_comp=None,
                            f_per_uav=4e9, f_local=1e9, t_max=0.8,
                            sic_eps=0.02, max_rounds=20):
    """Exchange-stable matching baseline (TES-style) with cross-UAV interference.

    Start from greedy pairing, then iteratively check all pairwise exchanges
    between any two pairs (within same UAV). A swap is accepted if BOTH pairs
    are weakly better off and at least one is strictly better (exchange stability).
    Unlike our swap_optimize which maximizes global deadline count, this only
    accepts swaps where both involved pairs benefit (stability condition).

    Cross-UAV interference is correctly modelled: when evaluating a pair on
    subcarrier sub_idx of UAV k, interference from all other UAVs' users on
    the same subcarrier is included in the SINR calculation.

    This is the standard approach from the NOMA matching literature.
    """
    N = h_gains.shape[0]
    K = h_gains.shape[1]

    # Start with greedy initialization
    pairs_per_uav = greedy_pairing(h_gains, active_mask, assignment, n_sub)

    bw_sub = channel.cfg.bandwidth / n_sub if channel else 1e6
    noise_sub = (channel.noise_linear * bw_sub / channel.cfg.bandwidth
                 if channel else 1e-14)
    p_max = channel.cfg.p_max if channel else 0.2

    def cross_interf_for_sub(sub_idx, k):
        """Sum received interference at UAV-k from other UAVs on subcarrier sub_idx."""
        cross = 0.0
        for k2 in range(K):
            if k2 == k:
                continue
            k2_pairs = pairs_per_uav.get(k2, [])
            if sub_idx < len(k2_pairs):
                i_s2, i_w2 = k2_pairs[sub_idx]
                if i_s2 is not None:
                    cross += p_max * h_gains[i_s2, k]
                if i_w2 is not None:
                    cross += p_max * h_gains[i_w2, k]
        return cross

    def pair_benefit(i_s, i_w, k, cross_interf):
        """Compute deadline-met count for a single pair given cross interference."""
        if i_s is None:
            return 0
        recv_s = p_max * h_gains[i_s, k]
        if i_w is not None:
            recv_w = p_max * h_gains[i_w, k]
            sinr_s = recv_s / (recv_w + cross_interf + noise_sub)
            sinr_w = recv_w / (sic_eps * recv_s + cross_interf + noise_sub)
            R_s = bw_sub * math.log2(1 + max(sinr_s, 1e-10))
            R_w = bw_sub * math.log2(1 + max(sinr_w, 1e-10))
        else:
            sinr_s = recv_s / (cross_interf + noise_sub)
            R_s = bw_sub * math.log2(1 + max(sinr_s, 1e-10))
            R_w = 0.0

        n_off_est = min(sum(1 for i in range(N)
                            if assignment[i] == k and active_mask[i]),
                        n_sub * 2)
        f_alloc = f_per_uav / max(n_off_est, 1)
        benefit = 0
        for user_i, R_i in [(i_s, R_s), (i_w, R_w)]:
            if user_i is None or R_i <= 1.0:
                continue
            if task_data is not None and task_comp is not None:
                alpha = kkt_alpha(task_data[user_i], task_comp[user_i],
                                  R_i, f_alloc, f_local)
                t_loc = (1 - alpha) * task_comp[user_i] / f_local
                t_off = ((1 + 0.1) * alpha * task_data[user_i] / R_i +
                         alpha * task_comp[user_i] / f_alloc) if alpha > 0 else 0
                delay = max(t_loc, t_off)
                if delay <= t_max:
                    benefit += 1
            else:
                benefit += 1
        return benefit

    # Iterative exchange stability check
    for _ in range(max_rounds):
        stable = True

        for k in range(K):
            pairs = pairs_per_uav[k]
            # Also try swaps with unserved users
            served_set = set()
            for pair in pairs:
                if pair[0] is not None:
                    served_set.add(pair[0])
                if pair[1] is not None:
                    served_set.add(pair[1])
            unserved = [i for i in range(N)
                        if assignment[i] == k and active_mask[i]
                        and i not in served_set]

            # Intra-UAV pairwise exchange
            for p1 in range(len(pairs)):
                for p2 in range(p1 + 1, len(pairs)):
                    # Current benefits including cross-UAV interference
                    ci_p1 = cross_interf_for_sub(p1, k)
                    ci_p2 = cross_interf_for_sub(p2, k)
                    b1_old = pair_benefit(pairs[p1][0], pairs[p1][1], k, ci_p1)
                    b2_old = pair_benefit(pairs[p2][0], pairs[p2][1], k, ci_p2)

                    # Try all user exchanges between pairs
                    for pos1 in [0, 1]:
                        for pos2 in [0, 1]:
                            if pairs[p1][pos1] is None or pairs[p2][pos2] is None:
                                continue
                            new_p1 = list(pairs[p1])
                            new_p2 = list(pairs[p2])
                            new_p1[pos1], new_p2[pos2] = new_p2[pos2], new_p1[pos1]

                            # Re-order by channel gain
                            if (new_p1[0] is not None and new_p1[1] is not None
                                    and h_gains[new_p1[0], k] < h_gains[new_p1[1], k]):
                                new_p1[0], new_p1[1] = new_p1[1], new_p1[0]
                            if (new_p2[0] is not None and new_p2[1] is not None
                                    and h_gains[new_p2[0], k] < h_gains[new_p2[1], k]):
                                new_p2[0], new_p2[1] = new_p2[1], new_p2[0]

                            b1_new = pair_benefit(new_p1[0], new_p1[1], k, ci_p1)
                            b2_new = pair_benefit(new_p2[0], new_p2[1], k, ci_p2)

                            # Exchange stability: both weakly better, at least one strict
                            if (b1_new >= b1_old and b2_new >= b2_old
                                    and b1_new + b2_new > b1_old + b2_old):
                                pairs[p1] = tuple(new_p1)
                                pairs[p2] = tuple(new_p2)
                                stable = False
                                break
                        if not stable:
                            break
                    if not stable:
                        break
                if not stable:
                    break

            # Replace: swap served user with unserved (if beneficial for the pair)
            if stable:
                for p_idx, pair in enumerate(pairs):
                    for pos in [0, 1]:
                        old_user = pair[pos]
                        if old_user is None:
                            continue
                        ci = cross_interf_for_sub(p_idx, k)
                        b_old = pair_benefit(pair[0], pair[1], k, ci)
                        for new_user in unserved:
                            new_pair = list(pair)
                            new_pair[pos] = new_user
                            if (new_pair[0] is not None and new_pair[1] is not None
                                    and h_gains[new_pair[0], k] < h_gains[new_pair[1], k]):
                                new_pair[0], new_pair[1] = new_pair[1], new_pair[0]
                            b_new = pair_benefit(new_pair[0], new_pair[1], k, ci)
                            if b_new > b_old:
                                pairs[p_idx] = tuple(new_pair)
                                unserved.remove(new_user)
                                unserved.append(old_user)
                                stable = False
                                break
                        if not stable:
                            break
                    if not stable:
                        break

            pairs_per_uav[k] = pairs

        if stable:
            break

    return pairs_per_uav


def ofdma_rates(h_gains, active_mask, assignment, channel, n_sub):
    """Baseline 6: Pure OFDMA (no NOMA, one user per subcarrier).

    Each UAV gets n_sub subcarriers, each subcarrier serves 1 user (no pairing).
    No intra-cell interference, no inter-UAV interference on different subcarriers.
    """
    N = h_gains.shape[0]
    K = h_gains.shape[1]
    rates = np.zeros(N)
    bw_sub = channel.cfg.bandwidth / n_sub
    noise_sub = channel.noise_linear * (bw_sub / channel.cfg.bandwidth)
    p_max = channel.cfg.p_max

    # Each UAV: top n_sub users by channel, one per subcarrier
    served = np.zeros(N, dtype=bool)
    for k in range(K):
        candidates = [i for i in range(N)
                       if assignment[i] == k and active_mask[i]]
        candidates.sort(key=lambda i: h_gains[i, k], reverse=True)
        selected = candidates[:n_sub]  # only 1 user per sub (OMA)

        for i in selected:
            served[i] = True
            recv = p_max * h_gains[i, k]
            # No intra interference (OMA), but cross-UAV interference still exists
            # since both UAVs share all subcarriers
            cross = 0.0
            for k2 in range(K):
                if k2 == k:
                    continue
                # Other UAV's user on same subcarrier
                cands_k2 = [j for j in range(N)
                             if assignment[j] == k2 and active_mask[j]]
                cands_k2.sort(key=lambda j: h_gains[j, k2], reverse=True)
                sub_idx = selected.index(i)
                if sub_idx < len(cands_k2):
                    j = cands_k2[sub_idx]
                    cross += p_max * h_gains[j, k]

            sinr = recv / (cross + noise_sub)
            rates[i] = bw_sub * math.log2(1 + max(sinr, 1e-10))

    return rates, served


def ofdma_ortho_rates(h_gains, active_mask, assignment, channel, n_sub):
    """Orthogonal-band OFDMA baseline: each UAV occupies a non-overlapping sub-band.

    The total bandwidth B is divided equally among K UAVs (each gets B/K).
    Within each UAV's sub-band there are n_sub subcarriers (each B/(K*n_sub) wide).
    There is NO cross-UAV interference because UAVs use disjoint frequency segments.
    Intra-UAV allocation is OFDMA (one user per subcarrier, no SIC needed).

    This baseline represents the engineering practice of orthogonal frequency
    partitioning to eliminate inter-UAV interference. Comparing against it isolates
    the net gain of NOMA's spectral reuse over interference-free OFDMA.
    """
    N = h_gains.shape[0]
    K = h_gains.shape[1]
    rates = np.zeros(N)
    # Each UAV's sub-band is B/K; each subcarrier within it is B/(K*n_sub)
    bw_sub = channel.cfg.bandwidth / (K * n_sub)
    noise_sub = channel.noise_linear * (bw_sub / channel.cfg.bandwidth)
    p_max = channel.cfg.p_max

    served = np.zeros(N, dtype=bool)
    for k in range(K):
        candidates = [i for i in range(N)
                       if assignment[i] == k and active_mask[i]]
        candidates.sort(key=lambda i: h_gains[i, k], reverse=True)
        selected = candidates[:n_sub]  # one user per subcarrier, OMA

        for i in selected:
            served[i] = True
            recv = p_max * h_gains[i, k]
            # No cross-UAV interference (orthogonal bands)
            sinr = recv / noise_sub
            rates[i] = bw_sub * math.log2(1 + max(sinr, 1e-10))

    return rates, served


def ofdma_fair_rates(h_gains, active_mask, assignment, channel, n_sub):
    """Fair OFDMA: same number of users as NOMA via TDMA within subcarrier.
    Each subcarrier serves 2 users in time-division (half time each).
    Effective bandwidth per user = B_s/2, no SIC required.
    Total served = 2*n_sub*K = same as NOMA.
    """
    N = h_gains.shape[0]
    K = h_gains.shape[1]
    rates = np.zeros(N)
    bw_eff = channel.cfg.bandwidth / (n_sub * 2)  # half subcarrier BW per user (TDMA)
    noise_eff = channel.noise_linear * (bw_eff / channel.cfg.bandwidth)
    p_max = channel.cfg.p_max

    served = np.zeros(N, dtype=bool)
    for k in range(K):
        candidates = [i for i in range(N) if assignment[i] == k and active_mask[i]]
        candidates.sort(key=lambda i: h_gains[i, k], reverse=True)
        selected = candidates[:n_sub * 2]  # 2 users per subcarrier

        for i in selected:
            served[i] = True
            recv = p_max * h_gains[i, k]
            # Cross-UAV interference (same subcarrier reuse)
            cross = 0.0
            for k2 in range(K):
                if k2 == k:
                    continue
                cands_k2 = [j for j in range(N) if assignment[j] == k2 and active_mask[j]]
                cands_k2.sort(key=lambda j: h_gains[j, k2], reverse=True)
                sub_idx = selected.index(i) // 2  # which subcarrier
                # Other UAV's 2 users on same subcarrier
                start = sub_idx * 2
                for j in cands_k2[start:start+2]:
                    cross += p_max * h_gains[j, k]

            sinr = recv / (cross + noise_eff)
            rates[i] = bw_eff * math.log2(1 + max(sinr, 1e-10))

    return rates, served


def ia_greedy_pairing(h_gains, active_mask, assignment, n_sub,
                      channel=None, task_data=None, task_comp=None,
                      f_per_uav=4e9, f_local=1e9, t_max=0.8, sic_eps=0.02):
    """Interference-Aware Greedy (IA-Greedy) baseline — required by Reviewer 3.

    At each greedy step, selects the candidate pair that minimises cross-UAV
    interference on its subcarrier, breaking ties by deadline benefit.
    No swap refinement is applied.

    Concretely:
      For each UAV k, iterate over subcarrier slots 0..n_sub-1.
      For each remaining slot, score all unused-user pairs by:
          score = -I_cross_estimate + beta * benefit
      where I_cross_estimate is the expected cross-interference contribution of
      this pair on the same subcarrier from the OTHER UAV's current best user
      assignment, and benefit is the deadline-met count for the pair.
      Beta = 0.1 keeps interference minimisation as primary criterion.

    This provides a strict greedy that considers cross-UAV interference during
    pair selection — distinct from BenefitGreedy (which ignores cross-interference)
    and from our full algorithm (which uses ILP init + swap refinement).
    """
    N = h_gains.shape[0]
    K = h_gains.shape[1]
    bw_sub = channel.cfg.bandwidth / n_sub if channel else 1e6
    noise_sub = (channel.noise_linear * bw_sub / channel.cfg.bandwidth
                 if channel else 1e-14)
    p_max = channel.cfg.p_max if channel else 0.2
    pairs_per_uav = {k: [] for k in range(K)}

    # For each UAV, build a per-subcarrier list of other UAVs' candidate pairs
    # so we can estimate cross-UAV interference when scoring.
    # Strategy: first assign each UAV's strongest user to each subcarrier
    # independently as a first-pass interference estimate.
    first_pass = {}
    for k in range(K):
        cands = sorted([i for i in range(N) if assignment[i] == k and active_mask[i]],
                       key=lambda i: h_gains[i, k], reverse=True)
        first_pass[k] = cands  # sorted by channel strength

    for k in range(K):
        candidates = [i for i in range(N)
                      if assignment[i] == k and active_mask[i]]
        if len(candidates) < 2:
            if candidates:
                pairs_per_uav[k] = [(candidates[0], None)]
            else:
                pairs_per_uav[k] = []
            continue

        n_off_est = min(len(candidates), n_sub * 2)
        f_alloc = f_per_uav / max(n_off_est, 1)
        used = set()
        pairs_k = []

        for sub_idx in range(n_sub):
            if len(used) >= len(candidates):
                break
            remaining = [i for i in candidates if i not in used]
            if len(remaining) < 2:
                if remaining:
                    pairs_k.append((remaining[0], None))
                    used.add(remaining[0])
                break

            # Estimate cross-UAV interference from other UAVs on this sub_idx
            # using the first-pass strongest-user estimate.
            cross_est = 0.0
            for k2 in range(K):
                if k2 == k:
                    continue
                fp = first_pass[k2]
                # 2 users per subcarrier from k2 on this sub (strong + weak)
                strong_idx = sub_idx * 2
                weak_idx = sub_idx * 2 + 1
                if strong_idx < len(fp):
                    cross_est += p_max * h_gains[fp[strong_idx], k]
                if weak_idx < len(fp):
                    cross_est += p_max * h_gains[fp[weak_idx], k]

            # Score all candidate pairs by interference reduction + benefit
            best_score = None
            best_pair = None
            for idx_a in range(len(remaining)):
                for idx_b in range(idx_a + 1, len(remaining)):
                    i, j = remaining[idx_a], remaining[idx_b]
                    i_s = i if h_gains[i, k] >= h_gains[j, k] else j
                    i_w = j if h_gains[i, k] >= h_gains[j, k] else i

                    recv_s = p_max * h_gains[i_s, k]
                    recv_w = p_max * h_gains[i_w, k]
                    sinr_s = recv_s / (recv_w + cross_est + noise_sub)
                    if sinr_s < 1.0:
                        continue  # SIC feasibility filter
                    sinr_w = recv_w / (sic_eps * recv_s + cross_est + noise_sub)
                    R_s = bw_sub * math.log2(1 + max(sinr_s, 1e-10))
                    R_w = bw_sub * math.log2(1 + max(sinr_w, 1e-10))

                    # Benefit (deadline-met count)
                    benefit = 0
                    for user_i, R_i in [(i_s, R_s), (i_w, R_w)]:
                        if task_data is not None and task_comp is not None:
                            alpha = kkt_alpha(task_data[user_i], task_comp[user_i],
                                              R_i, f_alloc, f_local)
                            t_loc = (1 - alpha) * task_comp[user_i] / f_local
                            t_off = ((1 + 0.1) * alpha * task_data[user_i] / R_i +
                                     alpha * task_comp[user_i] / f_alloc) if alpha > 0 else 0
                            if max(t_loc, t_off) <= t_max:
                                benefit += 1
                        else:
                            benefit += R_i

                    # IA-Greedy score: minimise interference on this subcarrier
                    # by picking the pair that adds least interference to OTHER UAVs.
                    # Proxy: minimise sum of h_gains of this pair to non-serving UAVs.
                    interf_contribution = 0.0
                    for k2 in range(K):
                        if k2 == k:
                            continue
                        interf_contribution += (p_max * h_gains[i_s, k2] +
                                                p_max * h_gains[i_w, k2])

                    # Primary: minimise interference; secondary: maximise benefit
                    score = -interf_contribution + 0.1 * benefit

                    if best_score is None or score > best_score:
                        best_score = score
                        best_pair = (i_s, i_w)

            if best_pair is None:
                # Fallback: pick by channel gain only
                remaining_sorted = sorted(remaining, key=lambda i: h_gains[i, k],
                                          reverse=True)
                best_pair = (remaining_sorted[0], remaining_sorted[1])
                if h_gains[best_pair[0], k] < h_gains[best_pair[1], k]:
                    best_pair = (best_pair[1], best_pair[0])

            pairs_k.append(best_pair)
            used.add(best_pair[0])
            if best_pair[1] is not None:
                used.add(best_pair[1])

        pairs_per_uav[k] = pairs_k

    return pairs_per_uav


def exchange_stable_cross_uav_pairing(h_gains, active_mask, assignment, n_sub,
                                       channel=None, task_data=None, task_comp=None,
                                       f_per_uav=4e9, f_local=1e9, t_max=0.8,
                                       sic_eps=0.02, max_rounds=20):
    """Exchange-stable matching WITH cross-UAV swaps (ablation R3-C3).

    Same bilateral-improvement stability criterion as exchange_stable_pairing,
    but extended to also consider cross-UAV swap moves (Type-3 swaps).

    This isolates whether ExStable's gap vs Ours is due to:
      (a) Lack of cross-UAV move capability, OR
      (b) Conservative bilateral improvement criterion.

    Accept a swap (intra or cross-UAV) if BOTH involved pairs are weakly
    better off and at least one is strictly better — the exchange stability
    condition. Unlike swap_optimize which maximises the global deadline count.
    """
    from .pairing import evaluate_pairing

    N = h_gains.shape[0]
    K = h_gains.shape[1]

    # Start with greedy initialization
    pairs_per_uav = greedy_pairing(h_gains, active_mask, assignment, n_sub)

    bw_sub = channel.cfg.bandwidth / n_sub if channel else 1e6
    noise_sub = (channel.noise_linear * bw_sub / channel.cfg.bandwidth
                 if channel else 1e-14)
    p_max = channel.cfg.p_max if channel else 0.2

    def cross_interf_for_sub(sub_idx, k, current_pairs):
        """Cross-UAV interference at UAV-k on subcarrier sub_idx."""
        cross = 0.0
        for k2 in range(K):
            if k2 == k:
                continue
            k2_pairs = current_pairs.get(k2, [])
            if sub_idx < len(k2_pairs):
                i_s2, i_w2 = k2_pairs[sub_idx]
                if i_s2 is not None:
                    cross += p_max * h_gains[i_s2, k]
                if i_w2 is not None:
                    cross += p_max * h_gains[i_w2, k]
        return cross

    def pair_benefit_with_interf(i_s, i_w, k, cross_interf):
        """Deadline-met count for a single pair given cross interference."""
        if i_s is None:
            return 0
        recv_s = p_max * h_gains[i_s, k]
        if i_w is not None:
            recv_w = p_max * h_gains[i_w, k]
            sinr_s = recv_s / (recv_w + cross_interf + noise_sub)
            sinr_w = recv_w / (sic_eps * recv_s + cross_interf + noise_sub)
            R_s = bw_sub * math.log2(1 + max(sinr_s, 1e-10))
            R_w = bw_sub * math.log2(1 + max(sinr_w, 1e-10))
        else:
            sinr_s = recv_s / (cross_interf + noise_sub)
            R_s = bw_sub * math.log2(1 + max(sinr_s, 1e-10))
            R_w = 0.0

        n_off_est = min(sum(1 for i in range(N)
                            if assignment[i] == k and active_mask[i]),
                        n_sub * 2)
        f_alloc = f_per_uav / max(n_off_est, 1)
        benefit = 0
        for user_i, R_i in [(i_s, R_s), (i_w, R_w)]:
            if user_i is None or R_i <= 1.0:
                continue
            if task_data is not None and task_comp is not None:
                alpha = kkt_alpha(task_data[user_i], task_comp[user_i],
                                  R_i, f_alloc, f_local)
                t_loc = (1 - alpha) * task_comp[user_i] / f_local
                t_off = ((1 + 0.1) * alpha * task_data[user_i] / R_i +
                         alpha * task_comp[user_i] / f_alloc) if alpha > 0 else 0
                delay = max(t_loc, t_off)
                if delay <= t_max:
                    benefit += 1
            else:
                benefit += 1
        return benefit

    # Iterative exchange stability check (intra + cross-UAV)
    for _ in range(max_rounds):
        stable = True

        # --- Intra-UAV exchanges (same as exchange_stable_pairing) ---
        for k in range(K):
            pairs = pairs_per_uav[k]
            served_set = set()
            for pair in pairs:
                if pair[0] is not None: served_set.add(pair[0])
                if pair[1] is not None: served_set.add(pair[1])
            unserved = [i for i in range(N)
                        if assignment[i] == k and active_mask[i]
                        and i not in served_set]

            # Intra-UAV pairwise exchange
            for p1 in range(len(pairs)):
                for p2 in range(p1 + 1, len(pairs)):
                    ci_p1 = cross_interf_for_sub(p1, k, pairs_per_uav)
                    ci_p2 = cross_interf_for_sub(p2, k, pairs_per_uav)
                    b1_old = pair_benefit_with_interf(pairs[p1][0], pairs[p1][1], k, ci_p1)
                    b2_old = pair_benefit_with_interf(pairs[p2][0], pairs[p2][1], k, ci_p2)

                    for pos1 in [0, 1]:
                        for pos2 in [0, 1]:
                            if pairs[p1][pos1] is None or pairs[p2][pos2] is None:
                                continue
                            new_p1 = list(pairs[p1])
                            new_p2 = list(pairs[p2])
                            new_p1[pos1], new_p2[pos2] = new_p2[pos2], new_p1[pos1]
                            if (new_p1[0] is not None and new_p1[1] is not None
                                    and h_gains[new_p1[0], k] < h_gains[new_p1[1], k]):
                                new_p1[0], new_p1[1] = new_p1[1], new_p1[0]
                            if (new_p2[0] is not None and new_p2[1] is not None
                                    and h_gains[new_p2[0], k] < h_gains[new_p2[1], k]):
                                new_p2[0], new_p2[1] = new_p2[1], new_p2[0]
                            b1_new = pair_benefit_with_interf(new_p1[0], new_p1[1], k, ci_p1)
                            b2_new = pair_benefit_with_interf(new_p2[0], new_p2[1], k, ci_p2)
                            if (b1_new >= b1_old and b2_new >= b2_old
                                    and b1_new + b2_new > b1_old + b2_old):
                                pairs[p1] = tuple(new_p1)
                                pairs[p2] = tuple(new_p2)
                                stable = False
                                break
                        if not stable:
                            break
                    if not stable:
                        break
                if not stable:
                    break

            # Intra-UAV replace: swap served with unserved (bilateral: the pair benefits)
            if stable:
                for p_idx, pair in enumerate(pairs):
                    for pos in [0, 1]:
                        old_user = pair[pos]
                        if old_user is None:
                            continue
                        ci = cross_interf_for_sub(p_idx, k, pairs_per_uav)
                        b_old = pair_benefit_with_interf(pair[0], pair[1], k, ci)
                        for new_user in unserved:
                            new_pair = list(pair)
                            new_pair[pos] = new_user
                            if (new_pair[0] is not None and new_pair[1] is not None
                                    and h_gains[new_pair[0], k] < h_gains[new_pair[1], k]):
                                new_pair[0], new_pair[1] = new_pair[1], new_pair[0]
                            b_new = pair_benefit_with_interf(new_pair[0], new_pair[1], k, ci)
                            if b_new > b_old:
                                pairs[p_idx] = tuple(new_pair)
                                unserved.remove(new_user)
                                unserved.append(old_user)
                                stable = False
                                break
                        if not stable:
                            break
                    if not stable:
                        break

            pairs_per_uav[k] = pairs
            if not stable:
                break

        # --- Cross-UAV exchanges (key extension over exchange_stable_pairing) ---
        if stable and K >= 2:
            for k1 in range(K):
                for k2 in range(k1 + 1, K):
                    pairs1 = pairs_per_uav[k1]
                    pairs2 = pairs_per_uav[k2]
                    for p1 in range(len(pairs1)):
                        for p2 in range(len(pairs2)):
                            for pos1 in [0, 1]:
                                for pos2 in [0, 1]:
                                    u1 = pairs1[p1][pos1]
                                    u2 = pairs2[p2][pos2]
                                    if u1 is None or u2 is None:
                                        continue

                                    # Cross-UAV interference BEFORE swap
                                    ci1_old = cross_interf_for_sub(p1, k1, pairs_per_uav)
                                    ci2_old = cross_interf_for_sub(p2, k2, pairs_per_uav)
                                    b1_old = pair_benefit_with_interf(
                                        pairs1[p1][0], pairs1[p1][1], k1, ci1_old)
                                    b2_old = pair_benefit_with_interf(
                                        pairs2[p2][0], pairs2[p2][1], k2, ci2_old)

                                    # Build proposed new pairs
                                    new_p1 = list(pairs1[p1])
                                    new_p2 = list(pairs2[p2])
                                    new_p1[pos1] = u2
                                    new_p2[pos2] = u1

                                    # Re-order by channel gain for serving UAV
                                    if (new_p1[0] is not None and new_p1[1] is not None
                                            and h_gains[new_p1[0], k1] < h_gains[new_p1[1], k1]):
                                        new_p1[0], new_p1[1] = new_p1[1], new_p1[0]
                                    if (new_p2[0] is not None and new_p2[1] is not None
                                            and h_gains[new_p2[0], k2] < h_gains[new_p2[1], k2]):
                                        new_p2[0], new_p2[1] = new_p2[1], new_p2[0]

                                    # Temporarily apply swap to compute new cross-interference
                                    orig_p1 = pairs_per_uav[k1][p1]
                                    orig_p2 = pairs_per_uav[k2][p2]
                                    pairs_per_uav[k1][p1] = tuple(new_p1)
                                    pairs_per_uav[k2][p2] = tuple(new_p2)

                                    ci1_new = cross_interf_for_sub(p1, k1, pairs_per_uav)
                                    ci2_new = cross_interf_for_sub(p2, k2, pairs_per_uav)
                                    b1_new = pair_benefit_with_interf(
                                        new_p1[0], new_p1[1], k1, ci1_new)
                                    b2_new = pair_benefit_with_interf(
                                        new_p2[0], new_p2[1], k2, ci2_new)

                                    # Exchange stability: bilateral improvement
                                    if (b1_new >= b1_old and b2_new >= b2_old
                                            and b1_new + b2_new > b1_old + b2_old):
                                        # Accept: pairs already updated
                                        stable = False
                                        break
                                    else:
                                        # Revert
                                        pairs_per_uav[k1][p1] = orig_p1
                                        pairs_per_uav[k2][p2] = orig_p2
                                if not stable:
                                    break
                            if not stable:
                                break
                        if not stable:
                            break
                    if not stable:
                        break
                if not stable:
                    break

        if stable:
            break

    return pairs_per_uav


def hungarian_pairing(h_gains, active_mask, assignment, n_sub,
                      users_per_sub=2, channel=None, task_data=None,
                      task_comp=None, f_per_uav=4e9, f_local=1e9,
                      t_max=0.8, sic_eps=0.02):
    """[ABLATION BASELINE ONLY] Polynomial-time user pairing via weighted bipartite matching.

    NOTE: This function is NOT used by the proposed algorithm (Ours = ILP+Swap).
    It is retained as an ablation baseline to show that Hungarian initialization
    achieves 71.6% vs. ILP initialization at 73.3% — demonstrating that ILP
    provides a better starting point for the swap refinement.

    Uses scipy.optimize.linear_sum_assignment (O(N^3) Hungarian algorithm).
    Performance gap vs ILP: ~1.7pp before swap, ~0pp after swap due to swap dominance.

    by scipy.optimize.linear_sum_assignment in O(N^3). For each UAV:
      1. Compute deadline benefit for all C(M_k,2) candidate pairs.
      2. Build an (n_sub x P) cost matrix (P = number of candidate pairs).
      3. Run linear_sum_assignment to assign subcarrier slots to pairs.
      4. Post-process to enforce user non-overlap (greedy selection by benefit).

    This replaces the ILP (scipy.optimize.milp) with a polynomial-time solver,
    maintaining O(N^2) pair enumeration + O(n_sub^3) assignment per UAV.
    """
    from scipy.optimize import linear_sum_assignment

    N = h_gains.shape[0]
    K = h_gains.shape[1]
    pairs_per_uav = {}

    for k in range(K):
        candidates = [i for i in range(N)
                      if assignment[i] == k and active_mask[i]]
        if len(candidates) < 2:
            pairs_per_uav[k] = [(candidates[0], None)] if candidates else []
            continue

        all_pairs = list(combinations(candidates, 2))
        bw_sub = channel.cfg.bandwidth / n_sub if channel else 1e6
        noise_sub = (channel.noise_linear * bw_sub / channel.cfg.bandwidth
                     if channel else 1e-14)
        p_max = channel.cfg.p_max if channel else 0.2
        n_off_est = min(len(candidates), n_sub * 2)
        f_alloc = f_per_uav / max(n_off_est, 1)

        # Compute deadline benefit for each candidate pair
        benefits = []
        valid_pairs = []
        for (i, j) in all_pairs:
            i_s = i if h_gains[i, k] >= h_gains[j, k] else j
            i_w = j if h_gains[i, k] >= h_gains[j, k] else i
            recv_s = p_max * h_gains[i_s, k]
            recv_w = p_max * h_gains[i_w, k]
            sinr_s = recv_s / (recv_w + noise_sub)
            if sinr_s < 1.0:  # SIC feasibility filter
                benefits.append(0.0)
                valid_pairs.append((i_s, i_w))
                continue
            sinr_w = recv_w / (sic_eps * recv_s + noise_sub)
            R_s = bw_sub * math.log2(1 + max(sinr_s, 1e-10))
            R_w = bw_sub * math.log2(1 + max(sinr_w, 1e-10))
            benefit = 0.0
            for user_i, R_i in [(i_s, R_s), (i_w, R_w)]:
                if task_data is not None and task_comp is not None and R_i > 1.0:
                    alpha = kkt_alpha(task_data[user_i], task_comp[user_i],
                                      R_i, f_alloc, f_local)
                    D = task_data[user_i]; C = task_comp[user_i]
                    t_loc = (1 - alpha) * C / f_local
                    t_off = ((1 + 0.1) * alpha * D / R_i +
                             alpha * C / f_alloc) if alpha > 0 else 0
                    if max(t_loc, t_off) <= t_max:
                        benefit += 1.0
                else:
                    benefit += R_s + R_w
            benefits.append(benefit)
            valid_pairs.append((i_s, i_w))

        benefits = np.array(benefits)
        P = len(valid_pairs)
        if P == 0:
            pairs_per_uav[k] = []
            continue

        # Build cost matrix: rows=subcarrier slots, cols=candidate pairs
        # linear_sum_assignment minimizes cost, so negate benefits
        n_slots = min(n_sub, P)
        cost_matrix = np.zeros((n_slots, P))
        for s in range(n_slots):
            cost_matrix[s, :] = -benefits  # uniform across subcarriers

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Post-process: enforce user non-overlap (greedy by benefit)
        selected = sorted([(col_ind[r], benefits[col_ind[r]])
                           for r in range(len(row_ind))],
                          key=lambda x: -x[1])
        chosen_pairs = []
        used = set()
        for p_idx, _ in selected:
            i_s, i_w = valid_pairs[p_idx]
            if i_s in used or i_w in used:
                continue
            chosen_pairs.append((i_s, i_w))
            used.add(i_s)
            used.add(i_w)
            if len(chosen_pairs) >= n_sub:
                break

        pairs_per_uav[k] = chosen_pairs

    return pairs_per_uav
