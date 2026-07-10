"""
NF (Noise Figure) and SIC γ_min sensitivity experiments.

Required by Reviewer 1:
  - NF sensitivity: NF = 5, 7, 9 dB
  - SIC γ_min sensitivity: γ_SIC_min = 1 (0dB), 3dB, 5dB (linear: 1, 2, ~3.16)

Usage: python -m uav_mec.run_nf_sic_sensitivity
"""
import time, json
import numpy as np
from pathlib import Path

from .model import ScenarioConfig, ChannelConfig
from .multi_uav_env import MultiUAVMECEnv
from .run_main_experiment import run_episode, strategy_ours, strategy_greedy


def make_cfg_with_nf(nf_db: float) -> ScenarioConfig:
    """Create ScenarioConfig with custom noise figure."""
    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                         user_mobility=True)
    # Override channel noise by setting custom noise_power_dbm
    # NF total: thermal_5MHz (-107 dBm) + NF, per-subcarrier: add 10*log10(5e6/S)
    # But our model uses noise_linear * (bw_sub/bandwidth) to get per-sub noise
    # So we set total 5MHz noise = thermal + NF in dBm, then the model scales per-sub
    total_noise_dbm = -107.0 + nf_db  # total noise = thermal + NF
    cfg.channel_override_noise_dbm = total_noise_dbm
    return cfg, total_noise_dbm


def run_episode_nf(cfg, seed, strategy_fn, noise_power_dbm):
    """Run episode with overridden noise figure."""
    import math
    from .multi_uav_env import MultiUAVMECEnv
    from .pairing import evaluate_pairing, compute_rates_for_pairing, kkt_alpha
    from .baselines_pairing import ilp_optimal_pairing

    # Override noise after reset (channel is initialized in reset())
    env = MultiUAVMECEnv(cfg)
    env.reset(seed=seed)
    noise_linear_override = 10 ** (noise_power_dbm / 10) * 1e-3  # dBm to W
    env.channel.noise_linear = noise_linear_override
    ch = env.channel  # reference after noise override
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

        _result = strategy_fn(h_gains, env.task_queue_active, assignment, env, ch)

        if isinstance(_result, tuple):
            met, total, rates, alphas, served = _result
        else:
            pairs = _result
            met, total, rates, alphas, served = evaluate_pairing(
                pairs, h_gains, ch, env.task_queue_data, env.task_queue_comp,
                env.task_queue_active, assignment, cfg.n_subcarriers,
                cfg.f_per_uav, cfg.task.f_local, cfg.task.t_max)

        total_met += met
        total_tasks += total

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
                    prob = (cfg.burst_rate if cid == env.burst_cluster else cfg.cool_rate)
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
    print("NF and SIC γ_min Sensitivity Experiments")
    print("=" * 60)

    cfg_base = ScenarioConfig(fixed_cluster_centers=True, random_burst=True,
                               user_mobility=True)
    seeds = list(range(42, 42 + 20))

    # ---- NF Sensitivity ----
    print("\n--- Noise Figure Sensitivity (NF = 5, 7, 9 dB) ---")
    print(f"{'NF(dB)':>8s}  {'Ours(%)':>10s}  {'Greedy(%)':>10s}  {'Gap(pp)':>8s}")

    nf_results = {}
    for nf_db in [5, 7, 9]:
        # Noise: per-subcarrier = (thermal_total + NF_contribution) / S
        # thermal_total = -107 dBm for 5MHz
        # with NF: effective total = -107 + nf_db dBm (as linear contribution)
        # per-sub noise_power_dbm: we use noise_linear at full BW = 10^((-107+nf)/10)*1e-3 W
        # then per-sub = noise_linear * (bw_sub/bw_total)
        total_noise_dbm = -107.0 + nf_db
        noise_linear_total = 10 ** (total_noise_dbm / 10) * 1e-3  # W

        ours_rates = []
        greedy_rates = []
        for seed in seeds:
            r_ours = run_episode_nf(cfg_base, seed, strategy_ours, total_noise_dbm)
            r_greedy = run_episode_nf(cfg_base, seed, strategy_greedy, total_noise_dbm)
            ours_rates.append(r_ours)
            greedy_rates.append(r_greedy)

        ours_mean = np.mean(ours_rates)
        greedy_mean = np.mean(greedy_rates)
        gap = ours_mean - greedy_mean
        nf_results[nf_db] = {"ours": round(ours_mean, 1), "greedy": round(greedy_mean, 1),
                              "gap": round(gap, 1), "ours_std": round(np.std(ours_rates), 1)}
        print(f"{nf_db:>8d}  {ours_mean:>9.1f}%  {greedy_mean:>9.1f}%  {gap:>+8.1f}pp")

    # ---- SIC γ_min Sensitivity ----
    print("\n--- SIC γ_min Sensitivity (0dB, 3dB, 5dB) ---")
    print(f"{'γ_min(dB)':>10s}  {'Ours(%)':>10s}  {'Greedy(%)':>10s}  {'Gap(pp)':>8s}")

    from .pairing import swap_optimize, evaluate_pairing
    from .baselines_pairing import ilp_optimal_pairing

    sic_results = {}
    for gamma_db in [0, 3, 5]:
        gamma_lin = 10 ** (gamma_db / 10)  # linear γ_SIC_min

        # To test SIC γ_min, we need to modify the SIC feasibility filter
        # in ilp_optimal_pairing. We patch the sic_feasibility_min parameter.
        def strategy_ours_sic(h, active, assign, env, ch, gmin=gamma_lin):
            cfg = env.cfg
            hp = ilp_optimal_pairing(h, active, assign, cfg.n_subcarriers, channel=ch,
                                     task_data=env.task_queue_data,
                                     task_comp=env.task_queue_comp,
                                     f_per_uav=cfg.f_per_uav, f_local=cfg.task.f_local,
                                     t_max=cfg.task.t_max, sic_eps=0.02,
                                     sic_gamma_min=gmin)
            pairs, _, _ = swap_optimize(hp, h, ch, active, assign,
                                         env.task_queue_data, env.task_queue_comp,
                                         cfg.n_subcarriers, cfg.f_per_uav,
                                         cfg.task.f_local, cfg.task.t_max)
            return pairs

        ours_rates = []
        greedy_rates = []
        for seed in seeds:
            r_ours = run_episode(cfg_base, seed, strategy_ours_sic)
            r_greedy = run_episode(cfg_base, seed, strategy_greedy)
            ours_rates.append(r_ours["deadline_rate"])
            greedy_rates.append(r_greedy["deadline_rate"])

        ours_mean = np.mean(ours_rates)
        greedy_mean = np.mean(greedy_rates)
        gap = ours_mean - greedy_mean
        sic_results[gamma_db] = {"ours": round(ours_mean, 1), "greedy": round(greedy_mean, 1),
                                  "gap": round(gap, 1)}
        print(f"{gamma_db:>10d}  {ours_mean:>9.1f}%  {greedy_mean:>9.1f}%  {gap:>+8.1f}pp")

    # Save
    out = Path(__file__).parent.parent / "results" / "nf_sic_sensitivity.json"
    with open(out, "w") as f:
        json.dump({"nf": nf_results, "sic_gamma": sic_results,
                   "date": time.strftime("%Y-%m-%d")}, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
