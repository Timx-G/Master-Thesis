"""
Phase 3: Generate all figures for the paper.
Fig.3: Main comparison bar chart (8 methods)
Fig.4: Ablation bar chart
Fig.5: Scalability (N=10-30)
Fig.6: t_max sensitivity
Fig.7: Per-slot time series (Ours vs Greedy vs Random)
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "master-thesis-matplotlib"),
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .statistics import paired_bootstrap_ci

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Hatching patterns for black/white distinguishability
HATCHES = ['///', '\\\\\\', 'xxx', '...', '+++', 'ooo', '|||', '---']
COLORS = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']

# Canonical experiment keys stay stable in JSON; only this mapping is public.
METHOD_DISPLAY_LABELS = {
    "exchange_stable": "Exchange-Stable",
    "fair_ofdma": "Fair OFDMA",
    "full": "EAD-JSRA (Proposed)",
    "greedy": "Greedy",
    "ia_greedy": "IA-Greedy",
    "ilp": "Per-UAV ILP",
    "intra_global": "Intra-Global",
    "local_only": "Local-Only",
}


def method_display_label(method: str) -> str:
    """Return a publication label without exposing an internal method key."""
    return METHOD_DISPLAY_LABELS.get(method, method.replace("_", " ").title())


def load_json(name):
    p = Path(__file__).parent.parent / "results" / name
    with open(p) as f:
        return json.load(f)


# ===== Fig.3: Main comparison (horizontal bar chart, all 16 methods) =====
def fig3_main_comparison():
    raw = load_json("phase2_main_experiment.json")
    data = raw["results"]

    # All 16 methods in Tab5 order (best first)
    methods_ordered = [
        'Ours(ILP+Swap)', 'BG+Swap', 'IA-Greedy+Swap', 'Greedy+Swap',
        'WorstCase+Swap', 'IntraSwapOnly', 'ILP+BlindSwap', 'ExStable+CrossUAV',
        'ExchangeStable', 'IA-Greedy', 'ILP-Optimal', 'BenefitGreedy',
        'OFDMA(OrthoBank)', 'OFDMA(FullReuse)', 'Random', 'Greedy',
    ]
    label_map = {
        'Ours(ILP+Swap)':    'Ours (ILP+Swap)',
        'BG+Swap':           'BG+Swap',
        'IA-Greedy+Swap':    'IA-Greedy+Swap',
        'Greedy+Swap':       'Greedy+Swap',
        'WorstCase+Swap':    'WorstCase+Swap',
        'IntraSwapOnly':     'IntraSwapOnly',
        'ILP+BlindSwap':     'ILP+BlindSwap',
        'ExStable+CrossUAV': 'ExStable+CrossUAV',
        'ExchangeStable':    'ExchangeStable',
        'IA-Greedy':         'IA-Greedy',
        'ILP-Optimal':       'Per-UAV ILP',
        'BenefitGreedy':     'BenefitGreedy',
        'OFDMA(OrthoBank)':  'OFDMA (OrthoBank)',
        'OFDMA(FullReuse)':  'OFDMA (FullReuse)',
        'Random':            'Random',
        'Greedy':            'Greedy',
    }
    color_map = {
        'Ours(ILP+Swap)':    '#1a508b',
        'BG+Swap':           '#4393c3',
        'IA-Greedy+Swap':    '#4393c3',
        'Greedy+Swap':       '#4393c3',
        'WorstCase+Swap':    '#4393c3',
        'IntraSwapOnly':     '#74b9d5',
        'ILP+BlindSwap':     '#74b9d5',
        'ExStable+CrossUAV': '#92c5de',
        'ExchangeStable':    '#92c5de',
        'IA-Greedy':         '#b8dced',
        'ILP-Optimal':       '#b8dced',
        'BenefitGreedy':     '#b8dced',
        'OFDMA(OrthoBank)':  '#d6604d',
        'OFDMA(FullReuse)':  '#d6604d',
        'Random':            '#e08070',
        'Greedy':            '#c0392b',
    }
    # Gap labels: Ours value + Greedy gap only (quieter)
    show_gap = {
        'Ours(ILP+Swap)', 'Greedy',
    }

    means = [data[m]['deadline_mean'] for m in methods_ordered]
    stds  = [data[m]['deadline_std']  for m in methods_ordered]
    ours_val = means[0]
    n = len(methods_ordered)

    # y positions: Ours at TOP (highest y), Greedy at bottom (y=0)
    # Reverse the order so best is at top visually
    # Also add gap between algorithmic (idx 0-11) and OFDMA/lower (idx 12-15)
    y = np.zeros(n)
    for i in range(n):
        # reversed: idx 0 (Ours) → highest y, idx 15 (Greedy) → y=0
        y[i] = (n - 1 - i)
    y[:12] += 0.5  # shift algorithmic group up to create gap vs reference group

    fig, ax = plt.subplots(figsize=(3.3, 4.7))

    for i, m in enumerate(methods_ordered):
        lw = 1.6 if m == 'Ours(ILP+Swap)' else 0.4
        ax.barh(y[i], means[i], xerr=stds[i], height=0.55,
                capsize=1.8, color=color_map[m], edgecolor='black', linewidth=lw,
                error_kw=dict(elinewidth=0.7, ecolor='#444444', capthick=0.7))

    # Reference line at Ours
    ax.axvline(ours_val, color='#1a508b', linestyle='--', linewidth=0.8, alpha=0.6)

    # Gap labels: Ours value inside bar + Greedy gap only
    for i, m in enumerate(methods_ordered):
        gap = ours_val - means[i]
        if m == 'Ours(ILP+Swap)':
            ax.text(means[i] - 0.4, y[i], f'{means[i]:.1f}%',
                    va='center', ha='right', fontsize=7,
                    fontweight='bold', color='white')
        elif m in show_gap:
            ax.text(means[i] - 0.5, y[i], f'\u2212{gap:.1f}',
                    va='center', ha='right', fontsize=6.5, color='#b22222')

    ax.set_yticks(y)
    ax.set_yticklabels([label_map[m] for m in methods_ordered], fontsize=7.5)
    ax.set_xlabel('Deadline Satisfaction Rate (%)', fontsize=8.5)
    ax.set_xlim(35, 84)
    ax.grid(axis='x', alpha=0.18, linestyle=':', linewidth=0.5)
    ax.tick_params(axis='y', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Thin separator line between algorithmic methods and OFDMA/reference
    sep_y = (y[11] + y[12]) / 2
    ax.axhline(sep_y, color='#cccccc', linewidth=0.8, linestyle='--', zorder=1)

    # No group labels — caption explains color grouping

    fig.tight_layout()
    fig.savefig(OUT / "fig3_main_comparison.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig3_main_comparison.png", bbox_inches='tight')
    print(f"  Fig.3 saved (clean horizontal bar, key gaps only)")
    plt.close()


# ===== Fig.4: Ablation =====
def fig4_ablation():
    raw = load_json("phase2_main_experiment.json")
    data = raw["results"]
    # Ablation: ILP init → BG init → Greedy init → no swap (ILP-Optimal) → Greedy baseline
    methods = ['Ours(ILP+Swap)', 'BG+Swap', 'Greedy+Swap', 'ILP-Optimal', 'Greedy']
    labels  = ['Ours\n(ILP+Swap)', 'BG\n+Swap', 'Greedy\n+Swap', 'Per-UAV\nILP\n(no swap)', 'Greedy\n(baseline)']
    means = [data[m]['deadline_mean'] for m in methods]
    stds  = [data[m]['deadline_std']  for m in methods]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(methods))
    colors_abl = ['#2166ac', '#92c5de', '#fddbc7', '#d6604d']
    bars = ax.bar(x, means, yerr=stds, width=0.6, capsize=3,
                  color=colors_abl, edgecolor='black', linewidth=0.5)
    for i, b in enumerate(bars):
        b.set_hatch(HATCHES[i])

    ax.set_ylabel('Deadline Satisfaction Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(55, 80)
    ax.grid(axis='y', alpha=0.3)

    # Annotate deltas
    for i in range(1, len(methods)):
        delta = means[i] - means[0]
        ax.annotate(f'{delta:.1f}pp', xy=(i, means[i] - 1.5),
                   ha='center', fontsize=7, color='red', weight='bold')

    fig.savefig(OUT / "fig4_ablation.pdf")
    fig.savefig(OUT / "fig4_ablation.png")
    print(f"  Fig.4 saved")
    plt.close()


# ===== Fig.5: Scalability =====
def fig5_scalability():
    """N-scalability: 3 curves (Ours / ExchangeStable / Greedy), N=10-30."""
    with open(Path(__file__).parent.parent / "results" / "n_scalability_3methods.json") as f:
        import json as _json
        raw = _json.load(f)
    scale = raw["n_scalability_3methods"]

    Ns = [10, 15, 20, 25, 30]
    ours_vals   = [scale[str(n)]["ours"]            for n in Ns]
    es_vals     = [scale[str(n)]["exchange_stable"] for n in Ns]
    greedy_vals = [scale[str(n)]["greedy"]          for n in Ns]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    ax.plot(Ns, ours_vals,   'o-',  color='#2166ac', label='Ours (ILP+Swap)',
            linewidth=1.8, markersize=5, zorder=4)
    ax.plot(Ns, es_vals,     'D--', color='#66a61e', label='ExchangeStable',
            linewidth=1.3, markersize=4.5, zorder=3)
    ax.plot(Ns, greedy_vals, 's:',  color='#d6604d', label='Greedy',
            linewidth=1.3, markersize=4.5, zorder=3)

    ax.set_xlabel('Number of Users ($N$)', fontsize=9)
    ax.set_ylabel('Deadline Satisfaction Rate (%)', fontsize=9)
    ax.set_xticks(Ns)
    ax.set_ylim(45, 98)
    ax.grid(alpha=0.25, linestyle=':', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9,
              edgecolor='#cccccc', labelspacing=0.3, handlelength=1.5)

    # Annotate gap vs Greedy at N=20 and N=30 only
    for n_ann, label in [(20, f'+{scale[str(20)]["gap_greedy"]:.1f}\u202fpp'),
                          (30, f'+{scale[str(30)]["gap_greedy"]:.1f}\u202fpp')]:
        idx = Ns.index(n_ann)
        mid_y = (ours_vals[idx] + greedy_vals[idx]) / 2
        ax.annotate(label,
                    xy=(n_ann, mid_y),
                    xytext=(n_ann + 1.5, mid_y),
                    fontsize=7, color='#2166ac',
                    arrowprops=dict(arrowstyle='-', lw=0.4, color='#aaaaaa'),
                    va='center')

    fig.tight_layout()
    fig.savefig(OUT / "fig5_scalability.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig5_scalability.png", bbox_inches='tight')
    print(f"  Fig.5 saved (3-line: Ours / ExStable / Greedy)")
    plt.close()


# ===== Fig.6: t_max Sensitivity =====
def fig6_tmax():
    """t_max sensitivity: 3 curves (Ours / ExchangeStable / Greedy)."""
    with open(Path(__file__).parent.parent / "results" / "tmax_sensitivity_3methods.json") as f:
        import json as _json
        raw = _json.load(f)
    data = raw["tmax_sensitivity_3methods"]

    tvals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ours   = [data[str(t)]["ours"]            for t in tvals]
    es     = [data[str(t)]["exchange_stable"] for t in tvals]
    greedy = [data[str(t)]["greedy"]          for t in tvals]

    fig, ax = plt.subplots(figsize=(4.0, 2.8))

    ax.plot(tvals, ours,   'o-',  color='#2166ac', label='Ours (ILP+Swap)',
            linewidth=1.8, markersize=5, zorder=4)
    ax.plot(tvals, es,     'D--', color='#66a61e', label='ExchangeStable',
            linewidth=1.3, markersize=4.5, zorder=3)
    ax.plot(tvals, greedy, 's:',  color='#d6604d', label='Greedy',
            linewidth=1.3, markersize=4.5, zorder=3)

    ax.set_xlabel('Deadline threshold $t_{\\max}$ (s)', fontsize=9)
    ax.set_ylabel('Deadline Satisfaction Rate (%)', fontsize=9)
    ax.set_xticks(tvals)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9,
              edgecolor='#cccccc', labelspacing=0.3)
    ax.grid(alpha=0.25, linestyle=':', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate gap vs Greedy at t=0.7 and t=0.8 only (2 sparse labels)
    for t_ann in [0.7, 0.8]:
        idx = tvals.index(t_ann)
        mid_y = (ours[idx] + greedy[idx]) / 2
        gap = data[str(t_ann)]["gap_greedy"]
        ax.annotate(f'+{gap:.1f}\u202fpp',
                    xy=(t_ann, mid_y), xytext=(t_ann + 0.04, mid_y),
                    fontsize=7, color='#2166ac', va='center',
                    arrowprops=dict(arrowstyle='-', lw=0.4, color='#aaaaaa'))

    fig.tight_layout()
    fig.savefig(OUT / "fig6_tmax_sensitivity.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig6_tmax_sensitivity.png", bbox_inches='tight')
    print(f"  Fig.6 saved (3-line: Ours / ExStable / Greedy)")
    plt.close()


# ===== Fig.7: Per-slot time series =====
def fig7_timeseries():
    # Re-run to get per-slot data
    import math
    from uav_mec.model import ScenarioConfig
    from uav_mec.multi_uav_env import MultiUAVMECEnv
    from uav_mec.pairing import (greedy_pairing, benefit_greedy_pairing,
                                  swap_optimize, evaluate_pairing)
    from uav_mec.run_main_experiment import strategy_ours, strategy_greedy, strategy_random

    cfg = ScenarioConfig(fixed_cluster_centers=True, random_burst=True, user_mobility=True)
    N = cfg.num_users

    series = {}
    for name, fn in [('Ours', strategy_ours), ('Greedy', strategy_greedy), ('Random', strategy_random)]:
        env = MultiUAVMECEnv(cfg)
        env.reset(seed=42)
        ch = env.channel
        uav_pos = [uav.position for uav in env.uavs]
        slot_rates = []

        for slot in range(cfg.num_slots):
            if cfg.user_mobility:
                for i in range(N):
                    a = env.rng.uniform(0, 2*math.pi)
                    env.user_positions[i,0] += cfg.user_speed*cfg.uav.dt*math.cos(a)
                    env.user_positions[i,1] += cfg.user_speed*cfg.uav.dt*math.sin(a)
                env.user_positions = np.clip(env.user_positions, 0, cfg.area_size)

            assign = np.zeros(N, dtype=int)
            for i in range(N):
                d0 = np.linalg.norm(np.array(uav_pos[0][:2])-env.user_positions[i])
                d1 = np.linalg.norm(np.array(uav_pos[1][:2])-env.user_positions[i])
                assign[i] = 0 if d0 <= d1 else 1
            h = np.zeros((N, 2))
            for i in range(N):
                if not env.task_queue_active[i]: continue
                for k in range(2):
                    h[i,k] = ch.path_loss(np.linalg.norm(np.array(uav_pos[k][:2])-env.user_positions[i]))

            pairs = fn(h, env.task_queue_active, assign, env, ch)
            met, total, rates, alphas, served = evaluate_pairing(
                pairs, h, ch, env.task_queue_data, env.task_queue_comp,
                env.task_queue_active, assign, cfg.n_subcarriers,
                cfg.f_per_uav, cfg.task.f_local, cfg.task.t_max)
            slot_rates.append(met / max(total, 1) * 100)

            # Update queue
            for i in range(N):
                if not env.task_queue_active[i]: continue
                if served[i] and rates[i] > 1:
                    al = alphas[i]; D = env.task_queue_data[i]; C = env.task_queue_comp[i]
                    nf = max(sum(1 for j in range(N) if served[j] and assign[j]==assign[i] and rates[j]>1), 1)
                    fa = max(cfg.f_per_uav/nf, 1e3)
                    tl = (1-al)*C/cfg.task.f_local; to = al*D/rates[i]+al*C/fa if al>0 else 0
                    delay = max(tl, to)
                else:
                    delay = env.task_queue_comp[i] / cfg.task.f_local
                if delay <= cfg.task.t_max:
                    env.task_queue_active[i] = False; env.task_queue_age[i] = 0
                else:
                    env.task_queue_age[i] += 1
                    if env.task_queue_age[i] >= cfg.max_queue_age:
                        env.task_queue_active[i] = False; env.task_queue_age[i] = 0
            env.slot += 1
            if env.slot < cfg.num_slots:
                nt = env.task_gen.generate(N)
                if cfg.random_burst:
                    if env.slot >= env.next_burst_switch:
                        env.burst_cluster = env.rng.integers(0, 3)
                        env.next_burst_switch = env.slot + env.rng.integers(
                            cfg.burst_period_min, cfg.burst_period_max+1)
                    for i in range(N):
                        cid = env.user_cluster_ids[i]
                        prob = cfg.burst_rate if cid == env.burst_cluster else cfg.cool_rate
                        nt['has_task'][i] = env.rng.random() < prob
                        if not nt['has_task'][i]:
                            nt['data_bits'][i] = 0; nt['comp_cycles'][i] = 0
                for i in range(N):
                    if nt['has_task'][i] and not env.task_queue_active[i]:
                        env.task_queue_data[i] = nt['data_bits'][i]
                        env.task_queue_comp[i] = nt['comp_cycles'][i]
                        env.task_queue_age[i] = 0; env.task_queue_active[i] = True

        series[name] = slot_rates

    # Plot
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    window = 5
    for name, color, ls in [('Ours', '#2166ac', '-'), ('Greedy', '#d6604d', '--'), ('Random', '#999999', ':')]:
        vals = np.convolve(series[name], np.ones(window)/window, mode='valid')
        ax.plot(range(len(vals)), vals, color=color, linestyle=ls, label=name, linewidth=1.3)

    ax.set_xlabel('Time Slot', fontsize=9)
    ax.set_ylabel('Deadline Rate (%, 5-slot avg)', fontsize=9)
    ax.legend(loc='lower left', fontsize=8.5)
    ax.set_ylim(30, 100)
    ax.grid(alpha=0.25)

    fig.savefig(OUT / "fig7_timeseries.pdf")
    fig.savefig(OUT / "fig7_timeseries.png")
    print(f"  Fig.7 saved")
    plt.close()


def fig_drl_convergence():
    """DRL convergence: convergence-evidence figure (not just comparison plot).
    Visually demonstrates: PPO plateaus well below Ours; MAPPO degrades (non-stationarity);
    both gaps are persistent, not transient.
    """
    import json as _json
    rdir = Path(__file__).parent.parent / "results"

    with open(rdir / "ppo_convergence_2k.json") as f:
        ppo_data = _json.load(f)
    with open(rdir / "mappo_convergence_history.json") as f:
        mappo_data = _json.load(f)

    ppo_raw   = np.array(ppo_data["history"])
    mappo_raw = np.array(mappo_data["history"])
    ours_val  = 73.4

    w = 50
    ppo_rm   = np.convolve(ppo_raw,   np.ones(w)/w, mode='valid')
    mappo_rm = np.convolve(mappo_raw, np.ones(w)/w, mode='valid')
    eps_rm   = np.arange(w, 2001)

    # Late-stage statistics (ep 1000–2000 of rolling mean)
    plateau_start_ep = 600            # ep where PPO visually levels off
    plateau_idx      = plateau_start_ep - w   # index in rolling array
    ppo_plateau_vals = ppo_rm[plateau_idx:]
    ppo_plat_mean    = ppo_plateau_vals.mean()     # 59.93
    ppo_plat_std     = ppo_plateau_vals.std()      # ~1.00
    mappo_late_vals  = mappo_rm[plateau_idx:]
    mappo_late_mean  = mappo_late_vals.mean()      # 46.15

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    # ── 1. Plateau region background (more subtle) ────────────────────────
    ax.axvspan(plateau_start_ep, 2000,
               alpha=0.03, color='#2166ac', zorder=0,
               label='_nolegend_')

    # ── 2. Raw traces (very faint background context) ─────────────────────
    ax.plot(np.arange(1, 2001), ppo_raw,
            color='#4393c3', alpha=0.06, lw=0.35, zorder=1)
    ax.plot(np.arange(1, 2001), mappo_raw,
            color='#d6604d', alpha=0.06, lw=0.35, zorder=1)

    # ── 3. PPO plateau confidence band (mean ± std, lighter) ─────────────
    ax.fill_between([plateau_start_ep, 2000],
                    ppo_plat_mean - ppo_plat_std,
                    ppo_plat_mean + ppo_plat_std,
                    color='#4393c3', alpha=0.12, zorder=2, label='_nolegend_')

    # ── 4. Rolling mean curves ────────────────────────────────────────────
    ax.plot(eps_rm, ppo_rm, color='#2166ac', lw=1.6,
            label='PPO (50-ep mean)', zorder=4)
    ax.plot(eps_rm, mappo_rm, color='#c0392b', lw=1.3, linestyle='--',
            label='MAPPO (50-ep mean)', zorder=4)

    # ── 5. Ours dominant reference ────────────────────────────────────────
    ax.axhline(ours_val, color='#1a508b', lw=2.5, zorder=5,
               label=f'Ours ({ours_val}%)')
    ax.text(1980, ours_val + 1.2, f'{ours_val}%',
            ha='right', va='bottom', fontsize=8,
            fontweight='bold', color='#1a508b', zorder=6)

    # ── 6. PPO plateau annotation (smaller, cleaner, no arrow) ───────────
    ax.text(1300, ppo_plat_mean + ppo_plat_std + 1.5,
            f'plateau ≈{ppo_plat_mean:.0f}%',
            fontsize=6.5, color='#4a78aa',
            ha='center', va='bottom', zorder=7)

    # ── 7. MAPPO decline note (text only, no arrow) ───────────────────────
    ax.text(1550, mappo_late_mean + 3.5,
            'degrades', fontsize=6.5, color='#c04040',
            ha='center', va='bottom', style='italic', zorder=7)

    # ── 8. Persistent-gap indicator (light, thin) ─────────────────────────
    gap_x = 1910
    ax.annotate('', xy=(gap_x, ours_val - 0.5), xytext=(gap_x, ppo_plat_mean + 0.5),
                arrowprops=dict(arrowstyle='<->', color='#888888',
                                lw=0.7), zorder=6)
    ax.text(gap_x + 28, (ours_val + ppo_plat_mean) / 2,
            f'{ours_val - ppo_plat_mean:.1f}pp',
            fontsize=6.5, color='#888888', va='center', ha='left', zorder=7)
    # ── 9. Axes ───────────────────────────────────────────────────────────
    ax.set_xlabel('Training Episode', fontsize=9)
    ax.set_ylabel('Deadline Rate (%)', fontsize=9)
    ax.set_xlim(1, 2000)
    ax.set_ylim(20, 85)
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.93,
              edgecolor='#cccccc', labelspacing=0.3,
              handlelength=1.5)
    ax.grid(alpha=0.15, linestyle=':', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_drl_convergence_comparison.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig_drl_convergence_comparison.png", bbox_inches='tight')
    print(f"  Fig.DRL saved (convergence-evidence: plateau={ppo_plat_mean:.1f}%±{ppo_plat_std:.1f}%, "
          f"gap={ours_val-ppo_plat_mean:.1f}pp persistent)")
    plt.close()


def _canonical_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def method_mean_ci(summary: dict, method: str) -> tuple[float, float, float]:
    """Return mean and deterministic bootstrap CI in percentage points."""
    values = (
        np.asarray(
            summary["methods"][method]["on_time_rate_by_seed"],
            dtype=float,
        )
        * 100.0
    )
    low, high = paired_bootstrap_ci(values, 10_000, 20260711)
    return float(values.mean()), low, high


def _save_figure(figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output.with_suffix(".pdf"),
        dpi=220,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        output.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def _bar_figure(
    summary: dict,
    methods: tuple[str, ...],
    labels: tuple[str, ...],
    output: Path,
) -> None:
    values = [method_mean_ci(summary, method) for method in methods]
    means = np.asarray([value[0] for value in values])
    errors = np.asarray(
        [
            [value[0] - value[1] for value in values],
            [value[2] - value[0] for value in values],
        ]
    )
    positions = np.arange(len(methods))
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    colors = (
        "#3B6FB6",
        "#E07A3F",
        "#3F8F68",
        "#8A5A9E",
        "#6B7280",
        "#C44E52",
        "#8172B3",
        "#4C9F9A",
    )
    axis.bar(
        positions,
        means,
        yerr=errors,
        capsize=4,
        color=colors[: len(methods)],
    )
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylabel("On-time completion rate (%)")
    axis.set_ylim(0.0, min(100.0, float(max(means + errors[1])) + 8.0))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, output)


def _suite_figure(
    root: Path,
    cases: dict[str, str],
    case_names: tuple[str, ...],
    labels: tuple[str, ...],
    output: Path,
) -> None:
    summaries = [
        _canonical_json(root / cases[name] / "summary.json")
        for name in case_names
    ]
    methods = tuple(
        name
        for name in ("greedy", "exchange_stable", "full")
        if name in summaries[0]["methods"]
    )
    styles = {
        "greedy": ("#3B6FB6", "o"),
        "exchange_stable": ("#E07A3F", "s"),
        "full": ("#3F8F68", "^"),
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for method in methods:
        values = [method_mean_ci(summary, method) for summary in summaries]
        means = np.asarray([value[0] for value in values])
        errors = np.asarray(
            [
                [value[0] - value[1] for value in values],
                [value[2] - value[0] for value in values],
            ]
        )
        color, marker = styles[method]
        axis.errorbar(
            labels,
            means,
            yerr=errors,
            label=method_display_label(method),
            color=color,
            marker=marker,
            capsize=3,
        )
    axis.set_ylabel("On-time completion rate (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, output)


def _suite_table(root: Path, index: dict) -> str:
    robustness = index.get("suites", {}).get("robustness", {})
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Case & Exchange-Stable (\\%) & EAD-JSRA (Proposed) (\\%) \\\\",
        "\\midrule",
    ]
    for case, relative in sorted(robustness.items()):
        summary = _canonical_json(root / relative / "summary.json")
        exchange = method_mean_ci(summary, "exchange_stable")[0]
        full = method_mean_ci(summary, "full")[0]
        escaped_case = case.replace("_", "\\_")
        lines.append(f"{escaped_case} & {exchange:.2f} & {full:.2f} \\\\")
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    return "\n".join(lines) + "\n"


def _statistics_table(statistics: dict, metrics: dict) -> str:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        (
            "Comparison & $\\Delta$ (pp) & 95\\% CI & "
            "$p_{\\mathrm{Holm}}$ & $r_{rb}$ \\\\"
        ),
        "\\midrule",
    ]
    for name, value in statistics.items():
        escaped_name = name.replace("_", "\\_")
        p_value = float(value["p_holm"])
        p_text = "$<0.0001$" if p_value < 0.0001 else f"{p_value:.4f}"
        lines.append(
            f"{escaped_name} & {value['mean_difference_pp']:.2f} & "
            f"[{value['ci_low_pp']:.2f}, {value['ci_high_pp']:.2f}] & "
            f"{p_text} & {value['rank_biserial']:.3f} \\\\"
        )
    lines.extend(
        (
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\bigskip",
            "",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            (
                "Method & On-time (\\%) & Drop (\\%) & Delay (s) & "
                "Jain & CPU (\\%) & P95 (ms) \\\\"
            ),
            "\\midrule",
        )
    )
    for method, value in metrics["main"].items():
        escaped_method = method_display_label(method).replace("_", "\\_")
        lines.append(
            f"{escaped_method} & {100.0 * value['on_time_rate']:.2f} & "
            f"{100.0 * value['drop_rate']:.2f} & "
            f"{value['mean_completed_delay_s']:.4f} & "
            f"{value['jain_user_on_time']:.4f} & "
            f"{100.0 * value['cpu_utilization']:.2f} & "
            f"{1000.0 * value['runtime_p95_s']:.2f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    return "\n".join(lines) + "\n"


def generate_assets(
    canonical_root: Path,
    figures_dir: Path,
    macros_path: Path,
    statistics_path: Path,
    robustness_path: Path,
) -> None:
    """Generate all thesis assets exclusively from canonical JSON evidence."""
    canonical_root = Path(canonical_root)
    index = _canonical_json(canonical_root / "index.json")
    summaries = {
        name: _canonical_json(canonical_root / relative / "summary.json")
        for name, relative in index["runs"].items()
    }
    analysis = canonical_root / index["analysis"]
    statistics = _canonical_json(analysis / "statistics.json")
    metrics = _canonical_json(analysis / "metrics.json")
    acceptance = _canonical_json(analysis / "acceptance.json")
    if not acceptance.get("all_gates_pass"):
        raise RuntimeError(
            "refusing to generate thesis assets from failed evidence gates"
        )

    figures_dir = Path(figures_dir)
    _bar_figure(
        summaries["main"],
        tuple(summaries["main"]["methods"]),
        tuple(
            method_display_label(name)
            for name in summaries["main"]["methods"]
        ),
        figures_dir / "main_on_time",
    )
    _bar_figure(
        summaries["pairing"],
        ("intra_bilateral", "cross_bilateral", "intra_global", "full"),
        (
            "Intra/Bilateral",
            "Cross/Bilateral",
            "Intra/Global",
            "Cross/Global\nEAD-JSRA",
        ),
        figures_dir / "pairing_ablation",
    )
    _bar_figure(
        summaries["resource"],
        ("full_fixed_half", "full_crossing", "full"),
        (
            "Fixed 0.5 / Equal",
            "Crossing / Equal",
            "Deadline/Min-CPU\nEAD-JSRA",
        ),
        figures_dir / "resource_ablation",
    )

    suite_index = index.get("suites", {})
    mechanism = suite_index.get("mechanism", {})
    if mechanism:
        _suite_figure(
            canonical_root,
            mechanism,
            ("coupling0", "coupling05", "coupling1"),
            ("0", "0.5", "1"),
            figures_dir / "mechanism_interference",
        )
    scalability = suite_index.get("scalability", {})
    if scalability:
        _suite_figure(
            canonical_root,
            scalability,
            ("n10", "n15", "n20", "n25", "n30", "n40", "n50"),
            ("10", "15", "20", "25", "30", "40", "50"),
            figures_dir / "scalability_users",
        )
        _suite_figure(
            canonical_root,
            scalability,
            ("k1", "k2", "k3", "k4", "k5"),
            ("1", "2", "3", "4", "5"),
            figures_dir / "scalability_uavs",
        )

    full_mean, full_low, full_high = method_mean_ci(summaries["main"], "full")
    sic_count = sum(
        int(value["sic_violations"])
        for value in acceptance["validation"].values()
    )
    exact = summaries["exact"]
    macros = {
        "FullOnTimePercent": f"{full_mean:.2f}\\%",
        "FullOnTimeCILowPercent": f"{full_low:.2f}\\%",
        "FullOnTimeCIHighPercent": f"{full_high:.2f}\\%",
        "InnovationOneCrossGainPP": (
            f"{statistics['innovation1_cross']['mean_difference_pp']:.2f}"
        ),
        "InnovationOneCrossCILowPP": (
            f"{statistics['innovation1_cross']['ci_low_pp']:.2f}"
        ),
        "InnovationOneCrossCIHighPP": (
            f"{statistics['innovation1_cross']['ci_high_pp']:.2f}"
        ),
        "InnovationOneAcceptanceGainPP": (
            f"{statistics['innovation1_acceptance']['mean_difference_pp']:.2f}"
        ),
        "InnovationOneAcceptanceCILowPP": (
            f"{statistics['innovation1_acceptance']['ci_low_pp']:.2f}"
        ),
        "InnovationOneAcceptanceCIHighPP": (
            f"{statistics['innovation1_acceptance']['ci_high_pp']:.2f}"
        ),
        "InnovationTwoGainPP": (
            f"{statistics['innovation2_deadline']['mean_difference_pp']:.2f}"
        ),
        "InnovationTwoCILowPP": (
            f"{statistics['innovation2_deadline']['ci_low_pp']:.2f}"
        ),
        "InnovationTwoCIHighPP": (
            f"{statistics['innovation2_deadline']['ci_high_pp']:.2f}"
        ),
        "StrongBaselineGainPP": (
            f"{statistics['baseline_exchange']['mean_difference_pp']:.2f}"
        ),
        "StrongBaselineCILowPP": (
            f"{statistics['baseline_exchange']['ci_low_pp']:.2f}"
        ),
        "StrongBaselineCIHighPP": (
            f"{statistics['baseline_exchange']['ci_high_pp']:.2f}"
        ),
        "FullRuntimePNinetyFiveMs": (
            f"{1000.0 * metrics['main']['full']['runtime_p95_s']:.2f}"
        ),
        "ExactOptimalPercent": (
            f"{100.0 * exact['optimal_slot_fraction']:.2f}\\%"
        ),
        "ExactMeanCompletionGap": f"{exact['mean_completion_gap']:.3f}",
        "SICViolationCount": str(sic_count),
        "ConfirmatorySeedCount": str(len(summaries["main"]["seeds"])),
    }
    macros_path = Path(macros_path)
    macros_path.parent.mkdir(parents=True, exist_ok=True)
    macros_path.write_text(
        "".join(
            f"\\newcommand{{\\{name}}}{{{value}}}\n"
            for name, value in macros.items()
        ),
        encoding="utf-8",
    )
    statistics_path = Path(statistics_path)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(
        _statistics_table(statistics, metrics),
        encoding="utf-8",
    )
    robustness_path = Path(robustness_path)
    robustness_path.parent.mkdir(parents=True, exist_ok=True)
    robustness_path.write_text(
        _suite_table(canonical_root, index),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis assets from canonical evidence",
    )
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=Path("figures/thesis"))
    parser.add_argument(
        "--macros",
        type=Path,
        default=Path("paper/generated_results.tex"),
    )
    parser.add_argument(
        "--statistics-table",
        type=Path,
        default=Path("paper/generated_statistics.tex"),
    )
    parser.add_argument(
        "--robustness-table",
        type=Path,
        default=Path("paper/generated_robustness.tex"),
    )
    args = parser.parse_args()
    generate_assets(
        args.canonical,
        args.figures,
        args.macros,
        args.statistics_table,
        args.robustness_table,
    )


if __name__ == "__main__":
    main()
