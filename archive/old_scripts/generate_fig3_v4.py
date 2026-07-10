"""
Fig.3 v5 — horizontal bar chart, two-panel, aligned with Table 1.
All 15 methods + 2 DRL. IEEE double-column safe (font >= 8pt).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent.parent / "figures"

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'axes.linewidth': 0.6,
})

OURS = 73.41

COLOR = {
    'ours':   '#1a5276',
    'swap':   '#2980b9',
    'match':  '#7fb3d3',
    'init':   '#aed6f1',
    'ofdma':  '#e74c3c',
    'random': '#f1948a',
    'drl':    '#7b241c',
    'greedy': '#c0392b',
}

# Panel (a): algorithmic comparison — 10 methods, sorted descending
alg = [
    ('Ours (ILP+Swap)',        73.41, 2.01, 'ours'),
    ('BG+Swap',                72.45, 2.12, 'swap'),
    ('Greedy+Swap',            71.64, 2.27, 'swap'),
    ('WorstCase+Swap',         71.28, 2.45, 'swap'),
    ('IntraSwapOnly',          70.79, 2.15, 'swap'),
    ('ILP+BlindSwap',          70.40, 1.93, 'swap'),
    ('ExStable+CrossUAV',      69.60, 1.76, 'match'),
    ('ExchStable',             69.17, 1.92, 'match'),
    ('Per-UAV ILP',            67.67, 2.24, 'init'),
    ('BenefitGreedy',          67.03, 2.18, 'init'),
]

# Panel (b): reference baselines
ref = [
    ('OFDMA (OrthoBank)',      65.50, 2.57, 'ofdma'),
    ('OFDMA (FullReuse)',      64.13, 1.81, 'ofdma'),
    ('Random',                 63.75, 2.10, 'random'),
    ('PPO (DRL)',               62.58, 2.52, 'drl'),
    ('OFDMA (Fair)',           62.55, 2.44, 'ofdma'),
    ('Greedy',                 62.12, 2.30, 'greedy'),
    ('MAPPO (DRL)',            46.61, 2.21, 'drl'),
]

def hbar_panel(ax, methods, title, xlim=(44, 78)):
    labels = [m[0] for m in methods]
    means  = np.array([m[1] for m in methods])
    stds   = np.array([m[2] for m in methods])
    colors = [COLOR[m[3]] for m in methods]

    # Reverse so highest is at top
    labels = labels[::-1]
    means  = means[::-1]
    stds   = stds[::-1]
    colors = colors[::-1]

    y = np.arange(len(labels))
    bars = ax.barh(y, means, xerr=stds, height=0.6,
                   color=colors, edgecolor='white', linewidth=0.4,
                   error_kw=dict(elinewidth=0.8, ecolor='#444', capthick=0.8, capsize=2.5),
                   zorder=3)

    # Gap annotation to the right of each bar
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        gap = mean - OURS
        if abs(gap) < 0.05:
            txt, col = '—', '#1a5276'
        else:
            txt = f'{gap:+.1f}'
            col = '#c0392b' if gap < 0 else '#1a5276'
        ax.text(mean + std + 0.2, bar.get_y() + bar.get_height() / 2,
                txt, va='center', ha='left',
                fontsize=7, color=col, fontweight='bold')

    ax.axvline(OURS, color='#c0392b', linestyle='--', linewidth=0.9,
               alpha=0.8, zorder=4, label=f'Ours = {OURS:.1f}%')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Deadline Satisfaction Rate (%)', fontsize=8.5)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.75,
              handlelength=1.5, borderpad=0.4)


# Two panels: height proportional to method count
fig, (ax_a, ax_b) = plt.subplots(
    1, 2,
    figsize=(7.1, 3.6),
    gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.55}
)

hbar_panel(ax_a, alg,  '(a) Algorithmic comparison', xlim=(60, 79))
hbar_panel(ax_b, ref,  '(b) Reference baselines',    xlim=(38, 79))

fig.savefig(OUT / 'fig3_main_comparison.pdf', format='pdf')
fig.savefig(OUT / 'fig3_main_comparison.png', format='png')
print("Saved.")
