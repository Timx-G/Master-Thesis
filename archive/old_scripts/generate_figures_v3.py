"""
Generate updated figures for paper revision.
Uses data from phase2_main_experiment.json + supplementary_experiments.json.

Fig.3: Main comparison (10 methods including ExchangeStable, Hungarian+Swap)
Fig.5: N-scalability (N=10,15,20,25,30 with Ours/Greedy/ExchangeStable)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)
RES = Path(__file__).parent.parent / "results"

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


def fig3_main_comparison():
    """Main comparison bar chart — 10 methods."""
    # Data from experiments (phase2_main_experiment.json + known values)
    methods = [
        ('Ours\n(Swap)',       72.9, 2.1),
        ('Hung.\n+Swap',       72.9, 2.1),
        ('Greedy\n+Swap',      72.2, 2.2),
        ('Exchange\nStable',   68.2, 2.0),
        ('Hungarian',          68.3, 2.1),
        ('Single\nUAV',        67.6, 1.8),
        ('Random',             64.7, 2.8),
        ('PPO\n(DRL)',         63.8, 2.4),
        ('Greedy',             62.7, 2.3),
        ('OFDMA',              46.4, 1.9),
    ]

    labels = [m[0] for m in methods]
    means = [m[1] for m in methods]
    stds = [m[2] for m in methods]

    colors = ['#1a5276', '#2471a3', '#5499c7', '#85c1e9',
              '#aed6f1', '#d4e6f1', '#fadbd8', '#f5b7b1',
              '#e74c3c', '#922b21']

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, width=0.7, capsize=2.5,
                  color=colors, edgecolor='black', linewidth=0.5)

    # Bold border for Ours
    bars[0].set_linewidth(2.0)

    ax.set_ylabel('Deadline Satisfaction Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(35, 82)
    ax.axhline(y=means[0], color='red', linestyle='--', linewidth=0.5, alpha=0.4)
    ax.grid(axis='y', alpha=0.3)

    # Annotate gap vs Ours
    for i in range(1, len(methods)):
        gap = means[0] - means[i]
        if gap > 0.5:
            ax.annotate(f'-{gap:.1f}', xy=(i, means[i] + stds[i] + 0.8),
                       ha='center', fontsize=6, color='red')

    fig.savefig(OUT / "fig3_main_comparison.pdf")
    fig.savefig(OUT / "fig3_main_comparison.png")
    print("  Fig.3 saved (10 methods)")
    plt.close()


def fig5_scalability():
    """N-scalability: Ours vs Greedy vs ExchangeStable, N=10..30."""
    # Data from supplementary_experiments.json
    with open(RES / "supplementary_experiments.json") as f:
        data = json.load(f)

    ns_data = data['n_scalability']
    Ns = [10, 15, 20, 25, 30]
    ours = [ns_data[str(n)]['ours_mean'] for n in Ns]
    greedy = [ns_data[str(n)]['greedy_mean'] for n in Ns]
    exstable = [ns_data[str(n)]['exchange_mean'] for n in Ns]
    gaps_g = [ns_data[str(n)]['gap_greedy'] for n in Ns]

    fig, ax1 = plt.subplots(figsize=(4.5, 3.0))

    ax1.plot(Ns, ours, 'o-', color='#1a5276', label='Ours (Swap)',
             linewidth=1.8, markersize=6, zorder=3)
    ax1.plot(Ns, exstable, 's--', color='#2e86c1', label='ExchangeStable',
             linewidth=1.5, markersize=5, zorder=2)
    ax1.plot(Ns, greedy, '^:', color='#e74c3c', label='Greedy',
             linewidth=1.5, markersize=5, zorder=2)

    ax1.set_xlabel('Number of Users ($N$)')
    ax1.set_ylabel('Deadline Satisfaction Rate (%)')
    ax1.set_xticks(Ns)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(45, 95)

    # Annotate gap vs Greedy
    for i, n in enumerate(Ns):
        ax1.annotate(f'+{gaps_g[i]:.1f}pp',
                    xy=(n, (ours[i] + greedy[i]) / 2),
                    ha='center', fontsize=6.5, color='#196f3d', weight='bold')

    fig.savefig(OUT / "fig5_scalability.pdf")
    fig.savefig(OUT / "fig5_scalability.png")
    print("  Fig.5 saved (N=10..30, 3 methods)")
    plt.close()


if __name__ == "__main__":
    print("=== Generating Updated Figures ===")
    fig3_main_comparison()
    fig5_scalability()
    print("\nDone. Figures saved to figures/")
