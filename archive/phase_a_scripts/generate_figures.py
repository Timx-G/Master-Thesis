"""
Generate paper figures from experiment results.

Figures:
  Fig3 - λd convergence curves (DC-PPO vs PPO-Lagrangian)
  Fig4 - Main results bar chart (6 algorithms)
  Fig5 - Scalability line plot (N=10,15,20,25)

Run:
    cd ~/my_workspace/my_project/V2X_MEC_Offloading_Thesis
    python3 -m uav_mec.generate_figures
"""

from __future__ import annotations
import os, sys, json, csv
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIG_DIR = os.path.join(_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
})

COLORS = {
    "DC-PPO":          "#1f77b4",   # blue
    "PPO-Lagrangian":  "#ff7f0e",   # orange
    "PPO":             "#2ca02c",   # green
    "TD3":             "#9467bd",   # purple
    "Random":          "#8c564b",   # brown
    "Greedy":          "#d62728",   # red
}


# ── Fig 3: λd Convergence ─────────────────────────────────────────────────────

def fig3_lambda_convergence():
    lam_path = os.path.join(_ROOT, "results", "lambda_trajectories.json")
    if not os.path.exists(lam_path):
        print("  [Fig3] lambda_trajectories.json not found, skip.")
        return

    with open(lam_path) as f:
        traj = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo, color, ls in [
        ("DC-PPO",        COLORS["DC-PPO"],         "-"),
        ("PPO-Lagrangian", COLORS["PPO-Lagrangian"], "--"),
    ]:
        if algo not in traj:
            continue
        seeds_data = traj[algo]           # {seed_str: [lambda per ep]}
        all_seqs = np.array(list(seeds_data.values()))   # (n_seeds, n_ep)
        ep = np.arange(1, all_seqs.shape[1] + 1)
        mean = all_seqs.mean(axis=0)
        std  = all_seqs.std(axis=0)
        label = "DC-PPO (Ours)" if algo == "DC-PPO" else "PPO-Lagrangian"
        ax.plot(ep, mean, color=color, ls=ls, lw=1.8, label=label)
        ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.15)

    # λmax reference lines
    ax.axhline(5.0,  color=COLORS["DC-PPO"],        ls=":", lw=1.2, alpha=0.7,
               label=r"$\lambda_{\max}=5$ (DC-PPO)")
    ax.axhline(20.0, color=COLORS["PPO-Lagrangian"], ls=":", lw=1.2, alpha=0.7,
               label=r"$\lambda_{\max}=20$ (PPO-Lag)")

    ax.set_xlabel("Training Episode")
    ax.set_ylabel(r"Lagrangian Multiplier $\lambda_d$")
    ax.set_title(r"Convergence of $\lambda_d$ (N=20, 5 seeds)")
    ax.legend(loc="center right", framealpha=0.9)
    ax.set_xlim(1, all_seqs.shape[1])
    ax.set_ylim(bottom=0)

    out = os.path.join(FIG_DIR, "fig3_lambda_convergence.pdf")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  [Fig3] saved → {out}")


# ── Fig 4: Main Results Bar Chart ─────────────────────────────────────────────

def fig4_main_results():
    csv_path = os.path.join(_ROOT, "results", "exp1_dcppo_baseline.csv")
    if not os.path.exists(csv_path):
        print("  [Fig4] exp1_dcppo_baseline.csv not found, skip.")
        return

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    order = ["DC-PPO", "PPO-Lagrangian", "PPO", "TD3", "Random", "Greedy"]
    labels = ["DC-PPO\n(Ours)", "PPO-\nLagrangian", "PPO", "TD3", "Random", "Greedy"]
    means, stds = [], []
    for algo in order:
        drs = [float(r["deadline_rate"]) for r in rows if r["algorithm"] == algo]
        means.append(np.mean(drs) * 100)
        stds.append(np.std(drs) * 100)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.6,
                  color=[COLORS.get(a, "#999999") for a in order],
                  edgecolor="white", linewidth=0.8, error_kw={"elinewidth": 1.5})

    # Annotate bar tops
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.8, f"{m:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold" if order[i] == "DC-PPO" else "normal")

    # Highlight ours
    bars[0].set_edgecolor("#1f4e79")
    bars[0].set_linewidth(2.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Deadline Satisfaction Rate (%)")
    ax.set_title(f"Algorithm Comparison (N=20, t_max=1.0 s, 5 seeds)")
    ax.set_ylim(0, 100)
    ax.axhline(78, color="gray", ls="--", lw=1.0, alpha=0.6, label="78% target")
    ax.legend(framealpha=0.9)

    out = os.path.join(FIG_DIR, "fig4_main_results.pdf")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  [Fig4] saved → {out}")


# ── Fig 5: Scalability ────────────────────────────────────────────────────────

def fig5_scalability():
    csv_path = os.path.join(_ROOT, "results", "exp3_scalability.csv")
    if not os.path.exists(csv_path):
        print("  [Fig5] exp3_scalability.csv not found, skip.")
        return

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    N_vals = [10, 15, 20, 25]
    algos  = ["DC-PPO", "PPO", "Random"]
    algo_labels = {"DC-PPO": "DC-PPO (Ours)", "PPO": "PPO", "Random": "Random"}
    linestyles  = {"DC-PPO": "-o", "PPO": "-s", "Random": "-^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: deadline_rate vs N
    for algo in algos:
        means, stds = [], []
        for N in N_vals:
            drs = [float(r["deadline_rate"]) for r in rows
                   if r["algo"] == algo and int(r["N"]) == N]
            means.append(np.mean(drs) * 100)
            stds.append(np.std(drs) * 100)
        ax1.errorbar(N_vals, means, yerr=stds, fmt=linestyles[algo],
                     color=COLORS[algo], lw=1.8, ms=6, capsize=4,
                     label=algo_labels[algo])

    ax1.set_xlabel("Number of Users N")
    ax1.set_ylabel("Deadline Satisfaction Rate (%)")
    ax1.set_title("Scalability: Deadline Rate vs N")
    ax1.set_xticks(N_vals)
    ax1.legend(framealpha=0.9)
    ax1.set_ylim(30, 105)

    # Right: margin (DC-PPO - PPO) vs N
    dc_means = []
    pp_means = []
    for N in N_vals:
        dc = np.mean([float(r["deadline_rate"]) for r in rows
                      if r["algo"] == "DC-PPO" and int(r["N"]) == N])
        pp = np.mean([float(r["deadline_rate"]) for r in rows
                      if r["algo"] == "PPO" and int(r["N"]) == N])
        dc_means.append(dc * 100)
        pp_means.append(pp * 100)

    margins = [d - p for d, p in zip(dc_means, pp_means)]
    bar_colors = ["#1f77b4" if m > 0 else "#d62728" for m in margins]
    ax2.bar(N_vals, margins, width=3, color=bar_colors, edgecolor="white",
            linewidth=0.8)
    for i, (N, m) in enumerate(zip(N_vals, margins)):
        ax2.text(N, m + 0.1 if m >= 0 else m - 0.5,
                 f"{m:+.1f}pp", ha="center", va="bottom" if m >= 0 else "top",
                 fontsize=9)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Number of Users N")
    ax2.set_ylabel("DC-PPO − PPO (pp)")
    ax2.set_title("DC-PPO Margin over PPO")
    ax2.set_xticks(N_vals)

    out = os.path.join(FIG_DIR, "fig5_scalability.pdf")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  [Fig5] saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Generating paper figures...")
    print("=" * 50)
    fig3_lambda_convergence()
    fig4_main_results()
    fig5_scalability()
    print("Done. Output → figures/")
