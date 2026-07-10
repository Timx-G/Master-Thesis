"""
Reconstruction scripts for Fig1 (system model) and Fig9 (swap visualization).
Both were originally generated with Matplotlib v3.10.8 on 2026-04-12.
Source scripts were lost; this file reconstructs them from the PDF content.

Usage:
    python -m uav_mec.generate_fig1_fig9
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size':   10,
    'figure.dpi':  300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Fixed cluster centres (from ScenarioConfig)
FIXED_CLUSTER_CENTERS = [
    (200, 200),   # Cluster A (lower-left)
    (550, 260),   # Cluster B (lower-right)
    (380, 590),   # Cluster C (upper-mid)
]
UAV_POSITIONS = [
    (267, 333),   # UAV_0
    (533, 267),   # UAV_1
]


# ============================================================
# Fig 1 — System Model
# ============================================================
def fig1_system_model(seed: int = 42):
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(6.0, 5.5))

    # ----- Cluster shading -----
    cluster_colors = ['#ffaaaa', '#aaccff', '#aaffcc']
    label_colors   = ['#cc4444', '#4488cc', '#33aa66']
    cluster_names  = ['Cluster A', 'Cluster B', 'Cluster C']
    for (cx, cy), col, lcol, name in zip(FIXED_CLUSTER_CENTERS,
                                          cluster_colors, label_colors, cluster_names):
        circle = mpatches.Circle((cx, cy), radius=130, color=col, alpha=0.25, zorder=1)
        ax.add_patch(circle)
        ax.text(cx, cy - 138, name, ha='center', va='top', fontsize=9,
                color=lcol, fontweight='bold', zorder=5)

    # ----- Users (2 per subcarrier × 3 subcarriers = 6 per cluster) -----
    sub_colors = ['#d62728', '#1f77b4', '#2ca02c']
    offsets_per_cluster = [
        [(-65,-55),(-25,-40), (-55,15),(25,-15), (-15,65),(50,25)],
        [(55,-45),(90,-15),   (-25,35),(45,55),   (-55,-5),(15,-65)],
        [(-45,55),(35,60),    (-65,-15),(45,-5),   (15,65),(-25,-55)],
    ]
    for c_idx, (cx, cy) in enumerate(FIXED_CLUSTER_CENTERS):
        for sub in range(3):
            for pt in range(2):
                dx, dy = offsets_per_cluster[c_idx][sub * 2 + pt]
                ax.scatter(cx + dx, cy + dy,
                           c=sub_colors[sub], s=45, zorder=4,
                           edgecolors='white', linewidths=0.4)

    # ----- UAVs (triangles + minimal labels) -----
    for k, (ux, uy) in enumerate(UAV_POSITIONS):
        ax.scatter(ux, uy, marker='^', s=300, c='#111111', zorder=6)

    ax.annotate(
        f'UAV$_0$',
        xy=(UAV_POSITIONS[0][0], UAV_POSITIONS[0][1]),
        xytext=(UAV_POSITIONS[0][0] - 70, UAV_POSITIONS[0][1] + 68),
        fontsize=8, ha='center',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbbbbb', lw=0.6),
        arrowprops=dict(arrowstyle='-', color='#bbbbbb', lw=0.5),
        zorder=7,
    )
    ax.annotate(
        f'UAV$_1$',
        xy=(UAV_POSITIONS[1][0], UAV_POSITIONS[1][1]),
        xytext=(UAV_POSITIONS[1][0] + 80, UAV_POSITIONS[1][1] + 58),
        fontsize=8, ha='center',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbbbbb', lw=0.6),
        arrowprops=dict(arrowstyle='-', color='#bbbbbb', lw=0.5),
        zorder=7,
    )
    # H=100m: small note, no arrow
    ax.text(UAV_POSITIONS[0][0] - 135, UAV_POSITIONS[0][1] + 120,
            'H\u2009=\u2009100\u2009m', fontsize=7, color='#bbbbbb',
            ha='center', va='center', zorder=7)

    # ----- Uplink arrows -----
    for c_idx, (cx, cy) in enumerate(FIXED_CLUSTER_CENTERS):
        d0 = (cx - UAV_POSITIONS[0][0])**2 + (cy - UAV_POSITIONS[0][1])**2
        d1 = (cx - UAV_POSITIONS[1][0])**2 + (cy - UAV_POSITIONS[1][1])**2
        ux, uy = UAV_POSITIONS[0] if d0 < d1 else UAV_POSITIONS[1]
        ax.annotate('', xy=(ux, uy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='#339966',
                                    lw=0.8, mutation_scale=10,
                                    connectionstyle='arc3,rad=0.08'),
                    zorder=5)

    # ----- Cross-UAV Interference (lighter box) -----
    u0x, u0y = UAV_POSITIONS[0]
    u1x, u1y = UAV_POSITIONS[1]
    ax.annotate(
        '', xy=(u1x - 12, u1y + 5), xytext=(u0x + 12, u0y - 5),
        arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.6,
                        linestyle='dashed', mutation_scale=12),
        zorder=6,
    )
    mid_x = (u0x + u1x) / 2 - 50
    mid_y = (u0y + u1y) / 2 + 55
    ax.text(mid_x, mid_y, 'Cross-UAV Interference',
            ha='center', va='bottom', fontsize=7.5, color='#cc0000',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                      ec='#cc0000', lw=0.7),
            zorder=8)

    # Legend — subcarrier color coding
    legend_elements = [
        mpatches.Patch(fc=sub_colors[0], ec='white', label='Sub 1 pairs'),
        mpatches.Patch(fc=sub_colors[1], ec='white', label='Sub 2 pairs'),
        mpatches.Patch(fc=sub_colors[2], ec='white', label='Sub 3 pairs'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              title='Grouped NOMA', title_fontsize=8,
              fontsize=7.5, framealpha=0.88, edgecolor='#cccccc',
              handlelength=1.0, labelspacing=0.25)

    # ----- Axes -----
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 800)
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('y (m)', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.18, linestyle='--', linewidth=0.4)
    # No 800m×800m text — stated in caption
    for spine in ax.spines.values():
        spine.set_linestyle('--')
        spine.set_color('#cccccc')

    fig.savefig(OUT / "fig1_system_model.pdf")
    fig.savefig(OUT / "fig1_system_model.png")
    print("  Fig.1 saved")
    plt.close()


# ============================================================
# Fig 9 — Cross-UAV Swap Visualization
# IEEE mechanism figure: structure-first, restrained highlighting
# ============================================================
def fig9_pairing_visualization():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.8))

    users_before = [
        [('A', False), ('B', False), ('G', False), ('H', False)],  # Sub 1
        [('C', False), ('D', False), ('I', False), ('J', False)],  # Sub 2
        [('E', False), ('F', False), ('K', False), ('L', False)],  # Sub 3
    ]
    users_after = [
        [('A', False), ('H', True), ('G', False), ('B', True)],   # Sub 1 — B↔H swapped
        [('C', False), ('D', False), ('I', False), ('J', False)],  # Sub 2
        [('E', False), ('F', False), ('K', False), ('L', False)],  # Sub 3
    ]

    # --- Layout ---
    col_xs  = [0.18, 0.38, 0.62, 0.82]
    row_ys  = [0.80, 0.50, 0.20]
    uav_ys  = 0.93                          # ↓ slightly: tighter top-to-Sub1 gap
    sub_xs  = 0.04

    # --- Visual hierarchy ---
    row_scale   = [1.10, 0.93, 0.93]        # Sub 2/3 接近 Sub 1 大小，层级差更小
    row_alpha   = [1.00, 0.85, 0.85]        # Sub 2/3 接近不透明，刚好浅于 Sub 1
    row_lbl_fs  = [11.0, 9.0,  9.0]
    row_lbl_col = ['#1a1a1a', '#555555', '#555555']
    row_ul_lw   = [1.8, 0.8, 0.8]          # Sub 1 underline stronger; Sub 2/3 slightly more visible

    node_w = 0.086
    node_h = 0.122

    # Node colours — restrained unified highlight: lighter teal fill + red border
    # (same teal family as regular nodes → cohesive; red ring → clearly marked)
    teal_fill   = '#3dbfa8'
    teal_ec     = '#228877'
    hi_fill     = '#a8ddd4'    # 更明显的浅 teal（与标准 teal 形成可见深浅差）
    hi_ec       = '#8b1a10'    # 更深的红框
    hi_lw       = 3.0          # 稍粗边框，识别更清晰

    uav_labels   = ['UAV$_0$', 'UAV$_1$']
    sub_labels   = ['Sub 1', 'Sub 2', 'Sub 3']
    panel_titles = ['Before', 'After Cross-UAV Swap on Sub\u00a01']

    for ax_idx, (ax, users, title) in enumerate(
            zip(axes, [users_before, users_after], panel_titles)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=3,
                     color='#111111')

        # --- UAV header boxes ---
        for label, x in zip(uav_labels, [np.mean(col_xs[:2]), np.mean(col_xs[2:])]):
            box = FancyBboxPatch((x - 0.105, uav_ys - 0.032), 0.21, 0.064,
                                 boxstyle='round,pad=0.01',
                                 fc='#edf5fb', ec='#5588aa', lw=0.9,
                                 transform=ax.transAxes, zorder=5)
            ax.add_patch(box)
            ax.text(x, uav_ys, label, ha='center', va='center',
                    fontsize=9.5, fontweight='bold',
                    color='#1a3a5c', transform=ax.transAxes, zorder=6)

        # --- Sub labels ---
        for s, (sy, slabel) in enumerate(zip(row_ys, sub_labels)):
            ax.text(sub_xs, sy, slabel, ha='left', va='center',
                    fontsize=row_lbl_fs[s], color=row_lbl_col[s],
                    fontweight='bold' if s == 0 else 'normal',
                    alpha=row_alpha[s],          # 与节点透明度一致
                    transform=ax.transAxes)

        # --- "unchanged" note (right panel only, between Sub 2 and 3) ---
        if ax_idx == 1:
            mid_y = (row_ys[1] + row_ys[2]) / 2
            ax.text(0.5, mid_y - 0.09, 'Sub\u00a02\u20133 unchanged',
                    ha='center', va='center', fontsize=7.5,
                    color='#999999', style='italic',
                    transform=ax.transAxes, zorder=7)

        # --- Rows ---
        for row_idx, row in enumerate(users):
            sy  = row_ys[row_idx]
            sc  = row_scale[row_idx]
            al  = row_alpha[row_idx]
            nw  = node_w * sc
            nh  = node_h * sc

            # User nodes
            for col_idx, (letter, highlighted) in enumerate(row):
                cx = col_xs[col_idx]
                if highlighted:
                    fc, ec_c, lw_c = hi_fill, hi_ec, hi_lw
                else:
                    fc, ec_c, lw_c = teal_fill, teal_ec, (0.8 if row_idx == 0 else 0.5)

                node = Ellipse((cx, sy), width=nw * 1.15, height=nh,
                               fc=fc, ec=ec_c, lw=lw_c, alpha=al,
                               transform=ax.transAxes, zorder=4)
                ax.add_patch(node)
                ax.text(cx, sy, letter, ha='center', va='center',
                        fontsize=10.5 * sc, fontweight='bold',
                        color='#111111', alpha=al,
                        transform=ax.transAxes, zorder=5)

            # NOMA pair underline (pair structure emphasis)
            for uav_start in [0, 2]:
                x_l = col_xs[uav_start]     - nw * 0.62
                x_r = col_xs[uav_start + 1] + nw * 0.62
                y_u = sy - nh * 0.70
                ax.plot([x_l, x_r], [y_u, y_u],
                        color='#2e6655' if row_idx == 0 else '#999999',
                        lw=row_ul_lw[row_idx], alpha=al,
                        transform=ax.transAxes, zorder=3)

            # Cross-UAV interference/coupling arrow
            gap = 0.012
            x_start = col_xs[1] + nw * 0.60 + gap
            x_end   = col_xs[2] - nw * 0.60 - gap
            if row_idx == 0:
                arw_color = '#aa2222'
                arw_lw    = 1.3
                arw_alpha = 1.0
                arw_ms    = 11
            else:
                arw_color = '#bbbbbb'
                arw_lw    = 0.7
                arw_alpha = al          # 与该行节点透明度一致
                arw_ms    = 8
            arw = FancyArrowPatch(
                (x_start, sy), (x_end, sy),
                arrowstyle='<->', color=arw_color,
                linestyle='dashed', lw=arw_lw,
                mutation_scale=arw_ms, alpha=arw_alpha,
                transform=ax.transAxes, zorder=6,
            )
            ax.add_patch(arw)

    # Tight layout — wspace=0.02 maximises cohesion between panels
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02, wspace=0.02)
    fig.savefig(OUT / "fig9_pairing_visualization.pdf", bbox_inches='tight')
    fig.savefig(OUT / "fig9_pairing_visualization.png", bbox_inches='tight')
    print("  Fig.9 saved (second-pass polish: cohesive panels, unified highlight)")
    plt.close()


if __name__ == "__main__":
    print("=== Generating Fig.1 and Fig.9 ===")
    fig1_system_model()
    fig9_pairing_visualization()
    print("Done.")
