#!/usr/bin/env python3
"""
Publication Figure 1: Overview of polynomial hint preconditioning results.
4-panel figure suitable for the main paper.

(a) Method comparison (main results bar chart)
(b) Degree sweep
(c) Per-horizon improvement
(d) Training curve (100K)

Usage:
    python analysis/figure1_overview.py
"""

import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_10K = 1.2421


def main():
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ========== (a) Main results ==========
    ax = fig.add_subplot(gs[0, 0])
    methods = ['Baseline', 'Cheb\n$d{=}4$', 'Cheb\n$d{=}6$', 'Cheb $d{=}4$\n+10% drop', 'MS\n$d{=}4{+}d{=}6$']
    mases = [1.2421, 1.1944, 1.1836, 1.1802, 1.1675]
    colors = ['#AAAAAA', '#4C72B0', '#4C72B0', '#55A868', '#C44E52']
    bars = ax.bar(methods, mases, color=colors, edgecolor='black', linewidth=0.5, width=0.7)
    ax.set_ylabel('Geometric Mean MASE')
    ax.set_title('(a) Main Results (10K steps)', fontsize=11)
    ax.set_ylim(1.14, 1.26)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add delta labels
    for bar, m in zip(bars, mases):
        if m != BASELINE_10K:
            delta = (m - BASELINE_10K) / BASELINE_10K * 100
            ax.text(bar.get_x() + bar.get_width()/2., m - 0.003,
                    f'{delta:.1f}%', ha='center', va='top', fontsize=8, fontweight='bold')

    # ========== (b) Degree sweep ==========
    ax = fig.add_subplot(gs[0, 1])
    degrees = [2, 3, 4, 5, 6, 7, 8]
    deg_mases = [1.2157, 1.2040, 1.1944, 1.2084, 1.1836, 1.2027, 1.2216]
    colors_d = ['#4C72B0' if d != 6 else '#C44E52' for d in degrees]
    ax.bar(degrees, deg_mases, color=colors_d, edgecolor='black', linewidth=0.5)
    ax.axhline(y=BASELINE_10K, color='gray', linestyle='--', linewidth=1, label='Baseline')
    ax.set_xlabel('Chebyshev Degree $d$')
    ax.set_ylabel('Geometric Mean MASE')
    ax.set_title('(b) Degree Sweep (single-scale)', fontsize=11)
    ax.set_xticks(degrees)
    ax.set_ylim(1.15, 1.26)
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ========== (c) Per-horizon improvement ==========
    ax = fig.add_subplot(gs[1, 0])
    horizons = ['Short\n(n=55)', 'Medium\n(n=21)', 'Long\n(n=21)']
    deltas = [-2.45, -8.51, -12.41]
    win_rates = [37/55*100, 18/21*100, 19/21*100]
    h_colors = ['#55A868', '#4C72B0', '#C44E52']

    x = np.arange(len(horizons))
    width = 0.35
    bars1 = ax.bar(x - width/2, deltas, width, color=h_colors, edgecolor='black', linewidth=0.5, label='$\\Delta$ MASE (%)')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('$\\Delta$ MASE (%)', color='black')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_title('(c) Improvement Scales with Horizon', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add win rate on secondary axis
    ax2 = ax.twinx()
    ax2.bar(x + width/2, win_rates, width, color=[c + '80' for c in ['#55A868', '#4C72B0', '#C44E52']],
            edgecolor='black', linewidth=0.5, alpha=0.5)
    ax2.axhline(y=50, color='gray', linestyle=':', linewidth=0.5)
    ax2.set_ylabel('Win Rate (%)', color='gray')
    ax2.set_ylim(0, 100)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='y', colors='gray')

    # ========== (d) Training curve ==========
    ax = fig.add_subplot(gs[1, 1])
    steps = [10, 20, 30, 50, 70, 100]
    bl = [1.2421, 1.3025, 1.3127, 1.2784, 1.2780, 1.2911]
    hd10 = [None, 1.2401, 1.2064, 1.2412, 1.2241, 1.1918]

    ax.plot(steps, bl, 'o-', color='#4C72B0', linewidth=2, markersize=5, label='Baseline')
    hd10_s = [s for s, v in zip(steps, hd10) if v is not None]
    hd10_v = [v for v in hd10 if v is not None]
    ax.plot(hd10_s, hd10_v, 's-', color='#C44E52', linewidth=2, markersize=5,
            label='Hint $d{=}4$ + 10% drop')
    ax.set_xlabel('Training Steps (K)')
    ax.set_ylabel('Geometric Mean MASE')
    ax.set_title('(d) Training Curve (1K warmup)', fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xticks(steps)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Fill between to show gap
    hd10_interp = np.interp(steps, hd10_s, hd10_v)
    ax.fill_between(steps, bl, hd10_interp, alpha=0.1, color='#C44E52')

    plt.savefig(os.path.join(FIG_DIR, "figure1_overview.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, "figure1_overview.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Wrote {FIG_DIR}/figure1_overview.pdf")


if __name__ == "__main__":
    main()
