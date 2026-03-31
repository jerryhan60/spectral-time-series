"""
Convergence advantage analysis.

Shows that hints achieve better results with less compute:
- MS46 at 10K beats baseline at any step up to 100K
- Equivalent to ~3x compute savings
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

FIGDIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"

# Baseline training curve (10K warmup)
bl_steps = [10000, 20000, 30000, 50000, 70000, 100000]
bl_mase = [1.2421, 1.3025, 1.3127, 1.2784, 1.2780, 1.2911]

# Baseline training curve (1K warmup)
bl1k_steps = [90000, 100000]
bl1k_mase = [1.2913, 1.2833]

# hd10 training curve (1K warmup, hint d=4 + 10% dropout)
hd_steps = [20000, 30000, 50000, 70000, 100000]
hd_mase = [1.2401, 1.2064, 1.2412, 1.2241, 1.1918]

# MS46 at 10K
ms46_10k = 1.1675

fig, ax = plt.subplots(figsize=(7, 4.5))

# Plot training curves
ax.plot(bl_steps, bl_mase, 'o-', color='#1f77b4', linewidth=2, markersize=7,
        label='Baseline (10K warmup)', zorder=3)
ax.plot(hd_steps, hd_mase, 's-', color='#ff7f0e', linewidth=2, markersize=7,
        label='Hint d=4 + 10% drop (1K warmup)', zorder=3)

# Baseline with 1K warmup
ax.plot(bl1k_steps, bl1k_mase, '^--', color='#9467bd', linewidth=1.5, markersize=6,
        label='Baseline (1K warmup)', zorder=3, alpha=0.8)

# MS46 point
ax.plot([10000], [ms46_10k], 'D', color='#2ca02c', markersize=10,
        label=f'MS46 d=4+d=6 (10K)', zorder=5, markeredgecolor='black', markeredgewidth=1)

# Horizontal line at MS46 level
ax.axhline(y=ms46_10k, color='#2ca02c', linewidth=1, linestyle='--', alpha=0.5)
ax.text(105000, ms46_10k + 0.003, f'MS46@10K: {ms46_10k:.4f}', fontsize=8,
        color='#2ca02c', va='bottom')

# Horizontal line at best baseline
best_bl = min(bl_mase)
best_bl_step = bl_steps[bl_mase.index(best_bl)]
ax.axhline(y=best_bl, color='#1f77b4', linewidth=1, linestyle=':', alpha=0.5)
ax.text(105000, best_bl - 0.003, f'Best BL: {best_bl:.4f}\n@{best_bl_step//1000}K', fontsize=8,
        color='#1f77b4', va='top')

# Arrow showing compute advantage
# MS46 at 10K (1.1675) vs baseline at 100K (1.2911)
# Even at 100K, baseline never reaches MS46's 10K performance
ax.annotate('', xy=(10000, ms46_10k), xytext=(100000, ms46_10k),
            arrowprops=dict(arrowstyle='<->', color='#2ca02c', linewidth=1.5))
ax.text(50000, ms46_10k - 0.012, 'MS46@10K beats\nbaseline at ANY step',
        ha='center', fontsize=8, color='#2ca02c', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))

# Show improvement annotation at 30K (best hd10 point)
ax.annotate(f'-8.1% vs BL', xy=(30000, hd_mase[1]),
            xytext=(35000, 1.17), fontsize=8, color='#ff7f0e',
            arrowprops=dict(arrowstyle='->', color='#ff7f0e', linewidth=1))

ax.set_xlabel('Training steps')
ax.set_ylabel('GIFT-Eval Geometric Mean MASE')
ax.set_title('Convergence Advantage of Hint Preconditioning')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 115000)
ax.set_ylim(1.14, 1.33)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/convergence_advantage.pdf')
plt.savefig(f'{FIGDIR}/convergence_advantage.png')
print("Saved convergence_advantage.pdf/png")
