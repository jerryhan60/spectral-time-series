"""
Seed consistency figure: HD10 vs BL across 5 seeds.

Shows per-seed MASE, improvement, and win rate to demonstrate robustness.

Generates:
  - analysis/figures/seed_consistency.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)

seeds = [0, 1, 2, 7, 42]
seed_labels = ['s0', 's1', 's2', 's7', 's42']

# Known results from 5-seed m2d evaluation
bl_mase  = np.array([1.2185, 1.1955, 1.2134, 1.2198, 1.2111])
hd_mase  = np.array([1.1577, 1.1902, 1.1798, 1.1698, 1.1852])
ms_mase  = np.array([1.1999, 1.1792, 1.1711, 1.2119, 1.1670])
mshd_mase = np.array([1.2029, 1.1729, 1.1876, 1.1885, 1.1794])

hd_wins  = np.array([66, 54, 70, 78, 61])
ms_wins  = np.array([60, 59, 72, 65, 74])
mshd_wins = np.array([54, 62, 64, 73, 62])

hd_pct  = (1 - hd_mase / bl_mase) * 100
ms_pct  = (1 - ms_mase / bl_mase) * 100
mshd_pct = (1 - mshd_mase / bl_mase) * 100

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 13, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) MASE by seed
ax = axes[0]
x = np.arange(len(seeds))
w = 0.2
ax.bar(x - 1.5*w, bl_mase, w, color='#1f77b4', label='Baseline', alpha=0.85)
ax.bar(x - 0.5*w, hd_mase, w, color='#ff7f0e', label='HD10', alpha=0.85)
ax.bar(x + 0.5*w, ms_mase, w, color='#2ca02c', label='MS46', alpha=0.85)
ax.bar(x + 1.5*w, mshd_mase, w, color='#d62728', label='MSHD10', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(seed_labels)
ax.set_ylabel('GIFT-Eval Geomean MASE')
ax.set_title('(a) MASE by Seed')
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(1.13, 1.25)

# (b) Improvement percentage
ax = axes[1]
ax.bar(x - w, hd_pct, w, color='#ff7f0e', label='HD10', alpha=0.85)
ax.bar(x, ms_pct, w, color='#2ca02c', label='MS46', alpha=0.85)
ax.bar(x + w, mshd_pct, w, color='#d62728', label='MSHD10', alpha=0.85)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(seed_labels)
ax.set_ylabel('MASE Improvement vs BL (%)')
ax.set_title('(b) Improvement by Seed')
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# Add annotations for pooled stats
ax.text(0.98, 0.02, f'HD10 pooled: {hd_pct.mean():.1f}% ± {hd_pct.std():.1f}%\n'
        f'MS46 pooled: {ms_pct.mean():.1f}% ± {ms_pct.std():.1f}%\n'
        f'MSHD10 pooled: {mshd_pct.mean():.1f}% ± {mshd_pct.std():.1f}%',
        transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

# (c) Win rate
ax = axes[2]
ax.bar(x - w, hd_wins, w, color='#ff7f0e', label='HD10', alpha=0.85)
ax.bar(x, ms_wins, w, color='#2ca02c', label='MS46', alpha=0.85)
ax.bar(x + w, mshd_wins, w, color='#d62728', label='MSHD10', alpha=0.85)
ax.axhline(y=48.5, color='gray', linestyle='--', linewidth=1, label='Random (50%)')
ax.set_xticks(x)
ax.set_xticklabels(seed_labels)
ax.set_ylabel('Configs Won (out of 97)')
ax.set_title('(c) Win Rate by Seed')
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(35, 85)

plt.suptitle('Seed Consistency: All Three Hint Methods (m2d, 10K steps)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIG_DIR, f'seed_consistency.{ext}'), bbox_inches='tight')
plt.close(fig)
print("Saved seed_consistency.pdf/png")

# Print summary statistics
print(f"\nHD10:   mean={hd_pct.mean():.2f}% ± {hd_pct.std():.2f}%, min={hd_pct.min():.2f}%, max={hd_pct.max():.2f}%")
print(f"MS46:   mean={ms_pct.mean():.2f}% ± {ms_pct.std():.2f}%, min={ms_pct.min():.2f}%, max={ms_pct.max():.2f}%")
print(f"MSHD10: mean={mshd_pct.mean():.2f}% ± {mshd_pct.std():.2f}%, min={mshd_pct.min():.2f}%, max={mshd_pct.max():.2f}%")
