"""
Publication-quality training curve figure: BL vs HD10 vs MSHD10 (1K-100K steps).

Includes early training (1K-5K) to show hint crossover dynamics.

Generates:
  - analysis/figures/training_curve_100k.pdf
  - analysis/figures/improvement_over_training.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Complete training curve data (seed 0, cosine schedule)
# BL: 1K-100K (13 points, 20K/30K missing from 100K-run evals)
bl_steps = np.array([1, 3, 5, 10, 40, 50, 60, 70, 80, 90, 100]) * 1000
bl_mase  = np.array([1.4151, 1.2302, 1.2295, 1.2393, 1.2564, 1.2362, 1.2352, 1.2309, 1.3038, 1.2737, 1.2783])

# HD10: 1K-100K (13 points)
hd_steps = np.array([1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) * 1000
hd_mase  = np.array([1.4638, 1.2242, 1.2015, 1.2160, 1.1762, 1.1775, 1.1970, 1.1770, 1.1831, 1.1903, 1.1982, 1.1780, 1.1739])

# MSHD10: 10K-100K (10 points)
ms_steps = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) * 1000
ms_mase  = np.array([1.2120, 1.2001, 1.1786, 1.1980, 1.1679, 1.1743, 1.1765, 1.1646, 1.1610, 1.1622])

# Win rates vs BL (at shared steps only)
# Steps where BL and HD10 both have data
hd_vs_bl_steps = np.array([1, 3, 5, 10, 40, 50, 60, 70, 80, 90, 100]) * 1000
hd_vs_bl_wins  = np.array([42, 54, 62, 56, 60, 66, 69, 63, 74, 76, 74])

# Steps where BL and MSHD10 both have data
ms_vs_bl_steps = np.array([10, 40, 50, 60, 70, 80, 90, 100]) * 1000
ms_vs_bl_wins  = np.array([70, 67, 72, 71, 76, 79, 83, 85])

plt.rcParams.update({
    "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
    "axes.titlesize": 13, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel (a): MASE over training
ax1.plot(bl_steps/1000, bl_mase, 'o-', color='#1f77b4', linewidth=2, markersize=5, label='Baseline', zorder=3)
ax1.plot(hd_steps/1000, hd_mase, 's-', color='#ff7f0e', linewidth=2, markersize=5, label='HD10', zorder=3)
ax1.plot(ms_steps/1000, ms_mase, 'D-', color='#d62728', linewidth=2, markersize=5, label='MSHD10', zorder=3)

# Mark key points
ax1.annotate('BL collapse\n(1.304)', xy=(80, 1.3038), xytext=(65, 1.35),
            fontsize=8, color='#1f77b4', ha='center',
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1))
ax1.annotate('Best: 1.161', xy=(90, 1.1610), xytext=(75, 1.13),
            fontsize=8, color='#d62728', ha='center',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1))
ax1.annotate('HD10 worse\nat 1K (warmup)', xy=(1, 1.4638), xytext=(12, 1.44),
            fontsize=7, color='#ff7f0e', ha='center',
            arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1))

ax1.set_xlabel('Training Steps (K)')
ax1.set_ylabel('GIFT-Eval Geometric Mean MASE')
ax1.set_title('(a) Evaluation MASE During Training')
ax1.legend(frameon=False, loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.set_ylim(1.12, 1.48)
ax1.set_xlim(-2, 105)

# Panel (b): Win rate vs BL over training
ax2.plot(hd_vs_bl_steps/1000, hd_vs_bl_wins, 's-', color='#ff7f0e', linewidth=2, markersize=5, label='HD10 vs BL')
ax2.plot(ms_vs_bl_steps/1000, ms_vs_bl_wins, 'D-', color='#d62728', linewidth=2, markersize=5, label='MSHD10 vs BL')
ax2.axhline(y=48.5, color='gray', linestyle='--', linewidth=1, label='Random (50%)')

# Shade regions
ax2.fill_between([0, 105], 48.5, 97, color='green', alpha=0.05)
ax2.fill_between([0, 105], 0, 48.5, color='red', alpha=0.05)

ax2.annotate('85/97\n(p=5e-15)', xy=(100, 85), xytext=(85, 92),
            fontsize=8, color='#d62728', ha='center',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1))
ax2.annotate('42/97\n(HD10 loses)', xy=(1, 42), xytext=(12, 35),
            fontsize=7, color='#ff7f0e', ha='center',
            arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1))

ax2.set_xlabel('Training Steps (K)')
ax2.set_ylabel('Configs Won vs Baseline (out of 97)')
ax2.set_title('(b) Win Rate Against Baseline')
ax2.legend(frameon=False, loc='lower right')
ax2.grid(True, alpha=0.2)
ax2.set_ylim(30, 97)
ax2.set_xlim(-2, 105)

plt.suptitle('Training Dynamics: Polynomial Hint Preconditioning (seed 0, cosine schedule)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIG_DIR, f'training_curve_100k.{ext}'), bbox_inches='tight')
plt.close(fig)
print("Saved training_curve_100k.pdf/png")

# Improvement over time figure (at shared steps only)
fig, ax = plt.subplots(figsize=(8, 4.5))

# HD10 improvement at shared BL steps
shared_steps_hd = [10, 40, 50, 60, 70, 80, 90, 100]
bl_at_shared_hd = np.array([1.2393, 1.2564, 1.2362, 1.2352, 1.2309, 1.3038, 1.2737, 1.2783])
hd_at_shared = np.array([1.2160, 1.1970, 1.1770, 1.1831, 1.1903, 1.1982, 1.1780, 1.1739])
hd_pct = (1 - hd_at_shared / bl_at_shared_hd) * 100

# MSHD10 at same BL steps
ms_at_shared = np.array([1.2120, 1.1980, 1.1679, 1.1743, 1.1765, 1.1646, 1.1610, 1.1622])
ms_pct = (1 - ms_at_shared / bl_at_shared_hd) * 100

ax.plot(shared_steps_hd, hd_pct, 's-', color='#ff7f0e', linewidth=2, markersize=6, label='HD10')
ax.plot(shared_steps_hd, ms_pct, 'D-', color='#d62728', linewidth=2, markersize=6, label='MSHD10')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.fill_between(shared_steps_hd, 0, 15, color='green', alpha=0.05)

# Annotate key values
for i, step in enumerate(shared_steps_hd):
    if step in [10, 50, 80, 100]:
        ax.annotate(f'{ms_pct[i]:.1f}%', xy=(step, ms_pct[i]),
                   xytext=(0, 8), textcoords='offset points',
                   fontsize=7, color='#d62728', ha='center')

ax.set_xlabel('Training Steps (K)')
ax.set_ylabel('MASE Improvement vs Baseline (%)')
ax.set_title('Relative Improvement Over Training\n(higher = better, seed 0, cosine schedule)', fontweight='bold')
ax.legend(frameon=False)
ax.grid(True, alpha=0.2)
ax.set_xlim(5, 105)
ax.set_ylim(-1, 12)

plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIG_DIR, f'improvement_over_training.{ext}'), bbox_inches='tight')
plt.close(fig)
print("Saved improvement_over_training.pdf/png")
