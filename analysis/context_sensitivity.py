"""
Context sensitivity analysis: MASE as a function of context length.

Shows that MS46 at reduced context (1000) still beats baseline at full context (4000).
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
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Results from context sweep evaluation
# MS46 @ different context lengths
ms46_ctx = {
    1000: 1.2074,
    # 2000: TBD (currently evaluating)
    4000: 1.1675,  # standard evaluation
}

# Baseline @ different context lengths
bl_ctx = {
    1000: 1.2512,
    # 2000: TBD
    4000: 1.2421,  # standard evaluation
}

fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

# Plot baseline
bl_x = sorted(bl_ctx.keys())
bl_y = [bl_ctx[x] for x in bl_x]
ax.plot(bl_x, bl_y, 's-', color='#1f77b4', markersize=8, linewidth=2, label='Baseline')

# Plot ms46
ms_x = sorted(ms46_ctx.keys())
ms_y = [ms46_ctx[x] for x in ms_x]
ax.plot(ms_x, ms_y, 'o-', color='#ff7f0e', markersize=8, linewidth=2, label='MS46 (d=4+d=6)')

# Highlight key comparison: MS46@1000 vs BL@4000
ax.annotate('', xy=(1000, ms46_ctx[1000]), xytext=(4000, bl_ctx[4000]),
            arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=1.5))
ax.text(2500, (ms46_ctx[1000] + bl_ctx[4000])/2 + 0.005,
        f'MS46@1K beats\nBL@4K by {(bl_ctx[4000] - ms46_ctx[1000])/bl_ctx[4000]*100:.1f}%',
        ha='center', va='bottom', fontsize=8, color='#2ca02c',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2ca02c', alpha=0.8))

ax.set_xlabel('Context length')
ax.set_ylabel('MASE (geometric mean)')
ax.set_title('Context length sensitivity')
ax.legend(loc='upper right', framealpha=0.8)
ax.set_xlim(500, 4500)
ax.grid(True, alpha=0.3)
ax.invert_yaxis()  # Lower MASE is better → inverted for visual clarity

plt.tight_layout()
plt.savefig('/scratch/gpfs/EHAZAN/jh1161/analysis/figures/context_sensitivity.pdf')
plt.savefig('/scratch/gpfs/EHAZAN/jh1161/analysis/figures/context_sensitivity.png')
print("Saved context_sensitivity.pdf/png")

# Print summary
print("\nContext Sensitivity Results:")
print(f"  MS46@1000: {ms46_ctx[1000]:.4f}")
print(f"  MS46@4000: {ms46_ctx[4000]:.4f} (improvement from more context: {(ms46_ctx[1000]-ms46_ctx[4000])/ms46_ctx[1000]*100:.1f}%)")
print(f"  BL@1000:   {bl_ctx[1000]:.4f}")
print(f"  BL@4000:   {bl_ctx[4000]:.4f}")
print(f"  MS46@1000 vs BL@4000: MS46 is {(bl_ctx[4000]-ms46_ctx[1000])/bl_ctx[4000]*100:.1f}% better")
print(f"  Effective context gain: hints at ctx=1000 equivalent to ctx>{4000}")
