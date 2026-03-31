"""
Comprehensive ablation summary figure for publication.

Shows MASE improvement (%) relative to baseline for all ablation experiments.
Groups: (1) Filter design, (2) Hint mechanism, (3) Training configuration.
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

BASELINE_MASE = 1.2421
FIGDIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"

# All ablation results — MASE values (None = pending)
ablations = {
    # --- Filter Design ---
    'Cheb d=4 (single)': 1.1802,
    'Cheb d=6 (single)': 1.1882,
    'Cheb d=8 (single)': 1.2219,
    'L2-opt d=6': 1.1784,
    'Multi-scale d=4+d=6': 1.1675,
    'Multi-scale d=4+d=5': 1.2203,
    'Multi-scale d=4+d=8': 1.2196,
    # --- Hint Mechanism ---
    'Duplicate input': 1.2342,
    'Learned 4-tap': 1.2351,
    'Learned 16-tap': 1.2775,
    'Zero hints': 1.3201,
    'Random hints': 1.2806,
    # --- Training Config ---
    'Hint d=4 + 10% drop': 1.1802,
    'MS46 + 10% drop': 1.1817,
    'MS46 + 5% drop': 1.2436,
    'MS46 + 20% drop': 1.2201,
    'Hint ramp 0→1 over 3K': 1.2308,
    'Variable prefix': 1.2128,
}

# Group labels and boundaries
groups = [
    ('Filter Design', ['Cheb d=4 (single)', 'Cheb d=6 (single)', 'Cheb d=8 (single)',
                        'L2-opt d=6', 'Multi-scale d=4+d=6',
                        'Multi-scale d=4+d=5', 'Multi-scale d=4+d=8']),
    ('Hint Mechanism', ['Duplicate input', 'Learned 4-tap', 'Learned 16-tap',
                        'Zero hints', 'Random hints']),
    ('Training Config', ['Hint d=4 + 10% drop', 'MS46 + 10% drop',
                          'MS46 + 5% drop', 'MS46 + 20% drop',
                          'Hint ramp 0→1 over 3K', 'Variable prefix']),
]

# Build ordered lists
labels = []
improvements = []
group_positions = []
group_labels_list = []

pos = 0
for group_name, group_items in groups:
    group_start = pos
    for item in group_items:
        mase = ablations[item]
        labels.append(item)
        if mase is not None:
            improvements.append((BASELINE_MASE - mase) / BASELINE_MASE * 100)
        else:
            improvements.append(None)
        pos += 1
    group_positions.append((group_start, pos - 1))
    group_labels_list.append(group_name)
    pos += 0.5  # gap between groups

# Create figure
fig, ax = plt.subplots(figsize=(7, 6))

y_positions = []
y = 0
group_y_ranges = []  # stores (start_idx, end_idx) into y_positions list
idx = 0
for group_name, group_items in groups:
    g_start_idx = idx
    for _ in group_items:
        y_positions.append(y)
        y += 1
        idx += 1
    group_y_ranges.append((g_start_idx, idx - 1))
    y += 0.8  # gap

y_positions = np.array(y_positions)
n = len(labels)

# Plot bars
for i in range(n):
    imp = improvements[i]
    if imp is None:
        # Pending: draw gray bar with "?"
        ax.barh(y_positions[i], 0, color='lightgray', edgecolor='gray',
                linewidth=0.5, height=0.6)
        ax.text(0.3, y_positions[i], '?', ha='left', va='center',
                fontsize=8, color='gray', fontstyle='italic')
    else:
        color = '#2ca02c' if imp > 0 else '#d62728'
        ax.barh(y_positions[i], imp, color=color, edgecolor='black',
                linewidth=0.5, height=0.6, alpha=0.8)
        # Value label
        offset = 0.3 if imp >= 0 else -0.3
        ha = 'left' if imp >= 0 else 'right'
        ax.text(imp + offset, y_positions[i], f'{imp:+.1f}%',
                ha=ha, va='center', fontsize=7)

# Best result highlight
best_idx = np.argmax([x if x is not None else -999 for x in improvements])
ax.barh(y_positions[best_idx], improvements[best_idx], color='#1f77b4',
        edgecolor='black', linewidth=1.5, height=0.6, alpha=0.9, zorder=5)

ax.set_yticks(y_positions)
ax.set_yticklabels(labels, fontsize=8)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('MASE improvement vs baseline (%)')
ax.set_title('Ablation Study: MASE Improvement at 10K Steps')

# Add group labels
for (g_start, g_end), g_name in zip(group_y_ranges, group_labels_list):
    mid = (y_positions[int(g_start)] + y_positions[int(g_end)]) / 2
    ax.text(-10, mid, g_name, ha='right', va='center', fontsize=9,
            fontweight='bold', fontstyle='italic', color='#333333')

# Add baseline reference line text
ax.text(0.02, 0.02, f'Baseline MASE: {BASELINE_MASE:.4f}',
        transform=ax.transAxes, fontsize=8, color='gray',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

ax.grid(True, alpha=0.2, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{FIGDIR}/ablation_summary.pdf')
plt.savefig(f'{FIGDIR}/ablation_summary.png')
print("Saved ablation_summary.pdf/png")
