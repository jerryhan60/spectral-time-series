"""
Computational overhead analysis of hint preconditioning.

Shows that hints add minimal parameters (+0.1%) and compute overhead.
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

# Model parameters
baseline_params = 11_393_712  # 11.39M
# MS46 adds: input projection goes from 2*16=32 to 4*16=64 input dims
# The linear projection is d_model x input_dim = 384 x 32 -> 384 x 64
# Extra params = 384 * 32 = 12,288
ms46_extra_params = 384 * 32  # 12,288
ms46_total = baseline_params + ms46_extra_params

# FIR computation cost (per sample)
# Chebyshev FIR: d=4 and d=6 convolutions at stride=16
# For sequence length L, each FIR costs O(L*d) multiply-adds
# Transformer self-attention: O(L^2 * d_model)
seq_len = 4000
d_model = 384
n_layers = 6
n_heads = 6

# FLOPs comparison
fir_flops = seq_len * (4 + 6)  # ~40K FLOPs for both FIR filters
attention_flops = n_layers * (4 * seq_len * d_model + 2 * seq_len**2 * (d_model // n_heads) * n_heads)
ffn_flops = n_layers * 2 * seq_len * d_model * (4 * d_model)  # FFN is 4x expansion

total_model_flops = attention_flops + ffn_flops

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

# Panel A: Parameter comparison
ax = axes[0]
labels = ['Baseline', 'MS46\n(+hints)']
params = [baseline_params / 1e6, ms46_total / 1e6]
colors = ['#1f77b4', '#2ca02c']
bars = ax.bar(labels, params, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
ax.set_ylabel('Parameters (M)')
ax.set_title('(a) Model parameters')

# Annotate
for bar, p in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{p:.2f}M', ha='center', va='bottom', fontsize=9)
ax.text(0.5, 0.85, f'+{ms46_extra_params:,} params\n(+{ms46_extra_params/baseline_params*100:.2f}%)',
        transform=ax.transAxes, ha='center', fontsize=9, color='#2ca02c',
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))
ax.set_ylim(0, max(params) * 1.2)

# Panel B: Compute overhead
ax = axes[1]
components = ['Attention\n+ FFN', 'FIR hints']
flops = [total_model_flops / 1e9, fir_flops / 1e9]
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(components, flops, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
ax.set_ylabel('GFLOPs per forward pass')
ax.set_title('(b) Compute cost')
ax.set_yscale('log')
ax.text(0.5, 0.85, f'FIR overhead:\n{fir_flops/total_model_flops*100:.4f}%',
        transform=ax.transAxes, ha='center', fontsize=9, color='#ff7f0e',
        bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))

# Panel C: Improvement per parameter
ax = axes[2]
methods = ['Baseline\n(11.39M)', 'MS46\n(11.41M)', 'hd10\n(11.39M)', 'STU Hybrid\n(~14M)']
mase_vals = [1.2421, 1.1675, 1.1802, 1.3044]
param_counts = [11.39, 11.41, 11.39, 14.0]
improvements = [(1.2421 - m) / 1.2421 * 100 for m in mase_vals]

colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
for i, (method, imp, pc, color) in enumerate(zip(methods, improvements, param_counts, colors)):
    ax.scatter(pc, imp, color=color, s=100, edgecolors='black', linewidth=0.5, zorder=5)
    ax.annotate(method, (pc, imp), textcoords='offset points',
               xytext=(5, 5), fontsize=7, color=color)

ax.axhline(y=0, color='black', linewidth=0.5, linestyle=':')
ax.set_xlabel('Parameters (M)')
ax.set_ylabel('MASE improvement (%)')
ax.set_title('(c) Efficiency: improvement per param')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/compute_overhead.pdf')
plt.savefig(f'{FIGDIR}/compute_overhead.png')
print("Saved compute_overhead.pdf/png")
print(f"\nBaseline: {baseline_params:,} params")
print(f"MS46: {ms46_total:,} params (+{ms46_extra_params:,} = +{ms46_extra_params/baseline_params*100:.2f}%)")
print(f"FIR FLOPs: {fir_flops:,}")
print(f"Model FLOPs: {total_model_flops:,}")
print(f"FIR overhead: {fir_flops/total_model_flops*100:.6f}%")
