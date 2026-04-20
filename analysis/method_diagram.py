"""
Method diagram: FIR hint preconditioning pipeline.

Shows a schematic of how Chebyshev FIR hints are computed and fed to the model.
This is a simplified illustration, not the actual architecture diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches

rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

FIGDIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Panel A: Time series → patches → model pipeline
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(a) Hint preconditioning pipeline', fontsize=11, fontweight='bold')

# Draw boxes
boxes = [
    (0.5, 4, 2, 1.2, 'Time\nSeries $x$', '#E8F4FD'),
    (3.5, 4, 2, 1.2, 'FIR Filter\n$h * x$', '#FFF3E0'),
    (3.5, 2, 2, 1.2, 'Residual\n$h*x - x$', '#E8F5E9'),
    (7, 3, 2.5, 2.2, 'Moirai2\nDecoder', '#F3E5F5'),
]

for x, y, w, h, text, color in boxes:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8)

# Arrows
arrow_style = dict(arrowstyle='->', color='black', linewidth=1.5)
# x → FIR filter
ax.annotate('', xy=(3.4, 4.6), xytext=(2.6, 4.6), arrowprops=arrow_style)
# FIR filter → residual
ax.annotate('', xy=(4.5, 3.3), xytext=(4.5, 3.9), arrowprops=arrow_style)
# x → model (direct)
ax.annotate('', xy=(6.9, 4.2), xytext=(2.6, 4.2),
            arrowprops=dict(arrowstyle='->', color='#1f77b4', linewidth=1.5,
                           connectionstyle='arc3,rad=0.3'))
ax.text(4.7, 5.2, 'target channel', fontsize=7, color='#1f77b4', ha='center')
# residual → model
ax.annotate('', xy=(6.9, 3.2), xytext=(5.6, 2.6),
            arrowprops=dict(arrowstyle='->', color='#2ca02c', linewidth=1.5))
ax.text(6.5, 2.4, 'hint channels', fontsize=7, color='#2ca02c', ha='center')
# model → prediction
ax.annotate('', xy=(9.5, 1.5), xytext=(8.25, 2.9),
            arrowprops=arrow_style)
ax.text(9.5, 1.2, '$\\hat{y}$', fontsize=12, ha='center')
# Note: no hint at inference
ax.text(5, 0.5, 'At inference: hint channels = 0', fontsize=8,
        ha='center', color='red', fontstyle='italic',
        bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))

# Panel B: Patching illustration
ax = axes[0, 1]
np.random.seed(42)
t = np.linspace(0, 4*np.pi, 256)
signal = np.sin(t) + 0.5*np.sin(5*t) + 0.3*np.sin(13*t) + 0.1*np.random.randn(len(t))
P = 16

ax.plot(t, signal, color='#1f77b4', linewidth=0.8, alpha=0.5)

# Show patches
for i in range(0, min(5, len(t)//P)):
    start = i * P
    end = start + P
    color = '#E8F4FD' if i % 2 == 0 else '#FFF3E0'
    ax.fill_between(t[start:end], signal[start:end].min()-0.3, signal[start:end].max()+0.3,
                    alpha=0.3, color=color)
    ax.axvline(t[start], color='gray', linewidth=0.3, linestyle=':')

ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('(b) Patch-size P=16 discretization', fontsize=11)
ax.text(0.5, 0.95, 'Each patch sees only P=16\nconsecutive samples',
        transform=ax.transAxes, fontsize=8, va='top', ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

# Panel C: FIR filter effect on signal
ax = axes[1, 0]

# Original signal with high-freq component
t_short = np.linspace(0, 2*np.pi, 128)
low_freq = np.sin(t_short)
high_freq = 0.4 * np.sin(8*t_short)
full = low_freq + high_freq

# Chebyshev d=4 FIR (approximate)
from numpy.polynomial.chebyshev import Chebyshev
d = 4
cheb = Chebyshev.basis(d, domain=[0, 1])
coeffs = cheb.coef
# Normalize to monic
coeffs = coeffs / coeffs[-1] if abs(coeffs[-1]) > 0 else coeffs
kernel = coeffs

# Apply FIR (simplified)
filtered = np.convolve(full, kernel/kernel.sum(), mode='same')
residual = filtered - full

ax.plot(t_short, full, color='#1f77b4', linewidth=1.5, label='Original $x$', alpha=0.8)
ax.plot(t_short, filtered, color='#ff7f0e', linewidth=1.5, label='$h * x$', alpha=0.8)
ax.plot(t_short, residual, color='#2ca02c', linewidth=1.5, label='Hint: $h*x - x$', alpha=0.8)

ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('(c) FIR hint = multi-scale residual', fontsize=11)
ax.legend(loc='upper right', framealpha=0.8, fontsize=8)
ax.grid(True, alpha=0.2)

# Panel D: Multi-channel input construction
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('(d) Multi-channel patch input', fontsize=11, fontweight='bold')

# Show how a single patch becomes 4 channels
channel_names = ['Target $x$', 'Mask', 'Hint $h_4*x - x$', 'Hint $h_6*x - x$']
channel_colors = ['#1f77b4', '#9467bd', '#2ca02c', '#d62728']
channel_dims = ['P=16', 'P=16', 'P=16', 'P=16']

for i, (name, color, dim) in enumerate(zip(channel_names, channel_colors, channel_dims)):
    y_pos = 5 - i * 1.2
    # Draw channel bar
    rect = FancyBboxPatch((1, y_pos), 5, 0.8, boxstyle="round,pad=0.05",
                          facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.8, y_pos + 0.4, name, ha='right', va='center', fontsize=8, color=color)
    ax.text(6.3, y_pos + 0.4, dim, ha='left', va='center', fontsize=7, color='gray')

# Bracket for total input
ax.annotate('', xy=(7.5, 5.8), xytext=(7.5, 1),
            arrowprops=dict(arrowstyle='|-|', color='black', linewidth=1))
ax.text(8, 3.4, 'Input dim\n= 4×16 = 64', ha='left', va='center', fontsize=9, fontweight='bold')

# Baseline comparison
ax.text(5, 0.3, 'Baseline: 2 channels (target + mask) = 32 dims',
        ha='center', va='center', fontsize=8, color='gray', fontstyle='italic')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/method_diagram.pdf')
plt.savefig(f'{FIGDIR}/method_diagram.png')
print("Saved method_diagram.pdf/png")
