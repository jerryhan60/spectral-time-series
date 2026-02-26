#!/usr/bin/env python3
"""Plot Chebyshev degree sweep GIFT-Eval results — raw and normalized MASE."""
import matplotlib.pyplot as plt
import numpy as np

degrees = [1, 2, 3, 4, 5, 6, 7]

# Normalized Geo Mean MASE at epoch 500 (50K steps)
norm_500 = [0.8836, 1.2447, 1.4321, 1.5823, 1.6725, 1.7496, 1.9944]

# Normalized Geo Mean MASE at epoch 1000 (100K steps)
norm_1000 = [0.9146, 1.2681, 1.4502, 1.5769, 1.7090, 1.7892, 2.0209]

fig, ax = plt.subplots(figsize=(8, 5))

# Plot epoch 500
d500 = [d for d, v in zip(degrees, norm_500) if v is not None]
v500 = [v for v in norm_500 if v is not None]
ax.plot(d500, v500, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Epoch 500 (50K steps)')

# Plot epoch 1000
d1000 = [d for d, v in zip(degrees, norm_1000) if v is not None]
v1000 = [v for v in norm_1000 if v is not None]
ax.plot(d1000, v1000, 's-', color='#F44336', linewidth=2, markersize=8, label='Epoch 1000 (100K steps)')

# Reference lines
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Seasonal naive (1.0)')
ax.axhline(y=0.9415, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='AK baseline (0.94)')

# Mark d=1 as control
ax.annotate('control\n(identity)', xy=(1, 0.8836), xytext=(1.8, 0.75),
            fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray'),
            color='gray')

ax.set_xlabel('Chebyshev Degree', fontsize=12)
ax.set_ylabel('Normalized Geo Mean MASE', fontsize=12)
ax.set_title('EXP-1: Chebyshev Preconditioning Degree Sweep\nGIFT-Eval (97 configs, normalized by seasonal naive)', fontsize=13)
ax.set_xticks(degrees)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.6, 2.2)

plt.tight_layout()
plt.savefig('/scratch/gpfs/EHAZAN/jh1161/chebyshev_degree_sweep.png', dpi=150)
print("Saved to chebyshev_degree_sweep.png")
