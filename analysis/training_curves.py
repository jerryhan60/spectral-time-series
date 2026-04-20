"""
Training loss curves comparison figure for publication.

Shows training loss vs steps for baseline, hd10, and ms46.
"""

import numpy as np
import json
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

# Load existing data
with open('/scratch/gpfs/EHAZAN/jh1161/training_loss_data.json') as f:
    data = json.load(f)

bl_loss = data['baseline']['train/PackedQuantileMAELoss_step']
hd_loss = data['hd10']['train/PackedQuantileMAELoss_step']

bl_steps = np.array(bl_loss['steps'])
bl_vals = np.array(bl_loss['values'])
hd_steps = np.array(hd_loss['steps'])
hd_vals = np.array(hd_loss['values'])

# Try to load ms46 TB data
ms46_steps, ms46_vals = None, None
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator('/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted/q_ms_d4d6_20260223_134653/logs/version_0')
    ea.Reload()
    tags = ea.Tags()['scalars']
    loss_tag = 'train/PackedQuantileMAELoss_step'
    if loss_tag in tags:
        events = ea.Scalars(loss_tag)
        ms46_steps = np.array([e.step for e in events])
        ms46_vals = np.array([e.value for e in events])
        print(f"Loaded ms46 TB data: {len(ms46_steps)} points, steps {ms46_steps[0]}-{ms46_steps[-1]}")
except Exception as e:
    print(f"Could not load ms46 TB data: {e}")
    # Try extracting from log files
    try:
        import re
        log_file = '/scratch/gpfs/EHAZAN/jh1161/logs/slurm-5181258.out'  # or another ms46 log
        # Will try an alternative approach later
    except:
        pass

# Smoothing function
def smooth(y, window=50):
    if len(y) <= window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel A: Full 100K training curves (baseline vs hd10)
ax = axes[0]

# Smooth
window = 50
bl_smooth = smooth(bl_vals, window)
hd_smooth = smooth(hd_vals, window)
bl_steps_smooth = bl_steps[window-1:]
hd_steps_smooth = hd_steps[window-1:]

ax.plot(bl_steps_smooth, bl_smooth, color='#1f77b4', linewidth=1.5, label='Baseline (10K warmup)', alpha=0.9)
ax.plot(hd_steps_smooth, hd_smooth, color='#ff7f0e', linewidth=1.5, label='Hint d=4 + 10% drop (1K warmup)', alpha=0.9)

if ms46_steps is not None:
    ms46_smooth = smooth(ms46_vals, window)
    ms46_steps_smooth = ms46_steps[window-1:]
    ax.plot(ms46_steps_smooth, ms46_smooth, color='#2ca02c', linewidth=1.5, label='Multi-scale d=4+d=6', alpha=0.9)

# Mark evaluation points
eval_points_bl = {10000: 1.2421, 20000: 1.3025, 30000: 1.3127, 50000: 1.2784, 70000: 1.2780, 100000: 1.2911}
eval_points_hd = {10000: None, 20000: 1.2401, 30000: 1.2064, 50000: 1.2412, 70000: 1.2241, 100000: 1.1918}

ax.set_xlabel('Training step')
ax.set_ylabel('Training loss')
ax.set_title('(a) Training loss over 100K steps')
ax.legend(loc='upper right', framealpha=0.8, fontsize=7)
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 100000)

# Panel B: Early training (first 10K steps) - zoomed
ax = axes[1]

mask_bl = bl_steps <= 10000
mask_hd = hd_steps <= 10000

bl_early = smooth(bl_vals[mask_bl], min(window, len(bl_vals[mask_bl])//3))
hd_early = smooth(hd_vals[mask_hd], min(window, len(hd_vals[mask_hd])//3))
bl_steps_early = bl_steps[mask_bl][len(bl_vals[mask_bl]) - len(bl_early):]
hd_steps_early = hd_steps[mask_hd][len(hd_vals[mask_hd]) - len(hd_early):]

ax.plot(bl_steps_early, bl_early, color='#1f77b4', linewidth=1.5, label='Baseline', alpha=0.9)
ax.plot(hd_steps_early, hd_early, color='#ff7f0e', linewidth=1.5, label='hd10', alpha=0.9)

if ms46_steps is not None:
    mask_ms = ms46_steps <= 10000
    ms_early = smooth(ms46_vals[mask_ms], min(window, len(ms46_vals[mask_ms])//3))
    ms_steps_early = ms46_steps[mask_ms][len(ms46_vals[mask_ms]) - len(ms_early):]
    ax.plot(ms_steps_early, ms_early, color='#2ca02c', linewidth=1.5, label='MS46', alpha=0.9)

# Mark the warmup period
ax.axvspan(0, 1000, alpha=0.05, color='orange', label='Warmup (1K)')
ax.axvline(x=1000, color='orange', linewidth=0.5, linestyle='--', alpha=0.5)

ax.set_xlabel('Training step')
ax.set_ylabel('Training loss')
ax.set_title('(b) Early training (first 10K steps)')
ax.legend(loc='upper right', framealpha=0.8, fontsize=7)
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 10000)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/training_curves.pdf')
plt.savefig(f'{FIGDIR}/training_curves.png')
print("Saved training_curves.pdf/png")

# Also create a MASE vs training steps figure
fig2, ax2 = plt.subplots(figsize=(6, 4))

bl_steps_eval = sorted(eval_points_bl.keys())
bl_mase = [eval_points_bl[s] for s in bl_steps_eval]
hd_steps_eval = [s for s in sorted(eval_points_hd.keys()) if eval_points_hd[s] is not None]
hd_mase = [eval_points_hd[s] for s in hd_steps_eval]

ax2.plot(bl_steps_eval, bl_mase, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Baseline (10K warmup)')
ax2.plot(hd_steps_eval, hd_mase, 's-', color='#ff7f0e', linewidth=2, markersize=6, label='Hint d=4 + 10% drop (1K warmup)')

# Add ms46 point if available
ax2.plot([10000], [1.1675], 'D', color='#2ca02c', markersize=8, label='MS46 d=4+d=6 (10K)', zorder=5)

# Improvement annotations
for step in hd_steps_eval:
    bl_val = eval_points_bl[step]
    hd_val = eval_points_hd[step]
    imp = (bl_val - hd_val) / bl_val * 100
    ax2.annotate(f'{imp:+.1f}%', xy=(step, hd_val), xytext=(0, -12),
                textcoords='offset points', fontsize=7, ha='center', color='#ff7f0e')

ax2.set_xlabel('Training step')
ax2.set_ylabel('GIFT-Eval Geo Mean MASE')
ax2.set_title('Evaluation MASE vs Training Steps')
ax2.legend(loc='upper right', framealpha=0.8)
ax2.grid(True, alpha=0.2)
ax2.set_xlim(0, 105000)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/mase_vs_steps.pdf')
plt.savefig(f'{FIGDIR}/mase_vs_steps.png')
print("Saved mase_vs_steps.pdf/png")
