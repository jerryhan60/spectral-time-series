#!/bin/bash
#SBATCH --job-name=trcurve
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/trcurve_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/trcurve_%j.err

module load anaconda3/2024.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

python3 << 'PYEOF'
"""Extract training loss from TensorBoard logs and generate figures."""
import os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2"
FIG = "/scratch/gpfs/EHAZAN/jh1161/paper/figures"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

def extract_loss(run_dir, tag="train/PackedQuantileMAELoss_epoch"):
    """Extract training loss from TensorBoard event files."""
    log_dirs = sorted(glob.glob(os.path.join(run_dir, "logs/version_*")))
    if not log_dirs:
        return [], []
    # Use latest version
    ea = EventAccumulator(log_dirs[-1])
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        # Try step loss
        tag = "train/PackedQuantileMAELoss_step"
        if tag not in ea.Tags().get('scalars', []):
            print(f"  No loss tag found in {log_dirs[-1]}")
            print(f"  Available tags: {ea.Tags().get('scalars', [])[:10]}")
            return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

# ============================================================
# Figure: Training loss for 10K runs (BL vs HD10 vs Dup vs Zero, seed 0)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

runs_10k = {
    'Baseline': 'baseline_m2d_seed0_10k',
    'Hint (ours)': 'hd10_m2d_seed0_10k',
    'Zero': 'zero_m2d_seed0_10k',
    'Duplicate': 'duplicate_m2d_seed0_10k',
}
colors_10k = {'Baseline': '#1b76d2', 'Hint (ours)': '#483e8c', 'Zero': '#8bb8e8', 'Duplicate': '#dc8969'}

for name, run in runs_10k.items():
    run_dir = os.path.join(BASE, run)
    if os.path.exists(run_dir):
        steps, values = extract_loss(run_dir)
        if steps:
            ax.plot(steps, values, label=name, color=colors_10k[name], linewidth=1.5, alpha=0.8)
            print(f"  {name}: {len(steps)} points, final={values[-1]:.4f}")
        else:
            print(f"  {name}: no data")

ax.set_xlabel('Training step')
ax.set_ylabel('Training loss')
ax.set_title('Training loss (10K steps, seed 0)')
ax.legend(frameon=False, fontsize=9)
ax.set_xlim(0, 10000)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_training_loss_10k.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG, 'fig_training_loss_10k.png'), bbox_inches='tight')
plt.close()
print("Saved fig_training_loss_10k")

# ============================================================
# Figure: Training loss for 100K runs (BL vs HD10, seed 0)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

runs_100k = {
    'Baseline': 'baseline_m2d_s0_100k',
    'Hint (ours)': 'hd10_m2d_s0_100k',
    'MSHD10': 'mshd10_m2d_s0_100k',
}
colors_100k = {'Baseline': '#1b76d2', 'Hint (ours)': '#483e8c', 'MSHD10': '#dc8969'}

for name, run in runs_100k.items():
    run_dir = os.path.join(BASE, run)
    if os.path.exists(run_dir):
        steps, values = extract_loss(run_dir)
        if steps:
            # Smooth with rolling average for readability
            if len(values) > 50:
                window = max(1, len(values) // 100)
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:]
                ax.plot(smooth_steps, smoothed, label=name, color=colors_100k[name], linewidth=1.5)
            else:
                ax.plot(steps, values, label=name, color=colors_100k[name], linewidth=1.5)
            print(f"  {name}: {len(steps)} points, final={values[-1]:.4f}")

ax.set_xlabel('Training step')
ax.set_ylabel('Training loss')
ax.set_title('Training loss (100K steps, seed 0)')
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_training_loss_100k.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG, 'fig_training_loss_100k.png'), bbox_inches='tight')
plt.close()
print("Saved fig_training_loss_100k")

print("\nDone.")
PYEOF
