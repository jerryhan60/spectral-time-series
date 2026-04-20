"""Extract training loss from TensorBoard event files and generate comparison plots."""
import os
import sys
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/lib/python3.12/site-packages')

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Define runs to compare
RUNS = {
    "Baseline (13.83M)": "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/moirai_small_baseline_20260126_163112/logs/version_0",
    "Approx STU (12.53M)": "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_stu/lotsa_v1_unweighted/moirai_small_stu_fast_20260127_181919/logs/version_0",
    "Sandwich (15.65M)": "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_stu_sandwich/lotsa_v1_unweighted/moirai_small_stu_sandwich_20260127_223958/logs/version_0",
    "Multi-Head STU (13.83M)": "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_multihead_stu/lotsa_v1_unweighted/moirai_small_multihead_stu_20260207_174302/logs/version_0",
    "Non-Approx STU (13.83M)": "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_nonapprox_stu/lotsa_v1_unweighted/moirai_small_nonapprox_stu_20260207_175645/logs/version_0",
    "Parallel STU+Attn (13.83M)": "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_parallel_stu/lotsa_v1_unweighted/moirai_small_parallel_stu_20260207_180153/logs/version_0",
}

LOSS_TAG = "train/PackedNLLLoss_epoch"

def extract_loss(log_dir):
    """Extract epoch-level training loss from TensorBoard events."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])

    # Try different tag names
    tag = None
    for candidate in [LOSS_TAG, "train/PackedNLLLoss", "train/PackedQuantileMAELoss_epoch"]:
        if candidate in tags:
            tag = candidate
            break

    if tag is None:
        print(f"  Available tags: {tags}")
        return None, None

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)


# Extract all data
data = {}
for name, log_dir in RUNS.items():
    if not os.path.exists(log_dir):
        print(f"SKIP {name}: dir not found")
        continue
    print(f"Loading {name}...")
    steps, values = extract_loss(log_dir)
    if steps is not None and len(steps) > 0:
        data[name] = (steps, values)
        print(f"  {len(steps)} epochs, latest: epoch {steps[-1]}, loss {values[-1]:.4f}")
    else:
        print(f"  No loss data found")

if not data:
    print("No data to plot!")
    sys.exit(1)

# Colors
colors = {
    "Baseline (13.83M)": "#333333",
    "Approx STU (12.53M)": "#1f77b4",
    "Sandwich (15.65M)": "#ff7f0e",
    "Multi-Head STU (13.83M)": "#2ca02c",
    "Non-Approx STU (13.83M)": "#d62728",
    "Parallel STU+Attn (13.83M)": "#9467bd",
}
linestyles = {
    "Baseline (13.83M)": "-",
    "Approx STU (12.53M)": "--",
    "Sandwich (15.65M)": "--",
    "Multi-Head STU (13.83M)": "-",
    "Non-Approx STU (13.83M)": "-",
    "Parallel STU+Attn (13.83M)": "-",
}

# ---- Plot 1: Full training loss curves ----
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
for name, (steps, values) in data.items():
    ax.plot(steps, values, label=name, color=colors.get(name, None),
            linestyle=linestyles.get(name, '-'), linewidth=1.5, alpha=0.85)

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Training Loss (PackedNLLLoss)", fontsize=13)
ax.set_title("Training Loss Comparison: STU Variants vs Baseline", fontsize=14)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig("/scratch/gpfs/EHAZAN/jh1161/training_loss_all.png", dpi=150)
print(f"\nSaved: training_loss_all.png")

# ---- Plot 2: Zoomed to common epoch range (new variants) ----
new_variants = [k for k in data if "Multi-Head" in k or "Non-Approx" in k or "Parallel" in k or "Baseline" in k]
if new_variants:
    min_max_epoch = min(data[k][0][-1] for k in new_variants)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for name in new_variants:
        steps, values = data[name]
        mask = steps <= min_max_epoch
        ax.plot(steps[mask], values[mask], label=name, color=colors.get(name, None),
                linestyle=linestyles.get(name, '-'), linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Training Loss (PackedNLLLoss)", fontsize=13)
    ax.set_title(f"Training Loss: New Variants vs Baseline (epochs 0-{int(min_max_epoch)})", fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig("/scratch/gpfs/EHAZAN/jh1161/training_loss_new_variants.png", dpi=150)
    print(f"Saved: training_loss_new_variants.png")

# ---- Plot 3: Last 100 epochs zoomed (if enough data) ----
new_with_data = [k for k in new_variants if k in data and len(data[k][0]) > 100]
if new_with_data:
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for name in new_with_data:
        steps, values = data[name]
        # Last 100 epochs
        mask = steps >= (steps[-1] - 100)
        ax.plot(steps[mask], values[mask], label=name, color=colors.get(name, None),
                linestyle=linestyles.get(name, '-'), linewidth=1.8, alpha=0.85)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Training Loss (PackedNLLLoss)", fontsize=13)
    ax.set_title("Training Loss: Last 100 Epochs (Zoomed)", fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/scratch/gpfs/EHAZAN/jh1161/training_loss_zoomed.png", dpi=150)
    print(f"Saved: training_loss_zoomed.png")

# ---- Summary table ----
print("\n" + "=" * 70)
print("TRAINING PROGRESS SUMMARY")
print("=" * 70)
print(f"{'Model':<32} {'Epochs':>7} {'Latest Loss':>12} {'Min Loss':>12}")
print("-" * 70)
for name, (steps, values) in sorted(data.items(), key=lambda x: x[1][1][-1]):
    print(f"{name:<32} {int(steps[-1]):>7} {values[-1]:>12.4f} {values.min():>12.4f}")
