"""Plot separate loss terms for EXP-1 degree experiments."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted"

runs = {
    "d1": "m2_precond_d1_cheb_20260208_194538",
    "d2": "m2_precond_d2_cheb_20260208_195702",
    "d3": "m2_precond_d3_cheb_20260208_195919",
    "d4": "m2_precond_d4_cheb_20260208_200423",
    "d5": "m2_precond_d5_cheb_20260208_200423",
    "d6": "m2_precond_d6_cheb_20260208_200423",
    "d7": "m2_precond_d7_cheb_20260208_211558",
}

metrics = {
    "quantile_loss": "train/quantile_loss_step",
    "fir_inverse_loss": "train/time_precondition_inverse_loss_step",
    "total_loss": "train/total_loss_step",
    "rejection_rate": "train/rejection_rate_step",
}


def load_scalar(logdir, tag):
    ea = EventAccumulator(logdir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    return steps, vals


def smooth(vals, window=100):
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode="valid")


data = {}
for label, run_name in runs.items():
    logdir = os.path.join(BASE, run_name, "logs", "version_0")
    if not os.path.isdir(logdir):
        continue
    data[label] = {}
    for name, tag in metrics.items():
        steps, vals = load_scalar(logdir, tag)
        if steps is not None:
            data[label][name] = (steps, vals)

colors = {
    "d1": "#1f77b4",
    "d2": "#ff7f0e",
    "d3": "#2ca02c",
    "d4": "#d62728",
    "d5": "#9467bd",
    "d6": "#8c564b",
    "d7": "#e377c2",
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("EXP-1: Loss Terms by Chebyshev Degree d1-d7 (100K steps)", fontsize=14, fontweight="bold")
w = 100

# 1) Quantile loss — clipped y-axis
ax = axes[0, 0]
ax.set_title("Quantile Loss (preconditioned space)")
for label in runs:
    if label in data and "quantile_loss" in data[label]:
        steps, vals = data[label]["quantile_loss"]
        s = smooth(vals, w)
        ax.plot(steps[w-1:len(s)+w-1], s, label=label, color=colors[label], alpha=0.8, linewidth=1.2)
ax.set_xlabel("Step")
ax.set_ylabel("Quantile MAE")
ax.set_ylim(0, 0.35)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2) FIR inverse loss
ax = axes[0, 1]
ax.set_title("FIR Inverse Loss\n(learnable 64-tap inverse filter MSE)\nd1=0 (identity, no residual to learn)")
for label in runs:
    if label == "baseline":
        continue
    if label in data and "fir_inverse_loss" in data[label]:
        steps, vals = data[label]["fir_inverse_loss"]
        s = smooth(vals, w)
        ax.plot(steps[w-1:len(s)+w-1], s, label=label, color=colors[label], alpha=0.8, linewidth=1.2)
ax.set_xlabel("Step")
ax.set_ylabel("FIR Inverse MSE")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3) Total loss — clipped
ax = axes[1, 0]
ax.set_title("Total Loss (quantile + 0.1 * FIR inverse)")
for label in runs:
    if label in data and "total_loss" in data[label]:
        steps, vals = data[label]["total_loss"]
        s = smooth(vals, w)
        ax.plot(steps[w-1:len(s)+w-1], s, label=label, color=colors[label], alpha=0.8, linewidth=1.2)
ax.set_xlabel("Step")
ax.set_ylabel("Total Loss")
ax.set_ylim(0, 0.45)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4) Rejection rate
ax = axes[1, 1]
ax.set_title("Anomaly Rejection Rate (z-score threshold=8.0)")
for label in runs:
    if label in data and "rejection_rate" in data[label]:
        steps, vals = data[label]["rejection_rate"]
        s = smooth(vals, w)
        ax.plot(steps[w-1:len(s)+w-1], s, label=label, color=colors[label], alpha=0.8, linewidth=1.2)
ax.set_xlabel("Step")
ax.set_ylabel("Rejection Rate")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/scratch/gpfs/EHAZAN/jh1161/exp1_loss_terms.png", dpi=150, bbox_inches="tight")
print("Saved to /scratch/gpfs/EHAZAN/jh1161/exp1_loss_terms.png")

# Also print final smoothed values for reference
print("\n=== Final smoothed values (last 100-step window) ===")
for label in runs:
    if label not in data:
        continue
    parts = []
    for name in ["quantile_loss", "fir_inverse_loss", "total_loss"]:
        if name in data[label]:
            _, vals = data[label][name]
            parts.append(f"{name}={vals[-200:].mean():.4f}")
    print(f"  {label}: {', '.join(parts)}")
