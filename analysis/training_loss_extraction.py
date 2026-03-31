#!/usr/bin/env python3
"""
Extract and compare training loss curves from TensorBoard event files.

Creates:
    - figures/loss_curves.pdf: Smoothed training loss comparison
    - figures/loss_convergence_rate.pdf: Relative convergence speed

Usage:
    python analysis/training_loss_extraction.py
"""

import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = "/scratch/gpfs/EHAZAN/jh1161/analysis/figures"
os.makedirs(FIG_DIR, exist_ok=True)

OUTPUTS = Path("/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted")

# Run name -> (label, color, linestyle)
RUNS = {
    "q_baseline_20260219_095606": ("Baseline", "#4C72B0", "-"),
    "q_ms_d4d6_20260223_134653": ("MS $d{=}4{+}d{=}6$", "#C44E52", "-"),
}


def load_tb_events(run_dir):
    """Load training loss from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed, trying tbparse...")
        return load_tb_events_tbparse(run_dir)

    log_dirs = list(run_dir.glob("logs/version_*/"))
    if not log_dirs:
        log_dirs = list(run_dir.glob("lightning_logs/version_*/"))
    if not log_dirs:
        return None, None

    log_dir = sorted(log_dirs)[-1]  # Latest version
    ea = EventAccumulator(str(log_dir))
    ea.Reload()

    # Find the training loss tag
    tags = ea.Tags().get('scalars', [])
    loss_tags = [t for t in tags if 'loss' in t.lower() and 'step' in t.lower()]
    if not loss_tags:
        loss_tags = [t for t in tags if 'loss' in t.lower()]
    if not loss_tags:
        return None, None

    loss_tag = loss_tags[0]
    events = ea.Scalars(loss_tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_tb_events_tbparse(run_dir):
    """Alternative loader using tbparse."""
    try:
        from tbparse import SummaryReader
    except ImportError:
        return load_tb_events_manual(run_dir)

    log_dirs = list(run_dir.glob("logs/version_*/"))
    if not log_dirs:
        return None, None

    log_dir = sorted(log_dirs)[-1]
    reader = SummaryReader(str(log_dir))
    df = reader.scalars

    loss_cols = df[df['tag'].str.contains('loss', case=False, na=False)]
    if loss_cols.empty:
        return None, None

    loss_tag = loss_cols['tag'].iloc[0]
    loss_data = df[df['tag'] == loss_tag]
    return loss_data['step'].values, loss_data['value'].values


def load_tb_events_manual(run_dir):
    """Manual event file parsing as last resort."""
    import struct

    log_dirs = list(run_dir.glob("logs/version_*/"))
    if not log_dirs:
        return None, None

    log_dir = sorted(log_dirs)[-1]
    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None, None

    # Use tensorboard protobuf parsing
    try:
        from tensorflow.core.util.event_pb2 import Event
        steps = []
        values = []
        for ef in event_files:
            with open(ef, 'rb') as f:
                while True:
                    header = f.read(8)
                    if len(header) < 8:
                        break
                    data_len = struct.unpack('<Q', header)[0]
                    _ = f.read(4)  # masked CRC of length
                    data = f.read(data_len)
                    _ = f.read(4)  # masked CRC of data

                    event = Event()
                    event.ParseFromString(data)
                    if event.HasField('summary'):
                        for v in event.summary.value:
                            if 'loss' in v.tag.lower():
                                steps.append(event.step)
                                values.append(v.simple_value)

        return np.array(steps), np.array(values)
    except ImportError:
        print("Cannot parse TensorBoard events without tensorboard or tensorflow")
        return None, None


def smooth(values, window=50):
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    alpha = 2 / (window + 1)
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def main():
    print("Extracting training loss curves...")

    all_data = {}
    for run_name, (label, color, ls) in RUNS.items():
        run_dir = OUTPUTS / run_name
        if not run_dir.exists():
            print(f"  Skip {run_name}: directory not found")
            continue

        steps, values = load_tb_events(run_dir)
        if steps is not None and len(steps) > 0:
            all_data[run_name] = (steps, values, label, color, ls)
            print(f"  {label}: {len(steps)} events, steps {steps[0]}-{steps[-1]}")
        else:
            print(f"  {label}: no events found")

    if not all_data:
        print("No training loss data found. Creating figure from SLURM logs instead...")
        create_loss_from_logs()
        return

    # Plot 1: Raw + smoothed loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for run_name, (steps, values, label, color, ls) in all_data.items():
        # Raw (light)
        ax1.plot(steps, values, color=color, alpha=0.1, linewidth=0.5)
        # Smoothed
        smoothed = smooth(values, window=100)
        ax1.plot(steps, smoothed, color=color, linewidth=2, linestyle=ls, label=label)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('(a) Training Loss Curves')
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(0, 10000)

    # Plot 2: Loss ratio / relative convergence
    keys = list(all_data.keys())
    if len(keys) >= 2:
        # Interpolate to common step grid
        max_step = min(all_data[keys[0]][0][-1], all_data[keys[1]][0][-1])
        common_steps = np.arange(0, min(max_step, 10000), 10)

        vals = {}
        for k in keys[:2]:
            s, v = all_data[k][0], smooth(all_data[k][1], window=100)
            vals[k] = np.interp(common_steps, s, v)

        ratio = vals[keys[1]] / vals[keys[0]]
        ax2.plot(common_steps, ratio, color='#2c3e50', linewidth=2)
        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8)
        ax2.fill_between(common_steps, ratio, 1.0, alpha=0.2,
                        color='#C44E52', where=ratio < 1.0)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss Ratio (Hint / Baseline)')
        ax2.set_title('(b) Relative Convergence')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlim(0, 10000)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'loss_curves.{ext}'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved loss_curves.pdf/png")


def create_loss_from_logs():
    """Create loss comparison from SLURM log output."""
    # Parse training loss from SLURM logs
    import re

    log_patterns = {
        "Baseline": "/scratch/gpfs/EHAZAN/jh1161/logs/",
        "MS d=4+d=6": "/scratch/gpfs/EHAZAN/jh1161/logs/",
    }

    # Find relevant log files
    log_dir = Path("/scratch/gpfs/EHAZAN/jh1161/logs")
    baseline_logs = sorted(log_dir.glob("*baseline*10k*.out"))
    ms46_logs = sorted(log_dir.glob("*ms*d4d6*.out")) + sorted(log_dir.glob("*ms46*.out"))

    all_series = {}

    for label, logs in [("Baseline", baseline_logs), ("MS46", ms46_logs)]:
        if not logs:
            continue
        log_file = logs[-1]  # Most recent
        steps_vals = []
        with open(log_file) as f:
            for line in f:
                m = re.search(r'Epoch\s+(\d+).*?PackedQuantileMAELoss_step=([0-9.]+)', line)
                if m:
                    epoch = int(m.group(1))
                    loss = float(m.group(2))
                    steps_vals.append((epoch, loss))

        if steps_vals:
            epochs, losses = zip(*steps_vals)
            all_series[label] = (np.array(epochs), np.array(losses))
            print(f"  {label}: {len(epochs)} loss values from {log_file.name}")

    if all_series:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {"Baseline": "#4C72B0", "MS46": "#C44E52"}
        for label, (epochs, losses) in all_series.items():
            smoothed = smooth(losses, window=50)
            ax.plot(epochs, smoothed, color=colors.get(label, 'gray'), linewidth=2, label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Comparison')
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(FIG_DIR, f'loss_curves.{ext}'), bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved loss_curves.pdf/png")
    else:
        print("No loss data found in logs either.")


if __name__ == "__main__":
    main()
