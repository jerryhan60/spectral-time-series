#!/usr/bin/env python3
"""
Plot and compare training curves from multiple runs.
Extracts data from TensorBoard event files and creates matplotlib plots.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Error: tensorboard package not found")
    print("Install with: pip install tensorboard")
    sys.exit(1)


def load_tensorboard_data(log_dir, tag="train/PackedNLLLoss"):
    """Load scalar data from TensorBoard logs."""
    print(f"Loading: {log_dir}")

    # Find event file
    event_files = list(Path(log_dir).glob("events.out.tfevents*"))
    if not event_files:
        print(f"  Warning: No event files found in {log_dir}")
        return None, None

    event_file = str(event_files[0])
    print(f"  Reading: {event_file}")

    # Load events
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # Check available tags
    available_tags = ea.Tags()
    print(f"  Available scalar tags: {available_tags.get('scalars', [])[:5]}...")

    if tag not in available_tags.get('scalars', []):
        print(f"  Warning: Tag '{tag}' not found!")
        return None, None

    # Extract data
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    print(f"  Loaded {len(steps)} data points")
    return np.array(steps), np.array(values)


def plot_comparison(runs, output_path="training_comparison.png", figsize=(12, 6)):
    """
    Plot training curves for multiple runs.

    Args:
        runs: dict mapping run_name -> (log_dir, color, linestyle)
        output_path: where to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Metrics to plot
    metrics = [
        ("train/PackedNLLLoss", "Training Loss"),
        ("val/PackedNLLLoss", "Validation Loss"),
    ]

    for ax, (tag, title) in zip(axes, metrics):
        for run_name, (log_dir, color, linestyle) in runs.items():
            steps, values = load_tensorboard_data(log_dir, tag)

            if steps is not None and len(steps) > 0:
                ax.plot(steps, values,
                       label=run_name,
                       color=color,
                       linestyle=linestyle,
                       linewidth=2,
                       alpha=0.8)

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Loss (NLL)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    # Also try to display if in interactive mode
    try:
        plt.show()
    except:
        pass


def plot_detailed_comparison(runs, output_dir="training_plots"):
    """Create detailed comparison plots."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Training loss only (large)
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name, (log_dir, color, linestyle) in runs.items():
        steps, values = load_tensorboard_data(log_dir, "train/PackedNLLLoss")
        if steps is not None and len(steps) > 0:
            ax.plot(steps, values,
                   label=run_name,
                   color=color,
                   linestyle=linestyle,
                   linewidth=2.5)

    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Training Loss (NLL)", fontsize=14)
    ax.set_title("Training Loss Comparison", fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_loss.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/training_loss.png")
    plt.close()

    # 2. Validation loss only
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name, (log_dir, color, linestyle) in runs.items():
        steps, values = load_tensorboard_data(log_dir, "val/PackedNLLLoss")
        if steps is not None and len(steps) > 0:
            ax.plot(steps, values,
                   label=run_name,
                   color=color,
                   linestyle=linestyle,
                   linewidth=2.5,
                   marker='o',
                   markersize=4,
                   markevery=max(1, len(steps)//20))  # Show ~20 markers

    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Validation Loss (NLL)", fontsize=14)
    ax.set_title("Validation Loss Comparison", fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/validation_loss.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/validation_loss.png")
    plt.close()

    # 3. Final values comparison (bar chart)
    fig, ax = plt.subplots(figsize=(8, 6))

    final_train = []
    final_val = []
    labels = []

    for run_name, (log_dir, color, linestyle) in runs.items():
        steps_train, values_train = load_tensorboard_data(log_dir, "train/PackedNLLLoss")
        steps_val, values_val = load_tensorboard_data(log_dir, "val/PackedNLLLoss")

        if steps_train is not None and len(steps_train) > 0:
            final_train.append(values_train[-1])
            final_val.append(values_val[-1] if values_val is not None else np.nan)
            labels.append(run_name)

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, final_train, width, label='Training', alpha=0.8)
    ax.bar(x + width/2, final_val, width, label='Validation', alpha=0.8)

    ax.set_ylabel('Final Loss (NLL)', fontsize=14)
    ax.set_title('Final Loss Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (train, val) in enumerate(zip(final_train, final_val)):
        ax.text(i - width/2, train, f'{train:.3f}',
               ha='center', va='bottom', fontsize=10)
        if not np.isnan(val):
            ax.text(i + width/2, val, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_loss_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/final_loss_comparison.png")
    plt.close()


def main():
    print("=" * 60)
    print("Training Comparison Plotter")
    print("=" * 60)

    # Define runs to compare
    base_path = Path("/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain")

    runs = {
        "Non-preconditioned": (
            str(base_path / "moirai_small/lotsa_v1_unweighted/pretrain_run_20251020_205126/logs/version_0"),
            "blue",
            "-"
        ),
        "Preconditioned": (
            str(base_path / "moirai_small_precond/lotsa_v1_unweighted/precond_default_20251102_102511/logs/version_0"),
            "red",
            "--"
        ),
    }

    print("\nRuns to compare:")
    for name, (log_dir, color, style) in runs.items():
        print(f"  - {name}: {Path(log_dir).parent.parent.name}")

    print("\n" + "=" * 60)
    print("Creating plots...")
    print("=" * 60 + "\n")

    # Create basic comparison
    plot_comparison(runs, output_path="training_comparison.png")

    # Create detailed plots
    plot_detailed_comparison(runs, output_dir="training_plots")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Print final values
    print("\nFinal loss values:")
    for run_name, (log_dir, _, _) in runs.items():
        steps_train, values_train = load_tensorboard_data(log_dir, "train/PackedNLLLoss")
        steps_val, values_val = load_tensorboard_data(log_dir, "val/PackedNLLLoss")

        if steps_train is not None and len(steps_train) > 0:
            final_train = values_train[-1]
            final_val = values_val[-1] if values_val is not None else "N/A"
            print(f"\n{run_name}:")
            print(f"  Training:   {final_train:.4f} (at step {steps_train[-1]})")
            if values_val is not None:
                print(f"  Validation: {final_val:.4f} (at step {steps_val[-1]})")

            # Improvement
            if run_name != "Non-preconditioned" and values_train is not None:
                _, base_train = load_tensorboard_data(
                    runs["Non-preconditioned"][0],
                    "train/PackedNLLLoss"
                )
                if base_train is not None:
                    improvement = base_train[-1] - final_train
                    pct = (improvement / base_train[-1]) * 100
                    print(f"  Improvement: {improvement:.4f} ({pct:.2f}%)")

    print("\n✓ All plots created successfully!")
    print(f"  - training_comparison.png (overview)")
    print(f"  - training_plots/training_loss.png")
    print(f"  - training_plots/validation_loss.png")
    print(f"  - training_plots/final_loss_comparison.png")


if __name__ == "__main__":
    main()
