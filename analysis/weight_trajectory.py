"""
Weight trajectory analysis across training checkpoints.

Tracks how model weights evolve during training for both baseline and ms46,
particularly focusing on the hint channel input projection weights.
This helps explain the training curve dynamics (why ms46 degrades after 10K steps
without dropout, but is stable with dropout).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

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

OUTPUTS = Path("/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_unweighted")

# Baseline checkpoints
BL_DIR = OUTPUTS / "m2_baseline_20260209_114203" / "checkpoints"
# ms46 10K checkpoints (has checkpoints at 1K-10K steps)
MS46_DIR = OUTPUTS / "q_ms_d4d6_20260223_134653" / "checkpoints"
# hd10 100K checkpoints (hint d=4 + 10% dropout)
HD10_DIR = OUTPUTS / "m2_hd10_100k_20260223_112829" / "checkpoints"

def load_weight_stats(ckpt_path, prefix="module.model."):
    """Extract key weight statistics from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    stats = {}
    for name, param in state_dict.items():
        if not name.startswith(prefix):
            continue
        short_name = name[len(prefix):]

        # Focus on key components
        p = param.float()
        stats[short_name] = {
            'norm': p.norm().item(),
            'mean': p.mean().item(),
            'std': p.std().item(),
            'max': p.abs().max().item(),
        }

    return stats

def extract_layer_norms(stats, pattern):
    """Extract norms for layers matching a pattern."""
    norms = {}
    for name, s in stats.items():
        if pattern in name:
            norms[name] = s['norm']
    return norms

def analyze_input_projection(ckpt_path):
    """Analyze the input projection weights (where hints enter the model)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    result = {}
    for name, param in state_dict.items():
        if 'in_proj' in name and 'weight' in name:
            p = param.float()
            result[name] = {
                'norm': p.norm().item(),
                'shape': list(p.shape),
                # If input dim > 1 (hint channels present), analyze channel split
                'channel_norms': [],
            }
            if p.shape[-1] > 1:
                for ch in range(p.shape[-1]):
                    result[name]['channel_norms'].append(p[..., ch].norm().item())

    return result

def main():
    figdir = Path("/scratch/gpfs/EHAZAN/jh1161/analysis/figures")
    figdir.mkdir(exist_ok=True)

    # Collect checkpoints at various steps
    steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # --- Analysis 1: QKV weight norm trajectory ---
    print("Analyzing QKV weight norm trajectory...")

    bl_qkv_norms = {s: [] for s in steps}
    ms46_qkv_norms = {s: [] for s in steps}

    for step in steps:
        epoch = step // 100 - 1
        ckpt_name = f"epoch_{epoch}-step_{step}.ckpt"

        # Baseline
        bl_ckpt = BL_DIR / ckpt_name
        if bl_ckpt.exists():
            stats = load_weight_stats(bl_ckpt)
            for layer in range(6):
                for comp in ['in_proj_weight']:
                    key = f"encoder.layers.{layer}.self_attn.{comp}"
                    if key in stats:
                        bl_qkv_norms[step].append(stats[key]['norm'])

        # MS46
        ms46_ckpt = MS46_DIR / ckpt_name
        if ms46_ckpt.exists():
            stats = load_weight_stats(ms46_ckpt)
            for layer in range(6):
                for comp in ['in_proj_weight']:
                    key = f"encoder.layers.{layer}.self_attn.{comp}"
                    if key in stats:
                        ms46_qkv_norms[step].append(stats[key]['norm'])

    # Plot QKV norm evolution
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    ax = axes[0]
    valid_steps = [s for s in steps if bl_qkv_norms[s] and ms46_qkv_norms[s]]
    if valid_steps:
        bl_total = [sum(bl_qkv_norms[s]) for s in valid_steps]
        ms_total = [sum(ms46_qkv_norms[s]) for s in valid_steps]
        ax.plot([s/1000 for s in valid_steps], bl_total, 'o-', color='#1f77b4', label='Baseline', markersize=4)
        ax.plot([s/1000 for s in valid_steps], ms_total, 's-', color='#ff7f0e', label='MS46 (hint)', markersize=4)
        ax.set_xlabel('Training steps (K)')
        ax.set_ylabel('Total QKV weight norm')
        ax.set_title('(a) QKV weight norm evolution')
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No matching checkpoints found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(a) QKV weight norm evolution')

    # Per-layer difference at final step
    ax = axes[1]
    final_step = max(valid_steps) if valid_steps else 10000
    if bl_qkv_norms.get(final_step) and ms46_qkv_norms.get(final_step):
        n_layers = min(len(bl_qkv_norms[final_step]), len(ms46_qkv_norms[final_step]))
        bl_norms = bl_qkv_norms[final_step][:n_layers]
        ms_norms = ms46_qkv_norms[final_step][:n_layers]
        pct_diff = [(m - b) / b * 100 for b, m in zip(bl_norms, ms_norms)]

        colors = ['#2ca02c' if d > 0 else '#d62728' for d in pct_diff]
        ax.bar(range(n_layers), pct_diff, color=colors, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Transformer layer')
        ax.set_ylabel('QKV norm change (%)')
        ax.set_title(f'(b) Per-layer QKV norm change at {final_step//1000}K')
        ax.set_xticks(range(n_layers))
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Missing checkpoint data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(b) Per-layer QKV norm change')

    plt.tight_layout()
    plt.savefig(figdir / 'weight_trajectory.pdf')
    plt.savefig(figdir / 'weight_trajectory.png')
    print(f"Saved weight_trajectory.pdf/png")

    # --- Analysis 2: Input projection channel analysis ---
    print("\nAnalyzing input projection channels...")

    ms46_10k = MS46_DIR / "epoch_99-step_10000.ckpt"
    if ms46_10k.exists():
        proj_stats = analyze_input_projection(ms46_10k)
        print("Input projection analysis (ms46 @ 10K):")
        for name, info in proj_stats.items():
            print(f"  {name}: shape={info['shape']}, norm={info['norm']:.4f}")
            if info['channel_norms']:
                for i, cn in enumerate(info['channel_norms']):
                    print(f"    Channel {i}: norm={cn:.4f}")

    # --- Analysis 3: Compare hint model vs baseline weight distributions ---
    print("\nWeight distribution comparison...")

    # Check which checkpoints exist for all three models
    hd10_steps = [10000, 20000, 30000, 50000, 70000, 100000]
    hd10_data = {}

    for step in hd10_steps:
        epoch = step // 100 - 1
        ckpt_name = f"epoch_{epoch}-step_{step}.ckpt"
        hd10_ckpt = HD10_DIR / ckpt_name
        if hd10_ckpt.exists():
            stats = load_weight_stats(hd10_ckpt)
            total_norm = sum(s['norm'] for s in stats.values())
            hd10_data[step] = total_norm
            print(f"  hd10 @ {step}: total norm = {total_norm:.2f}")

    if hd10_data:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        hd_steps = sorted(hd10_data.keys())
        hd_norms = [hd10_data[s] for s in hd_steps]
        ax.plot([s/1000 for s in hd_steps], hd_norms, 'o-', color='#2ca02c', markersize=5, label='hd10 (hint+dropout)')

        # Add baseline points if available
        bl_norms_long = {}
        for step in [10000, 20000, 50000, 100000]:
            epoch = step // 100 - 1
            bl_ckpt = BL_DIR / f"epoch_{epoch}-step_{step}.ckpt"
            if bl_ckpt.exists():
                stats = load_weight_stats(bl_ckpt)
                bl_norms_long[step] = sum(s['norm'] for s in stats.values())

        if bl_norms_long:
            bl_s = sorted(bl_norms_long.keys())
            bl_n = [bl_norms_long[s] for s in bl_s]
            ax.plot([s/1000 for s in bl_s], bl_n, 's-', color='#1f77b4', markersize=5, label='Baseline')

        ax.set_xlabel('Training steps (K)')
        ax.set_ylabel('Total model weight norm')
        ax.set_title('Weight norm evolution during long training')
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figdir / 'weight_norm_evolution.pdf')
        plt.savefig(figdir / 'weight_norm_evolution.png')
        print("Saved weight_norm_evolution.pdf/png")

    print("\nWeight trajectory analysis complete.")

if __name__ == '__main__':
    main()
