"""
Analyze how the model's input projection treats hint channels vs target channels.

The in_proj maps per-patch inputs (target + hint channels) to the model's hidden dim.
This analysis shows the model has learned to differentially weight the hint channels.
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
FIGDIR = Path("/scratch/gpfs/EHAZAN/jh1161/analysis/figures")
FIGDIR.mkdir(exist_ok=True)

def analyze_checkpoint(ckpt_path):
    """Extract input projection weights and analyze channel structure."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    results = {}
    for name, param in state_dict.items():
        if 'in_proj' in name and 'weight' in name:
            p = param.float()
            results[name] = p
    return results

def main():
    # Load ms46 (2 hint channels) at 10K
    ms46_ckpt = OUTPUTS / "q_ms_d4d6_20260223_134653/checkpoints/epoch_99-step_10000.ckpt"
    # Load baseline at 10K
    bl_ckpt = OUTPUTS / "q_baseline_20260219_095606/checkpoints/epoch_99-step_10000.ckpt"

    print("Loading checkpoints...")
    ms46_weights = analyze_checkpoint(ms46_ckpt)
    bl_weights = analyze_checkpoint(bl_ckpt)

    # The hidden_layer weight has shape [384, N_input]
    # For ms46: N_input = 64 (patch_size=16, 4 channels: target + 2 hints + ?)
    # For baseline: N_input should be smaller

    print("\nMS46 weight shapes:")
    for name, w in ms46_weights.items():
        print(f"  {name}: {w.shape}")

    print("\nBaseline weight shapes:")
    for name, w in bl_weights.items():
        print(f"  {name}: {w.shape}")

    # Focus on the hidden_layer (first layer of in_proj)
    ms46_hidden = None
    bl_hidden = None
    ms46_residual = None
    bl_residual = None

    for name, w in ms46_weights.items():
        if 'hidden_layer' in name:
            ms46_hidden = w
        if 'residual_layer' in name:
            ms46_residual = w

    for name, w in bl_weights.items():
        if 'hidden_layer' in name:
            bl_hidden = w
        if 'residual_layer' in name:
            bl_residual = w

    if ms46_hidden is None or bl_hidden is None:
        print("Could not find hidden_layer weights!")
        return

    patch_size = 16
    n_input_ms46 = ms46_hidden.shape[1]
    n_input_bl = bl_hidden.shape[1]
    n_channels_ms46 = n_input_ms46 // patch_size
    n_channels_bl = n_input_bl // patch_size

    print(f"\nMS46: input_dim={n_input_ms46}, channels={n_channels_ms46}")
    print(f"Baseline: input_dim={n_input_bl}, channels={n_channels_bl}")

    # Compute per-channel weight norms
    # Group by channel: each channel has patch_size consecutive entries
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # Panel A: Per-input-dim weight norm comparison
    ax = axes[0]

    # MS46 hidden layer per-input-dim norms
    ms46_input_norms = ms46_hidden.norm(dim=0).numpy()  # norm across output dim for each input
    bl_input_norms = bl_hidden.norm(dim=0).numpy()

    # Group by channel
    ms46_channel_norms = []
    for ch in range(n_channels_ms46):
        start = ch * patch_size
        end = start + patch_size
        ch_norm = ms46_input_norms[start:end].mean()
        ms46_channel_norms.append(ch_norm)

    bl_channel_norms = []
    for ch in range(n_channels_bl):
        start = ch * patch_size
        end = start + patch_size
        ch_norm = bl_input_norms[start:end].mean()
        bl_channel_norms.append(ch_norm)

    # Labels for ms46 channels
    if n_channels_ms46 == 4:
        labels = ['Target', 'Hint d=4', 'Hint d=6', 'Ch 3']
    elif n_channels_ms46 == 3:
        labels = ['Target', 'Hint d=4', 'Hint d=6']
    else:
        labels = [f'Ch {i}' for i in range(n_channels_ms46)]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:n_channels_ms46]
    x = np.arange(n_channels_ms46)
    ax.bar(x, ms46_channel_norms, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Mean weight norm')
    ax.set_title('(a) MS46 input projection norms')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Per-position within patch
    ax = axes[1]
    for ch in range(min(n_channels_ms46, 3)):
        start = ch * patch_size
        end = start + patch_size
        norms = ms46_input_norms[start:end]
        ax.plot(range(patch_size), norms, 'o-', color=colors[ch], markersize=3,
                label=labels[ch], linewidth=1.5)

    # Also plot baseline target for comparison
    bl_target_norms = bl_input_norms[:patch_size]
    ax.plot(range(patch_size), bl_target_norms, 's--', color='gray', markersize=3,
            label='BL target', linewidth=1, alpha=0.7)

    ax.set_xlabel('Position within patch')
    ax.set_ylabel('Weight norm')
    ax.set_title('(b) Per-position input importance')
    ax.legend(framealpha=0.8, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel C: Residual layer comparison
    ax = axes[2]
    if ms46_residual is not None and bl_residual is not None:
        ms46_res_norms = ms46_residual.norm(dim=0).numpy()
        bl_res_norms = bl_residual.norm(dim=0).numpy()

        ms46_res_ch_norms = []
        for ch in range(n_channels_ms46):
            start = ch * patch_size
            end = start + patch_size
            if end <= len(ms46_res_norms):
                ms46_res_ch_norms.append(ms46_res_norms[start:end].mean())

        bl_res_ch_norms = []
        for ch in range(n_channels_bl):
            start = ch * patch_size
            end = start + patch_size
            if end <= len(bl_res_norms):
                bl_res_ch_norms.append(bl_res_norms[start:end].mean())

        x_ms = np.arange(len(ms46_res_ch_norms))
        ax.bar(x_ms, ms46_res_ch_norms, color=colors[:len(ms46_res_ch_norms)],
               edgecolor='black', linewidth=0.5)
        if bl_res_ch_norms:
            ax.axhline(y=bl_res_ch_norms[0], color='gray', linestyle='--',
                      linewidth=1, label=f'BL target: {bl_res_ch_norms[0]:.3f}')
        ax.set_xticks(x_ms)
        ax.set_xticklabels(labels[:len(ms46_res_ch_norms)], rotation=15, ha='right')
        ax.set_ylabel('Mean weight norm')
        ax.set_title('(c) Residual connection norms')
        ax.legend(framealpha=0.8, fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGDIR / 'input_projection_analysis.pdf')
    plt.savefig(FIGDIR / 'input_projection_analysis.png')
    print("Saved input_projection_analysis.pdf/png")

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"MS46 target channel avg norm: {ms46_channel_norms[0]:.4f}")
    if len(ms46_channel_norms) > 1:
        print(f"MS46 hint d=4 avg norm: {ms46_channel_norms[1]:.4f} ({ms46_channel_norms[1]/ms46_channel_norms[0]*100:.1f}% of target)")
    if len(ms46_channel_norms) > 2:
        print(f"MS46 hint d=6 avg norm: {ms46_channel_norms[2]:.4f} ({ms46_channel_norms[2]/ms46_channel_norms[0]*100:.1f}% of target)")
    if bl_channel_norms:
        print(f"Baseline target avg norm: {bl_channel_norms[0]:.4f}")

if __name__ == '__main__':
    main()
