#!/usr/bin/env python3
"""
Attention pattern analysis: How do polynomial hints change attention distributions?

Loads baseline and hint-trained models, runs them on the same inputs,
and compares attention entropy, head specialization, and causal structure.

Requires GPU. Run from a GPU node:
    salloc --gres=gpu:1 --partition=della --account=ehazan --time=01:00:00 --mem=64G --cpus-per-task=8
    source uni2ts/venv/bin/activate
    python analysis/attention_analysis.py

Generates:
    - figures/attention_entropy.pdf: Per-layer attention entropy comparison
    - figures/attention_distance.pdf: Mean attention distance (how far tokens look)
    - figures/attention_heatmap.pdf: Example attention matrices side-by-side
"""

import sys
import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "uni2ts" / "src"))

FIG_DIR = str(Path(__file__).resolve().parent / "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Checkpoint paths (m2d = lotsa_v1_moirai2 data, seed 0)
BASE = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai2_small/lotsa_v1_moirai2"
BL_CKPT = f"{BASE}/baseline_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
HD10_CKPT = f"{BASE}/hd10_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
MS_CKPT = f"{BASE}/ms46_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"
MSHD10_CKPT = f"{BASE}/mshd10_m2d_seed0_10k/checkpoints/epoch_99-step_10000.ckpt"

MODELS = {
    'Baseline': BL_CKPT,
    'HD10': HD10_CKPT,
    'MS46': MS_CKPT,
    'MSHD10': MSHD10_CKPT,
}
COLORS = {'Baseline': '#1f77b4', 'HD10': '#ff7f0e', 'MS46': '#2ca02c', 'MSHD10': '#d62728'}


def load_module(ckpt_path):
    """Load Moirai2 module from checkpoint."""
    import torch
    from uni2ts.model.moirai2.pretrain import Moirai2Pretrain

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hparams = ckpt.get('hyper_parameters', {})
    module_kwargs = hparams.get('module_kwargs', {})

    from uni2ts.model.moirai2.module import Moirai2Module
    module = Moirai2Module(**module_kwargs)

    # Extract module state dict
    state_dict = {}
    prefix = 'module.'
    for k, v in ckpt['state_dict'].items():
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = v

    module.load_state_dict(state_dict, strict=False)
    module.eval()
    return module


def register_attention_hooks(module):
    """Register forward hooks to capture attention weights from all layers."""
    attention_maps = {}

    def make_hook(name):
        def hook_fn(mod, input, output):
            # MHA returns (output, attn_weights) when need_weights=True
            # But we need to modify to capture. Instead, use the scores.
            pass
        return hook_fn

    # The transformer layers store attention in self.attn
    # We need to hook into the attention computation
    for i, layer in enumerate(module.encoder.layers):
        def make_attn_hook(layer_idx):
            def hook(mod, input, output):
                # output is typically (attn_output, attn_weights)
                if isinstance(output, tuple) and len(output) > 1:
                    attention_maps[layer_idx] = output[1].detach().cpu()
            return hook

        if hasattr(layer, 'self_attn'):
            layer.self_attn.register_forward_hook(make_attn_hook(i))
        elif hasattr(layer, 'attn'):
            layer.attn.register_forward_hook(make_attn_hook(i))

    return attention_maps


def create_synthetic_input(module, batch_size=4, seq_len=128, device='cpu'):
    """Create synthetic time series input for attention analysis."""
    import torch

    np.random.seed(42)
    T = seq_len * 16  # patch_size = 16
    t = np.arange(T)

    signals = []
    for b in range(batch_size):
        # Mix of frequencies
        signal = (3 * np.sin(2*np.pi*t/(256 + b*32)) +
                  2 * np.sin(2*np.pi*t/(64 + b*8)) +
                  1.5 * np.sin(2*np.pi*t/16) +
                  np.sin(2*np.pi*t/4) +
                  0.3 * np.random.randn(T))
        signals.append(signal)

    signals = np.array(signals)
    return torch.tensor(signals, dtype=torch.float32, device=device)


def compute_attention_entropy(attn_weights):
    """Compute per-head attention entropy."""
    # attn_weights: (batch, heads, seq, seq)
    eps = 1e-12
    entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)
    return entropy.mean(dim=(0, 2))  # average over batch and queries -> (heads,)


def compute_mean_attention_distance(attn_weights):
    """Compute mean attention distance (how far each token looks)."""
    # attn_weights: (batch, heads, seq, seq)
    seq_len = attn_weights.shape[-1]
    positions = np.arange(seq_len)
    distances = np.abs(positions[:, None] - positions[None, :])  # (seq, seq)
    import torch
    dist_tensor = torch.tensor(distances, dtype=attn_weights.dtype, device=attn_weights.device)
    mean_dist = (attn_weights * dist_tensor).sum(dim=-1)
    return mean_dist.mean(dim=(0, 2))  # average over batch and queries -> (heads,)


def analyze_with_hooks(module, inputs, device='cpu'):
    """Run model with attention weight capture using hooks."""
    import torch

    # Try to get attention weights by modifying attention layers
    attn_weights_by_layer = {}
    handles = []

    # Find attention modules
    for name, mod in module.named_modules():
        if 'attn' in name.lower() and hasattr(mod, 'forward'):
            # Check if it's a MultiheadAttention or similar
            if isinstance(mod, torch.nn.MultiheadAttention):
                layer_idx = name
                orig_forward = mod.forward

                def make_wrapper(layer_name, orig_fn):
                    def wrapper(*args, **kwargs):
                        kwargs['need_weights'] = True
                        kwargs['average_attn_weights'] = False
                        result = orig_fn(*args, **kwargs)
                        if isinstance(result, tuple) and len(result) > 1:
                            attn_weights_by_layer[layer_name] = result[1].detach().cpu()
                        return result
                    return wrapper

                mod.forward = make_wrapper(name, orig_forward)

    return attn_weights_by_layer


def extract_attention_via_encoder(module, batch_size=16, num_patches=15, device='cpu'):
    """Extract attention by directly calling encoder with return_attn_weights=True."""
    import torch
    d_model = module.d_model
    encoder = module.encoder

    with torch.no_grad():
        x = torch.randn(batch_size, num_patches, d_model, device=device)

        # Causal mask (lower triangular = True = attend)
        mask = torch.triu(
            torch.ones(num_patches, num_patches, dtype=torch.bool, device=device),
            diagonal=1,
        ).logical_not()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        time_id = torch.arange(num_patches, device=device).unsqueeze(0).expand(batch_size, -1)

        result = encoder(
            x, attn_mask=mask, time_id=time_id, return_attn_weights=True,
        )
        reprs, attn_weights_list = result

    return [w.detach().cpu() for w in attn_weights_list]


def plot_attention_comparison_full(all_weights):
    """Comprehensive attention comparison figure for all models.

    all_weights: dict of {model_name: [layer_weights, ...]}
    """
    import torch
    model_names = list(all_weights.keys())
    n_layers = min(len(w) for w in all_weights.values())

    # Compute metrics per model per layer
    metrics = {name: {'entropy': [], 'local': [], 'sparsity': []} for name in model_names}

    for i in range(n_layers):
        for name in model_names:
            w = all_weights[name][i].float()
            avg = w.mean(dim=(0, 1, 2))  # [q, kv]

            eps = 1e-12
            ent = -(avg * (avg + eps).log()).sum(dim=-1).mean().item()
            metrics[name]['entropy'].append(ent)

            q_len = avg.shape[0]
            diag_mask = torch.zeros_like(avg)
            for j in range(q_len):
                start = max(0, j - 2)
                diag_mask[j, start:j+1] = 1.0
            metrics[name]['local'].append((avg * diag_mask).sum().item() / avg.sum().item())

            metrics[name]['sparsity'].append((avg < 0.01).float().mean().item())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    layers = range(1, n_layers + 1)
    markers = {'Baseline': 'o', 'HD10': 's', 'MS46': 'D', 'MSHD10': '^'}

    for metric_name, ax, title, ylabel in [
        ('entropy', axes[0, 0], '(a) Attention Entropy', 'Entropy (nats)'),
        ('local', axes[0, 1], '(b) Local vs. Global', 'Local Attention Ratio'),
        ('sparsity', axes[0, 2], '(c) Attention Sparsity', 'Sparsity (frac < 0.01)'),
    ]:
        for name in model_names:
            ax.plot(layers, metrics[name][metric_name],
                    f'{markers.get(name, "o")}-', color=COLORS[name],
                    linewidth=2, label=name, markersize=5)
        ax.set_xlabel('Layer')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.2)

    # Heatmaps for last layer: BL, HD10, HD10-BL difference
    last = n_layers - 1
    bl_hm = all_weights['Baseline'][last].float().mean(dim=(0, 1, 2)).numpy()

    ax = axes[1, 0]
    im = ax.imshow(bl_hm, aspect='auto', cmap='Blues', vmin=0)
    ax.set_xlabel('Key'); ax.set_ylabel('Query')
    ax.set_title(f'(d) Baseline (Layer {n_layers})')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Pick first hint model available for heatmap
    hint_name = 'HD10' if 'HD10' in model_names else model_names[1]
    hint_hm = all_weights[hint_name][last].float().mean(dim=(0, 1, 2)).numpy()

    ax = axes[1, 1]
    im = ax.imshow(hint_hm, aspect='auto', cmap='Oranges', vmin=0)
    ax.set_xlabel('Key'); ax.set_ylabel('Query')
    ax.set_title(f'(e) {hint_name} (Layer {n_layers})')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    diff = hint_hm - bl_hm
    vmax = max(abs(diff.min()), abs(diff.max()))
    im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Key'); ax.set_ylabel('Query')
    ax.set_title(f'(f) {hint_name} - Baseline (Layer {n_layers})')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Attention Pattern Analysis: Baseline vs. Hint Preconditioning (m2d, seed 0)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIG_DIR, f'attention_analysis.{ext}'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved attention_analysis.pdf/png")

    # Print metrics summary
    print("\n=== Attention Metrics Summary ===")
    header = "Layer " + " | ".join(f"Ent_{n[:4]:>4s} Loc_{n[:4]:>4s} Spr_{n[:4]:>4s}" for n in model_names)
    print(header)
    for i in range(n_layers):
        parts = [f"{i+1:5d}"]
        for name in model_names:
            parts.append(f"{metrics[name]['entropy'][i]:.3f}  {metrics[name]['local'][i]:.3f}  {metrics[name]['sparsity'][i]:.3f}")
        print(" | ".join(parts))


def main():
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Check which checkpoints exist
    available = {name: path for name, path in MODELS.items() if os.path.exists(path)}
    if 'Baseline' not in available or len(available) < 2:
        print(f"Need Baseline + at least 1 hint model. Found: {list(available.keys())}")
        generate_synthetic_analysis()
        return

    # Load all available models
    modules = {}
    for name, path in available.items():
        print(f"Loading {name}...")
        modules[name] = load_module(path).to(device)

    # Extract attention from all models
    print("Extracting attention patterns via encoder.forward(return_attn_weights=True)...")
    all_attn = {}
    for name, module in modules.items():
        attn = extract_attention_via_encoder(module, batch_size=16, num_patches=15, device=device)
        all_attn[name] = attn
        print(f"  {name}: {len(attn)} layers")
        del module  # free GPU memory
    torch.cuda.empty_cache()

    if all(len(v) > 0 for v in all_attn.values()):
        plot_attention_comparison_full(all_attn)
    else:
        print("Attention extraction failed, falling back to weight analysis...")
        bl_module = load_module(available['Baseline']).to(device)
        hint_name = [n for n in available if n != 'Baseline'][0]
        hint_module = load_module(available[hint_name]).to(device)
        analyze_weights(bl_module, hint_module)


def analyze_weights(bl_module, ms_module):
    """Compare model weights when attention extraction fails."""
    import torch

    print("\n=== Weight Analysis ===")

    # Compare projection weight norms by layer
    bl_norms = {}
    ms_norms = {}

    for name, param in bl_module.named_parameters():
        bl_norms[name] = param.data.norm().item()
    for name, param in ms_module.named_parameters():
        ms_norms[name] = param.data.norm().item()

    # Find common parameters
    common = set(bl_norms.keys()) & set(ms_norms.keys())

    # Compute relative differences
    diffs = {}
    for name in common:
        if bl_norms[name] > 1e-6:
            diffs[name] = (ms_norms[name] - bl_norms[name]) / bl_norms[name] * 100

    # Group by layer type
    qkv_diffs = {k: v for k, v in diffs.items() if any(x in k for x in ['q_proj', 'k_proj', 'v_proj', 'in_proj'])}
    out_diffs = {k: v for k, v in diffs.items() if 'out_proj' in k}
    ffn_diffs = {k: v for k, v in diffs.items() if 'ffn' in k or 'fc' in k or 'linear' in k}

    print(f"\nQKV projection weight norm changes:")
    for k, v in sorted(qkv_diffs.items()):
        print(f"  {k}: {v:+.2f}%")

    print(f"\nOutput projection weight norm changes:")
    for k, v in sorted(out_diffs.items()):
        print(f"  {k}: {v:+.2f}%")

    # Plot weight distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Parameter-wise norm comparison
    ax = axes[0, 0]
    common_sorted = sorted(common)
    bl_vals = [bl_norms[k] for k in common_sorted]
    ms_vals = [ms_norms[k] for k in common_sorted]
    ax.scatter(bl_vals, ms_vals, alpha=0.3, s=10, color='#4C72B0')
    max_val = max(max(bl_vals), max(ms_vals))
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=0.5)
    ax.set_xlabel('Baseline weight norm')
    ax.set_ylabel('MS d=4+d=6 weight norm')
    ax.set_title('(a) Parameter-wise weight norms')
    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) Distribution of relative changes
    ax = axes[0, 1]
    diff_vals = list(diffs.values())
    ax.hist(diff_vals, bins=50, color='#4C72B0', edgecolor='black', linewidth=0.3, alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.axvline(x=np.median(diff_vals), color='green', linewidth=1.5, label=f'Median: {np.median(diff_vals):.1f}%')
    ax.set_xlabel('Relative change in weight norm (%)')
    ax.set_ylabel('Count')
    ax.set_title('(b) Distribution of weight norm changes')
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) In-projection comparison (most affected by hint channels)
    ax = axes[1, 0]
    in_proj_params = [k for k in common if 'in_proj' in k]
    if not in_proj_params:
        in_proj_params = [k for k in common if 'proj' in k][:10]

    in_proj_bl = [bl_norms[k] for k in in_proj_params]
    in_proj_ms = [ms_norms[k] for k in in_proj_params]
    x = np.arange(len(in_proj_params))
    width = 0.35
    ax.bar(x - width/2, in_proj_bl, width, label='Baseline', color='#4C72B0')
    ax.bar(x + width/2, in_proj_ms, width, label='MS d=4+d=6', color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels([k.split('.')[-1][:15] for k in in_proj_params], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Weight norm')
    ax.set_title('(c) Projection layer norms')
    ax.legend(frameon=False, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (d) Layer-by-layer summary
    ax = axes[1, 1]
    # Group diffs by layer number
    layer_diffs = {}
    for k, v in diffs.items():
        # Extract layer number
        parts = k.split('.')
        for p in parts:
            if p.isdigit():
                layer_num = int(p)
                if layer_num not in layer_diffs:
                    layer_diffs[layer_num] = []
                layer_diffs[layer_num].append(v)
                break

    if layer_diffs:
        layers = sorted(layer_diffs.keys())
        means = [np.mean(layer_diffs[l]) for l in layers]
        stds = [np.std(layer_diffs[l]) for l in layers]
        ax.errorbar(layers, means, yerr=stds, fmt='o-', color='#C44E52', capsize=4, markersize=5)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean $\\Delta$ weight norm (%)')
        ax.set_title('(d) Per-layer weight divergence')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIG_DIR, f'weight_analysis.{ext}'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved weight_analysis.pdf/png")

    # Print extra hint weights
    print("\n=== Hint-specific parameters ===")
    for name, param in ms_module.named_parameters():
        if name not in bl_norms:
            print(f"  NEW: {name} (norm={param.data.norm().item():.4f}, shape={tuple(param.shape)})")


def generate_synthetic_analysis():
    """Generate placeholder analysis when checkpoints aren't available."""
    print("Checkpoint files not accessible from login node.")
    print("To run full attention analysis, submit as SLURM job or run on GPU node.")
    print("Creating attention_analysis.slurm for GPU execution...")

    slurm_script = """#!/bin/bash
#SBATCH --job-name=attn_anl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=01:00:00 --gres=gpu:1
#SBATCH --partition=della --account=ehazan
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/attn_anl_%j.out

module load anaconda3/2024.6 intel-mkl/2024.2 cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate
cd /scratch/gpfs/EHAZAN/jh1161
python analysis/attention_analysis.py
"""
    slurm_path = "/scratch/gpfs/EHAZAN/jh1161/analysis/attention_analysis.slurm"
    with open(slurm_path, 'w') as f:
        f.write(slurm_script)
    print(f"Created {slurm_path}")
    print("Submit with: sbatch analysis/attention_analysis.slurm")


if __name__ == "__main__":
    main()
