#!/usr/bin/env python3
"""
Analyze the architectural differences between baseline MOIRAI and STU-MOIRAI.
"""

import torch
import sys
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')


def count_parameters(model):
    """Count parameters by module type."""
    param_counts = {}
    for name, param in model.named_parameters():
        # Extract module type from name
        parts = name.split('.')
        if 'encoder' in name:
            if 'stu' in name.lower():
                module_type = 'STU layers'
            elif 'self_attn' in name or 'attn' in name:
                module_type = 'Attention layers'
            elif 'ffn' in name:
                module_type = 'FFN layers'
            elif 'norm' in name:
                module_type = 'Norm layers'
            else:
                module_type = 'Encoder (other)'
        elif 'embed' in name.lower() or 'patch' in name.lower():
            module_type = 'Embeddings'
        elif 'head' in name.lower() or 'proj' in name.lower():
            module_type = 'Output head'
        else:
            module_type = 'Other'

        if module_type not in param_counts:
            param_counts[module_type] = 0
        param_counts[module_type] += param.numel()

    return param_counts


def analyze_models():
    from uni2ts.model.moirai import MoiraiModule, MoiraiHybridModule
    from uni2ts.distribution import MixtureOutput, StudentTOutput, NormalFixedScaleOutput
    from uni2ts.distribution import NegativeBinomialOutput, LogNormalOutput

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Common config
    distr_output = MixtureOutput([
        StudentTOutput(),
        NormalFixedScaleOutput(),
        NegativeBinomialOutput(),
        LogNormalOutput(),
    ])

    d_model = 384
    num_layers = 6
    patch_sizes = (8, 16, 32, 64, 128)
    max_seq_len = 512

    print("\n" + "=" * 70)
    print("Architecture Comparison: Baseline vs STU-MOIRAI")
    print("=" * 70)

    # Create baseline model
    print("\n### Baseline MOIRAI ###")
    baseline = MoiraiModule(
        distr_output=distr_output,
        d_model=d_model,
        num_layers=num_layers,
        patch_sizes=patch_sizes,
        max_seq_len=max_seq_len,
        attn_dropout_p=0.0,
        dropout_p=0.0,
        scaling=True,
    )

    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"Total parameters: {baseline_params:,} ({baseline_params/1e6:.2f}M)")

    # Create STU model
    print("\n### STU-MOIRAI (alternating) ###")
    stu_model = MoiraiHybridModule(
        distr_output=distr_output,
        d_model=d_model,
        num_layers=num_layers,
        patch_sizes=patch_sizes,
        max_seq_len=max_seq_len,
        attn_dropout_p=0.0,
        dropout_p=0.0,
        scaling=True,
        stu_layer_pattern="alternating",
        num_eigh=24,
        use_hankel_L=False,
        use_approx=True,
    )

    stu_params = sum(p.numel() for p in stu_model.parameters())
    print(f"Total parameters: {stu_params:,} ({stu_params/1e6:.2f}M)")

    # Parameter breakdown
    print("\n" + "=" * 70)
    print("Parameter Breakdown")
    print("=" * 70)

    print("\n### Baseline ###")
    baseline_counts = count_parameters(baseline)
    for module_type, count in sorted(baseline_counts.items(), key=lambda x: -x[1]):
        print(f"  {module_type:<25} {count:>12,} ({100*count/baseline_params:>5.1f}%)")

    print("\n### STU-MOIRAI ###")
    stu_counts = count_parameters(stu_model)
    for module_type, count in sorted(stu_counts.items(), key=lambda x: -x[1]):
        print(f"  {module_type:<25} {count:>12,} ({100*count/stu_params:>5.1f}%)")

    # Layer-by-layer comparison
    print("\n" + "=" * 70)
    print("Layer Architecture")
    print("=" * 70)

    print("\n### Baseline: 6 Transformer layers ###")
    print("  All layers: Self-Attention -> FFN")
    print("  Attention: Q/K/V projections + output projection")
    print(f"  Per attention layer params: ~{d_model * d_model * 4 / 1e6:.2f}M (QKV + out)")

    print("\n### STU-MOIRAI: 3 STU + 3 Attention layers (alternating) ###")
    print("  Layer 0, 2, 4: STU -> FFN")
    print("  Layer 1, 3, 5: Self-Attention -> FFN")
    print("  STU (approx mode): M_inputs [384x384] + M_filters [24x384]")
    stu_layer_params = d_model * d_model + 24 * d_model
    print(f"  Per STU layer params: ~{stu_layer_params / 1e6:.3f}M")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nParameter reduction: {baseline_params - stu_params:,} ({100*(baseline_params-stu_params)/baseline_params:.1f}%)")
    print(f"\nArchitectural change:")
    print(f"  - 50% of attention layers replaced with STU")
    print(f"  - STU uses spectral (FFT) convolution instead of dot-product attention")
    print(f"  - STU has O(L log L) complexity vs O(L²) for attention")
    print(f"  - STU has fewer parameters per layer (~{stu_layer_params/1e3:.0f}K vs ~{d_model*d_model*4/1e3:.0f}K)")

    print("\n" + "=" * 70)
    print("Why similar training loss is expected:")
    print("=" * 70)
    print("""
1. SAME components preserved:
   - Patch embeddings (input representation)
   - FFN layers in all 6 layers (majority of computation)
   - Output distribution head
   - 50% of attention layers still present

2. STU is a valid sequence mixer:
   - Both attention and STU mix information across sequence positions
   - STU uses spectral basis (learned from Hankel matrix eigenvectors)
   - Mathematically, STU approximates linear dynamical systems

3. The model is primarily learning:
   - Time series patterns (trends, seasonality)
   - Distribution parameters for forecasting
   - These don't require the specific inductive bias of attention

4. Key insight:
   - Similar loss ≠ same model behavior
   - Need to check EVALUATION metrics (CRPS, MASE, etc.) to see real differences
   - STU might generalize differently on out-of-distribution data
""")


if __name__ == "__main__":
    analyze_models()
