"""
Phase 1 Test: STU Adapter Module Integration Tests

This script tests the core STU modules created in Phase 1:
1. SpectralFilters - filter computation and caching
2. STUCore - core spectral convolution
3. PackedSTU - packed sequence handling
4. STUEncoderLayer - TransformerEncoderLayer-compatible interface

Run with: python tests/test_stu_phase1.py
"""

import sys
import torch
import torch.nn as nn
import numpy as np

# Add the source directory to path
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')

def test_spectral_filters():
    """Test spectral filter computation."""
    print("=" * 60)
    print("Test 1: Spectral Filter Computation")
    print("=" * 60)

    from uni2ts.module.spectral_filters import (
        get_hankel_matrix,
        compute_spectral_filters,
        SpectralFilterBank,
        convolve_spectral,
    )

    # Test Hankel matrix construction
    print("\n1.1 Testing Hankel matrix construction...")
    Z_standard = get_hankel_matrix(64, use_hankel_L=False)
    Z_hankel_L = get_hankel_matrix(64, use_hankel_L=True)

    assert Z_standard.shape == (64, 64), f"Wrong shape: {Z_standard.shape}"
    assert Z_hankel_L.shape == (64, 64), f"Wrong shape: {Z_hankel_L.shape}"
    assert np.allclose(Z_standard, Z_standard.T), "Standard Hankel not symmetric"
    assert np.allclose(Z_hankel_L, Z_hankel_L.T), "Hankel-L not symmetric"
    print("   [PASS] Hankel matrices are symmetric and correct shape")

    # Test filter computation
    print("\n1.2 Testing spectral filter computation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phi = compute_spectral_filters(
        seq_len=128,
        num_eigh=24,
        use_hankel_L=False,
        device=device,
        dtype=torch.float32,
    )

    assert phi.shape == (128, 24), f"Wrong filter shape: {phi.shape}"
    assert phi.device.type == device.type, f"Wrong device: {phi.device}"
    print(f"   [PASS] Filters computed: shape={phi.shape}, device={phi.device}")

    # Test filter bank
    print("\n1.3 Testing SpectralFilterBank...")
    filter_bank = SpectralFilterBank(
        max_seq_len=2048,
        patch_sizes=(8, 16, 32, 64, 128),
        num_eigh=24,
    )

    # Get filters for different lengths
    phi_256 = filter_bank.get_filters(256, device=device)
    phi_128 = filter_bank.get_filters(128, device=device)
    phi_64 = filter_bank.get_filters(64, device=device)

    assert phi_256.shape == (256, 24), f"Wrong shape: {phi_256.shape}"
    assert phi_128.shape == (128, 24), f"Wrong shape: {phi_128.shape}"
    assert phi_64.shape == (64, 24), f"Wrong shape: {phi_64.shape}"
    print(f"   [PASS] Filter bank working for multiple lengths")

    # Test convolution
    print("\n1.4 Testing spectral convolution...")
    batch, seq_len, d_model = 2, 64, 32
    x = torch.randn(batch, seq_len, d_model, device=device)
    phi_proj = torch.randn(seq_len, d_model, device=device)  # Projected filters

    U_plus, U_minus = convolve_spectral(x, phi_proj, use_approx=True, return_both=True)

    assert U_plus.shape == (batch, seq_len, d_model), f"Wrong U_plus shape: {U_plus.shape}"
    assert U_minus.shape == (batch, seq_len, d_model), f"Wrong U_minus shape: {U_minus.shape}"
    print(f"   [PASS] Convolution output shapes correct")

    # Test single-branch convolution
    U_plus_single, U_minus_single = convolve_spectral(x, phi_proj, use_approx=True, return_both=False)
    assert U_plus_single.shape == (batch, seq_len, d_model)
    assert U_minus_single is None
    print(f"   [PASS] Single-branch (Hankel-L) convolution working")

    print("\n[PASS] All spectral filter tests passed!")
    return True


def test_stu_core():
    """Test STUCore module."""
    print("\n" + "=" * 60)
    print("Test 2: STUCore Module")
    print("=" * 60)

    from uni2ts.module.stu_adapter import STUCore

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test approximation mode
    print("\n2.1 Testing STUCore (approx mode)...")
    stu_approx = STUCore(
        d_model=256,
        max_seq_len=512,
        num_eigh=24,
        use_hankel_L=False,
        use_approx=True,
    ).to(device)

    x = torch.randn(2, 128, 256, device=device)
    y = stu_approx(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert not torch.isnan(y).any(), "NaN in output"
    print(f"   [PASS] Approx mode: input {x.shape} -> output {y.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in stu_approx.parameters())
    print(f"   Parameters (approx): {num_params:,}")

    # Test standard mode
    print("\n2.2 Testing STUCore (standard mode)...")
    stu_standard = STUCore(
        d_model=256,
        max_seq_len=512,
        num_eigh=24,
        use_hankel_L=False,
        use_approx=False,
    ).to(device)

    y_std = stu_standard(x)
    assert y_std.shape == x.shape
    assert not torch.isnan(y_std).any()

    num_params_std = sum(p.numel() for p in stu_standard.parameters())
    print(f"   [PASS] Standard mode: input {x.shape} -> output {y_std.shape}")
    print(f"   Parameters (standard): {num_params_std:,}")
    print(f"   Param ratio (standard/approx): {num_params_std / num_params:.1f}x")

    # Test Hankel-L mode
    print("\n2.3 Testing STUCore (Hankel-L mode)...")
    stu_hankel_L = STUCore(
        d_model=256,
        max_seq_len=512,
        num_eigh=24,
        use_hankel_L=True,
        use_approx=True,
    ).to(device)

    y_L = stu_hankel_L(x)
    assert y_L.shape == x.shape
    assert not torch.isnan(y_L).any()
    print(f"   [PASS] Hankel-L mode working")

    # Test gradient flow
    print("\n2.4 Testing gradient flow...")
    x = torch.randn(2, 128, 256, device=device, requires_grad=True)
    y = stu_approx(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "No gradient for input"
    assert stu_approx.M_inputs.grad is not None, "No gradient for M_inputs"
    assert stu_approx.M_filters.grad is not None, "No gradient for M_filters"
    print(f"   [PASS] Gradients flowing correctly")

    print("\n[PASS] All STUCore tests passed!")
    return True


def test_packed_stu():
    """Test PackedSTU for packed sequence handling."""
    print("\n" + "=" * 60)
    print("Test 3: PackedSTU Module")
    print("=" * 60)

    from uni2ts.module.stu_adapter import PackedSTU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create module
    packed_stu = PackedSTU(
        d_model=128,
        max_seq_len=256,
        num_eigh=16,
        use_approx=True,
    ).to(device)

    # Test without packing
    print("\n3.1 Testing without sample_id (no packing)...")
    x = torch.randn(2, 64, 128, device=device)
    y = packed_stu(x)

    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    print(f"   [PASS] No packing: {x.shape} -> {y.shape}")

    # Test with packing
    print("\n3.2 Testing with sample_id (packed sequences)...")
    # Create packed sequence: 3 samples of lengths 20, 30, 50
    total_len = 100
    x_packed = torch.randn(1, total_len, 128, device=device)
    sample_id = torch.cat([
        torch.full((20,), 0, dtype=torch.long),
        torch.full((30,), 1, dtype=torch.long),
        torch.full((50,), 2, dtype=torch.long),
    ]).unsqueeze(0).to(device)

    y_packed = packed_stu(x_packed, sample_id=sample_id)

    assert y_packed.shape == x_packed.shape
    assert not torch.isnan(y_packed).any()
    print(f"   [PASS] Packed sequence: {x_packed.shape} with 3 samples -> {y_packed.shape}")

    # Verify each sample processed independently
    print("\n3.3 Verifying sample independence...")
    # Process first sample manually
    x_sample0 = x_packed[:, :20, :]
    y_sample0_direct = packed_stu.stu_core(x_sample0)
    y_sample0_from_packed = y_packed[:, :20, :]

    assert torch.allclose(y_sample0_direct, y_sample0_from_packed, atol=1e-5), \
        "Packed output differs from direct output"
    print(f"   [PASS] Sample independence verified")

    # Test gradient flow through packed sequences
    print("\n3.4 Testing gradient flow through packed sequences...")
    x_packed = torch.randn(1, total_len, 128, device=device, requires_grad=True)
    y_packed = packed_stu(x_packed, sample_id=sample_id)
    loss = y_packed.sum()
    loss.backward()

    assert x_packed.grad is not None
    print(f"   [PASS] Gradients flowing through packed sequences")

    print("\n[PASS] All PackedSTU tests passed!")
    return True


def test_stu_encoder_layer():
    """Test STUEncoderLayer interface compatibility."""
    print("\n" + "=" * 60)
    print("Test 4: STUEncoderLayer")
    print("=" * 60)

    from uni2ts.module.stu_layer import STUEncoderLayer, VariateAwareSTUEncoderLayer
    from uni2ts.module.transformer import TransformerEncoderLayer
    from uni2ts.module.attention import GroupedQueryAttention
    from uni2ts.module.ffn import GatedLinearUnitFeedForward
    from uni2ts.module.norm import RMSNorm
    from functools import partial

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 256

    # Create STU layer
    print("\n4.1 Creating STUEncoderLayer...")
    stu_layer = STUEncoderLayer(
        d_model=d_model,
        max_seq_len=512,
        num_eigh=24,
        use_approx=True,
        pre_norm=True,
        dropout_p=0.1,
        norm_layer=RMSNorm,
        use_glu=True,
    ).to(device)

    num_params_stu = sum(p.numel() for p in stu_layer.parameters())
    print(f"   STU layer parameters: {num_params_stu:,}")

    # Test forward pass
    print("\n4.2 Testing forward pass...")
    x = torch.randn(2, 128, d_model, device=device)

    # Basic forward (matching TransformerEncoderLayer signature)
    y = stu_layer(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    print(f"   [PASS] Basic forward: {x.shape} -> {y.shape}")

    # Forward with optional args (for interface compatibility)
    y = stu_layer(x, attn_mask=None, var_id=None, time_id=None)
    assert y.shape == x.shape
    print(f"   [PASS] Forward with optional args")

    # Forward with sample_id
    sample_id = torch.zeros(2, 128, dtype=torch.long, device=device)
    y = stu_layer(x, sample_id=sample_id)
    assert y.shape == x.shape
    print(f"   [PASS] Forward with sample_id")

    # Test VariateAwareSTUEncoderLayer
    print("\n4.3 Testing VariateAwareSTUEncoderLayer...")
    var_stu_layer = VariateAwareSTUEncoderLayer(
        d_model=d_model,
        max_seq_len=512,
        max_variates=50,
        num_eigh=24,
    ).to(device)

    var_id = torch.randint(0, 10, (2, 128), device=device)
    y_var = var_stu_layer(x, var_id=var_id)
    assert y_var.shape == x.shape
    print(f"   [PASS] Variate-aware layer: {x.shape} -> {y_var.shape}")

    # Compare parameter counts with equivalent attention layer
    print("\n4.4 Parameter comparison with attention layer...")
    num_heads = d_model // 64
    attn = GroupedQueryAttention(
        dim=d_model,
        num_heads=num_heads,
        num_groups=num_heads,
        bias=False,
    )
    ffn = GatedLinearUnitFeedForward(in_dim=d_model)

    attn_params = sum(p.numel() for p in attn.parameters()) + sum(p.numel() for p in ffn.parameters())
    stu_params = num_params_stu

    print(f"   Attention layer (approx): {attn_params:,}")
    print(f"   STU layer: {stu_params:,}")
    print(f"   Ratio: {attn_params / stu_params:.2f}x")

    print("\n[PASS] All STUEncoderLayer tests passed!")
    return True


def test_integration():
    """Test STU integration with MOIRAI-like usage patterns."""
    print("\n" + "=" * 60)
    print("Test 5: Integration Test")
    print("=" * 60)

    from uni2ts.module.stu_layer import STUEncoderLayer
    from uni2ts.module.transformer import TransformerEncoderLayer
    from uni2ts.module.attention import GroupedQueryAttention
    from uni2ts.module.ffn import GatedLinearUnitFeedForward
    from uni2ts.module.norm import RMSNorm
    from functools import partial

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_model = 256
    num_layers = 4
    max_seq_len = 512

    # Build hybrid encoder (alternating STU and attention)
    print("\n5.1 Building hybrid encoder (alternating STU/Attention)...")

    layers = nn.ModuleList()
    for i in range(num_layers):
        if i % 2 == 0:
            # STU layer
            layers.append(STUEncoderLayer(
                d_model=d_model,
                max_seq_len=max_seq_len,
                num_eigh=24,
            ))
        else:
            # Attention layer
            attn = GroupedQueryAttention(
                dim=d_model,
                num_heads=d_model // 64,
                num_groups=d_model // 64,
                bias=False,
                norm_layer=RMSNorm,
            )
            ffn = GatedLinearUnitFeedForward(in_dim=d_model)
            layers.append(TransformerEncoderLayer(
                self_attn=attn,
                ffn=ffn,
                norm1=RMSNorm(d_model),
                norm2=RMSNorm(d_model),
                pre_norm=True,
            ))

    layers = layers.to(device)

    total_params = sum(p.numel() for p in layers.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Layers: STU -> Attn -> STU -> Attn")

    # Forward pass through hybrid encoder
    print("\n5.2 Testing forward pass through hybrid encoder...")
    x = torch.randn(2, 128, d_model, device=device)

    for i, layer in enumerate(layers):
        layer_type = "STU" if i % 2 == 0 else "Attn"
        x = layer(x)
        print(f"   Layer {i} ({layer_type}): output shape = {x.shape}")

    assert x.shape == (2, 128, d_model)
    assert not torch.isnan(x).any()
    print(f"   [PASS] Hybrid encoder forward pass")

    # Test with MOIRAI-like packed sequences
    print("\n5.3 Testing with MOIRAI-like packed sequences...")

    # Simulate packed batch: 2 time series packed together
    total_len = 200
    x_packed = torch.randn(1, total_len, d_model, device=device)
    sample_id = torch.cat([
        torch.full((80,), 0, dtype=torch.long),
        torch.full((120,), 1, dtype=torch.long),
    ]).unsqueeze(0).to(device)

    # Forward through hybrid encoder with packed data
    h = x_packed
    for i, layer in enumerate(layers):
        if i % 2 == 0:
            # STU layer: pass sample_id
            h = layer(h, sample_id=sample_id)
        else:
            # Attention layer: no sample_id needed (handled via attn_mask in MOIRAI)
            h = layer(h)

    assert h.shape == x_packed.shape
    assert not torch.isnan(h).any()
    print(f"   [PASS] Packed sequence processing: {x_packed.shape} -> {h.shape}")

    # Gradient test
    print("\n5.4 Testing gradient flow through hybrid encoder...")
    x = torch.randn(2, 64, d_model, device=device, requires_grad=True)
    h = x
    for layer in layers:
        h = layer(h)
    loss = h.sum()
    loss.backward()

    assert x.grad is not None
    print(f"   [PASS] Gradients flowing through hybrid encoder")

    print("\n[PASS] All integration tests passed!")
    return True


def main():
    """Run all Phase 1 tests."""
    print("\n" + "=" * 70)
    print("PHASE 1 TEST SUITE: STU Adapter Module for MOIRAI")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_passed = True

    try:
        all_passed &= test_spectral_filters()
        all_passed &= test_stu_core()
        all_passed &= test_packed_stu()
        all_passed &= test_stu_encoder_layer()
        all_passed &= test_integration()

    except Exception as e:
        print(f"\n[FAILED] Test error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL PHASE 1 TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
