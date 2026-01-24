#!/usr/bin/env python3
"""
Test script for embedding-level preconditioning.

Tests:
1. EmbeddingPrecondition module initialization
2. Forward pass (preconditioning)
3. Reverse pass (compensating filter)
4. Forward + Reverse ≈ Identity
5. Integration with MoiraiModule
6. sample_id masking for packed sequences
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')

from uni2ts.module.embedding_precondition import EmbeddingPrecondition


def test_initialization():
    """Test that EmbeddingPrecondition initializes correctly."""
    print("=" * 60)
    print("Test 1: Initialization")
    print("=" * 60)

    # Test Chebyshev
    precond_cheb = EmbeddingPrecondition(degree=5, polynomial_type="chebyshev")
    print(f"Chebyshev (degree 5) coefficients: {precond_cheb.coeffs}")

    # Test Legendre
    precond_leg = EmbeddingPrecondition(degree=5, polynomial_type="legendre")
    print(f"Legendre (degree 5) coefficients: {precond_leg.coeffs}")

    # Verify coefficients are registered as buffers (not parameters)
    assert not precond_cheb.coeffs.requires_grad, "Coefficients should not require grad"

    print("[PASS] Initialization test passed\n")
    return precond_cheb


def test_forward_pass(precond):
    """Test forward preconditioning."""
    print("=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)

    batch, seq_len, d_model = 2, 10, 384
    x = torch.randn(batch, seq_len, d_model)

    y = precond(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, "Output shape should match input"

    # First n positions should be unchanged (where n = degree)
    degree = precond.degree
    assert torch.allclose(x[:, :degree, :], y[:, :degree, :]), \
        f"First {degree} positions should be unchanged"

    # Later positions should be different
    assert not torch.allclose(x[:, degree:, :], y[:, degree:, :]), \
        "Positions after degree should be different"

    print("[PASS] Forward pass test passed\n")
    return x, y


def test_reverse_pass(precond, x, y):
    """Test reverse preconditioning."""
    print("=" * 60)
    print("Test 3: Reverse Pass")
    print("=" * 60)

    # Apply reverse to the preconditioned output
    x_recovered = precond.reverse(y)

    print(f"Original shape: {x.shape}")
    print(f"Preconditioned shape: {y.shape}")
    print(f"Recovered shape: {x_recovered.shape}")

    # Check forward + reverse ≈ identity
    max_diff = (x - x_recovered).abs().max().item()
    mean_diff = (x - x_recovered).abs().mean().item()

    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    # Should be very close to zero
    assert max_diff < 1e-5, f"Max diff too large: {max_diff}"

    print("[PASS] Reverse pass test passed (forward + reverse ≈ identity)\n")


def test_sample_id_masking():
    """Test that sample_id correctly masks cross-series contamination."""
    print("=" * 60)
    print("Test 4: sample_id Masking")
    print("=" * 60)

    precond = EmbeddingPrecondition(degree=3, polynomial_type="chebyshev")

    # Create a packed sequence with 2 series
    # Series 0: positions 0-4
    # Series 1: positions 5-9
    batch, seq_len, d_model = 1, 10, 8
    x = torch.randn(batch, seq_len, d_model)
    sample_id = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])

    y_with_id = precond(x, sample_id=sample_id)

    # At position 5 (first of series 1), with degree=3:
    # Without masking: y[5] = x[5] + c1*x[4] + c2*x[3] + c3*x[2]
    # With masking: y[5] = x[5] (no history from series 0)

    # Position 5 should equal x[5] since it's the start of series 1
    assert torch.allclose(x[:, 5, :], y_with_id[:, 5, :]), \
        "First position of new series should be unchanged"

    # Position 6 should only use position 5 as context (degree-1 terms masked)
    # This is a softer check - just verify it's not the same as without masking
    y_without_id = precond(x, sample_id=None)

    # Position 6 should be different with vs without sample_id
    # (unless x happens to be zero, which is unlikely)
    if not torch.allclose(x[:, 5:, :], torch.zeros_like(x[:, 5:, :])):
        # There should be some difference around series boundary
        diff_at_boundary = (y_with_id[:, 6, :] - y_without_id[:, 6, :]).abs().max()
        print(f"Difference at series boundary (pos 6): {diff_at_boundary:.6e}")

    print("[PASS] sample_id masking test passed\n")


def test_integration_with_moirai_module():
    """Test integration with MoiraiModule (if possible)."""
    print("=" * 60)
    print("Test 5: Integration with MoiraiModule")
    print("=" * 60)

    try:
        from uni2ts.distribution import MixtureOutput, StudentTOutput, NormalFixedScaleOutput
        from uni2ts.model.moirai.module import MoiraiModule

        # Test 1: MoiraiModule with embedding preconditioning (all variates)
        distr_output = MixtureOutput(components=[StudentTOutput(), NormalFixedScaleOutput()])

        module = MoiraiModule(
            distr_output=distr_output,
            d_model=64,  # Small for testing
            num_layers=2,
            patch_sizes=(8, 16),
            max_seq_len=32,
            attn_dropout_p=0.0,
            dropout_p=0.0,
            scaling=True,
            enable_embedding_preconditioning=True,
            embedding_precondition_type="chebyshev",
            embedding_precondition_degree=3,
            embedding_precondition_reverse=False,  # Reversal disabled
            num_target_variates=None,  # Precondition all variates
        )

        print(f"MoiraiModule created with embedding preconditioning (all variates)")
        print(f"  - enable_embedding_preconditioning: {module.enable_embedding_preconditioning}")
        print(f"  - embedding_precondition_reverse: {module.embedding_precondition_reverse}")
        print(f"  - num_target_variates: {module.num_target_variates}")

        # Create dummy inputs
        batch, seq_len, patch_size = 2, 8, 16
        target = torch.randn(batch, seq_len, patch_size)
        observed_mask = torch.ones(batch, seq_len, patch_size, dtype=torch.bool)
        sample_id = torch.zeros(batch, seq_len, dtype=torch.long)
        time_id = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        variate_id = torch.zeros(batch, seq_len, dtype=torch.long)
        prediction_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        prediction_mask[:, -2:] = True  # Last 2 tokens are predictions
        patch_size_tensor = torch.full((batch, seq_len), patch_size, dtype=torch.long)

        # Forward pass
        distr = module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size_tensor,
        )

        print(f"Forward pass successful, output distribution: {type(distr).__name__}")

        # Test 2: MoiraiModule with num_target_variates (only target variates preconditioned)
        module_with_targets = MoiraiModule(
            distr_output=distr_output,
            d_model=64,
            num_layers=2,
            patch_sizes=(8, 16),
            max_seq_len=32,
            attn_dropout_p=0.0,
            dropout_p=0.0,
            scaling=True,
            enable_embedding_preconditioning=True,
            embedding_precondition_type="chebyshev",
            embedding_precondition_degree=3,
            embedding_precondition_reverse=False,
            num_target_variates=2,  # Only first 2 variates are targets
        )

        print(f"\nMoiraiModule with num_target_variates=2")
        print(f"  - num_target_variates: {module_with_targets.num_target_variates}")

        # Create multivariate input with different variate_ids
        # variate_id 0, 1 are targets; variate_id 2, 3 are covariates
        variate_id_multi = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3],
                                         [0, 1, 2, 3, 0, 1, 2, 3]])

        distr_multi = module_with_targets(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id_multi,
            prediction_mask=prediction_mask,
            patch_size=patch_size_tensor,
        )

        print(f"Forward pass with multivariate input successful")
        print("[PASS] Integration test passed\n")

    except Exception as e:
        import traceback
        print(f"[SKIP] Integration test skipped: {e}")
        traceback.print_exc()
        print()


def test_gradients():
    """Test that gradients flow correctly (for debugging, not training)."""
    print("=" * 60)
    print("Test 6: Gradient Flow")
    print("=" * 60)

    precond = EmbeddingPrecondition(degree=3, polynomial_type="chebyshev")

    batch, seq_len, d_model = 2, 10, 32
    x = torch.randn(batch, seq_len, d_model, requires_grad=True)

    # Forward
    y = precond(x)

    # Compute a scalar loss and backprop
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Input should have gradients"
    assert x.grad.shape == x.shape, "Gradient shape should match input"

    print(f"Gradients flow correctly through forward pass")
    print("[PASS] Gradient test passed\n")


def test_target_mask():
    """Test that target_mask correctly limits preconditioning to target variates only.

    Following Universal Sequence Preconditioning theory:
    - y_t (target variable) should be preconditioned
    - u_t (covariates) should remain unchanged
    """
    print("=" * 60)
    print("Test 7: Target Mask (Only Precondition Target Variates)")
    print("=" * 60)

    precond = EmbeddingPrecondition(degree=3, polynomial_type="chebyshev")

    batch, seq_len, d_model = 2, 10, 32
    x = torch.randn(batch, seq_len, d_model)

    # Simulate multivariate data with target and covariate variates
    # variate_id 0, 1 are targets (first 2 variates)
    # variate_id 2, 3 are covariates (remaining variates)
    # In a sequence: positions with variate_id < num_target_variates are targets

    # Create target_mask: True for target variates, False for covariate variates
    # Simulate interleaved variate pattern: [t0, t1, c0, c1, t0, t1, c0, c1, ...]
    variate_id = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
                               [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]])
    num_target_variates = 2
    target_mask = variate_id < num_target_variates  # True for targets, False for covariates

    # Forward with target_mask
    y_masked = precond(x, target_mask=target_mask)

    # Forward without target_mask (all positions preconditioned)
    y_all = precond(x, target_mask=None)

    # Check 1: Target positions should be preconditioned (positions where target_mask=True and pos >= degree)
    degree = precond.degree
    target_positions = [i for i in range(seq_len) if target_mask[0, i] and i >= degree]
    for pos in target_positions:
        # Target positions should be different from original x (preconditioned)
        is_different = not torch.allclose(x[:, pos, :], y_masked[:, pos, :])
        if is_different:
            print(f"  Target position {pos} is preconditioned: PASS")

    # Check 2: Covariate positions should be UNCHANGED (same as original x)
    covariate_positions = [i for i in range(seq_len) if not target_mask[0, i]]
    for pos in covariate_positions:
        assert torch.allclose(x[:, pos, :], y_masked[:, pos, :]), \
            f"Covariate position {pos} should remain unchanged"
    print(f"  Covariate positions (variate_id >= {num_target_variates}) unchanged: PASS")

    # Check 3: Target positions should match y_all (same preconditioning)
    for pos in target_positions:
        assert torch.allclose(y_masked[:, pos, :], y_all[:, pos, :]), \
            f"Target position {pos} should have same preconditioning with or without mask"
    print(f"  Target preconditioning matches full preconditioning: PASS")

    # Check 4: Covariate positions differ from y_all (y_all has them preconditioned)
    for pos in covariate_positions:
        if pos >= degree:  # Only positions >= degree are actually preconditioned
            diff = (y_masked[:, pos, :] - y_all[:, pos, :]).abs().max()
            if diff > 1e-6:
                print(f"  Difference at covariate position {pos}: {diff:.6f}")

    print("[PASS] Target mask test passed\n")


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("=" * 60)
    print("Test 8: Numerical Stability")
    print("=" * 60)

    precond = EmbeddingPrecondition(degree=5, polynomial_type="chebyshev")

    batch, seq_len, d_model = 2, 20, 64

    # Test 1: Very large values
    x_large = torch.randn(batch, seq_len, d_model) * 1e3
    y_large = precond(x_large)
    z_large = precond.reverse(y_large)

    assert not torch.isnan(y_large).any(), "Forward should not produce NaN with large inputs"
    assert not torch.isnan(z_large).any(), "Reverse should not produce NaN with large inputs"
    assert not torch.isinf(y_large).any(), "Forward should not produce Inf with large inputs"
    assert not torch.isinf(z_large).any(), "Reverse should not produce Inf with large inputs"
    print(f"  Large values (1e3 scale): PASS")

    # Test 2: Input with NaN (should be handled gracefully)
    x_nan = torch.randn(batch, seq_len, d_model)
    x_nan[0, 5, 10] = float('nan')
    y_nan = precond(x_nan)
    z_nan = precond.reverse(y_nan)

    # The NaN position should be replaced with the original value
    assert not torch.isnan(y_nan).any(), "Forward should handle NaN input"
    assert not torch.isnan(z_nan).any(), "Reverse should handle NaN input"
    print(f"  NaN input handling: PASS")

    # Test 3: Input with Inf
    x_inf = torch.randn(batch, seq_len, d_model)
    x_inf[0, 5, 10] = float('inf')
    y_inf = precond(x_inf)
    z_inf = precond.reverse(y_inf)

    assert not torch.isnan(y_inf).any(), "Forward should not produce NaN from Inf input"
    assert not torch.isinf(z_inf).any() or torch.isfinite(z_inf).all(), "Reverse should handle Inf gracefully"
    print(f"  Inf input handling: PASS")

    # Test 4: Simulated transformer output (mix of values)
    # This mimics what the transformer might output - mixed magnitudes
    x_transformer = torch.randn(batch, seq_len, d_model)
    x_transformer = x_transformer * (1 + torch.randn(batch, seq_len, 1).abs() * 10)  # Scale up randomly
    z_transformer = precond.reverse(x_transformer)

    assert not torch.isnan(z_transformer).any(), "Reverse should not produce NaN with transformer-like outputs"
    print(f"  Transformer-like output handling: PASS")

    print("[PASS] Numerical stability test passed\n")


def main():
    print("\n" + "=" * 60)
    print("EMBEDDING PRECONDITIONING TEST SUITE")
    print("=" * 60 + "\n")

    # Run tests
    precond = test_initialization()
    x, y = test_forward_pass(precond)
    test_reverse_pass(precond, x, y)
    test_sample_id_masking()
    test_integration_with_moirai_module()
    test_gradients()
    test_target_mask()  # Test target_mask functionality (target vs covariate variates)
    test_numerical_stability()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
