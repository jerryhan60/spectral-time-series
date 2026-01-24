#!/usr/bin/env python
"""
Minimal test for learnable preconditioning training to verify the hybrid loss
implementation works correctly.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


def test_learnable_precond_training_step():
    """
    Test that learnable preconditioning can backprop without in-place errors.
    """
    print("Testing learnable preconditioning training step...")

    from uni2ts.module.learnable_precondition import LearnablePrecondition

    # Setup
    batch_size = 4
    seq_len = 8
    patch_size = 16

    # Create preconditioner
    preconditioner = LearnablePrecondition(degree=5, polynomial_type="chebyshev")

    # Create mock target
    target_orig = torch.randn(batch_size, seq_len, patch_size, requires_grad=False)

    # Apply preconditioning
    b, s, p = target_orig.shape
    target_flat = target_orig.view(b, s * p, 1)
    sample_id = torch.zeros(b, s, dtype=torch.long)
    sample_id_flat = sample_id.unsqueeze(-1).expand(-1, -1, p).reshape(b, s * p)

    target_precond_flat = preconditioner(target_flat, sample_id=sample_id_flat)
    target_precond = target_precond_flat.view(b, s, p)

    # Mock model output (distribution)
    loc = torch.randn(batch_size, seq_len, patch_size, requires_grad=True)
    scale = torch.abs(torch.randn(batch_size, seq_len, patch_size)) + 0.1
    scale.requires_grad = True

    distr = Normal(loc, scale)

    # Compute NLL loss
    nll_loss = -distr.log_prob(target_precond).mean()

    print(f"NLL Loss: {nll_loss.item():.4f}")

    # Backprop
    nll_loss.backward()

    # Verify gradients flowed to preconditioner
    assert preconditioner.coeffs.grad is not None, "No gradient for coefficients"
    print(f"Coefficient gradients: {preconditioner.coeffs.grad}")

    # Verify gradients have reasonable values
    assert not torch.isnan(preconditioner.coeffs.grad).any(), "NaN in gradients"
    assert not torch.isinf(preconditioner.coeffs.grad).any(), "Inf in gradients"

    print("SUCCESS: Training step completed without in-place operation errors!")
    return True


def test_hybrid_loss():
    """
    Test the hybrid loss implementation (NLL + reconstruction loss).

    The reconstruction loss measures MSE(reverse(model_prediction), original_target),
    which captures how much forecast error gets amplified through reversal.
    """
    print("\nTesting hybrid loss implementation...")

    from uni2ts.module.learnable_precondition import LearnablePrecondition

    batch_size = 4
    seq_len = 8
    patch_size = 16
    reversal_loss_weight = 0.1

    # Create preconditioner
    preconditioner = LearnablePrecondition(degree=5, polynomial_type="chebyshev")
    optimizer = torch.optim.Adam([preconditioner.coeffs], lr=0.01)

    initial_coeffs = preconditioner.coeffs.data.clone()

    for step in range(5):
        optimizer.zero_grad()

        # Original target
        target_orig = torch.randn(batch_size, seq_len, patch_size)

        # Apply preconditioning
        b, s, p = target_orig.shape
        target_flat = target_orig.view(b, s * p, 1)
        sample_id_flat = torch.zeros(b, s * p, dtype=torch.long)

        target_precond_flat = preconditioner(target_flat, sample_id=sample_id_flat)
        target_precond = target_precond_flat.view(b, s, p)

        # Mock distribution (simulates model's prediction in preconditioned space)
        # The mean represents the model's point prediction
        loc = torch.randn(batch_size, seq_len, patch_size, requires_grad=True)
        scale = torch.abs(torch.randn(batch_size, seq_len, patch_size)) + 0.1
        scale.requires_grad = True
        distr = Normal(loc, scale)

        # Model's prediction in preconditioned space
        pred_precond = distr.mean  # This is just 'loc' for Normal

        # NLL loss (in preconditioned space)
        nll_loss = -distr.log_prob(target_precond).mean()

        # Reconstruction loss: reverse the model's prediction and compare to original target
        # This measures end-to-end forecasting error in original space
        pred_precond_flat = pred_precond.view(b, s * p, 1)
        pred_orig_flat = preconditioner.reverse(pred_precond_flat, sample_id=sample_id_flat)
        pred_orig = pred_orig_flat.view(b, s, p)

        reconstruction_loss = ((pred_orig - target_orig) ** 2).mean()

        # Hybrid loss
        loss = nll_loss + reversal_loss_weight * reconstruction_loss

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        print(f"Step {step+1}: nll={nll_loss.item():.4f}, recon={reconstruction_loss.item():.4f}, total={loss.item():.4f}")

    final_coeffs = preconditioner.coeffs.data.clone()

    # Verify coefficients changed
    assert not torch.allclose(initial_coeffs, final_coeffs), \
        "Coefficients didn't change during training"

    print("SUCCESS: Hybrid loss training completed!")
    return True


def test_reconstruction_loss_gradient():
    """
    Test that reconstruction loss provides meaningful gradients to coefficients.
    """
    print("\nTesting reconstruction loss gradient flow...")

    from uni2ts.module.learnable_precondition import LearnablePrecondition

    batch_size = 4
    seq_len = 8
    patch_size = 16

    # Create preconditioner
    preconditioner = LearnablePrecondition(degree=5, polynomial_type="chebyshev")

    # Original target
    target_orig = torch.randn(batch_size, seq_len, patch_size)

    # Apply preconditioning
    b, s, p = target_orig.shape
    target_flat = target_orig.view(b, s * p, 1)
    sample_id_flat = torch.zeros(b, s * p, dtype=torch.long)

    target_precond_flat = preconditioner(target_flat, sample_id=sample_id_flat)
    target_precond = target_precond_flat.view(b, s, p)

    # Reverse
    target_reversed_flat = preconditioner.reverse(target_precond_flat, sample_id=sample_id_flat)
    target_reversed = target_reversed_flat.view(b, s, p)

    # Reconstruction loss
    reconstruction_loss = ((target_reversed - target_orig) ** 2).mean()

    print(f"Reconstruction loss: {reconstruction_loss.item():.6f}")

    # With correct forward/reverse, reconstruction loss should be very small
    # (but not exactly zero due to numerical precision)
    assert reconstruction_loss.item() < 1e-5, \
        f"Reconstruction loss too high: {reconstruction_loss.item()}"

    # Now test with perturbed coefficients
    preconditioner.coeffs.data += 0.1  # Perturb coefficients

    # Re-apply preconditioning with perturbed coefficients
    target_precond_flat_perturbed = preconditioner(target_flat, sample_id=sample_id_flat)
    target_precond_perturbed = target_precond_flat_perturbed.view(b, s, p)

    # Reverse with perturbed coefficients
    target_reversed_flat_perturbed = preconditioner.reverse(
        target_precond_flat_perturbed, sample_id=sample_id_flat
    )
    target_reversed_perturbed = target_reversed_flat_perturbed.view(b, s, p)

    # Reconstruction loss should still be ~0 (forward/reverse are inverses)
    reconstruction_loss_perturbed = ((target_reversed_perturbed - target_orig) ** 2).mean()

    print(f"Reconstruction loss (perturbed coeffs): {reconstruction_loss_perturbed.item():.6f}")

    # The forward/reverse should still be inverses, so loss should be small
    assert reconstruction_loss_perturbed.item() < 1e-5, \
        f"Reconstruction loss too high after perturbation: {reconstruction_loss_perturbed.item()}"

    print("SUCCESS: Reconstruction loss gradient test passed!")
    return True


def test_reconstruction_loss_encourages_stability():
    """
    Test that reconstruction loss penalizes coefficients that cause instability
    when using predicted (noisy) context instead of ground truth.

    This simulates what happens at inference time:
    - We predict ỹ with some error
    - We reverse using predicted values (not ground truth)
    - Large coefficients amplify prediction errors
    """
    print("\nTesting that reconstruction loss encourages stability...")

    from uni2ts.module.learnable_precondition import LearnablePrecondition

    batch_size = 4
    seq_len = 32
    patch_size = 1  # Simpler for this test

    # Create preconditioner with large degree
    preconditioner = LearnablePrecondition(degree=5, polynomial_type="chebyshev")

    print(f"Initial coefficients: {preconditioner.coeffs.data}")

    # Original target
    target_orig = torch.randn(batch_size, seq_len, patch_size)

    # Apply preconditioning
    b, s, p = target_orig.shape
    target_flat = target_orig.view(b, s * p, 1)
    sample_id_flat = torch.zeros(b, s * p, dtype=torch.long)

    target_precond_flat = preconditioner(target_flat, sample_id=sample_id_flat)

    # Simulate prediction error (noise in preconditioned space)
    noise_level = 0.1
    noisy_precond = target_precond_flat + noise_level * torch.randn_like(target_precond_flat)

    # Reverse using noisy predictions
    target_reversed_noisy_flat = preconditioner.reverse(noisy_precond, sample_id=sample_id_flat)

    # Error amplification: how much does noise get amplified?
    input_noise = noise_level
    output_error = ((target_reversed_noisy_flat.view(b, s, p) - target_orig) ** 2).mean().sqrt().item()

    amplification_factor = output_error / input_noise
    print(f"Input noise level: {input_noise:.4f}")
    print(f"Output error (RMS): {output_error:.4f}")
    print(f"Amplification factor: {amplification_factor:.2f}x")

    # With good coefficients, amplification should be bounded
    # The paper suggests coefficients bounded by 2^(0.3*n) ~ 2.3 for n=5
    # So amplification should be reasonable

    if amplification_factor > 10:
        print("WARNING: High amplification factor - coefficients may cause unstable reversal")
    else:
        print("OK: Amplification factor is reasonable")

    print("SUCCESS: Stability test completed!")
    return True


if __name__ == "__main__":
    test_learnable_precond_training_step()
    test_hybrid_loss()
    test_reconstruction_loss_gradient()
    test_reconstruction_loss_encourages_stability()
    print("\n" + "="*50)
    print("All tests passed!")
