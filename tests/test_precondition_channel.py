"""Tests for PreconditioningChannel module."""
import numpy as np
import pytest
import torch

from poly_precond.chebyshev import chebyshev_coefficients, compute_residual
from poly_precond.precondition_channel import PreconditioningChannel


def _make_inputs(B, T, P, *, n_samples=1, seed=0):
    """Create valid inputs for PreconditioningChannel.forward."""
    rng = torch.Generator().manual_seed(seed)
    target = torch.randn(B, T, P, generator=rng)
    observed_mask = torch.ones(B, T, P, dtype=torch.bool)
    if n_samples == 1:
        sample_id = torch.ones(B, T, dtype=torch.long)
    else:
        # Split patches evenly across samples
        sample_id = torch.zeros(B, T, dtype=torch.long)
        per = T // n_samples
        for i in range(n_samples):
            sample_id[:, i * per : (i + 1) * per] = i + 1
    variate_id = torch.ones(B, T, dtype=torch.long)
    time_id = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, -1)
    return target, observed_mask, sample_id, variate_id, time_id


def test_output_shape():
    B, T, P = 2, 20, 16
    mod = PreconditioningChannel(degree=4, stride=16, patch_size=P)
    target, mask, sid, vid, tid = _make_inputs(B, T, P)
    out = mod(target, mask, sid, vid, tid)
    assert out.shape == (B, T, P)


def test_first_d_patches_zero():
    """First d*s/P = d patches should be all zeros (insufficient lookback)."""
    B, T, P = 1, 20, 16
    d = 4
    mod = PreconditioningChannel(degree=d, stride=P, patch_size=P)
    target, mask, sid, vid, tid = _make_inputs(B, T, P)
    out = mod(target, mask, sid, vid, tid)
    # First d patches should be zero
    assert torch.allclose(out[:, :d, :], torch.zeros(B, d, P))


def test_no_cross_sample_leakage():
    """Two samples packed in sequence -- boundary should be respected."""
    B, T, P = 1, 20, 16
    d = 4
    mod = PreconditioningChannel(degree=d, stride=P, patch_size=P)

    # Create two samples: patches 0-9 = sample 1, patches 10-19 = sample 2
    target, mask, sid, vid, tid = _make_inputs(B, T, P, n_samples=2)

    out_packed = mod(target, mask, sid, vid, tid)

    # Now run each sample independently
    target1 = target[:, :10, :]
    sid1 = sid[:, :10]
    vid1 = vid[:, :10]
    tid1 = tid[:, :10]
    mask1 = mask[:, :10, :]
    out1 = mod(target1, mask1, sid1, vid1, tid1)

    target2 = target[:, 10:, :]
    sid2 = torch.ones_like(sid[:, 10:])  # renumber to 1
    vid2 = vid[:, 10:]
    tid2 = torch.arange(10, dtype=torch.long).unsqueeze(0).expand(B, -1)
    mask2 = mask[:, 10:, :]
    out2 = mod(target2, mask2, sid2, vid2, tid2)

    # Packed output for each sample region should match independent runs
    assert torch.allclose(out_packed[:, :10, :], out1, atol=1e-6)
    assert torch.allclose(out_packed[:, 10:, :], out2, atol=1e-6)


def test_dropout_train_eval():
    """Dropout active in train mode, inactive in eval mode."""
    B, T, P = 2, 20, 16
    mod = PreconditioningChannel(degree=4, stride=16, patch_size=P, dropout=0.99)
    target, mask, sid, vid, tid = _make_inputs(B, T, P)

    # Eval mode (default training=None -> self.training=False)
    mod.eval()
    out_eval = mod(target, mask, sid, vid, tid)

    # Train mode
    mod.train()
    torch.manual_seed(42)
    out_train = mod(target, mask, sid, vid, tid)

    # With 99% dropout, train output should have most patches zeroed
    nonzero_eval = (out_eval.abs() > 0).float().mean().item()
    nonzero_train = (out_train.abs() > 0).float().mean().item()
    # Eval should have more nonzero patches
    assert nonzero_eval > nonzero_train


def test_dropout_explicit_override():
    """Explicit training=True/False overrides module mode."""
    B, T, P = 2, 20, 16
    mod = PreconditioningChannel(degree=4, stride=16, patch_size=P, dropout=0.99)
    target, mask, sid, vid, tid = _make_inputs(B, T, P)

    # Module in eval mode, but force training=True
    mod.eval()
    torch.manual_seed(42)
    out_forced_train = mod(target, mask, sid, vid, tid, training=True)

    # Module in train mode, but force training=False
    mod.train()
    out_forced_eval = mod(target, mask, sid, vid, tid, training=False)

    nonzero_forced_train = (out_forced_train.abs() > 0).float().mean().item()
    nonzero_forced_eval = (out_forced_eval.abs() > 0).float().mean().item()
    assert nonzero_forced_eval > nonzero_forced_train


def test_observed_mask_bool_and_float():
    """Both bool and float masks should work identically."""
    B, T, P = 1, 20, 16
    mod = PreconditioningChannel(degree=4, stride=16, patch_size=P)
    target, mask_bool, sid, vid, tid = _make_inputs(B, T, P)

    mask_float = mask_bool.float()
    out_bool = mod(target, mask_bool, sid, vid, tid)
    out_float = mod(target, mask_float, sid, vid, tid)
    assert torch.allclose(out_bool, out_float)


def test_invalid_shape_raises():
    """Wrong ndim should raise ValueError."""
    mod = PreconditioningChannel(degree=4, stride=16, patch_size=16)
    # 2D target
    with pytest.raises(ValueError, match="target must have shape"):
        mod(torch.randn(2, 20), torch.ones(2, 20), torch.ones(2), torch.ones(2), torch.ones(2))


def test_future_values_do_not_affect_context():
    """Perturbing future patches should not change context hint values."""
    B, T, P = 1, 20, 16
    d = 4
    mod = PreconditioningChannel(degree=d, stride=P, patch_size=P)
    target, mask, sid, vid, tid = _make_inputs(B, T, P, seed=123)

    out_orig = mod(target, mask, sid, vid, tid)

    # Perturb last 5 patches
    target2 = target.clone()
    target2[:, -5:, :] = torch.randn(B, 5, P)
    out_pert = mod(target2, mask, sid, vid, tid)

    # Context region (patches d through T-5-d) should be unchanged
    ctx_end = T - 5 - d  # patches that cannot be affected by perturbed future
    if ctx_end > d:
        assert torch.allclose(out_orig[:, d:ctx_end, :], out_pert[:, d:ctx_end, :])


def test_numpy_matches_torch():
    """compute_residual (numpy) should match PreconditioningChannel for a simple case."""
    P = 16
    T = 10
    d = 4
    coeffs = chebyshev_coefficients(d)
    mod = PreconditioningChannel(degree=d, stride=P, poly_type="chebyshev", patch_size=P)

    # Create a simple single-sample, single-variate input
    np.random.seed(42)
    x_np = np.random.randn(T * P)
    x_torch = torch.from_numpy(x_np.reshape(1, T, P)).float()

    # NumPy reference
    residual_np = compute_residual(x_np, coeffs, P, full_lag_only=True)
    residual_np_patches = residual_np.reshape(1, T, P)

    # PyTorch module
    mask = torch.ones(1, T, P, dtype=torch.bool)
    sid = torch.ones(1, T, dtype=torch.long)
    vid = torch.ones(1, T, dtype=torch.long)
    tid = torch.arange(T, dtype=torch.long).unsqueeze(0)
    hint = mod(x_torch, mask, sid, vid, tid)

    np.testing.assert_allclose(hint.numpy(), residual_np_patches, atol=1e-5)


def test_backward_pass():
    """Gradients should flow through the module."""
    B, T, P = 2, 20, 16
    mod = PreconditioningChannel(degree=4, stride=16, patch_size=P)
    target = torch.randn(B, T, P, requires_grad=True)
    mask = torch.ones(B, T, P, dtype=torch.bool)
    sid = torch.ones(B, T, dtype=torch.long)
    vid = torch.ones(B, T, dtype=torch.long)
    tid = torch.arange(T, dtype=torch.long).unsqueeze(0).expand(B, -1)

    out = mod(target, mask, sid, vid, tid)
    loss = out.sum()
    loss.backward()
    assert target.grad is not None
    assert target.grad.shape == target.shape


def test_config_validation():
    """Invalid config values should raise ValueError."""
    with pytest.raises(ValueError, match="degree must be >= 1"):
        PreconditioningChannel(degree=0)
    with pytest.raises(ValueError, match="stride must be >= 1"):
        PreconditioningChannel(stride=0)
    with pytest.raises(ValueError, match="dropout must be in"):
        PreconditioningChannel(dropout=1.5)
    with pytest.raises(ValueError, match="dropout must be in"):
        PreconditioningChannel(dropout=-0.1)
