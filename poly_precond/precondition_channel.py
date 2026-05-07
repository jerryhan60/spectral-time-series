"""Polynomial preconditioning channel for patch-based time series transformers.

Computes the preconditioning residual r_t = sum_{k=1}^{d} c_k * x_{t-ks}
and returns it as a patch-aligned tensor ready to concatenate with
[target_patch, observation_mask] before the input projection. Respects
sample and variate boundaries in packed sequences.

sample_id=0 is treated as padding and excluded from residual computation,
following the Uni2TS packed-sequence convention.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from .chebyshev import (
    chebyshev_coefficients,
    differencing_coefficients,
    ema_coefficients,
    legendre_coefficients,
)

_COEFF_FN = {
    "chebyshev": chebyshev_coefficients,
    "legendre": legendre_coefficients,
    "ema": ema_coefficients,
    "differencing": differencing_coefficients,
}


class PreconditioningChannel(nn.Module):
    """Compute the polynomial hint channel for a batch of patched time series.

    Given a scaled target tensor of shape (B, T, P) -- batch, num_patches,
    patch_size -- this module:
      1. Flattens patches to raw time steps: (B, T*P).
      2. Applies the strided polynomial convolution respecting sample/variate boundaries.
      3. Extracts the residual (filtered - original).
      4. Re-patches to (B, T, P).
      5. Optionally applies per-patch dropout during training.

    sample_id=0 is treated as padding and excluded from residual computation,
    following the Uni2TS packed-sequence convention.

    Args:
        degree: Polynomial degree d.
        stride: Lag stride s (should equal patch_size P).
        poly_type: Polynomial basis -- "chebyshev", "legendre", "ema", or "differencing".
        dropout: Per-patch hint dropout probability during training.
        patch_size: Patch size P (used for re-patching).
    """

    def __init__(
        self,
        degree: int = 4,
        stride: int = 16,
        poly_type: str = "chebyshev",
        dropout: float = 0.0,
        patch_size: int = 16,
    ):
        super().__init__()
        if degree < 1:
            raise ValueError("degree must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in [0, 1]")
        if poly_type not in _COEFF_FN:
            raise ValueError(f"Unknown poly_type: {poly_type}. Choose from {list(_COEFF_FN)}")
        coeffs = _COEFF_FN[poly_type](degree)
        self.register_buffer("coeffs", torch.tensor(coeffs, dtype=torch.float32))
        self.stride = stride
        self.dropout = dropout
        self.patch_size = patch_size

    def forward(
        self,
        target: torch.Tensor,
        observed_mask: torch.Tensor,
        sample_id: torch.Tensor,
        variate_id: torch.Tensor,
        time_id: torch.Tensor,
        training: bool | None = None,
    ) -> torch.Tensor:
        """Compute the hint channel tensor.

        Args:
            target: Scaled target patches, shape (B, T, P).
            observed_mask: Observation mask, shape (B, T, P).
            sample_id: Sample indices for packed sequences, shape (B, T).
            variate_id: Variate indices, shape (B, T).
            time_id: Time-step indices (patch-level), shape (B, T).
            training: If True, apply hint dropout. If None (default),
                uses the module's training mode (self.training).

        Returns:
            Hint channel of shape (B, T, P), ready to concatenate with
            [target, observed_mask] along the last dimension.
        """
        if training is None:
            training = self.training

        # Shape validation
        if target.ndim != 3:
            raise ValueError(f"target must have shape (B, T, P), got {tuple(target.shape)}")
        if observed_mask.shape != target.shape:
            raise ValueError("observed_mask must have same shape as target")
        if sample_id.shape != target.shape[:2]:
            raise ValueError("sample_id must have shape (B, T)")
        if target.shape[-1] != self.patch_size:
            raise ValueError(f"target patch dim {target.shape[-1]} != patch_size {self.patch_size}")

        # Ensure bool mask
        observed_mask = observed_mask.bool()

        P = self.patch_size
        coeffs = self.coeffs.to(device=target.device, dtype=target.dtype)
        d = coeffs.numel()
        s = self.stride

        # Flatten patches to raw time steps
        flat = rearrange(target, "b t p -> b (t p)")
        flat_sid = sample_id.repeat_interleave(P, dim=1)
        flat_vid = variate_id.repeat_interleave(P, dim=1)
        offsets = torch.arange(P, device=time_id.device, dtype=time_id.dtype)
        offsets = offsets.repeat(time_id.shape[1]).unsqueeze(0).expand(time_id.shape[0], -1)
        flat_tid = time_id.repeat_interleave(P, dim=1) * P + offsets

        L = flat.shape[1]
        min_t = d * s
        if L <= min_t:
            hint = torch.zeros_like(target)
        else:
            result = flat.clone()
            base_mask = (flat_sid[:, min_t:] > 0) & (flat_tid[:, min_t:] >= min_t)
            valid_all = base_mask.clone()
            wsum = torch.zeros_like(flat[:, min_t:])
            for i in range(d):
                shift = (i + 1) * s
                lo = min_t - shift
                hi = L - shift
                prev = flat[:, lo:hi]
                valid_i = (
                    base_mask
                    & (flat_sid[:, min_t:] == flat_sid[:, lo:hi])
                    & (flat_vid[:, min_t:] == flat_vid[:, lo:hi])
                    & (flat_tid[:, min_t:] == flat_tid[:, lo:hi] + shift)
                )
                valid_all = valid_all & valid_i
                wsum = wsum + coeffs[i] * prev * valid_i
            result[:, min_t:] = torch.where(
                valid_all, flat[:, min_t:] + wsum, flat[:, min_t:]
            )
            preconditioned = rearrange(result, "b (t p) -> b t p", p=P)
            hint = preconditioned - target

        # Zero out unobserved positions
        hint = torch.where(observed_mask, hint, torch.zeros_like(hint))

        # Per-patch dropout during training
        if training and self.dropout > 0:
            drop_mask = torch.rand(hint.shape[:-1], device=hint.device) < self.dropout
            hint = hint.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        return hint
