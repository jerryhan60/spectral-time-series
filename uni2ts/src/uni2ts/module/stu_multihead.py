#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Multi-Head STU (Spectral Transform Unit) for MOIRAI.

Splits d_model into H heads, each with its own learned M_inputs and M_filters
projections but sharing the same spectral filters (phi). An output projection
W_out recombines the heads.

Parameter counts for d_model=384, H=6, head_dim=64, K=24:
    Per-head: M_inputs [64,64] + M_filters [24,64] = 5,632
    6 heads:  33,792
    W_out [384,384]: 147,456
    Total:    181,248
"""

from collections.abc import Callable
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int

from .spectral_filters import compute_spectral_filters, convolve_spectral
from .ffn import FeedForward, GatedLinearUnitFeedForward
from .norm import RMSNorm


class MultiHeadSTUCore(nn.Module):
    """
    Multi-head STU: splits d_model into H heads, each with own spectral filters.

    The spectral filters phi are shared across heads (they are data-independent
    eigenvectors of the Hankel matrix). Only M_inputs and M_filters are per-head.

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        max_seq_len: Maximum sequence length (for filter pre-computation)
        num_heads: Number of heads
        num_eigh: Number of spectral filters (K)
        use_hankel_L: Use single-branch Hankel-L vs dual-branch
        use_approx: Use approximation mode (recommended)
        dtype: Data type for parameters
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_heads: int = 6,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_approx: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        assert use_approx, "Multi-head STU only supports approx mode"

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.use_approx = use_approx

        # Shared spectral filters: phi [max_seq_len, num_eigh]
        phi = compute_spectral_filters(
            seq_len=max_seq_len,
            num_eigh=num_eigh,
            use_hankel_L=use_hankel_L,
            device=torch.device('cpu'),
            dtype=dtype,
        )
        self.register_buffer('phi', phi, persistent=False)

        # Per-head learnable projections
        # M_inputs: [H, head_dim, head_dim] - input projection per head
        # M_filters: [H, num_eigh, head_dim] - filter projection per head
        self.M_inputs = nn.Parameter(
            torch.empty(num_heads, self.head_dim, self.head_dim, dtype=dtype)
        )
        self.M_filters = nn.Parameter(
            torch.empty(num_heads, num_eigh, self.head_dim, dtype=dtype)
        )

        # Output projection: recombine heads
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self._init_params()

    def _init_params(self):
        """Initialize parameters with Xavier-like scaling."""
        std = 1.0 / math.sqrt(self.head_dim)
        nn.init.normal_(self.M_inputs, mean=0.0, std=std)
        nn.init.normal_(self.M_filters, mean=0.0, std=std)
        # out_proj is initialized by nn.Linear default (kaiming uniform)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """
        Apply multi-head spectral convolution.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        H = self.num_heads
        hd = self.head_dim

        # Truncate filters to actual sequence length
        phi = self.phi[:seq_len]  # [seq_len, K]

        return_both = not self.use_hankel_L

        # Split into heads: [batch, seq, d] -> [batch, seq, H, hd]
        x_heads = x.view(batch, seq_len, H, hd)

        # Per-head input projection using einsum:
        # x_heads [batch, seq, H, hd] @ M_inputs [H, hd, hd] -> [batch, seq, H, hd]
        x_proj = torch.einsum('bshd,hde->bshe', x_heads, self.M_inputs)

        # Per-head filter projection:
        # phi [seq, K] @ M_filters [H, K, hd] -> [H, seq, hd]
        phi_proj = torch.einsum('sk,hkd->hsd', phi, self.M_filters)

        # Convolve per head: convolve_spectral needs [batch, seq, d] and [seq, d]
        # Each head has its own phi_proj, so we loop over heads.
        output_heads = torch.zeros(
            batch, seq_len, H, hd, device=x.device, dtype=x.dtype,
        )

        for h in range(H):
            x_h = x_proj[:, :, h, :]  # [batch, seq, hd]
            phi_h = phi_proj[h]  # [seq, hd]

            spectral_plus, spectral_minus = convolve_spectral(
                x_h, phi_h, use_approx=True, return_both=return_both
            )

            if self.use_hankel_L:
                output_heads[:, :, h, :] = spectral_plus
            else:
                output_heads[:, :, h, :] = spectral_plus + spectral_minus

        # Concatenate heads: [batch, seq, H, hd] -> [batch, seq, d_model]
        output = output_heads.reshape(batch, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(output)

        return output

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"max_seq_len={self.max_seq_len}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"num_eigh={self.num_eigh}, "
            f"use_hankel_L={self.use_hankel_L}"
        )


class MultiHeadPackedSTU(nn.Module):
    """
    Multi-head STU that handles MOIRAI packed sequences.

    Wraps MultiHeadSTUCore with packed sequence handling using the same
    forward_batched pattern as PackedSTU in stu_adapter.py.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length per sample
        num_heads: Number of heads
        num_eigh: Number of spectral filters
        use_hankel_L: Use single-branch formulation
        use_approx: Use approximation mode (recommended)
        dtype: Data type for parameters
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_heads: int = 6,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_approx: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.stu_core = MultiHeadSTUCore(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            num_eigh=num_eigh,
            use_hankel_L=use_hankel_L,
            use_approx=use_approx,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "*batch seq_len d_model"],
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
    ) -> Float[torch.Tensor, "*batch seq_len d_model"]:
        """
        Apply multi-head STU with optional packed sequence handling.

        Args:
            x: Input tensor [*batch, seq_len, d_model]
            sample_id: Optional sample IDs for packed sequences [*batch, seq_len]

        Returns:
            Output tensor [*batch, seq_len, d_model]
        """
        if sample_id is None:
            return self.stu_core(x)

        return self.forward_batched(x, sample_id)

    def forward_batched(
        self,
        x: Float[torch.Tensor, "*batch seq_len d_model"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        max_samples: Optional[int] = None,
    ) -> Float[torch.Tensor, "*batch seq_len d_model"]:
        """
        Batched version for packed sequences.

        Pads all samples to same length for batched FFT, then unpads.

        Args:
            x: Input tensor
            sample_id: Sample IDs
            max_samples: Maximum expected samples (for pre-allocation)

        Returns:
            Output tensor
        """
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-2], x.shape[-1])
        sample_id_flat = sample_id.view(-1, sample_id.shape[-1])

        output = torch.zeros_like(x_flat)

        for batch_idx in range(x_flat.shape[0]):
            x_b = x_flat[batch_idx]
            sid_b = sample_id_flat[batch_idx]

            unique_samples = sid_b.unique()
            num_samples = len(unique_samples)

            # Collect sample lengths
            sample_xs = []
            sample_lengths = []
            for sid in unique_samples:
                mask = sid_b == sid
                sample_xs.append(x_b[mask])
                sample_lengths.append(mask.sum().item())

            # Pad to max length and batch
            max_len = max(sample_lengths)
            batched_x = torch.zeros(
                num_samples, max_len, x_b.shape[-1],
                device=x.device, dtype=x.dtype,
            )
            for i, sx in enumerate(sample_xs):
                batched_x[i, :len(sx)] = sx

            # Apply multi-head STU in batch
            batched_y = self.stu_core(batched_x)

            # Unpad and place back
            for i, (sid, length) in enumerate(zip(unique_samples, sample_lengths)):
                mask = sid_b == sid
                output[batch_idx, mask] = batched_y[i, :length]

        return output.view(original_shape)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_seq_len={self.max_seq_len}"


class MultiHeadSTUEncoderLayer(nn.Module):
    """
    Encoder layer using multi-head STU. Same interface as STUEncoderLayer.

    Uses MultiHeadPackedSTU instead of PackedSTU, with the same pre-norm + FFN
    structure. Accepts a d_ff parameter for wider FFN to match baseline param
    budgets.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length for spectral filters
        num_heads: Number of STU heads (default: 6)
        num_eigh: Number of spectral filters (default: 24)
        use_hankel_L: Use single-branch Hankel-L formulation
        use_approx: Use approximation mode (recommended)
        pre_norm: Use pre-normalization (default: True for MOIRAI)
        dropout_p: Dropout probability
        norm_layer: Normalization layer class
        activation: Activation function for FFN
        use_glu: Use gated linear unit in FFN
        d_ff: FFN hidden dimension (default: None = auto from GLU)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_heads: int = 6,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_approx: bool = True,
        pre_norm: bool = True,
        dropout_p: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = RMSNorm,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_glu: bool = True,
        d_ff: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.dropout_p = dropout_p

        # Multi-head STU module (replaces self-attention)
        self.stu = MultiHeadPackedSTU(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            num_eigh=num_eigh,
            use_hankel_L=use_hankel_L,
            use_approx=use_approx,
        )

        # Feed-forward network
        ffn_class = GatedLinearUnitFeedForward if use_glu else FeedForward
        self.ffn = ffn_class(
            in_dim=d_model,
            hidden_dim=d_ff,
            out_dim=None,
            activation=activation,
            bias=False,
            ffn_dropout_p=dropout_p,
        )

        # Normalization layers
        self.norm1 = norm_layer(d_model) if norm_layer else nn.Identity()
        self.norm2 = norm_layer(d_model) if norm_layer else nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]] = None,
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        sample_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        centroid: Optional[Float[torch.Tensor, "expert dim"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        """
        Forward pass matching TransformerEncoderLayer interface.

        Args:
            x: Input tensor [*batch, time_len, dim]
            attn_mask: Attention mask (unused, kept for interface compatibility)
            var_id: Variate IDs (unused, kept for interface compatibility)
            time_id: Time IDs (unused, kept for interface compatibility)
            sample_id: Sample IDs for packed sequences
            centroid: Expert centroids for MoE (unused, kept for interface)

        Returns:
            Output tensor [*batch, time_len, dim]
        """
        if self.pre_norm:
            x = x + self._stu_block(self.norm1(x), sample_id)
            x = x + self.ffn(self.norm2(x), centroid=centroid)
        else:
            x = self.norm1(x + self._stu_block(x, sample_id))
            x = self.norm2(x + self.ffn(x, centroid=centroid))

        return x

    def _stu_block(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        sample_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        """Apply multi-head STU with dropout."""
        x = self.stu(x, sample_id=sample_id)
        return self.dropout(x)
