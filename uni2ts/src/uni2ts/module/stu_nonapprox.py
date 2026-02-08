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
Non-Approximated STU Encoder Layer for MOIRAI.

This module provides NonApproxSTUEncoderLayer, which uses full (non-approximated)
spectral projection matrices M_phi_plus [K, d_in, d_out] instead of the factorized
approximation (M_inputs [d,d] + M_filters [K,d]).

With num_eigh=2 and dual-branch (use_hankel_L=False):
  M_phi_plus [2, 384, 384] = 294,912 params
  M_phi_minus [2, 384, 384] = 294,912 params
  Total: 589,824 params

This nearly exactly matches multi-head attention's 589,964 mixing params,
making it a clean apples-to-apples comparison with the same parameter budget.
"""

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn

from .stu_adapter import PackedSTU
from .ffn import FeedForward, GatedLinearUnitFeedForward
from .norm import RMSNorm


class NonApproxSTUEncoderLayer(nn.Module):
    """
    STU encoder layer using full (non-approximated) spectral projections.

    With num_eigh=2 and dual-branch, this has ~590K mixing params,
    matching multi-head attention exactly. The default FFN (GLU with
    d_ff = 8/3 * d_model) adds ~1.18M params, giving a per-layer total
    of ~1.77M which matches the attention-based TransformerEncoderLayer.

    This layer has the same forward signature as STUEncoderLayer and
    TransformerEncoderLayer, allowing drop-in replacement in the
    HybridTransformerSTUEncoder.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length for spectral filter pre-computation
        num_eigh: Number of spectral filters (default: 2 for parameter matching)
        use_hankel_L: Use single-branch Hankel-L (False = dual-branch, default)
        pre_norm: Use pre-normalization (default: True for MOIRAI)
        dropout_p: Dropout probability
        norm_layer: Normalization layer class
        activation: Activation function for FFN
        use_glu: Use gated linear unit in FFN
        d_ff: FFN hidden dimension (default: None = GLU default)

    Example:
        >>> layer = NonApproxSTUEncoderLayer(d_model=384, max_seq_len=512)
        >>> x = torch.randn(2, 256, 384)
        >>> y = layer(x)  # Same interface as TransformerEncoderLayer
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_eigh: int = 2,
        use_hankel_L: bool = False,
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

        # STU module with full (non-approx) spectral projections
        self.stu = PackedSTU(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_eigh=num_eigh,
            use_hankel_L=use_hankel_L,
            use_approx=False,
        )

        # Feed-forward network (same as TransformerEncoderLayer)
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
            attn_mask: Attention mask (unused in STU, kept for interface compatibility)
            var_id: Variate IDs (unused, kept for interface compatibility)
            time_id: Time IDs (unused, position is implicit in FFT)
            sample_id: Sample IDs for packed sequences
            centroid: Expert centroids for MoE (unused, kept for interface)

        Returns:
            Output tensor [*batch, time_len, dim]

        Note:
            attn_mask, var_id, time_id are kept for interface compatibility
            with TransformerEncoderLayer. STU handles position implicitly
            through spectral convolution.
        """
        if self.pre_norm:
            # Pre-norm: norm -> STU -> residual -> norm -> FFN -> residual
            x = x + self._stu_block(self.norm1(x), sample_id)
            x = x + self.ffn(self.norm2(x), centroid=centroid)
        else:
            # Post-norm: STU -> residual -> norm -> FFN -> residual -> norm
            x = self.norm1(x + self._stu_block(x, sample_id))
            x = self.norm2(x + self.ffn(x, centroid=centroid))

        return x

    def _stu_block(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        sample_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        """Apply STU with dropout."""
        x = self.stu(x, sample_id=sample_id)
        return self.dropout(x)
