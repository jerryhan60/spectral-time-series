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
Parallel STU+Attention Encoder Layer for MOIRAI.

This module provides ParallelSTUAttentionEncoderLayer, which runs both STU
(spectral convolution) and self-attention in parallel within the same layer,
combining their outputs via a learned per-dimension gate.

Architecture per layer:
    Input -> Norm1 -> [Attention(x) * sigmoid(gate) + STU(x) * (1-sigmoid(gate))] -> Residual
          -> Norm2 -> FFN -> Residual -> Output

The gate is a learnable parameter of shape [d_model] initialized to 0,
so sigmoid(0)=0.5 gives equal weighting initially.
"""

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn

from .attention import GroupedQueryAttention
from .stu_adapter import PackedSTU
from .ffn import FeedForward, GatedLinearUnitFeedForward
from .norm import RMSNorm


class ParallelSTUAttentionEncoderLayer(nn.Module):
    """
    Encoder layer that runs STU and Attention in parallel with learned gating.

    Both STU (spectral convolution) and self-attention operate on the same
    normalized input, and their outputs are combined via a learned gate:
        output = sigmoid(gate) * attention(x) + (1 - sigmoid(gate)) * stu(x)

    This allows the model to learn per-dimension which mechanism to emphasize.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length for spectral filters
        self_attn: GroupedQueryAttention instance (passed in from encoder)
        num_eigh: Number of spectral filters (default: 24)
        use_hankel_L: Use single-branch Hankel-L formulation
        use_approx: Use approximation mode (50x fewer params)
        pre_norm: Use pre-normalization (default: True for MOIRAI)
        dropout_p: Dropout probability
        norm_layer: Normalization layer class
        activation: Activation function for FFN
        use_glu: Use gated linear unit in FFN
        d_ff: FFN hidden dimension (default: None = auto)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        self_attn: GroupedQueryAttention,
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

        # Attention module (passed in, includes var_attn_bias etc.)
        self.self_attn = self_attn

        # STU module (spectral convolution)
        self.stu = PackedSTU(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_eigh=num_eigh,
            use_hankel_L=use_hankel_L,
            use_approx=use_approx,
        )

        # Learned gate: shape [d_model], initialized to 0 -> sigmoid(0)=0.5
        self.gate = nn.Parameter(torch.zeros(d_model))

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
        Forward pass with parallel STU + Attention and learned gating.

        Args:
            x: Input tensor [*batch, time_len, dim]
            attn_mask: Attention mask for attention branch
            var_id: Variate IDs for attention branch
            time_id: Time IDs for attention branch
            sample_id: Sample IDs for STU packed sequence handling
            centroid: Expert centroids for MoE (unused, kept for interface)

        Returns:
            Output tensor [*batch, time_len, dim]
        """
        if self.pre_norm:
            normed = self.norm1(x)
            # Run both branches on the same normalized input
            attn_out = self._sa_block(normed, attn_mask, var_id, time_id)
            stu_out = self._stu_block(normed, sample_id)
            # Gated combination
            g = torch.sigmoid(self.gate)
            mixed = g * attn_out + (1 - g) * stu_out
            x = x + self.dropout(mixed)
            x = x + self.ffn(self.norm2(x), centroid=centroid)
        else:
            attn_out = self._sa_block(x, attn_mask, var_id, time_id)
            stu_out = self._stu_block(x, sample_id)
            g = torch.sigmoid(self.gate)
            mixed = g * attn_out + (1 - g) * stu_out
            x = self.norm1(x + self.dropout(mixed))
            x = self.norm2(x + self.ffn(x, centroid=centroid))

        return x

    def _sa_block(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]],
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        """Apply self-attention (same calling convention as TransformerEncoderLayer)."""
        return self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            query_var_id=var_id,
            kv_var_id=var_id,
            query_time_id=time_id,
            kv_time_id=time_id,
        )

    def _stu_block(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        sample_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        """Apply STU spectral convolution."""
        return self.stu(x, sample_id=sample_id)
