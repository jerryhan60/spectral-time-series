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
STU Encoder Layer for MOIRAI.

This module provides STUEncoderLayer, which has the same interface as
TransformerEncoderLayer but uses spectral convolution instead of attention.

This allows drop-in replacement or hybrid architectures alternating between
attention and STU layers.
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


class STUEncoderLayer(nn.Module):
    """
    STU-based encoder layer matching TransformerEncoderLayer interface.

    This layer replaces self-attention with spectral convolution (STU),
    while maintaining the same pre/post-norm structure and FFN.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length for spectral filters
        num_eigh: Number of spectral filters (default: 24)
        use_hankel_L: Use single-branch Hankel-L formulation
        use_approx: Use approximation mode (50x fewer params)
        pre_norm: Use pre-normalization (default: True for MOIRAI)
        dropout_p: Dropout probability
        norm_layer: Normalization layer class
        activation: Activation function for FFN
        use_glu: Use gated linear unit in FFN
        d_ff: FFN hidden dimension (default: None = 4*d_model)

    Example:
        >>> layer = STUEncoderLayer(d_model=512, max_seq_len=2048)
        >>> x = torch.randn(2, 256, 512)
        >>> y = layer(x)  # Same interface as TransformerEncoderLayer
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
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

        # STU module (replaces self-attention)
        self.stu = PackedSTU(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_eigh=num_eigh,
            use_hankel_L=use_hankel_L,
            use_approx=use_approx,
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
            var_id: Variate IDs (unused in base STU, available for extensions)
            time_id: Time IDs (unused in STU, position is implicit in FFT)
            sample_id: Sample IDs for packed sequences
            centroid: Expert centroids for MoE (unused, kept for interface)

        Returns:
            Output tensor [*batch, time_len, dim]

        Note:
            attn_mask, var_id, time_id are kept for interface compatibility
            with TransformerEncoderLayer. STU handles position implicitly
            through spectral convolution and doesn't use variate-aware biases.
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


class SandwichedSTUEncoderLayer(nn.Module):
    """
    STU encoder layer with MLP sandwiching for increased expressiveness.

    Based on Flash STU paper: wraps STU with up/down projections allowing
    spectral convolution to operate in a higher-dimensional space.

    Architecture:
        Input -> Norm -> UpProject -> Activation -> STU -> DownProject -> Residual
                      -> Norm -> FFN -> Residual -> Output

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length for spectral filters
        sandwich_hidden_dim: Hidden dimension for sandwich MLP (default: 4*d_model)
        num_eigh: Number of spectral filters (default: 24)
        use_hankel_L: Use single-branch Hankel-L formulation
        use_approx: Use approximation mode (50x fewer params)
        pre_norm: Use pre-normalization (default: True)
        dropout_p: Dropout probability
        norm_layer: Normalization layer class
        activation: Activation function
        use_glu: Use gated linear unit in FFN
        d_ff: FFN hidden dimension

    Example:
        >>> layer = SandwichedSTUEncoderLayer(d_model=384, max_seq_len=512, sandwich_hidden_dim=1536)
        >>> x = torch.randn(2, 256, 384)
        >>> y = layer(x)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        sandwich_hidden_dim: Optional[int] = None,
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
        self.d_model = d_model

        # Sandwich hidden dimension (default: 4x d_model like typical FFN)
        self.sandwich_hidden_dim = sandwich_hidden_dim or (4 * d_model)

        # Sandwich MLP: up-project
        self.sandwich_up = nn.Linear(d_model, self.sandwich_hidden_dim, bias=False)
        self.sandwich_act = nn.SiLU()

        # STU operates in the higher-dimensional space
        self.stu = PackedSTU(
            d_model=self.sandwich_hidden_dim,  # STU in expanded space
            max_seq_len=max_seq_len,
            num_eigh=num_eigh,
            use_hankel_L=use_hankel_L,
            use_approx=use_approx,
        )

        # Sandwich MLP: down-project
        self.sandwich_down = nn.Linear(self.sandwich_hidden_dim, d_model, bias=False)

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
        """Forward pass with sandwiched STU."""
        if self.pre_norm:
            x = x + self._sandwiched_stu_block(self.norm1(x), sample_id)
            x = x + self.ffn(self.norm2(x), centroid=centroid)
        else:
            x = self.norm1(x + self._sandwiched_stu_block(x, sample_id))
            x = self.norm2(x + self.ffn(x, centroid=centroid))
        return x

    def _sandwiched_stu_block(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        sample_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        """Apply sandwiched STU: up-project -> activate -> STU -> down-project."""
        # Up-project to higher dimension
        h = self.sandwich_up(x)
        h = self.sandwich_act(h)

        # Apply STU in higher-dimensional space
        h = self.stu(h, sample_id=sample_id)

        # Down-project back to d_model
        h = self.sandwich_down(h)

        return self.dropout(h)


class VariateAwareSTUEncoderLayer(STUEncoderLayer):
    """
    STU encoder layer with variate-aware gating.

    This extension adds a gating mechanism that combines:
    1. Temporal branch: STU spectral convolution
    2. Variate branch: Cross-variate linear mixing

    The gate learns when to rely on temporal vs. cross-variate information.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        max_variates: Maximum number of variates (for variate embedding)
        num_eigh: Number of spectral filters
        **kwargs: Additional arguments passed to STUEncoderLayer
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        max_variates: int = 100,
        num_eigh: int = 24,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_eigh=num_eigh,
            **kwargs,
        )

        # Variate-aware components
        self.variate_embedding = nn.Embedding(max_variates, d_model)
        self.variate_proj = nn.Linear(d_model, d_model, bias=False)

        # Gating mechanism: learns to balance temporal vs. variate
        self.gate_proj = nn.Linear(d_model, 2, bias=False)

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
        Forward pass with variate-aware gating.

        If var_id is provided, uses gated combination of temporal (STU) and
        variate (learned embedding) pathways. Otherwise, falls back to standard STU.
        """
        if var_id is None:
            # No variate info: use standard STU
            return super().forward(x, attn_mask, var_id, time_id, sample_id, centroid)

        if self.pre_norm:
            x_norm = self.norm1(x)

            # Temporal branch: STU
            temporal = self._stu_block(x_norm, sample_id)

            # Variate branch: add variate embeddings and project
            var_emb = self.variate_embedding(var_id)
            variate = self.variate_proj(x_norm + var_emb)

            # Gated combination
            gate = F.softmax(self.gate_proj(x_norm), dim=-1)  # [*, L, 2]
            combined = gate[..., 0:1] * temporal + gate[..., 1:2] * variate

            x = x + combined
            x = x + self.ffn(self.norm2(x), centroid=centroid)
        else:
            # Post-norm version (analogous)
            temporal = self._stu_block(x, sample_id)
            var_emb = self.variate_embedding(var_id)
            variate = self.variate_proj(x + var_emb)
            gate = F.softmax(self.gate_proj(x), dim=-1)
            combined = gate[..., 0:1] * temporal + gate[..., 1:2] * variate
            x = self.norm1(x + combined)
            x = self.norm2(x + self.ffn(x, centroid=centroid))

        return x
