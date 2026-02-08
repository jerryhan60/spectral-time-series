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
Parallel STU+Attention Encoder for MOIRAI.

This module provides ParallelSTUAttentionEncoder, where every layer runs
both STU and Attention in parallel with a learned gate to combine outputs.
"""

from collections.abc import Callable
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn

from .attention import GroupedQueryAttention
from .ffn import FeedForward, GatedLinearUnitFeedForward
from .norm import RMSNorm
from .position import AttentionBias, QueryKeyProjection
from .stu_parallel import ParallelSTUAttentionEncoderLayer


class ParallelSTUAttentionEncoder(nn.Module):
    """
    Encoder where every layer runs STU and Attention in parallel.

    Each layer contains both a GroupedQueryAttention module and a PackedSTU
    module. Their outputs are combined via a learned per-dimension gate.
    The FFN hidden dimension can be reduced to stay within a parameter budget.

    Args:
        d_model: Model dimension
        num_layers: Total number of layers
        max_seq_len: Maximum sequence length for STU filter computation
        num_eigh: Number of spectral filters for STU (default: 24)
        use_hankel_L: Use single-branch Hankel-L for STU
        use_approx: Use approximation mode for STU (recommended)
        num_heads: Number of attention heads (default: d_model // 64)
        num_groups: Number of groups for grouped-query attention
        pre_norm: Use pre-normalization (default: True)
        attn_dropout_p: Attention dropout probability
        dropout_p: General dropout probability
        norm_layer: Normalization layer class
        activation: Activation function for FFN
        use_glu: Use gated linear unit in FFN
        use_qk_norm: Use query-key normalization
        var_attn_bias_layer: Factory for variate attention bias
        time_attn_bias_layer: Factory for time attention bias
        var_qk_proj_layer: Factory for variate QK projection
        time_qk_proj_layer: Factory for time QK projection
        shared_var_attn_bias: Share variate bias across layers
        shared_time_attn_bias: Share time bias across layers
        shared_var_qk_proj: Share variate QK projection
        shared_time_qk_proj: Share time QK projection
        d_ff: FFN hidden dimension (reduced for param budget matching)
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        max_seq_len: int,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_approx: bool = True,
        num_heads: Optional[int] = None,
        num_groups: Optional[int] = None,
        pre_norm: bool = True,
        attn_dropout_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: Optional[Callable[[int], nn.Module]] = RMSNorm,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_glu: bool = True,
        use_qk_norm: bool = True,
        var_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]] = None,
        time_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]] = None,
        var_qk_proj_layer: Optional[
            Callable[[int, int, int], QueryKeyProjection]
        ] = None,
        time_qk_proj_layer: Optional[
            Callable[[int, int, int], QueryKeyProjection]
        ] = None,
        shared_var_attn_bias: bool = False,
        shared_time_attn_bias: bool = False,
        shared_var_qk_proj: bool = False,
        shared_time_qk_proj: bool = False,
        d_ff: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        num_heads = num_heads or d_model // 64
        num_groups = num_groups or num_heads

        # Setup shared attention components (same as TransformerEncoder)
        var_attn_bias = self._get_layer(
            d_model, num_heads, num_groups, var_attn_bias_layer, shared_var_attn_bias
        )
        time_attn_bias = self._get_layer(
            d_model, num_heads, num_groups, time_attn_bias_layer, shared_time_attn_bias
        )
        var_qk_proj = self._get_layer(
            d_model, num_heads, num_groups, var_qk_proj_layer, shared_var_qk_proj
        )
        time_qk_proj = self._get_layer(
            d_model, num_heads, num_groups, time_qk_proj_layer, shared_time_qk_proj
        )

        # Factory for attention modules
        get_self_attn = partial(
            GroupedQueryAttention,
            dim=d_model,
            num_heads=num_heads,
            num_groups=num_groups,
            bias=False,
            norm_layer=norm_layer if use_qk_norm else None,
            softmax_scale=None,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )

        # Build all layers as parallel STU+Attention
        self.layers = nn.ModuleList(
            [
                ParallelSTUAttentionEncoderLayer(
                    d_model=d_model,
                    max_seq_len=max_seq_len,
                    self_attn=get_self_attn(),
                    num_eigh=num_eigh,
                    use_hankel_L=use_hankel_L,
                    use_approx=use_approx,
                    pre_norm=pre_norm,
                    dropout_p=dropout_p,
                    norm_layer=norm_layer,
                    activation=activation,
                    use_glu=use_glu,
                    d_ff=d_ff,
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.norm = norm_layer(d_model) if norm_layer else nn.Identity()

    @staticmethod
    def _get_layer(
        dim: int,
        num_heads: int,
        num_groups: int,
        layer: Optional[Callable],
        shared_layer: bool,
    ) -> Optional[Callable[[], nn.Module]]:
        """Get layer factory, optionally shared across layers."""
        if layer is None:
            return None
        if shared_layer:
            module = layer(dim=dim, num_heads=num_heads, num_groups=num_groups)
            return lambda: module
        return partial(layer, dim=dim, num_heads=num_heads, num_groups=num_groups)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]] = None,
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        sample_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        """
        Forward pass through parallel STU+Attention encoder.

        Args:
            x: Input tensor [*batch, time_len, dim]
            attn_mask: Attention mask for attention branches
            var_id: Variate IDs for attention branches
            time_id: Time IDs for attention branches
            sample_id: Sample IDs for STU packed sequence handling

        Returns:
            Output tensor [*batch, time_len, dim]
        """
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                var_id=var_id,
                time_id=time_id,
                sample_id=sample_id,
            )

        return self.norm(x)

    def get_layer_info(self) -> dict:
        """Get information about layer arrangement."""
        return {
            "num_layers": self.num_layers,
            "pattern": "parallel",
            "layer_types": ["parallel"] * self.num_layers,
        }

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"num_layers={self.num_layers}, "
            f"max_seq_len={self.max_seq_len}, "
            f"pattern=parallel"
        )
