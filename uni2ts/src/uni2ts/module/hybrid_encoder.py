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
Hybrid Transformer-STU Encoder for MOIRAI.

This module provides the HybridTransformerSTUEncoder that combines:
- STU layers for efficient long-range temporal processing (O(L log L))
- Attention layers for local patterns and cross-variate interactions (O(L²))

The hybrid design retains MOIRAI's any-variate attention capability while
gaining STU's efficiency for long sequences.
"""

from collections.abc import Callable
from functools import partial
from typing import Optional, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn

from .attention import GroupedQueryAttention
from .ffn import FeedForward, GatedLinearUnitFeedForward
from .norm import RMSNorm
from .position import AttentionBias, QueryKeyProjection
from .stu_layer import STUEncoderLayer, VariateAwareSTUEncoderLayer, SandwichedSTUEncoderLayer
from .transformer import TransformerEncoderLayer


class HybridTransformerSTUEncoder(nn.Module):
    """
    Hybrid encoder alternating between STU and Attention layers.

    This encoder replaces some transformer attention layers with STU layers
    to improve computational efficiency while preserving modeling capabilities:

    - STU layers: O(L log L) complexity, excellent for long-range temporal dependencies
    - Attention layers: O(L²) complexity, excellent for local patterns and cross-variate

    Args:
        d_model: Model dimension
        num_layers: Total number of layers
        max_seq_len: Maximum sequence length for STU filter computation
        num_eigh: Number of spectral filters for STU (default: 24)
        stu_layer_pattern: How to arrange STU vs attention layers:
            - "alternating": STU on even layers, attention on odd (default)
            - "first_half": STU for first half, attention for second half
            - "last_half": Attention for first half, STU for second half
            - "stu_only": All STU layers (no attention)
            - "attn_only": All attention layers (standard transformer)
        use_hankel_L: Use single-branch Hankel-L for STU (faster but less expressive)
        use_approx: Use approximation mode for STU (50x fewer params, recommended)
        use_variate_aware_stu: Use variate-aware gating in STU layers
        max_variates: Maximum variates for variate-aware STU
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
        d_ff: FFN hidden dimension

    Example:
        >>> encoder = HybridTransformerSTUEncoder(
        ...     d_model=512,
        ...     num_layers=12,
        ...     max_seq_len=2048,
        ...     stu_layer_pattern="alternating",
        ... )
        >>> x = torch.randn(2, 256, 512)
        >>> y = encoder(x)  # [2, 256, 512]
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        max_seq_len: int,
        num_eigh: int = 24,
        stu_layer_pattern: Literal[
            "alternating", "first_half", "last_half", "stu_only", "attn_only"
        ] = "alternating",
        use_hankel_L: bool = False,
        use_approx: bool = True,
        use_variate_aware_stu: bool = False,
        use_sandwiched_stu: bool = False,
        sandwich_hidden_dim: Optional[int] = None,
        max_variates: int = 100,
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
        self.stu_layer_pattern = stu_layer_pattern

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

        # Factory for attention layers
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

        # Factory for FFN
        get_ffn = partial(
            GatedLinearUnitFeedForward if use_glu else FeedForward,
            in_dim=d_model,
            hidden_dim=d_ff,
            out_dim=None,
            activation=activation,
            bias=False,
            ffn_dropout_p=dropout_p,
        )

        # Factory for layer norm
        get_norm = partial(norm_layer, d_model) if norm_layer else lambda: nn.Identity()

        # STU layer class selection
        if use_sandwiched_stu:
            stu_layer_class = SandwichedSTUEncoderLayer
        elif use_variate_aware_stu:
            stu_layer_class = VariateAwareSTUEncoderLayer
        else:
            stu_layer_class = STUEncoderLayer

        # Build layers
        self.layers = nn.ModuleList()
        self.layer_types = []  # Track which layers are STU vs attention

        for i in range(num_layers):
            use_stu = self._should_use_stu(i, num_layers, stu_layer_pattern)
            self.layer_types.append("stu" if use_stu else "attn")

            if use_stu:
                # STU layer
                stu_kwargs = dict(
                    d_model=d_model,
                    max_seq_len=max_seq_len,
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
                if use_variate_aware_stu:
                    stu_kwargs["max_variates"] = max_variates
                if use_sandwiched_stu:
                    stu_kwargs["sandwich_hidden_dim"] = sandwich_hidden_dim
                self.layers.append(stu_layer_class(**stu_kwargs))
            else:
                # Attention layer
                self.layers.append(
                    TransformerEncoderLayer(
                        self_attn=get_self_attn(),
                        ffn=get_ffn(),
                        norm1=get_norm(),
                        norm2=get_norm(),
                        pre_norm=pre_norm,
                        post_attn_dropout_p=dropout_p,
                    )
                )

        # Final normalization
        self.norm = norm_layer(d_model) if norm_layer else nn.Identity()

        # Count layers
        self.num_stu_layers = sum(1 for t in self.layer_types if t == "stu")
        self.num_attn_layers = num_layers - self.num_stu_layers

    @staticmethod
    def _should_use_stu(layer_idx: int, num_layers: int, pattern: str) -> bool:
        """Determine if a layer should be STU based on the pattern."""
        if pattern == "alternating":
            return layer_idx % 2 == 0
        elif pattern == "first_half":
            return layer_idx < num_layers // 2
        elif pattern == "last_half":
            return layer_idx >= num_layers // 2
        elif pattern == "stu_only":
            return True
        elif pattern == "attn_only":
            return False
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

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
        Forward pass through hybrid encoder.

        Args:
            x: Input tensor [*batch, time_len, dim]
            attn_mask: Attention mask for attention layers
            var_id: Variate IDs (used by attention layers and variate-aware STU)
            time_id: Time IDs (used by attention layers for position encoding)
            sample_id: Sample IDs for packed sequences (used by STU layers)

        Returns:
            Output tensor [*batch, time_len, dim]
        """
        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if layer_type == "stu":
                # STU layer: pass sample_id for packed sequence handling
                x = layer(x, var_id=var_id, sample_id=sample_id)
            else:
                # Attention layer: pass attention mask and position info
                x = layer(x, attn_mask=attn_mask, var_id=var_id, time_id=time_id)

        return self.norm(x)

    def get_layer_info(self) -> dict:
        """Get information about layer arrangement."""
        return {
            "num_layers": self.num_layers,
            "num_stu_layers": self.num_stu_layers,
            "num_attn_layers": self.num_attn_layers,
            "pattern": self.stu_layer_pattern,
            "layer_types": self.layer_types,
        }

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"num_layers={self.num_layers}, "
            f"max_seq_len={self.max_seq_len}, "
            f"pattern={self.stu_layer_pattern}, "
            f"stu_layers={self.num_stu_layers}, "
            f"attn_layers={self.num_attn_layers}"
        )


def create_moirai_hybrid_encoder(
    d_model: int,
    num_layers: int,
    max_seq_len: int,
    stu_layer_pattern: str = "alternating",
    num_eigh: int = 24,
    attn_dropout_p: float = 0.0,
    dropout_p: float = 0.0,
    var_attn_bias_layer: Optional[Callable] = None,
    time_qk_proj_layer: Optional[Callable] = None,
    shared_var_attn_bias: bool = False,
    shared_time_qk_proj: bool = True,
) -> HybridTransformerSTUEncoder:
    """
    Create a hybrid encoder with MOIRAI's default configuration.

    This is a convenience function that creates a HybridTransformerSTUEncoder
    with settings matching MOIRAI's TransformerEncoder defaults.

    Args:
        d_model: Model dimension
        num_layers: Number of layers
        max_seq_len: Maximum sequence length
        stu_layer_pattern: Layer pattern ("alternating", "first_half", etc.)
        num_eigh: Number of spectral filters
        attn_dropout_p: Attention dropout
        dropout_p: General dropout
        var_attn_bias_layer: Variate attention bias factory
        time_qk_proj_layer: Time QK projection factory (e.g., RotaryProjection)
        shared_var_attn_bias: Share variate bias
        shared_time_qk_proj: Share time projection

    Returns:
        HybridTransformerSTUEncoder configured for MOIRAI
    """
    return HybridTransformerSTUEncoder(
        d_model=d_model,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_eigh=num_eigh,
        stu_layer_pattern=stu_layer_pattern,
        use_hankel_L=False,
        use_approx=True,
        use_variate_aware_stu=False,
        num_heads=None,  # Auto: d_model // 64
        num_groups=None,  # Auto: same as num_heads
        pre_norm=True,
        attn_dropout_p=attn_dropout_p,
        dropout_p=dropout_p,
        norm_layer=RMSNorm,
        activation=F.silu,
        use_glu=True,
        use_qk_norm=True,
        var_attn_bias_layer=var_attn_bias_layer,
        time_attn_bias_layer=None,
        var_qk_proj_layer=None,
        time_qk_proj_layer=time_qk_proj_layer,
        shared_var_attn_bias=shared_var_attn_bias,
        shared_time_attn_bias=False,
        shared_var_qk_proj=False,
        shared_time_qk_proj=shared_time_qk_proj,
        d_ff=None,
    )
