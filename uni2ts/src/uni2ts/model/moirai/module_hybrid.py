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
Hybrid MOIRAI Module with STU Integration.

This module provides MoiraiHybridModule, which extends MoiraiModule with the
option to use a hybrid Transformer-STU encoder for improved efficiency on
long sequences.
"""

from functools import partial
from typing import Optional, Literal

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution

from uni2ts.common.torch_util import mask_fill, packed_attention_mask
from uni2ts.distribution import DistributionOutput
from uni2ts.module.norm import RMSNorm
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedStdScaler
from uni2ts.module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from uni2ts.module.ts_embed import MultiInSizeLinear
from uni2ts.module.hybrid_encoder import HybridTransformerSTUEncoder

from .module import encode_distr_output, decode_distr_output


class MoiraiHybridModule(
    nn.Module,
    PyTorchModelHubMixin,
    coders={DistributionOutput: (encode_distr_output, decode_distr_output)},
):
    """
    MOIRAI Module with Hybrid Transformer-STU Encoder.

    This extends MoiraiModule with the option to use a hybrid encoder that
    alternates between STU layers (for efficient long-range temporal processing)
    and attention layers (for cross-variate interactions).

    Args:
        distr_output: Distribution output object
        d_model: Model hidden dimension
        num_layers: Number of encoder layers
        patch_sizes: Sequence of patch sizes
        max_seq_len: Maximum sequence length
        attn_dropout_p: Attention dropout probability
        dropout_p: General dropout probability
        scaling: Whether to apply standardization
        stu_layer_pattern: How to arrange STU vs attention layers:
            - "alternating": STU on even layers, attention on odd (default)
            - "first_half": STU for first half, attention for second half
            - "last_half": Attention for first half, STU for second half
            - "stu_only": All STU layers (no attention)
            - "attn_only": All attention layers (standard transformer)
        num_eigh: Number of spectral filters for STU
        use_hankel_L: Use single-branch Hankel-L (faster)
        use_approx: Use approximation mode for STU (recommended)
        use_variate_aware_stu: Use variate-aware gating in STU layers

    Example:
        >>> module = MoiraiHybridModule(
        ...     distr_output=StudentTOutput(),
        ...     d_model=512,
        ...     num_layers=12,
        ...     patch_sizes=(8, 16, 32, 64, 128),
        ...     max_seq_len=2048,
        ...     stu_layer_pattern="alternating",
        ... )
    """

    def __init__(
        self,
        distr_output: DistributionOutput,
        d_model: int,
        num_layers: int,
        patch_sizes: tuple[int, ...],
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
        # STU-specific parameters
        stu_layer_pattern: Literal[
            "alternating", "first_half", "last_half", "stu_only", "attn_only"
        ] = "alternating",
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_approx: bool = True,
        use_variate_aware_stu: bool = False,
        max_variates: int = 100,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.max_seq_len = max_seq_len
        self.scaling = scaling
        self.stu_layer_pattern = stu_layer_pattern

        # Mask encoding (learnable mask token)
        self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)

        # Scaler
        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()

        # Input projection (patches -> embeddings)
        self.in_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )

        # Hybrid encoder
        self.encoder = HybridTransformerSTUEncoder(
            d_model=d_model,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            num_eigh=num_eigh,
            stu_layer_pattern=stu_layer_pattern,
            use_hankel_L=use_hankel_L,
            use_approx=use_approx,
            use_variate_aware_stu=use_variate_aware_stu,
            max_variates=max_variates,
            num_heads=None,  # Auto: d_model // 64
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=partial(BinaryAttentionBias),
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=None,
        )

        # Distribution output
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        """
        Forward pass of MoiraiHybridModule.

        The interface is identical to MoiraiModule, but uses the hybrid encoder
        internally for improved efficiency.

        Args:
            target: Input data [*batch, seq_len, max_patch]
            observed_mask: Missing value mask (1=observed)
            sample_id: Sample indices for packing
            time_id: Time indices
            variate_id: Variate indices
            prediction_mask: Prediction horizon mask (1=predict)
            patch_size: Patch size for each token

        Returns:
            Predictive distribution
        """
        # 1. Scale inputs
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale

        # 2. Project patches to embeddings
        reprs = self.in_proj(scaled_target, patch_size)

        # 3. Replace prediction horizon with learnable mask
        masked_reprs = mask_fill(reprs, prediction_mask, self.mask_encoding.weight)

        # 4. Apply hybrid encoder
        # Key difference: pass sample_id for STU layers
        reprs = self.encoder(
            masked_reprs,
            attn_mask=packed_attention_mask(sample_id),
            var_id=variate_id,
            time_id=time_id,
            sample_id=sample_id,  # For STU packed sequence handling
        )

        # 5. Project to distribution parameters
        distr_param = self.param_proj(reprs, patch_size)

        # 6. Create distribution with affine transform
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)

        return distr

    def get_encoder_info(self) -> dict:
        """Get information about the hybrid encoder configuration."""
        return self.encoder.get_layer_info()


def convert_moirai_to_hybrid(
    original_module,
    stu_layer_pattern: str = "alternating",
    num_eigh: int = 24,
    copy_weights: bool = True,
) -> MoiraiHybridModule:
    """
    Convert a standard MoiraiModule to MoiraiHybridModule.

    This utility function creates a hybrid module with the same configuration
    as the original, optionally copying compatible weights.

    Args:
        original_module: Original MoiraiModule instance
        stu_layer_pattern: STU layer arrangement pattern
        num_eigh: Number of spectral filters
        copy_weights: Whether to copy attention weights for attention layers

    Returns:
        MoiraiHybridModule with hybrid encoder

    Note:
        STU layers have different architectures, so their weights are
        initialized randomly. Attention layers can optionally copy weights
        from the original model.
    """
    hybrid_module = MoiraiHybridModule(
        distr_output=original_module.distr_output,
        d_model=original_module.d_model,
        num_layers=original_module.num_layers,
        patch_sizes=original_module.patch_sizes,
        max_seq_len=original_module.max_seq_len,
        attn_dropout_p=0.0,  # Will be overwritten if copying weights
        dropout_p=0.0,
        scaling=original_module.scaling,
        stu_layer_pattern=stu_layer_pattern,
        num_eigh=num_eigh,
    )

    if copy_weights:
        # Copy shared weights
        hybrid_module.mask_encoding.load_state_dict(
            original_module.mask_encoding.state_dict()
        )
        hybrid_module.in_proj.load_state_dict(
            original_module.in_proj.state_dict()
        )
        hybrid_module.param_proj.load_state_dict(
            original_module.param_proj.state_dict()
        )

        # Copy attention layer weights where applicable
        # This is a partial copy since STU layers have different architecture
        for i, layer_type in enumerate(hybrid_module.encoder.layer_types):
            if layer_type == "attn" and i < len(original_module.encoder.layers):
                try:
                    hybrid_module.encoder.layers[i].load_state_dict(
                        original_module.encoder.layers[i].state_dict()
                    )
                except Exception:
                    # Shape mismatch possible due to different configurations
                    pass

        # Copy final norm
        try:
            hybrid_module.encoder.norm.load_state_dict(
                original_module.encoder.norm.state_dict()
            )
        except Exception:
            pass

    return hybrid_module
