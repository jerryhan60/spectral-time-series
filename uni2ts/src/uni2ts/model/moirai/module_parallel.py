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
Parallel STU+Attention MOIRAI Module.

This module provides MoiraiParallelModule, which uses a ParallelSTUAttentionEncoder
where every layer runs both STU and Attention in parallel with a learned gate.
"""

from functools import partial
from typing import Optional

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
from uni2ts.module.parallel_encoder import ParallelSTUAttentionEncoder

from .module import encode_distr_output, decode_distr_output


class MoiraiParallelModule(
    nn.Module,
    PyTorchModelHubMixin,
    coders={DistributionOutput: (encode_distr_output, decode_distr_output)},
):
    """
    MOIRAI Module with Parallel STU+Attention Encoder.

    Every encoder layer runs both STU and Attention in parallel, combining
    outputs with a learned per-dimension gate. FFN dimension is reduced
    to match the baseline parameter budget.

    Args:
        distr_output: Distribution output object
        d_model: Model hidden dimension
        num_layers: Number of encoder layers
        patch_sizes: Sequence of patch sizes
        max_seq_len: Maximum sequence length
        attn_dropout_p: Attention dropout probability
        dropout_p: General dropout probability
        scaling: Whether to apply standardization
        num_eigh: Number of spectral filters for STU
        use_hankel_L: Use single-branch Hankel-L
        use_approx: Use approximation mode for STU
        parallel_d_ff: FFN hidden dimension (reduced for param matching)
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
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_approx: bool = True,
        parallel_d_ff: Optional[int] = None,
        # Accept and ignore hybrid-specific kwargs for config compatibility
        use_parallel_stu: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_sizes = patch_sizes
        self.max_seq_len = max_seq_len
        self.scaling = scaling

        # Mask encoding (learnable mask token)
        self.mask_encoding = nn.Embedding(num_embeddings=1, embedding_dim=d_model)

        # Scaler
        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()

        # Input projection (patches -> embeddings)
        self.in_proj = MultiInSizeLinear(
            in_features_ls=patch_sizes,
            out_features=d_model,
        )

        # Parallel STU+Attention encoder
        self.encoder = ParallelSTUAttentionEncoder(
            d_model=d_model,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            num_eigh=num_eigh,
            use_hankel_L=use_hankel_L,
            use_approx=use_approx,
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
            d_ff=parallel_d_ff,
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
        Forward pass of MoiraiParallelModule.

        Interface is identical to MoiraiModule / MoiraiHybridModule.
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

        # 4. Apply parallel encoder
        reprs = self.encoder(
            masked_reprs,
            attn_mask=packed_attention_mask(sample_id),
            var_id=variate_id,
            time_id=time_id,
            sample_id=sample_id,
        )

        # 5. Project to distribution parameters
        distr_param = self.param_proj(reprs, patch_size)

        # 6. Create distribution with affine transform
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)

        return distr

    def get_encoder_info(self) -> dict:
        """Get information about the encoder configuration."""
        return self.encoder.get_layer_info()
