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

import ast
import os
from collections.abc import Sequence
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Bool, Float, Int
from torch import nn

from uni2ts.common.precondition import compute_polynomial_coefficients
from uni2ts.common.torch_util import packed_causal_attention_mask
from uni2ts.module.norm import RMSNorm
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedStdScaler
from uni2ts.module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from uni2ts.module.transformer import TransformerEncoder
from uni2ts.module.ts_embed import ResidualBlock


class Moirai2Module(
    nn.Module,
    PyTorchModelHubMixin,
):
    """
    Contains components of Moirai, to ensure implementation is identical across models.
    Subclasses huggingface_hub.PyTorchModelHubMixin to support loading from HuggingFace Hub.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_layers: int,
        patch_size: int,
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        scaling: bool = True,
        time_precondition_enabled: bool = False,
        time_precondition_type: str = "chebyshev",
        time_precondition_degree: int = 5,
        time_precondition_stride: int = 1,
        time_precondition_coeffs_init: Sequence[float] | str | None = None,
        time_precondition_inverse_enabled: bool = False,
        time_precondition_inverse_length: int = 64,
        time_precondition_inverse_stride: int = 1,
        latent_precondition_enabled: bool = False,
        latent_precondition_type: str = "chebyshev",
        latent_precondition_degree: int = 5,
        latent_precondition_stride: int = 1,
        num_predict_token: int = 1,
        quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ):
        """
        :param d_model: model hidden dimensions
        :param num_layers: number of transformer layers
        :param patch_size: patch size
        :param max_seq_len: maximum sequence length for inputs
        :param attn_dropout_p: dropout probability for attention layers
        :param dropout_p: dropout probability for all other layers
        :param scaling: whether to apply scaling (standardization)
        :param num_quantiles: number of quantile levels
        """
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_predict_token = num_predict_token
        self.max_seq_len = max_seq_len
        self.scaling = scaling
        self.quantile_levels = quantile_levels
        self.num_quantiles = len(quantile_levels)
        self.latent_precondition_enabled = latent_precondition_enabled
        self.latent_precondition_stride = latent_precondition_stride
        self.time_precondition_enabled = time_precondition_enabled
        self.time_precondition_stride = time_precondition_stride
        self.time_precondition_inverse_enabled = time_precondition_inverse_enabled
        self.time_precondition_inverse_stride = time_precondition_inverse_stride

        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj = ResidualBlock(
            input_dims=patch_size * 2,
            hidden_dims=d_model,
            output_dims=d_model,
        )
        if time_precondition_enabled:
            if time_precondition_stride < 1:
                raise ValueError("time_precondition_stride must be >= 1")
            coeffs_init = self._parse_time_precondition_coeffs(
                time_precondition_coeffs_init
            )
            if coeffs_init is None:
                coeffs = compute_polynomial_coefficients(
                    time_precondition_type, time_precondition_degree
                )
                coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32)
            else:
                coeffs_tensor = coeffs_init
            self.register_buffer(
                "time_precondition_coeffs",
                coeffs_tensor,
                persistent=False,
            )
        else:
            if time_precondition_coeffs_init not in (None, "", "null", "None"):
                raise ValueError(
                    "time_precondition_coeffs_init requires time_precondition_enabled=true"
                )
            self.register_buffer(
                "time_precondition_coeffs",
                torch.empty(0),
                persistent=False,
            )
        if time_precondition_inverse_enabled:
            if not time_precondition_enabled:
                raise ValueError(
                    "time_precondition_inverse_enabled requires time_precondition_enabled=true"
                )
            if time_precondition_inverse_length < 1:
                raise ValueError("time_precondition_inverse_length must be >= 1")
            if time_precondition_inverse_stride < 1:
                raise ValueError("time_precondition_inverse_stride must be >= 1")
            self.time_precondition_inverse_coeffs = nn.Parameter(
                torch.zeros(time_precondition_inverse_length, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "time_precondition_inverse_coeffs",
                torch.empty(0),
                persistent=False,
            )
        if latent_precondition_enabled:
            if latent_precondition_stride < 1:
                raise ValueError("latent_precondition_stride must be >= 1")
            coeffs = compute_polynomial_coefficients(
                latent_precondition_type, latent_precondition_degree
            )
            self.register_buffer(
                "latent_precondition_coeffs",
                torch.tensor(coeffs, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer(
                "latent_precondition_coeffs",
                torch.empty(0),
                persistent=False,
            )
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=None,
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
            d_ff=d_ff,
        )
        self.out_proj = ResidualBlock(
            input_dims=d_model,
            hidden_dims=d_model,
            output_dims=num_predict_token * self.num_quantiles * patch_size,
        )

    @staticmethod
    def _parse_time_precondition_coeffs(
        coeffs_init: Sequence[float] | str | None,
    ) -> Optional[torch.Tensor]:
        if coeffs_init is None:
            return None
        if isinstance(coeffs_init, str):
            text = coeffs_init.strip()
            if text in ("", "null", "None"):
                return None
            if os.path.exists(text):
                with open(text, "r", encoding="utf-8") as handle:
                    text = handle.read().strip()
            if text in ("", "null", "None"):
                return None
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = [float(x) for x in text.replace(",", " ").split() if x]
            if isinstance(parsed, (int, float)):
                parsed = [parsed]
            if not isinstance(parsed, Sequence):
                raise ValueError("time_precondition_coeffs_init must be a sequence")
            return torch.tensor(list(parsed), dtype=torch.float32).flatten()
        if isinstance(coeffs_init, Sequence):
            return torch.tensor(list(coeffs_init), dtype=torch.float32).flatten()
        raise ValueError("time_precondition_coeffs_init must be a sequence or string")

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        training_mode: Bool = True,
        input_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]] = None,
    ):
        """
        Defines the forward pass of MoiraiDecoderModule.
        This method expects processed inputs.

        1. Apply scaling to observations
        2. Project from observations to representations
        3. Replace prediction window with learnable mask
        4. Apply transformer layers
        5. Project from representations to distribution parameters
        6. Return distribution object

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
        :param training_mode: whether to use training mode (inference mode)
        :param input_mask: binary mask for corrupted input patches
        :return: predictive distribution
        """
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        if self.time_precondition_enabled:
            scaled_target = self._apply_time_precondition(
                scaled_target, sample_id, variate_id, time_id
            )
        # Zero unobserved slots so the model sees [0, False] — the same pattern
        # it learned for masked patches during training.  Without this, the
        # prediction window carries −loc/scale which is far out-of-distribution
        # and corrupts autoregressive forecasts.
        scaled_target = torch.where(observed_mask, scaled_target, 0.0)
        masked_target, input_observed_mask = self.apply_input_mask(
            scaled_target, observed_mask, input_mask
        )
        input_tokens = torch.cat(
            [masked_target, input_observed_mask.to(torch.float32)], dim=-1
        )
        reprs = self.in_proj(input_tokens)
        if self.latent_precondition_enabled:
            reprs = self._apply_latent_precondition(
                reprs, sample_id, variate_id, time_id
            )

        reprs = self.encoder(
            reprs,
            packed_causal_attention_mask(sample_id, time_id),
            time_id=time_id,
            var_id=variate_id,
        )
        if self.latent_precondition_enabled:
            reprs = self._reverse_latent_precondition(
                reprs, sample_id, variate_id, time_id
            )
        preds = self.out_proj(reprs)
        if training_mode:
            return preds, scaled_target
        else:
            return preds * scale + loc

    @staticmethod
    def apply_input_mask(
        scaled_target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        input_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]],
    ) -> tuple[
        Float[torch.Tensor, "*batch seq_len patch"],
        Bool[torch.Tensor, "*batch seq_len patch"],
    ]:
        if input_mask is None:
            return scaled_target, observed_mask
        mask = input_mask.unsqueeze(-1)
        masked_target = scaled_target.masked_fill(mask, 0.0)
        input_observed_mask = observed_mask & ~mask
        return masked_target, input_observed_mask

    def _flatten_timepoints(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch time_len"],
        Int[torch.Tensor, "*batch time_len"],
        Int[torch.Tensor, "*batch time_len"],
        Int[torch.Tensor, "*batch time_len"],
    ]:
        patch_size = int(self.patch_size)
        flat_target = rearrange(target, "b t p -> b (t p)")
        flat_sample_id = sample_id.repeat_interleave(patch_size, dim=1)
        flat_variate_id = variate_id.repeat_interleave(patch_size, dim=1)
        offsets = torch.arange(
            patch_size, device=time_id.device, dtype=time_id.dtype
        ).repeat(time_id.shape[1])
        offsets = offsets.unsqueeze(0).expand(time_id.shape[0], -1)
        flat_time_id = time_id.repeat_interleave(patch_size, dim=1) * patch_size + offsets
        return flat_target, flat_sample_id, flat_variate_id, flat_time_id

    def _apply_time_precondition(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len patch"]:
        if self.time_precondition_coeffs.numel() == 0:
            return target
        reshape_back = None
        orig_target = target
        if target.dim() == 4:
            batch, quantiles, seq_len, patch = target.shape
            reshape_back = (batch, quantiles, seq_len, patch)
            target = target.reshape(batch * quantiles, seq_len, patch)
            sample_id = sample_id.reshape(batch * quantiles, seq_len)
            variate_id = variate_id.reshape(batch * quantiles, seq_len)
            time_id = time_id.reshape(batch * quantiles, seq_len)
        coeffs = self.time_precondition_coeffs.to(
            device=target.device, dtype=target.dtype
        )
        n = int(coeffs.numel())
        stride = int(self.time_precondition_stride)
        if stride < 1:
            raise ValueError("time_precondition_stride must be >= 1")

        flat_target, flat_sample_id, flat_variate_id, flat_time_id = (
            self._flatten_timepoints(target, sample_id, variate_id, time_id)
        )
        seq_len = flat_target.shape[1]
        min_time = n * stride
        if seq_len <= min_time:
            return orig_target
        result = flat_target.clone()
        base_mask = (flat_sample_id[:, min_time:] > 0) & (
            flat_time_id[:, min_time:] >= min_time
        )
        valid_all = base_mask.clone()
        weighted_sum = torch.zeros_like(flat_target[:, min_time:])
        for i in range(n):
            shift = (i + 1) * stride
            left_idx = min_time - shift
            right_idx = seq_len - shift
            prev = flat_target[:, left_idx:right_idx]
            valid_i = (
                base_mask
                & (flat_sample_id[:, min_time:] == flat_sample_id[:, left_idx:right_idx])
                & (
                    flat_variate_id[:, min_time:]
                    == flat_variate_id[:, left_idx:right_idx]
                )
                & (
                    flat_time_id[:, min_time:]
                    == flat_time_id[:, left_idx:right_idx] + shift
                )
            )
            valid_all = valid_all & valid_i
            weighted_sum = weighted_sum + coeffs[i] * prev * valid_i
        result[:, min_time:] = torch.where(
            valid_all,
            flat_target[:, min_time:] + weighted_sum,
            flat_target[:, min_time:],
        )
        result = rearrange(result, "b (t p) -> b t p", p=int(self.patch_size))
        if reshape_back is not None:
            return result.reshape(reshape_back)
        return result

    def _apply_time_precondition_inverse_fir(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch seq_len patch"],
        Bool[torch.Tensor, "*batch seq_len patch"],
    ]:
        if self.time_precondition_inverse_coeffs.numel() == 0:
            zeros = torch.zeros_like(target)
            return zeros, torch.zeros_like(target, dtype=torch.bool)
        reshape_back = None
        orig_target = target
        if target.dim() == 4:
            batch, quantiles, seq_len, patch = target.shape
            reshape_back = (batch, quantiles, seq_len, patch)
            target = target.reshape(batch * quantiles, seq_len, patch)
            sample_id = sample_id.reshape(batch * quantiles, seq_len)
            variate_id = variate_id.reshape(batch * quantiles, seq_len)
            time_id = time_id.reshape(batch * quantiles, seq_len)
        coeffs = self.time_precondition_inverse_coeffs.to(
            device=target.device, dtype=target.dtype
        )
        k = int(coeffs.numel())
        stride = int(self.time_precondition_inverse_stride)
        if stride < 1:
            raise ValueError("time_precondition_inverse_stride must be >= 1")

        flat_target, flat_sample_id, flat_variate_id, flat_time_id = (
            self._flatten_timepoints(target, sample_id, variate_id, time_id)
        )
        seq_len = flat_target.shape[1]
        min_time = k * stride
        if seq_len <= min_time:
            zeros = torch.zeros_like(orig_target)
            return zeros, torch.zeros_like(orig_target, dtype=torch.bool)

        result = torch.zeros_like(flat_target)
        valid_mask = torch.zeros_like(flat_target, dtype=torch.bool)

        fast_unpacked = (
            (sample_id.max(dim=1).values <= 1).all()
            and (variate_id.max(dim=1).values <= 1).all()
        )
        if fast_unpacked:
            weighted_sum = torch.zeros_like(flat_target[:, min_time:])
            for i in range(k):
                shift = (i + 1) * stride
                weighted_sum = weighted_sum + coeffs[i] * flat_target[
                    :, min_time - shift : seq_len - shift
                ]
            mask = (flat_sample_id[:, min_time:] > 0) & (
                flat_time_id[:, min_time:] >= min_time
            )
            result[:, min_time:] = torch.where(mask, weighted_sum, result[:, min_time:])
            valid_mask[:, min_time:] = mask
            result = rearrange(result, "b (t p) -> b t p", p=int(self.patch_size))
            valid_mask = rearrange(
                valid_mask, "b (t p) -> b t p", p=int(self.patch_size)
            )
            if reshape_back is not None:
                return result.reshape(reshape_back), valid_mask.reshape(reshape_back)
            return result, valid_mask

        base_mask = (flat_sample_id[:, min_time:] > 0) & (
            flat_time_id[:, min_time:] >= min_time
        )
        valid_all = base_mask.clone()
        weighted_sum = torch.zeros_like(flat_target[:, min_time:])
        for i in range(k):
            shift = (i + 1) * stride
            left_idx = min_time - shift
            right_idx = seq_len - shift
            prev = flat_target[:, left_idx:right_idx]
            valid_i = (
                base_mask
                & (flat_sample_id[:, min_time:] == flat_sample_id[:, left_idx:right_idx])
                & (
                    flat_variate_id[:, min_time:]
                    == flat_variate_id[:, left_idx:right_idx]
                )
                & (
                    flat_time_id[:, min_time:]
                    == flat_time_id[:, left_idx:right_idx] + shift
                )
            )
            valid_all = valid_all & valid_i
            weighted_sum = weighted_sum + coeffs[i] * prev * valid_i
        result[:, min_time:] = torch.where(
            valid_all, weighted_sum, result[:, min_time:]
        )
        valid_mask[:, min_time:] = valid_all
        result = rearrange(result, "b (t p) -> b t p", p=int(self.patch_size))
        valid_mask = rearrange(
            valid_mask, "b (t p) -> b t p", p=int(self.patch_size)
        )
        if reshape_back is not None:
            return result.reshape(reshape_back), valid_mask.reshape(reshape_back)
        return result, valid_mask

    def _apply_latent_precondition(
        self,
        tokens: Float[torch.Tensor, "*batch seq_len dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len dim"]:
        if self.latent_precondition_coeffs.numel() == 0:
            return tokens
        coeffs = self.latent_precondition_coeffs.to(
            device=tokens.device, dtype=tokens.dtype
        )
        n = int(coeffs.numel())
        stride = int(self.latent_precondition_stride)
        if stride < 1:
            raise ValueError("latent_precondition_stride must be >= 1")
        seq_len = tokens.shape[1]
        min_time = n * stride
        if seq_len <= min_time:
            return tokens
        start = min_time
        result = tokens.clone()
        if (sample_id.max(dim=1).values <= 1).all():
            weighted_sum = torch.zeros_like(tokens[:, start:, :])
            for i in range(n):
                shift = (i + 1) * stride
                weighted_sum = weighted_sum + coeffs[i] * tokens[
                    :, start - shift : seq_len - shift, :
                ]
            mask = sample_id[:, start:] > 0
            result[:, start:, :] = torch.where(
                mask.unsqueeze(-1),
                tokens[:, start:, :] + weighted_sum,
                tokens[:, start:, :],
            )
            return result
        base_mask = (sample_id[:, start:] > 0) & (time_id[:, start:] >= min_time)
        valid_all = base_mask.clone()
        weighted_sum = torch.zeros_like(tokens[:, start:, :])
        for i in range(n):
            shift = (i + 1) * stride
            left_idx = start - shift
            right_idx = seq_len - shift
            prev = tokens[:, left_idx:right_idx, :]
            valid_i = (
                base_mask
                & (sample_id[:, start:] == sample_id[:, left_idx:right_idx])
                & (variate_id[:, start:] == variate_id[:, left_idx:right_idx])
                & (time_id[:, start:] == time_id[:, left_idx:right_idx] + shift)
            )
            valid_all = valid_all & valid_i
            weighted_sum = weighted_sum + coeffs[i] * prev * valid_i.unsqueeze(-1)
        result[:, start:, :] = torch.where(
            valid_all.unsqueeze(-1),
            tokens[:, start:, :] + weighted_sum,
            tokens[:, start:, :],
        )
        return result

    def _reverse_latent_precondition(
        self,
        tokens: Float[torch.Tensor, "*batch seq_len dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len dim"]:
        if self.latent_precondition_coeffs.numel() == 0:
            return tokens
        coeffs = self.latent_precondition_coeffs.to(
            device=tokens.device, dtype=tokens.dtype
        )
        n = int(coeffs.numel())
        stride = int(self.latent_precondition_stride)
        if stride < 1:
            raise ValueError("latent_precondition_stride must be >= 1")
        seq_len = tokens.shape[1]
        min_time = n * stride
        if seq_len <= min_time:
            return tokens
        output = tokens.clone()
        if (sample_id.max(dim=1).values <= 1).all():
            for t in range(min_time, seq_len):
                mask = sample_id[:, t] > 0
                if not mask.any():
                    continue
                weighted_sum = torch.zeros_like(tokens[:, t, :])
                for i in range(n):
                    shift = (i + 1) * stride
                    weighted_sum = weighted_sum + coeffs[i] * output[:, t - shift, :]
                output[:, t, :] = torch.where(
                    mask.unsqueeze(-1),
                    tokens[:, t, :] - weighted_sum,
                    tokens[:, t, :],
                )
            return output

        base_mask = (sample_id[:, min_time:] > 0) & (time_id[:, min_time:] >= min_time)
        valid_all = base_mask.clone()
        for i in range(n):
            shift = (i + 1) * stride
            valid_i = (
                base_mask
                & (
                    sample_id[:, min_time:]
                    == sample_id[:, min_time - shift : seq_len - shift]
                )
                & (
                    variate_id[:, min_time:]
                    == variate_id[:, min_time - shift : seq_len - shift]
                )
                & (
                    time_id[:, min_time:]
                    == time_id[:, min_time - shift : seq_len - shift] + shift
                )
            )
            valid_all = valid_all & valid_i
        if not valid_all.any():
            return output
        for t in range(min_time, seq_len):
            mask = valid_all[:, t - min_time]
            if not mask.any():
                continue
            weighted_sum = torch.zeros_like(tokens[:, t, :])
            for i in range(n):
                shift = (i + 1) * stride
                weighted_sum = weighted_sum + coeffs[i] * output[:, t - shift, :]
            output[:, t, :] = torch.where(
                mask.unsqueeze(-1),
                tokens[:, t, :] - weighted_sum,
                tokens[:, t, :],
            )
        return output
