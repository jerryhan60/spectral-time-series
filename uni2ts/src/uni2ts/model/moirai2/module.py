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
from uni2ts.module.packed_scaler import PackedNOPScaler, PackedRobustScaler, PackedStdScaler
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
        scaler_type: str = "std",
        time_precondition_enabled: bool = False,
        time_precondition_type: str = "chebyshev",
        time_precondition_degree: int = 5,
        time_precondition_stride: int = 1,
        time_precondition_reg_lambda: float = 1.0,
        time_precondition_coeffs_init: Sequence[float] | str | None = None,
        time_precondition_learnable: bool = False,
        time_precondition_inverse_enabled: bool = False,
        time_precondition_inverse_length: int = 64,
        time_precondition_inverse_stride: int = 1,
        latent_precondition_enabled: bool = False,
        latent_precondition_type: str = "chebyshev",
        latent_precondition_degree: int = 5,
        latent_precondition_stride: int = 1,
        time_precondition_hint_mode: bool = False,
        time_precondition_dual_head: bool = False,
        attn_l1_lambda: float = 0.0,
        num_predict_token: int = 1,
        quantile_levels: tuple[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        patch_mask_ratio: float = 0.0,
        hint_dropout: float = 0.0,
        hint_embed_mode: str = "concat",
        hint_normalize: bool = False,
        time_precondition_extra_hints: str | None = None,
        stu_enabled: bool = False,
        stu_num_filters: int = 24,
        stu_gate_init: float = 0.0,
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
        self.patch_mask_ratio = patch_mask_ratio
        self.attn_l1_lambda = attn_l1_lambda
        self.hint_dropout = hint_dropout
        self.hint_embed_mode = hint_embed_mode
        self.hint_normalize = hint_normalize
        self.latent_precondition_enabled = latent_precondition_enabled
        self.latent_precondition_stride = latent_precondition_stride
        self.time_precondition_enabled = time_precondition_enabled
        self.time_precondition_stride = time_precondition_stride
        self.time_precondition_inverse_enabled = time_precondition_inverse_enabled
        self.time_precondition_inverse_stride = time_precondition_inverse_stride

        self.time_precondition_hint_mode = time_precondition_hint_mode
        self.time_precondition_dual_head = time_precondition_dual_head

        if not scaling:
            self.scaler = PackedNOPScaler()
        elif scaler_type == "robust":
            self.scaler = PackedRobustScaler()
        elif scaler_type == "std":
            self.scaler = PackedStdScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type!r}. Use 'std' or 'robust'.")
        # Parse extra hint scales (e.g. "6:16" or "2:16|6:16" or "l2_optimized:6:16")
        self._extra_hint_configs = []  # list of (degree, stride, poly_type)
        if time_precondition_extra_hints:
            # Support pipe, semicolon, or comma separators
            for ch in ("|", ";", ","):
                if ch in time_precondition_extra_hints:
                    sep = ch
                    break
            else:
                sep = ","
            for part in time_precondition_extra_hints.split(sep):
                part = part.strip()
                if not part:
                    continue
                tokens = part.split(":")
                if len(tokens) == 2:
                    # "degree:stride" — use primary polynomial type
                    self._extra_hint_configs.append(
                        (int(tokens[0]), int(tokens[1]), time_precondition_type)
                    )
                elif len(tokens) == 3:
                    # "poly_type:degree:stride" — explicit polynomial type
                    self._extra_hint_configs.append(
                        (int(tokens[1]), int(tokens[2]), tokens[0])
                    )
                else:
                    raise ValueError(
                        f"Invalid extra hint spec '{part}'. "
                        "Expected 'degree:stride' or 'poly_type:degree:stride'"
                    )
        in_proj_dims = patch_size * 2
        _hint_active = time_precondition_hint_mode and time_precondition_enabled
        num_hint_channels = 1 + len(self._extra_hint_configs) if _hint_active else 0
        if _hint_active and hint_embed_mode == "concat":
            in_proj_dims = patch_size * (2 + num_hint_channels)  # [target, mask, hint1, hint2, ...]
        self.in_proj = ResidualBlock(
            input_dims=in_proj_dims,
            hidden_dims=d_model,
            output_dims=d_model,
        )
        if _hint_active and hint_embed_mode == "separate":
            self.hint_proj = nn.Linear(patch_size * num_hint_channels, d_model)
            self.hint_gate = nn.Linear(d_model, 1)
        if time_precondition_enabled:
            if time_precondition_stride < 1:
                raise ValueError("time_precondition_stride must be >= 1")
            coeffs_init = self._parse_time_precondition_coeffs(
                time_precondition_coeffs_init
            )
            if coeffs_init is None:
                coeffs = compute_polynomial_coefficients(
                    time_precondition_type,
                    time_precondition_degree,
                    reg_lambda=time_precondition_reg_lambda,
                )
                coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32)
            else:
                coeffs_tensor = coeffs_init
            if time_precondition_learnable:
                self.time_precondition_coeffs = nn.Parameter(coeffs_tensor)
            else:
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
        # Register extra hint coefficient buffers
        for idx, (extra_deg, extra_stride, extra_poly_type) in enumerate(self._extra_hint_configs):
            extra_coeffs = compute_polynomial_coefficients(
                extra_poly_type, extra_deg
            )
            self.register_buffer(
                f"_extra_hint_coeffs_{idx}",
                torch.tensor(extra_coeffs, dtype=torch.float32),
                persistent=False,
            )
            # Store stride as int attribute
            setattr(self, f"_extra_hint_stride_{idx}", extra_stride)
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
            stu_enabled=stu_enabled,
            stu_num_filters=stu_num_filters,
            stu_max_seq_len=max_seq_len,
            stu_gate_init=stu_gate_init,
        )
        self.out_proj = ResidualBlock(
            input_dims=d_model,
            hidden_dims=d_model,
            output_dims=num_predict_token * self.num_quantiles * patch_size,
        )
        if time_precondition_dual_head and time_precondition_enabled:
            self.out_proj_raw = ResidualBlock(
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
        # Save raw scaled target before preconditioning for inference reversal
        scaled_target_raw = scaled_target

        # --- Hint mode: compute FIR residual as input feature, don't precondition ---
        _hint_separate = None  # hint tensor for "separate" embed mode
        if self.time_precondition_hint_mode and self.time_precondition_enabled:
            precond_target = self._apply_time_precondition(
                scaled_target, sample_id, variate_id, time_id
            )
            precond_hint = precond_target - scaled_target
            # Collect all hint channels (primary + extras)
            hint_channels = [precond_hint]
            for idx in range(len(self._extra_hint_configs)):
                extra_coeffs = getattr(self, f"_extra_hint_coeffs_{idx}")
                extra_stride = getattr(self, f"_extra_hint_stride_{idx}")
                extra_precond = self._apply_time_precondition(
                    scaled_target, sample_id, variate_id, time_id,
                    coeffs_override=extra_coeffs,
                    stride_override=extra_stride,
                )
                hint_channels.append(extra_precond - scaled_target)
            # Zero unobserved slots
            scaled_target_zeroed = torch.where(observed_mask, scaled_target, 0.0)
            hint_channels = [torch.where(observed_mask, h, 0.0) for h in hint_channels]
            masked_target, input_observed_mask = self.apply_input_mask(
                scaled_target_zeroed, observed_mask, input_mask
            )
            # Also mask hints for prediction window
            if input_mask is not None:
                hint_channels = [h.masked_fill(input_mask.unsqueeze(-1), 0.0) for h in hint_channels]
            # Per-sequence hint normalization
            if self.hint_normalize:
                hint_channels = [
                    h / h.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
                    for h in hint_channels
                ]
            # Hint dropout: randomly zero entire hint channel per patch
            if training_mode and self.hint_dropout > 0:
                for i in range(len(hint_channels)):
                    hint_mask = torch.rand(
                        hint_channels[i].shape[:-1], device=hint_channels[i].device
                    ) < self.hint_dropout
                    hint_channels[i] = hint_channels[i].masked_fill(hint_mask.unsqueeze(-1), 0.0)
            all_hints = torch.cat(hint_channels, dim=-1)  # (B, T, num_hints * patch_size)
            if self.hint_embed_mode == "separate":
                # Separate pathway: main in_proj sees [target, mask] only
                input_tokens = torch.cat(
                    [masked_target, input_observed_mask.to(torch.float32)],
                    dim=-1,
                )
                _hint_separate = all_hints
            else:
                # Default concat: in_proj sees [target, mask, hint1, hint2, ...]
                input_tokens = torch.cat(
                    [masked_target, input_observed_mask.to(torch.float32), all_hints],
                    dim=-1,
                )
        else:
            # Standard path: optionally precondition the target
            if self.time_precondition_enabled:
                scaled_target = self._apply_time_precondition(
                    scaled_target, sample_id, variate_id, time_id
                )
            # Zero unobserved slots so the model sees [0, False]
            scaled_target = torch.where(observed_mask, scaled_target, 0.0)
            masked_target, input_observed_mask = self.apply_input_mask(
                scaled_target, observed_mask, input_mask
            )
            input_tokens = torch.cat(
                [masked_target, input_observed_mask.to(torch.float32)], dim=-1
            )

        reprs = self.in_proj(input_tokens)
        # Separate hint embedding: project hint independently, gate, and add
        if _hint_separate is not None:
            hint_embed = self.hint_proj(_hint_separate)  # (B, T, d_model)
            gate = torch.sigmoid(self.hint_gate(reprs))  # (B, T, 1)
            reprs = reprs + gate * hint_embed
        if training_mode and self.patch_mask_ratio > 0:
            mask = torch.rand(
                reprs.shape[:-1], device=reprs.device
            ) < self.patch_mask_ratio
            reprs = reprs.masked_fill(mask.unsqueeze(-1), 0.0)
        if self.latent_precondition_enabled:
            reprs = self._apply_latent_precondition(
                reprs, sample_id, variate_id, time_id
            )

        _collect_attn = training_mode and self.attn_l1_lambda > 0
        encoder_result = self.encoder(
            reprs,
            packed_causal_attention_mask(sample_id, time_id),
            time_id=time_id,
            var_id=variate_id,
            return_attn_weights=_collect_attn,
        )
        if _collect_attn:
            reprs, attn_weights_list = encoder_result
        else:
            reprs = encoder_result
            attn_weights_list = None
        if self.latent_precondition_enabled:
            reprs = self._reverse_latent_precondition(
                reprs, sample_id, variate_id, time_id
            )

        # Store attention weights for L1 regularization (accessed by pretrain.py)
        self._last_attn_weights = attn_weights_list

        preds = self.out_proj(reprs)

        # --- Dual-head mode: second head for raw-space predictions ---
        if self.time_precondition_dual_head and self.time_precondition_enabled:
            preds_raw = self.out_proj_raw(reprs)
            if training_mode:
                # Return both heads' predictions and both targets
                return preds, preds_raw, scaled_target, scaled_target_raw
            else:
                # At inference, use raw head — no reversal needed
                return preds_raw * scale + loc

        # --- Hint mode: predict in raw space, no reversal ---
        if self.time_precondition_hint_mode and self.time_precondition_enabled:
            if training_mode:
                return preds, scaled_target_raw  # raw targets
            else:
                return preds * scale + loc  # no reversal

        # --- Standard mode ---
        if training_mode:
            return preds, scaled_target
        else:
            if self.time_precondition_enabled:
                preds = self._reverse_time_precondition_preds(
                    preds, scaled_target_raw, prediction_mask,
                    sample_id, variate_id, time_id,
                )
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
        coeffs_override: Optional[torch.Tensor] = None,
        stride_override: Optional[int] = None,
    ) -> Float[torch.Tensor, "*batch seq_len patch"]:
        use_coeffs = coeffs_override if coeffs_override is not None else self.time_precondition_coeffs
        use_stride = stride_override if stride_override is not None else int(self.time_precondition_stride)
        if use_coeffs.numel() == 0:
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
        coeffs = use_coeffs.to(
            device=target.device, dtype=target.dtype
        )
        n = int(coeffs.numel())
        stride = int(use_stride)
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

    def _reverse_time_precondition_preds(
        self,
        preds: Float[torch.Tensor, "*batch seq_len pred_dim"],
        scaled_target_raw: Float[torch.Tensor, "*batch seq_len patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len pred_dim"]:
        """Reverse time preconditioning on model predictions at inference.

        The model predicts in preconditioned space. This method reverses
        the preconditioning so that ``preds * scale + loc`` recovers
        values in the original data space.

        Uses the median quantile to propagate history for sequential
        reversal, applying the same correction to all quantiles.
        """
        if self.time_precondition_coeffs.numel() == 0:
            return preds

        coeffs = self.time_precondition_coeffs.to(
            device=preds.device, dtype=preds.dtype
        )
        n = int(coeffs.numel())
        stride = int(self.time_precondition_stride)
        ps = int(self.patch_size)
        npt = int(self.num_predict_token)
        nq = int(self.num_quantiles)

        # Flatten arbitrary leading batch dims into one
        batch_shape = preds.shape[:-2]
        S = preds.shape[-2]
        B = 1
        for s in batch_shape:
            B *= s
        preds_2d = preds.reshape(B, S, -1)
        raw_2d = scaled_target_raw.reshape(B, S, ps)
        pmask_2d = prediction_mask.reshape(B, S)

        # Reshape preds: (B, S, npt*nq*ps) -> (B, S, npt, nq, ps)
        preds_r = rearrange(
            preds_2d,
            "b s (npt nq ps) -> b s npt nq ps",
            npt=npt, nq=nq, ps=ps,
        ).clone()

        # Median quantile index
        levels = torch.tensor(self.quantile_levels, dtype=torch.float32)
        median_idx = int(torch.argmin(torch.abs(levels - 0.5)).item())

        # Process each horizon (h=1 first so its reversed values seed h=2, etc.)
        for h in range(1, npt + 1):
            if S <= h:
                continue

            # pred_h[:, t, q, :] is the preconditioned prediction for patch t+h
            # Shape: (B, S-h, nq, ps)
            pred_h = preds_r[:, :-h, h - 1, :, :]

            # Flatten to per-timepoint: (B, (S-h)*ps, nq)
            pred_h_flat = rearrange(pred_h, "b s q p -> b (s p) q")

            # Raw context flattened: (B, S*ps)
            raw_flat = rearrange(raw_2d, "b t p -> b (t p)")

            # Prediction mask flattened
            pmask_flat = pmask_2d.repeat_interleave(ps, dim=1)

            T_raw = raw_flat.shape[1]
            T_pred = pred_h_flat.shape[1]
            offset = h * ps
            min_lag = n * stride

            # Working copy of raw history (updated with reversed predictions)
            raw_history = raw_flat.clone()

            # Output
            reversed_flat = pred_h_flat.clone()

            start = max(offset, min_lag)

            # Skip directly to the first prediction timepoint to avoid
            # iterating over the entire context window (~4000 timepoints).
            any_pred = pmask_flat.any(dim=0)
            if any_pred[start:].any():
                first_pred_tp = int(
                    torch.nonzero(any_pred[start:], as_tuple=False)[0].item()
                ) + start
                start = max(start, first_pred_tp)
            else:
                continue  # No prediction timepoints for this horizon

            for tp in range(start, T_raw):
                src = tp - offset
                if src < 0 or src >= T_pred:
                    continue

                # Batch mask: only reverse at prediction positions
                if tp >= pmask_flat.shape[1]:
                    continue
                bmask = pmask_flat[:, tp]
                if not bmask.any():
                    continue

                # Check all lags are valid
                lags = [tp - (i + 1) * stride for i in range(n)]
                if any(l < 0 or l >= T_raw for l in lags):
                    continue

                # Vectorised correction: Σ c_i * raw_history[tp - (i+1)*s]
                lag_vals = torch.stack(
                    [raw_history[:, l] for l in lags], dim=1
                )  # (B, n)
                correction = (lag_vals * coeffs.unsqueeze(0)).sum(dim=1)  # (B,)

                # Reverse all quantiles with the same correction
                reversed_all_q = pred_h_flat[:, src, :] - correction.unsqueeze(-1)
                reversed_flat[:, src, :] = torch.where(
                    bmask.unsqueeze(-1),
                    reversed_all_q,
                    reversed_flat[:, src, :],
                )

                # Propagate median for subsequent steps
                raw_history[:, tp] = torch.where(
                    bmask,
                    reversed_all_q[:, median_idx],
                    raw_history[:, tp],
                )

            # Write back: (B, (S-h)*ps, nq) -> (B, S-h, nq, ps)
            reversed_h = rearrange(
                reversed_flat, "b (s p) q -> b s q p", p=ps
            )
            preds_r[:, :-h, h - 1, :, :] = reversed_h

        # Reshape back: (B, S, npt, nq, ps) -> (B, S, npt*nq*ps) -> orig shape
        result = rearrange(
            preds_r, "b s npt nq ps -> b s (npt nq ps)"
        )
        return result.reshape(*batch_shape, S, -1)

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
