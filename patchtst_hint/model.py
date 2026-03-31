"""
PatchTST with polynomial hint preconditioning.

Adds a fixed Chebyshev FIR-filtered hint channel to PatchTST's input embedding.
The only change is widening the input_embedding from Linear(patch_len, d_model) to
Linear(patch_len * 2, d_model). Everything else (encoder, head) is unchanged.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import PatchTSTForPrediction, PatchTSTConfig


def chebyshev_coefficients(degree: int) -> np.ndarray:
    """Compute Chebyshev polynomial T_d coefficients in monomial basis."""
    if degree == 0:
        return np.array([1.0])
    elif degree == 1:
        return np.array([0.0, 1.0])
    T_prev = np.zeros(degree + 1)
    T_prev[0] = 1.0
    T_curr = np.zeros(degree + 1)
    T_curr[1] = 1.0
    for _ in range(2, degree + 1):
        T_next = np.zeros(degree + 1)
        for j in range(degree):
            T_next[j + 1] += 2 * T_curr[j]
        for j in range(degree + 1):
            T_next[j] -= T_prev[j]
        T_prev = T_curr
        T_curr = T_next
    return T_curr


def apply_fir_hint(x: torch.Tensor, coeffs: torch.Tensor, stride: int) -> torch.Tensor:
    """Apply FIR filter and return the residual (hint).

    Args:
        x: (batch, seq_len, channels)
        coeffs: (degree+1,) polynomial coefficients
        stride: filter stride (= patch_length)
    Returns:
        hint: (batch, seq_len, channels)
    """
    B, T, C = x.shape
    degree = len(coeffs) - 1
    hint = torch.zeros_like(x)

    for k in range(degree + 1):
        lag = k * stride
        if lag == 0:
            hint += (coeffs[k] - 1.0) * x
        elif lag < T:
            hint[:, lag:, :] += coeffs[k] * x[:, :-lag, :]

    return hint


class HintPatchTSTForPrediction(nn.Module):
    """PatchTST with polynomial hint preconditioning.

    The hint is computed on the normalized time series, patchified identically to
    the target, and concatenated along the patch dimension before the embedding layer.
    A wider embedding layer (patch_len*2 -> d_model) replaces the standard one.

    Args:
        config: PatchTSTConfig
        hint_degree: Chebyshev polynomial degree (default 4)
        hint_stride: FIR filter stride (default = patch_length)
        hint_dropout: per-patch hint dropout probability (default 0.1)
        hint_mode: "hint" | "none" | "zero" | "duplicate"
    """

    def __init__(
        self,
        config: PatchTSTConfig,
        hint_degree: int = 4,
        hint_stride: int = None,
        hint_dropout: float = 0.1,
        hint_mode: str = "hint",
    ):
        super().__init__()
        self.hint_degree = hint_degree
        self.hint_stride = hint_stride or config.patch_length
        self.hint_dropout_rate = hint_dropout
        self.hint_mode = hint_mode
        self.patch_length = config.patch_length
        self.has_hint = (hint_mode != "none")

        # Build base model with ORIGINAL config, then widen the embedding
        self.model_config = config
        self.base = PatchTSTForPrediction(config)

        if self.has_hint:
            # Replace embedding: Linear(patch_len, d_model) -> Linear(patch_len*2, d_model)
            embedder = self.base.model.encoder.embedder
            if embedder.share_embedding:
                old = embedder.input_embedding
                embedder.input_embedding = nn.Linear(config.patch_length * 2, config.d_model)
            else:
                for i in range(len(embedder.input_embedding)):
                    embedder.input_embedding[i] = nn.Linear(config.patch_length * 2, config.d_model)
            # Replace patchifier with identity (we do our own)
            self.base.model.patchifier = _IdentityPatchifier(config)

        # Chebyshev coefficients
        coeffs = chebyshev_coefficients(hint_degree)
        self.register_buffer("hint_coeffs", torch.tensor(coeffs, dtype=torch.float32))

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify: (B, T, C) -> (B, C, num_patches, patch_len)"""
        B, T, C = x.shape
        P = self.patch_length
        # Trim to fit
        n_patches = (T - P) // P + 1
        start = T - P - P * (n_patches - 1)
        x = x[:, start:, :]  # (B, n_patches*P, C)
        # Unfold
        x = x.unfold(dimension=1, size=P, step=P)  # (B, n_patches, C, P)
        x = x.permute(0, 2, 1, 3)  # (B, C, n_patches, P)
        return x

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor = None,
        future_values: torch.Tensor = None,
        output_hidden_states: bool = None,
        output_attentions: bool = None,
        return_dict: bool = None,
    ):
        if not self.has_hint:
            return self.base(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                future_values=future_values,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )

        # 1. Scale (use the base model's scaler)
        model = self.base.model
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        scaled_past, loc, scale = model.scaler(past_values, past_observed_mask)

        # 2. Compute hints on scaled data
        if self.hint_mode == "zero":
            hints = torch.zeros_like(scaled_past)
        elif self.hint_mode == "duplicate":
            hints = scaled_past.clone()
        else:
            hints = apply_fir_hint(scaled_past, self.hint_coeffs, self.hint_stride)

        # 3. Hint dropout (per-patch, during training)
        if self.training and self.hint_dropout_rate > 0:
            B, T, C = hints.shape
            n_patches = T // self.patch_length
            mask = torch.rand(B, n_patches, 1, 1, device=hints.device) < self.hint_dropout_rate
            # Reshape hints to patch form, apply mask, reshape back
            P = self.patch_length
            start = T - P * n_patches
            hint_patches = hints[:, start:, :].reshape(B, n_patches, P, C)
            hint_patches = hint_patches.masked_fill(mask.expand_as(hint_patches), 0.0)
            hints[:, start:, :] = hint_patches.reshape(B, n_patches * P, C)

        # 4. Patchify both
        target_patches = self._patchify(scaled_past)   # (B, C, n_patches, P)
        hint_patches = self._patchify(hints)            # (B, C, n_patches, P)

        # 5. Concatenate along patch dimension: (B, C, n_patches, 2*P)
        combined = torch.cat([target_patches, hint_patches], dim=-1)

        # 6. Forward through masking + encoder + head
        # The base model's patchifier is replaced with identity, so we pass combined directly
        masked = model.masking(combined)

        encoder_output = model.encoder(
            masked,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        head = self.base.head
        if hasattr(encoder_output, 'last_hidden_state'):
            hidden = encoder_output.last_hidden_state
        elif isinstance(encoder_output, tuple):
            hidden = encoder_output[0]
        else:
            hidden = encoder_output

        y_hat = head(hidden)  # (B, pred_len, C)

        # Unscale
        y_hat = y_hat * scale + loc

        # Loss
        loss = None
        if future_values is not None:
            loss = nn.MSELoss()(y_hat, future_values)

        if return_dict is False:
            return (y_hat, loss) if loss is not None else (y_hat,)

        from dataclasses import dataclass
        @dataclass
        class Output:
            loss: torch.Tensor = None
            prediction_outputs: torch.Tensor = None
        return Output(loss=loss, prediction_outputs=y_hat)

    @property
    def device(self):
        return next(self.parameters()).device


class _IdentityPatchifier(nn.Module):
    """Dummy patchifier that passes through pre-patchified input."""
    def __init__(self, config):
        super().__init__()
        self.sequence_length = config.context_length
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_length
        self.num_patches = (config.context_length - config.patch_length) // config.patch_length + 1

    def forward(self, x):
        return x  # Already patchified
