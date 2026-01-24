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
Embedding-level preconditioning module.

Applies polynomial preconditioning to patch embeddings inside the model forward pass.
This operates on shape (batch, seq_len, d_model) after the patch projection layer.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from uni2ts.transform.precondition import PolynomialPrecondition


class EmbeddingPrecondition(nn.Module):
    """
    Static Polynomial Preconditioning for Patch Embeddings.

    Applies preconditioning to embeddings after patch projection (in_proj).
    Uses fixed Chebyshev or Legendre polynomial coefficients (not learnable).

    Forward: e'_k = e_k + Σᵢ₌₁ⁿ cᵢ · e_{k-i}  (Addition)
    Reverse: e_k = e'_k - Σᵢ₌₁ⁿ cᵢ · e_{k-i}  (Subtraction, autoregressive)

    The reverse after transformer is a "compensating filter" - not a true inverse
    due to the non-linear transformer mixing, but may still help empirically.
    """

    def __init__(
        self,
        degree: int = 5,
        polynomial_type: str = "chebyshev",
        stability_clamp: float = 1e4,
    ):
        """
        Initialize embedding preconditioning module.

        Args:
            degree: Polynomial degree (recommended 2-10)
            polynomial_type: "chebyshev" or "legendre"
            stability_clamp: Maximum absolute value for clamping during reverse (prevents NaN)
        """
        super().__init__()
        self.degree = degree
        self.polynomial_type = polynomial_type
        self.stability_clamp = stability_clamp

        # Compute coefficients using the existing logic from PolynomialPrecondition
        dummy_transform = PolynomialPrecondition(
            polynomial_type=polynomial_type,
            degree=degree,
            target_field="dummy",
            enabled=True
        )

        if polynomial_type == "chebyshev":
            coeffs_np = dummy_transform._chebyshev_coefficients(degree)
        elif polynomial_type == "legendre":
            coeffs_np = dummy_transform._legendre_coefficients(degree)
        else:
            raise ValueError(f"Unknown polynomial type: {polynomial_type}")

        # Fix negative strides issue from numpy
        coeffs_np = coeffs_np.copy()

        # Register as buffer (static, not learnable, but moves with model to GPU)
        self.register_buffer("coeffs", torch.tensor(coeffs_np, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        sample_id: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply forward preconditioning: e'_k = e_k + Σᵢ₌₁ⁿ cᵢ · e_{k-i}

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            sample_id: Tensor of shape [batch, seq_len] identifying samples for packed sequences.
                       Preconditioning is reset when sample_id changes to prevent cross-series contamination.
            target_mask: Optional tensor of shape [batch, seq_len] indicating which positions are
                        target variates (True) vs covariate variates (False). If provided, only
                        target positions are preconditioned; covariate positions remain unchanged.
                        This implements the Universal Sequence Preconditioning theory which applies
                        only to the target variable y_t, not covariates u_t.

        Returns:
            Preconditioned tensor of same shape
        """
        # Handle NaN/Inf in input - replace with zeros for stability
        x_safe = x.clone()
        x_safe = torch.where(torch.isnan(x_safe), torch.zeros_like(x_safe), x_safe)
        x_safe = torch.where(torch.isinf(x_safe), torch.zeros_like(x_safe), x_safe)

        result = self._apply_convolution(x_safe, self.coeffs, mode="add", sample_id=sample_id, target_mask=target_mask)

        # Clamp output for numerical stability
        result = torch.clamp(result, -self.stability_clamp, self.stability_clamp)

        # Final NaN/Inf check (shouldn't be needed but safety)
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
        result = torch.where(torch.isinf(result), torch.zeros_like(result), result)

        return result

    def reverse(
        self,
        x: torch.Tensor,
        sample_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply reverse preconditioning: e_k = e'_k - Σᵢ₌₁ⁿ cᵢ · e_{k-i}

        This is an autoregressive process where e_k depends on previously recovered values.
        Only positions >= degree are modified; earlier positions are unchanged.

        Note: When applied after the transformer, this is a "compensating filter" rather
        than a true inverse, since the transformer has non-linearly transformed the embeddings.

        Numerical stability: Values are clamped at each step to prevent NaN/Inf from
        the autoregressive feedback loop.

        Args:
            x: Input tensor (preconditioned) [batch, seq_len, d_model]
            sample_id: Optional sample ID tensor for packed sequences

        Returns:
            Reversed tensor
        """
        batch, time, dim = x.shape
        degree = self.degree
        coeffs = self.coeffs
        clamp_val = self.stability_clamp

        # If sequence is too short, no reversal needed
        if time <= degree:
            return x.clone()

        # Handle NaN/Inf in input - replace with zeros for stability
        x_safe = x.clone()
        x_safe = torch.where(torch.isnan(x_safe), torch.zeros_like(x_safe), x_safe)
        x_safe = torch.where(torch.isinf(x_safe), torch.zeros_like(x_safe), x_safe)

        # Use a list to accumulate results (avoid in-place operations for autograd)
        # First 'degree' positions are unchanged (but clamped for safety)
        y_list = [torch.clamp(x_safe[:, t, :], -clamp_val, clamp_val) for t in range(degree)]

        # Iterative implementation (autoregressive - can't vectorize over time)
        for t in range(degree, time):
            # e_k = e'_k - sum(coeffs[i] * e_{k-i})
            history_sum = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)

            for i in range(degree):
                # coeffs[i] corresponds to lag i+1
                lag = i + 1
                prev_t = t - lag
                prev_val = y_list[prev_t]

                # Apply mask if sample_id is present
                if sample_id is not None:
                    curr_id = sample_id[:, t]
                    prev_id = sample_id[:, prev_t]
                    mask = (curr_id == prev_id).unsqueeze(-1).float()
                    term = coeffs[i] * prev_val * mask
                else:
                    term = coeffs[i] * prev_val

                history_sum = history_sum + term

            # Compute e_k with clamping for numerical stability
            y_t = x_safe[:, t, :] - history_sum

            # Clamp to prevent runaway values in autoregressive loop
            y_t = torch.clamp(y_t, -clamp_val, clamp_val)

            # Replace any NaN/Inf that still occurred (shouldn't happen, but safety)
            y_t = torch.where(torch.isnan(y_t), torch.zeros_like(y_t), y_t)
            y_t = torch.where(torch.isinf(y_t), torch.zeros_like(y_t), y_t)

            y_list.append(y_t)

        # Stack the list into a tensor
        y = torch.stack(y_list, dim=1)
        return y

    def _apply_convolution(
        self,
        x: torch.Tensor,
        coeffs: torch.Tensor,
        mode: str,
        sample_id: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply convolution using sliding window with masking for packed sequences.

        This is a parallel (vectorized) implementation for the forward pass.
        Only positions >= degree are modified; earlier positions are unchanged.

        Args:
            x: [batch, seq_len, d_model]
            coeffs: [degree]
            mode: "add" for forward, "sub" for parallel subtraction
            sample_id: [batch, seq_len] optional sample IDs
            target_mask: [batch, seq_len] optional mask for target variates (True = target, False = covariate)

        Returns:
            Transformed tensor of same shape as x
        """
        batch, time, dim = x.shape
        degree = len(coeffs)

        # If sequence is too short, no preconditioning applied
        if time <= degree:
            return x.clone()

        # Compute weighted sum for positions >= degree only
        # weighted_sum has shape [batch, time - degree, dim]
        weighted_sum = torch.zeros(batch, time - degree, dim, device=x.device, dtype=x.dtype)

        # Vectorized implementation: sum_{i=1}^n c_i * e_{k-i} for k >= n
        for i in range(1, degree + 1):
            # For position k in [degree, time), we need x[:, k-i, :]
            # k-i ranges from [degree-i, time-i)
            # Which maps to weighted_sum index [0, time-degree)
            start = degree - i
            end = time - i
            weighted_sum = weighted_sum + coeffs[i-1] * x[:, start:end, :]

        # Apply sample_id masking if provided
        if sample_id is not None:
            # For each position k >= degree, check if sample_id[k] matches sample_id[k-i]
            # This requires iterating over lags again with masking
            weighted_sum_masked = torch.zeros_like(weighted_sum)
            for i in range(1, degree + 1):
                start = degree - i
                end = time - i
                term = coeffs[i-1] * x[:, start:end, :]

                # Check sample_id match: sample_id[:, degree:] vs sample_id[:, start:end]
                curr_ids = sample_id[:, degree:]  # [batch, time - degree]
                prev_ids = sample_id[:, start:end]  # [batch, time - degree]
                mask = (curr_ids == prev_ids).unsqueeze(-1).float()  # [batch, time - degree, 1]

                weighted_sum_masked = weighted_sum_masked + term * mask

            weighted_sum = weighted_sum_masked

        # Build result: first degree positions unchanged, rest get convolution
        result = x.clone()
        if mode == "add":
            result[:, degree:, :] = x[:, degree:, :] + weighted_sum
        elif mode == "sub":
            # Parallel subtraction (NOT autoregressive reversal)
            result[:, degree:, :] = x[:, degree:, :] - weighted_sum
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Apply target_mask: only keep preconditioning for target variates
        # For covariate variates (target_mask=False), restore original values
        if target_mask is not None:
            # target_mask: [batch, seq_len] - True for targets, False for covariates
            # Expand to [batch, seq_len, 1] for broadcasting with [batch, seq_len, d_model]
            target_mask_expanded = target_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            # Where target_mask is True: use preconditioned result
            # Where target_mask is False: use original x (covariate positions unchanged)
            result = result * target_mask_expanded + x * (1.0 - target_mask_expanded)

        return result

    def extra_repr(self) -> str:
        return f"degree={self.degree}, polynomial_type={self.polynomial_type}, stability_clamp={self.stability_clamp}"
