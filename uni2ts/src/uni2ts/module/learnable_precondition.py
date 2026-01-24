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

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from uni2ts.transform.precondition import PolynomialPrecondition

class LearnablePrecondition(nn.Module):
    """
    Learnable Polynomial Preconditioning Module.
    
    Initializes coefficients using Chebyshev or Legendre polynomials (same as the static transform),
    but makes them learnable parameters.
    
    Forward: ỹₜ = yₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ (Addition)
    Reverse: yₜ = ỹₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ (Subtraction)
    """
    def __init__(
        self,
        degree: int = 5,
        polynomial_type: str = "chebyshev",
        dim: int = 1, # Number of variates, usually applied per-variate so maybe not needed if broadcasting works
    ):
        super().__init__()
        self.degree = degree
        self.polynomial_type = polynomial_type
        
        # Initialize coefficients using the existing logic from PolynomialPrecondition
        # We use a dummy instance to get the coefficients
        dummy_transform = PolynomialPrecondition(
            polynomial_type=polynomial_type,
            degree=degree,
            target_field="dummy",
            enabled=True
        )
        
        # Get coefficients (numpy array)
        # The transform computes them on the fly or stores them. 
        # We can extract them by accessing the internal method or property if available.
        # Looking at precondition.py, it computes them in __call__ or we can use the helper methods.
        # Actually, PolynomialPrecondition has _chebyshev_coefficients and _legendre_coefficients methods.
        
        if polynomial_type == "chebyshev":
            coeffs_np = dummy_transform._chebyshev_coefficients(degree)
        elif polynomial_type == "legendre":
            coeffs_np = dummy_transform._legendre_coefficients(degree)
        else:
            raise ValueError(f"Unknown polynomial type: {polynomial_type}")
            
        # Fix negative strides issue
        coeffs_np = coeffs_np.copy()
            
        # coeffs_np is [c1, c2, ..., cn]
        # Register as learnable parameter
        self.coeffs = nn.Parameter(torch.tensor(coeffs_np, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor, sample_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply forward preconditioning: ỹₜ = yₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ
        
        Args:
            x: Input tensor of shape [batch, time, dim] or [batch, time]
            sample_id: Optional tensor of shape [batch, time] identifying samples for packed sequences.
                       If provided, preconditioning is reset when sample_id changes.
            
        Returns:
            Preconditioned tensor of same shape
        """
        return self._apply_convolution(x, self.coeffs, mode="add", sample_id=sample_id)
        
    def reverse(self, x: torch.Tensor, sample_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply reverse preconditioning: yₜ = ỹₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ

        This is an autoregressive process where yₜ depends on previously recovered values.

        Args:
            x: Input tensor (preconditioned) [batch, time, dim] or [batch, time]
            sample_id: Optional sample ID tensor

        Returns:
            Reversed tensor (original scale)
        """
        # Ensure x is [batch, time, dim]
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        batch, time, dim = x.shape
        degree = self.degree
        coeffs = self.coeffs

        # Use a list to accumulate results (avoid in-place operations for autograd)
        y_list = []

        # Iterative implementation
        # We can't easily vectorize over time because of the autoregressive dependency.
        # However, we can vectorize over batch and dim.

        for t in range(time):
            # y[t] = x[t] - sum(coeffs[i] * y[t-i-1])

            history_sum = torch.zeros(batch, dim, device=x.device, dtype=x.dtype)

            for i in range(degree):
                # Look back i+1 steps
                # coeffs[i] corresponds to lag i+1
                lag = i + 1
                if t - lag >= 0:
                    # Access from the list instead of tensor indexing
                    prev_val = y_list[t - lag]

                    # Apply mask if sample_id is present
                    if sample_id is not None:
                        # Check if sample_id matches
                        # sample_id: [batch, time]
                        curr_id = sample_id[:, t]
                        prev_id = sample_id[:, t - lag]
                        mask = (curr_id == prev_id).unsqueeze(-1).float()
                        term = coeffs[i] * prev_val * mask
                    else:
                        term = coeffs[i] * prev_val

                    history_sum = history_sum + term

            # Compute y[t] and append to list
            y_t = x[:, t, :] - history_sum
            y_list.append(y_t)

        # Stack the list into a tensor
        y = torch.stack(y_list, dim=1)
        return y

    def _apply_convolution(
        self, 
        x: torch.Tensor, 
        coeffs: torch.Tensor, 
        mode: str, 
        sample_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply convolution using sliding window with masking for packed sequences.
        
        x: [batch, time, dim]
        coeffs: [degree]
        sample_id: [batch, time]
        """
        # Ensure x is [batch, time, dim]
        if x.ndim == 2:
            x = x.unsqueeze(-1)
            
        batch, time, dim = x.shape
        degree = len(coeffs)
        
        # Output accumulator
        history_sum = torch.zeros_like(x)
        
        # Iterative implementation with masking
        # sum_{i=1}^n c_i * y_{t-i}
        for i in range(1, degree + 1):
            # Shift x by i positions to the right (time dimension)
            # padded: [batch, time + i, dim]
            # We want x[t-i], so we shift right.
            # shifted[:, t] = x[:, t-i]
            
            # Efficient shifting using slicing
            # shifted: [batch, time, dim]
            # Zeros at the beginning
            shifted = torch.zeros_like(x)
            shifted[:, i:, :] = x[:, :-i, :]
            
            # Apply coefficient
            term = coeffs[i-1] * shifted
            
            # Apply mask if sample_id is present
            if sample_id is not None:
                # Shift sample_id similarly
                shifted_sample_id = torch.full_like(sample_id, -1) # -1 as invalid ID
                shifted_sample_id[:, i:] = sample_id[:, :-i]
                
                # Mask: 1 if sample_id matches, 0 otherwise
                # [batch, time] -> [batch, time, 1]
                mask = (sample_id == shifted_sample_id).unsqueeze(-1).float()
                term = term * mask
                
            history_sum = history_sum + term
            
        if mode == "add":
            return x + history_sum
        elif mode == "sub":
            # Note: This is parallel subtraction, NOT autoregressive reversal.
            # Only use this if you know what you are doing (e.g. for loss calculation where input is ground truth).
            return x - history_sum
        else:
            raise ValueError(f"Unknown mode: {mode}")
