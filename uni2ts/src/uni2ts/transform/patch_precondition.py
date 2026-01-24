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

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from jaxtyping import Float, Int

from ._base import Transformation
from .precondition import PolynomialPrecondition

@dataclass
class PatchPolynomialPrecondition(PolynomialPrecondition):
    """
    Apply polynomial preconditioning to patched time series.
    
    This transform operates on data that has already been patched.
    Input shape is expected to be (..., time, patch_size).
    
    The preconditioning is applied along the 'time' dimension, treating
    each element of the patch as an independent channel.
    
    y'_t = y_t + sum_{i=1}^n c_i * y_{t-i}
    
    where y_t is a vector of size patch_size.
    """
    
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return data_entry

        if self.target_field not in data_entry:
            return data_entry

        target = data_entry[self.target_field]
        
        # Target is expected to be (..., time, patch_size)
        # It might be a numpy array or torch tensor, but usually numpy in transforms
        
        if isinstance(target, list):
            # List of arrays
            preconditioned = []
            for ts in target:
                if not isinstance(ts, np.ndarray):
                    ts = np.array(ts)
                preconditioned.append(self._apply_patch_convolution(ts, self.coeffs))
            data_entry[self.target_field] = preconditioned
            
        else:
            if not isinstance(target, np.ndarray):
                target = np.array(target)
                
            preconditioned = self._apply_patch_convolution(target, self.coeffs)
            data_entry[self.target_field] = preconditioned

        # Store metadata
        if self.store_original:
            data_entry[f"{self.target_field}_original"] = target
            
        data_entry["precondition_coeffs"] = self.coeffs
        data_entry["precondition_degree"] = self.degree
        data_entry["precondition_type"] = self.polynomial_type
        data_entry["precondition_enabled"] = True
        data_entry["precondition_is_patched"] = True

        return data_entry

    def _apply_patch_convolution(
        self,
        sequence: np.ndarray,
        coeffs: np.ndarray
    ) -> np.ndarray:
        """
        Apply convolution to patched sequence.
        
        Args:
            sequence: Array of shape (..., time, patch_size)
            coeffs: Array of coefficients
            
        Returns:
            Preconditioned array of same shape
        """
        # sequence shape: (..., time, patch_size)
        # We want to convolve along the -2 axis (time)
        
        n = len(coeffs)
        result = sequence.copy()
        
        time_dim = -2
        seq_len = sequence.shape[time_dim]
        
        if seq_len > n:
            # Vectorized implementation
            # result[t] = sequence[t] + sum(coeffs[i] * sequence[t-i-1])
            
            weighted_sum = np.zeros_like(sequence[..., n:, :])
            
            for i in range(n):
                # coeffs[i] corresponds to lag i+1
                # We need sequence[..., n-(i+1) : seq_len-(i+1), :]
                
                # Slicing along time dimension
                # We want to slice from n-i-1 to seq_len-i-1
                
                start = n - i - 1
                end = seq_len - i - 1
                
                # Create slice object for arbitrary dimensions
                # slice(None) for ...
                # slice(start, end) for time
                # slice(None) for patch_size
                
                # Construct slice tuple
                sl = [slice(None)] * sequence.ndim
                sl[time_dim] = slice(start, end)
                
                shifted_seq = sequence[tuple(sl)]
                
                weighted_sum += coeffs[i] * shifted_seq
                
            # Add to original values
            # Target slice: time >= n
            sl_target = [slice(None)] * sequence.ndim
            sl_target[time_dim] = slice(n, None)
            
            result[tuple(sl_target)] += weighted_sum
            
        return result


@dataclass
class PatchReversePrecondition(Transformation):
    """
    Reverse patch-level preconditioning.
    
    This transform reverses the convolution applied by PatchPolynomialPrecondition.
    It expects the input to be patched (..., time, patch_size).
    """
    target_field: str = "target"
    prediction_field: str = "prediction"
    enabled: bool = True

    def __call__(
        self,
        data_entry: dict[str, Any],
        context: Optional[np.ndarray] = None
    ) -> dict[str, Any]:
        if not self.enabled:
            return data_entry
            
        if not data_entry.get("precondition_enabled", False):
            return data_entry
            
        # Check if it was patched preconditioning
        if not data_entry.get("precondition_is_patched", False):
            # Fallback to standard reverse if available? 
            # Or just return if we strictly want patch reversal?
            # For now, let's assume we only handle patched reversal here.
            return data_entry

        if "precondition_coeffs" not in data_entry:
            return data_entry

        coeffs = data_entry["precondition_coeffs"]

        field_to_reverse = None
        if self.prediction_field in data_entry:
            field_to_reverse = self.prediction_field
        elif self.target_field in data_entry:
            field_to_reverse = self.target_field
        else:
            return data_entry

        preconditioned = data_entry[field_to_reverse]
        
        if not isinstance(preconditioned, np.ndarray):
            preconditioned = np.array(preconditioned)
            
        original_dtype = preconditioned.dtype
        
        # Apply reversal
        # Input shape: (..., time, patch_size)
        
        restored = self._reverse_patch_convolution(preconditioned, coeffs, context)
        
        restored = restored.astype(original_dtype)
        data_entry[field_to_reverse] = restored
        
        return data_entry

    def _reverse_patch_convolution(
        self,
        sequence: np.ndarray,
        coeffs: np.ndarray,
        context: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reverse convolution on patched sequence.
        
        Args:
            sequence: Preconditioned array (..., time, patch_size)
            coeffs: Coefficients
            context: Context array (..., context_time, patch_size)
            
        Returns:
            Restored array
        """
        n = len(coeffs)
        
        # We need to iterate along time dimension
        # Since it's recursive (y_t depends on previous restored y), we can't fully vectorize along time
        # But we can vectorize across batch/patch dimensions
        
        time_dim = -2
        seq_len = sequence.shape[time_dim]
        
        # Prepare buffer with context if available
        if context is not None:
            # Context shape should match sequence except for time dim
            # Concatenate along time dim
            
            # Ensure context is numpy
            if not isinstance(context, np.ndarray):
                context = np.array(context)
                
            full_seq = np.concatenate([context, sequence], axis=time_dim)
            start_idx = context.shape[time_dim]
            
            # We need to update full_seq in place
            # But sequence part of full_seq currently contains preconditioned values (y')
            # We need to convert them to y
            
            # Create a copy to hold restored values
            # Initialize with context (already y)
            # And sequence (currently y')
            y_buffer = full_seq.copy()
            
            # Iterate through sequence part
            for t in range(seq_len):
                t_full = start_idx + t
                
                # y_t = y'_t - sum(coeffs[i] * y_{t-i-1})
                # y'_t is in sequence[t] (which is y_buffer[t_full] initially)
                
                weighted_sum = 0.0
                for i in range(n):
                    # lag i+1
                    prev_idx = t_full - i - 1
                    
                    # Access y_buffer at prev_idx
                    # Need to handle arbitrary dimensions
                    sl = [slice(None)] * y_buffer.ndim
                    sl[time_dim] = prev_idx
                    
                    weighted_sum += coeffs[i] * y_buffer[tuple(sl)]
                    
                # Update y_buffer
                sl_curr = [slice(None)] * y_buffer.ndim
                sl_curr[time_dim] = t_full
                
                # y_buffer[t_full] currently holds y'_t
                # We want y_t = y'_t - weighted_sum
                y_buffer[tuple(sl_curr)] -= weighted_sum
                
            # Return only the sequence part
            sl_res = [slice(None)] * y_buffer.ndim
            sl_res[time_dim] = slice(start_idx, None)
            return y_buffer[tuple(sl_res)]
            
        else:
            # No context
            result = sequence.copy()
            
            for t in range(n, seq_len):
                # y_t = y'_t - sum(coeffs[i] * y_{t-i-1})
                
                weighted_sum = 0.0
                for i in range(n):
                    prev_idx = t - i - 1
                    
                    sl = [slice(None)] * result.ndim
                    sl[time_dim] = prev_idx
                    
                    weighted_sum += coeffs[i] * result[tuple(sl)]
                    
                sl_curr = [slice(None)] * result.ndim
                sl_curr[time_dim] = t
                
                result[tuple(sl_curr)] -= weighted_sum
                
            return result
