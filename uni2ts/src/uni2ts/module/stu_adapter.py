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
STU (Spectral Transform Unit) adapter for MOIRAI.

This module wraps the core STU spectral convolution operation in a form
compatible with MOIRAI's architecture, handling:
- Variable sequence lengths (via filter truncation)
- Packed sequence format (via sample_id-based processing)
- Both approx mode (50x fewer params) and standard mode

Based on the Flash-STU paper: "Flash STU: Fast Spectral Transform Units" (arXiv:2409.10489)
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from jaxtyping import Float, Int

from .spectral_filters import compute_spectral_filters, convolve_spectral


class STUCore(nn.Module):
    """
    Core STU module implementing spectral convolution.

    This is the fundamental building block that performs FFT-based convolution
    with learned spectral filter projections.

    Args:
        d_model: Model dimension (input and output)
        max_seq_len: Maximum sequence length (for filter pre-computation)
        num_eigh: Number of spectral filters (K)
        use_hankel_L: Use single-branch Hankel-L (faster) vs dual-branch (default)
        use_approx: Use approximation mode with 50x fewer parameters (recommended)
        dtype: Data type for parameters

    Example:
        >>> stu = STUCore(d_model=512, max_seq_len=2048, num_eigh=24)
        >>> x = torch.randn(2, 256, 512)  # [batch, seq, dim]
        >>> y = stu(x)  # [batch, seq, dim]
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_approx: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.use_approx = use_approx

        # Pre-compute spectral filters (will be moved to correct device later)
        # Note: We compute on CPU first and register as buffer
        phi = compute_spectral_filters(
            seq_len=max_seq_len,
            num_eigh=num_eigh,
            use_hankel_L=use_hankel_L,
            device=torch.device('cpu'),
            dtype=dtype,
        )
        self.register_buffer('phi', phi, persistent=False)

        # Initialize learnable projection matrices
        if use_approx:
            # Approximation mode: factorized projections (50x fewer params)
            # M_inputs: [d_model, d_model] - input projection
            # M_filters: [num_eigh, d_model] - filter projection
            self.M_inputs = nn.Parameter(torch.empty(d_model, d_model, dtype=dtype))
            self.M_filters = nn.Parameter(torch.empty(num_eigh, d_model, dtype=dtype))
            self._init_approx_params()
        else:
            # Standard mode: full projection matrices
            # M_phi_plus: [num_eigh, d_model, d_model]
            self.M_phi_plus = nn.Parameter(
                torch.empty(num_eigh, d_model, d_model, dtype=dtype)
            )
            if not use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    torch.empty(num_eigh, d_model, d_model, dtype=dtype)
                )
            self._init_standard_params()

    def _init_approx_params(self):
        """Initialize parameters for approximation mode."""
        # Xavier initialization scaled for STU
        std = 1.0 / math.sqrt(self.d_model)
        nn.init.normal_(self.M_inputs, mean=0.0, std=std)
        nn.init.normal_(self.M_filters, mean=0.0, std=std)

    def _init_standard_params(self):
        """Initialize parameters for standard mode."""
        std = 1.0 / math.sqrt(self.num_eigh * self.d_model)
        nn.init.normal_(self.M_phi_plus, mean=0.0, std=std)
        if not self.use_hankel_L:
            nn.init.normal_(self.M_phi_minus, mean=0.0, std=std)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"],
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        """
        Apply spectral convolution.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        seq_len = x.shape[1]

        # Truncate filters to actual sequence length
        phi = self.phi[:seq_len]

        # Determine whether to compute both branches
        return_both = not self.use_hankel_L

        if self.use_approx:
            # Approximation mode: project inputs and filters, then convolve
            x_proj = x @ self.M_inputs  # [batch, seq, d_model]
            phi_proj = phi @ self.M_filters  # [seq, d_model]

            spectral_plus, spectral_minus = convolve_spectral(
                x_proj, phi_proj, use_approx=True, return_both=return_both
            )
        else:
            # Standard mode: convolve, then contract
            U_plus, U_minus = convolve_spectral(
                x, phi, use_approx=False, return_both=return_both
            )
            # Contract over K and d_in dimensions
            spectral_plus = torch.tensordot(
                U_plus, self.M_phi_plus, dims=([2, 3], [0, 1])
            )
            if return_both:
                spectral_minus = torch.tensordot(
                    U_minus, self.M_phi_minus, dims=([2, 3], [0, 1])
                )

        if self.use_hankel_L:
            return spectral_plus
        else:
            return spectral_plus + spectral_minus

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"max_seq_len={self.max_seq_len}, "
            f"num_eigh={self.num_eigh}, "
            f"use_hankel_L={self.use_hankel_L}, "
            f"use_approx={self.use_approx}"
        )


class PackedSTU(nn.Module):
    """
    STU module that handles MOIRAI's packed sequence format.

    MOIRAI packs multiple samples into a single sequence with sample_id tracking.
    This module applies STU per-sample by:
    1. Unpacking by sample_id
    2. Applying STU to each sample
    3. Repacking into original format

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length per sample
        num_eigh: Number of spectral filters
        use_hankel_L: Use single-branch formulation
        use_approx: Use approximation mode (recommended)
        dtype: Data type for parameters

    Example:
        >>> stu = PackedSTU(d_model=512, max_seq_len=2048, num_eigh=24)
        >>> x = torch.randn(1, 500, 512)  # Packed sequence
        >>> sample_id = torch.tensor([[0]*100 + [1]*150 + [2]*250])
        >>> y = stu(x, sample_id=sample_id)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        use_approx: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.stu_core = STUCore(
            d_model=d_model,
            max_seq_len=max_seq_len,
            num_eigh=num_eigh,
            use_hankel_L=use_hankel_L,
            use_approx=use_approx,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "*batch seq_len d_model"],
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
    ) -> Float[torch.Tensor, "*batch seq_len d_model"]:
        """
        Apply STU with optional packed sequence handling.

        Args:
            x: Input tensor [*batch, seq_len, d_model]
            sample_id: Optional sample IDs for packed sequences [*batch, seq_len]

        Returns:
            Output tensor [*batch, seq_len, d_model]
        """
        if sample_id is None:
            # No packing: direct STU application
            return self.stu_core(x)

        # Handle packed sequences
        return self._forward_packed(x, sample_id)

    def _forward_packed(
        self,
        x: Float[torch.Tensor, "*batch seq_len d_model"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len d_model"]:
        """
        Apply STU to packed sequences by processing each sample separately.

        This unpacks by sample_id, applies STU per sample, then repacks.
        """
        # Flatten batch dimensions for simplicity
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-2], x.shape[-1])  # [B, L, D]
        sample_id_flat = sample_id.view(-1, sample_id.shape[-1])  # [B, L]

        output = torch.zeros_like(x_flat)

        for batch_idx in range(x_flat.shape[0]):
            x_b = x_flat[batch_idx]  # [L, D]
            sid_b = sample_id_flat[batch_idx]  # [L]

            unique_samples = sid_b.unique()
            for sid in unique_samples:
                mask = sid_b == sid
                x_sample = x_b[mask].unsqueeze(0)  # [1, L_sample, D]

                # Apply STU to this sample
                y_sample = self.stu_core(x_sample)  # [1, L_sample, D]

                # Place back
                output[batch_idx, mask] = y_sample.squeeze(0)

        return output.view(original_shape)

    def forward_batched(
        self,
        x: Float[torch.Tensor, "*batch seq_len d_model"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        max_samples: Optional[int] = None,
    ) -> Float[torch.Tensor, "*batch seq_len d_model"]:
        """
        More efficient batched version for packed sequences.

        Pads all samples to same length for batched FFT, then unpads.
        Use when number of samples is known and memory allows.

        Args:
            x: Input tensor
            sample_id: Sample IDs
            max_samples: Maximum expected samples (for pre-allocation)

        Returns:
            Output tensor
        """
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-2], x.shape[-1])
        sample_id_flat = sample_id.view(-1, sample_id.shape[-1])

        output = torch.zeros_like(x_flat)

        for batch_idx in range(x_flat.shape[0]):
            x_b = x_flat[batch_idx]
            sid_b = sample_id_flat[batch_idx]

            unique_samples = sid_b.unique()
            num_samples = len(unique_samples)

            # Collect sample lengths
            sample_xs = []
            sample_lengths = []
            for sid in unique_samples:
                mask = sid_b == sid
                sample_xs.append(x_b[mask])
                sample_lengths.append(mask.sum().item())

            # Pad to max length and batch
            max_len = max(sample_lengths)
            batched_x = torch.zeros(num_samples, max_len, x_b.shape[-1],
                                    device=x.device, dtype=x.dtype)
            for i, sx in enumerate(sample_xs):
                batched_x[i, :len(sx)] = sx

            # Apply STU in batch
            batched_y = self.stu_core(batched_x)

            # Unpad and place back
            for i, (sid, length) in enumerate(zip(unique_samples, sample_lengths)):
                mask = sid_b == sid
                output[batch_idx, mask] = batched_y[i, :length]

        return output.view(original_shape)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_seq_len={self.max_seq_len}"
