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
Spectral filter computation utilities for STU (Spectral Transform Units).

This module provides functions to compute spectral filters from Hankel matrices,
which form the basis for STU's FFT-based sequence processing.

Based on the Flash-STU paper: "Flash STU: Fast Spectral Transform Units" (arXiv:2409.10489)
"""

from typing import Optional, Dict, Tuple
import numpy as np
import torch


def get_hankel_matrix(seq_len: int, use_hankel_L: bool = False) -> np.ndarray:
    """
    Construct the Hankel matrix for spectral filter computation.

    The Hankel matrix encodes the impulse response structure of linear dynamical
    systems. Its eigendecomposition provides an optimal orthogonal basis for
    approximating LDS outputs.

    Args:
        seq_len: Sequence length (determines matrix size)
        use_hankel_L: If True, use Hankel-L formulation (single branch, faster).
                      If False, use standard Hankel (dual branch, more expressive).

    Returns:
        Z: Hankel matrix of shape [seq_len, seq_len]

    Theory:
        Standard Hankel: Z[i,j] = 2 / ((i+j)^3 - (i+j))
        Hankel-L:        Z[i,j] = sgn(i+j-2) * 8 / ((i+j+3)(i+j-1)(i+j+1))
    """
    entries = np.arange(1, seq_len + 1, dtype=np.float64)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        # Hankel-L: single branch formulation
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        # Standard Hankel: dual branch formulation
        Z = 2.0 / (i_plus_j ** 3 - i_plus_j)

    return Z


def compute_spectral_filters(
    seq_len: int,
    num_eigh: int,
    use_hankel_L: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute spectral filters from Hankel matrix eigendecomposition.

    This function performs eigendecomposition of the Hankel matrix and extracts
    the top-K eigenvectors scaled by the fourth root of eigenvalues.

    Args:
        seq_len: Sequence length
        num_eigh: Number of spectral filters (K) to keep
        use_hankel_L: Whether to use Hankel-L formulation
        device: Device to place filters on (default: cuda if available, else cpu)
        dtype: Data type for filters

    Returns:
        phi: Spectral filter tensor of shape [seq_len, num_eigh]

    Note:
        The eigendecomposition has O(n^3) complexity, but this is a one-time cost
        at initialization. Filters should be cached as buffers.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Compute Hankel matrix
    Z = get_hankel_matrix(seq_len, use_hankel_L)

    # Eigendecomposition (returns eigenvalues in ascending order)
    sigma, phi = np.linalg.eigh(Z)

    # Select top-K (largest) eigenvalues and corresponding eigenvectors
    sigma = sigma[-num_eigh:]
    phi = phi[:, -num_eigh:]

    # Scale eigenvectors by fourth root of eigenvalues
    # This balances the importance of different spectral components
    phi = phi * np.abs(sigma) ** 0.25

    return torch.tensor(phi, device=device, dtype=dtype)


class SpectralFilterBank:
    """
    Pre-computed bank of spectral filters for different sequence lengths.

    This class manages spectral filters for MOIRAI's multi-patch configuration,
    where different patch sizes result in different effective sequence lengths.

    Example:
        >>> filter_bank = SpectralFilterBank(
        ...     max_seq_len=2048,
        ...     patch_sizes=(8, 16, 32, 64, 128),
        ...     num_eigh=24
        ... )
        >>> phi = filter_bank.get_filters(seq_len=256, device='cuda')
    """

    def __init__(
        self,
        max_seq_len: int,
        patch_sizes: Tuple[int, ...] = (8, 16, 32, 64, 128),
        num_eigh: int = 24,
        use_hankel_L: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the filter bank.

        Args:
            max_seq_len: Maximum context length in tokens (before patching)
            patch_sizes: Tuple of supported patch sizes
            num_eigh: Number of spectral filters
            use_hankel_L: Whether to use Hankel-L formulation
            dtype: Data type for filters
        """
        self.max_seq_len = max_seq_len
        self.patch_sizes = patch_sizes
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.dtype = dtype

        # Compute filters for each patch configuration
        # Key: number of patches (seq_len // patch_size)
        self._filters: Dict[int, torch.Tensor] = {}
        self._precompute_filters()

    def _precompute_filters(self):
        """Pre-compute filters for all patch size configurations."""
        computed_lengths = set()

        for patch_size in self.patch_sizes:
            max_patches = self.max_seq_len // patch_size
            if max_patches not in computed_lengths and max_patches > 0:
                self._filters[max_patches] = compute_spectral_filters(
                    seq_len=max_patches,
                    num_eigh=self.num_eigh,
                    use_hankel_L=self.use_hankel_L,
                    device=torch.device('cpu'),  # Store on CPU, move to device on demand
                    dtype=self.dtype,
                )
                computed_lengths.add(max_patches)

    def get_filters(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get spectral filters for a given sequence length.

        Args:
            seq_len: Number of patches (not tokens)
            device: Device to place filters on

        Returns:
            phi: Spectral filters [seq_len, num_eigh]

        Note:
            If exact seq_len not pre-computed, uses the closest larger pre-computed
            length and truncates.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Find the smallest pre-computed length >= seq_len
        available_lengths = sorted(self._filters.keys())

        for length in available_lengths:
            if length >= seq_len:
                phi = self._filters[length][:seq_len].to(device)
                return phi

        # If seq_len is larger than any pre-computed, compute on the fly
        phi = compute_spectral_filters(
            seq_len=seq_len,
            num_eigh=self.num_eigh,
            use_hankel_L=self.use_hankel_L,
            device=device,
            dtype=self.dtype,
        )
        # Cache it
        self._filters[seq_len] = phi.cpu()
        return phi

    def to_module_buffers(self, module: torch.nn.Module, prefix: str = "phi"):
        """
        Register filters as module buffers for proper device handling.

        Args:
            module: The nn.Module to register buffers on
            prefix: Prefix for buffer names
        """
        for length, phi in self._filters.items():
            buffer_name = f"{prefix}_{length}"
            module.register_buffer(buffer_name, phi, persistent=False)


def convolve_spectral(
    u: torch.Tensor,
    phi: torch.Tensor,
    use_approx: bool = True,
    return_both: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    FFT-based convolution with spectral filters.

    This is the core operation of STU: convolving input sequences with
    pre-computed spectral filters in the frequency domain.

    Args:
        u: Input tensor [batch, seq_len, d_in]
        phi: Spectral filters [seq_len, num_eigh] or projected [seq_len, d_out]
        use_approx: If True, phi is already projected [seq_len, d_out].
                    If False, phi is raw filters [seq_len, K].
        return_both: If True, compute both positive and negative branches.
                     If False, compute only positive branch (Hankel-L).

    Returns:
        U_plus: Positive branch output [batch, seq_len, d_out]
        U_minus: Negative branch output (None if return_both=False)

    Complexity: O(L log L) via FFT
    """
    bsz, seq_len, d_in = u.shape
    input_dtype = u.dtype

    # Compute FFT length (next power of 2 for efficiency)
    n = 1 << (seq_len - 1).bit_length()

    # Sign pattern for negative branch: alternating +1, -1
    sgn = torch.ones(1, seq_len, 1, device=u.device, dtype=u.dtype)
    sgn[:, 1::2] *= -1

    if return_both:
        # Compute both branches (standard Hankel)
        if use_approx:
            _, d_out = phi.shape
            v = phi.view(1, -1, d_out, 1).to(torch.float32)
        else:
            _, K = phi.shape
            sgn = sgn.unsqueeze(-1)
            v = phi.view(1, -1, K, 1, 1).to(torch.float32)
            u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

        # FFT convolution
        v_fft = torch.fft.rfft(v, n=n, dim=1)
        U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
        U_fft = torch.fft.rfft(U, n=n, dim=1)
        U_conv = torch.fft.irfft(v_fft * U_fft, n=n, dim=1)[:, :seq_len]

        U_plus, U_minus = torch.unbind(U_conv, dim=-1)
        U_minus = U_minus * sgn

        return U_plus.to(input_dtype), U_minus.to(input_dtype)
    else:
        # Only compute positive branch (Hankel-L)
        if use_approx:
            _, d_out = phi.shape
            v = phi.view(1, -1, d_out).to(torch.float32)
        else:
            _, K = phi.shape
            v = phi.view(1, -1, K, 1).to(torch.float32)
            u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

        v_fft = torch.fft.rfft(v, n=n, dim=1)
        U_fft = torch.fft.rfft(u.to(torch.float32), n=n, dim=1)
        U_plus = torch.fft.irfft(v_fft * U_fft, n=n, dim=1)[:, :seq_len]

        return U_plus.to(input_dtype), None
