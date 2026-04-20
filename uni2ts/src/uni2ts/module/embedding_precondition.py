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
Embedding preconditioning module (stub).

This module provides a placeholder EmbeddingPrecondition class.
The actual implementation was removed in the spectral_non_precond branch.
"""

from typing import Optional

import torch
from jaxtyping import Bool, Float, Int
from torch import nn


class EmbeddingPrecondition(nn.Module):
    """
    Placeholder for embedding preconditioning module.

    This is a stub implementation that acts as an identity transform.
    The actual preconditioning logic was removed in the spectral_non_precond branch.

    Args:
        degree: Polynomial degree (unused in stub)
        polynomial_type: Type of polynomial basis (unused in stub)
    """

    def __init__(
        self,
        degree: int = 5,
        polynomial_type: str = "chebyshev",
    ):
        super().__init__()
        self.degree = degree
        self.polynomial_type = polynomial_type

    def forward(
        self,
        x: Float[torch.Tensor, "*batch seq_len d_model"],
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
        target_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]] = None,
    ) -> Float[torch.Tensor, "*batch seq_len d_model"]:
        """Identity transform - returns input unchanged."""
        return x

    def reverse(
        self,
        x: Float[torch.Tensor, "*batch seq_len d_model"],
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
    ) -> Float[torch.Tensor, "*batch seq_len d_model"]:
        """Identity reverse transform - returns input unchanged."""
        return x
