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
Polynomial preconditioning transform (stub).

This module provides a placeholder PolynomialPrecondition transform.
The actual implementation was removed in the spectral_non_precond branch.
"""

from typing import Any
from ._base import Transformation


class PolynomialPrecondition(Transformation):
    """
    Placeholder for polynomial preconditioning transform.

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
        **kwargs,
    ):
        self.degree = degree
        self.polynomial_type = polynomial_type

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Identity transform - returns data unchanged."""
        return data


class ReversePrecondition(Transformation):
    """
    Placeholder for reverse preconditioning transform.

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
        **kwargs,
    ):
        self.degree = degree
        self.polynomial_type = polynomial_type

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Identity transform - returns data unchanged."""
        return data
