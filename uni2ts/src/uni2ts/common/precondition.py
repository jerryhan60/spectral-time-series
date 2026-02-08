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

from __future__ import annotations

import warnings

import numpy as np


def compute_polynomial_coefficients(polynomial_type: str, degree: int) -> np.ndarray:
    """
    Compute monic polynomial coefficients for Universal Sequence Preconditioning.

    Returns coefficients in the order [c1, c2, ..., cN] where N = degree.
    """
    if polynomial_type not in ("chebyshev", "legendre"):
        raise ValueError(
            f"Unknown polynomial type: {polynomial_type}. "
            "Must be 'chebyshev' or 'legendre'."
        )
    if degree < 1:
        raise ValueError(f"Degree must be >= 1, got {degree}")
    if degree > 10:
        warnings.warn(
            f"Degree {degree} > 10 may cause numerical instability. "
            "Paper recommends degree <= 10."
        )

    if polynomial_type == "chebyshev":
        return _chebyshev_coefficients(degree)
    return _legendre_coefficients(degree)


def _chebyshev_coefficients(n: int) -> np.ndarray:
    """
    Compute monic Chebyshev polynomial coefficients of degree n.

    Returns coefficients [c1, c2, ..., cN] in power basis.
    """
    from numpy.polynomial import chebyshev, polynomial

    cheb = chebyshev.Chebyshev.basis(n)
    power_poly = cheb.convert(kind=polynomial.Polynomial)
    power_coeffs = power_poly.coef
    leading_coeff = power_coeffs[-1]
    monic_coeffs = power_coeffs / leading_coeff
    return monic_coeffs[:-1][::-1].copy()


def _legendre_coefficients(n: int) -> np.ndarray:
    """
    Compute monic Legendre polynomial coefficients of degree n.

    Returns coefficients [c1, c2, ..., cN] in power basis.
    """
    from numpy.polynomial import legendre, polynomial

    leg = legendre.Legendre.basis(n)
    power_poly = leg.convert(kind=polynomial.Polynomial)
    power_coeffs = power_poly.coef
    leading_coeff = power_coeffs[-1]
    monic_coeffs = power_coeffs / leading_coeff
    return monic_coeffs[:-1][::-1].copy()
