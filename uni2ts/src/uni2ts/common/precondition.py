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

VALID_POLYNOMIAL_TYPES = ("chebyshev", "legendre", "lyapunov", "l2_optimized")


def compute_polynomial_coefficients(
    polynomial_type: str, degree: int, *, reg_lambda: float = 1.0
) -> np.ndarray:
    """
    Compute monic polynomial coefficients for Universal Sequence Preconditioning.

    Returns coefficients in the order [c1, c2, ..., cN] where N = degree.

    Supported types:
      - "chebyshev": Monic Chebyshev polynomials (minimax optimal on [-1,1])
      - "legendre": Monic Legendre polynomials
      - "lyapunov": Optimized to jointly minimize max|p(z)| and noise gain
                    (Lyapunov equation P[0,0]) with trade-off controlled by reg_lambda
      - "l2_optimized": Optimized to jointly minimize max|p(z)| and L2 norm of
                        coefficients with trade-off controlled by reg_lambda
    """
    if polynomial_type not in VALID_POLYNOMIAL_TYPES:
        raise ValueError(
            f"Unknown polynomial type: {polynomial_type}. "
            f"Must be one of {VALID_POLYNOMIAL_TYPES}."
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
    elif polynomial_type == "legendre":
        return _legendre_coefficients(degree)
    elif polynomial_type == "lyapunov":
        return _optimized_coefficients(degree, reg_type="lyapunov", reg_lambda=reg_lambda)
    else:  # l2_optimized
        return _optimized_coefficients(degree, reg_type="l2", reg_lambda=reg_lambda)


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


def _solve_lyapunov_doubling(M, Q, max_iter=25):
    """Solve discrete Lyapunov equation P - M^T P M = Q via doubling."""
    import torch

    P = Q
    Mk = M
    for _ in range(max_iter):
        P = P + Mk.t() @ P @ Mk
        Mk = Mk @ Mk
        if not torch.isfinite(P).all():
            break
    return P


def _optimized_coefficients(
    n: int,
    reg_type: str = "lyapunov",
    reg_lambda: float = 1.0,
    num_iters: int = 15,
    grid_size: int = 500,
) -> np.ndarray:
    """
    Optimize monic polynomial coefficients by jointly minimizing
    max|p(z)| on [-1,1] and a regularization term via L-BFGS.

    reg_type: "lyapunov" penalizes noise gain (companion matrix Lyapunov P[0,0])
              "l2" penalizes sum of squared coefficients
    reg_lambda: weight on regularization term (higher = more regularization)
    """
    import torch

    z_grid = torch.linspace(-1, 1, grid_size, dtype=torch.float64)
    c = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([c], lr=0.1, max_iter=20, history_size=10)

    eye_n1 = torch.eye(n - 1, dtype=torch.float64) if n > 1 else None

    def closure():
        optimizer.zero_grad()
        # Build companion matrix
        M = torch.zeros((n, n), dtype=torch.float64)
        M[0, :] = -c
        if eye_n1 is not None:
            M[1:, :-1] = eye_n1

        # USP loss: max|p(z)| on [-1,1]
        z_n = z_grid**n
        powers = torch.stack([z_grid ** (n - i - 1) for i in range(n)])
        usp_loss = torch.max(torch.abs(z_n + c @ powers))

        # Regularization
        if reg_type == "lyapunov":
            eigvals = torch.linalg.eigvals(M)
            rho = torch.max(torch.abs(eigvals))
            if rho >= 0.98:
                reg_term = 1e8 * (rho - 0.98) + 1e6
            else:
                B = torch.zeros((n, 1), dtype=torch.float64)
                B[0, 0] = 1.0
                P = _solve_lyapunov_doubling(M, B @ B.t())
                reg_term = P[0, 0]
                if not torch.isfinite(reg_term):
                    reg_term = torch.tensor(1e12, dtype=torch.float64)
        else:  # l2
            reg_term = torch.sum(c**2)

        loss = usp_loss + reg_lambda * reg_term
        loss.backward()
        return loss

    for _ in range(num_iters):
        optimizer.step(closure)

    return c.detach().numpy()
