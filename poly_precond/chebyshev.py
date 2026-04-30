"""Polynomial coefficient computation and residual filtering.

Computes the preconditioning residual r_t = sum_{k=1}^{d} c_k * x_{t-ks}
from Equation (1) of the paper, where c_k are the non-leading coefficients
of a degree-d monic polynomial and s is the stride (= patch size P).
"""

from __future__ import annotations

import numpy as np


def chebyshev_coefficients(degree: int) -> list[float]:
    """Monic Chebyshev polynomial coefficients [c_1, ..., c_d] in descending powers.

    The degree-d monic Chebyshev polynomial is T_d(x) / 2^{d-1}.  In the
    power basis this is x^d + c_1 x^{d-1} + ... + c_d.  We return
    [c_1, ..., c_d].

    Example (d=4): x^4 - x^2 + 1/8 => coefficients [0, -1, 0, 0.125].
    """
    from numpy.polynomial import chebyshev, polynomial

    cheb = chebyshev.Chebyshev.basis(degree)
    power_coeffs = cheb.convert(kind=polynomial.Polynomial).coef
    monic = power_coeffs / power_coeffs[-1]
    return monic[:-1][::-1].tolist()


def legendre_coefficients(degree: int) -> list[float]:
    """Monic Legendre polynomial coefficients [c_1, ..., c_d]."""
    from numpy.polynomial import legendre, polynomial

    leg = legendre.Legendre.basis(degree)
    power_coeffs = leg.convert(kind=polynomial.Polynomial).coef
    monic = power_coeffs / power_coeffs[-1]
    return monic[:-1][::-1].tolist()


def ema_coefficients(degree: int) -> list[float]:
    """Exponential moving average coefficients with alpha = 2/(d+1)."""
    alpha = 2.0 / (degree + 1)
    return [alpha * (1 - alpha) ** i for i in range(degree)]


def differencing_coefficients(degree: int) -> list[float]:
    """N-th order differencing via binomial expansion: (-1)^k C(d,k)."""
    from math import comb

    return [(-1) ** k * comb(degree, k) for k in range(1, degree + 1)]


def compute_residual(x: np.ndarray, coeffs: list[float], stride: int) -> np.ndarray:
    """Compute the preconditioning residual r_t = sum_{k=1}^{d} c_k * x_{t-ks}.

    This is the cross-patch information that the preconditioning channel
    injects into each patch embedding.

    Args:
        x: Input time series of shape (..., T).
        coeffs: Polynomial coefficients [c_1, ..., c_d].
        stride: Lag stride s (= patch size P in the paper).

    Returns:
        Residual array of same shape as x, zero-padded for t < d*s.
    """
    d = len(coeffs)
    T = x.shape[-1]
    r = np.zeros_like(x)
    for k in range(d):
        shift = (k + 1) * stride
        if shift >= T:
            break
        r[..., shift:] += coeffs[k] * x[..., : T - shift]
    return r
