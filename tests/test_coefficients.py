"""Tests for polynomial coefficient computation."""
import numpy as np
import pytest

from poly_precond.chebyshev import (
    chebyshev_coefficients,
    differencing_coefficients,
    ema_coefficients,
    legendre_coefficients,
)


def test_chebyshev_d4_coefficients():
    coeffs = chebyshev_coefficients(4)
    expected = [0, -1, 0, 0.125]
    np.testing.assert_allclose(coeffs, expected, atol=1e-12)


def test_chebyshev_degrees_2_to_8():
    for d in range(2, 9):
        coeffs = chebyshev_coefficients(d)
        assert len(coeffs) == d
        # Even-degree monic Chebyshev has zero c_1 (no x^{d-1} term)
        if d % 2 == 0:
            assert abs(coeffs[0]) < 1e-12, f"d={d}: leading coeff should be zero, got {coeffs[0]}"


def test_legendre_d4():
    coeffs = legendre_coefficients(4)
    assert len(coeffs) == 4
    # Monic: verify by reconstructing polynomial and checking leading coeff is 1
    # p(x) = x^4 + c1*x^3 + c2*x^2 + c3*x + c4
    # Evaluate at x=1: 1 + sum(coeffs) should equal P_4(1)/leading = 1/leading
    from numpy.polynomial import legendre, polynomial

    leg = legendre.Legendre.basis(4)
    power = leg.convert(kind=polynomial.Polynomial).coef
    monic = power / power[-1]
    np.testing.assert_allclose(coeffs, monic[:-1][::-1].tolist(), atol=1e-12)


def test_differencing_d4():
    coeffs = differencing_coefficients(4)
    expected = [-4, 6, -4, 1]
    assert coeffs == expected


def test_ema_d4():
    coeffs = ema_coefficients(4)
    assert len(coeffs) == 4
    # All EMA coefficients should be positive
    assert all(c > 0 for c in coeffs)
    # alpha = 2/(4+1) = 0.4
    alpha = 0.4
    expected = [alpha * (1 - alpha) ** i for i in range(4)]
    np.testing.assert_allclose(coeffs, expected, atol=1e-12)
