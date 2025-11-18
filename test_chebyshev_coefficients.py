#!/usr/bin/env python3
"""
Test script to verify if the Chebyshev coefficient extraction is correct.

According to the paper:
- We need the n-th MONIC Chebyshev polynomial Mₙ(x) = Tₙ(x) / 2^(n-1)
- The coefficients should be in POWER BASIS (standard polynomial form)
- For the preconditioning formula: ỹₜ = yₜ - Σᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ
  We need coefficients [c₁, c₂, ..., cₙ] from the monic polynomial
"""

import numpy as np
from numpy.polynomial import chebyshev

def current_implementation(n):
    """Current implementation from precondition.py"""
    cheb = chebyshev.Chebyshev.basis(n)
    coeffs = cheb.coef
    return coeffs[1:]  # Skip c₀

def correct_monic_coefficients(n):
    """
    Correct implementation:
    1. Get Chebyshev polynomial Tₙ(x) in power basis
    2. Divide by 2^(n-1) to get monic form Mₙ(x)
    3. Extract coefficients [c₁, ..., cₙ] (excluding leading coefficient which should be 1)
    """
    # Create Chebyshev polynomial basis
    cheb_basis = chebyshev.Chebyshev.basis(n)

    # Convert to power basis (standard polynomial)
    power_poly = cheb_basis.convert(kind=np.polynomial.Polynomial)

    # Get coefficients in power basis (ascending order: c₀, c₁, ..., cₙ)
    power_coeffs = power_poly.coef

    # Make monic by dividing by leading coefficient (should be 2^(n-1) for Chebyshev)
    leading_coeff = power_coeffs[-1]  # Highest degree coefficient
    monic_coeffs = power_coeffs / leading_coeff

    # Return c₁, ..., cₙ (excluding c₀)
    # But for the preconditioning formula, we need coefficients in reverse order
    # Actually, let me check what order is expected...

    return monic_coeffs[1:], leading_coeff, power_coeffs

def test_chebyshev_polynomials():
    """Test for small degrees to verify correctness"""

    print("="*80)
    print("Testing Chebyshev Polynomial Coefficient Extraction")
    print("="*80)
    print()

    for n in [2, 3, 5]:
        print(f"\n{'='*60}")
        print(f"Degree n = {n}")
        print(f"{'='*60}")

        # Current implementation
        current_coeffs = current_implementation(n)
        print(f"\nCurrent implementation coefficients:")
        print(f"  Raw output: {current_coeffs}")
        print(f"  Length: {len(current_coeffs)}")

        # Correct monic coefficients
        monic_coeffs, leading, power_coeffs = correct_monic_coefficients(n)
        print(f"\nCorrect monic coefficients:")
        print(f"  Power basis coeffs (full): {power_coeffs}")
        print(f"  Leading coefficient: {leading}")
        print(f"  Expected 2^(n-1): {2**(n-1)}")
        print(f"  Monic coeffs (c₁...cₙ): {monic_coeffs}")
        print(f"  Monic coeffs (c₀...cₙ): {power_coeffs / leading}")

        # Check if monic (leading coefficient should be 1)
        is_monic = np.isclose(power_coeffs[-1] / leading, 1.0)
        print(f"\n  Is monic? {is_monic}")

        # Compare
        print(f"\n  Current vs Correct match? {np.allclose(current_coeffs, monic_coeffs)}")

        # What's in Chebyshev basis vs power basis?
        cheb_basis = chebyshev.Chebyshev.basis(n)
        print(f"\n  Chebyshev basis coeffs: {cheb_basis.coef}")
        print(f"  (These are weights on [T₀, T₁, ..., Tₙ])")

        # Manual calculation for verification
        # T₂(x) = 2x² - 1, so M₂(x) = T₂(x)/2 = x² - 1/2
        # Coefficients: [-1/2, 0, 1]  -> skip c₀: [0, 1]
        if n == 2:
            print(f"\n  Manual check for n=2:")
            print(f"    T₂(x) = 2x² - 1")
            print(f"    M₂(x) = T₂(x)/2 = x² - 1/2")
            print(f"    Power coeffs: [-1/2, 0, 1]")
            print(f"    c₁, c₂: [0, 1] (but we use c₁ for coefficient of x¹)")
            manual = np.array([-0.5, 0, 1])
            manual_monic = manual / manual[-1]
            print(f"    Manual monic (c₀, c₁, c₂): {manual_monic}")
            print(f"    Manual (c₁, c₂): {manual_monic[1:]}")

        # Check norm as mentioned in Lemma 3.2
        l1_norm = np.sum(np.abs(monic_coeffs))
        expected_bound = 2**(0.3 * n)
        print(f"\n  L1 norm of coefficients: {l1_norm:.6f}")
        print(f"  Paper's bound (2^(0.3n)): {expected_bound:.6f}")
        print(f"  Within bound? {l1_norm <= expected_bound}")

    # Test what the actual polynomial evaluations look like
    print(f"\n\n{'='*80}")
    print("Polynomial Evaluation Test")
    print(f"{'='*80}")

    n = 5
    x_test = np.array([0.5, 0.7, 0.9])

    # Using current implementation
    current_coeffs = current_implementation(n)

    # Using correct monic
    monic_coeffs, leading, power_coeffs = correct_monic_coefficients(n)
    power_basis_full = power_coeffs / leading  # Monic form

    print(f"\nFor n={n}, evaluating at x={x_test}")

    # Evaluate using numpy polynomial
    cheb_basis = chebyshev.Chebyshev.basis(n)
    monic_cheb = cheb_basis / (2**(n-1))

    print(f"\nMonic Chebyshev Mₙ(x) = Tₙ(x)/2^(n-1):")
    for x in x_test:
        val = monic_cheb(x)
        print(f"  M_{n}({x:.1f}) = {val:.6f}")

        # Check if |val| <= 2^(-n+2) as per Lemma 3.1 (when |arg(x)| is small)
        bound = 2**(-n+2)
        print(f"    Paper's bound (2^(-n+2)): {bound:.6f}, Within? {abs(val) <= bound}")


if __name__ == "__main__":
    test_chebyshev_polynomials()
