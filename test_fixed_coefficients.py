#!/usr/bin/env python3
"""
Test the fixed coefficient extraction from precondition.py
"""
import sys
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts')

from src.uni2ts.transform.precondition import PolynomialPrecondition
import numpy as np

def test_fixed_coefficients():
    print("="*80)
    print("Testing Fixed Coefficient Extraction")
    print("="*80)

    # Test Chebyshev
    print("\n" + "="*60)
    print("CHEBYSHEV POLYNOMIALS")
    print("="*60)

    for degree in [2, 3, 5, 7]:
        precond = PolynomialPrecondition(
            polynomial_type="chebyshev",
            degree=degree,
            enabled=True
        )

        coeffs = precond.coeffs
        print(f"\nDegree {degree}:")
        print(f"  Coefficients: {coeffs}")
        print(f"  Length: {len(coeffs)} (expected: {degree})")
        print(f"  L1 norm: {np.sum(np.abs(coeffs)):.6f}")
        print(f"  Max abs coeff: {np.max(np.abs(coeffs)):.6f}")

        # Verify against manual calculations
        if degree == 2:
            # T₂(x) = 2x² - 1, M₂(x) = x² - 0.5
            # Coeffs: [-0.5, 0, 1] -> [0, 1]
            expected = np.array([0, 1])
            match = np.allclose(coeffs, expected)
            print(f"  Expected: {expected}")
            print(f"  Match? {match} {'✓' if match else '✗'}")

        elif degree == 3:
            # T₃(x) = 4x³ - 3x, M₃(x) = x³ - 0.75x
            # Coeffs: [0, -0.75, 0, 1] -> [-0.75, 0, 1]
            expected = np.array([-0.75, 0, 1])
            match = np.allclose(coeffs, expected)
            print(f"  Expected: {expected}")
            print(f"  Match? {match} {'✓' if match else '✗'}")

        elif degree == 5:
            # T₅(x) = 16x⁵ - 20x³ + 5x, M₅(x) = x⁵ - 1.25x³ + 0.3125x
            # Coeffs: [0, 0.3125, 0, -1.25, 0, 1] -> [0.3125, 0, -1.25, 0, 1]
            expected = np.array([0.3125, 0, -1.25, 0, 1])
            match = np.allclose(coeffs, expected)
            print(f"  Expected: {expected}")
            print(f"  Match? {match} {'✓' if match else '✗'}")

    # Test Legendre
    print("\n" + "="*60)
    print("LEGENDRE POLYNOMIALS")
    print("="*60)

    for degree in [2, 3, 5, 7]:
        precond = PolynomialPrecondition(
            polynomial_type="legendre",
            degree=degree,
            enabled=True
        )

        coeffs = precond.coeffs
        print(f"\nDegree {degree}:")
        print(f"  Coefficients: {coeffs}")
        print(f"  Length: {len(coeffs)} (expected: {degree})")
        print(f"  L1 norm: {np.sum(np.abs(coeffs)):.6f}")
        print(f"  Max abs coeff: {np.max(np.abs(coeffs)):.6f}")

        # Verify against manual calculations
        if degree == 2:
            # P₂(x) = (3x² - 1)/2, leading coeff = 3/2
            # M₂(x) = x² - 1/3
            # Coeffs: [-1/3, 0, 1] -> [0, 1]
            expected = np.array([0, 1])
            match = np.allclose(coeffs, expected)
            print(f"  Expected: {expected}")
            print(f"  Match? {match} {'✓' if match else '✗'}")

    # Test coefficient growth (from Lemma 3.2)
    print("\n" + "="*60)
    print("COEFFICIENT GROWTH TEST (Lemma 3.2)")
    print("="*60)
    print("\nPaper states: max|cᵢ| ≤ 2^(0.3n)")

    for degree in [2, 3, 5, 7, 10]:
        precond_cheb = PolynomialPrecondition(
            polynomial_type="chebyshev",
            degree=degree,
            enabled=True
        )
        precond_leg = PolynomialPrecondition(
            polynomial_type="legendre",
            degree=degree,
            enabled=True
        )

        max_cheb = np.max(np.abs(precond_cheb.coeffs))
        max_leg = np.max(np.abs(precond_leg.coeffs))
        bound = 2**(0.3 * degree)

        print(f"\nDegree {degree}:")
        print(f"  Bound 2^(0.3n): {bound:.6f}")
        print(f"  Chebyshev max: {max_cheb:.6f} {'✓' if max_cheb <= bound else '✗'}")
        print(f"  Legendre max:  {max_leg:.6f} {'✓' if max_leg <= bound else '✗'}")

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)

if __name__ == "__main__":
    test_fixed_coefficients()
