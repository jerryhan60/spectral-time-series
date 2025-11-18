#!/usr/bin/env python3
"""
Test that forward preconditioning and reverse operations are inverses.
"""
import sys
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts')

from src.uni2ts.transform.precondition import PolynomialPrecondition, ReversePrecondition
import numpy as np

def test_forward_reverse_inverses():
    """Test that forward and reverse are true inverses"""

    print("="*80)
    print("Testing Forward/Reverse Preconditioning Are Inverses")
    print("="*80)

    # Create test data
    np.random.seed(42)
    test_sequence = np.random.randn(100)

    for poly_type in ["chebyshev", "legendre"]:
        print(f"\n{'='*60}")
        print(f"{poly_type.upper()} POLYNOMIAL")
        print(f"{'='*60}")

        for degree in [2, 3, 5]:
            # Create preconditioner
            precond = PolynomialPrecondition(
                polynomial_type=poly_type,
                degree=degree,
                enabled=True
            )

            # Create reverser
            reverse = ReversePrecondition(enabled=True)

            # Apply forward preconditioning
            data_entry = {"target": test_sequence.copy()}
            precond_entry = precond(data_entry)
            precond_seq = precond_entry["target"]

            # Apply reverse
            reverse_entry = reverse(precond_entry)
            reversed_seq = reverse_entry["target"]

            # Check if we get back original
            max_error = np.max(np.abs(reversed_seq - test_sequence))
            is_inverse = max_error < 1e-10

            print(f"\nDegree {degree}:")
            print(f"  Coefficients: {precond.coeffs}")
            print(f"  Max reconstruction error: {max_error:.2e}")
            print(f"  Is inverse? {is_inverse} {'✓' if is_inverse else '✗'}")

            if not is_inverse:
                print(f"  ERROR: Forward and reverse are NOT inverses!")
                print(f"  First 10 original:  {test_sequence[:10]}")
                print(f"  First 10 precond:   {precond_seq[:10]}")
                print(f"  First 10 reversed:  {reversed_seq[:10]}")

    # Test differencing specifically
    print(f"\n{'='*60}")
    print("DIFFERENCING TEST (n=2, c₁=-1)")
    print(f"{'='*60}")

    # For degree 2 Chebyshev, c₁ should be 0 (not -1!)
    # But we can manually test differencing
    test_seq = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0])

    # Manual differencing: y_t - y_{t-1}
    diff_manual = np.diff(test_seq)

    print(f"\nOriginal sequence: {test_seq}")
    print(f"Manual differencing: {diff_manual}")

    # For true differencing, we'd need coefficients [0, -1] for degree 2
    # But Chebyshev degree 2 gives [0, 1]
    # Let's verify what degree 2 Chebyshev actually does
    precond = PolynomialPrecondition(
        polynomial_type="chebyshev",
        degree=2,
        enabled=True
    )
    print(f"\nChebyshev deg 2 coefficients: {precond.coeffs}")
    print(f"Note: For differencing we'd need c₁=-1, but Chebyshev gives c₁=0")

    # Apply it anyway
    data_entry = {"target": test_seq.copy()}
    precond_entry = precond(data_entry)
    cheb2_result = precond_entry["target"]

    print(f"Chebyshev deg 2 result: {cheb2_result}")
    print(f"Formula: ỹ_t = y_t + 0·y_{t-1} + 1·y_{t-2} = y_t + y_{t-2}")

    # Check manually
    expected = test_seq.copy()
    expected[2:] = test_seq[2:] + 1 * test_seq[0:-2]  # y_t + y_{t-2}
    print(f"Expected: {expected}")
    print(f"Match? {np.allclose(cheb2_result, expected)}")

    print("\n" + "="*80)
    print("All forward/reverse tests completed!")
    print("="*80)

if __name__ == "__main__":
    test_forward_reverse_inverses()
