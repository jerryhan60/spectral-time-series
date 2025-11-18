"""
Test to verify correct coefficient ordering in preconditioning.

This test verifies:
1. Coefficients are extracted in the correct order [c₁, c₂, ..., cₙ]
2. Convolution applies c₁ to y_{t-1}, c₂ to y_{t-2}, etc.
3. Forward and reverse transformations are proper inverses
"""

import numpy as np
import sys
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')

from uni2ts.transform.precondition import PolynomialPrecondition

def test_chebyshev_coefficients():
    """Test Chebyshev coefficient extraction."""
    print("\n=== Testing Chebyshev Coefficients ===")

    # Test n=2: M₂(x) = x² - 0.5
    # p^c_2(x) = c₀x² + c₁x + c₂ = 1·x² + 0·x + (-0.5)
    # So c₁=0, c₂=-0.5
    precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=2, enabled=True)
    coeffs = precond.coeffs
    print(f"n=2: coeffs = {coeffs}")
    print(f"Expected: [0, -0.5], Got: {coeffs}")
    assert len(coeffs) == 2, f"Expected length 2, got {len(coeffs)}"
    assert np.isclose(coeffs[0], 0.0, atol=1e-10), f"c₁ should be 0, got {coeffs[0]}"
    assert np.isclose(coeffs[1], -0.5), f"c₂ should be -0.5, got {coeffs[1]}"
    print("✓ n=2 passed")

    # Test n=3: M₃(x) = x³ - 0.75x
    # p^c_3(x) = c₀x³ + c₁x² + c₂x + c₃ = 1·x³ + 0·x² + (-0.75)·x + 0
    # So c₁=0, c₂=-0.75, c₃=0
    precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=3, enabled=True)
    coeffs = precond.coeffs
    print(f"\nn=3: coeffs = {coeffs}")
    print(f"Expected: [0, -0.75, 0], Got: {coeffs}")
    assert len(coeffs) == 3, f"Expected length 3, got {len(coeffs)}"
    assert np.isclose(coeffs[0], 0.0, atol=1e-10), f"c₁ should be 0, got {coeffs[0]}"
    assert np.isclose(coeffs[1], -0.75), f"c₂ should be -0.75, got {coeffs[1]}"
    assert np.isclose(coeffs[2], 0.0, atol=1e-10), f"c₃ should be 0, got {coeffs[2]}"
    print("✓ n=3 passed")

    # Test n=5: M₅(x) = x⁵ - 1.25x³ + 0.3125x
    # p^c_5(x) = c₀x⁵ + c₁x⁴ + c₂x³ + c₃x² + c₄x + c₅
    #          = 1·x⁵ + 0·x⁴ + (-1.25)·x³ + 0·x² + 0.3125·x + 0
    # Expected: [c₁, c₂, c₃, c₄, c₅] = [0, -1.25, 0, 0.3125, 0]
    precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=5, enabled=True)
    coeffs = precond.coeffs
    print(f"\nn=5: coeffs = {coeffs}")
    print(f"Expected: [0, -1.25, 0, 0.3125, 0]")
    assert len(coeffs) == 5, f"Expected length 5, got {len(coeffs)}"
    # c₁=0, c₂=-1.25, c₃=0, c₄=0.3125, c₅=0
    assert np.isclose(coeffs[0], 0.0, atol=1e-10), f"c₁ should be 0, got {coeffs[0]}"
    assert np.isclose(coeffs[1], -1.25, atol=0.01), f"c₂ should be -1.25, got {coeffs[1]}"
    assert np.isclose(coeffs[2], 0.0, atol=1e-10), f"c₃ should be 0, got {coeffs[2]}"
    assert np.isclose(coeffs[3], 0.3125, atol=0.01), f"c₄ should be 0.3125, got {coeffs[3]}"
    assert np.isclose(coeffs[4], 0.0, atol=1e-10), f"c₅ should be 0, got {coeffs[4]}"
    print("✓ n=5 passed")

def test_convolution_ordering():
    """Test that convolution applies coefficients in the correct order."""
    print("\n=== Testing Convolution Ordering ===")

    # Create a simple test sequence
    sequence = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Test with n=2, coeffs = [c₁, c₂] = [0.5, 0.3]
    # For t=2 (index 2): ỹ₂ = y₂ + c₁·y₁ + c₂·y₀ = 3 + 0.5·2 + 0.3·1 = 3 + 1 + 0.3 = 4.3
    # For t=3 (index 3): ỹ₃ = y₃ + c₁·y₂ + c₂·y₁ = 4 + 0.5·3 + 0.3·2 = 4 + 1.5 + 0.6 = 6.1

    # Create a custom preconditioner with known coefficients
    precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=2, enabled=True)
    # Override coefficients with test values
    precond.coeffs = np.array([0.5, 0.3])

    result = precond._apply_convolution(sequence, precond.coeffs)

    print(f"Original sequence: {sequence}")
    print(f"Coefficients: c₁={precond.coeffs[0]}, c₂={precond.coeffs[1]}")
    print(f"Preconditioned sequence: {result}")

    # Check t < n (should be unchanged)
    assert result[0] == sequence[0], f"y₀ should be unchanged"
    assert result[1] == sequence[1], f"y₁ should be unchanged"

    # Check t >= n
    # t=2: ỹ₂ = 3 + 0.5·2 + 0.3·1 = 4.3
    expected_t2 = 3.0 + 0.5 * 2.0 + 0.3 * 1.0
    print(f"  t=2: y₂ + c₁·y₁ + c₂·y₀ = 3 + 0.5·2 + 0.3·1 = {expected_t2}")
    print(f"       Got: {result[2]}")
    assert np.isclose(result[2], expected_t2), f"t=2: Expected {expected_t2}, got {result[2]}"

    # t=3: ỹ₃ = 4 + 0.5·3 + 0.3·2 = 6.1
    expected_t3 = 4.0 + 0.5 * 3.0 + 0.3 * 2.0
    print(f"  t=3: y₃ + c₁·y₂ + c₂·y₁ = 4 + 0.5·3 + 0.3·2 = {expected_t3}")
    print(f"       Got: {result[3]}")
    assert np.isclose(result[3], expected_t3), f"t=3: Expected {expected_t3}, got {result[3]}"

    print("✓ Convolution ordering is correct: c₁ → y_{t-1}, c₂ → y_{t-2}")

def test_differencing():
    """Test that differencing works correctly with c₁=-1, c₂=0."""
    print("\n=== Testing Differencing (c₁=-1) ===")

    sequence = np.array([1.0, 3.0, 6.0, 10.0, 15.0])

    # Create preconditioner with custom differencing coefficients
    # For differencing: ỹₜ = yₜ + (-1)·yₜ₋₁ = yₜ - yₜ₋₁
    precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=1, enabled=True)
    # Override with differencing coefficient
    precond.coeffs = np.array([-1.0])

    result = precond._apply_convolution(sequence, precond.coeffs)

    print(f"Original: {sequence}")
    print(f"Differenced: {result}")
    print(f"Expected: [1.0, 2.0, 3.0, 4.0, 5.0] (first order differences)")

    # Check differencing: ỹₜ = yₜ - yₜ₋₁
    assert result[0] == sequence[0], "First element unchanged"
    for i in range(1, len(sequence)):
        expected = sequence[i] - sequence[i-1]
        print(f"  t={i}: y_{i} - y_{i-1} = {sequence[i]} - {sequence[i-1]} = {expected}, got {result[i]}")
        assert np.isclose(result[i], expected), f"Differencing failed at t={i}"

    print("✓ Differencing works correctly")

def test_forward_reverse_inverse():
    """Test that forward and reverse are proper inverses."""
    print("\n=== Testing Forward/Reverse Inverse Property ===")

    # Create random sequence
    np.random.seed(42)
    original_sequence = np.random.randn(100)

    # Test with Chebyshev degree 5
    precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=5, enabled=True)

    # Apply forward transformation
    data_entry = {"target": original_sequence.copy()}
    data_entry = precond(data_entry)
    preconditioned = data_entry["target"]
    coeffs = data_entry["precondition_coeffs"]

    print(f"Original sequence shape: {original_sequence.shape}")
    print(f"Coefficients: {coeffs}")

    # Apply reverse transformation
    from uni2ts.transform.precondition import ReversePrecondition
    reverse = ReversePrecondition(enabled=True)
    data_entry = reverse(data_entry)
    restored = data_entry["target"]

    print(f"Restored sequence shape: {restored.shape}")

    # Check that we get back the original
    max_error = np.max(np.abs(original_sequence - restored))
    mean_error = np.mean(np.abs(original_sequence - restored))

    print(f"Max error: {max_error}")
    print(f"Mean error: {mean_error}")

    assert np.allclose(original_sequence, restored, atol=1e-10), \
        f"Forward-Reverse should be identity! Max error: {max_error}"

    print("✓ Forward and reverse are proper inverses")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Coefficient Ordering in Preconditioning")
    print("=" * 60)

    try:
        test_chebyshev_coefficients()
        test_convolution_ordering()
        test_differencing()
        test_forward_reverse_inverse()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        return 1
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1

if __name__ == "__main__":
    exit(main())
