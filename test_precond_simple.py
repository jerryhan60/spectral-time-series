#!/usr/bin/env python3
"""
Simple test to verify preconditioning reversal works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "uni2ts" / "src"))

import numpy as np
from uni2ts.transform import PolynomialPrecondition, ReversePrecondition

def test_precondition_reversal():
    """Test that preconditioning and reversal are inverses."""
    print("="*60)
    print("Testing Preconditioning and Reversal")
    print("="*60)

    # Create test data
    np.random.seed(42)
    original_data = np.random.randn(100)

    print(f"\nOriginal data: mean={original_data.mean():.4f}, std={original_data.std():.4f}")

    # Apply preconditioning
    preconditioner = PolynomialPrecondition(
        polynomial_type="chebyshev",
        degree=5,
        target_field="target",
        enabled=True,
        store_original=True,
    )

    data_entry = {"target": original_data}
    preconditioned_entry = preconditioner(data_entry)
    preconditioned_data = preconditioned_entry["target"]

    print(f"Preconditioned data: mean={preconditioned_data.mean():.4f}, std={preconditioned_data.std():.4f}")

    # Apply reversal
    reverse_preconditioner = ReversePrecondition(
        target_field="target",
        enabled=True,
    )

    # Add metadata for reversal
    preconditioned_entry_for_reversal = {
        "target": preconditioned_data,
        "precondition_coeffs": preconditioner.coeffs,
        "precondition_degree": 5,
        "precondition_type": "chebyshev",
        "precondition_enabled": True,
    }

    reversed_entry = reverse_preconditioner(preconditioned_entry_for_reversal)
    reversed_data = reversed_entry["target"]

    print(f"Reversed data: mean={reversed_data.mean():.4f}, std={reversed_data.std():.4f}")

    # Check if reversal recovers original
    difference = np.abs(original_data - reversed_data)
    max_diff = difference.max()
    mean_diff = difference.mean()

    print(f"\nDifference between original and reversed:")
    print(f"  Max: {max_diff:.2e}")
    print(f"  Mean: {mean_diff:.2e}")

    if max_diff < 1e-10:
        print("\n✓ Reversal is accurate! (max diff < 1e-10)")
        return True
    else:
        print(f"\n✗ Reversal has errors! (max diff = {max_diff:.2e})")
        return False


if __name__ == "__main__":
    test_passed = test_precondition_reversal()

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Reversal test: {'✓ PASS' if test_passed else '✗ FAIL'}")

    if test_passed:
        print("\n✓ Test passed!")
        sys.exit(0)
    else:
        print("\n✗ Test failed!")
        sys.exit(1)
