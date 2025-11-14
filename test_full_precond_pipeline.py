#!/usr/bin/env python3
"""
Test the full preconditioning evaluation pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "uni2ts" / "src"))

import numpy as np
import torch
from uni2ts.transform import PolynomialPrecondition, ReversePrecondition

def test_reversal_with_context():
    """Test that reversal works correctly with context."""
    print("="*60)
    print("Testing Preconditioning Reversal with Context")
    print("="*60)

    # Create test data
    np.random.seed(42)
    original_data = np.random.randn(100)

    print(f"\nOriginal data shape: {original_data.shape}")
    print(f"Original data: mean={original_data.mean():.4f}, std={original_data.std():.4f}")

    # Apply preconditioning
    preconditioner = PolynomialPrecondition(
        polynomial_type="chebyshev",
        degree=5,
        target_field="target",
        enabled=True,
        store_original=True,
    )

    data_entry = {"target": original_data.copy()}
    preconditioned_entry = preconditioner(data_entry)
    preconditioned_data = preconditioned_entry["target"]

    print(f"\nPreconditioned data: mean={preconditioned_data.mean():.4f}, std={preconditioned_data.std():.4f}")

    # Simulate prediction: take last 50 points as "predictions" in preconditioned space
    context_length = 50
    prediction_length = 50

    original_context = original_data[-context_length-prediction_length:-prediction_length]
    preconditioned_predictions = preconditioned_data[-prediction_length:]

    print(f"\nContext length: {len(original_context)}")
    print(f"Prediction length: {len(preconditioned_predictions)}")
    print(f"Context (last 5 original values): {original_context[-5:]}")

    # Now reverse using context (simulating what the predictor does)
    degree = 5
    context = original_context[-degree:]  # Last 5 points for reversal

    # Prepend context to predictions
    full_sequence = np.concatenate([context, preconditioned_predictions])

    print(f"\nFull sequence shape: {full_sequence.shape}")

    # Apply reversal
    reverse_preconditioner = ReversePrecondition(
        target_field="target",
        enabled=True,
    )

    reversal_data = {
        "target": full_sequence,
        "precondition_coeffs": preconditioner.coeffs,
        "precondition_degree": 5,
        "precondition_type": "chebyshev",
        "precondition_enabled": True,
    }

    reversed_data = reverse_preconditioner(reversal_data)
    reversed_full = reversed_data["target"]

    # Extract predictions (remove context)
    reversed_predictions = reversed_full[degree:]

    print(f"Reversed predictions shape: {reversed_predictions.shape}")

    # Compare with ground truth
    ground_truth = original_data[-prediction_length:]

    difference = np.abs(ground_truth - reversed_predictions)
    max_diff = difference.max()
    mean_diff = difference.mean()

    print(f"\nComparison with ground truth:")
    print(f"  Ground truth mean: {ground_truth.mean():.4f}")
    print(f"  Reversed pred mean: {reversed_predictions.mean():.4f}")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-10:
        print("\n✓ Reversal with context is PERFECT! (max diff < 1e-10)")
        return True
    elif max_diff < 1e-6:
        print(f"\n✓ Reversal with context is GOOD! (max diff < 1e-6)")
        return True
    else:
        print(f"\n✗ Reversal has significant errors! (max diff = {max_diff:.2e})")
        return False


if __name__ == "__main__":
    test_passed = test_reversal_with_context()

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Reversal with context test: {'✓ PASS' if test_passed else '✗ FAIL'}")

    if test_passed:
        print("\n✓ Test passed! Reversal implementation is correct.")
        sys.exit(0)
    else:
        print("\n✗ Test failed! Check reversal implementation.")
        sys.exit(1)
