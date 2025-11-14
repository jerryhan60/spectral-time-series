#!/usr/bin/env python3
"""Test script to verify optimized preconditioning maintains numerical accuracy."""

import numpy as np
import time
from uni2ts.transform.precondition import PolynomialPrecondition


def original_apply_convolution(sequence, coeffs):
    """Original slow implementation for comparison."""
    n = len(coeffs)
    result = sequence.copy()

    for t in range(n, len(sequence)):
        weighted_sum = sum(
            coeffs[i-1] * sequence[t-i]
            for i in range(1, n+1)
        )
        result[t] = sequence[t] - weighted_sum

    return result


def test_correctness():
    """Test that optimized version produces identical results."""
    print("=" * 60)
    print("Testing Numerical Correctness")
    print("=" * 60)

    # Create test data
    np.random.seed(42)
    test_cases = [
        ("Short sequence", np.random.randn(100)),
        ("Long sequence", np.random.randn(10000)),
        ("Very long sequence", np.random.randn(100000)),
    ]

    # Test with different polynomial configurations
    configs = [
        ("Chebyshev degree 5", "chebyshev", 5),
        ("Chebyshev degree 10", "chebyshev", 10),
        ("Legendre degree 5", "legendre", 5),
    ]

    all_passed = True

    for config_name, poly_type, degree in configs:
        print(f"\n{config_name}:")

        # Create preconditioner
        precond = PolynomialPrecondition(
            polynomial_type=poly_type,
            degree=degree,
            enabled=True
        )

        for test_name, sequence in test_cases:
            # Compute with both methods
            result_original = original_apply_convolution(sequence, precond.coeffs)
            result_optimized = precond._apply_convolution(sequence, precond.coeffs)

            # Check equality
            max_diff = np.max(np.abs(result_original - result_optimized))

            if max_diff < 1e-10:
                print(f"  ✓ {test_name}: PASS (max diff: {max_diff:.2e})")
            else:
                print(f"  ✗ {test_name}: FAIL (max diff: {max_diff:.2e})")
                all_passed = False

    return all_passed


def test_performance():
    """Benchmark performance improvement."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    np.random.seed(42)

    # Create preconditioner
    precond = PolynomialPrecondition(
        polynomial_type="chebyshev",
        degree=5,
        enabled=True
    )

    # Test on different sequence lengths
    lengths = [1000, 10000, 100000]

    for length in lengths:
        sequence = np.random.randn(length)

        # Time original implementation
        start = time.time()
        for _ in range(10):
            result_original = original_apply_convolution(sequence, precond.coeffs)
        time_original = (time.time() - start) / 10

        # Time optimized implementation
        start = time.time()
        for _ in range(10):
            result_optimized = precond._apply_convolution(sequence, precond.coeffs)
        time_optimized = (time.time() - start) / 10

        speedup = time_original / time_optimized

        print(f"\nSequence length: {length}")
        print(f"  Original:  {time_original*1000:.3f} ms")
        print(f"  Optimized: {time_optimized*1000:.3f} ms")
        print(f"  Speedup:   {speedup:.1f}x")


def test_full_pipeline():
    """Test full preconditioning pipeline."""
    print("\n" + "=" * 60)
    print("Full Pipeline Test")
    print("=" * 60)

    np.random.seed(42)

    # Create sample data entry (like what comes from dataset)
    data_entry = {
        "target": np.random.randn(10000).astype(np.float32)
    }

    # Apply preconditioning
    precond = PolynomialPrecondition(
        polynomial_type="chebyshev",
        degree=5,
        enabled=True
    )

    start = time.time()
    result = precond(data_entry)
    elapsed = time.time() - start

    print(f"✓ Pipeline processed successfully in {elapsed*1000:.3f} ms")
    print(f"  Input shape:  {data_entry['target'].shape}")
    print(f"  Output shape: {result['target'].shape}")
    print(f"  Metadata stored: coeffs={len(result['precondition_coeffs'])}, "
          f"degree={result['precondition_degree']}, type={result['precondition_type']}")


if __name__ == "__main__":
    print("Preconditioning Optimization Test Suite")
    print()

    # Run correctness tests
    correctness_passed = test_correctness()

    # Run performance tests
    test_performance()

    # Run full pipeline test
    test_full_pipeline()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if correctness_passed:
        print("✓ All correctness tests PASSED")
        print("✓ Optimization is safe to use")
    else:
        print("✗ Some correctness tests FAILED")
        print("✗ DO NOT use this optimization")
