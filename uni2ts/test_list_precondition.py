#!/usr/bin/env python
"""
Quick test to verify preconditioning handles list of arrays.
This was the bug that caused the training error.
"""
import sys
import numpy as np

sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')

from uni2ts.transform.precondition import PolynomialPrecondition


def test_list_handling():
    """Test that preconditioning handles list of arrays."""
    print("Testing preconditioning with list of arrays...")

    # Create a list of time series (as returned by _flatten_data in dataset.py)
    target_list = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    ]

    precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=3)

    # This should NOT crash (it was crashing before the fix)
    data_entry = {"target": target_list}

    try:
        result = precond(data_entry)
        print("✓ Preconditioning succeeded on list of arrays")

        # Verify result is still a list
        assert isinstance(result["target"], list), "Result should be a list"
        assert len(result["target"]) == 3, "Result should have 3 series"

        # Verify each element is an array
        for i, ts in enumerate(result["target"]):
            assert isinstance(ts, np.ndarray), f"Series {i} should be ndarray"
            assert ts.shape == (10,), f"Series {i} should have shape (10,)"

        print("✓ Result structure is correct")

        # Verify preconditioning actually happened (first 3 values unchanged, rest modified)
        for i in range(3):
            original = target_list[i]
            preconditioned = result["target"][i]

            # First 'degree' values should be unchanged
            np.testing.assert_array_equal(
                preconditioned[:3],
                original[:3],
                err_msg=f"First 3 values of series {i} should be unchanged"
            )

            # Later values should be different
            assert not np.allclose(preconditioned[3:], original[3:]), \
                f"Later values of series {i} should be modified"

        print("✓ Preconditioning was applied correctly")
        print()
        print("All tests passed! ✓")
        return True

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_array_handling():
    """Test that regular array handling still works."""
    print("\nTesting preconditioning with regular arrays...")

    precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=3)

    # 1D array
    target_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    data_1d = {"target": target_1d}
    result_1d = precond(data_1d)
    assert isinstance(result_1d["target"], np.ndarray), "1D result should be array"
    print("✓ 1D array handling works")

    # 2D array
    target_2d = np.random.randn(10, 3)
    data_2d = {"target": target_2d}
    result_2d = precond(data_2d)
    assert isinstance(result_2d["target"], np.ndarray), "2D result should be array"
    assert result_2d["target"].shape == (10, 3), "2D shape should be preserved"
    print("✓ 2D array handling works")

    print("✓ All array tests passed")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Preconditioning Fix for List Handling")
    print("=" * 70)
    print()

    success1 = test_list_handling()
    test_array_handling()

    print()
    print("=" * 70)
    if success1:
        print("SUCCESS: Fix verified ✓")
        print()
        print("The training error should now be resolved.")
        print("You can resubmit: sbatch pretrain_moirai_precond_default.slurm")
        sys.exit(0)
    else:
        print("FAILED: Fix did not work ✗")
        sys.exit(1)
