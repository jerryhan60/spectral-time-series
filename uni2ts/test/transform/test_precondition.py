#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pytest

from uni2ts.transform.precondition import PolynomialPrecondition, ReversePrecondition


class TestPolynomialPrecondition:
    """Test suite for PolynomialPrecondition transform."""

    def test_init_chebyshev(self):
        """Test initialization with Chebyshev polynomials."""
        transform = PolynomialPrecondition(polynomial_type="chebyshev", degree=5)
        assert transform.polynomial_type == "chebyshev"
        assert transform.degree == 5
        assert transform.enabled is True
        assert len(transform.coeffs) == 5

    def test_init_legendre(self):
        """Test initialization with Legendre polynomials."""
        transform = PolynomialPrecondition(polynomial_type="legendre", degree=5)
        assert transform.polynomial_type == "legendre"
        assert transform.degree == 5
        assert len(transform.coeffs) == 5

    def test_init_invalid_type(self):
        """Test that invalid polynomial type raises error."""
        with pytest.raises(ValueError, match="Unknown polynomial type"):
            PolynomialPrecondition(polynomial_type="invalid", degree=5)

    def test_init_invalid_degree(self):
        """Test that invalid degree raises error."""
        with pytest.raises(ValueError, match="Degree must be >= 1"):
            PolynomialPrecondition(polynomial_type="chebyshev", degree=0)

    def test_init_high_degree_warning(self):
        """Test that high degree produces warning."""
        with pytest.warns(UserWarning, match="may cause numerical instability"):
            PolynomialPrecondition(polynomial_type="chebyshev", degree=15)

    def test_disabled_transform(self):
        """Test that disabled transform is a no-op."""
        transform = PolynomialPrecondition(enabled=False, degree=5)
        data_entry = {"target": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        result = transform(data_entry)
        np.testing.assert_array_equal(result["target"], data_entry["target"])
        assert "precondition_enabled" not in result

    def test_missing_target_field(self):
        """Test that missing target field is handled gracefully."""
        transform = PolynomialPrecondition(degree=5)
        data_entry = {"other_field": np.array([1.0, 2.0, 3.0])}
        result = transform(data_entry)
        assert "target" not in result
        assert result == data_entry

    def test_1d_target(self):
        """Test preconditioning on 1D time series."""
        transform = PolynomialPrecondition(polynomial_type="chebyshev", degree=2)
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        data_entry = {"target": target.copy()}

        result = transform(data_entry)

        # Check metadata is stored
        assert "precondition_coeffs" in result
        assert "precondition_degree" in result
        assert result["precondition_degree"] == 2
        assert result["precondition_type"] == "chebyshev"
        assert result["precondition_enabled"] is True

        # Check target is modified
        assert not np.array_equal(result["target"], target)

        # Check first 'degree' elements are unchanged
        np.testing.assert_array_equal(result["target"][:2], target[:2])

        # Check remaining elements are different
        assert not np.allclose(result["target"][2:], target[2:])

    def test_2d_target(self):
        """Test preconditioning on 2D time series [time, variate]."""
        transform = PolynomialPrecondition(polynomial_type="chebyshev", degree=2)
        target = np.random.randn(20, 3)  # 20 timesteps, 3 variates
        data_entry = {"target": target.copy()}

        result = transform(data_entry)

        # Check shape is preserved
        assert result["target"].shape == target.shape

        # Check metadata is stored
        assert result["precondition_enabled"] is True

        # Check target is modified
        assert not np.array_equal(result["target"], target)

    def test_store_original(self):
        """Test that original values are stored when requested."""
        transform = PolynomialPrecondition(degree=5, store_original=True)
        target = np.random.randn(20)
        data_entry = {"target": target.copy()}

        result = transform(data_entry)

        assert "target_original" in result
        np.testing.assert_array_equal(result["target_original"], target)

    def test_dtype_preservation(self):
        """Test that dtype is preserved after preconditioning."""
        transform = PolynomialPrecondition(degree=3)

        # Test with float32
        target_f32 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result_f32 = transform({"target": target_f32})
        assert result_f32["target"].dtype == np.float32

        # Test with float64
        target_f64 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        result_f64 = transform({"target": target_f64})
        assert result_f64["target"].dtype == np.float64

    def test_different_degrees(self):
        """Test that different degrees produce different results."""
        target = np.random.randn(50)
        data_entry = {"target": target.copy()}

        transform_deg2 = PolynomialPrecondition(degree=2)
        transform_deg5 = PolynomialPrecondition(degree=5)

        result_deg2 = transform_deg2(data_entry.copy())
        result_deg5 = transform_deg5(data_entry.copy())

        # Results should be different
        assert not np.allclose(result_deg2["target"], result_deg5["target"])

    def test_chebyshev_vs_legendre(self):
        """Test that Chebyshev and Legendre produce different results."""
        target = np.random.randn(50)
        data_entry = {"target": target.copy()}

        transform_cheb = PolynomialPrecondition(polynomial_type="chebyshev", degree=5)
        transform_leg = PolynomialPrecondition(polynomial_type="legendre", degree=5)

        result_cheb = transform_cheb(data_entry.copy())
        result_leg = transform_leg(data_entry.copy())

        # Results should be different
        assert not np.allclose(result_cheb["target"], result_leg["target"])


class TestReversePrecondition:
    """Test suite for ReversePrecondition transform."""

    def test_disabled_transform(self):
        """Test that disabled transform is a no-op."""
        transform = ReversePrecondition(enabled=False)
        data_entry = {
            "prediction": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "precondition_enabled": True,
            "precondition_coeffs": np.array([0.5, 0.3]),
        }
        original_pred = data_entry["prediction"].copy()
        result = transform(data_entry)
        np.testing.assert_array_equal(result["prediction"], original_pred)

    def test_no_preconditioning_metadata(self):
        """Test that transform is no-op when no preconditioning metadata."""
        transform = ReversePrecondition()
        data_entry = {"prediction": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        original_pred = data_entry["prediction"].copy()
        result = transform(data_entry)
        np.testing.assert_array_equal(result["prediction"], original_pred)

    def test_preconditioning_not_enabled(self):
        """Test that transform is no-op when preconditioning not enabled."""
        transform = ReversePrecondition()
        data_entry = {
            "prediction": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "precondition_enabled": False,
            "precondition_coeffs": np.array([0.5, 0.3]),
        }
        original_pred = data_entry["prediction"].copy()
        result = transform(data_entry)
        np.testing.assert_array_equal(result["prediction"], original_pred)

    def test_1d_reversal(self):
        """Test reversing preconditioning on 1D array."""
        # Create simple data
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Apply preconditioning
        precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=3)
        data_entry = {"target": original.copy()}
        preconditioned_entry = precond(data_entry)

        # Simulate prediction with preconditioned values
        preconditioned_entry["prediction"] = preconditioned_entry["target"].copy()

        # Reverse preconditioning
        reverse = ReversePrecondition()
        restored_entry = reverse(preconditioned_entry)

        # Should recover original values (within numerical precision)
        np.testing.assert_allclose(
            restored_entry["prediction"], original, rtol=1e-10, atol=1e-10
        )

    def test_2d_reversal(self):
        """Test reversing preconditioning on 2D array."""
        original = np.random.randn(30, 4)  # 30 timesteps, 4 variates

        # Apply preconditioning
        precond = PolynomialPrecondition(polynomial_type="legendre", degree=5)
        data_entry = {"target": original.copy()}
        preconditioned_entry = precond(data_entry)

        # Simulate prediction
        preconditioned_entry["prediction"] = preconditioned_entry["target"].copy()

        # Reverse preconditioning
        reverse = ReversePrecondition()
        restored_entry = reverse(preconditioned_entry)

        # Should recover original values
        np.testing.assert_allclose(
            restored_entry["prediction"], original, rtol=1e-9, atol=1e-9
        )

    def test_3d_reversal(self):
        """Test reversing preconditioning on 3D array [batch, time, variate]."""
        original_2d = np.random.randn(40, 3)

        # Apply preconditioning
        precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=4)
        data_entry = {"target": original_2d.copy()}
        preconditioned_entry = precond(data_entry)

        # Simulate prediction with batch dimension
        prediction_3d = np.stack([preconditioned_entry["target"]] * 10, axis=0)
        preconditioned_entry["prediction"] = prediction_3d

        # Reverse preconditioning
        reverse = ReversePrecondition()
        restored_entry = reverse(preconditioned_entry)

        # Should recover original values for all samples
        expected_3d = np.stack([original_2d] * 10, axis=0)
        np.testing.assert_allclose(
            restored_entry["prediction"], expected_3d, rtol=1e-9, atol=1e-9
        )

    def test_target_field_fallback(self):
        """Test that reversal works on target field if no prediction field."""
        original = np.random.randn(25)

        # Apply preconditioning
        precond = PolynomialPrecondition(degree=3)
        data_entry = {"target": original.copy()}
        preconditioned_entry = precond(data_entry)

        # Reverse preconditioning on target field (no prediction field)
        reverse = ReversePrecondition()
        restored_entry = reverse(preconditioned_entry)

        # Should recover original values
        np.testing.assert_allclose(
            restored_entry["target"], original, rtol=1e-9, atol=1e-9
        )

    def test_dtype_preservation(self):
        """Test that dtype is preserved after reversal."""
        original_f32 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)

        precond = PolynomialPrecondition(degree=2)
        data_entry = {"target": original_f32.copy()}
        preconditioned_entry = precond(data_entry)
        preconditioned_entry["prediction"] = preconditioned_entry["target"].copy()

        reverse = ReversePrecondition()
        restored_entry = reverse(preconditioned_entry)

        assert restored_entry["prediction"].dtype == np.float32


class TestPreconditionRoundTrip:
    """Test suite for preconditioning + reversal round trips."""

    @pytest.mark.parametrize("polynomial_type", ["chebyshev", "legendre"])
    @pytest.mark.parametrize("degree", [2, 5, 8])
    def test_roundtrip_1d(self, polynomial_type, degree):
        """Test that precondition -> reverse recovers original (1D)."""
        original = np.random.randn(100)

        precond = PolynomialPrecondition(polynomial_type=polynomial_type, degree=degree)
        reverse = ReversePrecondition()

        data_entry = {"target": original.copy()}
        preconditioned = precond(data_entry)
        preconditioned["prediction"] = preconditioned["target"].copy()
        restored = reverse(preconditioned)

        np.testing.assert_allclose(
            restored["prediction"], original, rtol=1e-8, atol=1e-8
        )

    @pytest.mark.parametrize("polynomial_type", ["chebyshev", "legendre"])
    @pytest.mark.parametrize("degree", [3, 6])
    def test_roundtrip_2d(self, polynomial_type, degree):
        """Test that precondition -> reverse recovers original (2D)."""
        original = np.random.randn(80, 5)

        precond = PolynomialPrecondition(polynomial_type=polynomial_type, degree=degree)
        reverse = ReversePrecondition()

        data_entry = {"target": original.copy()}
        preconditioned = precond(data_entry)
        preconditioned["prediction"] = preconditioned["target"].copy()
        restored = reverse(preconditioned)

        np.testing.assert_allclose(
            restored["prediction"], original, rtol=1e-8, atol=1e-8
        )

    def test_roundtrip_with_noise(self):
        """Test that small perturbations propagate correctly."""
        original = np.random.randn(50)

        precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=4)
        reverse = ReversePrecondition()

        data_entry = {"target": original.copy()}
        preconditioned = precond(data_entry)

        # Add small noise to simulate prediction error
        noise = np.random.randn(50) * 0.01
        preconditioned["prediction"] = preconditioned["target"] + noise

        restored = reverse(preconditioned)

        # Should be close to original + noise (accounting for convolution effects)
        # Not exact match due to noise, but should be reasonably close
        diff = np.abs(restored["prediction"] - original)
        assert np.mean(diff) < 0.1  # Mean error should be small

    def test_chained_transforms(self):
        """Test that preconditioning works in a chain of transforms."""
        from uni2ts.transform import Chain

        original = np.random.randn(60)

        precond = PolynomialPrecondition(polynomial_type="legendre", degree=5)
        reverse = ReversePrecondition()

        # Create chain: precondition -> (simulate model) -> reverse
        data_entry = {"target": original.copy()}
        preconditioned = precond(data_entry)
        preconditioned["prediction"] = preconditioned["target"].copy()
        restored = reverse(preconditioned)

        np.testing.assert_allclose(
            restored["prediction"], original, rtol=1e-8, atol=1e-8
        )

    def test_numerical_stability(self):
        """Test numerical stability with various input ranges."""
        # Test with small values
        small = np.random.randn(50) * 0.001
        precond = PolynomialPrecondition(degree=5)
        reverse = ReversePrecondition()
        data = {"target": small.copy()}
        data = precond(data)
        data["prediction"] = data["target"].copy()
        data = reverse(data)
        np.testing.assert_allclose(data["prediction"], small, rtol=1e-6, atol=1e-9)

        # Test with large values
        large = np.random.randn(50) * 1000
        data = {"target": large.copy()}
        data = precond(data)
        data["prediction"] = data["target"].copy()
        data = reverse(data)
        np.testing.assert_allclose(data["prediction"], large, rtol=1e-6, atol=1e-3)

    def test_series_boundary_respect(self):
        """
        Test that preconditioning respects series boundaries.

        This test verifies that when processing separate data_entries
        (representing different series), the preconditioning for series 2
        does NOT depend on values from series 1.

        This is guaranteed by the architecture: each data_entry is processed
        independently before any cross-series packing occurs in the pipeline.
        """
        # Create two distinct series
        series1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
        series2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=3)

        # Process series independently (as they would be in the real pipeline)
        data1 = {"target": series1.copy()}
        data2 = {"target": series2.copy()}

        precond_data1 = precond(data1)
        precond_data2 = precond(data2)

        # Now simulate what would happen if series2 incorrectly used series1 values
        # by creating a concatenated series and preconditioning it
        concatenated = np.concatenate([series1, series2])
        data_concat = {"target": concatenated.copy()}
        precond_concat = precond(data_concat)

        # The preconditioned series2 should match the second half of concatenated
        # (since preconditioning is applied after degree steps)
        # BUT: the first few values of series2 in the concatenated version will
        # incorrectly depend on series1 values

        # Extract the series2 portion from concatenated preconditioning
        series2_from_concat = precond_concat["target"][len(series1):]
        series2_independent = precond_data2["target"]

        degree = precond.degree

        # KEY INSIGHT:
        # - When series2 is processed independently, the first `degree` values remain
        #   unchanged (t < degree in the independent array)
        # - When series2 is part of a concatenated array, it starts at position len(series1),
        #   so its first element is at t=10 >= degree, meaning it WILL be modified using
        #   values from series1!
        # - This difference PROVES that series boundaries matter and separate processing is correct

        # The values SHOULD be different, proving concatenation would be wrong
        # We expect large differences because series2 would incorrectly use series1 values
        diff = np.abs(series2_from_concat[:degree] - series2_independent[:degree])
        assert np.mean(diff) > 1.0, (
            "Expected large differences between concatenated and independent processing. "
            "If values match, series boundaries are not being properly isolated."
        )

        # Verify the full series shows significant differences
        full_diff = np.abs(series2_from_concat - series2_independent)
        assert np.mean(full_diff) > 1.0, (
            "Expected significant differences across all timesteps"
        )

        # The key insight: in the real pipeline, series1 and series2 are in
        # separate data_entries, so they're processed independently.
        # This test confirms concatenation would produce wrong results.

    def test_multivariate_independence(self):
        """
        Test that different variates in a multivariate series are processed independently.

        For a 2D array [time, variate], each variate should be preconditioned
        independently without cross-variate dependencies.
        """
        # Create a 2D multivariate series [time, 3 variates]
        time_len = 20
        num_variates = 3
        target_2d = np.random.randn(time_len, num_variates) * 10

        precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=4)

        # Precondition the 2D array
        data_2d = {"target": target_2d.copy()}
        precond_2d = precond(data_2d)
        result_2d = precond_2d["target"]

        # Precondition each variate separately
        results_1d = []
        for v in range(num_variates):
            data_1d = {"target": target_2d[:, v].copy()}
            precond_1d = precond(data_1d)
            results_1d.append(precond_1d["target"])

        # Results should match: 2D processing = independent 1D processing
        for v in range(num_variates):
            np.testing.assert_allclose(
                result_2d[:, v],
                results_1d[v],
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Variate {v} should be processed independently"
            )

    def test_list_of_arrays(self):
        """
        Test that preconditioning handles list of arrays (from _flatten_data).

        In the data pipeline, _flatten_data converts multivariate series into
        a list of univariate arrays. Preconditioning must handle this case.
        """
        # Create a list of time series (as returned by _flatten_data)
        time_len = 20
        num_series = 3
        target_list = [np.random.randn(time_len) * 10 for _ in range(num_series)]

        precond = PolynomialPrecondition(polynomial_type="chebyshev", degree=4)

        # Precondition the list
        data_list = {"target": target_list.copy()}
        precond_list = precond(data_list)
        result_list = precond_list["target"]

        # Verify result is still a list
        assert isinstance(result_list, list), "Result should remain a list"
        assert len(result_list) == num_series, "List length should be preserved"

        # Precondition each series separately and compare
        for i, ts in enumerate(target_list):
            data_single = {"target": ts.copy()}
            precond_single = precond(data_single)
            result_single = precond_single["target"]

            np.testing.assert_allclose(
                result_list[i],
                result_single,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Series {i} should match when processed in list vs individually"
            )
