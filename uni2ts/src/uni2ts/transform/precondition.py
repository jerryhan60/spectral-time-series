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

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ._base import Transformation


@dataclass
class PolynomialPrecondition(Transformation):
    """
    Apply Chebyshev or Legendre polynomial preconditioning to time series.

    This implements Universal Sequence Preconditioning as described in:
    Marsden, A., & Hazan, E. (2025). Universal Sequence Preconditioning.
    arXiv:2502.06545.

    The transformation applies a polynomial convolution to improve the
    condition number of hidden transition matrices in time series dynamics.

    Mathematical formulation (as per Algorithm 1):
        ỹₜ = yₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  for t > n
        ỹₜ = yₜ                     for t ≤ n

    where cᵢ are the monic polynomial coefficients in power basis and n is the degree.

    For differencing (n=2, c₁=-1): ỹₜ = yₜ + (-1)·yₜ₋₁ = yₜ - yₜ₋₁

    Args:
        polynomial_type: "chebyshev" or "legendre"
        degree: Polynomial degree (recommended: 5-10, paper suggests <10 for stability)
        target_field: Field name to precondition (default: "target")
        enabled: Whether preconditioning is enabled (default: True)
        store_original: Whether to store original values (default: False)
    """

    polynomial_type: str = "chebyshev"
    degree: int = 5
    target_field: str = "target"
    enabled: bool = True
    store_original: bool = False

    def __post_init__(self):
        if not self.enabled:
            return

        # Validate inputs
        if self.polynomial_type not in ["chebyshev", "legendre"]:
            raise ValueError(
                f"Unknown polynomial type: {self.polynomial_type}. "
                "Must be 'chebyshev' or 'legendre'."
            )

        if self.degree < 1:
            raise ValueError(f"Degree must be >= 1, got {self.degree}")

        if self.degree > 10:
            import warnings
            warnings.warn(
                f"Degree {self.degree} > 10 may cause numerical instability. "
                "Paper recommends degree <= 10."
            )

        # Compute polynomial coefficients
        self.coeffs = self._compute_coefficients(self.polynomial_type, self.degree)

    def _compute_coefficients(self, polynomial_type: str, degree: int) -> np.ndarray:
        """
        Compute polynomial coefficients for the given type and degree.

        Returns coefficients in the order [c₁, c₂, ..., cₙ] where n = degree.
        """
        if polynomial_type == "chebyshev":
            return self._chebyshev_coefficients(degree)
        elif polynomial_type == "legendre":
            return self._legendre_coefficients(degree)
        else:
            raise ValueError(f"Unknown polynomial type: {polynomial_type}")

    def _chebyshev_coefficients(self, n: int) -> np.ndarray:
        """
        Compute monic Chebyshev polynomial coefficients of degree n.

        The monic Chebyshev polynomial is Mₙ(x) = Tₙ(x) / 2^(n-1).
        Returns coefficients [c₁, c₂, ..., cₙ] in power basis (standard polynomial form).

        For example:
        - n=2: T₂(x) = 2x² - 1, M₂(x) = x² - 0.5, coeffs = [0, -0.5]
        - n=3: T₃(x) = 4x³ - 3x, M₃(x) = x³ - 0.75x, coeffs = [0, -0.75, 0]

        Note: Paper notation p^c_n(x) = c₀x^n + c₁x^(n-1) + ... + cₙ where c₀=1 (monic).
        We return [c₁, c₂, ..., cₙ] (excluding c₀) for use in Algorithm 1.
        Numpy stores coefficients in increasing power order: [const, x¹, x², ..., x^n].
        """
        from numpy.polynomial import chebyshev, polynomial

        # Generate Chebyshev polynomial of degree n in Chebyshev basis
        cheb = chebyshev.Chebyshev.basis(n)

        # Convert to standard power basis (monomial form)
        power_poly = cheb.convert(kind=polynomial.Polynomial)

        # Get coefficients in power basis [c₀, c₁, ..., cₙ] where index=power
        # e.g., for M₃(x) = x³ - 0.75x: [0, -0.75, 0, 1] = [const, x¹, x², x³]
        power_coeffs = power_poly.coef

        # Make monic by dividing by leading coefficient (should be 2^(n-1) for Chebyshev)
        leading_coeff = power_coeffs[-1]
        monic_coeffs = power_coeffs / leading_coeff

        # Paper notation: p^c_n(x) = c₀x^n + c₁x^(n-1) + ... + cₙ
        # We need [c₁, c₂, ..., cₙ] which is [coeff(x^(n-1)), coeff(x^(n-2)), ..., coeff(x⁰)]
        # In numpy order this is [monic_coeffs[-2], monic_coeffs[-3], ..., monic_coeffs[0]]
        # Which is monic_coeffs[:-1] reversed!
        # But actually for the convolution, we want c₁ at index 0, c₂ at index 1, etc.
        # Since convolution applies coeffs[i] to y_{t-(i+1)}, we need:
        # coeffs[0] = c₁ = coeff of x^(n-1) = monic_coeffs[-2]
        # coeffs[1] = c₂ = coeff of x^(n-2) = monic_coeffs[-3]
        # This is monic_coeffs[-2::-1] or equivalently monic_coeffs[:-1][::-1]
        # Wait, let me reconsider...

        # Actually: monic_coeffs = [const, x¹, x², ..., x^(n-1), x^n]
        # Exclude leading coeff (x^n=c₀): monic_coeffs[:-1] = [const, x¹, x², ..., x^(n-1)]
        # Reverse to get [x^(n-1), x^(n-2), ..., x¹, const] = [c₁, c₂, ..., cₙ]
        return monic_coeffs[:-1][::-1]

    def _legendre_coefficients(self, n: int) -> np.ndarray:
        """
        Compute monic Legendre polynomial coefficients of degree n.

        The monic Legendre polynomial is Mₙ(x) = Pₙ(x) / leading_coeff.
        Returns coefficients [c₁, c₂, ..., cₙ] in power basis (standard polynomial form).

        The leading coefficient of Legendre polynomial Pₙ(x) is (2n)! / (2^n * n! * n!)

        Note: Paper notation p^c_n(x) = c₀x^n + c₁x^(n-1) + ... + cₙ where c₀=1 (monic).
        We return [c₁, c₂, ..., cₙ] (excluding c₀) for use in Algorithm 1.
        Numpy stores coefficients in increasing power order: [const, x¹, x², ..., x^n].
        """
        from numpy.polynomial import legendre, polynomial

        # Generate Legendre polynomial of degree n in Legendre basis
        leg = legendre.Legendre.basis(n)

        # Convert to standard power basis (monomial form)
        power_poly = leg.convert(kind=polynomial.Polynomial)

        # Get coefficients in power basis where index=power
        # [const, x¹, x², ..., x^(n-1), x^n]
        power_coeffs = power_poly.coef

        # Make monic by dividing by leading coefficient
        leading_coeff = power_coeffs[-1]
        monic_coeffs = power_coeffs / leading_coeff

        # Exclude leading coeff (x^n=c₀) and reverse to get [c₁, c₂, ..., cₙ]
        # monic_coeffs[:-1] = [const, x¹, ..., x^(n-1)]
        # Reversed: [x^(n-1), x^(n-2), ..., x¹, const] = [c₁, c₂, ..., cₙ]
        return monic_coeffs[:-1][::-1]

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        Apply preconditioning to target field.

        IMPORTANT: This transform operates on a single time series at a time.
        Each data_entry contains ONE time series (possibly multivariate).
        Multiple series are only packed together later in the pipeline via
        FlatPackCollection/PackCollate, ensuring we never precondition across
        series boundaries.
        """
        if not self.enabled:
            return data_entry

        if self.target_field not in data_entry:
            # If target field doesn't exist, return unchanged
            return data_entry

        target = data_entry[self.target_field]

        # Handle different target types:
        # 1. List of arrays (after _flatten_data in dataset pipeline)
        # 2. 2D array [time, variate] (multivariate time series)
        # 3. 1D array [time] (univariate time series)

        if isinstance(target, list):
            # Case 1: List of arrays (e.g., from _flatten_data)
            # Apply preconditioning to each array independently
            preconditioned = []
            for ts in target:
                if not isinstance(ts, np.ndarray):
                    ts = np.array(ts)
                preconditioned.append(self._apply_convolution(ts, self.coeffs))
            # Keep as list to maintain data structure
            data_entry[self.target_field] = preconditioned

        else:
            # Case 2 & 3: Single numpy array (1D or 2D)
            original_shape = target.shape
            original_dtype = target.dtype

            # Convert to numpy if needed
            if not isinstance(target, np.ndarray):
                target = np.array(target)

            # Apply preconditioning
            if target.ndim == 1:
                # 1D case: single univariate time series
                preconditioned = self._apply_convolution(target, self.coeffs)
            elif target.ndim == 2:
                # 2D case: [time, variate] - single multivariate time series
                # Process each variate INDEPENDENTLY to avoid cross-variate dependencies
                # Each variate represents a dimension of the SAME series, not different series
                preconditioned = np.stack([
                    self._apply_convolution(target[:, i], self.coeffs)
                    for i in range(target.shape[1])
                ], axis=1)
            else:
                raise ValueError(
                    f"Target field must be 1D or 2D, got shape {target.shape}"
                )

            # Restore original dtype
            preconditioned = preconditioned.astype(original_dtype)

            # Update target with preconditioned values
            data_entry[self.target_field] = preconditioned

        # Store metadata for reversal
        if self.store_original:
            data_entry[f"{self.target_field}_original"] = target
        data_entry["precondition_coeffs"] = self.coeffs
        data_entry["precondition_degree"] = self.degree
        data_entry["precondition_type"] = self.polynomial_type
        data_entry["precondition_enabled"] = True

        return data_entry

    def _apply_convolution(
        self,
        sequence: np.ndarray,
        coeffs: np.ndarray
    ) -> np.ndarray:
        """
        Apply polynomial convolution to a 1D sequence.

        Implements: ỹₜ = yₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ

        As per Algorithm 1 (page 2) of Universal Sequence Preconditioning paper:
        "y^{preconditioned}_t = y_t + Σⱼ₌₁ⁿ cⱼ y_{t-j}"

        This uses ADDITION to create the preconditioned target.
        For differencing (n=2, c₁=-1): ỹ_t = y_t + (-1)·y_{t-1} = y_t - y_{t-1} ✓

        IMPORTANT: This method operates on a single 1D sequence at a time.
        It only looks backward within the SAME sequence, never across
        different series. Series boundaries are naturally respected because
        this transform is applied to individual data_entries before any
        cross-series packing occurs in the pipeline.

        Args:
            sequence: 1D array of shape [time] representing ONE time series
            coeffs: 1D array of coefficients [c₁, c₂, ..., cₙ]

        Returns:
            Preconditioned sequence of same shape
        """
        n = len(coeffs)
        result = sequence.copy()

        # Vectorized implementation using array slicing
        # For t >= n, apply: result[t] = sequence[t] + ∑ᵢ₌₀ⁿ⁻¹ coeffs[i] · sequence[t-i-1]
        # This vectorizes the inner sum by using shifted array slices

        if len(sequence) > n:
            # Compute weighted sum for all positions t >= n simultaneously
            # weighted_sum[t-n] = coeffs[0]*seq[t-1] + coeffs[1]*seq[t-2] + ... + coeffs[n-1]*seq[t-n]
            weighted_sum = np.zeros(len(sequence) - n)
            for i in range(n):
                # coeffs[i] corresponds to sequence[t-(i+1)]
                # For all t in [n, len(sequence)), extract sequence[t-(i+1)]
                weighted_sum += coeffs[i] * sequence[n-i-1:len(sequence)-i-1]

            result[n:] = sequence[n:] + weighted_sum  # ADDITION (as per Algorithm 1)

        # For t < n, keep original values
        # (already copied in result = sequence.copy())

        return result


@dataclass
class ReversePrecondition(Transformation):
    """
    Reverse polynomial preconditioning after forecasting.

    This transformation reverses the preconditioning applied by
    PolynomialPrecondition, recovering the original scale predictions.

    Since forward preconditioning uses ADDITION:
        ỹₜ = yₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ

    Reverse uses SUBTRACTION:
        yₜ = ỹₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  for t > n
        yₜ = ỹₜ                     for t ≤ n

    Note: This requires the precondition_coeffs to be present in the
    data_entry, which should have been added by PolynomialPrecondition.

    Args:
        target_field: Field name to reverse precondition (default: "target")
        prediction_field: Field name for predictions (default: "prediction")
        enabled: Whether reversal is enabled (default: True)
    """

    target_field: str = "target"
    prediction_field: str = "prediction"
    enabled: bool = True

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """Reverse preconditioning on predictions."""
        if not self.enabled:
            return data_entry

        # Check if preconditioning was applied
        if not data_entry.get("precondition_enabled", False):
            return data_entry

        if "precondition_coeffs" not in data_entry:
            # No preconditioning metadata found
            return data_entry

        coeffs = data_entry["precondition_coeffs"]

        # Determine which field to reverse
        field_to_reverse = None
        if self.prediction_field in data_entry:
            field_to_reverse = self.prediction_field
        elif self.target_field in data_entry:
            field_to_reverse = self.target_field
        else:
            # No field to reverse
            return data_entry

        preconditioned = data_entry[field_to_reverse]

        # Handle different array shapes
        if not isinstance(preconditioned, np.ndarray):
            preconditioned = np.array(preconditioned)

        original_dtype = preconditioned.dtype

        # Apply reversal
        if preconditioned.ndim == 1:
            # 1D case: single time series
            restored = self._reverse_convolution(preconditioned, coeffs)
        elif preconditioned.ndim == 2:
            # 2D case: [time, variate] - apply to each variate
            restored = np.stack([
                self._reverse_convolution(preconditioned[:, i], coeffs)
                for i in range(preconditioned.shape[1])
            ], axis=1)
        elif preconditioned.ndim == 3:
            # 3D case: [batch/sample, time, variate] - common for predictions
            restored = np.stack([
                np.stack([
                    self._reverse_convolution(preconditioned[b, :, v], coeffs)
                    for v in range(preconditioned.shape[2])
                ], axis=1)
                for b in range(preconditioned.shape[0])
            ], axis=0)
        else:
            raise ValueError(
                f"Field must be 1D, 2D, or 3D, got shape {preconditioned.shape}"
            )

        # Restore original dtype
        restored = restored.astype(original_dtype)

        # Update field with restored values
        data_entry[field_to_reverse] = restored

        return data_entry

    def _reverse_convolution(
        self,
        sequence: np.ndarray,
        coeffs: np.ndarray
    ) -> np.ndarray:
        """
        Reverse polynomial convolution on a 1D sequence.

        Since forward preconditioning uses ADDITION:
            ỹₜ = yₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ

        The reverse uses SUBTRACTION to undo it:
            yₜ = ỹₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ

        This is computed iteratively from left to right, since yₜ depends
        on previously computed values yₜ₋₁, yₜ₋₂, etc.

        Args:
            sequence: 1D preconditioned array of shape [time]
            coeffs: 1D array of coefficients [c₁, c₂, ..., cₙ]

        Returns:
            Restored sequence of same shape
        """
        n = len(coeffs)
        result = sequence.copy()

        # For t > n, reverse the convolution iteratively
        for t in range(n, len(sequence)):
            # Compute weighted sum using already-reversed values: ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ
            weighted_sum = sum(
                coeffs[i-1] * result[t-i]
                for i in range(1, n+1)
            )
            result[t] = sequence[t] - weighted_sum  # SUBTRACTION (reverse of addition)

        # For t ≤ n, keep original values
        # (already copied in result = sequence.copy())

        return result
