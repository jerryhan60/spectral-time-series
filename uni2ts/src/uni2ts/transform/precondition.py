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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from uni2ts.common.precondition import compute_polynomial_coefficients

from ._base import Transformation


@dataclass
class PatchPolynomialPrecondition(Transformation):
    """
    Apply polynomial preconditioning along the patch-token axis.

    This operates on patchified targets with shape [var, time, patch] and
    applies a causal polynomial convolution across the time axis for each
    variate and patch element independently.
    """

    polynomial_type: str = "chebyshev"
    degree: int = 5
    target_field: str = "target"
    lag_stride: int = 1
    enabled: bool = True

    def __post_init__(self):
        if not self.enabled:
            self.coeffs = None
            return
        if self.lag_stride < 1:
            raise ValueError("lag_stride must be >= 1")
        self.coeffs = compute_polynomial_coefficients(
            self.polynomial_type, self.degree
        ).astype(np.float32)

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return data_entry
        if self.target_field not in data_entry:
            return data_entry

        target = data_entry[self.target_field]
        data_entry[self.target_field] = self._apply(target)
        return data_entry

    def _apply(
        self, target: np.ndarray | list[np.ndarray] | dict[str, np.ndarray]
    ) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        if isinstance(target, list):
            return [self._apply(arr) for arr in target]
        if isinstance(target, dict):
            return {k: self._apply(v) for k, v in target.items()}
        if not isinstance(target, np.ndarray):
            target = np.asarray(target)
        return self._apply_patch_convolution(target, self.coeffs, self.lag_stride)

    @staticmethod
    def _apply_patch_convolution(
        target: np.ndarray, coeffs: np.ndarray, lag_stride: int
    ) -> np.ndarray:
        if target.ndim == 2:
            target = target[None, ...]
            squeeze = True
        else:
            squeeze = False

        if target.ndim != 3:
            raise ValueError(
                f"Patch precondition expects ndim=2 or 3, got shape {target.shape}"
            )

        var, time, patch = target.shape
        result = target.copy()
        n = len(coeffs)
        stride = int(lag_stride)
        if stride < 1:
            raise ValueError("lag_stride must be >= 1")
        if time <= n * stride:
            return result.squeeze(0) if squeeze else result

        weighted_sum = np.zeros((var, time - n * stride, patch), dtype=result.dtype)
        for i in range(n):
            start = (n - i - 1) * stride
            end = time - (i + 1) * stride
            weighted_sum += coeffs[i] * target[:, start:end, :]
        result[:, n * stride :, :] = target[:, n * stride :, :] + weighted_sum

        return result.squeeze(0) if squeeze else result
