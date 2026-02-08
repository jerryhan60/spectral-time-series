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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from uni2ts.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import MapFuncMixin


@dataclass
class PatchCrop(MapFuncMixin, Transformation):
    """
    Crop fields in a data_entry in the temporal dimension based on a patch_size.
    :param rng: numpy random number generator
    :param min_time_patches: minimum number of patches for time dimension
    :param max_patches: maximum number of patches for time * dim dimension (if flatten)
    :param will_flatten: whether time series fields will be flattened subsequently
    :param offset: whether to offset the start of the crop
    :param fields: fields to crop
    """

    min_time_patches: int
    max_patches: int
    will_flatten: bool = False
    offset: bool = True
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)

    def __post_init__(self):
        assert (
            self.min_time_patches <= self.max_patches
        ), "min_patches must be <= max_patches"
        assert len(self.fields) > 0, "fields must be non-empty"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a:b] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        patch_size = data_entry["patch_size"]
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]
        nvar = (
            sum(len(data_entry[f]) for f in self.fields)
            + sum(len(data_entry[f]) for f in self.optional_fields if f in data_entry)
            if self.will_flatten
            else 1
        )

        offset = (
            np.random.randint(
                time % patch_size + 1
            )  # offset by [0, patch_size) so that the start is not always a multiple of patch_size
            if self.offset
            else 0
        )
        total_patches = (
            time - offset
        ) // patch_size  # total number of patches in time series

        # 1. max_patches should be divided by nvar if the time series is subsequently flattened
        # 2. cannot have more patches than total available patches
        max_patches = min(self.max_patches // nvar, total_patches)
        if max_patches < self.min_time_patches:
            raise ValueError(
                f"max_patches={max_patches} < min_time_patches={self.min_time_patches}"
            )

        num_patches = np.random.randint(
            self.min_time_patches, max_patches + 1
        )  # number of patches to consider
        first = np.random.randint(
            total_patches - num_patches + 1
        )  # first patch to consider

        start = offset + first * patch_size
        stop = start + num_patches * patch_size
        return start, stop


@dataclass
class FixedPatchCrop(MapFuncMixin, Transformation):
    """
    Crop fields to a fixed maximum number of patches using a random start.
    """

    max_patches: int
    offset: bool = False
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)

    def __post_init__(self):
        assert self.max_patches > 0, "max_patches must be > 0"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a:b] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        patch_size = data_entry["patch_size"]
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]

        offset = (
            np.random.randint(time % patch_size + 1) if self.offset else 0
        )
        total_patches = (time - offset) // patch_size
        if total_patches < 1:
            raise ValueError("time series too short for patching")

        num_patches = min(total_patches, self.max_patches)
        first = (
            0
            if total_patches == num_patches
            else np.random.randint(total_patches - num_patches + 1)
        )
        start = offset + first * patch_size
        stop = start + num_patches * patch_size
        return start, stop


@dataclass
class ResampleZScorePatchCrop(MapFuncMixin, Transformation):
    """
    Crop fields to a fixed maximum number of patches using a random start.
    If anomaly checks fail, resample crop boundaries up to max_attempts.
    """

    max_patches: int
    prefix_ratio: float
    zscore_threshold: float
    variance_ratio_threshold: float = 0.0
    variance_min_count: int = 2
    max_attempts: int = 1
    min_prefix_tokens: int = 1
    minimum_scale: float = 1e-5
    offset: bool = False
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)
    reject_field: str = "reject"

    def __post_init__(self):
        assert self.max_patches > 0, "max_patches must be > 0"
        if not 0.0 < self.prefix_ratio < 1.0:
            raise ValueError("prefix_ratio must be between 0 and 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        reject = False
        a = 0
        b = 0
        for _ in range(self.max_attempts):
            a, b = self._get_boundaries(data_entry)
            reject = self._should_reject(data_entry, a, b)
            if not reject:
                break
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        data_entry[self.reject_field] = bool(reject)
        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a:b] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        patch_size = data_entry["patch_size"]
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]

        offset = (
            np.random.randint(time % patch_size + 1) if self.offset else 0
        )
        total_patches = (time - offset) // patch_size
        if total_patches < 1:
            raise ValueError("time series too short for patching")

        num_patches = min(total_patches, self.max_patches)
        first = (
            0
            if total_patches == num_patches
            else np.random.randint(total_patches - num_patches + 1)
        )
        start = offset + first * patch_size
        stop = start + num_patches * patch_size
        return start, stop

    def _should_reject(self, data_entry: dict[str, Any], a: int, b: int) -> bool:
        if self.zscore_threshold <= 0 and self.variance_ratio_threshold <= 0:
            return False

        patch_size = data_entry["patch_size"]
        crop_len = b - a
        if crop_len <= 0 or crop_len % patch_size != 0:
            return False

        num_patches = crop_len // patch_size
        if num_patches < 2:
            return False

        prefix_patches = max(
            self.min_prefix_tokens, int(np.floor(num_patches * self.prefix_ratio))
        )
        prefix_patches = min(prefix_patches, num_patches - 1)
        prefix_len = prefix_patches * patch_size
        if prefix_len <= 0 or prefix_len >= crop_len:
            return False

        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        for ts in field:
            values = ts[a:b]
            if self._series_reject(values, prefix_len):
                return True
        return False

    def _series_reject(self, values: np.ndarray, prefix_len: int) -> bool:
        prefix = values[:prefix_len]
        suffix = values[prefix_len:]
        prefix_obs = prefix[~np.isnan(prefix)]
        suffix_obs = suffix[~np.isnan(suffix)]
        if self.zscore_threshold > 0 and prefix_obs.size and suffix_obs.size:
            mean = float(prefix_obs.mean())
            var = float(prefix_obs.var())
            scale = max(math.sqrt(var), self.minimum_scale)
            max_abs_z = float(np.max(np.abs((suffix_obs - mean) / scale)))
            if max_abs_z > self.zscore_threshold:
                return True

        if (
            self.variance_ratio_threshold > 0
            and prefix_obs.size >= self.variance_min_count
            and suffix_obs.size >= self.variance_min_count
        ):
            prefix_var = float(prefix_obs.var())
            suffix_var = float(suffix_obs.var())
            eps = 1e-6
            ratio = (suffix_var + eps) / (prefix_var + eps)
            if ratio > self.variance_ratio_threshold or ratio < (
                1.0 / self.variance_ratio_threshold
            ):
                return True
        return False


@dataclass
class EvalCrop(MapFuncMixin, Transformation):
    offset: int
    distance: int
    prediction_length: int
    context_length: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a : b or None] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]
        window = data_entry["window"]
        fcst_start = self.offset + window * self.distance
        a = fcst_start - self.context_length
        b = fcst_start + self.prediction_length

        if self.offset >= 0:
            assert time >= b > a >= 0
        else:
            assert 0 >= b > a >= -time

        return a, b


@dataclass
class FinetunePatchCrop(MapFuncMixin, Transformation):
    """
    Similar to EvalCrop, crop training samples based on specific context_length and prediction_length
    """

    distance: int
    prediction_length: int
    context_length: int
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )

        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a:b] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]  # num of time steps of one series
        window = data_entry["window"]
        fcst_start = self.context_length + window * self.distance
        a = fcst_start - self.context_length
        b = fcst_start + self.prediction_length

        assert time >= b > a >= 0

        return a, b
