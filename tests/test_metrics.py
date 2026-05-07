"""Tests for metric and statistics utilities."""
import numpy as np
import pytest

from poly_precond.metrics import (
    geometric_mean,
    paired_wins,
    relative_delta,
    two_sided_sign_test_pvalue,
)


def test_geometric_mean():
    # geo mean of [2, 8] = sqrt(16) = 4
    assert np.isclose(geometric_mean([2, 8]), 4.0)
    # geo mean of [1, 1, 1] = 1
    assert np.isclose(geometric_mean([1, 1, 1]), 1.0)
    # geo mean of single value
    assert np.isclose(geometric_mean([5.0]), 5.0)


def test_relative_delta():
    # 10% improvement: (0.9 - 1.0) / 1.0 * 100 = -10
    assert np.isclose(relative_delta(0.9, 1.0), -10.0)
    # No change
    assert np.isclose(relative_delta(1.0, 1.0), 0.0)
    # 50% worse
    assert np.isclose(relative_delta(1.5, 1.0), 50.0)


def test_paired_wins():
    method = [0.8, 1.2, 0.9, 1.1]
    baseline = [1.0, 1.0, 1.0, 1.0]
    assert paired_wins(method, baseline) == 2  # 0.8 < 1.0 and 0.9 < 1.0


def test_sign_test_pvalue():
    # 50/100 wins = perfectly balanced -> p ~ 1.0
    p = two_sided_sign_test_pvalue(50, 100)
    assert p > 0.9

    # 90/100 wins = very significant -> small p
    p = two_sided_sign_test_pvalue(90, 100)
    assert p < 0.001
