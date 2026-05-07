"""Metric and statistics utilities for paper table verification."""
import numpy as np


def geometric_mean(values):
    return np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64))))


def relative_delta(method, baseline):
    return (method - baseline) / baseline * 100


def paired_wins(method_values, baseline_values):
    return int(np.sum(np.asarray(method_values) < np.asarray(baseline_values)))


def two_sided_sign_test_pvalue(wins, n):
    from scipy.stats import binomtest
    return binomtest(wins, n, 0.5, alternative="two-sided").pvalue
