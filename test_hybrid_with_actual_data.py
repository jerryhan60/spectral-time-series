#!/usr/bin/env python3
"""
Test hybrid evaluation with simulated realistic predictions.
"""
import numpy as np

def reverse_precondition_with_base_context(
    precond_predictions,
    base_predictions,
    coeffs: np.ndarray,
) -> np.ndarray:
    """From eval_precond_hybrid.py"""
    if not isinstance(precond_predictions, np.ndarray):
        precond_predictions = np.array(precond_predictions)
    if not isinstance(base_predictions, np.ndarray):
        base_predictions = np.array(base_predictions)

    if precond_predictions.ndim == 1:
        return _reverse_1d(precond_predictions, base_predictions, coeffs)
    elif precond_predictions.ndim == 2:
        return np.stack([
            _reverse_1d(precond_predictions[i], base_predictions[i], coeffs)
            for i in range(precond_predictions.shape[0])
        ], axis=0)
    else:
        raise ValueError(
            f"precond_predictions must be 1D or 2D, got shape {precond_predictions.shape}"
        )


def _reverse_1d(precond_seq: np.ndarray, base_seq: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """From eval_precond_hybrid.py"""
    n = len(coeffs)
    result = precond_seq.copy()

    if len(precond_seq) > n:
        weighted_sum = np.zeros(len(precond_seq) - n)
        for i in range(n):
            weighted_sum += coeffs[i] * base_seq[n-i-1:len(base_seq)-i-1]

        result[n:] = precond_seq[n:] - weighted_sum

    return result


def forward_precondition(sequence, coeffs):
    """Apply forward preconditioning"""
    n = len(coeffs)
    result = sequence.copy()

    if len(sequence) > n:
        weighted_sum = np.zeros(len(sequence) - n)
        for i in range(n):
            weighted_sum += coeffs[i] * sequence[n-i-1:len(sequence)-i-1]

        result[n:] = sequence[n:] + weighted_sum

    return result


print("="*80)
print("TESTING HYBRID EVALUATION WITH REALISTIC SCENARIO")
print("="*80)

# Chebyshev degree 5 coefficients
coeffs = np.array([0.0, -1.25, 0.0, 0.3125, 0.0])
n = len(coeffs)

# Simulate true future values (ground truth)
np.random.seed(42)
true_future = np.random.randn(18) * 100 + 1000

print(f"\nGround truth (first 10 values): {true_future[:10]}")
print(f"Ground truth MAE would be 0 if predictions perfect")
print()

# Simulate base model predictions (reasonable but not perfect)
base_predictions = true_future + np.random.randn(18) * 20  # Some error

print(f"Base predictions (first 10): {base_predictions[:10]}")
base_mae = np.abs(base_predictions - true_future).mean()
print(f"Base model MAE: {base_mae:.2f}")
print()

# Simulate preconditioned ground truth
true_future_precond = forward_precondition(true_future, coeffs)

print(f"Preconditioned ground truth (first 10): {true_future_precond[:10]}")
print()

# Simulate preconditioned model predictions (in preconditioned space)
# This model should predict the preconditioned values
precond_predictions = true_future_precond + np.random.randn(18) * 20  # Some error in precond space

print(f"Precond predictions (first 10): {precond_predictions[:10]}")
precond_mae_in_precond_space = np.abs(precond_predictions - true_future_precond).mean()
print(f"Precond model MAE (in precond space): {precond_mae_in_precond_space:.2f}")
print()

# Now apply hybrid reversal
print("="*80)
print("APPLYING HYBRID REVERSAL")
print("="*80)

hybrid_predictions = reverse_precondition_with_base_context(
    precond_predictions,
    base_predictions,
    coeffs
)

print(f"Hybrid predictions (first 10): {hybrid_predictions[:10]}")
hybrid_mae = np.abs(hybrid_predictions - true_future).mean()
print(f"Hybrid MAE (in original space): {hybrid_mae:.2f}")
print()

print("="*80)
print("COMPARISON")
print("="*80)
print(f"Base model MAE:   {base_mae:.2f}")
print(f"Hybrid MAE:       {hybrid_mae:.2f}")
print(f"Ratio (hybrid/base): {hybrid_mae/base_mae:.2f}x")
print()

# Now let's test what SHOULD work: if we had perfect base predictions
print("="*80)
print("TESTING WITH PERFECT BASE PREDICTIONS")
print("="*80)

hybrid_with_perfect_base = reverse_precondition_with_base_context(
    precond_predictions,
    true_future,  # Use ground truth as "base predictions"
    coeffs
)

hybrid_perfect_base_mae = np.abs(hybrid_with_perfect_base - true_future).mean()
print(f"Hybrid MAE (with perfect base): {hybrid_perfect_base_mae:.2f}")
print(f"This should approximately equal precond_mae_in_precond_space: {precond_mae_in_precond_space:.2f}")
print(f"Difference: {abs(hybrid_perfect_base_mae - precond_mae_in_precond_space):.4f}")
print()

# The key insight:
print("="*80)
print("KEY INSIGHT")
print("="*80)
print("When we use imperfect base predictions to reverse the preconditioned predictions,")
print("we introduce additional error beyond what either model has individually.")
print()
print("The hybrid error depends on:")
print("1. Precond model's error in precond space")
print("2. Base model's error (gets propagated through the reversal)")
print("3. Interaction between these two sources of error")
print()
print("If the errors are uncorrelated or negatively correlated, hybrid can be worse!")
