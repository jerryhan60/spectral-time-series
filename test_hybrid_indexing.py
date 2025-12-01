#!/usr/bin/env python3
"""
Test to verify hybrid evaluation indexing is correct.
"""
import numpy as np

def forward_precondition(sequence, coeffs):
    """Forward preconditioning (from precondition.py)"""
    n = len(coeffs)
    result = sequence.copy()

    if len(sequence) > n:
        weighted_sum = np.zeros(len(sequence) - n)
        for i in range(n):
            # coeffs[i] corresponds to sequence[t-(i+1)]
            weighted_sum += coeffs[i] * sequence[n-i-1:len(sequence)-i-1]

        result[n:] = sequence[n:] + weighted_sum  # ADDITION

    return result

def hybrid_reverse(precond_seq, base_seq, coeffs):
    """Hybrid reversal (from eval_precond_hybrid.py)"""
    n = len(coeffs)
    result = precond_seq.copy()

    if len(precond_seq) > n:
        weighted_sum = np.zeros(len(precond_seq) - n)
        for i in range(n):
            # coeffs[i] corresponds to base_seq[t-(i+1)]
            weighted_sum += coeffs[i] * base_seq[n-i-1:len(base_seq)-i-1]

        result[n:] = precond_seq[n:] - weighted_sum  # SUBTRACTION

    return result

# Test case: Chebyshev degree 5
coeffs = np.array([0.0, -1.25, 0.0, 0.3125, 0.0])
n = len(coeffs)

print("="*80)
print("INDEXING VERIFICATION TEST")
print("="*80)
print(f"Coefficients: {coeffs}")
print(f"Degree: {n}")
print()

# Create a simple test sequence
original = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
print(f"Original sequence: {original}")
print()

# Apply forward preconditioning
preconditioned = forward_precondition(original, coeffs)
print(f"Preconditioned: {preconditioned}")
print()

# Manual calculation for first few preconditioned values
print("Manual verification of forward preconditioning:")
for t in range(n, min(n+3, len(original))):
    manual = original[t]
    terms = []
    for i in range(n):
        c_i = coeffs[i]
        y_past = original[t-(i+1)]
        manual += c_i * y_past
        terms.append(f"{c_i}*{y_past}")
    print(f"  ỹ[{t}] = {original[t]} + {' + '.join(terms)} = {manual:.4f}")
    print(f"  Code gives: {preconditioned[t]:.4f}")
    print()

print("="*80)
print("HYBRID REVERSAL TEST")
print("="*80)

# Simulate: use original as "base predictions" and try to reverse
base_predictions = original.copy()
hybrid_result = hybrid_reverse(preconditioned, base_predictions, coeffs)

print(f"Preconditioned: {preconditioned}")
print(f"Base predictions (using original): {base_predictions}")
print(f"Hybrid reversed: {hybrid_result}")
print(f"Original: {original}")
print()

# Check if we recovered original
print("Difference from original:")
diff = hybrid_result - original
print(f"  {diff}")
print(f"  Max abs difference: {np.abs(diff).max():.10f}")
print()

if np.allclose(hybrid_result, original, atol=1e-10):
    print("✓ PERFECT: Hybrid reversal correctly recovers original when base=original")
else:
    print("✗ ERROR: Hybrid reversal does NOT recover original!")
    print()
    print("Manual verification of hybrid reversal:")
    for t in range(n, min(n+3, len(preconditioned))):
        manual = preconditioned[t]
        terms = []
        for i in range(n):
            c_i = coeffs[i]
            y_base = base_predictions[t-(i+1)]
            manual -= c_i * y_base
            terms.append(f"{c_i}*{y_base}")
        print(f"  y[{t}] = {preconditioned[t]} - ({' + '.join(terms)}) = {manual:.4f}")
        print(f"  Code gives: {hybrid_result[t]:.4f}")
        print(f"  Should be: {original[t]:.4f}")
        print()

print("="*80)
print("TESTING WITH DIFFERENT BASE PREDICTIONS")
print("="*80)

# Now test with different base predictions (the actual hybrid case)
base_predictions_different = original * 0.9 + 0.5  # Slightly different predictions
print(f"Base predictions (different): {base_predictions_different}")

hybrid_result_2 = hybrid_reverse(preconditioned, base_predictions_different, coeffs)
print(f"Hybrid reversed (with different base): {hybrid_result_2}")
print(f"Original: {original}")
print()

diff2 = hybrid_result_2 - original
print(f"Difference from original: {diff2}")
print(f"Max abs difference: {np.abs(diff2).max():.4f}")
print()

print("This is expected to be different because base predictions ≠ original values")
print("The preconditioned model was trained expecting actual past values,")
print("but we're reversing with predicted values.")
