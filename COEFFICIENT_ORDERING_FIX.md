# Coefficient Ordering Fix - 2025-11-17

## Summary

Fixed a critical bug in the coefficient extraction for polynomial preconditioning. The coefficients were being extracted in the wrong order, causing the algorithm to apply transformations incorrectly.

## The Bug

### Paper's Notation (Equation 2, page 3)

```
p^c_n(x) = Σᵢ₌₀ⁿ cᵢ x^(n-i) = c₀x^n + c₁x^(n-1) + c₂x^(n-2) + ... + cₙ
```

Where:
- c₀ = 1 (leading coefficient, monic condition)
- c₁ = coefficient of x^(n-1)
- c₂ = coefficient of x^(n-2)
- cₙ = coefficient of x⁰ (constant term)

**Algorithm 1 uses:** `ỹₜ = yₜ + Σⱼ₌₁ⁿ cⱼ y_{t-j}`

So we need: **[c₁, c₂, ..., cₙ]** (excluding c₀)

### Numpy's Convention

`numpy.polynomial.Polynomial.coef` returns coefficients in **increasing order of powers**:

```
[const, x¹, x², ..., x^(n-1), x^n]
```

### Previous (Buggy) Implementation

```python
# WRONG: This included the leading coefficient!
return monic_coeffs[1:]
```

For M₃(x) = x³ - 0.75x (numpy order: `[0, -0.75, 0, 1]`):
- `monic_coeffs[1:]` gave `[-0.75, 0, 1]` ❌
- This treated the leading coefficient c₀=1 as c₃!

### Correct Implementation

```python
# CORRECT: Exclude leading coefficient and reverse order
return monic_coeffs[:-1][::-1]
```

For M₃(x) = x³ - 0.75x:
- `monic_coeffs = [0, -0.75, 0, 1]`
- `monic_coeffs[:-1] = [0, -0.75, 0]`
- `monic_coeffs[:-1][::-1] = [0, -0.75, 0]` ✓
- This gives [c₁, c₂, c₃] = [0, -0.75, 0] ✓

## Why the Reversal is Needed

After excluding the leading coefficient, numpy order gives:
```
[const, x¹, x², ..., x^(n-1)] = [cₙ, c_{n-1}, ..., c₂, c₁]
```

But we need:
```
[c₁, c₂, ..., c_{n-1}, cₙ] = [coeff(x^(n-1)), coeff(x^(n-2)), ..., coeff(x¹), const]
```

So we must reverse: `monic_coeffs[:-1][::-1]`

## Verification

### Example: M₃(x) = x³ - 0.75x

**Polynomial form:**
```
p^c_3(x) = 1·x³ + 0·x² + (-0.75)·x + 0
```

**Expected coefficients:**
- c₁ = 0 (coeff of x²)
- c₂ = -0.75 (coeff of x¹)
- c₃ = 0 (const)

**Result:** `[0, -0.75, 0]` ✓

### Example: M₅(x) = x⁵ - 1.25x³ + 0.3125x

**Polynomial form:**
```
p^c_5(x) = 1·x⁵ + 0·x⁴ + (-1.25)·x³ + 0·x² + 0.3125·x + 0
```

**Expected coefficients:**
- c₁ = 0 (coeff of x⁴)
- c₂ = -1.25 (coeff of x³)
- c₃ = 0 (coeff of x²)
- c₄ = 0.3125 (coeff of x¹)
- c₅ = 0 (const)

**Result:** `[0, -1.25, 0, 0.3125, 0]` ✓

## Convolution Verification

The convolution implementation was **already correct**:

```python
for i in range(n):
    weighted_sum += coeffs[i] * sequence[n-i-1:len(sequence)-i-1]
```

This applies:
- `coeffs[0]` = c₁ to `y_{t-1}` ✓
- `coeffs[1]` = c₂ to `y_{t-2}` ✓
- `coeffs[i]` = c_{i+1} to `y_{t-(i+1)}` ✓

## Files Modified

1. **uni2ts/src/uni2ts/transform/precondition.py:140**
   - Changed: `return monic_coeffs[1:]`
   - To: `return monic_coeffs[:-1][::-1]`
   - In: `_chebyshev_coefficients()`

2. **uni2ts/src/uni2ts/transform/precondition.py:174**
   - Changed: `return monic_coeffs[1:]`
   - To: `return monic_coeffs[:-1][::-1]`
   - In: `_legendre_coefficients()`

## Test Coverage

Created `test_coefficient_ordering.py` with:
1. ✓ Coefficient extraction for n=2, 3, 5
2. ✓ Convolution applies c₁→y_{t-1}, c₂→y_{t-2}
3. ✓ Differencing works (c₁=-1)
4. ✓ Forward/reverse are proper inverses

## Impact

### Previous Models
All models trained with preconditioning before 2025-11-17 used **incorrect coefficient ordering**. Combined with the previous bugs (wrong coefficients + wrong sign), this means:
- ✗ All preconditioned models need retraining
- ✓ Baseline models (no preconditioning) are unaffected

### This is the THIRD Critical Bug
1. **Bug #1 (2025-11-17):** Extracted Chebyshev/Legendre basis instead of power basis
2. **Bug #2 (2025-11-17):** Used subtraction instead of addition in forward pass
3. **Bug #3 (2025-11-17, THIS FIX):** Wrong coefficient ordering

All three bugs are now fixed.
