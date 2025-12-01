# Critical Bug Fix: Reversal Context Issue in Hybrid and GT Evaluations

**Date**: 2025-11-18
**Status**: FIXED

## Summary

Discovered and fixed a critical bug in both `eval_precond_gt.py` and `eval_precond_hybrid.py` where the reversal of preconditioning used **incomplete context**, leading to incorrect error metrics.

## The Bug

### Root Cause

The preconditioning reversal formula is:
```
y[t] = ỹ[t] - Σ(i=1 to n) c_i · y_context[t-i]
```

For the **first `n` prediction timesteps** (t=0, 1, ..., n-1), the reversal needs context from **before the prediction window starts** - i.e., from the **input window**.

### What Was Wrong

Both evaluation scripts were only using the **prediction window** as context, missing the input window entirely:

1. **`eval_precond_gt.py`** (BEFORE FIX):
   - Line 189-201: Only extracted `label_target` (prediction window)
   - Missing: The `input_target` needed for reversal at early timesteps
   - Result: Inflated errors (MSE 9.28M vs correct 2.08M on M3 Monthly)

2. **`eval_precond_hybrid.py`** (BEFORE FIX):
   - Line 213: Only used `base_fc.samples` (prediction window)
   - Missing: The input window needed for proper context
   - Lines 122-125: Tried to handle this with indexing tricks, but fundamentally flawed
   - Additional issue: Paired each stochastic precond sample with different stochastic base sample (unstable)

### Why This Matters

Consider degree=5 (n=5) preconditioning:
- At prediction timestep t=0, reversal needs context from positions [-5, -4, -3, -2, -1]
- These positions are in the **input window**, NOT the prediction window
- Without these values, the reversal is mathematically incorrect

### Observable Symptoms

**Before Fix** (`eval_precond_gt.py`):
```
Ground Truth Context Results (M3 Monthly):
  MSE[mean]: 9,281,208.72
  MAE[0.5]: 1,789.662
  MASE[0.5]: 3.863

Precond Space Results (M3 Monthly):
  MSE[mean]: 2,080,377.859
  MAE[0.5]: 645.464
  MASE[0.5]: 1.717
```

The GT context results were **WORSE** than precond space, which is mathematically impossible - perfect context reversal should give identical metrics.

**After Fix**:
```
Ground Truth Context Results: [same as Precond Space]
  MSE[mean]: 2,080,377.859
  MAE[0.5]: 645.464
  MASE[0.5]: 1.717
```

Metrics now match exactly! ✓

## The Fix

### Changes to `eval_precond_gt.py`

**1. Extract full context (input + label)** instead of just label:

```python
# BEFORE (lines 175-201):
ground_truths = []
for item in test_data_list:
    # ... extract label only ...
    label_target = label_dict['target']
    ground_truths.append(label_target)  # ❌ WRONG: Only prediction window

# AFTER (lines 175-220):
ground_truths = []
for item in test_data_list:
    # ... extract both input and label ...
    input_target = input_dict['target']
    label_target = label_dict['target']
    full_gt = np.concatenate([input_target, label_target], axis=0)  # ✓ CORRECT
    ground_truths.append(full_gt)
```

**2. Update reversal function signature**:

```python
# BEFORE:
def reverse_precondition_with_gt_context(
    precond_predictions,
    ground_truth,  # Only [pred_len]
    coeffs,
)

# AFTER:
def reverse_precondition_with_gt_context(
    precond_predictions,
    full_ground_truth,  # [input_len + pred_len]
    input_length,       # NEW: where predictions start
    coeffs,
)
```

**3. Implement proper reversal with full context**:

```python
def _reverse_1d_with_full_context(
    precond_seq,
    full_gt,
    input_len,
    coeffs
):
    n = len(coeffs)
    pred_len = len(precond_seq)
    result = precond_seq.copy()

    # For each prediction timestep t
    for t in range(pred_len):
        # Actual position in full sequence
        actual_pos = input_len + t

        if actual_pos >= n:
            # Apply reversal using full context
            for i in range(n):
                context_idx = actual_pos - i - 1
                result[t] -= coeffs[i] * full_gt[context_idx]

    return result
```

**4. Pass input_length to reversal**:

```python
# BEFORE:
gt_reversed_samples = reverse_precondition_with_gt_context(
    precond_predictions=precond_samples,
    ground_truth=gt,  # ❌ Only label
    coeffs=precondition_coeffs,
)

# AFTER:
pred_len = precond_samples.shape[-1]
input_len = len(full_gt) - pred_len

gt_reversed_samples = reverse_precondition_with_gt_context(
    precond_predictions=precond_samples,
    full_ground_truth=full_gt,  # ✓ Input + label
    input_length=input_len,     # ✓ NEW
    coeffs=precondition_coeffs,
)
```

### Changes to `eval_precond_hybrid.py`

**1. Extract input windows from test data**:

```python
# NEW (lines 232-259):
input_windows = []
for item in test_data_list:
    # ... extract input_dict ...
    input_target = input_dict['target']
    # ... convert to numpy ...
    input_windows.append(input_target)
```

**2. Create full base context for reversal**:

```python
# BEFORE (lines 206-221):
for base_fc, precond_fc in tqdm(...):
    base_samples = base_fc.samples  # ❌ Only predictions

    hybrid_samples = reverse_precondition_with_base_context(
        precond_predictions=precond_samples,
        base_predictions=base_samples,  # ❌ Missing input window
        coeffs=precondition_coeffs,
    )

# AFTER (lines 265-293):
for base_fc, precond_fc, input_window in tqdm(...):
    base_samples = base_fc.samples

    # Take MEDIAN of base samples for stable context (improved 2025-11-18)
    base_median = np.median(base_samples, axis=0)  # ✓ Single stable context

    # Create full context: input + median base prediction
    full_context = np.concatenate([input_window, base_median])

    hybrid_samples = reverse_precondition_with_base_context(
        precond_predictions=precond_samples,
        base_context=full_context,   # ✓ Input + median (stable for all samples)
        input_length=input_len,      # ✓ NEW
        coeffs=precondition_coeffs,
    )
```

**3. Update reversal function** (same pattern as GT):

```python
# Renamed from _reverse_1d to _reverse_1d_with_context
# Now takes full_base_context and input_len
# Implements proper indexing: full_base_context[actual_pos - i - 1]
```

**4. Median-based context (additional improvement)**:

Instead of pairing each stochastic preconditioned sample with a different stochastic base sample, we now:
- Compute the **median** of all base model samples (across the 100 stochastic samples)
- Use this single, stable median trajectory as context for reversing **all** preconditioned samples
- Benefits:
  - More stable reversal context (not dependent on specific stochastic sample pairing)
  - Reduces variance in hybrid predictions
  - Uses "best estimate" from base model as anchor

## Impact

### Scripts Affected
1. ✅ **`uni2ts/cli/eval_precond_gt.py`** - FIXED
2. ✅ **`uni2ts/cli/eval_precond_hybrid.py`** - FIXED

### Scripts NOT Affected
- `uni2ts/cli/eval.py` - Standard evaluation (no manual reversal)
- `uni2ts/cli/eval_precond_space.py` - Precond space evaluation (no reversal)
- `uni2ts/cli/eval_baseline_in_precond_space.py` - Baseline evaluation (uses forward preconditioning, not reversal)

### Integration with `comprehensive_evaluation.py`
Both fixed scripts are integrated into the comprehensive evaluation framework:
- Mode `precond-gt` calls fixed `eval_precond_gt.py`
- Mode `hybrid` calls fixed `eval_precond_hybrid.py`

## Verification

### Test Case: M3 Monthly Dataset

**Preconditioned Space Evaluation** (baseline):
```
MSE[mean]: 2,080,377.859
MAE[0.5]: 645.464
```

**Ground Truth Context** (AFTER FIX):
```
MSE[mean]: 2,080,377.859  ✓ MATCHES!
MAE[0.5]: 645.464         ✓ MATCHES!
```

Perfect match confirms the fix is correct!

## Key Insights

1. **Forward preconditioning is a convolution** across the full sequence (input + predictions)
2. **Reverse preconditioning must mirror this** - it needs the same full context
3. **Context boundary is critical** - predictions start at `input_length`, not position 0
4. **The first n timesteps are most sensitive** - they depend entirely on input window context

## Mathematical Correctness

Given a sequence of length L = input_len + pred_len:
- Forward: `ỹ[t] = y[t] + Σ(i=1 to n) c_i · y[t-i]` for t ∈ [0, L)
- Reverse: `y[t] = ỹ[t] - Σ(i=1 to n) c_i · y[t-i]` for t ∈ [0, L)

For predictions starting at position `input_len`:
- Prediction window: t ∈ [input_len, L)
- Context for reversal at t=input_len: positions [input_len-n, ..., input_len-1]
- These are in the **input window**, not the prediction window!

## Lessons Learned

1. **Always verify metrics match expected behavior** - GT context reversal should match precond space
2. **Context windows matter** - Don't assume predictions exist in isolation
3. **Test boundary conditions** - The first few timesteps exposed the bug
4. **Check indexing carefully** - Off-by-one errors compound in time series

## Next Steps

1. ✅ Test fixed `eval_precond_gt.py` on multiple datasets
2. ✅ Test fixed `eval_precond_hybrid.py` on multiple datasets
3. ⬜ Re-run comprehensive evaluations with fixed scripts
4. ⬜ Compare hybrid vs GT context performance (now that both are correct)

## Related Files

- `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/eval_precond_gt.py`
- `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/eval_precond_hybrid.py`
- `/scratch/gpfs/EHAZAN/jh1161/eval/comprehensive_evaluation.py`
- This document: `/scratch/gpfs/EHAZAN/jh1161/REVERSAL_CONTEXT_BUG_FIX.md`
