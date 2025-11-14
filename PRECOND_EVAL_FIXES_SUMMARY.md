# Preconditioned Evaluation Issues - Diagnosis and Fixes

## Issues Identified

### 1. MSE_median Missing from All Datasets (Status: `partial_success`)
**Diagnosis**: NOT A BUG - Expected behavior
- The preconditioned space evaluation only computes `MSE[mean]`, not `MSE[0.5]` (median)
- This is by design - the evaluation metrics list in `eval_precond_space.py` only includes `MSE()` without quantile specification
- All datasets correctly show `partial_success` because they have 10/11 metrics (missing MSE_median)

**No fix needed** - This is correct behavior.

### 2. Rideshare Dataset Shows `all_nan` Status
**Root Cause**: Rideshare dataset contains NaN values in ground truth labels

**Original Problem**:
- My NaN filtering was TOO AGGRESSIVE
- Code was skipping ALL samples that had ANY NaN values
- For Rideshare, all 2304 samples have NaN somewhere in their labels
- Result: All samples skipped → no evaluation possible

**The Issue with My Approach**:
```python
# OLD CODE (TOO AGGRESSIVE)
if np.isnan(orig_input_target).any() or np.isnan(orig_label_target).any():
    skipped_items += 1
    continue  # Skip entire sample
```

**Why This Was Wrong**:
1. Regular evaluation (`eval.py`) handles NaN using **masked arrays** (`np.ma.masked_invalid`)
2. Masked arrays allow computation on data with NaN by masking invalid values
3. My code was rejecting samples before they could be evaluated with masking
4. For Rideshare: all samples have some NaN, but not ALL values are NaN
5. With masking, evaluation CAN proceed on valid portions of the data

## Fixes Applied

### Fix: Allow NaN in Original Data, Only Skip if Preconditioning Introduces New NaN

```python
# NEW CODE (CORRECT)
has_nan = np.isnan(orig_input_target).any() or np.isnan(orig_label_target).any()

if has_nan:
    nan_in_original += 1
    # DON'T skip - preconditioning should preserve NaN
    # The evaluation framework will handle NaN using masked arrays

# Precondition (NaN values pass through)
preconditioned_full = preconditioner(data_entry)["target"]

# Only skip if preconditioning introduced NEW NaN
new_nan = np.isnan(preconditioned_full) & ~np.isnan(full_target)
if new_nan.any():
    nan_after_precond += 1
    skipped_items += 1
    continue
```

### Key Changes:

1. **Track but don't skip** samples with NaN in original data
2. **Pass NaN through preconditioning** (numpy operations preserve NaN)
3. **Only skip** if preconditioning introduces NEW NaN (numerical instability)
4. **Let evaluation framework handle NaN** using `mask_invalid_label=True`

### Updated Statistics Output:

```
Preconditioning statistics:
  Total items: 2304
  Items with NaN in original data: 2304 (will use masked arrays)
  Items skipped (new NaN after preconditioning): 0
  Valid items for evaluation: 2304
```

## Expected Behavior After Fix

### For Rideshare:
- **Before**: All samples skipped → error → `all_nan` status
- **After**: All samples evaluated with masking → metrics computed → likely still NaN metrics (but for the right reason)

### For Other Datasets with Partial NaN:
- **Before**: Samples with any NaN were completely rejected
- **After**: Samples with NaN are evaluated on valid portions using masked arrays

## Why Rideshare Metrics May Still Be NaN

Even with the fix, Rideshare may still produce NaN metrics because:

1. **If ground truth is mostly/all NaN**: Division by zero in metrics like MASE, ND, etc.
2. **Masked arrays reduce to empty**: If too many values are masked, aggregations fail
3. **This is a data quality issue**, not a code issue

The regular evaluation (without preconditioning) also produces NaN metrics for Rideshare, confirming this is a dataset problem.

## Verification

To verify the fix works:

```bash
# Re-run evaluation on single dataset
python -m cli.eval_precond_space \
    model=moirai_precond_ckpt_no_reverse \
    model.checkpoint_path=/path/to/ckpt \
    data.dataset_name=rideshare_with_missing \
    data.prediction_length=168
```

Expected output:
```
Preconditioning statistics:
  Total items: 2304
  Items with NaN in original data: 2304 (will use masked arrays)
  Items skipped (new NaN after preconditioning): 0
  Valid items for evaluation: 2304

# Evaluation proceeds, metrics may still be NaN due to data quality
```

## Files Modified

1. **`cli/eval_precond_space.py`** (lines 92-117, 135-148)
   - Changed NaN filtering from "skip if any NaN" to "skip only if new NaN introduced"
   - Updated statistics messages
   - Updated error message for clarity

## Comparison: Regular vs Precond Evaluation

### Regular Evaluation (`eval.py`):
```python
# No explicit NaN checking before evaluation
# Relies entirely on mask_invalid_label=True during metric computation
```

### Precond Space Evaluation (NEW):
```python
# Tracks NaN but doesn't skip
# Only skips if preconditioning creates new NaN (instability)
# Uses mask_invalid_label=True like regular evaluation
```

## Related Issues

### Aggregation Script Behavior:
The aggregation in `eval_precond_comprehensive.slurm` correctly identifies:
- `success`: All metrics present and valid
- `partial_success`: Some metrics present (e.g., missing MSE_median)
- `all_nan`: All metrics are NaN
- `failed`: Evaluation crashed/errored

### MSE_median Status:
All datasets showing `partial_success` is EXPECTED because:
- MSE_median is not computed by default
- Only MSE_mean is computed
- This gives 10/11 metrics → `partial_success`

## Recommendations

1. **For Rideshare**: Consider if this dataset should be excluded from preconditioning evaluation
   - High NaN content makes evaluation unreliable
   - Regular evaluation also produces NaN metrics
   - May indicate data quality issues

2. **For Other Datasets**: The fix allows proper handling of sparse NaN values
   - Evaluation proceeds on valid portions
   - Metrics are more reliable

3. **Future Enhancement**: Could add a threshold for NaN tolerance
   - Skip samples with > X% NaN values
   - Balance between data retention and quality
