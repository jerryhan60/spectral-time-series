# Evaluation in Preconditioned Space - Issues Analysis

## Summary

The `eval_precond_comprehensive.slurm` script has failures in 6 out of 30 datasets when evaluating in preconditioned space. This document analyzes the root causes and proposes solutions.

## Failed Datasets Analysis

### Category 1: Complete Failure - All Samples Skipped (NaN Introduction)

**Datasets**: Bitcoin, NN5_Daily, Rideshare, Sunspot

**Error Pattern**:
```
ValueError: All X items were skipped during preconditioning.
Preconditioning introduced NaN in all samples (numerical instability).
Dataset cannot be evaluated in preconditioned space.
```

**Root Cause**:
- These datasets already contain NaN values in the original data
- When preconditioning is applied (Chebyshev degree 5), the convolution operation introduces NEW NaN values in ALL samples
- The `eval_precond_space.py` script filters out any sample where preconditioning introduces new NaN (lines 113-117)
- Since all samples get new NaN, all samples are filtered out, resulting in zero valid samples

**Why This Happens**:
- Preconditioning formula: `ỹₜ = yₜ + Σᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ` (CORRECTED 2025-11-17: uses addition)
- If yₜ₋ᵢ contains NaN, and we multiply by coefficient cᵢ, the result is NaN
- This NaN propagates forward in the sequence (NaN + x = NaN, NaN - x = NaN)
- For degree 5, we need 5 previous timesteps, so if any of them have NaN, the result is NaN

**Examples**:
- Bitcoin: 18 samples, all have NaN in original data, all get new NaN after preconditioning
- NN5_Daily: 111 samples, all have NaN, all get new NaN
- Rideshare: 2304 samples, all have NaN, all get new NaN
- Sunspot: 1 sample, has NaN, gets new NaN

### Category 2: Partial Failure - Broadcasting Error (Shape Mismatch)

**Datasets**: Vehicle_Trips, Temperature_Rain

**Error Pattern**:
```
ValueError: operands could not be broadcast together with shapes (8,30) (32,30)
```

**Root Cause**:
- Preconditioning filters out SOME samples (but not all) that get new NaN
- Vehicle_Trips: 329 samples → 136 valid after filtering (193 skipped)
- Temperature_Rain: 32,072 samples → 1,159 valid after filtering (30,913 skipped)
- The script generates predictions for ALL original samples (using batch_size=32)
- But the preconditioned ground truth only contains the valid (filtered) samples
- When computing metrics, GluonTS tries to compute `label - prediction` but:
  - Label shape: (8, 30) - only 8 valid samples in this batch
  - Prediction shape: (32, 30) - full batch of 32 predictions
- **The shapes don't match**, causing the broadcast error

**Code Location** (`eval_precond_space.py`):
```python
# Line 230-233: Generate predictions for ALL samples
forecast_it = predictor.predict(
    get_inputs(test_data_list),  # <-- ALL original samples
    num_samples=100,
)

# Line 263-271: Evaluate using FILTERED preconditioned test data
res = evaluate_forecasts(
    forecasts=forecast_list,  # <-- Predictions for ALL samples
    test_data=preconditioned_test_data,  # <-- Only VALID samples (filtered)
    ...
)
```

**Why This Wasn't Caught Earlier**:
- Most datasets either:
  - Have NO NaN (all samples pass filtering) → no mismatch
  - Have ALL NaN that introduce new NaN (all samples filtered out) → caught by the "All items skipped" check
- Only datasets with PARTIAL filtering trigger this shape mismatch error

### Category 3: Partial Success - Too Few Valid Samples

**Dataset**: KDD_Cup_2018

**Result**: Metrics computed but all NaN

**Root Cause**:
- 270 total samples, 269 skipped, only 1 valid sample remaining
- Metrics computed on single sample produce NaN (insufficient data for meaningful statistics)
- Marked as `partial_success` in CSV

## Comparison with Working Scripts

The `eval_comprehensive.slurm` and standard evaluation scripts work because:

1. **They don't filter samples**: All samples are evaluated, even if they contain NaN
2. **GluonTS handles NaN**: The evaluation framework uses masked arrays to handle NaN in labels
3. **Preconditioning is reversed**: Predictions are transformed back to original space before evaluation
4. **Shape consistency**: Predictions and labels always have matching shapes

## Proposed Solutions

### Solution 1: Track Filtered Indices (Recommended)

Modify `eval_precond_space.py` to track which samples were filtered and only generate predictions for valid samples:

```python
def precondition_ground_truth(...):
    preconditioned_input = []
    preconditioned_label = []
    valid_indices = []  # <-- NEW: Track which samples are valid

    for idx, item in enumerate(tqdm(test_data, desc="Preconditioning ground truth")):
        # ... preconditioning logic ...

        if new_nan.any():
            skipped_items += 1
            continue

        valid_indices.append(idx)  # <-- NEW: Record valid index
        preconditioned_input.append(new_input)
        preconditioned_label.append(new_label)

    # Return valid indices along with preconditioned data
    return PreconditionedTestData(...), valid_indices

def evaluate_in_preconditioned_space(...):
    preconditioned_test_data, valid_indices = precondition_ground_truth(...)

    # Only generate predictions for valid samples
    test_data_list = list(test_data)
    valid_test_data = [test_data_list[i] for i in valid_indices]

    forecast_it = predictor.predict(get_inputs(valid_test_data), ...)
```

**Pros**:
- Maintains current filtering logic
- Ensures shape consistency
- Minimal code changes

**Cons**:
- More complex indexing logic
- Still can't evaluate datasets where all samples are filtered

### Solution 2: Use Masked Arrays Throughout (Better)

Modify preconditioning to use NumPy masked arrays and preserve all samples:

```python
def precondition_ground_truth(...):
    # Don't skip samples - use masked arrays instead
    for item in test_data:
        full_target = np.concatenate([orig_input_target, orig_label_target], axis=0)

        # Create masked array if there are NaN
        if np.isnan(full_target).any():
            full_target = np.ma.masked_invalid(full_target)

        # Apply preconditioning (will preserve mask)
        preconditioned_full = preconditioner({"target": full_target})["target"]

        # If new NaN introduced, mask them
        if not isinstance(preconditioned_full, np.ma.MaskedArray):
            preconditioned_full = np.ma.masked_invalid(preconditioned_full)

        # Keep ALL samples, even if they have NaN/masked values
        preconditioned_input.append(new_input)
        preconditioned_label.append(new_label)
```

**Pros**:
- Consistent with how standard evaluation handles NaN
- No shape mismatch issues
- Can evaluate more datasets

**Cons**:
- Requires updating preconditioning transform to handle masked arrays
- May report metrics for datasets with severe NaN issues

### Solution 3: Lower Polynomial Degree (Workaround)

Use a lower degree (e.g., degree 2 or 3) which requires fewer historical timesteps and is less prone to NaN propagation:

```bash
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt,PRECOND_DEGREE=2 eval_precond_comprehensive.slurm
```

**Pros**:
- Simple parameter change
- May reduce NaN propagation

**Cons**:
- Doesn't match training configuration if model was trained with degree 5
- May still fail on heavily NaN datasets
- Doesn't address the fundamental shape mismatch issue

### Solution 4: Skip Problematic Datasets (Quick Fix)

Modify the SLURM script to skip datasets known to have issues:

```bash
# Skip datasets with known NaN issues
if [[ "$dataset_name" == "bitcoin_with_missing" ]] || \
   [[ "$dataset_name" == "nn5_daily_with_missing" ]] || \
   [[ "$dataset_name" == "rideshare_with_missing" ]] || \
   [[ "$dataset_name" == "sunspot_with_missing" ]] || \
   [[ "$dataset_name" == "vehicle_trips_with_missing" ]] || \
   [[ "$dataset_name" == "temperature_rain_with_missing" ]]; then
    echo "  ⚠ $display_name has known NaN issues - skipping"
    echo "$display_name,,,,,,,,,,,skipped_known_nan_issue" >> "$CSV_FILE"
    continue
fi
```

**Pros**:
- Immediate workaround
- Allows evaluation to complete on remaining datasets

**Cons**:
- Doesn't solve the underlying issue
- Reduces benchmark coverage

## Recommended Action Plan

1. **Short term**: Implement Solution 4 (skip problematic datasets) to get results quickly
2. **Medium term**: Implement Solution 1 (track filtered indices) to handle partial filtering
3. **Long term**: Implement Solution 2 (masked arrays) for complete NaN handling

## Additional Notes

### Why Standard Eval Works

The working scripts (`eval_comprehensive.slurm` and `eval_results_official_moirai`) don't have this issue because:

1. They use `cli/eval.py` instead of `cli/eval_precond_space.py`
2. `eval.py` doesn't filter any samples - all samples go through
3. Preconditioning (if enabled) is reversed before evaluation
4. GluonTS evaluation naturally handles NaN using masked arrays
5. Predictions and labels always have the same shape (batch_size, pred_len)

### Dataset NaN Statistics from Logs

| Dataset | Total Samples | Original NaN | Skipped (New NaN) | Valid | Status |
|---------|--------------|--------------|-------------------|-------|--------|
| Bitcoin | 18 | 18 | 18 | 0 | Failed |
| NN5_Daily | 111 | 111 | 111 | 0 | Failed |
| Rideshare | 2304 | 2304 | 2304 | 0 | Failed |
| Sunspot | 1 | 1 | 1 | 0 | Failed |
| Vehicle_Trips | 329 | 207 | 193 | 136 | Broadcast Error |
| Temperature_Rain | 32072 | 30921 | 30913 | 1159 | Broadcast Error |
| KDD_Cup_2018 | 270 | 270 | 269 | 1 | Partial Success |

### Preconditioning Degree Impact

The paper recommends degree 2-10, with degree 5 being a good middle ground. However:
- Higher degrees require more historical timesteps
- More timesteps = more opportunities for NaN to propagate
- Datasets with sparse NaN may benefit from lower degrees (2-3)
- Datasets without NaN are unaffected by degree choice regarding NaN propagation
