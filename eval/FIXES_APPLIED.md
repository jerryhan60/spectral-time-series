# Fixes Applied to Evaluation Scripts

## Date: 2025-11-16

## Issues Fixed

### 1. eval_baseline_in_precond_space - Multiple Configuration Issues

#### Issue 1a: Hydra Configuration Errors (SLURM script)

**Problems**:
```
Could not override 'precond_type'.
Key 'precond_type' is not in struct

Could not override 'val_data'.
Key 'val_data' is not in struct
```

**Root Cause**:
The script was trying to override parameters that don't exist in the base eval config structure:
- `precond_type` and `precond_degree`: Expected as top-level params by the Python script (line 121: `cfg.get("precond_type", ...)`)
- `val_data`: Expected as top-level param by the Python script (line 133: `call(cfg.val_data)`)
- The default eval config (`conf/eval/default.yaml`) only defines: `model`, `data`, `batch_size`, `metrics`

**Fix Applied** - `eval_baseline_in_precond_space.slurm` (lines 206-207, 212-214):
```bash
+precond_type=$PRECOND_TYPE \
+precond_degree=$PRECOND_DEGREE \
+val_data=monash_cached \
+val_data.dataset_name=$dataset_name \
+val_data.prediction_length=$prediction_length \
```

The `+` prefix tells Hydra to **add** these parameters to the config rather than **override** existing nested parameters.

#### Issue 1b: Model Loading Error (Python script)

**Problem**:
```
TypeError: MoiraiForecast.__init__() missing 4 required positional arguments:
'prediction_length', 'target_dim', 'feat_dynamic_real_dim', and 'past_feat_dynamic_real_dim'
```

**Root Cause**:
The script called `call(cfg.model)` directly, but `load_from_checkpoint` requires these dimension arguments. The script needs to:
1. Load test data first to get metadata
2. Use `_partial_=True` to create a partial function
3. Call it with metadata arguments

**Fix Applied** - `eval_baseline_in_precond_space.py` (lines 129-162):

**Before**:
```python
model = call(cfg.model)  # Missing required arguments!
test_data_input = call(cfg.val_data)  # Wrong - returns string, not data object
```

**After**:
```python
# Load data config to get metadata (cfg.data returns tuple: test_data, metadata)
_, metadata = call(cfg.data)

# Load actual test data from val_data
test_data_input, _ = call(cfg.val_data)

# Create partial function and call with metadata
model = call(cfg.model, _partial_=True, _convert_="all")(
    prediction_length=metadata.prediction_length,
    target_dim=metadata.target_dim,
    feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
    past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
)
```

**Key Insight**:
- `cfg.data` is used to get metadata (both data and val_data should have same metadata)
- `cfg.val_data` is used to get the actual test data for evaluation
- Both return tuples: `(test_data, metadata)`

This matches how `eval.py` loads models (line 37-40).

**Impact**: The script can now successfully load baseline model checkpoints and evaluate them in preconditioned space.

---

### 2. eval_precond_space.py - NaN Propagation and Shape Mismatch

**Problem**:
6 out of 30 datasets were failing:
- **Complete failures**: Bitcoin, NN5_Daily, Rideshare, Sunspot (all samples filtered out)
- **Broadcast errors**: Vehicle_Trips, Temperature_Rain (shape mismatch between predictions and filtered labels)
- **Partial success**: KDD_Cup_2018 (only 1 valid sample)

**Root Causes**:

1. **NaN Propagation**: Preconditioning formula `ỹₜ = yₜ + Σᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ` requires n previous timesteps (CORRECTED 2025-11-17: uses addition)
   - If any previous timestep has NaN, the result becomes NaN (NaN + x = NaN)
   - Degree 5 preconditioning needs 5 previous values, so NaN spreads easily
   - Datasets with extensive NaN have all/most samples filtered out

2. **Shape Mismatch** (The Critical Bug):
   - Script generated predictions for ALL samples (e.g., 329 samples)
   - But preconditioned ground truth only contained VALID samples (e.g., 136 after filtering 193)
   - When computing metrics: prediction shape (32, 30) vs label shape (8, 30) → **broadcast error**
   - This was because predictions and labels were misaligned

**Fix Applied**:

Modified `eval_precond_space.py` to track valid sample indices and only generate predictions for those samples:

1. **Updated `precondition_ground_truth()` function** (lines 76, 84, 136, 177):
   - Track `valid_indices = []` list
   - Use `for idx, item in enumerate(...)` to track sample indices
   - Append `valid_indices.append(idx)` for each valid sample
   - Return `return PreconditionedTestData(...), valid_indices`

2. **Updated `evaluate_in_preconditioned_space()` function** (lines 209, 221, 238):
   - Unpack: `preconditioned_test_data, valid_indices = precondition_ground_truth(...)`
   - Filter: `valid_test_data = [test_data_list[i] for i in valid_indices]`
   - Predict only for valid: `predictor.predict(get_inputs(valid_test_data), ...)`

**Before**:
```python
# Generate predictions for ALL samples
forecast_it = predictor.predict(get_inputs(test_data_list), ...)  # 329 samples

# Compare against VALID labels only
res = evaluate_forecasts(
    forecasts=forecast_list,         # 329 predictions
    test_data=preconditioned_test_data,  # 136 labels
    ...
)
# --> SHAPE MISMATCH!
```

**After**:
```python
# Filter to valid samples
valid_test_data = [test_data_list[i] for i in valid_indices]  # 136 samples

# Generate predictions ONLY for valid samples
forecast_it = predictor.predict(get_inputs(valid_test_data), ...)  # 136 samples

# Compare aligned predictions and labels
res = evaluate_forecasts(
    forecasts=forecast_list,         # 136 predictions
    test_data=preconditioned_test_data,  # 136 labels
    ...
)
# --> SHAPES MATCH!
```

**Impact**:
- ✅ **Vehicle_Trips** now evaluates successfully (was broadcast error)
- ✅ **Temperature_Rain** now evaluates successfully (was broadcast error)
- ✅ Evaluation completes on 26/30 datasets (87% coverage, up from 80%)
- ❌ **Bitcoin, NN5_Daily, Rideshare, Sunspot** still fail (0 valid samples - expected)

These 4 datasets cannot be evaluated in preconditioned space with degree 5 due to 100% NaN propagation. They would need:
- Evaluation in original space (using `eval.py` with reversal), OR
- Lower polynomial degree (2-3 instead of 5), OR
- Accept they are unsuitable for preconditioned space evaluation

See `NAN_HANDLING_FIX.md` for complete technical details.

---

## Summary of Changes

| File | Lines Changed | Type of Fix | Status |
|------|---------------|-------------|--------|
| `eval_baseline_in_precond_space.slurm` | 206-207, 212-214 | Config parameter fix (Hydra + prefix) | ✅ Complete |
| `eval_baseline_in_precond_space.py` | 129-162 | Model loading with metadata, fixed data loading | ✅ Complete |
| `eval_precond_space.py` | 76, 84, 136, 177, 209, 221, 238 | Track valid indices, align predictions | ✅ Complete |
| `eval_precond_comprehensive.slurm` | (reverted) | Removed skip logic | ✅ No longer needed |

## Testing Recommendations

### Test eval_baseline_in_precond_space.slurm:
```bash
# Test with a single dataset first
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -m cli.eval_baseline_in_precond_space \
  model=moirai_lightning_ckpt \
  model.checkpoint_path=/path/to/baseline.ckpt \
  model.patch_size=32 \
  model.context_length=1000 \
  +precond_type=chebyshev \
  +precond_degree=5 \
  batch_size=32 \
  data=monash_cached \
  data.dataset_name=m1_monthly \
  data.prediction_length=18
```

### Test eval_precond_comprehensive.slurm:
```bash
# Submit with a preconditioned checkpoint
cd /scratch/gpfs/EHAZAN/jh1161
sbatch --export=MODEL_PATH=/path/to/precond_checkpoint.ckpt eval/eval_precond_comprehensive.slurm

# Monitor progress
tail -f logs/eval_precond_comprehensive_*.out
```

## Related Documentation

- `EVAL_PRECOND_SPACE_ISSUES.md`: Detailed analysis of NaN propagation issues and long-term solutions
- `CLAUDE.md`: Overall repository guide (section on evaluation approaches)
- `EVAL_PRECOND_README.md`: Guide to preconditioned space evaluation

## Future Work

1. **Implement index tracking** in `eval_precond_space.py` to handle partial filtering without shape mismatches
2. **Add masked array support** to preconditioning transform for better NaN handling
3. **Test lower degrees** (2-3) on NaN-heavy datasets to see if they can be evaluated
4. **Document dataset characteristics** to understand which datasets are unsuitable for preconditioned space evaluation
