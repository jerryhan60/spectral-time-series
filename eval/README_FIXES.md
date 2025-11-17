# Quick Reference: Evaluation Fixes

## What Was Fixed

### ✅ Issue 1: Hydra Config Errors in baseline evaluation
**File**: `eval_baseline_in_precond_space.slurm`
**Errors**:
- `Could not override 'precond_type'. Key 'precond_type' is not in struct`
- `Could not override 'val_data'. Key 'val_data' is not in struct`

**Fix**: Added `+` prefix to all new config parameters not in base config:
- `+precond_type=$PRECOND_TYPE`
- `+precond_degree=$PRECOND_DEGREE`
- `+val_data=monash_cached`
- `+val_data.dataset_name=$dataset_name`
- `+val_data.prediction_length=$prediction_length`

### ✅ Issue 2: Shape Mismatch / Broadcast Error in preconditioned evaluation
**Files**: `eval_precond_space.py`
**Error**: `ValueError: operands could not be broadcast together with shapes (8,30) (32,30)`
**Fix**: Track which samples pass preconditioning and only generate predictions for those samples

## How to Use

### Test the Fix
```bash
cd /scratch/gpfs/EHAZAN/jh1161/eval
bash test_nan_fix.sh
```

### Run Full Evaluation
```bash
cd /scratch/gpfs/EHAZAN/jh1161
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt eval/eval_precond_comprehensive.slurm
```

## What to Expect

### Previously Failing Datasets:
- ✅ **Vehicle_Trips**: Now works (was broadcast error)
- ✅ **Temperature_Rain**: Now works (was broadcast error)
- ❌ **Bitcoin**: Still fails (0 valid samples - expected)
- ❌ **NN5_Daily**: Still fails (0 valid samples - expected)
- ❌ **Rideshare**: Still fails (0 valid samples - expected)
- ❌ **Sunspot**: Still fails (0 valid samples - expected)

### Coverage:
- **Before**: 24/30 datasets (80%) with workaround skip logic
- **After**: 26/30 datasets (87%) with proper NaN handling

### The 4 Datasets That Still Fail:
These datasets have 100% NaN propagation (all samples get filtered out). This is expected and unavoidable with degree 5 preconditioning. They can be evaluated in original space using `eval.py` instead.

## Documentation

- **NAN_HANDLING_FIX.md**: Complete technical details of the fix
- **EVAL_PRECOND_SPACE_ISSUES.md**: Original problem analysis
- **FIXES_APPLIED.md**: Summary of all fixes with code examples

## Key Changes Made

1. **eval_precond_space.py**:
   - `precondition_ground_truth()` now returns `(data, valid_indices)` tuple
   - `evaluate_in_preconditioned_space()` filters inputs to valid samples before prediction
   - Predictions and labels are now perfectly aligned

2. **eval_baseline_in_precond_space.slurm**:
   - Uses `+precond_type` instead of `precond_type` for Hydra config

3. **eval_precond_comprehensive.slurm**:
   - Removed skip logic (no longer needed)

## Quick Commands

```bash
# Test single dataset
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -m cli.eval_precond_space \
  model=moirai_precond_ckpt_no_reverse \
  model.checkpoint_path=/path/to/checkpoint.ckpt \
  model.patch_size=32 \
  model.precondition_type=chebyshev \
  model.precondition_degree=5 \
  data=monash_cached \
  data.dataset_name=vehicle_trips_with_missing \
  data.prediction_length=30

# Run comprehensive evaluation
sbatch --export=MODEL_PATH=/path/to/checkpoint.ckpt eval/eval_precond_comprehensive.slurm

# Monitor progress
tail -f logs/eval_precond_comprehensive_*.out
```
