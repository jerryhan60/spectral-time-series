# Preconditioning Evaluation Solution

## Summary

This document describes the **complete solution** for properly evaluating models trained with preconditioning. The solution ensures fair comparison by:

1. **Applying preconditioning to test data** (same as training)
2. **Running inference** in preconditioned space
3. **Reversing preconditioning** on predictions (back to original scale)
4. **Computing metrics** in original scale

## Problem Recap

Models trained WITH preconditioning were being evaluated WITHOUT preconditioning, creating a distribution mismatch:

**Training Pipeline:**
```
Raw Data → PolynomialPrecondition → Standardize → Model Training
```

**Old Evaluation Pipeline (WRONG):**
```
Raw Data → Standardize → Model Inference → Metrics
```

**New Evaluation Pipeline (CORRECT):**
```
Raw Data → PolynomialPrecondition → Standardize → Model Inference
         → Reverse Preconditioning → Metrics (in original scale)
```

## Implementation Components

### 1. PreconditionReversingPredictor (New Class)

**File:** `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast_precond.py`

**Purpose:** Wraps a base predictor to apply reverse preconditioning on all predictions.

**Key Features:**
- Wraps any GluonTS predictor
- Extracts context data from input to properly initialize reversal
- Handles both univariate and multivariate time series
- Creates `ReversedForecast` objects with reversed samples

### 2. MoiraiForecastPrecond (Enhanced Class)

**File:** `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast_precond.py`

**Enhancements:**

#### `get_default_transform()` Override
```python
def get_default_transform(self) -> Transformation:
    """Adds PolynomialPrecondition to the transform chain."""
    transform = super().get_default_transform()
    
    if self.enable_preconditioning:
        precond_transform = PolynomialPrecondition(
            polynomial_type=self.precondition_type,
            degree=self.precondition_degree,
            target_field="target",
            enabled=True,
            store_original=True,
        )
        transform = precond_transform + transform
    
    return transform
```

#### `create_predictor()` Override
```python
def create_predictor(self, batch_size: int, device: str = "auto") -> Predictor:
    """Wraps predictor with preconditioning reversal."""
    base_predictor = super().create_predictor(batch_size, device)
    
    if not self.enable_preconditioning:
        return base_predictor
    
    return PreconditionReversingPredictor(
        base_predictor=base_predictor,
        precondition_coeffs=self.preconditioner.coeffs,
        precondition_degree=self.precondition_degree,
        precondition_type=self.precondition_type,
    )
```

### 3. Hydra Configuration

**File:** `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/model/moirai_precond_ckpt.yaml`

```yaml
_target_: uni2ts.model.moirai.forecast_precond.create_precond_forecast_from_checkpoint
checkpoint_path: ???
num_samples: 100
patch_size: ???
context_length: ???

# Preconditioning parameters
# IMPORTANT: These MUST match the training configuration!
enable_preconditioning: true
precondition_type: chebyshev  # "chebyshev" or "legendre"
precondition_degree: 5
```

### 4. Updated SLURM Script

**File:** `/scratch/gpfs/EHAZAN/jh1161/eval_precond_monash.slurm`

**Key Change:** Now uses `model=moirai_precond_ckpt` instead of `model=moirai_lightning_ckpt`

```bash
python -m cli.eval \
  run_name=eval_${CKPT_NAME}_${ds} \
  model=moirai_precond_ckpt \
  model.checkpoint_path=$CHECKPOINT_PATH \
  model.patch_size=$PATCH_SIZE \
  model.context_length=$CONTEXT_LENGTH \
  model.enable_preconditioning=true \
  model.precondition_type=$PRECOND_TYPE \
  model.precondition_degree=$PRECOND_DEGREE \
  batch_size=$BATCH_SIZE \
  data=monash_cached \
  data.dataset_name=$ds
```

## Usage Instructions

### For Standard Evaluation

**1. Run the SLURM script:**
```bash
sbatch eval_precond_monash.slurm
```

This will evaluate your preconditioned model on all Monash datasets with proper preconditioning reversal.

### For Custom Evaluation

**1. Use the Hydra CLI:**
```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

python -m cli.eval \
  run_name=my_precond_eval \
  model=moirai_precond_ckpt \
  model.checkpoint_path=/path/to/checkpoint.ckpt \
  model.patch_size=32 \
  model.context_length=1000 \
  model.enable_preconditioning=true \
  model.precondition_type=chebyshev \
  model.precondition_degree=5 \
  batch_size=32 \
  data=monash_cached \
  data.dataset_name=m4_monthly
```

**2. Use Python API:**
```python
from uni2ts.model.moirai.forecast_precond import create_precond_forecast_from_checkpoint
from uni2ts.eval_util.data import get_gluonts_test_dataset
from uni2ts.eval_util.evaluation import evaluate_model

# Load model
model = create_precond_forecast_from_checkpoint(
    checkpoint_path="/path/to/checkpoint.ckpt",
    prediction_length=24,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
    context_length=1000,
    patch_size=32,
    num_samples=100,
    enable_preconditioning=True,
    precondition_type="chebyshev",
    precondition_degree=5,
)

# Load data
test_data, metadata = get_gluonts_test_dataset(
    dataset_name="m4_monthly",
    use_lotsa_cache=True,
)

# Create predictor (automatically includes preconditioning reversal)
predictor = model.create_predictor(batch_size=32, device="cuda")

# Evaluate (preconditioning is handled automatically!)
metrics = evaluate_model(
    predictor,
    test_data=test_data,
    metrics=...,
    batch_size=32,
)

print(metrics)
```

## Important Configuration Notes

### Preconditioning Parameters MUST Match Training

The evaluation preconditioning parameters **MUST EXACTLY MATCH** the training configuration:

| Parameter | Description | Common Values |
|-----------|-------------|---------------|
| `enable_preconditioning` | Whether preconditioning is used | `true` or `false` |
| `precondition_type` | Polynomial type | `chebyshev` or `legendre` |
| `precondition_degree` | Polynomial degree | `5` (typical), `3-10` (range) |

**To find your training config:**
```bash
# Look in your training output directory
cat /scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/.../config.yaml
```

### Disable Preconditioning for Non-Preconditioned Models

For models trained WITHOUT preconditioning, either:

**Option 1:** Use the standard config:
```bash
model=moirai_lightning_ckpt  # Standard, no preconditioning
```

**Option 2:** Use precond config with it disabled:
```bash
model=moirai_precond_ckpt
model.enable_preconditioning=false
```

## Verification and Testing

### Test 1: Basic Preconditioning Reversal

```bash
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
python test_precond_simple.py
```

**Expected output:**
```
✓ Reversal is accurate! (max diff < 1e-10)
✓ Test passed!
```

### Test 2: End-to-End Evaluation (Small Dataset)

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate

python -m cli.eval \
  run_name=test_precond \
  model=moirai_precond_ckpt \
  model.checkpoint_path=/path/to/checkpoint.ckpt \
  model.patch_size=32 \
  model.context_length=1000 \
  model.enable_preconditioning=true \
  model.precondition_type=chebyshev \
  model.precondition_degree=5 \
  batch_size=32 \
  data=monash_cached \
  data.dataset_name=m4_monthly
```

## Technical Details

### How Reversal Works

The preconditioning transformation is:
```
ỹₜ = yₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ
```

The reversal is:
```
yₜ = ỹₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ
```

**Key insight:** Reversal is iterative and needs history. The first `n` timesteps use values from the context (previous `n` points from the time series).

### Context Handling

The `PreconditionReversingPredictor` automatically:
1. Extracts the last `degree` points from the input context
2. Prepends them to predictions before reversal
3. Removes them after reversal
4. Returns predictions in original scale

### Multivariate Support

Both preconditioning and reversal work with multivariate time series:
- Each variate is processed independently
- No cross-variate dependencies are introduced
- Shape handling: `[num_samples, pred_len, num_variates]`

## Comparison: Before vs After

### Before (Incorrect)

```python
# Model trained on: Preconditioned + Standardized data
# Model evaluated on: Standardized data (NO preconditioning!)
# ❌ Distribution mismatch! Metrics not comparable.
```

### After (Correct)

```python
# Model trained on: Preconditioned + Standardized data
# Model evaluated on: Preconditioned + Standardized data
# Then reversed to: Original scale
# ✅ Fair comparison! Metrics are meaningful.
```

## Troubleshooting

### Issue: Import Error

**Error:** `ModuleNotFoundError: No module named 'uni2ts.model.moirai.forecast_precond'`

**Solution:**
```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
source venv/bin/activate
python -c "from uni2ts.model.moirai.forecast_precond import create_precond_forecast_from_checkpoint"
```

### Issue: Wrong Preconditioning Parameters

**Symptom:** Poor metrics, much worse than training loss suggests

**Solution:** Verify parameters match training:
```bash
# Check training config
cat /path/to/training/output/config.yaml | grep -A3 precondit

# Ensure eval uses same parameters
```

### Issue: Predictions in Wrong Scale

**Symptom:** Predictions are orders of magnitude off

**Possible Causes:**
1. `enable_preconditioning=false` when it should be `true`
2. Wrong `precondition_type` or `precondition_degree`
3. Using `moirai_lightning_ckpt` instead of `moirai_precond_ckpt`

**Solution:** Use `moirai_precond_ckpt` with correct parameters.

## Files Modified/Created

### Modified Files
1. `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast_precond.py`
   - Added `PreconditionReversingPredictor` class
   - Enhanced `MoiraiForecastPrecond` with proper transforms
   - Fixed `forward()` method comments

2. `/scratch/gpfs/EHAZAN/jh1161/eval_precond_monash.slurm`
   - Updated to use `moirai_precond_ckpt` config
   - Added preconditioning parameters
   - Updated documentation

### Created Files
1. `/scratch/gpfs/EHAZAN/jh1161/test_precond_simple.py`
   - Simple test for preconditioning reversal
   - Verifies mathematical correctness

2. `/scratch/gpfs/EHAZAN/jh1161/PRECOND_EVAL_SOLUTION.md`
   - This documentation file

### Existing Files (Already Present)
1. `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/model/moirai_precond_ckpt.yaml`
   - Config for preconditioning-aware evaluation

2. `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/precondition.py`
   - `PolynomialPrecondition` and `ReversePrecondition` classes
   - Already well-implemented, no changes needed

## Next Steps

1. **Run full evaluation:**
   ```bash
   sbatch eval_precond_monash.slurm
   ```

2. **Monitor progress:**
   ```bash
   tail -f logs/eval_precond_*.out
   ```

3. **Analyze results:**
   ```bash
   # Results are saved in:
   ls uni2ts/outputs/eval/monash_cached/*/
   ```

4. **Compare with non-preconditioned model:**
   - Now metrics are directly comparable!
   - Check if preconditioning improved performance
   - Analyze per-dataset results

## References

- **Preconditioning Paper:** Marsden, A., & Hazan, E. (2025). Universal Sequence Preconditioning. arXiv:2502.06545
- **Original Issue:** `PRECOND_EVAL_ISSUE.md`
- **Code:** `uni2ts/src/uni2ts/model/moirai/forecast_precond.py`
