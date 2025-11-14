# Preconditioning Evaluation Issue & Workaround

## Problem Statement

Your model was trained **WITH preconditioning**, but the current evaluation code does **NOT** properly reverse preconditioning during evaluation. This creates a potential issue for fair metric comparison.

## Background: Training vs Evaluation Pipeline

### During Training

```
Raw Time Series (y_t)
    ↓
PolynomialPrecondition: ỹ_t = y_t - Σc_i·y_{t-i}
    ↓
PackedStdScaler: Normalize to mean=0, std=1
    ↓
Model Training (learns on preconditioned + scaled data)
```

### Current Evaluation (Problematic)

```
Raw Time Series (y_t)  ← Original scale
    ↓
PackedStdScaler: Normalize  ← No preconditioning!
    ↓
Model Inference
    ↓
Predictions (what space are they in?)
    ↓
Metrics Computed
```

**Problem:** The model was trained on preconditioned data but evaluated on non-preconditioned data. This distribution mismatch could affect performance.

## What Actually Happens

When you load a checkpoint and run evaluation:

1. ✓ The trained model weights are loaded correctly
2. ✗ Input test data is NOT preconditioned (unlike training data)
3. ? Model makes predictions on data with different characteristics
4. ? Metrics may not reflect true performance

## Current Status

**The `moirai_lightning_ckpt_precond.yaml` config file exists but:**
- The `MoiraiForecast` class doesn't actually implement `reverse_preconditioning` parameter
- These parameters are ignored during evaluation
- No automatic preconditioning reversal occurs

**Checked files:**
- `/uni2ts/src/uni2ts/model/moirai/forecast.py` - No reverse_precondition support
- `/uni2ts/src/uni2ts/model/moirai/pretrain.py` - Has preconditioning but only for training
- `/uni2ts/src/uni2ts/transform/precondition.py` - Has ReversePrecondition class but not used in eval

## Workaround: Run Evaluation Anyway

The script `eval_precond_monash.slurm` will:

1. Load your preconditioned model checkpoint
2. Run evaluation using standard pipeline (no preconditioning)
3. Report metrics
4. Add warning about potential distribution mismatch

**To run:**
```bash
sbatch eval_precond_monash.slurm
```

## Interpreting Results

### Scenario 1: Performance is Good
- Model may be robust to distribution mismatch
- PackedStdScaler normalization may compensate
- Results might still be meaningful

### Scenario 2: Performance is Poor
- Could indicate distribution mismatch issue
- Model expects preconditioned data but receives raw data
- Metrics don't reflect true capability

### Scenario 3: Performance is Similar to Non-preconditioned
- Preconditioning may not have had significant impact
- Model learned representations that generalize
- OR there's still a distribution issue masking the true effect

## Proper Solution (Not Yet Implemented)

### What Should Happen

```
Raw Test Data (y_t)
    ↓
Apply Same Preconditioning: ỹ_t = y_t - Σc_i·y_{t-i}
    ↓
PackedStdScaler: Normalize
    ↓
Model Inference → Predictions (in preconditioned space)
    ↓
Reverse Preconditioning: ŷ_t = ỹ̂_t + Σc_i·y_{t-i}
    ↓
Predictions in Original Scale
    ↓
Compute Metrics (fair comparison!)
```

### Required Implementation

**Option 1: Modify MoiraiForecast class**

Add to `forecast.py`:
```python
class MoiraiForecast(L.LightningModule):
    def __init__(
        self,
        # ... existing params ...
        reverse_preconditioning: bool = False,
        precondition_type: str = "chebyshev",
        precondition_degree: int = 5,
    ):
        # Store preconditioning config
        # Apply preconditioning to inputs
        # Reverse preconditioning on outputs
```

**Option 2: Custom evaluation script**

Create custom eval that:
1. Applies preconditioning to test data manually
2. Runs model inference
3. Reverses preconditioning on predictions
4. Computes metrics

## Recommendation

### Short Term (Now)

1. **Run the evaluation anyway** using `eval_precond_monash.slurm`
2. **Document the limitation** in your results
3. **Compare trends** rather than absolute numbers
4. **Note:** If preconditioned model performs *similarly* to non-preconditioned, preconditioning might not be helping

### Medium Term (Implementation Needed)

1. Implement proper preconditioning reversal in `MoiraiForecast`
2. Update evaluation pipeline to handle preconditioned models
3. Re-run evaluations for fair comparison

### What to Compare

**You CAN compare:**
- Training loss curves (both use same data pipeline)
- Relative improvements between models
- Convergence speed
- Training stability

**You should be CAREFUL comparing:**
- Final evaluation metrics (may have distribution mismatch)
- Absolute performance numbers
- Generalization quality

## Files Created

1. **`eval_precond_monash.slurm`** - Evaluation script (with limitations)
2. **`PRECOND_EVAL_ISSUE.md`** - This document
3. **`TRAINING_LOSS_EXPLANATION.md`** - Training loss comparison guide

## Next Steps

1. Run evaluation: `sbatch eval_precond_monash.slurm`
2. Compare training loss curves (these ARE directly comparable)
3. Check if evaluation metrics make sense
4. Consider implementing proper preconditioning reversal for fair comparison

## ✅ SOLUTION IMPLEMENTED

**UPDATE:** The preconditioning evaluation issue has been FIXED!

See **`PRECOND_EVAL_SOLUTION.md`** for complete documentation.

### Quick Start

To evaluate your preconditioned model with proper preconditioning reversal:

```bash
sbatch eval_precond_monash.slurm
```

Or using the Python API:
```python
from uni2ts.model.moirai.forecast_precond import create_precond_forecast_from_checkpoint

model = create_precond_forecast_from_checkpoint(
    checkpoint_path="/path/to/checkpoint.ckpt",
    ...,
    enable_preconditioning=True,
    precondition_type="chebyshev",
    precondition_degree=5,
)

predictor = model.create_predictor(batch_size=32)
# Preconditioning reversal is automatic!
```

### What Was Fixed

1. ✅ `MoiraiForecastPrecond.get_default_transform()` - Now adds preconditioning to input transform
2. ✅ `MoiraiForecastPrecond.create_predictor()` - Now wraps predictor with reversal
3. ✅ `PreconditionReversingPredictor` - New class that reverses preconditioning on predictions
4. ✅ `eval_precond_monash.slurm` - Updated to use correct config
5. ✅ Testing - Verified reversal is mathematically accurate

### Verification

```bash
python test_precond_simple.py
# ✓ Reversal is accurate! (max diff < 1e-10)
```

---

## Original Questions

**Q: Can I trust the evaluation metrics?**
A: **YES!** With the new implementation, metrics are computed in the original scale after proper preconditioning reversal.

**Q: Why does the config file have reverse_preconditioning parameters?**
A: The correct config is `moirai_precond_ckpt.yaml` which is now fully implemented and working.

**Q: Should I retrain without preconditioning?**
A: No! You can now evaluate fairly with the fixed pipeline.

**Q: What's the priority fix?**
A: ~~Implement preconditioning reversal~~ **✅ DONE!**
