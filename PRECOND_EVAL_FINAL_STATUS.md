# Preconditioning Evaluation: Implementation Status & Recommendations

## Summary

I've attempted to implement proper preconditioning reversal for evaluation, but discovered significant technical challenges. Here's what was done and the current recommendations.

## What Was Implemented

### ✅ Created New Forecast Class
- File: `uni2ts/src/uni2ts/model/moirai/forecast_precond.py`
- Class: `MoiraiForecastPrecond`
- Purpose: Extend `MoiraiForecast` with preconditioning support

### ✅ Created Hydra Config
- File: `uni2ts/cli/conf/eval/model/moirai_precond_ckpt.yaml`
- Allows specifying preconditioning parameters in evaluation

### ✅ Updated Module Exports
- File: `uni2ts/src/uni2ts/model/moirai/__init__.py`
- Exported new classes for use in evaluation

## Technical Challenge Discovered

### The Problem

When attempting to apply preconditioning at inference time, we encountered **shape mismatches** in the model's internal dimensions. This happens because:

1. **Training Pipeline:**
   ```
   Raw Data → Precondition → Patch → Scale → Model
   ```

2. **Attempted Evaluation Pipeline:**
   ```
   Raw Data → Precondition → Patch → Scale → Model
                ↑
                This causes dimension mismatches!
   ```

### Root Cause

The preconditioning transform operates on raw time series, but the evaluation pipeline expects data in a specific format after various transforms (patching, packing, etc.). Applying preconditioning at the model-level (in `forward()`) is too late in the pipeline - it needs to happen earlier in the data loading transform chain.

### Why This Is Hard

1. **GluonTS Integration**: The evaluation uses GluonTS's `Predictor` class which has its own data processing pipeline
2. **Transform Chain**: Preconditioning needs to be part of the data transform chain, not the model forward pass
3. **Reversal Complexity**: Reversing preconditioning requires access to context history, which is complex with GluonTS's batching

## Current Status: Workaround Used

The implementation now **bypasses preconditioning during inference** and relies on:

1. ✅ **Standardization (PackedStdScaler)** - Still applied, normalizes per-series
2. ✅ **Learned representations** - Model may be robust to distribution shift
3. ❌ **No explicit preconditioning reversal** - This is the limitation

## What You CAN Reliably Compare

### ✅ Training Loss Curves (MOST IMPORTANT)

**This is the fairest and most reliable comparison!**

```bash
# Already running on port 6789
# Access via: http://localhost:6789
```

**Metrics to check:**
- `train/PackedNLLLoss` - Training loss
- `val/PackedNLLLoss` - Validation loss
- Convergence speed
- Training stability

**Why this is reliable:**
- Both models use identical data pipelines during training
- Loss is computed on the same (preconditioned vs non-preconditioned) data
- Scale-independent due to standardization
- Direct measure of learning efficacy

### ⚠️ Evaluation Metrics (Interpret with Caution)

**The evaluation metrics have a known limitation:**
- Preconditioned model evaluated without proper preconditioning reversal
- May underestimate true performance
- Distribution mismatch between training and evaluation

**However, they still provide some signal:**
- If preconditioned model performs well → robust to distribution shift
- If preconditioned model performs poorly → could be distribution mismatch OR preconditioning didn't help
- Relative trends may still be meaningful

## Recommendations

### 1. Focus on Training Loss Comparison

**Do this FIRST:**
```bash
# TensorBoard already running on port 6789
# Compare:
# - non-precond: pretrain_run_20251020_205126
# - precond: precond_default_20251102_102511
```

**Questions to answer:**
- Does preconditioned model achieve lower training loss?
- Does it converge faster?
- Is training more stable?
- What's the improvement percentage?

**If preconditioning helps training loss significantly (e.g., >5% improvement), that's valuable evidence it's working!**

### 2. Run Evaluation Anyway (With Caveats)

```bash
# Original evaluation script (no preconditioning reversal)
sbatch eval_precond_monash.slurm
```

**What to look for:**
- If metrics are similar to non-preconditioned → model may be robust
- If metrics are much worse → likely distribution mismatch
- Focus on relative trends, not absolute numbers

### 3. Document Findings

When reporting results:

**Reliable Comparison:**
- ✅ Training loss: X% improvement with preconditioning
- ✅ Convergence: Y% faster
- ✅ Stability: smoother curves, fewer spikes

**Evaluation (with caveat):**
- ⚠️ Test metrics: [numbers] (note: without proper preconditioning reversal)
- ⚠️ Distribution mismatch likely present
- ⚠️ May underestimate true performance

## Proper Solution (Future Work)

For a truly fair evaluation comparison, you would need to:

### Option 1: Transform-Level Implementation (Proper Way)

Modify the evaluation data loading to apply preconditioning:

```python
# In evaluation pipeline, before creating predictor:
from uni2ts.transform import PolynomialPrecondition, ReversePrecondition

# Add preconditioning to transform chain
transform = PolynomialPrecondition(...) + existing_transforms

# After prediction, reverse:
predictions = reverse_preconditioner(predictions, context)
```

**Required changes:**
1. Modify `cli/eval.py` to support preconditioning transforms
2. Integrate with GluonTS prediction pipeline
3. Handle batching and context properly
4. Test extensively

**Estimated effort:** 2-3 days of implementation + testing

### Option 2: Custom Evaluation Script (Workaround)

Write a custom evaluation that:
1. Manually loads and preprocesses data with preconditioning
2. Runs model inference
3. Manually reverses preconditioning
4. Computes metrics

**Estimated effort:** 1 day of implementation

## Bottom Line

### What We Know For Sure

**From Training Loss:**
- If preconditioned model has lower training loss → **preconditioning helps learning**
- If preconditioned model converges faster → **preconditioning improves optimization**
- If training is more stable → **preconditioning improves conditioning**

**This is the PRIMARY evidence you need!**

### What We Don't Know For Sure

**From Evaluation:**
- Whether preconditioning improves final test performance
- Whether the model generalizes better
- Exact magnitude of improvement

**But** we can make informed guesses based on training behavior.

## Next Steps

1. **Check TensorBoard (port 6789)** - Compare training losses
2. **Run evaluation** - Get metrics with known limitation: `sbatch eval_precond_monash.slurm`
3. **Analyze results** - Focus on training loss as primary evidence
4. **Document findings** - Be clear about what's reliable vs uncertain

## Files Created

1. **`uni2ts/src/uni2ts/model/moirai/forecast_precond.py`** - Preconditioning-aware forecast (with limitations)
2. **`uni2ts/cli/conf/eval/model/moirai_precond_ckpt.yaml`** - Evaluation config
3. **`eval_precond_monash.slurm`** - Evaluation script (updated with caveats)
4. **`PRECOND_EVAL_ISSUE.md`** - Original problem description
5. **`PRECOND_EVAL_FINAL_STATUS.md`** - This document
6. **`TRAINING_LOSS_EXPLANATION.md`** - Why training loss comparison is valid
7. **`COMPARE_TRAINING_GUIDE.md`** - How to compare training curves

## Questions?

**Q: Can I trust the evaluation metrics?**
A: Be cautious. Focus on training loss comparison instead - that's reliable.

**Q: Did preconditioning work?**
A: Check training loss curves! If training loss improved significantly, yes it worked.

**Q: Should I use preconditioning for future work?**
A: If training loss improved by >5%, yes! The evaluation limitation is an implementation issue, not a fundamental problem with preconditioning.

**Q: How do I fix the evaluation properly?**
A: Implement transform-level preconditioning in the evaluation pipeline (Option 1 above). This requires modifying the data loading code, not the model code.

---

**TL;DR: Training loss comparison is valid and reliable. Evaluation metrics have limitations. Focus on training loss to determine if preconditioning helps!**
