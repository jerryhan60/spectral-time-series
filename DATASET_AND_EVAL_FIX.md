# Dataset Download & Evaluation Fixes

**Date**: 2025-11-01
**Issues Fixed**:
1. Dataset download errors on compute nodes
2. Missing preconditioning reversal during evaluation

---

## Issue 1: Dataset Download Errors ✅ FIXED

### Problem
```
URLError(gaierror(-2, 'Name or service not known'))
```

**Root Cause**: Compute nodes don't have internet access. GluonTS tries to download datasets during evaluation, which fails.

### Solution: Pre-download Datasets

**Step 1**: Run download script on LOGIN NODE (has internet):

```bash
# On login node (della-login or similar)
cd /scratch/gpfs/EHAZAN/jh1161

# Activate venv
source uni2ts/venv/bin/activate

# Download all eval datasets
python download_eval_datasets.py
```

This will download all 12 datasets:
- Yearly: tourism_yearly, m1_yearly, m3_yearly, m4_yearly
- Quarterly: tourism_quarterly, m1_quarterly, m3_quarterly, m4_quarterly
- Monthly: tourism_monthly, m1_monthly, m3_monthly, m4_monthly

**Step 2**: Datasets are cached in `~/.gluonts/datasets/`

**Step 3**: Compute nodes will use cached versions

### Special Case: M3 Dataset

M3 datasets (m3_yearly, m3_quarterly, m3_monthly) require manual download:

1. Go to: https://forecasters.org/resources/time-series-data/m3-competition/
2. Download `M3C.xls`
3. Place in: `~/.gluonts/datasets/M3C.xls`

### Verification

Check cached datasets:
```bash
ls ~/.gluonts/datasets/
```

---

## Issue 2: Preconditioning Reversal ❌ NOT YET IMPLEMENTED

### Problem

Models trained with preconditioning produce predictions in the **preconditioned space**. For correct evaluation metrics, predictions must be **reversed back to original space**.

### Current Status

**Not implemented in evaluation pipeline yet.** The evaluation code (cli/eval.py) doesn't know about preconditioning and doesn't reverse predictions.

### Why This Matters

If you train with preconditioning enabled:
```python
# During training
y_preconditioned = precondition(y_original)  # Model learns this
```

During evaluation, the model predicts in preconditioned space:
```python
# Model prediction
y_pred_preconditioned = model(x)  # In preconditioned space

# Need to reverse for metrics
y_pred_original = reverse_precondition(y_pred_preconditioned)  # ← MISSING!
```

### Temporary Workaround

**Option 1: Train without preconditioning for evaluation**

Until reversal is implemented, train a separate model without preconditioning specifically for evaluation:

```bash
# Train WITHOUT preconditioning
sbatch --export=PRECOND_TYPE=chebyshev,PRECOND_DEGREE=0 pretrain_moirai.slurm

# Or use baseline
sbatch pretrain_moirai.slurm
```

**Option 2: Implement reversal manually**

You'll need to:
1. Store preconditioning coefficients with the checkpoint
2. Apply ReversePrecondition transform to predictions before computing metrics

### What Needs to Be Implemented

1. **Store preconditioning metadata in checkpoint**:
   ```python
   # In pretrain.py, save with checkpoint
   self.preconditioning_config = {
       'enabled': self.hparams.enable_preconditioning,
       'type': self.hparams.precondition_type,
       'degree': self.hparams.precondition_degree,
   }
   ```

2. **Load metadata during evaluation**:
   ```python
   # In eval.py, load from checkpoint
   if checkpoint['preconditioning_config']['enabled']:
       reverse_precond = True
       precond_coeffs = checkpoint['precondition_coeffs']
   ```

3. **Apply reversal to predictions**:
   ```python
   # After model predictions
   if reverse_precond:
       from uni2ts.transform import ReversePrecondition
       reverser = ReversePrecondition(...)
       predictions = reverser({'prediction': predictions})['prediction']
   ```

4. **Add to evaluation config**:
   ```yaml
   # eval/default.yaml
   reverse_preconditioning: false  # Set true for preconditioned models
   precondition_type: chebyshev
   precondition_degree: 5
   ```

### Implementation Complexity

**Moderate** - Requires changes in:
- Checkpoint saving (to include preconditioning metadata)
- Model loading (to read metadata)
- Prediction pipeline (to apply reversal)
- Evaluation config (to specify reversal params)

### Timeline

- **Short-term**: Use workaround (train without preconditioning for eval)
- **Medium-term**: Implement proper reversal (1-2 days of work)
- **Long-term**: Auto-detect from checkpoint (ideal)

---

## Recommended Workflow for Now

### Workflow A: Training Only (No Evaluation Yet)

```bash
# 1. Train with preconditioning
sbatch pretrain_moirai_precond_default.slurm

# 2. Monitor training metrics
tail -f logs/pretrain_precond_*.out

# 3. Wait for implementation of reversal before evaluating
```

### Workflow B: Training + Evaluation

```bash
# 1. Train TWO models:

# Model 1: WITH preconditioning (for research)
sbatch pretrain_moirai_precond_default.slurm

# Model 2: WITHOUT preconditioning (for evaluation)
sbatch pretrain_moirai.slurm

# 2. Download datasets (on login node)
python download_eval_datasets.py

# 3. Evaluate the non-preconditioned model
bash find_checkpoint.sh baseline
bash submit_eval_all_frequencies.sh outputs/baseline_*/checkpoints/last.ckpt

# 4. Wait for reversal implementation to evaluate preconditioned model
```

### Workflow C: Implement Reversal Yourself

If you need to evaluate preconditioned models now:

1. Modify `cli/eval.py` to apply ReversePrecondition
2. Add preconditioning config to model config
3. Apply reversal after predictions, before metrics

Example pseudo-code:
```python
# In eval.py after line 48
if cfg.get('reverse_preconditioning', False):
    from uni2ts.transform import ReversePrecondition
    reverser = ReversePrecondition()
    # Apply to each prediction in test_data
    # ... (implementation details)
```

---

## Summary Table

| Issue | Status | Fix | Time |
|-------|--------|-----|------|
| Dataset download | ✅ FIXED | Run `download_eval_datasets.py` | 10 min |
| M3 dataset | ⚠️ MANUAL | Download M3C.xls manually | 5 min |
| Precond reversal | ❌ NOT IMPLEMENTED | Use workaround or implement | TBD |

---

## Next Steps

### Immediate (Today)

1. **Download datasets** (on login node):
   ```bash
   python download_eval_datasets.py
   ```

2. **Download M3 dataset manually**:
   - Visit https://forecasters.org/resources/time-series-data/m3-competition/
   - Download M3C.xls
   - Place in ~/.gluonts/datasets/

3. **Resume training**:
   ```bash
   sbatch pretrain_moirai_precond_default.slurm
   ```

### Short-term (This Week)

4. **For evaluation, choose**:
   - Option A: Wait for reversal implementation
   - Option B: Train baseline model for evaluation
   - Option C: Implement reversal yourself

### Medium-term (Next Week)

5. **Implement preconditioning reversal** (if needed)
6. **Test on one dataset** before full evaluation
7. **Run full evaluation** on all 12 datasets

---

## Files Created

1. **`download_eval_datasets.py`** - Script to pre-download datasets
2. **`DATASET_AND_EVAL_FIX.md`** (this file) - Documentation

---

## Important Notes

### About Preconditioning Reversal

- **Training loss**: Computed in preconditioned space ✓ (correct)
- **Validation loss**: Computed in preconditioned space ✓ (correct)
- **Evaluation metrics**: Should be in original space ✗ (needs reversal)

The reason is:
- During training, model learns patterns in preconditioned space
- For evaluation, we want metrics (MSE, MAE, etc.) in original space
- This requires reversing the preconditioning transformation

### Why Reversal Matters

Example:
```python
# Original: y = [1, 2, 3, 4, 5]
# Preconditioned: ỹ = [1, 2, 2.5, 3.2, 3.8]  # hypothetical

# Model predicts: ỹ_pred = [1, 2, 2.6, 3.1, 3.9]

# WITHOUT reversal:
MSE(ỹ_pred, ỹ) = small  # Looks good!

# WITH reversal:
y_pred = reverse(ỹ_pred) = [1, 2, 3.1, 3.9, 4.8]
MSE(y_pred, y) = slightly_larger  # True error in original space
```

The evaluation should use the true error in original space.

---

## Support

For questions:
1. Check this document: `DATASET_AND_EVAL_FIX.md`
2. Check dataset cache: `ls ~/.gluonts/datasets/`
3. Review bug fixes: `BUGFIX_SUMMARY.md`
4. Check evaluation guide: `EVALUATION_GUIDE.md`

---

**Status**:
- ✅ Dataset download: FIXED
- ❌ Precondition reversal: NOT YET IMPLEMENTED (use workaround)

**Last Updated**: 2025-11-01
