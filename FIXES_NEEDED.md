# Quick Fix Guide

**Both issues identified. One fixed, one needs implementation.**

---

## ✅ Issue 1: Dataset Download - FIXED

**Error**: `URLError(gaierror(-2, 'Name or service not known'))`

**Fix**: Pre-download datasets on login node

```bash
# On della-login (has internet)
cd /scratch/gpfs/EHAZAN/jh1161
source uni2ts/venv/bin/activate
python download_eval_datasets.py
```

**What it does**: Downloads all 12 evaluation datasets to `~/.gluonts/datasets/`

**Then**: Compute nodes will use cached datasets automatically

---

## ❌ Issue 2: Preconditioning Reversal - NOT IMPLEMENTED

**Problem**: Models trained with preconditioning produce predictions in preconditioned space. Evaluation metrics should be computed in original space (requires reversal).

**Status**: Not yet implemented in evaluation pipeline

**Workarounds**:

### Option A: Train Without Preconditioning for Evaluation
```bash
# Train baseline for evaluation
sbatch pretrain_moirai.slurm
```

### Option B: Wait for Implementation
- Continue training with preconditioning
- Wait for reversal to be implemented before evaluating

### Option C: Implement It Yourself
See `DATASET_AND_EVAL_FIX.md` for implementation details

---

## Recommended Action Plan

### Today (Immediate)

1. **Fix dataset downloads**:
   ```bash
   # On login node
   python download_eval_datasets.py
   ```

2. **Resume training**:
   ```bash
   sbatch pretrain_moirai_precond_default.slurm
   ```

### This Week (Short-term)

3. **Choose evaluation strategy**:
   - Wait for reversal implementation, OR
   - Train separate baseline model for evaluation

4. **Implement reversal** (if needed urgently)

---

## Files to Read

- `DATASET_AND_EVAL_FIX.md` - Detailed explanation of both issues
- `download_eval_datasets.py` - Script to pre-download datasets
- `BUGFIX_SUMMARY.md` - Previous bug fixes

---

## Summary

| Issue | Fixed? | Action Required |
|-------|--------|----------------|
| Dataset download | ✅ YES | Run download script on login node |
| Precond reversal | ❌ NO | Use workaround or implement |

**Critical**: You can fix dataset downloads immediately. Preconditioning reversal needs more work.
