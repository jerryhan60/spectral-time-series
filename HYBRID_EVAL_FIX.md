# Hybrid Evaluation Hydra Configuration Fix

## Problem

When running hybrid evaluation, you encountered this Hydra error:

```
Could not override 'base_model.checkpoint_path'.
To append to your config use +base_model.checkpoint_path=...
Key 'checkpoint_path' is not in struct
    full_key: base_model.checkpoint_path
    object_type=dict
```

## Root Cause

The `default_hybrid.yaml` configuration file was using `moirai_1.1_R_small` for the base_model, which is designed for official HuggingFace models and doesn't support custom checkpoint paths. Additionally, the command construction was trying to override the model config group while also overriding fields within it, causing conflicts.

## Changes Made

### 1. Fixed `uni2ts/cli/conf/eval/default_hybrid.yaml`

**Before:**
```yaml
defaults:
  - model@base_model: moirai_1.1_R_small  # Wrong: no checkpoint support
  - model@precond_model: moirai_precond_ckpt  # Wrong: doesn't disable reversal
```

**After:**
```yaml
defaults:
  - model@base_model: moirai_lightning_ckpt  # ✓ Supports custom checkpoints
  - model@precond_model: moirai_precond_ckpt_no_reverse  # ✓ No reversal
```

### 2. Fixed `eval/comprehensive_evaluation.py`

**Removed conflicting model config group overrides:**

**Before:**
```python
cmd = [
    'base_model=moirai_lightning_ckpt',  # ❌ Redundant, conflicts with defaults
    f'base_model.checkpoint_path={base_model_path}',
    'precond_model=moirai_precond_ckpt_no_reverse',  # ❌ Redundant
    f'precond_model.checkpoint_path={precond_model_path}',
    ...
]
```

**After:**
```python
cmd = [
    # No model config group overrides - use defaults from default_hybrid.yaml
    f'base_model.checkpoint_path={base_model_path}',  # ✓ Only override fields
    f'base_model.patch_size={dataset_patch_size}',
    f'base_model.context_length={context_length}',
    f'precond_model.checkpoint_path={precond_model_path}',
    f'precond_model.patch_size={dataset_patch_size}',
    f'precond_model.context_length={context_length}',
    ...
]
```

## Why This Fixes the Issue

1. **Correct Base Config**: `moirai_lightning_ckpt` has `checkpoint_path` defined, so Hydra allows overriding it
2. **No Config Group Conflicts**: We let the defaults in `default_hybrid.yaml` set the model configs, then only override specific fields
3. **Proper Struct Mode**: The loaded configs support the fields we're trying to override

## Testing

Try running the hybrid evaluation again:

```bash
cd /scratch/gpfs/EHAZAN/jh1161
python eval/comprehensive_evaluation.py
```

Or test with a single dataset:

```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
python -m cli.eval_precond_hybrid \
    run_name=test_hybrid \
    base_model.checkpoint_path=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/pretrain_run_20251020_205126/checkpoints/last.ckpt \
    base_model.patch_size=32 \
    base_model.context_length=1000 \
    precond_model.checkpoint_path=/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/precond_chebyshev_d5_20251117_204835/checkpoints/last.ckpt \
    precond_model.patch_size=32 \
    precond_model.context_length=1000 \
    precond_model.precondition_type=chebyshev \
    precond_model.precondition_degree=5 \
    data=monash_cached \
    data.dataset_name=monash_m3_monthly \
    data.prediction_length=18
```

## Verification

You can verify the command structure by running:
```bash
python test_hybrid_command.py
```

This will show you the exact command that would be generated without actually running the evaluation.

## Summary

The fix ensures that:
- ✓ Base model uses `moirai_lightning_ckpt` config (supports checkpoints)
- ✓ Preconditioned model uses `moirai_precond_ckpt_no_reverse` (no automatic reversal)
- ✓ Command only overrides field values, not config groups
- ✓ No Hydra struct mode conflicts

The hybrid evaluation should now work correctly!
