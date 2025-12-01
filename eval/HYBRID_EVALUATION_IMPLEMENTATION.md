# Hybrid Evaluation Implementation Summary

## Overview
This document summarizes the implementation of hybrid evaluation functionality in the comprehensive evaluation framework.

## Changes Made

### 1. Bug Check: `eval_precond_hybrid.py`
**Status**: ✅ No bugs found

Verified that the implementation is correct according to the CRITICAL_FIXES documentation:
- **Line 127**: Uses SUBTRACTION for reversal (`result[n:] = precond_seq[n:] - weighted_sum`) ✓
- **Lines 121-125**: Correctly computes weighted sum using base model predictions ✓
- **Forward/Reverse Logic**: Matches Algorithm 1 from the paper ✓
  - Forward: ỹ_t = y_t + Σ c_i · y[t-i] (ADDITION)
  - Reverse: y_t = ỹ_t - Σ c_i · y[t-i] (SUBTRACTION)

### 2. New Function: `eval_hybrid()` in `comprehensive_evaluation.py`
**Location**: Lines 724-881

Implemented a comprehensive hybrid evaluation function that:
1. Takes both base model and preconditioned model checkpoints
2. Iterates through datasets (with optional filtering)
3. Runs `cli.eval_precond_hybrid` for each dataset
4. Extracts and aggregates metrics
5. Saves results to CSV

**Key Features**:
- Follows the same pattern as other eval functions (eval_standard, eval_precond_space, eval_baseline_in_precond_space)
- Uses `moirai_precond_ckpt_no_reverse` config to ensure no automatic reversal in precond model
- Automatic patch size selection based on dataset frequency
- Progress tracking and error handling
- Comprehensive output logging

**Function Signature**:
```python
def eval_hybrid(
    base_model_path: str,
    precond_model_path: str,
    precond_type: str = 'chebyshev',
    precond_degree: int = 5,
    patch_size: int = 32,
    context_length: int = 1000,
    batch_size: int = 32,
    output_dir: Optional[str] = None,
    dataset_filter: Optional[List[str]] = None,
    datasets: Optional[List[Dict]] = None
) -> pd.DataFrame
```

### 3. CLI Updates
**Lines 960-973, 1084-1103**

Added support for hybrid mode in command-line interface:

**New Arguments**:
- `--mode hybrid`: Select hybrid evaluation mode
- `--base-model-path`: Path to base model checkpoint
- `--precond-model-path`: Path to preconditioned model checkpoint

**Mode Choices**: `['standard', 'precond', 'baseline-precond', 'hybrid', 'compare']`

### 4. Documentation Updates
**Lines 2-42**

Updated docstring with hybrid evaluation usage example:
```bash
python comprehensive_evaluation.py --mode hybrid \
    --base-model-path /path/to/baseline_checkpoint.ckpt \
    --precond-model-path /path/to/precond_checkpoint.ckpt \
    --precond-type chebyshev --precond-degree 5
```

### 5. Example Usage
**Lines 1014-1053**

Added example code in `new_main()` function showing how to use the hybrid evaluation programmatically:
```python
df = eval_hybrid(
    base_model_path=our_pretrained_model_path,
    precond_model_path=d5_precond_model_path,
    precond_type='chebyshev',
    precond_degree=5,
    output_dir="eval-runs/hybrid",
    dataset_filter=datasets
)
```

## Configuration Files Verified

The following configuration files exist and are correctly set up:
- `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/model/moirai_lightning_ckpt.yaml` ✓
- `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/model/moirai_precond_ckpt_no_reverse.yaml` ✓
- `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/default_hybrid.yaml` ✓

## Usage Examples

### Command Line Usage

**Test on single dataset**:
```bash
cd /scratch/gpfs/EHAZAN/jh1161
python eval/comprehensive_evaluation.py --mode hybrid \
    --base-model-path uni2ts/outputs/pretrain/.../baseline/checkpoints/last.ckpt \
    --precond-model-path uni2ts/outputs/pretrain/.../precond_d5/checkpoints/last.ckpt \
    --precond-type chebyshev \
    --precond-degree 5 \
    --datasets monash_m3_monthly
```

**Full evaluation on all datasets**:
```bash
python eval/comprehensive_evaluation.py --mode hybrid \
    --base-model-path uni2ts/outputs/pretrain/.../baseline/checkpoints/last.ckpt \
    --precond-model-path uni2ts/outputs/pretrain/.../precond_d5/checkpoints/last.ckpt \
    --precond-type chebyshev \
    --precond-degree 5 \
    --output-dir eval/outputs/hybrid_evaluation_results
```

### Programmatic Usage

```python
from eval.comprehensive_evaluation import setup_environment, eval_hybrid

setup_environment()

df = eval_hybrid(
    base_model_path="/path/to/baseline.ckpt",
    precond_model_path="/path/to/precond.ckpt",
    precond_type='chebyshev',
    precond_degree=5,
    output_dir="eval-runs/hybrid",
    dataset_filter=["monash_m3_monthly", "m1_quarterly"]  # Optional
)

print(df)
# Results saved to: eval-runs/hybrid/evaluation_metrics_hybrid.csv
```

## Output

The hybrid evaluation produces:
1. **CSV file**: `evaluation_metrics_hybrid.csv` with columns:
   - dataset
   - MSE[mean], MSE[0.5]
   - MAE[0.5], MASE[0.5], MAPE[0.5], sMAPE[0.5]
   - RMSE[mean], NRMSE[mean]
   - ND[0.5], MSIS
   - mean_weighted_sum_quantile_loss
   - status (success/failed/partial_success)

2. **Individual output files**: `{dataset_name}_output.txt` for each dataset

3. **Summary statistics**: Total datasets, successful, failed counts

## How Hybrid Evaluation Works

The hybrid approach combines two models:

1. **Base Model**: Pretrained without preconditioning (predicts in original space)
2. **Preconditioned Model**: Trained with preconditioning (predicts in preconditioned space, no reversal)

**Evaluation Flow**:
```
1. Generate base_predictions (y_base) from base model
2. Generate precond_predictions (ỹ_precond) from preconditioned model
3. Create hybrid_predictions using reversal with base context:
   y_hybrid[t] = ỹ_precond[t] - Σ(i=1 to n) c_i · y_base[t-i]
4. Compare hybrid_predictions to ground truth in original space
```

**Intuition**: The preconditioned model learns residuals/deltas, while the base model provides anchoring context.

## Testing

Syntax check passed:
```bash
python -m py_compile eval/comprehensive_evaluation.py
# No errors ✓
```

## Related Files

- `uni2ts/cli/eval_precond_hybrid.py` - Core hybrid evaluation logic
- `eval/comprehensive_evaluation.py` - Unified evaluation framework
- `uni2ts/src/uni2ts/transform/precondition.py` - Preconditioning implementation

## Next Steps

To use the hybrid evaluation:
1. Ensure you have both baseline and preconditioned model checkpoints
2. Choose the appropriate preconditioning parameters (type and degree)
3. Run the evaluation using either CLI or programmatic interface
4. Compare results with other evaluation modes (standard, precond_space, baseline_precond)

## Notes

- The hybrid evaluation is particularly useful for understanding how models can leverage complementary learning spaces
- Results are in the **original space** (not preconditioned), making them directly comparable to standard evaluation
- Make sure preconditioning parameters match the training configuration of the preconditioned model
