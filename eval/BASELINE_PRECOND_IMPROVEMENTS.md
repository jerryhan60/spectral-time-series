# Improvements to eval_baseline_in_precond_space.py

## Summary

Updated `eval_baseline_in_precond_space.py` to use the built-in `evaluate_forecasts` function with proper GluonTS metrics, matching the approach used in `eval_precond_space.py`.

## Changes Made

### 1. Added Imports
```python
from uni2ts.eval_util.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
```

### 2. Replaced Manual Metrics Computation

**Before:** Manually computed only 4 metrics (MSE[mean], MSE[0.5], MAE[0.5], RMSE[mean]) with other metrics as NaN placeholders.

**After:** Uses `evaluate_forecasts` with full metric suite:
- MSE (mean)
- MAE (median)
- MAPE (median)
- sMAPE (median)
- MASE (median)
- RMSE (mean)
- NRMSE (mean)
- ND (median)
- MSIS
- MeanWeightedSumQuantileLoss

### 3. Improved Preconditioning Logic

**Key improvement:** Now preconditions full sequences (context + prediction) for both ground truth and forecasts, then extracts only the prediction part. This ensures preconditioning is applied consistently across the full time series.

**For ground truth:**
1. Concatenate context + label
2. Apply preconditioning to full sequence
3. Split back into preconditioned context and preconditioned label

**For forecasts:**
1. For each sample in the forecast
2. Concatenate context + sample
3. Apply preconditioning to full sequence
4. Extract preconditioned prediction part
5. Create new `SampleForecast` with preconditioned samples

### 4. Created Preconditioned Test Data Structure

Uses the same `PreconditionedTestData` class pattern as `eval_precond_space.py` to properly wrap preconditioned inputs and labels for evaluation.

### 5. Standardized Output Format

**Before:** Custom print format with limited metrics
**After:** Standard table format matching `eval_precond_space.py`:

```
================================================================================
EVALUATION METRICS (BASELINE IN PRECONDITIONED SPACE)
================================================================================
          MSE[mean]  MAE[0.5]  MAPE[0.5]  sMAPE[0.5]  MASE[0.5]  ...
None       <value>   <value>    <value>     <value>     <value>   ...
================================================================================
```

## Benefits

1. **Consistency:** Now uses the same evaluation pipeline as `eval_precond_space.py`
2. **Completeness:** Provides all standard metrics (10 metrics) instead of just 4
3. **Accuracy:** Properly handles multivariate data and masked arrays for NaN values
4. **Comparability:** Output format matches other evaluation scripts, making comparison easier
5. **Maintainability:** Uses shared evaluation utilities instead of custom metric computation

## Impact on comprehensive_evaluation.py

Updated `extract_baseline_precond_metrics()` to use the same extraction logic as `extract_metrics_from_output()` since both now output the same 10-metric format.

## Testing

To test the improvements:

```bash
# Test baseline in precond space evaluation
python -m cli.eval_baseline_in_precond_space \
    model=moirai_lightning_ckpt \
    model.checkpoint_path=/path/to/baseline.ckpt \
    model.patch_size=32 \
    model.context_length=1000 \
    +precond_type=chebyshev \
    +precond_degree=5 \
    data=monash_cached \
    data.dataset_name=m1_monthly \
    data.prediction_length=18
```

Or use comprehensive_evaluation.py:

```python
python comprehensive_evaluation.py --mode baseline-precond \
    --model-path /path/to/baseline.ckpt \
    --precond-type chebyshev \
    --precond-degree 5 \
    --datasets m1_monthly
```

## Output Format

The script now outputs metrics in the same format as `eval_precond_space.py`:

```
Preconditioning Configuration:
  Type: chebyshev
  Degree: 5

Loading test data: m1_monthly
  Prediction length: 18
  ...

Generating forecasts from baseline model...
Generated 617 forecasts

Applying preconditioning to both predictions and ground truth...
Preconditioning: 100%|██████████| 617/617 [00:XX<00:00, XXX.XXit/s]
Preconditioned 617 forecasts and ground truth

Computing metrics in preconditioned space...
================================================================================

================================================================================
EVALUATION METRICS (BASELINE IN PRECONDITIONED SPACE)
================================================================================
          MSE[mean]  MAE[0.5]  MAPE[0.5]  sMAPE[0.5]  MASE[0.5]  RMSE[mean]  NRMSE[mean]  ND[0.5]  MSIS  mean_weighted_sum_quantile_loss
None  1234567.890  2345.678      3.456       1.234      0.765   3456.789       12.345    0.876 7.890                            0.654
================================================================================

Results saved to: /path/to/metrics_baseline_in_precond_space.csv
```

## Files Modified

1. `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/eval_baseline_in_precond_space.py` - Main evaluation logic
2. `/scratch/gpfs/EHAZAN/jh1161/eval/comprehensive_evaluation.py` - Metrics extraction function
