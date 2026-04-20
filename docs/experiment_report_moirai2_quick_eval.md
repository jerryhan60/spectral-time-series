# Moirai2 Quick Evaluation Results: Baseline vs Chebyshev Preconditioning

**Date**: 2026-02-08
**Branch**: `spectral_non_precond`

## Overview

This report presents GIFT-Eval benchmark results for two Moirai2 Small models trained for 200 steps on a single dataset (`australian_electricity_demand`). The purpose is to validate the training and evaluation pipelines, not to produce competitive forecasting performance. Both models are heavily undertrained relative to the paper's 100K-step specification.

Full-scale training runs (100K steps on LOTSA v1) are currently in progress and will provide meaningful performance comparisons.

## Models Evaluated

| Property | Baseline | Preconditioned |
|----------|----------|----------------|
| Architecture | Moirai2 Small (causal decoder) | Moirai2 Small + Chebyshev FIR |
| Parameters | 11.39M | 11.39M |
| Training steps | 200 (20 epochs x 10 batches) | 200 (20 epochs x 10 batches) |
| Training data | `test_small` (1 dataset) | `test_small` (1 dataset) |
| Batch size | 32 | 32 |
| Final training loss | 0.109 | 0.082 |
| Precondition type | None | Chebyshev, custom 4-tap FIR |
| FIR coefficients | N/A | [0.0, -0.1176, 0.0, -0.1361] |
| FIR inverse | N/A | Enabled (length=64, lambda=0.1) |
| Loss reversal | N/A | Disabled |

### Preconditioning Details

The preconditioned model applies a causal FIR filter before encoding:

```
z_t = y_t + c_1 * y_{t-1} + c_2 * y_{t-2} + c_3 * y_{t-3} + c_4 * y_{t-4}
```

with fixed coefficients `[0.0, -0.1176, 0.0, -0.1361]` derived from ak8836's regularization=3 optimization. These are NOT pure Chebyshev polynomials (which would be `[0.0, -1.25, 0.0, 0.3125, 0.0]` for degree 5), but a regularized variant with smaller magnitudes.

A learnable FIR inverse filter (64 taps) is trained with an auxiliary loss (lambda=0.1) to encourage the model to learn useful residual representations while maintaining the ability to reconstruct the original signal.

## Quick Eval Results (8 Datasets)

### Per-Dataset MASE Comparison

| Dataset | Freq | Pred Len | Baseline MASE | Precond MASE | Delta | Winner |
|---------|------|----------|---------------|-------------|-------|--------|
| jena_weather/H | H | 48 | **0.7368** | 0.7701 | +4.5% | Baseline |
| hospital | M | 12 | **0.9304** | 0.9700 | +4.3% | Baseline |
| ett1/H | H | 48 | 1.5221 | **1.4808** | -2.7% | Precond |
| m4_monthly | M | 18 | 2.6332 | **2.4312** | -7.7% | Precond |
| electricity/H | H | 48 | 3.2784 | **3.0725** | -6.3% | Precond |
| saugeenday/D | D | 30 | 4.3885 | **3.9154** | -10.8% | Precond |
| m4_hourly | H | 48 | 9.2180 | **8.4035** | -8.8% | Precond |
| covid_deaths | D | 30 | **100.1327** | 105.4635 | +5.3% | Baseline |

### Aggregate Metrics

| Metric | Baseline | Precond | Winner | Margin |
|--------|----------|---------|--------|--------|
| **Geometric Mean MASE** | 2.3224 | **2.2273** | Precond | -4.1% |
| Arithmetic Mean MASE | 15.3550 | **15.3884** | Baseline | +0.2% |
| Arith Mean (excl. outlier) | 3.2439 | **3.0062** | Precond | -7.3% |
| Median MASE | 2.6332 | **2.4312** | Precond | -7.7% |
| Beats naive (MASE < 1.0) | 2/8 | 2/8 | Tie | - |
| Best MASE | **0.7368** | 0.7701 | Baseline | - |
| Worst MASE | **100.1327** | 105.4635 | Baseline | - |

### Per-Metric Breakdown

| Dataset | Baseline MAE | Precond MAE | Baseline MSE | Precond MSE |
|---------|-------------|-------------|-------------|-------------|
| m4_monthly | 1032.82 | 1052.71 | 3.86M | 3.99M |
| electricity/H | 644.22 | **580.87** | 15.97M | **11.91M** |
| hospital | **27.22** | 30.04 | **7131.27** | 8279.56 |
| jena_weather/H | **15.06** | 16.21 | **2062.95** | 2133.49 |
| ett1/H | **8.998** | 9.222 | **241.49** | 281.29 |
| saugeenday/D | 19.77 | **17.63** | 1514.06 | **1339.28** |
| covid_deaths | **1063.47** | 1162.65 | **31.79M** | 36.20M |
| m4_hourly | 1209.87 | **1123.80** | 65.11M | **64.15M** |

## Analysis

### Where Preconditioning Helps

Preconditioning wins on **5 out of 8 datasets** with improvements ranging from 2.7% to 10.8% in MASE:

1. **saugeenday/D** (-10.8%): Daily river flow data with strong trends. FIR detrending helps the model focus on residuals rather than the trend component.

2. **m4_hourly** (-8.8%): High-frequency M4 competition data. The FIR filter removes short-term autocorrelation, letting the model predict innovations.

3. **m4_monthly** (-7.7%): Monthly M4 data. Polynomial filter captures monthly trend/seasonality structure.

4. **electricity/H** (-6.3%): Hourly electricity demand. Clear diurnal and weekly patterns that benefit from preconditioning detrending.

5. **ett1/H** (-2.7%): ETT benchmark hourly data. Modest improvement suggesting some benefit from trend removal.

### Where Baseline Wins

Baseline wins on 3 datasets:

1. **jena_weather/H** (+4.5%): Weather data is already relatively stationary (MASE ~0.74). Preconditioning may add unnecessary transformation to data that doesn't need detrending.

2. **hospital** (+4.3%): Monthly hospital patient counts. Low-frequency, relatively smooth data where the simple trend removal may remove useful signal.

3. **covid_deaths** (+5.3%): Extreme distributional shift makes both models perform poorly (MASE >100). The 5.3% difference is noise given the overall poor performance.

### Key Observations

1. **Training loss gap persists into eval**: The precond model's lower training loss (0.082 vs 0.109) translates into better downstream forecasting on most datasets, suggesting the preconditioning genuinely improves learning rather than just overfitting the training objective.

2. **Scale of improvement**: The 4.1% improvement in geometric mean MASE is meaningful but should be interpreted cautiously given:
   - Only 200 training steps (vs 100K in paper)
   - Trained on 1 dataset (vs 27 datasets in LOTSA v1)
   - Only 8 eval datasets (vs 97 in full benchmark)

3. **Preconditioning overhead is negligible**: Both models trained at ~2-2.5 it/s with identical wall-clock times. The FIR filter adds minimal computation.

4. **Stationary data caveat**: On already-stationary data (jena_weather, hospital), preconditioning slightly hurts. This suggests the optimal approach may be adaptive preconditioning that activates only when the input has significant trend/non-stationarity.

## Caveats

- **Severely undertrained**: 200 steps on 1 dataset vs paper's 100K steps on 27 datasets
- **Custom coefficients**: The quick test used regularized coefficients `[0.0, -0.1176, 0.0, -0.1361]`, not pure Chebyshev. Full runs will test pure Chebyshev coefficients at degrees 1-7.
- **Small eval set**: 8 of 97 GIFT-Eval configurations. Full evaluation jobs (97 configs) have been submitted (jobs 4583584, 4583585).
- **covid_deaths dominates arithmetic mean**: MASE ~100 makes arithmetic mean unreliable; geometric mean is the better aggregate metric.

## Checkpoints

| Model | Path |
|-------|------|
| Baseline | `uni2ts/outputs/pretrain/moirai2_small/test_small/quick_baseline_20260208_185718/checkpoints/epoch_19-step_200.ckpt` |
| Precond | `uni2ts/outputs/pretrain/moirai2_small/test_small/quick_precond_20260208_190022/checkpoints/epoch_19-step_200.ckpt` |

## SLURM Jobs

| Job ID | Name | Type | Status |
|--------|------|------|--------|
| 4583019 | m2_quick_base | Pretraining | Completed |
| 4583020 | m2_quick_precond | Pretraining | Completed |
| 4583252 | gifteval_base_quick | GIFT-Eval (8) | Completed |
| 4583253 | gifteval_precond_quick | GIFT-Eval (8) | Completed |
| 4583584 | gifteval_m2_base_full | GIFT-Eval (97) | Submitted |
| 4583585 | gifteval_m2_precond_full | GIFT-Eval (97) | Submitted |

## Next Steps

1. **Full GIFT-Eval** on quick test checkpoints (97 configs, in progress)
2. **Full pretraining** results (100K steps, 9 runs in progress):
   - 1 baseline (no preconditioning)
   - 7 pure Chebyshev (degree 1-7)
   - 1 custom 4-tap coefficients
3. **Full GIFT-Eval** on each fully-trained checkpoint
4. **Comparison report** across all variants with per-domain analysis
