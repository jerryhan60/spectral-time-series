# Inference-Time Preconditioning Ensemble

## Problem

Our best preconditioning results (hint mode, MASE 1.1802) require retraining the model with hint channels. Can we get preconditioning benefits at inference time only, using a frozen pretrained model?

## Approach

Run the frozen official Moirai2-small model multiple times on different views of the same input series — raw and preconditioned at various polynomial degrees — then combine the forecasts.

## Base Model

Official Moirai2-small from HuggingFace (frozen, no modification).

## Data Flow

```
Input: raw time series x (length T), prediction_length H

1. Z-score: z = (x - loc) / scale   (context window stats)

2. Raw pass: feed z as target → Q_raw(τ)

3. For each degree d in {2, 3, 4, 5, 6, 7, 8}:
   Precondition: p_d(z)[t] = z[t] + Σ_{i=1}^{d} c_i * z[t - i]
   (Chebyshev coefficients, stride=1)

   Strategy A1 (Replace + Reverse with raw anchor):
     Feed p_d(z) as target → Q_d(τ) in precond space
     Reverse: Q_d_raw(τ)[t] = Q_d(τ)[t] - Σ_{i=1}^{d} c_i * Q_raw(τ)[t - i]
     For first d prediction steps where t-i falls in context: use observed z values

   Strategy A2 (Residual):
     Compute r_d = p_d(z) - z
     Feed r_d as target → Q_r(τ)
     Final: Q_d_raw(τ) = Q_raw(τ) + Q_r(τ)

4. Combine forecasts using zero-parameter methods
```

### Reversal Design Choice

The FIR filter p(z) is invertible via recursive IIR, but autoregressive reversal on model outputs accumulates errors. Instead, we reverse using Q_raw as the anchor:

```
Q_d_raw(τ)[t] = Q_d(τ)[t] - Σ_{i=1}^{d} c_i * Q_raw(τ)[t - i]
```

This is non-recursive (single-pass FIR on output) and blends information from both passes.

## Combination Methods

| Method | Formula | Intuition |
|--------|---------|-----------|
| Uniform average | Q_combined = (1/K) Σ Q_k | Equal trust in all passes |
| Inverse-variance | w_k = 1/(IQR_k), normalize, weighted avg | Trust tighter predictions more |
| Quantile pool | Average full quantile distributions | Preserves distributional info |
| Oracle | min_k MASE(Q_k) per dataset | Upper bound (not a real method) |

## Experimental Plan

### Phase 1: Individual Pass Quality

Evaluate whether the model produces sensible output on preconditioned input.

- Raw pass (baseline)
- A1 (replace + reverse) x d={2,3,4,5,6,7,8}, s=1
- A2 (residual) x d={2,3,4,5,6,7,8}, s=1
- Total: 15 individual runs on full GIFT-Eval (97 configs)

Key question: do any individual preconditioned passes beat raw?

### Phase 2: Combination

- For each of {A1, A2}:
  - Best single degree vs raw
  - Raw + best degree (2-model ensemble)
  - Raw + all degrees (8-model ensemble)
- All 3 real combination methods x above subsets

Key question: does diversity help even when individuals are weaker?

### Phase 3: Per-Dataset Analysis

- Which dataset categories benefit? (high-freq, long-horizon, etc.)
- Does the pattern match training-time hint results?

Key question: is the benefit systematic or noise?

## Implementation Structure

Single new script: `gifteval/eval_inference_precond.py`

Wraps existing `Moirai2Forecast` as a black box. No modification to model code.

```
Functions:
  compute_chebyshev_coeffs(degree)        # reuse from precondition.py
  precondition_series(z, coeffs)          # FIR: p(z)[t] = z[t] + Σ c_i * z[t-i], stride=1
  reverse_with_raw_anchor(Q_precond, Q_raw, z_context, coeffs, context_len)
  run_single_pass(model, data, precond_fn=None)
  combine_forecasts(forecasts_dict, method)
  evaluate_gift_eval(...)                 # main loop
```

Output: same CSV format as existing eval_gifteval.py for direct comparison.

## Compute Budget

- Each full GIFT-Eval pass: ~20-25 min on H200
- Phase 1: 15 passes = ~6 hours (parallelizable across GPUs)
- Phase 2: post-processing only (seconds)

## Success Criteria

- Any individual preconditioned pass within 5% of raw: model handles preconditioned input sensibly
- Any combination beats raw by >1%: inference-time preconditioning has signal
- Consistent benefit on high-freq/long-horizon: matches training-time findings

## Preconditioning Configuration

- Polynomial type: Chebyshev (monic)
- Degrees: 2, 3, 4, 5, 6, 7, 8
- Stride: 1 (direct preconditioning on the time series)
- Coefficients: fixed, computed analytically (not learned)

## Results (Quick Evaluation — 8 datasets)

**Note**: Used Moirai v1 (moirai-1.1-R-small) since this is the official HuggingFace model. Moirai2 checkpoints require loading from our custom training runs.

### Strategy A1: Replace + Reverse with Raw Anchor

| Method | Geo Mean MASE | vs Raw | Win Rate |
|--------|--------------|--------|----------|
| raw | 1.8982 | — | — |
| A1_d2 (best individual) | 1.9494 | +2.70% | 2/8 |
| A1_d3 | 2.0403 | +7.49% | 1/8 |
| A1_d4 | 2.1123 | +11.28% | 1/8 |
| A1_d5 | 2.1841 | +15.06% | 2/8 |
| A1_d6 | 2.2605 | +19.09% | 0/8 |
| A1_d7 | 2.3376 | +23.15% | 0/8 |
| A1_d8 | 2.4366 | +28.37% | 0/8 |
| A1_invvar (best combo) | 1.9741 | +4.00% | 2/8 |
| A1_uniform | 2.0721 | +9.16% | 2/8 |
| A1_qpool | 2.0806 | +9.61% | 2/8 |

### Strategy A2: Residual Forecasting

| Method | Geo Mean MASE | vs Raw |
|--------|--------------|--------|
| raw | 1.8970 | — |
| A2_d2 | 9.9867 | +426% |
| A2_d4 | 16.4785 | +768% |
| A2_d6 | 18.0596 | +852% |
| A2_d8 | 18.3075 | +865% |

### Key Findings

1. **A1 (replace + reverse)**: All individual preconditioned passes are worse than raw. Best is d=2 (+2.7%). Degradation increases monotonically with degree. Combination methods cannot recover — best is invvar at +4.0%.

2. **A2 (residual)**: Catastrophically bad (5-18x worse). The FIR residual signal is not a valid time series — the model produces garbage when asked to forecast it.

3. **No frequency/horizon patterns**: Unlike training-time hint mode (which helps high-freq data), inference-time preconditioning hurts across all frequency categories.

4. **Conclusion**: Inference-time preconditioning does NOT work with a frozen model that was trained on raw data. The model's internal representations are calibrated for raw time series distributions. Preconditioning the input shifts the distribution in ways the model cannot handle, and the raw-anchor reversal cannot recover the lost information.

5. **Implication for training-time preconditioning**: This negative result strengthens the case for training-time hint mode. The model needs to be trained with preconditioning information to benefit from it — simply providing preconditioned data to a frozen model is not sufficient. The training process is necessary for the model to learn how to extract and use the spectral information provided by the polynomial filter.

### Approach C: Learned Adapter

Given the negative zero-parameter results, we investigated whether a learned adapter could find per-series signal.

#### Per-Series Oracle Analysis

Critically, while **dataset-level** averages show preconditioning always hurts, the **per-series** oracle reveals significant hidden signal:

| Metric | Value |
|--------|-------|
| Total series evaluated | 57,406 |
| Raw is best for | 24,786 (43.2%) |
| Some precond is better for | 32,620 (56.8%) |
| Oracle improvement vs always-raw | **6.77%** |
| Mean improvement when precond wins | 13.12% |
| Median improvement when precond wins | 10.08% |
| Max improvement | 82.53% |

Per-dataset oracle breakdown:

| Dataset | Raw MASE | Oracle MASE | Improvement | Raw Wins |
|---------|----------|-------------|-------------|----------|
| m4_monthly | 1.0589 | 0.9755 | +7.88% | 18513/48000 |
| electricity/H | 1.3430 | 1.3219 | +1.57% | 5370/7400 |
| hospital | 0.8099 | 0.7641 | +5.65% | 392/767 |
| jena_weather/H | 0.8722 | 0.8342 | +4.36% | 149/399 |
| ett1/H | 0.9261 | 0.9007 | +2.74% | 83/140 |
| saugeenday/D | 2.9801 | 2.8587 | +4.07% | 4/20 |
| covid_deaths | 31.8718 | 29.7636 | +6.61% | 73/266 |
| m4_hourly | 1.8997 | 1.7888 | +5.84% | 202/414 |

**Key insight**: The majority of individual series (56.8%) actually benefit from preconditioning. The negative dataset-level results are because the average-harming series dominate the mean. A perfect per-series selector would yield 6.77% improvement.

#### Learned Adapter Results (Leave-One-Dataset-Out Cross-Validation)

Two adapter architectures trained to combine raw + A1_d2/d4/d6 forecasts:

**Linear adapter** (4 softmax weights, ~3 params):

| Fold (val) | Adapter MASE | Raw MASE | Improvement |
|------------|-------------|----------|-------------|
| m4_monthly | 1.0577 | 1.0589 | +0.11% |
| electricity | 1.3594 | 1.3430 | -1.22% |
| hospital | 0.8155 | 0.8099 | -0.70% |
| jena_weather | 0.8639 | 0.8722 | +0.96% |
| ett1 | 0.9290 | 0.9261 | -0.32% |
| saugeenday | 2.9618 | 2.9801 | +0.61% |
| covid_deaths | 31.2114 | 31.8718 | +2.07% |
| m4_hourly | 1.8651 | 1.8997 | +1.82% |
| **AGGREGATE** | **1.2379** | **1.2401** | **+0.17%** |

Learned weights: ~70% raw, ~18% A1_d2, ~5% A1_d4, ~5% A1_d6 (global); datasets with fewer series (saugeenday, covid) learn more balanced weights.

**Feature-conditioned MLP adapter** (8 input features → 16 hidden → 4 weights, ~180 params):

| Fold (val) | Adapter MASE | Raw MASE | Improvement |
|------------|-------------|----------|-------------|
| m4_monthly | 1.0601 | 1.0589 | -0.11% |
| electricity | 1.3425 | 1.3430 | +0.04% |
| hospital | 0.8078 | 0.8099 | +0.25% |
| jena_weather | 0.8600 | 0.8722 | +1.40% |
| ett1 | 0.9240 | 0.9261 | +0.22% |
| saugeenday | 2.9594 | 2.9801 | +0.70% |
| covid_deaths | 31.2638 | 31.8718 | +1.91% |
| m4_hourly | 1.8730 | 1.8997 | +1.40% |
| **AGGREGATE** | **1.2379** | **1.2401** | **+0.18%** |

MLP wins 7/8 folds (vs linear 5/8), showing per-series feature conditioning helps.

#### Overall Conclusions

1. **Oracle shows 6.77% signal** — a large gap between the 0.17-0.18% the adapters achieve and the theoretical maximum. The per-series selection problem is hard.

2. **Both adapters achieve ~0.17-0.18% aggregate improvement**. While positive, this is marginal and not practically significant (within noise).

3. **MLP is more robust**: Wins 7/8 folds vs linear's 5/8, suggesting per-series features have predictive value for choosing forecast sources.

4. **The adapter essentially learns "mostly trust raw"**: Linear adapter gives 70% weight to raw. The signal from preconditioned passes is too weak and inconsistent to exploit reliably.

5. **Gap between oracle (6.77%) and adapter (0.18%) = 6.6% unrealized potential**. This suggests the problem is not lack of signal but lack of a good selector. Better features, more training data, or learning the selector jointly with the model could help.

6. **Final verdict on inference-time preconditioning**: The frozen-model approach has fundamental limitations. While per-series signal exists, it cannot be reliably extracted without either (a) training the model with preconditioning information (hint mode), or (b) a much more sophisticated adapter trained on vastly more data.

### Full Evaluation (97 configs)

Full runs submitted as SLURM jobs 5025175 (A1) and 5025176 (A2) on pli partition for zero-parameter methods.
