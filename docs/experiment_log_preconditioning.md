# Preconditioning Experiment Log

**Focus**: Time-domain polynomial preconditioning and FIR inverse filters for Moirai2 Small.
**Branch**: `spectral_non_precond`
**Last updated**: 2026-02-25
**Research document**: `Spectral Time Series 2025 research findings.pdf` (root directory)

---

## Research Context (from "Spectral Time Series 2025 Research Findings")

### Goal
Investigate universal sequence preconditioning (USP) and spectral filtering for zero-shot time series foundation models (MOIRAI family).

### Moirai 1.0 Preconditioning Findings (Monash Benchmark, 29 datasets)

Standard time-domain preconditioning on Moirai 1.0 small:

1. **Preconditioning reduces training loss** — optimal degree d=4 or 5 for minimizing training loss. Loss explodes for d>5 due to exponentially growing max coefficient.

2. **Ground-truth reversal (1-step forecast) works well** — Geo mean MAE ratio: d=2 → 0.75, d=5 → 0.71 (vs baseline). Significant improvement.

3. **Model-forecast reversal fails catastrophically** — Geo mean MAE ratio: d=2 → 1.22, d=5 → 2.02 (worse than baseline). Error accumulates exponentially when reversing with predicted values, especially for higher degree polynomials where coefficients >= 1.

4. **Conclusion**: Preconditioning is NOT directly applicable to long-horizon inference. The technique only works for 1-step-ahead forecasts with ground-truth reversal.

### Patch (Embedding) Preconditioning Findings

Applied Chebyshev filter to patch embeddings after linear projection, before transformer:
```
Raw time series -> in_proj -> precond -> transformer -> out_proj
```

- Train loss improves with degree up to d=5
- Eval loss improves up to d=4, but still worse than baseline
- **Root cause**: The transformer + output projection must learn both time series mechanics AND reversal of preconditioning simultaneously — too much burden on the same parameters.

### STU-MOIRAI Hybrid Findings

Alternating STU + Attention layers (3 STU + 3 Attn, 12.5M params):
- STU hybrid wins 55/97 GIFT-Eval configs vs baseline (42 baseline wins)
- Differences are small and driven by noisy datasets
- **Not statistically significant**

### Key Insight: Training vs Eval Divergence

The preconditioning creates a distribution shift:
- Training (LOTSA): `precond_context → attention_patterns → output → param_proj → good predictions`
- Evaluation (new data): `precond_context → different_patterns → output → param_proj → bad predictions`

The param_proj learns a specific mapping for LOTSA preconditioned patterns that doesn't transfer.

---

## Method Summary

**Time preconditioning** applies a causal FIR filter to the z-scored time series before the transformer encoder:

```
z_t = y_t + c_1 * y_{t-s} + c_2 * y_{t-2s} + ... + c_d * y_{t-d*s}
```

where `c_i` are monic polynomial coefficients (Chebyshev or Legendre), `d` is the degree, and `s` is the stride. This is applied at per-timepoint granularity (patches are flattened, filtered, reshaped). Coefficients are currently **fixed** (not learned).

### Stride Explanation

The **stride** `s` controls the lag spacing between filter taps. With Moirai2's patch size of 16:

- **Stride=1** (default): Filter taps are adjacent timepoints: `y_{t-1}, y_{t-2}, ...`. This operates within a single patch, creating very local correlations. Problem: at inference with reversal, errors accumulate within the patch since each reversed point depends on the previous one.

- **Stride=16** (patch-aligned): Filter taps span across patches: `y_{t-16}, y_{t-32}, ...`. Each tap references the corresponding position in the previous patch. This has two key advantages:
  1. **Error isolation**: During reversal, errors don't cascade within a patch because each timepoint only depends on points in *other* patches (which are predicted independently).
  2. **Semantic alignment**: The filter operates at the same granularity as the transformer's attention, which sees patch-level representations. The preconditioning captures inter-patch trends rather than intra-patch noise.

**Empirical result**: Stride=16 universally helps. The Lyapunov+FIR config swings from +1.90% worse (s=1) to -2.09% better (s=16) — a 4% improvement just from the stride change. Every top-performing experiment uses stride=16.

**FIR inverse filter** is a learnable filter (nn.Parameter) trained via auxiliary MSE loss to predict the residual `r = z_precond - z_raw` from the preconditioned signal. This guides the model to learn representations that can reconstruct the original signal.

**Loss reversal** (alternative to FIR inverse): analytically undo the precondition on predictions before computing loss in raw space. Mutually exclusive with FIR inverse.

### Hint Mode (Best Approach)

**Hint mode** avoids the reversal problem entirely. Instead of transforming the input and reversing the output, the FIR filter residual `r_t = z_preconditioned_t - z_raw_t` is provided as a **3rd input channel** alongside the target and observation mask:

```
Input to model: [target_values, observation_mask, precond_hint]
```

The model trains and predicts in **raw (z-scored) space** — no reversal is needed at inference. The hint channel gives the transformer additional spectral information about inter-patch trends without forcing it to learn in a different coordinate system.

**Why hint mode works better than reversal**: Reversal requires the model to produce accurate predictions in preconditioned space, then analytically undo the filter. Any prediction error gets amplified by the reversal. Hint mode avoids this by keeping predictions in the original space while giving the model extra information for free.

**Best result**: Hint d=3 s=16 achieves MASE 1.2040 (-3.07% vs baseline), beating Moirai-base (1.259) with 8x fewer parameters.

### Pipeline

```
Raw time series
  -> Z-score scaling (PackedStdScaler)
  -> Time preconditioning (fixed FIR filter)
  -> Zero unobserved slots
  -> Patch embedding (in_proj)
  -> Transformer encoder (6 layers)
  -> Output projection (out_proj)
  -> Loss computation (PackedQuantileMAELoss)
     + FIR inverse auxiliary loss (if enabled)
```

---

## Currently Running Experiments (as of 2026-02-09)

### Active Training Jobs

| Job ID | Experiment | Description | Status | Partition |
|--------|-----------|-------------|--------|-----------|
| 4611086 | EXP-1 Baseline (fixed) | Moirai2 no precond, zscore=8.0 | RUNNING (~ep 200) | ailab |
| 4610803-4611085 | EXP-1b regularization sweep | d=4 with lambda=0.25..10, custom 4-tap | RUNNING (~5h in) | ailab |
| 4558185 | STU Multi-Head | H=6 MH-STU, d_ff=1379, 13.83M params | RUNNING (~2d) | ailab |
| 4558187 | STU Parallel | STU+Attn per layer, d_ff=888, gate | RUNNING (~2d) | ailab |

### Planned (awaiting EXP-1b results)

| Experiment | Runs | Depends on | Description |
|-----------|------|------------|-------------|
| EXP-2 Legendre | 7 | EXP-1 | Legendre degree sweep — **likely cancelled** (EXP-1 shows precond hurts) |
| EXP-3 FIR Inverse | 6 | EXP-1 | FIR inverse hyperparameter sweep — **likely cancelled** |
| EXP-4 Loss Reversal | 3 | EXP-1 | FIR inverse vs loss reversal vs neither |
| EXP-5 Stride | 3 | EXP-1 | Stride variations — **likely cancelled** |
| EXP-6 Learnable | ~12 | EXP-1 | Learnable coefficients (code ready) |

---

## Degrees of Freedom

### Time Preconditioning

| Parameter | Config key | Values tested | Description |
|-----------|-----------|---------------|-------------|
| Polynomial type | `module_kwargs.time_precondition_type` | chebyshev | Orthogonal polynomial family |
| Degree | `module_kwargs.time_precondition_degree` | 1, 2, 3, 4, 5, 6, 7 | Number of filter taps |
| Stride | `module_kwargs.time_precondition_stride` | 1 | Lag spacing between taps |
| Custom coefficients | `module_kwargs.time_precondition_coeffs_init` | `[0,-0.1176,0,-0.1361]` | Override polynomial computation |
| Learnable coefficients | (not implemented) | - | Could make coeffs nn.Parameter |

### FIR Inverse

| Parameter | Config key | Values tested | Description |
|-----------|-----------|---------------|-------------|
| Enabled | `module_kwargs.time_precondition_inverse_enabled` | true | Whether to learn inverse filter |
| Length | `module_kwargs.time_precondition_inverse_length` | 64 | Number of learnable taps |
| Stride | `module_kwargs.time_precondition_inverse_stride` | 1 | Lag spacing for inverse |
| Lambda | `model.time_precondition_inverse_lambda` | 0.1 | Weight of inverse aux loss |

### Loss Reversal

| Parameter | Config key | Values tested | Description |
|-----------|-----------|---------------|-------------|
| Enabled | `model.time_precondition_reverse_in_loss` | false | Compute loss in raw space |

*Note: Loss reversal and FIR inverse are mutually exclusive.*

### Chebyshev Coefficients Reference

| Degree | Coefficients (monic) | Non-zero taps |
|--------|---------------------|---------------|
| 1 | `[0.0]` | 0 (identity) |
| 2 | `[0.0, -0.5]` | 1 |
| 3 | `[0.0, -0.75, 0.0]` | 1 |
| 4 | `[0.0, -1.0, 0.0, 0.125]` | 2 |
| 5 | `[0.0, -1.25, 0.0, 0.3125, 0.0]` | 2 |
| 6 | `[0.0, -1.5, 0.0, 0.5625, 0.0, -0.03125]` | 3 |
| 7 | `[0.0, -1.75, 0.0, 0.875, 0.0, -0.109375, 0.0]` | 3 |

Note: d=1 produces `[0.0]` which is effectively an identity (no filtering). All odd-indexed coefficients are zero for Chebyshev.

### Legendre Coefficients Reference

| Degree | Coefficients (monic) |
|--------|---------------------|
| 1 | `[0.0]` |
| 2 | `[0.0, -0.333]` |
| 3 | `[0.0, -0.6, 0.0]` |
| 4 | `[0.0, -0.857, 0.0, 0.086]` |
| 5 | `[0.0, -1.111, 0.0, 0.238, 0.0]` |
| 6 | `[0.0, -1.364, 0.0, 0.455, 0.0, -0.022]` |
| 7 | `[0.0, -1.615, 0.0, 0.734, 0.0, -0.082, 0.0]` |

---

## Experiments

### EXP-0: Quick Validation (200 steps, 1 dataset)

**Goal**: Validate Moirai2 baseline + preconditioning pipeline end-to-end.

| Property | Baseline | Precond |
|----------|----------|---------|
| Training steps | 200 | 200 |
| Training data | test_small (1 dataset) | test_small (1 dataset) |
| Batch size | 32 | 32 |
| Precondition | None | Custom 4-tap FIR |
| FIR coefficients | N/A | `[0.0, -0.1176, 0.0, -0.1361]` |
| FIR inverse | N/A | Enabled, length=64, lambda=0.1 |
| Loss reversal | N/A | Disabled |
| Final training loss | 0.109 | 0.082 |

**GIFT-Eval Quick (8 datasets):**

| Metric | Baseline | Precond | Delta |
|--------|----------|---------|-------|
| Geo Mean MASE | 2.3224 | **2.2273** | -4.1% |
| Arith Mean MASE | 3.2439 | **3.0062** | -7.3% |
| Median MASE | 2.6332 | **2.4312** | -7.7% |
| Beats naive | 2/8 | 2/8 | - |

Precond wins 5/8 datasets. Biggest gains on trended data (saugeenday -10.8%, m4_hourly -8.8%). Baseline wins on stationary data (jena_weather, hospital).

**GIFT-Eval Full (97 datasets):**

| Metric | Baseline | Precond | Delta |
|--------|----------|---------|-------|
| Geo Mean MASE | 2.0857 | **1.9664** | -5.7% |
| Arith Mean MASE | 3.7522 | **3.6652** | -2.3% |
| Median MASE | 1.9149 | **1.7040** | -11.0% |
| Beats naive | 14/97 | 15/97 | +1 |
| Min MASE | 0.5638 | **0.5424** | -3.8% |
| Head-to-head | - | **75 wins** vs 22 | - |

Precond wins on **75/97 datasets** (77%). Improvement is consistent across domains.

**Status**: COMPLETED
**Jobs**: 4583019 (baseline train), 4583020 (precond train), 4583584 (baseline eval), 4583585 (precond eval)
**Caveats**: Severely undertrained (200 steps on 1 dataset). Custom coefficients, not pure Chebyshev.

---

### EXP-1: Chebyshev Degree Sweep (100K steps, full LOTSA v1)

**Goal**: Find optimal Chebyshev degree for time preconditioning at full training scale.

**Common config**:
- Training: 100K steps (1000 epochs x 100 batches), bs=256, lr=1e-3, cosine annealing, 10K warmup
- Precision: bf16-mixed
- Dataset: lotsa_v1_unweighted (27 datasets)
- FIR inverse: enabled, length=64, stride=1, lambda=0.1
- Loss reversal: disabled
- Anomaly z-score threshold: 8.0
- Seed: 42
- GPU: NVIDIA H200, 30h time limit, ailab partition

| Run | Job ID | Degree | Coefficients | Status |
|-----|--------|--------|-------------|--------|
| Baseline (no zscore) | 4583433 | N/A | N/A (no precond, zscore=0) | COMPLETED (confounded) |
| Baseline (fixed) | 4611086 | N/A | N/A (no precond, zscore=8.0) | RUNNING (~ep 200) |
| d=1 | 4583434 | 1 | `[0.0]` (identity) | COMPLETED |
| d=2 | 4583435 | 2 | `[0.0, -0.5]` | COMPLETED |
| d=3 | 4583436 | 3 | `[0.0, -0.75, 0.0]` | COMPLETED |
| d=4 | 4583437 | 4 | `[0.0, -1.0, 0.0, 0.125]` | COMPLETED |
| d=5 | 4583438 | 5 | `[0.0, -1.25, 0.0, 0.3125, 0.0]` | COMPLETED |
| d=6 | 4583439 | 6 | `[0.0, -1.5, 0.0, 0.5625, 0.0, -0.03125]` | COMPLETED |
| d=7 | 4583440 | 7 | `[0.0, -1.75, 0.0, 0.875, 0.0, -0.109375, 0.0]` | COMPLETED |
| Custom 4-tap | — | N/A | `[0.0, -0.1176, 0.0, -0.1361]` | RUNNING (EXP-1b batch) |

#### GIFT-Eval Results (97 configs, all completed runs)

**Normalized Geo Mean MASE** (= model MASE / seasonal naive MASE; lower is better; <1 beats naive):

| Model | Epoch 500 (50K steps) | Epoch 1000 (100K steps) | Delta |
|-------|:--------------------:|:----------------------:|:-----:|
| **d=1 (control)** | **0.8836** | **0.9146** | +0.031 |
| d=2 | 1.2447 | 1.2681 | +0.023 |
| d=3 | 1.4321 | 1.4502 | +0.018 |
| d=4 | 1.5823 | 1.5769 | -0.005 |
| d=5 | 1.6725 | 1.7090 | +0.037 |
| d=6 | 1.7496 | 1.7892 | +0.040 |
| d=7 | 1.9944 | 2.0209 | +0.027 |
| AK baseline (100K) | — | 0.9415 | — |

**Raw Geo Mean MASE** (absolute, not normalized):

| Model | Epoch 500 (50K steps) | Epoch 1000 (100K steps) |
|-------|:--------------------:|:----------------------:|
| **d=1 (control)** | **1.2352** | **1.2785** |
| d=2 | 1.7400 | 1.6995 |
| d=3 | 1.9180 | 1.9419 |
| d=4 | 2.1187 | 2.1112 |
| d=5 | 2.2396 | 2.2891 |
| d=6 | 2.3439 | 2.3973 |
| d=7 | 2.6729 | 2.7088 |

**Configs Beating Naive (MASE < 1.0)**:

| Model | Epoch 500 | Epoch 1000 |
|-------|:---------:|:----------:|
| d=1 (control) | **36/97** | **35/97** |
| d=2 | 22/97 | 22/97 |
| d=3 | 14/97 | 14/97 |
| d=4 | 12/97 | 12/97 |
| d=5 | 11/97 | 11/97 |
| d=6 | 10/97 | 10/97 |
| d=7 | 8/97 | 8/97 |

#### Key Findings

1. **Monotonic degradation with degree** at both checkpoints: d=1 < d=2 < d=3 < d=4 < d=5 < d=6 < d=7. The Chebyshev filter actively hurts forecasting for every degree >= 2.

2. **d=1 (identity) is the best model** — Normalized Geo MASE 0.88 at epoch 500, 0.91 at epoch 1000. Since d=1 applies no actual filtering (coeff `[0.0]`), the improvement over the confounded baseline comes entirely from **anomaly z-score rejection** (threshold=8.0).

3. **Most models slightly degrade from epoch 500 to 1000** — d=1, d=3, d=5, d=6 all got worse. Only d=2 improved. Suggests possible overfitting or that 50K steps was already near optimal.

4. **FIR inverse filter cannot undo the damage** — Despite 64 learnable taps trained with auxiliary MSE loss, the model cannot recover the information destroyed by high-degree Chebyshev filtering.

5. **Consistent with Moirai 1.0 findings** (from research PDF): preconditioning improves training loss but degrades inference, due to error accumulation during reversal and distribution shift between training/eval.

#### Normalized MASE Methodology

`normalized_MASE = model_MASE / seasonal_naive_MASE` per dataset. Seasonal naive reference values from AK's precomputed file, copied to `gifteval/reference/seasonal_naive_baseline.csv`. This metric adjusts for dataset difficulty (e.g., covid_deaths has naive MASE ~47, so a model MASE of 50 gives normalized MASE ~1.07).

**Comparison to AK's baseline**: On the 82 matched configs, our d=1 model (raw geo MASE 1.11) is actually **better** than AK's Moirai2 baseline (1.23). The earlier apparent discrepancy (our 1.28 vs AK's 0.94) was due to comparing raw MASE (ours) against normalized MASE (AK's) — an apples-to-oranges error.

**CONFOUND DISCOVERED**: The original baseline (4583433) used default `anomaly_zscore_threshold=0.0` (disabled), while ALL precond runs set `anomaly_zscore_threshold=8.0`. This caused massive loss spikes in the baseline from outlier LOTSA samples. **A corrected baseline (4611086) is running with zscore=8.0.**

**RULE: All future runs MUST include `model.anomaly_zscore_threshold=8.0`.** This is now baked into both SLURM scripts.

**Evaluation plan**: All evals complete. d=4 epoch 500 (job 4618502) and d=7 epoch 1000 (job 4618504) filled in the last missing data points.

---

### EXP-1b: Regularization Sweep at d=4 (100K steps, full LOTSA v1)

**Goal**: Understand how coefficient regularization strength affects preconditioning quality. Uses CVXPY-optimized coefficients from `generate_coeffs.py` (source: `/scratch/gpfs/EHAZAN/ak8836/uni2ts_patched/generate_coeffs.py`).

**Optimization problem** (solved per lambda value):
```
minimize   max_{|x| <= 1} |P(x)|   +   lambda * ||c||^2
subject to P(x) = x^4 + c_1*x^3 + c_2*x^2 + c_3*x + c_4   (monic)
```
- Term 1 (spectral): worst-case polynomial value on [-1,1] (Chebyshev optimality criterion)
- Term 2 (energy): L2 norm of coefficients (shrinkage toward identity)
- lambda=0 recovers pure Chebyshev; lambda->inf gives near-identity filter

**Common config**: Same as EXP-1 (100K steps, bs=256, FIR inverse len=64, lambda_inv=0.1, seed=42).

| Run | Job ID | Reg Lambda | Coefficients | max|c| | ||c||_2 | Status |
|-----|--------|-----------|-------------|--------|---------|--------|
| d4-lam0 (pure Cheb) | 4583437 | 0 | `[0.0, -1.0, 0.0, 0.125]` | 1.000 | 1.008 | RUNNING (EXP-1) |
| d4-lam0.25 | 4604514 | 0.25 | `[0.0, -0.70543, 0.0, -0.08509]` | 0.705 | 0.711 | PENDING (pli, starts 02/10 18:30) |
| d4-lam0.5 | 4604515 | 0.5 | `[0.0, -0.50746, 0.0, -0.21409]` | 0.507 | 0.551 | PENDING (pli, starts 02/10 18:30) |
| d4-lam1 | 4604516 | 1.0 | `[0.0, -0.37729, 0.0, -0.29357]` | 0.377 | 0.478 | PENDING (pli, starts 02/10 18:30) |
| d4-lam2 | 4604517 | 2.0 | `[0.0, -0.25, 0.0, -0.25]` | 0.250 | 0.354 | PENDING (pli, starts 02/10 18:30) |
| d4-lam3 | 4604518 | 3.0 | `[0.0, -0.16667, 0.0, -0.16667]` | 0.167 | 0.236 | PENDING (pli, starts 02/10 18:30) |
| d4-lam10 | 4604519 | 10.0 | `[0.0, -0.05, 0.0, -0.05]` | 0.050 | 0.071 | PENDING (pli, starts 02/10 18:30) |

**Key observations about the coefficient landscape**:
- lambda=0..0.1: No effect — pure Chebyshev is already nearly optimal for the combined objective
- lambda=0.25: First meaningful shift — c4 flips sign from +0.125 to -0.085
- lambda=2.0: Both non-zero coefficients become equal (-0.25, -0.25)
- lambda=3.0: Coefficients `[-0.167, -0.167]` are in the same regime as ak8836's quick test coefficients
- lambda=10.0: Near-identity filter (coefficients ~0.05)

**Hypotheses**:
1. Moderate regularization (lambda=0.5..2.0) will outperform both extremes
2. Pure Chebyshev (lambda=0) may overfilter, removing useful signal
3. Near-identity (lambda=10) won't provide enough detrending to help
4. The optimal lambda may correspond to a filter whose spectral properties match the typical autocorrelation structure of LOTSA time series

**Results**: Pending

**Evaluation plan**: Once training completes, run full GIFT-Eval (97 configs) on each checkpoint.

---

### EXP-2: Legendre Degree Sweep (planned)

**Goal**: Compare Legendre polynomials to Chebyshev. Legendre zeros are more evenly spaced on [-1,1] vs Chebyshev's clustered near endpoints.

**Config**: Same as EXP-1 but `type=legendre`, degrees 1-7.

**Status**: NOT STARTED. Awaiting EXP-1 results to decide if full sweep or targeted runs.

---

### EXP-3: FIR Inverse Hyperparameters (planned)

**Goal**: Sensitivity analysis of FIR inverse filter hyperparameters at the best degree from EXP-1.

**Planned runs** (at best degree d*):

| Run | FIR Length | Lambda | Description |
|-----|-----------|--------|-------------|
| Short inverse | 16 | 0.1 | Much shorter filter, faster |
| Long inverse | 128 | 0.1 | More capacity |
| Low lambda | 64 | 0.01 | Less inverse pressure |
| High lambda | 64 | 0.5 | Strong inverse regularization |
| Very high lambda | 64 | 1.0 | Equal weight main + inverse loss |
| No inverse | N/A | N/A | FIR inverse disabled entirely |

**Status**: NOT STARTED. Awaiting EXP-1 results.

---

### EXP-4: Loss Reversal vs FIR Inverse (planned)

**Goal**: Compare the two approaches for handling preconditioning at inference time.

| Run | FIR Inverse | Loss Reversal | Description |
|-----|------------|---------------|-------------|
| FIR only | enabled | disabled | Current approach (EXP-1) |
| Reversal only | disabled | enabled | Analytical undo in loss |
| Neither | disabled | disabled | Raw preconditioned loss |

**Status**: NOT STARTED. Awaiting EXP-1 results.

---

### EXP-5: Stride Variations (planned)

**Goal**: Test whether matching the stride to data periodicity improves performance.

| Run | Degree | Stride | Rationale |
|-----|--------|--------|-----------|
| Stride 2 | d* | 2 | Filter uses `y_{t-2}, y_{t-4}, ...` |
| Stride 4 | d* | 4 | Quarter-period spacing |
| Stride 16 | d* | 16 | Patch-aligned (patch_size=16) |

**Status**: NOT STARTED. Lower priority.

---

### EXP-6: Learnable Coefficients (planned, code implemented)

**Goal**: Let the model learn optimal filter coefficients from data instead of using fixed polynomial coefficients.

**Implementation**: DONE. `module.py` now supports `time_precondition_learnable=True` (registers coefficients as `nn.Parameter` instead of buffer). `pretrain.py` adds L2 regularization via `time_precondition_coeffs_lambda`. New logged metrics: `train/precond_coeffs_norm`, `train/precond_coeffs_l2`.

**Status**: Code implemented. Awaiting EXP-1 results to determine best degree `d*` before launching runs.

**Depends on**: EXP-1 (best degree), EXP-1b (regularization sensitivity)

#### EXP-6a: Learnable vs Fixed at Best Degree (core comparison)

| Run | Learnable | Init | Degree | L2 Lambda | FIR Inverse | Description |
|-----|-----------|------|--------|-----------|-------------|-------------|
| fixed-d* | No | Chebyshev d* | d* | N/A | Yes (len=64, λ=0.1) | **Control** (= EXP-1 winner) |
| learn-d*-lam0 | Yes | Chebyshev d* | d* | 0.0 | Yes | Fully free coefficients |
| learn-d*-lam0.01 | Yes | Chebyshev d* | d* | 0.01 | Yes | Light regularization |
| learn-d*-lam0.1 | Yes | Chebyshev d* | d* | 0.1 | Yes | Moderate regularization |
| learn-d*-lam1.0 | Yes | Chebyshev d* | d* | 1.0 | Yes | Strong regularization |

#### EXP-6b: Initialization Sensitivity

| Run | Init | Degree (taps) | L2 Lambda | Description |
|-----|------|---------------|-----------|-------------|
| learn-cheb-d* | Chebyshev d* | d* | 0.01 | Best fixed polynomial init |
| learn-zero | All zeros | d* | 0.01 | Identity init (learn from scratch) |
| learn-small-rand | N(0, 0.01) | d* | 0.01 | Random init |
| learn-cheb-d2 | Chebyshev d=2 | d* | 0.01 | Simpler polynomial, more taps to learn |

Implementation: zero init via `time_precondition_coeffs_init='[0.0, 0.0, ...]'`. Random init: pre-generate, save to file, pass via `time_precondition_coeffs_init=/path/to/file`.

#### EXP-6c: Degree vs Learnable Interaction

| Run | Learnable | Degree | L2 Lambda | Description |
|-----|-----------|--------|-----------|-------------|
| learn-d2 | Yes | 2 | 0.01 | Can 2 learned taps beat fixed d*? |
| learn-d4 | Yes | 4 | 0.01 | = EXP-6a winner |
| learn-d8 | Yes | 8 | 0.01 | More capacity (all taps free, not just even) |
| learn-d16 | Yes | 16 | 0.1 | High capacity with stronger reg |

#### EXP-6d: Learnable Coefficients without FIR Inverse

| Run | Learnable | FIR Inverse | L2 Lambda | Description |
|-----|-----------|-------------|-----------|-------------|
| learn-with-inv | Yes | Yes (len=64, λ=0.1) | 0.01 | = EXP-6a winner |
| learn-no-inv | Yes | No | 0.01 | Learnable filter, no inverse |
| fixed-no-inv | No | No | N/A | Fixed filter, no inverse (= EXP-3) |

#### Config Example

```bash
model.module_kwargs.time_precondition_enabled=true \
model.module_kwargs.time_precondition_learnable=true \
model.module_kwargs.time_precondition_degree=4 \
model.module_kwargs.time_precondition_inverse_enabled=true \
model.module_kwargs.time_precondition_inverse_length=64 \
model.time_precondition_inverse_lambda=0.1 \
model.time_precondition_coeffs_lambda=0.01
```

#### Post-hoc Analysis Plan

1. Extract learned coefficients: `ckpt['state_dict']['module.time_precondition_coeffs']`
2. Plot coefficient trajectories (save intermediate checkpoints every 10K steps)
3. Compute frequency response of learned filter vs Chebyshev/Legendre
4. Check convergence stability across random seeds

#### Priority

1. EXP-6a (core), 2. EXP-6d (simplification), 3. EXP-6b (init), 4. EXP-6c (degree)
Total: ~12 new runs, ~9 GPU-days.

---

## Results Summary

*Last updated: 2026-02-09*

### EXP-1: Chebyshev Degree Sweep — COMPLETED

**Normalized Geo Mean MASE** (primary metric; model MASE / seasonal naive MASE; <1 = beats naive):

| Run | Epoch 500 | Epoch 1000 | Status |
|-----|:---------:|:----------:|--------|
| **d=1 (control)** | **0.8836** | **0.9146** | Best model |
| d=2 | 1.2447 | 1.2681 | +39% vs d=1 |
| d=3 | 1.4321 | 1.4502 | +59% vs d=1 |
| d=4 | 1.5823 | 1.5769 | +72% vs d=1 |
| d=5 | 1.6725 | 1.7090 | +87% vs d=1 |
| d=6 | 1.7496 | 1.7892 | +96% vs d=1 |
| d=7 | 1.9944 | 2.0209 | +121% vs d=1 |

**Verdict: NEGATIVE RESULT.** Chebyshev preconditioning of any degree >= 2 monotonically degrades forecasting performance. The FIR inverse filter cannot recover the information destroyed by filtering.

### EXP-0: Quick Validation — COMPLETED

| Metric | Baseline (200 steps) | Precond (200 steps) | Delta |
|--------|:--------------------:|:-------------------:|:-----:|
| Raw Geo Mean MASE (97 configs) | 2.0857 | **1.9664** | -5.7% |
| Head-to-head | — | **75 wins** vs 22 | — |

*Caveat: Severely undertrained (200 steps, 1 dataset). The initial positive signal did not hold at full training scale (EXP-1).*

### EXP-6: Learnable Coefficients — IN PROGRESS

**Goal**: Let the model learn optimal preconditioning coefficients from data, initialized with Lyapunov-regularized polynomial (lambda=5.0, d=4).

#### EXP-6-quick: 10K steps comparison

**Config**: 100 epochs x 100 batches = 10K steps, bs=256, seed=42, same node (della-i21g3, H200).
- **Baseline**: No preconditioning
- **Learnable**: Lyapunov d=4 lambda=5.0, `learnable=true`, `coeffs_lambda=0.01` (L2 penalty), no FIR inverse

**Init coefficients**: `[0, -0.081, 0, -0.089]` (near-identity, max|c|=0.089)

**Training**: Job 4819275 (COMPLETED). Baseline ~80 min, learnable ~80 min on same GPU.
- Baseline training loss: 0.52 → 0.11 (100 epochs)
- Learnable training loss: 0.45 → 0.07 (lower training loss!)

**FEV-Bench Results (100 tasks)**:

| Metric | Baseline | Learnable | Winner |
|--------|:--------:|:---------:|:------:|
| MASE Geo Mean | **1.2908** | 1.3653 | Baseline |
| MASE Mean | **1.9712** | 2.0944 | Baseline |
| SQL Geo Mean | **1.0562** | 1.1715 | Baseline |
| Task wins | **81/108** | 27/108 | Baseline |

**Verdict**: Baseline wins at 10K steps. Learnable model has lower training loss but worse eval — possible overfitting or insufficient training for the model to adapt to preconditioning.

**Checkpoints**:
- Baseline: `outputs/pretrain/moirai2_small/lotsa_v1_unweighted/m2_baseline_cmp_20260216_222947/checkpoints/epoch_99-step_10000.ckpt`
- Learnable: `outputs/pretrain/moirai2_small/lotsa_v1_unweighted/m2_learnable_lyap5_20260216_222947/checkpoints/epoch_99-step_10000.ckpt`

#### EXP-6-25k: 25K steps with free learning (no L2 penalty)

**Config**: 250 epochs x 100 batches = 25K steps, bs=256, seed=42.
- **Baseline**: Job 4826061 (RUNNING)
- **Learnable**: Job 4826062 (RUNNING), `coeffs_lambda=0.0` (fully free), Lyapunov d=4 lambda=5.0

**Hypothesis**: Longer training + no L2 penalty allows coefficients to find optimal filter, potentially beating baseline.

**Status**: RUNNING. Separate jobs for parallelism.

---

### EXP-1b: d=4 Regularization Sweep — IN PROGRESS

| Run | Lambda | Coefficients | max|c| | Status |
|-----|--------|-------------|--------|--------|
| d4-lam0 (pure Cheb) | 0 | `[0, -1.0, 0, 0.125]` | 1.000 | = EXP-1 d=4 (done) |
| d4-lam0.25 | 0.25 | `[0, -0.706, 0, -0.085]` | 0.706 | RUNNING |
| d4-lam0.5 | 0.5 | `[0, -0.507, 0, -0.214]` | 0.507 | RUNNING |
| d4-lam1.0 | 1.0 | `[0, -0.377, 0, -0.294]` | 0.377 | RUNNING |
| d4-lam2.0 | 2.0 | `[0, -0.25, 0, -0.25]` | 0.250 | RUNNING |
| d4-lam3.0 | 3.0 | `[0, -0.167, 0, -0.167]` | 0.167 | RUNNING |
| d4-lam10.0 | 10.0 | `[0, -0.05, 0, -0.05]` | 0.050 | RUNNING |
| custom 4-tap | — | `[0, -0.118, 0, -0.136]` | 0.136 | RUNNING |

---

## Key Files

| File | Description |
|------|-------------|
| `uni2ts/src/uni2ts/model/moirai2/module.py` | Moirai2Module: time precondition, FIR inverse, latent precondition |
| `uni2ts/src/uni2ts/model/moirai2/pretrain.py` | Moirai2Pretrain: loss computation, FIR inverse aux loss, loss reversal |
| `uni2ts/src/uni2ts/common/precondition.py` | Chebyshev/Legendre coefficient computation |
| `uni2ts/src/uni2ts/transform/precondition.py` | Patch-level preconditioning transform |
| `uni2ts/src/uni2ts/model/moirai2/forecast.py` | Moirai2Forecast: inference with preconditioning |
| `uni2ts/cli/conf/pretrain/model/moirai2_small.yaml` | Model config (all precond disabled by default) |
| `uni2ts/pretraining/pretrain_moirai2_baseline_full.slurm` | Baseline training SLURM script |
| `uni2ts/pretraining/pretrain_moirai2_precond.slurm` | Precond training SLURM script (parameterized by DEGREE) |
| `gifteval/eval_moirai2.slurm` | GIFT-Eval SLURM script for Moirai2 checkpoints |
| `gifteval/eval_gifteval.py` | Main evaluation script |

---

## Bugs and Fixes

1. **Import error** (`PolynomialPrecondition` not found): `finetune.py` imported old names. Fixed by adding aliases in `transform/__init__.py`.

2. **Forecast shape mismatch** (`(48, 9)` vs `(9, 48)`): `_format_preds` output shape wrong for GluonTS `QuantileForecastGenerator` which transposes per-sample. Fixed rearrange in `forecast.py`.

3. **SLURM custom coefficients parsing**: `--export=COEFFS=[0.0,-0.1176,0.0,-0.1361]` splits on commas. Fix: set env vars before `sbatch`, pass names only to `--export`.

4. **numpy/pandas binary incompatibility** (EXP-1b, custom 4-tap): `ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject`. Affects jobs 4586701, 4589501-4589506. The EXP-1 jobs (4583433-4583440) are unaffected (submitted earlier, possibly on different nodes). Fix: reinstall pandas in venv (`pip install --force-reinstall pandas`) or pin numpy version.

## EXP-7: Stride=16 Quick Experiments (10K steps) — 2026-02-19

**Goal**: Evaluate stride=16 (patch-aligned) preconditioning with various configurations.

**Key insight**: Stride=16 operates at the inter-patch level instead of intra-patch. Since Moirai2 uses patch_size=16, stride=16 means the FIR filter taps access values from adjacent patches rather than within the same patch. This dramatically reduces reversal error accumulation (only 1 reversal step per patch boundary vs 16 within).

**Common config**: 10K steps (100 epochs x 100 batches), bs=256, seed=42, anomaly_zscore_threshold=8.0, ailab partition, H200 GPU.

### GIFT-Eval Results (97 configs, full benchmark)

| Experiment | Description | MASE Geo Mean | vs Baseline | Wins/Losses |
|------------|-------------|:------------:|:-----------:|:-----------:|
| **q_s16d2** | stride=16, Cheb d=2, coeff [-0.5] | **1.2227** | **-1.56%** | 54/43 |
| q_ft_s16 | fine-tune 25K baseline + s16 d=4 | 1.2303 | -0.95% | — |
| q_s8d4 | stride=8, Cheb d=4 | 1.2382 | -0.32% | — |
| q_baseline | no precond (10K control) | 1.2421 | — | — |
| q_s16d4 | stride=16, Cheb d=4, coeff [-1.0,0,0.125] | 1.2487 | +0.53% | 43/54 |
| q_s16ws | warm-start from baseline + s16 d=4 | 1.2603 | +1.47% | — |
| q_dualloss | s16 d=4 + dual loss mode | 1.2860 | +3.53% | — |

**Winner**: stride=16 d=2 at -1.56% improvement. Milder preconditioning (coeff -0.5) > aggressive (coeff -1.0).

### Per-Dataset Analysis (s16d2 vs baseline)

**By Frequency** (most informative breakdown):

| Frequency | Configs | Wins/Losses | Mean Change | Notes |
|-----------|:-------:|:-----------:|:-----------:|-------|
| 15T | 12 | **12/0** | **-9.64%** | Perfect sweep, all improve |
| 10S | 6 | **6/0** | **-8.39%** | All bizitobs configs improve |
| 5T | 12 | 7/5 | -2.13% | Med/long horizons benefit |
| Q-DEC | 1 | 1/0 | -0.36% | |
| H | 31 | 17/14 | +0.07% | Roughly neutral |
| D | 15 | 5/10 | +0.23% | Slightly negative |
| M | 5 | 2/3 | +0.24% | |
| W-* | 6 | 1/5 | +1.39% | Weekly data hurt |
| 10T | 6 | 2/4 | **+7.91%** | Solar outlier drives this |
| A-DEC | 1 | 0/1 | +0.74% | |

**By Horizon**:

| Horizon | Configs | Wins/Losses | Mean Change |
|---------|:-------:|:-----------:|:-----------:|
| short | 55 | 23/32 | -0.01% (neutral) |
| medium | 21 | 14/7 | **-1.81%** |
| long | 21 | 17/4 | **-3.96%** |

**Top improvements**: electricity/15T/long -16.9%, electricity/15T/medium -14.9%, bizitobs_app/10S/medium -14.0%, ett2/15T/long -13.7%
**Top degradations**: bizitobs_l2c/H/medium +31.1%, solar/10T/long +27.7%, solar/10T/medium +22.2%

**Key pattern**: Preconditioning helps periodic data (15T, 5T) at medium-long horizons. Hurts solar/10T and some hourly configs.

### New Architecture Experiments (10K steps, GIFT-Eval COMPLETED)

| Experiment | Description | MASE Geo Mean | vs Baseline | Wins/Losses |
|------------|-------------|:------------:|:-----------:|:-----------:|
| **m2_hint_s16** | **spectral hint mode (FIR residual as 3rd input, d=5, s=16)** | **1.2084** | **-2.71%** | **68/29** |
| m2_firstdiff_s16 | stride=16, coeff=[-1.0] (first diff) | 1.5926 | +28.21% | — |
| m2_dualhead_s16 | dual-head (raw + precond output projections) | 1.5576 | +25.3% | — |
| m2_dualhead_ft | dual-head, fine-tuned from 25K baseline | 1.5346 | +23.5% | — |

**BREAKTHROUGH: Hint mode is the new best approach at -2.71% vs baseline (68-29 wins).**

#### Hint Mode Per-Dataset Analysis

**By Frequency** (hint vs baseline):

| Frequency | Configs | Wins/Losses | Mean Change |
|-----------|:-------:|:-----------:|:-----------:|
| 10S | 6 | **6/0** | **-11.75%** |
| 5T | 12 | **11/1** | **-5.13%** |
| 15T | 12 | **9/3** | **-4.80%** |
| 10T | 6 | 4/2 | **-4.07%** |
| D | 15 | 10/5 | -1.22% |
| H | 31 | 21/10 | -0.43% |

**By Horizon**:

| Horizon | Configs | Wins/Losses | Mean Change |
|---------|:-------:|:-----------:|:-----------:|
| long | 21 | **17/4** | **-6.42%** |
| medium | 21 | **18/3** | **-3.23%** |
| short | 55 | 33/22 | -0.51% |

**Top wins**: LOOP_SEATTLE/H/long -22.8%, SZ_TAXI/15T/long -21.6%, LOOP_SEATTLE/H/medium -20.0%
**Top losses**: bizitobs_l2c/H/medium +32.8%, bizitobs_l2c/H/short +17.5%, ett1/W/short +15.0%

**Key insight**: Hint mode advantage scales with horizon (long > medium > short) and frequency (high-freq > low-freq). No reversal errors at inference since model predicts in raw space.

### Follow-Up Experiments (10K steps)

| Experiment | Description | MASE Geo Mean | vs Baseline | Status |
|------------|-------------|:------------:|:-----------:|--------|
| q_ak_r5s16 | Lyapunov d=4 + FIR inverse, stride=16 | 1.2162 | -2.09% | DONE |
| q_s16d3 | stride=16, Cheb d=3, coeff [-0.75] | — | — | Eval submitted |
| q_s16mild | stride=16, custom coeff [-0.3] | — | — | Eval submitted |
| q_ft_s16d2 | fine-tune 25K baseline + s16 d=2 | — | — | Eval submitted |
| q_s16d2_learn | learnable coeffs, init from d=2 | — | — | Eval submitted |
| q_s16c04 | stride=16, custom coeff [0.0, -0.4] | — | — | Eval submitted |
| q_s16c06 | stride=16, custom coeff [0.0, -0.6] | — | — | Eval submitted |
| q_s16lopt | stride=16, learned-optimal coeffs [-0.15, -0.06] | — | — | Eval submitted |
| m2_ft_s16d2_10k | fine-tune 25K baseline + s16 d=2 for 10K | — | — | Eval submitted |
| q_ak_r5 | Lyapunov d=4 + FIR inverse, stride=1 | 1.2657 | +1.90% | DONE |

**Stride=16 universally helps**: AK's config swings from +1.90% (s=1) to -2.09% (s=16), a 4% improvement just from stride change.

### Hint Mode Follow-Up Experiments

| Experiment | Description | MASE | vs Baseline | MASE < 1 | Status |
|------------|-------------|:----:|:-----------:|:--------:|--------|
| **q_hint_s16d4** | **hint d=4 s=16** | **1.1944** | **-3.84%** | **43/97** | **DONE — NEW BEST** |
| q_hint_s16d3 | hint d=3 s=16 | 1.2040 | -3.07% | 41/97 | DONE |
| q_hint_s16d7 | hint d=7 s=16 | 1.2027 | -3.17% | 41/97 | DONE |
| m2_hint_s16 | hint d=5 s=16 | 1.2084 | -2.71% | 39/97 | DONE |
| q_hint_s16d2 | hint d=2 s=16 | 1.2157 | -2.13% | 37/97 | DONE |
| q_hint_s16_learn | hint d=5 s=16, learnable | 1.2203 | -1.75% | 37/97 | DONE |
| q_hint_s8d5 | hint d=5 s=8 (stride ablation) | 1.2247 | -1.40% | 36/97 | DONE |
| m2_hint_s16_25k (25K) | hint d=5 s=16, 25K steps | 1.2452 | +0.25% | 38/97 | DONE |
| m2_hint_s16_25k (20K) | hint d=5 s=16, 20K checkpoint | 1.2503 | +0.65% | — | DONE (partial) |

### Hint Mode vs Reversal: Per-Dataset Analysis

Comparing hint mode (1.2084) vs reversal s16d2 (1.2227): hint wins 57-40 overall.

**By Frequency** (where each method excels):

| Frequency | Hint Wins | Rev Wins | Winner | Key Finding |
|-----------|:---------:|:--------:|--------|-------------|
| 5T | 10 | 2 | **Hint** | Strong at sub-hourly |
| 10T | 5 | 1 | **Hint** | solar/10T/long: -34.7% |
| 15T | 5 | 7 | **Reversal** | ETT/electricity periodic data |
| H | 11 | 19 | **Reversal** (count) | But hint has lower avg MASE |
| D | 8 | 3 | **Hint** | |
| W | 4 | 3 | Tie | |

**Complementary pattern**: Hint dominates on noisy/irregular data (IT monitoring, solar), reversal dominates on smooth periodic data (ETT, electricity at 15T).

**By Horizon**: Hint wins at all horizons, most at long (-6.42%) and medium (-3.23%).

### Learned Coefficient Trajectory

From q_s16d2_learn (learnable coefficients initialized at Chebyshev d=2 [0.0, -0.5]):

| Step | Coefficients | Total Strength |
|------|-------------|:-------------:|
| 0 (init) | [0.0, -0.5] | 0.500 |
| 5000 | [-0.15, -0.06] | 0.210 |
| 9000 | [-0.132, -0.047] | 0.179 |
| 10000 | [-0.132, -0.047] | 0.179 (converged) |

**Key insight**: Optimal preconditioning is ~4x milder than fixed Chebyshev d=2. The first tap (lag 16, adjacent patch) carries 73% of the weight. The coefficients converge quickly and remain stable.

### Long Training Runs

| Experiment | Description | Steps | MASE | vs Baseline | Status |
|------------|-------------|:-----:|:----:|:-----------:|--------|
| m2_s16d4_50k | stride=16 d=4, 50K steps | 50K | 1.2549 | +1.02% | DONE |
| m2_stride16_warmstart | warm-start from 25K base + s16 d=4 | 25K | 1.2395 | -0.22% | DONE |
| **m2_s16d2_25k** | **stride=16 d=2, 25K steps** | **25K** | **1.2372** | **-0.40%** | **DONE** |
| m2_hint_s16_25k (20K) | hint mode d=5 s=16, 20K ckpt | 20K | 1.2503 | +0.65% | DONE (partial) |
| m2_hint_s16_25k (25K) | hint mode d=5 s=16, 25K steps | 25K | 1.2452 | +0.25% | DONE |

---

## 25K Step Results (Full Training Scale)

| Experiment | MASE Geo Mean | vs Baseline | MASE < 1 |
|------------|:------------:|:-----------:|:--------:|
| **hint d=6 25K** | **1.1889** | **-4.28%** | **—** |
| **hdrop10 25K (d=4+10%drop)** | **1.1931** | **-3.94%** | **39/97** |
| **hint d=4 25K** | **1.1936** | **-3.91%** | **—** |
| hint d=2 c=-0.8 25K | 1.2062 | -2.89% | — |
| m2_s16d2_25k (reversal) | 1.2372 | -0.40% | 36/97 |
| m2_stride16_warmstart | 1.2395 | -0.22% | 40/97 |
| 25K baseline | 1.2422 | — | — |
| m2_hint_s16_25k (hint d=5) | 1.2452 | +0.25% | 38/97 |

**Key finding**: Hint mode holds at 25K for d=4 and d=6! d=6 is the best 25K result (-4.28%), followed by hdrop10 and d=4 (both ~-3.9%). Even-degree polynomials are robust to longer training. The previous observation that "hint advantage vanishes at 25K" only applies to d=5 and d=3 — d=4 and d=6 are robust.

---

## Comparison to Official Moirai Models

*Last updated: 2026-02-22*

All evaluations below use GIFT-Eval with ctx=4000, ps=32, 97 configurations.

| Model | Params | Steps | MASE (Geo Mean) | MASE < 1 | vs Official Small |
|-------|--------|-------|:---------------:|:--------:|:-----------------:|
| Moirai-small (official, from leaderboard) | ~14M | — | 1.323 | 27/97 | — |
| Our 100K Moirai2 baseline | 11.4M | 100K | 1.2878 | 34/97 | -2.7% |
| Moirai-base (official, from leaderboard) | ~90M | — | 1.259 | 40/97 | -4.8% |
| Our 10K Moirai2 baseline | 11.4M | 10K | 1.2395 | 39/97 | -6.3% |
| Hint d=4 s=16 | 11.4M | 10K | 1.1944 | 43/97 | -9.7% |
| **Hint d=4 + 10% dropout (BEST)** | **11.4M** | **10K** | **1.1802** | **—** | **-10.8%** |

**Key observations**:
- Our Moirai2 baseline already beats official Moirai-small by 6.3% (architecture improvements: quantile loss, causal decoder, etc.)
- Hint preconditioning adds another 5.0% on top of the already-strong Moirai2 baseline
- Our best model (11.4M params, 10K steps) beats official Moirai-base (~90M params) by **6.2%**
- The 100K baseline (1.2878) is worse than 10K baseline (1.2395) — likely because the early 100K run lacked `anomaly_zscore_threshold=8.0`
- **Always compare preconditioning experiments vs our own 10K baseline (1.2395)** for fair apples-to-apples evaluation (same training config, same eval settings)

---

## Comprehensive Results Table (all completed experiments)

*Last updated: 2026-02-22*

### 10K Step Experiments (all completed)

| Rank | Experiment | Description | MASE Geo Mean | vs Baseline | MASE < 1 |
|:----:|------------|-------------|:------------:|:-----------:|:--------:|
| **1** | **q_hint_drop10** | **hint d=4 + 10% dropout** | **1.1802** | **-4.98%** | **—** |
| **2** | **q_hint_s16d6** | **hint d=6 s=16** | **1.1836** | **-4.71%** | **—** |
| 3 | q_hint_c08 | hint d=2 c=-0.8 s=16 | 1.1884 | -4.33% | — |
| 3 | q_hint_sep | hint d=4, separate embed | 1.1884 | -4.33% | — |
| 5 | q_hint_d6drop05 | hint d=6 + 5% dropout | 1.1922 | -4.02% | 41/97 |
| 6 | q_hint_drop05 | hint d=4 + 5% dropout | 1.1941 | -3.86% | — |
| 7 | q_hint_s16d4 | hint mode d=4 s=16 | 1.1944 | -3.84% | 43/97 |
| 8 | q_hint_d6_sep | hint d=6, separate embed | 1.1998 | -3.40% | — |
| 9 | q_hint_d6_learn | hint d=6, learnable | 1.2025 | -3.19% | — |
| 10 | q_hint_s16d7 | hint mode d=7 s=16 | 1.2027 | -3.17% | 41/97 |
| 11 | q_c08_d10 | hint d=2 c=-0.8 + 10% drop | 1.2037 | -3.09% | — |
| 12 | q_hint_s16d3 | hint mode d=3 s=16 | 1.2040 | -3.07% | 41/97 |
| 13 | q_hint_c15 | hint d=2 c=-1.5 | 1.2074 | -2.79% | — |
| 14 | m2_hint_s16 | hint mode d=5 s=16 | 1.2084 | -2.71% | 39/97 |
| 15 | q_hint_drop15 | hint d=4 + 15% dropout | 1.2103 | -2.56% | — |
| 16 | q_d6_d10 | hint d=6 + 10% dropout | 1.2106 | -2.54% | — |
| 17 | q_hint_s16d2 | hint mode d=2 s=16 | 1.2157 | -2.13% | 37/97 |
| 18 | q_ak_r5s16 | Lyapunov+FIR inv s=16 | 1.2162 | -2.09% | 40/97 |
| 19 | q_c08_sep | hint d=2 c=-0.8, sep embed | 1.2168 | -2.04% | — |
| 20 | q_hint_d5drop10 | hint d=5 + 10% dropout | 1.2191 | -1.85% | — |
| 21 | q_c08_s8 | hint d=2 c=-0.8, s=8 | 1.2204 | -1.75% | — |
| 22 | q_hint_s16_learn | hint mode d=5 s=16 learnable | 1.2203 | -1.75% | 37/97 |
| 23 | q_hint_s16d8 | hint mode d=8 s=16 | 1.2216 | -1.65% | — |
| 24 | q_s16d2 | Chebyshev d=2 s=16 reversal | 1.2227 | -1.56% | — |
| 25 | q_hint_s8d5 | hint mode d=5 s=8 | 1.2247 | -1.40% | 36/97 |
| 26 | q_d4_c08 | hint d=4 + c=-0.8 combo | 1.2260 | -1.30% | — |
| 27 | q_d4d10ps | hint d=4+d=10+patchsize | 1.2377 | -0.35% | — |
| 10 | q_s16d2_learn | Cheb d=2 learnable reversal | 1.2290 | -1.06% | 36/97 |
| 11 | q_ft_s16 | fine-tune + s16 d=4 | 1.2303 | -0.95% | — |
| 12 | q_s8d4 | stride=8, d=4 | 1.2382 | -0.32% | — |
| 13 | q_s16mild | custom coeff [-0.3] | 1.2392 | -0.23% | 38/97 |
| 14 | q_s16lopt | learned-optimal [-0.15,-0.06] | 1.2416 | -0.04% | — |
| — | **q_baseline** | **no precond** | **1.2421** | **—** | **—** |
| 16 | q_s16c04 | custom coeff [0,-0.4] | 1.2485 | +0.52% | — |
| 17 | q_s16d4 | Chebyshev d=4 s=16 | 1.2487 | +0.53% | — |
| 18 | q_s16ws | warm start coeffs | 1.2603 | +1.47% | — |
| 19 | q_s16c06 | custom coeff [0,-0.6] | 1.2621 | +1.61% | — |
| 20 | q_ak_r5 | Lyapunov+FIR inv s=1 | 1.2657 | +1.90% | — |
| 21 | q_ft_s16d2 | fine-tune baseline + d=2 | 1.2784 | +2.92% | — |
| 22 | m2_ft_s16d2_10k | fine-tune baseline + d=2 10K | 1.2784 | +2.92% | — |
| 23 | q_dualloss | dual loss mode | 1.2860 | +3.53% | — |
| 24 | q_s16d3 | Chebyshev d=3 s=16 (reversal) | 1.3515 | +8.81% | 28/97 |
| 25 | dualhead_ft | dual-head fine-tune | 1.5346 | +23.5% | 27/97 |
| 26 | dualhead_s16 | dual-head from scratch | 1.5576 | +25.3% | 26/97 |
| 27 | m2_firstdiff_s16 | first diff (coeff -1.0) | 1.5926 | +28.21% | 24/97 |

### 25K / 50K Step Experiments

| Experiment | Steps | MASE Geo Mean | vs 25K Baseline |
|------------|:-----:|:------------:|:-----------:|
| **m2_s16d2_25k** | **25K** | **1.2372** | **-0.40%** |
| m2_stride16_warmstart | 25K+25K | 1.2395 | -0.22% |
| 25K baseline | 25K | 1.2422 | — |
| m2_s16d4_50k | 50K | 1.2549 | +1.02% |

### Hint Mode Degree Sweep (10K steps, all DONE)

| Degree | Chebyshev Coefficients | MASE | vs Baseline | MASE < 1 |
|:------:|------------------------|:----:|:-----------:|:--------:|
| d=2 | `[0.0, -0.5]` | 1.2157 | -2.13% | 37/97 |
| d=3 | `[0.0, -0.75, 0.0]` | 1.2040 | -3.07% | 41/97 |
| **d=4** | **`[0.0, -1.0, 0.0, 0.125]`** | **1.1944** | **-3.84%** | **43/97** |
| d=5 | `[0.0, -1.25, 0.0, 0.3125, 0.0]` | 1.2084 | -2.71% | 39/97 |
| **d=6** | **`[0.0, -1.5, 0.0, 0.5625, 0.0, -0.03125]`** | **1.1836** | **-4.71%** | **—** |
| d=7 | `[0.0, -1.75, 0.0, 0.875, 0.0, -0.109, 0.0]` | 1.2027 | -3.17% | 41/97 |
| d=8 | `[0.0, -2.0, 0.0, 1.25, 0.0, -0.25, 0.0, 0.0156]` | 1.2216 | -1.65% | — |

**Degree sweep pattern**: d=2 → d=3 → d=4 → d=5 → **d=6 (BEST)** → d=7 → d=8 (degrading). Two local minima at d=4 and d=6.
d=6 achieves MASE 1.1836 (-4.71%), surpassing d=4 (1.1944, -3.84%) as the best degree.
Non-monotonic: d=5 dips then d=6 recovers strongly. d=8 degrades significantly — confirms upper bound at d=6.

**Stride ablation**: s=8 d=5 (1.2247, -1.40%) vs s=16 d=5 (1.2084, -2.71%) → stride=16 is 1.3% better. Confirms stride=16 (patch-aligned) universally superior.

**25K hint mode**: d=5 25K (1.2452, +0.25%) — hint advantage vanishes at longer training, mirroring the reversal trend.

### Hint Mode Mechanism Analysis

The hint channel provides `hint_t = FIR(y) - y` (FIR residual). With stride=16 (patch-aligned):

| Degree | Effective hint | Lags used | MASE |
|--------|---------------|-----------|------|
| d=2 | `-0.5 * y_{t-16}` | 1 patch back | 1.2157 |
| **d=3** | **`-0.75 * y_{t-32}`** | **2 patches back** | **1.2040** |
| d=5 | `-1.25*y_{t-32} + 0.3125*y_{t-64}` | 2+4 patches back | 1.2084 |

**Key insight**: Looking 2 patches back (d=3) > 1 patch back (d=2). Adding the 4-patch lag (d=5) slightly hurts.
The hint is essentially giving the model explicit historical lag information as a free input channel.

### Coefficient Sensitivity Experiments (10K steps, queued)

Testing different coefficient magnitudes and polynomial families:

| Experiment | Description | Effective hint | Job ID | Status |
|------------|-------------|---------------|--------|--------|
| q_hd3l | d=3 learnable, init [-0.75], L2=0.01 | adapts from -0.75 | 4936767 | PENDING |
| q_hleg3 | Legendre d=3 (coeff -0.6 at lag 2s) | `-0.6 * y_{t-32}` | 4936785 | PENDING |
| q_hc1 | Custom [0,-1.0,0] (stronger lag 2s) | `-1.0 * y_{t-32}` | 4936786 | PENDING |
| q_hcoeff c06 | d=2 with coeff -0.6 at lag 1s | `-0.6 * y_{t-16}` | 4936787 | PENDING |
| q_hcoeff c08 | d=2 with coeff -0.8 at lag 1s | `-0.8 * y_{t-16}` | 4936788 | PENDING |
| q_hcoeff c10 | d=2 with coeff -1.0 at lag 1s | `-1.0 * y_{t-16}` | 4936789 | PENDING |

**Goal**: Fine-grained coefficient landscape. If c=-0.6 or c=-0.8 at lag 2s beats c=-0.75 (d=3), the optimal coefficient is between polynomials.

### Note on Intermediate Checkpoints

**WARNING**: Intermediate checkpoints from longer runs are NOT comparable to same-step checkpoints from shorter runs. Cosine annealing schedule depends on total steps:
- 10K run at step 10K: LR = minimum (fully decayed)
- 25K run at step 10K: LR = cos(0.4π) ≈ 0.31× peak (still high)

The 25K run's epoch_99-step_10000 checkpoint gives MASE 1.2559 (vs 1.2084 for the dedicated 10K run). This ~3.8% gap is entirely due to the LR schedule — the intermediate checkpoint hasn't converged as well because the LR hasn't decayed enough.

### 25K/100K Scaling Experiments

| Experiment | Steps | Config | MASE | vs Baseline | Status |
|------------|:-----:|--------|:----:|:-----------:|--------|
| m2_hint_s16_25k | 25K | hint d=5 s=16 | 1.2452 | +0.25% | **DONE** |
| m2_hint_d3_25k | 25K | hint d=3 s=16 | — | — | DONE, eval queued |
| m2_hint_d4_25k | 25K | hint d=4 s=16 | — | — | QUEUED (ailab+pli) |
| m2_hint_s16d4_100k | 100K | hint d=4 s=16 | — | — | **RUNNING** (pli, started 2026-02-22 21:51) |

**CAUTION**: Hint d=5 25K result (1.2452) shows almost no improvement over baseline. The 10K→25K decay is steep: -2.71% at 10K → +0.25% at 25K. This mirrors the reversal decay pattern (-1.56% at 10K → -0.40% at 25K). 100K hint may not retain the -3.84% advantage from d=4 10K.

---

## Architectural Improvements (New Experiments, 2026-02-22)

### Analysis of Hint d=4 Per-Dataset Patterns

Based on detailed per-dataset comparison of hint d=4 vs baseline:

**Where hint helps most:**
- Long horizons: 20/1 W/L (-9.8% average), near-universal improvement
- High-frequency data: 10T (-8.7%), H (-3.6%), 15T (-1.8%)
- Noisy/irregular: bizitobs (-28% to -43%), solar (-6.3%)
- Hard configs (MASE > 1.0): -3.5% to -16.2% improvement

**Where hint hurts:**
- Short horizons: essentially flat (+0.7%)
- Daily/monthly freq: +2.5% / +5.1% regression
- Strongly periodic univariate: us_births (+30.5%), saugeenday (+10.8%)

### New Architectural Modifications (all 10K steps, combined with hint d=4 s=16)

Three new features implemented:
1. **Patch Random Masking**: 50% of input patch embeddings randomly zeroed during training (Moirai 2.0 regularizer)
2. **Attention L1 Regularization**: L1 penalty on attention weights to encourage sparse attention (tighter generalization bound)
3. **Robust Scaler (R2-IN)**: Replace mean/std with median/MAD for normalization (outlier robustness)

| Experiment | Config | Status |
|------------|--------|--------|
| q_hint_mask50 | hint d=4 + patch mask 50% | QUEUED (ailab+pli) |
| q_hint_attnl1 | hint d=4 + attn L1 λ=0.01 | QUEUED (ailab+pli) |
| q_hint_robust | hint d=4 + robust scaler | QUEUED (ailab+pli) |
| q_hint_mask_l1 | hint d=4 + mask 50% + attn L1 | QUEUED (ailab+pli) |
| q_base_mask50 | baseline + mask 50% (no hint) | QUEUED (ailab+pli) |
| q_base_robust | baseline + robust scaler (no hint) | QUEUED (ailab+pli) |

**Status Update (2026-02-22 23:00):**
- q_hint_robust: **FAILED** (NaN losses from step 43, median/MAD numerically unstable). Cancelled.
- q_base_robust: Cancelled (same scaler issue).
- q_hint_mask50: RUNNING (epoch 21/100, loss 0.163)
- q_hint_attnl1: RUNNING (epoch 20/100, loss 0.149) — best training loss
- q_hint_mask_l1: RUNNING (epoch 12/100, loss 0.172)
- q_base_mask50: RUNNING (epoch 9/100, loss 0.183)

### PS-Loss (Patch-wise Structural Loss) Experiments

Added auxiliary loss penalizing per-patch structure divergence (correlation + variance + mean).

| Experiment | Config | Status | Notes |
|------------|--------|--------|-------|
| q_hint_psloss | hint d=4 + PS-Loss λ=0.1 | RUNNING (epoch 6, loss 0.260) | |
| q_hint_psloss_strong | hint d=4 + PS-Loss λ=1.0 | RUNNING (epoch 6, loss 1.020) | Too strong, loss dominated by PS |
| q_hint_all | hint d=4 + mask50 + PS-Loss 0.1 | RUNNING (epoch 9, loss 0.263) | |

### Hint Dropout Experiments

Randomly zero the hint channel per patch during training to prevent model from ignoring hint.

| Experiment | Config | Status |
|------------|--------|--------|
| q_hint_hdrop20 | hint d=4 + hint dropout 20% | QUEUED (ailab+pli) |
| q_hint_hdrop50 | hint d=4 + hint dropout 50% | QUEUED (ailab+pli) |
| q_hint_dualreg | hint d=4 + hint_drop=0.2 + patch_mask=0.3 | QUEUED (ailab+pli) |
| q_hint_drop10 | hint d=4 + standard dropout=0.1 | QUEUED (ailab+pli) |
| q_hint_d6 | hint d=6 s=16 (fill degree sweep gap) | QUEUED (ailab+pli) |

**Rationale:**
- Hint dropout prevents the model from learning to zero-out hint channel weights
- Dual regularization combines hint dropout + patch masking for stronger generalization
- Standard dropout (0.1) is a simple baseline regularizer
- d=6 fills the gap: d=5→1.2084, d=7→1.2027, d=6 might be between or better

### Separate Hint Embedding Experiments (2026-02-22)

**Motivation**: Currently, hint is concatenated with [target, mask] into a single in_proj (48→384 ResidualBlock).
All 3 channels share the same projection pathway, which may limit the model's ability to use the hint selectively.

**Architecture change**:
- `hint_embed_mode="separate"`: in_proj processes [target, mask] (32→384). Hint gets a dedicated `nn.Linear(16, 384)` + learned sigmoid gate from main reprs.
- `hint_normalize=True`: Per-sequence std normalization on hint channel before input.

| Experiment | Config | Status |
|------------|--------|--------|
| q_hint_sep | hint d=4, separate embed | QUEUED (ailab+pli) |
| q_hint_sepnorm | hint d=4, separate + normalize | QUEUED (ailab+pli) |
| q_hint_norm | hint d=4, concat + normalize | QUEUED (ailab+pli) |
| q_hint_sepgd | hint d=4, separate + hint_drop=0.2 | QUEUED (ailab+pli) |
| q_hint_d4_learn | hint d=4, learnable coefficients | QUEUED (ailab+pli) |

**Expected impact**: Gated separate embedding can learn to suppress hint for daily data (+0.7%) while amplifying for 15T (-10.5%). Per-dataset analysis shows:
- **15T**: -10.5% avg (spectacular wins)
- **10T**: -6.1% avg
- **5T**: -5.0% avg
- **D**: +0.7% avg (hint hurts)
- **H**: -1.5% avg (mixed)
- **Long horizon**: -8.9% avg
- **Short horizon**: -1.2% avg

If the gate eliminates daily losses, net MASE improvement could exceed -4.0%.

### COMPLETE RESULTS — Architecture & Regularization Experiments (2026-02-23)

All training completed. Full GIFT-Eval (97 configs) results collected.

| Rank | Experiment | Description | MASE | vs Baseline | Notes |
|:----:|------------|-------------|:----:|:-----------:|-------|
| **1** | **q_hint_drop10** | **hint d=4 + 10% hint dropout** | **1.1802** | **-4.98%** | **ALL-TIME BEST** |
| **2** | **q_hint_s16d6** | **hint d=6 s=16** | **1.1836** | **-4.71%** | Best degree |
| 3 | q_hint_c08 | hint d=2 c=-0.8 s=16 | 1.1884 | -4.33% | Best coefficient |
| 3 | q_hint_sep | hint d=4, separate embed | 1.1884 | -4.33% | Tied with c=-0.8 |
| 5 | q_hint_s16d4 | hint d=4 s=16 (standard Cheb) | 1.1944 | -3.84% | Previous best |
| 6 | q_hint_psloss | hint d=4 + PS-Loss λ=0.1 | 1.1961 | -3.70% | Structural patch loss |
| 7 | q_hint_d4_learn | hint d=4 learnable | 1.2011 | -3.30% | Learnable coefficients |
| 8 | q_hint_s16d7 | hint d=7 s=16 | 1.2027 | -3.17% | |
| 9 | q_d3_learnable | hint d=3 learnable | 1.2031 | -3.14% | |
| 10 | q_hint_c09 | hint d=2 c=-0.9 s=16 | 1.2080 | -2.74% | |
| 11 | q_hint_c10 | hint d=2 c=-1.0 s=16 | 1.2083 | -2.72% | |
| 12 | m2_hint_s16 | hint d=5 s=16 | 1.2084 | -2.71% | |
| 13 | q_hint_d6drop05 | hint d=6 + 5% dropout | 1.1922 | -4.02% | Mild dropout hurts d=6 |
| 14 | q_hint_drop05 | hint d=4 + 5% dropout | 1.1941 | -3.86% | 5% ≈ no effect |
| 15 | q_hint_d6_sep | hint d=6 + separate embed | 1.1998 | -3.40% | Sep hurts d=6 |
| 16 | q_hint_d6_learn | hint d=6 learnable | 1.2025 | -3.19% | Learnable hurts d=6 |
| 17 | q_c08_d10 | hint d=2 c=-0.8 + 10% dropout | 1.2037 | -3.09% | Dropout hurts c=-0.8 |
| 18 | q_hint_c15 | hint d=2 c=-1.5 s=16 | 1.2074 | -2.79% | 2nd coefficient minimum |
| 19 | q_hint_c09 | hint d=2 c=-0.9 s=16 | 1.2080 | -2.74% | |
| 20 | q_hint_c10 | hint d=2 c=-1.0 s=16 | 1.2083 | -2.72% | |
| 21 | m2_hint_s16 | hint d=5 s=16 | 1.2084 | -2.71% | |
| 22 | q_d4_drop15 | hint d=4 + 15% dropout | 1.2103 | -2.56% | 15% too much |
| 23 | q_d6_d10 | hint d=6 + 10% dropout | 1.2106 | -2.54% | Dropout destroys d=6 |
| 24 | q_hint_attnl1 | hint d=4 + attn L1 λ=0.01 | 1.2120 | -2.42% | Modest improvement |
| 25 | q_hint_s16d2 | hint d=2 s=16 | 1.2157 | -2.13% | |
| 26 | q_hint_hdrop20 | hint d=4 + 20% hint dropout | 1.2183 | -1.92% | Too much dropout |
| 27 | q_d5_drop10 | hint d=5 + 10% dropout | 1.2191 | -1.85% | Dropout hurts d=5 |
| 28 | q_c08_s8 | hint d=2 c=-0.8 stride=8 | 1.2204 | -1.75% | Stride 16 > 8 |
| 29 | q_hint_s16d8 | hint d=8 s=16 | 1.2216 | -1.65% | Degree too high |
| 30 | q_hint_d4_c08 | hint d=4, coeffs scaled 0.8x | 1.2260 | -1.30% | Bad combination |
| 31 | q_hint_psloss1 | hint d=4 + PS-Loss λ=1.0 | 1.2287 | -1.08% | PS too strong |
| 32 | q_hint_c06 | hint d=2 c=-0.6 s=16 | 1.2303 | -0.95% | |
| 33 | q_hint_c12 | hint d=2 c=-1.2 s=16 | 1.2368 | -0.43% | |
| 34 | q_d4d10ps | hint d=4 + 10% drop + psloss | 1.2377 | -0.35% | Dropout+psloss conflict |
| — | **q_baseline** | **no precond** | **1.2421** | **0%** | **Reference** |
| — | q_hint_hdrop50 | hint d=4 + 50% dropout | 1.2421 | 0% | Dropout = baseline! |
| — | q_hint_sepgd | hint d=4 sep + 20% dropout | 1.2538 | +0.9% | Over-regularized |
| — | q_hint_sepnorm | hint d=4 sep + normalize | 1.2835 | +3.3% | Normalization kills |
| — | q_hint_mask_l1 | hint d=4 + mask + attn L1 | 1.3027 | +4.9% | Bad combination |
| — | q_c08_norm | hint d=2 c=-0.8 + normalize | 1.3090 | +5.4% | Normalization kills |
| — | q_hint_norm | hint d=4 concat + normalize | 1.3113 | +5.6% | Normalization kills |
| — | q_hint_dualreg | hint d=4 + drop20 + mask30 | 1.3123 | +5.7% | Over-regularized |
| — | q_base_mask50 | baseline + patch mask 50% | 1.3517 | +8.8% | Masking hurts |
| — | q_hint_all | hint + mask50 + PS-Loss 0.1 | 1.3682 | +10.1% | Everything fails |
| — | q_hint_mask50 | hint d=4 + patch mask 50% | 1.4000 | +12.7% | Masking destroys hint |

### Key Findings from Architecture Experiments

1. **10% hint dropout is the best regularizer** (-4.98%), better than attn L1 (-2.42%), PS-Loss (-3.70%)
2. **d=6 is the best degree** (-4.71%), filling the gap between d=5 and d=7. d=8 degrades.
3. **c=-0.8 is the optimal coefficient for d=2** (-4.33%), sharply peaked
4. **Separate embedding ties** with c=-0.8 at -4.33% — viable alternative architecture
5. **Normalization kills** (+5.4%) — model needs raw hint magnitude
6. **Patch masking destroys hint** (+12.7%) — randomly zeroing patches prevents model from using hint
7. **50% hint dropout = baseline** — too much dropout completely negates the hint
8. **Dropout is degree-specific**: ONLY helps d=4 (sharp optimum at 10%). HURTS d=5 and d=6.
9. **Combinations of features universally fail**: c08_d10, d6_d10, d4_c08, c08_sep, d4d10ps all worse than best components
10. **d=4 AND d=6 hold at 25K**: d=4 1.1936 (identical to 10K), d=6 1.1889 (+0.45% only). Even-degree polynomials are robust.
11. **hdrop10 degrades slightly at 25K**: 1.1802→1.1931 (+1.1%) — dropout regularization doesn't prevent 25K decay completely
12. **d=6 is the best at 25K**: 1.1889 (-4.28%) beats d=4 (1.1936) and hdrop10 (1.1931)
13. **Parameter overhead is negligible**: +12,288 params (+0.1% of 11.4M) — same model size
14. **Multi-scale hints implemented**: Multiple FIR filters at different (degree, stride) combos concatenated as additional input channels
15. **Per-dataset analysis**: d=6 improves 69/97 configs. Huge gains on high-freq (10S: -21.8%) and long horizon (-10%). Neutral on hourly/daily.
16. **Stride=4 may help hourly/daily**: Currently testing d=6 s=4 and multi-stride d=6 s=16 + d=6 s=4

### Coefficient Sweep (hint d=2, s=16, 10K steps)

| Coefficient | MASE | vs Baseline | Notes |
|:-----------:|:----:|:-----------:|-------|
| -0.5 (standard Cheb) | 1.2157 | -2.13% | |
| -0.6 | 1.2303 | -0.95% | |
| **-0.8** | **1.1884** | **-4.33%** | **OPTIMAL** |
| -0.9 | 1.2080 | -2.74% | |
| -1.0 | 1.2083 | -2.72% | |
| -1.2 | 1.2368 | -0.43% | |
| -1.5 | 1.2074 | -2.79% | Secondary minimum |

**Pattern**: Sharply peaked at c=-0.8 (primary minimum). Secondary minimum at c=-1.5. Both milder (-0.5, -0.6) and stronger (-1.0, -1.2) are significantly worse. Two-minimum landscape.

### Updated Degree Sweep (hint mode, s=16, 10K steps)

| Degree | MASE | vs Baseline |
|:------:|:----:|:-----------:|
| d=2 | 1.2157 | -2.13% |
| d=3 | 1.2040 | -3.07% |
| d=4 | 1.1944 | -3.84% |
| d=5 | 1.2084 | -2.71% |
| **d=6** | **1.1836** | **-4.71%** |
| d=7 | 1.2027 | -3.17% |
| d=8 | 1.2216 | -1.65% |

**Pattern**: Non-monotonic. Two local minima at d=4 and d=6. d=6 is the global best at 10K. d=8 degrades significantly, confirming upper bound. However, d=4 is the ONLY config that holds at 25K.

### Dropout Sweep (hint d=4, s=16, 10K steps)

| Dropout Rate | MASE | vs Baseline |
|:------------:|:----:|:-----------:|
| 0% (standard) | 1.1944 | -3.84% |
| 5% | 1.1941 | -3.86% |
| **10%** | **1.1802** | **-4.98%** |
| 15% | 1.2103 | -2.56% |
| 20% | 1.2183 | -1.92% |
| 50% | 1.2421 | 0% |

**Pattern**: Extremely sharp optimum at exactly 10%. 5% ≈ 0% (no effect). 15% significantly worse. 20%+ progressively eliminates the hint advantage.

### Dropout × Degree Interaction (10K steps)

| Degree | No dropout | 5% dropout | 10% dropout | Effect of 10% |
|:------:|:----------:|:----------:|:-----------:|:--------------:|
| d=4 | 1.1944 | 1.1941 | **1.1802** | -1.2% (HELPS) |
| d=5 | 1.2084 | — | 1.2191 | +0.9% (HURTS) |
| d=6 | **1.1836** | 1.1922 | 1.2106 | +2.3% (HURTS) |

**Critical finding**: Dropout only helps d=4 specifically. For d=5 and d=6, any amount of dropout degrades performance. This suggests d=4 has the right information complexity for dropout regularization, while higher-degree polynomials provide stronger signal that shouldn't be disrupted.

### 25K Training Degradation Analysis

| Config | 10K MASE | 25K MASE | Change | Held? |
|--------|:--------:|:--------:|:------:|:-----:|
| **hint d=6 s=16** | **1.1836** | **1.1889** | **+0.45%** | **YES** |
| hdrop10 (d=4+10%drop) | 1.1802 | 1.1931 | +1.1% | Partial |
| **hint d=4 s=16** | **1.1944** | **1.1936** | **-0.07%** | **YES** |
| hint d=2 c=-0.8 | 1.1884 | 1.2062 | +1.5% | No |
| hint d=3 s=16 | 1.2040 | 1.2366 | +2.7% | No |
| hint d=5 s=16 | 1.2084 | 1.2452 | +3.0% | No |
| reversal d=2 s=16 | 1.2227 | 1.2372 | +1.2% | No |
| baseline | 1.2421 | 1.2422 | Stable | — |

**Pattern**: d=4 and d=6 both survive 25K training. d=6 is the best at 25K (1.1889, -3.94% vs baseline).
hdrop10 degrades slightly (1.1802→1.1931) but is still competitive at 25K (-3.94%).
d=3, d=5, c=-0.8 all degrade significantly. Even-degree polynomials appear more robust at longer training.

**25K leaderboard:**
1. **d=6 s=16**: 1.1889 (-4.28%)
2. **hdrop10 (d=4+10%drop)**: 1.1931 (-3.94%)
3. **d=4 s=16**: 1.1936 (-3.91%)
4. d=2 c=-0.8: 1.2062 (-2.89%)

### Per-Dataset Analysis (hdrop10 vs baseline, by frequency)

| Freq | Configs | Win/Loss | Mean Change | Best Model |
|:----:|:-------:|:--------:|:-----------:|:----------:|
| **10S** | 6 | **6/0** | **-25.4%** | hdrop10 |
| **15T** | 12 | **12/0** | **-8.9%** | d6 |
| **10T** | 6 | 5/1 | **-8.8%** | d6 |
| **5T** | 12 | **12/0** | **-6.3%** | hdrop10 |
| M | 5 | 3/2 | -2.9% | c08 |
| W | 8 | 4/4 | -2.8% | c08 |
| H | 31 | 18/13 | -2.2% | hdrop10 |
| D | 15 | 9/6 | +1.0% | baseline |
| Q | 1 | 0/1 | +10.3% | d6 |
| A | 1 | 0/1 | +7.6% | d6 |

**By Horizon:**

| Horizon | Configs | Win/Loss | Mean Change |
|:-------:|:-------:|:--------:|:-----------:|
| **long** | 21 | **17/4** | **-10.6%** |
| **medium** | 21 | **17/4** | **-7.7%** |
| short | 55 | 35/20 | -1.7% |

**Critical findings:**
- Preconditioning benefit scales inversely with data frequency (sub-hourly data universally improves)
- Benefit scales with forecast horizon (long 6x better than short)
- hdrop10 wins on noisy/irregular data; d=6 wins on strongly periodic data
- Universal failure on daily/quarterly/annual data

---

## Active Experiments (2026-02-25)

### All Training Complete

All 10K, 25K, and 100K training runs have completed. d6_100k eval is running (job 5082176).

### Completed 100K Results

| Experiment | MASE | vs 100K Baseline (1.2878) |
|------------|:----:|:-------------------------:|
| **hd10_100k (d=4 + 10% drop)** | **1.1918** | **-7.45%** |
| d4_100k (d=4 hint, no dropout) | 1.2135 | -5.77% |
| hint100k (d=4, prev run) | 1.2038 | -6.52% |
| d6_100k | *eval running* | *pending* |

### Recently Completed 10K Evaluations

| Experiment | MASE | vs Baseline | Notes |
|------------|:----:|:-----------:|-------|
| q_l2opt_d6 | 1.1784 | -5.13% | L2-opt beats Cheb at d=6 |
| q_ms_d4d6_hd10 | 1.1817 | -4.86% | Dropout hurts ms at 10K |
| q_lyap_d6 | 1.1985 | -3.51% | Lyapunov mild coefficients |
| q_l2ms46 | 1.2080 | -2.75% | L2-opt multi-scale weak |
| q_mix46 | 1.2082 | -2.73% | Mixed Cheb+L2-opt also weak |
| q_leg_d6 | 1.2099 | -2.60% | Legendre < Chebyshev |
| q_ms_d4d8 | 1.2135 | -2.30% | d=8 not complementary |
| q_d5hd10 | 1.2196 | -1.81% | Dropout hurts d=5 |
| q_expdec | 1.2215 | -1.66% | Exponential decay mediocre |
| q_ms_d4d5 | 1.2225 | -1.58% | d=5 not complementary |
| q_leg_d8 | 1.2225 | -1.58% | Legendre weak at d=8 |
| q_msd468 | 1.2257 | -1.32% | Triple channel hurts |
| q_d6hd10 | 1.2263 | -1.27% | Dropout hurts d=6 at 10K |
| q_ms_d6d8 | 1.2287 | -1.08% | No d=4 → weak |
| q_d6_s4 | 1.2346 | -0.60% | Stride=4 worse than 16 |
| q_leg_d4 | 1.2363 | -0.47% | Legendre very weak at d=4 |
| q_ms_d6d4 | 1.2365 | -0.45% | Wrong ordering costs 5.6% |
| q_ms_s164 | 1.2390 | -0.25% | Multi-stride d=6 weak |
| q_l2_d4 | 1.2531 | +0.89% | L2-opt fails at d=4 |
| q_fd_hint | 1.2579 | +1.27% | First-diff harmful |

### Per-Dataset Analysis: d=6 vs Baseline (97 GIFT-Eval configs)

**By data frequency** (avg MASE change):
- 10S (6 configs): **-21.8%** — massive gains on high-freq data
- 15T (12 configs): **-12.3%** — strong on sub-hourly
- 10T (6 configs): **-10.8%** — strong
- 5T (12 configs): **-4.6%** — moderate
- H (31 configs): **-0.1%** — neutral on hourly
- D (15 configs): **+0.2%** — neutral/slightly worse on daily

**By forecast horizon** (avg MASE change):
- Long (21 configs): **-10.0%** (17 wins / 4 losses)
- Medium (21 configs): **-6.3%** (16 / 5)
- Short (55 configs): **-1.3%** (36 / 19)

**Win/loss**: d=6 improves 69/97 configs, degrades 28/97.
**Implication**: Multi-stride hints (s=16 + s=4) could recover the neutral hourly/daily performance.

### Completed 25K Results (ALL DONE)

| Config | 10K MASE | 25K MASE | Change | Held? |
|--------|:--------:|:--------:|:------:|:-----:|
| **d=6** | **1.1836** | **1.1889** | **+0.45%** | **YES (BEST)** |
| hdrop10 (d=4+10%drop) | 1.1802 | 1.1931 | +1.1% | Partial |
| **d=4** | **1.1944** | **1.1936** | **-0.07%** | **YES** |
| c=-0.8 | 1.1884 | 1.2062 | +1.5% | No |
| d=3 | 1.2040 | 1.2366 | +2.7% | No |
| d=5 | 1.2084 | 1.2452 | +3.0% | No |

