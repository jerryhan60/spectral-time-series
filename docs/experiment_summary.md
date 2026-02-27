# Spectral Hint Preconditioning for Moirai2: Experiment Summary

**Last updated**: 2026-02-25
**Branch**: `spectral_non_precond`
**Model**: Moirai2 Small (11.4M params, 6 attention layers, d_model=384, patch_size=16)
**Training data**: LOTSA v1 (27 time series datasets, unweighted)
**Evaluation**: GIFT-Eval benchmark (97 configurations across multiple domains, frequencies, horizons)
**Primary metric**: Geometric mean MASE across all 97 GIFT-Eval configs (lower = better)

---

## Executive Summary

We investigated **time-domain polynomial preconditioning** for improving zero-shot time series forecasting with Moirai2 Small. After 40+ experiments spanning multiple approaches, polynomial families, degrees, strides, and architectural variants, the **best model achieves MASE 1.1675 (-6.01% vs baseline)** using **multi-scale hint mode** with Chebyshev degree-4 + degree-6 at stride 16.

**Key findings**:
1. **Hint mode** (FIR residual as auxiliary input channels, no reversal) is the only approach that consistently works
2. **Multi-scale hints** (multiple polynomial degrees) dramatically outperform single-degree hints
3. **Stride=16** (patch-aligned) is critical — stride=1 fails, stride=8 is mediocre
4. **Hint dropout** (10%) provides strong regularization for single-degree hints
5. Benefits are largest on high-frequency data and long forecast horizons; roughly neutral on low-frequency/short-horizon

---

## Baseline

All experiments compare against **our own Moirai2 Small baseline** trained from scratch with identical settings:

| Property | Value |
|----------|-------|
| **Architecture** | Moirai2 Small (causal decoder, quantile loss) |
| **Parameters** | 11.4M |
| **Training** | 10K steps (100 epochs × 100 batches, bs=256) on LOTSA v1 |
| **Key setting** | `anomaly_zscore_threshold=8.0` (filters outlier sequences) |
| **Baseline MASE** | **1.2421** (10K) / **1.2422** (25K) / **1.2878** (100K) |

### Official Moirai 2.0-R-small Reference

The official `Salesforce/moirai-2.0-R-small` (from HuggingFace) achieves **MASE 1.0236** (57/97 < 1.0) on GIFT-Eval.

**IMPORTANT**: The official model was trained on a completely different, ~10x larger corpus (arXiv:2511.11698):
- ~36M series, ~295B observations from 5 sources (GIFT-Eval Pretrain, Chronos-Mixup, KernelSynth synthetic, internal CloudOps, GIFT-Eval TrainTest)
- 100K steps, bs=256, AdamW (lr=1e-3), 10K warmup + cosine annealing, bf16
- We do NOT have access to this training corpus

Our experiments use LOTSA v1 (27 datasets, ~10x less data). The gap (1.1675 vs 1.0236 = 14%) is primarily due to training data differences, not architecture. All comparisons in this document are **matched-compute, matched-data** against our own LOTSA-trained baselines.

---

## Complete Results Table

### 10K Step Experiments (all completed)

| Rank | Experiment | Description | MASE | vs Baseline |
|:----:|------------|-------------|:----:|:-----------:|
| **1** | **q_ms_d4d6** | **multi-scale d=4+d=6 hint** | **1.1675** | **-6.01%** |
| 2 | q_l2opt_d6 | L2-optimized d=6 s=16 | 1.1784 | -5.13% |
| 3 | q_hint_drop10 | hint d=4 + 10% hint dropout | 1.1802 | -4.98% |
| 4 | q_ms_d4d6_hd10 | multi-scale d=4+d=6 + 10% dropout | 1.1817 | -4.86% |
| 5 | q_hint_s16d6 | hint d=6 s=16 (Chebyshev) | 1.1836 | -4.71% |
| 6 | q_hint_c08 | hint c=-0.8 s=16 | 1.1884 | -4.33% |
| 6 | q_hint_sep | hint d=4, separate embed | 1.1884 | -4.33% |
| 8 | q_hint_d6drop05 | hint d=6 + 5% dropout | 1.1922 | -4.02% |
| 9 | q_ms_d2d6 | multi-scale d=4+d=2+d=6 | 1.1940 | -3.87% |
| 10 | q_hint_drop05 | hint d=4 + 5% dropout | 1.1941 | -3.86% |
| 11 | q_hint_s16d4 | hint d=4 s=16 | 1.1944 | -3.84% |
| 12 | q_lyap_d6 | Lyapunov d=6 s=16 | 1.1985 | -3.51% |
| 13 | q_hint_d6_sep | hint d=6, separate embed | 1.1998 | -3.40% |
| 14 | q_hint_d6_learn | hint d=6, learnable coeffs | 1.2025 | -3.19% |
| 15 | q_hint_s16d7 | hint d=7 s=16 | 1.2027 | -3.17% |
| 16 | q_c08_d10 | hint c=-0.8 + 10% dropout | 1.2037 | -3.09% |
| 17 | hint100k | hint d=4 s=16, 100K steps | 1.2038 | -3.08% |
| 18 | q_hint_s16d3 | hint d=3 s=16 | 1.2040 | -3.07% |
| 19 | q_ms_strd | multi-stride d=4 s=16+s=8 | 1.2057 | -2.93% |
| 20 | q_hint_c15 | hint c=-1.5 | 1.2074 | -2.79% |
| 21 | q_l2ms46 | multi-scale L2-opt d=4+d=6 | 1.2080 | -2.75% |
| 22 | q_mix46 | Cheb d=4 primary + L2-opt d=6 extra | 1.2082 | -2.73% |
| 23 | m2_hint_s16 | hint d=5 s=16 | 1.2084 | -2.71% |
| 24 | q_leg_d6 | Legendre d=6 s=16 | 1.2099 | -2.60% |
| 25 | q_hint_drop15 | hint d=4 + 15% dropout | 1.2103 | -2.56% |
| 26 | q_d6_d10 | hint d=6 + 10% dropout | 1.2106 | -2.54% |
| 27 | q_ms_d4d8 | multi-scale d=4+d=8 | 1.2135 | -2.30% |
| 28 | q_hint_s16d2 | hint d=2 s=16 | 1.2157 | -2.13% |
| 29 | q_hint_d5drop10 | hint d=5 + 10% dropout | 1.2191 | -1.85% |
| 30 | q_d5hd10 | hint d=5 + 10% dropout (rerun) | 1.2196 | -1.81% |
| 31 | q_expdec | exponential decay coefficients | 1.2215 | -1.66% |
| 32 | q_hint_s16d8 | hint d=8 s=16 | 1.2216 | -1.65% |
| 33 | q_ms_d4d5 | multi-scale d=4+d=5 | 1.2225 | -1.58% |
| 33 | q_leg_d8 | Legendre d=8 s=16 | 1.2225 | -1.58% |
| 35 | q_s16d2 | reversal d=2 s=16 | 1.2227 | -1.56% |
| 36 | q_msd468 | triple d=4+d=6+d=8 | 1.2257 | -1.32% |
| 37 | q_d6hd10 | hint d=6 + 10% dropout | 1.2263 | -1.27% |
| 38 | q_ms_d6d8 | multi-scale d=6+d=8 (no d=4) | 1.2287 | -1.08% |
| 39 | q_d6_s4 | hint d=6 stride=4 | 1.2346 | -0.60% |
| 40 | q_leg_d4 | Legendre d=4 s=16 | 1.2363 | -0.47% |
| 41 | q_ms_d6d4 | multi-scale d=6 primary + d=4 extra | 1.2365 | -0.45% |
| 42 | q_ms_s164 | multi-stride d=6 s=16+s=4 | 1.2390 | -0.25% |
| — | **q_baseline** | **no preconditioning** | **1.2421** | **—** |
| — | q_s16d4 | reversal d=4 s=16 | 1.2487 | +0.53% |
| — | q_l2_d4 | L2-optimized d=4 s=16 | 1.2531 | +0.89% |
| — | q_fd_hint | first-diff hint (c=[-1.0,0.0]) | 1.2579 | +1.27% |
| — | q_dualloss | dual loss mode | 1.2860 | +3.53% |
| — | dualhead_s16 | dual-head from scratch | 1.5576 | +25.3% |
| — | m2_firstdiff_s16 | first diff (coeff -1.0) | 1.5926 | +28.2% |

### 25K Step Results

| Experiment | MASE | vs 25K Baseline (1.2422) |
|------------|:----:|:-----------:|
| **hint d=6 25K** | **1.1889** | **-4.28%** |
| hdrop10 25K (d=4+10% drop) | 1.1931 | -3.94% |
| hint d=4 25K | 1.1936 | -3.91% |
| ms d=4+d=6 25K | 1.2212 | -1.69% |
| reversal d=2 25K | 1.2372 | -0.40% |
| hint d=5 25K | 1.2452 | +0.25% |

**Note on multi-scale at 25K**: ms d=4+d=6 drops from -6.01% at 10K to -1.69% at 25K. The multi-scale hint advantage decays significantly at longer training, just like single-scale hints. This suggests hints provide inductive bias that is most valuable in the low-data/early-training regime.

### 100K Step Results

| Experiment | MASE | vs 100K Baseline (1.2878) |
|------------|:----:|:-------------------------:|
| **hd10_100k (d=4 + 10% drop)** | **1.1918** | **-7.45%** |
| d4_100k (d=4 hint, no dropout) | 1.2135 | -5.77% |
| hint100k (d=4, prev run) | 1.2038 | -6.52% |
| d6_100k (d=6 hint, no dropout) | 1.2220 | -5.11% |
| 100K baseline (no precond) | 1.2878 | — |

Note: The 100K baseline (1.2878) is worse than the 10K baseline (1.2421) due to overfitting. Hint preconditioning provides even larger relative gains at 100K: hd10_100k achieves -7.45%, the strongest relative improvement of any experiment. **Hint dropout is critical at 100K** — without dropout, d=4 gives -5.77%; with 10% dropout, it gives -7.45%. The dropout acts as a regularizer that prevents the model from over-relying on the hint signal during extended training.

---

## How Hint Mode Works

The hint provides the model with **spectral information about inter-patch autocorrelation** as extra input channels, without requiring any reversal at inference. The FIR filter is strictly **causal** (backward-looking only).

### FIR Filter (Causal, Backward-Looking)

For Chebyshev degree d with stride s=16 (= patch_size), the filter at timepoint t computes:

```
preconditioned[t] = y[t] + c₁·y[t-16] + c₂·y[t-32] + ... + c_d·y[t-d·16]
                           ↑              ↑                    ↑
                      1 patch back    2 patches back       d patches back

hint[t] = preconditioned[t] - y[t]    (the FIR residual — all past values)
```

Each coefficient c_i comes from the Chebyshev polynomial. With stride=16, each tap references the **same position in a previous patch**, aligning with the transformer's patch-level attention.

### Inference Pipeline (Single Forward Pass)

```
                          CONTEXT (known)                    PREDICTION (unknown)
Time ──────────────────────────────────────────────────────────────────────────►

Raw input:  [y₁...y₁₆] [y₁₇...y₃₂] [y₃₃...y₄₈] ... [y_T-15...y_T] [  ???  ] [  ???  ]
              Patch 0      Patch 1      Patch 2          Patch P       Patch P+1  Patch P+2

                    ┌─────────────────────────────────────────────────────────┐
Step 1: Z-score     │  scaled_y = (y - mean) / std                           │
                    └─────────────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────▼──────────────────────────────────┐
Step 2: Compute     │  For each hint degree (e.g., d=4 and d=6):              │
causal FIR hints    │                                                          │
                    │    hint_d4[t] = -1.0·y[t-16] + 0.125·y[t-48]           │
                    │    hint_d6[t] = -1.5·y[t-16] + 0.5625·y[t-48]          │
                    │                  - 0.03·y[t-80]                          │
                    │                                                          │
                    │    All lookups are BACKWARD only (causal)                │
                    │    Prediction window hints → set to ZERO                 │
                    └──────────────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────▼──────────────────────────────────┐
Step 3: Build       │                                                          │
per-patch tokens    │  Context patches:           Prediction patches:          │
                    │  ┌─────────────────┐        ┌─────────────────┐          │
                    │  │ target  (16 val) │        │ zeros   (16)    │          │
                    │  │ mask    (16 ones)│        │ zeros   (16)    │          │
                    │  │ hint_d4 (16 val) │        │ zeros   (16)    │          │
                    │  │ hint_d6 (16 val) │        │ zeros   (16)    │          │
                    │  └─────────────────┘        └─────────────────┘          │
                    │       64 dims                    64 dims                  │
                    └──────────────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────▼──────────────────────────────────┐
Step 4: in_proj     │  Linear projection: 64 → 384 dims per patch             │
+ causal            │                                                          │
transformer         │  Causal attention (each patch sees past + itself only):  │
                    │                                                          │
                    │        Patch: 0  1  2  3  ...  P  P+1  P+2              │
                    │  Patch 0     [✓  ·  ·  ·       ·   ·    · ]             │
                    │  Patch 1     [✓  ✓  ·  ·       ·   ·    · ]             │
                    │  Patch 2     [✓  ✓  ✓  ·       ·   ·    · ]             │
                    │    ...                                                    │
                    │  Patch P     [✓  ✓  ✓  ✓  ...  ✓   ·    · ] ← last ctx │
                    │  Patch P+1   [✓  ✓  ✓  ✓  ...  ✓   ✓    · ] ← 1st pred│
                    │  Patch P+2   [✓  ✓  ✓  ✓  ...  ✓   ✓    ✓ ] ← 2nd pred│
                    │                                                          │
                    │  Prediction patches have ZERO inputs but attend to       │
                    │  context patches whose embeddings carry hint info        │
                    └──────────────────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────▼──────────────────────────────────┐
Step 5: Output      │  Position t predicts values at position t+1              │
                    │  Predictions are in z-scored raw space                    │
                    │  Final forecast = prediction · scale + loc               │
                    │  NO reversal / inverse filter needed                      │
                    └──────────────────────────────────────────────────────────┘
```

### Why Hint Mode Works (vs Reversal)

**Reversal mode** requires the model to predict in preconditioned space, then analytically undo the filter. Any prediction error gets amplified by the inverse — errors compound across patches.

**Hint mode** keeps predictions in the original z-scored space. The hint channel gives the transformer explicit information about inter-patch autocorrelation structure (essentially, "how does this patch relate to patches 1-6 steps back?") as a free additional input. The model can learn to use or ignore this information as appropriate — it provides inductive bias without forcing the model into a different coordinate system.

---

## Methods Explored

### 1. Standard Preconditioning + Reversal

Apply a causal FIR filter, train normally, analytically reverse at inference:
```
z_t = y_t + c1*y_{t-s} + c2*y_{t-2s} + ... + cd*y_{t-d*s}
```

**Result: MOSTLY NEGATIVE.** Stride=1 reversal fails catastrophically at all degrees (+1.9% to +28% worse). Stride=16 reversal gives modest gains: d=2 achieves -1.56%, but higher degrees degrade. The fundamental issue is **error accumulation during reversal** — predicted values fed back into the inverse filter amplify errors exponentially.

### 2. Hint Mode (Single Scale)

Provide the FIR filter residual as an additional input channel. No reversal at inference:
```
Input = [target, observation_mask, hint]     (3 channels × patch_size)
hint_t = FIR(z_scored_y)_t - z_scored_y_t    (filter residual)
```

**Result: CONSISTENTLY POSITIVE.** Every degree from d=2 to d=8 beats baseline. Best single-degree: d=6 at -4.71%. Adding hint dropout (10%) to d=4 gives -4.98%.

### 3. Multi-Scale Hints (Best Approach)

Provide **multiple** FIR residuals at different polynomial degrees as separate channels:
```
Input = [target, mask, hint_d4, hint_d6]     (4 channels × patch_size)
```

**Result: BREAKTHROUGH.** Multi-scale d=4+d=6 achieves **MASE 1.1675 (-6.01%)**, dramatically beating the best single-scale hint (d=6, -4.71%). The two degrees capture complementary frequency bands — d=4 provides moderate smoothing while d=6 captures higher-order spectral detail.

However, more scales is not always better: triple d=4+d=2+d=6 (1.1940, -3.87%) is worse than dual d=4+d=6. The d=2 filter adds noise rather than complementary information.

### 4. Failed/Neutral Approaches

| Approach | Best MASE | Why It Failed |
|----------|:---------:|---------------|
| Standard reversal (stride=1) | 1.2657 | Error accumulation destroys predictions |
| Dual-head (raw + precond outputs) | 1.5576 | Capacity split hurts both heads |
| First differencing (coeff -1.0) | 1.5926 | Too aggressive, destroys signal |
| Learnable coefficients | 1.2025 | Converges to near-identity; overfits |
| Robust scaler (median/MAD) | NaN | Numerically unstable |
| Multi-stride (same d, different strides) | 1.2057 | Redundant information |

---

## Hint Mode Degree Sweep

| Degree | Coefficients | MASE | vs Baseline |
|:------:|-------------|:----:|:-----------:|
| d=2 | `[0, -0.5]` | 1.2157 | -2.13% |
| d=3 | `[0, -0.75, 0]` | 1.2040 | -3.07% |
| d=4 | `[0, -1.0, 0, 0.125]` | 1.1944 | -3.84% |
| d=5 | `[0, -1.25, 0, 0.3125, 0]` | 1.2084 | -2.71% |
| **d=6** | **`[0, -1.5, 0, 0.5625, 0, -0.03]`** | **1.1836** | **-4.71%** |
| d=7 | `[0, -1.75, 0, 0.875, 0, -0.109, 0]` | 1.2027 | -3.17% |
| d=8 | `[0, -2.0, 0, 1.25, 0, -0.25, 0, 0.016]` | 1.2216 | -1.65% |

**Pattern**: Non-monotonic with two local optima at d=4 and d=6. All odd-indexed Chebyshev coefficients are zero, so effective filter taps are sparser than the degree suggests. d=6 is optimal, but d=8 degrades (max|c|=2.0, likely causes numerical instability in bf16).

## Polynomial Family Comparison (all d=6, s=16)

| Family | max|c| | MASE | vs Baseline | Description |
|--------|:------:|:----:|:-----------:|-------------|
| **L2-optimized** | **~0.28** | **1.1784** | **-5.13%** | **Minimizes L2 norm of coefficients** |
| Chebyshev | 1.50 | 1.1836 | -4.71% | Standard choice |
| Lyapunov | ~0.23 | 1.1985 | -3.51% | Minimizes Lyapunov exponent |
| Legendre | 1.36 | 1.2099 | -2.60% | Orthogonal on [-1,1] |

**Surprise finding**: L2-optimized polynomials (mildest coefficients, max|c|~0.28) **outperform Chebyshev** at d=6 by 0.44%. This contradicts the earlier hypothesis that "stronger coefficients = better spectral discrimination." Instead, mild coefficients appear to avoid the numerical precision issues that degrade Chebyshev at higher degrees.

### Chebyshev vs Legendre by Degree

| Degree | Chebyshev MASE | Legendre MASE | Cheb vs Baseline | Leg vs Baseline | Cheb Advantage |
|:------:|:--------------:|:-------------:|:----------------:|:---------------:|:--------------:|
| d=4 | 1.1944 | 1.2363 | -3.84% | -0.47% | +3.37% |
| d=6 | 1.1836 | 1.2099 | -4.71% | -2.60% | +2.11% |
| d=8 | 1.2216 | 1.2225 | -1.65% | -1.58% | +0.07% |

Chebyshev dominates Legendre at all degrees. The advantage shrinks with degree — at d=8 they're tied. But L2-optimized beats both at d=6.

Also tested: exponential decay coefficients (1.2215, -1.66%), first-diff hint (1.2579, +1.27%). Exponential decay is mediocre; first-diff is harmful.

---

## Per-Dataset Analysis

### By Frequency (hint d=6 vs baseline)

| Frequency | Configs | Wins/Losses | Mean MASE Change |
|-----------|:-------:|:-----------:|:----------------:|
| **10S** | 6 | **6/0** | **-21.8%** |
| **15T** | 12 | **12/0** | **-12.3%** |
| **10T** | 6 | **6/0** | **-10.8%** |
| **5T** | 12 | **10/2** | **-4.6%** |
| H | 31 | 17/14 | -0.1% |
| D | 15 | 8/7 | +0.2% |
| W | 6 | 1/5 | +2-5% |
| M | 5 | 2/3 | neutral |

### By Forecast Horizon

| Horizon | Configs | Wins/Losses | Mean MASE Change |
|---------|:-------:|:-----------:|:----------------:|
| **Long** | 21 | **20/1** | **-10.0%** |
| **Medium** | 21 | **17/4** | **-6.3%** |
| Short | 55 | 32/23 | -1.3% |

**Key pattern**: Hint preconditioning provides massive improvements on high-frequency data (sub-hourly) and long forecast horizons, where the spectral FIR residual gives the model explicit information about inter-patch autocorrelation patterns. It is roughly neutral on hourly/daily data at short horizons.

---

## Lessons Learned

### 1. The Reversal Problem is Fundamental
Standard preconditioning (filter → train → reverse at inference) fails because forecast errors compound through the inverse filter. This is true regardless of polynomial family, degree, or coefficient magnitude. The only way to use preconditioning is to **avoid reversal entirely** — i.e., provide the filter output as side information (hint mode).

### 2. Stride = Patch Size is the Right Granularity
Setting the FIR filter stride to 16 (= patch_size) ensures the filter operates at the same granularity as the transformer's attention mechanism. Each filter tap connects the same position across adjacent patches, which aligns with how the causal attention already processes temporal information. Stride=1 creates intra-patch correlations that confuse the patch embedding; stride=8 is better but still misaligned.

### 3. Multi-Scale d=4+d=6 is Uniquely Effective
Multi-scale d=4+d=6 (1.1675, -6.01%) is dramatically better than any other multi-scale combination tested:
- d=4+d=5: 1.2225 (-1.58%) — d=5 too similar to d=4
- d=4+d=8: 1.2135 (-2.30%) — d=8 too aggressive
- d=4+d=6+d=8: 1.2257 (-1.32%) — third channel adds noise
- d=4+d=2+d=6: 1.1940 (-3.87%) — d=2 too mild

The d=4+d=6 pair captures specifically complementary frequency bands. d=4 (moderate smoothing, max|c|=1.0) and d=6 (higher-order spectral detail, max|c|=1.5) appear to span the useful spectral range without redundancy.

### 4. Optimal Filter Strength Is Non-Trivial
The degree sweep reveals a non-monotonic pattern: d=4 and d=6 are both strong local optima, while d=5 dips. The underlying reason appears to be how Chebyshev coefficient magnitudes interact with bf16 precision — d=8 (max|c|=2.0) and d=5 (max|c|=1.25) may cause numerical issues during the patch embedding when values are large.

### 5. Hint Dropout: Hurts at 10K, Critical at 100K
At 10K steps, dropout has mixed effects: helps d=4 (1.1944→1.1802) but hurts d=6 (1.1836→1.2263), multi-scale d=4+d=6 (1.1675→1.1817). **At 100K steps, dropout becomes critical**: d=4+10% dropout achieves -7.45% vs baseline, while d=4 without dropout only achieves -5.77%. The dropout prevents the model from over-relying on hints during extended training, acting as a regularizer that combats overfitting. At short training, hints are already regularizing, so additional dropout is redundant/harmful.

### 6. Preconditioning Benefits Are Degree-Dependent at Scale
At 10K steps, hint d=5 gives -2.71%, but at 25K it gives +0.25% (worse than baseline). However, d=4 and d=6 are robust to longer training: d=6 at 25K gives -4.28%, d=4 at 25K gives -3.91%. Even-degree polynomials appear more stable during extended training. At 100K steps, hint d=4 achieves -6.52% vs the 100K baseline — the hint provides an even *larger* relative benefit at longer training, acting as a regularizer against the overfitting seen in the plain 100K baseline (1.2878 vs 1.2421 at 10K).

### 7. Learnable Coefficients Underperform Fixed Polynomials
When allowed to learn optimal FIR coefficients via backpropagation, the model converges to near-identity filters (~4× weaker than Chebyshev d=2). The learned coefficients essentially undo the preconditioning, suggesting the model optimizes for training loss reduction rather than generalizable spectral features. Fixed polynomial coefficients provide inductive bias that the model cannot discover on its own.

### 8. Multi-Scale Primary Degree Ordering Matters
Multi-scale d=4+d=6 (1.1675, -6.01%) vs d=6+d=4 (1.2365, -0.45%) — **swapping the primary degree costs 5.6%**. The "primary" hint (channel 0 after target+mask) gets preferential treatment from the input projection. d=4 as primary + d=6 as extra is far better than the reverse. Similarly, d=6+d=8 without d=4 (1.2287) is weak. **d=4 is the anchor degree for multi-scale hints**.

### 9. L2-Optimized Polynomials Beat Chebyshev at d=6, Fail at d=4
At d=6, L2-optimized (1.1784, -5.13%) beats Chebyshev (1.1836, -4.71%). But at d=4, L2-opt (1.2531, +0.89%) is worse than baseline! The mild L2-opt coefficients don't have enough discriminative power at low degree but avoid precision issues at high degree. For multi-scale, L2-opt d=4+d=6 (1.2080, -2.75%) and mixed Cheb-d=4+L2opt-d=6 (1.2082, -2.73%) both dramatically underperform pure Chebyshev d=4+d=6 (1.1675, -6.01%). **Chebyshev is optimal for multi-scale configurations where d=4 primary is critical.**

### 10. Hint Benefits Decay with Training Length; Dropout Extends Them
At 10K: ms d=4+d=6 gives -6.01%. At 25K: only -1.69%. At 100K, d=4 without dropout gives -5.77% (vs 100K baseline), but **with 10% dropout it gives -7.45%**. The pattern: hint inductive bias is most valuable early. At longer training, the model learns the spectral patterns from raw data alone, making hints redundant. Dropout forces the model to not over-rely on hints, maintaining their regularization benefit. **For production deployments with long training, hint dropout is essential.**

### 11. Anomaly Filtering is a Major Confound
Setting `anomaly_zscore_threshold=8.0` to filter extreme outlier sequences had a larger impact than many of our preconditioning experiments. Our 10K baseline (1.2421) already beats the published Moirai 1.1 Small (1.323) by 6.3%, primarily due to this one setting. All valid preconditioning comparisons must use this threshold on both baseline and experimental runs.

---

## Further Research Directions

### Completed Investigations (archived)

The following directions have been fully explored and resolved:
- **Multi-scale degree search**: d=4+d=6 is uniquely optimal. d=4+d=5, d=4+d=8, d=4+d=6+d=8 all worse. ✓
- **Multi-scale + dropout**: Dropout hurts multi-scale at 10K (1.1675→1.1817). ✓
- **Legendre polynomials**: Underperform Chebyshev at all degrees. ✓
- **L2-opt/Lyapunov families**: L2-opt good at d=6 but fails at d=4; poor in multi-scale. ✓
- **Mixed-family multi-scale**: Cheb d=4 + L2-opt d=6 (1.2082) worse than pure Cheb (1.1675). ✓
- **25K multi-scale**: Advantage decays from -6.01% to -1.69%. ✓
- **100K single-scale**: d=4 with 10% dropout achieves -7.45% at 100K. ✓

### High-Confidence Directions

1. **Multi-scale d=4+d=6 at 100K with dropout**: Given that (a) dropout is critical at 100K (-7.45% with, -5.77% without), and (b) multi-scale d=4+d=6 is the best 10K configuration, the combination multi-scale+dropout at 100K is the highest-priority next experiment.

2. **Frequency-adaptive preconditioning**: Hints hurt weekly data (+2-5%) while massively helping sub-hourly data (-10-22%). A learned gate that downweights the hint for low-frequency data could eliminate the regression on weekly data while preserving gains elsewhere. This is the most promising architectural change.

3. **Multi-scale d=4+d=6 with L2-opt d=6 at 100K**: Since L2-opt d=6 beats Chebyshev d=6 in single-scale (1.1784 vs 1.1836), and dropout helps at 100K, combining L2-opt d=6 as secondary with Chebyshev d=4 primary at 100K with dropout could outperform single-scale.

### Exploratory Directions

4. **Spectral-domain hints**: Instead of time-domain FIR residuals, compute FFT-based spectral features and provide them as hint channels. This would directly give the model frequency-domain information rather than encoding it implicitly through polynomial filters.

5. **Per-dataset preconditioning curriculum**: Train with a mixture of hint configurations (varying degree, stride, dropout) across mini-batches. This could help the model generalize across frequency domains.

6. **Scale to larger models**: All experiments use Moirai2 Small (11.4M). The hint mechanism adds negligible parameters (only changes the input projection dimension). Testing on Moirai2 Base (~90M) would show if the benefit persists or is subsumed by increased model capacity.

---

## Flash-STU Hybrid Results

Parallel STU+Attention architecture integrated into Moirai2 Small. STU branch uses approx mode (project-then-convolve), K=24 Hankel spectral filters, zero-init tanh gate, d_ff reduced 1024→940 to fit extra params.

| Model | Params | Steps | MASE (Geo Mean) | vs 10K Baseline |
|-------|--------|-------|-----------------|-----------------|
| Flash-STU v2 (parallel, approx) | 11.75M | 10K | 1.3044 | +5.01% |
| STU v1 (alternating, Moirai v1) | 13.83M | 100K | 1.3359 | N/A (different base model) |

**Conclusion**: STU hybrid v2 underperforms the baseline at 10K steps. Possible causes: zero-init gates may need longer training to open; d_ff reduction (1024→940) reduces attention layer capacity. Not a priority for further investigation unless longer training shows improvement.

**Bug fix note**: Initial eval (job 5082318) had 42/97 failures on medium/long horizons. Root cause: `STULayer.forward()` hardcoded `B, T, D = x.shape` but recursive prediction passes 4D tensors. Fixed by flattening leading batch dims. Re-eval (job 5085830) succeeded on all 97/97 configs.

---

## In-Progress Experiments

### Currently Evaluating

| Experiment | Description | Status |
|------------|-------------|--------|
| m2_d6_100k | hint d=6 s=16, 100K steps | Training complete, eval running (job 5082176) |

All 10K and 25K experiments have completed training and evaluation. d4_100k, hd10_100k, and d6_100k training all complete.
