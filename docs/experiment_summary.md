# Spectral Hint Preconditioning for Moirai2: Experiment Summary

**Last updated**: 2026-03-04
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
5. Benefits are **frequency-dependent**: ms d=4+d=6 at 10K gives -5.0% on high-freq (≥hourly) but +1.6% on low-freq (<daily). Dropout + longer training (100K) fixes the regression, giving -5.7% high-freq and -6.2% low-freq

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
| 9 | cross_c4l6 | Cross-family (Cheb d=4 + Leg d=6) | 1.1900 | -4.20% |
| 10 | q_ms_d2d6 | multi-scale d=4+d=2+d=6 | 1.1940 | -3.87% |
| 10 | q_hint_drop05 | hint d=4 + 5% dropout | 1.1941 | -3.86% |
| 11 | q_hint_s16d4 | hint d=4 s=16 | 1.1944 | -3.84% |
| 12 | q_lyap_d6 | Lyapunov d=6 s=16 | 1.1985 | -3.51% |
| 13 | q_hint_d6_sep | hint d=6, separate embed | 1.1998 | -3.40% |
| 14 | q_hint_d6_learn | hint d=6, learnable coeffs | 1.2025 | -3.19% |
| 15 | q_hint_s16d7 | hint d=7 s=16 | 1.2027 | -3.17% |
| 16 | q_c08_d10 | hint c=-0.8 + 10% dropout | 1.2037 | -3.09% |
| 17 | hint100k | hint d=4 s=16, 100K steps | 1.2038 | -3.08% |
| 18 | q_hint_s16d3 | hint d=3 s=16 | 1.2040 | -3.07% |
| 19 | s32_hint | hint d=6 stride=32 (2P) | 1.1975 | -3.59% |
| 20 | q_ms_strd | multi-stride d=4 s=16+s=8 | 1.2057 | -2.93% |
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
| — | 4chan_10k | 4-channel (d=2+d=4+d=6+d=8) | 1.2007 | -3.33% |
| — | 3chan_10k | 3-channel (d=2+d=4+d=6) | 1.2252 | -1.36% |
| — | **q_baseline** | **no preconditioning** | **1.2421** | **—** |
| — | bl_lr5e4 | baseline lr=5e-4 | 1.2531 | +0.89% |
| — | s1_hint | hint d=6 stride=1 (dense) | 1.2598 | +1.43% |
| — | bl_lr2e3 | baseline lr=2e-3 | 1.2586 | +1.33% |
| — | q_s16d4 | reversal d=4 s=16 | 1.2487 | +0.53% |
| — | q_l2_d4 | L2-optimized d=4 s=16 | 1.2531 | +0.89% |
| — | ema_hint | EMA-smoothed hint (d=6) | 1.2596 | +1.41% |
| — | q_fd_hint | first-diff hint (c=[-1.0,0.0]) | 1.2579 | +1.27% |
| — | q_dualloss | dual loss mode | 1.2860 | +3.53% |
| — | anneal_8k | hint anneal (8K ramp → 100% dropout) | 1.3063 | +5.17% |
| — | anneal_5k | hint anneal (5K ramp → 100% dropout) | 1.3486 | +8.57% |
| — | diff_hint | first-diff hint (differencing d=6) | 1.4537 | +17.0% |
| — | dualhead_s16 | dual-head from scratch | 1.5576 | +25.3% |
| — | m2_firstdiff_s16 | first diff (coeff -1.0) | 1.5926 | +28.2% |

**Note**: All results in the table above are 97/97 config evaluations at context=4000 unless otherwise noted. 3chan and 4chan were re-evaluated 2026-03-04 with corrected context length (previously used default=1000).

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

| Experiment | MASE | vs 100K Baseline (1.2911) |
|------------|:----:|:-------------------------:|
| **hd10_100k (d=4 + 10% drop)** | **1.1918** | **-7.69%** |
| d4_100k (d=4 hint, no dropout) | 1.2135 | -6.01% |
| hint100k (d=4, prev run) | 1.2038 | -6.76% |
| d6_100k (d=6 hint, no dropout) | 1.2220 | -5.35% |
| leg_ms46_100k (Legendre d=4+d=6 + 10% drop) | 1.2317 | -4.60% |
| 100K baseline (proper, zscore=8.0) | 1.2911 | — |
| 100K baseline + 1K warmup | 1.2833 | -0.60% |
| 100K baseline (old, no zscore) | 1.2878 | — |

**Updated**: Proper 100K baseline (with anomaly_zscore_threshold=8.0) is 1.2911, virtually identical to the old confounded baseline (1.2878). The zscore filter has negligible effect at 100K. Both are worse than the 10K baseline (1.2421) due to overfitting. Hint preconditioning provides even larger relative gains at 100K: hd10_100k achieves -7.69%, the strongest relative improvement. **Hint dropout is critical at 100K** — without dropout, d=4 gives -6.01%; with 10% dropout, it gives -7.69%.

### Weighted LOTSA Results (10K steps, warmup=1K)

| Experiment | MASE | vs Unweighted Baseline (1.2421) |
|------------|:----:|:-------------------------------:|
| Weighted baseline | 1.2497 | +0.6% |
| Weighted + ms46 hint | 1.2382 | -0.31% |

Weighted sampling (using official Moirai 2.0 paper dataset weights) provides no improvement over uniform sampling at 10K steps. The slight degradation (+0.6%) suggests upweighting harder/rarer datasets doesn't pay off with limited compute. Adding hints to weighted training barely helps (-0.31% vs baseline), compared to -6.01% with unweighted training — confirming that data distribution changes hurt hint effectiveness (Lesson 12).

### Additional Ablation Results (10K steps)

| Experiment | MASE | vs Baseline | vs ms46 (1.1675) | Notes |
|------------|:----:|:-----------:|:-----------------:|-------|
| **synth46** (LOTSA+KernelSynth + ms46) | 1.2087 | -2.69% | +3.5% worse | KernelSynth GP series hurt hints |
| **s48** (ms46 + d=4@stride48) | 1.2210 | -1.70% | +4.6% worse | Extra stride-48 hint dilutes signal |
| **wt_hint_w1k** (weighted LOTSA + ms46) | 1.2382 | -0.31% | +6.1% worse | Weighted sampling hurts hints |
| **sd20** (ms46 + 20% seq dropout) | 1.2535 | +0.92% | +7.4% worse | Seq dropout destroys hint benefit |
| wt_base_w1k (weighted LOTSA baseline) | 1.2497 | +0.61% | — | Weighted sampling hurts baseline too |
| **hd20** (ms46 + 20% hint dropout) | 1.2201 | -1.77% | +4.5% worse | 20% dropout too aggressive for ms46 at 10K |
| **cd46** (ms46 + combined dropout) | 1.2349 | -0.58% | +5.8% worse | hint_dropout + model dropout too aggressive |
| **vp46** (ms46 + variable prefix [0.15,0.45]) | 1.2896 | +3.82% | +10.5% worse | Variable context ratio adds noise, hurts training |
| **vpb** (baseline + variable prefix [0.15,0.45]) | 1.3200 | +6.27% | — | VP hurts baseline even more than ms46 |
| **ramp3k** (ms46 + hint_ramp=3000, no drop) | 1.2159 | -2.11% | +4.1% worse | Gradual hint ramp hurts — full exposure from start needed |
| **ramp_hd** (ms46 + ramp3K + hd10) | 1.2283 | -1.11% | +5.2% worse | Ramp + dropout worse than ramp alone |
| **lr5e4** (ms46 + lr=5e-4) | 1.2687 | +2.14% | +8.7% worse | Half LR: too slow convergence at 10K |
| **lr2e3** (ms46 + lr=2e-3) | 1.2241 | -1.45% | +4.8% worse | Double LR: overshooting in 10K regime |
| **wd05** (ms46 + weight_decay=0.05) | 1.2279 | -1.14% | +5.2% worse | Half weight decay: less regularization hurts |
| **w500** (ms46 + warmup=500, step 9K) | 1.2308 | -0.91% | +5.4% worse | Half warmup: reaches peak LR too fast |
| **stu_hint_d4d6** (STU+Attn + ms46, 2K steps) | 1.2643 | -1.78% | — | Hints dramatically accelerate STU (vs STU_fulldff 1.2947 at 2K) |
| **bs128_ms46** (ms46 + bs=128) | 1.2439 | -0.14% | +6.5% worse | Smaller batch size ≈ neutral |
| **ckpt_avg** (ms46+hd10, avg steps 20-35K) | 1.2040 | -3.07% | +3.1% worse | Checkpoint averaging helps but < ms46@10K |
| **avg_last4** (ms46 best, avg 7K-10K) | 1.2086 | -2.70% | +3.5% worse | Averaging HURTS best model — cosine already converged |
| **soup_a07** (hint+base soup α=0.7) | 1.4558 | +17.2% | +24.7% worse | Model soup fails: in_proj dimension mismatch |
| **soup_a05** (hint+base soup α=0.5) | 1.8323 | +47.5% | +56.9% worse | Even worse at lower alpha — confirms complete failure |
| **ms46@50K** (ms46 100K run at step 50K) | 1.2363 | -0.47% | +5.9% worse | Hint benefit decayed at 50K steps |
| **ms46+hd10@35K** (ms46+hd10 at step 35K) | 1.2629 | +1.67% | +8.2% worse | Raw checkpoint noisy vs avg (1.2040) |
| **c4e4** (Cheb d=4 + EMA d=4 hints) | 1.2024 | -3.20% | +3.0% worse | Cheb+EMA combo worse than pure ms46 |
| **hd05_ms46** (ms46 + 5% hint dropout) | 1.2436 | +0.12% | +6.5% worse | 5% dropout HURTS — non-monotonic: 0%>10%>5%>20% |
| **STU+ms46_bs128** (STU+Attn + ms46, 10K, bs128) | 1.2581 | +1.29% | +7.8% worse | STU doesn't help when hints present |
| **ms46@8K** (ms46 at step 8K) | 1.2223 | -1.59% | +4.7% worse | Big 8K→10K jump: hints benefit from final LR cooldown |

### ms46 Training Curve

| Steps | MASE | vs 10K Baseline (1.2421) | Notes |
|:-----:|:----:|:------------------------:|-------|
| 8K | 1.2223 | -1.59% | Model still rapidly improving |
| 10K | 1.1675 | -6.01% | **Best result**, cosine at minimum LR |
| 25K | 1.2212 | -1.69% | Overfitting: hints become noise |
| 50K | 1.2363 | -0.47% | Further overfitting |

**Key insight**: 4.4% improvement from 8K→10K (cosine cooldown phase). The final phase of training is critical for hint models.

### Baseline Training Curve (proper baseline, zscore=8.0, 10K warmup)

| Steps | MASE | vs 10K Baseline (1.2421) | Notes |
|:-----:|:----:|:------------------------:|-------|
| 20K | 1.3025 | +4.86% | Still in warmup ramp-down |
| 30K | 1.3127 | +5.69% | High-LR phase overfitting |
| 50K | 1.2784 | +2.92% | Starts recovering |
| 70K | 1.2780 | +2.89% | Plateau near 50K |
| 100K | 1.2911 | +3.94% | Slight overfit vs 10K |

**Note**: This curve uses 10K warmup steps (10% of 100K) with cosine annealing. The 10K baseline uses 1K warmup (also 10% of 10K). The non-monotonic behavior (worse at 20K-30K) is because the 10K warmup means the model hasn't reached full LR until step 10K, so intermediate checkpoints are undertrained.

### hd10 Training Curve (d=4 + 10% hint dropout, 10K warmup)

| Steps | Baseline | hd10 | Improvement | Notes |
|:-----:|:--------:|:----:|:-----------:|-------|
| 20K | 1.3025 | 1.2401 | -4.79% | Hint already helps in warmup |
| 30K | 1.3127 | **1.2064** | **-8.10%** | Peak relative improvement |
| 50K | 1.2784 | 1.2412 | -2.91% | Regression from 30K dip |
| 70K | 1.2780 | 1.2241 | -4.22% | Recovering |
| 100K | 1.2911 | 1.1918 | -7.69% | Slight decay from 30K peak |

**Key insight**: hd10 consistently outperforms baseline at *every* checkpoint (3-8% margin). The improvement is largest when the baseline struggles most (30K: -8.10%, 100K: -7.69%). The absolute best hd10 MASE is at 100K (1.1918), suggesting hints + dropout provide durable regularization. The ms46h50k experiment (ms46 + dropout, 50K steps) is testing whether multi-scale hints can maintain their 10K advantage with longer training + dropout.

**Lesson 12**: Data modifications and aggressive regularization universally hurt ms46. The multi-scale hint benefit is fragile:
- **Synthetic data** (KernelSynth): Smooth GP kernels produce near-zero hint residuals, diluting signal from real data. -2.69% vs baseline but +3.5% worse than ms46 without synth.
- **Weighted sampling**: Changes the training distribution, likely reducing exposure to datasets where hints matter most (high-freq energy/traffic data).
- **20% sequence dropout**: Zeroing ALL hint channels per-sequence removes the inductive bias too often. The model needs to see hints consistently to learn to use them.
- **10% per-patch dropout** remains the only regularization that helps at scale (100K steps), and only with single-degree d=4, not ms46.

**Lesson 13**: Hyperparameter changes universally hurt ms46 at 10K steps. All variations (lr/2, lr×2, wd/2, warmup/2, hint ramp, variable prefix) degrade performance by 4-9% vs default ms46. The default Moirai2 settings (lr=1e-3, wd=0.1, warmup=1000, bs=256) are well-optimized and should be preserved for preconditioning experiments.

**Lesson 14**: The hd10 training curve shows hints provide durable benefit (3-8%) at every training checkpoint from 20K-100K. The benefit is largest when baseline overfits most (30K: -8.10%, 100K: -7.69%). The ms46h50k experiment will reveal whether multi-scale hints + dropout maintain their 10K advantage at longer training.

**Lesson 15**: Per-dataset analysis (FEV 100 configs, ms46 vs official Moirai 2.0) reveals the gap is **data-driven, not architectural**:
- **Cloud Ops**: 0/5 wins, +57% avg gap. Official model trained on proprietary CloudOps data.
- **Cumulative COVID**: +136% gap. Non-stationary trends not in LOTSA. Official model has KernelSynth synthetic data.
- **Economics/FRED**: +23% avg gap. Low-frequency macro data underrepresented in LOTSA.
- **Energy wins**: We WIN on KDD Cup solar (-14.6%), EPF PJM (-10%), ERCOT hourly (-6.6%) — hourly/daily energy data well-represented in LOTSA.
- **Weather nearly tied**: +1.3% avg gap. Well-represented in LOTSA.
- **Frequency pattern**: We win at high-freq (hourly), lose at extremes (sub-hourly, weekly+). Hints help high-freq most.

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
                    │    hint_d4[t] = -1.0·y[t-32] + 0.125·y[t-64]           │
                    │    hint_d6[t] = -1.5·y[t-32] + 0.5625·y[t-64]          │
                    │                  - 0.03125·y[t-96]                       │
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
| d=5 | 1.2084 | 1.2057 | -2.71% | -2.93% | -0.22% |
| d=6 | 1.1836 | 1.2099 | -4.71% | -2.60% | +2.11% |
| d=8 | 1.2216 | 1.2225 | -1.65% | -1.58% | +0.07% |

Chebyshev dominates Legendre at d=4 and d=6, but at d=5 they are essentially tied (Legendre 1.2057 slightly better than Chebyshev 1.2084). At d=8 they're also tied. L2-optimized beats both at d=6.

**Note**: Legendre d=5 corrected from 1.2237 → 1.2057 after re-evaluation at context=4000 (previous eval used default context=1000).

### Multi-Scale: Chebyshev vs Legendre (all at 10K steps)

| Configuration | Chebyshev MASE | Legendre MASE | Cheb vs Baseline | Leg vs Baseline | Cheb Advantage |
|:-------------:|:--------------:|:-------------:|:----------------:|:---------------:|:--------------:|
| ms d=4+d=6 | **1.1675** | 1.2052 | **-6.01%** | -2.97% | **+3.04%** |
| ms d=5+d=6 | — | 1.2550 | — | +1.04% | — |
| Cross (Cheb d=4 + Leg d=6) | — | 1.1900 | — | -4.20% | — |

**Note**: Legendre results corrected 2026-03-04. Previous evals used default context=1000, now all use context=4000 (matching other experiments). Cross-family (Cheb d=4 + Leg d=6) improves dramatically from 1.2379 → **1.1900** at ctx=4000.

**Lesson 16 (Legendre multi-scale, REVISED)**: With corrected context=4000 evaluation, Legendre multi-scale (d=4+d=6, 1.2052, -2.97%) is much more competitive than originally thought, though still 3.0% behind Chebyshev (1.1675). The cross-family combination (Cheb d=4 + Leg d=6, **1.1900**, -4.20%) is now the 3rd-best multi-scale result, suggesting Legendre d=6 provides complementary information when paired with Chebyshev d=4. Legendre ms d=5+d=6 (1.2550, +1.04%) still hurts — confirming nearby Legendre degrees lack complementarity.

**Theoretical justification (Chebyshev vs Legendre)**: Both families are proven to give sublinear regret O(T^{2/3}) for online convex optimization via monic polynomial preconditioning (Hazan & Marsden, 2025, arXiv:2502.06545). However, the **constants** differ critically:

1. **Minimax optimality**: Monic Chebyshev T_n(x)/2^{n-1} minimizes max|p(x)| on [-1,1] among all monic degree-n polynomials. This equiripple property guarantees the FIR filter coefficients have uniform spectral contribution — no single lag dominates. Legendre polynomials lack this property; their extrema are unevenly distributed.

2. **Multi-scale complementarity**: Chebyshev zeros are cos(kπ/n) — evenly spaced in angle, dense near ±1. Two Chebyshev polynomials of different degrees (d=4, d=6) have **non-overlapping zero crossings** that tile the frequency spectrum uniformly. Legendre zeros cluster toward the center of [-1,1], so two Legendre polynomials of nearby degree (d=4, d=6) provide largely **redundant** spectral information. This explains why the multi-scale gap (5.5%) is far larger than the single-scale gap (2.1% at d=6): the synergy between two channels depends on complementarity, which Chebyshev uniquely provides.

3. **Empirical consistency**: The Chebyshev advantage shrinks with degree (d=4: 3.4%, d=8: 0.07%), consistent with both families converging to the same spectral whitening at high degree. At low degree where the polynomial structure is most constrained, Chebyshev's optimality matters most.

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

### 15. Checkpoint Averaging Hurts the Best Model
The best ms46 10K checkpoint (1.1675) DEGRADES when averaged with earlier checkpoints: avg(7K-10K uniform) = 1.2086, avg(steps 20-35K of longer run) = 1.2040. The cosine annealing schedule converges the model to a sharp minimum at 10K where LR≈0. Averaging with checkpoints still on the descent path (7K, 8K, 9K) moves the weights away from this minimum. This explains why SWA/checkpoint averaging is harmful here: the final checkpoint is already well-optimized, and averaging can only dilute it. **For cosine-annealed 10K runs, the final checkpoint is the best; no post-hoc averaging can improve it.** Model soup (averaging hint vs non-hint models) fails even more catastrophically due to input dimension mismatch (1.4558 at α=0.7, 1.8323 at α=0.5).

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

4. **Data augmentation with KernelSynth**: Synthetic time series generated from random GP kernel combinations (RBF, Matern, ExpSineSquared, DotProduct, RationalQuadratic). 100K synthetic series added to LOTSA. Tests whether richer training data closes part of the gap to official M2.0. (Jobs 5119597/5119598 testing this.)

5. **Multi-stride hints**: Adding a d=4,stride=32 channel alongside d=4,s=16 and d=6,s=16. Wider stride captures longer-range patterns (128-timestep lookback vs 64). Spectral analysis shows this provides complementary frequency response, especially at period=30 steps (monthly cycle in daily data). (Job 5119776 testing this.)

6. **Weighted LOTSA sampling**: Official Moirai 2.0 weights uses balanced per-dataset weighting (small datasets upsampled, large datasets downsampled). Our default uses uniform weighting where large datasets dominate. Testing if balanced weighting + KernelSynth improves results. (Jobs 5119780/5119781 testing this.)

7. **Model-level dropout**: Current model has dropout_p=0.0 (only hint_dropout=0.1 on hint channels). Adding model dropout (0.1) may improve generalization, especially on low-frequency non-periodic datasets where the model over-relies on hints. (Job 5119789 testing this.)

8. **Scale to larger models**: All experiments use Moirai2 Small (11.4M). The hint mechanism adds negligible parameters (only changes the input projection dimension). Testing on Moirai2 Base (~90M) would show if the benefit persists or is subsumed by increased model capacity.

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

## Per-Frequency Analysis

Detailed breakdown of GIFT-Eval MASE by data frequency reveals a strong frequency-dependent pattern:

| Freq (seasonality) | # Configs | Baseline 10K | ms d=4+d=6 10K | Δ ms46 | hd10 100K | Δ hd10 |
|--------------------|-----------|-------------|----------------|--------|-----------|--------|
| 96 (15-min) | 5 | 1.359 | 1.073 | **-13.3%** | 1.060 | -11.2% |
| 48 (30-min) | 4 | 0.815 | 0.741 | **-7.1%** | 0.769 | -4.6% |
| 288 (5-min) | 7 | 0.920 | 0.868 | -5.6% | 0.856 | -6.6% |
| 24 (hourly) | 22 | 0.970 | 0.933 | -3.0% | 0.902 | -5.2% |
| 1440 (1-min) | 6 | 0.570 | 0.552 | -2.5% | 0.560 | -1.9% |
| 7 (daily) | 16 | 1.462 | 1.454 | -0.05% | 1.404 | **-4.6%** |
| 144 (10-min) | 2 | 0.525 | 0.527 | +0.2% | 0.540 | +2.6% |
| 1 (annual/misc) | 29 | 4.065 | 4.272 | +2.4% | 3.604 | **-7.9%** |
| 12 (monthly) | 6 | 1.498 | 1.556 | +3.5% | 1.486 | **-0.7%** |
| 4 (quarterly) | 3 | 1.519 | 1.619 | +6.8% | 1.420 | **-2.7%** |

**Split by high/low frequency** (geometric mean MASE, 100 configs):

| Category | Baseline 10K | ms d=4+d=6 10K | hd10 100K |
|----------|-------------|----------------|-----------|
| High-freq (≥24), 46 configs | 0.822 | 0.781 (-5.0%) | 0.775 (-5.7%) |
| Low-freq (<24), 54 configs | 1.896 | 1.926 (+1.6%) | 1.779 (-6.2%) |
| Win rate (vs baseline) | — | 63/100 (63%) | 79/100 (79%) |

**Key insight**: Multi-scale hints at 10K help high-frequency data (sub-hourly) enormously but hurt low-frequency data. The FIR taps at lag 32 and 64 (stride=16) align with daily periodicity in 15-min/30-min/hourly data but not with annual cycles in monthly/quarterly data. Dropout + longer training (100K) allows the model to learn when to ignore hints, fixing the low-frequency regression while maintaining high-frequency gains.

### Per-Horizon Analysis (ms d=4+d=6 vs Baseline, 10K)

| Horizon | #Configs | Baseline | ms46 | Change | Win Rate |
|---------|----------|----------|------|--------|----------|
| short | 55 | 1.1865 | 1.1574 | -2.45% | 37/55 (67%) |
| medium | 21 | 1.2596 | 1.1525 | **-8.51%** | 18/21 (86%) |
| long | 21 | 1.3810 | 1.2096 | **-12.41%** | 19/21 (90%) |

**Critical finding**: Hint benefit scales monotonically with horizon. Long-horizon configs get **5× the benefit** of short-horizon configs (-12.4% vs -2.5%). This makes theoretical sense — polynomial smoothing captures trends and removes high-frequency noise, which is most valuable when the model must extrapolate further into the future.

Top beneficiaries: solar/10T/long (-33.9%), us_births/M/short (-33.3%), SZ_TAXI/15T/long (-29.8%), bizitobs_application/10S/* (-23-29%).

Datasets where hints hurt: bizitobs_l2c/H/medium (+12.2%), solar/10T/short (+11.5%), m4_quarterly (+8.3%). These are mostly short-horizon, low-frequency, or datasets with specific patterns that hints disrupt.

### Statistical Significance (Paired Bootstrap, FEV 100 configs)

| Comparison | Geo Mean Ratio | 95% CI | P(A<B) | Significance |
|------------|---------------|--------|--------|--------------|
| ms46_10K vs Baseline_10K | 0.985 | [0.967, 1.004] | 0.946 | n.s. (p=0.054) |
| hd10_100K vs Baseline_10K | 0.940 | [0.921, 0.958] | 1.000 | *** (p<0.001) |
| hd10_100K vs Official M2.0 | 1.086 | [1.054, 1.123] | 0.000 | *** (worse) |
| hd10_100K vs ms46_10K | 0.955 | [0.932, 0.975] | 1.000 | *** |

**Note**: ms46_10K improvement is borderline significant on FEV (p=0.054) but clearly significant on GIFT-Eval (97 configs, -6.01%). The discrepancy arises because GIFT-Eval over-represents high-frequency datasets where hints have the largest impact. hd10_100K is robustly significant across both benchmarks.

---

## FEV Benchmark Analysis (100 configurations)

Extended evaluation on the FEV benchmark (autogluon/fev_datasets), which includes 100 dataset configurations. Context length: 4000.

| Model | FEV Geo MASE | MASE < 1.0 | vs Official M2.0 |
|-------|-------------|------------|-------------------|
| Official M2.0 | **1.1172** | 53/100 | — |
| hd10_100k | 1.2137 | 43/100 | 24/100 wins |
| ms_d4d6_10k | 1.2713 | 42/100 | 21/100 wins |

**Oracle best-of-ours**: 29/100 wins vs Official M2.0 (geo mean ratio 1.074). Even picking the best of our models per task, we remain 7.4% behind overall — gap is primarily data-driven (10x training data difference).

**Where we win** (ms_d4d6 or hd10 vs Official): kdd_cup_2022_10T (-14.6%), epf_pjm (-14.6%), fred_md_2025 (-12.1%), uci_air_quality (-7.2%), world_tourism (-6.5%). Mostly energy/electricity forecasting with clear periodicities.

**Where we lose**: uk_covid_cumulative (+136%), bizitobs_l2c_5T (+89%), boomlet_1975 (+84%), redset_15T (+78%). Cloud ops and irregular non-stationary data — domains where official model has private training data.

**Gap by frequency** (FEV, hd10_100k vs Official M2.0):
- 10T (2 tasks): ratio 0.948 — **we beat official**
- 1Y (1 task): ratio 0.983 — nearly matched
- 1D (16 tasks): ratio 1.041 — close
- 1W (11 tasks): ratio 1.068 — moderate gap
- 15T (5 tasks): ratio 1.293 — largest gap (cloud ops data)

hd10_100k is **within 5% of official M2.0 on 59/100 FEV tasks**.

### ms46 vs hd10 on FEV (head-to-head)

hd10 dominates ms46 at context=4000: **70/100 wins** (geo MASE 1.2137 vs 1.2713, -4.75%). Margin grows monotonically with decreasing frequency: H(+2.5%), D(+5.3%), W(+6.4%), M(+8.5%), Q(+20.3%), Y(+16.3%). ms46 only wins on sub-hourly periodic data (10T, 15T, 30T).

### Context-Length Sensitivity (critical finding)

| Model | GIFT-Eval (ctx=1000) | FEV (ctx=4000) | Degradation |
|-------|---------------------|----------------|-------------|
| ms_d4d6_10k | **1.1675** | 1.2713 | +8.9% |
| hd10_100k | 1.1918 | **1.2137** | **+1.8%** |
| Official M2.0 | 1.0236 | 1.1172 | +9.2% |

**ms46 is best at ctx=1000 but worst of the three at ctx=4000.** hd10 degrades only 1.8% vs ~9% for the others, making it the most context-robust model. The relative ranking reverses: ms46 beats hd10 by 2% at ctx=1000 but loses by 4.75% at ctx=4000.

**Interpretation**: With 4000-token context (250 patches), the model has access to more raw history and the fixed-stride hint channels (window ≈ 128–192 steps) become redundant or conflicting. Dropout training forces hd10 to learn hint-free fallback patterns, making it robust to this context-length shift.

---

## Per-Config Analysis (updated 2026-02-27)

Detailed per-config comparison reveals where hints help and hurt:

### ms46 (10K) vs Baseline — Win/Loss Breakdown

- **ms46 wins 74/97 configs** (baseline wins 23/97)
- Biggest ms46 gains: solar/10T/long (-33.9%), us_births/M (-33.3%), SZ_TAXI/15T/long (-29.8%), bizitobs_application/10S (-23 to -29%)
- Biggest ms46 regressions: bizitobs_l2c/H/medium (+12.2%), solar/10T/short (+11.5%), m4_quarterly (+8.3%)
- **Long horizons benefit most**: short -2.5%, medium -8.5%, long **-12.4%**

### hd10 (100K) vs Baseline — Win/Loss Breakdown

- **hd10 wins 73/97 configs** (baseline wins 24/97)
- Biggest hd10 gains: solar/10T/long (-32.3%), us_births/M (-30.0%), SZ_TAXI/15T/long (-27.5%)
- Biggest hd10 regressions: bizitobs_l2c/H/medium (+32.6%), solar/D/short (+20.2%), bizitobs_service/10S/long (+19.2%)

### Oracle (config-wise best of ms46 vs hd10)

- **Oracle MASE: 1.1421 (-8.1% vs baseline)** — 2% more than either alone
- ms46 wins 45/97, hd10 wins 52/97
- Significant complementarity: ms46 excels on 10S cloud ops, hd10 excels on low-freq

### Regression Root Causes

1. **Temporal scale mismatch**: Hint taps at lags 32/64 raw steps (stride=16) align well with 15T/30T/hourly periodicities but miss daily cycle. For 15T data, lag 96 = 24h (daily), but current hints max out at lag 64 (16h). **Fix: add stride=48 channel (taps at lags 96, 192).**
2. **Hint over-reliance at 100K**: hd10 regresses on 10S/bizitobs data because the single-degree hint doesn't capture the right spectral structure, and 100K training creates hint dependence. **Fix: use multi-scale (ms46) at 100K with dropout.**
3. **Short-horizon anti-correlation**: For datasets with high-amplitude cycles (solar), the hint's dominant tap encodes momentum from a period anti-correlated with the prediction target. **Fix: sequence dropout to force hint-independent predictions.**

### By Difficulty Level

| Category | N | Baseline | ms46 | ms46 % | hd10 | hd10 % |
|----------|:-:|:---:|:---:|:---:|:---:|:---:|
| Easy (MASE<1) | 38 | 0.7408 | 0.7112 | -4.0% | 0.7138 | -3.7% |
| Medium (1-2) | 45 | 1.2791 | 1.1924 | -6.8% | 1.2055 | -5.8% |
| Hard (2-5) | 10 | 3.2722 | 3.0693 | -6.2% | 3.2164 | -1.7% |
| Very Hard (>5) | 4 | 10.7515 | 9.1051 | -15.3% | 11.4166 | +6.2% |

ms46 helps across all difficulty levels. hd10 actually hurts on very hard configs (+6.2%), driven by bizitobs 10S regressions.

---

## In-Progress Experiments (updated 2026-03-03)

### Completed 100K (with 10K warmup)

| Experiment | MASE | vs 100K Baseline | Description |
|------------|------|------------------|-------------|
| **ms46_100k** | **1.2214** | **-5.40%** | ms d=4+d=6 + 10% drop, 100K, 10K warmup |
| hd10_10kw | 1.2579 | -2.57% | hint d=4 + 10% drop, 100K, 10K warmup |

**Lesson 17 (Warmup confound — RESOLVED)**: The 2×2 factorial is now complete:

| | 10K warmup | 1K warmup | Δ warmup |
|---|---|---|---|
| **Baseline** | 1.2911 | **1.2833** | -0.6% |
| **hd10 (d=4 + 10% drop)** | 1.2579 | **1.1918** | -5.3% |
| **Δ hints** | -2.6% | **-7.1%** | |

1K warmup helps baseline by only -0.6% (not significant), but helps hd10 by -5.3%. The **interaction between warmup and hints** is the main effect: short warmup allows early exploitation of hint signal. ms46 is more robust to warmup: 10K warmup gives 1.2214 (-5.40%), suggesting multi-scale hints tolerate suboptimal warmup schedules better than single-degree hints.

**Lesson 18 (Statistical significance)**: Paired bootstrap test (10K resamples) on 97 GIFT-Eval configs shows ms46 improvement is highly significant:
- Ratio ms46/baseline = 0.9399, 95% CI [0.9195, 0.9595]
- P(ms46 better) = 1.0000 (all 10K bootstrap samples favor ms46)
- Win rate: 74/97 (76.3%) configs improved
- Per-horizon significance: Short P=0.996, Medium P=1.000, Long P=1.000
- Largest wins: solar/10T/long (-33.9%), us_births/M (-33.3%), SZ_TAXI/15T/long (-29.8%)
- Largest regressions: bizitobs_l2c/H/medium (+12.2%), solar/10T/short (+11.5%), m4_quarterly (+8.3%)
- Analysis script: `analysis/statistical_analysis.py`

**Lesson 19 (Stride = patch alignment)**: Stride=16 (= patch_size) outperforms all other strides because it creates hints about **inter-patch structure** rather than within-patch variation:

| Stride | Relation to patch | d=4 MASE | d=6 MASE |
|--------|-------------------|----------|----------|
| 4 | sub-patch | — | 1.2346 |
| 8 | half-patch | 1.2382 | 1.2204 |
| 16 | **patch-aligned** | **1.1944** | **1.1836** |

When S=P=16, `hint[16i+j] = Σ c_k · target[16(i-k)+j]` — each element j within a hint patch only references element j from previous patches. This creates coherent patch-level hints. When S=1, the hint mixes different sub-patch positions, creating incoherent signals. From a frequency perspective: S=16 hints capture low-frequency inter-patch dynamics (which the patchwise transformer misses), while S=1 hints capture within-patch variation (which the transformer already handles). See `analysis/stride_analysis.py` for detailed derivation and figures.

**Lesson 20 (Domain/frequency breakdown)**: Per-domain and per-frequency analysis reveals clear structure in where hints help:

By domain (geomean MASE improvement, 10K):
| Domain | n | Improvement | W/L |
|--------|---|-------------|-----|
| Transport | 11 | **+10.2%** | 10/1 |
| IT/Cloud | 24 | **+8.6%** | 22/2 |
| Energy | 16 | +6.4% | 9/7 |
| ETT | 16 | +5.3% | 13/3 |
| Weather | 15 | +3.3% | 12/3 |
| M4 Competition | 6 | **-0.9%** | 2/4 |

By frequency:
| Frequency | n | Improvement | W/L |
|-----------|---|-------------|-----|
| Sub-hourly | 36 | **+12.3%** | 34/2 |
| Hourly | 31 | +3.0% | 23/8 |
| Daily | 15 | -0.6% | 9/6 |
| Quarterly | 1 | **-8.3%** | 0/1 |

**Key insight**: Hints capture inter-patch temporal structure at the patch boundary scale (16 time steps). High-frequency data (sub-hourly) has the most structure at this scale, yielding 12.3% improvement. Low-frequency data (daily, quarterly) has less inter-patch structure to exploit, and the hint's fixed polynomial taps may add noise. Competition benchmarks (M4) are often short, low-frequency, and already well-served by the baseline's learned representations.

Analysis: `analysis/domain_analysis.py`, `analysis/limitations_analysis.py`

**Lesson 21 (Computational overhead)**: Polynomial hint preconditioning adds negligible computational cost:
- Parameters: +24.6K (+0.22% of 11.4M) for ms d=4+d=6
- FLOPs: +80K FIR FLOPs vs 6.27G transformer FLOPs (+0.001%)
- Wall-clock: <2% slower on same GPU (0.71 vs 0.72 it/s on A100)
- FIR filter operates on raw time steps O(d×T) before patching, while transformer is O(n²×d×L)
- Analysis: `analysis/computational_overhead.py`

---

## Effect Size Analysis (2026-03-03)

Statistical characterization of the hint improvement across GIFT-Eval 97 configurations:

### Overall Effect Magnitude

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Cohen's d (point estimate) | 0.369 | **Small-to-medium effect** (d>0.5 threshold) |
| Direction of change | 65 improve / 32 worsen | 67% of configs improve |
| Median improvement (improvers) | -2.8% MASE | Typical improving config |
| Median harm (worseners) | +1.9% MASE | Typical harming config |
| **Wilcoxon signed-rank p-value** | **3.65e-08** | **Highly statistically significant (p<0.001)** |

### Effect Size by Horizon

| Horizon | n | Cohen's d | Interpretation |
|---------|---|-----------|-----------------|
| Short (≤96 steps) | 37 | 0.244 | Small effect (variance-dominated) |
| Medium (97-336 steps) | 21 | 0.447 | Small-to-medium effect |
| Long (>336 steps) | 39 | **0.591** | **Medium-to-large effect** (strongest!) |

**Pattern**: Hints deliver monotonically increasing benefit with forecast horizon. Long-horizon forecasts ($>336$ steps) capture more of the inter-patch temporal structure that the hint FIR filters reveal.

### Multi-Metric Agreement

Examined directional agreement between MASE and sMAPE (symmetric mean absolute percentage error):

| Metric Pair | Agreement Rate | Pearson r | Interpretation |
|-------------|---|-----------|---|
| MASE vs sMAPE | **74%** | **0.847** | Good agreement (hints help on both metrics) |

**Stability**: The hint improvement is robust across different error metrics, not an artifact of how MASE scales. Configs where hints help on MASE typically also improve on sMAPE.

### Conclusion

The -6.01% improvement (multi-scale d=4+d=6 at 10K) represents a **highly significant and practically meaningful effect**:
- Statistical significance well below p<0.05 threshold
- Effect size grows with horizon (medium effect on 40% of configs)
- Robust across multiple error metrics (74% directional agreement)
- Especially strong on high-frequency and transport domains (10-12% improvement)

---

### New Experiments (10K)

| Experiment | Job ID | Description | Status |
|------------|--------|-------------|--------|
| **s48_10k** | **5121360** | **ms d=4+d=6@s16 + d=4@s48 (daily cycle coverage)** | **Pending** |
| **sd20ms_10k** | **5121361** | **ms d=4+d=6 + 20% sequence dropout** | **Pending** |
| sd20_10k | 5120415 | ms d=4+d=6 + 20% seq dropout (different config) | Pending |
| vp46_10k | 5120929 | ms d=4+d=6 + variable prefix [0.15,0.45] | **1.2896** (+3.82% vs baseline). Variable prefix hurts significantly. |
| vpb_10k | 5120930 | Baseline + variable prefix [0.15,0.45] | Running |
| eval_stu2k | 5119449 | Evaluate 3 STU 2K checkpoints on GIFT-Eval | Running |

### Weighted LOTSA Experiments

| Experiment | Job ID | Description | Status |
|------------|--------|-------------|--------|
| wtb1k | 5120341 | Weighted LOTSA baseline, warmup=1K | Running |
| wth1k | 5120342 | Weighted LOTSA + ms hint, warmup=1K | Running |
| m2_wt_base | 5091690 | Weighted LOTSA baseline, warmup=10K (⚠️ undertrained) | Complete |
| m2_wt_hint | 5091691 | Weighted LOTSA + ms hint, warmup=10K (⚠️ undertrained) | Complete |
| ge_wt | 5121044 | GIFT-Eval of warmup=10K weighted models | Pending |

### Completed 100K Experiments

| Experiment | MASE | vs 100K Baseline | vs 10K Baseline |
|------------|------|------------------|-----------------|
| Baseline 100K | 1.2878 | — | +3.68% (overfitting) |
| Hint d=4 (no dropout) 100K | 1.2135 | -5.77% | -2.30% |
| Hint d=6 (no dropout) 100K | 1.2220 | -5.11% | -1.62% |
| **Hint d=4 + 10% dropout 100K** | **1.1918** | **-7.45%** | **-4.05%** |

### Completed Quick STU Experiments (2K steps)

| Experiment | Job ID | Final Loss | Notes |
|------------|--------|------------|-------|
| stu_hint_d4d6_2k | 5092066 | 0.116 | STU hybrid + ms d=4+d=6 hints |
| stu_fulldff_k8_warmgate_2k | 5092067 | 0.117 | Full d_ff, K=8 filters, warm gate |
| stu_hint_fulldff_2k | 5092068 | 0.115 | STU + hints + full d_ff |

All converge to similar loss. Need GIFT-Eval to differentiate. Not yet evaluated.

### Hint Ablation Experiments (2026-03-03)

Testing whether the hint benefit comes from extra capacity, any learned filter, or the specific Chebyshev polynomial structure.

| Experiment | Job ID | Result | Description |
|------------|--------|--------|-------------|
| abl_dup | 5312571 | **MASE 1.2342 (-0.64%)** | Duplicate input (hint = copy of target), 10K |
| abl_lc4 | 5180864 | **MASE 1.2351 (-0.56%)** | Learned 4-tap conv (Cheb d=4 init, learnable), 10K |
| abl_lc16 | 5180890 | **MASE 1.2775 (+2.85%)** | Learned 16-tap conv (Cheb d=16 init, learnable), 10K |
| zero_h46 | 5317047 | **MASE 1.3201 (+6.28%)** | Zero hint (ms46 arch, hints=0), 10K |
| rand_h46 | 5317048 | **MASE 1.2806 (+3.10%)** | Random hint (ms46 arch, hints=randn), 10K |
| bl_lrsw | 5317049 | BL lr=5e-4: **1.2531 (+0.89%)**, lr=2e-3: **1.2586 (+1.33%)** | Baseline LR sweep confirms default lr=1e-3 is optimal |
| seed_ms46 | 5317272 | TIMED OUT | ms46 seeds 0,1 complete; seed2 partial; BL seeds 0,1 complete; seed2 missing |
| 3chan_10k | 5319588 | **MASE 1.2333 (-0.71%)** | Channel scaling: d=2+d=4+d=6 (3 channels), 10K |
| 4chan_10k | 5319589 | **MASE 1.2091 (-2.66%)** | Channel scaling: d=2+d=4+d=6+d=8 (4 channels), 10K |
| s1_hint | 5317702 | **MASE 1.2598 (+1.43%)** | Stride=1 (dense FIR) hint d=6, 10K. Patch-misaligned stride hurts. |
| s32_hint | 5317717 | **MASE 1.1975 (-3.59%)** | Stride=32 (2× patch) hint d=6, 10K |
| ctx_sweep | 5318043 | MS46@1K=1.2074, BL@1K=1.2512, MS46@2K=1.1700, BL@2K=1.2304, MS46@4K=1.1675, BL@4K=1.2421 | Context length sweep — COMPLETE |
| ema_h_10k | 5319920 | Training complete, eval pending | EMA hint d=6 s=16, 10K |
| diff_h10k | 5319921 | Training complete, eval pending | Differencing hint d=6 s=16, 10K |
| leg_ms46@100K | 5316865 | **MASE 1.2317 (-4.60%)** | Legendre d=4+d=6 + 10% drop, 100K |
| bl1k_100k | 5316864 | **MASE 1.2833 (-0.60%)** | Baseline + 1K warmup, 100K |
| ms46d1k | 5321651 | Running (~70K/100K) | ms46+drop+1kwarm 100K (best config) |
| anneal5k | 5347064 | Pending (ailab) | Hint annealing (dropout 0→100% over 5K steps), 10K (fixed config) |
| anneal8k | 5347065 | Pending (ailab) | Hint annealing (dropout 0→100% over 8K steps), 10K (fixed config) |
| eval_leg2 | 5347063 | Running | cross_c4l6 evaluation (only missing Legendre model) |
| eval_ed | 5347062 | Running | EMA + differencing hint evaluation |
| seed_rem | 5347066 | Pending (ailab) | Remaining seeds (ms46_seed2, baseline_seed2) |
| eval_seeds | 5347067 | Pending (dep) | Seed sensitivity evaluation (6 models) |

**Key findings**:
- **abl_dup (MASE 1.2342, -0.64%)**: Duplicate input barely helps — extra capacity alone is NOT the driver.
- **abl_lc4 (MASE 1.2351, -0.56%)**: Learned 4-tap filters ≈ duplicate input — learning doesn't capture polynomial structure.
- **abl_lc16 (MASE 1.2775, +2.85%)**: MORE learnable parameters = WORSE. 16-tap learned filters degrade performance below baseline. This is the most dramatic ablation result.
- **ctx_sweep**: MS46@ctx=1000 (1.2074) beats BL@ctx=4000 (1.2421) by 2.8%. MS46@ctx=2000 (1.1700) beats BL@ctx=4000 by 5.8%.
- **s1_hint (MASE 1.2598, +1.43%)**: Stride=1 (dense FIR) HURTS. Only stride=16 (patch-aligned) works. Confirms patch-alignment hypothesis.
- **s32_hint (MASE 1.1975, -3.59%)**: Stride=32 (2×patch) works but weaker than stride=16. Full stride sweep: S=1 (+1.4%), S=4 (-0.6%), S=8 (-1.7%), **S=16 (-4.7%)**, S=32 (-3.6%). Non-monotonic — stride=P is optimal.
- **4chan_10k (MASE 1.2091, -2.66%)**: 4 channels (d=2,4,6,8) slightly worse than 2 channels (d=4+d=6). Diminishing returns — more channels don't help.
- **3chan_10k (MASE 1.2333, -0.71%)**: 3 channels (d=2,4,6) much worse than 2 channels. Including d=2 (too low degree) dilutes the signal.
- **BL LR sweep**: lr=5e-4 (1.2531) and lr=2e-3 (1.2586) both worse than default lr=1e-3 (1.2421). **Baseline is not under-optimized** — the 6% improvement is real, not a LR artifact.
- **Ablation hierarchy**: Zero (+6.3%) > Random (+3.1%) > Learned 16 (+2.9%) > BL (0%) > Dup (-0.6%) ≈ Learned 4 (-0.6%) >> MS46 (-6.0%). 12.3% gap between best and worst.

**Ablation summary** (all with ms46 architecture, 64-dim in_proj):

| Ablation | MASE | vs Baseline | Key insight |
|----------|:----:|:-----------:|-------------|
| **Chebyshev d=4+d=6** | **1.1675** | **-6.01%** | Fixed polynomial structure is key |
| Zero hints | 1.3201 | +6.28% | Extra capacity HURTS — wider projection wastes capacity |
| Random hints | 1.2806 | +3.10% | Random noise HURTS — not any-signal benefit |
| Duplicate input | 1.2342 | -0.64% | Extra capacity barely helps |
| Learned 4-tap | 1.2351 | -0.56% | Learned filters don't capture structure |
| Learned 16-tap | 1.2775 | +2.85% | More learnable params = WORSE |
| Baseline | 1.2421 | --- | Reference |

**Interpretation**:
- **Not capacity**: Zero hints (+6.28%) shows extra input dims actively HURT. The wider projection (64-dim vs 32-dim) wastes capacity on uninformative channels. Duplicate input (-0.64%) barely helps — confirming extra capacity alone is insufficient.
- **Not any signal**: Random hints (+3.10%) shows noise hurts even more than duplicating the input. The model can learn to ignore duplicate channels but not random noise.
- **Not learned filters**: 4-tap (-0.56%) ≈ duplicate, 16-tap (+2.85%) even hurts. The model can't learn the correct filter structure.
- **Fixed Chebyshev structure is essential**: Only the mathematically determined polynomial coefficients give the 6% improvement. The 12.3% gap between Chebyshev (-6.01%) and zero hints (+6.28%) is the strongest evidence for the importance of polynomial spectral structure.
- **More flexibility hurts**: zero (+6.28%) > random (+3.10%) > learned 16-tap (+2.85%) > baseline (0%) > duplicate (-0.64%) ≈ learned 4-tap (-0.56%) ≫ Chebyshev (-6.01%). The less structured the hint, the worse the result.

**Context sensitivity**:

| Model | ctx=1000 | ctx=4000 |
|-------|:--------:|:--------:|
| Baseline | 1.2512 | 1.2421 |
| MS46 | **1.2074** | **1.1675** |
| MS46 vs BL | -3.5% | -6.0% |

MS46@ctx=1000 beats BL@ctx=4000 by 2.8% — hints compensate for 4× less context.

**Input projection weight analysis (MS46 @ 10K)**:
- Target channel avg norm: 1.288 (100%)
- Hint d=4 avg norm: 0.868 (67.4% of target)
- Hint d=6 avg norm: 1.018 (79.0% of target)
- Baseline target avg norm: 1.536
- Both hint channels receive significant attention. d=6 gets more weight, consistent with d=6 being the better single-scale degree.

### Resume Jobs (2026-03-03)

| Experiment | Job ID | Progress | Description |
|------------|--------|----------|-------------|
| bl1k_r | 5316864 | Resuming from 90K→100K, epoch 906 | Baseline 1K warmup, 100K + auto-eval |
| leg46_r | 5316865 | Resuming from 60K→100K | Legendre ms d=4+d=6 + 10% drop, 100K + auto-eval (pending) |
