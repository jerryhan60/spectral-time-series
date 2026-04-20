# Preconditioning Experiment Results Summary

**Last updated**: 2026-02-19
**Branch**: `spectral_non_precond`
**Model**: Moirai2 Small (11.39M params, 6 layers, d_model=384, patch_size=16)
**Training data**: LOTSA v1 unweighted (27 datasets)

---

## 1. Executive Summary

- **All preconditioning variants tested so far hurt forecasting performance compared to the no-preconditioning baseline.** The best learnable model (stride=16, d=4) achieves only +0.30% worse MASE on FEV-Bench, which is near-parity, but no configuration actually beats the baseline.

- **Stride=16 (patch-aligned) preconditioning is far superior to stride=1.** The stride=16 variants (d=1 and d=4) achieve MASE within 0.5% of baseline on FEV-Bench, while stride=1 variants degrade by 3-4%. This suggests that preconditioning at the patch boundary avoids disrupting intra-patch structure.

- **Learnable coefficients with no regularization (stride=1) diverge from identity toward a strong high-pass filter** (learned coeffs [-0.667, -0.126, -0.056, -0.045]), causing +4.3% MASE degradation on FEV-Bench and +12.5% on GIFT-Eval (97 configs). Strong L2 regularization (lambda=1.0) keeps coefficients near zero but still degrades by +3.1%.

- **Fixed Chebyshev preconditioning (EXP-1, 100K steps) shows monotonic degradation with degree**: d=2 is +39% worse than the d=1 control, and d=7 is +121% worse. The FIR inverse filter cannot recover information destroyed by the fixed polynomial filter.

- **Frequency matters**: Preconditioning with stride=16 shows competitive or slightly better performance at sub-hourly and monthly+ frequencies, but consistently underperforms at hourly frequencies. The learnable free model (stride=1) only beats the baseline in the monthly+ frequency band.

---

## 2. Experiment Overview Table

All experiments use Moirai2 Small, LOTSA v1 unweighted, bs=256, lr=1e-3, cosine annealing, anomaly_zscore_threshold=8.0, seed=42.

### Learnable Coefficient Experiments (EXP-6)

| # | Experiment | Steps | Stride | Degree | Learned Coefficients | FEV MASE Geo | GIFT MASE Geo | vs Baseline (FEV) | vs Baseline (GIFT) |
|---|-----------|-------|--------|--------|---------------------|:------------:|:-------------:|:-----------------:|:------------------:|
| 1 | **Baseline 10K** (no precond) | 10K | -- | -- | -- | 1.2908 | **1.2395** | -- | -- |
| 2 | **Baseline 25K** (no precond) | 25K | -- | -- | -- | **1.2869** | **1.2422** | -- | -- |
| 3 | Learnable, Lyap5 init (s=1) | 10K | 1 | 4 | not extracted | 1.3653 | 1.3428 | +5.78% | +8.33% |
| 4 | Learnable, free (s=1, lam=0) | 25K | 1 | 4 | [-0.667, -0.126, -0.056, -0.045] | 1.3422 | 1.3978 | +4.30% | +12.53% |
| 5 | Learnable, stride=16, d=4 | 25K | 16 | 4 | [-0.115, -0.039, -0.031, -0.001] | 1.2907 | -- | +0.30% | -- |
| 6 | Learnable, stride=16, d=1 | 25K | 16 | 1 | [-0.128] | 1.2931 | -- | +0.48% | -- |
| 7 | Learnable, strongL2 (s=1, lam=1.0) | 25K | 1 | 4 | [-0.001, -0.001, -0.001, -0.001] | 1.3268 | -- | +3.10% | -- |

### Fixed Chebyshev Degree Sweep (EXP-1, 100K steps)

| # | Experiment | Fixed Coefficients | GIFT-Eval MASE Geo (100K) | Normalized MASE | vs d=1 Control |
|---|-----------|-------------------|:-------------------------:|:---------------:|:--------------:|
| 8 | d=1 (identity control) | [0.0] | 1.2785 | 0.9146 | -- |
| 9 | d=2 | [0.0, -0.5] | 1.6995 | 1.2681 | +39% |
| 10 | d=3 | [0.0, -0.75, 0.0] | 1.9419 | 1.4502 | +59% |
| 11 | d=4 | [0.0, -1.0, 0.0, 0.125] | 2.1112 | 1.5769 | +72% |
| 12 | d=5 | [0.0, -1.25, 0.0, 0.3125, 0.0] | 2.2891 | 1.7090 | +87% |
| 13 | d=6 | [0.0, -1.5, 0.0, 0.5625, 0.0, -0.03125] | 2.3973 | 1.7892 | +96% |
| 14 | d=7 | [0.0, -1.75, 0.0, 0.875, 0.0, -0.109375, 0.0] | 2.7088 | 2.0209 | +121% |

---

## 3. Fixed vs Learnable Comparison

### Coefficient Analysis

The key question: can learned coefficients find a better filter than fixed polynomials?

| Configuration | Stride | Init Coefficients | Final Learned Coefficients | max|c| | Sum|c| |
|--------------|--------|-------------------|---------------------------|--------|--------|
| Fixed Chebyshev d=4 | 1 | [0, -1.0, 0, 0.125] | (not learnable) | 1.000 | 1.125 |
| Learnable free (s=1) | 1 | [0, -0.081, 0, -0.089] | **[-0.667, -0.126, -0.056, -0.045]** | 0.667 | 0.894 |
| Learnable stride16 d=4 | 16 | [0, -0.081, 0, -0.089] | **[-0.115, -0.039, -0.031, -0.001]** | 0.115 | 0.186 |
| Learnable stride16 d=1 | 16 | [0] (Lyapunov) | **[-0.128]** | 0.128 | 0.128 |
| StrongL2 (s=1, lam=1.0) | 1 | [0, -0.081, 0, -0.089] | **[-0.001, -0.001, -0.001, -0.001]** | 0.001 | 0.004 |

**Key observations:**

1. **Stride=1 free learning produces large coefficients** (max|c|=0.667). The optimizer aggressively moves toward a high-pass filter, which improves training loss but hurts generalization. This mirrors the fixed Chebyshev finding: stronger filtering = worse eval.

2. **Stride=16 naturally constrains coefficient magnitude** (max|c|=0.115-0.128). Because the filter operates at patch boundaries (every 16 timesteps), each coefficient has less influence, so the optimizer finds a gentle filter that barely modifies the signal.

3. **Strong L2 regularization collapses coefficients to near-zero** (max|c|=0.001), effectively disabling preconditioning. Yet FEV-Bench MASE is still +3.1% worse than baseline, suggesting the preconditioning infrastructure itself (even as a no-op) introduces some overhead or distributional shift.

### Side-by-Side: Same Degree, Different Approaches

All at 25K steps, degree 4:

| Approach | Coefficients | FEV MASE | vs Baseline |
|----------|-------------|:--------:|:-----------:|
| Baseline (no precond) | -- | **1.2869** | -- |
| Fixed Chebyshev d=4 (100K) | [0, -1.0, 0, 0.125] | ~2.11* | +64%* |
| Learnable free (s=1) | [-0.667, -0.126, -0.056, -0.045] | 1.3422 | +4.30% |
| Learnable stride16 d=4 | [-0.115, -0.039, -0.031, -0.001] | **1.2907** | **+0.30%** |
| StrongL2 (s=1) | [-0.001, -0.001, -0.001, -0.001] | 1.3268 | +3.10% |

*Note: Fixed Chebyshev d=4 was evaluated on GIFT-Eval at 100K steps, not FEV-Bench at 25K, so comparison is approximate.*

**Verdict**: Learnable coefficients with stride=16 produce a near-identity filter that achieves near-parity with baseline. But the question remains: if the optimal filter is near-identity, why use preconditioning at all?

---

## 4. Frequency-Dependent Analysis

### FEV-Bench MASE by Frequency Band (Geometric Mean, 25K steps)

| Frequency | n | Baseline | Learnable Free (s=1) | Stride16 d=4 | Stride16 d=1 | StrongL2 (s=1) |
|-----------|:-:|:--------:|:--------------------:|:------------:|:------------:|:--------------:|
| Sub-hourly | 24 | 0.7795 | 0.8779 (+12.6%) | **0.7752** (-0.6%) | **0.7741** (-0.7%) | 0.8095 (+3.8%) |
| Hourly | 22 | **0.8785** | 0.9534 (+8.5%) | 0.9068 (+3.2%) | 0.9155 (+4.2%) | 0.9297 (+5.8%) |
| Daily | 19 | 1.4382 | 1.5070 (+4.8%) | **1.4193** (-1.3%) | 1.4646 (+1.8%) | 1.4660 (+1.9%) |
| Weekly | 14 | 2.1496 | 2.2188 (+3.2%) | 2.2004 (+2.4%) | **2.1410** (-0.4%) | 2.1983 (+2.3%) |
| Monthly+ | 21 | 2.1874 | **2.0097** (-8.1%) | 2.1509 (-1.7%) | 2.1308 (-2.6%) | 2.2105 (+1.1%) |
| **Overall** | **100** | **1.2869** | 1.3422 (+4.3%) | 1.2907 (+0.3%) | 1.2931 (+0.5%) | 1.3268 (+3.1%) |

**Bold** = best in row (among all models including baseline).

### Head-to-Head Wins vs Baseline by Frequency (FEV-Bench)

| Frequency | Learnable Free | Stride16 d=4 | Stride16 d=1 | StrongL2 |
|-----------|:-------------:|:------------:|:------------:|:--------:|
| Sub-hourly (24) | 1-23 | 11-13 | 12-12 | 10-14 |
| Hourly (22) | 3-19 | 4-18 | 4-18 | 3-19 |
| Daily (19) | 4-15 | **10-9** | 5-14 | 4-15 |
| Weekly (14) | 5-9 | 5-9 | **8-6** | 6-8 |
| Monthly+ (21) | **12-9** | **14-7** | **14-7** | 6-15 |
| **Total (100)** | 25-75 | 44-56 | 43-57 | 29-71 |

Format: Wins-Losses (preconditioning wins - baseline wins).

### Key Frequency Insights

1. **Sub-hourly**: Stride=16 variants are competitive (near 50-50 head-to-head, slightly better geo mean). Stride=1 learnable catastrophically fails (1W-23L), likely because the filter disrupts the fine-grained temporal patterns within patches.

2. **Hourly**: All preconditioning variants lose badly (3-4W vs 18-19L). Hourly data appears most sensitive to any signal modification.

3. **Daily**: Stride=16 d=4 is the only variant competitive here (10W-9L), suggesting patch-aligned degree-4 filtering may help with daily seasonality patterns.

4. **Weekly/Monthly+**: Both stride=16 variants win more than they lose at monthly+ frequencies (14W-7L). The learnable free model also wins at monthly+ (12W-9L) despite losing overall -- its learned high-pass filter may help with slow-moving trends.

5. **Pattern**: Preconditioning hurts most at hourly frequencies, helps most at monthly+ frequencies. The crossover appears around daily frequency.

---

## 5. Key Insights

### Insight 1: Preconditioning Creates a Train-Eval Distribution Shift

All variants show lower training loss than baseline but worse eval metrics. The preconditioning transforms the signal distribution during training, and the model learns patterns specific to the transformed distribution that don't transfer to evaluation.

| Experiment | Training Loss (final) | FEV MASE (eval) |
|-----------|:--------------------:|:-----------:|
| Baseline 25K | ~0.11 | **1.2869** |
| Learnable free (s=1) 25K | ~0.07 | 1.3422 |

The 36% lower training loss translates to 4.3% *worse* eval MASE.

### Insight 2: Coefficient Magnitude Correlates with Degradation

Across all experiments, there is a clear monotonic relationship between filter coefficient magnitude and eval degradation:

| max|c| | Example | FEV MASE vs Baseline |
|---------|---------|:--------------------:|
| 0.000 | Baseline | 0.0% |
| 0.001 | StrongL2 | +3.1% |
| 0.115 | Stride16 d=4 | +0.3% |
| 0.128 | Stride16 d=1 | +0.5% |
| 0.667 | Learnable free (s=1) | +4.3% |
| 1.000 | Fixed Chebyshev d=4 | +64%* |

*Exception*: StrongL2 has near-zero coefficients but +3.1% degradation, suggesting that the preconditioning infrastructure itself (learnable parameters, gradient flow through the filter) introduces a cost even when the filter is effectively identity.

### Insight 3: Stride=16 is the Only Viable Configuration

Stride=1 preconditioning modifies every adjacent timestep, disrupting the fine-grained temporal structure within patches. Stride=16 (= patch_size) only modifies relationships between patches, preserving intra-patch structure. This makes stride=16 nearly harmless (+0.3%) but also nearly useless -- the filter learns to be near-identity.

### Insight 4: Fixed Chebyshev is Strictly Worse Than Learnable

The fixed Chebyshev filter uses coefficients determined by polynomial theory, which are too aggressive for time series forecasting. Learnable coefficients converge to much smaller magnitudes than any Chebyshev polynomial of degree >= 2 would produce, confirming that the "optimal" filter for this task is close to identity.

### Insight 5: The Learnable Free Model Only Wins on Slow Data

The stride=1 learnable model's strong high-pass filter (max|c|=0.667) helps with monthly+ data (-8.1% MASE) by removing slow trends, but catastrophically degrades sub-hourly data (+12.6% MASE). A frequency-adaptive preconditioning approach might capture these heterogeneous benefits.

---

## 6. Experiment Status and Next Steps

### Completed Experiments

| Experiment | Status | Key Result |
|-----------|--------|------------|
| EXP-0: Quick validation (200 steps) | COMPLETED | Precond wins 75/97 at 200 steps (did not hold at scale) |
| EXP-1: Fixed Chebyshev d=1..7 (100K steps) | COMPLETED | Monotonic degradation with degree |
| EXP-1b: d=4 regularization sweep (100K steps) | COMPLETED (training), eval PENDING | Lambda sweep from 0 to 10 |
| EXP-6-quick: Learnable 10K | COMPLETED | Baseline wins (1.2908 vs 1.3653) |
| EXP-6-25k: Learnable free 25K | COMPLETED | Baseline wins (1.2869 vs 1.3422) |
| Stride=16 d=1 and d=4 (25K) | COMPLETED | Near-parity (+0.3-0.5%) |
| StrongL2 (s=1, 25K) | COMPLETED | Baseline wins (1.2869 vs 1.3268) |

### Still Running / Pending

| Experiment | Status | Notes |
|-----------|--------|-------|
| EXP-1b GIFT-Eval (97 configs) | PENDING (pli queue) | Full GIFT-Eval on all regularization sweep checkpoints |
| EXP-6 stride=16 GIFT-Eval | NOT SUBMITTED | Need to run GIFT-Eval on stride=16 d=1 and d=4 checkpoints |
| StrongL2 GIFT-Eval | NOT SUBMITTED | Need to run GIFT-Eval on strongL2 checkpoint |

### Likely Cancelled Experiments

| Experiment | Reason |
|-----------|--------|
| EXP-2: Legendre degree sweep | Negative EXP-1 results; Legendre unlikely to differ fundamentally |
| EXP-3: FIR inverse hyperparameters | FIR inverse cannot recover filtered information |
| EXP-4: Loss reversal vs FIR inverse | Neither approach overcomes the fundamental train-eval shift |
| EXP-5: Stride variations (2, 4, 16) | Stride=16 tested in learnable experiments; s=2 and s=4 unlikely to improve |
| EXP-6b/c/d: Additional learnable variants | Core result is negative; init sensitivity and degree interaction are moot |

### Potential Future Directions

1. **Frequency-adaptive preconditioning**: Apply different filter strengths based on detected frequency, potentially benefiting monthly+ data without hurting hourly data.

2. **Evaluate stride=16 at 100K steps**: The stride=16 d=4 model (+0.30% at 25K) might converge to or beat baseline with longer training.

3. **Residual preconditioning**: Instead of replacing the signal, add the filter output as a residual. This bounds the maximum disruption.

4. **Abandon preconditioning**: The consistently negative results suggest this approach is fundamentally incompatible with the Moirai2 architecture's causal decoder + quantile loss. Research effort may be better spent elsewhere (e.g., STU hybrid architectures, attention modifications).

---

## Appendix: Data Sources

| File | Description |
|------|-------------|
| `gifteval/results/fev_bench_baseline_25k_20260217_065234.csv` | Baseline 25K FEV-Bench (100 tasks) |
| `gifteval/results/fev_bench_learnable_free_25k_20260217_065234.csv` | Learnable free 25K FEV-Bench |
| `gifteval/results/fev_bench_stride16_d4_25k_20260218_113256.csv` | Stride=16 d=4 25K FEV-Bench |
| `gifteval/results/fev_bench_stride16_d1_25k_20260218_113256.csv` | Stride=16 d=1 25K FEV-Bench |
| `gifteval/results/fev_bench_strongL2_25k_20260218_113256.csv` | StrongL2 25K FEV-Bench |
| `gifteval/results/fev_bench_baseline_10k_20260217_012709.csv` | Baseline 10K FEV-Bench |
| `gifteval/results/fev_bench_learnable_lyap5_10k_20260217_012709.csv` | Learnable Lyap5 10K FEV-Bench |
| `gifteval/results/gifteval_results_epoch_99-step_10000_20260217_082551.csv` | Baseline 10K GIFT-Eval (97 configs) |
| `gifteval/results/gifteval_results_epoch_99-step_10000_20260217_085433.csv` | Learnable 10K GIFT-Eval |
| `gifteval/results/gifteval_results_epoch_249-step_25000_20260217_083634.csv` | Baseline 25K GIFT-Eval |
| `gifteval/results/gifteval_results_epoch_249-step_25000_20260217_085901.csv` | Learnable 25K GIFT-Eval |
| `docs/experiment_log_preconditioning.md` | Full experiment log with EXP-1 fixed Chebyshev results |
| `slurm_job_log.md` | SLURM job tracking |
