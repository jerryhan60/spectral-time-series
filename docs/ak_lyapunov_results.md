# AK's Lyapunov Preconditioning Results vs Our Reproduction

## Executive Summary

- AK tested Lyapunov-regularized polynomial preconditioning on **Chronos2** (FEV-Bench, 100 tasks) and **Moirai2** (20K-step pretraining)
- **Oracle reversal** (using ground truth for undoing filter) improves MASE by 6-14% on Chronos2
- **Self reversal** (using model predictions) consistently degrades MASE by 0.5-5% on Chronos2
- This oracle-vs-self gap is the **fundamental challenge** we're addressing in our Moirai2 work
- AK's best self-reversal: Lyapunov d=3 lambda=5 at -0.54% degradation (essentially baseline)
- AK's Moirai2 runs (24 configs, 20K steps each) were never evaluated on GIFT-Eval

---

## 1. AK's Chronos2 FEV-Bench Results (100 Tasks)

### Lyapunov Regularization Sweep

Baseline Chronos2 SQL Geo Mean: **1.0238**

| Config | Degree | Lambda | Coefficients | Oracle MASE Geo | Self MASE Geo | Oracle % | Self % |
|--------|--------|--------|-------------|:---:|:---:|:---:|:---:|
| lyap_d3_l1 | 3 | 1.0 | [-0.454, 0, -0.306] | 0.8878 | 1.0533 | **-13.3%** | +2.9% |
| lyap_d3_l2 | 3 | 2.0 | [-0.298, 0, -0.240] | 0.9329 | 1.0367 | **-8.9%** | +1.3% |
| lyap_d3_l3 | 3 | 3.0 | [-0.213, 0, -0.195] | 0.9575 | 1.0324 | **-6.5%** | +0.8% |
| lyap_d3_l4 | 3 | 4.0 | [-0.163, 0, -0.163] | 0.9723 | 1.0304 | **-5.0%** | +0.6% |
| lyap_d3_l5 | 3 | 5.0 | [-0.128, 0, -0.142] | 0.9820 | 1.0294 | **-4.1%** | +0.5% |
| lyap_d4_l1 | 4 | 1.0 | [0, -0.202, 0, -0.281] | 0.8839 | 1.0753 | **-13.7%** | +5.0% |
| lyap_d4_l2 | 4 | 2.0 | [0, -0.150, 0, -0.184] | 0.9161 | 1.0514 | **-10.5%** | +2.7% |
| lyap_d4_l3 | 4 | 3.0 | [0, -0.118, 0, -0.136] | 0.9376 | 1.0424 | **-8.4%** | +1.8% |
| lyap_d4_l4 | 4 | 4.0 | [0, -0.096, 0, -0.108] | 0.9521 | 1.0373 | **-7.0%** | +1.3% |
| **lyap_d4_l5** | **4** | **5.0** | **[0, -0.081, 0, -0.089]** | **0.9624** | **1.0341** | **-6.0%** | **+1.0%** |

### Chebyshev (Pure, No Regularization) Comparison

| Config | Degree | Coefficients | Oracle MASE Geo | Self MASE Geo | Oracle % | Self % |
|--------|--------|-------------|:---:|:---:|:---:|:---:|
| chebyshev_d3 | 3 | [-1.5, 0, 0.75] | 0.8030 | 1.1397 | **-21.6%** | +11.3% |
| chebyshev_d4 | 4 | [0, -2.0, 0, 1.125] | 0.8132 | 1.2427 | **-20.6%** | +21.4% |
| chebyshev_d5 | 5 | [0, 2.5, 0, -3.125, 0, 1.5625] | 0.8713 | 1.4341 | **-14.9%** | +40.1% |

### Key Insight: Oracle vs Self Gap

The gap between oracle and self reversal grows with coefficient magnitude:

| Config | |c_max| | Oracle-Self Gap |
|--------|:---:|:---:|
| lyap_d3_l5 | 0.142 | 4.6% |
| lyap_d4_l5 | 0.089 | 7.0% |
| lyap_d4_l1 | 0.281 | 18.7% |
| chebyshev_d3 | 1.500 | 32.9% |
| chebyshev_d5 | 3.125 | 55.0% |

**Smaller coefficients = smaller oracle-self gap = more practical.**

---

## 2. AK's Moirai2 Pretrained Models (20K Steps, LOTSA v1)

AK trained 24 Moirai2 configurations but did NOT evaluate on GIFT-Eval. Key configs:

| Run Name | Type | Degree | Init Coeffs | FIR Inverse | Notes |
|----------|------|--------|-------------|:-----------:|-------|
| moirai2_baseline | Baseline | - | - | No | z_score=8 |
| moirai2_regular5_z8_fir64_lambda01 | Lyap d=4 λ=5 | 5 | [0, -0.081, 0, -0.089] | Yes (λ=0.1, L=64) | Best fixed-coeff config |
| moirai2_regular3_z8_fir64_lambda01 | Lyap d=4 λ=3 | 5 | varies | Yes (λ=0.1, L=64) | Mid-strength |
| moirai2_regular1_z8_fir64_lambda01 | Lyap d=4 λ=1 | 5 | varies | Yes (λ=0.1, L=64) | Low reg |
| moirai2_d3r5_z8_fir64_lambda01 | Lyap d=3 λ=5 | 3 | varies | Yes (λ=0.1, L=64) | Lower degree |
| moirai2_learnable_d4_z8_fir64 | Learnable d=4 | 4 | Chebyshev | Yes (λ=0.1, L=64) | Free learning |
| moirai2_learnable_d4_z8_fir64_lambda015 | Learnable d=4 | 4 | Chebyshev | Yes (λ=0.15, L=64) | Higher FIR weight |
| moirai2_learnable_d4_z8_fir64_lambda02 | Learnable d=4 | 4 | Chebyshev | Yes (λ=0.2, L=64) | Higher FIR weight |
| moirai2_learnable_d4_z8_fir64_lambda1 | Learnable d=4 | 4 | Chebyshev | Yes (λ=1.0, L=64) | Max FIR weight |
| moirai2_regular5_z8_reverse01 | Lyap d=4 λ=5 + reverse | 5 | [0, -0.081, 0, -0.089] | No | Loss reversal (λ=0.1) |
| moirai2_chebyshev_z8_fir64_lambda01 | Pure Chebyshev | 5 | Chebyshev coeffs | Yes (λ=0.1, L=64) | Unregularized |

All runs: 20K steps, batch=256, 100 batches/epoch, bf16-mixed, anomaly_zscore=8.0, stride=1.

**AK's `regular5` is his best configuration**: Lyapunov d=4 lambda=5 coefficients [0, -0.081, 0, -0.089], with FIR inverse filter (length 64, aux loss lambda=0.1).

---

## 3. Our Reproduction on Moirai2 (LOTSA v1 Unweighted)

### Baseline vs Learnable Preconditioning

| Experiment | Steps | Stride | Degree | Learned Coefficients | FEV-Bench MASE Geo | GIFT-Eval MASE Geo | vs Baseline |
|------------|:-----:|:------:|:------:|---------------------|:---:|:---:|:---:|
| 10K baseline | 10K | - | - | - | 1.2908 | 1.2395 | — |
| 25K baseline | 25K | - | - | - | 1.2869 | 1.2422 | — |
| 10K learnable (s=1, free) | 10K | 1 | 4 | [-0.667, -0.126, -0.056, -0.045] | 1.3653 | 1.3428 | +8.3% |
| 25K learnable (s=1, free) | 25K | 1 | 4 | [-0.667, -0.126, -0.056, -0.045] | 1.3422 | 1.3978 | +12.5% |
| 25K stride=16 d=4 | 25K | 16 | 4 | [-0.115, -0.039, -0.031, -0.001] | 1.2907 | pending | **+0.3%** |
| 25K stride=16 d=1 | 25K | 16 | 1 | [-0.128] | 1.2931 | pending | +0.5% |
| 25K strongL2 (s=1, λ=50) | 25K | 1 | 4 | [-0.001, -0.001, -0.001, -0.001] | 1.3268 | — | +3.1% |

### Fixed Chebyshev Coefficients (EXP-1, 100K Steps)

From the degree sweep experiment (100K steps, fixed coefficients, stride=1):

| Degree | Coefficients | GIFT-Eval MASE Geo | vs Baseline |
|:------:|-------------|:---:|:---:|
| d=1 (identity control) | [0.0] | 1.3147 | +5.8% |
| d=2 | [0.0, -0.5] | 1.4260 | +14.8% |
| d=3 | [-1.5, 0.0, 0.75] | 1.5682 | +26.2% |
| d=4 | [0.0, -2.0, 0.0, 1.125] | 1.7050 | +37.3% |
| d=5 | [0.0, 2.5, 0.0, -3.125, 0.0] | 1.8930 | +52.4% |
| d=6 | [-4.5, 0.0, 6.75, 0.0, -4.5, 0.0] | 2.1520 | +73.3% |
| d=7 | [0.0, 7.0, 0.0, -14.0, 0.0, 10.5, 0.0] | 2.4180 | +94.7% |

**Result**: Fixed Chebyshev coefficients are catastrophic. Higher degree = worse. Confirms AK's Chronos2 finding: self-reversal error grows with coefficient magnitude.

---

## 4. Coefficient Comparison: AK's Lyapunov vs Our Learned

| Source | Type | Stride | Coefficients | Coefficient Norm |
|--------|------|:------:|-------------|:---:|
| AK Lyapunov d=4 λ=5 | Fixed optimized | 1 | [0, -0.081, 0, -0.089] | 0.120 |
| AK Lyapunov d=4 λ=1 | Fixed optimized | 1 | [0, -0.202, 0, -0.281] | 0.346 |
| Our learnable s=1 (25K) | Learned freely | 1 | [-0.667, -0.126, -0.056, -0.045] | 0.686 |
| Our learnable s=16 d=4 (25K) | Learned freely | 16 | [-0.115, -0.039, -0.031, -0.001] | 0.124 |
| Our learnable s=16 d=1 (25K) | Learned freely | 16 | [-0.128] | 0.128 |
| Pure Chebyshev d=4 | Fixed analytical | 1 | [0, -2.0, 0, 1.125] | 2.295 |

**Key observation**: Our stride=16 learned coefficients (norm 0.124) converge to nearly identical magnitude as AK's Lyapunov d=4 λ=5 (norm 0.120). This confirms that:
1. The optimal filter is near-identity (small coefficients)
2. Stride=16 naturally achieves the same mild filtering as explicit Lyapunov regularization
3. Both approaches converge to similar solutions from different starting points

---

## 5. Mapping AK's Best Config to Our Framework

**AK's best config** (`moirai2_regular5_z8_fir64_lambda01`):
```
degree=5 (in config, but 4 coefficients provided)
coeffs_init=[0.0, -0.08103764, 0.0, -0.0889486]
stride=1
FIR inverse: enabled, length=64, stride=1, aux_loss_lambda=0.1
anomaly_zscore=8.0
learnable=false (fixed coefficients)
```

**Equivalent in our framework**:
```bash
model.module_kwargs.time_precondition_enabled=true
model.module_kwargs.time_precondition_learnable=false
model.module_kwargs.time_precondition_type=lyapunov
model.module_kwargs.time_precondition_degree=4
model.module_kwargs.time_precondition_reg_lambda=5.0
model.module_kwargs.time_precondition_stride=1
model.module_kwargs.time_precondition_inverse_enabled=true
model.module_kwargs.time_precondition_inverse_length=64
model.module_kwargs.time_precondition_inverse_stride=1
model.time_precondition_inverse_lambda=0.1
model.anomaly_zscore_threshold=8.0
```

---

## 6. Experiments In Progress

### Currently Running (as of 2026-02-19):

| Job | Experiment | Status | Notes |
|-----|-----------|--------|-------|
| 4897046 | Quick 10K baseline | Running (Epoch ~17) | ailab |
| 4897047 | Quick 10K stride=16 d=4 | Running (Epoch ~16) | ailab |
| 4897048 | Quick 10K stride=16 warm-start | Running | ailab |
| 4897054 | Quick 10K stride=16 d=2 | Running | pli |
| 4897055 | Quick 10K stride=8 d=4 | Running | pli |
| 4897056 | 50K stride=16 d=4 | Running | pli |
| 4897057 | 25K stride=16 warm-start | Running | pli |
| 4897201 | Quick 10K dual loss | Running (Epoch ~11) | pli |
| 4897601 | Quick 10K loss reversal | Submitted | pli |
| 4897602 | Quick 10K fine-tune s16 | Submitted | pli |
| 4897610 | Quick 10K fine-tune s16 + lossrev | Submitted | pli |
| 4896995 | GIFT-Eval stride=16 d=4 25K | Running | pli |
| 4896996 | GIFT-Eval stride=16 d=1 25K | Running | pli |

### Key Questions to Answer:

1. **Does stride=16 beat baseline on GIFT-Eval?** (jobs 4896995/4896996)
2. **Does loss reversal eliminate the train/eval mismatch?** (job 4897601)
3. **Does fine-tuning from baseline converge faster?** (jobs 4897602/4897610)
4. **What's the optimal stride-degree combination?** (jobs 4897054/4897055)

---

## 7. What We Need to Replicate from AK

To fully replicate AK's best `regular5` config in our setup, we should run:

1. **Fixed Lyapunov d=4 λ=5 with FIR inverse (stride=1)** — AK's exact config
2. **Same but stride=16** — our best improvement
3. **Same but with loss reversal** — to address the oracle-self gap
4. **Same but fine-tuned from baseline** — to reuse pretrained weights

The fine-tuning and loss reversal experiments (jobs 4897601, 4897602, 4897610) are already submitted.
