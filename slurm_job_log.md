# SLURM Job Log

This file tracks all SLURM jobs submitted for this project.

---

## Trained Models

| Model | Run Date | Epochs | Checkpoint Path | Config |
|-------|----------|--------|-----------------|--------|
| Baseline MOIRAI Small (prev) | 2026-01-25 | 100 | `uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/moirai_small_baseline_20260125_164605/checkpoints/epoch_epoch_0099.ckpt` | moirai_small, lotsa_v1_unweighted |
| STU-MOIRAI Small (prev) | 2026-01-25 | 100 | `uni2ts/outputs/pretrain/moirai_small_stu/lotsa_v1_unweighted/moirai_small_stu_20260125_164605/checkpoints/epoch_epoch_0099.ckpt` | moirai_small_stu, lotsa_v1_unweighted |
| Baseline MOIRAI Small (new) | 2026-01-26 | 1000 | `uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/moirai_small_baseline_20260126_163112/checkpoints/epoch_epoch_0999.ckpt` | moirai_small, lotsa_v1_unweighted, 100 batches/epoch, bs=128 |
| STU-MOIRAI Small (new) | 2026-01-26 | 599+ (running) | `uni2ts/outputs/pretrain/moirai_small_stu/lotsa_v1_unweighted/moirai_small_stu_20260126_163112/checkpoints/epoch_epoch_0599.ckpt` | moirai_small_stu, lotsa_v1_unweighted, 100 batches/epoch, bs=128 |
| STU-MOIRAI Fast | 2026-01-27 | 100 (running) | `uni2ts/outputs/pretrain/moirai_small_stu/lotsa_v1_unweighted/moirai_small_stu_fast_20260127_181919/checkpoints/epoch_epoch_0099.ckpt` | moirai_small_stu + forward_batched (2x faster) |
| STU-Only Sandwich | 2026-01-27 | pending | TBD | moirai_small_stu_only, 6 STU layers + MLP sandwiching (14.3M params) |
| Hybrid Sandwich | 2026-01-27 | pending | TBD | moirai_small_stu_sandwich, alternating + sandwiching (15.7M params) |

### Architecture Variants Summary

| Variant | Pattern | Sandwiching | Params | Description |
|---------|---------|-------------|--------|-------------|
| Baseline MOIRAI | 6 attn layers | No | 13.83M | Standard transformer encoder |
| Current Hybrid | 3 STU + 3 attn (alt) | No | 12.53M | 50% attention replaced with STU |
| STU-Only Sandwich | 6 STU layers | Yes (512 hidden) | 14.29M | All STU layers with MLP sandwich |
| Hybrid Sandwich | 3 STU + 3 attn (alt) | Yes (768 hidden) | 15.65M | Alternating with MLP sandwich on STU |
| **Multi-Head STU** | 3 MH-STU + 3 attn (alt) | No | 13.83M | H=6 heads, wider FFN (d_ff=1379) |
| **Non-Approx STU** | 3 full-STU + 3 attn (alt) | No | 13.83M | K=2, full M_phi[K,d,d] (590K mixing params) |
| **Parallel STU+Attn** | 6 parallel layers | No | 13.83M | Both STU+Attn per layer, d_ff=888, learned gate |

**MLP Sandwiching** (from Flash STU paper): Wraps STU with up/down projections:
- `Input -> UpProject -> Activate -> STU -> DownProject -> Output`
- Allows STU to operate in higher-dimensional space for increased expressiveness

---

## Active/Recent Jobs

| Job ID | Name | Type | Status | Submitted | Checkpoint/Input | Output Path |
|--------|------|------|--------|-----------|------------------|-------------|
| Job ID | Name | Type | Status | Submitted | Checkpoint/Input | Output Path |
|--------|------|------|--------|-----------|------------------|-------------|
| **EXP-1: Chebyshev Degree Sweep** | | | | | | |
| 4583433 | m2_baseline | Pretraining | **COMPLETED** (confounded) | 2026-02-08 | Moirai2 baseline, 100K steps, bs=256. **CONFOUNDED**: missing zscore=8.0 | `uni2ts/logs/m2_baseline_4583433.out` |
| 4583434 | m2_precond d=1 | Pretraining | **COMPLETED** | 2026-02-08 | Chebyshev d=1 (identity), FIR inv | `uni2ts/logs/m2_precond_d*_4583434.out` |
| 4583435 | m2_precond d=2 | Pretraining | **COMPLETED** | 2026-02-08 | Chebyshev d=2, FIR inv | `uni2ts/logs/m2_precond_d*_4583435.out` |
| 4583436 | m2_precond d=3 | Pretraining | **COMPLETED** | 2026-02-08 | Chebyshev d=3, FIR inv | `uni2ts/logs/m2_precond_d*_4583436.out` |
| 4583437 | m2_precond d=4 | Pretraining | **COMPLETED** | 2026-02-08 | Chebyshev d=4, FIR inv | `uni2ts/logs/m2_precond_d*_4583437.out` |
| 4583438 | m2_precond d=5 | Pretraining | **COMPLETED** | 2026-02-08 | Chebyshev d=5, FIR inv | `uni2ts/logs/m2_precond_d*_4583438.out` |
| 4583439 | m2_precond d=6 | Pretraining | **COMPLETED** | 2026-02-08 | Chebyshev d=6, FIR inv | `uni2ts/logs/m2_precond_d*_4583439.out` |
| 4583440 | m2_precond d=7 | Pretraining | **COMPLETED** | 2026-02-08 | Chebyshev d=7, FIR inv | `uni2ts/logs/m2_precond_d*_4583440.out` |
| **EXP-1b: d=4 Regularization Sweep (all COMPLETED, zscore=8.0)** | | | | | | |
| 4611086 | m2_baseline_fixed | Pretraining | **COMPLETED** | 2026-02-09 | Moirai2 baseline, zscore=8.0 | `m2_baseline_20260209_114203/` |
| 4610803 | m2_precond 4-tap custom | Pretraining | **COMPLETED** | 2026-02-09 | [0,-0.1176,0,-0.1361] | `m2_precond_d4_custom_20260209_111608/` |
| 4611080 | m2_d4_lam0.25 | Pretraining | **COMPLETED** | 2026-02-09 | [0,-0.70543,0,-0.08509] | `...112351/epoch_999-step_100000.ckpt` |
| 4611081 | m2_d4_lam0.5 | Pretraining | **COMPLETED** | 2026-02-09 | [0,-0.50746,0,-0.21409] | `...112351/epoch_999-step_100000-v3.ckpt` |
| 4611082 | m2_d4_lam1.0 | Pretraining | **COMPLETED** | 2026-02-09 | [0,-0.37729,0,-0.29357] | `...112351/epoch_999-step_100000-v1.ckpt` |
| 4611083 | m2_d4_lam2.0 | Pretraining | **COMPLETED** | 2026-02-09 | [0,-0.25,0,-0.25] | `...112351/epoch_999-step_100000-v2.ckpt` |
| 4611084 | m2_d4_lam3.0 | Pretraining | **COMPLETED** | 2026-02-09 | [0,-0.16667,0,-0.16667] | `...112351/epoch_999-step_100000-v4.ckpt` |
| 4611085 | m2_d4_lam10.0 | Pretraining | **COMPLETED** | 2026-02-09 | [0,-0.05,0,-0.05] | `m2_precond_d4_custom_20260209_113038/` |
| **EXP-1b: GIFT-Eval (full 97 configs, pli)** | | | | | | |
| 4632564-71 | ge_1b_* (old) | GIFT-Eval Full | **CANCELLED** | 2026-02-10 | context=1000 (wrong), replaced by ctx=4000 runs | — |
| **EXP-1a+1b: GIFT-Eval Re-eval (ctx=4000, pli)** | | | | | | |
| 4634793 | ge_baseline | GIFT-Eval Full | PENDING | 2026-02-10 | baseline ep999 (zscore=8.0), ctx=4000 | `logs/ge_baseline_4634793.out` |
| 4634794 | ge_d1 | GIFT-Eval Full | PENDING | 2026-02-10 | Cheb d=1 [0.0], ctx=4000 | `logs/ge_d1_4634794.out` |
| 4634795 | ge_d2 | GIFT-Eval Full | PENDING | 2026-02-10 | Cheb d=2, ctx=4000 | `logs/ge_d2_4634795.out` |
| 4634796 | ge_d3 | GIFT-Eval Full | PENDING | 2026-02-10 | Cheb d=3, ctx=4000 | `logs/ge_d3_4634796.out` |
| 4634797 | ge_d4 | GIFT-Eval Full | PENDING | 2026-02-10 | Cheb d=4, ctx=4000 | `logs/ge_d4_4634797.out` |
| 4634798 | ge_d5 | GIFT-Eval Full | PENDING | 2026-02-10 | Cheb d=5, ctx=4000 | `logs/ge_d5_4634798.out` |
| 4634799 | ge_d6 | GIFT-Eval Full | PENDING | 2026-02-10 | Cheb d=6, ctx=4000 | `logs/ge_d6_4634799.out` |
| 4634800 | ge_d7 | GIFT-Eval Full | PENDING | 2026-02-10 | Cheb d=7, ctx=4000 | `logs/ge_d7_4634800.out` |
| 4634801 | ge_4tap | GIFT-Eval Full | PENDING | 2026-02-10 | 4-tap [0,-0.1176,0,-0.1361], ctx=4000 | `logs/ge_4tap_4634801.out` |
| 4634802 | ge_lam025 | GIFT-Eval Full | PENDING | 2026-02-10 | lam=0.25, ctx=4000 | `logs/ge_lam025_4634802.out` |
| 4634803 | ge_lam1 | GIFT-Eval Full | PENDING | 2026-02-10 | lam=1.0, ctx=4000 | `logs/ge_lam1_4634803.out` |
| 4634804 | ge_lam2 | GIFT-Eval Full | PENDING | 2026-02-10 | lam=2.0, ctx=4000 | `logs/ge_lam2_4634804.out` |
| 4634805 | ge_lam05 | GIFT-Eval Full | PENDING | 2026-02-10 | lam=0.5, ctx=4000 | `logs/ge_lam05_4634805.out` |
| 4634806 | ge_lam3 | GIFT-Eval Full | PENDING | 2026-02-10 | lam=3.0, ctx=4000 | `logs/ge_lam3_4634806.out` |
| 4634807 | ge_lam10 | GIFT-Eval Full | PENDING | 2026-02-10 | lam=10.0, ctx=4000 | `logs/ge_lam10_4634807.out` |
| **STU Architecture Variants** | | | | | | |
| 4558185 | pretrain_multihead_stu | Pretraining | **TIMEOUT** (hit 2d8h wall) | 2026-02-07 | Multi-Head STU (H=6, d_ff=1379, 13.83M params) | `uni2ts/logs/pretrain_multihead_stu_4558185.out` |
| 4558186 | pretrain_nonapprox_stu | Pretraining | **COMPLETED** | 2026-02-07 | Non-Approx STU (K=2, full M_phi, 13.83M params) | `uni2ts/logs/pretrain_nonapprox_stu_4558186.out` |
| 4558187 | pretrain_parallel_stu | Pretraining | **COMPLETED** | 2026-02-07 | Parallel STU+Attn (d_ff=888, gate, 13.83M params) | `uni2ts/logs/pretrain_parallel_stu_4558187.out` |
| 4620192 | gifteval_nonapprox_stu | GIFT-Eval Full | **COMPLETED** | 2026-02-09 | Non-Approx STU ep999, **Geo MASE=1.3359** | `gifteval/results/report_epoch_epoch_0999_20260209_165950.md` |
| **Hint Mode Degree Sweep (10K steps, s=16)** | | | | | | |
| 4919073 | q_hintd7 | Pretraining | **COMPLETED** | 2026-02-20 | Hint d=7 s=16, 10K steps | `q_hint_s16d7_20260220_100857/checkpoints/epoch_99-step_10000.ckpt` |
| 4932905 | q_hintd4 | Pretraining | **COMPLETED** | 2026-02-20 | Hint d=4 s=16, 10K steps | `q_hint_s16d4_20260220_114346/checkpoints/epoch_99-step_10000.ckpt` |
| 4919074 | q_hints8 | Pretraining | **COMPLETED** | 2026-02-20 | Hint s=8 d=5, 10K steps (stride ablation) | `q_hint_s8d5_20260220_114241/checkpoints/epoch_99-step_10000.ckpt` |
| — | q_hint_s16d3 | Pretraining | **COMPLETED** | 2026-02-20 | Hint d=3 s=16, 10K steps, **MASE 1.2040 (-3.07%)** | `q_hint_s16d3_20260220_071659/checkpoints/epoch_99-step_10000.ckpt` |
| — | q_hint_s16d2 | Pretraining | **COMPLETED** | 2026-02-20 | Hint d=2 s=16, 10K steps, MASE 1.2157 (-2.13%) | `q_hint_s16d2_20260220_040343/checkpoints/epoch_99-step_10000.ckpt` |
| — | m2_hint_s16 | Pretraining | **COMPLETED** | 2026-02-19 | Hint d=5 s=16, 10K steps, MASE 1.2084 (-2.71%) | `m2_hint_s16_20260219_112524/checkpoints/epoch_99-step_10000.ckpt` |
| **Hint Mode 25K Training** | | | | | | |
| — | m2_hint_s16_25k | Pretraining | **COMPLETED** | 2026-02-20 | Hint d=5 s=16, 25K steps | `m2_hint_s16_25k_20260219_190429/checkpoints/epoch_249-step_25000.ckpt` |
| **Hint Mode GIFT-Eval (pli/hazan_intern)** | | | | | | |
| 4936772 | ge_d7 | GIFT-Eval Full | **COMPLETED** | 2026-02-20 | Hint d=7 s=16 10K, **MASE 1.2027** | — |
| 4936737 | ge_d4 | GIFT-Eval Full | **COMPLETED** | 2026-02-20 | Hint d=4 s=16 10K, **MASE 1.1944 (BEST)** | — |
| 4936739 | ge_s8d5 | GIFT-Eval Full | **COMPLETED** | 2026-02-20 | Hint s=8 d=5 10K, **MASE 1.2247** | — |
| 4936740 | ge_25k | GIFT-Eval Full | **COMPLETED** | 2026-02-20 | Hint d=5 s=16 25K, **MASE 1.2452** | — |
| **Hint Mode 25K/100K Training** | | | | | | |
| — | m2_hint_d3_25k | Pretraining | **COMPLETED** | 2026-02-20 | Hint d=3 s=16, 25K steps | `m2_hint_d3_25k_20260220_160459/checkpoints/epoch_249-step_25000.ckpt` |
| 4995249 | m2_hint100k | Pretraining | PENDING (ailab) | 2026-02-22 | Hint d=4 s=16, 100K steps | — |
| **New Coefficient Experiments (2026-02-20, all COMPLETED)** | | | | | | |
| — | q_hd3l | Pretraining | **COMPLETED** | 2026-02-20 | Hint d=3 learnable | `q_hint_d3_learn_20260220_161221/checkpoints/epoch_99-step_10000.ckpt` |
| — | q_hc06 | Pretraining | **COMPLETED** | 2026-02-20 | Hint coeff 0.6 at lag 1s | `q_hint_c06_20260220_161222/checkpoints/epoch_99-step_10000.ckpt` |
| — | q_hc08 | Pretraining | **COMPLETED** | 2026-02-20 | Hint coeff 0.8 at lag 1s | `q_hint_c08_20260220_162544/checkpoints/epoch_99-step_10000.ckpt` |
| — | q_hc09 | Pretraining | **COMPLETED** | 2026-02-20 | Hint coeff 0.9 at lag 1s | `q_hint_c09_20260220_162544/checkpoints/epoch_99-step_10000.ckpt` |
| — | q_hc10 | Pretraining | **COMPLETED** | 2026-02-20 | Hint coeff 1.0 at lag 1s | `q_hint_c10_20260220_163252/checkpoints/epoch_99-step_10000.ckpt` |
| **GIFT-Eval for Coefficient Experiments (2026-02-22, della/gpu)** | | | | | | |
| 4995268 | ge_hint_d3_25k | GIFT-Eval Full | PENDING | 2026-02-22 | Hint d=3 25K | — |
| 4995269 | ge_hc06 | GIFT-Eval Full | PENDING | 2026-02-22 | Hint coeff 0.6 | — |
| 4995270 | ge_hc08 | GIFT-Eval Full | PENDING | 2026-02-22 | Hint coeff 0.8 | — |
| 4995271 | ge_hc09 | GIFT-Eval Full | PENDING | 2026-02-22 | Hint coeff 0.9 | — |
| 4995272 | ge_hc10 | GIFT-Eval Full | PENDING | 2026-02-22 | Hint coeff 1.0 | — |
| 4995273 | ge_hd3l | GIFT-Eval Full | PENDING | 2026-02-22 | Hint d=3 learnable | — |
| **Completed** | | | | | | |
| 4583584 | gifteval_m2_base_full | GIFT-Eval Full | COMPLETED | 2026-02-08 | Moirai2 baseline 200-step ckpt, 97 configs | `gifteval/results/` |
| 4583585 | gifteval_m2_precond_full | GIFT-Eval Full | COMPLETED | 2026-02-08 | Moirai2 precond 200-step ckpt, 97 configs | `gifteval/results/` |
| 4576433 | moirai2_full | Pretraining | COMPLETED | 2026-02-08 | Moirai2 Small full paper specs, 100K steps | `uni2ts/logs/moirai2_small_full_4576433.out` |
| 4583019 | m2_quick_base | Quick Test | COMPLETED | 2026-02-08 | Baseline 200 steps, loss 0.540→0.109 | |
| 4583020 | m2_quick_precond | Quick Test | COMPLETED | 2026-02-08 | Precond 200 steps, loss 0.447→0.082 | |
| 4235319 | pretrain_stu_only | Pretraining | COMPLETED | 2026-01-27 21:50 | STU-only + sandwiching (14.3M params) | `uni2ts/logs/pretrain_stu_only_4235319.out` |
| 4235320 | pretrain_stu_sandwich | Pretraining | COMPLETED | 2026-01-27 21:50 | Hybrid + sandwiching (15.7M params) | `uni2ts/logs/pretrain_stu_sandwich_4235320.out` |
| 4229720 | pretrain_stu_fast | Pretraining | COMPLETED | 2026-01-27 18:18 | moirai_small_stu + forward_batched (2x faster) | `uni2ts/logs/pretrain_stu_fast_4229720.out` |
| 4184963 | pretrain_stu | Pretraining | COMPLETED (pli) | 2026-01-26 16:31 | moirai_small_stu config (slow _forward_packed) | `uni2ts/logs/pretrain_stu_4184963.out` |
| 4184956 | pretrain_baseline | Pretraining | COMPLETED | 2026-01-26 16:31 | moirai_small config | `uni2ts/logs/pretrain_baseline_4184956.out` |

### Full GIFT-Eval Jobs (97 configs) - With Markdown Reports

| Job ID | Name | Type | Status | Submitted | Checkpoint | Output Path |
|--------|------|------|--------|-----------|------------|-------------|
| 4307507 | gifteval_baseline | GIFT-Eval Full | PENDING (pli) | 2026-01-29 | moirai_small baseline ep999 (13.83M) | `logs/gifteval_full_4307507.out` |
| 4307508 | gifteval_stu | GIFT-Eval Full | PENDING (pli) | 2026-01-29 | moirai_small_stu ep999 (12.53M) | `logs/gifteval_full_4307508.out` |
| 4307510 | gifteval_sandwich | GIFT-Eval Full | PENDING (pli) | 2026-01-29 | moirai_small_stu_sandwich ep999 (15.65M) | `logs/gifteval_full_4307510.out` |

**Output includes:** CSV results, leaderboard-format CSV, config.json, and **markdown report** with model info + aggregate metrics.

#### Previous Jobs (for reference)

| Job ID | Name | Type | Status | Checkpoint |
|--------|------|------|--------|------------|
| 4235662 | gifteval_full | GIFT-Eval Full | COMPLETED | baseline ep999 |
| 4235663 | gifteval_full | GIFT-Eval Full | COMPLETED | STU ep599 |
| 4235661 | gifteval_full | GIFT-Eval Full | COMPLETED | STU ep099 |

### Completed Quick GIFT-Eval Jobs (8 configs)

| Job ID | Model | Epochs | Mean MASE (arith) | Geo Mean MASE | Notes |
|--------|-------|--------|-------------------|---------------|-------|
| 4233824 | Baseline | 100 | 7.63 | 2.19 | moirai_small_baseline_20260125 |
| 4233826 | Baseline | 1000 | 5.86 | **1.75** | moirai_small_baseline_20260126 (best) |
| 4233825 | STU | 100 | 9.20 | 2.33 | moirai_small_stu_20260125 |
| 4233827 | STU | ~590 | 6.06 | 1.78 | moirai_small_stu_20260126 |

**Key observations (Quick Eval - 8 datasets):**
- Both models improve significantly with more training
- Baseline 1000ep (1.75) slightly outperforms STU ~590ep (1.78)
- STU has higher variance on outlier datasets (covid_deaths)
- Geometric mean MASE < 2 indicates better than naive baseline overall

---

## Cancelled/Failed Jobs

| Job ID | Name | Status | Reason |
|--------|------|--------|--------|
| 4228811-4228814 | gifteval_* | CANCELLED | Replaced with ailab partition jobs |
| 4228872-4228876 | gifteval_* | FAILED | Incorrect dataset names in eval script (fixed) |
| 4229358-4229362 | gifteval_* | FAILED | to_univariate=True broke univariate datasets (fixed) |
| 4234283-4234285 | gifteval_full_* | CANCELLED | Resubmitted with updated script (adds CRPS metric) |

---

## Results

Results saved to `/scratch/gpfs/EHAZAN/jh1161/gifteval/results/`

### Quick Eval Summary (8 datasets)

| Model | Epochs | Geo Mean MASE |
|-------|--------|---------------|
| Baseline | 100 | 2.19 |
| Baseline | 1000 | **1.75** |
| STU | 100 | 2.33 |
| STU | ~590 | 1.78 |

### Full Eval Results (97 configs)

| Model | Params | Epochs | Geo Mean MASE | Arith Mean MASE | Beats Naive | W-L vs Baseline | Status |
|-------|--------|--------|---------------|-----------------|-------------|-----------------|--------|
| Baseline | 13.83M | 1000 | **1.3147** | 2.0095 | 33/97 | --- | ✓ Complete |
| Approx STU | 12.53M | 1000 | 1.3172 | 2.0355 | 35/97 | 55-42 | ✓ Complete |
| Sandwich STU | 15.65M | 1000 | **1.3128** | 1.9791 | 36/97 | 45-52 | ✓ Complete |
| Non-Approx STU | 13.83M | 1000 | 1.3359 | 2.0451 | 34/97 | 50-47 | ✓ Complete |
| Multi-Head STU | 13.83M | 1000 | - | - | - | - | Training (ep 774) |
| Parallel STU | 13.83M | 1000 | - | - | - | - | Training (ep 877) |

**Paired bootstrap (95% CI): All STU variants NOT significantly different from baseline.**
- Approx: ratio=1.0018 [0.987, 1.017]
- Sandwich: ratio=0.9986 [0.984, 1.013]
- Non-Approx: ratio=1.0161 [0.997, 1.037]

### GIFT-Eval Leaderboard Methodology

Based on research into the [GIFT-Eval leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval):

1. **Per-config ranking**: For each of 97 dataset configurations, models are ranked 1-N by MASE[0.5]
2. **MASE_Rank**: Arithmetic mean of per-config ranks (primary leaderboard metric)
3. **Geometric Mean MASE**: Alternative aggregation - geometric mean of MASE values across configs
4. **CRPS_Rank**: Same as MASE_Rank but using `mean_weighted_sum_quantile_loss` metric

**Interpretation:**
- MASE < 1.0 = beats seasonal naive baseline
- MASE_Rank ≈ 1 = consistently top performer
- Lower is better for all metrics

---

## Moirai2 Preconditioning Full Leaderboard (2026-02-22)

| MASE | vs Baseline | Experiment | Steps | Config |
|:----:|:-----------:|------------|:-----:|--------|
| **1.1944** | **-3.64%** | q_hint_s16d4 | 10K | hint d=4 s=16 **BEST** |
| 1.2027 | -2.97% | q_hint_s16d7 | 10K | hint d=7 s=16 |
| 1.2040 | -2.86% | q_hint_s16d3 | 10K | hint d=3 s=16 |
| 1.2084 | -2.51% | m2_hint_s16 | 10K | hint d=5 s=16 |
| 1.2157 | -1.92% | q_hint_s16d2 | 10K | hint d=2 s=16 |
| 1.2162 | -1.88% | q_ak_r5s16 | 10K | reversal AK reg5 s=16 |
| 1.2203 | -1.55% | q_hint_s16_learn | 10K | hint d=5 s=16 learnable |
| 1.2227 | -1.36% | q_s16d2 | 10K | reversal d=2 s=16 |
| 1.2247 | -1.19% | q_hint_s8d5 | 10K | hint d=5 s=8 |
| 1.2290 | -0.85% | q_s16d2_learn | 10K | reversal d=2 learnable |
| 1.2303 | -0.74% | q_ft_s16 | 10K | fine-tune s=16 |
| 1.2372 | -0.19% | m2_s16d2_25k | 25K | reversal d=2 s=16 |
| 1.2395 | 0.00% | **m2_baseline_cmp** | 10K | **BASELINE** |
| 1.2422 | +0.22% | m2_baseline_25k | 25K | baseline |
| 1.2452 | +0.46% | m2_hint_s16_25k | 25K | hint d=5 s=16 |
| 1.2878 | +3.90% | m2_baseline | 100K | baseline (no anomaly filter) |

### Active/Pending Jobs (2026-02-25)

| Job ID | Name | Partition | Status | Notes |
|--------|------|-----------|--------|-------|
| 5082176 | ge_eval | ailab | RUNNING | d6_100k eval (GIFT-Eval 97 configs) |

All preconditioning training (10K, 25K, 100K) and most evaluations are complete. Only d6_100k eval remains.

### Cancelled Jobs
| Job ID | Name | Reason |
|--------|------|--------|
| 4996027 | q_hint_robust | NaN losses from robust scaler (step 43+) |
| 4996030 | q_base_robust | Same scaler issue |
| 4995268-73, 4995345-50 | gifteval_full | Broken --export=CHECKPOINT (no value) |

### Flash-STU Hybrid (Spectral + Attention)
| Job ID | Name | Type | Status | Submitted | Details | Output Path |
|--------|------|------|--------|-----------|---------|-------------|
| 5028774 | stu_hybrid_10k | Pretraining | **CANCELLED** | 2026-02-23 | Moved to della-gpu (ailab busy) | -- |
| 5028842 | stu_hybrid_10k | Pretraining | **COMPLETED** | 2026-02-23 | Moirai2 Small + approx STU (K=24, d_ff=940, 11.4M params), 10K steps, ~3h, loss 0.51→0.11 | `logs/stu_hybrid_5028842.out` |
| 5082166 | gifteval_full | Evaluation | **CANCELLED** | 2026-02-25 | Resubmitted (bad --export syntax) | -- |
| 5082318 | gifteval_full | Evaluation | **PENDING** | 2026-02-25 | GIFT-Eval of STU hybrid epoch_99-step_10000.ckpt, ailab | `logs/gifteval_5082318.out` |
| 5082360 | gifteval_full | Evaluation | **PENDING** | 2026-02-25 | GIFT-Eval of STU hybrid epoch_99-step_10000.ckpt, della-gpu (backup) | `logs/gifteval_5082360.out` |
