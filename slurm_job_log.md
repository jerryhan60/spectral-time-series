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
| 4583019 | m2_quick_base | Quick Test | PENDING (ailab) | 2026-02-08 | Moirai2 baseline quick test: 200 steps (20ep x 10bat), test_small, bs=32 | `slurm-m2-quick-base-4583019.out` |
| 4583020 | m2_quick_precond | Quick Test | PENDING (ailab) | 2026-02-08 | Moirai2 precond quick test: 200 steps, Chebyshev deg5, FIR inv len64, λ=0.1 | `slurm-m2-quick-precond-4583020.out` |
| 4576542 | gifteval_moirai2 | GIFT-Eval Full | PENDING (ailab) | 2026-02-08 | Moirai2 Small 10K-step checkpoint (11.39M params), full 97 configs, 12h time limit. Fixed QuantileForecastGenerator unpack bug. | `logs/gifteval_moirai2_4576542.out` |
| 4576438 | gifteval_moirai2 | GIFT-Eval Full | FAILED (ailab) | 2026-02-08 | All 97 evals failed: Moirai2Forecast.forward() returned tensor, QuantileForecastGenerator expected ((preds,), loc, scale) tuple | `logs/gifteval_moirai2_4576438.out` |
| 4576433 | moirai2_full | Pretraining | RUNNING (pli) | 2026-02-08 | Moirai2 Small full paper specs: 100K steps (1000ep x 100bat), bs=256, 10K warmup, lr=1e-3, bf16-mixed, 11.39M params | `uni2ts/logs/moirai2_small_full_4576433.out` |
| 4558185 | pretrain_multihead_stu | Pretraining | RUNNING (ailab) | 2026-02-07 | Multi-Head STU (H=6, d_ff=1379, 13.83M params) | `uni2ts/logs/pretrain_multihead_stu_4558185.out` |
| 4558186 | pretrain_nonapprox_stu | Pretraining | RUNNING (ailab) | 2026-02-07 | Non-Approx STU (K=2, full M_phi, 13.83M params) | `uni2ts/logs/pretrain_nonapprox_stu_4558186.out` |
| 4558187 | pretrain_parallel_stu | Pretraining | RUNNING (ailab) | 2026-02-07 | Parallel STU+Attn (d_ff=888, gate, 13.83M params) | `uni2ts/logs/pretrain_parallel_stu_4558187.out` |
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

| Model | Epochs | Geo Mean MASE | Arith Mean MASE | Beats Naive | Status |
|-------|--------|---------------|-----------------|-------------|--------|
| Baseline | 100 | **1.4870** | 2.2768 | 22/97 | ✓ Complete |
| Baseline | 1000 | - | - | - | Pending |
| STU | 100 | - | - | - | Pending |
| STU | ~600 | - | - | - | Pending |
| STU Fast | 100 | - | - | - | Pending |

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
