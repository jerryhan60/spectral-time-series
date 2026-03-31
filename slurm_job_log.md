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

## OLMo USP Results (2026-03-08)

Experiments on OLMo3-190M (12-layer transformer, d_model=768, 7630 steps, 1B tokens).
All experiments add a pre-attention DV2-style block (depthwise conv + SiLU gating).

### Final Leaderboard (all completed at 7630 steps, avg last 50 CE)
| Rank | Run | Description | CE Loss | vs Baseline |
|------|-----|-------------|---------|-------------|
| **1** | **sel_dx2_k32** | **Selective gating DV2 + k=32** | **2.8398** | **-2.43%** |
| 2 | mix2_dx2 | Mixture of 2 EMA kernels + dual + expand | 2.8463 | -2.21% |
| 3 | layer_dx2 | Layer-adaptive kernels (4→32) + dual + expand | 2.8465 | -2.20% |
| 4 | dv2_expand_2x | DV2 Expand 2x (wider gate path) | 2.8427 | -1.76% |
| 5 | dv2_ema_k16 | DV2 EMA init k=16 | 2.8490 | -1.54% |
| 6 | stacked_dv2_4_16 | Stacked DV2 (k=4 + k=16) | 2.8538 | -1.38% |
| 7 | dv2_poly_r01 | DV2 + Polynomial Residual | 2.8606 | -1.14% |
| 8 | deep_v2_k16 | Deep Conv v2 (standard DV2) | 2.8614 | -1.12% |
| — | baseline_v4 | OLMo3-190M baseline | 2.9105 | — |

### v17+ Running Experiments (2026-03-08)
| Job ID | Run | Step | CE Loss | Δ vs BL | Status |
|--------|-----|------|---------|---------|--------|
| 5506261 | casc_dx2 | 7630 | 2.8447 | -2.26% | **DONE** |
| 5506609 | dx2_k8 | 4718 | 3.0116 | +3.47% | Training (early) |
| 5506543 | gelu_dx2 | 5216 | 2.9641 | +1.84% | Training (early) |
| 5506818 | sel_gelu_dx2 | 4537 | 3.0233 | +3.88% | Training (early) |
| 5506819 | conv_dx2 | 1113 | 4.6986 | — | Training (warmup) |
| 5507208 | ema_gconv_dx2 | 790 | 5.4364 | — | Training (warmup) |

### v18 Experiments (submitted, PENDING)
| Job ID | Run | Description |
|--------|-----|-------------|
| 5489986 | ema_expand_2x | EMA init + 2x expand (combine top two) |
| 5489987 | dv2_ema_k32 | DV2 EMA with k=32 |
| 5489988 | ema_stacked_4_16 | EMA init + stacked (k=4 + k=16) |
| 5489989 | ema_orth_k16 | EMA init + orthogonal projections |
| 5489990 | ema_layerscale_k16 | EMA init + learnable LayerScale |
| 5489991 | ema_rmsnorm_k16 | EMA init + RMSNorm on conv output |
| 5489992 | ema_expand_orth_2x | EMA + Expand + Orth (triple combo) |
| 5489993 | ema_expand_4x | EMA init + 4x expand |

### v19 Experiments (submitted, PENDING)
| Job ID | Run | Description |
|--------|-----|-------------|
| 5490154 | learnable_ema_k16 | Learnable alpha EMA (S4D-style) |
| 5490155 | freq_filter_64 | Frequency-domain filter + gating |
| 5490156 | multi_alpha_ema_3 | Multi-alpha EMA (3 alphas/channel) |
| 5490157 | multi_alpha_ema_5 | Multi-alpha EMA (5 alphas/channel) |

---

## Active/Pending Jobs (updated 2026-03-10 16:00)

### ALL PENDING (cluster maintenance)

| Job ID | Name | Partition | Description | Priority |
|--------|------|-----------|-------------|----------|
| 5553059 | lr_ctrl | pli | LR control 100K: cosine-floor + WSD for BL | CRITICAL |
| 5553061 | hd_lrc | ailab | LR control 100K: cosine-floor + WSD for HD10 | CRITICAL |
| 5560420 | bl_lr2 | pli | Higher-LR BL: LR=2e-3, LR=3e-3 (10K) | HIGH |
| 5560421 | bl_lr2 | ailab | Higher-LR BL backup | HIGH |
| 5553030_[2,7] | hd100_sN | pli | HD10+BL multi-seed 100K (s2,s7) | HIGH |
| 5553031_[2,7] | hd100_sN | ailab | HD10+BL multi-seed 100K backup | HIGH |
| 5560425_[2,7] | ms100_s | pli | MSHD10 multi-seed 100K (s2,s7) | HIGH |
| 5560426_[2,7] | ms100_s | ailab | MSHD10 multi-seed 100K backup | HIGH |
| 5560455_[0,1] | base_sd | pli | Base model (46M) BL+HD10 seeds 0,1 | HIGH |
| 5560456_[0,1] | base_sd | ailab | Base model backup | HIGH |
| 5560316 | bx_2e3 | ailab | OLMo bx_lr2e3 | MEDIUM |
| 5560317 | bl_2e3 | pli | OLMo baseline_lr2e3 | MEDIUM |
| 5560400 | bx15w5 | pli | OLMo bx_lr1.5e3_w500 | MEDIUM |
| 5560401 | bx2w5 | pli | OLMo bx_lr2e3_w500 | MEDIUM |

### COMPLETED
| Job ID | Name | Status | Result |
|--------|------|--------|--------|
| 5560429 | attn_anl | DONE | HD10 more local attention in last layer; figures saved |

---

## ARCHIVE: Previous Active Jobs (2026-03-08 21:30)

### CANCELLED Jobs (confounded by lotsa_v1_unweighted data config)
| Job ID | Name | Status | Reason |
|--------|------|--------|--------|
| 5443329 | s0_100k | CANCELLED | Used lotsa_v1_unweighted + wrong warmup |
| 5443330 | s7_100k | CANCELLED | Used lotsa_v1_unweighted + wrong warmup |
| 5473603 | hd10_s0 | CANCELLED | Used lotsa_v1_unweighted |
| 5473604 | hd10_s7 | CANCELLED | Used lotsa_v1_unweighted |
| 5473605 | s13_res | CANCELLED | Used lotsa_v1_unweighted |

### CLEAN Experiments — lotsa_v1_moirai2 data (RESUBMITTED after config fix)
**BUG FIX**: `variate_proportional` not valid SampleTimeSeriesType. Changed to `proportional` in lotsa_v1_moirai2.yaml.
Old jobs (5474999-5475129) all CANCELLED — would have failed with same error.

| Job ID | Name | Partition/Account | Status | Description |
|--------|------|-------------------|--------|-------------|
| Job ID | Name | Partition/Account | Status | Description |
|--------|------|-------------------|--------|-------------|
| 5475358-60 | core_m2d | various | DONE | baseline+hd10+ms46hd20 seed=0 (10K, moirai2) — all trained |
| 5475361-63 | seeds_m2d | various | DONE | hd10+baseline seeds 1,2 (10K, moirai2) — all trained |
| 5475364-65 | drop_var | pli+ailab | RUNNING (ep95/99) | hd30,sd30,hd10sd20,ms46hd20 — training DONE, 2nd job still running |
| 5475366 | lat_prec | pli | RUNNING | lat_c4s1 DONE, lat_c4s4 at ep9 (very slow on m2d) |
| 5475368 | hint_var | pli | RUNNING | ms46 trained+eval DONE. hd10_drop20 eval at 1/97. hd2,hd3 eval pending |

### Evaluation Jobs — Running
| Job ID | Name | Status | Model | Progress |
|--------|------|--------|-------|----------|
| 5485789 | ev_h10s | RUNNING | hd10_seed0 (unweighted) | 33/97 |
| 5486179 | ev_msm2 | RUNNING | mshd10_m2d_seed0 | 33/97 |
| 5484693 | ev_m2dn | RUNNING | hd10_drop20_m2d_seed0 | 37/97 |
| 5484699 | ev_mshd | RUNNING | mshd10_seed0 (unweighted, dup) | 33/97 |

### Evaluation Jobs — Newly Submitted
| Job ID | Name | Partition | Description |
|--------|------|-----------|-------------|
| 5491028 | ev_m2ds | pli | baseline_m2d_s2, hd10_m2d s0/1/2, mshd10_m2d_s1 |
| 5491029 | ev_m2da | pli | Ablations: hd30,sd30,hd10sd20,ms46hd20,lat_c4s1 (m2d) |
| 5491654 | ev_hd10 | pli | HD10 seeds 1,2 (unweighted) |

### 100K Training — Submitted
| Job ID | Name | Partition | Description |
|--------|------|-----------|-------------|
| 5490688 | mshd10_100k | pli | MSHD10+BL 100K on moirai2 data, seed 0 |
| 5490689 | hd10_100k | pli | HD10 100K on moirai2 data, seed 0 |

### New Ablation — Submitted 2026-03-08
| Job ID | Name | Partition | Description |
|--------|------|-----------|-------------|
| 5495765 | deg_ema | pli | CANCELLED (redundant, primary 5500244 running) |
| 5499998 | deg_ema | della | CANCELLED (redundant, primary 5500244 running) |
| 5495972 | leg_dif | della | Legendre d=6, d=4, diff d=4 — ALL TRAINED. Eval started (leg6 at 2/97) but will timeout at 18:02. |
| 5499448 | str_swp | pli | CANCELLED (redundant) |
| 5499449 | str_swp | della | CANCELLED (redundant, primary 5500228 ahead) |
| 5500003 | hd_bl_s7 | della | **BL_s7: 1.2198. HD10_s7: 1.1698 (-4.11%, 78/97, p=5.8e-10).** DONE. |
| 5500041 | dup_poly | della | CANCELLED (redundant, primary 5500229 completed both seeds) |
| 5500228 | str_swp | della/ailab | Stride sweep DONE: s4=1.1819(-3.0%), s8=1.1977(-1.7%), **s32=1.1929(-2.1%)**. multi_stride eval in progress. |
| 5500229 | dup_poly | della/ailab | dup_hd10 m2d seeds 1,2 — **DONE. s1: 1.2353 (+3.33%), s2: 1.2202 (+0.56%). Pooled: 78/194 (40.2%), p=.004 WORSE than BL** |
| 5500244 | deg_ema | della/ailab | **DONE.** hd6=1.1733(-3.71%), dup_hd10=1.2068(-0.96%), ema_hint=1.1711(-3.89%), **hd8=1.1828(-2.92%)**. |
| 5500668 | base_hd1 | della | **CANCELLED** — stuck on MIG 40GB partition (8/97 after 8h). Resubmitted as ev_base. |
| 5517797 | ev_base | pli | 46M base model eval (BL+HD10), batch=16 to avoid OOM. PENDING. |
| 5517798 | ev_base | ailab | **BL_base DONE: 1.2038. HD10_base DONE: 1.1812 (-1.88%, 60/94, p=0.005)** |
| 5519844 | ev_base_bl | ailab | **DONE. BL_base(46M)=1.1825 (84 valid configs, 13 OOM errors).** |
| 5519845 | ev_base_hd | ailab | **DONE. HD10_base(46M)=1.1584 (-2.04%, 54/84, p=0.012). HD10_small(11.4M)=1.1765 BEATS BL_base(46M)=1.1825.** |
| 5501449 | ms46hd10 | della | MSHD10 m2d — seed 0,1 DONE, seed 2 training epoch 0. (Redundant) |
| 5503509 | ev_msh3 | della/ailab | **DONE.** s0=1.2029, s1=1.1729, s2=1.1876 |
| 5506380 | ev_duph | ailab | **dup_hd10_s0 DONE: 1.2068 (-0.96% ns)**. hd6_s0 eval at [70/97] (redundant). |
| 5507799 | ev_strd | ailab | **ALL DONE. s4=1.1819, s8=1.1977, s32=1.1929** (all re-evals match originals) |
| 5507800 | ev_strd | pli | Stride eval backup on pli |
| 5508460 | ms46_xsd | ailab | **ALL DONE. CORRECTION: 234803 is BL_s7 duplicate (not MS46_s42). Real MS46_s42: 015542=1.1670 (-3.63%, 74/97, p=1e-7).** |
| 5508461 | ms46_xsd | pli | MS46+BL m2d seeds 7,42 backup |
| 5508657 | ds_m2d | ailab | Dual-stride d=4@s16+d=4@s8, 10%drop, m2d — seed0 DONE, seed1 training |
| 5508658 | tri_m2d | ailab | **FAILED**: Hydra parse error on `|` in extra_hints. Need `;` separator. | |
| 5509003 | ev_s7m2 | ailab | **ALL DONE. BL_s7=1.2198, HD10_s7=1.1698** (re-eval matches original) |
| 5516334 | hd10_s42 | ailab | **ALL DONE. BL_s42: 1.2111. HD10_s42: 1.1852 (-2.13%, 61/97, p=0.007)** |
| 5524805 | ev_lat | ailab | **ALL DONE. lat_c4s1: 1.6192 (+33%), lat_c4s4: 1.7126 (+41%). BOTH FAILED.** |
| 5526046 | ev_s42 | ailab | **DONE. BL_s42=1.2111, HD10_s42=1.1852 (-2.13%, 61/97, p=0.014).** |
| 5526061 | ev_s42 | pli | Backup on pli |
| 5525025 | ev_100kc | ailab | **DONE. BL@20K=1.2442, BL@30K=1.1918.** HD10@10K(1.1577) beats BL@30K → 2-3x compute savings |
| 5516335 | hd10_s42 | pli | HD10+BL seed 42 m2d backup (PENDING) |
| 5509070 | ev_legn | ailab | **ALL DONE. Leg4=1.1810 (-3.07%), Leg6=1.1806 (-3.10%).** Leg d=6 ≈ d=4. |
| 5517973 | ev_tcurve | pli | Training curve eval: BL+HD10 at steps 1K, 3K, 5K (PENDING) |
| 5519700 | ev_tcurve | ailab | BL@1K=1.4151, HD10@1K=1.4638, BL@3K=1.2302, HD10@3K=1.2242 DONE. BL@5K at 31/97. Then HD10@5K. |
| 5515047 | ev_basis | ailab | **DONE.** diff4=1.2048(-1.12%, ns), leg4=1.1810(-3.07%), leg6=1.1806(-3.10%) |
| 5515048 | ev_basis | pli | **CANCELLED** (redundant) |
| 5520482 | ev_newm | ailab | **ALL DONE. MS46_s7: 1.2119. DS48_s0 re-eval: 1.1795. MSHD10_s0 re-eval: 1.1880. MSHD10_s1 re-eval: 1.2033** |
| 5520488 | ev_newm | pli | NEW: same as above, backup (PENDING) |
| 5516841 | ev_ctx | pli | Context length ablation: BL+HD10 at ctx=512,1000,2000 (PENDING) |
| 5519701 | ev_ctx | ailab | BL@512=1.3261, HD10@512=1.2634, BL@1000=1.2887, HD10@1000=1.2240, BL@2000=1.2391 DONE. HD10@2000 at 29/97. |
| 5490688 | ms10_100k | pli | **BL 100K DONE (100K steps). Now training MSHD10 100K (just started).** |
| 5490689 | hd10_100k | pli | **HD10 100K at epoch 899 (~90K steps). ETA ~1.5h.** |
| 5508657 | ds_m2d | ailab | **DS48_s0 DONE: 1.1795 (-3.20%, p=0.004). DS48_s1 DONE: 1.1685 (-2.26%, 64/97, p=1.08e-3). Pooled 126/194 (65%), p=1.89e-5** |
| 5528202 | ev_h100c | ailab | **DONE. HD10@20K = 1.1762 (-5.47% vs BL@20K=1.2442, 76/97 wins, p=1.7e-8). Advantage GROWS during training (4.99%→5.47%).** |
| 5528203 | ev_h100c | pli | HD10 100K curve eval backup |
| 5535310-12 | ev_hcurv/h10k | — | **CANCELLED, replaced by comprehensive eval** |
| 5536137 | ev_hfull | pli | Full 100K curve: HD10@30K/40K/50K + BL@40K/50K/60K |
| 5536138 | ev_hfull | ailab | **DONE (timed out after BL@40K started). HD10@30K=1.1775, HD10@40K=1.1970, HD10@50K=1.1770, BL@40K=1.2564.** |
| 5545545 | ev_100r | pli | BL@50K/60K/70K/80K/90K/100K + HD10@60K/70K/80K/90K/100K |
| 5545546 | ev_100r | ailab | Same as above backup |
| 5528222 | mshd_s7 | ailab | **DONE. MSHD10_s7=1.1885 (-2.57%, 73/97, p=6.4e-7).** |
| 5528223 | mshd_s7 | pli | MSHD10 seed 7 backup |
| 5528580 | ev_ms42 | ailab | **DONE. MS46_s42=1.1670 (-3.63%, 74/97, p<0.0001). CORRECTED from old mislabeled file.** |
| 5528581 | ev_ms42 | pli | MS46 seed 42 eval backup |
| 5529171 | mshd42 | ailab | **DONE. MSHD10_s42=1.1794 (-2.61%, 62/97, p=0.008). 5-SEED POOLED: 315/485 (64.9%), p=4.48e-11.** |
| 5529172 | mshd42 | pli | MSHD10 seed 42 backup |
| 5500867 | lrn_fix | pli | **ALL DONE. learned_cheb: 1.1715 (-3.85%, 68/97, p=4.7e-5). learned_zero: 1.1751 (-3.57%). Both beat BL but fixed HD10 (1.1577) slightly better (ns, 51/97, p=0.34).** |
| 5497981 | lat_cont | pli | Latent precond: lat_c6s1 at ep77/100. Init loss 3.08 (6x baseline). Likely FAIL. |

### Learned FIR Coefficients — 2026-03-09 (MAJOR FINDING)
Two models with same architecture (Chebyshev d=4, stride=16, 10% dropout) but different coefficient initialization:

| Coeff | Fixed Chebyshev | Learned (Cheb init) | Learned (Zero init) |
|-------|----------------|---------------------|---------------------|
| c[0] | 0.000 | -0.026 | **+0.308** |
| c[1] | -1.000 | -1.032 | **+0.177** |
| c[2] | 0.000 | **-0.123** | +0.057 |
| c[3] | 0.125 | **0.007** | +0.045 |

- **Cheb-init**: Converged to a DIFFERENCE filter (c[1]≈-1 dominates). Eliminated c[3] (3-stride lookback).
- **Zero-init**: Converged to a SMOOTHING filter (all positive, decreasing). Weighted moving average.
- **Both achieve identical training loss** (0.136). Two distinct filter basins.
- **Coefficients stabilize by step ~7K** (70% of training).
- **Learned (Cheb init) eval: 1.1715 (-3.85% vs BL, 68/97, p=4.7e-5)**
- **vs Fixed HD10: 51/97 (p=0.34, NOT significant). Fixed is 1.20% better in geo mean.**
- **Fixed Chebyshev is a strong inductive bias that slightly outperforms optimization.**
- Learned (Zero init) eval: PENDING (at 29/97).

### Context Length Ablation — 2026-03-09 (COMPLETE with ctx=512!)
| Context | BL      | HD10    | HD10 Δ% | HD10@C beats BL@... |
|---------|---------|---------|---------|---------------------|
| 512     | 1.3261  | 1.2634  | -4.73%  | HD10@512 < BL@1000  |
| 1000    | 1.2887  | 1.2240  | -5.02%  | HD10@1000 < BL@2000 |
| 2000    | 1.2391  | 1.1713  | -5.47%  | HD10@2000 < BL@4000 |
| 4000    | 1.2185  | 1.1577  | -4.99%  | —                   |
**Hint benefit CONSISTENT (~5%) across all context lengths. Hints = 2x free context.**

### Training Curve Eval — 2026-03-09 (in progress)
| Step | BL      | HD10    | HD10 Δ% |
|------|---------|---------|---------|
| 1K   | 1.4151  | 1.4638  | +3.44% (hints HURT early) |
| 3K   | 1.2302  | 1.2242  | -0.49% (neutral, crossover point) |
| 5K   | 1.2295  | 1.2015  | -2.28% (p=0.008, significant!) |
| 10K  | 1.2185  | 1.1577  | -4.99%  |
| 20K  | 1.2442  | 1.1762  | -5.47%  |
| 30K  | 1.1918  | 1.1775  | -1.20% (p=0.15 ns — gap NARROWS!) |

### Stride Sweep — 2026-03-08 (PARTIAL: s4 done, s8/s32/multi pending)
| Stride | MASE | vs BL (1.2185) | vs HD10 s=16 (1.1577) | p vs s16 |
|--------|------|---------------|----------------------|----------|
| s=4 | 1.1819 | -3.00% | +2.09% | 0.0002 (s16 wins 66/97) |
| s=8 | 1.1977 | -1.70% | +3.45% | 0.004 (s16 wins 62/97) |
| s=16 (HD10) | 1.1577 | -4.99% | — | — |
| s=32 | 1.1929 | -2.10% | +3.04% | 0.002 (s16 wins 63/97) |
| multi (s4+s8+s16+s32) | 1.1725 | -3.77% | +1.28% | 0.004 (s16 wins 62/97) |
**Lookback alignment CONFIRMED:** s=16 optimal for 15T (92%), H (77%). s=4 better for 10S (83%).

### Degree Sweep — 2026-03-08 (COMPLETE)
| Degree | MASE | vs BL (1.2185) | Wins | p | Sig |
|--------|------|---------------|------|---|-----|
| d=0 (dup) | 1.2068 | -0.96% | 42/97 | 0.111 | ns |
| **d=4 (HD10)** | **1.1577** | **-4.99%** | **66/97** | **0.0002** | **\*\*\*** |
| d=6 | 1.1733 | -3.71% | 69/97 | 1.9e-5 | \*\*\* |
| d=8 | 1.1828 | -2.92% | 62/97 | 0.004 | \*\* |
**d=4 is optimal.** Monotonic degradation with higher degree: d=4 > d=6 > d=8 >> d=0.
Higher degree adds noise; lower degree lacks information. d=4 matches patch-level temporal resolution.

### Duplicate Channel Ablation — 2026-03-08 (3 seeds on m2d, COMPLETE)
| Seed | dup MASE | vs BL | Wins | p |
|------|---------|-------|------|---|
| 0 | 1.2068 | -0.96% | 42/97 | 0.11 ns |
| 1 | 1.2353 | +3.33% | 34/97 | 0.002 ** |
| 2 | 1.2202 | +0.56% | 44/97 | 0.21 ns |
| **Pooled** | — | — | **120/291 (41.2%)** | **0.002** |
**Duplicate HURTS on m2d.** HD10 wins 206/291 (70.8%) vs duplicate (p<1e-10).
→ Polynomial FIR structure essential, not just extra capacity.

### Completed Inference Ablation — 2026-03-08
| Model | Normal MASE | Zero-hints | Random | Duplicate | Degradation |
|-------|------------|------------|--------|-----------|-------------|
| HD10_m2d_s0 (10%drop) | 1.1577 | 1.2658 (+9.3%) | 1.3254 (+14.5%) | 1.5043 (+29.9%) | Moderate |
| HD10_m2d_s2 (10%drop) | 1.1798 | 1.3687 (+16.0%) | — | — | Higher than s0 |
| MS46_m2d_s0 (0%drop) | 1.1999 | 1.6301 (+35.9%) | — | — | Extreme |
| MS46_m2d_s2 (0%drop) | 1.1711 | 1.6156 (+37.9%) | — | — | Extreme |
| HD2_m2d_s0 (2%drop) | 1.1794 | 1.2648 (+7.2%) | — | — | Low |
| HD30_m2d_s0 (30%drop) | 1.2034 | 1.2329 (+2.5%) | — | — | Minimal |
**Dropout controls dependence: 0%→36%, 2%→7%, 10%→9%, 30%→2.5%**

### Context Length Sweep — 2026-03-08
| Context | HD10_m2d_s0 | BL_m2d_s0 | HD10 Advantage |
|---------|------------|-----------|----------------|
| 1000 | 1.2240 | 1.2887 | -5.02% |
| 2000 | 1.1713 | 1.2391 | -5.47% |
| 4000 | 1.1577 | 1.2185 | -4.99% |
**HD10@ctx=1000 beats BL@ctx=2000 — polynomial hints = 2x free context**

### Completed Soup Eval — 2026-03-08
| Model | MASE | vs BL_m2d_s0 | Wins |
|-------|------|-------------|------|
| soup_hd10_bl_s0 (70%) | 1.3619 | +11.8% | 16/97 |
| soup_hd10_bl_s0_50 (50%) | 1.7174 | +41.0% | 5/97 |
**Both variants catastrophically fail.** Weight averaging destroys co-adaptation.
50% average is WORSE than 70% — confirming deep structural incompatibility.

### Seed Robustness — lotsa_v1_unweighted (ms46+hd10)
| Job ID | Name | Partition/Account | Status | Description |
|--------|------|-------------------|--------|-------------|
| 5475186 | mshd10_sd | pli/eladgroup | PENDING | ms46+hd10 seeds 0,1,2 (10K, lotsa_v1_unweighted) |
| 5475187 | mshd10_sd | ailab/ehazan | PENDING | Same as above, backup |
| 5475188 | mshd10_sd | della/ehazan | PENDING | Same as above, backup |

**Note**: Cancel duplicates once one starts running. Old hd10_m2d jobs (5474313-5474821) CANCELLED — superseded by core_m2d.

### Completed Evals — Key Results (2026-03-08)
| Model | Checkpoint | MASE | Configs | Context |
|-------|-----------|------|---------|---------|
| MSHD10_s2 (unweighted) | mshd10_seed2_10k | **1.2158** | 97 | 4000 |
| BL_m2d_s0 | baseline_m2d_seed0 | **1.2185** | 97 | 4000 |
| MS46_m2d_s0 | ms46_m2d_seed0 | **1.1999** | 97 | 4000 |

### Completed Seed Sensitivity (2026-03-07)
| Job ID | Name | Result | Description |
|--------|------|--------|-------------|
| 5437880 | eval_seeds | See below | Eval 6 seed checkpoints (ms46+baseline, seeds 0,1,2) |
| 5437881 | seed_extra | TIMEOUT (s7 done, s13 partial) | Train seeds 7,13. Seed 7 all done, seed 13 at ep40 |

**10K Seed Sensitivity Results (MS46 vs Baseline, lotsa_v1_unweighted)**:
| Seed | MS46 MASE | Baseline MASE | Δ |
|------|-----------|---------------|---|
| 0 | 1.2369 | 1.2080 | +2.4% (BL wins) |
| 1 | 1.2549 | 1.2500 | +0.4% (BL wins) |
| 2 | 1.2059 | 1.2397 | -2.7% (MS46 wins) |
| 42 (original) | 1.1675 | 1.2422 | -6.0% (MS46 wins) |
| **Mean** | **1.2326±0.020** | **1.2326±0.018** | **0.00%** |
| t-test | | | p=0.999 (NOT significant) |

**Paired Bootstrap Analysis (per-seed, 10K resamples)**:
| Seed | Delta (BL-MS46) | 95% CI | p-value | Win Rate MS46 | Significant? |
|------|-----------------|--------|---------|---------------|-------------|
| 0 | -0.0289 | [-0.060, -0.004] | 0.989 | 38.1% | NO (BL better) |
| 1 | -0.0049 | [-0.047, +0.027] | 0.594 | 57.7% | NO |
| 2 | +0.0338 | [+0.012, +0.057] | 0.001 | 67.0% | YES (MS46 better) |
**Conclusion**: MS46 NOT robust on lotsa_v1_unweighted. Awaiting lotsa_v1_moirai2 results.

**Cross-Seed Robustness (ms46+hd10 with dropout vs ALL 3 baseline seeds)**:
| Model (single seed=42) | vs bl_s0 | vs bl_s1 | vs bl_s2 | Robust? |
|-------------------------|----------|----------|----------|---------|
| **ms46+hd10** (1.1817) | +2.2% p=0.002* | +5.5% p=0.000* | +4.7% p=0.000* | **YES** |
| cross_c4l6 (1.2379) | -2.5% p=0.998 | +1.0% p=0.064 | +0.1% p=0.432 | NO |
| ms46_s0 (1.2369) | -2.4% p=0.989 | +1.1% p=0.140 | +0.2% p=0.400 | NO |
| ms46_s2 (1.2059) | +0.2% p=0.424 | +3.5% p=0.000* | +2.7% p=0.001* | Partial |

**KEY FINDING**: ms46+hd10 (multi-scale Cheb d=4+d=6 + 10% hint dropout) is the ONLY model
that significantly beats ALL 3 baseline seeds. Dropout is the critical ingredient.

### Recently Completed (2026-03-04)
| Job ID | Name | Result | Description |
|--------|------|--------|-------------|
| 5347062 | eval_ed | EMA=**1.2596** (+1.4%), diff=**1.4537** (+17.0%) | EMA & differencing hint eval (97/97) |
| 5347063 | eval_leg2 | cross_c4l6=**1.1900** (-4.2%) | Cross-family Cheb d=4 + Leg d=6 (97/97, ctx=4000) |
| 5347064 | anneal5k | COMPLETED | Hint annealing 5K ramp training |
| 5347065 | anneal8k | COMPLETED | Hint annealing 8K ramp training |
| 5349372 | eval_lc4k | leg_ms46=**1.2052** (-3.0%), leg_d5=**1.2057** (-2.9%), leg_ms56=**1.2550** (+1.0%) | Legendre re-eval at ctx=4000 (97/97) |
| 5349377 | eval_cc4k | 3chan=**1.2252** (-1.4%), 4chan=**1.2007** (-3.3%) | Channel re-eval at ctx=4000 (97/97) |
| 5349691 | eval_ann | anneal5k=**1.3486** (+8.6%), anneal8k=**1.3063** (+5.2%) | Anneal model eval (97/97) |
| 5316865 | leg46_r | MASE **1.2317** (-4.6% vs 100K BL) | Legendre ms46 100K steps |
| 5316864 | bl1k_r | MASE **1.2833** (-0.6% vs 100K BL) | BL 1K warmup 100K |
| 5317049 | bl_lrsw | lr=5e-4: 1.2531, lr=2e-3: 1.2586 | Baseline LR sweep (97-config auto-eval) |
| 5318043 | ctx_sweep | COMPLETED | BL@1K=1.2512, BL@2K=1.2304, MS46@1K=1.2074, MS46@2K=1.1700 |
| 5317702 | s1_hint | MASE **1.2598** (+1.4%) | Stride=1 training + eval |
| 5317717 | s32_hint | MASE **1.1975** (-3.6%) | Stride=32 training + eval (97-config auto-eval) |

### Completed — Legendre Investigation (2026-02-28, corrected 2026-03-04)
| Job ID | Name | Old Result (ctx=1K) | Corrected (ctx=4K) | Description |
|--------|------|---------------------|---------------------|-------------|
| 5181258 | leg_ms46 | 1.2356 | **1.2052** (-3.0%) | L1: Legendre d=4+d=6 multi-scale, 10K |
| 5181259 | leg_d5 | 1.2237 | **1.2057** (-2.9%) | L2: Legendre d=5 single-scale, 10K |
| 5181276 | leg_ms56 | 1.2627 | **1.2550** (+1.0%) | L3: Legendre d=5+d=6 multi-scale, 10K |
| 5181277 | cross_c4l6 | 1.2379 | **1.1900** (-4.2%) | L4: Cross-family Cheb d=4 + Leg d=6, 10K |
| 5181080 | leg46_100k | **TIMEOUT** epoch 610/1000 (60K) | L5: Legendre d=4+d=6 + 10% drop, 100K. Resumed as 5312574 |

### Completed — Hint Ablation Training (2026-02-28)
| Job ID | Name | Status | Description |
|--------|------|--------|-------------|
| 5180864 | abl_lc4 | COMPLETED | Learned 4-tap conv (Cheb d=4 init), 10K. Eval pending (5312570) |
| 5180890 | abl_lc16 | COMPLETED | Learned 16-tap conv (Cheb d=16 init), 10K. Eval pending (5312570) |
| 5180862 | abl_dup | **TIMEOUT** epoch 89/99 (9K) | Duplicate input, 10K. Eval@9K in 5312570. Resume as 5312571 |
| 5181262 | eval_abl | **FAILED** | Path bug: looked at wrong directory. Fixed as 5312570 |

### Completed — Long Training
| Job ID | Name | Result | Description |
|--------|------|--------|-------------|
| 5148815 | bl1k_100k | **TIMEOUT** epoch 932/1000 (90K) | Baseline + 1K warmup, 100K. Eval@90K in 5312570. Resume as 5312573 |
| 5105464 | ms46_100k | MASE **1.2214** (-5.40%) | ms d=4+d=6 + 10% drop, 100K, 10K warmup |
| 5105462 | hd10_10kw | MASE **1.2579** (+2.70% vs hd10 1K) | hint d=4 + 10% drop, 100K, 10K warmup |

### Pending — New Experiments
| Job ID | Name | Partition | Description |
|--------|------|-----------|-------------|
| 5145687 | bf16_46 | ailab | ms d=4+d=6 + **bf16-mixed precision**, 10K |
| 5145958 | moe_46 | ailab | ms d=4+d=6 + **MoE** (4 experts, top-2, d_ff=512), 10K |
| **5147212** | **ft_off46** | **della** | **Fine-tune official Moirai 2.0 + ms46 hints**, 10K |
| **5147215** | **ft_offbl** | **della** | **Fine-tune official Moirai 2.0 baseline** (no hints), 10K |
| 5147856 | wr_ms46 | della | **Warm restart** from ms46@10K with lr=3e-4, 5K steps |
| 5148708 | ms46_12k | — | ms d=4+d=6, **12K steps** (more cosine cooldown) |
| 5148709 | polyd_46 | — | ms d=4+d=6 + **polynomial decay** (power=2), 10K |
| 5148741 | rest_46 | — | ms d=4+d=6 + **cosine with 2 restarts**, warmup=500, 10K |
| 5148742 | w200_46 | — | ms d=4+d=6 + **warmup=200** (more decay time), 10K |
| 5148750 | l2d6d_46 | — | **L2-opt d=6 + 10% dropout**, 10K |
| 5148752 | d12k_46 | — | ms d=4+d=6 + 10% dropout, **12K steps** |
| **5148812** | **zero_h46** | **della** | **ABLATION: Zero hint** (ms46 arch, hints=0), 10K |
| **5148814** | **rand_h46** | **della** | **ABLATION: Random hint** (ms46 arch, hints=randn), 10K |
| **5148815** | **bl1k_100k** | **della** | **ABLATION: Baseline 1K warmup**, 100K steps |

### Pending — Evaluations
| Job ID | Name | Partition | Description |
|--------|------|-----------|-------------|
| 5148574 | ge_quick | pli | Eval: ema, wsd, fnoise, minlr, phd |
| 5148633 | ge_m100k | della (dep) | Eval both 100K runs after completion |
| 5148724 | ge_retry | — | Retry: ms46+drop@10K (ge_ms4d failed) |
| 5148767 | ge_nb | — (dep) | Batch: accum2, tsmixup, domshuffle, freqmask |
| 5145920 | ge_md10 | della | ms46+drop step 10K eval |
| 5145923 | ge_mc | della | ms46+drop training curve (30K, 50K) |
| 5143805 | ge_batch | della | Batch: ema6, ms46e4, swa, flip, cosr |
| 5147589 | ge_lat | della | Batch: phd, minlr, wsd, ema, fnoise |
| 5147815 | ge_bat3 | della | Batch: accum2, stu+ms46, mix, ds5, fm10 |
| **5148816** | **ge_ablat** | **della (dep)** | **ABLATION eval**: zero_hint + random_hint (afterany:5148812,5148814) |

### Recently Completed (this session, 2026-02-27 evening)
| Job ID | Name | Result |
|--------|------|--------|
| 5140402 | accum2_46 | **TIMEOUT** at epoch 152/200 (6h). Ckpts at 2K-14K. 10K ckpt exists for eval. |
| 5140784 | wsd_46 | Training COMPLETE epoch 99. Eval pending (ge_quick + ge_lat). |
| 5140794 | ema_46 | Training COMPLETE epoch 99. Eval pending (ge_quick + ge_lat). |
| 5140817 | fnoise_46 | Training COMPLETE epoch 99. Eval pending (ge_quick + ge_lat). |
| 5140783 | minlr_46 | Training COMPLETE epoch 99. Eval pending (ge_quick + ge_lat). |
| 5140721 | phd46_10k | Training COMPLETE epoch 99. Eval pending (ge_quick + ge_lat). |
| 5145734 | ge_ms4d | **FAILED** in 20s — loaded checkpoint but eval loop didn't run. Resubmitted as 5148724. |
| 5142928 | ge_ps16 | **TIMEOUT** at 78/97 configs (2h limit). Need resubmit with 4h. |

### Previously Completed
| Job ID | Name | Result |
|--------|------|--------|
| 5143464 | ge_8k9k | 8K=**1.2223** (97cfg), 9K=**TIMEOUT** at 79/97 (LOOP_SEATTLE). |
| 5143524 | ge_s128 | STU+ms46 bs128: **1.2581** (+1.29%). STU doesn't help with hints. |
| 5139460 | stu46_10k | Training COMPLETE epoch 99. Eval in ge_bat3. |
| 5140721 | phd46_10k | Training COMPLETE epoch 99. Eval in ge_lat. |
| 5140783 | minlr_46 | Training COMPLETE epoch 99. Eval in ge_lat. |
| 5140677 | ms46h50kr | **TIMEOUT** at epoch 453/500 (45K of target 50K). Ckpt at 45K saved. |
| 5143788 | ge_hd05 | MASE **1.2436** (+0.12%). 5% hint dropout HURTS. 10% optimal. |
| 5142992 | ge_c4e4 | MASE **1.2024** (-3.20%). Cheb d=4 + EMA d=4 not as good as ms46. |
| 5142928 | ge_ps16 | **TIMEOUT** at 78/97. Partial MASE ~1.2071. Eval patch_size=16 worse. |
| 5142719 | avg_ms46 | **TIMEOUT**. Round 1 (avg_last4): **1.2086** (-2.70%). Round 2 incomplete. |
| 5140438 | avg_ms46h | **TIMEOUT** at 79/97. |
| 5140523 | soup_46 | **TIMEOUT**. α=0.7: **1.4558**, α=0.5: **1.8323**. Model soup FAILS. |
| 5140544 | flip_46 | Training COMPLETE epoch 99/100. Ckpt saved. Eval in ge_batch. |
| 5140415 | cosr_46 | Training COMPLETE epoch 99/100. Ckpt saved. Eval in ge_batch. |
| 5141870 | hd05_46r | Training COMPLETE. Eval: ge_hd05=**1.2436**. |
| 5139463 | stu46bs128 | Training COMPLETE. Eval running as ge_s128. |
| 5138943 | ema6_10k | Training COMPLETE. Batch eval (ge_batch). |
| 5139474 | ms46e4_10k | Training COMPLETE. Batch eval (ge_batch). |
| 5140249 | ms46swa | Training COMPLETE. Batch eval (ge_batch). |
| 5138928 | c4ema4_10k | Training COMPLETE. Eval: ge_c4e4=**1.2024** (-3.20%). |
| 5132017 | hd05_46 | TIMEOUT at epoch 77/100. Resumed as 5141870. |
| 5140437 | ge_bs128 | ms46 bs=128: MASE **1.2439** (-0.14%). Batch size ~neutral. |
| 5138617 | ge_wt_w1k | Weighted baseline=**1.2497** (+0.61%), weighted+hint=**1.2382** (-0.31%). |
| 5140674 | ge_ms46_50k | ms46 @ 50K steps: MASE **1.2363** (-0.47%). Hint decay confirmed. |
| 5140675 | ge_ms46h35k | ms46+hd10 @ 35K: MASE **1.2629** (+1.67%). Raw ckpt noisy. |
| 5140438 | avg_ms46h | Ckpt avg (steps 20-35K): **1.2040** (-3.07%). Second eval (35K raw) timed out at 79/97. |
| 5140523 | soup_46 | α=0.7: **1.4558**, α=0.5: **1.8323** (FAILS: in_proj dim mismatch). α=0.3: running. |
| 5125274 | ms46h50k | TIMEOUT at epoch 349/500 (35K/50K steps). Checkpoints saved. Resume as 5140677. |
| 5131992 | bs128_46 | ✅ Training complete, epoch 49/50 (10K steps). Eval as 5140437. |
| 5128061 | ge_h70k | ✅ hd10 step 70K: MASE **1.2241** (-4.22%). Full hd10 curve complete. |
| 5127109 | ge_hp2 | ✅ lr5e4=**1.2687** (+2.14%), wd05=**1.2279** (-1.14%). HP changes hurt. |
| 5128212 | ge_hp3 | ✅ lr2e3=**1.2241** (-1.45%), w500=**1.2308** (-0.91%). HP changes hurt. |
| 5122074 | ge_crit | ✅ Completed baseline 5-step curve + hd10 at 20K/30K/50K. **Timed out** during hd10@70K. |
| 5126443 | ge_ramp | ✅ ramp3k=**1.2159** (-2.11%), ramp_hd=**1.2283** (-1.11%). Hint ramp hurts. |
| 5124619 | ramp3k | ✅ Training complete, epoch 99. |
| 5124620 | ramp_hd | ✅ Training complete, epoch 99. |
| 5124905 | lr5e4_46 | ✅ Training complete, epoch 99. Eval=**1.2687** (+2.14%). Half LR hurts. |
| 5124906 | lr2e3_46 | ✅ Training complete, epoch 99. Eval pending (ge_hp3). |
| 5125291 | wd05_46 | ✅ Training complete, epoch 99. Eval pending (ge_hp2). |
| 5125099 | w500_46 | ⚠️ Timed out at epoch 97. Using epoch_89 (step 9K) for eval. |
| 5124812 | ge_vp | ✅ vp46=**1.2896** (+3.82%), vpb=**1.3200** (+6.27%). Variable prefix definitively harmful. |
| 5124904 | ge_cd46 | ✅ cd46 eval: MASE **1.2349** (-0.58%). Combined dropout hurts ms46. |
| 5123302 | hd20_10k | ✅ ms d=4+d=6 + 20% hint dropout, epoch 99, loss 0.098. Eval as 5125613. |
| 5123301 | cd46_10k | ✅ ms d=4+d=6 + combined dropout (hint=0.1 + seq=0.1), epoch 99, loss 0.139. Eval as 5124904. |
| 5122941 | s48_10k | ✅ ms d=4+d=6 + d=4@s48, epoch 99, loss 0.134. |
| 5120929 | vp46_10k | ✅ ms d=4+d=6 + variable prefix [0.15,0.45]. MASE **1.2896** (+3.82%). Variable prefix hurts. |
| 5120930 | vpb_10k | ✅ Baseline + variable prefix [0.15,0.45], eval running as part of ge_vp. |
| 5120415 | sd20_10k | ✅ ms d=4+d=6 + 20% seq dropout. MASE **1.2535** (+0.92%). Seq dropout hurts ms46. |
| 5122392 | synth46 | ✅ LOTSA+KernelSynth + ms46. MASE **1.2087** (-2.69% vs baseline, +3.5% vs ms46). |
| 5120341 | wtb1k | ✅ Weighted LOTSA baseline, warmup=1K. Training complete. Eval pending (ge_wt_w1k). |
| 5120342 | wth1k | ✅ Weighted LOTSA + ms hint, warmup=1K. Training complete. Eval pending (ge_wt_w1k). |
| 5121044 | ge_wt | ✅ Weighted LOTSA 10K warmup: baseline=**1.3106**, hint=**1.2941**. Too much warmup. |
| 5119449 | eval_stu2k | ⏰ TIMEOUT at 4h. STU fulldff=**1.2947**, k8_gate=**1.2986**, hint_d4d6=timeout. Resubmitted as 5124790. |
| 5124515 | ge_s48 | ✅ s48 eval: MASE **1.2210** (-1.70%). Third hint channel dilutes signal. |

### ge_crit Training Curve Results
#### Baseline
| Steps | Checkpoint | MASE | Notes |
|-------|-----------|------|-------|
| 100K | epoch_999-step_100000 | 1.2911 | Proper baseline (zscore=8.0) |
| 20K | epoch_199-step_20000 | 1.3025 | From 100K run (10K warmup schedule) |
| 30K | epoch_299-step_30000 | 1.3127 | Still in high-LR phase |
| 50K | epoch_499-step_50000 | 1.2784 | First improvement over 10K |
| 70K | epoch_699-step_70000 | 1.2780 | Plateau near 50K |

#### hd10 (d=4 + 10% hint dropout)
| Steps | Checkpoint | MASE | vs Baseline | Notes |
|-------|-----------|------|-------------|-------|
| 20K | epoch_199-step_20000 | 1.2401 | -4.79% | |
| 30K | epoch_299-step_30000 | **1.2064** | **-8.10%** | Peak improvement! |
| 50K | epoch_499-step_50000 | 1.2412 | -2.91% | Regression from 30K |
| 70K | epoch_699-step_70000 | 1.2241 | -4.22% | Recovering |
| 100K | (known) | 1.1918 | -7.69% | |

### Cancelled Jobs (2026-02-27)
| Job ID | Reason |
|--------|--------|
| 5124415 | ge_vp on ailab: Resubmitted on della as 5124812 |
| 5124581, 5124582 | ramp3k/ramp_hd: Missing `+` prefix for Hydra. Resubmitted as 5124619/5124620. |
| 5122202 | ms46_100k on ailab: Redundant |
| 5122211 | ms46_100k on pli: Redundant |
| 5121884, 5121940 | ARM/x86 incompatibility on grace |
| 5121361 | sd20ms_10k: Duplicate |

### Prepared (Not Yet Submitted)
| Script | Description | When to Submit |
|--------|-------------|----------------|
| `/tmp/quick_ms46_ctx40_10k.slurm` | ms46 + prefix_ratio=0.4, 10K | After vp results (likely skip - vp46 failed) |
| `/tmp/quick_ms46_bs128_10k.slurm` | ms46 + batch_size=128, 10K | When GPU available |
| `/tmp/ge_ramp.slurm` | Eval ramp3k + ramp_hd | After ramp training completes |
| `/tmp/ge_hpsweep.slurm` | Eval lr5e4 + lr2e3 + w500 | After HP sweep training completes |
| `pretraining/continue_ms46_100k.slurm` | Resume ms46_100k from last.ckpt | After job 5105464 times out |

### Previous Jobs
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

### Completed Evals (2026-02-25)

| Job ID | Name | Result | Notes |
|--------|------|--------|-------|
| 5082176 | ge_d6_100k | MASE 1.2220 | Hint d=6, 100K steps, -5.11% vs 100K baseline |
| 5089146 | ge_m2_hf | MASE 1.0236 | Official Moirai 2.0-R-small (57/97 < 1.0) |
| 5085830 | ge_stu_hybrid | MASE 1.3044 | STU v2 hybrid, 10K steps, +5.01% vs baseline |

### Active/Pending Jobs (2026-02-27)

| Job ID | Name | Partition | Status | Notes |
|--------|------|-----------|--------|-------|
| 5105462 | hd10_10kw | della/gpu | PENDING (~01:08 est.) | Hint d=4+10%drop, 100K steps, 10K warmup |
| 5105463 | ge_hd10_10kw | della/gpu | PENDING (dep) | GIFT-Eval of above (afterok:5105462) |
| 5105464 | ms46_100k | della/gpu | PENDING | ms d=4+d=6 + 10% drop, 100K steps |
| 5119449 | eval_stu2k | della/gpu | PENDING (~01:16 est.) | Eval 3 STU 2K checkpoints on GIFT-Eval |
| 5119597 | synth_b10k | della/gpu | PENDING (~01:18 est.) | LOTSA+KernelSynth baseline, 10K steps |
| 5119598 | synth_h10k | della/gpu | PENDING (~01:22 est.) | LOTSA+KernelSynth + ms d=4+d=6 hint, 10K steps |
| 5119776 | mstrd_10k | della/gpu | PENDING | Multi-stride hint (d=4s16 + d=6s16 + d=4s32), 10K |
| 5119780 | wsh_10k | della/gpu | PENDING | Weighted LOTSA + KernelSynth + ms hint, 10K |
| 5119781 | wsb_10k | della/gpu | PENDING | Weighted LOTSA + KernelSynth baseline, 10K |
| 5119789 | ms46d_10k | della/gpu | PENDING | ms d=4+d=6 + model dropout 0.1, 10K |

### Recently Completed (2026-02-26)

| Job ID | Name | Result | Notes |
|--------|------|--------|-------|
| 5091690 | m2_wt_base | MASE 1.2426 | **⚠️ INVALID**: 10K warmup on 10K steps |
| 5091691 | m2_wt_hint | MASE 1.2534 | **⚠️ INVALID**: same warmup bug |
| 5092066 | stu_hint_d4d6_2k | loss 0.116 | STU + ms hint, 2K steps |
| 5092067 | stu_fulldff_k8_warmgate_2k | loss 0.117 | Full d_ff + K=8 + warm gate, 2K steps |
| 5092068 | stu_hint_fulldff_2k | loss 0.115 | STU + hints + full d_ff, 2K steps |
| 5091644-47 | FEV evals | COMPLETED | ms_d4d6, hd10_100k, official Moirai 2.0 all evaluated |
| 5091692 | ge_wt_base | COMPLETED | Weighted baseline GIFT-Eval (invalid warmup) |
| 5091700 | ge_wt_hint | COMPLETED | Weighted hint GIFT-Eval (invalid warmup) |

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

## OLMo USP v5 Experiments (2026-03-07)

### Spectral Gated Preconditioning (Fixed kernel + SiLU projections, Deep Conv v2 arch)

| Job ID | Name | Mode | Details | Status |
|--------|------|------|---------|--------|
| 5474427 | sg_cheb16 | spectral_gated | Chebyshev k=16 | PENDING |
| 5474428 | sg_hank16 | spectral_gated | Hankel k=16 | PENDING |
| 5474429 | sgms_c4c16 | spectral_gated_ms | Cheb d=3 + Cheb d=15 | PENDING |
| 5474430 | sg_leg16 | spectral_gated | Legendre k=16 | PENDING |
| 5474431 | ms_sg_cheb | ms_spectral_gated | MS embed + spectral gated per-layer | PENDING |
| 5474432 | sgms_mixed | spectral_gated_ms | Cheb d=3 + Leg d=7 + Cheb d=15 | PENDING |

### CET (Conditioned Embedded Tokens — SVD-based condition number reduction)

| Job ID | Name | Mode | Details | Status |
|--------|------|------|---------|--------|
| 5474527 | cet_svd_k2 | cet_svd | target_kappa=2.0 | PENDING |
| 5474528 | cet_svd_k5 | cet_svd | target_kappa=5.0 (gentler) | PENDING |
| 5474529 | cet_sg_cheb | cet_svd_spectral_gated | CET + spectral gated Cheb k=16 | PENDING |

### Hadamard Preconditioning

| Job ID | Name | Mode | Details | Status |
|--------|------|------|---------|--------|
| 5474588 | had_perlyr | hadamard_perlayer | Per-layer block Hadamard (zero params) | PENDING |
| 5474589 | cet_had | cet_hadamard | CET SVD + per-layer Hadamard | PENDING |

## OLMo USP v6: Faithful USP (Marsden-Hazan, 2026-03-07)

Based directly on "Universal Sequence Preconditioning" (arXiv:2502.06545).
Per-layer causal polynomial convolution with fixed monic coefficients + scalar gate (1 param/layer).

| Job ID | Name | Mode | Details | Status |
|--------|------|------|---------|--------|
| 5474807-5474813 | v6 base | various | CANCELLED — resubmitted as 5475018+ |
| 5475018 | usp_cheb5_pl | usp_perlayer | Chebyshev d=5, per-layer (paper's sweet spot) | **RUNNING** |
| 5475019 | usp_cheb10_pl | usp_perlayer | Chebyshev d=10, per-layer | **RUNNING** |
| 5475020 | usp_leg5_pl | usp_perlayer | Legendre d=5, per-layer | PENDING |
| 5475021 | usp_cheb2_pl | usp_perlayer | Chebyshev d=2 (= differencing), per-layer | PENDING |
| 5475022 | usp_cheb5_both | usp_both | Chebyshev d=5, embedding + per-layer | PENDING |
| 5475023 | usp_cheb5_strong | usp_perlayer | Chebyshev d=5, stronger gate (init=-1.0, α≈0.27) | PENDING |
| 5475024 | usp_cheb5_anneal | usp_anneal | Cosine annealing (strong early, decay late) | PENDING |
| 5475029 | usp_c5_dv2 | usp_perlayer_v2 | Chebyshev d=5 + Deep Conv v2 (k=16) | PENDING |
| 5475030 | usp_adaptive_k8 | usp_adaptive | Adaptive spectral filter (Levinson-Durbin) | PENDING |

## OLMo USP v6.1: Layerwise Variants (2026-03-07)

| Job ID | Name | Mode | Details | Status |
|--------|------|------|---------|--------|
| 5475054 | usp_graduated | usp_graduated | Degree 2→8 across layers | PENDING |
| 5475055 | usp_channelwise | usp_channelwise | Cheb d=[2,5,8] on channel groups | PENDING |
| 5475056 | usp_grad_channelwise | usp_grad_channelwise | Graduated + channelwise combined | PENDING |
| 5475057 | usp_cw_legendre | usp_channelwise | Legendre d=[2,5,8] on channel groups | PENDING |
| 5475058 | usp_cw_strong | usp_channelwise | Cheb d=[2,5,8], stronger gate=-1.5 | PENDING |

## OLMo USP v6.2: Poly-Gated (Fixed kernel + SiLU, 2026-03-07)

Fixed Marsden-Hazan polynomial kernel + Mamba-style SiLU projections.
Tests whether fixed polynomial kernel + learnable gating can match fully learnable Deep Conv v2.

| Job ID | Name | Mode | Details | Status |
|--------|------|------|---------|--------|
| 5475075 | usp_poly_gated_c5 | usp_poly_gated | Cheb d=5 + SiLU projections | PENDING |
| 5475076 | usp_poly_gated_c8 | usp_poly_gated | Cheb d=8 + SiLU projections | PENDING |
| 5475077 | usp_poly_gated_multi | usp_poly_gated_multi | Multi-poly d=[2,5,8] + SiLU projections | PENDING |

### v6.2 Results (2026-03-07 late)
- **Poly-gated advantage FADES**: -2.8% at step 1000 but converges to +0.1% by step 2800
- ALL fixed polynomial approaches (channelwise, graduated, CW_legendre) → neutral or worse
- Fixed polynomial gives a fast "warm start" but learnable DV2 kernel catches up and wins long-term
- DV2 at -1.02% (smoothed) remains the best approach

## OLMo USP v7: DV2 Variants (2026-03-07)

Goal: Beat DV2's -1.02% by improving the DV2 architecture itself, informed by polynomial theory.

| Job ID(s) | Name | Mode | Details | Status |
|-----------|------|------|---------|--------|
| 5481508,5481517 | dv2_poly_r01 | dv2_poly_residual | DV2 + fixed poly residual (alpha=0.1, Cheb d=5) | PENDING |
| 5481511 | dv2_poly_r03 | dv2_poly_residual | DV2 + fixed poly residual (alpha=0.3) | PENDING |
| 5481509,5481518 | dv2_dual_4_16 | dv2_dual_scale | Dual-scale DV2 (k=4 + k=16) | PENDING |
| 5481512 | dv2_dual_4_32 | dv2_dual_scale | Dual-scale DV2 (k=4 + k=32) | PENDING |
| 5481510,5481519 | dv2_k32 | dv2_k32 | Standard DV2 with kernel_size=32 | PENDING |
| 5481570,5481572 | dv2_expand_2x | dv2_expand | DV2 with expand_factor=2 (wider inner dim) | PENDING |
| 5481571,5481573 | dv2_both_k16 | dv2_both | DV2 before attention AND before FFN | PENDING |
| 5481613,5481614 | dv2_gradprecond_c5 | dv2_gradprecond | DV2 forward + Cheb d=5 gradient smoothing | PENDING |
| 5481641,5481642 | dv2_grad_4to32 | dv2_graduated | Layer-varying kernel size (4→32) | PENDING |
| 5481664,5481665 | dv2_polybasis_b6 | dv2_polybasis | Polynomial basis kernel (6 Cheb basis functions) | PENDING |

**Note (2026-03-07)**: ALL v7 jobs initially FAILED due to `import rich` (module not installed). Fixed with try/except. Old job IDs cancelled, resubmitted as:

| Job ID | Name | Mode | Details | Status |
|--------|------|------|---------|--------|
| 5481877 | dv2_poly_r01 | dv2_poly_residual | DV2 + fixed poly residual (alpha=0.1) | PENDING |
| 5481878 | dv2_poly_r03 | dv2_poly_residual | DV2 + fixed poly residual (alpha=0.3) | PENDING |
| 5481879 | dv2_dual_4_16 | dv2_dual_scale | Dual-scale DV2 (k=4 + k=16) | PENDING |
| 5481880 | dv2_dual_4_32 | dv2_dual_scale | Dual-scale DV2 (k=4 + k=32) | PENDING |
| 5481881 | dv2_k32 | dv2_k32 | DV2 with kernel_size=32 | PENDING |
| 5481882 | dv2_expand_2x | dv2_expand | DV2 with expand_factor=2 | PENDING |
| 5481883 | dv2_both_k16 | dv2_both | DV2 before attention AND before FFN | PENDING |
| 5481884 | dv2_gp5 | dv2_gradprecond | DV2 + Cheb d=5 gradient smoothing | PENDING |
| 5481885 | dv2_grad | dv2_graduated | Layer-varying kernel 4→32 | PENDING |
| 5481886 | dv2_pb6 | dv2_polybasis | Polynomial basis kernel (6 basis) | PENDING |
| 5481888 | dv2_ema | dv2_ema | DV2 with EMA kernel init | PENDING |
| 5481889 | logit_pc | logit_precond | Pure logit preconditioning (no arch change) | PENDING |
| 5481890 | dv2_lp | dv2_logit_precond | DV2 + logit preconditioning | PENDING |

## OLMo USP v8: Creative Preconditioning (2026-03-07)

Goal: Beat DV2's -1.11% with more expressive or better-structured conv preprocessing.

| Job ID | Name | Mode | Details | Status |
|--------|------|------|---------|--------|
| 5482026 | mhconv | multihead_conv | Multi-head conv [4,8,16,32] kernel sizes | PENDING |
| 5482027 | stk_dv2 | stacked_dv2 | Two sequential DV2 blocks (k=4 then k=16) | PENDING |
| 5482028 | dv2_n | dv2_norm | DV2 with RMSNorm before conv | PENDING |
| 5482029 | decay | decay_kernel | Kernel as sum of 4 learnable exponential decays | PENDING |
| 5482030 | cmix | conv_mixer | DV2 + cross-channel mixing bottleneck | PENDING |

**v8 RESUBMITTED** (after import fix + venv fix):

| Job ID | Name | Mode | Run Name | Status |
|--------|------|------|----------|--------|
| 5482506 | mhconv | multihead_conv | mhconv_4_8_16_32 | RUNNING (pli) |
| 5482507 | stk_dv2 | stacked_dv2 | stacked_dv2_4_16 | RUNNING (pli) |
| 5482508 | decay | decay_kernel | decay_kernel_d4 | RUNNING (pli) |
| 5482509 | dv2k32 | deep_conv_v2 k=32 | dv2_k32 | RUNNING (pli) |
| 5482510 | dv2ex | dv2_expand | dv2_expand_2x | RUNNING (pli) |
| 5482511 | dv2ema | dv2_ema | dv2_ema_k16 | RUNNING (pli) |

## OLMo USP v9: MHConv Improvements (2026-03-08)

Based on mhconv's strong results (-0.95% vs DV2 at step 3000).

| Job ID | Name | Mode | Run Name | Status |
|--------|------|------|----------|--------|
| 5485900 | mhc_ema | mhconv_ema | mhconv_ema_k4816 | PENDING (ailab) |
| 5485901 | mhc_6 | mhconv_6 | mhconv_6_scales | PENDING (ailab) |
| 5485902 | pyra | pyramid_conv | pyramid_conv_4_32 | PENDING (ailab) |
| 5485903 | mhc_dec | mhconv_decay | mhconv_decay_d3 | PENDING (ailab) |
| 5485925 | mhcema_p | mhconv_ema | mhconv_ema_k4816_p | PENDING (pli) |
| 5485926 | mhc6_p | mhconv_6 | mhconv_6_scales_p | PENDING (pli) |

## OLMo USP v10: Constrained EMA (2026-03-08)

Testing structured/constrained kernel parameterizations vs free kernels.

| Job ID | Name | Mode | Run Name | Status |
|--------|------|------|----------|--------|
| 5486039 | c_ema | constrained_ema | constrained_ema_k16 | PENDING (ailab) |
| 5486040 | dv2_k4 | dv2_short | dv2_short_k4 | PENDING (ailab) |
| 5486041 | mg_conv | multigate_conv | multigate_conv_4816 | PENDING (ailab) |
| 5486042 | ema_sc | ema_short_conv | ema_short_conv_k16 | PENDING (ailab) |

**Cancelled** (redundant with better alternatives):
- 5482493 dv2_pr03, 5482595 dv2_pr05: Poly residual higher alpha (pr01 already matching DV2)
- 5482494 dv2_ds16, 5482495 dv2_ds32: Dual-scale DV2 (mhconv is better designed)

## OLMo USP v11: Placement Experiments (2026-03-08)

Key finding: No pre-attention DV2 variant beats standard DV2 with proper smoothing.
Testing different PLACEMENT of the conv block in the transformer.

| Job ID | Name | Mode | Run Name | Status |
|--------|------|------|----------|--------|
| 5486166 | post_at | post_attn | post_attn_k16 | PENDING (ailab) |
| 5486167 | dual_dv | dual_dv2 | dual_dv2_k16 | PENDING (ailab) |
| 5486168 | post_ff | post_ffn | post_ffn_k16 | PENDING (ailab) |
| 5486169 | sel_dv2 | selective_dv2 | selective_dv2_k16 | PENDING (ailab) |
| 5486170 | par_dv2 | parallel_dv2 | parallel_dv2_k16 | PENDING (ailab) |

## OLMo USP v12: Literature-Inspired (2026-03-08)

Based on research survey: k=2-4 is universally optimal across Mamba/Based/RWKV/GLA.

| Job ID | Name | Mode | Run Name | Status |
|--------|------|------|----------|--------|
| 5486308 | tshift | token_shift | token_shift | PENDING (ailab) |
| 5486309 | mamba4 | dv2_k4_expand | dv2_k4_expand2 | PENDING (ailab) |
| 5486310 | hyena | hyena_style | hyena_style_k16 | PENDING (ailab) |

## CRITICAL FINDING: DV2 EMA beats DV2 by 0.5-0.7% at every step (2026-03-08)

DV2 EMA (EMA-initialized conv kernel) consistently outperforms standard DV2 at all matched steps:
- Step 1000: EMA -1.95% vs DV2 -1.38% (0.57% better)
- Step 2000: EMA -1.53% vs DV2 -0.98% (0.55% better)
- Step 3000: EMA -1.41% vs DV2 -0.79% (0.62% better)
- Step 3500: EMA -1.35% vs DV2 -0.82% (0.53% better)

Same param count as DV2 — only difference is conv kernel initialization.
Training was killed at step 3500 by SLURM time limit. Resume jobs submitted.

## Resume Jobs (2026-03-08)

| Job ID | Name | Mode | Run Name | Resume From | Status |
|--------|------|------|----------|-------------|--------|
| 5487039 | ema_res | dv2_ema | dv2_ema_k16 | step 3500 | PENDING (ailab) |
| 5487040 | exp_res | dv2_expand | dv2_expand_2x | step 3000 | PENDING (ailab) |
| 5487041 | ema_res | dv2_ema | dv2_ema_k16 | step 3500 | PENDING (pli) |
| 5487042 | exp_res | dv2_expand | dv2_expand_2x | step 3000 | PENDING (pli) |

## OLMo USP v15: Gradient Preconditioning + Ablations (2026-03-08)

True mathematical preconditioning via gradient hooks (Muon-style Newton-Schulz).

| Job ID | Name | Mode | Run Name | Status |
|--------|------|------|----------|--------|
| 5486883 | muon_f | muon_full | muon_full | PENDING (ailab) |
| 5486884 | muon_d | muon_diag | muon_diag | PENDING (ailab) |
| 5486885 | frz_cv | frozen_conv | frozen_conv_k16 | PENDING (ailab) |
| 5486886 | fix_em | fixed_ema | fixed_ema_k16 | PENDING (ailab) |

## OLMo USP v16: Init Ablation (2026-03-08)

| Job ID | Name | Mode | Run Name | Status |
|--------|------|------|----------|--------|
| 5487236 | i_uavg | init_uniform_avg | init_uniform_avg | PENDING (ailab) |
| 5487237 | i_diff | init_diff | init_diff | PENDING (ailab) |
| 5487238 | i_salp | init_same_alpha | init_same_alpha | PENDING (ailab) |
| 5487239 | i_cosi | init_cosine_decay | init_cosine_decay | PENDING (ailab) |

## OLMo USP v17: Spectral Conditioning + Wide Gate (2026-03-08)

Based on ablation insight: SiLU gating is doing most of the work in DV2.
Wide gate variants + Chebyshev-basis convolution (true preconditioning).

| Job ID | Name | Mode | Run Name | Expand | Status |
|--------|------|------|----------|--------|--------|
| 5489210 | ema_w4 | ema_wide | ema_wide_4x | 4.0 | PENDING (ailab) |
| 5489211 | ew4_ls | ema_wide_ls | ema_wide_ls_4x | 4.0 | PENDING (ailab) |
| 5489212 | wg_4x | wide_gate | wide_gate_4x | 4.0 | PENDING (ailab) |
| 5489213 | ew_rms | ema_wide_rms | ema_wide_rms_4x | 4.0 | PENDING (ailab) |
| 5489219 | ema_w2 | ema_wide | ema_wide_2x | 2.0 | PENDING (ailab) |
| 5489273 | cheb8 | cheb_conv | cheb_conv_d8_k16 | 1.0 | PENDING (ailab) |
| 5489274 | cheb4 | cheb_conv | cheb_conv_d4_k16 | 1.0 | PENDING (ailab) |
| 5489275 | chebw | cheb_conv_wide | cheb_conv_wide_d8 | 2.0 | PENDING (ailab) |

## Completed Results Summary (OLMo USP, avg last 100 CE)

| Model | CE (avg100) | vs Baseline | Steps | Notes |
|-------|-------------|-------------|-------|-------|
| **Baseline** | **2.9089** | — | 7630 | Reference |
| deep_v2_k16 (DV2) | 2.8765 | **-1.11%** | 7630 | Previous champion |
| dv2_pr01 (poly res) | 2.8757 | **-1.14%** | 7630 | Just completed, new champion |
| v8_dv2ema | ~2.82* | **-1.6%*** | 5853 | *Projected, time-limited |
| v8_dv2ex (expand 2x) | ~2.81* | **-1.7%*** | 5473 | *Projected, time-limited |

### Key Ablation Findings (at step 2500 vs baseline 3.4270)
- gp_abl (gate only): 3.3945 (-0.95%) → **SiLU gating is the primary mechanism**
- co_abl (conv only): 3.4443 (+0.50%) → **Conv alone is harmful**
- dv2_gp5 (grad precond): CE=6.7 → **Gradient preconditioning FAILED catastrophically**

| 5496567 | ctx_swp | pli | Context length sweep: HD10+BL m2d s0 at ctx=500,1000,2000,3000 |
| 5496762 | zh_eval | pli | Zero-hints ablation: HD10_m2d_s0 with hints zeroed at inference |
| 5497345 | hd_bl_s7 | pli | HD10+BL m2d seed=7 (4th m2d seed) |
| 5497951 | ev_latp | pli | Eval lat_c4s1, lat_c4s4, lat_c6s1, hd10_lat4 (latent precond) |
| 5497981 | lat_cont | pli | Continue latent precond training: lat_c6s1 + hd10_lat4 |

## Key Results (2026-03-08)

### Model Soup — FAILED
- soup_hd10_bl_s0 (alpha=0.5 weight avg of HD10 + baseline): **MASE=1.3619, 16/97 wins, +11.8% vs baseline**
- Weight averaging destroys co-adaptation between shared params and hint projections
- soup_50 variant also evaluating, early results equally bad

### Redundant Running Jobs (wasting GPUs)
| Job ID | Name | Status | Why Redundant |
|--------|------|--------|---------------|
| 5475363 | seeds_m2d | ep38, will timeout | Re-training hd10_m2d_s2 (already exists) |
| 5475360 | core_m2d | evaluating 53/97 | Re-evaluating models with existing results |
| 5483069 | hd10_sd | ep75, tight | Re-training hd10 uwt seeds (already exist) |
| 5482533 | mshd_m2d | ep76, will timeout | Re-training mshd10_m2d_s2 (already exists) |

### New Experiments — Stride Sweep (2026-03-08)
| Job ID | Name | Partition | Description |
|--------|------|-----------|-------------|
| 5499448 | str_swp | pli/hazan_intern | Stride sweep: s4, s8, s32, multi-stride (m2d seed 0) |
| 5499449 | str_swp | della/ehazan | Backup stride sweep on della |

### Base Scale Experiment (2026-03-08)
| Job ID | Name | Partition | Description |
|--------|------|-----------|-------------|
| 5500667 | base_hd10 | pli/eladgroup | HD10 vs BL at BASE scale (d=768, L=12, ~87M params), seed=42, batch=64, 10K steps |
| 5500668 | base_hd10 | della/ehazan | Backup: same as above |
**Rationale**: Highest-impact unanswered question — does hint benefit persist at larger model sizes?
If base model benefits LESS → mechanism is attention compensation (cross-patch info leakage)
If equally → mechanism is more fundamental (input-level spectral diversity)

### Mechanism Theory Update
**Cross-Patch Information Leakage**: Chebyshev d=4 hint = -x[t-2s] + 0.125*x[t-4s]
Each patch gets info from 2 and 4 patches back. This pre-computes temporal relationships
that the transformer would otherwise need to learn through self-attention.
- Explains frequency dependence: sub-hourly benefits because lookback matches correlation scale
- Predicts: base model (more attn) should benefit LESS, shorter stride helps H data
- Stride sweep, ctx_hd10, and base_hd10 will test these predictions

### New Experiments Submitted (2026-03-08 08:15)

| Job ID | Name | Partition | Status | Description |
|--------|------|-----------|--------|-------------|
| 5501446 | inf_abl | pli | PENDING | Inference-time hint ablation: eval HD10/MS46 with --zero-hints |
| 5501447 | inf_abl | ailab | PENDING | Backup on ailab |
| 5501448 | ms46hd10 | pli | PENDING | MS46 + 10% hint dropout on m2d seeds 0,1,2 |
| 5501449 | ms46hd10 | della | PENDING | Backup on della |

### Comprehensive M2D Results (2026-03-08)

**HD10 vs Baseline (m2d, 3 seeds):**
| Seed | HD10 MASE | BL MASE | Δ% | Wins | p (sign) | p (Wilcoxon) |
|------|-----------|---------|-----|------|----------|--------------|
| 0 | 1.1577 | 1.2185 | -4.99% | 66/97 | 0.0005 | <0.0001 |
| 1 | 1.1902 | 1.1955 | -0.44% | 54/97 | 0.31 | 0.11 |
| 2 | 1.1798 | 1.2134 | -2.77% | 70/97 | <0.0001 | <0.0001 |
| **Pooled** | — | — | **-2.75%** | **190/291 (65.3%)** | **2.0e-7** | — |
| **95% CI** | — | — | **[-3.89%, -1.66%]** | — | — | — |

**MS46 vs Baseline (m2d, 3 seeds):**
| Seed | MS46 MASE | BL MASE | Δ% | Wins | p (sign) |
|------|-----------|---------|-----|------|----------|
| 0 | 1.1999 | 1.2185 | -1.52% | 60/97 | 0.025 |
| 1 | 1.1792 | 1.1955 | -1.37% | 59/97 | 0.042 |
| 2 | 1.1711 | 1.2134 | -3.48% | 72/97 | <0.0001 |
| **Pooled** | — | — | **-2.13%** | **191/291 (65.6%)** | **1.1e-7** |
| **95% CI** | — | — | **[-3.14%, -1.13%]** | — | — |

**Cross-seed consistency: MS46 CoV=1.0% vs HD10 CoV=1.9%**
MS46 is 2x more consistent. MS46 significant on ALL 3 seeds. HD10 fails on seed 1.

**Overhead: <0.3% params, <1% training time, 0.1% FLOPs**

### New Experiments Submitted (2026-03-08 09:00)

| Job ID | Name | Partition | Status | Description |
|--------|------|-----------|--------|-------------|
| 5501980 | inf_ext | ailab | PENDING | Extended inference ablation: random-hints and duplicate-hints on HD10 m2d s0 |
| 5501984 | inf_ext | pli | PENDING | Backup on pli |

### Early Inference Ablation Results (2026-03-08, HD10 m2d s0, 28/97 configs)

**MAJOR FINDING**: Zero-hint model is WORSE than baseline, not just worse than normal HD10.
- HD10 (normal): geomean MASE = 1.5836
- HD10 (zero-hints): geomean MASE = 1.7383 (+9.8% vs normal, +4.3% vs baseline)
- Baseline: geomean MASE = 1.6665
- Zero-hints wins only 3/28 vs HD10, 6/28 vs baseline
- **Model genuinely USES hints at inference — not just a training regularizer**
- Degradation concentrated in high-frequency long-horizon configs (bizitobs +44%, elec/15T +24%)
- Low-frequency short-horizon configs barely affected (hierarchical_sales/D: +0.0%)

### Dropout Variant Results (m2d, seed 0, complete)

| Method | GeoMASE | Δ% vs BL | Wins | Sign p |
|--------|---------|----------|------|--------|
| HD10 (10%) | 1.1577 | -4.99% | 66/97 | 0.0005 |
| HD2 (2%) | 1.1794 | -3.20% | 63/97 | 0.004 |
| SD30 (seq 30%) | 1.1805 | -3.11% | 52/97 | 0.54 (ns!) |
| HD20 (20%) | 1.1842 | -2.81% | 67/97 | 0.0002 |
| HD3 (3%) | 1.1947 | -1.95% | 55/97 | 0.22 (ns) |
| MS46HD20 | 1.1916 | -2.20% | 61/97 | 0.014 |
| HD10+SD20 | 1.1953 | -1.90% | 61/97 | 0.014 |
| HD30 (30%) | 1.2034 | -1.23% | 61/97 | 0.014 |

**Model soups FAIL**: 0.75*HD10+0.25*BL = +12%, 0.50*HD10+0.50*BL = +41% (catastrophic)

### 100K Training Curve — COMPLETE (2026-03-09)

| Steps | BL gmMASE | HD10 gmMASE | Δ% | Wins | p |
|-------|-----------|-------------|-----|------|---|
| 5K | 1.2295 | 1.2015 | -2.28% | 62/97 | 0.008 |
| 10K | 1.2185 | 1.1577 | -4.99% | 66/97 | 0.0005 |
| 20K | 1.2442 | 1.1762 | -5.47% | 76/97 | 1.7e-8 |
| 30K | 1.1918 | 1.1775 | -1.20% | 56/97 | 0.15 ns |
| 40K | 1.2564 | 1.1970 | -4.73% | 60/97 | 0.025 |
| 50K | 1.2362 | 1.1770 | -4.79% | 66/97 | 0.0005 |
| 60K | 1.2352 | 1.1831 | -4.22% | 69/97 | 5e-5 |
| 70K | 1.2309 | 1.1903 | -3.29% | 63/97 | 0.004 |
| 80K | 1.3038 | 1.1982 | -8.10% | 74/97 | 2e-7 |
| 90K | 1.2737 | 1.1780 | -7.51% | 76/97 | 2e-8 |
| 100K | 1.2783 | 1.1739 | -8.17% | 74/97 | 2e-7 |

**KEY**: BL collapses at 80K (LR=9.7e-5). BL@100K WORSE than BL@10K. HD10@100K is BEST ckpt.

### OLMo-USP 2B Runs — COMPLETE (2026-03-09)
- Precond 2B: CE=2.6533 vs Baseline 2B: CE=2.7290 → **-2.77%** at 2B tokens
- Improvement grows: 1.6% at 2K steps → 2.8% at 14K steps → 2.77% at end

### New Jobs Submitted (2026-03-09)

| Job ID | Name | Partition | Description |
|--------|------|-----------|-------------|
| 5547493 | ev_hd100 | pli | **DONE.** HD10@60K-100K evals. All results collected. |
| 5553030 | hd100_sN | pli (array 7,2) | Multi-seed 100K: BL+HD10 for seeds 7 and 2 (~30h each) |
| 5553031 | hd100_sN | ailab (array 7,2) | Same as above, backup |
| 5553032 | ev_ms100 | pli | MSHD10 100K curve eval (10K-100K as available) |
| 5553059 | lr_ctrl | pli | **LR control**: BL cosine floor=10% + BL WSD schedule (100K). PENDING maint. |
| 5553061 | hd_lrc | ailab | **LR control**: HD10 cosine floor=10% + HD10 WSD schedule (100K). PENDING maint. |
| 5553671 | ev_10k | ailab | Eval BL+HD10 @10K from 100K runs (for MSHD10 comparison) |
| 5553673 | ev_10k | pli | Same as above, backup |

### OLMo-USP Updated Results (2026-03-09 17:45)

Best completed 1B experiments (all at 7630 steps):
| Run | Mode | LR | CE | vs BL@1e-3 (2.739) |
|-----|------|-----|------|-------------------|
| precond_lr8e4 | sel_dx2 x3 | 8e-4 | 2.700 | **-1.42%** |
| precond_lr1e3 | sel_dx2 x3 | 1e-3 | 2.701 | -1.39% |
| bx_lr5e4_w500 | sel_dx2_bx x3 | 5e-4 | 2.707 | -1.17% |
| bx_lr5e4 | sel_dx2_bx x3 | 5e-4 | 2.715 | -0.88% |

Best 2B result: sel_dx2_x3_2B CE=2.647 vs baseline_2B CE=2.723 = **-2.79%**

Running: bx_lr8e4 (3734/7630), bx_lr1e3 (4167/7630), sel_dx2_x3_2B repeat (14857/15259)

### OLMo-USP Leaderboard (2026-03-10 02:00)

| Rank | Run | LR | Warmup | avg20 CE | vs BL (2.878) |
|------|-----|----|--------|---------|--------------|
| 1 | **bx_lr1e3_w500** | 1e-3 | 500 | **2.679** | **-6.90%** |
| 2 | bx_lr1.5e3 | 1.5e-3 | 250 | 2.690 | -6.53% |
| 3 | bx_lr1e3 | 1e-3 | 250 | 2.694 | -6.41% |
| 4 | bx_lr8e4 | 8e-4 | 250 | 2.698 | -6.24% |
| 5 | precond_lr8e4 | 8e-4 | 250 | 2.709 | -5.87% |
| 6 | baseline_lr1.5e3 | 1.5e-3 | 250 | 2.728 | -5.21% |
| 7 | baseline_lr1e3 | 1e-3 | 250 | 2.748 | -4.52% |

Round 13 completed (all 3 runs done):
| Job | Name | avg20 CE | Status |
|-----|------|----------|--------|
| 5555809 | bx_lr1e3_w500 | 2.679 (-6.90%) | **DONE** — NEW #1 |
| 5555810 | bx_lr1.5e3 | 2.690 (-6.53%) | **DONE** |
| 5555811 | baseline_lr1.5e3 | 2.728 (-5.21%) | **DONE** |

Round 14 (PENDING, cluster maintenance until 6 PM):
| Job | Name | Config | Status |
|-----|------|--------|--------|
| 5560316 | bx_lr2e3 | sel_dx2_bx LR=2e-3 | PENDING |
| 5560317 | bl_lr2e3 | baseline LR=2e-3 | PENDING |
| 5560400 | bx_lr1.5e3_w500 | sel_dx2_bx LR=1.5e-3 warm=500 | PENDING |
| 5560401 | bx_lr2e3_w500 | sel_dx2_bx LR=2e-3 warm=500 | PENDING |

### THREE-WAY Training Curve (BL vs HD10 vs MSHD10, m2d seed 0) — COMPLETE

| Steps | BL | HD10 (Δ%) | HD w | MSHD10 (Δ%) | MS w | Leader |
|-------|------|-----------|------|-------------|------|--------|
| 10K | 1.2393 | 1.2160 (-1.9%) | 56 | 1.2120 (-2.2%***) | 70 | MSHD10 |
| 20K | 1.2442 | 1.1762 (-5.5%***) | 76 | 1.2001 (-3.5%***) | 69 | HD10 |
| 30K | 1.1918 | 1.1775 (-1.2% ns) | 56 | 1.1786 (-1.1%*) | 59 | tied |
| 40K | 1.2564 | 1.1970 (-4.7%*) | 60 | 1.1980 (-4.6%***) | 67 | tied |
| 50K | 1.2362 | 1.1770 (-4.8%***) | 66 | 1.1679 (-5.5%***) | 72 | MSHD10 |
| 60K | 1.2352 | 1.1831 (-4.2%***) | 69 | 1.1743 (-4.9%***) | 71 | MSHD10 |
| 70K | 1.2309 | 1.1903 (-3.3%**) | 63 | 1.1765 (-4.4%***) | 76 | MSHD10 |
| 80K | 1.3038 | 1.1982 (-8.1%***) | 74 | 1.1646 (-10.7%***) | 79 | MSHD10 |
| 90K | 1.2737 | 1.1780 (-7.5%***) | 76 | **1.1610 (-8.8%***)** | **83** | **MSHD10** |
| 100K | 1.2783 | 1.1739 (-8.2%***) | 74 | 1.1622 (-9.1%***) | **85** | **MSHD10** |

**KEY FINDINGS:**
- **MSHD10@90K = 1.1610 = ABSOLUTE BEST** checkpoint of any method
- **MSHD10@100K = 1.1622, -9.09%, 85/97 wins (p=1.04e-14)** — highest win rate
- HD10@100K = 1.1739, -8.17%, 74/97 wins (p=2.03e-7)
- MSHD10 overtakes HD10 from 50K onward with consistently MORE per-config wins
- MSHD10 vs HD10 head-to-head: 43/97 at 100K (p=0.31, marginal)
- BL UNSTABLE: range 1.19-1.30. HD10: 1.17-1.20. MSHD10: 1.16-1.21.

Eval jobs:
| Job | Evaluating | Status |
|-----|-----------|--------|
| 5553032 | ev_ms100 (MSHD10 10K-100K) | DONE (all 10 checkpoints) |
| 5558079 | ev_ms90 (MSHD10 90K+100K) | DONE (90K+100K) |
| 5490688 | ms10_100k (MSHD10 training) | DONE (epoch 999, 100K steps) |

### New Experiments Submitted (2026-03-10)

| Job ID | Name | Partition | Description | Status |
|--------|------|-----------|-------------|--------|
| 5560420 | bl_lr2 | pli | BL LR=2e-3 + LR=3e-3 (10K, seed 0) | PENDING |
| 5560421 | bl_lr2 | ailab | Same as above (backup) | PENDING |
| 5560425 | ms100_s | pli (array 2,7) | MSHD10 100K seeds 2,7 | PENDING |
| 5560426 | ms100_s | ailab (array 2,7) | MSHD10 100K seeds 2,7 (backup) | PENDING |
