# Moirai 2.0 Official Training Configuration Research

**Date**: 2026-02-25
**Sources**: Moirai 2.0 paper (arXiv:2511.11698), HuggingFace model card, Salesforce blog, uni2ts codebase

## Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Training steps | **100,000** | Paper |
| Batch size | **256** | Paper |
| Optimizer | **AdamW** | Paper / model config |
| Learning rate | **1e-3** | Paper / model config |
| Weight decay | **1e-1** | Paper / model config |
| Beta1 | **0.9** | Paper / model config |
| Beta2 | **0.98** | Paper / model config |
| Warmup steps | **10,000** (linear) | Paper / model config |
| LR schedule | **Cosine annealing** after warmup | Paper |
| Precision | **bf16 mixed** | Paper |
| Inference GPU | Single H200 | Paper (benchmark only) |
| Training GPUs | Not disclosed | -- |

## Model Sizes

| Variant | Parameters |
|---------|-----------|
| Small | 11.4M |
| Base | 87.1M |
| Large | 305M |

## Architecture (Moirai 2.0 vs 1.0)

- **Decoder-only** transformer with causal attention (vs encoder in 1.0)
- **Single patch size** (vs multi-patch in 1.0)
- **Quantile loss** (pinball) with 9 quantile levels: {0.1, 0.2, ..., 0.9} (vs distributional loss in 1.0)
- **Multi-token prediction** (vs single-token in 1.0)
- Patch token embedding includes **missing value information**
- **Patch-level random masking** (50%) during training
- **Instance normalization**: statistics computed from first 30% of each sample
- **Z-score anomaly detection** to filter outlier distributions

## Training Data -- NOT LOTSA

**CRITICAL FINDING**: Moirai 2.0 was NOT trained on LOTSA. It uses a completely different training corpus.

### Moirai 2.0 Training Corpus (36M series, ~295B observations)

| Source | Series Count | Observations | Description |
|--------|-------------|--------------|-------------|
| GIFT-Eval Pretrain | 3.25M | 230B | Non-leaking historical context from GIFT-Eval benchmark |
| GIFT-Eval TrainTest | 144K | -- | Train/test splits from GIFT-Eval |
| Chronos-Mixup | 30M | 63B | Mixup data generated from non-leaking subsets of Chronos Dataset |
| KernelSynth | 11M | 1.02B | Synthetic time series via KernelSynth (from Chronos paper) |
| Internal Salesforce CloudOps | 2.15M | 1.48B | Internal Salesforce operational data |

### Comparison: LOTSA vs Moirai 2.0 Corpus

| Aspect | LOTSA (Moirai 1.0) | Moirai 2.0 Corpus |
|--------|--------------------|--------------------|
| Total series | ~27B observations | ~295B observations (~10x more) |
| Synthetic data | None | KernelSynth (11M series) + Chronos-Mixup (30M series) |
| Internal data | None | Salesforce CloudOps (2.15M series) |
| Data source | Public datasets only | Public + synthetic + proprietary |
| Domains | 9 diverse domains | Multiple via GIFT-Eval + synthetic |

### What This Means for Our Experiments

Since we don't have access to the Moirai 2.0 training corpus (GIFT-Eval Pretrain, Chronos-Mixup, KernelSynth, Salesforce internal data), we can only train on LOTSA. The `lotsa_v1_moirai2.yaml` config in the codebase appears to be Salesforce's adaptation of LOTSA with variate-proportional sampling (not the actual Moirai 2.0 training data), but it references 22 datasets we don't have.

**Our options**:
1. `lotsa_v1_weighted.yaml`: All 170 datasets available, uses "proportional" sampling + per-dataset weights. **Best option for us.**
2. `lotsa_v1_moirai2.yaml`: 22/169 datasets missing, uses "variate_proportional" sampling. Would fail at runtime.
3. `lotsa_v1_unweighted.yaml`: All datasets available, no weights, uniform sampling. What we've been using.

## Moirai-MoE Training (for Reference)

From the Moirai-MoE paper (arXiv:2410.10469):

| Parameter | Value |
|-----------|-------|
| GPUs | 16x A100 (40G) |
| Batch size | 1,024 |
| Precision | bfloat16 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-1, beta2=0.98) |
| Warmup | 10,000 steps (linear) |
| LR schedule | Cosine annealing |
| Steps (Small) | 50,000 |
| Steps (Base) | 250,000 |
| Experts | 32 per layer |
| Total params (Small) | 117M |
| Total params (Base) | 935M |

## Local Codebase Config Verification

The `moirai2_small.yaml` Hydra config matches the paper:
- `lr: 1e-3`
- `weight_decay: 1e-1`
- `beta1: 0.9`
- `beta2: 0.98`
- `num_warmup_steps: 10_000`
- `d_model: 384`, `d_ff: 1024`, `num_layers: 6`, `patch_size: 16`
- Schedule: cosine annealing (via `num_training_steps` computed from epochs * batches)

## Key Differences Between Our Training and Official

| Aspect | Our Training | Official Moirai 2.0 |
|--------|-------------|---------------------|
| Data | LOTSA (unweighted) | 36M series corpus (10x larger) |
| Steps | 10K-100K | 100K |
| Batch size | 128-256 | 256 |
| Warmup | 1,000 | 10,000 |
| GPUs | 1 | Not disclosed |

**NOTE**: Our warmup has been 1,000 steps (matching earlier experiments), while the official config uses 10,000. This is a significant difference worth testing.

## References

- Paper: https://arxiv.org/abs/2511.11698
- HuggingFace: https://huggingface.co/Salesforce/moirai-2.0-R-small
- Blog: https://www.salesforce.com/blog/moirai-2-0/
- Moirai-MoE Paper: https://arxiv.org/abs/2410.10469
- GitHub: https://github.com/SalesforceAIResearch/uni2ts
