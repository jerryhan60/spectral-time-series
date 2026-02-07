# STU-MOIRAI Hybrid Model Experiment

## Overview

This document describes the experimental setup comparing the **STU-MOIRAI Hybrid** model against the **Baseline MOIRAI** model for time series forecasting pretraining.

---

## Model Architectures

### Baseline: MOIRAI Small

The baseline model follows the original MOIRAI architecture from Woo et al. (2024).

| Parameter | Value |
|-----------|-------|
| Model Class | `MoiraiPretrain` → `MoiraiModule` |
| Hidden Dimension (`d_model`) | 384 |
| Number of Layers | 6 |
| Attention Heads | 6 (d_model // 64) |
| Patch Sizes | [8, 16, 32, 64, 128] |
| Max Sequence Length | 512 |
| Total Parameters | ~12.5M |
| Encoder Type | Standard Transformer (all attention layers) |

**Architecture:**
```
Input → Patch Embedding → [Transformer Block × 6] → Distribution Head → Output
                              ↑
                    (Self-Attention + FFN)
```

### STU-MOIRAI Hybrid

The hybrid model replaces alternating transformer layers with Spectral Transform Unit (STU) layers.

| Parameter | Value |
|-----------|-------|
| Model Class | `MoiraiHybridPretrain` → `MoiraiHybridModule` |
| Hidden Dimension (`d_model`) | 384 |
| Number of Layers | 6 |
| Attention Heads | 6 (for attention layers) |
| Patch Sizes | [8, 16, 32, 64, 128] |
| Max Sequence Length | 512 |
| Total Parameters | ~12.5M |
| Encoder Type | Hybrid (alternating STU + Attention) |
| STU Layer Pattern | `alternating` |
| Number of Spectral Filters (`num_eigh`) | 24 |
| Use Approximation Mode | True |

**Architecture:**
```
Input → Patch Embedding → [STU Block] → [Attn Block] → [STU Block] → ... → Distribution Head → Output
                              ↑              ↑
                    (FFT Convolution)  (Self-Attention + FFN)
```

**Layer Pattern (alternating):**
- Layer 0: STU
- Layer 1: Attention
- Layer 2: STU
- Layer 3: Attention
- Layer 4: STU
- Layer 5: Attention

---

## STU (Spectral Transform Unit) Details

The STU layer is based on the Flash STU paper (arXiv:2409.10489) and provides efficient long-range sequence modeling via spectral convolution.

### Key Components

1. **Spectral Filters**: Derived from Hankel matrix eigendecomposition
   - `num_eigh = 24` top eigenvectors used
   - Precomputed for sequence length 512

2. **FFT Convolution**: O(L log L) complexity vs O(L²) for attention
   - Input projected to filter space
   - Convolution via FFT
   - Output projected back

3. **Approximation Mode**: Uses factorized parameter matrices
   - ~50x fewer parameters per STU layer
   - `M_phi`: [num_eigh, d_model]
   - `M_inputs`, `M_filters`: projection matrices

### Packed Sequence Handling

MOIRAI uses packed sequences with `sample_id` to batch multiple time series. The STU adapter (`PackedSTU`) handles this by:
1. Unpacking sequences by sample_id
2. Applying STU convolution to each sequence
3. Repacking into batch format

---

## Training Configuration

### Common Settings

| Parameter | Value |
|-----------|-------|
| Dataset | LOTSA v1 (unweighted) |
| Max Epochs | 100 |
| Batches per Epoch | 100 |
| Batch Size | 128 |
| Learning Rate | 1e-3 |
| Weight Decay | 0.1 |
| Optimizer | AdamW (β1=0.9, β2=0.98) |
| Scheduler | Cosine with Restarts |
| Warmup Steps | 10,000 |
| Gradient Clipping | 1.0 (norm) |
| Precision | FP32 |
| Loss Function | PackedNLLLoss |

### Data Transforms

Both models use identical data preprocessing:
1. `SampleDimension`: Sample up to 128 variates
2. `GetPatchSize`: Determine patch size per frequency
3. `PatchCrop`: Crop to max 512 patches
4. `PackFields`: Pack variates into sequences
5. `AddObservedMask`: Create missing value masks
6. `ImputeTimeSeries`: Fill missing values with 0
7. `Patchify`: Convert to patches
8. `AddVariateIndex` / `AddTimeIndex`: Add positional info
9. `MaskedPrediction`: Create prediction targets (15-50% mask ratio)
10. `FlatPackCollection`: Flatten for batching

### Distribution Output

Mixture distribution with 4 components:
- Student-T
- Normal (fixed scale)
- Negative Binomial
- Log-Normal

---

## Experiment Results

### Training Runs

| Model | Job ID | Start Time | Status |
|-------|--------|------------|--------|
| STU Hybrid | 4163304 | 2026-01-25 16:46 | Completed (100 epochs) |
| Baseline | 4163305 | 2026-01-25 16:46 | Completed (100 epochs) |

### Checkpoint Locations

**Baseline:**
```
/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/moirai_small_baseline_20260125_164605/checkpoints/
```

**STU Hybrid:**
```
/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_stu/lotsa_v1_unweighted/moirai_small_stu_20260125_164605/checkpoints/
```

### Training Speed

| Model | Speed (it/s) | Time per Epoch |
|-------|--------------|----------------|
| Baseline | ~1.1-1.3 | ~80s |
| STU Hybrid | ~0.55-0.57 | ~180s |

The STU model is ~2x slower due to FFT convolution overhead (without Flash FFT Conv optimization).

### Loss Progression

See TensorBoard logs at:
```
outputs/pretrain/moirai_small*/lotsa_v1_unweighted/*/logs/
```

---

## File Locations

### Model Code

| File | Description |
|------|-------------|
| `src/uni2ts/model/moirai/module_hybrid.py` | MoiraiHybridModule class |
| `src/uni2ts/model/moirai/pretrain_hybrid.py` | MoiraiHybridPretrain Lightning module |
| `src/uni2ts/module/hybrid_encoder.py` | HybridTransformerSTUEncoder |
| `src/uni2ts/module/stu_layer.py` | STUEncoderLayer |
| `src/uni2ts/module/stu_adapter.py` | STUCore and PackedSTU |
| `src/uni2ts/module/spectral_filters.py` | Spectral filter computation |

### Configuration

| File | Description |
|------|-------------|
| `cli/conf/pretrain/model/moirai_small.yaml` | Baseline config |
| `cli/conf/pretrain/model/moirai_small_stu.yaml` | STU hybrid config |
| `cli/conf/pretrain/default.yaml` | Training defaults |

### Scripts

| File | Description |
|------|-------------|
| `pretraining/pretrain_moirai_small_baseline.slurm` | Baseline training job |
| `pretraining/pretrain_moirai_small_stu.slurm` | STU training job |
| `check_training.sh` | Quick status check |
| `monitor_training.py` | Live loss monitoring |
| `launch_tensorboard.sh` | TensorBoard launcher |

---

## Next Steps

1. **Evaluate on benchmarks**: Run evaluation on LSF and Monash datasets
2. **Compare forecasting performance**: CRPS, MSE, MAE metrics
3. **Analyze learned representations**: Compare attention patterns vs spectral filters
4. **Optimize STU speed**: Install Flash FFT Conv for faster training
5. **Hyperparameter tuning**: Experiment with num_eigh, layer patterns

---

## References

1. Woo et al. (2024). "Unified Training of Universal Time Series Forecasting Transformers." ICML 2024.
2. Flash STU (2024). "Fast Spectral Transform Units." arXiv:2409.10489.
3. Uni2TS Repository: https://github.com/SalesforceAIResearch/uni2ts
