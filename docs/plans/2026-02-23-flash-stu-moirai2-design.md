# Flash-STU 2.0 + Moirai2 Hybrid Design

**Date:** 2026-02-23
**Goal:** Integrate Flash-STU spectral layers into Moirai2 Small as a Hybrid-Multi architecture, matching baseline param count (11.4M), targeting MASE improvement on GIFT-Eval.

## Architecture

Approach C: Lightweight STU Augmentation. Each of the 6 transformer layers gets a parallel STU branch with gated residual fusion.

### Per-Layer Structure

```
Input (B, T, 384)
  |
  +-- Pre-Norm -> GroupedQueryAttention -> Dropout --+
  |                                                   | Residual Add
  +-- RMSNorm -> STU(approx, K=24) -> gate * scale --+
  |
  v (B, T, 384)
  +-- Pre-Norm -> FFN(d_ff=940, GLU) -> Dropout -> Residual Add
  |
  v
Output (B, T, 384)
```

### STU Layer Details

- **Mode:** Approx (project-then-convolve)
- **Spectral filters:** K=24 Hankel eigenvectors, precomputed once at init
- **Two-branch:** Positive + negative Hankel (sign-alternated input)
- **Convolution:** torch.fft.rfft/irfft (no FlashFFTConv dependency)
- **Gate:** Learned scalar, initialized to 0 (zero-init for smooth warmup)
- **Params per STU layer:** K*d + d^2 = 9,216 + 147,456 = 156,672
- **Total STU params (6 layers):** ~942K including norms and gates

### Parameter Budget

| Component | Baseline | STU Variant |
|-----------|----------|-------------|
| Attention (6 layers) | 2.06M | 2.06M (unchanged) |
| FFN (6 layers, GLU) | 9.44M (d_ff=1024) | 8.68M (d_ff=940) |
| STU (6 layers, approx) | 0 | 0.94M |
| Norms + misc | ~0.05M | ~0.06M |
| In/Out projections | ~0.25M | ~0.25M |
| **Total** | **~11.4M** | **~11.4M** |

### Spectral Filter Computation

1. Build Hankel matrix H[i,j] = 2 / ((i+j)^3 - i-j) for i,j in [1, max_seq_len]
2. Eigendecompose: eigvals, eigvecs = eigh(H)
3. Take top K=24 eigenvectors, scale by eigenvalue^0.25
4. Store as buffer (not parameter), shape (max_seq_len, K)
5. Pad to nearest power-of-two for FFT efficiency

### Causality

STU's Hankel-based spectral filtering operates on the full sequence via FFT convolution. The Hankel matrix is constructed from temporal indices capturing historical structure. The resulting filters are applied causally by construction (past-to-present convolution pattern). Combined with the causal attention mask already in Moirai2, the model maintains autoregressive properties.

### Key Design Decisions

1. **Zero-init gate:** Training starts exactly like baseline Moirai2. STU contribution grows organically as the gate learns to open.
2. **Shared spectral filters:** All 6 layers share the same precomputed Hankel eigenvectors. Only learned projection matrices differ per layer.
3. **d_ff reduction (1024 -> 940):** Compensates for STU parameter overhead. The 8% FFN reduction is minor compared to the inductive bias STU provides.
4. **Approx mode:** ~50x fewer STU params than standard mode, leaving budget for attention.

### Files to Modify/Create

1. **NEW** `uni2ts/src/uni2ts/module/stu_layer.py` - STU layer adapted from flash-stu-2
2. **MODIFY** `uni2ts/src/uni2ts/module/transformer.py` - Add STU branch to TransformerEncoderLayer
3. **MODIFY** `uni2ts/src/uni2ts/model/moirai2/module.py` - Spectral filter init, pass STU config
4. **MODIFY** `uni2ts/src/uni2ts/model/moirai2/pretrain.py` - Accept STU hyperparams
5. **MODIFY** `uni2ts/src/uni2ts/model/moirai2/forecast.py` - Accept STU hyperparams
6. **NEW** `uni2ts/cli/conf/pretrain/model/moirai2_small_stu.yaml` - Hydra config
7. **NEW** `uni2ts/pretraining/quick_stu_hybrid.slurm` - Training script

### Evaluation Plan

1. Quick validation: CPU test with 3 batches to verify shapes and gradients
2. 10K step training run on LOTSA
3. GIFT-Eval full benchmark (97 configs)
4. Compare MASE geometric mean vs baseline (1.2421 at 10K)
