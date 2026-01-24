# Embedding Preconditioning Analysis Report

**Date**: 2025-12-06
**Models Analyzed**: Baseline, Embedding Precond D3, Embedding Precond D4
**Datasets**: 29 Monash benchmark datasets

## Executive Summary

Embedding-level preconditioning (d=3, d=4) shows improved training NLL but **degraded evaluation performance across almost all metrics**. This analysis reveals that the approach has fundamental theoretical issues that prevent it from achieving the benefits demonstrated in the Universal Sequence Preconditioning paper.

### Key Findings

1. **Performance Degradation is Widespread**
   - D3: 22/29 datasets have worse MAE (76%), 24/27 have worse CRPS (89%)
   - D4: 21/28 datasets have worse MAE (75%), 21/27 have worse CRPS (78%)
   - Mean MAE change: +14.4% (D3), +15.5% (D4)
   - Mean CRPS change: +16.7% (D3), +21.4% (D4)

2. **Not Just Point Estimate vs Distribution Tradeoff**
   - Both MAE (point estimate) AND CRPS (probabilistic) degrade together
   - Correlation between MAE and CRPS changes: r ≈ 0.8-0.9

3. **Fundamental Theoretical Mismatch**
   - USP theory applies to 1-D time series, not 384-D learned embeddings
   - Embeddings lack the LDS-like properties that make preconditioning effective

## Detailed Metric Comparison

### Dataset-Level Results

| Category | D3 MAE | CRPS | Examples |
|----------|--------|------|----------|
| **Improved** (2) | -6% to -9% | -6% to -9% | M1 Monthly, Aus. Elec. Demand |
| **Neutral** (16) | -5% to +10% | Varies | NN5, Tourism, Vehicle Trips |
| **Degraded** (11) | +10% to +137% | +9% to +123% | M3 Other, COVID Deaths, M4 Hourly |

### Worst Performing Datasets

| Dataset | MAE Change | CRPS Change | Characteristics |
|---------|------------|-------------|-----------------|
| M3 Other | +137% | +123% | Mixed frequencies, 174 series |
| COVID Deaths | +46% | +41% | Non-stationary pandemic data |
| Rideshare | +46% | +41% | Hourly, 2304 series, missing values |
| M4 Hourly | +35% | +32% | High frequency, 414 series |
| Traffic Weekly | +32% | +27% | Weekly, 862 series |

### Best Performing Datasets

| Dataset | MAE Change | CRPS Change | Characteristics |
|---------|------------|-------------|-----------------|
| M1 Monthly | -9% | -9% | Monthly, 617 series, clean data |
| Aus. Elec. Demand | -6.5% | -6.5% | 30-min, 5 series, strong seasonality |

## Pattern Analysis

### By Frequency

| Frequency | Mean MAE Change | Count | Pattern |
|-----------|-----------------|-------|---------|
| Hourly | +16% | 6 | Consistently degrades |
| Weekly | +20% | 3 | Degrades significantly |
| Daily | +8% | 10 | Mixed results |
| Monthly | +6% | 8 | Mixed results |

### By Domain

| Domain | Mean MAE Change | Count | Pattern |
|--------|-----------------|-------|---------|
| Healthcare | +30% | 2 | Severe degradation |
| Competition/Mixed | +29% | 7 | Highly variable |
| Traffic | +20% | 5 | Consistent degradation |
| Finance | +10% | 4 | Moderate degradation |
| Energy | -6.5% | 1 | Improvement |

### By Series Count

| Category | Mean MAE Change | Count | Pattern |
|----------|-----------------|-------|---------|
| Few (1-10) | +2% | 4 | Neutral/slight improvement |
| Moderate (11-500) | +20% | 14 | Variable, often degrades |
| Many (501-2000) | +12% | 5 | Moderate degradation |
| Very Many (2000+) | +12% | 6 | Moderate degradation |

## Root Cause Analysis

### 1. USP Theory Does Not Apply to Embeddings

The Universal Sequence Preconditioning paper shows that for **1-dimensional time series** from Linear Dynamical Systems:

```
ỹ_t = y_t + Σ c_i * y_{t-i}
```

This transformation applies a polynomial to the transition matrix, shrinking its spectral radius and making learning easier.

**For embedding-level preconditioning:**
```
ẽ_k = e_k + Σ c_i * e_{k-i}
```

Where `e_k ∈ R^{384}` are learned representations from `in_proj(scaled_target)`.

**Why this fails:**
- Embeddings are NOT time series outputs from an LDS
- Each dimension is a complex, learned combination of input features
- The spectral shrinkage intuition does not transfer
- There's no theoretical justification for this approach

### 2. Missing Reversal Creates Learning Complexity

The config explicitly disables reversal (`embedding_precondition_reverse: false`).

**Without reversal:**
- The `param_proj` layer must implicitly learn to "undo" preconditioning
- This adds unnecessary complexity to the learning task
- The transformer's non-linear mixing makes this extremely difficult
- The model essentially has to learn two things: forecasting AND un-preconditioning

### 3. High-Dimensional Preconditioning Amplifies Issues

- Preconditioning is applied independently to 384 embedding dimensions
- Any issues are amplified 384x compared to 1-D preconditioning
- Cross-dimension correlations in embeddings are not respected

## Hypotheses Tested

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| NLL vs MAE tradeoff | **Rejected** | CRPS also degrades alongside MAE |
| High frequency hurts | **Partial** | Hourly degrades except Aus. Elec. Demand |
| Short horizons worse | **Weak** | Some correlation but not strong (r=-0.1) |
| Many series worse | **Weak** | Slight positive correlation (r=0.07) |
| Non-stationary data hurts | **Supported** | COVID Deaths (+46%) is highly non-stationary |

## Recommendations

### Immediate: Abandon Embedding-Level Preconditioning

The theoretical and empirical evidence strongly suggests that embedding-level preconditioning is fundamentally flawed. The approach should be replaced with alternatives that have proper theoretical grounding.

### Alternative 1: Data-Level Preconditioning (Recommended)

Apply preconditioning to raw target values BEFORE the model:
- Has exact mathematical forward/reverse operations
- Theory from USP paper applies directly
- Already implemented in `uni2ts/src/uni2ts/transform/precondition.py`

**Configuration:**
```yaml
enable_preconditioning: true
precondition_type: chebyshev
precondition_degree: 5
embedding_preconditioning: false  # Disable embedding-level
```

### Alternative 2: Learnable Preconditioning

Use the learnable preconditioning module that can adapt coefficients during training:
- Coefficients are nn.Parameters that get optimized
- Can add reconstruction loss to ensure reversal quality
- Already implemented in `uni2ts/src/uni2ts/module/learnable_precondition.py`

### Alternative 3: Enable Embedding Reversal (Less Recommended)

If embedding-level must be used, enable the compensating filter:
```yaml
embedding_precondition_reverse: true
```

While not a true mathematical inverse, it may partially mitigate the issues.

### Future Research Directions

1. **Understand successful cases**: Why did M1 Monthly and Aus. Elec. Demand improve?
   - Both have relatively simple, seasonal patterns
   - Few series or strong cross-series structure

2. **Theory development**: Can USP theory be extended to high-dimensional embeddings?
   - Would require showing embeddings have LDS-like properties
   - May need different coefficient designs

3. **Hybrid approaches**: Apply preconditioning only to datasets where it helps
   - Use meta-learning to predict when preconditioning will help

## Files Referenced

| File | Purpose |
|------|---------|
| `/scratch/gpfs/EHAZAN/jh1161/uni2ts/eval-runs/standard/evaluation_metrics.csv` | Baseline metrics |
| `/scratch/gpfs/EHAZAN/jh1161/uni2ts/eval-runs/embedding_precond_d3/evaluation_metrics_embedding_precond.csv` | D3 metrics |
| `/scratch/gpfs/EHAZAN/jh1161/uni2ts/eval-runs/embedding_precond_d4/evaluation_metrics_embedding_precond.csv` | D4 metrics |
| `/scratch/gpfs/EHAZAN/jh1161/eval/comparison_table.csv` | Side-by-side comparison |
| `uni2ts/src/uni2ts/module/embedding_precondition.py` | Embedding preconditioning implementation |
| `uni2ts/cli/conf/pretrain/model/moirai_small_embedding_precond.yaml` | Model config |
