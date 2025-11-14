# Training Loss Explanation: PackedNLLLoss

## Summary

**The training loss reported is `train/PackedNLLLoss` (Negative Log-Likelihood Loss).**

### Quick Answer to Your Questions:

1. **What is the training loss?** PackedNLLLoss (Negative Log-Likelihood)
2. **Is it scale-independent?** **YES** ✓ (thanks to automatic per-series standardization)
3. **Can you compare preconditioned vs non-preconditioned?** **YES** ✓ (fair comparison)

---

## How the Training Loss Works

### 1. Data Flow with Scaling

```
Raw Time Series (y_t)
    ↓
[Optional: Preconditioning Transform]  ← Applied in data pipeline
    ↓ (if enabled)
Preconditioned Series (ỹ_t)
    ↓
PackedStdScaler: Standardize per series
    ↓
loc, scale = compute_statistics(ỹ_t)
scaled = (ỹ_t - loc) / scale
    ↓
Model Forward Pass (on scaled data)
    ↓
Predicted Distribution: p(scaled_target | context)
    ↓
PackedNLLLoss: -log p(scaled_target)
    ↓
Loss (scale-independent!)
```

### 2. The Scaler (PackedStdScaler)

**Location:** `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/module/packed_scaler.py:78-122`

**What it does:**
```python
class PackedStdScaler(PackedScaler):
    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        # Computes per-series standardization
        # loc = mean(y_t) for each series
        # scale = std(y_t) for each series (with minimum_scale = 1e-5)
```

**Key properties:**
- **Per-series normalization**: Each time series is standardized independently
- **Minimum scale**: `1e-5` prevents division by zero for flat series
- **Bessel's correction**: Uses `n-1` for unbiased variance estimate

**Example:**
```
Series A: [100, 200, 300] → loc=200, scale=100 → [-1, 0, 1]
Series B: [1, 2, 3]       → loc=2,   scale=1   → [-1, 0, 1]
```

Both become the same scale after standardization!

### 3. The Loss Function (PackedNLLLoss)

**Location:** `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/loss/packed/distribution.py:23-33`

**Implementation:**
```python
class PackedNLLLoss(PackedDistributionLoss):
    def _loss_func(self, pred: Distribution, target, ...) -> torch.Tensor:
        return -pred.log_prob(target)  # Negative log-likelihood
```

**What it computes:**
- Model outputs a distribution: `p(y_scaled | context)`
- Loss = `-log p(y_scaled = target_scaled)`
- This is equivalent to maximum likelihood estimation (MLE)

**Loss aggregation** (`_base.py:76-106`):
1. Compute per-token loss: `-log_prob(target)`
2. Normalize by number of observations per series
3. Average across all series in the batch

**Mathematical form:**
```
Loss = -1/N ∑_{series} ∑_{t in prediction_window} log p(y_scaled[t] | context)

where N = total number of predicted tokens
```

---

## Why the Loss is Scale-Independent

### The Magic: Two-Stage Normalization

1. **Stage 1: Scaler normalizes the data**
   ```
   y_scaled = (y - mean(y)) / std(y)
   ```

2. **Stage 2: NLL loss on normalized data**
   ```
   loss = -log p(y_scaled)
   ```

**Result:** The loss operates on standardized values (mean=0, std=1), making it independent of the original data scale!

### Example: Tourism vs M4

**Tourism Monthly (large values):**
- Raw data: visitor counts in thousands (e.g., 5000, 10000, 15000)
- After scaling: standardized to ~[-2, 0, 2]
- Loss computed on standardized values

**M4 Monthly (smaller values):**
- Raw data: economic indicators (e.g., 5, 10, 15)
- After scaling: standardized to ~[-2, 0, 2]
- Loss computed on standardized values

**Both series have comparable loss values** because they're normalized to the same scale!

---

## Preconditioning and Training Loss

### Does Preconditioning Affect Loss Comparability?

**Answer: YES, you can fairly compare!** Here's why:

### Data Flow Comparison

**Without Preconditioning:**
```
Raw Data (y_t)
    ↓
PackedStdScaler: (y_t - μ) / σ
    ↓
Model → NLL Loss
```

**With Preconditioning:**
```
Raw Data (y_t)
    ↓
PolynomialPrecondition: ỹ_t = y_t - ∑ c_i y_{t-i}
    ↓
PackedStdScaler: (ỹ_t - μ̃) / σ̃
    ↓
Model → NLL Loss
```

### Key Insight

**Both pipelines apply standardization AFTER their respective transforms:**
1. Non-preconditioned: standardizes raw data
2. Preconditioned: standardizes preconditioned data

The scaler ensures both operate on data with:
- Mean ≈ 0
- Std ≈ 1

Therefore, **loss magnitudes are directly comparable!**

### What the Loss Curves Tell You

If you see different loss curves:

**Lower loss with preconditioning** → Preconditioning helps!
- The preconditioned series are easier to model
- Model can achieve better likelihood

**Higher loss with preconditioning** → Preconditioning may hurt
- The preconditioned series are harder to model
- Or optimization hasn't converged yet

**Similar loss** → Preconditioning has neutral effect
- Model performs similarly on both

---

## Configuration

### Where is scaling configured?

**File:** `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/pretrain/model/moirai_small.yaml:17`

```yaml
scaling: true  # Uses PackedStdScaler
# If false, uses PackedNOPScaler (no scaling)
```

**Code:** `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/module.py:121`

```python
self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
```

### Loss function configuration

**File:** `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/pretrain/model/moirai_small.yaml:22-23`

```yaml
loss_func:
  _target_: uni2ts.loss.packed.PackedNLLLoss
```

---

## Practical Implications

### 1. Comparing Training Curves

**You CAN fairly compare:**
- Training loss between preconditioned and non-preconditioned models
- Validation loss between different models
- Loss across different datasets (within the same training run)

**Example comparison:**
```bash
# View tensorboard logs
tensorboard --logdir uni2ts/outputs/pretrain/

# Compare:
# - pretrain_run_A (no preconditioning)
# - pretrain_run_B (with preconditioning)
```

### 2. What to Look For

**In training loss curves:**

1. **Convergence speed**: Does preconditioning converge faster?
2. **Final loss value**: Does preconditioning achieve lower loss?
3. **Stability**: Is training more stable with preconditioning?
4. **Overfitting**: Check train vs validation gap

**Good signs for preconditioning:**
- ✓ Lower final training loss
- ✓ Lower validation loss
- ✓ Faster convergence
- ✓ Similar train/val gap (not overfitting)

**Warning signs:**
- ✗ Higher validation loss (preconditioning may hurt generalization)
- ✗ Larger train/val gap (overfitting to preconditioned space)
- ✗ Training instability

### 3. Example Analysis

Let's say you see:

**Non-preconditioned:**
- Training loss: 4.31 (final)
- Validation loss: 4.35

**Preconditioned:**
- Training loss: 4.04 (final)
- Validation loss: 4.08

**Interpretation:**
- ✓ Preconditioning achieves ~0.27 lower loss (both train & val)
- ✓ Train/val gap similar (0.04 vs 0.04) → no overfitting
- ✓ Preconditioning helps the model!

---

## Technical Details

### NLL Loss Properties

1. **Range**: [0, ∞)
   - Lower is better
   - 0 = perfect prediction
   - ∞ = model assigns 0 probability to observed value

2. **Units**: nats (natural logarithm)
   - To convert to bits: multiply by 1/log(2) ≈ 1.443

3. **Interpretation**:
   ```
   Loss = 4.0 means:
   - Average log probability = -4.0
   - Average probability ≈ e^(-4) ≈ 0.018 = 1.8%
   ```

### Why NLL instead of MSE?

**NLL advantages:**
1. **Probabilistic**: Outputs full distribution, not just point estimate
2. **Uncertainty quantification**: Model learns prediction confidence
3. **Better for forecasting**: Can sample multiple futures
4. **Proper scoring rule**: Encourages calibrated predictions

**MSE disadvantages:**
1. Only learns mean (no uncertainty)
2. Sensitive to outliers
3. Assumes Gaussian errors

---

## Summary Table

| Aspect | Value | Scale-Independent? |
|--------|-------|-------------------|
| **Loss Function** | PackedNLLLoss | ✓ Yes |
| **Scaler** | PackedStdScaler | ✓ Normalizes per-series |
| **Minimum Scale** | 1e-5 | Prevents div-by-zero |
| **Applied Before** | Model forward pass | ✓ Yes |
| **Preconditioning** | Before scaling | ✓ Still normalized |
| **Comparable?** | Precond vs non-precond | ✓ Yes |

---

## Conclusion

### Can You Compare Training Loss? ✓ YES!

**The training loss (PackedNLLLoss) is scale-independent because:**

1. ✓ Data is standardized per-series before computing loss
2. ✓ Standardization happens AFTER preconditioning
3. ✓ Loss operates on normalized values (mean=0, std=1)
4. ✓ Both preconditioned and non-preconditioned models see comparable scales

**Therefore, you can directly compare:**
- Training loss curves
- Validation loss curves
- Final loss values
- Convergence speed

**To compare your models:**

```bash
# View logs
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts
tensorboard --logdir outputs/pretrain/ --port 6006

# Look at:
# - train/PackedNLLLoss
# - val/PackedNLLLoss
```

The model with lower loss (both train and val) is performing better!
