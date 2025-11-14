# How Training Loss is Computed During Pretraining

## Overview

The Moirai model uses **Negative Log-Likelihood (NLL) loss** on a **mixture of distributions** during pretraining. This is a probabilistic approach where the model learns to predict full probability distributions rather than just point estimates.

---

## Loss Function: Packed NLL Loss

### Configuration
**File:** `uni2ts/cli/conf/pretrain/model/moirai_small.yaml:22-23`
```yaml
loss_func:
  _target_: uni2ts.loss.packed.PackedNLLLoss
```

### Implementation
**File:** `uni2ts/src/uni2ts/loss/packed/distribution.py:23-33`

```python
class PackedNLLLoss(PackedDistributionLoss):
    def _loss_func(
        self,
        pred: Distribution,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        return -pred.log_prob(target)
```

**Key Point:** The loss is simply the **negative log probability** of the true target values under the predicted distribution.

---

## Distribution: Mixture of 4 Components

### Configuration
**File:** `uni2ts/cli/conf/pretrain/model/moirai_small.yaml:4-10`

```yaml
distr_output:
  _target_: uni2ts.distribution.MixtureOutput
  components:
    - _target_: uni2ts.distribution.StudentTOutput          # Heavy-tailed
    - _target_: uni2ts.distribution.NormalFixedScaleOutput  # Gaussian
    - _target_: uni2ts.distribution.NegativeBinomialOutput  # Count data
    - _target_: uni2ts.distribution.LogNormalOutput         # Positive skewed
```

### Why a Mixture?

Different time series have different characteristics:
- **Student-T**: Handles outliers and heavy tails (financial data, rare events)
- **Normal**: Standard Gaussian distribution (smooth, regular patterns)
- **Negative Binomial**: Count data (discrete events like sales, arrivals)
- **Log-Normal**: Positive-only data with right skew (prices, sizes)

The model learns to **automatically select** which distribution(s) best fit each time series through learned mixture weights.

---

## Mixture Distribution Log Probability

### Implementation
**File:** `uni2ts/src/uni2ts/distribution/mixture.py:81-119`

```python
def log_prob(self, value: torch.Tensor) -> torch.Tensor:
    # Get mixture weights (learned by model)
    weights_log_probs = self.weights.logits.expand(...)

    # Compute log probability under each component distribution
    components_log_probs = torch.stack([
        torch.where(
            comp.support.check(value),  # Check if value is valid for this component
            comp.log_prob(value),       # Log prob under this component
            float("-inf"),              # Invalid: zero probability
        )
        for comp in self.components     # For each of 4 distributions
    ])

    # Combine via log-sum-exp (mixture formula)
    return torch.logsumexp(weights_log_probs + components_log_probs, dim=0)
```

### Mathematical Formula

For a mixture of K distributions:

```
p(y | x) = Σᵢ₌₁ᴷ wᵢ · pᵢ(y | x)

log p(y | x) = log(Σᵢ₌₁ᴷ wᵢ · pᵢ(y | x))
             = log-sum-exp(log wᵢ + log pᵢ(y | x))
```

Where:
- `wᵢ` = weight for component i (learned, sums to 1)
- `pᵢ(y | x)` = probability of y under distribution i
- K = 4 (Student-T, Normal, Negative Binomial, Log-Normal)

**Loss** = `-log p(y | x)` (negative log-likelihood)

---

## Training Step Flow

### Code Flow
**File:** `uni2ts/src/uni2ts/model/moirai/pretrain.py:149-189`

```python
def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
    # 1. Forward pass: Model predicts distribution
    distr = self(
        **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
    )
    # distr is a Mixture distribution with 4 components

    # 2. Compute NLL loss
    loss = self.hparams.loss_func(
        pred=distr,                    # Predicted mixture distribution
        **{
            field: batch[field]
            for field in [
                "target",              # True values (ground truth)
                "prediction_mask",     # Which tokens to predict
                "observed_mask",       # Which values are observed (not missing)
                "sample_id",           # Sample identifier
                "variate_id",          # Variate identifier
            ]
        },
    )

    # 3. Log the loss
    self.log(f"train/{self.hparams.loss_func.__class__.__name__}", loss, ...)

    # 4. Return loss for backpropagation
    return loss
```

### Detailed Steps

1. **Model Forward Pass** (`self(...)`)
   - Input: Masked time series patches
   - Output: Mixture distribution parameters (means, scales, degrees of freedom, weights)

2. **Compute Component Log Probabilities**
   - For each of 4 distributions, compute `log pᵢ(y | x)`
   - Check if target value is in support of each distribution

3. **Compute Mixture Log Probability**
   - Combine using: `log p(y | x) = log-sum-exp(log wᵢ + log pᵢ(y | x))`

4. **Apply Masking and Reduction**
   - Only compute loss on **predicted tokens** (where `prediction_mask=1`)
   - Only compute loss on **observed values** (where `observed_mask=1`)
   - Average over valid tokens

5. **Negative Log-Likelihood**
   - Loss = `-log p(y | x)`
   - Lower loss = higher probability = better predictions

---

## Packed Loss: Handling Variable-Length Sequences

### Why "Packed"?

The model handles multiple time series of different lengths in a single batch. The "packed" format efficiently handles this:

- **sample_id**: Identifies which sample each token belongs to
- **variate_id**: Identifies which variable (for multivariate data)
- **prediction_mask**: Marks which tokens should be predicted
- **observed_mask**: Marks which values are actually observed (not NaN)

### Reduction Formula

**File:** `uni2ts/src/uni2ts/loss/packed/_base.py:76-106`

```python
def reduce_loss(self, loss, prediction_mask, observed_mask, sample_id, variate_id):
    # Create mask for same sample + same variate
    id_mask = (sample_id.unsqueeze(-1) == sample_id.unsqueeze(-2)) &
              (variate_id.unsqueeze(-1) == variate_id.unsqueeze(-2))

    # Combine prediction mask and observed mask
    mask = prediction_mask.unsqueeze(-1) * observed_mask

    # Count observations per time series
    tobs = (id_mask * mask.sum(dim=-1, keepdim=True)).sum(dim=-2, keepdim=True)

    # Count prediction tokens
    nobs = (id_mask * prediction_mask.unsqueeze(-1)).sum(dim=-2, keepdim=True)
    nobs = torch.where(nobs == 0, nobs, 1 / nobs).sum()

    # Normalize loss by observation count
    loss = loss / (tobs * nobs)

    # Sum masked loss
    return (loss * mask).sum()
```

This ensures:
1. Loss is computed only on predicted, observed tokens
2. Loss is normalized by number of observations per time series
3. Fair weighting across samples of different lengths

---

## Effect of Preconditioning

### When Preconditioning is Enabled

**File:** `uni2ts/src/uni2ts/model/moirai/pretrain.py:386-399`

If `enable_preconditioning=True`:
```python
# Data transformation pipeline includes:
+ PolynomialPrecondition(
    polynomial_type="chebyshev",  # or "legendre"
    degree=5,
    target_field="target",
)
```

**Impact on Loss:**
1. **Target values are transformed** before being fed to the model:
   ```
   ỹₜ = yₜ - Σᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ
   ```
   where `cᵢ` are Chebyshev/Legendre polynomial coefficients

2. **Model predicts distribution over transformed targets** `ỹₜ`

3. **Loss is computed on transformed space:**
   ```
   Loss = -log p(ỹₜ | x)
   ```

4. **During inference:** Predictions must be **reversed** back to original space

### Why This Helps

Polynomial preconditioning can:
- Remove strong autocorrelations
- Make the distribution more Gaussian-like
- Simplify the prediction task for the model
- Potentially improve generalization

---

## Summary

### Loss Computation Pipeline

```
Input Batch
    ↓
[Preconditioning] (if enabled)
    ↓ ỹₜ
Model Forward Pass
    ↓ Mixture Distribution: p(y|x) = Σᵢ wᵢ · pᵢ(y|x)
    ├─ Student-T distribution
    ├─ Normal distribution
    ├─ Negative Binomial distribution
    └─ Log-Normal distribution
    ↓
Compute Log Probability: log p(y|x) = log-sum-exp(log wᵢ + log pᵢ(y|x))
    ↓
Apply Masks (prediction_mask, observed_mask)
    ↓
Negative Log-Likelihood: -log p(y|x)
    ↓
Normalize by observation count
    ↓
Scalar Loss Value
    ↓
Backpropagation
```

### Key Takeaways

1. **Probabilistic Approach**: Model learns full distributions, not just point estimates
2. **Mixture Model**: Automatically adapts to different data characteristics
3. **NLL Loss**: Maximizes likelihood of observed data under predicted distribution
4. **Packed Format**: Efficiently handles variable-length, multivariate time series
5. **Masked Prediction**: Only predicts on specified tokens (supports various masking strategies)
6. **Preconditioning**: Optional transformation to simplify prediction task

---

## Related Files

### Core Loss Implementation
- `uni2ts/src/uni2ts/loss/packed/distribution.py` - PackedNLLLoss
- `uni2ts/src/uni2ts/loss/packed/_base.py` - Base class with reduction logic

### Distribution Implementation
- `uni2ts/src/uni2ts/distribution/mixture.py` - Mixture distribution
- `uni2ts/src/uni2ts/distribution/student_t.py` - Student-T component
- `uni2ts/src/uni2ts/distribution/normal.py` - Normal component
- `uni2ts/src/uni2ts/distribution/negative_binomial.py` - Neg. Binomial component
- `uni2ts/src/uni2ts/distribution/log_normal.py` - Log-Normal component

### Training Logic
- `uni2ts/src/uni2ts/model/moirai/pretrain.py` - MoiraiPretrain.training_step()
- `uni2ts/cli/conf/pretrain/model/moirai_small.yaml` - Config

### Preconditioning
- `uni2ts/src/uni2ts/transform/precondition.py` - PolynomialPrecondition

---

*Last Updated: 2025-11-05*
*Location: /scratch/gpfs/EHAZAN/jh1161/TRAINING_LOSS_COMPUTATION.md*
