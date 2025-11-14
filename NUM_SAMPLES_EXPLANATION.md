# How `num_samples` Works During Evaluation

## Quick Answer

**`num_samples`** determines how many **random samples** to draw from the predicted probability distribution to form an empirical forecast distribution.

- **Default:** `num_samples = 100`
- **Purpose:** Convert probabilistic predictions into point estimates and uncertainty quantification

---

## The Evaluation Pipeline

### Step 1: Model Predicts a Distribution

**File:** `uni2ts/src/uni2ts/model/moirai/forecast.py:345`

```python
distr = self._get_distr(...)  # Model outputs: Mixture(StudentT, Normal, NegBinom, LogNorm)
```

The model outputs a **probability distribution**, not a single prediction:
```
p(y | x) = w₁·StudentT + w₂·Normal + w₃·NegBinom + w₄·LogNorm
```

### Step 2: Sample from the Distribution

**File:** `uni2ts/src/uni2ts/model/moirai/forecast.py:345`

```python
preds = distr.sample(torch.Size((num_samples,)))
# Shape: [num_samples, prediction_length]
# e.g., [100, 18] for 100 samples of 18-step ahead predictions
```

**What this does:**
- Draws `num_samples` independent random trajectories from `p(y | x)`
- Each sample is one possible future realization
- Together, they form an **empirical distribution**

### Step 3: Create SampleForecast

**GluonTS converts samples into a forecast object:**

```python
forecast = SampleForecast(
    samples=preds,  # [num_samples, prediction_length]
    start_date=...,
    item_id=...
)
```

### Step 4: Compute Metrics

Different metrics use the samples differently:

```python
# Point metrics (deterministic)
forecast.mean        # Mean of samples: samples.mean(axis=0)
forecast.median      # Median: np.median(samples, axis=0)

# Quantile metrics
forecast.quantile(0.1)  # 10th percentile
forecast.quantile(0.9)  # 90th percentile

# Probabilistic metrics
CRPS(forecast, target)  # Uses ALL samples for empirical CDF
```

---

## How Metrics Use `num_samples`

### **Deterministic Metrics** (MSE, MAE, RMSE)

**Config:** `uni2ts/cli/conf/eval/default.yaml:10-18`

```yaml
metrics:
  - _target_: gluonts.ev.metrics.MSE      # Mean Squared Error
  - _target_: gluonts.ev.metrics.MAE      # Mean Absolute Error
  - _target_: gluonts.ev.metrics.RMSE     # Root Mean Squared Error
```

**Computation:**
```python
# Use MEAN of samples as point prediction
y_pred = forecast.mean  # = samples.mean(axis=0)
MSE = ((y_true - y_pred) ** 2).mean()
MAE = abs(y_true - y_pred).mean()
```

**Effect of `num_samples`:**
- More samples → more accurate estimate of true mean
- Law of Large Numbers: `mean(samples) → E[Y | X]` as `num_samples → ∞`
- **Recommended:** ≥ 100 samples for stable mean estimate

---

### **Quantile Metrics** (MSIS, Mean Weighted Quantile Loss)

```yaml
metrics:
  - _target_: gluonts.ev.metrics.MSIS  # Mean Scaled Interval Score
  - _target_: gluonts.ev.metrics.MeanWeightedSumQuantileLoss
    quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
```

**Computation:**
```python
# Use quantiles of samples
q10 = forecast.quantile(0.1)  # = np.quantile(samples, 0.1, axis=0)
q90 = forecast.quantile(0.9)  # = np.quantile(samples, 0.9, axis=0)

MSIS = 2 * (q90 - q10) + penalties_for_missed_coverage
```

**Effect of `num_samples`:**
- More samples → smoother, more accurate quantile estimates
- **Rule of thumb:** For `k` quantiles, need `≥ 10/k` samples per quantile
  - For 0.1 quantile: need ≥ 10/0.1 = 100 samples
  - For 0.05 quantile: need ≥ 200 samples
- **Recommended:** 100-200 samples for standard quantile evaluation

---

### **Probabilistic Metrics** (CRPS - if used)

**CRPS (Continuous Ranked Probability Score):**
```python
# Measures distance between empirical CDF and true value
# Uses ALL samples to build empirical distribution
F_empirical(y) = (1/num_samples) * Σᵢ 𝟙(sampleᵢ ≤ y)

CRPS = ∫ (F_empirical(y) - 𝟙(y ≥ y_true))² dy
```

**Effect of `num_samples`:**
- CRPS is **most sensitive** to `num_samples`
- More samples → better approximation of true distribution
- **Recommended:** 200-1000 samples for accurate CRPS

---

## Visualization: What Samples Represent

### Example: Predicting Next 18 Months

```python
num_samples = 100
prediction_length = 18

# Model predicts distribution
distribution = model(context)  # p(y | x)

# Draw 100 trajectories
samples = distribution.sample((100,))
# Shape: [100, 18]
# samples[0]: [102.3, 105.1, 107.8, ..., 145.2]  ← Trajectory 1
# samples[1]: [101.8, 104.5, 108.2, ..., 142.7]  ← Trajectory 2
# samples[2]: [103.1, 106.3, 109.1, ..., 148.9]  ← Trajectory 3
# ...
# samples[99]: [102.5, 105.9, 107.3, ..., 146.1]  ← Trajectory 100

# Compute statistics
mean_forecast = samples.mean(axis=0)    # [18,] average trajectory
std_forecast = samples.std(axis=0)      # [18,] uncertainty at each step
q10_forecast = np.quantile(samples, 0.1, axis=0)  # 10th percentile
q90_forecast = np.quantile(samples, 0.9, axis=0)  # 90th percentile
```

**Visual representation:**
```
Future Time →
       t+1   t+2   t+3   ...  t+18
     ┌─────┬─────┬─────┬─────┬─────┐
s₁   │102.3│105.1│107.8│ ... │145.2│  ← Sample 1
s₂   │101.8│104.5│108.2│ ... │142.7│  ← Sample 2
s₃   │103.1│106.3│109.1│ ... │148.9│  ← Sample 3
...  │ ... │ ... │ ... │ ... │ ... │
s₁₀₀ │102.5│105.9│107.3│ ... │146.1│  ← Sample 100
     └─────┴─────┴─────┴─────┴─────┘
       ↓     ↓     ↓           ↓
Mean: 102.4 105.5 108.0  ...  146.0   ← Point forecast
Std:   0.8   1.2   1.5   ...   3.2   ← Uncertainty grows
```

---

## Effect of Changing `num_samples`

### **Low `num_samples` (e.g., 10)**

**Pros:**
- ✅ Fast evaluation
- ✅ Less memory

**Cons:**
- ❌ Noisy estimates of mean/quantiles
- ❌ High variance across runs
- ❌ Unreliable CRPS/probabilistic metrics

**Example:**
```
With 10 samples:
  Mean estimate: 105.3 ± 2.1  (high variance)
  0.1 quantile: 99.2 or 101.5 (unstable, depends on lucky draws)
```

---

### **Medium `num_samples` (e.g., 100)** ⭐ **DEFAULT**

**Pros:**
- ✅ Good balance of speed and accuracy
- ✅ Stable mean/median estimates
- ✅ Reasonable quantile estimates
- ✅ Acceptable for most metrics

**Cons:**
- ⚠️ May be noisy for extreme quantiles (0.01, 0.99)
- ⚠️ CRPS can have some variance

**Recommended for:** Standard evaluation benchmarks

---

### **High `num_samples` (e.g., 1000)**

**Pros:**
- ✅ Very accurate mean/quantile estimates
- ✅ Stable CRPS and probabilistic metrics
- ✅ Low variance across runs
- ✅ Captures tail behavior well

**Cons:**
- ❌ 10× slower evaluation
- ❌ 10× more memory

**Recommended for:** Final benchmark results, paper submissions

---

## Configuration Files

### **Evaluation Config**

**File:** `uni2ts/cli/conf/eval/model/moirai_1.1_R_small.yaml`
```yaml
_target_: uni2ts.model.moirai.MoiraiForecast
prediction_length: ???
context_length: ???
patch_size: ???
num_samples: 100  ← Controls sampling during evaluation
```

### **Pretraining Config**

**File:** `uni2ts/cli/conf/pretrain/model/moirai_small.yaml`
```yaml
_target_: uni2ts.model.moirai.MoiraiPretrain
num_samples: 100  ← Used for validation during training
```

---

## Common Misconceptions

### ❌ **Myth 1:** "More samples = better predictions"

**Reality:** More samples give better **estimates** of the mean/quantiles, but don't change the underlying prediction quality. The model's distribution `p(y|x)` is fixed.

---

### ❌ **Myth 2:** "I should set num_samples=1 for point forecasts"

**Reality:** Even for MSE/MAE (point metrics), you want `num_samples ≥ 100` to get a stable estimate of the mean. With 1 sample, you get a random draw, not the expected value.

---

### ❌ **Myth 3:** "num_samples affects training"

**Reality:** During **pretraining**, the model is trained via NLL loss, which doesn't require sampling. `num_samples` is only used during **validation** and **evaluation**.

---

## When to Change `num_samples`

### **Use Higher `num_samples` (200-1000) When:**

1. Computing **CRPS** or probabilistic metrics
2. Estimating **extreme quantiles** (0.01, 0.99)
3. Running **final benchmarks** for publication
4. Dataset has **high variance** or **heavy tails**
5. You need **low variance** across multiple runs

### **Use Lower `num_samples` (50) When:**

1. **Quick validation** during development
2. **Memory constrained** environments
3. Only care about **mean/median** (not quantiles/CRPS)
4. Running **many experiments** (need speed)

### **Keep Default (100) When:**

1. Standard **MSE/MAE/MASE** evaluation
2. Typical **quantile metrics** (0.1, 0.5, 0.9)
3. **Benchmark comparisons** with other papers
4. **Good enough** for most purposes

---

## Code Example: Changing `num_samples`

### **Via Command Line:**

```bash
# Quick evaluation with 50 samples
python -m cli.eval \
  model=moirai_1.1_R_small \
  model.num_samples=50 \
  data=monash_cached \
  data.dataset_name=tourism_monthly

# High-quality evaluation with 500 samples
python -m cli.eval \
  model=moirai_1.1_R_small \
  model.num_samples=500 \
  data=monash_cached \
  data.dataset_name=tourism_monthly
```

### **Via SLURM Script:**

```bash
NUM_SAMPLES=${NUM_SAMPLES:-100}

python -m cli.eval \
  model=moirai_1.1_R_small \
  model.num_samples=$NUM_SAMPLES \
  ...

# Run with different settings:
sbatch --export=NUM_SAMPLES=200 eval_script.slurm
```

---

## Performance Trade-offs

| num_samples | Evaluation Time | Memory | MSE Stability | CRPS Accuracy | Recommended Use |
|-------------|----------------|--------|---------------|---------------|-----------------|
| 10          | ~10s           | Low    | ±5%          | Poor          | Debug only      |
| 50          | ~30s           | Low    | ±2%          | Moderate      | Quick checks    |
| **100**     | **~60s**       | **Med**| **±1%**      | **Good**      | **Default** ⭐  |
| 200         | ~120s          | Med    | ±0.5%        | Very Good     | Final results   |
| 500         | ~300s          | High   | ±0.2%        | Excellent     | Publications    |
| 1000        | ~600s          | High   | ±0.1%        | Excellent     | Benchmarks      |

*Times are approximate for a single dataset on 1 GPU*

---

## Summary

### **Key Points:**

1. **`num_samples` controls sampling from predicted distribution**
   - Not the number of predictions
   - Number of random trajectories drawn from `p(y|x)`

2. **Different metrics have different sample requirements**
   - MSE/MAE: ≥100 for stable mean
   - Quantiles: ≥100-200 for smooth estimates
   - CRPS: ≥200-1000 for accuracy

3. **Default (100) is usually good enough**
   - Balances speed and accuracy
   - Stable for most metrics
   - Standard in research papers

4. **More samples ≠ better model**
   - Just better estimates of mean/quantiles
   - Model quality determined by training

5. **Trade-off: Accuracy vs Speed**
   - 10× samples → 10× slower, but only ~√10× more accurate
   - Diminishing returns beyond 200-500 samples

### **Rule of Thumb:**

- **Development/debugging:** 50 samples
- **Standard evaluation:** 100 samples ⭐
- **Final benchmarks:** 200-500 samples
- **Probabilistic evaluation (CRPS):** 500-1000 samples

---

*Last Updated: 2025-11-05*
*Location: /scratch/gpfs/EHAZAN/jh1161/NUM_SAMPLES_EXPLANATION.md*
