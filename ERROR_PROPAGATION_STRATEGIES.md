# Strategies to Reduce Error Propagation in Hybrid Reversal

**Problem**: When reversing preconditioning using base model predictions as context, errors in base predictions propagate and amplify through the reversal process.

**Context**: With degree 5 Chebyshev coefficients `[0, -1.25, 0, 0.3125, 0]`, the reversal formula is:
```
y[t] = ỹ[t] + 1.25·y_context[t-2] - 0.3125·y_context[t-4]
```

If `y_context` comes from an inaccurate base model, errors compound at each timestep.

---

## Strategy 1: Autoregressive Reversal ⭐ **RECOMMENDED**

**Idea**: Use previously reversed hybrid predictions as context, not base model predictions throughout.

**Implementation**:
```python
# For t < degree: Must use base model (need bootstrap)
# For t >= degree: Use own previously reversed predictions as context

hybrid_context = [input_window, ...empty prediction slots...]
for t in range(pred_len):
    if t < degree:
        # Use base model for positions we haven't reversed yet
        y[t] = reverse_using_mixed_context(base_model, hybrid_context)
    else:
        # Use our own predictions as context
        y[t] = reverse_using_hybrid_context(hybrid_context)

    hybrid_context[input_len + t] = y[t]  # Store for future use
```

**Benefits**:
- Reduces error propagation (uses our own predictions, which may be better)
- Self-correcting if hybrid predictions are more accurate than base
- Still uses base model for initial bootstrap (necessary)

**Implemented in**: `eval_precond_hybrid_autoregressive.py`

**Test with**:
```bash
python -m cli.eval_precond_hybrid_autoregressive \
  base_model.checkpoint_path=/path/to/base.ckpt \
  precond_model.checkpoint_path=/path/to/precond.ckpt \
  data.dataset_name=monash_m3_monthly
```

---

## Strategy 2: Partial Ground Truth Context (Oracle Initialization)

**Idea**: Use ground truth for the first `k` predictions, then switch to base/hybrid for remaining predictions.

**Rationale**:
- Early predictions (t=0,1,2,3) need context from positions that overlap with ground truth
- Using GT for these reduces initial error accumulation
- Later predictions can rely on earlier (more accurate) hybrid predictions

**Implementation**:
```python
k = degree  # or degree + margin

for t in range(pred_len):
    if t < k:
        # Use ground truth as context where available
        context = [input_window, ground_truth[:t]]
    else:
        # Use hybrid predictions as context
        context = [input_window, hybrid[:t]]

    y[t] = reverse_using_context(ỹ[t], context, coeffs)
```

**When to use**:
- Development/debugging: Understand upper bound with partial oracle
- Two-stage training: Train with GT context, fine-tune without
- Semi-supervised: Have GT for some test sequences

**Limitation**: Requires ground truth during evaluation (not pure zero-shot)

---

## Strategy 3: Lower Polynomial Degree

**Idea**: Use degree 2 or 3 instead of degree 5.

**Why it helps**:
- Fewer past timesteps needed as context
- Less error accumulation over time
- Degree 2: Only needs y[t-1] and y[t-2]
- Degree 5: Needs y[t-1] through y[t-5] (more chances for errors)

**Trade-off**:
- ✅ Less error propagation
- ❌ May lose preconditioning benefits (degree 5-10 usually optimal per paper)

**Test different degrees**:
```bash
# Degree 2
python -m cli.eval_precond_hybrid \
  precond_model.precondition_degree=2 ...

# Degree 3
python -m cli.eval_precond_hybrid \
  precond_model.precondition_degree=3 ...
```

**Expected pattern**:
- Degree 2: Less error propagation, but weaker preconditioning effect
- Degree 5: Stronger preconditioning, but more error propagation
- Optimal: Degree 3-4 may balance both concerns

---

## Strategy 4: Ensemble Multiple Base Models

**Idea**: Use predictions from multiple base models, average/median them for more robust context.

**Implementation**:
```python
# Train multiple base models with different seeds
base_models = [base_model_1, base_model_2, base_model_3]

# Get predictions from all
base_preds = [model.predict(input) for model in base_models]

# Ensemble (median or mean)
ensemble_pred = np.median(base_preds, axis=0)

# Use ensemble as context
hybrid = reverse(precond_pred, context=ensemble_pred)
```

**Benefits**:
- More robust context (reduces variance)
- Smooths out individual model errors
- Works with existing infrastructure

**Cost**: Need to run multiple base models (3-5x compute)

---

## Strategy 5: Iterative Refinement

**Idea**: Multiple passes - each iteration uses previous iteration's output as context.

**Algorithm**:
```python
# Iteration 0: Use base model as context
hybrid_v0 = reverse(precond_pred, context=base_pred)

# Iteration 1: Use hybrid_v0 as context
hybrid_v1 = reverse(precond_pred, context=hybrid_v0)

# Iteration 2: Use hybrid_v1 as context
hybrid_v2 = reverse(precond_pred, context=hybrid_v1)

# Continue until convergence or max_iterations
```

**Intuition**: If hybrid predictions are better than base, using them as context improves reversal.

**Stopping criterion**:
- Fixed iterations (e.g., 3-5)
- Convergence: `||hybrid_v{n+1} - hybrid_v{n}|| < ε`
- Validation metric improvement

**When it works**: When precond model captures dynamics better than base model.

**When it fails**: If initial reversal is poor, iterations may diverge.

---

## Strategy 6: Weighted Combination

**Idea**: Don't fully commit to reversal - blend base and reversed predictions.

**Formula**:
```python
# Standard hybrid
hybrid = reverse(precond_pred, base_context)

# Weighted combination
final = α * base_pred + (1-α) * hybrid
```

**Learning α**:
- Fixed: α = 0.5 (equal weight)
- Per-dataset: Optimize α on validation set
- Time-varying: α(t) decreases as t increases (more confident in hybrid later)
- Learned: Train a small network to predict α based on context uncertainty

**Example**:
```python
# More trust in base model early, more trust in hybrid later
α = lambda t: 0.7 * exp(-t/10)  # Decay from 0.7 to 0
final[t] = α(t) * base[t] + (1-α(t)) * hybrid[t]
```

---

## Strategy 7: Uncertainty-Weighted Context

**Idea**: When building context, weight contributions by prediction confidence.

**Implementation**:
```python
# Get prediction uncertainty from base model (std of samples)
base_std = np.std(base_samples, axis=0)  # [pred_len]
confidence = 1 / (1 + base_std)  # Higher confidence = lower std

# Use confidence-weighted context
for t in range(pred_len):
    weighted_context = []
    for i in range(degree):
        pos = t - i - 1
        if pos >= 0:
            weighted_context.append(confidence[pos] * context[pos])

    y[t] = reverse_with_weighted_context(ỹ[t], weighted_context, coeffs)
```

**Benefits**:
- Downweights unreliable base predictions
- Naturally handles heteroscedastic uncertainty
- Requires uncertainty estimates (available from sample variance)

---

## Strategy 8: Direct Prediction Without Reversal

**Idea**: Train the preconditioned model to predict in **original space**, not preconditioned space.

**How**:
- Training: Apply preconditioning to inputs, but loss in original space
- Architecture adds a learned reversal layer
- Model learns to "undo" preconditioning internally

**Benefits**:
- No external reversal needed
- Model can learn optimal reversal strategy
- Avoids error propagation from external context

**Challenge**: Requires retraining models

---

## Strategy 9: Hybrid Training with Base Model Context

**Idea**: Train the preconditioned model **knowing it will use base model context**.

**Training approach**:
```python
# During training:
1. Get base model predictions on training data
2. Train precond model to predict in precond space
3. During loss computation:
   - Reverse using base model predictions as context
   - Compute loss in original space against GT

# This teaches the model to account for base model errors
```

**Benefits**:
- Model learns to compensate for base model weaknesses
- Explicitly optimizes for the hybrid use case
- Can learn to predict "corrections" rather than absolute values

**Challenge**: Requires joint training or fine-tuning stage

---

## Comparison Table

| Strategy | Complexity | Compute Cost | Requires Retraining | Expected Improvement |
|----------|-----------|--------------|---------------------|---------------------|
| **Autoregressive** ⭐ | Medium | 1x | No | High |
| Partial GT | Low | 1x | No | High (if GT available) |
| Lower Degree | Low | 1x | Yes (new model) | Medium |
| Ensemble Base | Low | 3-5x | No | Medium |
| Iterative Refinement | Medium | 2-5x | No | Medium-High |
| Weighted Combination | Low | 1x | No | Medium |
| Uncertainty Weighting | Medium | 1x | No | Medium |
| Direct Prediction | High | 1x | Yes | High |
| Hybrid Training | High | 1.5x | Yes | Very High |

---

## Recommended Approach

**Phase 1: Quick Wins (No Retraining)**
1. ✅ **Try Autoregressive Reversal** - Implemented in `eval_precond_hybrid_autoregressive.py`
2. Test degree 2-4 (less error propagation)
3. Try weighted combination (α=0.5 as baseline)

**Phase 2: If Results Improve**
1. Implement iterative refinement (3-5 iterations)
2. Ensemble multiple base models
3. Optimize α per dataset

**Phase 3: Retrain if Necessary**
1. Train models at degree 3-4 (sweet spot)
2. Implement hybrid training (train knowing base context will be used)

---

## Testing the Strategies

Created new evaluation script: `eval_precond_hybrid_autoregressive.py`

**Test autoregressive reversal**:
```bash
cd /scratch/gpfs/EHAZAN/jh1161/uni2ts

python -m cli.eval_precond_hybrid_autoregressive \
  run_name=test_autoregressive \
  base_model.checkpoint_path=/path/to/base.ckpt \
  precond_model.checkpoint_path=/path/to/precond.ckpt \
  data=monash_cached \
  data.dataset_name=monash_m3_monthly
```

**Compare all approaches**:
```bash
# Standard hybrid (current)
python -m cli.eval_precond_hybrid ...

# Autoregressive hybrid (new)
python -m cli.eval_precond_hybrid_autoregressive ...

# Ground truth context (upper bound)
python -m cli.eval_precond_gt ...
```

---

## Expected Results

**If autoregressive works well**:
```
GT Context:       MAE = 645  (upper bound)
Autoregressive:   MAE = 700-800  (close to GT)
Standard Hybrid:  MAE = 1294  (current)
```

**Why**: Autoregressive uses its own (potentially better) predictions as context instead of relying on base model throughout the prediction horizon.

---

## Mathematical Analysis

**Error propagation in standard hybrid**:
```
e_hybrid[t] = e_precond[t] + Σ c_i · e_base[t-i]
```

Where:
- `e_precond[t]`: Error in preconditioned prediction
- `e_base[t-i]`: Error in base model at position t-i
- Errors accumulate over time

**Error propagation in autoregressive**:
```
e_hybrid[t] = e_precond[t] + Σ c_i · e_hybrid[t-i]
```

Key difference: Uses `e_hybrid` (own errors) not `e_base`.

**If hybrid is better than base** (`|e_hybrid| < |e_base|`):
- Autoregressive propagates smaller errors
- Self-correcting over time

**If hybrid is worse than base**:
- Errors may amplify
- Fall back to standard hybrid or lower degree

---

## Future Work

1. **Learned reversal**: Train a small network to predict optimal reversal weights
2. **Conditional reversal**: Use different strategies based on input characteristics
3. **Multi-model ensemble**: Combine predictions from degree 2, 3, 5 models
4. **Adaptive degree**: Start with high degree, fall back to lower if errors accumulate
5. **Kalman filtering**: Use state-space model to optimally combine base and precond predictions
