# Covariate Preconditioning Guide

**Status**: Optional Extension (Not in Current Implementation)

---

## Current Behavior

The current implementation **only preconditions the target variable** (`target` field).

Covariates (`past_feat_dynamic_real`) are **NOT preconditioned**.

---

## How to Enable Covariate Preconditioning

If you want to experiment with preconditioning covariates, you have two options:

### Option 1: Add Second Preconditioning Transform (Recommended)

Modify `src/uni2ts/model/moirai/pretrain.py`:

```python
def default_train_transform():
    # Precondition target
    transform = PolynomialPrecondition(
        polynomial_type=self.hparams.precondition_type,
        degree=self.hparams.precondition_degree,
        target_field="target",
        enabled=self.hparams.enable_preconditioning,
        store_original=False,
    )

    # NEW: Precondition covariates (if enabled)
    if self.hparams.get("precondition_covariates", False):
        transform = transform + PolynomialPrecondition(
            polynomial_type=self.hparams.precondition_type,
            degree=self.hparams.precondition_degree,
            target_field="past_feat_dynamic_real",  # ← Different field
            enabled=True,
            store_original=False,
        )

    # Rest of transforms...
```

Add to config (`cli/conf/pretrain/model/moirai_small.yaml`):
```yaml
precondition_covariates: false  # Set to true to enable
```

### Option 2: Multi-Field Preconditioning Transform

Create a new transform that handles multiple fields:

```python
# In src/uni2ts/transform/precondition.py

@dataclass
class MultiFieldPrecondition(Transformation):
    """
    Apply preconditioning to multiple fields.

    Args:
        fields: List of field names to precondition
        optional_fields: Fields that may not exist
        polynomial_type: "chebyshev" or "legendre"
        degree: Polynomial degree
    """
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = tuple()
    polynomial_type: str = "chebyshev"
    degree: int = 5
    enabled: bool = True

    def __post_init__(self):
        if not self.enabled:
            return
        # Create a PolynomialPrecondition instance
        self.precond = PolynomialPrecondition(
            polynomial_type=self.polynomial_type,
            degree=self.degree,
            enabled=True,
        )

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return data_entry

        # Apply to all specified fields
        for field in self.fields:
            if field in data_entry:
                self.precond.target_field = field
                data_entry = self.precond(data_entry)

        # Apply to optional fields if present
        for field in self.optional_fields:
            if field in data_entry:
                self.precond.target_field = field
                data_entry = self.precond(data_entry)

        return data_entry
```

Usage:
```python
transform = MultiFieldPrecondition(
    fields=("target",),
    optional_fields=("past_feat_dynamic_real",),
    polynomial_type="chebyshev",
    degree=5,
    enabled=True,
)
```

---

## Important Considerations

### 1. Reversal During Inference

If you precondition covariates, you need to:
- Store the preconditioning coefficients for each field
- Reverse the preconditioning on predictions
- Handle covariates during evaluation

### 2. Different Temporal Characteristics

Covariates may have:
- Different frequencies than the target
- Different stationarity properties
- Different optimal polynomial degrees

**Recommendation**: Use different degrees for target vs covariates if needed.

### 3. Inference Complexity

Preconditioning covariates adds complexity:
- Must precondition covariates at inference time
- Must track which fields were preconditioned
- Must reverse correctly for evaluation

---

## Experimental Design

If you want to test covariate preconditioning:

### Experiment 1: Target Only (Baseline)
```bash
sbatch --export=precondition_covariates=false pretrain_moirai_precond.slurm
```

### Experiment 2: Target + Covariates
```bash
sbatch --export=precondition_covariates=true pretrain_moirai_precond.slurm
```

### Comparison
Compare performance on datasets with rich covariates:
- Electricity demand (with temperature, hour-of-day)
- Traffic (with day-of-week, weather)
- Energy (with seasonality features)

---

## Theoretical Justification

### Why Target-Only is Standard

1. **Paper Focus**: The original paper preconditions the **output dynamics**
2. **Autoregressive Models**: Target has strongest temporal dependencies
3. **Simplicity**: Easier to implement and validate
4. **Proven**: Target-only preconditioning has demonstrated results

### When Covariates Might Help

1. **Strong Temporal Structure**: If covariates have AR-like patterns
2. **Similar Dynamics**: If covariates evolve similarly to target
3. **Shared Conditioning**: If both suffer from ill-conditioning

### When Covariates Might NOT Help

1. **Static Features**: Time-invariant covariates (no temporal structure)
2. **Different Frequencies**: Covariates at different timescales
3. **Already Well-Conditioned**: If covariate dynamics are simple
4. **Exogenous Shocks**: If covariates are white noise or random

---

## Recommendation

**Start with target-only preconditioning** (current implementation):

✅ **Pros**:
- Matches the paper's approach
- Simpler to implement and debug
- Proven to work
- Easier to interpret results

❓ **Consider covariates IF**:
- You have strong evidence they need preconditioning
- Initial experiments show room for improvement
- Covariates have clear temporal dependencies
- You're willing to add implementation complexity

---

## Code Location

Current implementation (target-only):
- `src/uni2ts/transform/precondition.py` - Transform implementation
- `src/uni2ts/model/moirai/pretrain.py` line 388 - Where it's applied
- `cli/conf/pretrain/model/moirai_small_precond.yaml` - Configuration

To modify:
1. Edit `pretrain.py` to add covariate preconditioning
2. Update config files to add `precondition_covariates` flag
3. Test on small dataset first
4. Compare results with target-only

---

## Related Questions

**Q: What about static covariates?**
A: Static covariates (e.g., store ID, category) have no temporal structure, so preconditioning doesn't apply.

**Q: What about time features (hour, day, etc.)?**
A: Time features are usually categorical or cyclical. Preconditioning is designed for continuous time series, so probably not beneficial.

**Q: Should we use the same degree for target and covariates?**
A: Not necessarily. Optimal degrees might differ based on:
- Stationarity of each series
- Noise levels
- Temporal structure

**Q: Does the paper mention covariate preconditioning?**
A: The paper focuses on preconditioning the **target sequence** (outputs), not inputs. Extending to covariates is a research question.

---

## Summary

| Aspect | Current (Target-Only) | With Covariates |
|--------|----------------------|-----------------|
| Complexity | ✅ Simple | ⚠️ More complex |
| Paper Support | ✅ Yes | ❌ Not covered |
| Implementation | ✅ Ready | ⚠️ Needs modification |
| Use Case | ✅ General | ⚠️ Specific datasets |
| Recommendation | ✅ **Start here** | ⚠️ Experiment later |

**Default: Use target-only preconditioning** (current implementation)

**Future work**: If results are promising, experiment with covariate preconditioning as an ablation study.

---

**Last Updated**: 2025-11-01
