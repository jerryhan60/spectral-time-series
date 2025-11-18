# Preconditioning Usage Guide

This guide explains how to use the **Universal Sequence Preconditioning** feature in Uni2TS, based on the paper by Marsden & Hazan (2025).

## ⚠️ CRITICAL UPDATE (2025-11-17)

**The implementation was fixed on 2025-11-17 to correct two critical bugs:**
1. Coefficient extraction now uses power basis (not Chebyshev/Legendre basis)
2. Forward preconditioning now uses ADDITION (not subtraction) as per Algorithm 1

**All models trained before this date need to be retrained.** See `CRITICAL_FIXES_PRECONDITIONING.md`.

## What is Preconditioning?

Preconditioning applies polynomial transformations (Chebyshev or Legendre) to time series data before training. This improves the condition number of hidden transition matrices and can lead to better forecasting performance.

**Mathematical formulation (as per Algorithm 1 of the paper):**
```
Forward:  ỹₜ = yₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  for t > n  (ADDITION)
Reverse:  yₜ = ỹₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  for t > n  (SUBTRACTION)
```

where `cᵢ` are monic polynomial coefficients in power basis and `n` is the degree.

## Implementation

The preconditioning feature has been integrated into:
- `/src/uni2ts/transform/precondition.py` - Transform classes
- `/src/uni2ts/model/moirai/pretrain.py` - Pretraining model
- `/src/uni2ts/model/moirai/finetune.py` - Finetuning model
- `/test/transform/test_precondition.py` - Unit tests

## Quick Start

### Option 1: Use Preconfigured Model

```bash
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_experiment \
  model=moirai_small_precond \
  data=lotsa_v1_unweighted
```

### Option 2: Override Parameters

```bash
python -m cli.train \
  -cp conf/pretrain \
  run_name=my_precond_run \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  model.enable_preconditioning=true \
  model.precondition_type=chebyshev \
  model.precondition_degree=5
```

### Option 3: Test Different Configurations

```bash
# Chebyshev degree 2
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_cheb_2 \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  model.enable_preconditioning=true \
  model.precondition_type=chebyshev \
  model.precondition_degree=2

# Chebyshev degree 10
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_cheb_10 \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  model.enable_preconditioning=true \
  model.precondition_type=chebyshev \
  model.precondition_degree=10

# Legendre degree 5
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_leg_5 \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  model.enable_preconditioning=true \
  model.precondition_type=legendre \
  model.precondition_degree=5
```

## Configuration Parameters

### `enable_preconditioning` (bool, default: false)
- **Description**: Enable or disable preconditioning
- **Values**: `true` or `false`
- **Usage**: Toggle this to compare preconditioned vs. baseline training

### `precondition_type` (str, default: "chebyshev")
- **Description**: Type of polynomial to use
- **Values**: `"chebyshev"` or `"legendre"`
- **Notes**: Paper suggests both perform similarly

### `precondition_degree` (int, default: 5)
- **Description**: Degree of the polynomial
- **Values**: 1-10 (recommended: 2-10)
- **Notes**:
  - Higher degrees may improve conditioning but increase numerical instability
  - Paper recommends degree ≤ 10
  - Degrees > 10 will trigger a warning

## How It Works

### Training Pipeline

1. **Data Loading**: Time series loaded from dataset
2. **Preconditioning Applied**: `PolynomialPrecondition` transform is applied **first** in the transform chain
3. **Normal Training**: Remaining transforms and model training proceed as usual
4. **Metadata Stored**: Preconditioning coefficients and parameters are stored with each batch

### Transform Chain Order

```python
# With preconditioning enabled
PolynomialPrecondition(degree=5, enabled=True)
  → SampleDimension()
  → GetPatchSize()
  → PatchCrop()
  → ... (remaining transforms)
```

### Evaluation/Inference

During evaluation and finetuning:
- The **same preconditioning** is applied to validation/test data
- Predictions are made in the preconditioned space
- **Reversal** can be applied using `ReversePrecondition` transform (for post-processing)

## Experiment Recommendations

Based on the plan document, here's a recommended experiment structure:

### Phase 1: Baseline
```bash
python -m cli.train \
  -cp conf/pretrain \
  run_name=baseline_small \
  model=moirai_small \
  data=lotsa_v1_unweighted
```

### Phase 2: Preconditioning Sweep

```bash
# Primary experiment: Chebyshev degree 5
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_cheb_5 \
  model=moirai_small_precond \
  data=lotsa_v1_unweighted

# Lower degree
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_cheb_2 \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  model.enable_preconditioning=true \
  model.precondition_degree=2

# Higher degree
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_cheb_10 \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  model.enable_preconditioning=true \
  model.precondition_degree=10

# Legendre comparison
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_leg_5 \
  model=moirai_small \
  data=lotsa_v1_unweighted \
  model.enable_preconditioning=true \
  model.precondition_type=legendre \
  model.precondition_degree=5
```

## Testing the Implementation

### Basic Functionality Test

```python
# In Python/IPython
from uni2ts.transform import PolynomialPrecondition, ReversePrecondition
import numpy as np

# Create transform
precond = PolynomialPrecondition(
    polynomial_type="chebyshev",
    degree=5,
    enabled=True
)

# Test on synthetic data
data = {"target": np.random.randn(100)}
preconditioned = precond(data)

print(f"Original shape: {data['target'].shape}")
print(f"Preconditioned shape: {preconditioned['target'].shape}")
print(f"Metadata stored: {preconditioned.keys()}")

# Test reversal
reverse = ReversePrecondition()
preconditioned["prediction"] = preconditioned["target"].copy()
restored = reverse(preconditioned)

# Verify round-trip
np.testing.assert_allclose(
    restored["prediction"],
    data["target"],
    rtol=1e-8
)
print("✓ Round-trip test passed!")
```

### Unit Tests

```bash
# Activate your environment
source venv/bin/activate

# Run preconditioning tests (requires pytest)
python -m pytest test/transform/test_precondition.py -v

# Run specific test
python -m pytest test/transform/test_precondition.py::TestPreconditionRoundTrip -v
```

## Expected Outcomes

According to the paper and experiment plan:

### Success Indicators
- ✓ Training completes without NaN/Inf values
- ✓ Loss curves are stable
- ✓ Improved MSE/MAE/MASE on evaluation datasets
- ✓ Better performance on datasets with complex dynamics

### Potential Improvements
- Reduced forecasting error (MSE, MAE, MASE)
- Better probabilistic forecasts (lower CRPS)
- Improved performance on long-horizon forecasting
- Benefits strongest for datasets with high temporal dependencies

## Troubleshooting

### Issue: NaN or Inf during training
**Solution**: Try lower degree (e.g., 2-5) or disable preconditioning

### Issue: No improvement over baseline
**Explanation**: Preconditioning may not benefit all datasets equally. Datasets with simple dynamics or weak temporal dependencies may see minimal gains.

### Issue: Training slower than baseline
**Explanation**: Preconditioning adds minimal overhead (O(n·d) per sequence). If significantly slower, check for other bottlenecks.

### Issue: Import errors
**Solution**: Ensure transforms are properly exported in `__init__.py`:
```python
from uni2ts.transform import PolynomialPrecondition, ReversePrecondition
```

## Advanced Usage

### Custom Preconditioning in Code

```python
from uni2ts.transform import PolynomialPrecondition, Chain
from uni2ts.transform import SampleDimension, GetPatchSize

# Create custom transform chain
transform = (
    PolynomialPrecondition(
        polynomial_type="chebyshev",
        degree=7,
        enabled=True,
        store_original=True  # Keep original for debugging
    )
    + SampleDimension(max_dim=128)
    + GetPatchSize(...)
    # ... other transforms
)
```

### Disable Preconditioning at Runtime

```python
# In your config or code
precond = PolynomialPrecondition(
    enabled=False  # Acts as identity transform
)
```

### Using Different Degrees for Different Datasets

Modify the model's `train_transform_map` to use dataset-specific degrees:

```python
def train_transform_map(self):
    def dataset_specific_transform(dataset_name):
        # Use higher degree for complex datasets
        degree = 10 if "complex" in dataset_name else 5

        return (
            PolynomialPrecondition(
                degree=degree,
                enabled=self.hparams.enable_preconditioning
            )
            + ... # remaining transforms
        )

    return {"complex_dataset": dataset_specific_transform}
```

## Performance Tips

1. **Start with degree 5**: Good balance of conditioning improvement and stability
2. **Test on subset first**: Validate on small dataset before full training
3. **Monitor gradients**: Use gradient clipping if instability occurs
4. **Compare systematically**: Always run baseline for comparison
5. **Dataset-specific tuning**: Some datasets may benefit from different degrees

## Citation

If you use this preconditioning implementation, please cite:

```bibtex
@article{marsden2025universal,
  title={Universal Sequence Preconditioning},
  author={Marsden, Annie and Hazan, Elad},
  journal={arXiv preprint arXiv:2502.06545},
  year={2025}
}
```

## Related Files

- Implementation: `/src/uni2ts/transform/precondition.py`
- Tests: `/test/transform/test_precondition.py`
- Pretrain Model: `/src/uni2ts/model/moirai/pretrain.py`
- Finetune Model: `/src/uni2ts/model/moirai/finetune.py`
- Config: `/cli/conf/pretrain/model/moirai_small.yaml`
- Precond Config: `/cli/conf/pretrain/model/moirai_small_precond.yaml`
- Experiment Plan: `/PRECONDITIONING_EXPERIMENT_PLAN.md`

## Support

For issues or questions:
1. Check the unit tests for examples
2. Review the experiment plan document
3. Verify configuration parameters
4. Test with `enable_preconditioning=false` to isolate issues
