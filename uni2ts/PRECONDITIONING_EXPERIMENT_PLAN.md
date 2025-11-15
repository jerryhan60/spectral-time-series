# Preconditioning Experiment Plan for Uni2TS

## Executive Summary

This document outlines a two-phase experiment to evaluate the impact of **Universal Sequence Preconditioning** (using Chebyshev/Legendre polynomials) on time series forecasting performance in the Uni2TS (Moirai) framework.

**Hypothesis**: Applying polynomial preconditioning to training data and reversing it during inference will improve forecasting accuracy by better conditioning the hidden transition matrices in the time series dynamics.

---

## Background: Universal Sequence Preconditioning

### Paper Reference
- **Title**: Universal Sequence Preconditioning
- **Authors**: Annie Marsden, Elad Hazan (Google DeepMind, Princeton University)
- **arXiv**: 2502.06545 (February 2025)
- **URL**: https://arxiv.org/abs/2502.06545

### Key Concepts

**What is Preconditioning?**
Preconditioning involves convolving input sequences with the coefficients of Chebyshev or Legendre polynomials. Mathematically, this translates to applying a polynomial transformation to the unknown transition matrix in the hidden space.

**Mathematical Formulation:**
```
ỹₜ = -∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ + yₜ
```

Where:
- `yₜ` is the original sequence at time t
- `ỹₜ` is the preconditioned sequence
- `cᵢ` are the Chebyshev or Legendre polynomial coefficients
- `n` is the polynomial degree (hyperparameter)

**Why It Works:**
- Improves the condition number of the hidden transition matrix
- Provides sublinear regret bounds that are hidden dimension independent
- Generalizes across different learning algorithms (including neural networks)
- Effective even with asymmetric transition matrices

**Key Benefits:**
- First method to achieve dimension-independent regret bounds
- Works for both linear and non-linear dynamical systems
- Simple to implement (just a convolution operation)
- No additional learnable parameters required

---

## Experimental Design

### Phase 1: Baseline Evaluation

**Objective**: Establish baseline performance metrics for the smallest Moirai model.

#### Configuration
- **Model**: Moirai Small (6 layers, 384 d_model)
- **Dataset**: LOTSA v1 (unweighted, all datasets)
- **Training Settings**:
  - Patch sizes: [8, 16, 32, 64, 128]
  - Max sequence length: 512
  - Learning rate: 1e-3
  - Weight decay: 1e-1
  - Dropout: 0.0
  - Attention dropout: 0.0
  - Min patches: 2
  - Mask ratio: 0.15-0.5
  - Max dim: 128

#### Steps
1. Verify LOTSA v1 data is fully downloaded
2. Run pretraining using existing configuration:
   ```bash
   python -m cli.train \
     -cp conf/pretrain \
     run_name=baseline_small \
     model=moirai_small \
     data=lotsa_v1_unweighted
   ```
3. Run evaluation on standard benchmarks:
   - Monash datasets
   - LSF (Long Sequence Forecasting) benchmarks
   - Custom evaluation sets

#### Expected Outputs
- Trained model checkpoint
- Training metrics (loss curves, etc.)
- Evaluation metrics:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MASE (Mean Absolute Scaled Error)
  - CRPS (Continuous Ranked Probability Score)
  - sMAPE (symmetric Mean Absolute Percentage Error)

#### Timeline
- Training: ~3-7 days (depending on hardware)
- Evaluation: ~1 day

---

### Phase 2: Preconditioning Evaluation

**Objective**: Evaluate if polynomial preconditioning improves forecasting accuracy.

#### Implementation Strategy

##### 1. Create Preconditioning Transform Module

**Location**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/precondition.py`

**Implementation**:
```python
import numpy as np
import torch
from dataclasses import dataclass
from uni2ts.transform._base import Transformation

@dataclass
class PolynomialPrecondition(Transformation):
    """
    Apply Chebyshev or Legendre polynomial preconditioning to time series.

    Args:
        polynomial_type: "chebyshev" or "legendre"
        degree: Polynomial degree (recommended: 5-10)
        target_field: Field name to precondition (default: "target")
    """
    polynomial_type: str = "chebyshev"
    degree: int = 5
    target_field: str = "target"

    def __post_init__(self):
        # Compute polynomial coefficients
        if self.polynomial_type == "chebyshev":
            self.coeffs = self._chebyshev_coefficients(self.degree)
        elif self.polynomial_type == "legendre":
            self.coeffs = self._legendre_coefficients(self.degree)
        else:
            raise ValueError(f"Unknown polynomial type: {self.polynomial_type}")

    def _chebyshev_coefficients(self, n: int) -> np.ndarray:
        """Compute Chebyshev polynomial coefficients of degree n"""
        # Use numpy.polynomial.chebyshev
        from numpy.polynomial import chebyshev
        coeffs = chebyshev.chebcoef(n)
        return coeffs

    def _legendre_coefficients(self, n: int) -> np.ndarray:
        """Compute Legendre polynomial coefficients of degree n"""
        # Use numpy.polynomial.legendre
        from numpy.polynomial import legendre
        coeffs = legendre.legcoef(n)
        return coeffs

    def __call__(self, data_entry: dict) -> dict:
        """Apply preconditioning to target field"""
        target = data_entry[self.target_field]

        # Apply convolution with polynomial coefficients
        preconditioned = self._apply_convolution(target, self.coeffs)

        # Store original for potential reversal
        data_entry[f"{self.target_field}_original"] = target
        data_entry[self.target_field] = preconditioned
        data_entry["precondition_coeffs"] = self.coeffs
        data_entry["precondition_degree"] = self.degree

        return data_entry

    def _apply_convolution(self, sequence, coeffs):
        """Apply polynomial convolution to sequence"""
        # Implement: ỹₜ = -∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ + yₜ
        n = len(coeffs)
        result = sequence.copy()

        for t in range(n, len(sequence)):
            weighted_sum = sum(coeffs[i] * sequence[t-i] for i in range(1, n+1))
            result[t] = sequence[t] - weighted_sum

        return result


@dataclass
class ReversePrecondition(Transformation):
    """
    Reverse polynomial preconditioning after forecasting.
    """
    target_field: str = "target"

    def __call__(self, data_entry: dict) -> dict:
        """Reverse preconditioning on predictions"""
        if "precondition_coeffs" not in data_entry:
            return data_entry  # No preconditioning was applied

        predictions = data_entry[self.target_field]
        coeffs = data_entry["precondition_coeffs"]

        # Reverse the transformation
        restored = self._reverse_convolution(predictions, coeffs)
        data_entry[self.target_field] = restored

        return data_entry

    def _reverse_convolution(self, sequence, coeffs):
        """Reverse polynomial convolution"""
        # Solve: yₜ = ỹₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ
        n = len(coeffs)
        result = sequence.copy()

        for t in range(n, len(sequence)):
            weighted_sum = sum(coeffs[i] * result[t-i] for i in range(1, n+1))
            result[t] = sequence[t] + weighted_sum

        return result
```

##### 2. Integrate into Data Pipeline

**Modify**: Training data configuration to include preconditioning transform

**New file**: `cli/conf/pretrain/data/lotsa_v1_preconditioned.yaml`
```yaml
_target_: uni2ts.data.builder.ConcatDatasetBuilder
_args_:
  - _target_: uni2ts.data.builder.lotsa_v1.Buildings900KDatasetBuilder
    datasets: ${cls_getattr:${._target_},dataset_list}
    transform:
      _target_: uni2ts.transform.precondition.PolynomialPrecondition
      polynomial_type: chebyshev  # or legendre
      degree: 5
      target_field: target
  # ... (repeat for all dataset builders)
```

##### 3. Modify Evaluation Pipeline

**Update**: Forecast generation to reverse preconditioning

**Location**: Modify `uni2ts/model/moirai/forecast.py` or create wrapper

**Approach**: Add post-processing step that applies `ReversePrecondition` to model outputs

#### Hyperparameter Grid

Test multiple configurations:

| Experiment | Polynomial Type | Degree | Notes |
|------------|----------------|---------|-------|
| precond_cheb_2 | Chebyshev | 2 | Low degree baseline |
| precond_cheb_5 | Chebyshev | 5 | Recommended |
| precond_cheb_10 | Chebyshev | 10 | Higher degree |
| precond_leg_5 | Legendre | 5 | Compare polynomials |
| precond_leg_10 | Legendre | 10 | Higher degree Legendre |

**Note**: Paper indicates degree > 10 may cause performance degradation.

#### Training Commands

```bash
# Chebyshev degree 5 (primary experiment)
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_cheb_5 \
  model=moirai_small \
  data=lotsa_v1_preconditioned \
  data.transform.polynomial_type=chebyshev \
  data.transform.degree=5

# Legendre degree 5 (comparison)
python -m cli.train \
  -cp conf/pretrain \
  run_name=precond_leg_5 \
  model=moirai_small \
  data=lotsa_v1_preconditioned \
  data.transform.polynomial_type=legendre \
  data.transform.degree=5
```

#### Evaluation Commands

```bash
# Evaluate with proper reversal
python -m cli.eval \
  run_name=eval_precond_cheb_5 \
  model=moirai_small \
  model.checkpoint_path=checkpoints/precond_cheb_5/last.ckpt \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96 \
  reverse_precondition=true \
  precondition_coeffs_path=configs/cheb_5_coeffs.npy
```

#### Expected Outcomes

**Success Metrics**:
- Improved MSE/MAE/MASE compared to baseline
- Reduced CRPS (better probabilistic forecasts)
- Better performance on datasets with complex dynamics

**Potential Observations**:
- Chebyshev and Legendre should yield similar results (per paper)
- Degree 5-10 likely optimal
- Greatest improvements on datasets with:
  - High temporal dependencies
  - Asymmetric dynamics
  - Complex transition patterns

---

## Implementation Roadmap

### Week 1-2: Baseline Training
- [ ] Verify LOTSA v1 data download complete
- [ ] Launch baseline training
- [ ] Monitor training progress
- [ ] Run baseline evaluations
- [ ] Document baseline metrics

### Week 3: Preconditioning Implementation
- [ ] Implement `PolynomialPrecondition` transform class
- [ ] Implement `ReversePrecondition` transform class
- [ ] Add unit tests for preconditioning/reversal
- [ ] Validate on small synthetic dataset
- [ ] Create preconditioned data configurations

### Week 4: Integration & Testing
- [ ] Integrate transforms into data pipeline
- [ ] Modify forecast pipeline for reversal
- [ ] Test end-to-end pipeline
- [ ] Create experiment configurations
- [ ] Prepare logging/monitoring

### Week 5-6: Preconditioning Experiments
- [ ] Launch Chebyshev degree 2,5,10 experiments
- [ ] Launch Legendre degree 5,10 experiments
- [ ] Monitor training
- [ ] Run evaluations with proper reversal
- [ ] Compare metrics against baseline

### Week 7: Analysis & Documentation
- [ ] Statistical significance testing
- [ ] Analyze per-dataset improvements
- [ ] Identify which dataset types benefit most
- [ ] Create visualization of results
- [ ] Write final report

---

## Technical Considerations

### 1. Numerical Stability
- Polynomial coefficients can grow rapidly (Chebyshev: ~2^0.3n)
- May need gradient clipping or careful scaling
- Monitor for NaN/Inf during training

### 2. Sequence Length Requirements
- Preconditioning requires history of length ≥ degree
- Handle edge cases for short sequences
- Consider padding strategies

### 3. Computational Cost
- Convolution adds minimal overhead (O(n·d) per sequence)
- Negligible compared to transformer computation
- May increase data loading time slightly

### 4. Memory Considerations
- Store original sequences if needed for reversal
- Coefficient storage is minimal (n floats)
- Consider caching preconditioned data

### 5. Distribution Shift
- Preconditioning changes data distribution
- May affect learned scaling parameters
- Consider recomputing scaling statistics on preconditioned data

---

## Evaluation Metrics

### Primary Metrics
1. **MSE** (Mean Squared Error) - standard forecasting metric
2. **MAE** (Mean Absolute Error) - robust to outliers
3. **MASE** (Mean Absolute Scaled Error) - scale-independent

### Secondary Metrics
1. **CRPS** - probabilistic forecast quality
2. **sMAPE** - percentage error
3. **Quantile Loss** - distribution calibration

### Analysis Dimensions
- **By Dataset**: Which datasets benefit most?
- **By Horizon**: Short vs long-term forecasting
- **By Frequency**: Hourly, daily, monthly, etc.
- **By Complexity**: Simple vs complex dynamics

---

## Expected Challenges & Mitigations

### Challenge 1: Reversal Complexity
**Issue**: Correctly reversing preconditioning on forecasts may be non-trivial
**Mitigation**:
- Implement careful tests with synthetic data
- Verify: precondition → reverse = identity
- Consider iterative reversal methods if needed

### Challenge 2: Hyperparameter Sensitivity
**Issue**: Optimal degree may vary by dataset
**Mitigation**:
- Test multiple degrees (2, 5, 10)
- Consider adaptive degree selection
- Analyze correlation with dataset properties

### Challenge 3: Training Instability
**Issue**: Preconditioning may affect gradient flow
**Mitigation**:
- Use gradient clipping
- Monitor gradient norms
- Consider warmup with/without preconditioning

### Challenge 4: Computational Resources
**Issue**: Multiple experiments require significant compute
**Mitigation**:
- Prioritize Chebyshev degree 5 as primary experiment
- Run baseline and one preconditioned experiment in parallel
- Use smaller subset of LOTSA for initial validation

---

## Success Criteria

### Minimum Success
- Successfully implement preconditioning transforms
- Complete baseline + 1 preconditioned experiment
- Demonstrate numerical correctness of reversal
- Document implementation for future work

### Expected Success
- Complete baseline + 2-3 preconditioned experiments
- Show statistically significant improvement on ≥30% of datasets
- Identify dataset characteristics that benefit from preconditioning
- Publish findings internally

### Exceptional Success
- Show consistent improvements across majority of datasets
- Achieve state-of-the-art results on key benchmarks
- Derive theoretical explanation for improvements in this context
- Prepare paper/blog post for publication

---

## Resource Requirements

### Computational
- GPU: 8× A100 or V100 (for parallel training)
- Storage: ~5TB for LOTSA data + checkpoints
- Time: ~100-200 GPU-days total

### Personnel
- 1× ML Engineer/Researcher (primary implementer)
- 1× Advisor (review & guidance)
- Access to paper authors for clarifications (optional)

### Software
- Python 3.12+
- PyTorch 2.0+
- NumPy (for polynomial coefficients)
- Uni2TS codebase
- Standard ML tools (tensorboard, wandb, etc.)

---

## References

1. Marsden, A., & Hazan, E. (2025). Universal Sequence Preconditioning. arXiv:2502.06545.
2. Woo, G., et al. (2024). Unified Training of Universal Time Series Forecasting Transformers. ICML 2024.
3. Salesforce AI Research. (2024). Uni2TS: Universal Time Series Transformers. https://github.com/SalesforceAIResearch/uni2ts

---

## Appendix A: Mathematical Details

### Chebyshev Polynomials
The Chebyshev polynomials of the first kind are defined recursively:
```
T₀(x) = 1
T₁(x) = x
Tₙ₊₁(x) = 2x·Tₙ(x) - Tₙ₋₁(x)
```

### Legendre Polynomials
The Legendre polynomials are defined by Rodrigues' formula:
```
Pₙ(x) = 1/(2ⁿn!) · dⁿ/dxⁿ[(x² - 1)ⁿ]
```

### Convolution Operation
For a sequence y = [y₁, y₂, ..., yₜ] and coefficients c = [c₁, c₂, ..., cₙ]:
```
ỹₜ = yₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  for t > n
ỹₜ = yₜ                     for t ≤ n
```

### Reversal Operation
To recover original sequence from preconditioned:
```
yₜ = ỹₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  for t > n
yₜ = ỹₜ                     for t ≤ n
```

This can be computed iteratively from left to right.

---

## Appendix B: Code Snippets

### Computing Chebyshev Coefficients
```python
import numpy as np
from numpy.polynomial import chebyshev

def get_chebyshev_coefficients(degree: int) -> np.ndarray:
    """Get coefficients for Chebyshev polynomial of given degree."""
    # Generate Chebyshev polynomial
    cheb = chebyshev.Chebyshev.basis(degree)
    # Get coefficients in standard polynomial basis
    coeffs = cheb.coef
    return coeffs
```

### Computing Legendre Coefficients
```python
import numpy as np
from numpy.polynomial import legendre

def get_legendre_coefficients(degree: int) -> np.ndarray:
    """Get coefficients for Legendre polynomial of given degree."""
    # Generate Legendre polynomial
    leg = legendre.Legendre.basis(degree)
    # Get coefficients in standard polynomial basis
    coeffs = leg.coef
    return coeffs
```

### Applying Convolution (Vectorized)
```python
import numpy as np
from scipy.signal import convolve

def apply_polynomial_preconditioning(
    sequence: np.ndarray,
    coefficients: np.ndarray
) -> np.ndarray:
    """Apply polynomial preconditioning via convolution."""
    n = len(coefficients)
    result = sequence.copy()

    # Pad for boundary conditions
    padded = np.pad(sequence, (n, 0), mode='edge')

    # Apply convolution
    for t in range(n, len(sequence)):
        weighted_sum = np.dot(coefficients[::-1], padded[t:t+n])
        result[t] = sequence[t] - weighted_sum

    return result
```

---

## Appendix C: Experiment Tracking Template

```yaml
experiment_name: precond_cheb_5
date: 2025-XX-XX
status: [planned/running/completed]

configuration:
  model: moirai_small
  polynomial_type: chebyshev
  degree: 5
  dataset: lotsa_v1_unweighted

baseline_metrics:
  mse: X.XXX
  mae: X.XXX
  mase: X.XXX
  crps: X.XXX

preconditioned_metrics:
  mse: X.XXX
  mae: X.XXX
  mase: X.XXX
  crps: X.XXX

improvements:
  mse_improvement: X.X%
  mae_improvement: X.X%
  mase_improvement: X.X%
  crps_improvement: X.X%

notes: |
  - Observations during training
  - Any issues encountered
  - Interesting findings
```

---

**Document Version**: 1.0
**Date**: 2025-10-19
**Author**: Experimental Planning System
**Status**: Ready for Implementation
