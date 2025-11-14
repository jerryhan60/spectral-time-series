# Preconditioning in Evaluation: Quick Reference Guide

**Quick Lookup for Understanding the Issue and Implementation**

---

## THE CORE PROBLEM IN 30 SECONDS

**Training**: Data → PolynomialPrecondition → Standardize → Model  
**Evaluation**: Data → Standardize → Model (NO preconditioning!)  

**Result**: Unfair comparison. Model trained on preconditioned space evaluated on original space.

---

## FILE LOCATIONS AT A GLANCE

```
├── Preconditioning Implementation
│   └── /scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/precondition.py
│       ├── PolynomialPrecondition (line 24)   - Apply preconditioning
│       └── ReversePrecondition (line 250)     - Reverse preconditioning
│
├── Model Classes
│   ├── /scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast.py
│   │   └── MoiraiForecast (line 72)           - Standard model (NO precond support)
│   └── /scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast_precond.py
│       └── MoiraiForecastPrecond (line 30)    - Attempted fix (INCOMPLETE)
│
├── Evaluation Pipeline
│   ├── /scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/eval.py
│   │   └── main() - Entry point (line 30)
│   └── /scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/eval_util/evaluation.py
│       ├── evaluate_model() (line 228)        - Main evaluation function
│       ├── evaluate_forecasts() (line 176)    - Compare forecasts with labels
│       └── _get_data_batch() (line 57)        - Batch preparation
│
├── Training Pipeline (For Reference)
│   └── /scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/pretrain.py
│       └── MoiraiPretrain.__init__() (line 84) - Uses preconditioning correctly
│
├── Configuration
│   ├── /scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/default.yaml
│   │   └── Default evaluation configuration
│   ├── /scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/model/moirai_lightning_ckpt_precond.yaml
│   │   └── Config with preconditioning parameters (NOT USED!)
│   └── /scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/data/monash.yaml
│       └── Data loading configuration
│
└── Data Pipeline
    └── /scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/eval_util/data.py
        ├── get_gluonts_test_dataset() (line 59)
        ├── get_lsf_test_dataset() (line 190)
        └── get_custom_eval_dataset() (line 215)
```

---

## KEY CLASSES AND METHODS

### PolynomialPrecondition
**File**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/precondition.py` (line 24)

**What it does**: Applies polynomial preconditioning: ỹₜ = yₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ

**Key Methods**:
- `__call__(data_entry)` - Main entry point
- `_apply_convolution(sequence, coeffs)` - Vectorized convolution
- `_chebyshev_coefficients(n)` - Compute Chebyshev coefficients
- `_legendre_coefficients(n)` - Compute Legendre coefficients

**Parameters**:
- `polynomial_type`: "chebyshev" or "legendre"
- `degree`: 5-10 recommended
- `enabled`: Boolean
- `target_field`: "target" by default

**Output**: 
- Preconditioned target
- Stores: `precondition_coeffs`, `precondition_degree`, `precondition_type`, `precondition_enabled`

---

### ReversePrecondition
**File**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/precondition.py` (line 250)

**What it does**: Reverses polynomial preconditioning: yₜ = ỹₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ

**Key Methods**:
- `__call__(data_entry)` - Main entry point
- `_reverse_convolution(sequence, coeffs)` - Iterative reversal (NOT vectorized!)

**Parameters**:
- `target_field`: "target" by default
- `prediction_field`: "prediction" by default
- `enabled`: Boolean

**Requirements**:
- Input must have: `precondition_enabled`, `precondition_coeffs`

---

### MoiraiForecast
**File**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast.py` (line 72)

**Critical Methods**:

#### `create_predictor(batch_size, device)` (line 123)
- Creates PyTorchPredictor
- Builds transformation chain via `get_default_transform()`
- **ISSUE**: No preconditioning in transforms

```python
def create_predictor(self, batch_size: int, device: str = "auto") -> PyTorchPredictor:
    instance_splitter = TFTInstanceSplitter(...)
    return PyTorchPredictor(
        input_names=self.prediction_input_names,
        prediction_net=self,
        batch_size=batch_size,
        prediction_length=self.hparams.prediction_length,
        input_transform=self.get_default_transform() + instance_splitter,  # MISSING PRECONDITIONING!
        device=device,
    )
```

#### `get_default_transform()` (line 941)
- Builds input transformation pipeline
- Converts target to numpy, adds observed values indicator
- **ISSUE**: Doesn't include PolynomialPrecondition

```python
def get_default_transform(self) -> Transformation:
    transform = AsNumpyArray(field="target", ...)
    if self.hparams.target_dim == 1:
        transform += ExpandDimArray(field="target", axis=0)
    transform += AddObservedValuesIndicator(...)
    # ... MORE TRANSFORMS ...
    return transform
    # MISSING: PolynomialPrecondition!
```

---

### MoiraiForecastPrecond
**File**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast_precond.py` (line 30)

**Status**: INCOMPLETE - forward() bypasses preconditioning

**Key Methods**:

#### `_apply_preconditioning_numpy()` (line 97)
- Applies PolynomialPrecondition to batch of numpy data
- Handles [batch, time, dim] or [batch, time] shapes
- Returns: (preconditioned_data, coeffs)

#### `_reverse_preconditioning_numpy()` (line 147)
- Reverses PolynomialPrecondition on predictions
- Handles [batch, sample, time, dim] prediction shapes
- Requires context data for iterative reversal

#### `forward()` (line 204)
**CRITICAL ISSUE**: Currently doesn't apply preconditioning!

```python
def forward(self, past_target, past_observed_target, past_is_pad, ...):
    """
    IMPORTANT NOTE: The current implementation discovered that applying preconditioning
    at inference time creates shape mismatches...
    
    For now, we bypass preconditioning during inference...
    """
    # For now, just call parent forward without preconditioning
    return super().forward(...)
```

---

## EVALUATION PIPELINE

### Entry Point: `evaluate_model()` (line 228 in evaluation.py)

```python
def evaluate_model(
    model: Predictor,      # PyTorchPredictor instance
    test_data: TestData,   # GluonTS test data
    metrics,               # List of metric objects
    batch_size=100,
    ...
) -> pd.DataFrame:
    forecasts = model.predict(test_data.input)  # ← Gets predictions (potentially wrong space!)
    
    return evaluate_forecasts(
        forecasts=forecasts,
        test_data=test_data,
        metrics=metrics,
        ...
    )
```

**Problem**: No reversal of preconditioning on forecasts

---

### Data Flow

```
Raw Test Data
    ↓
PyTorchPredictor.predict()
    ├─ input_transform: AsNumpyArray + ExpandDimArray + AddObservedValuesIndicator + TFTInstanceSplitter
    │  (NO PolynomialPrecondition!)
    ├─ model.forward()
    └─ Returns predictions (in original space if trained, but model was trained in preconditioned space!)
    ↓
evaluate_forecasts()
    ├─ Compare predictions with test labels
    ├─ Compute metrics
    └─ (NO ReversePrecondition applied!)
    ↓
Metrics DataFrame
    └─ INCORRECT VALUES because model was trained in different space!
```

---

## TRAINING VS EVALUATION: THE MISMATCH

### Training (Correct)
```python
# In MoiraiPretrain
transform = PolynomialPrecondition(
    polynomial_type=self.hparams.precondition_type,
    degree=self.hparams.precondition_degree,
    target_field="target",
    enabled=self.hparams.enable_preconditioning,  # TRUE when enabled
)
# Then: ImputeTimeSeries, Patchify, etc.

# Data flows: Original → Preconditioned → Standardized → Model learns on preconditioned
```

### Evaluation (Incorrect)
```python
# In MoiraiForecast.get_default_transform()
transform = AsNumpyArray(field="target", ...)
transform += ExpandDimArray(field="target", axis=0)
transform += AddObservedValuesIndicator(...)
# NO PolynomialPrecondition!

# Data flows: Original → Standardized → Model predicts on original space (but was trained on preconditioned!)
```

---

## WHAT NEEDS TO CHANGE

### Change 1: Add Preconditioning to Input Transform
**Where**: `MoiraiForecast.get_default_transform()` (line 941)

**How**: Include PolynomialPrecondition BEFORE other transforms
```python
def get_default_transform(self) -> Transformation:
    # ADD THIS:
    if hasattr(self.hparams, 'enable_preconditioning') and self.hparams.enable_preconditioning:
        transform = PolynomialPrecondition(
            polynomial_type=self.hparams.precondition_type,
            degree=self.hparams.precondition_degree,
            enabled=True,
        )
    else:
        transform = Identity()
    
    # THEN rest of transforms...
    transform += AsNumpyArray(field="target", ...)
    # ... etc
    return transform
```

### Change 2: Add Reversal After Predictions
**Where**: `evaluate_model()` in evaluation.py (line 228)

**How**: Apply ReversePrecondition to forecasts
```python
def evaluate_model(model, test_data, metrics, ...):
    forecasts = model.predict(test_data.input)
    
    # ADD THIS:
    # If model was trained with preconditioning, reverse it
    # (Need to get coefficients from somewhere)
    
    return evaluate_forecasts(forecasts, test_data, metrics, ...)
```

### Change 3: Make Preconditioning Parameters Available
**Where**: Throughout the pipeline

**How**: 
- Store preconditioning parameters in model __init__ and hparams
- Pass from config to model
- Make available in create_predictor()

### Change 4: Fix MoiraiForecastPrecond
**Where**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast_precond.py`

**How**: Debug and fix the forward() method to properly handle preconditioning

---

## MATHEMATICAL QUICK REFERENCE

### Preconditioning (Forward)
```
ỹₜ = yₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  for t > n
ỹₜ = yₜ                     for t ≤ n

Where:
- yₜ = original value at time t
- ỹₜ = preconditioned value
- cᵢ = polynomial coefficient i
- n = polynomial degree
```

### Reversal (Inverse)
```
yₜ = ỹₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  for t > n
yₜ = ỹₜ                     for t ≤ n

Computed iteratively from left to right:
for t in range(n, len(sequence)):
    weighted_sum = ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  (using already-computed y values)
    yₜ = ỹₜ + weighted_sum
```

---

## DEBUGGING CHECKLIST

- [ ] Is preconditioning enabled during training? Check `enable_preconditioning` in training config
- [ ] Are preconditioning parameters in model checkpoint? Check model hparams
- [ ] Is `get_default_transform()` including PolynomialPrecondition? Check MoiraiForecast line 941
- [ ] Is `evaluate_model()` reversing preconditioning? Check evaluation.py line 228
- [ ] Are preconditioning coefficients available during reversal? Check data flow
- [ ] Do prediction shapes match reversal expectations? Check shapes at each step

---

## QUICK COMMANDS

**Find preconditioning in code**:
```bash
grep -rn "PolynomialPrecondition\|ReversePrecondition\|precondition" uni2ts/src/ --include="*.py"
```

**Check if model has preconditioning**:
```bash
grep -n "enable_preconditioning" uni2ts/src/uni2ts/model/moirai/*.py
```

**Find all evaluation entry points**:
```bash
grep -rn "def evaluate_model\|def create_predictor" uni2ts/src/ --include="*.py"
```

**Check training vs evaluation transforms**:
```bash
grep -A 10 "get_default_transform\|build.*transform" uni2ts/src/uni2ts/model/moirai/*.py
```

---

**Last Updated**: 2025-11-02
**For More Details**: See EVALUATION_PRECONDITIONING_ANALYSIS.md
