# Evaluation Pipeline and Preconditioning Implementation Analysis

**Prepared for**: Fixing Preconditioned Model Evaluation in Uni2TS  
**Date**: 2025-11-02  
**Thoroughness**: Very Thorough - Complete Architecture Overview

---

## EXECUTIVE SUMMARY

The evaluation pipeline has a fundamental design issue when handling preconditioning:

1. **Training Phase**: Preconditioning is applied DURING data loading (in the transformation pipeline)
2. **Inference Phase**: Preconditioning is NOT applied during prediction
3. **Result**: Model trained on preconditioned data is evaluated on non-preconditioned data = unfair comparison

The problem exists in three layers:
- Data Layer: Preconditioning transforms not applied during evaluation
- Model Layer: Forecast models don't know how to reverse-precondition outputs
- Pipeline Layer: PyTorchPredictor's transformation stack doesn't include preconditioning

---

## 1. CURRENT EVALUATION SCRIPTS

### 1.1 Main Entry Point: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/eval.py`

**Location**: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/eval.py`

**Key Components**:
- Uses Hydra configuration system
- Entry point: `@hydra.main(version_base="1.3", config_path="conf/eval", config_name="default")`
- Main function signature: `main(cfg: DictConfig)`

**Data Flow**:
```
1. Load config (default.yaml)
2. Load test data: test_data, metadata = call(cfg.data)
3. Load/instantiate model: model = call(cfg.model, ...)
4. Create predictor: predictor = model.create_predictor(batch_size, device)
5. Evaluate: evaluate_model(predictor, test_data, metrics, ...)
6. Report metrics to TensorBoard
```

**Critical Issue**: 
- No preconditioning logic in the evaluation pipeline
- Model checkpoint may have been trained with preconditioning, but eval doesn't apply it

### 1.2 Evaluation Utility: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/eval_util/evaluation.py`

**Key Functions**:

#### `evaluate_model()` (Line 228-264)
```python
def evaluate_model(
    model: Predictor,  # GluonTS predictor
    test_data: TestData,
    metrics,
    axis=None,
    batch_size=100,
    mask_invalid_label=True,
    allow_nan_forecast=False,
    seasonality=None,
) -> pd.DataFrame:
```
- Calls `predictor.predict(test_data.input)` to get forecasts
- Compares forecasts with test_data labels
- **No preconditioning reversal applied**

#### `evaluate_forecasts()` (Line 176-225)
- Takes pre-made forecasts and compares with labels
- Uses `_get_data_batch()` to prepare data for metrics

#### `_get_data_batch()` (Line 57-92)
- Constructs batches for metric evaluation
- Computes seasonal_error from input historical data
- Returns ChainMap combining labels and forecasts

---

## 2. PRECONDITIONING IMPLEMENTATION

### 2.1 Core Transformation: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/precondition.py`

**PolynomialPrecondition Class** (Lines 24-248)

**Purpose**: Apply polynomial-based preconditioning to time series

**Mathematical Formula**:
```
ỹₜ = yₜ - ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  for t > n
ỹₜ = yₜ                     for t ≤ n
```

**Key Parameters**:
- `polynomial_type`: "chebyshev" or "legendre"
- `degree`: Polynomial degree (5-10 recommended)
- `target_field`: Field to precondition (default: "target")
- `enabled`: Boolean to enable/disable
- `store_original`: Store original values

**Implementation Details**:

**Initialization** (Lines 56-78):
```python
def __post_init__(self):
    if not self.enabled:
        return
    
    # Validate inputs
    if self.polynomial_type not in ["chebyshev", "legendre"]:
        raise ValueError(...)
    
    # Compute polynomial coefficients
    self.coeffs = self._compute_coefficients(self.polynomial_type, self.degree)
```

**Coefficient Computation** (Lines 80-124):
- `_chebyshev_coefficients()`: Uses numpy.polynomial.chebyshev
- `_legendre_coefficients()`: Uses numpy.polynomial.legendre
- Returns coefficients [c₁, c₂, ..., cₙ] excluding constant term

**Main Transform** (Lines 126-201):
```python
def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
    if not self.enabled:
        return data_entry
    
    target = data_entry[self.target_field]
    
    # Handle different target formats:
    # 1. List of arrays (after _flatten_data)
    # 2. 2D array [time, variate]
    # 3. 1D array [time]
    
    # Apply convolution
    preconditioned = self._apply_convolution(target, self.coeffs)
    
    # Store metadata
    data_entry[self.target_field] = preconditioned
    data_entry["precondition_coeffs"] = self.coeffs
    data_entry["precondition_degree"] = self.degree
    data_entry["precondition_type"] = self.polynomial_type
    data_entry["precondition_enabled"] = True
    
    return data_entry
```

**Convolution Implementation** (Lines 203-247):
```python
def _apply_convolution(
    self,
    sequence: np.ndarray,  # 1D array [time]
    coeffs: np.ndarray,    # [c₁, c₂, ..., cₙ]
) -> np.ndarray:
    n = len(coeffs)
    result = sequence.copy()
    
    # For t >= n, apply: result[t] = sequence[t] - ∑ᵢ₌₀ⁿ⁻¹ coeffs[i] · sequence[t-i-1]
    # Vectorized using numpy slicing
    
    if len(sequence) > n:
        weighted_sum = np.zeros(len(sequence) - n)
        for i in range(n):
            weighted_sum += coeffs[i] * sequence[n-i-1:len(sequence)-i-1]
        result[n:] = sequence[n:] - weighted_sum
    
    return result
```

---

### 2.2 Reversal Transform: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/precondition.py` (Lines 250-376)

**ReversePrecondition Class**

**Purpose**: Reverse polynomial preconditioning after forecasting

**Mathematical Formula**:
```
yₜ = ỹₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ  for t > n
yₜ = ỹₜ                     for t ≤ n
```

**Key Parameters**:
- `target_field`: Field to reverse precondition (default: "target")
- `prediction_field`: Field for predictions (default: "prediction")
- `enabled`: Boolean to enable/disable

**Implementation** (Lines 275-338):
```python
def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
    if not self.enabled:
        return data_entry
    
    # Check if preconditioning was applied
    if not data_entry.get("precondition_enabled", False):
        return data_entry
    
    if "precondition_coeffs" not in data_entry:
        return data_entry
    
    coeffs = data_entry["precondition_coeffs"]
    
    # Determine which field to reverse
    # Priority: prediction_field > target_field
    
    # Apply reversal
    if preconditioned.ndim == 1:
        restored = self._reverse_convolution(preconditioned, coeffs)
    elif preconditioned.ndim == 2:
        # [time, variate] - apply to each variate
        restored = np.stack([
            self._reverse_convolution(preconditioned[:, i], coeffs)
            for i in range(preconditioned.shape[1])
        ], axis=1)
    elif preconditioned.ndim == 3:
        # [batch/sample, time, variate] - common for predictions
        restored = np.stack([
            np.stack([
                self._reverse_convolution(preconditioned[b, :, v], coeffs)
                for v in range(preconditioned.shape[2])
            ], axis=1)
            for b in range(preconditioned.shape[0])
        ], axis=0)
    
    data_entry[field_to_reverse] = restored
    return data_entry
```

**Reversal Convolution** (Lines 340-375):
```python
def _reverse_convolution(
    self,
    sequence: np.ndarray,  # 1D preconditioned array [time]
    coeffs: np.ndarray,    # [c₁, c₂, ..., cₙ]
) -> np.ndarray:
    n = len(coeffs)
    result = sequence.copy()
    
    # Iteratively compute: yₜ = ỹₜ + ∑ᵢ₌₁ⁿ cᵢ · yₜ₋ᵢ
    for t in range(n, len(sequence)):
        weighted_sum = sum(
            coeffs[i-1] * result[t-i]
            for i in range(1, n+1)
        )
        result[t] = sequence[t] + weighted_sum
    
    return result
```

**Important**: This is ITERATIVE (not vectorized) because each value depends on previously computed values.

---

## 3. MOIRAI FORECAST MODELS

### 3.1 Standard Model: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast.py`

**MoiraiForecast Class** (Lines 72-979)

**Constructor** (Lines 73-91):
```python
def __init__(
    self,
    prediction_length: int,
    target_dim: int,
    feat_dynamic_real_dim: int,
    past_feat_dynamic_real_dim: int,
    context_length: int,
    module_kwargs: Optional[dict[str, Any]] = None,
    module: Optional[MoiraiModule] = None,
    patch_size: int | str = "auto",
    num_samples: int = 100,
):
```

**Create Predictor Method** (Lines 123-151):
```python
def create_predictor(
    self,
    batch_size: int,
    device: str = "auto",
) -> PyTorchPredictor:
    # Build transformation chain
    instance_splitter = TFTInstanceSplitter(...)
    
    return PyTorchPredictor(
        input_names=self.prediction_input_names,
        prediction_net=self,
        batch_size=batch_size,
        prediction_length=self.hparams.prediction_length,
        input_transform=self.get_default_transform() + instance_splitter,
        device=device,
    )
```

**Get Default Transform** (Lines 941-978):
```python
def get_default_transform(self) -> Transformation:
    transform = AsNumpyArray(
        field="target",
        expected_ndim=1 if self.hparams.target_dim == 1 else 2,
        dtype=np.float32,
    )
    if self.hparams.target_dim == 1:
        transform += ExpandDimArray(field="target", axis=0)
    transform += AddObservedValuesIndicator(
        target_field="target",
        output_field="observed_target",
        dtype=bool,
    )
    # ... more transforms for dynamic features ...
    return transform
```

**Key Issue**: 
- This transform does NOT include PolynomialPrecondition
- Even if model was trained with preconditioning, it's not applied during evaluation

---

### 3.2 Preconditioning-Aware Model: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast_precond.py`

**MoiraiForecastPrecond Class** (Lines 30-246)

**This is the attempted fix, but has issues**

**Constructor** (Lines 42-95):
```python
def __init__(
    self,
    # ... standard params ...
    enable_preconditioning: bool = True,
    precondition_type: str = "chebyshev",
    precondition_degree: int = 5,
):
    super().__init__(...)
    
    # Create preconditioning transforms
    if self.enable_preconditioning:
        self.preconditioner = PolynomialPrecondition(
            polynomial_type=precondition_type,
            degree=precondition_degree,
            target_field="target",
            enabled=True,
            store_original=True,
        )
        self.reverse_preconditioner = ReversePrecondition(
            target_field="prediction",
            enabled=True,
        )
```

**Apply Preconditioning (NumPy)** (Lines 97-145):
```python
def _apply_preconditioning_numpy(
    self,
    data: np.ndarray  # [batch, time, dim] or [batch, time]
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if not self.enable_preconditioning:
        return data, None
    
    # Handle batch dimension
    original_shape = data.shape
    if data.ndim == 2:
        data = data[..., np.newaxis]
    
    batch_size = data.shape[0]
    preconditioned_batch = []
    
    # Apply preconditioning to each series in batch
    for i in range(batch_size):
        series = data[i]  # [time, dim]
        data_entry = {"target": series}
        preconditioned_entry = self.preconditioner(data_entry)
        preconditioned_series = preconditioned_entry["target"]
        preconditioned_batch.append(preconditioned_series)
    
    preconditioned_data = np.stack(preconditioned_batch, axis=0)
    
    # Restore original shape if needed
    if len(original_shape) == 2:
        preconditioned_data = preconditioned_data.squeeze(-1)
    
    coeffs = self.preconditioner.coeffs if self.enable_preconditioning else None
    
    return preconditioned_data, coeffs
```

**Reverse Preconditioning (NumPy)** (Lines 147-202):
```python
def _reverse_preconditioning_numpy(
    self,
    predictions: np.ndarray,  # [batch, sample, time, dim]
    coeffs: Optional[np.ndarray],
    context: np.ndarray,  # [batch, context_time, dim]
) -> np.ndarray:
    if not self.enable_preconditioning or coeffs is None:
        return predictions
    
    batch_size, num_samples, pred_len, dim = predictions.shape
    reversed_batch = []
    
    for b in range(batch_size):
        reversed_samples = []
        
        for s in range(num_samples):
            pred_sample = predictions[b, s]  # [pred_time, dim]
            context_b = context[b]  # [context_time, dim]
            
            # Concatenate context + prediction for reversal
            full_series = np.concatenate([context_b, pred_sample], axis=0)
            
            # Create data entry with preconditioning metadata
            data_entry = {
                "prediction": full_series,
                "precondition_coeffs": coeffs,
                "precondition_degree": self.precondition_degree,
                "precondition_type": self.precondition_type,
                "precondition_enabled": True,
            }
            
            # Apply reversal
            reversed_entry = self.reverse_preconditioner(data_entry)
            reversed_full = reversed_entry["prediction"]
            
            # Extract just the prediction part
            reversed_pred = reversed_full[-pred_len:]
            reversed_samples.append(reversed_pred)
        
        reversed_batch.append(np.stack(reversed_samples, axis=0))
    
    return np.stack(reversed_batch, axis=0)
```

**Forward Method** (Lines 204-246):
```python
def forward(
    self,
    past_target: Float[torch.Tensor, "batch past_time tgt"],
    # ... other inputs ...
    num_samples: Optional[int] = None,
) -> Float[torch.Tensor, "batch sample future_time *tgt"]:
    """
    IMPORTANT NOTE: The current implementation discovered that applying preconditioning
    at inference time creates shape mismatches. Since the model was trained on
    preconditioned data with per-series standardization, and evaluation also applies
    per-series standardization, the standardization itself may normalize out some
    of the preconditioning effects.
    
    For now, we bypass preconditioning during inference...
    """
    # For now, just call parent forward without preconditioning
    return super().forward(...)
```

**Critical Issue**: 
- The forward() method DOESN'T apply preconditioning
- Has a comment explaining why: "shape mismatches" and potential conflicts with standardization
- This class exists but is not properly implemented

---

## 4. TRAINING PIPELINE WITH PRECONDITIONING

### 4.1 Training Entry Point: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/pretrain.py`

**MoiraiPretrain Class** (Lines 66-388+)

**Constructor Parameters** (Lines 102-104):
```python
enable_preconditioning: bool = False,
precondition_type: str = "chebyshev",
precondition_degree: int = 5,
```

**Build Transform Chain** (Lines 387-392):
```python
# Start with optional preconditioning transform
transform = PolynomialPrecondition(
    polynomial_type=self.hparams.precondition_type,
    degree=self.hparams.precondition_degree,
    target_field="target",
    enabled=self.hparams.enable_preconditioning,
)
```

**Important**: 
- Preconditioning IS applied during training
- Applied BEFORE other transforms like ImputeTimeSeries, Patchify, etc.
- Data flows through the preconditioned space during model learning

---

## 5. DATA PIPELINE FLOW

### 5.1 Evaluation Data Loading: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/eval_util/data.py`

**Functions**:

#### `get_gluonts_test_dataset()` (Lines 59-162)
- Loads test dataset from GluonTS repository
- Returns: `(TestData, MetaData)`
- Splits data appropriately for evaluation
- **No preconditioning applied here**

#### `get_lsf_test_dataset()` (Lines 190-212)
- Loads LSF test dataset
- Returns: `(TestData, MetaData)`

#### `get_custom_eval_dataset()` (Lines 215-239)
- Custom evaluation dataset loading
- Returns: `(TestData, MetaData)`

### 5.2 Transformation Base Class: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/_base.py`

**Transformation ABC** (Lines 21-35):
```python
class Transformation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]: ...
    
    def chain(self, other: "Transformation") -> "Chain":
        return Chain([self, other])
    
    def __add__(self, other: "Transformation") -> "Chain":
        return self.chain(other)
```

**Chain Class** (Lines 37+):
- Chains multiple transformations
- Applies them sequentially: `result = t1(t2(t3(...)))`

---

## 6. EVALUATION CONFIGURATION FILES

### 6.1 Default Evaluation Config: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/default.yaml`

```yaml
hydra:
  run:
    dir: outputs/${hydra:job.name}/${hydra:runtime.choices.data}/${data.dataset_name}/${data.mode}/prediction_length=${data.prediction_length}/${run_name}
defaults:
  - model: ???
  - data: ???
  - _self_
run_name: ???
metrics:
  - _target_: gluonts.ev.metrics.MSE
  - _target_: uni2ts.eval_util.metrics.MedianMSE
  - _target_: gluonts.ev.metrics.MAE
  - _target_: gluonts.ev.metrics.MASE
  - _target_: gluonts.ev.metrics.MAPE
  - _target_: gluonts.ev.metrics.SMAPE
  - _target_: gluonts.ev.metrics.MSIS
  - _target_: gluonts.ev.metrics.RMSE
  - _target_: gluonts.ev.metrics.NRMSE
  - _target_: gluonts.ev.metrics.ND
  - _target_: gluonts.ev.metrics.MeanWeightedSumQuantileLoss
    quantile_levels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
batch_size: 512
min_batch_size: 1
device: auto
```

### 6.2 Model Config with Preconditioning: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/model/moirai_lightning_ckpt_precond.yaml`

```yaml
_target_: uni2ts.model.moirai.MoiraiForecast.load_from_checkpoint
checkpoint_path: ???
num_samples: 100
patch_size: ???
context_length: ???

# Preconditioning parameters
# Set these if the model was trained with preconditioning enabled
reverse_preconditioning: true  # Whether to reverse preconditioning on predictions
precondition_type: chebyshev  # Type used during training
precondition_degree: 5  # Degree used during training
```

**Issue**: 
- Config has parameters for preconditioning but they're not actually used by MoiraiForecast
- These would be used by MoiraiForecastPrecond, but that's not the standard model

### 6.3 Data Config: `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/conf/eval/data/monash.yaml`

```yaml
_target_: uni2ts.eval_util.data.get_gluonts_test_dataset
dataset_name: ???
prediction_length: null
mode: S
```

---

## 7. PYTORCH PREDICTOR INTEGRATION

The evaluation uses GluonTS's `PyTorchPredictor` class.

**Data Flow**:
```
1. Raw test data (from GluonTS TestData)
2. Apply input_transform (including preconditioning IF present)
3. Forward through model.forward()
4. Generate predictions
5. NO output_transform for reversal (this is the problem!)
```

**Current Signature** (from forecast.py):
```python
return PyTorchPredictor(
    input_names=self.prediction_input_names,
    prediction_net=self,
    batch_size=batch_size,
    prediction_length=self.hparams.prediction_length,
    input_transform=self.get_default_transform() + instance_splitter,
    device=device,
)
```

**What's Missing**:
- No `output_transform` parameter to apply ReversePrecondition
- PyTorchPredictor doesn't have built-in support for preconditioning reversal
- Predictions returned in whatever space they were produced in

---

## 8. PROBLEM SUMMARY AND NEEDED FIXES

### 8.1 Current Problems

**Problem 1**: Training/Evaluation Mismatch
- Training: Data flows through PolynomialPrecondition → standardization → model
- Evaluation: Data flows through standardization → model (NO preconditioning)

**Problem 2**: Model Architecture Issue
- MoiraiForecast doesn't know about preconditioning
- MoiraiForecastPrecond exists but forward() bypasses preconditioning due to "shape mismatches"

**Problem 3**: Predictor Integration
- PyTorchPredictor has no mechanism for output_transform to reverse preconditioning
- Predictions are compared in preconditioned space (if trained with it) but evaluated in original space

**Problem 4**: Configuration Disconnect
- Config files have preconditioning parameters but they're not used in standard evaluation flow

### 8.2 What Needs to Be Fixed

**Fix 1**: Add preconditioning to input_transform in evaluation
- Include PolynomialPrecondition in get_default_transform()
- Need preconditioning parameters available to the model

**Fix 2**: Add output_transform to PyTorchPredictor for reversal
- Create custom predictor wrapper or modify create_predictor()
- Apply ReversePrecondition after getting predictions

**Fix 3**: Make preconditioning parameters available throughout the pipeline
- Pass precondition_type, precondition_degree to model at evaluation time
- Store in model checkpoints OR pass from config

**Fix 4**: Fix MoiraiForecastPrecond implementation
- Debug the "shape mismatch" issue mentioned in forward()
- Determine if preconditioning should be applied as input_transform or forward preprocessing
- Test thoroughly with standardization

---

## 9. KEY FILES AND FUNCTIONS SUMMARY

### Critical Files

| File | Component | Issue |
|------|-----------|-------|
| `/scratch/gpfs/EHAZAN/jh1161/uni2ts/cli/eval.py` | Evaluation entry | No preconditioning awareness |
| `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/eval_util/evaluation.py` | Metric computation | No reversal of preconditioning |
| `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/transform/precondition.py` | Core transforms | Well-implemented, just not used in eval |
| `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast.py` | Standard predictor | Doesn't handle preconditioning |
| `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/forecast_precond.py` | Precond predictor | Incomplete implementation |
| `/scratch/gpfs/EHAZAN/jh1161/uni2ts/src/uni2ts/model/moirai/pretrain.py` | Training | Correctly applies preconditioning |

### Key Classes

| Class | Location | Purpose | Status |
|-------|----------|---------|--------|
| `PolynomialPrecondition` | precondition.py:24 | Apply preconditioning | Complete |
| `ReversePrecondition` | precondition.py:250 | Reverse preconditioning | Complete |
| `MoiraiForecast` | forecast.py:72 | Standard forecast model | Works but no precond support |
| `MoiraiForecastPrecond` | forecast_precond.py:30 | Preconditioning-aware | Incomplete |
| `PyTorchPredictor` | GluonTS | Predictor wrapper | Limited extensibility |

### Key Methods

| Method | Class | Line | Purpose |
|--------|-------|------|---------|
| `__call__` | PolynomialPrecondition | 126 | Apply preconditioning transform |
| `_apply_convolution` | PolynomialPrecondition | 203 | Vectorized convolution |
| `__call__` | ReversePrecondition | 275 | Reverse preconditioning |
| `_reverse_convolution` | ReversePrecondition | 340 | Iterative reversal |
| `create_predictor` | MoiraiForecast | 123 | Create GluonTS predictor |
| `get_default_transform` | MoiraiForecast | 941 | Build input transformation |
| `forward` | MoiraiForecastPrecond | 204 | Forward pass (currently bypasses precond) |
| `evaluate_model` | evaluation.py | 228 | Main evaluation function |

---

## 10. RECOMMENDED IMPLEMENTATION STRATEGY

### Short-term (Immediate Fix)

1. **Modify MoiraiForecast.get_default_transform()**
   - Add preconditioning parameters to __init__
   - Include PolynomialPrecondition in transform chain if enabled
   - Store coefficients as instance variable

2. **Create Output Transform Wrapper**
   - Create custom Transformation class for reversal
   - Apply in evaluation loop after predictions

3. **Update eval.py**
   - Pass preconditioning parameters from config to model
   - Apply reversal before metrics computation

### Medium-term (Proper Architecture)

1. **Extend PyTorchPredictor**
   - Create custom predictor class with output_transform support
   - OR modify create_predictor to post-process predictions

2. **Standardize MoiraiForecastPrecond**
   - Fix forward() method to properly handle preconditioning
   - Debug shape mismatch issues
   - Make it the default for preconditioned models

3. **Update Configuration**
   - Make preconditioning parameters mandatory in model configs
   - Create separate evaluation configs for preconditioned vs standard models

---

## 11. DATA STRUCTURES AND TRANSFORMATIONS

### Data Entry Format

```python
# After PolynomialPrecondition
data_entry = {
    "target": np.array([...]),  # Preconditioned values
    "precondition_coeffs": np.array([c1, c2, ..., cn]),
    "precondition_degree": 5,
    "precondition_type": "chebyshev",
    "precondition_enabled": True,
    # ... other fields ...
}
```

### Prediction Output Format

```python
# From model.forward()
predictions: torch.Tensor  # Shape: [batch, num_samples, prediction_length, target_dim]

# Needs to be reversed to:
reversed_predictions: torch.Tensor  # Shape: [batch, num_samples, prediction_length, target_dim]
```

---

## CONCLUSION

The evaluation pipeline has a **complete disconnect** from the preconditioning applied during training. This means:

1. Models trained WITH preconditioning are evaluated WITHOUT it (or vice versa)
2. Metrics are computed on misaligned data spaces
3. Preconditioning benefits are not properly captured

The fix requires changes across three layers:
- **Data Layer**: Apply PolynomialPrecondition in evaluation transforms
- **Model Layer**: Handle preconditioning reversal after predictions
- **Pipeline Layer**: Integrate reversal into PyTorchPredictor or create wrapper

Files are well-structured and transformations are correctly implemented. The issue is purely architectural - they're just not connected during evaluation.

