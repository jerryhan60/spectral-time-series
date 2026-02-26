# Inference-Time Preconditioning Ensemble — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evaluate whether running the frozen official Moirai2-small model on preconditioned inputs, then combining forecasts, improves accuracy without any retraining.

**Architecture:** A single new evaluation script that wraps `Moirai2Forecast` as a black box. For each dataset, we run the model once on raw data and once per preconditioning degree (d=2..8, Chebyshev, stride=1). We reverse preconditioned forecasts using the raw forecast as anchor, then combine via zero-parameter methods (uniform average, inverse-variance weighting, quantile pooling). Output in the same CSV format as existing `eval_gifteval.py`.

**Tech Stack:** PyTorch, GluonTS (QuantileForecast), uni2ts (Moirai2Forecast, precondition.py), numpy, pandas, scipy.stats.gmean

---

## Reference Files

- **Existing eval script**: `gifteval/eval_gifteval.py` — dataset configs, metric computation, CSV output format
- **Precondition coefficients**: `uni2ts/src/uni2ts/common/precondition.py` — `compute_polynomial_coefficients()`
- **Forecast class**: `uni2ts/src/uni2ts/model/moirai2/forecast.py` — `Moirai2Forecast.predict()` takes list of numpy arrays, returns `(batch, num_quantiles, prediction_length, target_dim)`
- **Evaluation utility**: `uni2ts/src/uni2ts/eval_util/evaluation.py` — `evaluate_model()` uses GluonTS predictor pipeline
- **GIFT-Eval data**: `gifteval/gift-eval/src/gift_eval/data.py` — `Dataset` class

Key interface: `Moirai2Forecast.predict(past_target=[np.array])` → `np.array(batch, num_quantiles, prediction_length, target_dim)`. The model internally z-scores, patches, runs transformer, de-normalizes. We precondition **before** handing data to `.predict()` and post-process the output.

**Important**: The model does its own z-scoring internally (via `PackedStdScaler`). So we precondition the **raw** series, not a pre-z-scored series. The model will z-score the preconditioned series with its own context-window statistics. This means the reversal operates in the model's output space (already de-normalized).

---

### Task 1: Core preconditioning utilities

**Files:**
- Create: `gifteval/eval_inference_precond.py`

**Step 1: Write the preconditioning and reversal functions**

```python
#!/usr/bin/env python3
"""
Inference-time preconditioning ensemble for Moirai2.

Runs a frozen model on raw + preconditioned inputs, combines forecasts.
No retraining required.
"""

import sys
import os
import numpy as np
from typing import List, Optional, Dict, Tuple

# Reuse existing coefficient computation
sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/uni2ts/src")
from uni2ts.common.precondition import compute_polynomial_coefficients


def precondition_series(series: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Apply causal FIR preconditioning to a 1D series.

    p(x)[t] = x[t] + sum_{i=1}^{d} c_i * x[t - i]

    First d values are left unchanged (not enough history).

    Args:
        series: 1D array of shape (T,) or 2D of shape (T, 1)
        coeffs: array of shape (d,) with coefficients [c1, c2, ..., cd]

    Returns:
        Preconditioned series, same shape as input
    """
    squeeze = False
    if series.ndim == 1:
        series = series[:, np.newaxis]
        squeeze = True

    T = series.shape[0]
    d = len(coeffs)
    result = series.copy()

    for t in range(d, T):
        for i in range(d):
            result[t] += coeffs[i] * series[t - (i + 1)]

    if squeeze:
        result = result.squeeze(-1)
    return result


def reverse_with_raw_anchor(
    Q_precond: np.ndarray,
    Q_raw: np.ndarray,
    z_context_tail: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Reverse preconditioning on forecast using raw forecast as anchor.

    Q_reversed[t] = Q_precond[t] - sum_{i=1}^{d} c_i * anchor[t - i]

    Where anchor[t] = Q_raw[t] for t >= 0 (prediction window),
                      z_context_tail[t] for t < 0 (context window).

    Args:
        Q_precond: shape (num_quantiles, H, 1) — preconditioned forecast
        Q_raw: shape (num_quantiles, H, 1) — raw forecast (anchor)
        z_context_tail: shape (d,) — last d values of raw context series
        coeffs: shape (d,) — [c1, ..., cd]

    Returns:
        Reversed forecast, shape (num_quantiles, H, 1)
    """
    num_q, H, tgt_dim = Q_precond.shape
    d = len(coeffs)
    result = Q_precond.copy()

    for t in range(H):
        correction = np.zeros((num_q, tgt_dim))
        for i in range(d):
            lag_t = t - (i + 1)
            if lag_t >= 0:
                # Use raw forecast as anchor
                correction += coeffs[i] * Q_raw[:, lag_t, :]
            else:
                # Use observed context values
                ctx_idx = len(z_context_tail) + lag_t
                if ctx_idx >= 0:
                    correction += coeffs[i] * z_context_tail[ctx_idx]
        result[:, t, :] -= correction

    return result
```

**Step 2: Verify coefficient computation works**

Run interactively:
```bash
python -c "
from uni2ts.common.precondition import compute_polynomial_coefficients
import numpy as np
for d in range(2, 9):
    c = compute_polynomial_coefficients('chebyshev', d)
    print(f'd={d}: {c}')
"
```

Expected: prints coefficient arrays for each degree (odd indices ~0 for Chebyshev).

**Step 3: Commit**

```bash
git add gifteval/eval_inference_precond.py
git commit -m "feat: add core preconditioning utilities for inference-time ensemble"
```

---

### Task 2: Single-pass evaluation function

**Files:**
- Modify: `gifteval/eval_inference_precond.py`

**Step 1: Add the single-dataset evaluation function that returns raw quantile arrays**

This bypasses the GluonTS predictor pipeline and calls `Moirai2Forecast.predict()` directly, so we can intercept inputs and outputs.

```python
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import gmean

# Add paths
sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/gifteval/gift-eval/src")

from dotenv import load_dotenv
load_dotenv("/scratch/gpfs/EHAZAN/jh1161/uni2ts/.env")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from gift_eval.data import Dataset
from gluonts.ev.metrics import MAE, MSE, MASE, SMAPE
from gluonts.time_feature import get_seasonality
from gluonts.model.forecast import QuantileForecast


def load_moirai2_model(prediction_length: int, context_length: int = 1000):
    """Load official Moirai2-small from HuggingFace."""
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    module = Moirai2Module.from_pretrained("Salesforce/moirai-1.1-R-small")
    model = Moirai2Forecast(
        prediction_length=prediction_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=context_length,
        module=module,
    )
    return model


def run_single_pass(
    model,
    past_targets: List[np.ndarray],
    precond_coeffs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run model on a list of series, optionally preconditioning inputs.

    Args:
        model: Moirai2Forecast instance (already on correct device)
        past_targets: list of np.array, each shape (T,) or (T, 1)
        precond_coeffs: if provided, precondition each series before feeding to model

    Returns:
        predictions: np.array of shape (batch, num_quantiles, H, 1)
    """
    if precond_coeffs is not None:
        past_targets = [precondition_series(s, precond_coeffs) for s in past_targets]

    # Ensure 2D
    past_targets_2d = []
    for s in past_targets:
        s = np.asarray(s, dtype=np.float32)
        if s.ndim == 1:
            s = s[:, np.newaxis]
        past_targets_2d.append(s)

    with torch.no_grad():
        preds = model.predict(past_targets_2d)

    return preds  # (batch, num_quantiles, H, 1)
```

**Step 2: Verify the model loads and runs on a tiny example**

```bash
# On a GPU node:
python -c "
import sys, os, numpy as np, torch
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/uni2ts/src')
sys.path.insert(0, '/scratch/gpfs/EHAZAN/jh1161/gifteval/gift-eval/src')
os.environ['HF_HUB_OFFLINE'] = '1'
from dotenv import load_dotenv
load_dotenv('/scratch/gpfs/EHAZAN/jh1161/uni2ts/.env')
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
module = Moirai2Module.from_pretrained('Salesforce/moirai-1.1-R-small')
model = Moirai2Forecast(prediction_length=24, target_dim=1,
    feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    context_length=100, module=module)
model = model.to('cuda').eval()
x = [np.random.randn(100, 1).astype(np.float32)]
preds = model.predict(x)
print(f'Output shape: {preds.shape}')  # expect (1, 9, 24, 1)
"
```

Expected: `Output shape: (1, 9, 24, 1)`

**Step 3: Commit**

```bash
git add gifteval/eval_inference_precond.py
git commit -m "feat: add single-pass evaluation with model loading"
```

---

### Task 3: Combination methods

**Files:**
- Modify: `gifteval/eval_inference_precond.py`

**Step 1: Add forecast combination functions**

```python
def combine_forecasts_uniform(
    forecasts: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Uniform average of quantile forecasts.

    Args:
        forecasts: {name: array of shape (batch, num_quantiles, H, 1)}

    Returns:
        Combined forecast, same shape
    """
    arrays = list(forecasts.values())
    return np.mean(arrays, axis=0)


def combine_forecasts_inv_variance(
    forecasts: Dict[str, np.ndarray],
    quantile_levels: List[float],
) -> np.ndarray:
    """
    Inverse-variance weighted average. Variance proxy = IQR (Q0.9 - Q0.1).

    Args:
        forecasts: {name: array of shape (batch, num_quantiles, H, 1)}
        quantile_levels: list of quantile levels (e.g. [0.1, 0.2, ..., 0.9])

    Returns:
        Combined forecast, same shape
    """
    # Find indices for Q0.1 and Q0.9
    q10_idx = quantile_levels.index(0.1)
    q90_idx = quantile_levels.index(0.9)

    weights = {}
    for name, preds in forecasts.items():
        # IQR per sample, averaged over horizon
        iqr = np.mean(np.abs(preds[:, q90_idx, :, :] - preds[:, q10_idx, :, :]))
        weights[name] = 1.0 / max(iqr, 1e-8)

    total_w = sum(weights.values())
    result = np.zeros_like(list(forecasts.values())[0])
    for name, preds in forecasts.items():
        result += (weights[name] / total_w) * preds

    return result


def combine_forecasts_quantile_pool(
    forecasts: Dict[str, np.ndarray],
    quantile_levels: List[float],
) -> np.ndarray:
    """
    Pool all quantile predictions, then re-extract quantiles.

    For each (batch, horizon, target_dim) position, gather all
    num_quantiles * K values from K forecasts, then compute new quantiles.

    Args:
        forecasts: {name: array of shape (batch, num_quantiles, H, 1)}
        quantile_levels: list of quantile levels

    Returns:
        Combined forecast, shape (batch, num_quantiles, H, 1)
    """
    # Stack all forecasts: (K, batch, num_quantiles, H, 1)
    stacked = np.stack(list(forecasts.values()), axis=0)
    # Reshape to pool quantiles: (batch, K * num_quantiles, H, 1)
    K, B, Q, H, D = stacked.shape
    pooled = stacked.transpose(1, 0, 2, 3, 4).reshape(B, K * Q, H, D)

    # Compute new quantiles along the pooled dimension
    result = np.quantile(pooled, quantile_levels, axis=1)
    # result shape: (num_quantiles, B, H, D) -> transpose to (B, num_quantiles, H, D)
    result = result.transpose(1, 0, 2, 3)

    return result
```

**Step 2: Test combination functions with synthetic data**

```bash
python -c "
import numpy as np

# Synthetic forecasts: 2 models, batch=1, 9 quantiles, horizon=10, dim=1
np.random.seed(42)
f1 = np.random.randn(1, 9, 10, 1)
f2 = np.random.randn(1, 9, 10, 1)
forecasts = {'raw': f1, 'd4': f2}

# Uniform
uniform = np.mean([f1, f2], axis=0)
assert uniform.shape == (1, 9, 10, 1)
print(f'Uniform: shape={uniform.shape}, mean={uniform.mean():.4f}')

# Quick check inv-var
ql = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
iqr1 = np.mean(np.abs(f1[:, 8, :, :] - f1[:, 0, :, :]))
iqr2 = np.mean(np.abs(f2[:, 8, :, :] - f2[:, 0, :, :]))
print(f'IQR1={iqr1:.4f}, IQR2={iqr2:.4f}')
print('OK')
"
```

Expected: prints shapes and IQR values, ends with "OK".

**Step 3: Commit**

```bash
git add gifteval/eval_inference_precond.py
git commit -m "feat: add forecast combination methods (uniform, inv-var, quantile pool)"
```

---

### Task 4: Full evaluation loop with metric computation

**Files:**
- Modify: `gifteval/eval_inference_precond.py`

**Step 1: Add the per-dataset evaluation function and metric computation**

The key challenge: the existing `evaluate_model()` uses the GluonTS predictor pipeline, but we need to run the model multiple times per dataset (raw + each degree) and combine. So we compute metrics manually using GluonTS metric classes.

```python
from gluonts.ev.metrics import MAE, MSE, MASE, SMAPE


def compute_mase(
    forecasts: np.ndarray,
    labels: np.ndarray,
    past_data: np.ndarray,
    seasonality: int,
    quantile_idx: int = 4,  # index of 0.5 quantile in [0.1, ..., 0.9]
) -> float:
    """
    Compute MASE at median quantile.

    Args:
        forecasts: (batch, num_quantiles, H, 1)
        labels: (batch, H) or (batch, H, 1) — ground truth
        past_data: (batch, T) or (batch, T, 1) — historical data for seasonal error
        seasonality: seasonal period
        quantile_idx: index of median quantile

    Returns:
        MASE value (scalar)
    """
    # Extract median forecast
    median_forecast = forecasts[:, quantile_idx, :, 0]  # (batch, H)

    if labels.ndim == 3:
        labels = labels[:, :, 0]
    if past_data.ndim == 3:
        past_data = past_data[:, :, 0]

    # MAE
    mae = np.nanmean(np.abs(median_forecast - labels))

    # Seasonal naive error (per series, then average)
    seasonal_errors = []
    for i in range(past_data.shape[0]):
        s = past_data[i]
        s = s[~np.isnan(s)]
        if len(s) > seasonality:
            naive_errors = np.abs(s[seasonality:] - s[:-seasonality])
            se = np.mean(naive_errors)
        else:
            se = np.nanmean(np.abs(np.diff(s)))
        seasonal_errors.append(max(se, 1e-8))

    mean_seasonal_error = np.mean(seasonal_errors)

    return mae / mean_seasonal_error


# Import the GIFTEVAL_CONFIGS from existing script
from eval_gifteval import GIFTEVAL_CONFIGS, QUICK_CONFIGS


def evaluate_single_dataset_ensemble(
    module,
    dataset_name: str,
    term: str,
    degrees: List[int],
    strategies: List[str],
    context_length: int = 1000,
    device: str = "cuda",
) -> Dict:
    """
    Run raw + preconditioned passes on a single dataset, combine, compute metrics.

    Args:
        module: Moirai2Module (shared across all passes)
        dataset_name: GIFT-Eval dataset name
        term: "short", "medium", or "long"
        degrees: list of polynomial degrees to try
        strategies: list of strategies ("A1", "A2")
        context_length: context length for model
        device: "cuda" or "cpu"

    Returns:
        Dict with per-pass and combined MASE scores
    """
    from uni2ts.model.moirai2 import Moirai2Forecast

    try:
        # Load dataset
        dataset_check = Dataset(name=dataset_name, term=term, to_univariate=False)
        is_multivariate = dataset_check.target_dim > 1
        dataset = Dataset(name=dataset_name, term=term, to_univariate=is_multivariate)
        prediction_length = dataset.prediction_length

        try:
            seasonality = get_seasonality(dataset.freq)
        except:
            seasonality = 1

        # Build model wrapper
        model = Moirai2Forecast(
            prediction_length=prediction_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=context_length,
            module=module,
        )
        model = model.to(device).eval()

        # Extract test data as numpy arrays
        test_data = dataset.test_data
        past_targets = []
        labels = []
        for input_entry, label_entry in zip(test_data.input, test_data.label):
            past_targets.append(input_entry["target"])
            labels.append(label_entry["target"])

        labels = np.stack(labels, axis=0)  # (batch, H) or (batch, H, D)

        # --- Raw pass ---
        Q_raw = run_single_pass(model, past_targets, precond_coeffs=None)
        # Q_raw shape: (batch, num_quantiles, H, 1)

        raw_mase = compute_mase(Q_raw, labels,
                                np.stack([np.asarray(s) for s in past_targets]),
                                seasonality)

        results = {
            "dataset": dataset_name,
            "term": term,
            "prediction_length": prediction_length,
            "raw_MASE": raw_mase,
        }

        # --- Preconditioned passes ---
        all_forecasts = {"raw": Q_raw}

        for d in degrees:
            coeffs = compute_polynomial_coefficients("chebyshev", d)

            if "A1" in strategies:
                # Strategy A1: replace + reverse with raw anchor
                Q_precond = run_single_pass(model, past_targets, precond_coeffs=coeffs)

                # Reverse using raw anchor
                Q_a1 = np.zeros_like(Q_precond)
                for b in range(Q_precond.shape[0]):
                    s = np.asarray(past_targets[b]).flatten()
                    context_tail = s[-d:] if len(s) >= d else s
                    Q_a1[b] = reverse_with_raw_anchor(
                        Q_precond[b], Q_raw[b], context_tail, coeffs
                    )

                a1_mase = compute_mase(Q_a1, labels,
                                       np.stack([np.asarray(s) for s in past_targets]),
                                       seasonality)
                results[f"A1_d{d}_MASE"] = a1_mase
                all_forecasts[f"A1_d{d}"] = Q_a1

            if "A2" in strategies:
                # Strategy A2: forecast residual, add back
                residuals = []
                for s in past_targets:
                    s_arr = np.asarray(s, dtype=np.float32)
                    if s_arr.ndim == 1:
                        s_arr = s_arr[:, np.newaxis]
                    p = precondition_series(s_arr, coeffs)
                    residuals.append(p - s_arr)

                Q_residual = run_single_pass(model, residuals, precond_coeffs=None)
                Q_a2 = Q_raw + Q_residual

                a2_mase = compute_mase(Q_a2, labels,
                                       np.stack([np.asarray(s) for s in past_targets]),
                                       seasonality)
                results[f"A2_d{d}_MASE"] = a2_mase
                all_forecasts[f"A2_d{d}"] = Q_a2

        # --- Combinations ---
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for strategy in strategies:
            prefix = strategy
            strategy_forecasts = {k: v for k, v in all_forecasts.items()
                                  if k == "raw" or k.startswith(prefix)}

            if len(strategy_forecasts) > 1:
                # Uniform
                Q_uniform = combine_forecasts_uniform(strategy_forecasts)
                results[f"{prefix}_uniform_MASE"] = compute_mase(
                    Q_uniform, labels,
                    np.stack([np.asarray(s) for s in past_targets]),
                    seasonality)

                # Inverse variance
                Q_invvar = combine_forecasts_inv_variance(strategy_forecasts, quantile_levels)
                results[f"{prefix}_invvar_MASE"] = compute_mase(
                    Q_invvar, labels,
                    np.stack([np.asarray(s) for s in past_targets]),
                    seasonality)

                # Quantile pool
                Q_pool = combine_forecasts_quantile_pool(strategy_forecasts, quantile_levels)
                results[f"{prefix}_qpool_MASE"] = compute_mase(
                    Q_pool, labels,
                    np.stack([np.asarray(s) for s in past_targets]),
                    seasonality)

                # Oracle (best single pass including raw)
                best_name = min(strategy_forecasts.keys(),
                                key=lambda k: results.get(f"{k}_MASE", results.get("raw_MASE", 999)))
                results[f"{prefix}_oracle_name"] = best_name

        return results

    except Exception as e:
        return {
            "dataset": dataset_name,
            "term": term,
            "error": str(e),
        }
    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
```

**Step 2: Verify on a single quick dataset (hospital/short)**

```bash
# On GPU node:
cd /scratch/gpfs/EHAZAN/jh1161
python -c "
import sys
sys.path.insert(0, 'gifteval')
from eval_inference_precond import *
from uni2ts.model.moirai2 import Moirai2Module
module = Moirai2Module.from_pretrained('Salesforce/moirai-1.1-R-small')
result = evaluate_single_dataset_ensemble(
    module, 'hospital', 'short', degrees=[4], strategies=['A1'], device='cuda')
print(result)
"
```

Expected: Dict with `raw_MASE` and `A1_d4_MASE` values (both should be finite, reasonable numbers ~0.5-3.0).

**Step 3: Commit**

```bash
git add gifteval/eval_inference_precond.py
git commit -m "feat: add full evaluation loop with metric computation and combination"
```

---

### Task 5: Main function and CLI

**Files:**
- Modify: `gifteval/eval_inference_precond.py`

**Step 1: Add the main function with argument parsing and output**

```python
import argparse
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="Inference-time preconditioning ensemble evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick subset (8 datasets)")
    parser.add_argument("--degrees", type=str, default="2,3,4,5,6,7,8",
                        help="Comma-separated polynomial degrees")
    parser.add_argument("--strategies", type=str, default="A1,A2",
                        help="Comma-separated strategies (A1, A2)")
    parser.add_argument("--context-length", type=int, default=1000)
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/gpfs/EHAZAN/jh1161/gifteval/results")
    args = parser.parse_args()

    degrees = [int(d) for d in args.degrees.split(",")]
    strategies = args.strategies.split(",")
    configs = QUICK_CONFIGS if args.quick else GIFTEVAL_CONFIGS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Degrees: {degrees}")
    print(f"Strategies: {strategies}")
    print(f"Configs: {len(configs)}")

    # Load module once
    from uni2ts.model.moirai2 import Moirai2Module
    module = Moirai2Module.from_pretrained("Salesforce/moirai-1.1-R-small")
    print("Model loaded")

    # Run evaluation
    all_results = []
    for i, (dataset_name, term) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {dataset_name}/{term}")
        result = evaluate_single_dataset_ensemble(
            module, dataset_name, term, degrees, strategies,
            args.context_length, device)
        all_results.append(result)

        if "error" not in result:
            print(f"  raw={result['raw_MASE']:.4f}", end="")
            for d in degrees:
                for s in strategies:
                    key = f"{s}_d{d}_MASE"
                    if key in result:
                        print(f"  {s}_d{d}={result[key]:.4f}", end="")
            print()
        else:
            print(f"  ERROR: {result['error']}")

    # Save results
    df = pd.DataFrame(all_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"inference_precond_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Geometric Mean MASE across datasets")
    print("=" * 80)

    success_df = df[~df.get("error", pd.Series([None]*len(df))).notna()]
    mase_cols = [c for c in success_df.columns if c.endswith("_MASE")]

    for col in sorted(mase_cols):
        valid = success_df[col].dropna()
        valid = valid[(valid > 0) & (valid < 100)]
        if len(valid) > 0:
            print(f"  {col:30s}: {gmean(valid):.4f} (n={len(valid)})")


if __name__ == "__main__":
    main()
```

**Step 2: Test CLI with --quick flag on GPU**

```bash
cd /scratch/gpfs/EHAZAN/jh1161
python gifteval/eval_inference_precond.py --quick --degrees 4 --strategies A1
```

Expected: Runs on 8 datasets, prints per-dataset raw and A1_d4 MASE, saves CSV.

**Step 3: Commit**

```bash
git add gifteval/eval_inference_precond.py
git commit -m "feat: add CLI and summary output for inference precond ensemble"
```

---

### Task 6: SLURM job script

**Files:**
- Create: `gifteval/eval_inference_precond.slurm`

**Step 1: Write the SLURM script**

```bash
#!/bin/bash
#SBATCH --job-name=inf_precond
#SBATCH --output=/scratch/gpfs/EHAZAN/jh1161/logs/inf_precond_%j.out
#SBATCH --error=/scratch/gpfs/EHAZAN/jh1161/logs/inf_precond_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=pli-low
#SBATCH --account=eladgroup

module load anaconda3/2024.6
module load intel-mkl/2024.2
module load cudatoolkit/12.6
source /scratch/gpfs/EHAZAN/jh1161/uni2ts/venv/bin/activate

cd /scratch/gpfs/EHAZAN/jh1161

# Default: full sweep, all strategies
DEGREES="${DEGREES:-2,3,4,5,6,7,8}"
STRATEGIES="${STRATEGIES:-A1,A2}"
QUICK="${QUICK:-}"

echo "=== Inference-Time Preconditioning Ensemble ==="
echo "Degrees: $DEGREES"
echo "Strategies: $STRATEGIES"
echo "Quick: $QUICK"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

ARGS="--degrees $DEGREES --strategies $STRATEGIES"
if [ -n "$QUICK" ]; then
    ARGS="$ARGS --quick"
fi

python gifteval/eval_inference_precond.py $ARGS

echo "=== Done ==="
date
```

**Step 2: Submit a quick test job**

```bash
QUICK=1 DEGREES=4 STRATEGIES=A1 sbatch gifteval/eval_inference_precond.slurm
```

Expected: Job submits, completes in ~10 min, output in `logs/inf_precond_*.out`.

**Step 3: Commit**

```bash
git add gifteval/eval_inference_precond.slurm
git commit -m "feat: add SLURM script for inference-time preconditioning evaluation"
```

---

### Task 7: Quick validation run and results analysis

**Step 1: Run quick validation (8 datasets, d=4 only, A1 strategy)**

```bash
# Interactive GPU:
cd /scratch/gpfs/EHAZAN/jh1161
python gifteval/eval_inference_precond.py --quick --degrees 4 --strategies A1
```

**Step 2: Check results**

Examine the CSV output. Key questions:
- Is `raw_MASE` consistent with our known baseline (~1.24 geo mean on full, maybe ~1.1-1.5 range on individual datasets)?
- Is `A1_d4_MASE` finite and in a reasonable range?
- Does the uniform combination help or hurt?

**Step 3: If results look sensible, run full Phase 1**

```bash
# Submit full evaluation job
sbatch gifteval/eval_inference_precond.slurm
```

---

### Task 8: Results analysis script

**Files:**
- Create: `gifteval/analyze_inference_precond.py`

**Step 1: Write analysis script for Phase 2 and Phase 3**

```python
#!/usr/bin/env python3
"""Analyze inference-time preconditioning results."""

import sys
import pandas as pd
import numpy as np
from scipy.stats import gmean
from pathlib import Path


def analyze(csv_path: str):
    df = pd.read_csv(csv_path)
    success = df[df.get("error", pd.Series([None]*len(df))).isna()].copy()

    print(f"Loaded {len(success)} successful evaluations from {csv_path}")

    mase_cols = sorted([c for c in success.columns if c.endswith("_MASE")])

    # Geometric mean MASE per method
    print("\n=== Geometric Mean MASE ===")
    for col in mase_cols:
        valid = success[col].dropna()
        valid = valid[(valid > 0) & (valid < 100)]
        if len(valid) > 0:
            gm = gmean(valid)
            vs_raw = (gm / gmean(success["raw_MASE"].dropna()) - 1) * 100
            print(f"  {col:35s}: {gm:.4f} ({vs_raw:+.2f}% vs raw)")

    # Per-dataset: how often does each method beat raw?
    print("\n=== Win Rate vs Raw ===")
    for col in mase_cols:
        if col == "raw_MASE":
            continue
        valid = success[["raw_MASE", col]].dropna()
        wins = (valid[col] < valid["raw_MASE"]).sum()
        print(f"  {col:35s}: {wins}/{len(valid)} datasets")

    # Per-frequency analysis
    if "dataset" in success.columns:
        print("\n=== Results by Frequency Category ===")
        # Categorize by freq pattern in dataset name
        def categorize(name):
            if "/5T" in name or "/10T" in name or "/15T" in name:
                return "high_freq"
            elif "/H" in name:
                return "hourly"
            elif "/D" in name:
                return "daily"
            elif "/W" in name or "/M" in name:
                return "low_freq"
            else:
                return "other"

        success["freq_cat"] = success["dataset"].apply(categorize)
        for cat in ["high_freq", "hourly", "daily", "low_freq", "other"]:
            subset = success[success["freq_cat"] == cat]
            if len(subset) == 0:
                continue
            print(f"\n  {cat} ({len(subset)} configs):")
            for col in mase_cols[:5]:  # Top 5 methods
                valid = subset[col].dropna()
                valid = valid[(valid > 0) & (valid < 100)]
                if len(valid) > 0:
                    print(f"    {col:30s}: {gmean(valid):.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_inference_precond.py <results.csv>")
        sys.exit(1)
    analyze(sys.argv[1])
```

**Step 2: Run analysis on results**

```bash
python gifteval/analyze_inference_precond.py gifteval/results/inference_precond_<timestamp>.csv
```

**Step 3: Commit**

```bash
git add gifteval/analyze_inference_precond.py
git commit -m "feat: add analysis script for inference-time preconditioning results"
```
