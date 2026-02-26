#!/usr/bin/env python3
"""
Inference-time preconditioning ensemble evaluation on GIFT-Eval.

Runs a frozen pretrained model (Moirai v1 from HuggingFace) on raw and
preconditioned inputs, reverses the preconditioned forecasts, and combines
them via zero-parameter methods.

Strategies:
  A1: precondition input -> model -> reverse forecast using raw forecast as anchor
  A2: feed residual (preconditioned - raw) -> model -> add raw forecast back

Usage:
    python eval_inference_precond.py --quick
    python eval_inference_precond.py --degrees 2,4,6
    python eval_inference_precond.py --quick --strategies A1
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/uni2ts/src")
sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/gifteval/gift-eval/src")
sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/gifteval")

from dotenv import load_dotenv
load_dotenv("/scratch/gpfs/EHAZAN/jh1161/uni2ts/.env")

import numpy as np
import pandas as pd
import torch
from gluonts.time_feature import get_seasonality
from scipy.stats import gmean

from gift_eval.data import Dataset
from uni2ts.common.precondition import compute_polynomial_coefficients
from eval_gifteval import GIFTEVAL_CONFIGS, QUICK_CONFIGS

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN_IDX = 4  # index of 0.5 in QUANTILE_LEVELS


# ---------------------------------------------------------------------------
# 1. Preconditioning utilities
# ---------------------------------------------------------------------------

def precondition_series(series: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Apply causal FIR filter: p(x)[t] = x[t] + sum_{i=1}^{d} c_i * x[t-i].
    First d values are left unchanged (not enough history)."""
    d = len(coeffs)
    out = series.copy()
    for t in range(d, len(series)):
        for i in range(d):
            out[t] += coeffs[i] * series[t - i - 1]
    return out


def reverse_with_raw_anchor(
    Q_precond: np.ndarray,
    Q_raw: np.ndarray,
    z_context_tail: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Reverse preconditioning on forecast quantiles.

    Q_reversed[t] = Q_precond[t] - sum_{i=1}^{d} c_i * anchor[t-i]
    where anchor[t] = Q_raw[t] for t >= 0, z_context_tail for t < 0.

    Args:
        Q_precond: preconditioned forecast, shape (..., H)
        Q_raw: raw forecast (anchor for future steps), shape (..., H)
        z_context_tail: last d values of the raw input series, shape (d,)
        coeffs: FIR coefficients, shape (d,)

    Returns:
        Reversed forecast, same shape as Q_precond.
    """
    d = len(coeffs)
    H = Q_precond.shape[-1]
    # Build full anchor: [z_context_tail, Q_raw]
    # z_context_tail has shape (d,), Q_raw has shape (..., H)
    # We need anchor at indices t-1, t-2, ..., t-d for each forecast step t
    out = Q_precond.copy()
    for t in range(H):
        for i in range(d):
            anchor_idx = t - i - 1  # index into forecast (0-based)
            if anchor_idx < 0:
                # Use context tail: z_context_tail[d + anchor_idx]
                ctx_idx = d + anchor_idx
                if ctx_idx >= 0:
                    out[..., t] -= coeffs[i] * z_context_tail[ctx_idx]
            else:
                out[..., t] -= coeffs[i] * Q_raw[..., anchor_idx]
    return out


# ---------------------------------------------------------------------------
# 2. Model loading and single-pass evaluation
# ---------------------------------------------------------------------------

def load_model(model_name: str, prediction_length: int, context_length: int = 1000):
    """Load a MoiraiForecast model from HuggingFace and build Moirai2Forecast wrapper.

    Returns a Moirai2Forecast instance with a .predict() method.
    """
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    # Try Moirai2Module first
    try:
        module = Moirai2Module.from_pretrained(f"Salesforce/{model_name}")
    except Exception:
        # Fall back to loading MoiraiModule (v1) and wrapping with Moirai2Forecast
        from uni2ts.model.moirai import MoiraiModule
        module_v1 = MoiraiModule.from_pretrained(f"Salesforce/{model_name}")

        # Create a thin wrapper that adapts MoiraiModule to work with Moirai2Forecast
        # by extracting the module and using Moirai2Forecast.predict()-like logic.
        # Since MoiraiModule is not directly compatible with Moirai2Forecast,
        # we return (module_v1, "v1") and handle it in run_single_pass.
        return module_v1, "v1"

    forecast = Moirai2Forecast(
        prediction_length=prediction_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=context_length,
        module=module,
    )
    return forecast, "v2"


def _prepare_inputs_v1(past_targets: List[np.ndarray], context_length: int):
    """Prepare numpy arrays into padded/sliced arrays + pad masks for v1 model."""
    from uni2ts.transform.imputation import CausalMeanImputation
    impute = CausalMeanImputation()

    processed = []
    pad_masks = []
    for arr in past_targets:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a[:, np.newaxis]
        if np.isnan(a).any():
            a = impute(a)

        pad_length = max(0, context_length - a.shape[0])
        if a.shape[0] > context_length:
            a = a[-context_length:]
            pad_mask = np.zeros(context_length, dtype=bool)
        elif pad_length > 0:
            pad_block = np.full((pad_length, 1), a[0, 0], dtype=np.float32)
            a = np.concatenate([pad_block, a], axis=0)
            pad_mask = np.zeros(context_length, dtype=bool)
            pad_mask[:pad_length] = True
        else:
            pad_mask = np.zeros(context_length, dtype=bool)

        processed.append(a)
        pad_masks.append(pad_mask)

    return np.array(processed), np.array(pad_masks)


def _predict_v1(module, past_targets: List[np.ndarray],
                prediction_length: int, context_length: int,
                device: str, patch_size: int = 32,
                num_samples: int = 100,
                batch_size: int = 32) -> np.ndarray:
    """Run prediction using MoiraiModule (v1) by calling MoiraiForecast.forward().

    Builds input tensors directly, calls forward to get sample-based forecasts,
    then computes empirical quantiles. Processes in mini-batches.

    Returns: (total_series, num_quantiles, H, 1)
    """
    from uni2ts.model.moirai import MoiraiForecast

    forecast = MoiraiForecast(
        module=module,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    forecast = forecast.to(device)
    forecast.eval()

    all_processed, all_pad_masks = _prepare_inputs_v1(past_targets, context_length)
    N = len(past_targets)
    all_quantiles = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_data = all_processed[start:end]
        batch_pads = all_pad_masks[start:end]

        past_target = torch.tensor(batch_data, device=device, dtype=torch.float32)
        past_observed = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.tensor(batch_pads, device=device, dtype=torch.bool)

        with torch.no_grad():
            # Returns (batch, num_samples, H, 1) for fixed patch_size
            samples = forecast(
                past_target=past_target,
                past_observed_target=past_observed,
                past_is_pad=past_is_pad,
            )

        samples_np = samples.cpu().numpy()  # (B, num_samples, H) or (B, S, H, D)
        if samples_np.ndim == 3:
            samples_np = samples_np[..., np.newaxis]  # (B, S, H, 1)
        B_cur, S, H, D = samples_np.shape
        batch_result = np.zeros((B_cur, len(QUANTILE_LEVELS), H, D))
        for q_idx, q_level in enumerate(QUANTILE_LEVELS):
            batch_result[:, q_idx, :, :] = np.quantile(samples_np, q_level, axis=1)
        all_quantiles.append(batch_result)

    del forecast
    return np.concatenate(all_quantiles, axis=0)


def _predict_v2(forecast_model, past_targets: List[np.ndarray],
                device: str, batch_size: int = 32) -> np.ndarray:
    """Run prediction using Moirai2Forecast.predict() in mini-batches.

    Returns: (total_series, num_quantiles, H, 1)
    """
    forecast_model = forecast_model.to(device)
    forecast_model.eval()
    N = len(past_targets)
    all_preds = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = past_targets[start:end]
        with torch.no_grad():
            preds = forecast_model.predict(batch)
        if preds.ndim == 3:
            preds = preds[..., np.newaxis]  # (B, Q, H) -> (B, Q, H, 1)
        all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


def run_single_pass(
    model,
    model_type: str,
    past_targets: List[np.ndarray],
    prediction_length: int,
    context_length: int,
    device: str = "cuda",
    precond_coeffs: Optional[np.ndarray] = None,
    patch_size: int = 32,
    batch_size: int = 32,
) -> np.ndarray:
    """Run model on a list of series, optionally preconditioning first.

    Args:
        model: The loaded module (MoiraiModule for v1, Moirai2Forecast for v2).
        model_type: "v1" or "v2".
        past_targets: List of np.ndarray each shape (T, 1).
        prediction_length: Forecast horizon H.
        context_length: Context window size.
        device: Torch device.
        precond_coeffs: If given, precondition each series before feeding to model.
        patch_size: Patch size for v1 model.
        batch_size: Mini-batch size for GPU processing.

    Returns: (batch, num_quantiles, H, 1)
    """
    if precond_coeffs is not None:
        processed = []
        for arr in past_targets:
            flat = arr.flatten()
            precond_flat = precondition_series(flat, precond_coeffs)
            processed.append(precond_flat[:, np.newaxis].astype(np.float32))
        past_targets = processed

    if model_type == "v1":
        return _predict_v1(model, past_targets, prediction_length,
                           context_length, device, patch_size=patch_size,
                           batch_size=batch_size)
    else:
        return _predict_v2(model, past_targets, device, batch_size=batch_size)


# ---------------------------------------------------------------------------
# 3. Combination methods
# ---------------------------------------------------------------------------

def combine_forecasts_uniform(forecasts: Dict[str, np.ndarray]) -> np.ndarray:
    """Simple average across all forecast variants.
    Each value in forecasts: (batch, num_quantiles, H, 1)."""
    stacked = np.stack(list(forecasts.values()), axis=0)
    return stacked.mean(axis=0)


def combine_forecasts_inv_variance(
    forecasts: Dict[str, np.ndarray],
    quantile_levels: List[float],
) -> np.ndarray:
    """Inverse-variance weighted combination at each quantile.

    Weight each forecast inversely proportional to its variance across
    quantile levels (proxy for uncertainty). More confident forecasts
    get higher weight."""
    keys = list(forecasts.keys())
    stacked = np.stack([forecasts[k] for k in keys], axis=0)  # (K, B, Q, H, 1)

    # Compute variance across quantile dimension for each forecast
    # shape: (K, B, 1, H, 1)
    variances = stacked.var(axis=2, keepdims=True)
    variances = np.clip(variances, a_min=1e-10, a_max=None)
    weights = 1.0 / variances  # (K, B, 1, H, 1)
    weights = weights / weights.sum(axis=0, keepdims=True)  # normalize

    return (stacked * weights).sum(axis=0)


def combine_forecasts_quantile_pool(
    forecasts: Dict[str, np.ndarray],
    quantile_levels: List[float],
) -> np.ndarray:
    """Pool all forecast variants and re-extract quantiles.

    For each (batch, time_step), pool the K median forecasts and take
    empirical quantiles from them. This can capture multimodality."""
    keys = list(forecasts.keys())
    K = len(keys)
    # stacked: (K, B, Q, H, 1)
    stacked = np.stack([forecasts[k] for k in keys], axis=0)
    B, Q, H, D = stacked.shape[1:]

    # For each quantile level, pool all K forecasts at that quantile
    # and recompute empirical quantiles across the K samples
    # stacked[:, :, q, :, :] has shape (K, B, H, D)
    # We want to compute quantiles across the K dimension
    result = np.zeros((B, Q, H, D))
    for q_idx, q_level in enumerate(quantile_levels):
        # Gather all K forecasts at this quantile level: (K, B, H, D)
        pool = stacked[:, :, q_idx, :, :]
        result[:, q_idx, :, :] = np.quantile(pool, q_level, axis=0)

    return result


# ---------------------------------------------------------------------------
# 4. MASE computation
# ---------------------------------------------------------------------------

def compute_mase(
    forecasts: np.ndarray,
    labels: List[np.ndarray],
    past_data: List[np.ndarray],
    seasonality: int,
    quantile_idx: int = MEDIAN_IDX,
) -> float:
    """Compute MASE at specified quantile across all series.

    Args:
        forecasts: (batch, num_quantiles, H, 1)
        labels: list of np.ndarray, each shape (H,) or (H, 1)
        past_data: list of np.ndarray (raw historical data for naive scaling)
        seasonality: seasonal period for naive forecast
        quantile_idx: which quantile to evaluate (default: median)

    Returns:
        MASE value (scalar).
    """
    mase_values = []
    for i in range(len(labels)):
        # Point forecast at the median quantile
        pred = forecasts[i, quantile_idx, :, 0]
        label = np.asarray(labels[i]).flatten()
        H = min(len(pred), len(label))
        pred = pred[:H]
        label = label[:H]

        # Absolute errors
        ae = np.abs(pred - label)

        # Naive seasonal forecast error (scaling factor)
        past = np.asarray(past_data[i]).flatten()
        past_valid = past[~np.isnan(past)]
        if len(past_valid) > seasonality and seasonality > 0:
            naive_errors = np.abs(past_valid[seasonality:] - past_valid[:-seasonality])
            scale = np.mean(naive_errors)
        else:
            scale = np.mean(np.abs(np.diff(past_valid))) if len(past_valid) > 1 else 1.0

        if scale < 1e-10:
            scale = 1.0

        mase_values.append(np.mean(ae) / scale)

    return float(np.mean(mase_values))


# ---------------------------------------------------------------------------
# 5. Per-dataset ensemble evaluation
# ---------------------------------------------------------------------------

def evaluate_single_dataset_ensemble(
    module,
    model_type: str,
    dataset_name: str,
    term: str,
    degrees: List[int],
    strategies: List[str],
    context_length: int = 1000,
    device: str = "cuda",
    patch_size: int = 32,
    batch_size: int = 32,
) -> Dict:
    """Evaluate raw + preconditioned ensemble on one GIFT-Eval dataset.

    Returns dict with MASE scores for each variant and combination.
    """
    try:
        # Load dataset
        dataset_check = Dataset(name=dataset_name, term=term, to_univariate=False)
        is_multivariate = dataset_check.target_dim > 1
        dataset = Dataset(name=dataset_name, term=term, to_univariate=is_multivariate)
        prediction_length = dataset.prediction_length

        try:
            freq = dataset.freq
            seasonality = get_seasonality(freq)
        except Exception:
            seasonality = 1

        # Collect all test inputs and labels
        past_targets = []
        labels = []
        for inp, lab in dataset.test_data:
            target = np.asarray(inp["target"]).flatten()
            # Take last context_length values
            if len(target) > context_length:
                target = target[-context_length:]
            past_targets.append(target[:, np.newaxis].astype(np.float32))
            labels.append(np.asarray(lab["target"]).flatten())

        if len(past_targets) == 0:
            return {"dataset": dataset_name, "term": term,
                    "prediction_length": prediction_length,
                    "error": "No test data"}

        # Build model wrapper for v2 with correct prediction_length
        if model_type == "v2":
            from uni2ts.model.moirai2 import Moirai2Forecast
            forecast_model = Moirai2Forecast(
                prediction_length=prediction_length,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
                context_length=context_length,
                module=module,
            )
        else:
            forecast_model = module  # MoiraiModule, passed to _predict_v1

        # ---- Raw pass ----
        print(f"    Running raw pass ({len(past_targets)} series, H={prediction_length})...")
        Q_raw = run_single_pass(
            forecast_model, model_type, past_targets,
            prediction_length, context_length, device,
            patch_size=patch_size, batch_size=batch_size,
        )
        raw_mase = compute_mase(Q_raw, labels, past_targets, seasonality)

        result = {
            "dataset": dataset_name,
            "term": term,
            "prediction_length": prediction_length,
            "raw_MASE": raw_mase,
        }

        # Collect forecasts for combination
        all_A1_forecasts = {"raw": Q_raw}
        all_A2_forecasts = {"raw": Q_raw}

        # ---- Preconditioned passes ----
        for d in degrees:
            coeffs = compute_polynomial_coefficients("chebyshev", d)

            if "A1" in strategies:
                # A1: precondition -> model -> reverse
                print(f"    A1 d={d}: precondition -> model -> reverse...")
                Q_precond = run_single_pass(
                    forecast_model, model_type, past_targets,
                    prediction_length, context_length, device,
                    precond_coeffs=coeffs,
                    patch_size=patch_size, batch_size=batch_size,
                )

                # Reverse each series using raw forecast as anchor
                Q_reversed = np.zeros_like(Q_precond)
                for i in range(len(past_targets)):
                    raw_series = past_targets[i].flatten()
                    tail = raw_series[-d:] if d <= len(raw_series) else np.zeros(d)
                    # Reverse for all quantiles: Q_precond[i] has shape (Q, H, 1)
                    for q in range(Q_precond.shape[1]):
                        Q_reversed[i, q, :, 0] = reverse_with_raw_anchor(
                            Q_precond[i, q, :, 0],
                            Q_raw[i, q, :, 0],
                            tail,
                            coeffs,
                        )

                a1_mase = compute_mase(Q_reversed, labels, past_targets, seasonality)
                result[f"A1_d{d}_MASE"] = a1_mase
                all_A1_forecasts[f"A1_d{d}"] = Q_reversed

            if "A2" in strategies:
                # A2: compute residual, feed to model, add raw back
                print(f"    A2 d={d}: residual -> model -> add raw...")
                residual_targets = []
                for arr in past_targets:
                    flat = arr.flatten()
                    precond_flat = precondition_series(flat, coeffs)
                    resid = (precond_flat - flat)[:, np.newaxis].astype(np.float32)
                    residual_targets.append(resid)

                Q_resid = run_single_pass(
                    forecast_model, model_type, residual_targets,
                    prediction_length, context_length, device,
                    patch_size=patch_size, batch_size=batch_size,
                )
                # Final forecast = Q_raw + Q_resid
                Q_a2 = Q_raw + Q_resid

                a2_mase = compute_mase(Q_a2, labels, past_targets, seasonality)
                result[f"A2_d{d}_MASE"] = a2_mase
                all_A2_forecasts[f"A2_d{d}"] = Q_a2

        # ---- Combination methods ----
        if "A1" in strategies and len(all_A1_forecasts) > 1:
            print("    Computing A1 combinations...")
            combined_uniform = combine_forecasts_uniform(all_A1_forecasts)
            result["A1_uniform_MASE"] = compute_mase(
                combined_uniform, labels, past_targets, seasonality)

            combined_invvar = combine_forecasts_inv_variance(
                all_A1_forecasts, QUANTILE_LEVELS)
            result["A1_invvar_MASE"] = compute_mase(
                combined_invvar, labels, past_targets, seasonality)

            combined_qpool = combine_forecasts_quantile_pool(
                all_A1_forecasts, QUANTILE_LEVELS)
            result["A1_qpool_MASE"] = compute_mase(
                combined_qpool, labels, past_targets, seasonality)

        if "A2" in strategies and len(all_A2_forecasts) > 1:
            print("    Computing A2 combinations...")
            combined_uniform = combine_forecasts_uniform(all_A2_forecasts)
            result["A2_uniform_MASE"] = compute_mase(
                combined_uniform, labels, past_targets, seasonality)

            combined_invvar = combine_forecasts_inv_variance(
                all_A2_forecasts, QUANTILE_LEVELS)
            result["A2_invvar_MASE"] = compute_mase(
                combined_invvar, labels, past_targets, seasonality)

            combined_qpool = combine_forecasts_quantile_pool(
                all_A2_forecasts, QUANTILE_LEVELS)
            result["A2_qpool_MASE"] = compute_mase(
                combined_qpool, labels, past_targets, seasonality)

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "term": term,
            "prediction_length": 0,
            "error": str(e),
        }
    finally:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inference-time preconditioning ensemble evaluation on GIFT-Eval")
    parser.add_argument("--model", type=str, default="moirai-1.1-R-small",
                        help="HuggingFace model name")
    parser.add_argument("--quick", action="store_true",
                        help="Run on 8-dataset quick subset")
    parser.add_argument("--degrees", type=str, default="2,3,4,5,6,7,8",
                        help="Comma-separated Chebyshev degrees")
    parser.add_argument("--strategies", type=str, default="A1,A2",
                        help="Comma-separated strategies (A1, A2)")
    parser.add_argument("--context-length", type=int, default=1000,
                        help="Context length for model input")
    parser.add_argument("--patch-size", type=int, default=32,
                        help="Patch size for v1 model (ignored for v2)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size for GPU inference")
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/gpfs/EHAZAN/jh1161/gifteval/results",
                        help="Output directory for results")
    args = parser.parse_args()

    degrees = [int(d) for d in args.degrees.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = QUICK_CONFIGS if args.quick else GIFTEVAL_CONFIGS

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Degrees: {degrees}")
    print(f"Strategies: {strategies}")
    print(f"Configs: {len(configs)}")
    print(f"Context length: {args.context_length}")
    print()

    # Load model once
    print(f"Loading model {args.model}...")
    model, model_type = load_model(args.model, prediction_length=1,
                                    context_length=args.context_length)
    print(f"Model type: {model_type}")
    if model_type == "v1":
        model = model.to(device)

    # Evaluate each dataset
    results = []
    for i, (dataset_name, term) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {dataset_name}/{term}")
        result = evaluate_single_dataset_ensemble(
            model, model_type, dataset_name, term,
            degrees, strategies, args.context_length, device,
            patch_size=args.patch_size, batch_size=args.batch_size,
        )
        results.append(result)

        if "error" not in result:
            print(f"  raw_MASE: {result['raw_MASE']:.4f}", end="")
            for key in sorted(result.keys()):
                if key.startswith(("A1_d", "A2_d")) and key.endswith("_MASE"):
                    print(f"  {key}: {result[key]:.4f}", end="")
            print()
        else:
            print(f"  ERROR: {result['error']}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(results)
    csv_path = output_dir / f"inference_precond_{args.model}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE-TIME PRECONDITIONING ENSEMBLE RESULTS")
    print("=" * 80)

    success_df = df[~df.get("error", pd.Series([None] * len(df))).notna()]
    if len(success_df) == 0:
        print("No successful evaluations.")
        return

    print(f"\nSuccessful: {len(success_df)}/{len(configs)}")

    # Geometric mean MASE for each column
    mase_cols = [c for c in success_df.columns if c.endswith("_MASE")]
    print(f"\n{'Method':<30} {'Geo Mean MASE':>15} {'Mean MASE':>12} {'<1.0':>6}")
    print("-" * 65)
    for col in sorted(mase_cols):
        valid = success_df[col].dropna()
        valid = valid[valid > 0]
        if len(valid) == 0:
            continue
        gm = gmean(valid)
        am = valid.mean()
        below_1 = (valid < 1.0).sum()
        print(f"{col:<30} {gm:>15.4f} {am:>12.4f} {below_1:>3}/{len(valid)}")

    # Highlight best
    if len(mase_cols) > 1:
        geo_means = {}
        for col in mase_cols:
            valid = success_df[col].dropna()
            valid = valid[valid > 0]
            if len(valid) > 0:
                geo_means[col] = gmean(valid)
        if geo_means:
            best_col = min(geo_means, key=geo_means.get)
            raw_gm = geo_means.get("raw_MASE", float("inf"))
            best_gm = geo_means[best_col]
            print(f"\nBest method: {best_col} ({best_gm:.4f})")
            if raw_gm < float("inf") and best_gm < raw_gm:
                pct = (raw_gm - best_gm) / raw_gm * 100
                print(f"Improvement over raw: {pct:.2f}%")
            elif raw_gm < float("inf"):
                pct = (best_gm - raw_gm) / raw_gm * 100
                print(f"Degradation vs raw: {pct:.2f}%")


if __name__ == "__main__":
    main()
