#!/usr/bin/env python3
"""
Learned adapter for inference-time preconditioning ensemble.

Phase 1: Oracle analysis
  - Run raw + preconditioned forecasts, save per-series results
  - Compute per-series oracle (best single forecast per series)
  - Report oracle upper bound

Phase 2: Learned adapter
  - Train a simple model to combine raw + preconditioned forecasts
  - Architectures: (a) Global linear weights, (b) Feature-conditioned MLP
  - Validation: Leave-one-dataset-out cross-validation

Usage:
    python eval_adapter_precond.py --quick --phase oracle
    python eval_adapter_precond.py --quick --phase adapter
    python eval_adapter_precond.py --quick --phase all
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
import torch.nn as nn
from gluonts.time_feature import get_seasonality
from scipy.stats import gmean

from gift_eval.data import Dataset
from uni2ts.common.precondition import compute_polynomial_coefficients
from eval_gifteval import GIFTEVAL_CONFIGS, QUICK_CONFIGS

# Reuse utilities from existing script
from eval_inference_precond import (
    precondition_series,
    reverse_with_raw_anchor,
    load_model,
    run_single_pass,
    QUANTILE_LEVELS,
    MEDIAN_IDX,
)


# ---------------------------------------------------------------------------
# 1. Collect per-series forecasts + labels
# ---------------------------------------------------------------------------

def collect_all_forecasts(
    module,
    model_type: str,
    configs: List[Tuple[str, str]],
    degrees: List[int],
    context_length: int = 1000,
    device: str = "cuda",
    patch_size: int = 32,
    batch_size: int = 32,
) -> Dict:
    """Run raw + A1 preconditioned passes for all datasets, return per-series data.

    Returns dict with keys:
        'datasets': list of dataset info dicts
        'forecasts': dict mapping method_name -> list of per-series forecast arrays
        'labels': list of per-series label arrays
        'past_data': list of per-series past data arrays
        'seasonalities': list of seasonality values per series
        'dataset_indices': list of dataset index per series (for LOOCV)
    """
    all_forecasts = {}  # method_name -> list of (Q, H, 1) arrays
    all_labels = []
    all_past_data = []
    all_seasonalities = []
    all_dataset_indices = []
    dataset_info = []

    for ds_idx, (dataset_name, term) in enumerate(configs):
        print(f"\n[{ds_idx+1}/{len(configs)}] {dataset_name}/{term}")

        try:
            dataset_check = Dataset(name=dataset_name, term=term, to_univariate=False)
            is_multivariate = dataset_check.target_dim > 1
            dataset = Dataset(name=dataset_name, term=term, to_univariate=is_multivariate)
            prediction_length = dataset.prediction_length

            try:
                freq = dataset.freq
                seasonality = get_seasonality(freq)
            except Exception:
                seasonality = 1

            # Collect test data
            past_targets = []
            labels = []
            for inp, lab in dataset.test_data:
                target = np.asarray(inp["target"]).flatten()
                if len(target) > context_length:
                    target = target[-context_length:]
                past_targets.append(target[:, np.newaxis].astype(np.float32))
                labels.append(np.asarray(lab["target"]).flatten())

            if len(past_targets) == 0:
                print(f"  SKIP: no test data")
                continue

            n_series = len(past_targets)
            dataset_info.append({
                "name": dataset_name, "term": term,
                "prediction_length": prediction_length,
                "n_series": n_series, "seasonality": seasonality,
            })

            # Build model wrapper
            if model_type == "v2":
                from uni2ts.model.moirai2 import Moirai2Forecast
                forecast_model = Moirai2Forecast(
                    prediction_length=prediction_length,
                    target_dim=1, feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                    context_length=context_length, module=module,
                )
            else:
                forecast_model = module

            # Raw pass
            print(f"    Raw pass ({n_series} series, H={prediction_length})...")
            Q_raw = run_single_pass(
                forecast_model, model_type, past_targets,
                prediction_length, context_length, device,
                patch_size=patch_size, batch_size=batch_size,
            )

            # Initialize method forecasts
            if "raw" not in all_forecasts:
                all_forecasts["raw"] = []
            for i in range(n_series):
                all_forecasts["raw"].append(Q_raw[i])

            # Preconditioned passes
            for d in degrees:
                method_name = f"A1_d{d}"
                if method_name not in all_forecasts:
                    all_forecasts[method_name] = []

                coeffs = compute_polynomial_coefficients("chebyshev", d)
                print(f"    A1 d={d}: precondition -> model -> reverse...")
                Q_precond = run_single_pass(
                    forecast_model, model_type, past_targets,
                    prediction_length, context_length, device,
                    precond_coeffs=coeffs,
                    patch_size=patch_size, batch_size=batch_size,
                )

                for i in range(n_series):
                    raw_series = past_targets[i].flatten()
                    tail = raw_series[-d:] if d <= len(raw_series) else np.zeros(d)
                    Q_reversed_i = np.zeros_like(Q_precond[i])
                    for q in range(Q_precond.shape[1]):
                        Q_reversed_i[q, :, 0] = reverse_with_raw_anchor(
                            Q_precond[i, q, :, 0],
                            Q_raw[i, q, :, 0],
                            tail, coeffs,
                        )
                    all_forecasts[method_name].append(Q_reversed_i)

            # Collect labels, past data, metadata
            for i in range(n_series):
                all_labels.append(labels[i])
                all_past_data.append(past_targets[i].flatten())
                all_seasonalities.append(seasonality)
                all_dataset_indices.append(ds_idx)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")
            continue
        finally:
            torch.cuda.empty_cache()

    return {
        "datasets": dataset_info,
        "forecasts": all_forecasts,
        "labels": all_labels,
        "past_data": all_past_data,
        "seasonalities": all_seasonalities,
        "dataset_indices": all_dataset_indices,
    }


# ---------------------------------------------------------------------------
# 2. Per-series MASE computation
# ---------------------------------------------------------------------------

def compute_per_series_mase(
    forecast: np.ndarray,  # (Q, H, 1)
    label: np.ndarray,     # (H,)
    past_data: np.ndarray, # (T,)
    seasonality: int,
    quantile_idx: int = MEDIAN_IDX,
) -> float:
    """Compute MASE for a single series."""
    pred = forecast[quantile_idx, :, 0]
    label = label.flatten()
    H = min(len(pred), len(label))
    pred = pred[:H]
    label = label[:H]
    ae = np.abs(pred - label)

    past = past_data.flatten()
    past_valid = past[~np.isnan(past)]
    if len(past_valid) > seasonality and seasonality > 0:
        naive_errors = np.abs(past_valid[seasonality:] - past_valid[:-seasonality])
        scale = np.mean(naive_errors)
    else:
        scale = np.mean(np.abs(np.diff(past_valid))) if len(past_valid) > 1 else 1.0

    if scale < 1e-10:
        scale = 1.0

    return float(np.mean(ae) / scale)


# ---------------------------------------------------------------------------
# 3. Oracle analysis
# ---------------------------------------------------------------------------

def oracle_analysis(data: Dict) -> pd.DataFrame:
    """For each series, find the best single forecast method. Report statistics."""
    methods = list(data["forecasts"].keys())
    n_series = len(data["labels"])

    print(f"\n{'='*80}")
    print("ORACLE ANALYSIS: Per-Series Best Method")
    print(f"{'='*80}")
    print(f"Total series: {n_series}")
    print(f"Methods: {methods}")

    # Compute per-series MASE for each method
    per_series_mase = {m: [] for m in methods}
    for i in range(n_series):
        for m in methods:
            mase = compute_per_series_mase(
                data["forecasts"][m][i],
                data["labels"][i],
                data["past_data"][i],
                data["seasonalities"][i],
            )
            per_series_mase[m].append(mase)

    # Convert to numpy for easy manipulation
    mase_matrix = np.array([per_series_mase[m] for m in methods])  # (K, N)

    # Oracle: best method per series
    oracle_idx = np.argmin(mase_matrix, axis=0)  # (N,)
    oracle_mase = mase_matrix[oracle_idx, np.arange(n_series)]

    # Always-raw baseline
    raw_idx = methods.index("raw")
    raw_mase_all = mase_matrix[raw_idx]

    # Statistics
    raw_mean = np.mean(raw_mase_all)
    oracle_mean = np.mean(oracle_mase)
    oracle_improvement = (raw_mean - oracle_mean) / raw_mean * 100

    print(f"\nPer-series MASE (mean across all series):")
    for m_idx, m in enumerate(methods):
        m_mean = np.mean(mase_matrix[m_idx])
        wins = np.sum(oracle_idx == m_idx)
        pct_wins = wins / n_series * 100
        print(f"  {m:<15} mean={m_mean:.4f}  chosen by oracle: {wins}/{n_series} ({pct_wins:.1f}%)")

    print(f"\n  {'raw (always)':<15} mean={raw_mean:.4f}")
    print(f"  {'oracle':<15} mean={oracle_mean:.4f}  improvement: {oracle_improvement:.2f}%")

    # How often does raw win?
    raw_wins = np.sum(oracle_idx == raw_idx)
    print(f"\n  Raw is best for {raw_wins}/{n_series} series ({raw_wins/n_series*100:.1f}%)")
    print(f"  Some precond is better for {n_series - raw_wins}/{n_series} series ({(n_series-raw_wins)/n_series*100:.1f}%)")

    # Improvement distribution when precond wins
    precond_wins_mask = oracle_idx != raw_idx
    if precond_wins_mask.any():
        improvements = (raw_mase_all[precond_wins_mask] - oracle_mase[precond_wins_mask]) / raw_mase_all[precond_wins_mask] * 100
        print(f"\n  When precond wins:")
        print(f"    Mean improvement: {np.mean(improvements):.2f}%")
        print(f"    Median improvement: {np.median(improvements):.2f}%")
        print(f"    Max improvement: {np.max(improvements):.2f}%")
        print(f"    Min improvement: {np.min(improvements):.2f}%")

    # Per-dataset breakdown
    print(f"\n  Per-dataset oracle analysis:")
    ds_indices = np.array(data["dataset_indices"])
    for ds_idx, ds_info in enumerate(data["datasets"]):
        mask = ds_indices == ds_idx
        if not mask.any():
            continue
        ds_raw_mean = np.mean(raw_mase_all[mask])
        ds_oracle_mean = np.mean(oracle_mase[mask])
        ds_improv = (ds_raw_mean - ds_oracle_mean) / ds_raw_mean * 100
        ds_raw_wins = np.sum(oracle_idx[mask] == raw_idx)
        ds_total = mask.sum()
        print(f"    {ds_info['name']:<25} raw={ds_raw_mean:.4f} oracle={ds_oracle_mean:.4f} "
              f"improv={ds_improv:+.2f}%  raw_wins={ds_raw_wins}/{ds_total}")

    # Build results dataframe
    rows = []
    for i in range(n_series):
        row = {
            "series_idx": i,
            "dataset_idx": data["dataset_indices"][i],
            "dataset": data["datasets"][data["dataset_indices"][i]]["name"],
            "seasonality": data["seasonalities"][i],
            "oracle_method": methods[oracle_idx[i]],
        }
        for m in methods:
            row[f"{m}_MASE"] = per_series_mase[m][i]
        row["oracle_MASE"] = oracle_mase[i]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Learned adapter: global linear weights
# ---------------------------------------------------------------------------

class LinearAdapter(nn.Module):
    """Learn global softmax weights over K forecasts.
    Q_combined = sum_k softmax(w)[k] * Q_k
    """
    def __init__(self, n_methods: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_methods))

    def forward(self, forecasts_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            forecasts_stack: (batch, K, Q, H) — K forecast methods
        Returns:
            combined: (batch, Q, H)
        """
        weights = torch.softmax(self.logits, dim=0)  # (K,)
        return (forecasts_stack * weights[None, :, None, None]).sum(dim=1)


# ---------------------------------------------------------------------------
# 5. Feature-conditioned adapter
# ---------------------------------------------------------------------------

def compute_series_features(past_data: np.ndarray, seasonality: int) -> np.ndarray:
    """Compute features of input series for conditioning the adapter."""
    x = past_data.flatten()
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.zeros(8, dtype=np.float32)

    features = []
    # 1. Length (log)
    features.append(np.log(len(x) + 1))
    # 2. Mean
    features.append(np.mean(x))
    # 3. Std
    features.append(np.std(x) + 1e-8)
    # 4. Coefficient of variation
    features.append(np.std(x) / (np.abs(np.mean(x)) + 1e-8))
    # 5. Autocorrelation at lag 1
    if len(x) > 1:
        x_centered = x - np.mean(x)
        ac1 = np.correlate(x_centered[:-1], x_centered[1:])[0] / (np.sum(x_centered**2) + 1e-8)
        features.append(ac1)
    else:
        features.append(0.0)
    # 6. Autocorrelation at seasonal lag
    if len(x) > seasonality and seasonality > 0:
        x_centered = x - np.mean(x)
        ac_s = np.correlate(x_centered[:-seasonality], x_centered[seasonality:])[0] / (np.sum(x_centered**2) + 1e-8)
        features.append(ac_s)
    else:
        features.append(0.0)
    # 7. Trend strength (linear regression slope / std)
    t = np.arange(len(x), dtype=np.float32)
    if np.std(t) > 0 and np.std(x) > 0:
        slope = np.polyfit(t, x, 1)[0]
        features.append(slope / (np.std(x) + 1e-8))
    else:
        features.append(0.0)
    # 8. Spectral entropy (rough proxy for complexity)
    if len(x) > 4:
        fft_vals = np.abs(np.fft.rfft(x - np.mean(x)))
        fft_vals = fft_vals / (fft_vals.sum() + 1e-8)
        fft_vals = fft_vals[fft_vals > 0]
        entropy = -np.sum(fft_vals * np.log(fft_vals + 1e-8))
        features.append(entropy)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


class FeatureConditionedAdapter(nn.Module):
    """MLP that maps series features -> combination weights."""
    def __init__(self, n_features: int, n_methods: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_methods),
        )

    def forward(self, features: torch.Tensor, forecasts_stack: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, n_features)
            forecasts_stack: (batch, K, Q, H)
        Returns:
            combined: (batch, Q, H)
        """
        logits = self.net(features)  # (batch, K)
        weights = torch.softmax(logits, dim=-1)  # (batch, K)
        return (forecasts_stack * weights[:, :, None, None]).sum(dim=1)


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------

def train_adapter(
    forecasts_stack: np.ndarray,  # (N, K, Q, H)
    labels: np.ndarray,           # (N, H)
    past_data_list: List[np.ndarray],
    seasonalities: np.ndarray,    # (N,)
    dataset_indices: np.ndarray,  # (N,)
    adapter_type: str = "linear",  # "linear" or "mlp"
    n_epochs: int = 200,
    lr: float = 0.01,
) -> Dict:
    """Train adapter with leave-one-dataset-out cross-validation.

    Uses pinball loss at the median (equivalent to MAE) as training objective,
    then evaluates MASE.

    Returns dict with per-fold and aggregate results.
    """
    N, K, Q, H = forecasts_stack.shape
    unique_ds = np.unique(dataset_indices)
    n_features = 8  # from compute_series_features

    results = {"folds": [], "adapter_type": adapter_type}

    for val_ds in unique_ds:
        train_mask = dataset_indices != val_ds
        val_mask = dataset_indices == val_ds

        train_forecasts = torch.tensor(forecasts_stack[train_mask], dtype=torch.float32)
        val_forecasts = torch.tensor(forecasts_stack[val_mask], dtype=torch.float32)
        train_labels = torch.tensor(labels[train_mask], dtype=torch.float32)
        val_labels = torch.tensor(labels[val_mask], dtype=torch.float32)

        if adapter_type == "mlp":
            train_features = torch.tensor(
                np.array([compute_series_features(past_data_list[i], int(seasonalities[i]))
                          for i in np.where(train_mask)[0]]),
                dtype=torch.float32,
            )
            val_features = torch.tensor(
                np.array([compute_series_features(past_data_list[i], int(seasonalities[i]))
                          for i in np.where(val_mask)[0]]),
                dtype=torch.float32,
            )
            # Normalize features
            feat_mean = train_features.mean(dim=0)
            feat_std = train_features.std(dim=0) + 1e-8
            train_features = (train_features - feat_mean) / feat_std
            val_features = (val_features - feat_mean) / feat_std

            model = FeatureConditionedAdapter(n_features, K, hidden_dim=16)
        else:
            train_features = val_features = None
            model = LinearAdapter(K)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train on median MAE (pinball loss at q=0.5 is MAE/2)
        best_val_loss = float("inf")
        best_state = None
        patience = 30
        no_improve = 0

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            if adapter_type == "mlp":
                combined = model(train_features, train_forecasts)
            else:
                combined = model(train_forecasts)

            # MAE at median quantile
            pred_median = combined[:, MEDIAN_IDX, :]  # (N_train, H)
            loss = torch.mean(torch.abs(pred_median - train_labels))
            loss.backward()
            optimizer.step()

            # Validation loss
            model.eval()
            with torch.no_grad():
                if adapter_type == "mlp":
                    val_combined = model(val_features, val_forecasts)
                else:
                    val_combined = model(val_forecasts)
                val_pred_median = val_combined[:, MEDIAN_IDX, :]
                val_loss = torch.mean(torch.abs(val_pred_median - val_labels)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Load best model and evaluate
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            if adapter_type == "mlp":
                val_combined = model(val_features, val_forecasts)
            else:
                val_combined = model(val_forecasts)
            val_combined_np = val_combined.numpy()

        # Compute MASE for adapter and raw baseline on val set
        val_indices = np.where(val_mask)[0]
        adapter_mases = []
        raw_mases = []
        for j, idx in enumerate(val_indices):
            # Adapter MASE
            adapter_forecast = val_combined_np[j]  # (Q, H)
            adapter_mase = compute_per_series_mase(
                adapter_forecast[:, :, np.newaxis],
                labels[idx],
                past_data_list[idx],
                int(seasonalities[idx]),
            )
            adapter_mases.append(adapter_mase)

            # Raw MASE
            raw_forecast = forecasts_stack[idx, 0]  # (Q, H) — raw is index 0
            raw_mase = compute_per_series_mase(
                raw_forecast[:, :, np.newaxis],
                labels[idx],
                past_data_list[idx],
                int(seasonalities[idx]),
            )
            raw_mases.append(raw_mase)

        # Get learned weights
        if adapter_type == "linear":
            weights = torch.softmax(model.logits, dim=0).detach().numpy()
        else:
            weights = None

        fold_result = {
            "val_dataset": int(val_ds),
            "adapter_mean_mase": float(np.mean(adapter_mases)),
            "raw_mean_mase": float(np.mean(raw_mases)),
            "n_val_series": len(val_indices),
            "learned_weights": weights.tolist() if weights is not None else None,
            "epochs_trained": epoch + 1,
        }
        results["folds"].append(fold_result)

    return results


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Learned adapter for inference-time preconditioning ensemble")
    parser.add_argument("--model", type=str, default="moirai-1.1-R-small")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--degrees", type=str, default="2,4,6",
                        help="Chebyshev degrees (fewer for speed)")
    parser.add_argument("--context-length", type=int, default=1000)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["oracle", "adapter", "all"])
    parser.add_argument("--adapter-type", type=str, default="both",
                        choices=["linear", "mlp", "both"])
    parser.add_argument("--output-dir", type=str,
                        default="/scratch/gpfs/EHAZAN/jh1161/gifteval/results")
    parser.add_argument("--load-forecasts", type=str, default=None,
                        help="Load pre-computed forecasts from .npz file instead of running model")
    args = parser.parse_args()

    degrees = [int(d) for d in args.degrees.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = QUICK_CONFIGS if args.quick else GIFTEVAL_CONFIGS

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Degrees: {degrees}")
    print(f"Phase: {args.phase}")
    print(f"Configs: {len(configs)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Collect forecasts (or load from disk)
    if args.load_forecasts:
        print(f"\nLoading pre-computed forecasts from {args.load_forecasts}...")
        import pickle
        with open(args.load_forecasts, "rb") as f:
            data = pickle.load(f)
    else:
        print(f"\nLoading model {args.model}...")
        model, model_type = load_model(args.model, prediction_length=1,
                                        context_length=args.context_length)
        print(f"Model type: {model_type}")
        if model_type == "v1":
            model = model.to(device)

        data = collect_all_forecasts(
            model, model_type, configs, degrees,
            context_length=args.context_length, device=device,
            patch_size=args.patch_size, batch_size=args.batch_size,
        )

        # Save forecasts to disk for reuse (pickle handles variable-length arrays)
        import pickle
        forecast_path = output_dir / f"adapter_forecasts_{args.model}_{timestamp}.pkl"
        with open(forecast_path, "wb") as f:
            pickle.dump(data, f)
        print(f"\nForecasts saved to: {forecast_path}")

    # Step 2: Oracle analysis
    if args.phase in ("oracle", "all"):
        oracle_df = oracle_analysis(data)
        oracle_path = output_dir / f"adapter_oracle_{args.model}_{timestamp}.csv"
        oracle_df.to_csv(oracle_path, index=False)
        print(f"\nOracle results saved to: {oracle_path}")

    # Step 3: Learned adapter
    if args.phase in ("adapter", "all"):
        methods = list(data["forecasts"].keys())
        n_series = len(data["labels"])

        # Build forecasts_stack: (N, K, Q, H)
        # Need to align H across series (different datasets have different H)
        # Group by dataset for training
        print(f"\n{'='*80}")
        print("LEARNED ADAPTER TRAINING")
        print(f"{'='*80}")
        print(f"Methods: {methods}")
        print(f"Total series: {n_series}")

        # Since different datasets have different H, we train per-horizon-group
        # or pad to max H. Simpler: operate on each dataset group independently
        # and aggregate results.

        dataset_indices = np.array(data["dataset_indices"])
        seasonalities = np.array(data["seasonalities"])
        unique_ds = np.unique(dataset_indices)

        # Group series by prediction length for batched training
        def _build_H_groups(indices, data, methods):
            """Group series indices by prediction length for efficient batching."""
            groups = {}
            for idx in indices:
                H_i = len(data["labels"][idx])
                if H_i not in groups:
                    groups[H_i] = []
                groups[H_i].append(idx)
            return groups

        def _build_batched_tensors(indices, data, methods):
            """Pre-build tensors grouped by H for fast training."""
            groups = _build_H_groups(indices, data, methods)
            K = len(methods)
            batched = {}
            for H_i, group_indices in groups.items():
                n = len(group_indices)
                fc_np = np.zeros((n, K, len(QUANTILE_LEVELS), H_i), dtype=np.float32)
                lab_np = np.zeros((n, H_i), dtype=np.float32)
                for j, idx in enumerate(group_indices):
                    for k, m in enumerate(methods):
                        fc_np[j, k] = data["forecasts"][m][idx][:, :H_i, 0]
                    lab_np[j] = data["labels"][idx][:H_i]
                batched[H_i] = {
                    "forecasts": torch.tensor(fc_np),
                    "labels": torch.tensor(lab_np),
                    "indices": group_indices,
                }
            return batched

        adapter_types = [args.adapter_type] if args.adapter_type != "both" else ["linear", "mlp"]

        # Pre-compute features for MLP adapter
        all_features = None
        if "mlp" in adapter_types:
            print("  Computing series features...")
            all_features = np.array([
                compute_series_features(data["past_data"][i], int(seasonalities[i]))
                for i in range(n_series)
            ], dtype=np.float32)

        for adapter_type in adapter_types:
            print(f"\n--- Adapter type: {adapter_type} ---")

            all_fold_adapter_mases = []
            all_fold_raw_mases = []
            fold_results = []

            for val_ds_idx in unique_ds:
                val_mask = dataset_indices == val_ds_idx
                train_mask = ~val_mask
                val_ds_name = data["datasets"][int(val_ds_idx)]["name"]
                train_indices = np.where(train_mask)[0]
                val_indices = np.where(val_mask)[0]
                K = len(methods)

                # Pre-build batched tensors for train and val
                train_batches = _build_batched_tensors(train_indices, data, methods)
                val_batches = _build_batched_tensors(val_indices, data, methods)

                if adapter_type == "linear":
                    model_adapter = LinearAdapter(K)
                else:
                    model_adapter = FeatureConditionedAdapter(8, K, hidden_dim=16)
                    feat_mean = all_features[train_mask].mean(axis=0)
                    feat_std = all_features[train_mask].std(axis=0) + 1e-8

                optimizer = torch.optim.Adam(model_adapter.parameters(), lr=0.01)

                best_val_loss = float("inf")
                best_state = None
                patience = 30
                no_improve = 0

                for epoch in range(200):
                    model_adapter.train()
                    optimizer.zero_grad()
                    total_loss = 0.0
                    total_groups = 0

                    for H_i, batch in train_batches.items():
                        fc = batch["forecasts"]    # (n, K, Q, H_i)
                        lab = batch["labels"]      # (n, H_i)

                        if adapter_type == "mlp":
                            feat_idx = batch["indices"]
                            feat_raw = all_features[feat_idx]
                            feat_t = torch.tensor((feat_raw - feat_mean) / feat_std, dtype=torch.float32)
                            combined = model_adapter(feat_t, fc)
                        else:
                            combined = model_adapter(fc)

                        pred_median = combined[:, MEDIAN_IDX, :]  # (n, H_i)
                        loss = torch.mean(torch.abs(pred_median - lab))
                        total_loss += loss * len(batch["indices"])
                        total_groups += len(batch["indices"])

                    avg_loss = total_loss / total_groups
                    avg_loss.backward()
                    optimizer.step()

                    # Validation
                    model_adapter.eval()
                    with torch.no_grad():
                        val_total = 0.0
                        val_count = 0
                        for H_i, batch in val_batches.items():
                            fc = batch["forecasts"]
                            lab = batch["labels"]
                            if adapter_type == "mlp":
                                feat_idx = batch["indices"]
                                feat_raw = all_features[feat_idx]
                                feat_t = torch.tensor((feat_raw - feat_mean) / feat_std, dtype=torch.float32)
                                val_combined = model_adapter(feat_t, fc)
                            else:
                                val_combined = model_adapter(fc)
                            val_pred = val_combined[:, MEDIAN_IDX, :]
                            val_total += torch.mean(torch.abs(val_pred - lab)).item() * len(batch["indices"])
                            val_count += len(batch["indices"])
                        avg_val_loss = val_total / val_count

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_state = {k: v.clone() for k, v in model_adapter.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            break

                # Load best and evaluate MASE
                if best_state is not None:
                    model_adapter.load_state_dict(best_state)
                model_adapter.eval()

                adapter_mases = []
                raw_mases = []
                with torch.no_grad():
                    for H_i, batch in val_batches.items():
                        fc = batch["forecasts"]
                        if adapter_type == "mlp":
                            feat_idx = batch["indices"]
                            feat_raw = all_features[feat_idx]
                            feat_t = torch.tensor((feat_raw - feat_mean) / feat_std, dtype=torch.float32)
                            combined = model_adapter(feat_t, fc)
                        else:
                            combined = model_adapter(fc)
                        combined_np = combined.numpy()

                        for j, idx in enumerate(batch["indices"]):
                            combined_j = combined_np[j][:, :, np.newaxis]  # (Q, H, 1)
                            adapter_mase = compute_per_series_mase(
                                combined_j, data["labels"][idx],
                                data["past_data"][idx], int(seasonalities[idx]),
                            )
                            adapter_mases.append(adapter_mase)

                            raw_j = data["forecasts"]["raw"][idx][:, :H_i, :]
                            raw_mase = compute_per_series_mase(
                                raw_j, data["labels"][idx],
                                data["past_data"][idx], int(seasonalities[idx]),
                            )
                            raw_mases.append(raw_mase)

                all_fold_adapter_mases.extend(adapter_mases)
                all_fold_raw_mases.extend(raw_mases)

                adapter_mean = np.mean(adapter_mases)
                raw_mean = np.mean(raw_mases)
                improv = (raw_mean - adapter_mean) / raw_mean * 100

                if adapter_type == "linear":
                    weights = torch.softmax(model_adapter.logits, dim=0).detach().numpy()
                    weights_str = ", ".join([f"{methods[k]}:{w:.3f}" for k, w in enumerate(weights)])
                else:
                    weights_str = "(feature-conditioned)"

                print(f"  Fold val={val_ds_name:<25} adapter={adapter_mean:.4f} raw={raw_mean:.4f} "
                      f"improv={improv:+.2f}%  epochs={epoch+1}  weights=[{weights_str}]")

                fold_results.append({
                    "val_dataset": val_ds_name,
                    "adapter_mase": adapter_mean,
                    "raw_mase": raw_mean,
                    "improvement_pct": improv,
                })

            # Aggregate results
            agg_adapter = np.mean(all_fold_adapter_mases)
            agg_raw = np.mean(all_fold_raw_mases)
            agg_improv = (agg_raw - agg_adapter) / agg_raw * 100

            print(f"\n  AGGREGATE ({adapter_type}): adapter={agg_adapter:.4f} raw={agg_raw:.4f} "
                  f"improvement={agg_improv:+.2f}%")

            # Save fold results
            fold_df = pd.DataFrame(fold_results)
            fold_path = output_dir / f"adapter_{adapter_type}_{args.model}_{timestamp}.csv"
            fold_df.to_csv(fold_path, index=False)
            print(f"  Results saved to: {fold_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
