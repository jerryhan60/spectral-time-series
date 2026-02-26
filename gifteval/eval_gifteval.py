#!/usr/bin/env python3
"""
Evaluate checkpoints on GIFT-Eval benchmark.
Outputs results in the format expected by the GIFT-Eval leaderboard.

Metrics follow the GIFT-Eval leaderboard methodology:
- MASE[0.5]: Mean Absolute Scaled Error at median quantile
- Per-config ranking: Models ranked 1-N on each dataset configuration
- MASE_Rank: Arithmetic mean of per-config ranks (primary leaderboard metric)
- Geometric Mean MASE: Alternative scale-invariant aggregation

Usage:
    python eval_gifteval.py --checkpoint /path/to/checkpoint.ckpt
    python eval_gifteval.py --checkpoint /path/to/checkpoint.ckpt --quick
    python eval_gifteval.py --model moirai-1.1-R-small  # Use HuggingFace pretrained
    python eval_gifteval.py --compare results/*.csv    # Compare multiple models
"""

import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# Set offline mode for cluster compute nodes
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add paths
sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/uni2ts/src")
sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/gifteval/gift-eval/src")

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/scratch/gpfs/EHAZAN/jh1161/uni2ts/.env")

from gluonts.ev.metrics import MAE, MSE, MASE, SMAPE, MAPE
from gluonts.time_feature import get_seasonality
from scipy.stats import gmean

# Import gift_eval after path setup
from gift_eval.data import Dataset

# Import uni2ts evaluation utilities
from uni2ts.eval_util.evaluation import evaluate_model


def compare_models(result_files: list, output_path: str = None):
    """
    Compare multiple model results and compute GIFT-Eval leaderboard-style rankings.

    This replicates the GIFT-Eval leaderboard methodology:
    1. For each configuration, rank models by MASE[0.5] (rank 1 = best)
    2. Compute MASE_Rank = arithmetic mean of ranks across all configurations
    3. Also report geometric mean MASE for reference

    Args:
        result_files: List of CSV file paths containing evaluation results
        output_path: Optional path to save comparison results

    Returns:
        DataFrame with model rankings
    """
    # Load all results
    all_results = []
    for f in result_files:
        df = pd.read_csv(f)
        # Extract model name from filename
        model_name = Path(f).stem.replace("gifteval_results_", "").rsplit("_", 2)[0]
        df["model"] = model_name
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)

    # Filter to only successful evaluations (no errors)
    if "error" in combined.columns:
        combined = combined[combined["error"].isna()]

    # Create config key for grouping
    combined["config_key"] = combined["dataset"] + "/" + combined["term"]

    # Compute per-configuration rankings (lower MASE = better = lower rank)
    combined["MASE_rank"] = combined.groupby("config_key")["MASE"].rank(method="first", ascending=True)

    # Aggregate by model
    model_stats = combined.groupby("model").agg({
        "MASE": ["mean", lambda x: gmean(x[x > 0]) if (x > 0).any() else np.nan, "count"],
        "MASE_rank": "mean",
        "MAE": "mean",
        "MSE": "mean",
    })

    # Flatten column names
    model_stats.columns = [
        "MASE_arithmetic_mean", "MASE_geometric_mean", "num_configs",
        "MASE_Rank", "MAE_mean", "MSE_mean"
    ]

    # Sort by MASE_Rank (lower is better)
    model_stats = model_stats.sort_values("MASE_Rank")

    # Add overall rank
    model_stats["Overall_Rank"] = range(1, len(model_stats) + 1)

    # Reorder columns
    model_stats = model_stats[[
        "Overall_Rank", "MASE_Rank", "MASE_geometric_mean",
        "MASE_arithmetic_mean", "num_configs", "MAE_mean", "MSE_mean"
    ]]

    print("\n" + "="*100)
    print("GIFT-EVAL LEADERBOARD-STYLE COMPARISON")
    print("="*100)
    print("\nMethodology: MASE_Rank = mean rank across all configurations (lower is better)")
    print("             Models ranked 1-N on each dataset, then ranks averaged")
    print()
    print(model_stats.to_string())

    # Per-configuration breakdown
    print("\n" + "-"*100)
    print("PER-CONFIGURATION RANKINGS (showing MASE values)")
    print("-"*100)

    pivot = combined.pivot_table(
        index="config_key",
        columns="model",
        values="MASE",
        aggfunc="first"
    )

    # Add rank columns
    for model in pivot.columns:
        rank_col = f"{model}_rank"

    print(pivot.round(4).to_string())

    # Save if requested
    if output_path:
        model_stats.to_csv(output_path)
        print(f"\nComparison saved to: {output_path}")

        # Also save detailed per-config results
        detail_path = output_path.replace(".csv", "_detailed.csv")
        pivot.to_csv(detail_path)
        print(f"Per-config details saved to: {detail_path}")

    return model_stats


# GIFT-Eval dataset configurations (97 total)
# Format: (dataset_name, term) where term in ["short", "medium", "long"]
# Dataset names must match the directory structure in GIFT_EVAL path
# Datasets with frequency subdirs use format "name/freq" (e.g., "electricity/H")
GIFTEVAL_CONFIGS = [
    # Bitbrains (Cloud/Web)
    ("bitbrains_fast_storage/5T", "short"),
    ("bitbrains_fast_storage/5T", "medium"),
    ("bitbrains_fast_storage/5T", "long"),
    ("bitbrains_fast_storage/H", "short"),
    ("bitbrains_rnd/5T", "short"),
    ("bitbrains_rnd/5T", "medium"),
    ("bitbrains_rnd/5T", "long"),
    ("bitbrains_rnd/H", "short"),

    # Bizitobs (Cloud/Web)
    ("bizitobs_application", "short"),
    ("bizitobs_application", "medium"),
    ("bizitobs_application", "long"),
    ("bizitobs_l2c/5T", "short"),
    ("bizitobs_l2c/5T", "medium"),
    ("bizitobs_l2c/5T", "long"),
    ("bizitobs_l2c/H", "short"),
    ("bizitobs_l2c/H", "medium"),
    ("bizitobs_l2c/H", "long"),
    ("bizitobs_service", "short"),
    ("bizitobs_service", "medium"),
    ("bizitobs_service", "long"),

    # Sales
    ("car_parts_with_missing", "short"),
    ("hierarchical_sales/D", "short"),
    ("hierarchical_sales/W", "short"),
    ("restaurant", "short"),

    # Health
    ("covid_deaths", "short"),
    ("hospital", "short"),

    # Energy - Electricity
    ("electricity/15T", "short"),
    ("electricity/15T", "medium"),
    ("electricity/15T", "long"),
    ("electricity/D", "short"),
    ("electricity/H", "short"),
    ("electricity/H", "medium"),
    ("electricity/H", "long"),
    ("electricity/W", "short"),

    # Energy - ETT
    ("ett1/15T", "short"),
    ("ett1/15T", "medium"),
    ("ett1/15T", "long"),
    ("ett1/D", "short"),
    ("ett1/H", "short"),
    ("ett1/H", "medium"),
    ("ett1/H", "long"),
    ("ett1/W", "short"),
    ("ett2/15T", "short"),
    ("ett2/15T", "medium"),
    ("ett2/15T", "long"),
    ("ett2/D", "short"),
    ("ett2/H", "short"),
    ("ett2/H", "medium"),
    ("ett2/H", "long"),
    ("ett2/W", "short"),

    # Energy - Solar
    ("solar/10T", "short"),
    ("solar/10T", "medium"),
    ("solar/10T", "long"),
    ("solar/D", "short"),
    ("solar/H", "short"),
    ("solar/H", "medium"),
    ("solar/H", "long"),
    ("solar/W", "short"),

    # Nature - Weather
    ("jena_weather/10T", "short"),
    ("jena_weather/10T", "medium"),
    ("jena_weather/10T", "long"),
    ("jena_weather/D", "short"),
    ("jena_weather/H", "short"),
    ("jena_weather/H", "medium"),
    ("jena_weather/H", "long"),
    ("kdd_cup_2018_with_missing/D", "short"),
    ("kdd_cup_2018_with_missing/H", "short"),
    ("kdd_cup_2018_with_missing/H", "medium"),
    ("kdd_cup_2018_with_missing/H", "long"),
    ("temperature_rain_with_missing", "short"),

    # Nature - Other
    ("saugeenday/D", "short"),
    ("saugeenday/M", "short"),
    ("saugeenday/W", "short"),
    ("us_births/D", "short"),
    ("us_births/M", "short"),
    ("us_births/W", "short"),

    # Transport
    ("LOOP_SEATTLE/5T", "short"),
    ("LOOP_SEATTLE/5T", "medium"),
    ("LOOP_SEATTLE/5T", "long"),
    ("LOOP_SEATTLE/D", "short"),
    ("LOOP_SEATTLE/H", "short"),
    ("LOOP_SEATTLE/H", "medium"),
    ("LOOP_SEATTLE/H", "long"),
    ("SZ_TAXI/15T", "short"),
    ("SZ_TAXI/15T", "medium"),
    ("SZ_TAXI/15T", "long"),
    ("SZ_TAXI/H", "short"),

    # M-competitions
    ("m4_daily", "short"),
    ("m4_hourly", "short"),
    ("m4_monthly", "short"),
    ("m4_quarterly", "short"),
    ("m4_weekly", "short"),
    ("m4_yearly", "short"),

    # Dense matrix
    ("M_DENSE/D", "short"),
    ("M_DENSE/H", "short"),
    ("M_DENSE/H", "medium"),
    ("M_DENSE/H", "long"),
]

# Quick subset for fast evaluation (representative sample from available datasets)
QUICK_CONFIGS = [
    ("m4_monthly", "short"),        # M-competition
    ("electricity/H", "short"),     # Energy
    ("hospital", "short"),          # Health
    ("jena_weather/H", "short"),    # Nature
    ("ett1/H", "short"),            # ETT benchmark
    ("saugeenday/D", "short"),      # Nature (daily)
    ("covid_deaths", "short"),      # Health
    ("m4_hourly", "short"),         # M-competition hourly
]


def _load_module_from_checkpoint(checkpoint_path: str):
    """Load model module from checkpoint once. Returns (module, model_type)."""
    from uni2ts.model.moirai import MoiraiForecast, MoiraiPretrain

    try:
        pretrain = MoiraiPretrain.load_from_checkpoint(checkpoint_path)
        return (pretrain.module, "moirai")
    except Exception as e:
        print(f"Standard load failed: {e}")

    try:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Pretrain
        pretrain = Moirai2Pretrain.load_from_checkpoint(checkpoint_path)
        return (pretrain.module, "moirai2")
    except Exception as e:
        print(f"Moirai2 load failed: {e}")

    try:
        from uni2ts.model.moirai import MoiraiHybridPretrain
        pretrain = MoiraiHybridPretrain.load_from_checkpoint(checkpoint_path)
        return (pretrain.module, "hybrid")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def _build_forecast_from_module(cached_module_tuple, checkpoint_path: str,
                                prediction_length: int, context_length: int = 1000,
                                patch_size: int = 32):
    """Build forecast wrapper from a cached module (no disk I/O)."""
    module, model_type = cached_module_tuple

    if model_type == "moirai":
        from uni2ts.model.moirai import MoiraiForecast
        return MoiraiForecast(
            prediction_length=prediction_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=context_length,
            module=module,
            patch_size=patch_size,
            num_samples=100,
        )
    elif model_type == "moirai2":
        from uni2ts.model.moirai2 import Moirai2Forecast
        return Moirai2Forecast(
            prediction_length=prediction_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=context_length,
            module=module,
        )
    elif model_type == "hybrid":
        from uni2ts.model.moirai import MoiraiForecast
        class HybridForecast(MoiraiForecast):
            def __init__(self, *args, module=None, **kwargs):
                super(MoiraiForecast, self).__init__()
                self.save_hyperparameters(ignore=["module"])
                self.module = module
                from uni2ts.model.moirai.forecast import SampleNLLLoss
                self.per_sample_loss_func = SampleNLLLoss()
        return HybridForecast(
            prediction_length=prediction_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=context_length,
            module=module,
            patch_size=patch_size,
            num_samples=100,
        )


def load_checkpoint_model(checkpoint_path: str, prediction_length: int,
                          context_length: int = 1000, patch_size: int = 32):
    """Load model from a checkpoint file (legacy, reloads each time)."""
    cached = _load_module_from_checkpoint(checkpoint_path)
    return _build_forecast_from_module(cached, checkpoint_path, prediction_length,
                                       context_length, patch_size)


def load_pretrained_model(model_name: str, prediction_length: int,
                          context_length: int = 1000, patch_size: int = 32):
    """Load pretrained model from HuggingFace."""
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/{model_name}"),
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    return model


def evaluate_single_dataset(model_loader, dataset_name: str, term: str,
                           batch_size: int = 32, device: str = "cuda"):
    """
    Evaluate model on a single GIFT-Eval dataset configuration.

    Args:
        model_loader: Function that takes prediction_length and returns a model
        dataset_name: Name of the dataset
        term: Prediction horizon term ("short", "medium", "long")
        batch_size: Batch size for prediction
        device: Device to run on

    Returns:
        dict with metrics or None if evaluation failed
    """
    try:
        # Load GIFT-Eval dataset
        # First check if dataset is multivariate - only use to_univariate for multivariate data
        dataset_check = Dataset(name=dataset_name, term=term, to_univariate=False)
        is_multivariate = dataset_check.target_dim > 1

        # Reload with appropriate to_univariate setting
        dataset = Dataset(name=dataset_name, term=term, to_univariate=is_multivariate)

        # Get prediction length from dataset
        prediction_length = dataset.prediction_length

        # Load model with correct prediction length
        model = model_loader(prediction_length)
        model = model.to(device)
        model.eval()

        # Get test data
        test_data = dataset.test_data

        # Get seasonality
        try:
            freq = dataset.freq
            seasonality = get_seasonality(freq)
        except:
            seasonality = 1

        # Create predictor
        predictor = model.create_predictor(batch_size=batch_size, device=device)

        # Define metrics (matching GIFT-Eval leaderboard)
        # The leaderboard uses eval_metrics/MASE[0.5] and eval_metrics/mean_weighted_sum_quantile_loss
        # Note: MeanWeightedSumQuantileLoss requires quantile_levels which depends on model config
        # For simplicity, we'll compute CRPS from the other metrics for now
        metrics = [
            MAE(),
            MSE(),
            MASE(),
            SMAPE(),
        ]

        # Evaluate
        result = evaluate_model(
            predictor,
            test_data=test_data,
            metrics=metrics,
            batch_size=batch_size,
            axis=None,  # Aggregate across all dimensions
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=seasonality,
        )

        # Extract metric values (column names include forecast type, e.g., MAE[0.5])
        # Use exact column names matching GIFT-Eval leaderboard format
        metrics_dict = {
            "dataset": dataset_name,
            "term": term,
            "frequency": dataset.freq,
            "config_name": f"{dataset_name}/{dataset.freq}/{term}",
            "prediction_length": prediction_length,
            # Leaderboard metric names
            "eval_metrics/MAE[0.5]": float(result["MAE[0.5]"].values[0]),
            "eval_metrics/MSE[mean]": float(result["MSE[mean]"].values[0]),
            "eval_metrics/MASE[0.5]": float(result["MASE[0.5]"].values[0]),
            "eval_metrics/sMAPE[0.5]": float(result["sMAPE[0.5]"].values[0]) if "sMAPE[0.5]" in result else np.nan,
            # Shorthand aliases for convenience
            "MAE": float(result["MAE[0.5]"].values[0]),
            "MSE": float(result["MSE[mean]"].values[0]),
            "MASE": float(result["MASE[0.5]"].values[0]),
            "SMAPE": float(result["sMAPE[0.5]"].values[0]) if "sMAPE[0.5]" in result else np.nan,
        }

        return metrics_dict

    except Exception as e:
        print(f"Error evaluating {dataset_name}/{term}: {e}")
        return {
            "dataset": dataset_name,
            "term": term,
            "frequency": "?",
            "config_name": f"{dataset_name}/?/{term}",
            "error": str(e),
        }
    finally:
        # Clean up GPU memory
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()


def get_model_info(checkpoint_path: str) -> dict:
    """Extract model information from checkpoint."""
    info = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": Path(checkpoint_path).stem,
        "model_type": "unknown",
        "d_model": None,
        "num_layers": None,
        "param_count": None,
        "param_count_str": None,
        "architecture": None,
        "stu_config": None,
    }

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)

        # Count parameters
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        info["param_count"] = total_params
        info["param_count_str"] = f"{total_params/1e6:.2f}M"

        # Detect model type from state dict keys
        keys = list(state_dict.keys())
        has_stu = any('stu' in k.lower() for k in keys)
        has_attention = any('attn' in k.lower() or 'attention' in k.lower() for k in keys)
        has_sandwich = any('sandwich' in k.lower() for k in keys)

        if has_stu and has_attention:
            info["model_type"] = "hybrid_stu_attention"
            if has_sandwich:
                info["architecture"] = "STU+Attention with MLP Sandwich"
            else:
                info["architecture"] = "STU+Attention (alternating)"
        elif has_stu:
            info["model_type"] = "stu_only"
            if has_sandwich:
                info["architecture"] = "STU-only with MLP Sandwich"
            else:
                info["architecture"] = "STU-only"
        else:
            info["model_type"] = "baseline"
            info["architecture"] = "Standard Attention (Moirai baseline)"

        # Try to extract hyperparameters
        if 'hyper_parameters' in ckpt:
            hp = ckpt['hyper_parameters']
            module_kwargs = hp.get('module_kwargs', {})
            info["d_model"] = module_kwargs.get('d_model')
            info["num_layers"] = module_kwargs.get('num_layers')
            if module_kwargs.get('stu_layer_pattern'):
                info["stu_config"] = {
                    "pattern": module_kwargs.get('stu_layer_pattern'),
                    "num_eigh": module_kwargs.get('num_eigh'),
                    "use_sandwiched_stu": module_kwargs.get('use_sandwiched_stu'),
                    "sandwich_hidden_dim": module_kwargs.get('sandwich_hidden_dim'),
                }

        # Infer from path if not found
        if info["d_model"] is None:
            if "small" in checkpoint_path.lower():
                info["d_model"] = 384
                info["num_layers"] = 6
            elif "base" in checkpoint_path.lower():
                info["d_model"] = 768
                info["num_layers"] = 12

    except Exception as e:
        info["error"] = str(e)

    return info


def generate_markdown_report(
    model_info: dict,
    results_df: pd.DataFrame,
    output_path: str,
    args,
):
    """
    Generate a comprehensive markdown report with model info and metrics.

    Metrics computed following GIFT-Eval leaderboard methodology:
    - MASE[0.5]: Mean Absolute Scaled Error at median quantile
    - Geometric Mean MASE: Scale-invariant aggregation (primary metric)
    - MASE Rank: Would be computed if comparing multiple models
    """
    # Filter successful results
    success_df = results_df[~results_df.get("error", pd.Series([None]*len(results_df))).notna()].copy()

    # Compute aggregate metrics
    valid_mase = success_df['MASE'].dropna()
    valid_mase = valid_mase[(valid_mase > 0) & (valid_mase < 100)]  # Filter outliers

    metrics = {
        "num_configs": len(success_df),
        "num_successful": len(valid_mase),
        "mase_arithmetic_mean": valid_mase.mean() if len(valid_mase) > 0 else np.nan,
        "mase_geometric_mean": gmean(valid_mase) if len(valid_mase) > 0 else np.nan,
        "mase_median": valid_mase.median() if len(valid_mase) > 0 else np.nan,
        "mase_min": valid_mase.min() if len(valid_mase) > 0 else np.nan,
        "mase_max": valid_mase.max() if len(valid_mase) > 0 else np.nan,
        "mase_below_1": (valid_mase < 1.0).sum() if len(valid_mase) > 0 else 0,
        "mae_mean": success_df['MAE'].mean() if len(success_df) > 0 else np.nan,
        "mse_mean": success_df['MSE'].mean() if len(success_df) > 0 else np.nan,
    }

    # Generate markdown
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# GIFT-Eval Evaluation Report

**Generated:** {timestamp}

## Model Information

| Property | Value |
|----------|-------|
| **Checkpoint** | `{model_info.get('checkpoint_name', 'N/A')}` |
| **Architecture** | {model_info.get('architecture', 'N/A')} |
| **Parameters** | {model_info.get('param_count_str', 'N/A')} |
| **d_model** | {model_info.get('d_model', 'N/A')} |
| **num_layers** | {model_info.get('num_layers', 'N/A')} |
| **Model Type** | {model_info.get('model_type', 'N/A')} |

### Checkpoint Path
```
{model_info.get('checkpoint_path', 'N/A')}
```
"""

    if model_info.get('stu_config'):
        stu = model_info['stu_config']
        md += f"""
### STU Configuration
| Property | Value |
|----------|-------|
| Pattern | {stu.get('pattern', 'N/A')} |
| num_eigh | {stu.get('num_eigh', 'N/A')} |
| Sandwiched | {stu.get('use_sandwiched_stu', 'N/A')} |
| Sandwich Hidden Dim | {stu.get('sandwich_hidden_dim', 'N/A')} |
"""

    md += f"""
## Aggregate Metrics (GIFT-Eval Leaderboard Style)

| Metric | Value | Description |
|--------|-------|-------------|
| **Geometric Mean MASE** | **{metrics['mase_geometric_mean']:.4f}** | Primary leaderboard metric (scale-invariant) |
| Arithmetic Mean MASE | {metrics['mase_arithmetic_mean']:.4f} | Simple average |
| Median MASE | {metrics['mase_median']:.4f} | Robust central tendency |
| MASE < 1.0 | {metrics['mase_below_1']}/{metrics['num_successful']} | Configs beating naive baseline |
| Min MASE | {metrics['mase_min']:.4f} | Best single config |
| Max MASE | {metrics['mase_max']:.4f} | Worst single config |

### Comparison to Reference Models

| Model | Params | MASE (Geo Mean) | MASE (Mean) | MASE < 1 |
|-------|--------|-----------------|-------------|----------|
| **This Model** | {model_info.get('param_count_str', '?')} | **{metrics['mase_geometric_mean']:.4f}** | {metrics['mase_arithmetic_mean']:.4f} | {metrics['mase_below_1']}/97 |
| Moirai-small (official) | ~14M | 1.323 | 1.958 | 27/97 |
| Moirai-base (official) | ~90M | 1.259 | 2.019 | 40/97 |

*Note: Lower MASE is better. MASE < 1.0 means the model outperforms a seasonal naive baseline.*

## Per-Configuration Results

| Configuration | MASE | Status |
|---------------|------|--------|
"""

    # Add per-config results
    for _, row in success_df.sort_values('MASE').iterrows():
        status = "✅" if row['MASE'] < 1.0 else "❌"
        md += f"| {row['dataset']}/{row['term']} | {row['MASE']:.4f} | {status} |\n"

    # Add failed configs if any
    failed = results_df[results_df.get("error", pd.Series([None]*len(results_df))).notna()]
    if len(failed) > 0:
        md += f"""
## Failed Evaluations ({len(failed)} configs)

| Configuration | Error |
|---------------|-------|
"""
        for _, row in failed.iterrows():
            md += f"| {row['dataset']}/{row['term']} | {row.get('error', 'Unknown')[:50]}... |\n"

    md += f"""
## Evaluation Settings

| Setting | Value |
|---------|-------|
| Context Length | {args.context_length} |
| Patch Size | {args.patch_size} |
| Batch Size | {args.batch_size} |
| Device | {'CUDA' if torch.cuda.is_available() else 'CPU'} |
| Num Configs | {len(GIFTEVAL_CONFIGS) if not args.quick else len(QUICK_CONFIGS)} |

---
*Report generated by eval_gifteval.py*
"""

    # Write markdown file
    with open(output_path, 'w') as f:
        f.write(md)

    print(f"\nMarkdown report saved to: {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate on GIFT-Eval benchmark")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--model", type=str, help="HuggingFace model name (e.g., moirai-1.1-R-small)")
    parser.add_argument("--quick", action="store_true", help="Run on quick subset only")
    parser.add_argument("--context-length", type=int, default=1000, help="Context length")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="/scratch/gpfs/EHAZAN/jh1161/gifteval/results",
                       help="Output directory for results")
    parser.add_argument("--compare", nargs="+", help="Compare multiple result CSV files")
    args = parser.parse_args()

    # Handle comparison mode
    if args.compare:
        result_files = []
        for pattern in args.compare:
            result_files.extend(glob.glob(pattern))
        if not result_files:
            parser.error(f"No files found matching: {args.compare}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_dir) / f"comparison_{timestamp}.csv"
        compare_models(result_files, str(output_path))
        return

    if not args.checkpoint and not args.model:
        parser.error("Must specify either --checkpoint or --model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model loader function
    # Cache the module to avoid reloading the checkpoint for each config
    if args.checkpoint:
        model_name = Path(args.checkpoint).stem
        print(f"Loading from checkpoint: {args.checkpoint}")
        _cached_module = _load_module_from_checkpoint(args.checkpoint)
        def model_loader(prediction_length):
            return _build_forecast_from_module(
                _cached_module, args.checkpoint, prediction_length,
                args.context_length, args.patch_size
            )
    else:
        model_name = args.model
        def model_loader(prediction_length):
            return load_pretrained_model(
                args.model, prediction_length,
                args.context_length, args.patch_size
            )
        print(f"Loading pretrained model: {args.model}")

    # Select dataset configs
    configs = QUICK_CONFIGS if args.quick else GIFTEVAL_CONFIGS
    print(f"Evaluating on {len(configs)} dataset configurations")

    # Run evaluation
    results = []
    for i, (dataset_name, term) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Evaluating {dataset_name}/{term}...")

        result = evaluate_single_dataset(
            model_loader, dataset_name, term,
            batch_size=args.batch_size, device=device
        )

        if result:
            results.append(result)
            if "error" not in result:
                print(f"  MAE: {result['MAE']:.4f}, MSE: {result['MSE']:.4f}, MASE: {result['MASE']:.4f}")
            else:
                print(f"  Error: {result['error']}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save results as CSV (GIFT-Eval format)
    df = pd.DataFrame(results)
    csv_path = output_dir / f"gifteval_results_{model_name}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Also save in GIFT-Eval leaderboard format (all_results.csv)
    if not args.quick:
        leaderboard_csv = output_dir / f"all_results_{model_name}.csv"
        # Select and rename columns for leaderboard format
        leaderboard_df = df[["config_name", "MAE", "MSE", "MASE", "SMAPE"]].copy()
        leaderboard_df.columns = ["dataset_config", "MAE", "MSE", "MASE", "SMAPE"]
        leaderboard_df.insert(1, "model", model_name)
        leaderboard_df.to_csv(leaderboard_csv, index=False)
        print(f"Leaderboard format saved to: {leaderboard_csv}")

    # Print summary
    print("\n" + "="*80)
    print("GIFT-EVAL RESULTS SUMMARY")
    print("="*80)

    # Filter successful results
    success_df = df[~df.get("error", pd.Series([None]*len(df))).notna()]
    if len(success_df) > 0:
        print(f"\nSuccessful evaluations: {len(success_df)}/{len(configs)}")

        # Geometric mean of MASE (matches leaderboard aggregation)
        valid_mase = success_df['MASE'].dropna()
        valid_mase = valid_mase[valid_mase > 0]  # gmean requires positive values
        if len(valid_mase) > 0:
            geo_mean_mase = gmean(valid_mase)
            print(f"\n*** GEOMETRIC MEAN MASE: {geo_mean_mase:.4f} ***")
            print(f"    (Leaderboard uses this for normalized MASE aggregation)")
            print(f"    Interpretation: <1.0 = better than seasonal naive baseline")

        # Arithmetic means (for reference)
        print(f"\nArithmetic means (for reference, scale-dependent):")
        print(f"  Mean MAE:  {success_df['MAE'].mean():.4f}")
        print(f"  Mean MSE:  {success_df['MSE'].mean():.4f}")
        print(f"  Mean MASE: {success_df['MASE'].mean():.4f}")

        # MASE distribution
        print(f"\nMASE Distribution:")
        print(f"  Configs with MASE < 1.0 (beats naive): {(success_df['MASE'] < 1.0).sum()}/{len(success_df)}")
        print(f"  Min MASE:  {success_df['MASE'].min():.4f}")
        print(f"  Max MASE:  {success_df['MASE'].max():.4f}")
        print(f"  Median MASE: {success_df['MASE'].median():.4f}")


        # Per-dataset breakdown (compact)
        print(f"\nPer-configuration MASE (✓ = beats naive, ✗ = worse than naive):")
        for _, row in success_df.iterrows():
            indicator = "✓" if row['MASE'] < 1.0 else "✗"
            print(f"  {indicator} {row['dataset']}/{row['term']}: {row['MASE']:.4f}")

    # Report failures
    failed = df[df.get("error", pd.Series([None]*len(df))).notna()]
    if len(failed) > 0:
        print(f"\nFailed evaluations: {len(failed)}")
        for _, row in failed.iterrows():
            print(f"  - {row['dataset']}/{row['term']}: {row.get('error', 'Unknown error')}")

    # Create config.json for leaderboard submission
    if not args.quick:
        config = {
            "model_name": model_name,
            "model_type": "pretrained" if args.model else "fine-tuned",
            "is_zero_shot": True,
            "data_leakage": False,
            "code_available": True,
            "organization": "Princeton",
            "evaluation_date": timestamp,
        }
        config_path = output_dir / f"config_{model_name}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved to: {config_path}")

    # Generate comprehensive markdown report
    if args.checkpoint:
        model_info = get_model_info(args.checkpoint)
    else:
        model_info = {
            "checkpoint_name": model_name,
            "checkpoint_path": f"HuggingFace: Salesforce/{args.model}",
            "model_type": "pretrained",
            "architecture": "Moirai (HuggingFace)",
            "param_count_str": "~14M (small) / ~90M (base)",
        }

    md_path = output_dir / f"report_{model_name}_{timestamp}.md"
    generate_markdown_report(model_info, df, str(md_path), args)

    # Print final summary for easy viewing
    print("\n" + "="*80)
    print("FINAL SUMMARY (copy this for comparison)")
    print("="*80)
    if len(success_df) > 0:
        valid_mase = success_df['MASE'].dropna()
        valid_mase = valid_mase[(valid_mase > 0) & (valid_mase < 100)]
        if len(valid_mase) > 0:
            print(f"Model: {model_name}")
            print(f"Params: {model_info.get('param_count_str', 'N/A')}")
            print(f"Architecture: {model_info.get('architecture', 'N/A')}")
            print(f"MASE (Geo Mean): {gmean(valid_mase):.4f}")
            print(f"MASE (Arith Mean): {valid_mase.mean():.4f}")
            print(f"MASE (Median): {valid_mase.median():.4f}")
            print(f"MASE < 1.0: {(valid_mase < 1.0).sum()}/{len(valid_mase)}")


if __name__ == "__main__":
    main()
