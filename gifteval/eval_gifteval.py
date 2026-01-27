#!/usr/bin/env python3
"""
Evaluate checkpoints on GIFT-Eval benchmark.
Outputs results in the format expected by the GIFT-Eval leaderboard.

Usage:
    python eval_gifteval.py --checkpoint /path/to/checkpoint.ckpt
    python eval_gifteval.py --checkpoint /path/to/checkpoint.ckpt --quick
    python eval_gifteval.py --model moirai-1.1-R-small  # Use HuggingFace pretrained
"""

import argparse
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

# Import gift_eval after path setup
from gift_eval.data import Dataset

# Import uni2ts evaluation utilities
from uni2ts.eval_util.evaluation import evaluate_model


# GIFT-Eval dataset configurations
# Format: (dataset_name, term) where term in ["short", "medium", "long"]
GIFTEVAL_CONFIGS = [
    # Energy domain
    ("australian_electricity_demand", "short"),
    ("australian_electricity_demand", "medium"),
    ("electricity_hourly", "short"),
    ("electricity_hourly", "medium"),
    ("electricity_weekly", "short"),
    ("solar_10_minutes", "short"),
    ("solar_weekly", "short"),

    # Transport domain
    ("traffic_hourly", "short"),
    ("traffic_hourly", "medium"),
    ("traffic_weekly", "short"),
    ("pedestrian_counts", "short"),
    ("uber_tlc_hourly", "short"),

    # Nature domain
    ("weather", "short"),
    ("weather", "medium"),
    ("temperature_rain_with_missing", "short"),
    ("sunspot_with_missing", "short"),
    ("saugeenday", "short"),

    # Economic domain
    ("fred_md", "short"),
    ("nn5_daily_with_missing", "short"),
    ("nn5_weekly", "short"),
    ("exchange_rate", "short"),
    ("exchange_rate", "medium"),

    # Health domain
    ("hospital", "short"),
    ("covid_deaths", "short"),

    # Sales domain
    ("dominick", "short"),
    ("car_parts_with_missing", "short"),

    # Web/Tech domain
    ("kdd_cup_2018_with_missing", "short"),

    # M-competitions
    ("m1_monthly", "short"),
    ("m1_quarterly", "short"),
    ("m1_yearly", "short"),
    ("monash_m3_monthly", "short"),
    ("monash_m3_quarterly", "short"),
    ("monash_m3_yearly", "short"),
    ("monash_m3_other", "short"),
    ("m4_hourly", "short"),
    ("m4_daily", "short"),
    ("m4_weekly", "short"),
    ("m4_monthly", "short"),
    ("m4_quarterly", "short"),
    ("m4_yearly", "short"),

    # Tourism
    ("tourism_monthly", "short"),
    ("tourism_quarterly", "short"),
    ("tourism_yearly", "short"),

    # CIF
    ("cif_2016_6", "short"),
    ("cif_2016_12", "short"),
]

# Quick subset for fast evaluation (representative sample)
QUICK_CONFIGS = [
    ("monash_m3_monthly", "short"),
    ("electricity_hourly", "short"),
    ("traffic_hourly", "short"),
    ("weather", "short"),
    ("nn5_daily_with_missing", "short"),
    ("hospital", "short"),
    ("tourism_monthly", "short"),
    ("m4_hourly", "short"),
]


def load_checkpoint_model(checkpoint_path: str, prediction_length: int,
                          context_length: int = 1000, patch_size: int = 32):
    """Load model from a checkpoint file."""
    from uni2ts.model.moirai import MoiraiForecast, MoiraiPretrain

    # Try loading as standard Moirai checkpoint
    try:
        pretrain = MoiraiPretrain.load_from_checkpoint(checkpoint_path)
        module = pretrain.module

        forecast = MoiraiForecast(
            prediction_length=prediction_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=context_length,
            module=module,
            patch_size=patch_size,
            num_samples=100,
        )
        return forecast
    except Exception as e:
        print(f"Standard load failed: {e}")

    # Try loading as hybrid STU checkpoint
    try:
        from uni2ts.model.moirai import MoiraiHybridPretrain

        pretrain = MoiraiHybridPretrain.load_from_checkpoint(checkpoint_path)
        module = pretrain.module

        class HybridForecast(MoiraiForecast):
            def __init__(self, *args, module=None, **kwargs):
                super(MoiraiForecast, self).__init__()
                self.save_hyperparameters(ignore=["module"])
                self.module = module
                from uni2ts.model.moirai.forecast import SampleNLLLoss
                self.per_sample_loss_func = SampleNLLLoss()

        forecast = HybridForecast(
            prediction_length=prediction_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=context_length,
            module=module,
            patch_size=patch_size,
            num_samples=100,
        )
        return forecast
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


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
        dataset = Dataset(name=dataset_name, term=term, to_univariate=True)

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

        # Extract metric values
        metrics_dict = {
            "dataset": dataset_name,
            "term": term,
            "config_name": f"{dataset_name}/{dataset.freq}/{term}",
            "prediction_length": prediction_length,
            "MAE": float(result["MAE"].values[0]),
            "MSE": float(result["MSE"].values[0]),
            "MASE": float(result["MASE"].values[0]),
            "SMAPE": float(result["sMAPE"].values[0]) if "sMAPE" in result else np.nan,
        }

        return metrics_dict

    except Exception as e:
        print(f"Error evaluating {dataset_name}/{term}: {e}")
        return {
            "dataset": dataset_name,
            "term": term,
            "config_name": f"{dataset_name}/?/{term}",
            "error": str(e),
        }
    finally:
        # Clean up GPU memory
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()


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
    args = parser.parse_args()

    if not args.checkpoint and not args.model:
        parser.error("Must specify either --checkpoint or --model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model loader function
    if args.checkpoint:
        model_name = Path(args.checkpoint).stem
        def model_loader(prediction_length):
            return load_checkpoint_model(
                args.checkpoint, prediction_length,
                args.context_length, args.patch_size
            )
        print(f"Loading from checkpoint: {args.checkpoint}")
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
    print("SUMMARY")
    print("="*80)

    # Filter successful results
    success_df = df[~df.get("error", pd.Series([None]*len(df))).notna()]
    if len(success_df) > 0:
        print(f"\nSuccessful evaluations: {len(success_df)}/{len(configs)}")
        print(f"Mean MAE:  {success_df['MAE'].mean():.4f}")
        print(f"Mean MSE:  {success_df['MSE'].mean():.4f}")
        print(f"Mean MASE: {success_df['MASE'].mean():.4f}")

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


if __name__ == "__main__":
    main()
