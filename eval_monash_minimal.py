#!/usr/bin/env python3
"""
Minimal evaluation on cached Monash datasets.
Uses MoiraiForecast for both baseline and STU models.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/uni2ts/src")

import torch
import pandas as pd
import numpy as np
from gluonts.time_feature import get_seasonality

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Set gluonts cache directory explicitly
os.environ["GLUONTS_DATASETS"] = "/home/jh1161/.gluonts/datasets"

from uni2ts.eval_util.data import get_gluonts_test_dataset
from uni2ts.eval_util.evaluation import evaluate_model
from gluonts.ev.metrics import MAE as GluonTSMAE

# Use datasets that are cached locally
DATASETS = {
    "monash_m3_monthly": {"prediction_length": 18},
}


def load_baseline_model(checkpoint_path, prediction_length, context_length=1000, patch_size=32):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiPretrain
    pretrain = MoiraiPretrain.load_from_checkpoint(checkpoint_path)
    forecast = MoiraiForecast(
        prediction_length=prediction_length, target_dim=1, feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0, context_length=context_length,
        module=pretrain.module, patch_size=patch_size, num_samples=100,
    )
    return forecast


def load_stu_model(checkpoint_path, prediction_length, context_length=1000, patch_size=32):
    """Load STU model using MoiraiForecast wrapper."""
    from uni2ts.model.moirai import MoiraiHybridPretrain, MoiraiForecast

    pretrain = MoiraiHybridPretrain.load_from_checkpoint(checkpoint_path)
    module = pretrain.module

    forecast = MoiraiForecast(
        prediction_length=prediction_length, target_dim=1, feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0, context_length=context_length,
        module=module, patch_size=patch_size, num_samples=100,
    )
    return forecast


def evaluate_single_dataset(model, dataset_name, prediction_length, batch_size=32, device="cuda"):
    try:
        test_data, metadata = get_gluonts_test_dataset(
            dataset_name=dataset_name, prediction_length=prediction_length, mode="S"
        )
        predictor = model.create_predictor(batch_size=batch_size, device=device)

        # Create the MAE metric
        mae_metric = GluonTSMAE()
        print(f"    Metric name: {mae_metric.__class__.__name__}")

        result = evaluate_model(
            predictor, test_data=test_data, metrics=[mae_metric], batch_size=batch_size,
            axis=None, mask_invalid_label=True, allow_nan_forecast=False,
            seasonality=get_seasonality(metadata.freq)
        )

        print(f"    Result columns: {result.columns.tolist()}")
        print(f"    Result: {result}")

        # Try to get MAE from any matching column
        mae_col = None
        for col in result.columns:
            if 'mae' in col.lower() or 'MAE' in col:
                mae_col = col
                break

        if mae_col is None:
            mae_col = result.columns[0]  # Just use first column

        return result[mae_col].values[0], None
    except Exception as e:
        import traceback
        return None, f"{e}\n{traceback.format_exc()}"


def main():
    baseline_ckpt = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/moirai_small_baseline_20260125_164605/checkpoints/epoch_epoch_0099.ckpt"
    stu_ckpt = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_stu/lotsa_v1_unweighted/moirai_small_stu_20260125_164605/checkpoints/epoch_epoch_0099.ckpt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Baseline: {baseline_ckpt}")
    print(f"STU: {stu_ckpt}")
    print(f"GLUONTS_DATASETS: {os.environ.get('GLUONTS_DATASETS', 'not set')}")
    print()

    results = []
    for dataset_name, config in DATASETS.items():
        prediction_length = config["prediction_length"]
        print(f"=== {dataset_name} (prediction_length={prediction_length}) ===")
        row = {"dataset": dataset_name, "prediction_length": prediction_length}

        # Baseline
        try:
            print("  Loading baseline...")
            model = load_baseline_model(baseline_ckpt, prediction_length)
            model = model.to(device).eval()
            with torch.no_grad():
                mae, err = evaluate_single_dataset(model, dataset_name, prediction_length, device=device)
            if mae is not None:
                print(f"  Baseline MAE: {mae:.4f}")
                row["baseline_mae"] = mae
            else:
                print(f"  Baseline Error: {err[:500]}")
                row["baseline_mae"] = np.nan
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"  Baseline failed: {e}")
            traceback.print_exc()
            row["baseline_mae"] = np.nan

        # STU
        try:
            print("  Loading STU...")
            model = load_stu_model(stu_ckpt, prediction_length)
            model = model.to(device).eval()
            with torch.no_grad():
                mae, err = evaluate_single_dataset(model, dataset_name, prediction_length, device=device)
            if mae is not None:
                print(f"  STU MAE: {mae:.4f}")
                row["stu_mae"] = mae
            else:
                print(f"  STU Error: {err[:500]}")
                row["stu_mae"] = np.nan
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"  STU failed: {e}")
            traceback.print_exc()
            row["stu_mae"] = np.nan

        results.append(row)
        print()

    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("SUMMARY: MAE Results on Monash Datasets")
    print("="*70)
    print(df.to_string(index=False))

    df.to_csv("/scratch/gpfs/EHAZAN/jh1161/monash_minimal_results.csv", index=False)
    print("\nSaved to monash_minimal_results.csv")


if __name__ == "__main__":
    main()
