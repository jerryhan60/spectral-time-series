#!/usr/bin/env python3
"""
Full evaluation on all cached Monash datasets.
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
os.environ["GLUONTS_DATASETS"] = "/home/jh1161/.gluonts/datasets"

from uni2ts.eval_util.data import get_gluonts_test_dataset
from uni2ts.eval_util.evaluation import evaluate_model
from gluonts.ev.metrics import MAE as GluonTSMAE

# Available cached datasets with prediction lengths from Moirai paper
DATASETS = {
    # M3 Competition
    "monash_m3_monthly": {"prediction_length": 18},
    # M4 Competition datasets
    "m4_monthly": {"prediction_length": 18},
    "m4_quarterly": {"prediction_length": 8},
    "m4_yearly": {"prediction_length": 6},
    # Electricity
    "electricity": {"prediction_length": 168},
    # Traffic
    "traffic": {"prediction_length": 168},
    # Weather
    "temperature_rain_without_missing": {"prediction_length": 30},
    # Financial
    "bitcoin_with_missing": {"prediction_length": 30},
    # Rideshare
    "rideshare_with_missing": {"prediction_length": 168},
    # Sunspot
    "sunspot_without_missing": {"prediction_length": 30},
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
        mae_metric = GluonTSMAE()

        result = evaluate_model(
            predictor, test_data=test_data, metrics=[mae_metric], batch_size=batch_size,
            axis=None, mask_invalid_label=True, allow_nan_forecast=False,
            seasonality=get_seasonality(metadata.freq)
        )

        # Get MAE from any matching column
        mae_col = None
        for col in result.columns:
            if 'mae' in col.lower() or 'MAE' in col:
                mae_col = col
                break
        if mae_col is None:
            mae_col = result.columns[0]

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
                print(f"  Baseline Error: {err[:200]}")
                row["baseline_mae"] = np.nan
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Baseline failed: {e}")
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
                print(f"  STU Error: {err[:200]}")
                row["stu_mae"] = np.nan
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  STU failed: {e}")
            row["stu_mae"] = np.nan

        results.append(row)
        print()

        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv("/scratch/gpfs/EHAZAN/jh1161/monash_full_results.csv", index=False)

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL SUMMARY: MAE Results on Monash Datasets")
    print("="*80)
    print(df.to_string(index=False))

    valid_baseline = df["baseline_mae"].dropna()
    valid_stu = df["stu_mae"].dropna()
    print("\n" + "-"*80)
    if len(valid_baseline) > 0:
        print(f"Mean Baseline MAE: {valid_baseline.mean():.4f} (n={len(valid_baseline)})")
    if len(valid_stu) > 0:
        print(f"Mean STU MAE: {valid_stu.mean():.4f} (n={len(valid_stu)})")

    # Win rate (excluding bitcoin which has anomalous values)
    both_valid = df[df["baseline_mae"].notna() & df["stu_mae"].notna()]
    # Filter out bitcoin due to extreme values
    both_valid_clean = both_valid[both_valid["dataset"] != "bitcoin_with_missing"]

    if len(both_valid) > 0:
        stu_wins = (both_valid["stu_mae"] < both_valid["baseline_mae"]).sum()
        print(f"\nSTU wins: {stu_wins}/{len(both_valid)} datasets ({100*stu_wins/len(both_valid):.1f}%)")

    if len(both_valid_clean) > 0:
        stu_wins_clean = (both_valid_clean["stu_mae"] < both_valid_clean["baseline_mae"]).sum()
        print(f"STU wins (excl. bitcoin): {stu_wins_clean}/{len(both_valid_clean)} datasets ({100*stu_wins_clean/len(both_valid_clean):.1f}%)")

    df.to_csv("/scratch/gpfs/EHAZAN/jh1161/monash_full_results.csv", index=False)
    print(f"\nResults saved to: monash_full_results.csv")


if __name__ == "__main__":
    main()
