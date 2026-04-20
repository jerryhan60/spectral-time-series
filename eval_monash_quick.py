#!/usr/bin/env python3
"""
Quick evaluation of baseline and STU models on subset of Monash datasets.
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
from gluonts.ev.metrics import MAE

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from uni2ts.eval_util.data import get_gluonts_test_dataset
from uni2ts.eval_util.evaluation import evaluate_model

# Smaller set of core Monash datasets for quick testing
MONASH_DATASETS = {
    "monash_m3_monthly": {"prediction_length": 18},
    "monash_m3_quarterly": {"prediction_length": 8},
    "monash_m3_yearly": {"prediction_length": 6},
    "monash_m3_other": {"prediction_length": 8},
    "nn5_weekly": {"prediction_length": 8},
    "tourism_monthly": {"prediction_length": 24},
    "tourism_quarterly": {"prediction_length": 8},
    "tourism_yearly": {"prediction_length": 4},
    "hospital": {"prediction_length": 12},
    "weather": {"prediction_length": 30},
}


def load_baseline_model(checkpoint_path, prediction_length, context_length=1000, patch_size=32):
    """Load the baseline Moirai model from checkpoint."""
    from uni2ts.model.moirai import MoiraiForecast, MoiraiPretrain

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


def load_stu_model(checkpoint_path, prediction_length, context_length=1000, patch_size=32):
    """Load the STU hybrid model from checkpoint."""
    from uni2ts.model.moirai import MoiraiHybridPretrain
    from uni2ts.model.moirai.forecast import MoiraiForecast, SampleNLLLoss
    import lightning as L

    pretrain = MoiraiHybridPretrain.load_from_checkpoint(checkpoint_path)
    module = pretrain.module

    class HybridForecast(MoiraiForecast):
        def __init__(self, prediction_length, target_dim, feat_dynamic_real_dim,
                     past_feat_dynamic_real_dim, context_length, module, patch_size, num_samples):
            L.LightningModule.__init__(self)
            self.save_hyperparameters(ignore=["module"])
            self.module = module
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


def evaluate_single_dataset(model, dataset_name, prediction_length, batch_size=32, device="cuda"):
    """Evaluate model on a single dataset and return MAE."""
    try:
        test_data, metadata = get_gluonts_test_dataset(
            dataset_name=dataset_name,
            prediction_length=prediction_length,
            mode="S"
        )

        predictor = model.create_predictor(batch_size=batch_size, device=device)
        metrics = [MAE()]
        result = evaluate_model(
            predictor,
            test_data=test_data,
            metrics=metrics,
            batch_size=batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=get_seasonality(metadata.freq),
        )

        mae = result["MAE"].values[0]
        return mae, None
    except Exception as e:
        import traceback
        return None, f"{e}\n{traceback.format_exc()}"


def main():
    baseline_ckpt = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small/lotsa_v1_unweighted/moirai_small_baseline_20260125_164605/checkpoints/epoch_epoch_0099.ckpt"
    stu_ckpt = "/scratch/gpfs/EHAZAN/jh1161/uni2ts/outputs/pretrain/moirai_small_stu/lotsa_v1_unweighted/moirai_small_stu_20260125_164605/checkpoints/epoch_epoch_0099.ckpt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Baseline checkpoint: {baseline_ckpt}")
    print(f"STU checkpoint: {stu_ckpt}")
    print()

    results = []

    for dataset_name, config in MONASH_DATASETS.items():
        prediction_length = config["prediction_length"]
        print(f"Evaluating {dataset_name} (prediction_length={prediction_length})...")

        row = {"dataset": dataset_name, "prediction_length": prediction_length}

        # Baseline
        try:
            print(f"  Loading baseline model...")
            baseline_model = load_baseline_model(baseline_ckpt, prediction_length)
            baseline_model = baseline_model.to(device)
            baseline_model.eval()

            print(f"  Running baseline evaluation...")
            mae, err = evaluate_single_dataset(baseline_model, dataset_name, prediction_length, device=device)
            if err:
                print(f"    Baseline error: {err[:200]}")
                row["baseline_mae"] = np.nan
            else:
                print(f"    Baseline MAE: {mae:.4f}")
                row["baseline_mae"] = mae

            del baseline_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Baseline failed: {e}")
            row["baseline_mae"] = np.nan

        # STU
        try:
            print(f"  Loading STU model...")
            stu_model = load_stu_model(stu_ckpt, prediction_length)
            stu_model = stu_model.to(device)
            stu_model.eval()

            print(f"  Running STU evaluation...")
            mae, err = evaluate_single_dataset(stu_model, dataset_name, prediction_length, device=device)
            if err:
                print(f"    STU error: {err[:200]}")
                row["stu_mae"] = np.nan
            else:
                print(f"    STU MAE: {mae:.4f}")
                row["stu_mae"] = mae

            del stu_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    STU failed: {e}")
            row["stu_mae"] = np.nan

        results.append(row)
        print()

    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY: MAE Results on Monash Datasets")
    print("="*80)
    print(df[["dataset", "prediction_length", "baseline_mae", "stu_mae"]].to_string(index=False))

    print("\n" + "-"*80)
    valid_baseline = df["baseline_mae"].dropna()
    valid_stu = df["stu_mae"].dropna()
    if len(valid_baseline) > 0:
        print(f"Mean Baseline MAE: {valid_baseline.mean():.4f} (n={len(valid_baseline)})")
    if len(valid_stu) > 0:
        print(f"Mean STU MAE: {valid_stu.mean():.4f} (n={len(valid_stu)})")

    output_path = "/scratch/gpfs/EHAZAN/jh1161/monash_comparison_results_quick.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
