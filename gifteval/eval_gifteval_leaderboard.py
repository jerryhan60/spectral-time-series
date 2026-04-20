#!/usr/bin/env python3
"""
Evaluate Moirai-2 on GIFT-Eval using the EXACT same code path as the official leaderboard.

Key difference from eval_gifteval.py: uses model.predict() (first-value padding)
instead of model.create_predictor() (zero padding via gluonts TFTInstanceSplitter).

This matches the official moirai2.ipynb notebook from gift-eval exactly.
"""
import argparse
import csv
import json
import logging
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
from gluonts.ev.metrics import MAE, MASE, MSE, SMAPE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model import evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality

# Add uni2ts src to path
sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/uni2ts/src")
from gift_eval.data import Dataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module, Moirai2Pretrain

warnings.filterwarnings("ignore")
logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)


class MoiraiQuantilePredictor:
    """
    Exact replica of the official GIFT-Eval Moirai2 predictor.
    Uses model.predict() which pads short series with first value (not zeros).
    """
    def __init__(self, model, batch_size=512,
                 quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
        self.model = model
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.prediction_length = model.hparams.prediction_length

    def predict(self, test_data_input):
        forecast_quantiles = []
        for batch in batcher(test_data_input, batch_size=self.batch_size):
            past_target = [entry["target"] for entry in batch]
            forecasts = self.model.predict(past_target)
            forecast_quantiles.append(forecasts)
        forecast_quantiles = np.concatenate(forecast_quantiles)

        quantile_forecasts = []
        for item, ts in zip(forecast_quantiles, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            # item shape from model.predict(): (num_quantiles, future_time, tgt) or (future_time, num_quantiles)
            # QuantileForecast expects: (num_quantiles, future_time)
            arr = np.squeeze(item)  # remove tgt dim if present
            if arr.ndim == 2 and arr.shape[0] != len(self.quantile_levels) and arr.shape[1] == len(self.quantile_levels):
                arr = arr.T  # transpose (future_time, num_quantiles) -> (num_quantiles, future_time)
            quantile_forecasts.append(
                QuantileForecast(
                    item_id=ts["item_id"],
                    forecast_arrays=arr,
                    start_date=forecast_start_date,
                    forecast_keys=list(map(str, self.quantile_levels)),
                )
            )
        return quantile_forecasts


# Config name mapping to match leaderboard conventions
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# All 97 GIFT-Eval configs
SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W "
    "hospital covid_deaths "
    "us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
    "car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D "
    "SZ_TAXI/15T SZ_TAXI/H "
    "M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W "
    "ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
).split()

MED_LONG_DATASETS = (
    "electricity/15T electricity/H "
    "solar/10T solar/H "
    "kdd_cup_2018_with_missing/H "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "SZ_TAXI/15T "
    "M_DENSE/H "
    "ett1/15T ett1/H "
    "ett2/15T ett2/H "
    "jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T "
    "bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
).split()

# Load dataset properties
PROPS_PATH = "/scratch/gpfs/EHAZAN/jh1161/gifteval/gift-eval/notebooks/dataset_properties.json"


def get_leaderboard_config_name(ds_name, term, dataset_properties_map):
    """Generate config name matching leaderboard convention."""
    if "/" in ds_name:
        ds_key = ds_name.split("/")[0].lower()
        ds_freq = ds_name.split("/")[1]
    else:
        ds_key = ds_name.lower()
        ds_freq = dataset_properties_map[PRETTY_NAMES.get(ds_key, ds_key)]["frequency"]
    ds_key = PRETTY_NAMES.get(ds_key, ds_key)
    return f"{ds_key}/{ds_freq}/{term}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to .ckpt file or HuggingFace model name")
    parser.add_argument("--model-name", type=str, default="moirai2")
    parser.add_argument("--context-length", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str,
                       default="/scratch/gpfs/EHAZAN/jh1161/gifteval/results")
    args = parser.parse_args()

    dataset_properties_map = json.load(open(PROPS_PATH))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"lb_results_{args.model_name}_{timestamp}.csv")

    # Determine all configs
    all_datasets = list(set(SHORT_DATASETS + MED_LONG_DATASETS))

    # Load module ONCE outside the loop
    print(f"Loading model from: {args.checkpoint}", flush=True)
    if args.checkpoint.endswith(".ckpt"):
        pretrain = Moirai2Pretrain.load_from_checkpoint(args.checkpoint)
        module = pretrain.module
    else:
        hf_path = f"Salesforce/{args.checkpoint}" if "/" not in args.checkpoint else args.checkpoint
        module = Moirai2Module.from_pretrained(hf_path)
    print(f"Module loaded successfully", flush=True)

    results = []

    for ds_name in sorted(all_datasets):
        terms = ["short"]
        if ds_name in MED_LONG_DATASETS:
            terms.extend(["medium", "long"])

        for term in terms:
            config_name = get_leaderboard_config_name(ds_name, term, dataset_properties_map)
            print(f"Evaluating: {config_name}", flush=True)

            try:
                # Load dataset (match official: to_univariate for multivariate)
                dataset_check = Dataset(name=ds_name, term=term, to_univariate=False)
                is_mv = dataset_check.target_dim > 1
                dataset = Dataset(name=ds_name, term=term, to_univariate=is_mv)

                pred_len = dataset.prediction_length
                seasonality = get_seasonality(dataset.freq)

                # Create forecast wrapper with this prediction length (reuse module)
                model = Moirai2Forecast(
                    module=module,
                    prediction_length=pred_len,
                    context_length=args.context_length,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=dataset.past_feat_dynamic_real_dim,
                ).to(device)
                model.eval()

                # Use official predictor (model.predict with first-value padding)
                predictor = MoiraiQuantilePredictor(model, batch_size=args.batch_size)

                metrics = [
                    MSE(forecast_type="mean"),
                    MSE(forecast_type=0.5),
                    MAE(),
                    MASE(),
                    SMAPE(),
                    MeanWeightedSumQuantileLoss(
                        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    ),
                ]

                res = evaluate_model(
                    predictor,
                    test_data=dataset.test_data,
                    metrics=metrics,
                    batch_size=args.batch_size,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=seasonality,
                )

                row = {
                    "dataset": config_name,
                    "model": args.model_name,
                    "eval_metrics/MASE[0.5]": float(res["MASE[0.5]"].values[0]),
                    "eval_metrics/MAE[0.5]": float(res["MAE[0.5]"].values[0]),
                    "eval_metrics/MSE[mean]": float(res["MSE[mean]"].values[0]),
                    "eval_metrics/mean_weighted_sum_quantile_loss": float(res["mean_weighted_sum_quantile_loss"].values[0]),
                }
                results.append(row)
                print(f"  MASE={row['eval_metrics/MASE[0.5]']:.4f}", flush=True)

            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
                results.append({
                    "dataset": config_name,
                    "model": args.model_name,
                    "eval_metrics/MASE[0.5]": float("nan"),
                })

    # Write CSV
    fieldnames = ["dataset", "model", "eval_metrics/MASE[0.5]", "eval_metrics/MAE[0.5]",
                  "eval_metrics/MSE[mean]", "eval_metrics/mean_weighted_sum_quantile_loss"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute geo mean MASE
    mases = [r["eval_metrics/MASE[0.5]"] for r in results
             if not math.isnan(r["eval_metrics/MASE[0.5]"])]
    geo_mean = math.exp(sum(math.log(m) for m in mases) / len(mases))

    # Normalize by seasonal naive
    naive_path = "/scratch/gpfs/EHAZAN/jh1161/gift-eval-space/results/seasonal_naive/all_results.csv"
    with open(naive_path) as f:
        naive = {r["dataset"]: float(r["eval_metrics/MASE[0.5]"]) for r in csv.DictReader(f)}

    matched_norm = []
    for r in results:
        if r["dataset"] in naive and not math.isnan(r["eval_metrics/MASE[0.5]"]):
            matched_norm.append(r["eval_metrics/MASE[0.5]"] / naive[r["dataset"]])

    if matched_norm:
        geo_norm = math.exp(sum(math.log(n) for n in matched_norm) / len(matched_norm))
    else:
        geo_norm = float("nan")

    print(f"\n{'='*60}")
    print(f"Results: {csv_path}")
    print(f"Configs evaluated: {len(mases)}/97")
    print(f"Raw geo mean MASE: {geo_mean:.6f}")
    print(f"Normalized geo mean MASE: {geo_norm:.6f}")
    print(f"(Official Moirai 2.0 paper reports: 0.728)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
