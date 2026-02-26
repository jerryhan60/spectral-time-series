#!/usr/bin/env python3
"""
Evaluate Moirai-2 on fev-bench benchmark.

Based on ak's implementation at /scratch/gpfs/EHAZAN/ak8836/moirai2/project/moirai-2/eval/fev_bench_moirai2.py
Adapted to be self-contained (no dependency on ak's gift_eval_moirai2.py).

Usage:
    python fev_bench_moirai2.py --ckpt /path/to/checkpoint.ckpt --output results.csv
    python fev_bench_moirai2.py --ckpt /path/to/HF_dir --output results.csv --num-tasks 5
"""
import argparse
import gc
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import pandas as pd
import torch

# Add uni2ts src to path
sys.path.insert(0, "/scratch/gpfs/EHAZAN/jh1161/uni2ts/src")
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

import fev

hf_datasets.disable_progress_bars()


def predict_with_model(
    task: fev.Task,
    module: Moirai2Module,
    context_length: int = 4000,
    batch_size: int = 512,
    device: str = "cuda",
    seed: int = 123,
) -> tuple[list[hf_datasets.DatasetDict], float, dict]:
    torch.manual_seed(seed)
    gts_logger = logging.getLogger("gluonts")
    gts_logger.setLevel(100)

    model = Moirai2Forecast(
        module=module,
        prediction_length=task.horizon,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=batch_size, device=device)

    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows():
        _, prediction_dataset = fev.convert_input_data(
            window, adapter="gluonts", as_univariate=True
        )
        start_time = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            forecasts = list(predictor.predict(prediction_dataset))
        inference_time += time.monotonic() - start_time

        predictions_dict = {"predictions": np.stack([f.mean for f in forecasts])}
        for q in task.quantile_levels:
            predictions_dict[str(q)] = np.stack([f.quantile(q) for f in forecasts])
        predictions_per_window.append(
            fev.utils.combine_univariate_predictions_to_multivariate(
                hf_datasets.Dataset.from_dict(predictions_dict),
                target_columns=task.target_columns,
            )
        )

    extra_info = {
        "model_config": {
            "context_length": context_length,
            "batch_size": batch_size,
            "device": device,
            "seed": seed,
        }
    }

    return predictions_per_window, inference_time, extra_info


def main():
    parser = argparse.ArgumentParser(description="Evaluate Moirai-2 on fev-bench")
    parser.add_argument("--ckpt", required=True, help="Path to HF checkpoint or .ckpt file.")
    parser.add_argument("--output", default="fev_bench_results.csv", help="Output CSV path.")
    parser.add_argument("--context-length", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-tasks", type=int, default=None, help="Number of tasks (default: all 100).")
    parser.add_argument("--model-name", default="moirai2", help="Model name for results CSV.")
    parser.add_argument("--tasks-yaml", default=None, help="Path to fev_bench_tasks.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Output: {args.output}")

    # Load model
    print("Loading model...")
    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_file() and ckpt_path.suffix == ".ckpt":
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = {}
        for k, v in ckpt["state_dict"].items():
            if k.startswith("module."):
                state_dict[k[len("module."):]] = v
        module_kwargs = dict(ckpt.get("hyper_parameters", {}).get("module_kwargs", {}))
        module = Moirai2Module(**module_kwargs)
        module.load_state_dict(state_dict, strict=True)
        module = module.to(device)
    else:
        module = Moirai2Module.from_pretrained(args.ckpt).to(device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in module.parameters()):,}")

    # Load benchmark tasks
    print("Loading fev-bench tasks...")
    if args.tasks_yaml:
        tasks_yaml = args.tasks_yaml
    else:
        # Try local copy first, fall back to ak's
        local_yaml = Path(__file__).parent / "fev_bench_tasks.yaml"
        ak_yaml = Path("/scratch/gpfs/EHAZAN/ak8836/moirai2/project/moirai-2/eval/fev_bench_tasks.yaml")
        if local_yaml.exists():
            tasks_yaml = str(local_yaml)
        elif ak_yaml.exists():
            tasks_yaml = str(ak_yaml)
        else:
            raise FileNotFoundError("Cannot find fev_bench_tasks.yaml")

    benchmark = fev.Benchmark.from_yaml(str(tasks_yaml))
    tasks = benchmark.tasks
    if args.num_tasks is not None:
        tasks = tasks[:args.num_tasks]
    print(f"Evaluating {len(tasks)} tasks")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task.dataset_config} (horizon={task.horizon})")
        try:
            predictions, inference_time, extra_info = predict_with_model(
                task,
                module=module,
                context_length=args.context_length,
                batch_size=args.batch_size,
                device=device,
            )
            evaluation_summary = task.evaluation_summary(
                predictions,
                model_name=args.model_name,
                inference_time_s=inference_time,
                extra_info=extra_info,
            )
            sql_val = evaluation_summary.get("SQL", "N/A")
            mase_val = evaluation_summary.get("MASE", "N/A")
            sql_str = f"{sql_val:.4f}" if isinstance(sql_val, (int, float)) else str(sql_val)
            mase_str = f"{mase_val:.4f}" if isinstance(mase_val, (int, float)) else str(mase_val)
            print(f"  SQL={sql_str}, MASE={mase_str}, time={inference_time:.1f}s")
            summaries.append(evaluation_summary)
            pd.DataFrame(summaries).to_csv(output_path, index=False)

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            summaries.append({
                "dataset": task.dataset_config,
                "model": args.model_name,
                "error": str(e),
            })
            pd.DataFrame(summaries).to_csv(output_path, index=False)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final save
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_path, index=False)
    print(f"\nDone! Results saved to {output_path}")
    print(f"Successful: {len([s for s in summaries if 'error' not in s])}/{len(tasks)}")

    # Print aggregate metrics
    if "MASE" in summary_df.columns:
        mase_vals = pd.to_numeric(summary_df["MASE"], errors="coerce").dropna()
        if len(mase_vals) > 0:
            print(f"\nAggregate MASE:")
            print(f"  Geo Mean: {np.exp(np.log(mase_vals).mean()):.4f}")
            print(f"  Mean: {mase_vals.mean():.4f}")
            print(f"  Median: {mase_vals.median():.4f}")
    if "SQL" in summary_df.columns:
        sql_vals = pd.to_numeric(summary_df["SQL"], errors="coerce").dropna()
        if len(sql_vals) > 0:
            print(f"\nAggregate SQL:")
            print(f"  Geo Mean: {np.exp(np.log(sql_vals).mean()):.4f}")
            print(f"  Mean: {sql_vals.mean():.4f}")
            print(f"  Median: {sql_vals.median():.4f}")


if __name__ == "__main__":
    main()
