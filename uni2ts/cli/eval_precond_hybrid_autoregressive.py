#!/usr/bin/env python3
"""
Autoregressive hybrid evaluation for preconditioned models.

Key difference from standard hybrid:
- Standard: Uses base model predictions as context for ALL timesteps
- Autoregressive: Uses previously reversed hybrid predictions as context

This reduces error propagation because we use our own (hopefully better)
predictions as context instead of relying on base model throughout.

Reversal strategy:
1. For t < degree: Use base model context (no choice, need initial context)
2. For t >= degree: Use previously reversed hybrid predictions as context
"""

import hydra
import numpy as np
import pandas as pd
import torch
from gluonts.time_feature import get_seasonality
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.eval_util.evaluation import evaluate_forecasts
from gluonts.ev.metrics import (
    MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
    MeanWeightedSumQuantileLoss,
)


def reverse_autoregressive(
    precond_predictions: np.ndarray,
    base_context: np.ndarray,
    input_length: int,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Autoregressive reversal: use previously reversed predictions as context.

    For each timestep t:
    - If t < degree: Use base_context (need initial bootstrap)
    - If t >= degree: Use previously reversed hybrid predictions as context

    Args:
        precond_predictions: [num_samples, pred_len] or [pred_len]
        base_context: [input_len + pred_len] - base model input + median predictions
        input_length: Where predictions start
        coeffs: [degree] preconditioning coefficients

    Returns:
        Hybrid predictions with autoregressive reversal
    """
    if not isinstance(precond_predictions, np.ndarray):
        precond_predictions = np.array(precond_predictions)
    if not isinstance(base_context, np.ndarray):
        base_context = np.array(base_context)

    # Handle 2D case (multiple samples)
    if precond_predictions.ndim == 2:
        return np.stack([
            _reverse_autoregressive_1d(precond_predictions[i], base_context, input_length, coeffs)
            for i in range(precond_predictions.shape[0])
        ], axis=0)
    else:
        return _reverse_autoregressive_1d(precond_predictions, base_context, input_length, coeffs)


def _reverse_autoregressive_1d(
    precond_seq: np.ndarray,
    base_context: np.ndarray,
    input_len: int,
    coeffs: np.ndarray
) -> np.ndarray:
    """
    Autoregressive reversal for single sequence.

    Key insight: Build hybrid_context incrementally:
    - Start with [input_window, ...empty...]
    - Fill in reversed predictions one by one
    - Use filled-in values as context for future predictions
    """
    n = len(coeffs)
    pred_len = len(precond_seq)
    result = precond_seq.copy()

    # Build hybrid context incrementally
    # Start with input window + space for predictions
    hybrid_context = np.zeros(input_len + pred_len)
    hybrid_context[:input_len] = base_context[:input_len]  # Copy input window

    # For first n timesteps, we must use base model predictions as context
    # (we don't have enough reversed predictions yet)
    for t in range(min(n, pred_len)):
        actual_pos = input_len + t

        if actual_pos >= n:
            # Use available context (mix of input window and base predictions)
            for i in range(n):
                context_idx = actual_pos - i - 1
                # For positions before prediction start, use input window
                # For positions in prediction window but not yet reversed, use base
                if context_idx < input_len:
                    result[t] -= coeffs[i] * hybrid_context[context_idx]
                else:
                    result[t] -= coeffs[i] * base_context[context_idx]

        # Store reversed prediction in hybrid context
        hybrid_context[actual_pos] = result[t]

    # For remaining timesteps, use hybrid context (our own predictions)
    for t in range(n, pred_len):
        actual_pos = input_len + t

        # Now we have enough reversed predictions to use as context
        for i in range(n):
            context_idx = actual_pos - i - 1
            result[t] -= coeffs[i] * hybrid_context[context_idx]

        # Store reversed prediction
        hybrid_context[actual_pos] = result[t]

    return result


def evaluate_autoregressive_hybrid(
    base_predictor,
    precond_predictor,
    test_data,
    precondition_coeffs: np.ndarray,
    batch_size: int = 100,
    seasonality: int = None,
):
    """
    Evaluate autoregressive hybrid model.

    Uses base model context only for initialization, then feeds back
    its own reversed predictions as context for subsequent timesteps.
    """
    test_data_list = list(test_data)

    def get_inputs(test_data_list):
        for item in test_data_list:
            if isinstance(item, tuple):
                yield item[0]
            elif hasattr(item, 'input'):
                yield item.input
            else:
                raise ValueError(f"Unknown test data format: {type(item)}")

    # Generate base model predictions
    print("Generating base model predictions...")
    base_forecast_it = base_predictor.predict(get_inputs(test_data_list), num_samples=100)
    base_forecasts = list(tqdm(base_forecast_it, total=len(test_data_list), desc="Base model"))

    # Generate preconditioned model predictions
    print("Generating preconditioned model predictions...")
    precond_forecast_it = precond_predictor.predict(get_inputs(test_data_list), num_samples=100)
    precond_forecasts = list(tqdm(precond_forecast_it, total=len(test_data_list), desc="Precond model"))

    # Extract input windows
    print("Extracting input windows...")
    input_windows = []
    for item in test_data_list:
        if isinstance(item, tuple):
            input_dict = item[0]
        elif hasattr(item, 'input'):
            input_dict = item.input
        else:
            raise ValueError(f"Unknown test data format: {type(item)}")

        if hasattr(input_dict, 'target'):
            input_target = input_dict.target
        elif isinstance(input_dict, dict) and 'target' in input_dict:
            input_target = input_dict['target']
        else:
            raise ValueError(f"Cannot extract target from input: {type(input_dict)}")

        if torch.is_tensor(input_target):
            input_target = input_target.cpu().numpy()
        elif not isinstance(input_target, np.ndarray):
            input_target = np.array(input_target)

        input_windows.append(input_target)

    # Create autoregressive hybrid forecasts
    print("Creating autoregressive hybrid forecasts...")
    hybrid_forecasts = []

    for base_fc, precond_fc, input_window in tqdm(
        zip(base_forecasts, precond_forecasts, input_windows),
        total=len(base_forecasts),
        desc="Autoregressive reversal"
    ):
        base_samples = base_fc.samples
        precond_samples = precond_fc.samples

        # Take median of base samples for stable initial context
        base_median = np.median(base_samples, axis=0)
        input_len = len(input_window)

        # Create full base context (input + base median)
        full_context = np.concatenate([input_window, base_median], axis=0)

        # Apply autoregressive reversal
        hybrid_samples = reverse_autoregressive(
            precond_predictions=precond_samples,
            base_context=full_context,
            input_length=input_len,
            coeffs=precondition_coeffs,
        )

        # Create forecast object
        import copy
        hybrid_fc = copy.copy(base_fc)
        hybrid_fc.samples = hybrid_samples
        hybrid_forecasts.append(hybrid_fc)

    print(f"Created {len(hybrid_forecasts)} autoregressive hybrid forecasts")

    # Evaluate
    print("Computing metrics...")
    metrics = [
        MSE(), MAE(), MAPE(), SMAPE(), MASE(), RMSE(), NRMSE(), ND(), MSIS(),
        MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    ]

    res = evaluate_forecasts(
        forecasts=hybrid_forecasts,
        test_data=test_data,
        metrics=metrics,
        batch_size=batch_size,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=seasonality,
    )

    return res


@hydra.main(version_base="1.3", config_path="conf/eval", config_name="default_hybrid")
def main(cfg: DictConfig):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.options.display.float_format = "{:.3f}".format

    test_data, metadata = call(cfg.data)
    batch_size = cfg.batch_size

    precondition_type = cfg.precond_model.get("precondition_type", "chebyshev")
    precondition_degree = cfg.precond_model.get("precondition_degree", 5)

    print(f"Autoregressive Hybrid Evaluation:")
    print(f"  Base model: {cfg.base_model._target_}")
    print(f"  Precond model: {cfg.precond_model._target_}")
    print(f"  Preconditioning: {precondition_type} degree {precondition_degree}")
    print(f"  Strategy: Autoregressive (use own predictions as context)")
    print()

    from uni2ts.transform import PolynomialPrecondition
    preconditioner = PolynomialPrecondition(
        polynomial_type=precondition_type,
        degree=precondition_degree,
        target_field="target",
        enabled=True,
    )
    coeffs = preconditioner.coeffs
    print(f"Coefficients: {coeffs}")
    print()

    while True:
        print("Loading models...")
        base_model = call(cfg.base_model, _partial_=True, _convert_="all")(
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
        )

        precond_model = call(cfg.precond_model, _partial_=True, _convert_="all")(
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
        )

        try:
            base_predictor = base_model.create_predictor(batch_size, cfg.device)
            precond_predictor = precond_model.create_predictor(batch_size, cfg.device)

            res = evaluate_autoregressive_hybrid(
                base_predictor=base_predictor,
                precond_predictor=precond_predictor,
                test_data=test_data,
                precondition_coeffs=coeffs,
                batch_size=batch_size,
                seasonality=get_seasonality(metadata.freq),
            )

            print("\n" + "="*80)
            print("AUTOREGRESSIVE HYBRID EVALUATION METRICS")
            print("="*80)
            print(res)
            print("="*80 + "\n")

            output_dir = HydraConfig.get().runtime.output_dir
            writer = SummaryWriter(log_dir=output_dir)
            for name, metric in res.to_dict("records")[0].items():
                writer.add_scalar(f"{metadata.split}_metrics_autoregressive_hybrid/{name}", metric)
            writer.close()

            csv_path = f"{output_dir}/metrics_autoregressive_hybrid.csv"
            res.to_csv(csv_path, index=False)
            print(f"Metrics saved to: {csv_path}")

            break
        except torch.cuda.OutOfMemoryError:
            print(f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size//2}")
            batch_size //= 2
            if batch_size < cfg.get("min_batch_size", 1):
                print(f"batch_size {batch_size} smaller than min_batch_size, ending evaluation")
                break


if __name__ == "__main__":
    main()
