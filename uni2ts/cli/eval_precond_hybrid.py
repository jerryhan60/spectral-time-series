#!/usr/bin/env python3
"""
Hybrid evaluation script for preconditioned models.

This script evaluates a "hybrid" approach that combines:
1. Base pretrained model forecasts (in original space)
2. Preconditioned model forecasts (in preconditioned space)

The hybrid forecast is computed by:
1. Generate base model predictions y_base in original space
2. Generate preconditioned model predictions ỹ_precond in preconditioned space
3. Reverse the preconditioning using base model outputs as context:
   y_hybrid[t] = ỹ_precond[t] + Σ(i=1 to n) c_i · y_context[t-i]
   where y_context comes from the base model's predictions

This allows the preconditioned model to predict "residuals" or "deltas"
while leveraging the base model's predictions as anchors.

Usage:
    python -m cli.eval_precond_hybrid \
        base_model=moirai_1.1_R_small \
        precond_model=moirai_precond_ckpt \
        precond_model.checkpoint_path=/path/to/ckpt \
        data=monash_cached \
        data.dataset_name=m1_monthly
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
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)


def reverse_precondition_with_base_context(
    precond_predictions,
    base_predictions,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Reverse preconditioning using base model predictions as context.

    This implements the hybrid reversal:
        y_hybrid[t] = ỹ_precond[t] + Σ(i=1 to n) c_i · y_base[t-i]

    Args:
        precond_predictions: Predictions from preconditioned model (in precond space)
                            Shape: [num_samples, prediction_length] or [prediction_length]
        base_predictions: Predictions from base model (in original space)
                         Shape: [num_samples, prediction_length] or [prediction_length]
        coeffs: Preconditioning coefficients [c_1, c_2, ..., c_n]

    Returns:
        Hybrid predictions in original space, same shape as input
    """
    # Ensure inputs are numpy arrays
    if not isinstance(precond_predictions, np.ndarray):
        precond_predictions = np.array(precond_predictions)
    if not isinstance(base_predictions, np.ndarray):
        base_predictions = np.array(base_predictions)

    # Handle different shapes
    if precond_predictions.ndim == 1:
        # Single prediction: [prediction_length]
        return _reverse_1d(precond_predictions, base_predictions, coeffs)
    elif precond_predictions.ndim == 2:
        # Multiple samples: [num_samples, prediction_length]
        return np.stack([
            _reverse_1d(precond_predictions[i], base_predictions[i], coeffs)
            for i in range(precond_predictions.shape[0])
        ], axis=0)
    else:
        raise ValueError(
            f"precond_predictions must be 1D or 2D, got shape {precond_predictions.shape}"
        )


def _reverse_1d(precond_seq: np.ndarray, base_seq: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """
    Reverse preconditioning for a single 1D sequence using base context.

    Implements: y_t = ỹ_t + Σ(i=1 to n) c_i · y_base[t-i]

    For the first few timesteps where we don't have enough history from base model,
    we use the preconditioned prediction directly (no reversal).
    """
    n = len(coeffs)
    result = precond_seq.copy()

    # For t >= n, we can apply full reversal using base model history
    if len(precond_seq) > n:
        # Compute weighted sum for all positions t >= n
        weighted_sum = np.zeros(len(precond_seq) - n)
        for i in range(n):
            # coeffs[i] corresponds to base_seq[t-(i+1)]
            # For all t in [n, len(precond_seq)), extract base_seq[t-(i+1)]
            weighted_sum += coeffs[i] * base_seq[n-i-1:len(base_seq)-i-1]

        result[n:] = precond_seq[n:] + weighted_sum

    # For t < n, keep preconditioned values (we don't have enough base history)
    # This is analogous to how forward preconditioning keeps original values for t < n

    return result


def evaluate_hybrid_model(
    base_predictor,
    precond_predictor,
    test_data,
    precondition_coeffs: np.ndarray,
    batch_size: int = 100,
    seasonality: int = None,
):
    """
    Evaluate hybrid model combining base and preconditioned predictions.

    The evaluation flow is:
    1. Generate predictions from base model (in original space)
    2. Generate predictions from preconditioned model (in precond space, no reversal)
    3. Create hybrid predictions by reversing precond predictions using base context
    4. Evaluate hybrid predictions against ground truth

    Args:
        base_predictor: Base model predictor (normal model)
        precond_predictor: Preconditioned model predictor (with reverse_output=False)
        test_data: Test dataset
        precondition_coeffs: Preconditioning coefficients
        batch_size: Batch size for prediction
        seasonality: Seasonality for MASE metric

    Returns:
        DataFrame with metrics
    """
    # Convert test_data to list to allow multiple iterations
    test_data_list = list(test_data)

    # Helper to get inputs from test data
    def get_inputs(test_data_list):
        for item in test_data_list:
            if isinstance(item, tuple):
                yield item[0]
            elif hasattr(item, 'input'):
                yield item.input
            else:
                raise ValueError(f"Unknown test data format: {type(item)}")

    print("Generating base model predictions...")
    base_forecast_it = base_predictor.predict(
        get_inputs(test_data_list),
        num_samples=100,
    )
    base_forecasts = list(tqdm(base_forecast_it, total=len(test_data_list), desc="Base model"))

    print("Generating preconditioned model predictions...")
    precond_forecast_it = precond_predictor.predict(
        get_inputs(test_data_list),
        num_samples=100,
    )
    precond_forecasts = list(tqdm(precond_forecast_it, total=len(test_data_list), desc="Precond model"))

    # Check we got same number of forecasts
    if len(base_forecasts) != len(precond_forecasts):
        raise ValueError(
            f"Mismatch in forecast counts: base={len(base_forecasts)}, "
            f"precond={len(precond_forecasts)}"
        )

    if len(base_forecasts) == 0:
        raise ValueError("No forecasts were generated.")

    print(f"Generated {len(base_forecasts)} forecast pairs")

    # Create hybrid forecasts
    print("Creating hybrid forecasts...")
    hybrid_forecasts = []

    for base_fc, precond_fc in tqdm(
        zip(base_forecasts, precond_forecasts),
        total=len(base_forecasts),
        desc="Hybridizing"
    ):
        # Get samples from forecasts
        # Forecast objects have .samples attribute with shape [num_samples, prediction_length, ...]
        base_samples = base_fc.samples  # [num_samples, pred_len]
        precond_samples = precond_fc.samples  # [num_samples, pred_len]

        # Apply hybrid reversal for each sample
        hybrid_samples = reverse_precondition_with_base_context(
            precond_predictions=precond_samples,
            base_predictions=base_samples,
            coeffs=precondition_coeffs,
        )

        # Create a new forecast object with hybrid samples
        # We can reuse the base forecast object and just replace the samples
        import copy
        hybrid_fc = copy.copy(base_fc)
        hybrid_fc.samples = hybrid_samples

        hybrid_forecasts.append(hybrid_fc)

    print(f"Created {len(hybrid_forecasts)} hybrid forecasts")

    # Compute metrics using the standard evaluate_forecasts function
    print("Computing metrics...")

    # Define metrics to compute
    metrics = [
        MSE(),
        MAE(),
        MAPE(),
        SMAPE(),
        MASE(),
        RMSE(),
        NRMSE(),
        ND(),
        MSIS(),
        MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    ]

    # Evaluate hybrid forecasts
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
    # Set display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.options.display.float_format = "{:.3f}".format

    # Load test data
    test_data, metadata = call(cfg.data)
    batch_size = cfg.batch_size

    # Get preconditioning parameters from precond_model config
    precondition_type = cfg.precond_model.get("precondition_type", "chebyshev")
    precondition_degree = cfg.precond_model.get("precondition_degree", 5)

    print(f"Hybrid Evaluation Configuration:")
    print(f"  Base model: {cfg.base_model._target_}")
    print(f"  Precond model: {cfg.precond_model._target_}")
    print(f"  Preconditioning type: {precondition_type}")
    print(f"  Preconditioning degree: {precondition_degree}")
    print()

    # Compute preconditioning coefficients
    from uni2ts.transform import PolynomialPrecondition
    preconditioner = PolynomialPrecondition(
        polynomial_type=precondition_type,
        degree=precondition_degree,
        target_field="target",
        enabled=True,
    )
    coeffs = preconditioner.coeffs
    print(f"Preconditioning coefficients: {coeffs}")
    print()

    while True:
        # Instantiate base model
        print("Loading base model...")
        base_model = call(cfg.base_model, _partial_=True, _convert_="all")(
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
        )

        # Instantiate preconditioned model (with reverse_output=False)
        print("Loading preconditioned model...")
        precond_model = call(cfg.precond_model, _partial_=True, _convert_="all")(
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
        )

        try:
            base_predictor = base_model.create_predictor(batch_size, cfg.device)
            precond_predictor = precond_model.create_predictor(batch_size, cfg.device)

            # Evaluate hybrid model
            res = evaluate_hybrid_model(
                base_predictor=base_predictor,
                precond_predictor=precond_predictor,
                test_data=test_data,
                precondition_coeffs=coeffs,
                batch_size=batch_size,
                seasonality=get_seasonality(metadata.freq),
            )

            print("\n" + "="*80)
            print("HYBRID EVALUATION METRICS")
            print("="*80)
            print(res)
            print("="*80 + "\n")

            # Save results
            output_dir = HydraConfig.get().runtime.output_dir
            writer = SummaryWriter(log_dir=output_dir)
            for name, metric in res.to_dict("records")[0].items():
                writer.add_scalar(f"{metadata.split}_metrics_hybrid/{name}", metric)
            writer.close()

            # Also save as CSV
            csv_path = f"{output_dir}/metrics_hybrid.csv"
            res.to_csv(csv_path, index=False)
            print(f"Metrics saved to: {csv_path}")

            break
        except torch.cuda.OutOfMemoryError:
            print(
                f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size//2}"
            )
            batch_size //= 2
            if batch_size < cfg.get("min_batch_size", 1):
                print(
                    f"batch_size {batch_size} smaller than "
                    f"min_batch_size {cfg.get('min_batch_size', 1)}, ending evaluation"
                )
                break


if __name__ == "__main__":
    main()
