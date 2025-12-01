#!/usr/bin/env python3
"""
Hybrid evaluation script for preconditioned models.

This script evaluates a "hybrid" approach that combines:
1. Base pretrained model forecasts (in original space)
2. Preconditioned model forecasts (in preconditioned space)

The hybrid forecast is computed by:
1. Generate base model predictions y_base in original space (stochastic samples)
2. Compute median of base model predictions across samples for stable context
3. Generate preconditioned model predictions ỹ_precond in preconditioned space
4. Reverse the preconditioning using median base model outputs as context:
   Since forward preconditioning uses: ỹ_t = y_t + Σ c_i · y[t-i]
   Reverse uses: y_hybrid[t] = ỹ_precond[t] - Σ(i=1 to n) c_i · y_median[t-i]
   where y_median is the median of base model's predictions

This allows the preconditioned model to predict "residuals" or "deltas"
while leveraging the base model's median prediction as a stable anchor.

Key Design Choice:
  - Base model predictions are stochastic (100 samples)
  - Instead of pairing each precond sample with a different base sample,
    we use the MEDIAN of all base samples as context for ALL precond samples
  - This provides a stable, deterministic context for reversal

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
    base_context,
    input_length: int,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Reverse preconditioning using base model context (input + predictions).

    Since forward preconditioning uses ADDITION:
        ỹ_t = y_t + Σ c_i · y[t-i]

    This implements the hybrid reversal using SUBTRACTION:
        y_hybrid[t] = ỹ_precond[t] - Σ(i=1 to n) c_i · y_base[t-i]

    The key insight: predictions start at position `input_length` in the full sequence.
    For the first few prediction timesteps, we need context from the input window.

    Args:
        precond_predictions: Predictions from preconditioned model (in precond space)
                            Shape: [num_samples, prediction_length] or [prediction_length]
        base_context: Full base context (input + base predictions, in original space)
                     Shape: [input_length + prediction_length] or [num_samples, input_length + prediction_length]
        input_length: Length of the input window (where predictions start)
        coeffs: Preconditioning coefficients [c_1, c_2, ..., c_n]

    Returns:
        Hybrid predictions in original space, same shape as precond_predictions
    """
    # Ensure inputs are numpy arrays
    if not isinstance(precond_predictions, np.ndarray):
        precond_predictions = np.array(precond_predictions)
    if not isinstance(base_context, np.ndarray):
        base_context = np.array(base_context)

    # Handle different shapes
    if precond_predictions.ndim == 1:
        # Single prediction: [prediction_length]
        return _reverse_1d_with_context(precond_predictions, base_context, input_length, coeffs)
    elif precond_predictions.ndim == 2:
        # Multiple samples: [num_samples, prediction_length]
        if base_context.ndim == 1:
            # Same context for all samples
            return np.stack([
                _reverse_1d_with_context(precond_predictions[i], base_context, input_length, coeffs)
                for i in range(precond_predictions.shape[0])
            ], axis=0)
        elif base_context.ndim == 2:
            # Different context per sample
            return np.stack([
                _reverse_1d_with_context(precond_predictions[i], base_context[i], input_length, coeffs)
                for i in range(precond_predictions.shape[0])
            ], axis=0)
        else:
            raise ValueError(f"base_context must be 1D or 2D when precond_predictions is 2D, got shape {base_context.shape}")
    else:
        raise ValueError(
            f"precond_predictions must be 1D or 2D, got shape {precond_predictions.shape}"
        )


def _reverse_1d_with_context(
    precond_seq: np.ndarray,
    full_base_context: np.ndarray,
    input_len: int,
    coeffs: np.ndarray
) -> np.ndarray:
    """
    Reverse preconditioning for a single 1D sequence using full base context.

    Since forward preconditioning uses ADDITION:
        ỹ_t = y_t + Σ(i=1 to n) c_i · y[t-i]

    Reverse uses SUBTRACTION with base context:
        y_t = ỹ_t - Σ(i=1 to n) c_i · y_base[t-i]

    The predictions start at position `input_len` in the full sequence.
    For early prediction timesteps, we use context from the input window.

    Args:
        precond_seq: Preconditioned predictions [pred_len]
        full_base_context: Full base context [input_len + pred_len]
                          (input window + base model predictions concatenated)
        input_len: Where predictions start in full sequence
        coeffs: Preconditioning coefficients [c_1, ..., c_n]

    Returns:
        Reversed predictions in original space [pred_len]
    """
    n = len(coeffs)
    pred_len = len(precond_seq)
    result = precond_seq.copy()

    # For each prediction timestep t
    for t in range(pred_len):
        # Actual position in full sequence
        actual_pos = input_len + t

        # Check if we have enough history
        if actual_pos >= n:
            # Apply reversal: result[t] = precond_seq[t] - Σ coeffs[i] * full_base_context[actual_pos - i - 1]
            for i in range(n):
                context_idx = actual_pos - i - 1
                result[t] -= coeffs[i] * full_base_context[context_idx]
        # else: keep preconditioned value (not enough history, though this is rare with large context windows)

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
    1. Generate predictions from base model (in original space, stochastic samples)
    2. Compute median of base model predictions for stable context
    3. Generate predictions from preconditioned model (in precond space, no reversal)
    4. Create hybrid predictions by reversing precond predictions using median base context
    5. Evaluate hybrid predictions against ground truth

    Key approach: Uses median of base model predictions as a stable, deterministic
    context for reversing ALL preconditioned samples, rather than pairing stochastic
    samples with stochastic samples.

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

    # Extract input windows from test data for proper context
    print("Extracting input windows from test data...")
    input_windows = []
    for item in test_data_list:
        if isinstance(item, tuple):
            input_dict = item[0]
        elif hasattr(item, 'input'):
            input_dict = item.input
        else:
            raise ValueError(f"Unknown test data format: {type(item)}")

        # Extract input target
        if hasattr(input_dict, 'target'):
            input_target = input_dict.target
        elif isinstance(input_dict, dict) and 'target' in input_dict:
            input_target = input_dict['target']
        else:
            raise ValueError(f"Cannot extract target from input: {type(input_dict)}")

        # Convert to numpy if needed
        if torch.is_tensor(input_target):
            input_target = input_target.cpu().numpy()
        elif not isinstance(input_target, np.ndarray):
            input_target = np.array(input_target)

        input_windows.append(input_target)

    print(f"Extracted {len(input_windows)} input windows")

    # Create hybrid forecasts
    print("Creating hybrid forecasts...")
    hybrid_forecasts = []

    for base_fc, precond_fc, input_window in tqdm(
        zip(base_forecasts, precond_forecasts, input_windows),
        total=len(base_forecasts),
        desc="Hybridizing"
    ):
        # Get samples from forecasts
        # Forecast objects have .samples attribute with shape [num_samples, prediction_length, ...]
        base_samples = base_fc.samples  # [num_samples, pred_len]
        precond_samples = precond_fc.samples  # [num_samples, pred_len]

        # Take median of base model predictions across all samples
        # This provides a stable, deterministic context for reversing all preconditioned samples
        # Rather than pairing stochastic samples with stochastic samples
        base_median = np.median(base_samples, axis=0)  # [pred_len]

        input_len = len(input_window)

        # Create full base context using median: input_window + median base prediction
        # This single context will be used for reversing ALL preconditioned samples
        full_context = np.concatenate([input_window, base_median], axis=0)  # [input_len + pred_len]

        # Apply hybrid reversal for all samples using the same median-based context
        hybrid_samples = reverse_precondition_with_base_context(
            precond_predictions=precond_samples,
            base_context=full_context,  # Single median context for all samples
            input_length=input_len,
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
