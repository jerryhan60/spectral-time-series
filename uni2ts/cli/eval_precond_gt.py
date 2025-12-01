#!/usr/bin/env python3
"""
Ground truth context evaluation script for preconditioned models.

This script evaluates preconditioned models by reversing the preconditioning
using ground truth as context, rather than base model predictions.

The evaluation approach:
1. Generate preconditioned model predictions ỹ_precond in preconditioned space
2. Reverse the preconditioning using ground truth as context:
   Since forward preconditioning uses: ỹ_t = y_t + Σ c_i · y[t-i]
   Reverse uses: y_gt_reversed[t] = ỹ_precond[t] - Σ(i=1 to n) c_i · y_gt[t-i]
   where y_gt comes from the actual ground truth values

This represents the "best case" scenario where the model has perfect
context for reversal, useful for understanding the upper bound of
performance with this architecture.

Usage:
    python -m cli.eval_precond_gt \
        model=moirai_precond_ckpt \
        model.checkpoint_path=/path/to/ckpt \
        model.precondition_type=chebyshev \
        model.precondition_degree=5 \
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


def reverse_precondition_with_gt_context(
    precond_predictions,
    full_ground_truth,
    input_length: int,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Reverse preconditioning using full ground truth (input + label) as context.

    Since forward preconditioning uses ADDITION:
        ỹ_t = y_t + Σ c_i · y[t-i]

    This implements reversal using ground truth context with SUBTRACTION:
        y_gt_reversed[t] = ỹ_precond[t] - Σ(i=1 to n) c_i · y_gt[t-i]

    The key insight: predictions start at position `input_length` in the full sequence,
    so we need context from the input window for the first few prediction timesteps.

    Args:
        precond_predictions: Predictions from preconditioned model (in precond space)
                            Shape: [num_samples, prediction_length] or [prediction_length]
        full_ground_truth: Full ground truth sequence (input + label, in original space)
                          Shape: [input_length + prediction_length]
        input_length: Length of the input window (where predictions start)
        coeffs: Preconditioning coefficients [c_1, c_2, ..., c_n]

    Returns:
        GT-context-reversed predictions in original space, same shape as precond_predictions
    """
    # Ensure inputs are numpy arrays
    if not isinstance(precond_predictions, np.ndarray):
        precond_predictions = np.array(precond_predictions)
    if not isinstance(full_ground_truth, np.ndarray):
        full_ground_truth = np.array(full_ground_truth)

    # Handle different shapes
    if precond_predictions.ndim == 1:
        # Single prediction: [prediction_length]
        return _reverse_1d_with_full_context(precond_predictions, full_ground_truth, input_length, coeffs)
    elif precond_predictions.ndim == 2:
        # Multiple samples: [num_samples, prediction_length]
        # Ground truth is the same for all samples
        return np.stack([
            _reverse_1d_with_full_context(precond_predictions[i], full_ground_truth, input_length, coeffs)
            for i in range(precond_predictions.shape[0])
        ], axis=0)
    else:
        raise ValueError(
            f"precond_predictions must be 1D or 2D, got shape {precond_predictions.shape}"
        )


def _reverse_1d_with_full_context(
    precond_seq: np.ndarray,
    full_gt: np.ndarray,
    input_len: int,
    coeffs: np.ndarray
) -> np.ndarray:
    """
    Reverse preconditioning for a single 1D sequence using full ground truth context.

    Since forward preconditioning uses ADDITION:
        ỹ_t = y_t + Σ(i=1 to n) c_i · y[t-i]

    Reverse uses SUBTRACTION with ground truth:
        y_t = ỹ_t - Σ(i=1 to n) c_i · y_gt[t-i]

    The predictions start at position `input_len` in the full sequence.
    For early prediction timesteps, we use context from the input window.

    Args:
        precond_seq: Preconditioned predictions [pred_len]
        full_gt: Full ground truth [input_len + pred_len]
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
            # Apply reversal: result[t] = precond_seq[t] - Σ coeffs[i] * full_gt[actual_pos - i - 1]
            for i in range(n):
                context_idx = actual_pos - i - 1
                result[t] -= coeffs[i] * full_gt[context_idx]
        # else: keep preconditioned value (not enough history, though this is rare with large context windows)

    return result


def evaluate_gt_context_model(
    precond_predictor,
    test_data,
    precondition_coeffs: np.ndarray,
    batch_size: int = 100,
    seasonality: int = None,
):
    """
    Evaluate preconditioned model with ground truth context reversal.

    The evaluation flow is:
    1. Generate predictions from preconditioned model (in precond space, no reversal)
    2. Extract ground truth from test data
    3. Create GT-context-reversed predictions by reversing using ground truth
    4. Evaluate GT-context-reversed predictions against ground truth

    Args:
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

    # Extract ground truth from test data
    # IMPORTANT: We need BOTH input and label to have proper context for reversal
    # The first few prediction timesteps need context from the input window
    print("Extracting ground truth (input + label) from test data...")
    ground_truths = []
    for item in test_data_list:
        if isinstance(item, tuple):
            # Tuple format: (input, label)
            input_dict, label_dict = item
        elif hasattr(item, 'input') and hasattr(item, 'label'):
            input_dict = item.input
            label_dict = item.label
        else:
            raise ValueError(f"Cannot extract input/label from test data format: {type(item)}")

        # Extract target values from input and label
        if hasattr(input_dict, 'target'):
            input_target = input_dict.target
        elif isinstance(input_dict, dict) and 'target' in input_dict:
            input_target = input_dict['target']
        else:
            raise ValueError(f"Cannot extract target from input: {type(input_dict)}")

        if hasattr(label_dict, 'target'):
            label_target = label_dict.target
        elif isinstance(label_dict, dict) and 'target' in label_dict:
            label_target = label_dict['target']
        else:
            raise ValueError(f"Cannot extract target from label: {type(label_dict)}")

        # Convert to numpy if needed
        if torch.is_tensor(input_target):
            input_target = input_target.cpu().numpy()
        elif not isinstance(input_target, np.ndarray):
            input_target = np.array(input_target)

        if torch.is_tensor(label_target):
            label_target = label_target.cpu().numpy()
        elif not isinstance(label_target, np.ndarray):
            label_target = np.array(label_target)

        # Concatenate input + label to get full context
        # This is needed for proper reversal at the prediction boundary
        full_gt = np.concatenate([input_target, label_target], axis=0)

        ground_truths.append(full_gt)

    print(f"Extracted {len(ground_truths)} ground truth sequences")

    print("Generating preconditioned model predictions...")
    precond_forecast_it = precond_predictor.predict(
        get_inputs(test_data_list),
        num_samples=100,
    )
    precond_forecasts = list(tqdm(precond_forecast_it, total=len(test_data_list), desc="Precond model"))

    # Check we got same number of forecasts and ground truths
    if len(precond_forecasts) != len(ground_truths):
        raise ValueError(
            f"Mismatch in counts: forecasts={len(precond_forecasts)}, "
            f"ground_truths={len(ground_truths)}"
        )

    if len(precond_forecasts) == 0:
        raise ValueError("No forecasts were generated.")

    print(f"Generated {len(precond_forecasts)} forecasts")

    # Create GT-context-reversed forecasts
    print("Creating GT-context-reversed forecasts...")
    gt_reversed_forecasts = []

    for precond_fc, full_gt in tqdm(
        zip(precond_forecasts, ground_truths),
        total=len(precond_forecasts),
        desc="GT reversal"
    ):
        # Get samples from forecast
        # Forecast objects have .samples attribute with shape [num_samples, prediction_length, ...]
        precond_samples = precond_fc.samples  # [num_samples, pred_len]

        # Calculate input length from full ground truth and prediction length
        pred_len = precond_samples.shape[-1]
        input_len = len(full_gt) - pred_len

        # Apply GT context reversal for each sample
        gt_reversed_samples = reverse_precondition_with_gt_context(
            precond_predictions=precond_samples,
            full_ground_truth=full_gt,
            input_length=input_len,
            coeffs=precondition_coeffs,
        )

        # Create a new forecast object with GT-reversed samples
        import copy
        gt_reversed_fc = copy.copy(precond_fc)
        gt_reversed_fc.samples = gt_reversed_samples

        gt_reversed_forecasts.append(gt_reversed_fc)

    print(f"Created {len(gt_reversed_forecasts)} GT-context-reversed forecasts")

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

    # Evaluate GT-context-reversed forecasts
    res = evaluate_forecasts(
        forecasts=gt_reversed_forecasts,
        test_data=test_data,
        metrics=metrics,
        batch_size=batch_size,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=seasonality,
    )

    return res


@hydra.main(version_base="1.3", config_path="conf/eval", config_name="default_precond_gt")
def main(cfg: DictConfig):
    # Set display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.options.display.float_format = "{:.3f}".format

    # Load test data
    test_data, metadata = call(cfg.data)
    batch_size = cfg.batch_size

    # Get preconditioning parameters from model config
    precondition_type = cfg.model.get("precondition_type", "chebyshev")
    precondition_degree = cfg.model.get("precondition_degree", 5)

    print(f"Ground Truth Context Evaluation Configuration:")
    print(f"  Precond model: {cfg.model._target_}")
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
        # Instantiate preconditioned model (with reverse_output=False)
        print("Loading preconditioned model...")
        precond_model = call(cfg.model, _partial_=True, _convert_="all")(
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
        )

        try:
            precond_predictor = precond_model.create_predictor(batch_size, cfg.device)

            # Evaluate with ground truth context
            res = evaluate_gt_context_model(
                precond_predictor=precond_predictor,
                test_data=test_data,
                precondition_coeffs=coeffs,
                batch_size=batch_size,
                seasonality=get_seasonality(metadata.freq),
            )

            print("\n" + "="*80)
            print("GROUND TRUTH CONTEXT EVALUATION METRICS")
            print("="*80)
            print(res)
            print("="*80 + "\n")

            # Save results
            output_dir = HydraConfig.get().runtime.output_dir
            writer = SummaryWriter(log_dir=output_dir)
            for name, metric in res.to_dict("records")[0].items():
                writer.add_scalar(f"{metadata.split}_metrics_gt_context/{name}", metric)
            writer.close()

            # Also save as CSV
            csv_path = f"{output_dir}/metrics_gt_context.csv"
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
