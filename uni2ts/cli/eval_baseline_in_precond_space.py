#!/usr/bin/env python3
"""
Evaluate BASELINE model predictions in PRECONDITIONED SPACE.

This script:
1. Loads a BASELINE (non-preconditioned) model
2. Generates predictions in original space
3. Applies preconditioning to BOTH predictions and ground truth
4. Computes error metrics in the preconditioned/transformed space

This allows fair comparison between:
- Baseline model (trained on original data, predictions preconditioned post-hoc)
- Preconditioned model (trained on preconditioned data, predictions already in precond space)

Usage:
    python -m cli.eval_baseline_in_precond_space \
        model=moirai_1.1_R_small \
        model.patch_size=32 \
        model.context_length=1000 \
        precond_type=chebyshev \
        precond_degree=5 \
        data=monash_cached \
        data.dataset_name=m1_monthly \
        data.prediction_length=18

Or with custom checkpoint:
    python -m cli.eval_baseline_in_precond_space \
        model=moirai_lightning_ckpt \
        model.checkpoint_path=/path/to/baseline.ckpt \
        model.patch_size=32 \
        precond_type=chebyshev \
        precond_degree=5 \
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
from tqdm import tqdm

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.transform import PolynomialPrecondition
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
from gluonts.model.forecast import SampleForecast


def apply_preconditioning_to_array(
    data: np.ndarray,
    precondition_type: str,
    precondition_degree: int,
) -> np.ndarray:
    """
    Apply preconditioning to a numpy array.

    Args:
        data: Array of shape (time_steps,) or (variates, time_steps)
        precondition_type: "chebyshev" or "legendre"
        precondition_degree: Polynomial degree

    Returns:
        Preconditioned array of same shape
    """
    preconditioner = PolynomialPrecondition(
        polynomial_type=precondition_type,
        degree=precondition_degree,
        target_field="target",
        enabled=True,
        store_original=False,
    )

    # Create a dummy data dict for the transformation
    if data.ndim == 1:
        # Univariate: shape (time_steps,) -> (1, time_steps)
        data_2d = data[np.newaxis, :]
    else:
        # Multivariate: already (variates, time_steps)
        data_2d = data

    data_dict = {"target": data_2d}

    # Apply transformation
    precond_dict = preconditioner(data_dict.copy())

    # Extract result
    precond_data = precond_dict["target"]

    # Return in original shape
    if data.ndim == 1:
        return precond_data[0, :]
    else:
        return precond_data


@hydra.main(version_base="1.3", config_path="conf/eval", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation function for baseline models in preconditioned space.
    """
    # Set pandas display options to avoid truncation (IMPORTANT for metric parsing)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.options.display.float_format = "{:.3f}".format

    print("=" * 80)
    print("Baseline Model Evaluation in Preconditioned Space")
    print("=" * 80)
    print()

    # Get preconditioning configuration
    precond_type = cfg.get("precond_type", "chebyshev")
    precond_degree = cfg.get("precond_degree", 5)

    print(f"Preconditioning Configuration:")
    print(f"  Type: {precond_type}")
    print(f"  Degree: {precond_degree}")
    print()

    # Load test data and metadata
    print(f"Loading test data: {cfg.data.dataset_name}")
    print(f"  Prediction length: {cfg.data.prediction_length}")

    # Load data (returns tuple: test_data, metadata)
    test_data_input, metadata = call(cfg.data)

    print(f"  Target dim: {metadata.target_dim}")
    print(f"  Feat dynamic real dim: {metadata.feat_dynamic_real_dim}")
    print(f"  Past feat dynamic real dim: {metadata.past_feat_dynamic_real_dim}")
    print()

    # Load model (BASELINE model without preconditioning)
    # Use _partial_=True to create a partial function, then call with metadata
    print(f"Loading baseline model: {cfg.model._target_}")
    print(f"  Patch size: {cfg.model.patch_size}")
    print(f"  Context length: {cfg.model.context_length}")

    if hasattr(cfg.model, "checkpoint_path"):
        print(f"  Checkpoint: {cfg.model.checkpoint_path}")

    model = call(cfg.model, _partial_=True, _convert_="all")(
        prediction_length=metadata.prediction_length,
        target_dim=metadata.target_dim,
        feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
        past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
    )
    model.eval()
    print()

    # Generate forecasts from BASELINE model (in original space)
    print("Generating forecasts from baseline model...")
    predictor = model.create_predictor(
        batch_size=cfg.get("batch_size", 32),
        device=cfg.get("device", "auto"),
    )
    num_samples = cfg.get("num_samples", 100)

    forecast_it = predictor.predict(test_data_input.input)

    # Collect forecasts - convert to list first to know total
    forecast_list = list(forecast_it)
    print(f"Generated {len(forecast_list)} forecasts")
    print()

    # Apply preconditioning to predictions and ground truth
    print("Applying preconditioning to both predictions and ground truth...")

    # Create preconditioned test data and forecasts
    preconditioned_inputs = []
    preconditioned_labels = []
    preconditioned_forecasts = []

    for item, forecast in tqdm(
        zip(test_data_input, forecast_list),
        total=len(forecast_list),
        desc="Preconditioning",
    ):
        # Handle both tuple and object formats
        if isinstance(item, tuple):
            input_dict, label_dict = item
        else:
            input_dict = item.input
            label_dict = item.label

        # Get ground truth and context
        gt = label_dict["target"]  # Shape: (variates, pred_len) or (pred_len,)
        context = input_dict["target"]  # Shape: (variates, context_len) or (context_len,)

        # Concatenate context and ground truth for preconditioning
        # This ensures we apply preconditioning to the full sequence
        if gt.ndim == 1:
            full_sequence = np.concatenate([context, gt])
        else:
            full_sequence = np.concatenate([context, gt], axis=-1)

        # Precondition the full sequence
        full_precond = apply_preconditioning_to_array(
            full_sequence, precond_type, precond_degree
        )

        # Split back into context and label
        context_len = context.shape[-1]
        if full_precond.ndim == 1:
            precond_context = full_precond[:context_len]
            precond_label = full_precond[context_len:]
        else:
            precond_context = full_precond[:, :context_len]
            precond_label = full_precond[:, context_len:]

        # Store preconditioned context and label
        new_input = dict(input_dict)
        new_input["target"] = precond_context

        new_label = dict(label_dict)
        new_label["target"] = precond_label

        preconditioned_inputs.append(new_input)
        preconditioned_labels.append(new_label)

        # Precondition forecast samples
        # forecast.samples shape: (num_samples, variates, pred_len) or (num_samples, pred_len)
        precond_samples_list = []
        for sample in forecast.samples:
            # For each sample, we need to precondition context + sample
            if sample.ndim == 1:
                sample_full_sequence = np.concatenate([context, sample])
            else:
                sample_full_sequence = np.concatenate([context, sample], axis=-1)

            sample_full_precond = apply_preconditioning_to_array(
                sample_full_sequence, precond_type, precond_degree
            )

            # Extract only the prediction part
            if sample_full_precond.ndim == 1:
                sample_precond = sample_full_precond[context_len:]
            else:
                sample_precond = sample_full_precond[:, context_len:]

            precond_samples_list.append(sample_precond)

        # Stack samples back
        precond_samples = np.stack(precond_samples_list, axis=0)

        # Create new forecast with preconditioned samples
        precond_forecast = SampleForecast(
            samples=precond_samples,
            start_date=forecast.start_date,
            item_id=forecast.item_id,
        )

        preconditioned_forecasts.append(precond_forecast)

    print(f"Preconditioned {len(preconditioned_forecasts)} forecasts and ground truth")
    print()

    # Create preconditioned test data structure
    class PreconditionedTestData:
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        @property
        def input(self):
            return iter(self.inputs)

        @property
        def label(self):
            return iter(self.labels)

        def __iter__(self):
            for inp, lab in zip(self.inputs, self.labels):
                yield type('TestDataEntry', (), {'input': inp, 'label': lab})()

        def __len__(self):
            return len(self.inputs)

    preconditioned_test_data = PreconditionedTestData(
        preconditioned_inputs, preconditioned_labels
    )

    # Compute metrics using evaluate_forecasts (same as eval_precond_space.py)
    print("Computing metrics in preconditioned space...")
    print("=" * 80)

    # Define metrics (same as eval_precond_space.py)
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

    # Use evaluate_forecasts with preconditioned test data
    res = evaluate_forecasts(
        forecasts=preconditioned_forecasts,
        test_data=preconditioned_test_data,
        metrics=metrics,
        batch_size=cfg.get("batch_size", 32),
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=get_seasonality(metadata.freq),
    )

    print("\n" + "="*80)
    print("EVALUATION METRICS (BASELINE IN PRECONDITIONED SPACE)")
    print("="*80)
    print(res)
    print("="*80 + "\n")

    # Save results
    output_dir = HydraConfig.get().runtime.output_dir
    results_file = f"{output_dir}/metrics_baseline_in_precond_space.csv"

    res.to_csv(results_file, index=False)

    print(f"Results saved to: {results_file}")
    print()


if __name__ == "__main__":
    main()
