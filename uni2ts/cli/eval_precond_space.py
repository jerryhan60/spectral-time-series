#!/usr/bin/env python3
"""
Evaluation script for preconditioned models that evaluates in the TRANSFORMED SPACE.

This script:
1. Loads a preconditioned model (with reverse_output=False)
2. Applies preconditioning to ground truth data
3. Compares predictions vs ground truth in the preconditioned space
4. Outputs metrics for the transformed space evaluation

Usage:
    python -m cli.eval_precond_space \
        model.checkpoint_path=/path/to/ckpt \
        model.patch_size=32 \
        model.context_length=1000 \
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


def precondition_ground_truth(
    test_data,
    precondition_type: str = "chebyshev",
    precondition_degree: int = 5,
):
    """
    Apply preconditioning to ground truth data.

    Args:
        test_data: GluonTS TestData object
        precondition_type: Type of polynomial ("chebyshev" or "legendre")
        precondition_degree: Degree of polynomial

    Returns:
        preconditioned_test_data: TestData with preconditioned targets
        valid_indices: List of indices that passed preconditioning (for tracking which samples to predict)
    """
    preconditioner = PolynomialPrecondition(
        polynomial_type=precondition_type,
        degree=precondition_degree,
        target_field="target",
        enabled=True,
        store_original=False,
    )

    # Apply preconditioning to each item
    preconditioned_input = []
    preconditioned_label = []
    valid_indices = []  # Track which indices are valid

    # Track statistics for debugging
    total_items = 0
    skipped_items = 0
    nan_in_original = 0
    nan_after_precond = 0

    for idx, item in enumerate(tqdm(test_data, desc="Preconditioning ground truth")):
        total_items += 1

        # Handle both tuple format (input, label) and object format
        if isinstance(item, tuple):
            input_dict, label_dict = item
        else:
            input_dict = item.input
            label_dict = item.label

        # Check for NaN in original data
        orig_input_target = input_dict["target"]
        orig_label_target = label_dict["target"]

        has_nan = np.isnan(orig_input_target).any() or np.isnan(orig_label_target).any()

        if has_nan:
            nan_in_original += 1
            # DON'T skip - preconditioning should preserve NaN
            # The evaluation framework will handle NaN using masked arrays

        # Precondition the full target (input + label)
        # NaN values will be preserved through preconditioning
        full_target = np.concatenate([orig_input_target, orig_label_target], axis=0)

        data_entry = {"target": full_target}
        preconditioned_entry = preconditioner(data_entry)
        preconditioned_full = preconditioned_entry["target"]

        # Only skip if preconditioning introduced NEW NaN that weren't there before
        # This can happen if preconditioning has numerical instability
        new_nan = np.isnan(preconditioned_full) & ~np.isnan(full_target)
        if new_nan.any():
            nan_after_precond += 1
            skipped_items += 1
            continue

        # Split back into input and label
        input_len = len(orig_input_target)
        preconditioned_input_target = preconditioned_full[:input_len]
        preconditioned_label_target = preconditioned_full[input_len:]

        # Create new input/label dicts
        new_input = dict(input_dict)
        new_input["target"] = preconditioned_input_target

        new_label = dict(label_dict)
        new_label["target"] = preconditioned_label_target

        # Store as tuple (input, label) and record the valid index
        preconditioned_input.append(new_input)
        preconditioned_label.append(new_label)
        valid_indices.append(idx)

    # Print statistics
    print(f"Preconditioning statistics:")
    print(f"  Total items: {total_items}")
    print(f"  Items with NaN in original data: {nan_in_original} (will use masked arrays)")
    print(f"  Items skipped (new NaN after preconditioning): {skipped_items}")
    print(f"  Valid items for evaluation: {len(preconditioned_input)}")

    # Check if we have any valid data left
    if len(preconditioned_input) == 0:
        raise ValueError(
            f"All {total_items} items were skipped during preconditioning. "
            f"Preconditioning introduced NaN in all samples (numerical instability). "
            f"Dataset cannot be evaluated in preconditioned space."
        )

    # Create new TestData-like structure
    # We'll use a simple class to mimic TestData interface
    class PreconditionedTestData:
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        @property
        def input(self):
            """Return iterator over input data entries."""
            return iter(self.inputs)

        @property
        def label(self):
            """Return iterator over label data entries."""
            return iter(self.labels)

        def __iter__(self):
            for inp, lab in zip(self.inputs, self.labels):
                yield type('TestDataEntry', (), {'input': inp, 'label': lab})()

        def __len__(self):
            return len(self.inputs)

    return PreconditionedTestData(preconditioned_input, preconditioned_label), valid_indices


def evaluate_in_preconditioned_space(
    predictor,
    test_data,
    precondition_type: str,
    precondition_degree: int,
    batch_size: int = 100,
    seasonality: int = None,
):
    """
    Evaluate model in preconditioned space.

    The evaluation flow is:
    1. Feed ORIGINAL inputs to model (model applies preconditioning internally)
    2. Model outputs predictions in PRECONDITIONED space (reverse_output=False)
    3. Precondition ground truth labels to match prediction space
    4. Compare preconditioned predictions vs preconditioned ground truth

    Args:
        predictor: Model predictor (with reverse_output=False)
        test_data: Original test data
        precondition_type: Type of preconditioning used
        precondition_degree: Degree of preconditioning used
        batch_size: Batch size for prediction
        seasonality: Seasonality for MASE metric

    Returns:
        DataFrame with metrics
    """
    print("Applying preconditioning to ground truth labels only...")
    preconditioned_test_data, valid_indices = precondition_ground_truth(
        test_data,
        precondition_type=precondition_type,
        precondition_degree=precondition_degree,
    )

    print("Generating predictions (in preconditioned space)...")
    # Convert to list to avoid iterator exhaustion issues
    test_data_list = list(test_data)

    # Only process samples that passed preconditioning (valid_indices)
    # This ensures predictions align with preconditioned ground truth
    valid_test_data = [test_data_list[i] for i in valid_indices]

    print(f"Generating predictions for {len(valid_test_data)} valid samples (out of {len(test_data_list)} total)...")

    # Get predictions using ORIGINAL inputs (not preconditioned) for VALID samples only
    # The model will apply preconditioning internally and output in preconditioned space
    # Create a proper generator that handles both tuple and object formats
    def get_inputs(test_data_list):
        for item in test_data_list:
            if isinstance(item, tuple):
                yield item[0]
            elif hasattr(item, 'input'):
                yield item.input
            else:
                raise ValueError(f"Unknown test data format: {type(item)}")

    forecast_it = predictor.predict(
        get_inputs(valid_test_data),  # Only predict for valid samples
        num_samples=100,
    )

    # Collect forecasts
    print("Collecting predictions and preconditioned ground truth...")
    forecast_list = list(tqdm(forecast_it, total=len(valid_test_data), desc="Processing forecasts"))

    # Check if we actually got any data
    if len(forecast_list) == 0:
        raise ValueError("No forecasts were generated. Check if the test data is empty or if there's an issue with the predictor.")

    print(f"Generated {len(forecast_list)} forecasts for {len(valid_test_data)} valid samples")

    # Compute metrics using the standard evaluate_forecasts function
    print("Computing metrics in preconditioned space...")

    # Define metrics to compute (instantiate each metric)
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
        forecasts=forecast_list,
        test_data=preconditioned_test_data,
        metrics=metrics,
        batch_size=batch_size,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=seasonality,
    )

    return res


@hydra.main(version_base="1.3", config_path="conf/eval", config_name="default")
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

    print(f"Preconditioning parameters:")
    print(f"  Type: {precondition_type}")
    print(f"  Degree: {precondition_degree}")

    while True:
        # Instantiate model with reverse_output=False
        model = call(cfg.model, _partial_=True, _convert_="all")(
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
        )

        try:
            predictor = model.create_predictor(batch_size, cfg.device)

            # Evaluate in preconditioned space
            res = evaluate_in_preconditioned_space(
                predictor=predictor,
                test_data=test_data,
                precondition_type=precondition_type,
                precondition_degree=precondition_degree,
                batch_size=batch_size,
                seasonality=get_seasonality(metadata.freq),
            )

            print("\n" + "="*80)
            print("EVALUATION METRICS (IN PRECONDITIONED SPACE)")
            print("="*80)
            print(res)
            print("="*80 + "\n")

            # Save results
            output_dir = HydraConfig.get().runtime.output_dir
            writer = SummaryWriter(log_dir=output_dir)
            for name, metric in res.to_dict("records")[0].items():
                writer.add_scalar(f"{metadata.split}_metrics_precond_space/{name}", metric)
            writer.close()

            # Also save as CSV
            csv_path = f"{output_dir}/metrics_precond_space.csv"
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
