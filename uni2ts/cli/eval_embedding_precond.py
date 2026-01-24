#!/usr/bin/env python3
"""
Evaluation script for embedding-preconditioned Moirai models.

For embedding-level preconditioning, the forward/reverse operations happen
INSIDE the model (MoiraiModule), so evaluation is straightforward:
1. Load checkpoint (contains all preconditioning parameters)
2. Run standard evaluation
3. Metrics compare model predictions to ground truth in original space

Key difference from data-level preconditioning:
- Data-level: Needs special handling to reverse predictions after model
- Embedding-level: Predictions are already in original space (if reversal enabled)

Usage:
    python -m cli.eval_embedding_precond \
        checkpoint_path=/path/to/checkpoint.ckpt \
        patch_size=32 \
        context_length=1000 \
        data=monash_cached \
        data.dataset_name=m1_monthly
"""

import hydra
import pandas as pd
import torch
from gluonts.time_feature import get_seasonality
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.eval_util.evaluation import evaluate_model
from uni2ts.model.moirai.forecast import MoiraiForecast


def load_embedding_precond_model(
    checkpoint_path: str,
    prediction_length: int,
    target_dim: int,
    feat_dynamic_real_dim: int,
    past_feat_dynamic_real_dim: int,
    context_length: int,
    patch_size: int,
    num_samples: int = 100,
):
    """
    Load an embedding-preconditioned model from checkpoint.

    The embedding preconditioning parameters (enable_embedding_preconditioning,
    embedding_precondition_type, embedding_precondition_degree, embedding_precondition_reverse)
    are stored in the checkpoint and automatically loaded.

    Args:
        checkpoint_path: Path to Lightning checkpoint (.ckpt)
        prediction_length: Forecast horizon
        target_dim: Number of target variables
        feat_dynamic_real_dim: Number of dynamic real features
        past_feat_dynamic_real_dim: Number of past dynamic real features
        context_length: Input context window size
        patch_size: Patch size for tokenization
        num_samples: Number of samples for probabilistic forecasting

    Returns:
        MoiraiForecast model ready for evaluation
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract module state dict (MoiraiModule weights)
    state_dict = checkpoint["state_dict"]
    module_state = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
        if k.startswith("module.")
    }

    # Extract hyperparameters to reconstruct MoiraiModule
    hparams = checkpoint.get("hyper_parameters", {})
    module_kwargs = hparams.get("module_kwargs", {})

    # Handle nested instantiation if needed
    if "_target_" in module_kwargs:
        # Remove hydra target, we'll construct directly
        module_kwargs = {k: v for k, v in module_kwargs.items() if not k.startswith("_")}

    # Instantiate distribution output if needed
    if "distr_output" in module_kwargs and isinstance(module_kwargs["distr_output"], dict):
        from hydra.utils import instantiate
        module_kwargs["distr_output"] = instantiate(module_kwargs["distr_output"], _convert_="all")

    # Create MoiraiModule with embedding preconditioning parameters
    from uni2ts.model.moirai.module import MoiraiModule
    module = MoiraiModule(**module_kwargs)
    module.load_state_dict(module_state)

    # Print embedding preconditioning status
    if hasattr(module, "enable_embedding_preconditioning"):
        print(f"Embedding preconditioning enabled: {module.enable_embedding_preconditioning}")
        if module.enable_embedding_preconditioning:
            print(f"  - Reversal: {module.embedding_precondition_reverse}")
            if hasattr(module, "num_target_variates"):
                print(f"  - Num target variates: {module.num_target_variates} (None = all)")
            if module.embedding_preconditioner is not None:
                print(f"  - Degree: {module.embedding_preconditioner.degree}")
                print(f"  - Type: {module.embedding_preconditioner.polynomial_type}")

    # Create MoiraiForecast wrapper
    model = MoiraiForecast(
        prediction_length=prediction_length,
        target_dim=target_dim,
        feat_dynamic_real_dim=feat_dynamic_real_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        context_length=context_length,
        module=module,
        patch_size=patch_size,
        num_samples=num_samples,
    )

    return model


@hydra.main(version_base="1.3", config_path="conf/eval", config_name="embedding_precond")
def main(cfg: DictConfig):
    # Set display options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.options.display.float_format = "{:.3f}".format

    print("=" * 60)
    print("Embedding-Preconditioned Model Evaluation")
    print("=" * 60)

    # Load test data
    test_data, metadata = call(cfg.data)
    print(f"Dataset: {cfg.data.dataset_name if hasattr(cfg.data, 'dataset_name') else 'unknown'}")
    print(f"Prediction length: {metadata.prediction_length}")
    print(f"Target dim: {metadata.target_dim}")

    batch_size = cfg.batch_size
    while True:
        # Load model
        model = load_embedding_precond_model(
            checkpoint_path=cfg.checkpoint_path,
            prediction_length=metadata.prediction_length,
            target_dim=metadata.target_dim,
            feat_dynamic_real_dim=metadata.feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=metadata.past_feat_dynamic_real_dim,
            context_length=cfg.context_length,
            patch_size=cfg.patch_size,
            num_samples=cfg.num_samples,
        )

        metrics = instantiate(cfg.metrics, _convert_="all")

        try:
            predictor = model.create_predictor(batch_size, cfg.device)
            res = evaluate_model(
                predictor,
                test_data=test_data,
                metrics=metrics,
                batch_size=cfg.batch_size,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=get_seasonality(metadata.freq),
            )
            print("\n" + "=" * 60)
            print("Results:")
            print("=" * 60)
            print(res)

            # Write to TensorBoard
            output_dir = HydraConfig.get().runtime.output_dir
            writer = SummaryWriter(log_dir=output_dir)
            for name, metric in res.to_dict("records")[0].items():
                writer.add_scalar(f"{metadata.split}_metrics/{name}", metric)
            writer.close()
            break

        except torch.cuda.OutOfMemoryError:
            print(
                f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size//2}"
            )
            batch_size //= 2
            if batch_size < cfg.min_batch_size:
                print(
                    f"batch_size {batch_size} smaller than "
                    f"min_batch_size {cfg.min_batch_size}, ending evaluation"
                )
                break


if __name__ == "__main__":
    main()
