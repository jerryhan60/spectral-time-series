#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution

from uni2ts.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedNLLLoss,
    PackedPointLoss,
)
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear
from uni2ts.optim import SchedulerType, get_scheduler
from uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DefaultPatchSizeConstraints,
    DummyValueImputation,
    ExtendMask,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    PolynomialPrecondition,
    SampleDimension,
    SelectFields,
    SequencifyField,
    Transformation,
)

from uni2ts.module.learnable_precondition import LearnablePrecondition
from .module import MoiraiModule


class MoiraiPretrain(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
    }

    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[MoiraiModule] = None,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        enable_preconditioning: bool = False,
        precondition_type: str = "chebyshev",
        precondition_degree: int = 5,
        loss_in_original_space: bool = False,
        learnable_preconditioning: bool = False,
        precondition_weight_decay: float = 0.0,
        reversal_loss_weight: float = 0.1,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = MoiraiModule(**module_kwargs) if module is None else module
        # Initialize learnable preconditioning module
        # If learnable_preconditioning is True, gradients are enabled.
        # If False, we still create it (frozen) to use for reversal in loss_in_original_space if needed.
        self.preconditioner_module = LearnablePrecondition(
            degree=precondition_degree,
            polynomial_type=precondition_type,
        )
        if not learnable_preconditioning:
            self.preconditioner_module.coeffs.requires_grad = False

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        """
        Redirects to the forward function of MoiraiModule.

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
        :param patch_size: patch size for each token
        :return: predictive distribution
        """
        distr = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size,
        )
        return distr

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Implements LightningModule training_step. Logs training loss.

        :param batch: batched inputs
        :param batch_idx: index of current batch
        :return: training loss for current batch
        """
        # Handle learnable preconditioning
        target_orig = batch["target"]
        if self.hparams.learnable_preconditioning:
            # Apply preconditioning on-the-fly
            # Flatten target: [batch, seq_len, patch_size] -> [batch, seq_len * patch_size, 1]
            b, s, p = target_orig.shape
            target_flat = target_orig.view(b, s * p, 1)
            
            # Expand sample_id: [batch, seq_len] -> [batch, seq_len * patch_size]
            sample_id = batch["sample_id"]
            sample_id_flat = sample_id.unsqueeze(-1).expand(-1, -1, p).reshape(b, s * p)
            
            # Apply forward
            target_precond_flat = self.preconditioner_module(target_flat, sample_id=sample_id_flat)
            target_precond = target_precond_flat.view(b, s, p)
            
            # Update batch for forward pass
            batch["target"] = target_precond
        
        distr = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )
        
        # Handle loss calculation
        #
        # For NLL loss with pure location shift (preconditioning), loss in original space
        # equals loss in preconditioned space due to mathematical equivalence:
        # log_prob(y_orig | distr_shifted) = log_prob(y_orig + correction | distr) = log_prob(y_precond | distr)
        #
        # This means we can compute the loss directly with the preconditioned target,
        # avoiding TransformedDistribution which caused in-place modification issues
        # with the Mixture distribution's internal caching.
        nll_loss = self.hparams.loss_func(
            pred=distr,
            **{
                field: batch[field]
                for field in [
                    "target",
                    "prediction_mask",
                    "observed_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )

        # Compute hybrid loss for learnable preconditioning with loss_in_original_space
        # This adds a reconstruction loss term that penalizes coefficients causing unstable reversal
        if self.hparams.learnable_preconditioning and self.hparams.loss_in_original_space:
            # target_precond is in batch["target"], target_orig was saved earlier
            target_precond = batch["target"]
            b, s, p = target_precond.shape

            sample_id = batch["sample_id"]
            sample_id_flat = sample_id.unsqueeze(-1).expand(-1, -1, p).reshape(b, s * p)

            # Get model's prediction in preconditioned space
            # Use the distribution mean as the point prediction
            pred_precond = distr.mean  # [batch, seq_len, patch_size]

            # Flatten for reversal
            pred_precond_flat = pred_precond.view(b, s * p, 1)

            # Reverse the model's prediction to original space
            pred_orig_flat = self.preconditioner_module.reverse(
                pred_precond_flat, sample_id=sample_id_flat
            )
            pred_orig = pred_orig_flat.view(b, s, p)

            # Compute reconstruction loss: MSE between reversed prediction and original ground truth
            # This measures end-to-end forecasting error in original space
            # Coefficients that amplify forecast errors will have higher reconstruction loss
            prediction_mask = batch["prediction_mask"]  # [batch, seq_len]
            observed_mask = batch["observed_mask"]  # [batch, seq_len, patch_size]

            # Combine masks: we care about observed values in the prediction window
            combined_mask = prediction_mask.unsqueeze(-1) & observed_mask.bool()

            # Compute masked MSE
            diff = (pred_orig - target_orig) ** 2
            if combined_mask.any():
                reconstruction_loss = diff[combined_mask].mean()
            else:
                reconstruction_loss = diff.mean()

            # Combine losses
            loss = nll_loss + self.hparams.reversal_loss_weight * reconstruction_loss

            # Log individual components
            batch_size = (
                batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
            )
            self.log(
                "train/nll_loss",
                nll_loss,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
            self.log(
                "train/reconstruction_loss",
                reconstruction_loss,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
        else:
            loss = nll_loss

        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"train/{self.hparams.loss_func.__class__.__name__}",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Implements LightningModule validation_step. Logs validation loss and additional metrics from val_metric.

        :param batch:
        :param batch_idx:
        :param dataloader_idx:
        :return: validation loss for current batch
        """
        # Handle learnable preconditioning
        target_orig = batch["target"]
        if self.hparams.learnable_preconditioning:
            b, s, p = target_orig.shape
            target_flat = target_orig.view(b, s * p, 1)
            sample_id = batch["sample_id"]
            sample_id_flat = sample_id.unsqueeze(-1).expand(-1, -1, p).reshape(b, s * p)
            target_precond_flat = self.preconditioner_module(target_flat, sample_id=sample_id_flat)
            target_precond = target_precond_flat.view(b, s, p)
            batch["target"] = target_precond
            
        distr = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]}
        )
        
        # For NLL loss with pure location shift (preconditioning), loss in original space
        # equals loss in preconditioned space due to mathematical equivalence:
        # log_prob(y_orig | distr_shifted) = log_prob(y_precond | distr)
        distr_for_loss = distr
        target_for_metric = batch["target"]

        nll_loss = self.hparams.loss_func(
            pred=distr_for_loss,
            **{
                field: batch[field] if field != "target" else target_for_metric
                for field in [
                    "target",
                    "prediction_mask",
                    "observed_mask",
                    "sample_id",
                    "variate_id",
                ]
            },
        )

        # Compute hybrid loss for learnable preconditioning with loss_in_original_space
        if self.hparams.learnable_preconditioning and self.hparams.loss_in_original_space:
            target_precond = batch["target"]
            b, s, p = target_precond.shape

            sample_id = batch["sample_id"]
            sample_id_flat = sample_id.unsqueeze(-1).expand(-1, -1, p).reshape(b, s * p)

            # Get model's prediction in preconditioned space
            pred_precond = distr.mean  # [batch, seq_len, patch_size]

            # Flatten for reversal
            pred_precond_flat = pred_precond.view(b, s * p, 1)

            # Reverse the model's prediction to original space
            pred_orig_flat = self.preconditioner_module.reverse(
                pred_precond_flat, sample_id=sample_id_flat
            )
            pred_orig = pred_orig_flat.view(b, s, p)

            # Compute reconstruction loss: MSE between reversed prediction and original ground truth
            prediction_mask = batch["prediction_mask"]
            observed_mask = batch["observed_mask"]
            combined_mask = prediction_mask.unsqueeze(-1) & observed_mask.bool()

            diff = (pred_orig - target_orig) ** 2
            if combined_mask.any():
                reconstruction_loss = diff[combined_mask].mean()
            else:
                reconstruction_loss = diff.mean()

            val_loss = nll_loss + self.hparams.reversal_loss_weight * reconstruction_loss

            # Log individual components
            batch_size = (
                batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
            )
            self.log(
                "val/nll_loss",
                nll_loss,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
            self.log(
                "val/reconstruction_loss",
                reconstruction_loss,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
        else:
            val_loss = nll_loss

        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"val/{self.hparams.loss_func.__class__.__name__}",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        if self.hparams.val_metric is not None:
            val_metrics = (
                self.hparams.val_metric
                if isinstance(self.hparams.val_metric, list)
                else [self.hparams.val_metric]
            )
            for metric_func in val_metrics:
                if isinstance(metric_func, PackedPointLoss):
                    # For point loss, we need to sample/mean from the distribution
                    # If we are in original space, we should sample from distr_for_loss (which is shifted)
                    pred = distr_for_loss.sample(torch.Size((self.hparams.num_samples,)))
                    pred = torch.median(pred, dim=0).values
                elif isinstance(metric_func, PackedDistributionLoss):
                    pred = distr_for_loss
                else:
                    raise ValueError(f"Unsupported loss function: {metric_func}")

                metric = metric_func(
                    pred=pred,
                    **{
                        field: batch[field] if field != "target" else target_for_metric
                        for field in [
                            "target",
                            "prediction_mask",
                            "observed_mask",
                            "sample_id",
                            "variate_id",
                        ]
                    },
                )

                self.log(
                    f"val/{metric_func.__class__.__name__}",
                    metric,
                    on_step=self.hparams.log_on_step,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    rank_zero_only=True,
                )

        return val_loss

    def configure_optimizers(self) -> dict:
        """
        Implements LightningModule configure_optimizers which defines the configuration of optimizer and learning rate
        scheduler.

        :return: dictionary of optimizers and learning rate schedulers
        """
        decay = set()
        no_decay = set()

        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            nn.Linear,
            # Add LearnablePrecondition to whitelist for decay? Or no decay?
            # Usually coefficients are small, maybe decay is good.
            # But they are parameters.
            LearnablePrecondition,
        )
        blacklist_params = (
            BinaryAttentionBias,
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)
                # Handle LearnablePrecondition coeffs
                elif isinstance(m, LearnablePrecondition) and pn.endswith("coeffs"):
                     # Maybe no decay for coefficients? Or decay?
                     # Let's put them in no_decay for now to be safe, or decay.
                     # If we put in decay, we need to ensure it's in whitelist.
                     # 'coeffs' is not 'weight'.
                     decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }

    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        """
        Get a dictionary of Transforms, with a default Transform as defined:
        SampleDimension: Subsample the variate dimension of a time series
        GetPatchSize: Get patch size for a given time series
        PatchCrop: Perform cropping on the time series
        PackFields: Pack each feature columns, including 'target' and 'past_feat_dynamic_real'.
        AddObservedMask: Add the observed_mask feature
        ImputeTimeSeries: Imputes missing values with 0
        Patchify: Perform patching
        AddVariateIndex: Add variate_id feature
        AddTimeIndex: Add time_id feature
        MaskedPrediction: Specify the task,
            i.e., sample the total input length, as well as sample the proportion of look-back window and prediction window length.
        ExtendMask: Add an auxiliary mask.
        FlatPackCollection: Pack/Merge along 'variate_id, time_id, prediction_mask, observed_mask, and target' dimensions.
        FlatPackFields: Pack/Merge 'target'.
        SequencifyField: sequencify the 'patch_size' field.
        SelectFields: Output the data of predefined fields

        :return: defaultdict with default Transform
        """

        def default_train_transform():
            # Start with optional preconditioning transform
            # Disable static preconditioning if learnable is enabled
            static_precond_enabled = self.hparams.enable_preconditioning and not self.hparams.learnable_preconditioning
            
            transform = PolynomialPrecondition(
                polynomial_type=self.hparams.precondition_type,
                degree=self.hparams.precondition_degree,
                target_field="target",
                enabled=static_precond_enabled,
                store_original=False,
            )

            # Add remaining transforms
            transform = (
                transform
                + SampleDimension(
                    max_dim=self.hparams.max_dim,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=self.module.patch_sizes,
                    patch_size_constraints=DefaultPatchSizeConstraints(),
                    offset=True,
                )
                + PatchCrop(
                    min_time_patches=self.hparams.min_patches,
                    max_patches=self.module.max_seq_len,
                    will_flatten=True,
                    offset=True,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + PackFields(
                    output_field="past_feat_dynamic_real",
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=False,
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=max(self.module.patch_sizes),
                    fields=("target", "observed_mask"),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=True,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + MaskedPrediction(
                    min_mask_ratio=self.hparams.min_mask_ratio,
                    max_mask_ratio=self.hparams.max_mask_ratio,
                    target_field="target",
                    truncate_fields=("variate_id", "time_id", "observed_mask"),
                    optional_truncate_fields=("past_feat_dynamic_real",),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + ExtendMask(
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )

            return transform

        return defaultdict(lambda: default_train_transform)
