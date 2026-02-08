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
from functools import partial
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int

from uni2ts.loss.packed import PackedQuantileLoss, PackedQuantileMAELoss
from uni2ts.optim import SchedulerType, get_scheduler
from uni2ts.common.precondition import compute_polynomial_coefficients
from uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    ApplyRejectMask,
    CausalPredictionMask,
    CopyField,
    DummyValueImputation,
    FlatPackCollection,
    FlatPackFields,
    ImputeTimeSeries,
    Identity,
    LambdaSetFieldIfNotPresent,
    PackFields,
    Patchify,
    PatchPolynomialPrecondition,
    ResampleZScorePatchCrop,
    SampleDimension,
    SelectFields,
    Transformation,
)

from .module import Moirai2Module


def _get_patch_size(_: Any, patch_size: int) -> int:
    return patch_size


class Moirai2Pretrain(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "reject_mask",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "reject_mask": np.zeros,
    }

    def __init__(
        self,
        prefix_ratio: float,
        mask_ratio: float,
        anomaly_zscore_threshold: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        anomaly_variance_ratio_threshold: float = 0.0,
        anomaly_variance_min_count: int = 2,
        anomaly_resample_attempts: int = 5,
        patch_precondition_enabled: bool = False,
        patch_precondition_type: str = "chebyshev",
        patch_precondition_degree: int = 5,
        patch_precondition_stride: int = 1,
        patch_precondition_in_forward: bool = False,
        patch_precondition_reverse_in_loss: bool = False,
        time_precondition_reverse_in_loss: bool = False,
        time_precondition_inverse_lambda: float = 0.1,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[Moirai2Module] = None,
        loss_func: Optional[PackedQuantileLoss] = None,
        beta1: float = 0.9,
        beta2: float = 0.98,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.module = Moirai2Module(**module_kwargs) if module is None else module
        if loss_func is None:
            loss_func = PackedQuantileMAELoss(self.module.quantile_levels)
        self.save_hyperparameters(ignore=["module"])
        if self.hparams.patch_precondition_reverse_in_loss and not (
            self.hparams.patch_precondition_enabled
        ):
            raise ValueError(
                "patch_precondition_reverse_in_loss requires patch_precondition_enabled=true"
            )
        if (
            self.hparams.patch_precondition_reverse_in_loss
            and not self.hparams.patch_precondition_in_forward
        ):
            raise ValueError(
                "patch_precondition_reverse_in_loss requires patch_precondition_in_forward=true"
            )
        if (
            self.hparams.time_precondition_reverse_in_loss
            and not self.module.time_precondition_enabled
        ):
            raise ValueError(
                "time_precondition_reverse_in_loss requires module time_precondition_enabled=true"
            )
        if (
            self.module.time_precondition_inverse_enabled
            and self.hparams.time_precondition_reverse_in_loss
        ):
            raise ValueError(
                "time_precondition_inverse_enabled cannot be combined with time_precondition_reverse_in_loss"
            )
        if (
            self.hparams.time_precondition_reverse_in_loss
            and self.hparams.patch_precondition_reverse_in_loss
        ):
            raise ValueError(
                "time_precondition_reverse_in_loss is not supported with patch_precondition_reverse_in_loss"
            )
        if self.hparams.patch_precondition_stride < 1:
            raise ValueError("patch_precondition_stride must be >= 1")
        if (
            self.hparams.patch_precondition_enabled
            or self.hparams.patch_precondition_reverse_in_loss
        ):
            coeffs = compute_polynomial_coefficients(
                self.hparams.patch_precondition_type,
                self.hparams.patch_precondition_degree,
            ).astype(np.float32)
            self.register_buffer(
                "patch_precondition_coeffs",
                torch.tensor(coeffs),
                persistent=False,
            )
        else:
            self.register_buffer(
                "patch_precondition_coeffs",
                torch.empty(0),
                persistent=False,
            )
        self._median_quantile_idx = self._resolve_median_quantile_idx(
            self.module.quantile_levels
        )

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        input_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds, scaled_target = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            training_mode=True,
            input_mask=input_mask,
        )
        return preds, scaled_target

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input_mask = self.sample_patch_mask(batch["sample_id"])
        preds, scaled_target = self(
            target=batch["target"],
            observed_mask=batch["observed_mask"],
            sample_id=batch["sample_id"],
            time_id=batch["time_id"],
            variate_id=batch["variate_id"],
            prediction_mask=batch["prediction_mask"],
            input_mask=input_mask,
        )
        loc = None
        scale = None
        scaled_target_raw = scaled_target
        if (
            self.module.time_precondition_enabled
            or self.hparams.time_precondition_reverse_in_loss
            or self.hparams.patch_precondition_reverse_in_loss
        ):
            loc, scale = self.module.scaler(
                batch["target"],
                batch["observed_mask"] * ~batch["prediction_mask"].unsqueeze(-1),
                batch["sample_id"],
                batch["variate_id"],
            )
            scaled_target_raw = (batch["target"] - loc) / scale
        prefilter_mask = batch.get(
            "reject_mask",
            torch.zeros_like(batch["sample_id"], dtype=torch.bool),
        )
        postfilter_mask, _, _ = self.compute_rejection_mask(
            scaled_target_raw,
            batch["observed_mask"],
            batch["prediction_mask"],
            batch["sample_id"],
        )
        combined_reject_mask = prefilter_mask | postfilter_mask
        loss_prediction_mask = batch["prediction_mask"] & ~combined_reject_mask
        if self.hparams.time_precondition_reverse_in_loss:
            base_loss = self.multi_token_loss_time_precondition_reverse(
                preds=preds,
                target_raw=scaled_target_raw,
                target_precond=scaled_target,
                observed_mask=batch["observed_mask"],
                prediction_mask=batch["prediction_mask"],
                loss_prediction_mask=loss_prediction_mask,
                sample_id=batch["sample_id"],
                variate_id=batch["variate_id"],
                time_id=batch["time_id"],
            )
        elif self.hparams.patch_precondition_reverse_in_loss:
            base_loss = self.multi_token_loss_original(
                preds=preds,
                target_original=batch["target_original"],
                observed_mask=batch["observed_mask"],
                prediction_mask=loss_prediction_mask,
                sample_id=batch["sample_id"],
                variate_id=batch["variate_id"],
                loc=loc,
                scale=scale,
            )
        else:
            base_loss = self.multi_token_loss(
                preds=preds,
                target=scaled_target,
                observed_mask=batch["observed_mask"],
                prediction_mask=loss_prediction_mask,
                sample_id=batch["sample_id"],
                variate_id=batch["variate_id"],
            )
        loss = base_loss
        batch_size = batch["sample_id"].max(dim=1).values.sum()
        self.log(
            "train/quantile_loss",
            base_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        if (
            self.module.time_precondition_inverse_enabled
            and self.hparams.time_precondition_inverse_lambda > 0
        ):
            aux_loss = self.time_precondition_inverse_aux_loss(
                preds=preds,
                target_precond=scaled_target,
                target_raw=scaled_target_raw,
                observed_mask=batch["observed_mask"],
                prediction_mask=batch["prediction_mask"],
                loss_prediction_mask=loss_prediction_mask,
                sample_id=batch["sample_id"],
                variate_id=batch["variate_id"],
                time_id=batch["time_id"],
            )
            loss = loss + self.hparams.time_precondition_inverse_lambda * aux_loss
            self.log(
                "train/time_precondition_inverse_loss",
                aux_loss,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
        rejected, total = self.count_rejected_samples(
            combined_reject_mask, batch["sample_id"]
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
        self.log(
            "train/total_loss",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log(
            "train/rejection_rate",
            torch.tensor(rejected / max(total, 1), device=loss.device),
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        if "reject_mask" in batch:
            pre_rejected, pre_total = self.count_rejected_samples(
                prefilter_mask, batch["sample_id"]
            )
            self.log(
                "train/rejection_rate_prefilter",
                torch.tensor(pre_rejected / max(pre_total, 1), device=loss.device),
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
        return loss

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Ensure parameters without a standard weight/bias suffix (e.g. inverse coeffs)
        # are still optimized.
        missing = set(param_dict.keys()) - decay - no_decay
        if missing:
            no_decay.update(missing)
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
            SchedulerType.COSINE,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": f"train/{self.hparams.loss_func.__class__.__name__}",
                "interval": "step",
            },
        }

    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        def default_train_transform():
            return (
                SampleDimension(
                    max_dim=self.hparams.max_dim,
                    fields=("target",),
                )
                + LambdaSetFieldIfNotPresent(
                    field="patch_size",
                    get_value=partial(_get_patch_size, patch_size=self.module.patch_size),
                )
                + ResampleZScorePatchCrop(
                    max_patches=self.module.max_seq_len,
                    prefix_ratio=self.hparams.prefix_ratio,
                    zscore_threshold=self.hparams.anomaly_zscore_threshold,
                    variance_ratio_threshold=self.hparams.anomaly_variance_ratio_threshold,
                    variance_min_count=self.hparams.anomaly_variance_min_count,
                    max_attempts=self.hparams.anomaly_resample_attempts,
                    fields=("target",),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + AddObservedMask(
                    fields=("target",),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=self.module.patch_size,
                    fields=("target", "observed_mask"),
                )
                + CopyField(
                    source_field="target",
                    target_field="target_original",
                    enabled=self.hparams.patch_precondition_reverse_in_loss,
                )
                + PatchPolynomialPrecondition(
                    polynomial_type=self.hparams.patch_precondition_type,
                    degree=self.hparams.patch_precondition_degree,
                    lag_stride=self.hparams.patch_precondition_stride,
                    target_field="target",
                    enabled=self.hparams.patch_precondition_enabled,
                )
                + AddVariateIndex(
                    fields=("target",),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=False,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + CausalPredictionMask(
                    prefix_ratio=self.hparams.prefix_ratio,
                    target_field="target",
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                    allow_short=True,
                )
                + ApplyRejectMask(
                    reject_field="reject",
                    prediction_mask_field="prediction_mask",
                    reject_mask_field="reject_mask",
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackFields(
                    output_field="prediction_mask",
                    fields=("prediction_mask",),
                    feat=False,
                )
                + FlatPackFields(
                    output_field="reject_mask",
                    fields=("reject_mask",),
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    feat=True,
                )
                + (
                    FlatPackFields(
                        output_field="target_original",
                        fields=("target_original",),
                        feat=True,
                    )
                    if self.hparams.patch_precondition_reverse_in_loss
                    else Identity()
                )
                + SelectFields(fields=list(self.seq_fields))
            )

        return defaultdict(lambda: default_train_transform)

    @property
    def val_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        return self.train_transform_map

    def sample_patch_mask(
        self, sample_id: Int[torch.Tensor, "*batch seq_len"]
    ) -> Bool[torch.Tensor, "*batch seq_len"]:
        if self.hparams.mask_ratio <= 0:
            return torch.zeros_like(sample_id, dtype=torch.bool)
        rand = torch.rand(sample_id.shape, device=sample_id.device)
        return (rand < self.hparams.mask_ratio) & (sample_id > 0)

    @staticmethod
    def build_horizon_mask(
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        horizon: int,
    ) -> Bool[torch.Tensor, "*batch seq_len"]:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if sample_id.shape[1] <= horizon:
            return torch.zeros(
                sample_id.shape[0],
                max(sample_id.shape[1] - horizon, 0),
                dtype=torch.bool,
                device=sample_id.device,
            )
        return (
            (sample_id[:, :-horizon] == sample_id[:, horizon:])
            & (sample_id[:, horizon:] > 0)
            & prediction_mask[:, horizon:]
        )

    def multi_token_loss(
        self,
        preds: Float[torch.Tensor, "*batch seq_len pred"],
        target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> torch.Tensor:
        preds = rearrange(
            preds,
            "... (predict_token num_quantiles patch_size) -> ... predict_token num_quantiles patch_size",
            predict_token=self.module.num_predict_token,
            num_quantiles=self.module.num_quantiles,
            patch_size=self.module.patch_size,
        )
        losses = []
        for horizon in range(1, self.module.num_predict_token + 1):
            if target.shape[1] <= horizon:
                continue
            horizon_mask = self.build_horizon_mask(
                sample_id, prediction_mask, horizon
            )
            if not horizon_mask.any():
                continue
            pred_h = rearrange(
                preds[:, :-horizon, horizon - 1],
                "... num_quantiles patch_size -> ... (num_quantiles patch_size)",
            )
            target_h = target[:, horizon:]
            observed_h = observed_mask[:, horizon:]
            sample_id_h = sample_id[:, horizon:]
            variate_id_h = variate_id[:, horizon:]
            losses.append(
                self.hparams.loss_func(
                    pred=pred_h,
                    target=target_h,
                    prediction_mask=horizon_mask,
                    observed_mask=observed_h,
                    sample_id=sample_id_h,
                    variate_id=variate_id_h,
                )
            )
        if len(losses) == 0:
            return torch.zeros((), device=preds.device)
        return torch.stack(losses).mean()

    def multi_token_loss_original(
        self,
        preds: Float[torch.Tensor, "*batch seq_len pred"],
        target_original: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        loc: Float[torch.Tensor, "*batch seq_len 1"],
        scale: Float[torch.Tensor, "*batch seq_len 1"],
    ) -> torch.Tensor:
        target_scaled = (target_original - loc) / scale
        return self.multi_token_loss(
            preds=preds,
            target=target_scaled,
            observed_mask=observed_mask,
            prediction_mask=prediction_mask,
            sample_id=sample_id,
            variate_id=variate_id,
        )

    @staticmethod
    def _resolve_median_quantile_idx(quantile_levels: Sequence[float]) -> int:
        levels = np.asarray(quantile_levels, dtype=np.float32)
        return int(np.argmin(np.abs(levels - 0.5)))

    def _flatten_timepoints(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch time_len"],
        Int[torch.Tensor, "*batch time_len"],
        Int[torch.Tensor, "*batch time_len"],
        Int[torch.Tensor, "*batch time_len"],
    ]:
        patch_size = int(self.module.patch_size)
        flat_target = rearrange(target, "b t p -> b (t p)")
        flat_sample_id = sample_id.repeat_interleave(patch_size, dim=1)
        flat_variate_id = variate_id.repeat_interleave(patch_size, dim=1)
        offsets = torch.arange(
            patch_size, device=time_id.device, dtype=time_id.dtype
        ).repeat(time_id.shape[1])
        offsets = offsets.unsqueeze(0).expand(time_id.shape[0], -1)
        flat_time_id = time_id.repeat_interleave(patch_size, dim=1) * patch_size + offsets
        return flat_target, flat_sample_id, flat_variate_id, flat_time_id

    def _reverse_time_precondition_median_history(
        self,
        precond_full: Float[torch.Tensor, "*batch seq_len num_quantiles patch"],
        target_raw: Float[torch.Tensor, "*batch seq_len patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len num_quantiles patch"]:
        if self.module.time_precondition_coeffs.numel() == 0:
            return precond_full
        coeffs = self.module.time_precondition_coeffs.to(
            device=target_raw.device, dtype=target_raw.dtype
        )
        n = int(coeffs.numel())
        stride = int(self.module.time_precondition_stride)
        if stride < 1:
            raise ValueError("time_precondition_stride must be >= 1")

        patch_size = int(self.module.patch_size)
        precond_flat = rearrange(precond_full, "b t q p -> b (t p) q")
        raw_flat = rearrange(target_raw, "b t p -> b (t p)")
        pred_mask_flat = prediction_mask.repeat_interleave(patch_size, dim=1)
        raw_expanded = raw_flat.unsqueeze(-1).expand_as(precond_flat)
        output = torch.where(pred_mask_flat.unsqueeze(-1), precond_flat, raw_expanded)
        output_median = raw_flat.clone()

        seq_len = precond_flat.shape[1]
        min_time = n * stride
        if seq_len <= min_time:
            return rearrange(
                output, "b (t p) q -> b t q p", p=patch_size
            )

        if not pred_mask_flat[:, min_time:].any():
            return rearrange(
                output, "b (t p) q -> b t q p", p=patch_size
            )

        fast_unpacked = (
            (sample_id.max(dim=1).values <= 1).all()
            and (variate_id.max(dim=1).values <= 1).all()
        )
        if fast_unpacked:
            any_pred_t = pred_mask_flat[:, min_time:].any(dim=0)
            first_pred_offset = int(torch.nonzero(any_pred_t, as_tuple=False)[0])
            start_idx = min_time + first_pred_offset
            median_idx = int(self._median_quantile_idx)
            for t in range(start_idx, seq_len):
                mask = pred_mask_flat[:, t]
                if not mask.any():
                    continue
                weighted_sum = torch.zeros_like(output_median[:, t])
                for i in range(n):
                    shift = (i + 1) * stride
                    weighted_sum = weighted_sum + coeffs[i] * output_median[:, t - shift]
                output_median[:, t] = torch.where(
                    mask,
                    precond_flat[:, t, median_idx] - weighted_sum,
                    output_median[:, t],
                )
                output[:, t, :] = torch.where(
                    mask.unsqueeze(-1),
                    precond_flat[:, t, :] - weighted_sum.unsqueeze(-1),
                    output[:, t, :],
                )
            return rearrange(
                output, "b (t p) q -> b t q p", p=patch_size
            )

        raw_flat, flat_sample_id, flat_variate_id, flat_time_id = (
            self._flatten_timepoints(target_raw, sample_id, variate_id, time_id)
        )
        base_mask = (flat_sample_id[:, min_time:] > 0) & (
            flat_time_id[:, min_time:] >= min_time
        )
        valid_all = base_mask.clone()
        for i in range(n):
            shift = (i + 1) * stride
            left_idx = min_time - shift
            right_idx = seq_len - shift
            valid_i = (
                base_mask
                & (
                    flat_sample_id[:, min_time:]
                    == flat_sample_id[:, left_idx:right_idx]
                )
                & (
                    flat_variate_id[:, min_time:]
                    == flat_variate_id[:, left_idx:right_idx]
                )
                & (
                    flat_time_id[:, min_time:]
                    == flat_time_id[:, left_idx:right_idx] + shift
                )
            )
            valid_all = valid_all & valid_i

        median_idx = int(self._median_quantile_idx)
        any_pred_t = pred_mask_flat[:, min_time:].any(dim=0)
        first_pred_offset = int(torch.nonzero(any_pred_t, as_tuple=False)[0])
        start_idx = min_time + first_pred_offset
        for t in range(start_idx, seq_len):
            mask = valid_all[:, t - min_time] & pred_mask_flat[:, t]
            if not mask.any():
                continue
            weighted_sum = torch.zeros_like(output_median[:, t])
            for i in range(n):
                shift = (i + 1) * stride
                weighted_sum = weighted_sum + coeffs[i] * output_median[:, t - shift]
            output_median[:, t] = torch.where(
                mask,
                precond_flat[:, t, median_idx] - weighted_sum,
                output_median[:, t],
            )
            output[:, t, :] = torch.where(
                mask.unsqueeze(-1),
                precond_flat[:, t, :] - weighted_sum.unsqueeze(-1),
                output[:, t, :],
            )

        return rearrange(
            output, "b (t p) q -> b t q p", p=patch_size
        )

    def multi_token_loss_time_precondition_reverse(
        self,
        preds: Float[torch.Tensor, "*batch seq_len pred"],
        target_raw: Float[torch.Tensor, "*batch seq_len patch"],
        target_precond: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        loss_prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> torch.Tensor:
        preds = rearrange(
            preds,
            "... (predict_token num_quantiles patch_size) -> ... predict_token num_quantiles patch_size",
            predict_token=self.module.num_predict_token,
            num_quantiles=self.module.num_quantiles,
            patch_size=self.module.patch_size,
        )
        losses = []
        for horizon in range(1, self.module.num_predict_token + 1):
            if target_raw.shape[1] <= horizon:
                continue
            horizon_mask = self.build_horizon_mask(
                sample_id, loss_prediction_mask, horizon
            )
            if not horizon_mask.any():
                continue
            pred_h = preds[:, :-horizon, horizon - 1]
            precond_full = (
                target_precond.unsqueeze(-2)
                .expand(-1, -1, self.module.num_quantiles, -1)
                .clone()
            )
            mask_h = prediction_mask[:, horizon:]
            if mask_h.any():
                precond_full[:, horizon:] = torch.where(
                    mask_h.unsqueeze(-1).unsqueeze(-1),
                    pred_h,
                    precond_full[:, horizon:],
                )
            reversed_full = self._reverse_time_precondition_median_history(
                precond_full,
                target_raw,
                prediction_mask,
                sample_id,
                variate_id,
                time_id,
            )
            pred_h_reversed = rearrange(
                reversed_full[:, horizon:],
                "... num_quantiles patch_size -> ... (num_quantiles patch_size)",
            )
            target_h = target_raw[:, horizon:]
            observed_h = observed_mask[:, horizon:]
            sample_id_h = sample_id[:, horizon:]
            variate_id_h = variate_id[:, horizon:]
            losses.append(
                self.hparams.loss_func(
                    pred=pred_h_reversed,
                    target=target_h,
                    prediction_mask=horizon_mask,
                    observed_mask=observed_h,
                    sample_id=sample_id_h,
                    variate_id=variate_id_h,
                )
            )
        if len(losses) == 0:
            return torch.zeros((), device=preds.device)
        return torch.stack(losses).mean()

    def time_precondition_inverse_aux_loss(
        self,
        preds: Float[torch.Tensor, "*batch seq_len pred"],
        target_precond: Float[torch.Tensor, "*batch seq_len patch"],
        target_raw: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        loss_prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> torch.Tensor:
        if self.module.time_precondition_inverse_coeffs.numel() == 0:
            return torch.zeros((), device=preds.device)
        preds = rearrange(
            preds,
            "... (predict_token num_quantiles patch_size) -> ... predict_token num_quantiles patch_size",
            predict_token=self.module.num_predict_token,
            num_quantiles=self.module.num_quantiles,
            patch_size=self.module.patch_size,
        )
        r_true = target_precond - target_raw
        median_idx = int(self._median_quantile_idx)
        losses = []
        for horizon in range(1, self.module.num_predict_token + 1):
            if target_precond.shape[1] <= horizon:
                continue
            horizon_mask = self.build_horizon_mask(
                sample_id, loss_prediction_mask, horizon
            )
            if not horizon_mask.any():
                continue
            pred_h = preds[:, :-horizon, horizon - 1, median_idx, :]
            z_mixed = target_precond.clone()
            mask_h = prediction_mask[:, horizon:]
            if mask_h.any():
                z_mixed[:, horizon:] = torch.where(
                    mask_h.unsqueeze(-1),
                    pred_h,
                    z_mixed[:, horizon:],
                )
            r_hat, valid_mask = self.module._apply_time_precondition_inverse_fir(
                z_mixed, sample_id, variate_id, time_id
            )
            diff = r_hat[:, horizon:] - r_true[:, horizon:]
            mask = (
                horizon_mask.unsqueeze(-1)
                & observed_mask[:, horizon:, :]
                & valid_mask[:, horizon:, :]
            )
            if mask.any():
                losses.append((diff[mask] ** 2).mean())
        if len(losses) == 0:
            return torch.zeros((), device=preds.device)
        return torch.stack(losses).mean()

    def compute_rejection_mask(
        self,
        scaled_target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[torch.Tensor, int, int]:
        reject_mask = torch.zeros_like(sample_id, dtype=torch.bool)
        rejected = 0
        total = 0
        use_zscore = self.hparams.anomaly_zscore_threshold > 0
        use_variance = self.hparams.anomaly_variance_ratio_threshold > 0
        eps = 1e-6
        for batch_idx in range(sample_id.shape[0]):
            sample_ids = torch.unique(sample_id[batch_idx])
            for sid in sample_ids:
                if sid == 0:
                    continue
                total += 1
                sid_mask = sample_id[batch_idx] == sid
                suffix_mask = sid_mask & prediction_mask[batch_idx]
                if not suffix_mask.any():
                    continue
                elem_mask = suffix_mask.unsqueeze(-1) & observed_mask[batch_idx]
                if not elem_mask.any():
                    continue

                reject = False
                if use_zscore:
                    max_abs_z = torch.where(
                        elem_mask,
                        scaled_target[batch_idx].abs(),
                        torch.tensor(float("-inf"), device=scaled_target.device),
                    ).max()
                    if max_abs_z > self.hparams.anomaly_zscore_threshold:
                        reject = True

                if use_variance and not reject:
                    prefix_mask = sid_mask & ~prediction_mask[batch_idx]
                    prefix_elem_mask = (
                        prefix_mask.unsqueeze(-1) & observed_mask[batch_idx]
                    )
                    if prefix_elem_mask.any():
                        prefix_vals = scaled_target[batch_idx][prefix_elem_mask]
                        suffix_vals = scaled_target[batch_idx][elem_mask]
                        if (
                            prefix_vals.numel() >= self.hparams.anomaly_variance_min_count
                            and suffix_vals.numel()
                            >= self.hparams.anomaly_variance_min_count
                        ):
                            prefix_var = prefix_vals.var(unbiased=False)
                            suffix_var = suffix_vals.var(unbiased=False)
                            ratio = (suffix_var + eps) / (prefix_var + eps)
                            if (
                                ratio > self.hparams.anomaly_variance_ratio_threshold
                                or ratio
                                < 1.0 / self.hparams.anomaly_variance_ratio_threshold
                            ):
                                reject = True

                if reject:
                    reject_mask[batch_idx, sid_mask] = True
                    rejected += 1
        return reject_mask, rejected, total

    @staticmethod
    def count_rejected_samples(
        reject_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[int, int]:
        rejected = 0
        total = 0
        for batch_idx in range(sample_id.shape[0]):
            sample_ids = torch.unique(sample_id[batch_idx])
            for sid in sample_ids:
                if sid == 0:
                    continue
                total += 1
                sid_mask = sample_id[batch_idx] == sid
                if reject_mask[batch_idx][sid_mask].any():
                    rejected += 1
        return rejected, total
