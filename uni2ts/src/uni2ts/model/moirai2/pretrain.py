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
        time_precondition_coeffs_lambda: float = 0.0,
        time_precondition_dual_head_lambda: float = 1.0,
        ps_loss_lambda: float = 0.0,
        prefix_ratio_jitter: float = 0.0,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[Moirai2Module] = None,
        loss_func: Optional[PackedQuantileLoss] = None,
        beta1: float = 0.9,
        beta2: float = 0.98,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        scheduler_type: str = "cosine",
        scheduler_num_cycles: int = 1,
        min_lr_ratio: float = 0.0,
        wsd_stable_ratio: float = 0.7,
        sign_flip_prob: float = 0.0,
        freq_noise_sigma: float = 0.0,
        tsmixup_prob: float = 0.0,
        dominant_shuffle_k: int = 0,
        freq_mask_rate: float = 0.0,
        init_from_pretrained: str = "",
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.module = Moirai2Module(**module_kwargs) if module is None else module
        # Optionally initialize from a pretrained HuggingFace model
        if init_from_pretrained:
            self._load_pretrained_weights(init_from_pretrained)
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

    def _load_pretrained_weights(self, model_name: str):
        """Load weights from a pretrained model, expanding in_proj if needed.

        Supports both HuggingFace model names and local checkpoint files (.ckpt).
        """
        import logging
        log = logging.getLogger(__name__)
        log.info(f"Loading pretrained weights from {model_name}")

        if model_name.endswith(".ckpt"):
            # Load from Lightning checkpoint file
            ckpt = torch.load(model_name, map_location="cpu")
            raw_state = ckpt.get("state_dict", ckpt)
            # Strip "module." prefix from Lightning state dict keys
            pre_state = {}
            for k, v in raw_state.items():
                key = k.replace("module.", "", 1) if k.startswith("module.") else k
                pre_state[key] = v
            del ckpt
        else:
            # Load from HuggingFace
            pretrained = Moirai2Module.from_pretrained(model_name)
            pre_state = pretrained.state_dict()

        our_state = self.module.state_dict()
        loaded, skipped, expanded = 0, 0, 0
        for key in pre_state:
            if key not in our_state:
                skipped += 1
                continue
            if our_state[key].shape == pre_state[key].shape:
                our_state[key] = pre_state[key]
                loaded += 1
            elif pre_state[key].dim() == 2 and our_state[key].dim() == 2:
                # Weight matrix with different input dims (e.g. in_proj expansion)
                pre_w = pre_state[key]
                our_w = our_state[key]
                if our_w.shape[0] == pre_w.shape[0] and our_w.shape[1] > pre_w.shape[1]:
                    padded = torch.zeros_like(our_w)
                    padded[:, :pre_w.shape[1]] = pre_w
                    our_state[key] = padded
                    expanded += 1
                    log.info(f"  Expanded {key}: {pre_w.shape} -> {our_w.shape}")
                else:
                    skipped += 1
                    log.warning(f"  Skipped {key}: incompatible shapes {pre_w.shape} vs {our_w.shape}")
            else:
                skipped += 1
                log.warning(f"  Skipped {key}: shape mismatch {pre_state[key].shape} vs {our_state[key].shape}")
        self.module.load_state_dict(our_state)
        if not model_name.endswith(".ckpt"):
            del pretrained
        log.info(f"Pretrained init: {loaded} loaded, {expanded} expanded, {skipped} skipped")

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        input_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]] = None,
    ):
        result = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            training_mode=True,
            input_mask=input_mask,
            global_step=self.global_step,
        )
        return result

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # Sign flip augmentation: negate target with probability p
        target = batch["target"]
        if self.hparams.sign_flip_prob > 0:
            B = target.shape[0]
            flip = torch.rand(B, device=target.device) < self.hparams.sign_flip_prob
            target = torch.where(flip[:, None, None], -target, target)
            batch = {**batch, "target": target}
        # TSMixup augmentation: mix K=3 series within the batch
        if self.hparams.tsmixup_prob > 0:
            target = batch["target"]
            B = target.shape[0]
            n_mix = int(B * self.hparams.tsmixup_prob)
            if n_mix > 0 and B >= 3:
                mix_idx = torch.randperm(B, device=target.device)[:n_mix]
                # Normalize each series to zero-mean, unit-std
                means = target.mean(dim=1, keepdim=True)
                stds = target.std(dim=1, keepdim=True).clamp(min=1e-8)
                normalized = (target - means) / stds
                # For each mixed sample, average K=3 random series
                K = 3
                weights = torch.distributions.Dirichlet(
                    torch.ones(K, device=target.device)
                ).sample((n_mix,))
                src_idx = torch.stack([
                    torch.randint(B, (n_mix,), device=target.device) for _ in range(K)
                ], dim=1)
                mixed = sum(
                    weights[:, k, None, None] * normalized[src_idx[:, k]]
                    for k in range(K)
                )
                target = target.clone()
                target[mix_idx] = mixed
                batch = {**batch, "target": target}
        # Frequency noise augmentation: perturb frequency magnitudes
        if self.hparams.freq_noise_sigma > 0:
            target = batch["target"]
            X = torch.fft.rfft(target, dim=1)
            noise = 1.0 + self.hparams.freq_noise_sigma * torch.randn(
                X.shape[0], X.shape[1], 1, device=X.device
            )
            X = X * noise
            target = torch.fft.irfft(X, n=target.shape[1], dim=1)
            batch = {**batch, "target": target}
        # Dominant shuffle: shuffle amplitudes/phases of top-k dominant frequencies
        if self.hparams.dominant_shuffle_k > 0:
            target = batch["target"]
            X = torch.fft.rfft(target, dim=1)
            amp = X.abs()
            k = self.hparams.dominant_shuffle_k
            # Get top-k dominant frequencies (exclude DC=0)
            topk_vals, topk_idx = torch.topk(amp[:, 1:, :].mean(dim=-1), k, dim=-1)
            topk_idx = topk_idx + 1  # offset for DC
            X_aug = X.clone()
            for b in range(X.shape[0]):
                idx = topk_idx[b]
                perm = idx[torch.randperm(k, device=idx.device)]
                X_aug[b, idx] = X[b, perm]
            target = torch.fft.irfft(X_aug, n=target.shape[1], dim=1)
            batch = {**batch, "target": target}
        # Frequency masking: randomly zero out frequency bins
        if self.hparams.freq_mask_rate > 0:
            target = batch["target"]
            X = torch.fft.rfft(target, dim=1)
            n_freq = X.shape[1]
            n_mask = int(n_freq * self.hparams.freq_mask_rate)
            if n_mask > 0:
                mask = torch.ones(X.shape[0], n_freq, 1, device=X.device)
                for b in range(X.shape[0]):
                    idx = torch.randperm(n_freq, device=X.device)[:n_mask]
                    mask[b, idx] = 0
                X = X * mask
                target = torch.fft.irfft(X, n=target.shape[1], dim=1)
                batch = {**batch, "target": target}
        input_mask = self.sample_patch_mask(batch["sample_id"])
        result = self(
            target=batch["target"],
            observed_mask=batch["observed_mask"],
            sample_id=batch["sample_id"],
            time_id=batch["time_id"],
            variate_id=batch["variate_id"],
            prediction_mask=batch["prediction_mask"],
            input_mask=input_mask,
        )
        # Unpack based on mode
        _dual_head = (
            self.module.time_precondition_dual_head
            and self.module.time_precondition_enabled
        )
        if _dual_head:
            preds, preds_raw, scaled_target, scaled_target_raw = result
        else:
            preds, scaled_target = result
            preds_raw = None

        loc = None
        scale = None
        if not _dual_head:
            scaled_target_raw = scaled_target
        if (
            self.module.time_precondition_enabled
            or self.hparams.time_precondition_reverse_in_loss
            or self.hparams.patch_precondition_reverse_in_loss
        ):
            if not _dual_head:
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
        if _dual_head:
            # Dual-head: precond head loss + raw head loss
            precond_loss = self.multi_token_loss(
                preds=preds,
                target=scaled_target,
                observed_mask=batch["observed_mask"],
                prediction_mask=loss_prediction_mask,
                sample_id=batch["sample_id"],
                variate_id=batch["variate_id"],
            )
            raw_loss = self.multi_token_loss(
                preds=preds_raw,
                target=scaled_target_raw,
                observed_mask=batch["observed_mask"],
                prediction_mask=loss_prediction_mask,
                sample_id=batch["sample_id"],
                variate_id=batch["variate_id"],
            )
            base_loss = raw_loss + self.hparams.time_precondition_dual_head_lambda * precond_loss
        elif self.hparams.time_precondition_reverse_in_loss:
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
        # Patch-wise Structural Loss (PS-Loss)
        if self.hparams.ps_loss_lambda > 0:
            ps_loss = self._compute_ps_loss(
                preds=preds,
                target=scaled_target,
                observed_mask=batch["observed_mask"],
                prediction_mask=loss_prediction_mask,
                sample_id=batch["sample_id"],
            )
            loss = loss + self.hparams.ps_loss_lambda * ps_loss
        # MoE load balancing loss
        moe_aux_loss = 0.0
        for m in self.module.modules():
            if hasattr(m, '_aux_loss') and m._aux_loss != 0.0:
                moe_aux_loss = moe_aux_loss + m._aux_loss
                m._aux_loss = 0.0  # reset
        if isinstance(moe_aux_loss, torch.Tensor):
            loss = loss + 0.01 * moe_aux_loss
        batch_size = batch["sample_id"].max(dim=1).values.sum()
        if _dual_head:
            self.log(
                "train/raw_loss",
                raw_loss,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
            self.log(
                "train/precond_loss",
                precond_loss,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
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
        if (
            self.module.time_precondition_coeffs.numel() > 0
            and self.hparams.time_precondition_coeffs_lambda > 0
        ):
            coeffs_l2 = (self.module.time_precondition_coeffs ** 2).sum()
            loss = loss + self.hparams.time_precondition_coeffs_lambda * coeffs_l2
            self.log(
                "train/precond_coeffs_l2",
                coeffs_l2,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
        if self.module.time_precondition_coeffs.numel() > 0:
            self.log(
                "train/precond_coeffs_norm",
                self.module.time_precondition_coeffs.norm(),
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
        if (
            self.module.attn_l1_lambda > 0
            and self.module._last_attn_weights is not None
        ):
            attn_l1 = torch.stack(
                [w.abs().mean() for w in self.module._last_attn_weights]
            ).mean()
            loss = loss + self.module.attn_l1_lambda * attn_l1
            self.log(
                "train/attn_l1",
                attn_l1,
                on_step=self.hparams.log_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
                rank_zero_only=True,
            )
        if self.hparams.ps_loss_lambda > 0:
            self.log(
                "train/ps_loss",
                ps_loss,
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
        # STU projection matrices should have weight decay
        missing = set(param_dict.keys()) - decay - no_decay
        for fpn in list(missing):
            if "M_inputs" in fpn or "M_filters" in fpn:
                decay.add(fpn)
                missing.discard(fpn)
        # Remaining missing params (e.g. gate, inverse coeffs) go to no_decay
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
        sched_type = SchedulerType(self.hparams.scheduler_type)
        sched_kwargs = {}
        if sched_type == SchedulerType.COSINE_WITH_RESTARTS:
            sched_kwargs["num_cycles"] = self.hparams.scheduler_num_cycles
        if sched_type == SchedulerType.COSINE and self.hparams.min_lr_ratio > 0:
            sched_kwargs["min_lr_ratio"] = self.hparams.min_lr_ratio
        if sched_type == SchedulerType.WSD:
            sched_kwargs["stable_ratio"] = self.hparams.wsd_stable_ratio
        scheduler = get_scheduler(
            sched_type,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
            scheduler_specific_kwargs=sched_kwargs,
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
                    prefix_ratio_jitter=self.hparams.prefix_ratio_jitter,
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
        output_init = torch.where(
            pred_mask_flat.unsqueeze(-1), precond_flat, raw_expanded
        )

        seq_len = precond_flat.shape[1]
        min_time = n * stride
        if seq_len <= min_time:
            return rearrange(
                output_init, "b (t p) q -> b t q p", p=patch_size
            )

        if not pred_mask_flat[:, min_time:].any():
            return rearrange(
                output_init, "b (t p) q -> b t q p", p=patch_size
            )

        # Split into per-timestep lists to avoid in-place ops (autograd safe)
        output_slices = list(output_init.unbind(dim=1))
        median_slices = list(raw_flat.unbind(dim=1))

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
                weighted_sum = torch.zeros_like(median_slices[t])
                for i in range(n):
                    shift = (i + 1) * stride
                    weighted_sum = weighted_sum + coeffs[i] * median_slices[t - shift]
                median_slices[t] = torch.where(
                    mask,
                    precond_flat[:, t, median_idx] - weighted_sum,
                    median_slices[t],
                )
                output_slices[t] = torch.where(
                    mask.unsqueeze(-1),
                    precond_flat[:, t, :] - weighted_sum.unsqueeze(-1),
                    output_slices[t],
                )
            output = torch.stack(output_slices, dim=1)
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
            weighted_sum = torch.zeros_like(median_slices[t])
            for i in range(n):
                shift = (i + 1) * stride
                weighted_sum = weighted_sum + coeffs[i] * median_slices[t - shift]
            median_slices[t] = torch.where(
                mask,
                precond_flat[:, t, median_idx] - weighted_sum,
                median_slices[t],
            )
            output_slices[t] = torch.where(
                mask.unsqueeze(-1),
                precond_flat[:, t, :] - weighted_sum.unsqueeze(-1),
                output_slices[t],
            )

        output = torch.stack(output_slices, dim=1)
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

    def _compute_ps_loss(
        self,
        preds: Float[torch.Tensor, "*batch seq_len pred"],
        target: Float[torch.Tensor, "*batch seq_len patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len patch"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> torch.Tensor:
        """Patch-wise Structural Loss: correlation + variance + mean components."""
        npt = self.module.num_predict_token
        nq = self.module.num_quantiles
        ps = self.module.patch_size
        median_idx = self._median_quantile_idx

        # Reshape preds: (B, S, npt, nq, ps)
        preds_reshaped = rearrange(
            preds,
            "... (predict_token num_quantiles patch_size) -> ... predict_token num_quantiles patch_size",
            predict_token=npt,
            num_quantiles=nq,
            patch_size=ps,
        )

        all_losses = []
        for horizon in range(1, npt + 1):
            if target.shape[1] <= horizon:
                continue
            horizon_mask = self.build_horizon_mask(
                sample_id, prediction_mask, horizon
            )
            if not horizon_mask.any():
                continue

            # Median quantile prediction for this horizon: (B, S-h, ps)
            pred_h = preds_reshaped[:, :-horizon, horizon - 1, median_idx, :]
            target_h = target[:, horizon:]
            observed_h = observed_mask[:, horizon:]

            # Combined mask: horizon valid AND all elements in patch observed
            # horizon_mask: (B, S-h), observed_h: (B, S-h, ps)
            patch_fully_observed = observed_h.all(dim=-1)  # (B, S-h)
            valid = horizon_mask & patch_fully_observed  # (B, S-h)

            if not valid.any():
                continue

            # Select valid patches: flatten and gather
            pred_patches = pred_h[valid]   # (N, ps)
            tgt_patches = target_h[valid]  # (N, ps)

            if pred_patches.shape[0] == 0 or ps < 2:
                continue

            # --- Mean loss: |mean(y) - mean(yhat)| ---
            mean_loss = (tgt_patches.mean(dim=-1) - pred_patches.mean(dim=-1)).abs().mean()

            # --- Correlation loss: 1 - pearson_corr ---
            tgt_centered = tgt_patches - tgt_patches.mean(dim=-1, keepdim=True)
            pred_centered = pred_patches - pred_patches.mean(dim=-1, keepdim=True)
            tgt_std = tgt_centered.norm(dim=-1)    # (N,)
            pred_std = pred_centered.norm(dim=-1)   # (N,)
            # Avoid division by zero for constant patches
            valid_corr = (tgt_std > 1e-8) & (pred_std > 1e-8)
            if valid_corr.any():
                dot = (tgt_centered[valid_corr] * pred_centered[valid_corr]).sum(dim=-1)
                corr = dot / (tgt_std[valid_corr] * pred_std[valid_corr])
                corr_loss = (1.0 - corr).mean()
            else:
                corr_loss = torch.zeros((), device=preds.device)

            # --- Variance loss: KL(softmax(y), softmax(yhat)) ---
            tgt_logprob = torch.log_softmax(tgt_patches, dim=-1)
            pred_logprob = torch.log_softmax(pred_patches, dim=-1)
            tgt_prob = tgt_logprob.exp()
            # KL(p||q) = sum p * (log p - log q)
            kl = (tgt_prob * (tgt_logprob - pred_logprob)).sum(dim=-1)  # (N,)
            var_loss = kl.mean()

            all_losses.append(corr_loss + var_loss + mean_loss)

        if len(all_losses) == 0:
            return torch.zeros((), device=preds.device)
        return torch.stack(all_losses).mean()

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
