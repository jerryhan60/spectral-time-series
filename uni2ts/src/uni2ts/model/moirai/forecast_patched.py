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

from typing import Optional, Any, Iterable
import numpy as np
import torch
from jaxtyping import Float, Bool, Int
from gluonts.dataset import DataEntry
from gluonts.model import Forecast, Predictor
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import TestSplitSampler, Transformation

from uni2ts.transform.patch_precondition import PatchPolynomialPrecondition, PatchReversePrecondition
from .forecast import MoiraiForecast
from .forecast_precond import PreconditionReversingPredictor

class MoiraiForecastPatched(MoiraiForecast):
    """
    MoiraiForecast with patch-level preconditioning.
    
    This class handles inference with patch-level preconditioning.
    It applies preconditioning to the patched input within _convert,
    and handles reversal on the output.
    """
    
    def __init__(
        self,
        prediction_length: int,
        target_dim: int,
        feat_dynamic_real_dim: int,
        past_feat_dynamic_real_dim: int,
        context_length: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[Any] = None,
        patch_size: int | str = "auto",
        num_samples: int = 100,
        # Preconditioning parameters
        enable_preconditioning: bool = True,
        precondition_type: str = "chebyshev",
        precondition_degree: int = 5,
        reverse_output: bool = True,
    ):
        super().__init__(
            prediction_length=prediction_length,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
            context_length=context_length,
            module_kwargs=module_kwargs,
            module=module,
            patch_size=patch_size,
            num_samples=num_samples,
        )
        
        self.enable_preconditioning = enable_preconditioning
        self.precondition_type = precondition_type
        self.precondition_degree = precondition_degree
        self.reverse_output = reverse_output
        
        if self.enable_preconditioning:
            self.preconditioner = PatchPolynomialPrecondition(
                polynomial_type=precondition_type,
                degree=precondition_degree,
                target_field="target",
                enabled=True,
                store_original=True,
            )
            self.reverse_preconditioner = PatchReversePrecondition(
                target_field="prediction",
                enabled=True,
            )

    def _convert(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        future_target: Optional[Float[torch.Tensor, "batch future_time tgt"]] = None,
        future_observed_target: Optional[
            Bool[torch.Tensor, "batch future_time tgt"]
        ] = None,
        future_is_pad: Optional[Bool[torch.Tensor, "batch future_time"]] = None,
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> tuple[
        Float[torch.Tensor, "batch combine_seq patch"],  # target
        Bool[torch.Tensor, "batch combine_seq patch"],  # observed_mask
        Int[torch.Tensor, "batch combine_seq"],  # sample_id
        Int[torch.Tensor, "batch combine_seq"],  # time_id
        Int[torch.Tensor, "batch combine_seq"],  # variate_id
        Bool[torch.Tensor, "batch combine_seq"],  # prediction_mask
    ]:
        from einops import rearrange, repeat, reduce
        
        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []
        prediction_mask = []
        dim_count = 0

        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target
        )

        if future_target is None:
            future_target = torch.zeros(
                batch_shape
                + (
                    self.hparams.prediction_length,
                    past_target.shape[-1],
                ),
                dtype=past_target.dtype,
                device=device,
            )
        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(patch_size, past_target, -2, left=True),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )
        if future_observed_target is None:
            future_observed_target = torch.ones(
                batch_shape
                + (
                    self.hparams.prediction_length,
                    past_observed_target.shape[-1],
                ),
                dtype=torch.bool,
                device=device,
            )
        observed_mask.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_target, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_observed_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )
        if future_is_pad is None:
            future_is_pad = torch.zeros(
                batch_shape + (self.hparams.prediction_length,),
                dtype=torch.long,
                device=device,
            )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=self.prediction_token_length(patch_size),
                ),
            ]
        )
        dim_count += past_target.shape[-1]
        prediction_mask.extend(
            [
                torch.zeros(
                    batch_shape
                    + (self.context_token_length(patch_size) * past_target.shape[-1],),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    batch_shape
                    + (
                        self.prediction_token_length(patch_size)
                        * past_target.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )

        if feat_dynamic_real is not None:
            if observed_feat_dynamic_real is None:
                raise ValueError(
                    "observed_feat_dynamic_real must be provided if feat_dynamic_real is provided"
                )

            target.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., : self.hparams.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., self.hparams.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                ]
            )
            observed_mask.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., : self.hparams.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., self.hparams.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, self.max_patch_size - patch_size),
                    ),
                ]
            )
            sample_id.extend(
                [
                    repeat(
                        reduce(
                            (
                                self._patched_seq_pad(
                                    patch_size, past_is_pad, -1, left=True
                                )
                                == 0
                            ).int(),
                            "... (seq patch) -> ... seq",
                            "max",
                            patch=patch_size,
                        ),
                        "... seq -> ... (dim seq)",
                        dim=feat_dynamic_real.shape[-1],
                    ),
                    torch.ones(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )
            time_id.extend(
                [past_seq_id] * feat_dynamic_real.shape[-1]
                + [future_seq_id] * feat_dynamic_real.shape[-1]
            )
            variate_id.extend(
                [
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                        past=self.context_token_length(patch_size),
                    ),
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                        future=self.prediction_token_length(patch_size),
                    ),
                ]
            )
            dim_count += feat_dynamic_real.shape[-1]
            prediction_mask.extend(
                [
                    torch.zeros(
                        batch_shape
                        + (
                            self.context_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.zeros(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )

        if past_feat_dynamic_real is not None:
            if past_observed_feat_dynamic_real is None:
                raise ValueError(
                    "past_observed_feat_dynamic_real must be provided if past_feat_dynamic_real is provided"
                )
            target.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                )
            )
            observed_mask.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                )
            )
            sample_id.append(
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_feat_dynamic_real.shape[-1],
                )
            )
            time_id.extend([past_seq_id] * past_feat_dynamic_real.shape[-1])

            variate_id.append(
                repeat(
                    torch.arange(past_feat_dynamic_real.shape[-1], device=device)
                    + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                )
            )
            dim_count += past_feat_dynamic_real.shape[-1]
            prediction_mask.append(
                torch.zeros(
                    batch_shape
                    + (
                        self.context_token_length(patch_size)
                        * past_feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                )
            )

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        prediction_mask = torch.cat(prediction_mask, dim=-1)
        
        if self.enable_preconditioning:
            # Apply patch-level preconditioning
            target = self._apply_precond_tensor(target, variate_id, sample_id)
            
        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        )

    def _apply_precond_tensor(
        self,
        target: torch.Tensor,
        variate_id: torch.Tensor,
        sample_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply preconditioning to tensor, respecting sample and variate boundaries.
        """
        # target: (batch, seq, patch)
        # variate_id: (batch, seq)
        # sample_id: (batch, seq)
        
        device = target.device
        # We can iterate over batch
        # For each batch, we can iterate over unique (sample_id, variate_id) pairs?
        # Or just assume sample_id is constant per batch item (it's not necessarily).
        
        # Actually, in _convert, sample_id is constructed.
        
        device = target.device
        target_np = target.cpu().numpy()
        variate_id_np = variate_id.cpu().numpy()
        sample_id_np = sample_id.cpu().numpy()
        
        preconditioned_np = np.zeros_like(target_np)
        
        batch_size = target.shape[0]
        for b in range(batch_size):
            # Identify segments
            # A segment is defined by constant (sample_id, variate_id)
            # Since they are packed, they should be contiguous blocks?
            # In _convert, they are constructed by extending lists and then cat.
            # So yes, they are blocks.
            
            # We can just group by (sample_id, variate_id)
            
            # Create a unique ID for each segment
            # Assuming IDs are small enough
            # Or just iterate
            
            # Let's use a simple approach: iterate through the sequence and detect changes
            
            seq_len = target.shape[1]
            if seq_len == 0:
                continue
                
            current_start = 0
            current_key = (sample_id_np[b, 0], variate_id_np[b, 0])
            
            for t in range(1, seq_len):
                key = (sample_id_np[b, t], variate_id_np[b, t])
                if key != current_key:
                    # Process previous segment
                    segment = target_np[b, current_start:t]
                    preconditioned_np[b, current_start:t] = self.preconditioner._apply_patch_convolution(
                        segment, self.preconditioner.coeffs
                    )
                    
                    current_start = t
                    current_key = key
            
            # Process last segment
            segment = target_np[b, current_start:]
            preconditioned_np[b, current_start:] = self.preconditioner._apply_patch_convolution(
                segment, self.preconditioner.coeffs
            )
            
        return torch.from_numpy(preconditioned_np).to(device)

    def _format_preds(
        self,
        patch_size: int,
        preds: Float[torch.Tensor, "sample batch combine_seq patch"],
        target_dim: int,
    ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:
        
        if self.enable_preconditioning and self.reverse_output:
            # Reverse preconditioning on predictions
            # preds: (sample, batch, combine_seq, patch)
            
            # We need to reverse using the same logic (respecting boundaries)
            # But we also need context (history).
            # The history is in the 'combine_seq' dimension (it includes past and future).
            # Wait, MoiraiModule output 'distr' covers the whole sequence?
            # No, MoiraiModule forward returns distribution for the whole sequence?
            # Let's check module.py.
            # forward returns 'distr'.
            # In forecast.py: distr.sample(...) returns preds.
            # The preds cover the whole sequence (past + future) because of how it's constructed in _convert?
            # In _convert, target includes past and future.
            # So preds likely covers the whole sequence or just the masked part?
            # In MoiraiForecast._format_preds:
            # start = target_dim * context_token_length
            # preds = preds[..., start:end, :patch_size]
            # This implies preds covers the whole sequence.
            
            # So we should reverse the WHOLE sequence first, then slice.
            # This gives us the context for free (the past part of the sequence).
            
            # However, preds are samples from the distribution.
            # The distribution is over the whole sequence?
            # MoiraiModule forward:
            # reprs = self.encoder(...)
            # distr_param = self.param_proj(reprs, patch_size)
            # distr = self.distr_output.distribution(distr_param, ...)
            # Yes, it covers the whole sequence.
            
            # So we can reverse in place on the whole sequence.
            
            # We need to iterate over samples too.
            num_samples = preds.shape[0]
            batch_size = preds.shape[1]
            
            # To avoid slow loops, we can reshape samples into batch?
            # (sample * batch, seq, patch)
            
            preds_flat = preds.reshape(-1, preds.shape[2], preds.shape[3])
            
            # We need variate_id and sample_id for the batch.
            # They are the same for all samples.
            # We can repeat them.
            
            # But wait, we don't have access to variate_id here easily?
            # _format_preds doesn't take variate_id.
            # We might need to store it in self or pass it.
            # But _format_preds is called from forward.
            # forward calls _convert, then _get_distr, then _format_preds.
            # _convert returns variate_id.
            # But _get_distr calls _convert internally again?
            # No, _val_loss calls _convert.
            # forward calls _val_loss (loop) or _get_distr (single).
            # _get_distr calls _convert.
            # So we don't have variate_id in _format_preds unless we pass it.
            
            # We should override forward to handle this better.
            pass

        return super()._format_preds(patch_size, preds, target_dim)

    def forward(
        self,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        num_samples: Optional[int] = None,
    ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:
        
        # We need to reimplement forward to handle reversal properly
        # because we need access to the full sequence and IDs for reversal.
        
        if self.hparams.patch_size == "auto":
            # Loop over patch sizes
            # This is tricky because we need to select the best patch size.
            # If we reverse, we should compute loss on reversed data?
            # Or just compute loss on preconditioned data (as proxy)?
            # Usually we want loss on original data.
            
            # For now, let's assume we use the default logic (loss on preconditioned data)
            # but we return reversed predictions.
            
            val_loss = []
            preds = []
            
            for patch_size in self.module.patch_sizes:
                # Calculate loss (on preconditioned data)
                val_loss.append(
                    self._val_loss(
                        patch_size=patch_size,
                        target=past_target[..., : self.past_length, :],
                        observed_target=past_observed_target[..., : self.past_length, :],
                        is_pad=past_is_pad[..., : self.past_length],
                        feat_dynamic_real=(feat_dynamic_real[..., : self.past_length, :] if feat_dynamic_real is not None else None),
                        observed_feat_dynamic_real=(observed_feat_dynamic_real[..., : self.past_length, :] if observed_feat_dynamic_real is not None else None),
                        past_feat_dynamic_real=(past_feat_dynamic_real[..., : self.hparams.context_length, :] if past_feat_dynamic_real is not None else None),
                        past_observed_feat_dynamic_real=(past_observed_feat_dynamic_real[..., : self.hparams.context_length, :] if past_observed_feat_dynamic_real is not None else None),
                    )
                )
                
                # Get distribution and sample
                # We need to manually call _convert to get IDs for reversal
                (
                    target,
                    observed_mask,
                    sample_id,
                    time_id,
                    variate_id,
                    prediction_mask,
                ) = self._convert(
                    patch_size,
                    past_target,
                    past_observed_target,
                    past_is_pad,
                    feat_dynamic_real=feat_dynamic_real,
                    observed_feat_dynamic_real=observed_feat_dynamic_real,
                    past_feat_dynamic_real=past_feat_dynamic_real,
                    past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
                )

                distr = self.module(
                    target,
                    observed_mask,
                    sample_id,
                    time_id,
                    variate_id,
                    prediction_mask,
                    torch.ones_like(time_id, dtype=torch.long) * patch_size,
                )
                
                samples = distr.sample(torch.Size((num_samples or self.hparams.num_samples,)))

                if self.enable_preconditioning and self.reverse_output:
                    # Reverse samples using both variate_id and sample_id
                    samples = self._reverse_samples(samples, variate_id, sample_id)

                preds.append(
                    self._format_preds_simple(
                        patch_size,
                        samples,
                        past_target.shape[-1]
                    )
                )
                
            val_loss = torch.stack(val_loss)
            preds = torch.stack(preds)
            idx = val_loss.argmin(dim=0)
            return preds[idx, torch.arange(len(idx), device=idx.device)]
            
        else:
            # Single patch size
            patch_size = self.hparams.patch_size
            
            (
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
            ) = self._convert(
                patch_size,
                past_target,
                past_observed_target,
                past_is_pad,
                feat_dynamic_real=feat_dynamic_real,
                observed_feat_dynamic_real=observed_feat_dynamic_real,
                past_feat_dynamic_real=past_feat_dynamic_real,
                past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
            )

            distr = self.module(
                target,
                observed_mask,
                sample_id,
                time_id,
                variate_id,
                prediction_mask,
                torch.ones_like(time_id, dtype=torch.long) * patch_size,
            )
            
            samples = distr.sample(torch.Size((num_samples or self.hparams.num_samples,)))

            if self.enable_preconditioning and self.reverse_output:
                samples = self._reverse_samples(samples, variate_id, sample_id)

            return self._format_preds_simple(
                patch_size,
                samples,
                past_target.shape[-1]
            )

    def _reverse_samples(
        self,
        samples: torch.Tensor,
        variate_id: torch.Tensor,
        sample_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Reverse preconditioning on samples.
        samples: (num_samples, batch, seq, patch)
        variate_id: (batch, seq)
        sample_id: (batch, seq) - optional, used for segment detection
        """
        num_samples = samples.shape[0]
        batch_size = samples.shape[1]
        seq_len = samples.shape[2]
        patch_size = samples.shape[3]

        device = samples.device
        samples_np = samples.cpu().numpy()
        variate_id_np = variate_id.cpu().numpy()
        sample_id_np = sample_id.cpu().numpy() if sample_id is not None else None

        reversed_np = np.zeros_like(samples_np)

        # Iterate over batch
        for b in range(batch_size):
            # Find segments using both sample_id and variate_id (like _apply_precond_tensor)
            current_start = 0
            if sample_id_np is not None:
                current_key = (sample_id_np[b, 0], variate_id_np[b, 0])
            else:
                current_key = variate_id_np[b, 0]

            for t in range(1, seq_len):
                if sample_id_np is not None:
                    key = (sample_id_np[b, t], variate_id_np[b, t])
                else:
                    key = variate_id_np[b, t]

                if key != current_key:
                    # Process segment for all samples
                    segment = samples_np[:, b, current_start:t, :]
                    # Reverse each sample
                    for s in range(num_samples):
                        reversed_np[s, b, current_start:t, :] = self.reverse_preconditioner._reverse_patch_convolution(
                            segment[s], self.preconditioner.coeffs
                        )

                    current_start = t
                    current_key = key

            # Last segment
            segment = samples_np[:, b, current_start:, :]
            for s in range(num_samples):
                reversed_np[s, b, current_start:, :] = self.reverse_preconditioner._reverse_patch_convolution(
                    segment[s], self.preconditioner.coeffs
                )

        return torch.from_numpy(reversed_np).to(device)

    def _format_preds_simple(
        self,
        patch_size: int,
        preds: Float[torch.Tensor, "sample batch combine_seq patch"],
        target_dim: int,
    ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:
        # Simplified version of _format_preds that assumes preds are already processed/reversed
        # and just does the slicing and reshaping.
        
        # Note: In original _format_preds, it slices first then reshapes.
        # Here we have preds covering the whole sequence.
        
        start = target_dim * self.context_token_length(patch_size)
        end = start + target_dim * self.prediction_token_length(patch_size)
        
        # Slice to get prediction window
        preds_slice = preds[..., start:end, :patch_size]
        
        # Reshape to (sample, batch, future_time, target_dim)
        # Original: "sample ... (dim seq) patch -> ... sample (seq patch) dim"
        # Here ... is batch.
        
        # We need to be careful about the order.
        # The combine_seq is (dim * seq).
        # We want to separate dim and seq.
        
        # preds_slice shape: (sample, batch, dim*pred_seq, patch)
        
        from einops import rearrange
        preds_reshaped = rearrange(
            preds_slice,
            "sample batch (dim seq) patch -> batch sample (seq patch) dim",
            dim=target_dim,
        )
        
        # Crop to exact prediction length
        preds_final = preds_reshaped[..., : self.hparams.prediction_length, :]
        
        return preds_final.squeeze(-1)

    def create_predictor(
        self,
        batch_size: int,
        device: str = "auto",
    ) -> Predictor:
        # Override to ensure we don't use the default wrapper if we want patch preconditioning
        # But wait, create_predictor is used for GluonTS evaluation.
        # The default implementation uses get_default_transform + instance_splitter.
        # And wraps with PyTorchPredictor.
        
        # If we use MoiraiForecastPatched as the prediction_net, it will handle preconditioning internally
        # IF we pass raw data.
        # But get_default_transform usually prepares data.
        
        # In MoiraiForecast, get_default_transform does NOT apply preconditioning.
        # So we are good.
        
        # However, we need to make sure the input to the model is NOT preconditioned by any external transform.
        
        return super().create_predictor(batch_size, device)

def create_patched_forecast_from_checkpoint(
    checkpoint_path: str,
    prediction_length: int,
    target_dim: int,
    feat_dynamic_real_dim: int,
    past_feat_dynamic_real_dim: int,
    context_length: int = 1000,
    patch_size: int | str = "auto",
    num_samples: int = 100,
    enable_preconditioning: bool = True,
    precondition_type: str = "chebyshev",
    precondition_degree: int = 5,
    reverse_output: bool = True,
) -> MoiraiForecastPatched:
    model = MoiraiForecastPatched.load_from_checkpoint(
        checkpoint_path,
        prediction_length=prediction_length,
        target_dim=target_dim,
        feat_dynamic_real_dim=feat_dynamic_real_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        enable_preconditioning=enable_preconditioning,
        precondition_type=precondition_type,
        precondition_degree=precondition_degree,
        reverse_output=reverse_output,
        strict=False,
    )
    return model
