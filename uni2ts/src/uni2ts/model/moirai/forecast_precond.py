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

"""
Preconditioning-aware forecast model.
Extends MoiraiForecast to properly handle preconditioning during evaluation.
"""

from typing import Optional, Any, Iterable
import numpy as np
import torch
from jaxtyping import Float, Bool
from gluonts.dataset import DataEntry
from gluonts.model import Forecast, Predictor
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import TestSplitSampler, Transformation
from gluonts.transform.split import TFTInstanceSplitter

from uni2ts.transform import PolynomialPrecondition, ReversePrecondition
from .forecast import MoiraiForecast


class PreconditionReversingPredictor(Predictor):
    """
    Predictor wrapper that reverses preconditioning on outputs.

    This wraps a base predictor and applies ReversePrecondition transform
    to all predictions, converting them back to the original scale.
    """

    def __init__(
        self,
        base_predictor: Predictor,
        precondition_coeffs: np.ndarray,
        precondition_degree: int,
        precondition_type: str,
        context_wrapper: Optional[Any] = None,
        reverse_output: bool = True,
    ):
        self.base_predictor = base_predictor
        self.precondition_coeffs = precondition_coeffs
        self.precondition_degree = precondition_degree
        self.precondition_type = precondition_type
        self.context_wrapper = context_wrapper
        self.reverse_output = reverse_output
        self.reverse_transform = ReversePrecondition(
            target_field="target",
            enabled=True,
        )

    @property
    def prediction_length(self) -> int:
        return self.base_predictor.prediction_length

    @property
    def lead_time(self) -> int:
        return self.base_predictor.lead_time

    def predict(self, dataset: Iterable[DataEntry], **kwargs) -> Iterable[Forecast]:
        """
        Make predictions and optionally reverse preconditioning on outputs.

        This method:
        1. Clears any stored contexts from previous predictions
        2. Gets predictions from the base predictor (in preconditioned space)
           - The transform pipeline stores original contexts during processing
        3. If reverse_output is True: Reverses preconditioning on each prediction using stored contexts
        4. Returns forecasts in original scale (if reversed) or preconditioned space (if not reversed)
        """
        # Clear stored contexts from any previous prediction run
        if self.context_wrapper is not None:
            self.context_wrapper.stored_contexts = []

        # Get predictions from base predictor (in preconditioned space)
        # The transform pipeline will populate stored_contexts during processing
        forecasts = self.base_predictor.predict(dataset, **kwargs)

        # If reversal is disabled, return forecasts as-is (in preconditioned space)
        if not self.reverse_output:
            yield from forecasts
            return

        # Get stored contexts (populated during transform)
        stored_contexts = self.context_wrapper.stored_contexts if self.context_wrapper else []
        context_iter = iter(stored_contexts)

        # Process each forecast with reversal, yielding immediately
        for forecast in forecasts:
            # Get the corresponding context
            try:
                context = next(context_iter)
            except StopIteration:
                # No more contexts, return forecast as-is
                yield forecast
                continue

            if context is None:
                # No valid context, return forecast as-is
                yield forecast
                continue

            # Get forecast samples (in preconditioned space)
            samples = forecast.samples  # [num_samples, pred_len] or [num_samples, pred_len, dim]

            # Reverse preconditioning for each sample
            reversed_samples_list = []

            for sample_idx in range(samples.shape[0]):
                sample = samples[sample_idx]  # [pred_len] or [pred_len, dim]

                # Apply reversal using ReversePrecondition transform with explicit context
                # We pass the prediction sample and the context separately

                # Prepare data entry for reversal
                reversal_data = {
                    "target": sample,
                    "precondition_coeffs": self.precondition_coeffs,
                    "precondition_degree": self.precondition_degree,
                    "precondition_type": self.precondition_type,
                    "precondition_enabled": True,
                }

                # Apply reversal
                reversed_data = self.reverse_transform(reversal_data, context=context)
                reversed_sample = reversed_data["target"]

                reversed_samples_list.append(reversed_sample)

            # Stack reversed samples
            reversed_samples = np.stack(reversed_samples_list, axis=0)

            # Create new forecast with reversed samples
            class ReversedForecast(Forecast):
                """Forecast with preconditioning reversal applied."""
                def __init__(self, original_forecast, reversed_samples):
                    self._original_forecast = original_forecast
                    self._reversed_samples = reversed_samples

                @property
                def samples(self):
                    return self._reversed_samples

                @property
                def mean(self):
                    return self._reversed_samples.mean(axis=0)

                def quantile(self, q):
                    return np.quantile(self._reversed_samples, q, axis=0)

                def __getitem__(self, key):
                    if key == "mean":
                        return self.mean
                    elif key == "samples":
                        return self.samples
                    # Delegate to original forecast for other keys
                    return self._original_forecast[key]

                @property
                def start_date(self):
                    return self._original_forecast.start_date

                @property
                def item_id(self):
                    return self._original_forecast.item_id

                @property
                def freq(self):
                    return self._original_forecast.freq

                @property
                def prediction_length(self):
                    return self._original_forecast.prediction_length

                @property
                def index(self):
                    return self._original_forecast.index

            yield ReversedForecast(forecast, reversed_samples)


class MoiraiForecastPrecond(MoiraiForecast):
    """
    Extended MoiraiForecast that properly handles preconditioning.

    This class wraps MoiraiForecast to:
    1. Apply preconditioning to input data (same as training)
    2. Make predictions in preconditioned space
    3. Optionally reverse preconditioning on outputs (back to original scale)

    This ensures fair evaluation when the model was trained with preconditioning.
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
        """
        Initialize MoiraiForecastPrecond.

        Args:
            enable_preconditioning: Whether to apply preconditioning (must match training!)
            precondition_type: "chebyshev" or "legendre" (must match training!)
            precondition_degree: Polynomial degree (must match training!)
            reverse_output: Whether to reverse preconditioning on outputs (True: output in original scale, False: output in preconditioned space)
        """
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

        # Save preconditioning config
        self.enable_preconditioning = enable_preconditioning
        self.precondition_type = precondition_type
        self.precondition_degree = precondition_degree
        self.reverse_output = reverse_output

        # Create preconditioning transforms
        if self.enable_preconditioning:
            self.preconditioner = PolynomialPrecondition(
                polynomial_type=precondition_type,
                degree=precondition_degree,
                target_field="target",
                enabled=True,
                store_original=True,  # Store for reversal
            )
            self.reverse_preconditioner = ReversePrecondition(
                target_field="prediction",
                enabled=True,
            )

    def _apply_preconditioning_numpy(
        self,
        data: np.ndarray
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply preconditioning to numpy array.

        Args:
            data: Input array of shape [batch, time, dim] or [batch, time]

        Returns:
            preconditioned_data: Preconditioned array
            coeffs: Preconditioning coefficients for reversal
        """
        if not self.enable_preconditioning:
            return data, None

        # Handle batch dimension
        original_shape = data.shape
        if data.ndim == 2:
            # [batch, time] -> add dim
            data = data[..., np.newaxis]

        batch_size = data.shape[0]
        preconditioned_batch = []

        # Apply preconditioning to each series in batch
        for i in range(batch_size):
            series = data[i]  # [time, dim]

            # Create data entry for transform
            data_entry = {"target": series}

            # Apply preconditioning
            preconditioned_entry = self.preconditioner(data_entry)
            preconditioned_series = preconditioned_entry["target"]

            preconditioned_batch.append(preconditioned_series)

        preconditioned_data = np.stack(preconditioned_batch, axis=0)

        # Restore original shape if needed
        if len(original_shape) == 2:
            preconditioned_data = preconditioned_data.squeeze(-1)

        # Get coefficients for reversal
        coeffs = self.preconditioner.coeffs if self.enable_preconditioning else None

        return preconditioned_data, coeffs

    def _reverse_preconditioning_numpy(
        self,
        predictions: np.ndarray,
        coeffs: Optional[np.ndarray],
        context: np.ndarray,
    ) -> np.ndarray:
        """
        Reverse preconditioning on predictions.

        Args:
            predictions: Predictions in preconditioned space [batch, sample, time, dim]
            coeffs: Preconditioning coefficients
            context: Context data for computing reversal [batch, context_time, dim]

        Returns:
            reversed_predictions: Predictions in original scale
        """
        if not self.enable_preconditioning or coeffs is None:
            return predictions

        # predictions shape: [batch, sample, pred_time, dim]
        batch_size, num_samples, pred_len, dim = predictions.shape

        reversed_batch = []

        for b in range(batch_size):
            reversed_samples = []

            for s in range(num_samples):
                pred_sample = predictions[b, s]  # [pred_time, dim]
                context_b = context[b]  # [context_time, dim]

                # Create data entry with preconditioning metadata
                data_entry = {
                    "prediction": pred_sample,
                    "precondition_coeffs": coeffs,
                    "precondition_degree": self.precondition_degree,
                    "precondition_type": self.precondition_type,
                    "precondition_enabled": True,
                }

                # Apply reversal with explicit context
                reversed_entry = self.reverse_preconditioner(data_entry, context=context_b)
                reversed_pred = reversed_entry["prediction"]

                reversed_samples.append(reversed_pred)

            reversed_batch.append(np.stack(reversed_samples, axis=0))

        return np.stack(reversed_batch, axis=0)

    def create_predictor(
        self,
        batch_size: int,
        device: str = "auto",
    ) -> Predictor:
        """
        Create predictor with preconditioning support.

        This creates a predictor with:
        1. Input preconditioning transform (if enabled)
        2. Standard forecast transforms
        3. Output preconditioning reversal (if enabled)
        """
        # If preconditioning is disabled, return standard predictor
        if not self.enable_preconditioning:
            return super().create_predictor(batch_size, device)

        # Create the standard transform and instance splitter manually
        # (can't use super().create_predictor() due to transform incompatibility)
        from gluonts.transform import Chain as GluonTSChain

        ts_fields = []
        if self.hparams.feat_dynamic_real_dim > 0:
            ts_fields.append("feat_dynamic_real")
            ts_fields.append("observed_feat_dynamic_real")
        past_ts_fields = []
        if self.hparams.past_feat_dynamic_real_dim > 0:
            past_ts_fields.append("past_feat_dynamic_real")
            past_ts_fields.append("past_observed_feat_dynamic_real")

        instance_splitter = TFTInstanceSplitter(
            instance_sampler=TestSplitSampler(),
            past_length=self.past_length,
            future_length=self.hparams.prediction_length,
            observed_value_field="observed_target",
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )

        # Create input transform with preconditioning
        # We need to apply preconditioning BEFORE the standard transforms
        # Since uni2ts and GluonTS Transformation classes are incompatible,
        # we create a wrapper that inherits from GluonTS Transformation

        from gluonts.transform import Transformation as GluonTSTransformation

        class PreconditioningWrapper(GluonTSTransformation):
            """Wrapper to apply preconditioning within GluonTS transform chain."""
            def __init__(self, preconditioner, base_transform, degree):
                self.preconditioner = preconditioner
                self.base_transform = base_transform
                self.degree = degree
                self.stored_contexts = []  # Store contexts for reversal

            def __call__(self, data_it: Iterable[DataEntry], is_train: bool) -> Iterable[DataEntry]:
                """
                Apply preconditioning and base transforms to a stream of data entries.

                Args:
                    data_it: Iterable of data entries
                    is_train: Whether in training mode

                Yields:
                    Transformed data entries
                """
                # First, apply preconditioning to all items and store contexts
                def precondition_and_store():
                    for data_entry in data_it:
                        # Store original context BEFORE preconditioning for reversal
                        if "target" in data_entry:
                            original_target = np.array(data_entry["target"])
                            if original_target.ndim == 1:
                                context = original_target[-self.degree:]
                            elif original_target.ndim == 2:
                                context = original_target[-self.degree:, :]
                            else:
                                context = None
                            # Store context in list
                            self.stored_contexts.append(context)

                        # Apply preconditioning first (uni2ts transform, doesn't need is_train)
                        preconditioned_entry = self.preconditioner(data_entry)
                        yield preconditioned_entry

                # Then apply base transforms to the preconditioned data
                # base_transform is a GluonTS Chain that expects an iterable
                yield from self.base_transform(precondition_and_store(), is_train)

        # Create preconditioning transform
        precond_transform = PolynomialPrecondition(
            polynomial_type=self.precondition_type,
            degree=self.precondition_degree,
            target_field="target",
            enabled=True,
            store_original=True,
        )

        # Get base transforms
        base_transform = super().get_default_transform()

        # Wrap them together
        input_transform_with_precond = PreconditioningWrapper(
            precond_transform,
            base_transform,
            self.precondition_degree
        )

        # Chain with instance splitter (instance_splitter is GluonTS transform)
        input_transform = GluonTSChain([input_transform_with_precond, instance_splitter])

        # Create base predictor
        base_predictor = PyTorchPredictor(
            input_names=self.prediction_input_names,
            prediction_net=self,
            batch_size=batch_size,
            prediction_length=self.hparams.prediction_length,
            input_transform=input_transform,
            device=device,
        )

        # Wrap with preconditioning reversal predictor
        # Pass the wrapper so we can access stored contexts
        return PreconditionReversingPredictor(
            base_predictor=base_predictor,
            precondition_coeffs=self.preconditioner.coeffs,
            precondition_degree=self.precondition_degree,
            precondition_type=self.precondition_type,
            context_wrapper=input_transform_with_precond,
            reverse_output=self.reverse_output,
        )

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
        """
        Forward pass - preconditioning is now handled in the transform pipeline.

        The input data has already been preconditioned by get_default_transform(),
        so we just call the parent forward method. The output will be in the
        preconditioned space, and will be reversed by PreconditionReversingPredictor.
        """
        # Input is already preconditioned by the transform pipeline
        # Output will be reversed by the predictor wrapper
        return super().forward(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
            num_samples=num_samples,
        )


def create_precond_forecast_from_checkpoint(
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
) -> MoiraiForecastPrecond:
    """
    Create MoiraiForecastPrecond from a checkpoint.

    This function loads a model checkpoint and wraps it with preconditioning support.

    IMPORTANT: The preconditioning parameters must match those used during training!

    Args:
        checkpoint_path: Path to model checkpoint
        enable_preconditioning: Must match training config
        precondition_type: Must match training config
        precondition_degree: Must match training config
        reverse_output: Whether to reverse preconditioning on outputs (True: output in original scale, False: output in preconditioned space)
    """
    # Load base model from checkpoint
    model = MoiraiForecastPrecond.load_from_checkpoint(
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
        strict=False,  # Allow loading with extra/missing keys
    )

    return model
