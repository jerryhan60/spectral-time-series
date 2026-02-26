"""
Mid-training GIFT-Eval callback.

Runs GIFT-Eval quick evaluation (8 datasets) every N epochs during
pretraining and logs MASE metrics to TensorBoard. Runs the full
97-config evaluation at the end of training.
"""

import sys
import warnings
from contextlib import contextmanager

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

# GIFT-Eval quick subset (matches eval_gifteval.py QUICK_CONFIGS)
QUICK_CONFIGS = [
    ("m4_monthly", "short"),
    ("electricity/H", "short"),
    ("hospital", "short"),
    ("jena_weather/H", "short"),
    ("ett1/H", "short"),
    ("saugeenday/D", "short"),
    ("covid_deaths", "short"),
    ("m4_hourly", "short"),
]

# Full 97-config GIFT-Eval benchmark (matches eval_gifteval.py GIFTEVAL_CONFIGS)
FULL_CONFIGS = [
    ("bitbrains_fast_storage/5T", "short"), ("bitbrains_fast_storage/5T", "medium"),
    ("bitbrains_fast_storage/5T", "long"), ("bitbrains_fast_storage/H", "short"),
    ("bitbrains_rnd/5T", "short"), ("bitbrains_rnd/5T", "medium"),
    ("bitbrains_rnd/5T", "long"), ("bitbrains_rnd/H", "short"),
    ("bizitobs_application", "short"), ("bizitobs_application", "medium"),
    ("bizitobs_application", "long"), ("bizitobs_l2c/5T", "short"),
    ("bizitobs_l2c/5T", "medium"), ("bizitobs_l2c/5T", "long"),
    ("bizitobs_l2c/H", "short"), ("bizitobs_l2c/H", "medium"),
    ("bizitobs_l2c/H", "long"), ("bizitobs_service", "short"),
    ("bizitobs_service", "medium"), ("bizitobs_service", "long"),
    ("car_parts_with_missing", "short"), ("hierarchical_sales/D", "short"),
    ("hierarchical_sales/W", "short"), ("restaurant", "short"),
    ("covid_deaths", "short"), ("hospital", "short"),
    ("electricity/15T", "short"), ("electricity/15T", "medium"),
    ("electricity/15T", "long"), ("electricity/D", "short"),
    ("electricity/H", "short"), ("electricity/H", "medium"),
    ("electricity/H", "long"), ("electricity/W", "short"),
    ("ett1/15T", "short"), ("ett1/15T", "medium"), ("ett1/15T", "long"),
    ("ett1/D", "short"), ("ett1/H", "short"), ("ett1/H", "medium"),
    ("ett1/H", "long"), ("ett1/W", "short"),
    ("ett2/15T", "short"), ("ett2/15T", "medium"), ("ett2/15T", "long"),
    ("ett2/D", "short"), ("ett2/H", "short"), ("ett2/H", "medium"),
    ("ett2/H", "long"), ("ett2/W", "short"),
    ("solar/10T", "short"), ("solar/10T", "medium"), ("solar/10T", "long"),
    ("solar/D", "short"), ("solar/H", "short"), ("solar/H", "medium"),
    ("solar/H", "long"), ("solar/W", "short"),
    ("jena_weather/10T", "short"), ("jena_weather/10T", "medium"),
    ("jena_weather/10T", "long"), ("jena_weather/D", "short"),
    ("jena_weather/H", "short"), ("jena_weather/H", "medium"),
    ("jena_weather/H", "long"),
    ("kdd_cup_2018_with_missing/D", "short"),
    ("kdd_cup_2018_with_missing/H", "short"),
    ("kdd_cup_2018_with_missing/H", "medium"),
    ("kdd_cup_2018_with_missing/H", "long"),
    ("temperature_rain_with_missing", "short"),
    ("saugeenday/D", "short"), ("saugeenday/M", "short"),
    ("saugeenday/W", "short"), ("us_births/D", "short"),
    ("us_births/M", "short"), ("us_births/W", "short"),
    ("LOOP_SEATTLE/5T", "short"), ("LOOP_SEATTLE/5T", "medium"),
    ("LOOP_SEATTLE/5T", "long"), ("LOOP_SEATTLE/D", "short"),
    ("LOOP_SEATTLE/H", "short"), ("LOOP_SEATTLE/H", "medium"),
    ("LOOP_SEATTLE/H", "long"),
    ("SZ_TAXI/15T", "short"), ("SZ_TAXI/15T", "medium"),
    ("SZ_TAXI/15T", "long"), ("SZ_TAXI/H", "short"),
    ("m4_daily", "short"), ("m4_hourly", "short"), ("m4_monthly", "short"),
    ("m4_quarterly", "short"), ("m4_weekly", "short"), ("m4_yearly", "short"),
    ("M_DENSE/D", "short"), ("M_DENSE/H", "short"),
    ("M_DENSE/H", "medium"), ("M_DENSE/H", "long"),
]


@contextmanager
def _eval_mode(module):
    """Temporarily set module to eval mode, then restore training mode."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        if was_training:
            module.train()


class GIFTEvalCallback(Callback):
    def __init__(
        self,
        eval_every_n_epochs: int = 200,
        context_length: int = 1000,
        batch_size: int = 32,
    ):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.context_length = context_length
        self.batch_size = batch_size
        self._gift_eval_available = None

    def _ensure_imports(self):
        """Lazy-import gift_eval and gluonts metrics (paths may not be ready at init)."""
        if self._gift_eval_available is not None:
            return self._gift_eval_available

        try:
            gift_eval_src = "/scratch/gpfs/EHAZAN/jh1161/gifteval/gift-eval/src"
            if gift_eval_src not in sys.path:
                sys.path.insert(0, gift_eval_src)

            from dotenv import load_dotenv
            load_dotenv("/scratch/gpfs/EHAZAN/jh1161/uni2ts/.env")

            from gift_eval.data import Dataset  # noqa: F401
            from gluonts.ev.metrics import MAE, MSE, MASE  # noqa: F401
            from gluonts.time_feature import get_seasonality  # noqa: F401
            from uni2ts.eval_util.evaluation import evaluate_model  # noqa: F401

            self._gift_eval_available = True
        except ImportError as e:
            warnings.warn(f"GIFTEvalCallback: imports failed, disabling callback: {e}")
            self._gift_eval_available = False

        return self._gift_eval_available

    def _run_eval(self, trainer, pl_module, configs, prefix, label):
        """Run GIFT-Eval on a list of (dataset_name, term) configs."""
        if not trainer.is_global_zero:
            return
        if not self._ensure_imports():
            return

        from gift_eval.data import Dataset
        from gluonts.ev.metrics import MAE, MSE, MASE
        from gluonts.time_feature import get_seasonality
        from scipy.stats import gmean

        from uni2ts.eval_util.evaluation import evaluate_model
        from uni2ts.model.moirai2.forecast import Moirai2Forecast

        device = pl_module.device
        module = pl_module.module
        mase_values = {}

        print(f"\n[GIFTEvalCallback] {label} ({len(configs)} configs)...")

        with _eval_mode(module), torch.no_grad():
            for i, (dataset_name, term) in enumerate(configs):
                try:
                    ds_check = Dataset(name=dataset_name, term=term, to_univariate=False)
                    is_mv = ds_check.target_dim > 1
                    dataset = Dataset(name=dataset_name, term=term, to_univariate=is_mv)
                    prediction_length = dataset.prediction_length

                    forecast = Moirai2Forecast(
                        prediction_length=prediction_length,
                        target_dim=1,
                        feat_dynamic_real_dim=0,
                        past_feat_dynamic_real_dim=0,
                        context_length=self.context_length,
                        module=module,
                    )
                    forecast = forecast.to(device)

                    try:
                        seasonality = get_seasonality(dataset.freq)
                    except Exception:
                        seasonality = 1

                    predictor = forecast.create_predictor(
                        batch_size=self.batch_size, device=device
                    )

                    metrics = [MAE(), MSE(), MASE()]
                    result = evaluate_model(
                        predictor,
                        test_data=dataset.test_data,
                        metrics=metrics,
                        batch_size=self.batch_size,
                        axis=None,
                        mask_invalid_label=True,
                        allow_nan_forecast=False,
                        seasonality=seasonality,
                    )

                    mase_val = float(result["MASE[0.5]"].values[0])
                    key = f"{dataset_name}/{term}"
                    mase_values[key] = mase_val
                    print(f"  [{i+1}/{len(configs)}] {key}: MASE={mase_val:.4f}")

                    del forecast, predictor
                except Exception as e:
                    print(f"  [{i+1}/{len(configs)}] {dataset_name}/{term}: ERROR - {e}")
                finally:
                    torch.cuda.empty_cache()

        # Log metrics
        if mase_values and trainer.logger is not None:
            step = trainer.global_step
            valid = [v for v in mase_values.values() if v > 0 and np.isfinite(v)]
            if valid:
                geo_mean = float(gmean(valid))
                trainer.logger.log_metrics(
                    {f"{prefix}/geo_mean_mase": geo_mean}, step=step
                )
                print(f"  Geo Mean MASE: {geo_mean:.4f}")
            for key, mase in mase_values.items():
                safe_name = key.replace("/", "_")
                trainer.logger.log_metrics(
                    {f"{prefix}/mase_{safe_name}": mase}, step=step
                )

        print(f"[GIFTEvalCallback] {label} done.\n")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.eval_every_n_epochs != 0:
            return
        self._run_eval(
            trainer, pl_module, QUICK_CONFIGS, prefix="eval",
            label=f"Quick eval at epoch {epoch}",
        )

    def on_train_end(self, trainer, pl_module):
        self._run_eval(
            trainer, pl_module, FULL_CONFIGS, prefix="eval_full",
            label="Full GIFT-Eval at end of training",
        )
