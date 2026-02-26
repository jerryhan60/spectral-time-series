# Inference-Time Preconditioning: Experiment Summary

## Research Question

Can we improve forecasting accuracy of a frozen pretrained model (Moirai v1 small) by running it on both raw and preconditioned inputs, then combining the forecasts — without any retraining?

## Setup

- **Base model**: Salesforce/moirai-1.1-R-small (frozen, from HuggingFace)
- **Preconditioning**: Chebyshev monic polynomial FIR filter, degrees 2-8, stride 1
- **Benchmark**: GIFT-Eval (8-dataset quick subset, 57,406 total series)
- **Strategies tested**:
  - A1: Replace input with preconditioned series, forecast, reverse using raw forecast as anchor
  - A2: Feed FIR residual (preconditioned - raw) to model, add raw forecast back
  - Approach C: Learned adapter combining raw + preconditioned forecasts

## Results

### Zero-Parameter Methods (Approach A)

**A1 (Replace + Reverse)**: All individual preconditioned passes worse than raw. Best is d=2 at +2.7% worse. Degradation increases monotonically with degree. Best combination method (inverse-variance) is +4.0% worse.

**A2 (Residual)**: Catastrophically bad — 5-18x worse than raw. The FIR residual is not a valid time series; the model produces garbage on it.

| Method | Geo Mean MASE | vs Raw |
|--------|--------------|--------|
| Raw (baseline) | 1.898 | — |
| A1_d2 (best individual) | 1.949 | +2.7% |
| A1_invvar (best combo) | 1.974 | +4.0% |
| A2_d2 | 9.987 | +426% |

### Per-Series Oracle Analysis (Key Finding)

While dataset-level averages all favor raw, the **per-series** picture is very different:

| Metric | Value |
|--------|-------|
| Series where raw is best | 24,786 / 57,406 (43.2%) |
| Series where some precond is better | 32,620 / 57,406 **(56.8%)** |
| Oracle improvement (perfect selector) | **6.77%** |
| Mean improvement when precond wins | 13.1% |
| Max improvement on a single series | 82.5% |

**Every single dataset** has series that benefit from preconditioning (oracle improvement 1.6% - 7.9% per dataset). The negative dataset-level results happen because the harming-series dominate the mean.

### Learned Adapter (Approach C)

Trained to combine raw + A1_d2/d4/d6 forecasts via leave-one-dataset-out cross-validation:

| Adapter | Params | Aggregate Improvement | Positive Folds |
|---------|--------|----------------------|----------------|
| Linear (global softmax weights) | 3 | +0.17% | 5/8 |
| Feature-conditioned MLP | ~180 | +0.18% | 7/8 |

Linear adapter learns ~70% raw, ~18% d=2, ~5% d=4, ~5% d=6. MLP uses 8 series features (autocorrelation, variance, spectral entropy, trend, etc.) to condition weights per-series.

## Key Takeaways

1. **Frozen model + preconditioned input does not work** at the dataset level. The model's internal representations are calibrated for raw distributions.

2. **But per-series signal exists** (6.77% oracle ceiling). The majority of individual series benefit from preconditioning — the problem is identifying which ones.

3. **Simple adapters capture only ~0.18% of the 6.77% potential.** The per-series selection problem is hard — basic statistical features of the input series don't discriminate well.

4. **This strengthens the case for training-time hint mode.** Our best trained model (hint d=4 s=16 with 10% dropout, MASE 1.1802) achieves -5.0% improvement over baseline — the model learns during training how to use preconditioning information. A frozen model cannot.

5. **The 6.6% gap (oracle minus adapter)** suggests that with a much more sophisticated selector (e.g., trained on orders of magnitude more data, or jointly with the model), inference-time preconditioning could work. This remains an open direction.

## Implications for Future Work

- **Training-time preconditioning is necessary**: Simply feeding preconditioned data to a frozen model is insufficient.
- **Per-series diversity is real**: Different series genuinely benefit from different polynomial degrees — this motivates multi-scale or adaptive preconditioning at training time.
- **Adapter on top of trained hint model**: An interesting follow-up would be to apply the adapter approach to a model already trained with hint mode, where the individual passes should be much stronger.

## Files

- `gifteval/eval_inference_precond.py` — Zero-parameter evaluation script
- `gifteval/eval_adapter_precond.py` — Oracle analysis + learned adapter training
- `gifteval/results/adapter_forecasts_moirai-1.1-R-small_20260223_202227.pkl` — Saved per-series forecasts (462MB)
- `gifteval/results/adapter_oracle_moirai-1.1-R-small_20260223_211054.csv` — Per-series oracle results
- `docs/plans/2026-02-23-inference-time-preconditioning-design.md` — Full design document with all results
