# GIFT-Eval Evaluation Report

**Generated:** 2026-02-25 22:09:10

## Model Information

| Property | Value |
|----------|-------|
| **Checkpoint** | `moirai-2.0-R-small` |
| **Architecture** | Moirai (HuggingFace) |
| **Parameters** | ~14M (small) / ~90M (base) |
| **d_model** | N/A |
| **num_layers** | N/A |
| **Model Type** | pretrained |

### Checkpoint Path
```
HuggingFace: Salesforce/moirai-2.0-R-small
```

## Aggregate Metrics (GIFT-Eval Leaderboard Style)

| Metric | Value | Description |
|--------|-------|-------------|
| **Geometric Mean MASE** | **1.0236** | Primary leaderboard metric (scale-invariant) |
| Arithmetic Mean MASE | 1.5394 | Simple average |
| Median MASE | 0.9071 | Robust central tendency |
| MASE < 1.0 | 57/97 | Configs beating naive baseline |
| Min MASE | 0.2906 | Best single config |
| Max MASE | 38.2072 | Worst single config |

### Comparison to Reference Models

| Model | Params | MASE (Geo Mean) | MASE (Mean) | MASE < 1 |
|-------|--------|-----------------|-------------|----------|
| **This Model** | ~14M (small) / ~90M (base) | **1.0236** | 1.5394 | 57/97 |
| Moirai-small (official) | ~14M | 1.323 | 1.958 | 27/97 |
| Moirai-base (official) | ~90M | 1.259 | 2.019 | 40/97 |

*Note: Lower MASE is better. MASE < 1.0 means the model outperforms a seasonal naive baseline.*

## Per-Configuration Results

| Configuration | MASE | Status |
|---------------|------|--------|
| bizitobs_l2c/5T/short | 0.2906 | ✅ |
| jena_weather/10T/short | 0.3154 | ✅ |
| us_births/D/short | 0.3733 | ✅ |
| bizitobs_l2c/H/short | 0.5036 | ✅ |
| bizitobs_l2c/H/medium | 0.5080 | ✅ |
| SZ_TAXI/15T/long | 0.5303 | ✅ |
| bizitobs_l2c/5T/medium | 0.5360 | ✅ |
| jena_weather/H/short | 0.5365 | ✅ |
| LOOP_SEATTLE/5T/short | 0.5418 | ✅ |
| SZ_TAXI/15T/short | 0.5464 | ✅ |
| SZ_TAXI/15T/medium | 0.5520 | ✅ |
| SZ_TAXI/H/short | 0.5643 | ✅ |
| bizitobs_l2c/5T/long | 0.5799 | ✅ |
| bizitobs_l2c/H/long | 0.6147 | ✅ |
| jena_weather/10T/medium | 0.6510 | ✅ |
| solar/10T/short | 0.6575 | ✅ |
| restaurant/short | 0.6809 | ✅ |
| M_DENSE/D/short | 0.6851 | ✅ |
| bitbrains_fast_storage/5T/short | 0.6867 | ✅ |
| ett1/15T/short | 0.6870 | ✅ |
| M_DENSE/H/medium | 0.7039 | ✅ |
| M_DENSE/H/long | 0.7182 | ✅ |
| hierarchical_sales/W/short | 0.7257 | ✅ |
| jena_weather/10T/long | 0.7331 | ✅ |
| ett2/H/short | 0.7354 | ✅ |
| saugeenday/M/short | 0.7358 | ✅ |
| hierarchical_sales/D/short | 0.7468 | ✅ |
| ett2/15T/short | 0.7664 | ✅ |
| hospital/short | 0.7719 | ✅ |
| M_DENSE/H/short | 0.7930 | ✅ |
| us_births/M/short | 0.8040 | ✅ |
| m4_hourly/short | 0.8122 | ✅ |
| jena_weather/H/medium | 0.8173 | ✅ |
| car_parts_with_missing/short | 0.8328 | ✅ |
| ett1/H/short | 0.8362 | ✅ |
| electricity/15T/medium | 0.8413 | ✅ |
| solar/H/medium | 0.8482 | ✅ |
| LOOP_SEATTLE/H/short | 0.8503 | ✅ |
| LOOP_SEATTLE/5T/medium | 0.8646 | ✅ |
| electricity/15T/short | 0.8698 | ✅ |
| us_births/W/short | 0.8715 | ✅ |
| bizitobs_service/short | 0.8739 | ✅ |
| solar/H/short | 0.8794 | ✅ |
| solar/H/long | 0.8834 | ✅ |
| electricity/H/short | 0.8869 | ✅ |
| LOOP_SEATTLE/D/short | 0.8908 | ✅ |
| bitbrains_fast_storage/5T/long | 0.8908 | ✅ |
| LOOP_SEATTLE/H/long | 0.9038 | ✅ |
| electricity/15T/long | 0.9071 | ✅ |
| ett2/15T/medium | 0.9196 | ✅ |
| LOOP_SEATTLE/5T/long | 0.9282 | ✅ |
| LOOP_SEATTLE/H/medium | 0.9398 | ✅ |
| m4_monthly/short | 0.9435 | ✅ |
| ett2/15T/long | 0.9591 | ✅ |
| kdd_cup_2018_with_missing/H/short | 0.9620 | ✅ |
| bitbrains_fast_storage/5T/medium | 0.9919 | ✅ |
| solar/10T/medium | 0.9995 | ✅ |
| solar/10T/long | 1.0108 | ❌ |
| kdd_cup_2018_with_missing/H/long | 1.0269 | ❌ |
| jena_weather/H/long | 1.0388 | ❌ |
| ett1/15T/medium | 1.0390 | ❌ |
| ett2/H/long | 1.0515 | ❌ |
| bitbrains_fast_storage/H/short | 1.0667 | ❌ |
| ett1/15T/long | 1.0669 | ❌ |
| solar/D/short | 1.0743 | ❌ |
| jena_weather/D/short | 1.0746 | ❌ |
| kdd_cup_2018_with_missing/H/medium | 1.0750 | ❌ |
| electricity/H/medium | 1.0843 | ❌ |
| ett2/H/medium | 1.0853 | ❌ |
| bizitobs_service/medium | 1.1320 | ❌ |
| ett2/W/short | 1.1894 | ❌ |
| m4_quarterly/short | 1.1929 | ❌ |
| kdd_cup_2018_with_missing/D/short | 1.2421 | ❌ |
| electricity/H/long | 1.2622 | ❌ |
| ett1/H/medium | 1.3043 | ❌ |
| ett2/D/short | 1.3261 | ❌ |
| temperature_rain_with_missing/short | 1.3497 | ❌ |
| bizitobs_service/long | 1.3712 | ❌ |
| solar/W/short | 1.3834 | ❌ |
| electricity/D/short | 1.3838 | ❌ |
| saugeenday/W/short | 1.4070 | ❌ |
| bizitobs_application/short | 1.4386 | ❌ |
| ett1/H/long | 1.4650 | ❌ |
| electricity/W/short | 1.5730 | ❌ |
| ett1/W/short | 1.5910 | ❌ |
| bitbrains_rnd/5T/short | 1.6499 | ❌ |
| ett1/D/short | 1.7933 | ❌ |
| m4_weekly/short | 2.1290 | ❌ |
| bizitobs_application/medium | 2.4790 | ❌ |
| saugeenday/D/short | 2.7014 | ❌ |
| m4_daily/short | 3.0793 | ❌ |
| bitbrains_rnd/5T/long | 3.3276 | ❌ |
| bizitobs_application/long | 3.4279 | ❌ |
| m4_yearly/short | 3.5002 | ❌ |
| bitbrains_rnd/5T/medium | 4.3861 | ❌ |
| bitbrains_rnd/H/short | 5.8560 | ❌ |
| covid_deaths/short | 38.2072 | ❌ |

## Evaluation Settings

| Setting | Value |
|---------|-------|
| Context Length | 4000 |
| Patch Size | 32 |
| Batch Size | 64 |
| Device | CUDA |
| Num Configs | 97 |

---
*Report generated by eval_gifteval.py*
