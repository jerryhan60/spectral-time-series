# Hint Preconditioning: Seed Robustness Analysis

## Method Overview

**Base model**: Moirai2-Small (11.4M params), a universal time series forecasting transformer.
It splits input time series into fixed-length patches (size 16) and processes them with a 6-layer transformer encoder.

**The intervention ("hint preconditioning")**: Before patching, we apply a fixed Chebyshev polynomial FIR filter
to the raw time series and concatenate the filtered signal as an extra input channel. Concretely:

```
h[t] = T_d(B^s) x[t]   where B is the backshift operator, s=16 (stride = patch size)

For degree d=4:
h[t] = x[t] - 2*x[t-32] + 0.5*x[t-64]
```

This "hint" channel injects information from neighboring patches directly into each patch's embedding —
the term `x[t-32]` reaches back 2 patches, `x[t-64]` reaches back 4 patches. The model sees both the
original signal and this filtered version as input channels.

**Three variants tested:**

| Variant | Description | Extra Params |
|---------|-------------|-------------|
| **HD10** | Single Chebyshev filter (degree 4, stride 16), 10% hint dropout during training | 12K (0.11%) |
| **MS46** | Two filters (degree 4 + degree 6, stride 16), no dropout | 24K (0.21%) |
| **MSHD10** | Two filters (degree 4 + degree 6, stride 16), 10% hint dropout | 24K (0.21%) |

**Training**: All models trained for 10,000 steps on LOTSA v1 (Moirai2 config) with cosine LR schedule,
LR=1e-3, warmup=1000 steps, batch size=256. Only difference between runs is the random seed.

**Evaluation**: GIFT-Eval benchmark — 97 dataset×horizon configurations. Primary metric: geometric mean MASE
(lower is better). Statistical test: paired sign test per configuration.

---

## 1. Absolute MASE by Seed

Geometric mean MASE across 97 GIFT-Eval configurations. Lower is better.

| Seed | Baseline | HD10 | MS46 | MSHD10 |
|------|----------|------|------|--------|
| 0 | 1.2185 | **1.1577** | 1.1999 | 1.2029 |
| 1 | 1.1955 | 1.1902 | 1.1792 | **1.1729** |
| 2 | 1.2134 | 1.1798 | **1.1711** | 1.1876 |
| 7 | 1.2198 | **1.1698** | 1.2119 | 1.1885 |
| 42 | 1.2111 | 1.1852 | **1.1670** | 1.1794 |
| **Mean** | **1.2116** | **1.1765** | **1.1858** | **1.1863** |
| Std | 0.0087 | 0.0116 | 0.0173 | 0.0101 |

---

## 2. Pairwise Comparison vs Baseline (per seed)

### HD10 vs Baseline

| Seed | BL MASE | HD10 MASE | Improvement | Wins/97 | p-value | Sig |
|------|---------|------------|-------------|---------|---------|-----|
| 0 | 1.2185 | 1.1577 | +4.99% | 66/97 | 2.45e-04 | *** |
| 1 | 1.1955 | 1.1902 | +0.44% | 54/97 | 1.55e-01 | ns |
| 2 | 1.2134 | 1.1798 | +2.77% | 70/97 | 7.38e-06 | *** |
| 7 | 1.2198 | 1.1698 | +4.11% | 78/97 | 5.77e-10 | *** |
| 42 | 1.2111 | 1.1852 | +2.13% | 61/97 | 7.19e-03 | ** |
| **Pooled** | | | | **329/485** | **1.46e-15** | ******* |

### MS46 vs Baseline

| Seed | BL MASE | MS46 MASE | Improvement | Wins/97 | p-value | Sig |
|------|---------|------------|-------------|---------|---------|-----|
| 0 | 1.2185 | 1.1999 | +1.52% | 60/97 | 1.25e-02 | * |
| 1 | 1.1955 | 1.1792 | +1.37% | 59/97 | 2.09e-02 | * |
| 2 | 1.2134 | 1.1711 | +3.48% | 72/97 | 9.59e-07 | *** |
| 7 | 1.2198 | 1.2119 | +0.65% | 65/97 | 5.25e-04 | *** |
| 42 | 1.2111 | 1.1670 | +3.63% | 74/97 | 1.01e-07 | *** |
| **Pooled** | | | | **330/485** | **6.83e-16** | ******* |

### MSHD10 vs Baseline

| Seed | BL MASE | MSHD10 MASE | Improvement | Wins/97 | p-value | Sig |
|------|---------|------------|-------------|---------|---------|-----|
| 0 | 1.2185 | 1.2029 | +1.28% | 54/97 | 1.55e-01 | ns |
| 1 | 1.1955 | 1.1729 | +1.89% | 62/97 | 3.98e-03 | ** |
| 2 | 1.2134 | 1.1876 | +2.12% | 64/97 | 1.08e-03 | ** |
| 7 | 1.2198 | 1.1885 | +2.57% | 73/97 | 3.20e-07 | *** |
| 42 | 1.2111 | 1.1794 | +2.61% | 62/97 | 3.98e-03 | ** |
| **Pooled** | | | | **315/485** | **2.24e-11** | ******* |

---

## 3. Per-Config Robustness (HD10 vs Baseline, averaged across 5 seeds)

When we average each config's MASE across all 5 seeds to remove seed noise:

- **HD10 wins 72/97 configs** on seed-averaged MASE
- Sign test: p = 9.59e-07
- Wilcoxon signed-rank: p = 9.04e-09
- Paired t-test: t = 4.844, p = 2.44e-06
- Mean improvement: **2.77%**
- Median improvement: **1.47%**

### Seed-win distribution (per config, how many of 5 seeds does HD10 win?)

| HD10 wins X/5 seeds | Configs | % |
|---------------------|---------|---|
| 0/5 | 2 | 2.1% |
| 1/5 | 6 | 6.2% |
| 2/5 | 15 | 15.5% |
| 3/5 | 26 | 26.8% |
| 4/5 | 25 | 25.8% |
| 5/5 | 23 | 23.7% |

- **74/97 configs (76.3%)**: HD10 wins majority of seeds
- **23 configs**: HD10 wins all 5 seeds
- **2 configs**: BL wins all 5 seeds

### Full per-config results (seed-averaged MASE)

| Config | BL Mean | HD10 Mean | Δ% | HD10 wins X/5 seeds |
|--------|---------|-----------|-----|---------------------|
| solar/10T/long | 1.5383 | 1.0983 | +28.6% | 4/5 |
| solar/10T/medium | 1.2806 | 1.0509 | +17.9% | 4/5 |
| LOOP_SEATTLE/H/long | 1.3470 | 1.1059 | +17.9% | 5/5 |
| LOOP_SEATTLE/H/medium | 1.3278 | 1.1209 | +15.6% | 5/5 |
| us_births/M/short | 0.9678 | 0.8230 | +15.0% | 3/5 |
| bizitobs_l2c/H/medium | 1.0014 | 0.8812 | +12.0% | 4/5 |
| LOOP_SEATTLE/5T/long | 0.7822 | 0.7049 | +9.9% | 4/5 |
| bizitobs_l2c/H/short | 0.7975 | 0.7223 | +9.4% | 4/5 |
| m4_hourly/short | 1.1274 | 1.0227 | +9.3% | 5/5 |
| LOOP_SEATTLE/5T/medium | 0.7388 | 0.6705 | +9.2% | 4/5 |
| LOOP_SEATTLE/H/short | 1.0354 | 0.9540 | +7.9% | 5/5 |
| us_births/D/short | 0.5199 | 0.4807 | +7.5% | 5/5 |
| ett1/15T/long | 1.1361 | 1.0623 | +6.5% | 4/5 |
| us_births/W/short | 1.5519 | 1.4515 | +6.5% | 5/5 |
| electricity/15T/medium | 1.1138 | 1.0453 | +6.2% | 5/5 |
| M_DENSE/H/long | 0.7772 | 0.7294 | +6.1% | 5/5 |
| bizitobs_l2c/H/long | 1.1064 | 1.0398 | +6.0% | 4/5 |
| m4_yearly/short | 4.3857 | 4.1267 | +5.9% | 4/5 |
| m4_weekly/short | 2.6386 | 2.4874 | +5.7% | 5/5 |
| electricity/15T/long | 1.1568 | 1.0905 | +5.7% | 5/5 |
| SZ_TAXI/15T/long | 0.5797 | 0.5471 | +5.6% | 5/5 |
| jena_weather/H/long | 1.1454 | 1.0849 | +5.3% | 3/5 |
| SZ_TAXI/15T/medium | 0.5970 | 0.5664 | +5.1% | 5/5 |
| bizitobs_application/medium | 6.4487 | 6.1241 | +5.0% | 3/5 |
| ett2/15T/medium | 1.0325 | 0.9825 | +4.8% | 4/5 |
| ett2/15T/short | 0.8529 | 0.8133 | +4.6% | 4/5 |
| electricity/W/short | 2.0229 | 1.9303 | +4.6% | 5/5 |
| electricity/H/long | 1.3767 | 1.3145 | +4.5% | 5/5 |
| M_DENSE/H/medium | 0.7477 | 0.7142 | +4.5% | 5/5 |
| ett2/15T/long | 1.0565 | 1.0096 | +4.4% | 5/5 |
| solar/H/long | 1.0653 | 1.0186 | +4.4% | 3/5 |
| ett1/15T/medium | 1.1066 | 1.0601 | +4.2% | 5/5 |
| electricity/H/medium | 1.2021 | 1.1576 | +3.7% | 5/5 |
| bizitobs_service/short | 1.6294 | 1.5735 | +3.4% | 4/5 |
| bizitobs_application/short | 2.6795 | 2.5916 | +3.3% | 3/5 |
| ett1/W/short | 1.5605 | 1.5118 | +3.1% | 3/5 |
| bitbrains_fast_storage/5T/long | 1.0342 | 1.0048 | +2.8% | 4/5 |
| bizitobs_service/medium | 3.4919 | 3.3930 | +2.8% | 3/5 |
| ett1/15T/short | 0.7328 | 0.7134 | +2.6% | 4/5 |
| SZ_TAXI/H/short | 0.5893 | 0.5746 | +2.5% | 5/5 |
| ett2/H/long | 1.1511 | 1.1238 | +2.4% | 3/5 |
| SZ_TAXI/15T/short | 0.5689 | 0.5571 | +2.1% | 5/5 |
| bizitobs_application/long | 7.1303 | 6.9962 | +1.9% | 2/5 |
| M_DENSE/D/short | 0.7915 | 0.7786 | +1.6% | 3/5 |
| ett2/H/medium | 1.1084 | 1.0911 | +1.6% | 3/5 |
| saugeenday/D/short | 3.0328 | 2.9860 | +1.5% | 4/5 |
| electricity/D/short | 1.5412 | 1.5176 | +1.5% | 5/5 |
| bitbrains_fast_storage/5T/medium | 1.1046 | 1.0882 | +1.5% | 4/5 |
| car_parts_with_missing/short | 0.8776 | 0.8647 | +1.5% | 4/5 |
| kdd_cup_2018_with_missing/H/medium | 1.0378 | 1.0234 | +1.4% | 2/5 |
| ett2/H/short | 0.7877 | 0.7770 | +1.4% | 4/5 |
| m4_daily/short | 3.4606 | 3.4141 | +1.3% | 4/5 |
| hierarchical_sales/W/short | 0.7623 | 0.7521 | +1.3% | 5/5 |
| bitbrains_rnd/5T/long | 3.5212 | 3.4743 | +1.3% | 4/5 |
| ett1/H/short | 0.8699 | 0.8584 | +1.3% | 3/5 |
| jena_weather/D/short | 1.2911 | 1.2766 | +1.1% | 3/5 |
| saugeenday/W/short | 1.4382 | 1.4236 | +1.0% | 3/5 |
| covid_deaths/short | 47.1146 | 46.6648 | +1.0% | 3/5 |
| bitbrains_fast_storage/H/short | 1.2138 | 1.2023 | +0.9% | 5/5 |
| kdd_cup_2018_with_missing/H/long | 1.0203 | 1.0110 | +0.9% | 3/5 |
| M_DENSE/H/short | 0.8169 | 0.8102 | +0.8% | 4/5 |
| bitbrains_rnd/5T/medium | 4.5576 | 4.5239 | +0.7% | 4/5 |
| electricity/H/short | 1.0099 | 1.0025 | +0.7% | 4/5 |
| restaurant/short | 0.6963 | 0.6917 | +0.7% | 4/5 |
| bitbrains_rnd/5T/short | 1.7972 | 1.7862 | +0.6% | 4/5 |
| kdd_cup_2018_with_missing/H/short | 0.9416 | 0.9361 | +0.6% | 3/5 |
| m4_monthly/short | 0.9987 | 0.9952 | +0.4% | 2/5 |
| ett1/D/short | 1.6748 | 1.6693 | +0.3% | 3/5 |
| m4_quarterly/short | 1.3537 | 1.3500 | +0.3% | 3/5 |
| solar/H/medium | 0.9618 | 0.9595 | +0.2% | 3/5 |
| bitbrains_fast_storage/5T/short | 0.7904 | 0.7887 | +0.2% | 3/5 |
| bitbrains_rnd/H/short | 5.9342 | 5.9282 | +0.1% | 3/5 |
| hierarchical_sales/D/short | 0.7467 | 0.7474 | -0.1% | 1/5 |
| LOOP_SEATTLE/D/short | 0.9053 | 0.9072 | -0.2% | 2/5 |
| temperature_rain_with_missing/short | 1.3572 | 1.3610 | -0.3% | 3/5 |
| jena_weather/H/short | 0.5488 | 0.5506 | -0.3% | 3/5 |
| hospital/short | 0.7650 | 0.7679 | -0.4% | 2/5 |
| kdd_cup_2018_with_missing/D/short | 1.2103 | 1.2172 | -0.6% | 1/5 |
| saugeenday/M/short | 0.7418 | 0.7460 | -0.6% | 2/5 |
| LOOP_SEATTLE/5T/short | 0.5542 | 0.5575 | -0.6% | 2/5 |
| jena_weather/10T/long | 0.7068 | 0.7114 | -0.7% | 2/5 |
| bizitobs_service/long | 3.7477 | 3.7747 | -0.7% | 3/5 |
| solar/H/short | 0.9288 | 0.9365 | -0.8% | 2/5 |
| ett1/H/medium | 1.3286 | 1.3408 | -0.9% | 2/5 |
| bizitobs_l2c/5T/short | 0.2818 | 0.2845 | -1.0% | 2/5 |
| jena_weather/H/medium | 0.7896 | 0.7988 | -1.2% | 1/5 |
| ett1/H/long | 1.4629 | 1.4805 | -1.2% | 2/5 |
| ett2/D/short | 1.3844 | 1.4015 | -1.2% | 3/5 |
| jena_weather/10T/medium | 0.6371 | 0.6468 | -1.5% | 2/5 |
| jena_weather/10T/short | 0.3042 | 0.3094 | -1.7% | 1/5 |
| electricity/15T/short | 1.2286 | 1.2646 | -2.9% | 0/5 |
| solar/D/short | 1.0236 | 1.0555 | -3.1% | 2/5 |
| bizitobs_l2c/5T/long | 1.1314 | 1.1793 | -4.2% | 1/5 |
| bizitobs_l2c/5T/medium | 0.7615 | 0.8106 | -6.4% | 0/5 |
| solar/W/short | 1.5126 | 1.6111 | -6.5% | 1/5 |
| solar/10T/short | 1.2773 | 1.3637 | -6.8% | 3/5 |
| ett2/W/short | 1.2567 | 1.4067 | -11.9% | 2/5 |

---

## 4. Head-to-Head Between Hint Methods (5 seeds pooled)

| Comparison | Wins | % | p-value | Sig |
|------------|------|---|---------|-----|
| HD10 beats MS46 | 251/485 | 51.8% | 0.468 | ns |
| HD10 beats MSHD10 | 257/485 | 53.0% | 0.204 | ns |
| MS46 beats MSHD10 | 255/485 | 52.6% | 0.276 | ns |

All three methods are statistically indistinguishable from each other.

---

## 5. Confound Analysis

### Controlled confounds

| Confound | Control | Result |
|----------|---------|--------|
| Extra capacity (12K params) | Duplicate channel ablation (concat copy of input) | Duplicate **hurts** (41.2% pooled wins = worse than BL) |
| Data order / batch composition | 5 random seeds tested (0, 1, 2, 7, 42) | All pooled results significant at p < 1e-10 |
| Training compute | Measured wall-clock overhead on 9 same-node runs | ~1.6% overhead, negligible |
| Anomaly filtering | Identical settings (variance_ratio=0.0, zscore=8.0) | Controlled |

### Partially confounded

| Confound | Evidence | Status |
|----------|----------|--------|
| Implicit LR effect | BL@LR=2e-3 = 1.1844 (-2.79%), explains ~54% of HD10 benefit | HD10@LR=2e-3 experiment running |

### Known limitations

- **Seed 1 HD10 is not individually significant** (54/97 wins, p=0.15), but BL seed 1 is the strongest baseline (1.1955 vs mean 1.2116)
- **Effect is heterogeneous**: strong for high-frequency data (transport +7%, energy +4%), weak for weather/healthcare (~0%)
- **Base model (46M params) shows no consistent benefit** — hints help capacity-constrained models most
- **MSHD10 has eval stochasticity** from hint dropout active at inference time (2-3% variance)
