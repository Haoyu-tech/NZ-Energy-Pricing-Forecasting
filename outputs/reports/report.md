# NZEM Spot Price Forecasting — Results Report

- Generated: 2026-05-29 13:57
- Node: `BEN2201`  Data: 2023-01-01 ~ 2023-12-31 (EMI real)
- Modelling frequency: D  Hold-out test: 90 points

## 1. Data insights (EDA)
- **Seasonality**: Intraday price peaks around 17.5h; weekday mean 115.0 vs weekend 100.6 NZD/MWh.
- **Distribution**: Price is right-skewed (skew 0.25) with heavy-tailed spikes; log transform is more symmetric — fine to leave raw for tree models, consider log target for linear/SARIMAX.
- **Correlation**: Price-only dataset (real EMI final energy prices); no exogenous variables — add demand / hydro-storage datasets from EMI to enable cross-correlation analysis.

## 2. Model performance (hold-out backtest, one-step-ahead)

| model         |    MAE |   RMSE |   sMAPE% |
|:--------------|-------:|-------:|---------:|
| SARIMAX       | 16.585 | 23.828 |    12.28 |
| LightGBM      | 20.177 | 29.211 |    14.63 |
| SeasonalNaive | 31.849 | 46.397 |    23.87 |

**Best model: `SARIMAX`**, MAE=16.59 NZD/MWh, a **47.9%** improvement over the seasonal-naive baseline.

## 3. Robustness (walk-forward expanding CV)

| model         |   MAE_mean |   MAE_std |   folds |
|:--------------|-----------:|----------:|--------:|
| SARIMAX       |     15.939 |     0.647 |       2 |
| SeasonalNaive |     28.072 |     3.778 |       2 |
| LightGBM      |     28.975 |     8.798 |       2 |

## 4. LightGBM top features

|               |   importance |
|:--------------|-------------:|
| price_rstd7   |          484 |
| price_lag1    |          437 |
| price_lag14   |          360 |
| price_rstd14  |          332 |
| price_lag7    |          310 |
| price_rstd30  |          243 |
| price_rmean14 |          224 |
| price_rmean7  |          222 |
| doy_sin       |          221 |
| price_rmean30 |          220 |

## 5. Figures

See `outputs/figures/`: 01 time series, 02 seasonality, 03 distribution, 04 correlation, 05 forecast comparison.

## 6. Limitations & next steps
- Price spikes (dry + high-load) remain the main error source; add scarcity indicators and quantile regression for prediction intervals.
- Daily granularity here for a fast demo; set `target_freq` to `30min` for half-hourly (LightGBM is the better fit there).
- After plugging in real EMI data, re-check field mapping and time zone handling.