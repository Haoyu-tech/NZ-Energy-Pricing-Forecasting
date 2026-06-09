# TODO: build domain knowledge of the NZ electricity market

> Goal: build domain knowledge of the NZEM (New Zealand Electricity Market) using
> the Electricity Authority + EMI data platform, and run the models end-to-end on
> real spot price data myself.
> So I can say in an interview: "I tried model XX on NZEM data, and the result was ..."

---

## Stage 1: understand the market structure (know what you're analysing)

- [ ] Read through the Electricity Authority website to understand the regulator's role: https://www.ea.govt.nz
- [ ] Understand the NZEM market structure: who the generators are (Meridian / Contact /
      Genesis / Mercury / Manawa), who the retailers are, and Transpower's role as
      System Operator
- [ ] Understand the spot market mechanism: half-hourly settlement, nodal pricing (LMP),
      ~280 pricing nodes, energy price vs transmission constraints
- [ ] Understand the central impact of hydro dominance + drought risk (hydro storage / inflows) on prices
- [ ] Keep a glossary of key terms: spot price, final price, reserve, FK/GWAP, HVDC, scarcity pricing

## Stage 2: get the data (EMI data platform)

- [ ] Register for / get familiar with the EMI platform: https://www.emi.ea.govt.nz
- [ ] Find and download the spot price dataset (final pricing / wholesale prices); choose a
      time range and node (start with a representative node such as BEN2201 / OTA2201)
- [ ] Also download related explanatory variables: demand, hydro storage, generation by fuel
- [ ] Land the raw CSVs in the `02_data` directory and record a data dictionary
      (field meanings, units, time granularity, time zone NZST/NZDT)
- [ ] Data cleaning: handle missing values, align half-hourly timestamps, DST changeover, outlier spikes

## Stage 3: exploratory analysis (understand the data before modelling)

- [ ] Plot time series: intraday / weekly / seasonal price patterns
- [ ] Analyse the relationship between price spikes and drought / low storage / high demand
- [ ] Price distribution (heavy tails, occasional negative / extreme-high prices); assess whether a log transform is needed
- [ ] Correlation analysis: price vs demand / storage / wind generation

## Stage 4: run the models (the core interview talking point)

- [ ] Baseline: naive / seasonal-naive forecast, to set the evaluation benchmark (MAE / RMSE / MAPE)
- [ ] Classic time series: SARIMA / SARIMAX (with exogenous variables: demand, storage)
- [ ] Machine learning: gradient boosting (XGBoost / LightGBM) for multi-step forecasting; feature engineering is key
- [ ] (Optional, advanced) Deep learning: LSTM / Temporal Fusion Transformer for multivariate time series
- [ ] Rigorous evaluation: time-series cross-validation (walk-forward), no future-data leakage
- [ ] Write a "conclusion": which model is best, why, how large the error is, and what the limitations are

## Stage 5: turn it into interview material

- [ ] Put together a 1-page project summary: problem -> data -> method -> results -> insights
- [ ] Prepare a story you can tell out loud: data source (EMI), why these features, model comparison conclusion
- [ ] Tidy the code into a repo (README + notebook) for easy show-and-tell in interviews
- [ ] Be ready for follow-up questions: hydro market specifics, why price spikes are hard to predict, model limitations

---

## Key links

- Electricity Authority (regulator): https://www.ea.govt.nz
- EMI data platform (data downloads): https://www.emi.ea.govt.nz
- Transpower (system operator): https://www.transpower.co.nz

## Suggested directory layout

```
electricity_price_forecasting/
├── 01_market_knowledge/   <- this folder (notes + glossary + this TODO)
├── 02_data/               <- raw and cached data downloaded from EMI
├── src/                   <- data / feature / model / evaluation code
├── outputs/               <- figures + reports
├── run_pipeline.py        <- one-click end-to-end
└── config.yaml
```
