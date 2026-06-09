"""SARIMAX: classic time-series model with seasonality and exogenous variables.

endog = price; exog = demand, storage (market fundamentals).
Evaluation uses one-step-ahead rolling: fit on the training segment, then
forecast(1) point by point, appending the true value each step, to obtain a
one-step-ahead prediction comparable to the other models. Uses numpy arrays to
avoid timezone/frequency warnings.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.models.base import TARGET, BaseForecaster

EXOG = ["demand", "storage"]


class SarimaxForecaster(BaseForecaster):
    name = "SARIMAX"

    def __init__(self, order=(1, 1, 1), seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order  # if None, derived from cfg's seasonal_period

    def _exog(self, df: pd.DataFrame):
        cols = [c for c in EXOG if c in df.columns]
        return (df[cols].to_numpy(), cols) if cols else (None, [])

    def fit(self, train, feature_cols, cfg):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        s = cfg["modeling"].get("seasonal_period", 7)
        seasonal = self.seasonal_order or (1, 0, 1, s)
        y = train[TARGET].to_numpy(dtype=float)
        X, self._exog_cols = self._exog(train)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = SARIMAX(
                y, exog=X, order=self.order, seasonal_order=seasonal,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
        return self

    def forecast(self, test, feature_cols, cfg):
        y = test[TARGET].to_numpy(dtype=float)
        X = test[self._exog_cols].to_numpy() if self._exog_cols else None

        preds = np.empty(len(y))
        res = self.res
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(y)):
                xf = X[i:i + 1] if X is not None else None
                preds[i] = float(res.forecast(steps=1, exog=xf)[0])
                # Append the real observation to the state space and advance one step (no parameter re-estimation)
                res = res.append(y[i:i + 1], exog=xf, refit=False)
        return pd.Series(preds, index=test.index, name=self.name)
