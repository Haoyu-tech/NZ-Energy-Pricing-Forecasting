"""Unified model interface.

All models are evaluated one-step-ahead: to predict the price at time t, only
real information from before t may be used. This keeps the three models directly
comparable and mirrors the real day-ahead forecasting setting.
"""
from __future__ import annotations

import pandas as pd

TARGET = "price"


class BaseForecaster:
    name = "base"

    def fit(self, train: pd.DataFrame, feature_cols: list[str], cfg: dict) -> "BaseForecaster":
        raise NotImplementedError

    def forecast(self, test: pd.DataFrame, feature_cols: list[str], cfg: dict) -> pd.Series:
        """Return a one-step-ahead prediction series sharing the test index."""
        raise NotImplementedError
