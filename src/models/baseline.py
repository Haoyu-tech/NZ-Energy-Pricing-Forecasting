"""Baseline model: seasonal-naive forecast.

Predicts today's price = the actual price from the previous same-position period
(at daily granularity, lag7 = same day last week). This is one of the most common
and hardest-to-beat baselines in electricity/energy forecasting; every more
complex model should first prove it can beat it.
"""
from __future__ import annotations

import pandas as pd

from src.models.base import TARGET, BaseForecaster


class SeasonalNaive(BaseForecaster):
    name = "SeasonalNaive"

    def fit(self, train, feature_cols, cfg):
        self.s = cfg["modeling"].get("seasonal_period", 7)
        return self

    def forecast(self, test, feature_cols, cfg):
        # Prefer the seasonal lag column (already real historical values); fall back to lag1 if missing
        col = f"{TARGET}_lag{self.s}"
        if col not in test:
            col = f"{TARGET}_lag1"
        return test[col].rename(self.name)
