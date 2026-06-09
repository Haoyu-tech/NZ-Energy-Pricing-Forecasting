"""LightGBM gradient-boosting regressor.

Uses the engineered tabular features (lags / rolling / calendar / exogenous) for
one-step-ahead prediction. Tree models are naturally suited to heavy tails,
non-linearity, and variable interactions, making them a strong baseline-beater
in electricity price forecasting.
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from src.models.base import TARGET, BaseForecaster


class LGBMForecaster(BaseForecaster):
    name = "LightGBM"

    def __init__(self, **params):
        self.params = dict(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        self.params.update(params)
        self.model: lgb.LGBMRegressor | None = None

    def fit(self, train, feature_cols, cfg):
        self.feature_cols = feature_cols
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(train[feature_cols], train[TARGET])
        return self

    def forecast(self, test, feature_cols, cfg):
        pred = self.model.predict(test[feature_cols])
        return pd.Series(pred, index=test.index, name=self.name)

    def feature_importance(self) -> pd.Series:
        return (pd.Series(self.model.feature_importances_, index=self.feature_cols)
                .sort_values(ascending=False))
