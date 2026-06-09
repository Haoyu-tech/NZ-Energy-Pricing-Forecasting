"""Feature engineering.

Builds tabular features for the tree / linear models:
  - calendar features (day of week, month, position in year, encoded cyclically with sin/cos)
  - price lags
  - rolling price statistics (mean / std, capturing recent level and volatility)
  - exogenous variables (demand, storage) and their lags / changes
All lags use only past information to avoid leakage; warm-up rows containing NaN
are dropped after construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "price"


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    df["dow"] = idx.dayofweek
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["month"] = idx.month
    doy = idx.dayofyear
    # Cyclical encoding: position in the year + day of week
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    if "hour" not in df and hasattr(idx, "hour") and idx.to_series().dt.hour.nunique() > 1:
        h = idx.hour + idx.minute / 60
        df["hour_sin"] = np.sin(2 * np.pi * h / 24)
        df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    return df


def add_lags_rolling(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    for lag in cfg["features"]["lags"]:
        df[f"{TARGET}_lag{lag}"] = df[TARGET].shift(lag)
    # Rolling stats are computed on the shift(1) window to ensure they exclude the current period
    base = df[TARGET].shift(1)
    for w in cfg["features"]["rolling_windows"]:
        df[f"{TARGET}_rmean{w}"] = base.rolling(w).mean()
        df[f"{TARGET}_rstd{w}"] = base.rolling(w).std()
    # Lags and changes of exogenous variables (a falling storage trend is a spike precursor)
    for col in ("demand", "storage"):
        if col in df:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_chg1"] = df[col].shift(1) - df[col].shift(2)
    return df


def build(model_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Return (feature dataset with warm-up rows dropped, feature column names)."""
    df = add_calendar(model_df)
    df = add_lags_rolling(df, cfg)

    # In a day-ahead setting, forecasts for the current period's demand/storage are
    # usually available, so we allow current demand/storage as known exogenous inputs;
    # all other derived features come from the past only.
    drop_cols = {TARGET, "price_max", "demand_peak", "spike_count", "price_robust_z", "is_spike"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    before = len(df)
    df = df.dropna(subset=feature_cols + [TARGET])
    print(f"[features] Built {len(feature_cols)} features; dropped {before - len(df)} warm-up rows, {len(df)} remain")
    return df, feature_cols


if __name__ == "__main__":
    from src.utils.io import load_config
    from src.data.download import get_raw
    from src.data.preprocess import run as prep
    cfg = load_config()
    feat, cols = build(prep(get_raw(cfg), cfg), cfg)
    print(cols)
