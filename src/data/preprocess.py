"""Data cleaning layer.

Input: raw half-hourly data (from download.get_raw).
Output:
  - a clean half-hourly series (missing/duplicate/outlier handling, full time grid)
  - a modelling dataset aggregated to target_freq (daily by default)

Real EMI data gets its schema mapped here; synthetic data already uses the
canonical column names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import abspath, ensure_dir

EXPECTED = ["price", "demand", "storage"]


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map column names from various sources onto price/demand/storage and ensure a datetime index.

    Synthetic data already has the canonical structure; add field-mapping rules
    for real EMI data here.
    """
    df = df.copy()
    # Datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = next((c for c in df.columns if c.lower() in
                       ("timestamp", "trading_datetime", "trading_date", "datetime")), None)
        if ts_col is None:
            raise ValueError("No time column found; cannot build a datetime index.")
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
    df.index.name = "timestamp"
    # Time zone: normalise to NZ local time
    if df.index.tz is None:
        df.index = df.index.tz_localize("Pacific/Auckland", nonexistent="shift_forward",
                                        ambiguous="NaT")
    else:
        df.index = df.index.tz_convert("Pacific/Auckland")
    return df


def clean_halfhourly(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Clean the half-hourly series."""
    df = _coerce_schema(raw)

    # Keep only the numeric columns we need
    cols = [c for c in EXPECTED if c in df.columns]
    df = df[cols].apply(pd.to_numeric, errors="coerce")

    # Drop duplicate timestamps (keep first) and rebuild the full time grid (DST changes create gaps)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(),
                             freq=cfg["data"]["freq_raw"], tz="Pacific/Auckland")
    df = df.reindex(full_idx)
    df.index.name = "timestamp"

    # Missing values: time-linear interpolation first, then forward/back fill at the endpoints
    df = df.interpolate(method="time", limit=6).ffill().bfill()

    # Outlier flagging: robust z-score on price (median / MAD). Flag, never delete —
    # price spikes are genuine market signal, not noise to be discarded.
    med = df["price"].median()
    mad = (df["price"] - med).abs().median() + 1e-9
    df["price_robust_z"] = (df["price"] - med) / (1.4826 * mad)
    df["is_spike"] = (df["price_robust_z"] > 6).astype(int)

    n_spike = int(df["is_spike"].sum())
    print(f"[preprocess] Half-hourly cleaning done: {df.shape}, flagged {n_spike} spikes "
          f"({n_spike/len(df)*100:.2f}%)")
    return df


def aggregate(df_hh: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aggregate to the modelling granularity (target_freq). At daily granularity: mean price plus the daily peak."""
    freq = cfg["modeling"]["target_freq"]
    has = lambda c: c in df_hh.columns
    if freq in ("30min", "30T"):
        keep = [c for c in ("price", "demand", "storage") if has(c)]
        out = df_hh[keep].copy()
    else:
        g = df_hh.resample(freq)
        cols = {
            "price": g["price"].mean(),
            "price_max": g["price"].max(),    # daily peak price (spike intensity)
            "spike_count": g["is_spike"].sum(),
        }
        # demand / storage are exogenous variables only present in other EMI datasets;
        # skipped automatically when the price-only file has none
        if has("demand"):
            cols["demand"] = g["demand"].mean()
            cols["demand_peak"] = g["demand"].max()
        if has("storage"):
            cols["storage"] = g["storage"].mean()
        out = pd.DataFrame(cols)
    out = out.dropna(subset=["price"])
    out.index.name = "timestamp"
    return out


def run(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Clean + aggregate, write to the processed directory, and return the modelling dataset."""
    hh = clean_halfhourly(raw, cfg)
    model_df = aggregate(hh, cfg)

    proc_dir = abspath(cfg["data"]["processed_path"])
    ensure_dir(proc_dir)
    hh.to_csv(proc_dir / f"clean_halfhourly_{cfg['data']['node']}.csv")
    model_df.to_csv(proc_dir / f"model_{cfg['modeling']['target_freq']}_{cfg['data']['node']}.csv")
    print(f"[preprocess] Modelling dataset ({cfg['modeling']['target_freq']}): {model_df.shape}")
    return model_df


if __name__ == "__main__":
    from src.utils.io import load_config
    from src.data.download import get_raw
    cfg = load_config()
    run(get_raw(cfg), cfg)
