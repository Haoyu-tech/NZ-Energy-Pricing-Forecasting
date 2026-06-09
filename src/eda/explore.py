"""Exploratory data analysis: figures + short written takeaways.

Outputs (outputs/figures/):
  1) price time-series overview
  2) intraday / weekly seasonality curves (using the half-hourly data)
  3) price distribution (with a log-transform comparison)
  4) correlation heatmap of price vs storage/demand
All conclusions are written to the returned dict for the pipeline to roll up
into the report.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend so figures render on a server / in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import abspath, ensure_dir

plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.unicode_minus"] = False


def run(hh: pd.DataFrame, model_df: pd.DataFrame, cfg: dict) -> dict:
    fig_dir = abspath(cfg["output"]["figures"])
    ensure_dir(fig_dir)
    node = cfg["data"]["node"]
    notes: dict[str, str] = {}

    # 1) Time-series overview (daily mean price + daily peak price)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(model_df.index, model_df["price"], lw=0.8, label="Daily mean")
    if "price_max" in model_df:
        ax.plot(model_df.index, model_df["price_max"], lw=0.5, alpha=0.5, label="Daily max")
    ax.set_title(f"{node} spot price time series")
    ax.set_ylabel("NZD/MWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "01_price_timeseries.png")
    plt.close(fig)

    # 2) Seasonality: intraday / weekly
    hh_local = hh.copy()
    hh_local["hour"] = hh_local.index.hour + hh_local.index.minute / 60
    hh_local["dow"] = hh_local.index.dayofweek
    by_hour = hh_local.groupby("hour")["price"].mean()
    by_dow = hh_local.groupby("dow")["price"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(by_hour.index, by_hour.values, marker="o", ms=3)
    axes[0].set_title("Intraday mean price (half-hourly)")
    axes[0].set_xlabel("Hour of day"); axes[0].set_ylabel("NZD/MWh")
    axes[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(by_dow)], by_dow.values)
    axes[1].set_title("Mean price by day of week")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_seasonality.png")
    plt.close(fig)

    peak_hour = float(by_hour.idxmax())
    notes["Seasonality"] = (
        f"Intraday price peaks around {peak_hour:.1f}h; "
        f"weekday mean {by_dow.iloc[:5].mean():.1f} vs weekend {by_dow.iloc[5:].mean():.1f} NZD/MWh."
    )

    # 3) Price distribution (raw vs log)
    pos = model_df["price"].clip(lower=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(model_df["price"], bins=60)
    axes[0].set_title("Price distribution (raw, heavy-tailed)")
    axes[1].hist(np.log(pos), bins=60)
    axes[1].set_title("Price distribution (log, more symmetric)")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_distribution.png")
    plt.close(fig)

    skew = float(model_df["price"].skew())
    notes["Distribution"] = (
        f"Price is right-skewed (skew {skew:.2f}) with heavy-tailed spikes; "
        f"log transform is more symmetric — fine to leave raw for tree models, "
        f"consider log target for linear/SARIMAX."
    )

    # 4) Correlation
    num = model_df.select_dtypes("number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlation of numeric variables")
    fig.tight_layout()
    fig.savefig(fig_dir / "04_correlation.png")
    plt.close(fig)

    parts = []
    if "storage" in corr.columns:
        parts.append(f"price vs hydro storage {corr.loc['price', 'storage']:.2f} "
                     f"(dry years push prices up -> negative, as expected)")
    if "demand" in corr.columns:
        parts.append(f"price vs demand {corr.loc['price', 'demand']:.2f}")
    notes["Correlation"] = (
        "; ".join(parts) + "." if parts else
        "Price-only dataset (real EMI final energy prices); no exogenous variables — "
        "add demand / hydro-storage datasets from EMI to enable cross-correlation analysis."
    )

    print("[eda] Wrote 4 figures to", fig_dir)
    for k, v in notes.items():
        print(f"  - {k}: {v}")
    return notes


if __name__ == "__main__":
    from src.utils.io import load_config
    from src.data.download import get_raw
    from src.data.preprocess import clean_halfhourly, aggregate
    cfg = load_config()
    raw = get_raw(cfg)
    hh = clean_halfhourly(raw, cfg)
    run(hh, aggregate(hh, cfg), cfg)
