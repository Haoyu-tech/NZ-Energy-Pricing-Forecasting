"""Data acquisition layer.

Two sources:
1) Real EMI data: download spot price (and other) CSVs from the Electricity
   Authority's EMI platform. EMI publishes its datasets as static files, so they
   can be fetched month by month via a predictable URL. Note: EMI occasionally
   restructures its directory layout; on any download failure we fall back to
   synthetic data automatically.
2) Synthetic data: a built-in half-hourly generator that mimics NZEM
   characteristics so the whole pipeline can run end-to-end without internet —
   handy for validating the pipeline before swapping in the real data.

See the "Real EMI data (wired in)" section of the README for details.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.utils.io import abspath, ensure_dir

# EMI "Final energy prices" monthly file (half-hourly, all nodes, public direct link).
# Real-world file schema from the EMI platform:
# columns = TradingDate / TradingPeriod / PointOfConnection / DollarsPerMegawattHour
EMI_MONTHLY = ("https://www.emi.ea.govt.nz/Wholesale/Datasets/DispatchAndPricing/"
               "FinalEnergyPrices/ByMonth/{ym}_FinalEnergyPrices.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (research; NZEM price forecasting)"}


def _download_month(ym: str, cache_dir: Path) -> pd.DataFrame:
    """Download (or read from cache) a single month's all-node price file.

    Each monthly file is ~10 MB; cache locally to avoid re-downloading.
    """
    cache = cache_dir / f"{ym}_FinalEnergyPrices.csv"
    if cache.exists():
        return pd.read_csv(cache)
    url = EMI_MONTHLY.format(ym=ym)
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    ensure_dir(cache)
    cache.write_bytes(r.content)
    print(f"[download]   downloaded {ym} ({len(r.content)/1e6:.1f} MB)")
    return pd.read_csv(io.BytesIO(r.content))


def _periods_to_timestamp(date: pd.Series, period: pd.Series) -> pd.Series:
    """TradingPeriod (1..48/49/50) -> local start time of day. Period p maps to (p-1)*30 minutes.

    DST changeover days have 46/50 periods; we use a linear mapping and let the
    preprocess step's "rebuild full grid + interpolate" absorb the offset on the
    time grid.
    """
    base = pd.to_datetime(date)
    return base + pd.to_timedelta((period.astype(int) - 1) * 30, unit="m")


def _try_download_emi(cfg: dict) -> pd.DataFrame | None:
    """Download real EMI final energy prices, filter to the chosen node, and return a half-hourly price series.

    On failure (network / node not found / schema change) returns None and the
    caller falls back to synthetic data. Note: this dataset contains price only,
    not demand/storage — those are separate EMI datasets, and the downstream
    pipeline is robust to missing exogenous variables (see README).
    """
    node = cfg["data"]["node"]
    start = pd.Timestamp(cfg["data"]["start"])
    end = pd.Timestamp(cfg["data"]["end"])
    cache_dir = abspath(cfg["data"]["raw_path"]) / "emi_cache"
    try:
        frames = []
        months = pd.date_range(start, end, freq="MS")
        print(f"[download] Fetching real prices from EMI: node={node}, {len(months)} monthly files ...")
        for month in months:
            df = _download_month(month.strftime("%Y%m"), cache_dir)
            sub = df[df["PointOfConnection"] == node]
            if not sub.empty:
                frames.append(sub)
        if not frames:
            avail = sorted(df["PointOfConnection"].unique())[:10]
            raise ValueError(f"Node {node} has no records in the EMI data; examples: {avail}")

        raw = pd.concat(frames, ignore_index=True)
        ts = _periods_to_timestamp(raw["TradingDate"], raw["TradingPeriod"])
        out = (pd.DataFrame({"timestamp": ts, "price": raw["DollarsPerMegawattHour"].astype(float)})
               .dropna(subset=["timestamp"])
               .drop_duplicates(subset=["timestamp"])
               .set_index("timestamp")
               .sort_index())
        out["node"] = node
        # Clip to the configured range (inclusive of the end day)
        out = out.loc[start:end + pd.Timedelta(days=1)]
        print(f"[download] EMI real prices ready: {out.shape}, {out.index.min()} ~ {out.index.max()}")
        return out
    except Exception as e:  # any network/path/schema problem -> fall back
        print(f"[download] EMI download failed, falling back to synthetic data: {e}")
        return None


def make_synthetic(cfg: dict) -> pd.DataFrame:
    """Generate half-hourly synthetic data that mimics NZEM characteristics.

    Deliberately reproduces a few real market features so the analysis/modelling
    later has a clear business narrative:
      - Demand: intraday double peak (morning & evening) + weekend dip + higher
        in winter (NZ electricity use peaks Jun-Aug)
      - Storage: a slowly mean-reverting random walk, lower in winter (dry season)
      - Price: rises with demand, falls with storage; spikes when storage is low
        and load is high; heavy-tailed with occasional extreme highs
    """
    rng = np.random.default_rng(cfg.get("seed", 42))
    idx = pd.date_range(cfg["data"]["start"], cfg["data"]["end"],
                        freq=cfg["data"]["freq_raw"], tz="Pacific/Auckland")
    n = len(idx)

    hour = idx.hour + idx.minute / 60.0
    doy = idx.dayofyear.to_numpy()
    dow = idx.dayofweek.to_numpy()  # 0 = Monday

    # --- Demand (MW) ---
    # Intraday double peak: 8am and 6pm
    daily = (np.exp(-((hour - 8) ** 2) / 4) + 1.2 * np.exp(-((hour - 18) ** 2) / 5))
    daily = 0.5 + 0.9 * daily / daily.max()
    weekly = np.where(dow >= 5, 0.85, 1.0)          # weekends 15% lower
    # NZ is in the southern hemisphere: winter (mid-year) demand is high -> doy peaks near ~180
    seasonal_d = 1.0 + 0.18 * np.cos((doy - 180) / 365 * 2 * np.pi + np.pi)
    demand = 4500 * daily * weekly * seasonal_d
    demand += rng.normal(0, 80, n)
    demand = np.clip(demand, 1500, None)

    # --- Storage (relative level 0..1): mean-reverting random walk, lower in winter (dry) ---
    storage = np.empty(n)
    storage[0] = 0.6
    seasonal_s = 0.5 + 0.18 * np.cos((doy - 180) / 365 * 2 * np.pi)  # low in winter
    step = rng.normal(0, 0.004, n)
    for t in range(1, n):
        # slowly revert toward the seasonal mean
        storage[t] = storage[t - 1] + 0.02 * (seasonal_s[t] - storage[t - 1]) + step[t]
    storage = np.clip(storage, 0.08, 0.98)

    # --- Price (NZD/MWh) ---
    # Baseline rises with demand and falls with storage (low storage -> pricier marginal units)
    base = 30 + 0.018 * (demand - 3000) + 90 * (0.6 - storage)
    base = np.clip(base, 0, None)
    # Dry + high load -> higher spike probability
    scarcity = np.clip((0.35 - storage), 0, None) * (demand > 5200)
    spike_prob = 0.002 + 4.0 * scarcity
    spikes = rng.random(n) < np.clip(spike_prob, 0, 0.4)
    spike_mag = rng.gamma(2.0, 350, n) * spikes
    noise = rng.normal(0, 8, n)
    price = base + spike_mag + noise
    # Occasional negative prices (a simplified stand-in for high-wind/low-load conditions)
    neg = (rng.random(n) < 0.004) & (demand < 3200)
    price = np.where(neg, rng.uniform(-20, 0, n), price)
    price = np.clip(price, -50, 20000)

    df = pd.DataFrame(
        {"price": price.round(2), "demand": demand.round(1), "storage": storage.round(4)},
        index=idx,
    )
    df.index.name = "timestamp"
    df["node"] = cfg["data"]["node"]
    return df


def get_raw(cfg: dict) -> pd.DataFrame:
    """Return raw half-hourly data (real first, synthetic fallback) and write it to the raw directory."""
    raw = None
    if not cfg["data"].get("use_synthetic", True):
        raw = _try_download_emi(cfg)
    if raw is None:
        raw = make_synthetic(cfg)

    out = abspath(cfg["data"]["raw_path"]) / f"raw_{cfg['data']['node']}.csv"
    ensure_dir(out)
    raw.to_csv(out)
    print(f"[download] Raw data saved: {out}  shape={raw.shape}")
    return raw


if __name__ == "__main__":
    from src.utils.io import load_config
    get_raw(load_config())
