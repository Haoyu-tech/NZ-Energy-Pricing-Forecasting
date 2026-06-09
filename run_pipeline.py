"""NZEM electricity spot price forecasting — end-to-end pipeline.

Chains together: data acquisition -> cleaning -> EDA -> feature engineering ->
multi-model training -> hold-out backtest + walk-forward CV -> figures +
Markdown report.

Usage:
    python run_pipeline.py                 # use default config.yaml
    python run_pipeline.py --config x.yaml # use a specific config
    python run_pipeline.py --no-cv         # skip walk-forward CV (faster)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import pandas as pd

# The Windows console defaults to GBK; switch to UTF-8 so symbols print cleanly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.data.download import get_raw
from src.data.preprocess import clean_halfhourly, aggregate
from src.eda import explore
from src.evaluation import backtest as bt
from src.features.build_features import build
from src.models.baseline import SeasonalNaive
from src.models.lgbm_model import LGBMForecaster
from src.models.sarimax_model import SarimaxForecaster
from src.utils.io import abspath, ensure_dir, load_config

MODEL_FACTORIES = {
    "SeasonalNaive": SeasonalNaive,
    "SARIMAX": SarimaxForecaster,
    "LightGBM": LGBMForecaster,
}


def write_report(cfg, eda_notes, score, cv, top_features) -> str:
    rep_dir = abspath(cfg["output"]["reports"])
    ensure_dir(rep_dir)
    path = rep_dir / "report.md"
    best = score.index[0]
    naive_mae = score.loc["SeasonalNaive", "MAE"] if "SeasonalNaive" in score.index else None
    best_mae = score.loc[best, "MAE"]
    lift = (1 - best_mae / naive_mae) * 100 if naive_mae else float("nan")

    lines = [
        f"# NZEM Spot Price Forecasting — Results Report",
        f"",
        f"- Generated: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Node: `{cfg['data']['node']}`  Data: {cfg['data']['start']} ~ {cfg['data']['end']}"
        f" ({'synthetic' if cfg['data']['use_synthetic'] else 'EMI real'})",
        f"- Modelling frequency: {cfg['modeling']['target_freq']}  "
        f"Hold-out test: {cfg['modeling']['test_days']} points",
        f"",
        f"## 1. Data insights (EDA)",
    ]
    for k, v in eda_notes.items():
        lines.append(f"- **{k}**: {v}")

    lines += [
        f"",
        f"## 2. Model performance (hold-out backtest, one-step-ahead)",
        f"",
        score.to_markdown(),
        f"",
        f"**Best model: `{best}`**, MAE={best_mae:.2f} NZD/MWh"
        + (f", a **{lift:.1f}%** improvement over the seasonal-naive baseline." if naive_mae else "."),
        f"",
        f"## 3. Robustness (walk-forward expanding CV)",
        f"",
        cv.to_markdown() if cv is not None and len(cv) else "_(skipped)_",
        f"",
        f"## 4. LightGBM top features",
        f"",
        top_features.head(10).to_markdown() if top_features is not None else "_(none)_",
        f"",
        f"## 5. Figures",
        f"",
        f"See `outputs/figures/`: 01 time series, 02 seasonality, 03 distribution, "
        f"04 correlation, 05 forecast comparison.",
        f"",
        f"## 6. Limitations & next steps",
        f"- Price spikes (dry + high-load) remain the main error source; add scarcity indicators "
        f"and quantile regression for prediction intervals.",
        f"- Daily granularity here for a fast demo; set `target_freq` to `30min` for half-hourly "
        f"(LightGBM is the better fit there).",
        f"- After plugging in real EMI data, re-check field mapping and time zone handling.",
    ]
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    print("[report] Report written to:", path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-cv", action="store_true", help="skip walk-forward cross-validation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("=" * 70)
    print(" NZEM electricity spot price forecasting — pipeline start")
    print("=" * 70)

    # 1) Data
    raw = get_raw(cfg)
    hh = clean_halfhourly(raw, cfg)
    model_df = aggregate(hh, cfg)

    # 2) EDA
    eda_notes = explore.run(hh, model_df, cfg)

    # 3) Features
    feat, feature_cols = build(model_df, cfg)

    # 4) Hold-out backtest
    score, pred_df = bt.run_holdout(feat, feature_cols, MODEL_FACTORIES, cfg)
    bt.plot_predictions(pred_df, cfg)

    # 5) Walk-forward CV
    cv = None if args.no_cv else bt.run_walk_forward(feat, feature_cols, MODEL_FACTORIES, cfg)

    # 6) LightGBM feature importance (refit once on the full training segment)
    train, _ = bt.holdout_split(feat, cfg["modeling"]["test_days"])
    lgbm = LGBMForecaster().fit(train, feature_cols, cfg)
    top_features = lgbm.feature_importance().rename("importance").to_frame()

    # 7) Report
    write_report(cfg, eda_notes, score, cv, top_features)
    print("\n[done] Finished. Open outputs/reports/report.md to view the results.")


if __name__ == "__main__":
    main()
