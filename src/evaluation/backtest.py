"""Evaluation layer: metrics + hold-out backtest + walk-forward expanding CV.

Convention: every model forecasts one step ahead and never uses future information.
  - Hold-out backtest: use the last test_days points for out-of-sample testing,
    producing each model's prediction series and metrics.
  - Walk-forward: split the data into cv_splits expanding folds, retraining and
    evaluating fold by fold, reporting mean +/- std of the metric. This reflects
    robustness better than a single hold-out (the right replacement for random
    K-fold in time series).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.base import TARGET
from src.utils.io import abspath, ensure_dir

plt.rcParams["axes.unicode_minus"] = False


# ---------- Metrics ----------
def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    # sMAPE: more robust to zero/negative prices (electricity prices can go negative; plain MAPE blows up)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(np.abs(err) / np.where(denom == 0, np.nan, denom)) * 100
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "sMAPE%": round(float(np.nanmean(smape)), 2)}


# ---------- Hold-out backtest ----------
def holdout_split(df: pd.DataFrame, test_days: int):
    cut = len(df) - test_days
    return df.iloc[:cut], df.iloc[cut:]


def run_holdout(df, feature_cols, model_factories: dict, cfg):
    """Run one hold-out backtest per model; return (metrics table, predictions DataFrame)."""
    train, test = holdout_split(df, cfg["modeling"]["test_days"])
    print(f"[backtest] Hold-out backtest: train {len(train)} / test {len(test)}")
    rows, preds = [], {"actual": test[TARGET]}
    for name, factory in model_factories.items():
        model = factory().fit(train, feature_cols, cfg)
        yhat = model.forecast(test, feature_cols, cfg)
        preds[name] = yhat
        m = metrics(test[TARGET], yhat)
        m["model"] = name
        rows.append(m)
        print(f"  - {name:14s} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  sMAPE={m['sMAPE%']:.2f}%")
    score = pd.DataFrame(rows).set_index("model").sort_values("MAE")
    return score, pd.DataFrame(preds)


# ---------- Walk-forward expanding CV ----------
def run_walk_forward(df, feature_cols, model_factories: dict, cfg):
    n_splits = cfg["modeling"]["cv_splits"]
    test_size = cfg["modeling"]["test_days"]
    n = len(df)
    # Fold test segments march backward from the tail; the training segment is everything before it (expanding window)
    results = {name: [] for name in model_factories}
    for k in range(n_splits):
        test_end = n - k * test_size
        test_start = test_end - test_size
        if test_start <= test_size:  # stop once the training segment gets too short
            break
        train = df.iloc[:test_start]
        test = df.iloc[test_start:test_end]
        for name, factory in model_factories.items():
            model = factory().fit(train, feature_cols, cfg)
            yhat = model.forecast(test, feature_cols, cfg)
            results[name].append(metrics(test[TARGET], yhat)["MAE"])

    rows = []
    for name, maes in results.items():
        if maes:
            rows.append({"model": name, "MAE_mean": round(np.mean(maes), 3),
                         "MAE_std": round(np.std(maes), 3), "folds": len(maes)})
    cv = pd.DataFrame(rows).set_index("model").sort_values("MAE_mean")
    print(f"[backtest] walk-forward done ({cv['folds'].max() if len(cv) else 0} folds)")
    return cv


# ---------- Plotting ----------
def plot_predictions(pred_df: pd.DataFrame, cfg: dict, last_n: int = 90):
    fig_dir = abspath(cfg["output"]["figures"])
    ensure_dir(fig_dir)
    sub = pred_df.iloc[-last_n:]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sub.index, sub["actual"], color="black", lw=1.6, label="Actual")
    for col in sub.columns:
        if col != "actual":
            ax.plot(sub.index, sub[col], lw=1.0, alpha=0.85, label=col)
    ax.set_title("Out-of-sample forecast (last %d test points)" % last_n)
    ax.set_ylabel("NZD/MWh"); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    out = fig_dir / "05_predictions.png"
    fig.savefig(out)
    plt.close(fig)
    print("[backtest] Forecast comparison figure:", out)
