"""Reproducible development comparison. Never promotes or replaces a model.

Final calendar month is excluded from all fitting and evaluation. This is a
development study of previously observed history, not a fresh holdout claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
import features as F
import labeling as L
import validation as V


def paired_daily_brier(y, p, baseline, index):
    """Paired day-block bootstrap, not an IID-minute certainty claim."""
    onehot = np.eye(3)[y]
    delta = ((baseline - onehot) ** 2).sum(axis=1) - ((p - onehot) ** 2).sum(axis=1)
    daily = pd.Series(delta, index=index).groupby(index.floor("D")).mean().to_numpy()
    rng = np.random.default_rng(42)
    means = rng.choice(daily, (1000, len(daily)), replace=True).mean(axis=1)
    return {"mean_daily_brier_gain": float(daily.mean()),
            "day_bootstrap_95pct": np.quantile(means, [.025, .975]).tolist(),
            "days": len(daily)}


def run(estimators=150):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sol = F._to_indexed(pd.read_parquet(config.DATA_STORE / "candles_SOL_1m_binance.parquet"))
    btc = F._to_indexed(pd.read_parquet(config.DATA_STORE / "candles_BTC_1m_binance.parquet"))
    feats = F.compute_model_features(sol, btc)
    # Reindex prices to the exact grid used by feature calculations.
    sol = sol.reindex(feats.index)
    report = {"input_contract": "closed-ohlcv-v2", "seed": 42,
              "estimators": estimators, "source": "Binance reference data, not Jupiter Perps execution validation",
              "reserved_from": str(feats.index[-1].normalize().replace(day=1)),
              "raw_rows": len(sol), "valid_feature_rows": int(feats.notna().all(axis=1).sum()),
              "folds": []}
    # Reuse one feature specification across all targets; no feature search.
    for name, horizon in config.HORIZONS_MIN.items():
        label = L.forward_return_labels(sol, horizon, config.ROUND_TRIP_COST)
        valid = feats.notna().all(axis=1) & label.notna()
        X, y = feats.loc[valid], L.labels_to_classes(label.loc[valid])
        splits = V.purged_walk_forward(X.index, horizon)
        for s in splits[:-1]:
            tr = np.flatnonzero((X.index >= s.train_start) & (X.index < s.train_end))
            te = np.flatnonzero((X.index >= s.test_start) & (X.index < s.test_end))
            if len(tr) < 50000 or len(te) < 10000:
                continue
            train_x, train_y = X.iloc[tr], y[tr]
            parts = V.temporal_partitions(train_x.index, horizon)
            fit, es, cal = (parts[k] for k in ("fit", "early_stop", "calibration"))
            priors = np.bincount(train_y[fit], minlength=3).astype(float) + 1
            priors /= priors.sum()
            base_p = np.tile(priors, (len(te), 1))
            params = {k: v for k, v in config.LGBM_PARAMS.items() if k != "num_class"}
            params["n_estimators"] = estimators
            model = LGBMClassifier(**params)
            model.fit(train_x.iloc[fit], train_y[fit],
                      sample_weight=V.recency_weights(train_x.iloc[fit].index),
                      eval_set=[(train_x.iloc[es], train_y[es])],
                      callbacks=[early_stopping(30, verbose=False), log_evaluation(0)])
            calibrated = V.fit_isotonic_multiclass(model, train_x.iloc[cal], train_y[cal])
            linear = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=300))
            linear.fit(train_x.iloc[fit], train_y[fit])
            predictions = {"prior": base_p, "linear": linear.predict_proba(X.iloc[te]),
                           "lgb_raw": model.predict_proba(X.iloc[te]),
                           "lgb_isotonic": calibrated.predict_proba(X.iloc[te])}
            fold = {"horizon": name, "month": str(s.test_start), "n_test": len(te),
                    "train_end": str(s.train_end), "fit_class_counts": np.bincount(train_y[fit], minlength=3).tolist(),
                    "test_class_counts": np.bincount(y[te], minlength=3).tolist(),
                    "best_iteration": model.best_iteration_, "models": {}}
            for kind, p in predictions.items():
                fold["models"][kind] = {**V.probability_metrics(y[te], p),
                                        **paired_daily_brier(y[te], p, base_p, X.index[te])}
            report["folds"].append(fold)
            print(name, s.test_start.strftime("%Y-%m"),
                  {k: round(v["log_loss"], 5) for k, v in fold["models"].items()}, flush=True)
    out = config.LOGS_DIR / "prediction_audit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimators", type=int, default=150)
    args = parser.parse_args()
    run(args.estimators)