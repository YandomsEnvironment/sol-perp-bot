"""Feature ablation + candidate v5 indicators: are more indicators signal?

Three questions, answered on the selection folds only:
  1. What does the current model actually use? (gain importances)
  2. Does pruning to the top-K features hurt? (if not, extras are noise)
  3. Do NEW indicator families (VWAP/session anchors) add edge?

Usage: python scripts/ablate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import labeling as LAB  # noqa: E402
import train as T  # noqa: E402
import validation as V  # noqa: E402
from experiment import build_dataset, feature_columns, fit_recipe  # noqa: E402
import features as F  # noqa: E402
import lightgbm as lgb  # noqa: E402

HORIZON_NAME, DEADBAND = "slow", 0.0020
SEEDS = [42, 137, 2718]


def add_v5(feats: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """VWAP distance + session/week anchor features (all causal)."""
    out = feats.copy()
    tp = (candles["high"] + candles["low"] + candles["close"]) / 3.0
    vol = candles["volume"].replace(0, np.nan)
    day = candles.index.normalize()
    pv = tp * vol
    vwap_day = pv.groupby(day).cumsum() / vol.groupby(day).cumsum()
    out["vwap_dist_day"] = candles["close"] / vwap_day - 1.0
    out["vwap_dist_1440"] = candles["close"] / (pv.rolling(1440).sum()
                                                / vol.rolling(1440).sum()) - 1.0
    out["dist_day_open"] = (candles["close"] /
                            candles["close"].groupby(day).transform("first") - 1.0)
    week = candles.index.to_period("W")
    out["dist_week_open"] = (candles["close"] /
                             candles["close"].groupby(week).transform("first") - 1.0)
    out["vwap_slope_60"] = vwap_day.pct_change(60)
    return out


def evaluate(ds, candles, cols, label):
    y = LAB.labels_to_classes(ds.label)
    X = F.prepare_model_input(ds[cols], cols)
    splits = V.purged_walk_forward(X.index, horizon_min=config.HORIZONS_MIN[HORIZON_NAME])
    evaluated = splits[:-1][:2]
    all_y, all_p, all_prior = [], [], []
    grids = {(t, d): [] for t in (0.40, 0.45, 0.50) for d in (0.05, 0.10)}
    for s in evaluated:
        tr = X.index[(X.index >= s.train_start) & (X.index < s.train_end)]
        te = X.index[(X.index >= s.test_start) & (X.index < s.test_end)]
        Xtr, ytr = X.loc[tr], y[X.index.get_indexer(tr)]
        fitted, _ = fit_recipe(Xtr, ytr, config.HORIZONS_MIN[HORIZON_NAME],
                               estimators=1200, lgbm_over={}, class_weight=None,
                               seeds=SEEDS)
        p = np.asarray(fitted.predict_proba(X.loc[te]), dtype=float)
        yt = y[X.index.get_indexer(te)]
        counts = np.bincount(ytr, minlength=3)
        all_y.extend(yt.tolist()); all_p.extend(p.tolist())
        all_prior.extend(np.tile(counts / counts.sum(), (len(yt), 1)).tolist())
        te_atr = ds.loc[te, "atr_pct"].to_numpy()
        h = config.HORIZONS_MIN[HORIZON_NAME]
        for t in (0.40, 0.45, 0.50):
            for d in (0.05, 0.10):
                pred = V.decisive_classes(p, tau=t, delta=d)
                r = T._path(candles, te, pred, te_atr, h,
                            cost=config.ROUND_TRIP_COST, barrier_scale=h ** 0.5)
                sel = (pred != 0) & np.isfinite(r)
                grids[(t, d)].extend(r[sel].tolist())
    pooled = V.probability_metrics(np.asarray(all_y), np.asarray(all_p))
    prior = V.probability_metrics(np.asarray(all_y), np.asarray(all_prior))
    best_pf, best_cell = 0.0, None
    for k, rr in grids.items():
        rr = np.asarray(rr, dtype=float)
        if len(rr) >= 30:
            pf = V.profit_factor(rr)
            if pf > best_pf:
                best_pf, best_cell = pf, (k, int(len(rr)), float(np.mean(rr > 0)))
    print(f"{label:16s} d_ll={prior['log_loss']-pooled['log_loss']:+.4f} "
          f"ll={pooled['log_loss']:.4f} bal={pooled['balanced_accuracy']:.4f} "
          f"best_pf={best_pf:.2f}@{best_cell}", flush=True)
    return pooled, prior


def main():
    data = T.load_data()
    ds, candles = build_dataset(data, config.HORIZONS_MIN[HORIZON_NAME], DEADBAND)
    v3_cols = feature_columns(ds, "v3")

    # 1) importance audit from a quick single-seed fit on fold 1
    y = LAB.labels_to_classes(ds.label)
    splits = V.purged_walk_forward(ds.index, horizon_min=config.HORIZONS_MIN[HORIZON_NAME])
    s = splits[0]
    tr = ds.index[(ds.index >= s.train_start) & (ds.index < s.train_end)]
    params = {k: v for k, v in config.LGBM_PARAMS.items() if k != "num_class"}
    params.update(n_jobs=4, random_state=42, deterministic=True,
                  force_col_wise=True, n_estimators=300)
    m = lgb.LGBMClassifier(**params)
    pos = ds.index.get_indexer(tr)
    m.fit(ds[v3_cols].iloc[pos], y[pos], sample_weight=V.recency_weights(tr))
    imp = pd.Series(m.feature_importances_, index=v3_cols).sort_values(ascending=False)
    print(f"== IMPORTANCE AUDIT ({len(v3_cols)} v3 features) ==")
    print(imp.head(10).to_string())
    dead = int((imp == 0).sum())
    share_top10 = imp.head(10).sum() / imp.sum()
    print(f"zero-gain features: {dead}/{len(v3_cols)}; top-10 share of gain: {share_top10:.0%}\n")

    print("== ABLATION (slow, db20, 3-seed, selection folds) ==")
    evaluate(ds, candles, v3_cols, "full-v3")
    evaluate(ds, candles, list(imp.head(15).index), "top15-v3")
    evaluate(ds, candles, list(imp.head(8).index), "top8-v3")

    ds5 = add_v5(ds, candles).dropna()
    v5_cols = v3_cols + ["vwap_dist_day", "vwap_dist_1440", "dist_day_open",
                         "dist_week_open", "vwap_slope_60"]
    print("\n== NEW INDICATORS (v5 = v3 + VWAP/anchors) ==")
    evaluate(ds5, candles, v5_cols, "v5-full")
    evaluate(ds5, candles, v5_cols[-5:], "v5-only")


if __name__ == "__main__":
    main()
