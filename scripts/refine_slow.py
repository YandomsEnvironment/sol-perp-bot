"""Slow-horizon refinements on selection folds: exit ratio + persistence.

The 2:1 TP/SL ratio was inherited from config, never optimized; entry
persistence (lean must hold k bars) is a whipsaw filter. Both are evaluated
on the Jul/Aug selection folds only.

Usage: python scripts/refine_slow.py
"""
from __future__ import annotations

import json
import sys
import time
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
from experiment import build_dataset, feature_columns, fit_recipe, LOG  # noqa: E402
import features as F  # noqa: E402

H, DEADBAND, RULE = 60, 0.0020, (0.45, 0.10)
TPSL = [(2.0, 1.0), (1.5, 1.5), (2.5, 1.0), (3.0, 1.5), (1.5, 1.0), (3.0, 1.0)]
PERSIST = (1, 2, 3, 5)


def persisted(lean: np.ndarray, k: int) -> np.ndarray:
    """True where the lean has held the same value for the last k bars."""
    out = np.ones(len(lean), dtype=bool)
    for j in range(1, k):
        out[j:] &= lean[j:] == lean[:-j]
    return out


def main():
    data = T.load_data()
    ds, candles = build_dataset(data, H, DEADBAND)
    y = LAB.labels_to_classes(ds.label)
    cols = feature_columns(ds, "v3")
    X = F.prepare_model_input(ds[cols], cols)
    splits = V.purged_walk_forward(X.index, horizon_min=H)
    evaluated = splits[:-1][:2]
    grids = {("tpsl", t, s): [] for t, s in TPSL}
    grids.update({("persist", k, 0): [] for k in PERSIST})
    started = time.time()
    for s in evaluated:
        tr = X.index[(X.index >= s.train_start) & (X.index < s.train_end)]
        te = X.index[(X.index >= s.test_start) & (X.index < s.test_end)]
        Xtr, ytr = X.loc[tr], y[X.index.get_indexer(tr)]
        fitted, _ = fit_recipe(Xtr, ytr, H, estimators=1200, lgbm_over={},
                               class_weight=None, seeds=[42, 137, 2718])
        p = np.asarray(fitted.predict_proba(X.loc[te]), dtype=float)
        pred = V.decisive_classes(p, tau=RULE[0], delta=RULE[1])
        lean = np.where(p[:, 1] >= p[:, 2], 1, 2)
        te_atr = ds.loc[te, "atr_pct"].to_numpy()
        for t, sl_ in TPSL:
            r = T._path(candles, te, pred, te_atr, H,
                        cost=config.ROUND_TRIP_COST, barrier_scale=H ** 0.5,
                        tp=t, sl=sl_)
            sel = (pred != 0) & np.isfinite(r)
            grids[("tpsl", t, sl_)].extend(r[sel].tolist())
        for k in PERSIST:
            keep = persisted(lean, k)
            pred_k = np.where(keep, pred, 0)
            r = T._path(candles, te, pred_k, te_atr, H,
                        cost=config.ROUND_TRIP_COST, barrier_scale=H ** 0.5)
            sel = (pred_k != 0) & np.isfinite(r)
            grids[("persist", k, 0)].extend(r[sel].tolist())
        print(f"{s.test_start:%Y-%m}: fold done", flush=True)

    table = []
    for key, rr in grids.items():
        rr = np.asarray(rr)
        if len(rr) < 30:
            continue
        table.append({"kind": key[0], "a": key[1], "b": key[2], "n": int(len(rr)),
                      "win": float(np.mean(rr > 0)), "mean": float(rr.mean()),
                      "pf": float(V.profit_factor(rr))})
    record = {"name": "refine-slow", "mode": "refine", "ts": pd.Timestamp.now(tz="UTC").isoformat(),
              "horizon": "slow", "rule": {"tau": RULE[0], "delta": RULE[1]},
              "sim": "barrier_scale=sqrt(h)", "grid": table,
              "seconds": round(time.time() - started, 1)}
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    for row in sorted(table, key=lambda r: -r["pf"]):
        print(f"{row['kind']:8s} a={row['a']} b={row['b']}: n={row['n']:4d} "
              f"win={row['win']:.3f} mean={row['mean']:+.5f} pf={row['pf']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
