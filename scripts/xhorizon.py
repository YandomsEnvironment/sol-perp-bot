"""Cross-horizon agreement: slow calls confirmed by the mid model's lean.

The slow model's calls are only taken when the independently-trained mid
(15m) model agrees on direction. Evaluated on selection folds only; the
simulation runs at the slow horizon.

Usage: python scripts/xhorizon.py
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

SLOW_DB, MID_DB, H_SLOW, H_MID = 0.0020, 0.0020, 60, 15
SEEDS = [42, 137, 2718]


def main():
    data = T.load_data()
    ds_s, candles = build_dataset(data, H_SLOW, SLOW_DB)
    ds_m, _ = build_dataset(data, H_MID, MID_DB)
    y_s = LAB.labels_to_classes(ds_s.label)
    cols = feature_columns(ds_s, "v3")
    Xs = F.prepare_model_input(ds_s[cols], cols)
    Xm = F.prepare_model_input(ds_m[cols], cols)
    splits = V.purged_walk_forward(Xs.index, horizon_min=H_SLOW)
    evaluated = splits[:-1][:2]

    CELLS = [(tau, d, mode) for tau in (0.40, 0.45) for d in (0.05, 0.10)
             for mode in ("sign", "decisive")]
    grids = {c: [] for c in CELLS}
    base = []
    started = time.time()
    for s in evaluated:
        tr_s = Xs.index[(Xs.index >= s.train_start) & (Xs.index < s.train_end)]
        te_s = Xs.index[(Xs.index >= s.test_start) & (Xs.index < s.test_end)]
        tr_m = Xm.index[(Xm.index >= s.train_start) & (Xm.index < s.train_end)]
        te_m = Xm.index[(Xm.index >= s.test_start) & (Xm.index < s.test_end)]
        slow, _ = fit_recipe(Xs.loc[tr_s], y_s[Xs.index.get_indexer(tr_s)], H_SLOW,
                             estimators=1200, lgbm_over={}, class_weight=None, seeds=SEEDS)
        mid, _ = fit_recipe(Xm.loc[tr_m],
                            LAB.labels_to_classes(ds_m.loc[tr_m, "label"]), H_MID,
                            estimators=1200, lgbm_over={}, class_weight=None, seeds=SEEDS)
        ps = np.asarray(slow.predict_proba(Xs.loc[te_s]), dtype=float)
        pm = np.asarray(mid.predict_proba(Xm.loc[te_m]), dtype=float)
        te_atr = ds_s.loc[te_s, "atr_pct"].to_numpy()
        mid_sign = np.where(pm[:, 1] >= pm[:, 2], 1, 2)
        mid_dec = V.decisive_classes(pm, tau=0.40, delta=0.05)
        for tau, d, mode in CELLS:
            pred = V.decisive_classes(ps, tau=tau, delta=d)
            if mode == "sign":
                pred = np.where((pred != 0) & (mid_sign == pred), pred, 0)
            else:
                pred = np.where((pred != 0) & (mid_dec == pred), pred, 0)
            r = T._path(candles, te_s, pred, te_atr, H_SLOW,
                        cost=config.ROUND_TRIP_COST, barrier_scale=H_SLOW ** 0.5)
            sel = (pred != 0) & np.isfinite(r)
            grids[(tau, d, mode)].extend(r[sel].tolist())
        base_pred = V.decisive_classes(ps, tau=0.45, delta=0.10)
        r = T._path(candles, te_s, base_pred, te_atr, H_SLOW,
                    cost=config.ROUND_TRIP_COST, barrier_scale=H_SLOW ** 0.5)
        base.extend(r[(base_pred != 0) & np.isfinite(r)].tolist())
        print(f"{s.test_start:%Y-%m}: fold done", flush=True)

    base = np.asarray(base)
    print(f"baseline (slow alone, .45/.10): n={len(base)} "
          f"win={np.mean(base > 0):.3f} pf={V.profit_factor(base):.2f}", flush=True)
    table = []
    for c, rr in grids.items():
        rr = np.asarray(rr)
        if len(rr) < 30:
            continue
        table.append({"tau": c[0], "delta": c[1], "mode": c[2], "n": int(len(rr)),
                      "win": float(np.mean(rr > 0)), "mean": float(rr.mean()),
                      "pf": float(V.profit_factor(rr))})
    record = {"name": "xhorizon", "mode": "xhorizon", "ts": pd.Timestamp.now(tz="UTC").isoformat(),
              "sim": "barrier_scale=sqrt(h)", "baseline_pf": float(V.profit_factor(base)),
              "grid": table, "seconds": round(time.time() - started, 1)}
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    for row in sorted(table, key=lambda r: -r["pf"]):
        print(f"tau={row['tau']} d={row['delta']} {row['mode']:8s}: n={row['n']:4d} "
              f"win={row['win']:.3f} mean={row['mean']:+.5f} pf={row['pf']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
