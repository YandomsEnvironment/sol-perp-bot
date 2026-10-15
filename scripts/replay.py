"""One-shot confirmation replay on the reserved unseen window.

Trains the chosen recipe on ALL data before --train-through, then simulates
every minute after it through the decisive-direction rule and the fee-aware
path simulator. This is the holdout: run it once per recipe, report whatever
it says, and do not tune against it.

Usage:
  python scripts/replay.py --name v3cw --tau 0.50 --delta 0.10 \
      --deadband 0.0008 --features v3 --class-weight balanced --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features as F  # noqa: E402
import labeling as LAB  # noqa: E402
import validation as V  # noqa: E402
from experiment import build_dataset, feature_columns, fit_recipe  # noqa: E402


def summarize(returns: np.ndarray) -> dict:
    r = pd.Series(returns).dropna()
    if not len(r):
        return {"signals": 0}
    return {"signals": int(len(r)), "win_rate": float((r > 0).mean()),
            "mean_return": float(r.mean()),
            "total_return": float((1 + r).prod() - 1),
            "profit_factor": float(V.profit_factor(r)),
            "max_drawdown": float(V.max_drawdown(r)),
            "gates": V.gates_check(r)}


def weekly_breakdown(index: pd.DatetimeIndex, returns: np.ndarray) -> list:
    s = pd.Series(returns, index=index).dropna()
    if not len(s):
        return []
    out = []
    for week, grp in s.groupby(pd.Grouper(freq="W")):
        if len(grp):
            out.append({"week": str(week.date()), **summarize(grp.to_numpy())})
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--horizon", choices=list(config.HORIZONS_MIN), default="mid")
    ap.add_argument("--train-through", default="2026-09-01")
    ap.add_argument("--deadband", type=float, default=config.ROUND_TRIP_COST)
    ap.add_argument("--features", choices=("v2", "v3"), default="v3")
    ap.add_argument("--class-weight", default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--estimators", type=int, default=1200)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--save-bundle", action="store_true")
    a = ap.parse_args()

    horizon = config.HORIZONS_MIN[a.horizon]
    import train as T
    data = T.load_data()
    ds, candles = build_dataset(data, horizon, a.deadband)
    y = LAB.labels_to_classes(ds.label)
    cols = feature_columns(ds, a.features)
    X = F.prepare_model_input(ds[cols], cols)
    cutoff = pd.Timestamp(a.train_through, tz="UTC")

    tr = X.index[X.index < cutoff - pd.Timedelta(minutes=horizon)]
    te = X.index[X.index >= cutoff]
    if len(tr) < 1000 or len(te) < 1000:
        raise SystemExit(f"bad split: train={len(tr)} test={len(te)}")
    Xtr, ytr = X.loc[tr], y[X.index.get_indexer(tr)]
    print(f"training on {len(Xtr)} rows through {cutoff} ...", flush=True)
    started = time.time()
    cw = "balanced" if a.class_weight == "balanced" else (
        json.loads(a.class_weight) if a.class_weight else None)
    seeds = [42] if a.seeds == 1 else [42, 137, 2718][:a.seeds]
    fitted, _ = fit_recipe(Xtr, ytr, horizon, estimators=a.estimators,
                           lgbm_over={}, class_weight=cw, seeds=seeds)
    print(f"fit done in {time.time() - started:.0f}s", flush=True)

    p = np.asarray(fitted.predict_proba(X.loc[te]), dtype=float)
    yt = y[X.index.get_indexer(te)]
    prior_counts = np.bincount(ytr, minlength=3)
    prior = np.tile(prior_counts / prior_counts.sum(), (len(yt), 1))
    metrics = V.probability_metrics(yt, p)
    metrics["majority_prior_baseline"] = V.probability_metrics(yt, prior)
    pred = V.decisive_classes(p, tau=a.tau, delta=a.delta)
    te_atr = ds.loc[te, "atr_pct"].to_numpy()

    result = {"name": a.name, "horizon": a.horizon, "horizon_min": horizon,
              "train_through": str(cutoff), "test_start": str(te[0]),
              "test_end": str(te[-1]), "n_test": int(len(te)),
              "deadband": a.deadband, "features": a.features,
              "class_weight": cw, "seeds": seeds, "estimators": a.estimators,
              "tau": a.tau, "delta": a.delta,
              "calibration": fitted.calibration_selection["choice"],
              "probability_metrics": metrics,
              "directional_calls": int((pred != 0).sum()),
              "coverage": float((pred != 0).mean())}
    for label, cost in (("taker", config.ROUND_TRIP_COST),
                        ("maker", config.MAKER_ROUND_TRIP_COST)):
        r = T._path(candles, te, pred, te_atr, horizon, cost=cost,
                    barrier_scale=horizon ** 0.5)
        taken = (pred != 0) & np.isfinite(r)
        result[label] = summarize(r[taken])
        if label == "taker":
            result["weekly"] = weekly_breakdown(te[taken], r[taken])
    result["sim"] = "barrier_scale=sqrt(h)"
    out = ROOT / "logs" / f"replay_{a.name}_{a.horizon}.json"
    out.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in result.items()
                      if k in ("probability_metrics", "directional_calls",
                               "coverage", "taker", "maker")}, indent=2))
    for wk in result.get("weekly", []):
        print(f"  {wk['week']}: n={wk.get('signals', 0):4d} "
              f"win={wk.get('win_rate', 0):.3f} "
              f"pf={wk.get('profit_factor', 0):.2f}")
    print(f"saved {out}", flush=True)

    if a.save_bundle:
        outdir = ROOT / "models" / "candidates" / f"replay_{a.name}"
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        contract = F.CONTRACT_V3 if a.features == "v3" else F.CONTRACT_V2
        bundle = {"model": fitted, "feature_names": cols,
                  "horizon_name": a.horizon, "horizon_min": horizon,
                  "conf_threshold": 1.01, "decisive_rule": {"tau": a.tau, "delta": a.delta},
                  "input_contract": contract, "target": "forward",
                  "target_deadband": a.deadband, "trained_at": pd.Timestamp.now(tz="UTC"),
                  "training_cutoff": str(cutoff),
                  "calibration_selection": fitted.calibration_selection,
                  "aggregate": {"oos_probability_metrics": metrics,
                                "gate_pass": False,
                                "gate_reason": config.VENUE_VALIDATION_REASON},
                  "gate_pass": False, "research_gate_pass": False,
                  "promotion_block_reason": config.VENUE_VALIDATION_REASON}
        path = outdir / f"{a.horizon}_forward_{stamp}.pkl"
        joblib.dump(bundle, path)
        (outdir / f"{a.horizon}_forward_{stamp}_diagnostics.json").write_text(
            json.dumps(result, default=str, indent=2), encoding="utf-8")
        print(f"bundle saved: {path}", flush=True)


if __name__ == "__main__":
    main()

