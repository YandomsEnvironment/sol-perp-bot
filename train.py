"""Purged walk-forward research trainer. Earlier folds are development OOS, not a fresh holdout; artifacts are never promoted automatically."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features as F  # noqa: E402
import labeling as LAB  # noqa: E402
import validation as V  # noqa: E402


def load_data() -> dict:
    store = config.DATA_STORE
    sol = pd.read_parquet(store / "candles_SOL_1m_binance.parquet")
    btc = pd.read_parquet(store / "candles_BTC_1m_binance.parquet")
    fpath = store / "funding_SOL.parquet"
    mpath = store / "metrics_SOL.parquet"
    return {"sol": sol, "btc": btc,
            "funding": pd.read_parquet(fpath) if fpath.exists() else None,
            "metrics": pd.read_parquet(mpath) if mpath.exists() else None}


def _indexed(candles):
    if hasattr(F, "_to_indexed"):
        return F._to_indexed(candles)
    out = candles.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        col = next((c for c in ("timestamp", "time", "datetime") if c in out), None)
        if col is None:
            raise ValueError("Candles require a DatetimeIndex or timestamp column")
        out = out.set_index(col)
    out.index = pd.to_datetime(out.index, utc=True)
    return out.sort_index()


def build_dataset(data: dict, horizon_min: int, target: str = "forward") -> pd.DataFrame:
    """Construct OHLCV-only features, a shared ATR14 barrier scale and labels."""
    if target not in ("forward", "barrier"):
        raise ValueError("target must be forward or barrier")
    candles = _indexed(data["sol"])
    feats = F.compute_model_features(data["sol"], btc_candles=data.get("btc"))
    feats = feats.reindex(candles.index)
    # The same ATR14 percentage accompanies either target and the path simulator.
    if hasattr(F, "atr"):
        atr_pct = F.atr(candles, period=14) / candles["close"]
    else:
        tr = pd.concat([candles.high - candles.low,
                        (candles.high - candles.close.shift()).abs(),
                        (candles.low - candles.close.shift()).abs()], axis=1).max(axis=1)
        atr_pct = tr.rolling(14, min_periods=14).mean() / candles.close
    if target == "forward":
        labels = LAB.forward_return_labels(
            candles, horizon_min, deadband=config.ROUND_TRIP_COST)
    else:
        labels = LAB.triple_barrier_labels(
            candles, atr_pct * candles.close, horizon=horizon_min,
            tp_mult=config.TP_ATR, sl_mult=config.SL_ATR)
    ds = feats.copy()
    ds["label"] = labels
    ds["atr_pct"] = atr_pct
    ds = ds.join(candles[["close"]], how="left")
    ds = ds.dropna()
    print(f"dataset: {len(ds)} usable rows; target={target}", flush=True)
    return ds


def make_model(estimators=None):
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    params = {k: v for k, v in config.LGBM_PARAMS.items() if k != "num_class"}
    if estimators is not None:
        params["n_estimators"] = estimators
    params.update(n_jobs=4, random_state=42, deterministic=True, force_col_wise=True)
    return LGBMClassifier(**params), lgb


def train_one(model, lgb, X_tr, y_tr, horizon_min=60, target="barrier"):
    parts = V.temporal_partitions(X_tr.index, horizon_min)
    fit, stop = parts["fit"], parts["early_stop"]
    model.fit(X_tr.iloc[fit], y_tr[fit],
              sample_weight=V.recency_weights(X_tr.index[fit]),
              eval_set=[(X_tr.iloc[stop], y_tr[stop])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    cal = V.fit_isotonic_multiclass(model, X_tr.iloc[parts["calibration"]],
                                    y_tr[parts["calibration"]])
    selection = parts["selection"]
    y_selection = y_tr[selection]
    raw_metrics = V.probability_metrics(y_selection, model.predict_proba(X_tr.iloc[selection]))
    calibrated_metrics = V.probability_metrics(y_selection, cal.predict_proba(X_tr.iloc[selection]))
    use_calibration = calibrated_metrics["log_loss"] < raw_metrics["log_loss"]
    chosen = cal if use_calibration else model
    chosen.calibration_selection = {"choice": "isotonic" if use_calibration else "raw",
                                    "raw": raw_metrics, "isotonic": calibrated_metrics,
                                    "selection_reused_for_threshold": True}
    return chosen, parts


def _path(candles, index, pred, atr, horizon, cost=None, target="barrier",
          barrier_scale=1.0, tp=None, sl=None):
    """Simulate against the complete raw minute grid, retaining future bars."""
    pred = np.asarray(pred)
    signals = np.zeros(len(candles), dtype=int)
    atr_full = np.full(len(candles), np.nan, dtype=float)
    positions = candles.index.get_indexer(index)
    valid = positions >= 0
    signals[positions[valid]] = pred[valid]
    atr_full[positions[valid]] = np.asarray(atr)[valid]
    kw = {} if tp is None and sl is None else {"tp": config.TP_ATR if tp is None else tp,
                                               "sl": config.SL_ATR if sl is None else sl}
    full = np.asarray(V.path_returns(
        candles, signals, horizon, atr_pct=atr_full if target == "barrier" else None,
        cost=config.ROUND_TRIP_COST if cost is None else cost,
        barrier_scale=barrier_scale, **kw), dtype=float)
    result = np.full(len(index), np.nan, dtype=float)
    result[valid] = full[positions[valid]]
    return result


def _metrics(y, p, prior_counts):
    result = V.probability_metrics(y, p)
    counts = np.asarray(prior_counts, dtype=float)
    if counts.sum() <= 0:
        counts = np.ones(3, dtype=float)
    prior = np.tile(counts / counts.sum(), (len(y), 1))
    result["majority_prior_baseline"] = V.probability_metrics(y, prior)
    return result


def _threshold_diagnostics(y, p, candles, index, atr, horizon, thresholds, target="barrier"):
    pred, conf = p.argmax(axis=1), p.max(axis=1)
    out = []
    for threshold in thresholds:
        accepted = (pred != 0) & (conf >= threshold)
        gated_pred = np.where(accepted, pred, 0)
        returns = _path(candles, index, gated_pred, atr, horizon, target=target)
        selected = accepted & np.isfinite(returns)
        r = returns[selected]
        out.append({"threshold": float(threshold), "signals": int(selected.sum()),
                    "win_rate": float(np.mean(r > 0)) if len(r) else None,
                    "mean_return": float(np.mean(r)) if len(r) else None,
                    "profit_factor": float(V.profit_factor(r)) if len(r) else None})
    return out


def run(horizon_name: str, threshold: float, *, target="forward", estimators=None,
        max_folds=None, output_dir=None):
    horizon = config.HORIZONS_MIN[horizon_name]
    data = load_data()
    candles = _indexed(data["sol"])
    ds = build_dataset(data, horizon, target)
    y = LAB.labels_to_classes(ds.label)
    feature_cols = [c for c in ds if c not in ("label", "atr_pct", "close")]
    X = F.prepare_model_input(ds[feature_cols], feature_cols)
    atr = ds.atr_pct.to_numpy()
    splits = V.purged_walk_forward(X.index, horizon_min=horizon)
    # Reserve the final month regardless of cap: it is never a selection fold.
    evaluated = splits[:-1] if len(splits) > 1 else []
    if max_folds is not None:
        evaluated = evaluated[:max(0, max_folds)]
    records, all_y, all_p, all_prior, all_r = [], [], [], [], []
    fold_taus = []
    for s in evaluated:
        tr = X.index[(X.index >= s.train_start) & (X.index < s.train_end)]
        te = X.index[(X.index >= s.test_start) & (X.index < s.test_end)]
        if len(tr) < 500 or len(te) < 100:
            continue
        Xtr = X.loc[tr]
        ytr = y[X.index.get_indexer(tr)]
        model, lgb = make_model(estimators)
        fitted, parts = train_one(model, lgb, Xtr, ytr, horizon, target)
        sel = parts["selection"]
        psel = fitted.predict_proba(Xtr.iloc[sel])
        ysel = ytr[sel]
        sel_idx = Xtr.index[sel]
        sel_atr = ds.loc[sel_idx, "atr_pct"].to_numpy()
        sel_pred = psel.argmax(axis=1)
        # Threshold selection uses honest, non-overlapping path PnL, not labels.
        candidates = _threshold_diagnostics(
            ysel, psel, candles, sel_idx, sel_atr, horizon,
            [0.30, .35, .40, .45, .50, .55, .60, .65, .70], target)
        viable = [d for d in candidates
                  if d["threshold"] >= threshold and d["signals"] >= 30
                  and d["profit_factor"] is not None and d["profit_factor"] > 1.3]
        tau = max(viable, key=lambda d: d["profit_factor"])["threshold"] if viable else 1.01
        fold_taus.append(tau)
        raw_p = model.predict_proba(X.loc[te])
        p = fitted.predict_proba(X.loc[te])
        yt = y[X.index.get_indexer(te)]
        pred, confidence = p.argmax(axis=1), p.max(axis=1)
        pred = np.where((pred != 0) & (confidence >= tau), pred, 0)
        returns = _path(candles, te, pred, ds.loc[te, "atr_pct"].to_numpy(), horizon, target=target)
        all_y.extend(yt.tolist()); all_p.extend(p.tolist()); all_r.extend(returns.tolist())
        fit_counts = np.bincount(ytr[parts["fit"]], minlength=3)
        fit_prior = fit_counts / fit_counts.sum() if fit_counts.sum() else np.full(3, 1 / 3)
        all_prior.extend([fit_prior.tolist()] * len(yt))
        records.append({"test_start": str(s.test_start), "n": len(yt),
                        "probability_metrics": _metrics(yt, p, fit_counts), "threshold": tau,
                        "raw_probability_metrics": _metrics(yt, raw_p, fit_counts),
                        "calibration_selection": fitted.calibration_selection,
                        "threshold_diagnostics": _threshold_diagnostics(
                            yt, p, candles, te, ds.loc[te, "atr_pct"].to_numpy(), horizon,
                            [threshold, tau, .4, .5, .6], target)})
        print(f"{s.test_start:%Y-%m}: n={len(yt)} tau={tau:.2f} "
              f"accuracy={records[-1]['probability_metrics']['accuracy']:.4f}", flush=True)
    pooled = V.probability_metrics(np.asarray(all_y, dtype=int), np.asarray(all_p, dtype=float)) if all_y else {}
    if all_y:
        pooled["majority_prior_baseline"] = V.probability_metrics(
            np.asarray(all_y, dtype=int), np.asarray(all_prior, dtype=float))
    # Training cutoff anchored to last evaluated fold, never the reserved test month,
    # and unaffected by --max-folds (which must not leak later fold data).
    if not records:
        raise RuntimeError("No eligible evaluated folds; cannot train candidate")
    last_eval = next(s for s in reversed(evaluated) if str(s.test_start) == records[-1]["test_start"])
    cutoff = last_eval.train_end
    final_index = X.index[X.index < cutoff]
    model, lgb = make_model(estimators)
    final, _ = train_one(model, lgb, X.loc[final_index], y[X.index.get_indexer(final_index)], horizon, target)
    outdir = Path(output_dir) if output_dir else config.MODELS_DIR / "candidates"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    stem = f"{horizon_name}_{target}_{stamp}"
    import joblib
    bundle = {"model": final, "feature_names": feature_cols, "horizon_name": horizon_name,
              "horizon_min": horizon, "conf_threshold": 1.01,
              "evaluated_fold_thresholds": fold_taus,
              "threshold_note": "No final-model threshold fitted; promotion disabled",
              "input_contract": "closed-ohlcv-v2", "target": target,
              "target_deadband": config.ROUND_TRIP_COST if target == "forward" else None,
              "target_barriers": {"tp_atr": config.TP_ATR, "sl_atr": config.SL_ATR} if target == "barrier" else None,
              "trained_at": pd.Timestamp.now(tz="UTC"), "training_cutoff": str(cutoff),
              "calibration_selection": final.calibration_selection,
              "aggregate": {"oos_probability_metrics": pooled,
                            "path_signal_count": int(np.isfinite(all_r).sum()),
                            "path_mean_return": float(np.nanmean(all_r)) if np.isfinite(all_r).any() else None,
                            "gate_pass": False,
                            "gate_reason": config.VENUE_VALIDATION_REASON},
              "gate_pass": False, "research_gate_pass": False,
              "promotion_block_reason": config.VENUE_VALIDATION_REASON}
    model_path = outdir / f"{stem}.pkl"
    joblib.dump(bundle, model_path)
    diagnostics = {"input_contract": "closed-ohlcv-v2", "target": target,
                   "horizon_min": horizon, "folds": records, "all_oos_probability_metrics": pooled,
                   "all_oos_path_returns": {"signals": int(np.isfinite(all_r).sum()),
                                            "mean": float(np.nanmean(all_r)) if np.isfinite(all_r).any() else None},
                   "gate_pass": False, "research_gate_pass": False,
                   "promotion_block_reason": config.VENUE_VALIDATION_REASON,
                   "artifact": model_path.name}
    (outdir / f"{stem}_diagnostics.json").write_text(json.dumps(diagnostics, default=str, indent=2))
    print(f"candidate saved: {model_path}\ndiagnostics: {outdir / (stem + '_diagnostics.json')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", choices=list(config.HORIZONS_MIN), default="mid")
    ap.add_argument("--threshold", type=float, default=config.CONF_THRESHOLD)
    ap.add_argument("--estimators", type=int, default=None)
    ap.add_argument("--target", choices=("forward", "barrier"), default="forward")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    started = time.time()
    run(args.horizon, args.threshold, target=args.target, estimators=args.estimators,
        max_folds=args.max_folds, output_dir=args.output_dir)
    print(f"done in {time.time() - started:.0f}s", flush=True)