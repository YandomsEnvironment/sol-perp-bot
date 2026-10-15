"""Walk-forward experiment harness for recipe iteration.

Evaluates a training recipe on the SAME purged walk-forward selection folds
train.py uses (everything except the final reserved month). The final month is
never touched here -- scripts/replay.py is the one-shot confirmation on it.

Discipline: iterate on selection folds only; pick the winner; replay once.

Usage:
  python scripts/experiment.py --name base --horizon mid --folds 2
  python scripts/experiment.py --name v3cw --features v3 --class-weight balanced \
      --deadband 0.0008 --seeds 3 --folds 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import features as F  # noqa: E402
import labeling as LAB  # noqa: E402
import train as T  # noqa: E402
import validation as V  # noqa: E402

TAUS = (0.40, 0.45, 0.50, 0.55, 0.60)
DELTAS = (0.05, 0.10, 0.15, 0.20)
LOG = ROOT / "logs" / "experiments.jsonl"


def build_dataset(data: dict, horizon: int, deadband: float,
                  target: str = "forward") -> tuple:
    """Mirror train.build_dataset; barrier target matches the path simulator."""
    candles = T._indexed(data["sol"])
    funding = data.get("funding")
    oi = data.get("metrics")
    feats = F.compute_model_features(data["sol"], btc_candles=data.get("btc"),
                                     funding=funding, oi=oi)
    feats = feats.reindex(candles.index)
    atr_pct = F.atr(candles, period=14) / candles["close"]
    if target == "barrier":
        labels = LAB.triple_barrier_labels(
            candles, atr_pct * candles["close"], horizon=horizon,
            tp_mult=config.TP_ATR, sl_mult=config.SL_ATR)
    else:
        labels = LAB.forward_return_labels(candles, horizon, deadband=deadband)
    ds = feats.copy()
    ds["label"] = labels
    ds["atr_pct"] = atr_pct
    ds = ds.join(candles[["close"]], how="left").dropna()
    return ds, candles


def feature_columns(ds: pd.DataFrame, version: str) -> list[str]:
    cols = [c for c in ds if c not in ("label", "atr_pct", "close")]
    flow = F.V4_COLUMNS
    pos = F.V5_COLUMNS
    if version == "v2":
        cols = [c for c in cols if c not in F.V3_COLUMNS + flow + pos]
    elif version == "v3":
        cols = [c for c in cols if c not in flow + pos]
    elif version == "v4":
        cols = [c for c in cols if c not in pos]
    elif version == "v5":
        cols = [c for c in cols if c not in flow]
    elif version not in ("v2", "v3", "v4", "v5"):
        raise ValueError("features must be v2, v3, v4 or v5")
    return cols


def fit_recipe(X_tr: pd.DataFrame, y_tr: np.ndarray, horizon: int, *,
               estimators: int, lgbm_over: dict, class_weight, seeds,
               half_life: float = None) -> tuple:
    """train_one-style fit: early stop, optional seed bag, calibration choice."""
    import lightgbm as lgb

    parts = V.temporal_partitions(X_tr.index, horizon)
    fit, stop = parts["fit"], parts["early_stop"]

    def make(seed):
        params = {k: v for k, v in config.LGBM_PARAMS.items() if k != "num_class"}
        params.update(lgbm_over or {})
        params.update(n_jobs=4, random_state=int(seed), deterministic=True,
                      force_col_wise=True)
        params["n_estimators"] = int(estimators)
        if class_weight:
            params["class_weight"] = class_weight
        return lgb.LGBMClassifier(**params)

    weights = (V.recency_weights(X_tr.index[fit]) if half_life is None
               else V.recency_weights(X_tr.index[fit], half_life_days=half_life))
    models = []
    for seed in seeds:
        m = make(seed)
        m.fit(X_tr.iloc[fit], y_tr[fit], sample_weight=weights,
              eval_set=[(X_tr.iloc[stop], y_tr[stop])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        models.append(m)
    bag = V.SeedBag(models) if len(models) > 1 else models[0]

    cal = V.fit_isotonic_multiclass(bag, X_tr.iloc[parts["calibration"]],
                                    y_tr[parts["calibration"]])
    cal.classes_ = np.array([0, 1, 2])
    sel = parts["selection"]
    ysel = y_tr[sel]
    raw_m = V.probability_metrics(ysel, bag.predict_proba(X_tr.iloc[sel]))
    cal_m = V.probability_metrics(ysel, cal.predict_proba(X_tr.iloc[sel]))
    use_cal = cal_m["log_loss"] < raw_m["log_loss"]
    chosen = cal if use_cal else bag
    if not hasattr(chosen, "classes_"):
        chosen.classes_ = np.array([0, 1, 2])
    chosen.calibration_selection = {"choice": "isotonic" if use_cal else "raw",
                                    "raw": raw_m, "isotonic": cal_m}
    return chosen, parts

def run_experiment(name: str, horizon_name: str, *, folds: int, deadband: float,
                   features: str, class_weight, seeds, estimators: int,
                   lgbm_over: dict, smoke: bool = False,
                   target: str = "forward", placebo: bool = False,
                   half_life: float = None) -> dict:
    horizon = config.HORIZONS_MIN[horizon_name]
    data = T.load_data()
    ds, candles = build_dataset(data, horizon, deadband, target=target)
    y = LAB.labels_to_classes(ds.label)
    if placebo:
        rng = np.random.default_rng(123)
        periods = ds.index.to_period("M").to_numpy()
        y = y.copy()
        for per in np.unique(periods):
            mask = periods == per
            y[mask] = rng.permutation(y[mask])
    cols = feature_columns(ds, features)
    X = F.prepare_model_input(ds[cols], cols)
    splits = V.purged_walk_forward(X.index, horizon_min=horizon)
    evaluated = splits[:-1][:folds] if len(splits) > 1 else []

    fold_rows, all_y, all_p, all_prior = [], [], [], []
    grids = {(tau, d): [] for tau in TAUS for d in DELTAS}
    started = time.time()
    for s in evaluated:
        tr = X.index[(X.index >= s.train_start) & (X.index < s.train_end)]
        te = X.index[(X.index >= s.test_start) & (X.index < s.test_end)]
        if len(tr) < 500 or len(te) < 100:
            continue
        Xtr, ytr = X.loc[tr], y[X.index.get_indexer(tr)]
        fitted, _ = fit_recipe(
            Xtr, ytr, horizon, estimators=estimators, lgbm_over=lgbm_over,
            class_weight=class_weight, seeds=seeds, half_life=half_life)
        p = np.asarray(fitted.predict_proba(X.loc[te]), dtype=float)
        yt = y[X.index.get_indexer(te)]
        counts = np.bincount(ytr, minlength=3)
        prior = np.tile(counts / counts.sum(), (len(yt), 1))
        all_y.extend(yt.tolist()); all_p.extend(p.tolist())
        all_prior.extend(prior.tolist())
        metrics = V.probability_metrics(yt, p)
        metrics["majority_prior_baseline"] = V.probability_metrics(yt, prior)
        te_atr = ds.loc[te, "atr_pct"].to_numpy()
        for tau in TAUS:
            for d in DELTAS:
                pred = V.decisive_classes(p, tau=tau, delta=d)
                r = T._path(candles, te, pred, te_atr, horizon,
                            cost=config.ROUND_TRIP_COST,
                            barrier_scale=horizon ** 0.5)
                sel = (pred != 0) & np.isfinite(r)
                grids[(tau, d)].extend(r[sel].tolist())
        fold_rows.append({"test_start": str(s.test_start), "n": len(yt),
                          "metrics": metrics,
                          "calibration": fitted.calibration_selection["choice"]})
        print(f"{s.test_start:%Y-%m}: n={len(yt)} acc={metrics['accuracy']:.4f} "
              f"bal={metrics['balanced_accuracy']:.4f} ll={metrics['log_loss']:.4f} "
              f"(prior ll={metrics['majority_prior_baseline']['log_loss']:.4f})",
              flush=True)

    pooled = V.probability_metrics(np.asarray(all_y), np.asarray(all_p))
    pooled["majority_prior_baseline"] = V.probability_metrics(
        np.asarray(all_y), np.asarray(all_prior))
    pooled_grid = []
    for tau in TAUS:
        for d in DELTAS:
            rr = np.asarray(grids[(tau, d)])
            pooled_grid.append({"tau": tau, "delta": d, "signals": int(len(rr)),
                                "win_rate": float(np.mean(rr > 0)) if len(rr) else None,
                                "mean_return": float(np.mean(rr)) if len(rr) else None,
                                "profit_factor": float(V.profit_factor(rr)) if len(rr) else None})
    record = {"name": name, "ts": pd.Timestamp.now(tz="UTC").isoformat(),
              "horizon": horizon_name, "horizon_min": horizon,
              "deadband": deadband, "features": features, "target": target,
              "placebo": placebo,
              "sim": "barrier_scale=sqrt(h)",
              "class_weight": class_weight, "seeds": list(seeds),
              "estimators": estimators, "lgbm_over": lgbm_over,
              "folds": fold_rows, "pooled": pooled, "grid": pooled_grid,
              "seconds": round(time.time() - started, 1)}
    if not smoke:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    print(f"\n== {name} pooled == acc={pooled['accuracy']:.4f} "
          f"bal={pooled['balanced_accuracy']:.4f} ll={pooled['log_loss']:.4f} "
          f"prior_ll={pooled['majority_prior_baseline']['log_loss']:.4f}", flush=True)
    for g in pooled_grid:
        if g["signals"]:
            print(f"  tau={g['tau']:.2f} d={g['delta']:.2f}: n={g['signals']:5d} "
                  f"win={g['win_rate']:.3f} mean={g['mean_return']:+.5f} "
                  f"pf={g['profit_factor']:.2f}", flush=True)
    return record


MARGINS = (0.05, 0.10, 0.15, 0.20, 0.30)


def fit_directional(X_tr: pd.DataFrame, y_bin: np.ndarray, horizon: int, *,
                    estimators: int, lgbm_over: dict, class_weight, seeds) -> tuple:
    """Binary up/down model on meaningful-move bars, early stop + calibration."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import log_loss

    parts = V.temporal_partitions(X_tr.index, horizon)
    fit, stop = parts["fit"], parts["early_stop"]

    def make(seed):
        params = {k: v for k, v in config.LGBM_PARAMS.items()
                  if k not in ("num_class", "objective")}
        params["objective"] = "binary"
        params.update(lgbm_over or {})
        params.update(n_jobs=4, random_state=int(seed), deterministic=True,
                      force_col_wise=True)
        params["n_estimators"] = int(estimators)
        if class_weight:
            params["class_weight"] = class_weight
        return lgb.LGBMClassifier(**params)

    weights = V.recency_weights(X_tr.index[fit])
    models = []
    for seed in seeds:
        m = make(seed)
        m.fit(X_tr.iloc[fit], y_bin[fit], sample_weight=weights,
              eval_set=[(X_tr.iloc[stop], y_bin[stop])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        models.append(m)
    bag = V.SeedBag(models) if len(models) > 1 else models[0]
    if isinstance(bag, V.SeedBag):
        bag.classes_ = np.array([0, 1])

    sel = parts["selection"]
    ysel = y_bin[sel]
    raw_p = np.asarray(bag.predict_proba(X_tr.iloc[sel]), dtype=float)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(raw_p[:, 1], ysel.astype(float))
    cal = V.CalibratedBinary(bag, ir)
    cal_p = np.asarray(cal.predict_proba(X_tr.iloc[sel]), dtype=float)
    use_cal = log_loss(ysel, cal_p[:, 1], labels=[0, 1]) < \
        log_loss(ysel, raw_p[:, 1], labels=[0, 1])
    chosen = cal if use_cal else bag
    if isinstance(chosen, (V.SeedBag, V.CalibratedBinary)):
        chosen.classes_ = np.array([0, 1])
    chosen.calibration_selection = {"choice": "isotonic" if use_cal else "raw",
                                    "raw_ll": float(log_loss(ysel, raw_p[:, 1], labels=[0, 1])),
                                    "isotonic_ll": float(log_loss(ysel, cal_p[:, 1], labels=[0, 1]))}
    return chosen, parts


def margin_classes(p_up: np.ndarray, margin: float) -> np.ndarray:
    """LONG/SHORT/FLAT from a binary P(up) and a decision margin."""
    p_up = np.asarray(p_up, dtype=float)
    out = np.zeros(len(p_up), dtype=np.int64)
    out[p_up - 0.5 >= margin] = 1
    out[0.5 - p_up >= margin] = 2
    return out

def run_directional(name: str, horizon_name: str, *, folds: int, deadband: float,
                    features: str, class_weight, seeds, estimators: int,
                    lgbm_over: dict, smoke: bool = False,
                    target: str = "forward") -> dict:
    from sklearn.metrics import roc_auc_score, log_loss

    horizon = config.HORIZONS_MIN[horizon_name]
    data = T.load_data()
    ds, candles = build_dataset(data, horizon, deadband)
    meaningful = ds.label.isin([-1.0, 1.0]).to_numpy()
    cols = feature_columns(ds, features)
    X_all = F.prepare_model_input(ds[cols], cols)
    Xm = X_all.loc[meaningful]
    y_bin = (ds.label[meaningful] == 1.0).astype(int).to_numpy()
    splits = V.purged_walk_forward(Xm.index, horizon_min=horizon)
    evaluated = splits[:-1][:folds] if len(splits) > 1 else []

    fold_rows, all_y, all_p = [], [], []
    CELLS = [(margin, vol_mult, agree)
             for margin in (0.05, 0.08)
             for vol_mult in (0.0, 1.5, 2.0)
             for agree in (False, True)]
    META = [(q, vol_mult) for q in (0.35, 0.45, 0.55) for vol_mult in (0.0, 1.5)]
    grids = {c: [] for c in CELLS + [("meta", q, v) for q, v in META]}
    mom_returns = []
    started = time.time()
    for s in evaluated:
        tr = Xm.index[(Xm.index >= s.train_start) & (Xm.index < s.train_end)]
        me = (Xm.index >= s.test_start) & (Xm.index < s.test_end)
        if len(tr) < 500 or me.sum() < 100:
            continue
        Xtr = Xm.loc[tr]
        ytr = y_bin[Xm.index.get_indexer(tr)]
        fitted, _ = fit_directional(Xtr, ytr, horizon, estimators=estimators,
                                    lgbm_over=lgbm_over, class_weight=class_weight,
                                    seeds=seeds)
        te_meaningful = Xm.index[me]
        pm = np.asarray(fitted.predict_proba(Xm.loc[te_meaningful]), dtype=float)
        ym = y_bin[Xm.index.get_indexer(te_meaningful)]
        all_y.extend(ym.tolist()); all_p.extend(pm[:, 1].tolist())
        try:
            auc = float(roc_auc_score(ym, pm[:, 1]))
        except ValueError:
            auc = None
        acc = float((np.asarray(pm[:, 1] > 0.5, dtype=int) == ym).mean())
        te_all = X_all.index[(X_all.index >= s.test_start) & (X_all.index < s.test_end)]
        p_all = np.asarray(fitted.predict_proba(X_all.loc[te_all]), dtype=float)
        te_atr = ds.loc[te_all, "atr_pct"].to_numpy()
        # Companion triclass fit on the SAME fold's training window: its job is
        # chop detection, so it can veto directional calls it considers noise.
        tr_all_idx = X_all.index[(X_all.index >= s.train_start) & (X_all.index < s.train_end)]
        tri, _ = fit_recipe(X_all.loc[tr_all_idx],
                            LAB.labels_to_classes(ds.loc[tr_all_idx, "label"]),
                            horizon, estimators=estimators, lgbm_over=lgbm_over,
                            class_weight=class_weight, seeds=seeds)
        p3 = np.asarray(tri.predict_proba(X_all.loc[te_all]), dtype=float)
        tri_pred = V.decisive_classes(p3, tau=0.40, delta=0.05)
        p3_move = p3[:, 1] + p3[:, 2]
        exp_move = te_atr * np.sqrt(horizon)
        for margin, vol_mult, agree in CELLS:
            pred = margin_classes(p_all[:, 1], margin)
            if vol_mult > 0:
                pred = np.where(exp_move >= vol_mult * config.ROUND_TRIP_COST,
                                pred, 0)
            if agree:
                pred = np.where((tri_pred != 0) & (tri_pred == pred), pred, 0)
            r = T._path(candles, te_all, pred, te_atr, horizon,
                        cost=config.ROUND_TRIP_COST)
            sel = (pred != 0) & np.isfinite(r)
            grids[(margin, vol_mult, agree)].extend(r[sel].tolist())
        # Meta cells: triclass P(a move is coming) picks WHEN, the binary
        # model's lean picks WHICH WAY.
        for q, vol_mult in META:
            pred = margin_classes(p_all[:, 1], 0.03)
            pred = np.where(p3_move >= q, pred, 0)
            if vol_mult > 0:
                pred = np.where(exp_move >= vol_mult * config.ROUND_TRIP_COST,
                                pred, 0)
            r = T._path(candles, te_all, pred, te_atr, horizon,
                        cost=config.ROUND_TRIP_COST)
            sel = (pred != 0) & np.isfinite(r)
            grids[("meta", q, vol_mult)].extend(r[sel].tolist())
        mom = V.momentum_baseline_classes(ds.loc[te_all], "ret_15m")
        mr = T._path(candles, te_all, mom, te_atr, horizon,
                     cost=config.ROUND_TRIP_COST)
        mom_returns.extend(mr[np.isfinite(mr) & (mom != 0)].tolist())
        fold_rows.append({"test_start": str(s.test_start),
                          "n_meaningful": int(me.sum()), "auc": auc,
                          "dir_accuracy": acc,
                          "calibration": fitted.calibration_selection["choice"]})
        print(f"{s.test_start:%Y-%m}: n={int(me.sum())} auc={auc:.4f} "
              f"dir_acc={acc:.4f}", flush=True)
    yv, pv = np.asarray(all_y), np.asarray(all_p)
    pooled = {"auc": float(roc_auc_score(yv, pv)) if len(yv) else None,
              "dir_accuracy": float(((pv > 0.5).astype(int) == yv).mean()),
              "log_loss": float(log_loss(yv, pv, labels=[0, 1]))}
    grid = [{"margin": c[0], "vol_mult": c[1], "agree": c[2],
             "signals": int(len(grids[c])),
             "win_rate": float(np.mean(np.asarray(grids[c]) > 0)) if len(grids[c]) else None,
             "mean_return": float(np.mean(grids[c])) if len(grids[c]) else None,
             "profit_factor": float(V.profit_factor(grids[c])) if len(grids[c]) else None}
            for c in list(CELLS) + [("meta", q, v) for q, v in META]]
    record = {"name": name, "mode": "directional", "ts": pd.Timestamp.now(tz="UTC").isoformat(),
              "horizon": horizon_name, "horizon_min": horizon, "deadband": deadband,
              "features": features, "class_weight": class_weight, "seeds": list(seeds),
              "estimators": estimators, "lgbm_over": lgbm_over, "folds": fold_rows,
              "pooled": pooled, "grid": grid,
              "momentum_baseline": {"signals": int(len(mom_returns)),
                                    "win_rate": float(np.mean(np.asarray(mom_returns) > 0)) if mom_returns else None,
                                    "mean_return": float(np.mean(mom_returns)) if mom_returns else None,
                                    "profit_factor": float(V.profit_factor(mom_returns)) if mom_returns else None},
              "seconds": round(time.time() - started, 1)}
    if not smoke:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    print(f"\n== {name} pooled == auc={pooled['auc']:.4f} "
          f"dir_acc={pooled['dir_accuracy']:.4f} | momentum pf="
          f"{record['momentum_baseline']['profit_factor']:.2f}", flush=True)
    for g in grid:
        if g["signals"]:
            print(f"  m={g['margin']:.2f} vol={g['vol_mult']:.1f} "
                  f"agree={int(g['agree'])}: n={g['signals']:5d} "
                  f"win={g['win_rate']:.3f} mean={g['mean_return']:+.5f} "
                  f"pf={g['profit_factor']:.2f}", flush=True)
    return record


META_FEATURES = ["atr_pct", "atr_pct_z", "rv_60m", "vol_z_60", "ret_15m",
                 "ret_1h", "range_pos_60", "ema_x", "hour_sin", "hour_cos"]


def _fit_meta(Xm, ym, seed=42):
    import lightgbm as lgb
    params = dict(objective="binary", n_estimators=300, learning_rate=0.05,
                  num_leaves=7, max_depth=3, min_data_in_leaf=30,
                  lambda_l2=5.0, verbosity=-1, random_state=seed,
                  deterministic=True, force_col_wise=True, n_jobs=4)
    m = lgb.LGBMClassifier(**params)
    m.fit(Xm, ym)
    return m


def run_metalabel(name: str, horizon_name: str, *, folds: int, deadband: float,
                  features: str, seeds, estimators: int, lgbm_over: dict,
                  class_weight=None, smoke: bool = False,
                  primary_tau: float = 0.45,
                  primary_delta: float = 0.05, target: str = "forward") -> dict:
    """Meta-labeling: a shallow second model scores the primary's own calls.

    Primary = triclass + decisive rule. Meta trains on the primary's
    SELECTION-slice calls (inside the training window, never test data),
    predicting whether a call would have netted a positive path return.
    """
    from sklearn.metrics import roc_auc_score
    horizon = config.HORIZONS_MIN[horizon_name]
    data = T.load_data()
    ds, candles = build_dataset(data, horizon, deadband, target=target)
    y = LAB.labels_to_classes(ds.label)
    cols = feature_columns(ds, features)
    X = F.prepare_model_input(ds[cols], cols)
    splits = V.purged_walk_forward(X.index, horizon_min=horizon)
    evaluated = splits[:-1][:folds] if len(splits) > 1 else []
    QGRID = (0.40, 0.45, 0.50, 0.55, 0.60)
    base_returns, meta_returns = [], {q: [] for q in QGRID}
    meta_n, meta_aucs = [], []
    started = time.time()
    for s in evaluated:
        tr = X.index[(X.index >= s.train_start) & (X.index < s.train_end)]
        te = X.index[(X.index >= s.test_start) & (X.index < s.test_end)]
        if len(tr) < 500 or len(te) < 100:
            continue
        Xtr, ytr = X.loc[tr], y[X.index.get_indexer(tr)]
        primary, parts = fit_recipe(Xtr, ytr, horizon, estimators=estimators,
                                    lgbm_over=lgbm_over, class_weight=None,
                                    seeds=seeds)
        sel = parts["selection"]
        sel_idx = Xtr.index[sel]
        p3_sel = np.asarray(primary.predict_proba(Xtr.iloc[sel]), dtype=float)
        calls = V.decisive_classes(p3_sel, tau=primary_tau, delta=primary_delta)
        call_pos = np.flatnonzero(calls != 0)
        if len(call_pos) < 40:
            print(f"{s.test_start:%Y-%m}: too few selection calls "
                  f"({len(call_pos)}), fold skipped", flush=True)
            continue
        r_sel = T._path(candles, sel_idx, calls,
                        ds.loc[sel_idx, "atr_pct"].to_numpy(), horizon,
                        cost=config.ROUND_TRIP_COST,
                        barrier_scale=horizon ** 0.5)
        ok = call_pos[np.isfinite(r_sel[call_pos])]
        if len(ok) < 40:
            continue
        meta_cols = [c for c in META_FEATURES if c in Xtr.columns]
        meta_X = np.column_stack([Xtr.iloc[sel].iloc[ok][meta_cols].to_numpy(float),
                                  p3_sel[ok]])
        meta_y = (r_sel[ok] > 0).astype(int)
        meta = _fit_meta(meta_X, meta_y)
        meta_n.append(int(len(ok)))
        meta_aucs.append(float(roc_auc_score(meta_y, meta.predict_proba(meta_X)[:, 1])))
        p3 = np.asarray(primary.predict_proba(X.loc[te]), dtype=float)
        pred = V.decisive_classes(p3, tau=primary_tau, delta=primary_delta)
        te_atr = ds.loc[te, "atr_pct"].to_numpy()
        r = T._path(candles, te, pred, te_atr, horizon,
                    cost=config.ROUND_TRIP_COST,
                    barrier_scale=horizon ** 0.5)
        taken = (pred != 0) & np.isfinite(r)
        base_returns.extend(r[taken].tolist())
        meta_test = np.column_stack([X.loc[te][meta_cols].to_numpy(float), p3])
        mp = meta.predict_proba(meta_test)[:, 1]
        for q in QGRID:
            keep = (pred != 0) & (mp >= q) & np.isfinite(r)
            meta_returns[q].extend(r[keep].tolist())
        print(f"{s.test_start:%Y-%m}: base n={int(taken.sum())} "
              f"meta_train_n={len(ok)}", flush=True)
    def _sum(returns):
        rr = np.asarray(returns)
        if not len(rr):
            return {"signals": 0}
        return {"signals": int(len(rr)),
                "win_rate": float(np.mean(rr > 0)),
                "mean_return": float(np.mean(rr)),
                "profit_factor": float(V.profit_factor(rr)),
                "gates": V.gates_check(rr)}

    record = {"name": name, "mode": "metalabel", "ts": pd.Timestamp.now(tz="UTC").isoformat(),
              "horizon": horizon_name, "horizon_min": horizon, "features": features,
              "seeds": list(seeds), "estimators": estimators, "target": target,
              "primary_rule": {"tau": primary_tau, "delta": primary_delta},
              "meta_train_sizes": meta_n, "meta_train_auc_insample": meta_aucs,
              "baseline": _sum(base_returns),
              "grid": [{"q": q, **_sum(meta_returns[q])} for q in QGRID],
              "seconds": round(time.time() - started, 1)}
    if not smoke:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    b = record["baseline"]
    print(f"\n== {name} == base: n={b.get('signals', 0)} "
          f"win={b.get('win_rate', 0):.3f} pf={b.get('profit_factor', 0):.2f} "
          f"| meta_train_n={meta_n}", flush=True)
    for g in record["grid"]:
        if g["signals"]:
            print(f"  q={g['q']:.2f}: n={g['signals']:5d} win={g['win_rate']:.3f} "
                  f"mean={g['mean_return']:+.5f} pf={g['profit_factor']:.2f}", flush=True)
    return record


def run_factored(name: str, horizon_name: str, *, folds: int, deadband: float,
                 features: str, seeds, estimators: int, lgbm_over: dict,
                 class_weight=None, smoke: bool = False,
                 target: str = "forward") -> dict:
    """Factored ensemble: P(move) from triclass x P(up|move) from binary.

    p_flat = P_triclass(flat); p_long = (1-p_flat)*p_up; p_short = rest.
    Compared head-to-head against the direct triclass on the same folds.
    """
    horizon = config.HORIZONS_MIN[horizon_name]
    data = T.load_data()
    ds, candles = build_dataset(data, horizon, deadband, target=target)
    y = LAB.labels_to_classes(ds.label)
    cols = feature_columns(ds, features)
    X_all = F.prepare_model_input(ds[cols], cols)
    meaningful = ds.label.isin([-1.0, 1.0]).to_numpy()
    Xm = X_all.loc[meaningful]
    y_bin = (ds.label[meaningful] == 1.0).astype(int).to_numpy()
    splits = V.purged_walk_forward(X_all.index, horizon_min=horizon)
    evaluated = splits[:-1][:folds] if len(splits) > 1 else []

    fact_y, fact_p, dir_y, dir_p = [], [], [], []
    grids = {(tau, d): [] for tau in TAUS for d in DELTAS}
    started = time.time()
    for s in evaluated:
        tr = X_all.index[(X_all.index >= s.train_start) & (X_all.index < s.train_end)]
        te = X_all.index[(X_all.index >= s.test_start) & (X_all.index < s.test_end)]
        if len(tr) < 500 or len(te) < 100:
            continue
        ytr3 = y[X_all.index.get_indexer(tr)]
        tri, _ = fit_recipe(X_all.loc[tr], ytr3, horizon, estimators=estimators,
                            lgbm_over=lgbm_over, class_weight=None, seeds=seeds)
        mtr = Xm.index[(Xm.index >= s.train_start) & (Xm.index < s.train_end)]
        fitm, _ = fit_directional(Xm.loc[mtr], y_bin[Xm.index.get_indexer(mtr)],
                                  horizon, estimators=estimators,
                                  lgbm_over=lgbm_over, class_weight=None,
                                  seeds=seeds)
        p3 = np.asarray(tri.predict_proba(X_all.loc[te]), dtype=float)
        pu = np.asarray(fitm.predict_proba(X_all.loc[te]), dtype=float)[:, 1]
        pf_ = np.column_stack([p3[:, 0], (1 - p3[:, 0]) * pu,
                               (1 - p3[:, 0]) * (1 - pu)])
        yt = y[X_all.index.get_indexer(te)]
        fact_y.extend(yt.tolist()); fact_p.extend(pf_.tolist())
        dir_y.extend(yt.tolist()); dir_p.extend(p3.tolist())
        te_atr = ds.loc[te, "atr_pct"].to_numpy()
        for tau in TAUS:
            for d in DELTAS:
                pred = V.decisive_classes(pf_, tau=tau, delta=d)
                r = T._path(candles, te, pred, te_atr, horizon,
                            cost=config.ROUND_TRIP_COST)
                sel = (pred != 0) & np.isfinite(r)
                grids[(tau, d)].extend(r[sel].tolist())
        print(f"{s.test_start:%Y-%m}: factored fold done", flush=True)
    pooled_f = V.probability_metrics(np.asarray(fact_y), np.asarray(fact_p))
    pooled_d = V.probability_metrics(np.asarray(dir_y), np.asarray(dir_p))
    grid = [{"tau": t, "delta": d, "signals": int(len(grids[(t, d)])),
             "win_rate": float(np.mean(np.asarray(grids[(t, d)]) > 0)) if len(grids[(t, d)]) else None,
             "mean_return": float(np.mean(grids[(t, d)])) if len(grids[(t, d)]) else None,
             "profit_factor": float(V.profit_factor(grids[(t, d)])) if len(grids[(t, d)]) else None}
            for t in TAUS for d in DELTAS]
    record = {"name": name, "mode": "factored", "ts": pd.Timestamp.now(tz="UTC").isoformat(),
              "horizon": horizon_name, "horizon_min": horizon, "features": features,
              "seeds": list(seeds), "estimators": estimators, "target": target,
              "pooled": pooled_f, "direct_triclass": pooled_d, "grid": grid,
              "seconds": round(time.time() - started, 1)}
    if not smoke:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    print(f"\n== {name} == factored ll={pooled_f['log_loss']:.4f} "
          f"bal={pooled_f['balanced_accuracy']:.4f} | direct ll="
          f"{pooled_d['log_loss']:.4f} bal={pooled_d['balanced_accuracy']:.4f}",
          flush=True)
    for g in grid:
        if g["signals"]:
            print(f"  tau={g['tau']:.2f} d={g['delta']:.2f}: n={g['signals']:5d} "
                  f"win={g['win_rate']:.3f} mean={g['mean_return']:+.5f} "
                  f"pf={g['profit_factor']:.2f}", flush=True)
    return record


def run_regression(name: str, horizon_name: str, *, folds: int, deadband: float,
                   features: str, seeds, estimators: int, lgbm_over: dict,
                   class_weight=None, smoke: bool = False,
                   target: str = "forward") -> dict:
    """Expected-move regression: trade when predicted move clears cost.

    LightGBM regressor on the continuous forward return; direction = sign of
    the prediction; gate = |prediction| >= k x round-trip cost, optionally
    vetoed by the companion triclass chop detector (P(move) >= 0.45).
    """
    import lightgbm as lgb

    horizon = config.HORIZONS_MIN[horizon_name]
    data = T.load_data()
    ds, candles = build_dataset(data, horizon, deadband, target="forward")
    close = ds["close"]
    fwd = (close.shift(-horizon) / close - 1).to_numpy()
    y = LAB.labels_to_classes(ds.label)
    cols = feature_columns(ds, features)
    X = F.prepare_model_input(ds[cols], cols)
    splits = V.purged_walk_forward(X.index, horizon_min=horizon)
    evaluated = splits[:-1][:folds] if len(splits) > 1 else []

    KGRID = (1.0, 1.5, 2.0, 2.5)
    grids = {(k, veto): [] for k in KGRID for veto in (False, True)}
    ic_list, all_pred, all_true = [], [], []
    started = time.time()
    for s in evaluated:
        tr = X.index[(X.index >= s.train_start) & (X.index < s.train_end)]
        te = X.index[(X.index >= s.test_start) & (X.index < s.test_end)]
        if len(tr) < 500 or len(te) < 100:
            continue
        pos_tr = X.index.get_indexer(tr)
        y_reg = fwd[pos_tr]
        # The last `horizon` training rows have unknown future: purge them.
        parts = V.temporal_partitions(tr, horizon)
        fit, stop = parts["fit"], parts["early_stop"]
        weights = V.recency_weights(tr[fit])

        def make_reg(seed):
            params = {k: v for k, v in config.LGBM_PARAMS.items()
                      if k not in ("num_class", "objective")}
            params["objective"] = "regression"
            params.update(lgbm_over or {})
            params.update(n_jobs=4, random_state=int(seed), deterministic=True,
                          force_col_wise=True)
            params["n_estimators"] = int(estimators)
            return lgb.LGBMRegressor(**params)

        regs = []
        for seed in seeds:
            r = make_reg(seed)
            r.fit(X.iloc[pos_tr[fit]], y_reg[fit], sample_weight=weights,
                  eval_set=[(X.iloc[pos_tr[stop]], y_reg[stop])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            regs.append(r)
        pred_move = np.mean([r.predict(X.loc[te]) for r in regs], axis=0)
        tri, _ = fit_recipe(X.loc[tr], y[pos_tr], horizon, estimators=estimators,
                            lgbm_over=lgbm_over, class_weight=None, seeds=seeds)
        p3 = np.asarray(tri.predict_proba(X.loc[te]), dtype=float)
        p_move = p3[:, 1] + p3[:, 2]
        pos_te = X.index.get_indexer(te)
        true_move = fwd[pos_te]
        ok = np.isfinite(true_move) & np.isfinite(pred_move)
        ic_list.append(float(pd.Series(pred_move[ok]).corr(
            pd.Series(true_move[ok]), method="spearman")))
        all_pred.extend(pred_move[ok].tolist())
        all_true.extend(true_move[ok].tolist())
        te_atr = ds.loc[te, "atr_pct"].to_numpy()
        for k in KGRID:
            for veto in (False, True):
                pred = np.zeros(len(te), dtype=int)
                pred[pred_move >= k * config.ROUND_TRIP_COST] = 1
                pred[pred_move <= -k * config.ROUND_TRIP_COST] = 2
                if veto:
                    pred = np.where(p_move >= 0.45, pred, 0)
                r = T._path(candles, te, pred, te_atr, horizon,
                            cost=config.ROUND_TRIP_COST)
                sel = (pred != 0) & np.isfinite(r)
                grids[(k, veto)].extend(r[sel].tolist())
        print(f"{s.test_start:%Y-%m}: ic={ic_list[-1]:.4f}", flush=True)

    pooled_ic = float(pd.Series(all_pred).corr(pd.Series(all_true), method="spearman"))
    grid = [{"k": k, "veto": v, "signals": int(len(grids[(k, v)])),
             "win_rate": float(np.mean(np.asarray(grids[(k, v)]) > 0)) if len(grids[(k, v)]) else None,
             "mean_return": float(np.mean(grids[(k, v)])) if len(grids[(k, v)]) else None,
             "profit_factor": float(V.profit_factor(grids[(k, v)])) if len(grids[(k, v)]) else None}
            for k in KGRID for v in (False, True)]
    record = {"name": name, "mode": "regression", "ts": pd.Timestamp.now(tz="UTC").isoformat(),
              "horizon": horizon_name, "horizon_min": horizon, "features": features,
              "seeds": list(seeds), "estimators": estimators, "target": "forward",
              "pooled": {"spearman_ic": pooled_ic, "fold_ics": ic_list},
              "grid": grid, "seconds": round(time.time() - started, 1)}
    if not smoke:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    print(f"\n== {name} == spearman_ic={pooled_ic:.4f}", flush=True)
    for g in grid:
        if g["signals"]:
            print(f"  k={g['k']} veto={int(g['veto'])}: n={g['signals']:5d} "
                  f"win={g['win_rate']:.3f} mean={g['mean_return']:+.5f} "
                  f"pf={g['profit_factor']:.2f}", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--horizon", choices=list(config.HORIZONS_MIN), default="mid")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--deadband", type=float, default=config.ROUND_TRIP_COST)
    ap.add_argument("--features", choices=("v2", "v3", "v4", "v5"), default="v2")
    ap.add_argument("--class-weight", default=None)
    ap.add_argument("--seeds", default="1",
                    help="1, 3, or comma list like 42,7,99")
    ap.add_argument("--estimators", type=int, default=1200)
    ap.add_argument("--lgbm-json", default="{}")
    ap.add_argument("--mode", choices=("triclass", "directional", "metalabel",
                                       "factored", "regression"),
                    default="triclass")
    ap.add_argument("--target", choices=("forward", "barrier"), default="forward")
    ap.add_argument("--primary-tau", type=float, default=0.45)
    ap.add_argument("--primary-delta", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--half-life", type=float, default=None,
                    help="recency sample-weight half-life in days")
    ap.add_argument("--placebo", action="store_true",
                    help="shuffle labels within months: noise-floor control")
    a = ap.parse_args()
    cw = None
    if a.class_weight == "balanced":
        cw = "balanced"
    elif a.class_weight:
        cw = json.loads(a.class_weight)
    if a.seeds == "1":
        seeds = [42]
    elif str(a.seeds).isdigit():
        seeds = [42, 137, 2718][:int(a.seeds)]
    else:
        seeds = [int(s) for s in str(a.seeds).split(",")]
    runners = {"triclass": run_experiment, "directional": run_directional,
               "metalabel": run_metalabel, "factored": run_factored,
               "regression": run_regression}
    runner = runners[a.mode]
    kwargs = {}
    if a.mode == "metalabel":
        kwargs = {"primary_tau": a.primary_tau, "primary_delta": a.primary_delta}
    if a.placebo and a.mode != "triclass":
        raise SystemExit("--placebo is only supported for --mode triclass")
    if a.mode == "triclass":
        kwargs["placebo"] = a.placebo
        if a.half_life is not None:
            kwargs["half_life"] = a.half_life
    runner(a.name, a.horizon, folds=a.folds, deadband=a.deadband,
           features=a.features, class_weight=cw, seeds=seeds,
           estimators=a.estimators, lgbm_over=json.loads(a.lgbm_json),
           smoke=a.smoke, target=a.target, **kwargs)


if __name__ == "__main__":
    main()

