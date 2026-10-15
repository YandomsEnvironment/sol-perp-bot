"""Overfitting defenses, centralized (plan M1-M3, M7, R2, §9).

Nothing here learns. It only enforces HONEST MEASUREMENT, so any number
train.py reports is one you can trust with money:
  - purged & embargoed walk-forward splits (no future can leak into training)
  - recency-weighted samples (old regimes don't outvote the current one)
  - fee-aware trade simulation (no paper profits that ignore costs)
  - calibration report (claimed confidence vs realized win rate)
  - promotion gates (R5) and the dumb-rule baseline (M7)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import config

CLASS_FLAT, CLASS_LONG, CLASS_SHORT = 0, 1, 2


def temporal_partitions(index: pd.DatetimeIndex, horizon_min: int) -> dict:
    """Four disjoint chronological roles, purged at every internal boundary."""
    index = pd.DatetimeIndex(index)
    if len(index) < 100 or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Need at least 100 unique chronological observations")
    cuts = [0, int(len(index) * .60), int(len(index) * .75),
            int(len(index) * .875), len(index)]
    gap = pd.Timedelta(minutes=horizon_min + 1)
    out = {}
    for k, name in enumerate(("fit", "early_stop", "calibration", "selection")):
        positions = np.arange(cuts[k], cuts[k + 1])
        if k < 3:
            positions = positions[index[positions] < index[cuts[k + 1]] - gap]
        if len(positions) < 10:
            raise ValueError("Insufficient purged observations for " + name)
        out[name] = positions
    return out


def probability_metrics(y, proba) -> dict:
    """All-prediction metrics; Brier is sum of squared class errors per row."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
    y = np.asarray(y, dtype=int)
    p = np.asarray(proba, dtype=float)
    if p.shape != (len(y), 3) or not len(y) or not np.isfinite(p).all():
        raise ValueError("Expected finite N by 3 probabilities and nonempty labels")
    if (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Invalid probability simplex")
    return {"accuracy": float(accuracy_score(y, p.argmax(axis=1))),
            "balanced_accuracy": float(balanced_accuracy_score(y, p.argmax(axis=1))),
            "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
            "brier": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))}


def path_returns(candles, pred_classes, horizon, atr_pct=None,
                 cost=config.ROUND_TRIP_COST, tp=config.TP_ATR,
                 sl=config.SL_ATR, barrier_scale: float = 1.0):
    """One position at a time, next-open entry, conservative OHLC exits.

    Fractions of unlevered notional, not account equity. Maker cost scenarios
    are sensitivity analysis, not proof of fills. Flat/overlap/unmatured rows
    return NaN. No intrabar price ordering is invented: stop wins ties.

    barrier_scale multiplies the ATR distances: config's documented design is
    "TP/SL = mult x ATR(horizon)", so callers simulating an H-minute horizon
    should pass sqrt(H) (Brownian scaling) -- the raw 1m ATR alone makes the
    take-profit smaller than the fee at longer horizons.
    """
    if horizon < 1 or cost < 0:
        raise ValueError("Positive horizon and nonnegative cost required")
    pred = np.asarray(pred_classes)
    if len(candles) != len(pred):
        raise ValueError("Predictions must align with candles")
    a = None if atr_pct is None else np.asarray(atr_pct, dtype=float)
    if a is not None and len(a) != len(pred):
        raise ValueError("ATR must align with candles")
    prices = candles[["open", "high", "low", "close"]].to_numpy(dtype=float)
    out = np.full(len(pred), np.nan)
    next_free = 0
    index = candles.index
    for i in range(len(pred) - horizon):
        if pred[i] not in (1, 2) or i < next_free:
            continue
        end = i + horizon
        if isinstance(index, pd.DatetimeIndex):
            if index[end] - index[i] != pd.Timedelta(minutes=horizon):
                continue
        window = prices[i + 1:end + 1]
        if not np.isfinite(window).all() or (window <= 0).any():
            continue
        entry = window[0, 0]
        direction = 1 if pred[i] == 1 else -1
        exit_price, exit_row = window[-1, 3], end
        if a is not None:
            if not np.isfinite(a[i]) or a[i] <= 0:
                continue
            target = entry * (1 + direction * tp * a[i] * barrier_scale)
            stop = entry * (1 - direction * sl * a[i] * barrier_scale)
            for j, (op, high, low, close) in enumerate(window, i + 1):
                if (direction == 1 and op <= stop) or (direction == -1 and op >= stop):
                    exit_price, exit_row = op, j
                    break
                if (direction == 1 and low <= stop) or (direction == -1 and high >= stop):
                    exit_price, exit_row = stop, j
                    break
                if (direction == 1 and high >= target) or (direction == -1 and low <= target):
                    exit_price, exit_row = target, j
                    break
        out[i] = direction * (exit_price / entry - 1) - cost
        next_free = exit_row
    return out


# ----------------------------------------------------------------------------
# Walk-forward splitting with purge + embargo (M1)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    """[train_start, train_end) and [test_start, test_end), all tz-aware UTC."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_month_splits(
    index: pd.DatetimeIndex,
    train_months: int = config.TRAIN_MONTHS,
    step_months: int = 1,
) -> list[Split]:
    """Rolling-origin folds: train on N consecutive months, test the next one.

    Raw splits (before purge): train_end == test_start (exclusive boundary).
    """
    periods = pd.PeriodIndex(index, freq="M").unique().sort_values()
    splits: list[Split] = []
    i = 0
    while i + train_months < len(periods):
        tr = periods[i : i + train_months]
        te = periods[i + train_months]
        splits.append(Split(
            train_start=tr[0].start_time.tz_localize("UTC"),
            train_end=te.start_time.tz_localize("UTC"),
            test_start=te.start_time.tz_localize("UTC"),
            test_end=(te + 1).start_time.tz_localize("UTC"),
        ))
        i += step_months
    return splits


def apply_purge_embargo(
    splits: Sequence[Split],
    horizon_min: int,
    embargo_min: int = config.EMBARGO_MIN,
) -> list[Split]:
    """Shrink splits so no information can cross the train/test boundary.

    PURGE (applied always): training labels look `horizon_min` into the future,
    so the last `horizon_min` of training is removed -- otherwise those labels
    literally contain the first minutes of the test period (the #1 silent
    backtest bug).

    EMBARGO (not applied in walk-forward): in strict walk-forward, training
    data always PRECEDES its own test month, so post-test leakage cannot
    occur; a later fold reusing an earlier fold's test month as training data
    is correct and mirrors live retraining. Embargo becomes necessary only
    for combinatorial/purged K-fold CV, where train segments can follow test
    segments in time. Kept as a parameter for that future use.
    """
    td_h = pd.Timedelta(minutes=horizon_min)
    out: list[Split] = []
    for s in splits:
        out.append(replace(s, train_end=s.test_start - td_h))
    return out


def purged_walk_forward(
    index: pd.DatetimeIndex,
    horizon_min: int,
    train_months: int = config.TRAIN_MONTHS,
) -> list[Split]:
    """One-call helper: raw rolling splits + purge/embargo."""
    return apply_purge_embargo(
        rolling_month_splits(index, train_months=train_months),
        horizon_min=horizon_min,
    )


# ----------------------------------------------------------------------------
# Sample weighting (M2): recency half-life
# ----------------------------------------------------------------------------

def recency_weights(
    train_index: pd.DatetimeIndex,
    half_life_days: float = config.SAMPLE_HALF_LIFE_DAYS,
) -> np.ndarray:
    """Exponential decay: a sample half_life_days old counts 0.5x."""
    t = pd.DatetimeIndex(train_index)
    age_days = np.asarray((t[-1] - t).total_seconds()) / 86400.0
    return 0.5 ** (age_days / float(half_life_days))


# ----------------------------------------------------------------------------
# Fee-aware trade simulation + metrics (R1/R5)
# ----------------------------------------------------------------------------

def simulate_trades(
    pred_classes: np.ndarray,
    y_true: np.ndarray,
    atr_pct: np.ndarray,
    cost: float = config.ROUND_TRIP_COST,
    tp: float = config.TP_ATR,
    sl: float = config.SL_ATR,
) -> np.ndarray:
    """Realistic per-trade returns for taken signals (class != flat).

    pred == truth -> +tp*atr_pct - cost (barrier win)
    truth == flat -> -cost (chop: time-exit, still pay costs)
    pred != truth -> -sl*atr_pct - cost (stopped out)
    Not taken (pred flat) or unknown truth -> NaN.
    """
    pred = np.asarray(pred_classes)
    y = np.asarray(y_true)
    atr_pct = np.asarray(atr_pct)
    r = np.full(len(pred), np.nan, dtype=np.float64)
    taken = np.flatnonzero((pred != CLASS_FLAT) & np.isfinite(y))
    for i in taken:
        if pred[i] == y[i]:
            r[i] = tp * atr_pct[i] - cost
        elif y[i] == CLASS_FLAT:
            r[i] = -cost
        else:
            r[i] = -sl * atr_pct[i] - cost
    return r


def profit_factor(returns: Iterable[float]) -> float:
    r = pd.Series(list(returns)).dropna()
    if len(r) == 0:
        return 0.0
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def max_drawdown(returns: Iterable[float]) -> float:
    r = pd.Series(list(returns)).dropna()
    if len(r) == 0:
        return 0.0
    # Include initial capital, otherwise a first-trade loss disappears.
    equity = np.r_[1., np.cumprod(1. + r.to_numpy(dtype=float))]
    peak = np.maximum.accumulate(equity)
    return float(np.max(1. - equity / peak))


def hit_rate(pred_classes: np.ndarray, y_true: np.ndarray) -> tuple[float, int]:
    """Accuracy over taken trades only, plus how many were taken."""
    pred = np.asarray(pred_classes)
    y = np.asarray(y_true)
    taken = (pred != CLASS_FLAT) & np.isfinite(y)
    n = int(taken.sum())
    if n == 0:
        return 0.0, 0
    return float((pred[taken] == y[taken]).mean()), n


def gates_check(
    returns: Iterable[float],
    pf_min: float = 1.3,
    dd_max: float = 0.15,
    min_trades: int = 30,
) -> dict:
    """R5 backtest gate: all numeric, no vibes."""
    r = pd.Series(list(returns)).dropna()
    pf = profit_factor(r)
    dd = max_drawdown(r)
    ok = len(r) >= min_trades and pf > pf_min and dd < dd_max
    return {"n_trades": int(len(r)), "profit_factor": pf,
            "max_drawdown": dd, "gate_pass": bool(ok)}


# ----------------------------------------------------------------------------
# Calibration (R2): claimed confidence vs realized win rate
# ----------------------------------------------------------------------------

def calibration_report(
    claimed_prob: Sequence[float],
    won: Sequence[bool],
    bins: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 1.01),
) -> pd.DataFrame:
    """If the model is calibrated, 'actual_win_rate' ≈ 'claimed' per bucket.
    A model claiming 80% and winning 55% is overfitting made visible."""
    df = pd.DataFrame({"claimed": pd.to_numeric(pd.Series(claimed_prob)),
                       "won": pd.Series(won).astype(bool)})
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (df["claimed"] >= lo) & (df["claimed"] < hi)
        if int(m.sum()) > 0:
            rows.append({
                "bucket": f"{lo:.0%}-{hi:.0%}",
                "n": int(m.sum()),
                "claimed": float(df.loc[m, "claimed"].mean()),
                "actual_win_rate": float(df.loc[m, "won"].mean()),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Lightweight multiclass probability calibration (R2)
# ----------------------------------------------------------------------------

class CalibratedModel:
    """Wraps a fitted classifier with per-class isotonic calibration.

    Fits one IsotonicRegression per class on VALIDATION predictions
    Calibration is an estimate, not a guarantee of future reliability.
    Cheaper and simpler than sklearn's CalibratedClassifierCV (which
    cross-fits and multiplies model copies in the pickle).
    """

    def __init__(self, model, calibrators: dict):
        self.model = model                    # underlying fitted classifier
        self.calibrators = calibrators        # {class_idx: IsotonicRegression}

    def predict_proba(self, X) -> np.ndarray:
        p = np.asarray(self.model.predict_proba(X), dtype=np.float64)
        out = np.zeros_like(p)
        for k, ir in self.calibrators.items():
            out[:, k] = ir.predict(p[:, k])
        tot = out.sum(axis=1, keepdims=True)
        empty = tot[:, 0] <= 0
        out[empty] = p[empty]
        tot = out.sum(axis=1, keepdims=True)
        tot[tot <= 0] = 1.0
        return out / tot


def fit_isotonic_multiclass(model, X_val, y_val) -> CalibratedModel:
    """Calibrate `model`'s probabilities on a validation slice."""
    from sklearn.isotonic import IsotonicRegression
    p = np.asarray(model.predict_proba(X_val), dtype=np.float64)
    cals = {}
    for k in range(p.shape[1]):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(p[:, k], (np.asarray(y_val) == k).astype(float))
        cals[k] = ir
    return CalibratedModel(model, cals)


# ----------------------------------------------------------------------------
# Dumb-rule baseline (M7): the model must beat this after costs
# ----------------------------------------------------------------------------

class SeedBag:
    """Averaged ensemble of identically-configured classifiers over seeds.

    Variance reduction for probability outputs; classes stay {0,1,2}.
    Defined here (validation is imported by the dashboard) so bundles unpickle.
    """

    classes_ = np.array([CLASS_FLAT, CLASS_LONG, CLASS_SHORT])

    def __init__(self, models: list):
        if not models:
            raise ValueError("SeedBag needs at least one model")
        self.models = list(models)

    def predict_proba(self, X) -> np.ndarray:
        p = np.mean([np.asarray(m.predict_proba(X), dtype=np.float64)
                     for m in self.models], axis=0)
        return p / p.sum(axis=1, keepdims=True)


def fit_seed_bag(make_model_fn, X_tr, y_tr, seeds=(42, 137, 2718), **fit_kwargs):
    """Fit one classifier per seed and average. make_model_fn(seed) -> model."""
    models = []
    for seed in seeds:
        m = make_model_fn(seed)
        m.fit(X_tr, y_tr, **fit_kwargs)
        models.append(m)
    return SeedBag(models)


class CalibratedBinary:
    """Binary wrapper: classifier + isotonic calibration of P(up)."""

    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        p = np.asarray(self.model.predict_proba(X), dtype=np.float64)[:, 1]
        c = np.clip(self.calibrator.predict(p), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - c, c])


def decisive_classes(proba, tau: float = 0.50, delta: float = 0.05) -> np.ndarray:
    """Directional call only when the LONG/SHORT margin is decisive.

    tau:   minimum probability of the favored side (self-confidence)
    delta: minimum |p_long - p_short| margin (edge over the other side)
    Anything else is FLAT -- indecision is a first-class output.
    """
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("Expected an N by 3 probability array")
    long, short = p[:, CLASS_LONG], p[:, CLASS_SHORT]
    fav = np.maximum(long, short)
    take = (fav >= tau) & (np.abs(long - short) >= delta)
    out = np.full(len(p), CLASS_FLAT, dtype=np.int64)
    out[take & (long >= short)] = CLASS_LONG
    out[take & (short > long)] = CLASS_SHORT
    return out


def momentum_baseline_classes(
    feats: pd.DataFrame,
    col: str = "ret_15m",
    window: int = 5000,
    quantile: float = 0.8,
) -> np.ndarray:
    """Trade with 15m momentum only when |return| is in its top 20%.
    If LightGBM can't beat this after fees, the features have no signal."""
    s = feats[col]
    thr = s.abs().rolling(window, min_periods=200).quantile(quantile)
    pred = np.where(s > thr, CLASS_LONG, np.where(s < -thr, CLASS_SHORT, CLASS_FLAT))
    return pred.astype(np.int64)
