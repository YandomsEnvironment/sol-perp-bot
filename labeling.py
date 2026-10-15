"""Triple-barrier labels (plan §2 / M5).

At each bar we ask: within `horizon` minutes, does price first touch
  +tp_mult * ATR  (long wins, label +1)
  -sl_mult * ATR  (short wins, label -1)
  neither          (chop, label 0)

ATR scaling makes labels regime-adaptive and encodes "is this move bigger
than costs?". Rows too close to the end of the data get NaN (future unknown).
Classes for LightGBM: 0 = flat, 1 = long, 2 = short (see labels_to_classes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    candles: pd.DataFrame,
    atr: pd.Series,
    horizon: int,
    tp_mult: float = 1.0,
    sl_mult: float = 1.0,
) -> pd.Series:
    """Return float labels (+1/0/-1) aligned to candles.index, NaN at the tail."""
    close = candles["close"].to_numpy(dtype=np.float64)
    high = candles["high"].to_numpy(dtype=np.float64)
    low = candles["low"].to_numpy(dtype=np.float64)
    atr_np = atr.to_numpy(dtype=np.float64)
    n = len(candles)
    if horizon < 1 or len(atr_np) != n or tp_mult <= 0 or sl_mult <= 0:
        raise ValueError("Positive horizon/barriers and aligned ATR required")

    labels = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - horizon):
        a = atr_np[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = close[i]
        tp = entry + tp_mult * a
        sl = entry - sl_mult * a
        hi = high[i + 1 : i + 1 + horizon]
        lo = low[i + 1 : i + 1 + horizon]
        if not np.isfinite(entry) or not np.isfinite(hi).all() or not np.isfinite(lo).all():
            continue
        if isinstance(candles.index, pd.DatetimeIndex):
            if candles.index[i + horizon] - candles.index[i] != pd.Timedelta(minutes=horizon):
                continue

        up_first = np.flatnonzero(hi >= tp)
        down_first = np.flatnonzero(lo <= sl)
        up = int(up_first[0]) if up_first.size else -1
        down = int(down_first[0]) if down_first.size else -1

        if up == -1 and down == -1:
            labels[i] = 0.0        # neither barrier hit: chop
        elif down == -1:
            labels[i] = 1.0        # only up barrier
        elif up == -1:
            labels[i] = -1.0       # only down barrier
        elif up < down:
            labels[i] = 1.0
        elif down < up:
            labels[i] = -1.0
        else:
            labels[i] = 0.0        # both touched in the same bar: ambiguous

    return pd.Series(labels, index=candles.index, name="label")


def labels_to_classes(labels: pd.Series) -> np.ndarray:
    """Map {-1, 0, +1} -> {0 flat, 1 long, 2 short}. NaN rows become -1 (drop)."""
    out = np.full(len(labels), -1, dtype=np.int64)
    arr = labels.to_numpy()
    valid = np.isfinite(arr)
    out[valid & (arr == 0)] = 0
    out[valid & (arr == 1)] = 1
    out[valid & (arr == -1)] = 2
    return out


def forward_return_labels(candles: pd.DataFrame, horizon: int,
                          deadband: float) -> pd.Series:
    """Symmetric endpoint target at a declared horizon, not barrier frequency.

    deadband is a fractional price move, e.g. round-trip cost. It is not a
    profitability guarantee: executable entries are evaluated separately.
    """
    if horizon < 1 or deadband < 0:
        raise ValueError("Positive horizon and nonnegative deadband required")
    close = candles["close"].astype(float)
    move = close.shift(-horizon) / close - 1
    out = pd.Series(np.nan, index=candles.index, name="label")
    ok = np.isfinite(move) & (close > 0)
    out.loc[ok] = np.where(move[ok] > deadband, 1.,
                           np.where(move[ok] < -deadband, -1., 0.))
    if isinstance(candles.index, pd.DatetimeIndex):
        times = candles.index.to_series()
        contiguous = times.shift(-horizon) - times == pd.Timedelta(minutes=horizon)
        out.loc[~contiguous] = np.nan
    return out
