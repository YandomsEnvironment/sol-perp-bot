"""Unit tests for labeling.py — crafted price paths with KNOWN outcomes (T1).

If these pass, the triple-barrier math is correct; if the bot ever loses on a
path these tests guarantee, the bug is elsewhere.
"""
import numpy as np
import pandas as pd

from labeling import labels_to_classes, triple_barrier_labels


def _mk(high, low, close):
    n = len(high)
    ts = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts.asi8 // 10**6,
        "open": close, "high": high, "low": low,
        "close": close, "volume": np.ones(n),
    })


def test_rising_path_is_long():
    # entry 100, ATR 1 -> tp 101 / sl 99. Future highs 105 clear tp immediately
    n, h = 12, 6
    high = np.full(n, 105.0)
    low = np.full(n, 99.5)          # never touches sl
    close = np.full(n, 100.0)
    close[h:] = 105.0
    df = _mk(high, low, close)
    atr = pd.Series(np.full(n, 1.0))
    labels = triple_barrier_labels(df, atr, horizon=h)
    assert labels.iloc[0] == 1.0


def test_falling_path_is_short():
    n, h = 12, 6
    high = np.full(n, 100.5)        # never touches tp
    low = np.full(n, 95.0)          # clears sl
    close = np.full(n, 100.0)
    close[h:] = 95.0
    df = _mk(high, low, close)
    atr = pd.Series(np.full(n, 1.0))
    labels = triple_barrier_labels(df, atr, horizon=h)
    assert labels.iloc[0] == -1.0


def test_chop_is_flat():
    n, h = 12, 6
    high = np.full(n, 100.5)        # neither barrier ever touched
    low = np.full(n, 99.5)
    close = np.full(n, 100.0)
    df = _mk(high, low, close)
    atr = pd.Series(np.full(n, 1.0))
    labels = triple_barrier_labels(df, atr, horizon=h)
    assert labels.iloc[0] == 0.0


def test_stop_beats_profit_when_hit_first():
    # both barriers reachable, but low dips below sl BEFORE high clears tp
    n, h = 12, 6
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    high[3] = 105.0                 # tp touched at t+3
    low[1] = 95.0                   # sl touched at t+1 -> short wins
    close = np.full(n, 100.0)
    df = _mk(high, low, close)
    atr = pd.Series(np.full(n, 1.0))
    labels = triple_barrier_labels(df, atr, horizon=h)
    assert labels.iloc[0] == -1.0


def test_tail_is_nan():
    n, h = 10, 4
    df = _mk(np.full(n, 105.0), np.full(n, 95.0), np.full(n, 100.0))
    atr = pd.Series(np.full(n, 1.0))
    labels = triple_barrier_labels(df, atr, horizon=h)
    assert labels.iloc[: n - h].notna().all()
    assert labels.iloc[n - h :].isna().all()


def test_classes_mapping():
    s = pd.Series([1.0, 0.0, -1.0, np.nan])
    classes = labels_to_classes(s)
    assert classes.tolist() == [1, 0, 2, -1]
