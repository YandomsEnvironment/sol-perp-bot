import numpy as np
import pandas as pd

import train


def test_path_uses_full_candles_and_forward_fixed_horizon(monkeypatch):
    idx = pd.date_range("2026-01-01", periods=10, freq="min", tz="UTC")
    candles = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.}, index=idx)
    seen = {}

    def fake_path(frame, pred, horizon, atr_pct=None, cost=None, barrier_scale=1.0):
        seen.update(length=len(frame), atr=atr_pct, pred=np.asarray(pred).copy())
        return np.arange(len(frame), dtype=float)

    monkeypatch.setattr(train.V, "path_returns", fake_path)
    got = train._path(candles, idx[[2, 4]], np.array([1, 2]), np.array([.01, .02]), 3,
                      target="forward")
    assert seen["length"] == len(candles)
    assert seen["atr"] is None
    assert seen["pred"][2] == 1 and seen["pred"][4] == 2
    np.testing.assert_array_equal(got, [2., 4.])


def test_threshold_replays_only_accepted_signals(monkeypatch):
    idx = pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC")
    candles = pd.DataFrame(index=idx)
    replayed = []

    def fake_path(candles, index, pred, atr, horizon, cost=None, target="barrier"):
        replayed.append(np.asarray(pred).copy())
        return np.ones(len(pred))

    monkeypatch.setattr(train, "_path", fake_path)
    p = np.array([[.1, .8, .1], [.2, .4, .4]])
    train._threshold_diagnostics(np.array([1, 2]), p, candles, idx[:2], np.ones(2), 1,
                                 [.5, .7], target="forward")
    np.testing.assert_array_equal(replayed[0], [1, 0])
    np.testing.assert_array_equal(replayed[1], [1, 0])