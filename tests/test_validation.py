"""Tests for validation.py — the overfitting defenses themselves must be
provably leak-free (purge/embargo math) and metric-correct (known numbers).
"""
import numpy as np
import pandas as pd
import pytest

import validation as v


def _index(months: int = 14, freq: str = "15min") -> pd.DatetimeIndex:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    return pd.date_range(start, periods=months * 4 * 24 * 30, freq=freq)


# --- walk-forward splitting ---

def test_splits_rolling_and_ordered():
    splits = v.rolling_month_splits(_index(14), train_months=9)
    assert len(splits) == 5                      # 14 months - 9 = 5 test months
    for s in splits:
        assert s.train_start < s.train_end <= s.test_start < s.test_end
    tests = [s.test_start for s in splits]
    assert tests == sorted(tests)
    # each test month appears exactly once
    assert len({s.test_start.month for s in splits}) == len(splits)


def test_purge_pulls_train_end_back_by_horizon():
    splits = v.apply_purge_embargo(v.rolling_month_splits(_index(14), 9),
                                   horizon_min=60, embargo_min=60)
    for s in splits:
        gap = s.test_start - s.train_end
        assert gap == pd.Timedelta(minutes=60)


def test_train_strictly_precedes_its_own_test():
    # In walk-forward, train data must always be in the past relative to the
    # fold's own test month (rolling windows MAY include earlier folds' test
    # months as training data -- that mirrors live retraining and is correct).
    splits = v.purged_walk_forward(_index(14), horizon_min=5)
    for s in splits:
        assert s.train_end <= s.test_start


def test_no_train_label_can_touch_test_period():
    horizon = 60
    splits = v.purged_walk_forward(_index(14), horizon_min=horizon)
    idx = _index(14)
    for s in splits:
        train_rows = idx[(idx >= s.train_start) & (idx < s.train_end)]
        assert len(train_rows) > 0
        # latest possible label window end is train_time + horizon
        assert train_rows[-1] + pd.Timedelta(minutes=horizon) <= s.test_start


# --- sample weights ---

def test_recency_weights_monotonic_and_halflife():
    idx = pd.date_range("2025-01-01", periods=1000, freq="1h", tz="UTC")
    w = v.recency_weights(idx, half_life_days=10.0)
    assert (np.diff(w) >= 0).all()               # older -> smaller
    assert w[-1] == pytest.approx(1.0)
    # a sample exactly 10 days old should weigh 0.5
    t = idx[-1] - pd.Timedelta(days=10)
    pos = idx.get_indexer([t], method="nearest")[0]
    assert w[pos] == pytest.approx(0.5, rel=1e-3)


# --- trade simulation and metrics ---

def test_simulate_trades_known_outcomes():
    pred = np.array([1, 1, 2, 0, 1])             # long, long, short, flat, long
    y =    np.array([1, 2, 0, 1, 1])             # win, loss, chop, -, win
    atr =  np.full(5, 0.01)
    # pin symmetric barriers here so the test checks simulate_trades' own
    # contract regardless of config default asymmetry (tp=2/sl=1 in prod)
    r = v.simulate_trades(pred, y, atr, cost=0.001, tp=1.0, sl=1.0)
    assert r[0] == pytest.approx(0.01 - 0.001)   # win: +1 ATR - cost
    assert r[1] == pytest.approx(-0.01 - 0.001)  # loss: -1 ATR - cost
    assert r[2] == pytest.approx(-0.001)         # chop: time exit, pay cost
    assert np.isnan(r[3])                        # not taken
    assert r[4] == pytest.approx(0.01 - 0.001)
    # and with the asymmetric default geometry (tp=2, sl=1 from config)
    r2 = v.simulate_trades(pred, y, atr, cost=0.001)
    assert r2[0] == pytest.approx(2 * 0.01 - 0.001)
    assert r2[1] == pytest.approx(-0.01 - 0.001)


def test_profit_factor_and_drawdown_math():
    r = [0.01, -0.005, 0.01]
    assert v.profit_factor(r) == pytest.approx(4.0)
    assert v.max_drawdown(r) == pytest.approx(0.005)   # dip after first win
    assert v.max_drawdown([0.01, 0.02]) == 0.0
    assert v.profit_factor([]) == 0.0


def test_gates_check_numeric():
    good = [0.01] * 40 + [-0.004] * 30
    res = v.gates_check(good)
    assert res["n_trades"] == 70
    assert res["gate_pass"] is True
    bad = [0.001] * 40 + [-0.004] * 30           # PF < 1 after costs
    assert v.gates_check(bad)["gate_pass"] is False
    assert v.gates_check(good[:10])["gate_pass"] is False  # too few trades


def test_hit_rate_taken_only():
    pred = np.array([1, 1, 2, 0])
    y =    np.array([1, 2, 2, 1])
    rate, n = v.hit_rate(pred, y)
    assert n == 3 and rate == pytest.approx(2 / 3)


# --- calibration ---

def test_calibration_report_perfect_and_overconfident():
    claimed = [0.71] * 100
    won = [True] * 71 + [False] * 29
    rep = v.calibration_report(claimed, won)
    row = rep[rep["bucket"] == "70%-80%"].iloc[0]
    assert row["n"] == 100
    assert row["actual_win_rate"] == pytest.approx(0.71)
    # overconfident model: claims 80%, wins 55% -> the report must SHOW it
    won2 = [True] * 55 + [False] * 45
    rep2 = v.calibration_report([0.81] * 100, won2)
    row2 = rep2[rep2["bucket"] == "80%-90%"].iloc[0]
    assert row2["claimed"] > row2["actual_win_rate"] + 0.2


# --- baseline ---

def test_momentum_baseline_takes_few_trades():
    rng = np.random.default_rng(0)
    n = 6000
    feats = pd.DataFrame({"ret_15m": rng.normal(0, 0.002, n)})
    pred = v.momentum_baseline_classes(feats)
    assert set(np.unique(pred)) <= {0, 1, 2}
    frac_taken = (pred != 0).mean()
    assert 0.0 < frac_taken < 0.25               # selective by construction
