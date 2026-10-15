"""Tests for experiment machinery: decisive gating, seed bagging, v3 features."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import features as F
import validation as V


def test_decisive_classes_requires_both_confidence_and_margin():
    p = np.array([
        [0.2, 0.55, 0.25],   # margin 0.30 >= delta, fav 0.55 >= tau
        [0.2, 0.52, 0.28],   # margin 0.24 but fav 0.52 < tau
        [0.34, 0.40, 0.26],  # fav 0.40 < tau
        [0.1, 0.30, 0.60],   # short favored
        [0.6, 0.2, 0.2],     # flat-favored, never directional
    ])
    out = V.decisive_classes(p, tau=0.55, delta=0.05)
    assert out.tolist() == [1, 0, 0, 2, 0]


def test_decisive_classes_rejects_bad_shape():
    with pytest.raises(ValueError):
        V.decisive_classes(np.array([0.5, 0.5]))


def test_margin_classes_thresholds():
    from scripts.experiment import margin_classes
    p = np.array([0.72, 0.58, 0.5, 0.42, 0.28])
    assert margin_classes(p, 0.15).tolist() == [1, 0, 0, 0, 2]
    assert margin_classes(p, 0.05).tolist() == [1, 1, 0, 2, 2]


class _Const:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


def test_seed_bag_averages_and_normalizes():
    bag = V.SeedBag([_Const([[0.2, 0.5, 0.3]]), _Const([[0.4, 0.3, 0.3]])])
    out = bag.predict_proba(np.zeros((3, 1)))
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out, [[0.3, 0.4, 0.3]] * 3, atol=1e-12)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_seed_bag_requires_models():
    with pytest.raises(ValueError):
        V.SeedBag([])


def _candles(n=3200, seed=7):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.001, n)
    close = 100 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0008, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"timestamp": np.arange(n) * 60_000, "open": open_, "high": high,
         "low": low, "close": close, "volume": vol})


def test_v3_features_present_and_causal():
    candles = _candles()
    btc = _candles(seed=11)
    btc["timestamp"] = candles["timestamp"]
    feats = F.compute_model_features(candles, btc_candles=btc)
    for col in F.V3_COLUMNS:
        assert col in feats.columns, col
    # After warmup every v3 feature is finite on a clean grid.
    tail = feats.iloc[F.WARMUP_BARS + 10:]
    assert tail[F.V3_COLUMNS].notna().all().all()
    # Causality: corrupting the last 500 bars must not change earlier rows.
    sabotaged = candles.copy()
    sabotaged.loc[sabotaged.index[-500:], "close"] *= 1.5
    sabotaged.loc[sabotaged.index[-500:], "high"] *= 1.6
    sabotaged.loc[sabotaged.index[-500:], "low"] *= 0.4
    before = F.compute_model_features(candles, btc_candles=btc).iloc[:2000]
    after = F.compute_model_features(sabotaged, btc_candles=btc).iloc[:2000]
    pd.testing.assert_frame_equal(before, after)


def test_v5_positioning_features_flow_through():
    candles = _candles()
    btc = _candles(seed=11)
    btc["timestamp"] = candles["timestamp"]
    n = len(candles)
    funding = pd.DataFrame({
        "timestamp": candles["timestamp"][::60],
        "funding_rate": np.linspace(1e-5, 2e-5, n // 6 + 1)[:len(candles[::60])],
    })
    oi = pd.DataFrame({
        "timestamp": candles["timestamp"][::5],
        "open_interest": np.linspace(1e6, 2e6, len(candles[::5])),
    })
    feats = F.compute_model_features(candles, btc_candles=btc,
                                     funding=funding, oi=oi)
    for col in F.V5_COLUMNS:
        assert col in feats.columns, col
    tail = feats.iloc[F.WARMUP_BARS + 10:]
    assert tail[F.V5_COLUMNS].notna().all().all()
    # Without the frames the columns must NOT appear (contract stays closed).
    bare = F.compute_model_features(candles, btc_candles=btc)
    assert not set(F.V5_COLUMNS) & set(bare.columns)
    # Positioning changes are causal: corrupting future OI leaves history fixed.
    oi2 = oi.copy()
    oi2.loc[oi2.index[-30:], "open_interest"] *= 3
    before = F.compute_model_features(candles, btc_candles=btc, funding=funding, oi=oi).iloc[:1500]
    after = F.compute_model_features(candles, btc_candles=btc, funding=funding, oi=oi2).iloc[:1500]
    pd.testing.assert_frame_equal(before, after)


def test_feature_columns_v2_excludes_v3():
    cols = ["ret_1m", "close", "label", "atr_pct", "ret_2h", "btc_rsi14"]
    df = pd.DataFrame(columns=cols)
    out = None
    from scripts.experiment import feature_columns
    out = feature_columns(df, "v2")
    assert "ret_2h" not in out and "btc_rsi14" not in out and "ret_1m" in out
    with pytest.raises(ValueError):
        feature_columns(df, "v9")
