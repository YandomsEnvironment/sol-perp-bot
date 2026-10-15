"""Unit tests for features.py — the feature-parity guarantee (M6).

Run with:  python -m pytest tests -v
"""
import numpy as np
import pandas as pd

from features import WARMUP_BARS, compute_features, rsi


def _candles(n: int = 2000, seed: int = 7, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.001, n)
    close = start * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, n)))
    vol = rng.uniform(10, 100, n)
    ts = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    # Distance in ms since epoch, robust across pandas 2 (ns) / 3 (us) eras
    epoch = pd.Timestamp(0, tz="UTC")
    ms = ((ts - epoch).total_seconds() * 1000.0).astype("int64")
    return pd.DataFrame({
        "timestamp": ms,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
        "taker_buy_volume": vol * rng.uniform(0.3, 0.7, n),
    })


CORE_COLUMNS = [
    "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_1h", "ret_4h", "ret_1d",
    "rsi14", "atr_pct", "rv_15m", "rv_60m",
    "ema20_dist", "ema50_dist", "ema_x", "range_pos_60",
    "vol_z_60", "vol_ratio_5_60", "hour_sin", "hour_cos",
]


def test_core_columns_present_and_finite_after_warmup():
    feats = compute_features(_candles())
    for col in CORE_COLUMNS:
        assert col in feats.columns, f"missing feature {col}"
    tail = feats.iloc[WARMUP_BARS + 50 :]
    for col in CORE_COLUMNS:
        assert tail[col].notna().all(), f"{col} has NaN after warmup"
        assert np.isfinite(tail[col].to_numpy()).all(), f"{col} non-finite"


def test_rsi_bounds():
    close = pd.Series(np.linspace(100, 200, 300))
    r = rsi(close)
    assert ((r >= 0) & (r <= 100)).all()
    assert r.iloc[-1] > 90          # relentless uptrend -> near 100
    down = rsi(pd.Series(np.linspace(200, 100, 300)))
    assert down.iloc[-1] < 10


def test_deterministic_same_input_same_output():
    """M6 parity depends on determinism: identical input -> identical output."""
    a = compute_features(_candles(seed=3))
    b = compute_features(_candles(seed=3))
    pd.testing.assert_frame_equal(a, b)


def test_btc_features_aligned():
    sol = _candles(2000, seed=1)
    btc = _candles(2000, seed=2, start=50000.0)
    feats = compute_features(sol, btc_candles=btc)
    for col in ["btc_ret_1m", "btc_ret_5m", "btc_ret_15m", "btc_ret_30m",
                "sol_btc_rs_15m", "sol_btc_rs_1h"]:
        assert col in feats.columns
    tail = feats.iloc[WARMUP_BARS + 50 :]
    for col in ["btc_ret_1m", "btc_ret_5m", "btc_ret_15m", "btc_ret_30m",
                "sol_btc_rs_15m", "sol_btc_rs_1h"]:
        assert tail[col].notna().all()


def test_taker_flow_features():
    feats = compute_features(_candles())
    for col in ["tbr_5m", "tbr_15m", "tbr_60m", "tbr_x"]:
        assert col in feats.columns
    tail = feats.iloc[WARMUP_BARS + 60:]
    for col in ["tbr_5m", "tbr_15m", "tbr_60m", "tbr_x"]:
        assert tail[col].notna().all()
        assert np.isfinite(tail[col].to_numpy()).all()
    # buy-share ratios must stay in [0, 1]
    for col in ["tbr_5m", "tbr_15m", "tbr_60m"]:
        assert ((tail[col] >= 0) & (tail[col] <= 1)).all()


def test_funding_and_oi_features():
    sol = _candles(2000, seed=5)
    idx = sol["timestamp"].to_numpy()[::60]  # hourly funding stamps (values!)
    funding = pd.DataFrame({
        "timestamp": idx,
        "funding_rate": np.linspace(0.0001, -0.0001, len(idx)),
    })
    oi = pd.DataFrame({
        "timestamp": idx,
        "open_interest": np.linspace(1e6, 2e6, len(idx)),
    })
    feats = compute_features(sol, funding=funding, oi=oi)
    tail = feats.iloc[WARMUP_BARS + 60 :]
    for col in ["funding", "funding_chg", "oi_chg_5m", "oi_chg_60m"]:
        assert col in feats.columns
        assert tail[col].notna().all()
    # semantic check: between hourly stamps the ffilled OI is constant. Each
    # stamp jump stays inside the 5-bar lookback for 5 minutes -> ~34 stamps x
    # 5 bars = ~170 nonzero rows out of 2000 (~8.5%), so >90% exact zeros.
    vals = feats["oi_chg_5m"].fillna(0.0)
    assert vals.abs().max() < 0.0304
    assert (vals.abs() < 1e-12).mean() > 0.90
