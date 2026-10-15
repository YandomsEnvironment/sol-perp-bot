"""Feature engineering — the SINGLE source of truth (plan M6).

Training and live prediction use compute_model_features() with a versioned
OHLCV-only contract. Shared code does not itself guarantee source-data parity.
Retrain and validate before deploying a changed feature specification.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Bars of history needed before features are valid (ret_1d needs 1440 1m bars).
WARMUP_BARS = 1440

RETURN_WINDOWS = {1: "ret_1m", 5: "ret_5m", 15: "ret_15m", 30: "ret_30m",
                  60: "ret_1h", 240: "ret_4h", 1440: "ret_1d"}
RV_WINDOWS = (15, 60)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI, bounded 0-100 (50 = neutral).

    Uses 100*avgGain/(avgGain+avgLoss): equivalent to the classic formula but
    correct in the degenerate cases (all-gains -> 100, all-losses -> 0,
    no movement -> 50) instead of dividing by zero.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    denom = avg_gain + avg_loss
    out = pd.Series(50.0, index=close.index)
    nz = denom > 0
    out.loc[nz] = 100.0 * avg_gain[nz] / denom[nz]
    return out.clip(0.0, 100.0)  # kill float hair like 100.00000000000001


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range. df needs high/low/close columns.

    Used both as a feature (atr_pct) and by labeling.py for barrier distances,
    so labels and live TP/SL share the same volatility yardstick.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _to_indexed(candles: pd.DataFrame) -> pd.DataFrame:
    """Normalize a candle frame: datetime UTC index from `timestamp` (ms), sorted."""
    df = candles.copy()
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if np.issubdtype(ts.dtype, np.number):
            df.index = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            df.index = pd.to_datetime(ts, utc=True)
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("Candles require a timezone-aware timestamp index")
    if df.index.has_duplicates:
        raise ValueError("Duplicate candle timestamps")
    return df


def _returns(close: pd.Series) -> pd.DataFrame:
    out = {}
    for w, name in RETURN_WINDOWS.items():
        out[name] = close.pct_change(w)
    return pd.DataFrame(out)


def compute_features(
    candles: pd.DataFrame,
    btc_candles: Optional[pd.DataFrame] = None,
    funding: Optional[pd.DataFrame] = None,
    oi: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute all features. Returns a DataFrame aligned to `candles`' index.

    candles:    columns timestamp(ms), open, high, low, close, volume
    btc_candles: same schema, reference market for lead-lag features
    funding:    columns timestamp(ms), funding_rate
    oi:         columns timestamp(ms), open_interest
    Rows within WARMUP_BARS of the start contain NaN — callers drop or ignore.
    """
    df = _to_indexed(candles)
    # Row windows must mean minutes. Preserve holes as missing observations;
    # never silently turn a two-minute price change into ret_1m.
    df = df.asfreq("1min")
    close = df["close"]
    feats = _returns(close)

    # Momentum / oscillators
    feats["rsi14"] = rsi(close, 14)

    # Volatility (as % of price so it's comparable across regimes)
    a = atr(df, 14)
    feats["atr_pct"] = a / close
    for w in RV_WINDOWS:
        feats[f"rv_{w}m"] = close.pct_change().rolling(w).std()

    # Trend structure: distance from EMAs and EMA crossover slope
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    feats["ema20_dist"] = close / ema20 - 1.0
    feats["ema50_dist"] = close / ema50 - 1.0
    feats["ema_x"] = ema20 / ema50 - 1.0

    # Position in recent range: 0 = at 60-bar low, 1 = at 60-bar high
    roll_lo = close.rolling(60).min()
    roll_hi = close.rolling(60).max()
    feats["range_pos_60"] = (close - roll_lo) / (roll_hi - roll_lo)

    # Volume context: burst vs drought
    vmean = df["volume"].rolling(60).mean()
    vstd = df["volume"].rolling(60).std()
    feats["vol_z_60"] = (df["volume"] - vmean) / vstd
    feats["vol_ratio_5_60"] = df["volume"].rolling(5).mean() / vmean

    # Session (crypto volatility is strongly time-of-day dependent)
    hours = df.index.hour + df.index.minute / 60.0
    feats["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # Taker flow (microstructure): aggressive buy share of volume.
    # Historical: Binance Vision klines carry taker_buy_volume (SOL file).
    # Live parity: aggregate Hyperliquid WS trades (side field) per minute
    # into the same column — same function, same numbers (M6 guarantee).
    if "taker_buy_volume" in df.columns:
        tbr = (df["taker_buy_volume"] / df["volume"].replace(0, np.nan)
               ).fillna(0.5)                    # zero-volume bar -> neutral
        for w in (5, 15, 60):
            feats[f"tbr_{w}m"] = tbr.rolling(w).mean()
        feats["tbr_x"] = feats["tbr_5m"] - feats["tbr_60m"]
        # trade-size proxy: volume per aggressive trade (bigger = institutional)
        ntb = (df["taker_buy_volume"] / df["taker_buy_volume"].rolling(15)
               .mean()).fillna(1.0)
        feats["taker_size_15m"] = ntb

    # Trade-count burst (parity: HL WS trades can rebuild per-minute counts)
    if "count" in df.columns:
        cnt = df["count"].astype(float)
        cm, cs = cnt.rolling(60).mean(), cnt.rolling(60).std()
        feats["n_trades_z_60"] = (cnt - cm) / cs.replace(0, np.nan)

    # BTC lead-lag (SOL follows BTC on short windows) + relative strength
    if btc_candles is not None:
        btc = _to_indexed(btc_candles)
        btc_rets = _returns(btc["close"])[["ret_1m", "ret_5m", "ret_15m",
                                           "ret_30m"]]
        btc_rets.columns = ["btc_ret_1m", "btc_ret_5m", "btc_ret_15m",
                            "btc_ret_30m"]
        btc_close = btc["close"].reindex(df.index)
        btc_rets = btc_rets.reindex(df.index)
        for c in btc_rets.columns:
            feats[c] = btc_rets[c]
        # SOL vs BTC relative strength: when SOL outperforms BTC, momentum
        # is SOL-native (stronger signal) rather than market beta
        spread = np.log(close) - np.log(btc_close)
        feats["sol_btc_rs_15m"] = spread.diff(15)
        feats["sol_btc_rs_1h"] = spread.diff(60)

    # Perp-specific: funding level + change since last stamp (crowd positioning)
    if funding is not None:
        f = pd.DataFrame({
            "timestamp": pd.to_numeric(
                pd.Series(funding["timestamp"]).reset_index(drop=True), errors="coerce"),
            "funding_rate": pd.Series(funding["funding_rate"]).reset_index(drop=True),
        }).dropna(subset=["timestamp"])
        f.index = pd.to_datetime(f["timestamp"], unit="ms", utc=True)
        f = f.sort_index()["funding_rate"]
        f = f.reindex(df.index, method="ffill")
        feats["funding"] = f
        feats["funding_chg"] = f.diff()

    # Perp-specific: open interest deltas (positions building / unwinding).
    # IMPORTANT: ffill onto the 1m candle grid FIRST, then pct-change, so
    # "5m"/"60m" mean minutes — not 5/60 stamps of irregular spacing.
    if oi is not None:
        o = pd.DataFrame({
            "timestamp": pd.to_numeric(
                pd.Series(oi["timestamp"]).reset_index(drop=True), errors="coerce"),
            "open_interest": pd.Series(oi["open_interest"]).reset_index(drop=True),
        }).dropna(subset=["timestamp"])
        o.index = pd.to_datetime(o["timestamp"], unit="ms", utc=True)
        o = o.sort_index()["open_interest"]
        o = o.reindex(df.index, method="ffill")
        feats["oi_chg_5m"] = o.pct_change(5)
        feats["oi_chg_60m"] = o.pct_change(60)

    # A missing minute invalidates all lookbacks that could cross it.
    valid = df[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
    continuous = valid.rolling(WARMUP_BARS + 1, min_periods=WARMUP_BARS + 1).sum()
    feats.loc[continuous != WARMUP_BARS + 1, :] = np.nan
    return feats.replace([np.inf, -np.inf], np.nan)


def compute_model_features(candles: pd.DataFrame,
                           btc_candles: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Versioned, venue-portable inputs available from both OHLCV feeds.

    This deliberately excludes Binance-only flow/counts and funding from a
    different venue. Source-venue differences still require separate validation.
    """
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    use = [c for c in columns if c in candles]
    btc = None if btc_candles is None else btc_candles[
        [c for c in columns if c in btc_candles]]
    return compute_features(candles[use], btc_candles=btc)


def prepare_model_input(feats: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Reject unknown or unavailable inputs; zero is not a missing-data policy."""
    if feats.empty or not feature_names:
        raise ValueError("No usable model inputs")
    absent = sorted(set(feature_names) - set(feats.columns))
    if absent:
        raise ValueError("Missing model features: " + ", ".join(absent))
    X = feats.reindex(columns=feature_names)
    bad = X.columns[~np.isfinite(X.to_numpy(dtype=float)).all(axis=0)].tolist()
    if bad:
        raise ValueError("Nonfinite model features: " + ", ".join(bad))
    return X
