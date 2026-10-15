"""Deep 1m history backfill from Binance Vision bulk dumps (plan §4 fallback).

WHY THIS EXISTS (discovered 2026-10): Hyperliquid's REST candleSnapshot only
serves a rolling ~5000-bar window per interval (1m ≈ 3.5 days, 1h ≈ 208 days).
For 12+ months of 1m training data we use Binance Vision public dumps:
  https://data.binance.vision/data/futures/um/{monthly|daily}/klines/<SYM>/1m/...

Venue caveat: Binance history is reference data, not Jupiter execution history.
Feature transfer and trading costs require separate validation.

Output: data_store/candles_{COIN}_1m_binance.parquet with columns
timestamp, open, high, low, close, volume, quote_volume, count,
taker_buy_volume  (timestamp = ms UTC, bar open time).
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

BASE = "https://data.binance.vision/data/futures/um"
COLS = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
        "ignore"]
KEEP = ["timestamp", "open", "high", "low", "close", "volume",
        "quote_volume", "count", "taker_buy_volume"]

BINANCE_SYMBOLS = {config.COIN: "SOLUSDT", config.BTC_COIN: "BTCUSDT"}


def _month_iter(start: pd.Timestamp, end: pd.Timestamp):
    cur = start.to_period("M")
    last_full = (end - pd.offsets.MonthBegin(1)).to_period("M")
    while cur <= last_full:
        yield cur
        cur += 1


def _day_iter(start: pd.Timestamp, end: pd.Timestamp):
    cur = start.normalize()
    while cur <= end.normalize():
        yield cur
        cur += pd.Timedelta(days=1)


def _fetch_zip(url: str) -> pd.DataFrame | None:
    """Download one Binance Vision zip -> DataFrame, or None if 404/missing."""
    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    raw = zf.read(zf.namelist()[0]).decode("utf-8")
    skiprows = 1 if raw.lower().startswith("open_time") else 0
    df = pd.read_csv(io.StringIO(raw), header=None, names=COLS,
                     skiprows=skiprows)
    return df[KEEP]


def backfill(symbol: str, coin: str, months: float) -> pd.DataFrame:
    """Fetch monthly zips for full months + daily zips for the current month."""
    out_path = config.DATA_STORE / f"candles_{coin}_1m_binance.parquet"
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=months * 30.44)
    # period math on tz-naive copies (pandas drops tz in Period conversion)
    end_n = end.tz_localize(None)
    start_n = start.tz_localize(None)
    done_through = None
    if out_path.exists():
        old = pd.read_parquet(out_path)
        if len(old):
            done_through = pd.Timestamp(old["timestamp"].max(), unit="ms")
            print(f"[{symbol}] existing data through {done_through}", flush=True)

    frames: list[pd.DataFrame] = []
    n_months = n_days = 0
    for m in _month_iter(start_n, end_n):
        url = f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{m}.zip"
        df = _fetch_zip(url)
        if df is None:
            print(f"[{symbol}] {m}: not available (skipped)", flush=True)
            continue
        if done_through is not None and pd.Timestamp(
                df["timestamp"].max(), unit="ms") <= done_through:
            n_months += 1
            continue
        frames.append(df)
        n_months += 1
        print(f"[{symbol}] {m}: {len(df)} bars", flush=True)
        time.sleep(0.3)

    first_of_month = (end_n.to_period("M").start_time).tz_localize("UTC")
    for d in _day_iter(first_of_month, end - pd.Timedelta(days=1)):
        url = f"{BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{d:%Y-%m-%d}.zip"
        df = _fetch_zip(url)
        if df is None:
            continue
        frames.append(df)
        n_days += 1
        time.sleep(0.3)
    print(f"[{symbol}] months={n_months} days={n_days}", flush=True)

    if not frames:
        print(f"[{symbol}] nothing new", flush=True)
        return pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()

    new = (pd.concat(frames, ignore_index=True)
           .drop_duplicates("timestamp").sort_values("timestamp"))
    if out_path.exists():
        old = pd.read_parquet(out_path)
        new = (pd.concat([old, new], ignore_index=True)
               .drop_duplicates("timestamp").sort_values("timestamp"))
    new.to_parquet(out_path, index=False)
    print(f"[{symbol}] total stored: {len(new)} bars -> {out_path.name}",
          flush=True)
    return new


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=float, default=12.0)
    args = ap.parse_args()
    for coin, sym in BINANCE_SYMBOLS.items():
        backfill(sym, coin, args.months)
