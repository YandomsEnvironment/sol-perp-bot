"""Bulk historical data downloader (plan P1).

Source: Hyperliquid REST API — reference market data, NOT Jupiter execution
data. Historical model training also uses Binance; transfer must be evaluated.
  - 1m candles for SOL (target) and BTC (lead-lag feature)
  - funding history for SOL (crowd-positioning feature)

Resumable: if a parquet file already exists, download continues from its
last stored timestamp. Output: data_store/*.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

INFO_URL = config.HL_API_URL + "/info"
INTERVAL = "1m"  # verified against Hyperliquid API 2026-10


def _post(payload: dict, retries: int = 3) -> dict | list:
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(INFO_URL, json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - log & retry, then give up
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Hyperliquid info request failed: {payload} -> {last_err}")


def fetch_candles(coin: str, start_ms: int, end_ms: int,
                  interval: str = INTERVAL, verbose: bool = True) -> pd.DataFrame:
    """Page through candleSnapshot until end. Returns df[t,o,h,l,c,v] (ms ints).

    Response format (verified): a LIST of row dicts with STRING ohlcv values,
    e.g. {"t": ms, "T": ms, "s": "SOL", "i": "1m", "o": "109.53", ...}
    interval: "1m" (default) or "5m"/"15m"/"1h"/"4h" for display timeframes.
    """
    frames = []
    cur = int(start_ms)
    while cur < end_ms:
        data = _post({"type": "candleSnapshot", "req": {
            "coin": coin, "interval": interval,
            "startTime": cur, "endTime": int(end_ms)}})
        if not (isinstance(data, list) and data):
            break
        df = pd.DataFrame({
            "timestamp": [int(r["t"]) for r in data],
            "open": [float(r["o"]) for r in data],
            "high": [float(r["h"]) for r in data],
            "low": [float(r["l"]) for r in data],
            "close": [float(r["c"]) for r in data],
            "volume": [float(r["v"]) for r in data],
        })
        frames.append(df)
        last_t = int(df["timestamp"].max())
        if verbose:
            print(f"    {coin} {interval}: ...{pd.to_datetime(last_t, unit='ms')}"
                  f"  (+{len(df)} bars)", flush=True)
        if last_t <= cur:
            break
        cur = last_t + 1
        time.sleep(0.15)                        # be polite to the API
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low",
                                     "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("timestamp").sort_values("timestamp")


def fetch_funding(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Funding history rows -> df[timestamp(ms), funding_rate]."""
    rows = []
    cur = int(start_ms)
    while cur < end_ms:
        data = _post({"type": "fundingHistory", "coin": coin,
                      "startTime": cur, "endTime": int(end_ms)})
        if not isinstance(data, list) or not data:
            break
        for it in data:
            rows.append({"timestamp": int(it["time"]),
                         "funding_rate": float(it["fundingRate"])})
        last_t = max(r["timestamp"] for r in rows)
        if last_t <= cur:
            break
        cur = last_t + 1
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    return (pd.DataFrame(rows).drop_duplicates("timestamp")
            .sort_values("timestamp"))


def _missing_range(path: Path, desired_start_ms: int, now_ms: int):
    """Return (start, end) ms of what needs fetching, or None if complete.

    Handles BOTH directions:
      - backfill: desired_start older than stored data -> fetch older gap
      - incremental: now newer than stored data -> fetch the new tail
    Middle holes are not handled (v1: data comes from one continuous source).
    """
    if not path.exists():
        return desired_start_ms, now_ms
    old = pd.read_parquet(path)
    if len(old) == 0:
        return desired_start_ms, now_ms
    lo = int(old["timestamp"].min())
    hi = int(old["timestamp"].max())
    if desired_start_ms < lo:
        return desired_start_ms, hi          # extend history backwards
    if hi < now_ms:
        return hi + 1, now_ms                # incremental update
    return None                              # already complete


def _merge_save(path: Path, new: pd.DataFrame, key: str) -> int:
    if new is None or len(new) == 0:
        return 0
    if path.exists():
        old = pd.read_parquet(path)
        new = (pd.concat([old, new], ignore_index=True)
               .drop_duplicates(key).sort_values(key))
    new.to_parquet(path, index=False)
    return len(new)


def run(days: int, coins: list[str], funding_coin: str) -> None:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 86_400_000
    for coin in coins:
        cpath = config.DATA_STORE / f"candles_{coin}_1m.parquet"
        rng = _missing_range(cpath, start_ms, now_ms)
        if rng is None:
            print(f"[{coin}] candles already complete", flush=True)
            continue
        s, e = rng
        print(f"[{coin}] candles {pd.to_datetime(s, unit='ms')} -> "
              f"{pd.to_datetime(e, unit='ms')} ...", flush=True)
        candles = fetch_candles(coin, s, e)
        n = _merge_save(cpath, candles, "timestamp")
        print(f"[{coin}] saved {n} new bars -> {cpath.name}", flush=True)
    fpath = config.DATA_STORE / f"funding_{funding_coin}.parquet"
    rng = _missing_range(fpath, start_ms, now_ms)
    if rng is None:
        print(f"[{funding_coin}] funding already complete", flush=True)
        return
    s, e = rng
    print(f"[{funding_coin}] funding {pd.to_datetime(s, unit='ms')} -> "
          f"{pd.to_datetime(e, unit='ms')} ...", flush=True)
    funding = fetch_funding(funding_coin, s, e)
    n = _merge_save(fpath, funding, "timestamp")
    print(f"[{funding_coin}] saved {n} new funding rows -> {fpath.name}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=0, help="days of history")
    ap.add_argument("--months", type=float, default=12.0,
                    help="months of history (default 12, used if --days=0)")
    args = ap.parse_args()
    days = args.days if args.days > 0 else int(args.months * 30.44)
    run(days=days, coins=[config.COIN, config.BTC_COIN],
        funding_coin=config.COIN)
