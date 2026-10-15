"""Backfill Binance Vision futures metrics (open interest etc.) for SOL.

Downloads daily 5-minute metrics dumps (sum_open_interest, taker long/short
volume ratio) and writes data_store/metrics_SOL.parquet with ms timestamps.
Live parity: Hyperliquid ctx serves openInterest + funding in real time.

Usage: python data/download_metrics.py [--start 2025-10-01] [--end 2026-10-11]
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_store" / "metrics_SOL.parquet"
BASE = "https://data.binance.vision/data/futures/um/daily/metrics/SOLUSDT"
COLS = ["create_time", "sum_open_interest", "sum_open_interest_value",
        "sum_taker_long_short_vol_ratio"]


def fetch(day: date, tries: int = 3):
    import time
    url = f"{BASE}/SOLUSDT-metrics-{day.isoformat()}.zip"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), usecols=COLS)
            df["timestamp"] = (pd.to_datetime(df["create_time"], utc=True)
                               .astype("datetime64[ns]").astype("int64") // 10**6)
            return df.drop(columns="create_time")
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    start = date.fromisoformat(a.start)
    end = date.fromisoformat(a.end) if a.end else date.today() - timedelta(days=1)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    print(f"fetching {len(days)} daily metric files ...", flush=True)
    frames = [None] * len(days)
    done_count = 0
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(fetch, d): i for i, d in enumerate(days)}
        for fut in futures:
            pass
        from concurrent.futures import as_completed
        for fut in as_completed(futures):
            i = futures[fut]
            frames[i] = fut.result()
            done_count += 1
            if done_count % 25 == 0:
                print(f"  {done_count}/{len(days)} ...", flush=True)
    ok = [f for f in frames if f is not None and len(f)]
    print(f"got {len(ok)}/{len(days)} files", flush=True)
    if not ok:
        sys.exit("no data downloaded")
    out = (pd.concat(ok, ignore_index=True)
           .drop_duplicates("timestamp").sort_values("timestamp")
           .rename(columns={"sum_open_interest": "open_interest",
                            "sum_open_interest_value": "open_interest_usd",
                            "sum_taker_long_short_vol_ratio": "taker_ls_ratio"}))
    OUT.parent.mkdir(exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} rows, "
          f"{pd.to_datetime(out.timestamp.min(), unit='ms')} -> "
          f"{pd.to_datetime(out.timestamp.max(), unit='ms')}", flush=True)


if __name__ == "__main__":
    main()
