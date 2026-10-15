"""True forward test: predict on candles fetched live AFTER the data file ends.

Nothing in this window exists in any parquet, model, or prior evaluation.
"""
import json
import sys
import urllib.request
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\lumin\Desktop\sol-perp-bot")
sys.path.insert(0, str(ROOT))
import features as F
import labeling as LAB
import validation as V
import config


def fetch(sym, start_ms):
    rows, cursor = [], start_ms
    while True:
        u = (f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}"
             f"&interval=1m&startTime={cursor}&limit=1000")
        with urllib.request.urlopen(u, timeout=20) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 60_000
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v", "ct", "qv",
                                     "n", "tbv", "tqv", "ig"])
    out = pd.DataFrame({
        "timestamp": df["t"].astype(np.int64),
        "open": df["o"].astype(float), "high": df["h"].astype(float),
        "low": df["l"].astype(float), "close": df["c"].astype(float),
        "volume": df["v"].astype(float),
        "taker_buy_volume": df["tbv"].astype(float)})
    return out


def main():
    sol = pd.read_parquet(ROOT / "data_store/candles_SOL_1m_binance.parquet")
    btc = pd.read_parquet(ROOT / "data_store/candles_BTC_1m_binance.parquet")
    file_end = int(sol["timestamp"].max())
    fetch_from = file_end - 6 * 3600_000  # small overlap for sanity
    sol_f = fetch("SOLUSDT", fetch_from)
    btc_f = fetch("BTCUSDT", fetch_from)
    # Sanity: overlapping bars must match the file exactly.
    ov = sol_f[sol_f.timestamp <= file_end].set_index("timestamp")["close"]
    ovf = sol.set_index("timestamp")["close"].reindex(ov.index)
    assert np.allclose(ov, ovf), "overlap mismatch vs file"
    print(f"overlap check: {len(ov)} bars exact-match vs file")

    sol_all = (pd.concat([sol, sol_f[sol_f.timestamp > file_end]])
               .drop_duplicates("timestamp").sort_values("timestamp"))
    btc_all = (pd.concat([btc, btc_f[btc_f.timestamp > file_end]])
               .drop_duplicates("timestamp").sort_values("timestamp"))
    fresh_start = file_end + 60_000
    print(f"fresh window: {pd.to_datetime(fresh_start, unit='ms')} -> "
          f"{pd.to_datetime(sol_all.timestamp.max(), unit='ms')} "
          f"({len(sol_all[sol_all.timestamp > file_end])} new bars)")

    feats = F.compute_model_features(sol_all, btc_candles=btc_all)
    mapping = json.loads((ROOT / "research_models.json").read_text())
    bundles = {"fast": "fast", "mid": "mid", "slow": "slow"}
    for key in bundles:
        path = ROOT / "models" / mapping[key]
        b = joblib.load(path)
        h = b["horizon_min"]
        dead = float(b.get("target_deadband") or config.ROUND_TRIP_COST)
        candles = F._to_indexed(sol_all)
        labels = LAB.forward_return_labels(candles, h, deadband=dead)
        idx = feats.index[(feats.index >= pd.Timestamp(fresh_start, unit="ms", tz="UTC"))]
        usable = idx[labels.reindex(idx).notna()]
        try:
            X = F.prepare_model_input(feats.loc[usable], b["feature_names"])
        except ValueError as e:
            print(f"{key}: unavailable on fresh window ({e})")
            continue
        p = np.asarray(b["model"].predict_proba(X), dtype=float)
        y = LAB.labels_to_classes(labels.loc[usable])
        prior = np.bincount(y, minlength=3) / len(y)
        m = V.probability_metrics(y, p)
        base = V.probability_metrics(y, np.tile(prior, (len(y), 1)))
        rule = b.get("decisive_rule") or {"tau": 0.45, "delta": 0.10}
        pred = V.decisive_classes(p, tau=rule["tau"], delta=rule["delta"])
        import train as T
        atr = (F.atr(candles, 14) / candles["close"]).loc[usable].to_numpy()
        r = T._path(candles, usable, pred, atr, h,
                    cost=config.ROUND_TRIP_COST, barrier_scale=h ** 0.5)
        taken = (pred != 0) & np.isfinite(r)
        rr = r[taken]
        calls = int((pred != 0).sum())
        print(f"{key} (h={h}m, deadband={dead}): n={len(y)} "
              f"ll={m['log_loss']:.4f} vs prior {base['log_loss']:.4f} "
              f"(d={base['log_loss']-m['log_loss']:+.4f}) acc={m['accuracy']:.3f} "
              f"bal={m['balanced_accuracy']:.3f}")
        if calls:
            print(f"   calls={calls} fills={int(taken.sum())} "
                  f"win={np.mean(rr>0):.3f} pf={V.profit_factor(rr):.2f}")
        else:
            print("   calls=0 (no decisive signals in window)")


if __name__ == "__main__":
    main()
