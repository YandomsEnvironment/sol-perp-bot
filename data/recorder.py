"""Live WebSocket recorder (plan P1b / §4) — the data nobody gives away.

Records from Hyperliquid's public WebSocket (free, no key):
  - trades   : raw fills with SIDE -> rebuilds taker_buy_volume per minute
               (live parity with the Binance taker-flow features, M6)
  - book     : L2 orderbook FEATURES at each update (spread, depth and
               imbalance within 10bp of mid) — compact instead of terabytes
  - ctx      : funding / open interest / premium whenever they change

Output: data_store/ws/{trades|book|ctx}_YYYYMMDD.parquet, merged each flush.
Run:    python data/recorder.py    (keep it running for days/weeks)
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from websocket import WebSocketApp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

OUT = config.DATA_STORE / "ws"
OUT.mkdir(exist_ok=True)
FLUSH_S = 60


class Recorder:
    def __init__(self, coin: str = config.COIN, url: str = config.HL_WS_URL):
        self.coin = coin
        self.url = url
        self.buf: dict[str, list[dict]] = {"trades": [], "book": [], "ctx": []}
        self.lock = threading.Lock()
        self.counts = {"trades": 0, "book": 0, "ctx": 0}

    # ------------------------------------------------------------------ ws
    def _on_open(self, ws: WebSocketApp) -> None:
        for sub in (
            {"type": "trades", "coin": self.coin},
            {"type": "l2Book", "coin": self.coin},
            {"type": "activeAssetCtx", "coin": self.coin},
        ):
            ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
        print(f"[recorder] subscribed: {self.coin}", flush=True)

        def keepalive():
            while True:
                time.sleep(30)
                try:
                    ws.send(json.dumps({"method": "ping"}))
                except Exception:
                    return

        threading.Thread(target=keepalive, daemon=True).start()

    def _on_message(self, ws: WebSocketApp, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        ch, data = msg.get("channel"), msg.get("data")
        try:
            if ch == "trades" and isinstance(data, list):
                rows = [{"time": int(t["time"]), "px": float(t["px"]),
                         "sz": float(t["sz"]), "side": str(t["side"])}
                        for t in data if t.get("coin") == self.coin]
                if rows:
                    with self.lock:
                        self.buf["trades"] += rows
            elif ch == "l2Book" and isinstance(data, dict) \
                    and data.get("coin") == self.coin:
                row = self._book_row(data)
                if row:
                    with self.lock:
                        self.buf["book"].append(row)
            elif ch == "activeAssetCtx" and isinstance(data, dict):
                ctx = data.get("ctx", data)
                row = {"time": int(time.time() * 1000),
                       "funding": float(ctx.get("funding", "nan")),
                       "open_interest": float(ctx.get("openInterest", "nan")),
                       "premium": float(ctx.get("premium", "nan"))}
                with self.lock:
                    self.buf["ctx"].append(row)
        except Exception as e:  # never die on one bad message
            print(f"[recorder] bad message ch={ch} "
                  f"type={type(data).__name__}: {e}", flush=True)

    def _book_row(self, data: dict) -> dict | None:
        levels = data.get("levels", {})
        # Verified 2026-10: levels is a LIST [bids, asks]. Older docs show a
        # dict {"bids": [...], "asks": [...]} — support both.
        if isinstance(levels, dict):
            bids, asks = levels.get("bids", []), levels.get("asks", [])
        elif isinstance(levels, list) and len(levels) >= 2:
            bids, asks = levels[0], levels[1]
        else:
            bids, asks = [], []
        if not bids or not asks:
            return None
        bb, ba = float(bids[0]["px"]), float(asks[0]["px"])
        mid = (bb + ba) / 2.0
        if mid <= 0:
            return None

        def depth_10bp(lvls) -> float:
            tot = 0.0
            for lv in lvls[:100]:
                px = float(lv["px"])
                if abs(px - mid) / mid <= 0.001:
                    tot += float(lv["sz"]) * px
            return tot

        bd, ad = depth_10bp(bids), depth_10bp(asks)
        return {"time": int(data.get("time", time.time() * 1000)),
                "mid": mid, "spread_bps": (ba - bb) / mid * 1e4,
                "bid_dep_10bp": bd, "ask_dep_10bp": ad,
                "imb_10bp": (bd - ad) / (bd + ad) if bd + ad > 0 else 0.0}

    # --------------------------------------------------------------- disk
    def _flush(self) -> None:
        with self.lock:
            swapped = {k: v for k, v in self.buf.items()}
            self.buf = {k: [] for k in self.buf}
        for name, rows in swapped.items():
            if not rows:
                continue
            df = pd.DataFrame(rows).drop_duplicates()
            day = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
            path = OUT / f"{name}_{day}.parquet"
            if path.exists():
                old = pd.read_parquet(path)
                df = (pd.concat([old, df], ignore_index=True)
                      .drop_duplicates(subset=["time"]))
            df.to_parquet(path, index=False)
            self.counts[name] += len(rows)
            print(f"[recorder] {name}: +{len(rows)} rows (total "
                  f"{self.counts[name]})", flush=True)

    # ---------------------------------------------------------------- run
    def run(self) -> None:
        threading.Thread(target=self._flush_loop, daemon=True).start()
        while True:
            app = WebSocketApp(self.url,
                               on_open=self._on_open,
                               on_message=self._on_message,
                               on_error=lambda ws, e: print(
                                   f"[recorder] ws error: {e}", flush=True),
                               on_close=lambda ws, c, m: print(
                                   "[recorder] ws closed", flush=True))
            try:
                app.run_forever(ping_interval=20, ping_timeout=15)
            except Exception as e:
                print(f"[recorder] run_forever died: {e}", flush=True)
            self._flush()
            print("[recorder] reconnecting in 5s...", flush=True)
            time.sleep(5)

    def _flush_loop(self) -> None:
        while True:
            time.sleep(FLUSH_S)
            self._flush()


if __name__ == "__main__":
    Recorder().run()
