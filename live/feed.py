"""Realtime market feed for the dashboard — one WebSocket, everything live.

Subscriptions (Hyperliquid public WS, free, no key):
  - allMids         -> mid price pushes (sub-second source)
  - trades:<coin>   -> every fill (price + side + size): also the freshest
                       price between allMids pushes, and live taker flow
  - activeAssetCtx  -> funding / open interest / mark, pushed on change
  - candle 1m       -> the FORMING candle (t,o,h,l,c,v) in realtime

Runs as a daemon thread inside the Streamlit server process. All state is
in-memory deques behind a lock; readers never block on the network.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

from websocket import WebSocketApp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


class MarketFeed:
    def __init__(self, coin: str = config.COIN, url: str = config.HL_WS_URL,
                 max_mids: int = 43_200, max_trades: int = 30_000):
        self.coin = coin
        self.url = url
        self.mids: deque[tuple[int, float]] = deque(maxlen=max_mids)
        self.trades: deque[tuple[int, float, float, str]] = deque(
            maxlen=max_trades)                       # (time, px, sz, side)
        self.candles: deque[dict] = deque(maxlen=2_880)   # 48h of 1m bars
        self.ctx: dict = {}
        self.ctx_time: int = 0
        self.lock = threading.Lock()
        self.connected = False

    # ------------------------------------------------------------------ ws
    def start(self) -> "MarketFeed":
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self) -> None:
        while True:
            app = WebSocketApp(self.url,
                               on_open=self._on_open,
                               on_message=self._on_message,
                               on_error=lambda ws, e: None,
                               on_close=lambda ws, c, m: self._set(False))
            try:
                app.run_forever(ping_interval=20, ping_timeout=15)
            except Exception:
                pass
            self._set(False)
            time.sleep(2)                        # reconnect loop, forever

    def _set(self, v: bool) -> None:
        with self.lock:
            self.connected = v

    def _on_open(self, ws: WebSocketApp) -> None:
        for sub in (
            {"type": "allMids"},
            {"type": "trades", "coin": self.coin},
            {"type": "activeAssetCtx", "coin": self.coin},
            {"type": "candle", "coin": self.coin, "interval": "1m"},
        ):
            ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
        self._set(True)

    def _on_message(self, ws: WebSocketApp, raw: str) -> None:
        try:
            m = json.loads(raw)
        except Exception:
            return
        ch, data = m.get("channel"), m.get("data")
        try:
            if ch == "allMids":
                px = (data or {}).get("mids", {}).get(self.coin)
                if px:
                    with self.lock:
                        self.mids.append((int(time.time() * 1000), float(px)))
            elif ch == "trades" and isinstance(data, list):
                with self.lock:
                    for t in data:
                        if t.get("coin") != self.coin:
                            continue
                        ts, px, sz = int(t["time"]), float(t["px"]), float(t["sz"])
                        side = str(t["side"])
                        self.trades.append((ts, px, sz, side))
                        # a fill IS the freshest price
                        self.mids.append((ts, px))
            elif ch == "activeAssetCtx" and isinstance(data, dict):
                ctx = data.get("ctx", data)
                with self.lock:
                    self.ctx = dict(ctx)
                    self.ctx_time = int(time.time() * 1000)
            elif ch == "candle" and isinstance(data, dict):
                row = {"timestamp": int(data["t"]),
                       "open": float(data["o"]), "high": float(data["h"]),
                       "low": float(data["l"]), "close": float(data["c"]),
                       "volume": float(data["v"])}
                with self.lock:
                    if self.candles and self.candles[-1]["timestamp"] == row["timestamp"]:
                        self.candles[-1] = row        # forming bar update
                    elif not self.candles or row["timestamp"] > self.candles[-1]["timestamp"]:
                        self.candles.append(row)      # new bar
        except Exception:
            pass                                   # never die on one message

    # ------------------------------------------------------------- readers
    def latest_mid(self) -> tuple[int, float] | None:
        with self.lock:
            return self.mids[-1] if self.mids else None

    def mid_age_ms(self) -> int | None:
        t = self.latest_mid()
        return int(time.time() * 1000) - t[0] if t else None

    def mids_since(self, ms_ago: int) -> list[tuple[int, float]]:
        cutoff = int(time.time() * 1000) - ms_ago
        with self.lock:
            return [x for x in self.mids if x[0] >= cutoff]

    def latest_ctx(self) -> dict:
        with self.lock:
            return dict(self.ctx)

    def taker_flow(self, minutes: int = 15) -> dict:
        """Aggressive buy vs sell size over the trailing window."""
        cutoff = int(time.time() * 1000) - minutes * 60_000
        buy = sell = 0.0
        with self.lock:
            for ts, _px, sz, side in self.trades:
                if ts < cutoff:
                    continue
                if side == "B":
                    buy += sz
                else:
                    sell += sz
        return {"buy_sz": buy, "sell_sz": sell,
                "buy_share": buy / (buy + sell) if buy + sell > 0 else None}

    def candles_tail(self, n: int = 3) -> "pd.DataFrame":
        import pandas as pd
        with self.lock:
            rows = list(self.candles)[-n:]
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low",
                                         "close", "volume"])
        return pd.DataFrame(rows)

    @property
    def is_connected(self) -> bool:
        with self.lock:
            return self.connected


MidFeed = MarketFeed      # backward-compatible alias


if __name__ == "__main__":                       # standalone smoke test
    import pandas as pd  # noqa: F401
    feed = MarketFeed().start()
    time.sleep(10)
    t = feed.latest_mid()
    print(f"connected={feed.is_connected} mids={len(feed.mids)} "
          f"trades={len(feed.trades)} candles={len(feed.candles)} "
          f"ctx_keys={len(feed.latest_ctx())}")
    if t:
        print("latest mid:", t, f"age={feed.mid_age_ms()}ms")
    print("taker flow 15m:", feed.taker_flow(15))
    if feed.candles:
        print("last candle:", feed.candles[-1])
