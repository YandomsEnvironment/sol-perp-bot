"""Client for Lighter (app.lighter.xyz) perpetual DEX on Solana / zkLighter.

Provides access to Lighter REST API:
- Market metadata, fees, and tick sizes (/api/v1/orderBooks)
- Order book depth, best bid/ask, and spread (/api/v1/orderBookOrders)
- Recent live trades and flow (/api/v1/recentTrades)
- Native OHLCV candlestick data (/api/v1/candles)

Web application: https://app.lighter.xyz/trade/SOL
API documentation: https://apidocs.lighter.xyz/
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = getattr(config, "LIGHTER_API_BASE_URL", "https://mainnet.zklighter.elliot.ai")
DEFAULT_MARKET_ID = getattr(config, "LIGHTER_MARKET_ID", 2)
DEFAULT_TIMEOUT_SEC = 5.0
DEFAULT_USER_AGENT = "sol-perp-bot/1.0"


def _fetch_json(url: str, timeout: float = DEFAULT_TIMEOUT_SEC) -> Optional[Dict[str, Any]]:
    """Fetches and decodes JSON from a Lighter REST endpoint."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8")
                return json.loads(body)
    except Exception as e:
        logger.warning(f"Failed to query Lighter endpoint {url}: {e}")
    return None


def get_market_details(
    market_id: int = DEFAULT_MARKET_ID,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Optional[Dict[str, Any]]:
    """Returns order book details and metadata for the specified market (SOL perps default: market_id=2)."""
    url = f"{base_url}/api/v1/orderBooks?market_id={market_id}"
    data = _fetch_json(url, timeout=timeout)
    if not data or "order_books" not in data:
        return None

    for ob in data.get("order_books", []):
        if ob.get("market_id") == market_id or ob.get("symbol") == "SOL":
            return {
                "symbol": ob.get("symbol", "SOL"),
                "market_id": ob.get("market_id", market_id),
                "market_type": ob.get("market_type", "perp"),
                "status": ob.get("status", "active"),
                "taker_fee": float(ob.get("taker_fee", "0.0")),
                "maker_fee": float(ob.get("maker_fee", "0.0")),
                "min_base_amount": float(ob.get("min_base_amount", "0.055")),
                "min_quote_amount": float(ob.get("min_quote_amount", "10.0")),
                "supported_price_decimals": ob.get("supported_price_decimals", 3),
                "supported_size_decimals": ob.get("supported_size_decimals", 3),
                "open_interest": float(ob.get("open_interest", "0.0")),
                "venue": "Lighter",
                "app_url": getattr(config, "LIGHTER_WEB_URL", "https://app.lighter.xyz/trade/SOL"),
            }
    return None


def get_order_book(
    market_id: int = DEFAULT_MARKET_ID,
    limit: int = 20,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Optional[Dict[str, Any]]:
    """Returns the current order book depth and spread from Lighter."""
    url = f"{base_url}/api/v1/orderBookOrders?market_id={market_id}&limit={limit}"
    data = _fetch_json(url, timeout=timeout)
    if not data or data.get("code") != 200:
        return None

    asks_raw = data.get("asks", [])
    bids_raw = data.get("bids", [])

    asks = [
        {"price": float(a["price"]), "size": float(a.get("remaining_base_amount", a.get("initial_base_amount", 0)))}
        for a in asks_raw if "price" in a
    ]
    bids = [
        {"price": float(b["price"]), "size": float(b.get("remaining_base_amount", b.get("initial_base_amount", 0)))}
        for b in bids_raw if "price" in b
    ]

    asks.sort(key=lambda x: x["price"])
    bids.sort(key=lambda x: x["price"], reverse=True)

    best_ask = asks[0]["price"] if asks else None
    best_bid = bids[0]["price"] if bids else None
    mid_price = (best_ask + best_bid) / 2.0 if (best_ask and best_bid) else None
    spread_usd = (best_ask - best_bid) if (best_ask and best_bid) else None
    spread_bps = (spread_usd / mid_price * 10000.0) if (spread_usd and mid_price) else 0.0

    return {
        "market_id": market_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "spread_usd": spread_usd,
        "spread_bps": spread_bps,
        "asks": asks[:limit],
        "bids": bids[:limit],
        "total_asks": len(asks),
        "total_bids": len(bids),
    }


def get_recent_trades(
    market_id: int = DEFAULT_MARKET_ID,
    limit: int = 20,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Optional[List[Dict[str, Any]]]:
    """Returns recent trades from Lighter."""
    url = f"{base_url}/api/v1/recentTrades?market_id={market_id}&limit={limit}"
    data = _fetch_json(url, timeout=timeout)
    if not data or data.get("code") != 200:
        return None

    trades = []
    for t in data.get("trades", []):
        trades.append({
            "trade_id": str(t.get("trade_id", "")),
            "price": float(t.get("price", 0.0)),
            "size": float(t.get("size", 0.0)),
            "usd_amount": float(t.get("usd_amount", 0.0)),
            "timestamp": t.get("timestamp", 0),
            "is_maker_ask": t.get("is_maker_ask", False),
            "taker_fee": float(t.get("taker_fee", 0.0)),
        })
    return trades


def get_candles(
    resolution: str = "1h",
    count_back: int = 50,
    market_id: int = DEFAULT_MARKET_ID,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Optional[List[Dict[str, Any]]]:
    """Returns recent historical OHLCV candles from Lighter."""
    now_ts = int(time.time())
    url = (
        f"{base_url}/api/v1/candles?"
        f"market_id={market_id}&resolution={resolution}"
        f"&start_timestamp=0&end_timestamp={now_ts}&count_back={count_back}"
    )
    data = _fetch_json(url, timeout=timeout)
    if not data or "c" not in data:
        return None

    candles = []
    for c in data.get("c", []):
        candles.append({
            "time_ms": c.get("t"),
            "open": float(c.get("o", 0.0)),
            "high": float(c.get("h", 0.0)),
            "low": float(c.get("l", 0.0)),
            "close": float(c.get("c", 0.0)),
            "volume_base": float(c.get("v", 0.0)),
            "volume_usd": float(c.get("V", 0.0)),
        })
    return candles
