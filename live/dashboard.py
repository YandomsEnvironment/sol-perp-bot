"""Read-only JSON API powering the native SOL-PERP web dashboard."""
from __future__ import annotations

import bisect
import glob
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import features as F  # noqa: E402
from data.download_hist import fetch_candles, fetch_funding  # noqa: E402
from live.paper_trader import paper_trader  # noqa: E402

WEB_ROOT = Path(__file__).resolve().parent / "web"
SIGNALS_LOG = config.LOGS_DIR / "signals.jsonl"
CLASS_NAMES = {0: "FLAT", 1: "LONG", 2: "SHORT"}
TIMEFRAMES = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
              "1h": 3_600_000, "4h": 14_400_000}
HORIZONS = {"5m": ("fast", 5), "15m": ("mid", 15), "60m": ("slow", 60)}
_CACHE: dict[str, tuple[float, object]] = {}
_CACHE_LOCK = threading.Lock()
_FEED = None


def cached(key: str, ttl: float, factory):
    now = time.monotonic()
    with _CACHE_LOCK:
        found = _CACHE.get(key)
        if found and now - found[0] < ttl:
            return found[1]
    value = factory()
    with _CACHE_LOCK:
        _CACHE[key] = (now, value)
    return value


def load_settings() -> dict:
    path = ROOT / "settings.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        saved = {}
    tf = saved.get("tf", "1m")
    hz = saved.get("hz", "15m")
    return {"tf": tf if tf in TIMEFRAMES else "1m",
            "hz": hz if hz in HORIZONS else "15m",
            "sol_wallet": str(saved.get("sol_wallet", os.environ.get(
                "SOL_WALLET", ""))).strip()}


def save_settings(payload: dict) -> dict:
    current = load_settings()
    tf, hz = payload.get("tf", current["tf"]), payload.get("hz", current["hz"])
    wallet = str(payload.get("sol_wallet", current["sol_wallet"])).strip()
    if tf not in TIMEFRAMES or hz not in HORIZONS:
        raise ValueError("Unsupported chart timeframe or forecast horizon")
    if len(wallet) > 64:
        raise ValueError("Wallet address is too long")
    if wallet and not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", wallet):
        raise ValueError("Wallet must be a valid Solana public address")
    value = {"tf": tf, "hz": hz, "sol_wallet": wallet}
    (ROOT / "settings.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value


def get_feed():
    global _FEED
    if _FEED is None:
        from live.feed import MarketFeed
        _FEED = MarketFeed().start()
    return _FEED


_LAST_DATA = None
_LAST_CHART = {}


def get_data():
    global _LAST_DATA
    def fetch():
        global _LAST_DATA
        now = int(time.time() * 1000)
        start = now - 2 * 86_400_000
        try:
            sol = fetch_candles(config.COIN, start, now, verbose=False)
            btc = fetch_candles(config.BTC_COIN, start, now, verbose=False)
            funding = fetch_funding(config.COIN, start, now)
        except Exception:
            if _LAST_DATA is not None:
                return _LAST_DATA
            sol = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            btc = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            funding = pd.DataFrame(columns=["timestamp", "funding_rate"])
        try:
            r = requests.post(config.HL_API_URL + "/info",
                              json={"type": "metaAndAssetCtxs"}, timeout=15)
            r.raise_for_status()
            ctx = r.json()
        except Exception:
            ctx = None
        _LAST_DATA = (sol, btc, funding, ctx)
        return _LAST_DATA
    return cached("market-data", config.DASHBOARD_CACHE_TTL, fetch)


def get_chart_candles(tf: str) -> pd.DataFrame:
    global _LAST_CHART
    days = {"1m": 2, "5m": 4, "15m": 7, "1h": 14, "4h": 30}[tf]
    def fetch():
        now = int(time.time() * 1000)
        try:
            df = fetch_candles(config.COIN, now - days * 86_400_000, now,
                               interval=tf, verbose=False)
            _LAST_CHART[tf] = df
            return df
        except Exception:
            if tf in _LAST_CHART:
                return _LAST_CHART[tf]
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return cached(f"chart-{tf}", config.DASHBOARD_CACHE_TTL, fetch).copy()


def get_bundle(key: str) -> dict:
    """Explicit local research selection, never promotion or latest-file guessing."""
    try:
        selection = json.loads((ROOT / "research_models.json").read_text(encoding="utf-8-sig"))
        relative = selection[key]
        root = config.MODELS_DIR.resolve()
        path = (root / relative).resolve()
        if root not in path.parents or path.suffix != ".pkl":
            raise ValueError("Research model must be a .pkl file inside models")
        stamp = path.stat().st_mtime_ns
        def load():
            import joblib
            import validation as V
            bundle = dict(joblib.load(path))
            if (bundle.get("input_contract") not in F.SUPPORTED_CONTRACTS
                    or bundle.get("horizon_name") != key
                    or bundle.get("horizon_min") != config.HORIZONS_MIN[key]
                    or bundle.get("target") != "forward"
                    or not bundle.get("feature_names")):
                raise ValueError("Research model contract, target or horizon is incompatible")
            model = bundle["model"]
            if isinstance(model, tuple):
                model = model[0]
            if isinstance(model, V.CalibratedModel):
                model = model.model
            if list(model.classes_) != [0, 1, 2]:
                raise ValueError("Research model class order must be FLAT, LONG, SHORT")
            bundle.update(model=model, research_only=True, model_id=path.relative_to(root).as_posix(),
                          probability_kind="raw", conf_threshold=1.01, gate_pass=False)
            bundle["aggregate"] = {**bundle.get("aggregate", {}), "gate_pass": False}
            return bundle
        return cached(f"research-model-{key}-{path}-{stamp}", 3600, load)
    except Exception as exc:
        return {"model": None, "feature_names": [], "research_only": True,
                "load_error": f"Research model unavailable: {exc}"}


def live_ctx(ctx, coin: str) -> dict | None:
    if isinstance(ctx, list) and len(ctx) == 2:
        meta, ctxs = ctx
        universe = meta.get("universe", []) if isinstance(meta, dict) else meta
    elif isinstance(ctx, dict):
        universe, ctxs = ctx.get("universe", []), ctx.get("assetCtxs", [])
    else:
        return None
    for i, item in enumerate(universe):
        name = item if isinstance(item, str) else item.get("name")
        if name == coin and i < len(ctxs):
            c = ctxs[i]
            mark = float(c.get("markPx", "nan"))
            return {"funding": float(c["funding"]),
                    "open_interest_usd": float(c["openInterest"]) * mark,
                    "mark": mark}
    return None


def recorder_snapshots() -> dict:
    out = {}
    try:
        ctx_files = sorted(glob.glob(str(config.DATA_STORE / "ws" / "ctx_*.parquet")))
        if ctx_files:
            hist = (pd.concat([pd.read_parquet(f) for f in ctx_files[-2:]],
                              ignore_index=True).drop_duplicates("time")
                    .sort_values("time"))
            if len(hist) > 5:
                past = hist[hist["time"] <= int(hist["time"].max()) - 3_600_000]
                if len(past):
                    out["oi_chg_1h"] = (float(hist["open_interest"].iloc[-1]) /
                                        float(past["open_interest"].iloc[-1]) - 1)
    except Exception:
        pass
    try:
        files = sorted(glob.glob(str(config.DATA_STORE / "ws" / "trades_*.parquet")))
        if files:
            trades = pd.read_parquet(files[-1])
            cutoff = int(trades["time"].max()) - 15 * 60_000
            window = trades[trades["time"] > cutoff]
            buy = float(window.loc[window["side"] == "B", "sz"].sum())
            sell = float(window.loc[window["side"] == "A", "sz"].sum())
            if buy + sell:
                out["taker_buy_share_15m"] = buy / (buy + sell)
                out["taker_buy_size_15m"] = buy
                out["taker_sell_size_15m"] = sell
    except Exception:
        pass
    return out


def build_evidence(f: pd.Series, snap: dict, cross: dict | None = None) -> tuple[pd.DataFrame, dict]:
    f = f.reindex([*f.index, *[name for name in
        ("ema_x", "ema20_dist", "rsi14", "ret_1h", "ret_4h", "btc_ret_15m",
         "funding", "ret_15m", "vol_z_60", "ret_5m") if name not in f.index]])
    rows = []
    if cross:
        rows.append(("Multi-horizon consensus", cross["reading"], cross["vote"], cross["why"]))
    trend = "LONG" if f["ema_x"] > 0 and f["ema20_dist"] > 0 else (
        "SHORT" if f["ema_x"] < 0 and f["ema20_dist"] < 0 else "NEUTRAL")
    rows.append(("Trend (EMA20/50)", f"EMA spread {f['ema_x'] * 100:+.2f}%",
                 trend, "EMA alignment"))
    rsi = float(f["rsi14"])
    rsi_vote = "LONG" if rsi < 35 else "SHORT" if rsi > 65 else "NEUTRAL"
    rows.append(("RSI (14)", f"{rsi:.1f}", rsi_vote,
                 "oversold" if rsi < 35 else "overbought" if rsi > 65 else "mid-range"))
    r1h, r4h = float(f["ret_1h"]), float(f["ret_4h"])
    if r1h > 0.001 and r4h > 0.002:
        vote, why = "LONG", "higher-timeframe momentum agrees"
    elif r1h < -0.001 and r4h < -0.002:
        vote, why = "SHORT", "higher-timeframe momentum agrees"
    else:
        vote, why = "NEUTRAL", "timeframes disagree or are flat"
    rows.append(("Higher timeframe (1h/4h)", f"{r1h*100:+.2f}% / {r4h*100:+.2f}%", vote, why))
    btc = float(f["btc_ret_15m"])
    rows.append(("BTC lead (15m)", f"{btc*100:+.2f}%",
                 "LONG" if btc > .0015 else "SHORT" if btc < -.0015 else "NEUTRAL",
                 "BTC momentum" if abs(btc) > .0015 else "BTC quiet"))
    fund = float(f["funding"])
    rows.append(("Funding (hourly)", f"{fund:+.5f}" if np.isfinite(fund) else "Unavailable",
                 "SHORT" if fund > .00002 else "LONG" if fund < -.00002 else "NEUTRAL",
                 "crowd positioning" if np.isfinite(fund) else "not a model input"))
    oi = snap.get("oi_chg_1h")
    oi_vote = ("SHORT" if oi > 0 and f["ret_15m"] < 0 else
               "LONG" if oi > 0 and f["ret_15m"] > 0 else "NEUTRAL") if oi is not None else "NEUTRAL"
    rows.append(("Open interest change (1h)", f"{oi*100:+.1f}%" if oi is not None else "—",
                 oi_vote, "recorded OI/price context" if oi is not None else "no recorder snapshot"))
    tbs = snap.get("taker_buy_share_15m")
    tbs_vote = "LONG" if tbs is not None and tbs > .55 else (
        "SHORT" if tbs is not None and tbs < .45 else "NEUTRAL")
    rows.append(("Taker flow (15m)", f"{tbs:.0%} buys" if tbs is not None else "—",
                 tbs_vote, "aggressive buy/sell balance"))
    vz, r5 = float(f["vol_z_60"]), float(f["ret_5m"])
    vol_vote = ("LONG" if r5 > .001 else "SHORT" if r5 < -.001 else "NEUTRAL") if vz > 2 else "NEUTRAL"
    rows.append(("Volume confirmation", f"z={vz:.1f}", vol_vote,
                 "volume burst" if vz > 2 else "no volume climax"))
    frame = pd.DataFrame(rows, columns=["signal", "reading", "vote", "why"])
    tally = {side: int((frame["vote"] == side).sum()) for side in ("LONG", "SHORT", "NEUTRAL")}
    return frame, tally


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def market_snapshot(settings: dict) -> dict:
    tf, hz = settings["tf"], settings["hz"]
    sol, btc, funding, ctx = get_data()
    if sol is None:
        sol = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    model_key, horizon = HORIZONS[hz]
    bundle = get_bundle(model_key)
    model = bundle["model"]
    if isinstance(model, tuple):
        model = model[0]
    delay_ms = 2_000
    cutoff = (int(time.time() * 1000) - delay_ms) // TIMEFRAMES["1m"] * TIMEFRAMES["1m"]
    quality_errors = [bundle["load_error"]] if bundle.get("load_error") else []
    sol = sol.loc[sol["timestamp"] + TIMEFRAMES["1m"] <= cutoff].copy()
    if btc is not None and not btc.empty and "timestamp" in btc:
        btc = btc.loc[btc["timestamp"] + TIMEFRAMES["1m"] <= cutoff].copy()
    else:
        btc = pd.DataFrame(columns=sol.columns)
    for name, frame in (("SOL", sol), ("BTC", btc)):
        if frame is None or frame.empty:
            quality_errors.append(f"{name} closed candles are unavailable")
            continue
        ts = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy()
        if len(ts) > 1 and np.any(np.diff(ts) != TIMEFRAMES["1m"]):
            quality_errors.append(f"{name} candle input is stale or noncontiguous")
        if cutoff - int(ts[-1]) > 2 * TIMEFRAMES["1m"]:
            quality_errors.append(f"{name} latest closed candle is stale")
    if len(sol) < F.WARMUP_BARS + 60:
        quality_errors.append(f"Need at least {F.WARMUP_BARS + 60} closed 1m candles; got {len(sol)}")
    if bundle.get("input_contract") not in F.SUPPORTED_CONTRACTS:
        quality_errors.append("Model bundle input contract must be closed-ohlcv-v2 or closed-ohlcv-v3")
    if sol.empty:
        quality_errors.append("SOL closed candles are unavailable")
        feats = pd.DataFrame([{name: np.nan for name in
                               ("ret_15m", "atr_pct", "range_pos_60", "rsi14", "ret_1h", "ret_4h", "btc_ret_15m")}])
    else:
        feats = F.compute_model_features(sol, btc_candles=btc)
    proba = np.zeros(3, dtype=float)
    pred = None
    if not quality_errors:
        try:
            X = F.prepare_model_input(feats.iloc[[-1]], bundle["feature_names"])
            proba = np.asarray(model.predict_proba(X)[0], dtype=float)
            if (proba.shape != (3,) or not np.isfinite(proba).all()
                    or np.any(proba < 0) or not np.isclose(proba.sum(), 1.0, atol=1e-5)):
                raise ValueError("model returned invalid probabilities")
            pred = int(np.argmax(proba))
        except ValueError as exc:
            quality_errors.append(str(exc))
    feed = get_feed()
    price = float(sol["close"].iloc[-1]) if not sol.empty else 0.0
    tick = feed.latest_mid()
    tick_age = feed.mid_age_ms()
    if tick and tick_age is not None and tick_age < 15_000:
        price = float(tick[1])
    context = live_ctx(ctx, config.COIN) or {}
    live_feed_ctx = feed.latest_ctx()
    if live_feed_ctx.get("funding") is not None:
        try:
            mark = float(live_feed_ctx.get("markPx", price) or price)
            context = {"funding": float(live_feed_ctx["funding"]),
                       "open_interest_usd": float(live_feed_ctx.get("openInterest", 0) or 0) * mark,
                       "mark": mark}
        except (TypeError, ValueError):
            pass
    cross_info = None
    if not quality_errors and not feats.empty and pred is not None:
        confirm_key = "mid" if model_key == "fast" else ("fast" if model_key == "mid" else "mid")
        try:
            conf_bundle = get_bundle(confirm_key)
            conf_model = conf_bundle.get("model")
            if conf_model and not conf_bundle.get("load_error"):
                if isinstance(conf_model, tuple):
                    conf_model = conf_model[0]
                X_conf = F.prepare_model_input(feats.iloc[[-1]], conf_bundle["feature_names"])
                p_conf = np.asarray(conf_model.predict_proba(X_conf)[0], dtype=float)
                pred_conf = int(np.argmax(p_conf))

                agree = bool(pred != 0 and pred == pred_conf)
                diverge = bool(pred != 0 and pred_conf != 0 and pred != pred_conf)

                if agree:
                    vote = CLASS_NAMES[pred]
                    why = f"Consensus: {model_key} + {confirm_key} both forecast {vote}"
                elif diverge:
                    vote = "NEUTRAL"
                    why = f"Divergence: {model_key} ({CLASS_NAMES[pred]}) contradicts {confirm_key} ({CLASS_NAMES[pred_conf]})"
                elif pred == 0 and pred_conf == 0:
                    vote = "NEUTRAL"
                    why = f"Consensus: both {model_key} and {confirm_key} forecast FLAT"
                else:
                    active_side = CLASS_NAMES[pred if pred != 0 else pred_conf]
                    vote = "NEUTRAL"
                    why = f"Unconfirmed: {active_side} on only one horizon"

                reading = (f"{model_key} {CLASS_NAMES[pred]} ({proba[pred]*100:.0f}%) · "
                           f"{confirm_key} {CLASS_NAMES[pred_conf]} ({p_conf[pred_conf]*100:.0f}%)")
                cross_info = {
                    "reading": reading,
                    "vote": vote,
                    "why": why,
                    "confirm_key": confirm_key,
                    "confirm_lean": CLASS_NAMES[pred_conf],
                    "confirm_proba": {"flat": float(p_conf[0]), "long": float(p_conf[1]), "short": float(p_conf[2])},
                    "agree": agree,
                    "diverge": diverge
                }
        except Exception:
            cross_info = None

    snap = recorder_snapshots()
    evidence, tally = build_evidence(feats.iloc[-1], snap, cross_info)
    diff = tally["LONG"] - tally["SHORT"]
    evidence_lean = "LONG" if diff >= 2 else "SHORT" if diff <= -2 else "MIXED"
    model_lean = CLASS_NAMES[pred] if pred is not None and not quality_errors else "UNAVAILABLE"
    aggregate = bundle.get("aggregate", {})
    gate_pass = bool(aggregate.get("gate_pass", False)) and config.EXECUTION_COSTS_VERIFIED
    threshold = float(bundle.get("conf_threshold", 1.0))
    threshold_met = pred is not None and pred != 0 and float(proba[pred]) >= threshold
    actionable = config.EXECUTION_ENABLED and gate_pass and threshold_met and evidence_lean == model_lean
    chart = get_chart_candles(tf)
    if tf == "1m":
        live_candles = feed.candles_tail(3)
        if len(live_candles):
            chart = (pd.concat([chart[chart["timestamp"] < int(live_candles["timestamp"].min())],
                                live_candles]).drop_duplicates("timestamp", keep="last")
                     .sort_values("timestamp").reset_index(drop=True))
    chart = chart.tail({"1m": 2880, "5m": 1152, "15m": 672, "1h": 336, "4h": 180}[tf])
    chart_rows = chart.to_dict(orient="records")
    signals = []
    if not quality_errors and SIGNALS_LOG.exists():
        try:
            history = pd.read_json(SIGNALS_LOG, lines=True)
            if not history.empty and "horizon" in history:
                bar_ms = TIMEFRAMES[tf]
                history = history[(history["horizon"] == horizon) &
                                  history["pred"].isin(["LONG", "SHORT"])].copy()
                if not history.empty:
                    history["bucket"] = (history["time"] // bar_ms) * bar_ms
                    history = history.drop_duplicates("bucket", keep="last")
                    lower = int(chart["timestamp"].iloc[0]) if not chart.empty else 0
                    upper = int(chart["timestamp"].iloc[-1]) + bar_ms if not chart.empty else 0
                    history = history[(history["bucket"] >= lower) & (history["bucket"] < upper)]
                    signals = history[["time", "price", "pred", "p_long", "p_short"]].to_dict("records")
        except Exception:
            signals = []
    price_change = float(feats["ret_15m"].iloc[-1])
    atr = float(feats["atr_pct"].iloc[-1])
    atr_percentile = float((feats["atr_pct"] < atr).mean() * 100)

    min_atr_pct = float(getattr(config, "MIN_ATR_PERCENTILE", 25.0))
    chop_enabled = bool(getattr(config, "CHOP_FILTER_ENABLED", True))
    is_chop = bool(chop_enabled and atr_percentile < min_atr_pct)

    if cross_info and is_chop and cross_info.get("agree"):
        cross_info["chop_filtered"] = True
        cross_info["why"] = (cross_info.get("why") or "") + f" [Suppressed: ATR percentile {atr_percentile:.1f}% < {min_atr_pct:.1f}%]"

    chop_info = {
        "enabled": chop_enabled,
        "min_atr_percentile": min_atr_pct,
        "current_percentile": atr_percentile,
        "is_chop": is_chop,
        "status": "CHOP (Filtering active)" if is_chop else "EXPANDED (Clear to trade)",
    }
    if is_chop:
        actionable = False

    context_data = {**context}
    flow = feed.taker_flow(15)
    if flow.get("buy_share") is not None:
        snap.setdefault("taker_buy_share_15m", flow["buy_share"])
        snap.setdefault("taker_buy_size_15m", flow["buy_sz"])
        snap.setdefault("taker_sell_size_15m", flow["sell_sz"])
    now_ms = int(time.time() * 1000)
    if not quality_errors:
        with SIGNALS_LOG.open("a", encoding="utf-8") as signal_file:
            signal_file.write(json.dumps({
            "time": now_ms, "price": price,
            "p_long": float(proba[1]), "p_flat": float(proba[0]),
            "p_short": float(proba[2]), "pred": CLASS_NAMES[pred],
            "threshold": threshold, "horizon": horizon,
            }) + "\n")

    snap_for_paper = {
        "model_available": not bool(quality_errors),
        "cross_horizon": cross_info,
        "chop_filter": chop_info,
        "atr_pct": atr,
        "horizon_min": horizon,
        "probabilities": {"flat": float(proba[0]), "long": float(proba[1]), "short": float(proba[2])},
    }
    try:
        paper_trader.update_market(price, snap_for_paper, now_ms=now_ms)
        paper_summary = paper_trader.get_summary(price)
    except Exception as exc:
        paper_summary = {"enabled": False, "error": str(exc)}
    return clean({
        "updated_at": now_ms, "refresh_seconds": config.DASHBOARD_REFRESH_S,
        "coin": config.COIN, "tf": tf, "hz": hz, "horizon_min": horizon,
        "execution_venue": config.EXECUTION_VENUE,
        "execution_enabled": config.EXECUTION_ENABLED,
        "reference_source": config.LIVE_REFERENCE_SOURCE,
        "historical_source": config.HISTORICAL_SOURCE,
        "venue_validation_reason": config.VENUE_VALIDATION_REASON,
        "price": price, "price_change_15m": price_change,
        "funding": context_data.get("funding"),
        "open_interest_usd": context_data.get("open_interest_usd"),
        "atr_pct": atr, "atr_percentile": atr_percentile,
        "tick_age_ms": tick_age,
        "feed_connected": feed.is_connected,
        "flow": {"buy_share": flow.get("buy_share"), "buy_size": flow.get("buy_sz"),
                 "sell_size": flow.get("sell_sz")},
        "decision": "STAND ASIDE" if not actionable else model_lean,
        "model_available": not bool(quality_errors), "data_quality_errors": quality_errors,
        "model_id": bundle.get("model_id"), "model_trained_at": bundle.get("trained_at"),
        "model_target": bundle.get("target"), "probability_kind": bundle.get("probability_kind", "raw"),
        "research_only": bundle.get("research_only", False),
        "model_lean": model_lean, "evidence_lean": evidence_lean,
        "evidence_tally": tally, "model_probability": float(proba[pred]) if pred is not None else 0.0,
        "probabilities": {"flat": float(proba[0]), "long": float(proba[1]), "short": float(proba[2])},
        "gate_pass": gate_pass, "threshold": threshold,
        "qualifying_trades": int(aggregate.get("n_trades", 0) or 0),
        "gate_reason": "Walk-forward gate failed; probabilities are not validated confidence.",
        "evidence": evidence.to_dict(orient="records"),
        "cross_horizon": cross_info,
        "chop_filter": chop_info,
        "paper_trading": paper_summary,
        "market_context": {
            "atr_percentile": atr_percentile,
            "range_position_1h": float(feats["range_pos_60"].iloc[-1]),
            "rsi14": float(feats["rsi14"].iloc[-1]),
            "ret_1h": float(feats["ret_1h"].iloc[-1]),
            "ret_4h": float(feats["ret_4h"].iloc[-1]),
            "btc_ret_15m": float(feats["btc_ret_15m"].iloc[-1]),
            "oi_change_1h": snap.get("oi_chg_1h")},
        "candles": chart_rows, "signals": signals,
        "wallet": settings["sol_wallet"],
    })


def score_signals(sol: pd.DataFrame, horizon_min: int):
    try:
        history = pd.read_json(SIGNALS_LOG, lines=True)
    except Exception:
        return None
    if history.empty:
        return None
    ts, close = sol["timestamp"].to_numpy(), sol["close"].to_numpy()
    cutoff = int(time.time() * 1000) - horizon_min * 60_000
    rows = []
    for _, row in history.dropna(subset=["time", "price", "p_long", "p_short"]).iterrows():
        if int(row["time"]) > cutoff:
            continue
        i = bisect.bisect_left(ts, int(row["time"]) + horizon_min * 60_000)
        if i < len(ts):
            rows.append({"time": pd.Timestamp(int(row["time"]), unit="ms", tz="UTC"),
                         "pred": row.get("pred", ""), "p_long": row["p_long"],
                         "p_short": row["p_short"], "price": row["price"],
                         "px_after": float(close[i]),
                         "chg": float(close[i]) / float(row["price"]) - 1})
    return pd.DataFrame(rows) if rows else None


def account_activity(address: str) -> dict:
    if not address:
        return {"positions": None, "trades": None, "error": "No public wallet address configured."}
    if not re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", address):
        return {"positions": None, "trades": None,
                "error": "Wallet must be a valid Solana public address."}
    def get(endpoint):
        r = requests.get(f"{config.JUPITER_PERPS_API_URL}/{endpoint}",
                         params={"walletAddress": address}, timeout=10)
        r.raise_for_status()
        return r.json()
    try:
        positions = cached(f"jupiter-positions-{address}", config.DASHBOARD_CACHE_TTL,
                           lambda: get("positions"))
    except Exception:
        positions = None
    try:
        trades = cached(f"jupiter-trades-{address}", config.DASHBOARD_CACHE_TTL,
                        lambda: get("trades"))
    except Exception:
        trades = None
    return {"positions": positions, "trades": trades}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.log_date_time_string()} {fmt % args}", flush=True)

    def send_json(self, value, status=200):
        body = json.dumps(clean(value), separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return self.send_json({"ok": True})
        if parsed.path == "/api/settings":
            return self.send_json(load_settings())
        if parsed.path == "/api/snapshot":
            try:
                return self.send_json(market_snapshot(load_settings()))
            except Exception as exc:
                return self.send_json({"error": str(exc)}, 503)
        if parsed.path == "/api/account":
            address = parse_qs(parsed.query).get("address", [load_settings()["sol_wallet"]])[0]
            return self.send_json(account_activity(address))
        if parsed.path == "/api/details":
            try:
                sol, *_ = get_data()
                _, horizon = HORIZONS[load_settings()["hz"]]
                scored = score_signals(sol, horizon)
                return self.send_json({
                    "scored": scored.tail(100).to_dict("records") if scored is not None else [],
                    "horizon_min": horizon,
                })
            except Exception as exc:
                return self.send_json({"error": str(exc)}, 503)
        if parsed.path == "/api/paper_trades":
            feed = get_feed()
            price = 0.0
            tick = feed.latest_mid()
            if tick and feed.mid_age_ms() is not None and feed.mid_age_ms() < 15_000:
                price = float(tick[1])
            return self.send_json(paper_trader.get_summary(current_price=price))
        relative = "index.html" if parsed.path in ("", "/") else parsed.path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        root = WEB_ROOT.resolve()
        if root not in target.parents and target != root:
            return self.send_error(404)
        if not target.is_file():
            return self.send_json({"error": "Not found"}, 404)
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }.get(target.suffix.lower(), "application/octet-stream")
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/paper_trades/reset":
            paper_trader.reset()
            return self.send_json({"ok": True, "message": "Paper trading state reset successfully."})
        if parsed.path != "/api/settings":
            return self.send_json({"error": "Not found"}, 404)
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            payload = json.loads(self.rfile.read(length))
            return self.send_json(save_settings(payload))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self.send_json({"error": str(exc)}, 400)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SOL-PERP native dashboard: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()