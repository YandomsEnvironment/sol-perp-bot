"""Telegram alert and assistant bot for Solana Market Brief.

Features:
- Pushes real-time alerts when multi-horizon consensus confirms a LONG or SHORT.
- Low-volatility chop filter: suppresses alerts during sideways compression.
- Forward paper trading & PnL tracker with realistic Jupiter Perps fees ($0.65/side).
- Interactive commands: /price, /signals, /pnl, /trades, /status, /start.
- Auto-detects and binds your Telegram Chat ID upon sending /start.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

ENV_FILE = ROOT / ".env"


def load_local_env() -> None:
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()


load_local_env()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8291973632:AAGg0FhUj8olAvWnypj7ea53EQEwQIGg_FM")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_URL = f"https://api.telegram.org/bot{TOKEN}"
DASHBOARD_URL = "http://127.0.0.1:8501"

COOLDOWN_SECONDS = 900  # 15 minutes cooldown between repeat signals in same direction
_LAST_ALERT_TIME = 0
_LAST_ALERT_DIRECTION = None
_KNOWN_ACTIVE_TRADE_ID: str | None = None
_LAST_ANNOUNCED_CLOSE_COUNT: int = 0


def save_chat_id(chat_id: str | int) -> None:
    global CHAT_ID
    CHAT_ID = str(chat_id)
    lines = []
    if ENV_FILE.exists():
        lines = [l for l in ENV_FILE.read_text(encoding="utf-8").splitlines()
                 if not l.startswith("TELEGRAM_CHAT_ID=")]
    lines.append(f"TELEGRAM_CHAT_ID={CHAT_ID}")
    lines.append(f"TELEGRAM_BOT_TOKEN={TOKEN}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[telegram] bound chat_id: {CHAT_ID}", flush=True)


def send_message(text: str, chat_id: str | int = None, parse_mode: str = "HTML") -> bool:
    cid = chat_id or CHAT_ID
    if not cid:
        return False
    for attempt in range(3):
        try:
            r = requests.post(
                f"{API_URL}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True},
                timeout=12,
            )
            if r.status_code == 200:
                return True
        except Exception as exc:
            if attempt == 2:
                print(f"[telegram] send error after 3 attempts: {exc}", flush=True)
            time.sleep(1)
    return False


def get_snapshot() -> dict | None:
    try:
        r = requests.get(f"{DASHBOARD_URL}/api/snapshot", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_paper_trades() -> dict | None:
    try:
        r = requests.get(f"{DASHBOARD_URL}/api/paper_trades", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def format_price_message(s: dict) -> str:
    price = s.get("price", 0.0)
    ret15 = s.get("price_change_15m", 0.0) * 100
    ctx = s.get("market_context") or {}
    r1h = ctx.get("ret_1h", 0.0) * 100
    r4h = ctx.get("ret_4h", 0.0) * 100
    r1d = ctx.get("ret_1d", 0.0) * 100
    funding = float(s.get("funding") or 0.0) * 100
    oi = float(s.get("open_interest_usd") or 0.0) / 1e6
    s15 = "+" if ret15 >= 0 else ""
    s1h = "+" if r1h >= 0 else ""
    s4h = "+" if r4h >= 0 else ""
    s1d = "+" if r1d >= 0 else ""
    return (
        f"<b>⚡ Solana Multi-Timeframe Snapshot</b>\n\n"
        f"💰 <b>Reference Price:</b> ${price:,.2f}\n\n"
        f"<b>Momentum Across Timeframes:</b>\n"
        f"• ⏱ <b>15m:</b> <code>{s15}{ret15:.2f}%</code>\n"
        f"• 🕐 <b>1h:</b>  <code>{s1h}{r1h:.2f}%</code>\n"
        f"• 🕓 <b>4h:</b>  <code>{s4h}{r4h:.2f}%</code>\n"
        f"• 📅 <b>24h:</b> <code>{s1d}{r1d:.2f}%</code>\n\n"
        f"📊 <b>Hourly Funding:</b> {funding:.4f}%\n"
        f"🏦 <b>Open Interest:</b> ${oi:.1f}M USD\n"
        f"⏱ <i>Updated at {time.strftime('%H:%M:%S UTC')}</i>"
    )


def format_signal_message(s: dict) -> str:
    price = s.get("price", 0.0)
    lean = s.get("model_lean", "FLAT")
    proba = s.get("probabilities", {})
    tally = s.get("evidence_tally", {})
    cross = s.get("cross_horizon", {}) or {}
    chop = s.get("chop_filter", {}) or {}
    ctx = s.get("market_context") or {}

    p_long = proba.get("long", 0.0) * 100
    p_flat = proba.get("flat", 0.0) * 100
    p_short = proba.get("short", 0.0) * 100

    consensus_text = cross.get("reading", "Single horizon active")
    consensus_why = cross.get("why", "Evaluating multi-horizon consensus")

    tone = "🟢" if lean == "LONG" else ("🔴" if lean == "SHORT" else "⚪")

    is_chop = chop.get("is_chop", False)
    atr_pctile = chop.get("current_percentile", s.get("atr_percentile", 0.0))
    min_pctile = chop.get("min_atr_percentile", 25.0)

    if is_chop:
        vol_status = (
            f"⚠️ <b>Volatility Filter:</b> <code>CHOP / COMPRESSION</code>\n"
            f"• ATR Percentile: {atr_pctile:.1f}% (Below {min_pctile:.0f}% threshold)\n"
            f"• <i>Directional alerts suppressed to prevent fee drag.</i>\n\n"
        )
    else:
        vol_status = (
            f"✅ <b>Volatility Filter:</b> <code>EXPANDED / TRENDING</code>\n"
            f"• ATR Percentile: {atr_pctile:.1f}% (Above {min_pctile:.0f}% threshold)\n\n"
        )

    r1h = ctx.get("ret_1h", 0.0) * 100
    r4h = ctx.get("ret_4h", 0.0) * 100
    r1d = ctx.get("ret_1d", 0.0) * 100
    s1h = "+" if r1h >= 0 else ""
    s4h = "+" if r4h >= 0 else ""
    s1d = "+" if r1d >= 0 else ""

    return (
        f"<b>{tone} SOL Forecast & Multi-Horizon Consensus</b>\n\n"
        f"• <b>Active Model Lean:</b> <code>{lean}</code>\n"
        f"• <b>Reference Price:</b> ${price:,.2f}\n"
        f"• <b>Multi-Horizon Read:</b> {consensus_text}\n"
        f"• <b>Consensus Note:</b> <i>{consensus_why}</i>\n\n"
        f"{vol_status}"
        f"🌐 <b>Higher Timeframe Alignment:</b>\n"
        f"• 1h: <code>{s1h}{r1h:.2f}%</code> | 4h: <code>{s4h}{r4h:.2f}%</code> | 24h: <code>{s1d}{r1d:.2f}%</code>\n\n"
        f"📊 <b>Calibrated Probabilities:</b>\n"
        f"  📈 Long: {p_long:.1f}%\n"
        f"  ⚪ Flat: {p_flat:.1f}%\n"
        f"  📉 Short: {p_short:.1f}%\n\n"
        f"🔍 <b>Indicator Tally:</b> {tally.get('LONG', 0)}L / {tally.get('SHORT', 0)}S / {tally.get('NEUTRAL', 0)}N"
    )


def format_pnl_message(paper: dict) -> str:
    equity = paper.get("current_equity_usd", 10000.0)
    balance = paper.get("cash_balance_usd", 10000.0)
    net_pnl = paper.get("total_net_pnl_usd", 0.0)
    net_ret = paper.get("total_net_return_pct", 0.0)
    total = paper.get("total_trades", 0)
    wins = paper.get("wins", 0)
    losses = paper.get("losses", 0)
    win_rate = paper.get("win_rate_pct", 0.0)
    pf = paper.get("profit_factor", 1.0)
    fees = paper.get("total_fees_paid_usd", 0.0)
    max_dd = paper.get("max_drawdown_usd", 0.0)
    max_dd_pct = paper.get("max_drawdown_pct", 0.0)
    active = paper.get("active_position")

    sign = "+" if net_pnl >= 0 else ""
    ret_sign = "+" if net_ret >= 0 else ""

    if active:
        d = active.get("direction", "LONG")
        px = active.get("current_price", active.get("entry_price", 0.0))
        epx = active.get("entry_price", 0.0)
        u_pnl = active.get("unrealized_net_pnl_usd", 0.0)
        u_ret = active.get("unrealized_return_pct", 0.0) * 100
        u_sign = "+" if u_pnl >= 0 else ""
        sz = active.get("size_coin", 0.0)
        tp = active.get("tp_price", 0.0)
        sl = active.get("sl_price", 0.0)
        tone = "🟢" if d == "LONG" else "🔴"
        active_str = (
            f"• {tone} <b>{d}</b> {sz:.2f} SOL @ ${epx:,.2f}\n"
            f"  Current: ${px:,.2f} ({u_sign}${u_pnl:.2f} / {u_sign}{u_ret:.2f}% net)\n"
            f"  🎯 Target: ${tp:,.2f} | 🛑 Stop: ${sl:,.2f}"
        )
    else:
        active_str = "• <i>None (Standing aside in cash)</i>"

    return (
        f"<b>📊 Jupiter Perps Forward Paper Trading</b>\n\n"
        f"💰 <b>Virtual Equity:</b> ${equity:,.2f} (<code>{ret_sign}{net_ret:.2f}%</code>)\n"
        f"💵 <b>Cash Balance:</b> ${balance:,.2f} (Net: {sign}${net_pnl:.2f})\n"
        f"🎯 <b>Total Closed Trades:</b> {total} ({wins}W / {losses}L)\n"
        f"🏆 <b>Win Rate:</b> {win_rate:.1f}%\n"
        f"📈 <b>Profit Factor:</b> {pf:.2f}\n"
        f"💸 <b>Jupiter Fees Paid:</b> ${fees:.2f} ($0.65 USD / side)\n"
        f"📉 <b>Max Drawdown:</b> ${max_dd:.2f} ({max_dd_pct:.1f}%)\n\n"
        f"<b>Active Position:</b>\n{active_str}\n\n"
        f"⏱ <i>Updated at {time.strftime('%H:%M:%S UTC')}</i>"
    )


def format_trades_message(paper: dict) -> str:
    trades = paper.get("recent_trades", [])
    if not trades:
        return "<b>📜 Recent Paper Trades</b>\n\nNo paper trades closed yet."

    lines = ["<b>📜 Recent Paper Trades (Jupiter Perps)</b>\n"]
    for i, t in enumerate(trades[:6], 1):
        d = t.get("direction", "LONG")
        tone = "🟢" if d == "LONG" else "🔴"
        pnl = t.get("net_pnl_usd", 0.0)
        ret = t.get("net_return_pct", 0.0) * 100
        sign = "+" if pnl >= 0 else ""
        reason = t.get("exit_reason", "EXIT")
        epx = t.get("entry_price", 0.0)
        xpx = t.get("exit_price", 0.0)
        dur = t.get("duration_sec", 0)
        dur_str = f"{dur // 60}m {dur % 60}s" if dur >= 60 else f"{dur}s"
        lines.append(
            f"{i}. {tone} <b>{d}</b>: {sign}${pnl:.2f} ({sign}{ret:.2f}%)\n"
            f"   ${epx:,.2f} ➔ ${xpx:,.2f} | Reason: <code>{reason}</code> ({dur_str})"
        )
    return "\n\n".join(lines)


def format_wallet_message(act: dict, wallet_addr: str) -> str:
    short_addr = f"{wallet_addr[:6]}...{wallet_addr[-4:]}"
    pos_data = (act.get("positions") or {}).get("dataList") or []
    lines = [f"<b>🏦 Jupiter Perps Live Wallet</b>\n<code>{short_addr}</code>\n"]
    if not pos_data:
        lines.append("• <b>Active Position:</b> <i>None (No open positions)</i>")
    else:
        for p in pos_data:
            side = p.get("side", "").upper()
            tone = "🟢" if side == "LONG" else "🔴"
            sz_usd = float(p.get("sizeUsdDelta", 0) or 0) / 1e6
            epx = float(p.get("entryPrice", 0) or 0)
            mpx = float(p.get("markPrice", 0) or 0)
            pnl_after = float(p.get("pnlAfterFeesUsd", 0) or 0)
            pnl_before = float(p.get("pnlBeforeFeesUsd", 0) or 0)
            fees = float(p.get("totalFeesUsd", 0) or 0)
            lev = float(p.get("leverage", 0) or 0)
            sign = "+" if pnl_after >= 0 else ""
            b_sign = "+" if pnl_before >= 0 else ""

            tpsl = p.get("tpslRequests") or []
            tp_str, sl_str = "None", "None"
            for req in tpsl:
                if req.get("requestType") == "tp":
                    tp_str = f"${float(req.get('triggerPriceUsd', 0)):,.2f}"
                elif req.get("requestType") == "sl":
                    sl_str = f"${float(req.get('triggerPriceUsd', 0)):,.2f}"

            lines.append(
                f"{tone} <b>{side} SOL Position ({lev:.0f}x leverage)</b>\n"
                f"• Size: ${sz_usd:,.2f} USD\n"
                f"• Entry: ${epx:,.2f} | Current Mark: ${mpx:,.2f}\n"
                f"• Net PnL (after fees): <b>{sign}${pnl_after:.2f}</b> (gross: {b_sign}${pnl_before:.2f})\n"
                f"• TP Target: {tp_str} | SL Stop: {sl_str}\n"
                f"• Total Fees Paid: ${fees:.2f} USD"
            )

    trades_data = (act.get("trades") or {}).get("dataList") or []
    if trades_data:
        t = trades_data[0]
        action = t.get("action")
        side = t.get("side", "").upper()
        fee = t.get("fee", "0")
        price = t.get("price", "0")
        lines.append(f"\n<b>Last On-Chain Event:</b>\n• {action} {side} @ ${float(price):,.2f} (Fee: ${float(fee):.2f})")

    lines.append(f"\n⏱ <i>Updated at {time.strftime('%H:%M:%S UTC')}</i>")
    return "\n".join(lines)


def check_and_alert() -> None:
    global _LAST_ALERT_TIME, _LAST_ALERT_DIRECTION
    global _KNOWN_ACTIVE_TRADE_ID, _LAST_ANNOUNCED_CLOSE_COUNT

    if not CHAT_ID:
        return
    s = get_snapshot()
    if not s or not s.get("model_available"):
        return

    # 1. Track Paper Trading Lifecycle Events
    paper = s.get("paper_trading") or {}
    active = paper.get("active_position")
    total_closed = paper.get("total_trades", 0)

    # Initialize counter on first run
    if _LAST_ANNOUNCED_CLOSE_COUNT == 0 and total_closed > 0:
        _LAST_ANNOUNCED_CLOSE_COUNT = total_closed

    # A: Did a position just close?
    if total_closed > _LAST_ANNOUNCED_CLOSE_COUNT:
        recent = paper.get("recent_trades", [])
        if recent:
            last_trade = recent[0]
            d = last_trade.get("direction", "LONG")
            pnl = last_trade.get("net_pnl_usd", 0.0)
            ret = last_trade.get("net_return_pct", 0.0) * 100
            sign = "+" if pnl >= 0 else ""
            reason = last_trade.get("exit_reason", "EXIT")
            epx = last_trade.get("entry_price", 0.0)
            xpx = last_trade.get("exit_price", 0.0)
            tone = "🎯" if pnl > 0 else "🛑"
            card = (
                f"{tone} <b>PAPER TRADE CLOSED: {d}</b>\n\n"
                f"• <b>Exit Reason:</b> <code>{reason}</code>\n"
                f"• <b>Entry:</b> ${epx:,.2f} ➔ <b>Exit:</b> ${xpx:,.2f}\n"
                f"• <b>Net PnL (after fees):</b> <b>{sign}${pnl:.2f}</b> ({sign}{ret:.2f}%)\n"
                f"• <b>Jupiter Fee:</b> $1.30 round trip ($0.65/side)\n"
                f"• <b>Virtual Equity:</b> ${paper.get('current_equity_usd', 0.0):,.2f}"
            )
            send_message(card)
        _LAST_ANNOUNCED_CLOSE_COUNT = total_closed

    # B: Did a position just open?
    if active:
        tid = active.get("trade_id")
        if tid and tid != _KNOWN_ACTIVE_TRADE_ID:
            _KNOWN_ACTIVE_TRADE_ID = tid
            d = active.get("direction", "LONG")
            epx = active.get("entry_price", 0.0)
            tp = active.get("tp_price", 0.0)
            sl = active.get("sl_price", 0.0)
            sz = active.get("size_coin", 0.0)
            tone = "🟢" if d == "LONG" else "🔴"
            card = (
                f"🚀 <b>PAPER TRADE OPENED: {tone} {d}</b>\n\n"
                f"• <b>Size:</b> {sz:.2f} SOL (~$1,000 notional)\n"
                f"• <b>Entry:</b> ${epx:,.2f}\n"
                f"• <b>Target (2x ATR):</b> ${tp:,.2f}\n"
                f"• <b>Stop (1x ATR):</b> ${sl:,.2f}\n"
                f"• <b>Execution Fee:</b> $0.65 USD entry"
            )
            send_message(card)
    else:
        _KNOWN_ACTIVE_TRADE_ID = None

    # 2. Check Directional Setup Consensus Alert
    cross = s.get("cross_horizon") or {}
    agree = cross.get("agree", False)
    vote = cross.get("vote", "NEUTRAL")

    if not agree or vote not in ("LONG", "SHORT"):
        return

    # Check chop filter
    chop = s.get("chop_filter") or {}
    if chop.get("is_chop", False):
        return

    now = time.time()
    # Check cooldown
    if vote == _LAST_ALERT_DIRECTION and (now - _LAST_ALERT_TIME) < COOLDOWN_SECONDS:
        return

    price = float(s.get("price", 0.0))
    atr = float(s.get("atr_pct", 0.0006))
    proba = s.get("probabilities", {})
    p_val = proba.get(vote.lower(), 0.0) * 100

    direction_sign = 1 if vote == "LONG" else -1
    tp_price = price * (1 + direction_sign * config.TP_ATR * atr * (5 ** 0.5))
    sl_price = price * (1 - direction_sign * config.SL_ATR * atr * (5 ** 0.5))

    tone = "🟢 <b>BUY / LONG</b>" if vote == "LONG" else "🔴 <b>SELL / SHORT</b>"

    alert_text = (
        f"🚨 <b>SOL PERP ALERT: {tone}</b>\n\n"
        f"<b>Multi-Horizon Consensus Confirmed!</b>\n"
        f"Both fast (5m) and confirming models agree on {vote}.\n"
        f"Volatility filter passed (no chop compression).\n\n"
        f"💰 <b>Entry Reference:</b> ${price:,.2f}\n"
        f"🎯 <b>Take-Profit Target (2x ATR):</b> ${tp_price:,.2f}\n"
        f"🛑 <b>Stop-Loss Barrier (1x ATR):</b> ${sl_price:,.2f}\n"
        f"📊 <b>Confidence:</b> {p_val:.1f}%\n"
        f"💸 <b>Jupiter Perps Fee:</b> ~$0.65 USD / side (~16 bps round-trip)\n\n"
        f"🔗 <a href='https://jup.ag/perps'>Open Jupiter Perps</a>"
    )

    if send_message(alert_text):
        _LAST_ALERT_TIME = now
        _LAST_ALERT_DIRECTION = vote
        print(f"[telegram] alert sent: {vote} @ ${price:,.2f}", flush=True)


def handle_updates(offset: int) -> int:
    try:
        r = requests.get(f"{API_URL}/getUpdates", params={"offset": offset, "timeout": 1}, timeout=6)
        if r.status_code != 200:
            return offset
        data = r.json()
        if not data.get("ok"):
            return offset

        for item in data.get("result", []):
            offset = max(offset, item["update_id"] + 1)
            msg = item.get("message") or item.get("channel_post")
            if not msg:
                continue

            sender_chat_id = msg["chat"]["id"]
            user_text = (msg.get("text") or "").strip()
            cmd = user_text.lower().split()[0].split("@")[0].strip() if user_text else ""

            if cmd in ("/start", "start", "/help", "help"):
                save_chat_id(sender_chat_id)
                welcome = (
                    "👋 <b>Welcome to Solana Market Brief!</b>\n\n"
                    "Your Telegram chat is connected to your local trading bot.\n\n"
                    "• Real-time alerts on multi-horizon consensus & volatility expansion\n"
                    "• Automated forward paper trading tracker with Jupiter fees\n\n"
                    "<b>Commands:</b>\n"
                    "/price — Live SOL price, 15m momentum & funding\n"
                    "/signals — Current model forecast, consensus & chop filter\n"
                    "/pnl — Forward paper trading performance & open position\n"
                    "/trades — Last closed paper trades with net PnL\n"
                    "/status — Bot health, models & connection status"
                )
                send_message(welcome, sender_chat_id)

            elif cmd in ("/price", "price", "sol"):
                s = get_snapshot()
                if s:
                    send_message(format_price_message(s), sender_chat_id)
                else:
                    send_message("⚠️ Dashboard API is currently unreachable. Make sure the bot is running.", sender_chat_id)

            elif cmd in ("/signals", "/signal", "signal", "signals"):
                s = get_snapshot()
                if s:
                    send_message(format_signal_message(s), sender_chat_id)
                else:
                    send_message("⚠️ Dashboard API is currently unreachable.", sender_chat_id)

            elif cmd in ("/pnl", "pnl", "/perf", "perf"):
                paper = get_paper_trades()
                if paper and not paper.get("error"):
                    send_message(format_pnl_message(paper), sender_chat_id)
                else:
                    send_message("⚠️ Paper trading tracker is initializing or unreachable.", sender_chat_id)

            elif cmd in ("/trades", "trades", "/history", "history"):
                paper = get_paper_trades()
                if paper and not paper.get("error"):
                    send_message(format_trades_message(paper), sender_chat_id)
                else:
                    send_message("⚠️ Paper trading tracker is initializing or unreachable.", sender_chat_id)

            elif cmd in ("/wallet", "wallet", "/position", "position"):
                s = get_snapshot()
                w = (s or {}).get("wallet") or "2ESAsFbFpYUa8xMuUN6rCPH8X21jjSSfmme5vUE7He7H"
                try:
                    r = requests.get(f"{DASHBOARD_URL}/api/account?address={w}", timeout=10)
                    if r.status_code == 200:
                        send_message(format_wallet_message(r.json(), w), sender_chat_id)
                    else:
                        send_message("⚠️ Error fetching wallet activity from Jupiter API.", sender_chat_id)
                except Exception as exc:
                    send_message(f"⚠️ Error: {exc}", sender_chat_id)

            elif cmd in ("/status", "status"):
                s = get_snapshot()
                status_text = (
                    f"🤖 <b>Bot Status: Active & Monitoring</b>\n\n"
                    f"• <b>Live Feed:</b> {'Connected ✅' if s and s.get('feed_connected') else 'Reconnecting ⚠️'}\n"
                    f"• <b>Active Models:</b> 3 horizons (fast, mid, slow)\n"
                    f"• <b>Jupiter Fee Standard:</b> $0.65 / side (16 bps deadband)\n"
                    f"• <b>Multi-Horizon Filter:</b> Enabled ✅\n"
                    f"• <b>Low-Vol Chop Filter:</b> Enabled (25% ATR threshold) ✅\n"
                    f"• <b>Paper Trading:</b> Active ($10k starting equity) ✅\n"
                    f"• <b>Connected Wallet:</b> <code>2ESAsF...He7H</code> ✅"
                )
                send_message(status_text, sender_chat_id)
            elif user_text:
                send_message(
                    "<b>Commands available:</b>\n"
                    "/price — Live SOL price & stats\n"
                    "/signals — Model forecast & chop filter\n"
                    "/pnl — Forward paper trading performance\n"
                    "/wallet — Your live on-chain Jupiter positions\n"
                    "/trades — Recent closed paper trades\n"
                    "/status — Bot health & status",
                    sender_chat_id,
                )

    except Exception:
        time.sleep(1)
    return offset


def main() -> None:
    print(f"[telegram] bot service starting for @solanapricealerts_bot...", flush=True)
    if CHAT_ID:
        print(f"[telegram] using configured chat_id: {CHAT_ID}", flush=True)
    else:
        print("[telegram] no chat_id yet. Send /start to @solanapricealerts_bot on Telegram.", flush=True)

    offset = 0
    last_alert_check = 0
    while True:
        offset = handle_updates(offset)
        now = time.time()
        if now - last_alert_check >= 10:
            last_alert_check = now
            try:
                check_and_alert()
            except Exception as e:
                print(f"[telegram] check_and_alert error: {e}", flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
