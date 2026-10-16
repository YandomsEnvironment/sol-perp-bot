"""Paper trading engine and forward PnL tracker for Jupiter Perps.

Simulates execution against live reference price feeds with accurate fee accounting:
- Fixed Jupiter Perps pool execution fee: $0.65 USD per trade / side ($1.30 round trip)
- Realistic slippage: 2 bps per side
- Triple-barrier exits: 2x ATR Take-Profit, 1x ATR Stop-Loss, and max duration timeout
- Persistent state saved to data_store/paper_trading_state.json and paper_trades.jsonl
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import config

STATE_FILE = config.DATA_STORE / "paper_trading_state.json"
HISTORY_FILE = config.DATA_STORE / "paper_trades.jsonl"


def _iso_now(ms: int | None = None) -> str:
    t = (ms / 1000.0) if ms is not None else time.time()
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


class PaperTrader:
    def __init__(
        self,
        state_file: Path = STATE_FILE,
        history_file: Path = HISTORY_FILE,
        starting_balance: float = config.PAPER_STARTING_BALANCE_USD,
        position_notional: float = config.PAPER_POSITION_USD,
        fee_per_side: float = config.PAPER_FEE_USD,
        slippage_rate: float = config.PAPER_SLIPPAGE,
    ) -> None:
        self.state_file = Path(state_file)
        self.history_file = Path(history_file)
        self.starting_balance = float(starting_balance)
        self.position_notional = float(position_notional)
        self.fee_per_side = float(fee_per_side)
        self.slippage_rate = float(slippage_rate)

        self.cash_balance = self.starting_balance
        self.peak_equity = self.starting_balance
        self.active_position: dict[str, Any] | None = None
        self.closed_trades: list[dict[str, Any]] = []
        self._listeners: list[Callable[[str, dict[str, Any]], None]] = []

        self.load()

    def add_listener(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        self._listeners.append(callback)

    def _notify(self, event_type: str, trade: dict[str, Any]) -> None:
        for cb in self._listeners:
            try:
                cb(event_type, trade)
            except Exception as exc:
                print(f"[paper_trader] listener error: {exc}", flush=True)

    def reset(self) -> None:
        self.cash_balance = self.starting_balance
        self.peak_equity = self.starting_balance
        self.active_position = None
        self.closed_trades = []
        self.save()

    def load(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.starting_balance = float(data.get("starting_balance", self.starting_balance))
                self.cash_balance = float(data.get("cash_balance", self.starting_balance))
                self.peak_equity = float(data.get("peak_equity", self.cash_balance))
                self.active_position = data.get("active_position")
                self.closed_trades = data.get("closed_trades", [])
                return
            except Exception as exc:
                print(f"[paper_trader] error loading state: {exc}", flush=True)

        # Fallback to history file if present
        if self.history_file.exists():
            trades = []
            try:
                for line in self.history_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        trades.append(json.loads(line))
                self.closed_trades = trades
                net_sum = sum(t.get("net_pnl_usd", 0.0) for t in trades)
                self.cash_balance = self.starting_balance + net_sum
            except Exception:
                pass

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": int(time.time() * 1000),
            "starting_balance": self.starting_balance,
            "cash_balance": self.cash_balance,
            "peak_equity": self.peak_equity,
            "active_position": self.active_position,
            "closed_trades": self.closed_trades[-100:],  # keep last 100 in state json
        }
        tmp_file = self.state_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_file.replace(self.state_file)

    def _append_history(self, trade: dict[str, Any]) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trade) + "\n")

    def open_position(
        self,
        direction: str,
        price: float,
        atr_pct: float,
        horizon_min: int = 15,
        model_confidence: float = 0.0,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        if self.active_position is not None:
            raise RuntimeError("Position already open; close existing position first.")

        now_ms = now_ms or int(time.time() * 1000)
        direction = direction.upper()
        if direction not in ("LONG", "SHORT"):
            raise ValueError(f"Invalid direction: {direction}")

        # Horizon scaled ATR: 2x ATR(h) take profit, 1x ATR(h) stop loss
        h_mult = math.sqrt(horizon_min)
        atr_barrier = atr_pct * h_mult
        tp_pct = config.TP_ATR * atr_barrier
        sl_pct = config.SL_ATR * atr_barrier

        direction_sign = 1 if direction == "LONG" else -1
        tp_price = price * (1.0 + direction_sign * tp_pct)
        sl_price = price * (1.0 - direction_sign * sl_pct)

        max_hold_ms = horizon_min * config.PAPER_MAX_HOLD_BARS * 60 * 1000
        size_coin = self.position_notional / price if price > 0 else 0.0

        trade_id = f"PT-{direction}-{now_ms}"
        pos = {
            "trade_id": trade_id,
            "direction": direction,
            "entry_time": now_ms,
            "entry_time_iso": _iso_now(now_ms),
            "entry_price": price,
            "notional_usd": self.position_notional,
            "size_coin": size_coin,
            "atr_pct": atr_pct,
            "horizon_min": horizon_min,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "max_hold_ms": max_hold_ms,
            "expires_at": now_ms + max_hold_ms,
            "model_confidence": model_confidence,
            "entry_fee_usd": self.fee_per_side,
            "status": "OPEN",
        }
        self.active_position = pos
        self.save()
        self._notify("OPEN", pos)
        return pos

    def close_position(
        self,
        exit_price: float,
        reason: str,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        if self.active_position is None:
            raise RuntimeError("No active position to close.")

        now_ms = now_ms or int(time.time() * 1000)
        pos = self.active_position
        direction = pos["direction"]
        entry_price = float(pos["entry_price"])
        notional = float(pos["notional_usd"])

        # Price return
        if direction == "LONG":
            gross_return_pct = (exit_price / entry_price) - 1.0
        else:
            gross_return_pct = 1.0 - (exit_price / entry_price)

        gross_pnl_usd = gross_return_pct * notional

        # Jupiter Perps realistic execution costs:
        # $0.65 entry + $0.65 exit = $1.30 fixed fee
        # plus slippage (0.02% per side)
        slippage_cost = notional * (2 * self.slippage_rate)
        total_fees_usd = (2 * self.fee_per_side) + slippage_cost
        net_pnl_usd = gross_pnl_usd - total_fees_usd
        net_return_pct = net_pnl_usd / notional

        duration_sec = max(0, int((now_ms - pos["entry_time"]) / 1000))

        closed_record = {
            "trade_id": pos["trade_id"],
            "direction": direction,
            "entry_time": pos["entry_time"],
            "entry_time_iso": pos["entry_time_iso"],
            "entry_price": entry_price,
            "exit_time": now_ms,
            "exit_time_iso": _iso_now(now_ms),
            "exit_price": exit_price,
            "exit_reason": reason,
            "duration_sec": duration_sec,
            "notional_usd": notional,
            "size_coin": pos["size_coin"],
            "tp_price": pos["tp_price"],
            "sl_price": pos["sl_price"],
            "gross_return_pct": gross_return_pct,
            "gross_pnl_usd": gross_pnl_usd,
            "fixed_fees_usd": 2 * self.fee_per_side,
            "slippage_cost_usd": slippage_cost,
            "total_fees_usd": total_fees_usd,
            "net_pnl_usd": net_pnl_usd,
            "net_return_pct": net_return_pct,
            "model_confidence": pos.get("model_confidence", 0.0),
            "is_win": net_pnl_usd > 0,
        }

        self.cash_balance += net_pnl_usd
        self.closed_trades.append(closed_record)
        self._append_history(closed_record)
        self.active_position = None
        self.save()

        self._notify("CLOSE", closed_record)
        return closed_record

    def update_market(
        self,
        current_price: float,
        snapshot: dict[str, Any],
        now_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Evaluates active position exits and new consensus entries."""
        now_ms = now_ms or int(time.time() * 1000)

        # 1. If we have an active position, check for exit barriers
        if self.active_position is not None:
            pos = self.active_position
            direction = pos["direction"]
            entry_price = float(pos["entry_price"])
            tp_price = float(pos["tp_price"])
            sl_price = float(pos["sl_price"])
            expires_at = int(pos["expires_at"])

            cross = snapshot.get("cross_horizon") or {}
            consensus_vote = cross.get("vote", "NEUTRAL")
            consensus_agree = cross.get("agree", False)

            exit_reason = None
            if direction == "LONG":
                if current_price >= tp_price:
                    exit_reason = "TAKE_PROFIT"
                elif current_price <= sl_price:
                    exit_reason = "STOP_LOSS"
                elif now_ms >= expires_at:
                    exit_reason = "HORIZON_TIMEOUT"
                elif consensus_agree and consensus_vote == "SHORT":
                    exit_reason = "CONSENSUS_REVERSAL"
            else:  # SHORT
                if current_price <= tp_price:
                    exit_reason = "TAKE_PROFIT"
                elif current_price >= sl_price:
                    exit_reason = "STOP_LOSS"
                elif now_ms >= expires_at:
                    exit_reason = "HORIZON_TIMEOUT"
                elif consensus_agree and consensus_vote == "LONG":
                    exit_reason = "CONSENSUS_REVERSAL"

            if exit_reason:
                return self.close_position(current_price, exit_reason, now_ms)
            return None

        # 2. If NO active position, check entry conditions
        if not getattr(config, "PAPER_TRADING_ENABLED", True):
            return None

        if not snapshot.get("model_available"):
            return None

        # Check chop filter
        chop = snapshot.get("chop_filter") or {}
        if chop.get("is_chop", False):
            return None

        cross = snapshot.get("cross_horizon") or {}
        if not cross.get("agree", False):
            return None

        vote = cross.get("vote")
        if vote not in ("LONG", "SHORT"):
            return None

        atr_pct = float(snapshot.get("atr_pct", 0.001))
        horizon_min = int(snapshot.get("horizon_min", 15))
        proba = snapshot.get("probabilities", {})
        conf = float(proba.get(vote.lower(), 0.0))

        return self.open_position(
            direction=vote,
            price=current_price,
            atr_pct=atr_pct,
            horizon_min=horizon_min,
            model_confidence=conf,
            now_ms=now_ms,
        )

    def get_summary(self, current_price: float | None = None) -> dict[str, Any]:
        trades = self.closed_trades
        n_trades = len(trades)
        wins = sum(1 for t in trades if t["is_win"])
        losses = sum(1 for t in trades if not t["is_win"])
        win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0

        gross_pnl = sum(t["gross_pnl_usd"] for t in trades)
        total_fees = sum(t["total_fees_usd"] for t in trades)
        net_pnl = sum(t["net_pnl_usd"] for t in trades)

        gross_gains = sum(t["gross_pnl_usd"] for t in trades if t["gross_pnl_usd"] > 0)
        gross_losses = sum(abs(t["gross_pnl_usd"]) for t in trades if t["gross_pnl_usd"] < 0)
        profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else (gross_gains if gross_gains > 0 else 1.0)

        # Drawdown calculation
        equity_curve = [self.starting_balance]
        running = self.starting_balance
        max_dd_usd = 0.0
        max_dd_pct = 0.0
        peak = self.starting_balance

        for t in trades:
            running += t["net_pnl_usd"]
            equity_curve.append(running)
            if running > peak:
                peak = running
            dd_usd = peak - running
            dd_pct = (dd_usd / peak * 100.0) if peak > 0 else 0.0
            if dd_usd > max_dd_usd:
                max_dd_usd = dd_usd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        active = None
        unrealized_net = 0.0
        if self.active_position is not None:
            pos = dict(self.active_position)
            px = current_price or pos["entry_price"]
            entry = float(pos["entry_price"])
            notional = float(pos["notional_usd"])
            direction = pos["direction"]

            if direction == "LONG":
                u_ret = (px / entry) - 1.0
            else:
                u_ret = 1.0 - (px / entry)

            u_gross_pnl = u_ret * notional
            u_fees = (2 * self.fee_per_side) + (notional * 2 * self.slippage_rate)
            u_net_pnl = u_gross_pnl - u_fees
            unrealized_net = u_net_pnl

            # Progress to barriers
            tp_px = float(pos["tp_price"])
            sl_px = float(pos["sl_price"])
            if direction == "LONG":
                dist_to_tp_pct = ((tp_px - px) / px) * 100.0
                dist_to_sl_pct = ((px - sl_px) / px) * 100.0
            else:
                dist_to_tp_pct = ((px - tp_px) / px) * 100.0
                dist_to_sl_pct = ((sl_px - px) / px) * 100.0

            pos["current_price"] = px
            pos["unrealized_gross_pnl_usd"] = u_gross_pnl
            pos["unrealized_net_pnl_usd"] = u_net_pnl
            pos["unrealized_return_pct"] = u_net_pnl / notional
            pos["dist_to_tp_pct"] = dist_to_tp_pct
            pos["dist_to_sl_pct"] = dist_to_sl_pct
            active = pos

        current_equity = self.cash_balance + unrealized_net
        total_net_return_pct = ((current_equity - self.starting_balance) / self.starting_balance) * 100.0

        return {
            "enabled": getattr(config, "PAPER_TRADING_ENABLED", True),
            "starting_balance_usd": self.starting_balance,
            "cash_balance_usd": self.cash_balance,
            "current_equity_usd": current_equity,
            "total_net_pnl_usd": net_pnl + unrealized_net,
            "total_net_return_pct": total_net_return_pct,
            "total_trades": n_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": win_rate,
            "gross_pnl_usd": gross_pnl,
            "total_fees_paid_usd": total_fees,
            "profit_factor": profit_factor,
            "max_drawdown_usd": max_dd_usd,
            "max_drawdown_pct": max_dd_pct,
            "fee_structure": {
                "venue": "Jupiter Perps",
                "pool_fee_per_trade_usd": self.fee_per_side,
                "round_trip_fee_usd": 2 * self.fee_per_side,
                "slippage_bps": self.slippage_rate * 10000,
            },
            "active_position": active,
            "recent_trades": trades[-10:][::-1],  # newest first
        }


# Global singleton instance
paper_trader = PaperTrader()
