"""Tests for Forward Paper Trading and PnL Tracker."""
import math
from pathlib import Path
import pytest
import config
from live.paper_trader import PaperTrader


@pytest.fixture
def temp_trader(tmp_path):
    state_file = tmp_path / "paper_state.json"
    history_file = tmp_path / "paper_history.jsonl"
    return PaperTrader(
        state_file=state_file,
        history_file=history_file,
        starting_balance=10000.0,
        position_notional=1000.0,
        fee_per_side=0.65,
        slippage_rate=0.0002,
    )


def test_paper_trader_initial_state(temp_trader):
    assert temp_trader.cash_balance == 10000.0
    assert temp_trader.active_position is None
    assert len(temp_trader.closed_trades) == 0

    summary = temp_trader.get_summary()
    assert summary["starting_balance_usd"] == 10000.0
    assert summary["current_equity_usd"] == 10000.0
    assert summary["total_trades"] == 0
    assert summary["win_rate_pct"] == 0.0
    assert summary["total_fees_paid_usd"] == 0.0
    assert summary["active_position"] is None


def test_open_position_long(temp_trader):
    pos = temp_trader.open_position(
        direction="LONG",
        price=180.0,
        atr_pct=0.002,  # 0.2% 1m ATR
        horizon_min=15,
        model_confidence=0.72,
        now_ms=1700000000000,
    )

    assert pos["direction"] == "LONG"
    assert pos["entry_price"] == 180.0
    assert pos["notional_usd"] == 1000.0
    assert pos["size_coin"] == pytest.approx(1000.0 / 180.0)

    # 15m ATR barrier: 2x ATR(15m) TP, 1x ATR(15m) SL
    h_mult = math.sqrt(15)
    expected_tp = 180.0 * (1.0 + config.TP_ATR * 0.002 * h_mult)
    expected_sl = 180.0 * (1.0 - config.SL_ATR * 0.002 * h_mult)

    assert pos["tp_price"] == pytest.approx(expected_tp)
    assert pos["sl_price"] == pytest.approx(expected_sl)
    assert temp_trader.active_position is not None

    # Cannot open another position while one is active
    with pytest.raises(RuntimeError):
        temp_trader.open_position(direction="SHORT", price=180.0, atr_pct=0.002)


def test_close_position_take_profit(temp_trader):
    temp_trader.open_position(
        direction="LONG",
        price=180.0,
        atr_pct=0.002,
        horizon_min=15,
        now_ms=1700000000000,
    )

    # Take profit price
    tp_price = temp_trader.active_position["tp_price"]

    closed = temp_trader.close_position(
        exit_price=tp_price,
        reason="TAKE_PROFIT",
        now_ms=1700000600000,
    )

    assert closed["exit_reason"] == "TAKE_PROFIT"
    assert closed["is_win"] is True
    assert closed["gross_pnl_usd"] > 0
    # Fees should be $0.65 entry + $0.65 exit = $1.30, plus slippage
    assert closed["fixed_fees_usd"] == pytest.approx(1.30)
    assert closed["net_pnl_usd"] == pytest.approx(closed["gross_pnl_usd"] - closed["total_fees_usd"])
    assert temp_trader.cash_balance == pytest.approx(10000.0 + closed["net_pnl_usd"])
    assert temp_trader.active_position is None

    summary = temp_trader.get_summary()
    assert summary["total_trades"] == 1
    assert summary["wins"] == 1
    assert summary["win_rate_pct"] == 100.0
    assert summary["profit_factor"] > 1.0


def test_close_position_stop_loss(temp_trader):
    temp_trader.open_position(
        direction="LONG",
        price=180.0,
        atr_pct=0.002,
        horizon_min=15,
        now_ms=1700000000000,
    )

    sl_price = temp_trader.active_position["sl_price"]
    closed = temp_trader.close_position(
        exit_price=sl_price,
        reason="STOP_LOSS",
        now_ms=1700000300000,
    )

    assert closed["exit_reason"] == "STOP_LOSS"
    assert closed["is_win"] is False
    assert closed["net_pnl_usd"] < 0
    assert temp_trader.cash_balance < 10000.0

    summary = temp_trader.get_summary()
    assert summary["total_trades"] == 1
    assert summary["losses"] == 1
    assert summary["win_rate_pct"] == 0.0
    assert summary["max_drawdown_usd"] > 0


def test_update_market_triggers_barriers(temp_trader):
    temp_trader.open_position(
        direction="SHORT",
        price=180.0,
        atr_pct=0.002,
        horizon_min=15,
        now_ms=1700000000000,
    )

    tp_price = temp_trader.active_position["tp_price"]

    # Price moves halfway — no exit
    mid_price = 180.0 - 0.05
    res = temp_trader.update_market(
        current_price=mid_price,
        snapshot={},
        now_ms=1700000100000,
    )
    assert res is None
    assert temp_trader.active_position is not None

    # Price touches TP on SHORT
    res_tp = temp_trader.update_market(
        current_price=tp_price - 0.01,
        snapshot={},
        now_ms=1700000200000,
    )
    assert res_tp is not None
    assert res_tp["exit_reason"] == "TAKE_PROFIT"
    assert temp_trader.active_position is None


def test_paper_trading_persistence(tmp_path):
    state_file = tmp_path / "paper_state.json"
    history_file = tmp_path / "paper_history.jsonl"

    trader1 = PaperTrader(
        state_file=state_file,
        history_file=history_file,
        starting_balance=10000.0,
    )
    trader1.open_position(direction="LONG", price=100.0, atr_pct=0.001)
    trader1.close_position(exit_price=105.0, reason="TAKE_PROFIT")

    assert trader1.cash_balance > 10000.0

    # Load in new trader instance
    trader2 = PaperTrader(
        state_file=state_file,
        history_file=history_file,
        starting_balance=10000.0,
    )
    assert trader2.cash_balance == pytest.approx(trader1.cash_balance)
    assert len(trader2.closed_trades) == 1
    assert trader2.closed_trades[0]["exit_reason"] == "TAKE_PROFIT"
