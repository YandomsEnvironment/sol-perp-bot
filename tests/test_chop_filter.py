"""Tests for the Low-Volatility Chop Filter."""
import pytest
import config
from live import dashboard


def test_chop_filter_config_defaults():
    assert hasattr(config, "CHOP_FILTER_ENABLED")
    assert config.CHOP_FILTER_ENABLED is True
    assert hasattr(config, "MIN_ATR_PERCENTILE")
    assert config.MIN_ATR_PERCENTILE == 25.0


def test_chop_filter_flags_low_atr():
    atr_percentile = 18.5
    min_atr = config.MIN_ATR_PERCENTILE
    is_chop = bool(config.CHOP_FILTER_ENABLED and atr_percentile < min_atr)
    assert is_chop is True

    atr_percentile_high = 42.0
    is_chop_high = bool(config.CHOP_FILTER_ENABLED and atr_percentile_high < min_atr)
    assert is_chop_high is False


def test_telegram_signal_format_contains_chop_status():
    from live.telegram_bot import format_signal_message

    snap_chop = {
        "price": 185.50,
        "model_lean": "LONG",
        "probabilities": {"flat": 0.2, "long": 0.7, "short": 0.1},
        "evidence_tally": {"LONG": 4, "SHORT": 0, "NEUTRAL": 2},
        "cross_horizon": {"reading": "fast LONG · mid LONG", "why": "consensus", "agree": True, "vote": "LONG"},
        "chop_filter": {
            "enabled": True,
            "min_atr_percentile": 25.0,
            "current_percentile": 15.2,
            "is_chop": True,
        },
    }

    msg = format_signal_message(snap_chop)
    assert "CHOP / COMPRESSION" in msg
    assert "suppressed" in msg.lower()

    snap_expanded = dict(snap_chop)
    snap_expanded["chop_filter"] = {
        "enabled": True,
        "min_atr_percentile": 25.0,
        "current_percentile": 55.0,
        "is_chop": False,
    }
    msg_expanded = format_signal_message(snap_expanded)
    assert "EXPANDED" in msg_expanded
