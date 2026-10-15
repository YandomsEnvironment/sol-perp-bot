"""Focused tests for the local dashboard JSON API and static frontend."""
import json
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from live import dashboard


def _mock_inputs(monkeypatch, tmp_path):
    candles = pd.DataFrame({
        "timestamp": [60_000, 120_000],
        "open": [100.0, 101.0], "high": [102.0, 103.0],
        "low": [99.0, 100.0], "close": [101.0, 102.0],
        "volume": [10.0, 12.0],
    })
    monkeypatch.setattr(dashboard, "get_data", lambda: (
        closed, closed.copy(),
        pd.DataFrame(), None))
    closed = pd.concat([candles] * 220, ignore_index=True)
    cutoff = (int(time.time() * 1000) - 2000) // 60_000 * 60_000
    closed["timestamp"] = range(cutoff - len(closed) * 60_000, cutoff, 60_000)
    monkeypatch.setattr(dashboard.F, "WARMUP_BARS", 0)
    feature_rows = 220
    mock_features = pd.DataFrame({
        "ret_15m": [0.001] * feature_rows, "atr_pct": [0.002] * feature_rows,
        "range_pos_60": [0.5] * feature_rows, "rsi14": [50.] * feature_rows,
        "ret_1h": [0.] * feature_rows, "ret_4h": [0.] * feature_rows,
        "btc_ret_15m": [0.] * feature_rows,
    })
    monkeypatch.setattr(dashboard.F, "compute_model_features", lambda *a, **k: mock_features.copy())
    monkeypatch.setattr(dashboard.F, "prepare_model_input", lambda feats, names: feats.reindex(columns=names))
    model_bundle = {"model": type("Model", (), {
        "predict_proba": lambda self, x: [[0.1, 0.2, 0.7]]})(),
        "feature_names": ["ret_15m"], "conf_threshold": 0.6,
        "input_contract": "closed-ohlcv-v2",
        "aggregate": {"gate_pass": False, "n_trades": 0}}
    monkeypatch.setattr(dashboard, "get_bundle", lambda _: model_bundle)
    monkeypatch.setattr(dashboard, "build_evidence", lambda *a: (
        pd.DataFrame([{"signal": "trend", "reading": "+1%", "vote": "SHORT", "why": "test"}]),
        {"LONG": 0, "SHORT": 1, "NEUTRAL": 0}))
    monkeypatch.setattr(dashboard, "recorder_snapshots", lambda: {})
    monkeypatch.setattr(dashboard, "SIGNALS_LOG", tmp_path / "signals.jsonl")
    monkeypatch.setattr(dashboard, "get_chart_candles", lambda _: candles)
    monkeypatch.setattr(dashboard, "get_feed", lambda: type("Feed", (), {
        "latest_mid": lambda self: None, "mid_age_ms": lambda self: 123,
        "latest_ctx": lambda self: {}, "is_connected": True,
        "taker_flow": lambda self, n: {"buy_share": .47, "buy_sz": 47., "sell_sz": 53.},
        "candles_tail": lambda self, n: pd.DataFrame()})())


def test_snapshot_keeps_failed_gate_at_stand_aside(monkeypatch, tmp_path):
    _mock_inputs(monkeypatch, tmp_path)
    data = dashboard.market_snapshot({"hz": "15m", "tf": "1m", "sol_wallet": ""})
    assert data["model_lean"] == "SHORT"
    assert data["model_probability"] == pytest.approx(0.7)
    assert data["decision"] == "STAND ASIDE"
    assert data["gate_pass"] is False
    assert data["candles"][0]["close"] == 101.0
    assert data["flow"]["buy_share"] == pytest.approx(0.47)
    assert data["execution_venue"] == "Jupiter Perps"
    assert data["reference_source"] == "Hyperliquid"
    assert data["execution_enabled"] is False


def test_legacy_gate_cannot_authorize_jupiter_trading(monkeypatch, tmp_path):
    _mock_inputs(monkeypatch, tmp_path)
    dashboard.get_bundle("mid")["aggregate"]["gate_pass"] = True
    data = dashboard.market_snapshot({"hz": "15m", "tf": "1m", "sol_wallet": ""})
    assert data["model_available"] is True
    assert data["probabilities"]["short"] == pytest.approx(.7)
    assert data["gate_pass"] is False
    assert data["decision"] == "STAND ASIDE"


def test_snapshot_chart_and_signal_rows_are_json_safe(monkeypatch, tmp_path):
    _mock_inputs(monkeypatch, tmp_path)
    data = dashboard.market_snapshot({"hz": "15m", "tf": "1m", "sol_wallet": ""})
    assert data["candles"][0]["close"] == 101.0
    assert data["flow"]["buy_share"] == pytest.approx(.47)
    assert "decision" in data


def test_invalid_features_fail_closed_without_logging(monkeypatch, tmp_path):
    _mock_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(dashboard.F, "prepare_model_input",
                        lambda *a: (_ for _ in ()).throw(ValueError("Missing model features: ret_15m")))
    data = dashboard.market_snapshot({"hz": "15m", "tf": "1m", "sol_wallet": ""})
    assert data["model_available"] is False
    assert data["data_quality_errors"] == ["Missing model features: ret_15m"]
    assert data["model_lean"] == "UNAVAILABLE"
    assert data["decision"] == "STAND ASIDE"
    assert data["model_probability"] == 0
    assert data["probabilities"] == {"flat": 0, "long": 0, "short": 0}
    assert not (tmp_path / "signals.jsonl").exists()


@pytest.mark.parametrize("kind", ["unfinished", "stale"])
def test_bad_candle_freshness_fails_closed(monkeypatch, tmp_path, kind):
    _mock_inputs(monkeypatch, tmp_path)
    sol, btc, funding, ctx = dashboard.get_data()
    if kind == "unfinished":
        current_open = int(time.time() * 1000) // 60_000 * 60_000
        sol.loc[len(sol)] = {"timestamp": current_open, "open": 200., "high": 201.,
                             "low": 199., "close": 200., "volume": 999.}
        captured = {}
        original = dashboard.F.compute_model_features
        def capture(candles, **kwargs):
            captured["last_close"] = candles["close"].iloc[-1]
            return original(candles, **kwargs)
        monkeypatch.setattr(dashboard.F, "compute_model_features", capture)
    else:
        sol.loc[sol.index[-1], "timestamp"] -= 3 * 60_000
    monkeypatch.setattr(dashboard, "get_data", lambda: (sol, btc, funding, ctx))
    data = dashboard.market_snapshot({"hz": "15m", "tf": "1m", "sol_wallet": ""})
    if kind == "unfinished":
        assert data["model_available"] is True
        assert captured["last_close"] == 102.0
        assert data["model_lean"] == "SHORT"
    else:
        assert data["model_available"] is False
        assert data["model_lean"] == "UNAVAILABLE"
        assert data["decision"] == "STAND ASIDE"
        assert data["probabilities"] == {"flat": 0, "long": 0, "short": 0}
        assert not (tmp_path / "signals.jsonl").exists()


@pytest.mark.parametrize("which", ["sol", "btc"])
def test_empty_market_feed_returns_unavailable_snapshot(monkeypatch, tmp_path, which):
    _mock_inputs(monkeypatch, tmp_path)
    sol, btc, funding, ctx = dashboard.get_data()
    empty = sol.iloc[:0]
    monkeypatch.setattr(dashboard, "get_data", lambda: ((empty if which == "sol" else sol),
                                                          (empty if which == "btc" else btc), funding, ctx))
    calls = []
    monkeypatch.setattr(type(dashboard.get_bundle("mid")["model"]), "predict_proba",
                        lambda *args: calls.append(args))
    data = dashboard.market_snapshot({"hz": "15m", "tf": "1m", "sol_wallet": ""})
    assert data["model_available"] is False
    assert data["model_lean"] == "UNAVAILABLE"
    assert data["decision"] == "STAND ASIDE"
    assert data["probabilities"] == {"flat": 0, "long": 0, "short": 0}
    assert data["data_quality_errors"]
    assert not calls
    assert not (tmp_path / "signals.jsonl").exists()


def test_legacy_contract_fails_closed(monkeypatch, tmp_path):
    _mock_inputs(monkeypatch, tmp_path)
    dashboard.get_bundle("mid").pop("input_contract")
    data = dashboard.market_snapshot({"hz": "15m", "tf": "1m", "sol_wallet": ""})
    assert data["model_available"] is False
    assert any("closed-ohlcv-v2" in error for error in data["data_quality_errors"])
    assert data["model_lean"] == "UNAVAILABLE"
    assert not (tmp_path / "signals.jsonl").exists()


def test_json_serialization_handles_nonfinite_values():
    assert dashboard.clean(pd.Timestamp("2026-01-01", tz="UTC")) == "2026-01-01T00:00:00+00:00"
    assert dashboard.clean(float("inf")) is None


def test_evidence_tolerates_absent_optional_features():
    evidence, tally = dashboard.build_evidence(pd.Series(dtype=float), {})
    assert not evidence.empty
    assert evidence.loc[evidence.signal == "Funding (hourly)", "reading"].iloc[0] == "Unavailable"
    assert tally["LONG"] == 0 and tally["SHORT"] == 0


def test_static_frontend_has_two_tabs_and_canvas_chart():
    html = (Path(dashboard.WEB_ROOT) / "index.html").read_text(encoding="utf-8-sig")
    js = (Path(dashboard.WEB_ROOT) / "app.js").read_text(encoding="utf-8-sig")
    assert 'id="overviewTab"' in html
    assert 'id="detailsTab"' in html
    assert 'id="chart"' in html
    assert 'getContext("2d")' in js


def test_chart_interval_controls_replace_preferences_selector():
    html = (Path(dashboard.WEB_ROOT) / "index.html").read_text(encoding="utf-8-sig")
    assert 'id="chartIntervals"' in html
    for interval in ("1m", "5m", "15m"):
        assert f'data-tf="{interval}"' in html
    assert 'id="timeframeSelect"' not in html
    assert 'id="chartIntervalStatus"' in html


def test_chart_interval_save_preserves_other_preferences(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    dashboard.save_settings({"hz": "60m", "tf": "1m", "sol_wallet": ""})
    assert dashboard.save_settings({"tf": "5m"}) == {
        "hz": "60m", "tf": "5m", "sol_wallet": ""}
    assert dashboard.load_settings()["tf"] == "5m"


def test_wallet_settings_reject_non_public_key(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="valid Solana public address"):
        dashboard.save_settings({"hz": "15m", "tf": "1m", "sol_wallet": "not-a-wallet"})


def test_static_routes_prevent_path_traversal():
    from http.server import ThreadingHTTPServer
    import threading
    import urllib.error
    import urllib.request

    server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        root = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(root + "/api/health", timeout=2) as response:
            assert json.loads(response.read()) == {"ok": True}
        with urllib.request.urlopen(root + "/", timeout=2) as response:
            assert response.headers["Content-Type"].startswith("text/html")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(root + "/../settings.json", timeout=2)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)