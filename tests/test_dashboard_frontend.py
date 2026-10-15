"""Exercise frontend account arithmetic with Node, without network access."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("trades, expected", [
    ([], {"gross": 0, "fees": 0, "net": 0, "decreases": 0, "profits": 0, "losses": 0}),
    ([{"action": "Increase", "pnl": "0", "fee": "2"},
      {"action": "Decrease", "pnl": "100", "fee": "3"},
      {"action": "Decrease", "pnl": "-20", "fee": "1"}],
     {"gross": 80, "fees": 6, "net": 74, "decreases": 2, "profits": 100, "losses": -20}),
    ([{"action": "Increase", "fee": "2"}],
     {"gross": 0, "fees": 2, "net": -2, "decreases": 0, "profits": 0, "losses": 0}),
    ([{"action": "Decrease", "pnl": "10", "fee": None}],
     {"gross": 10, "fees": None, "net": None, "decreases": 1, "profits": 10, "losses": 0}),
    ([{"action": "Decrease", "pnl": "", "fee": "3"}],
     {"gross": None, "fees": 3, "net": None, "decreases": 1, "profits": None, "losses": None}),
    ([{"action": "Decrease", "pnl": "NaN", "fee": "Infinity"}],
     {"gross": None, "fees": None, "net": None, "decreases": 1, "profits": None, "losses": None}),
])
def test_account_totals(trades, expected):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute frontend arithmetic")
    path = Path(__file__).resolve().parents[1] / "live" / "web" / "app.js"
    script = (
        'global.document={addEventListener:()=>{}};\n'
        'global.matchMedia=()=>({matches:false,addEventListener:()=>{}});\n'
        + path.read_text(encoding="utf-8-sig")
        + f'\nconsole.log(JSON.stringify(accountTotals({json.dumps(trades)})));'
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


def test_account_card_is_on_overview():
    path = Path(__file__).resolve().parents[1] / "live" / "web" / "index.html"
    html = path.read_text(encoding="utf-8-sig")
    assert html.index('id="overviewPanel"') < html.index('id="tradeRows"') < html.index('id="detailsPanel"')
    assert html.count('id="accountSummary"') == 1
    assert "not lifetime profits" in html
    assert "may double-count" in html


def test_appearance_controls_are_local_and_accessible():
    root = Path(__file__).resolve().parents[1] / "live" / "web"
    html = (root / "index.html").read_text(encoding="utf-8-sig")
    js = (root / "app.js").read_text(encoding="utf-8-sig")
    for name in ("themeSelect", "chartStyle", "chartHeight", "chartGrid", "chartPriceLine", "chartSignals"):
        assert f'id="{name}"' in html
    assert 'aria-controls="chartSettings"' in html
    assert 'localStorage.setItem("marketBrief.appearance"' in js
    assert 'prefers-color-scheme: dark' in js


def test_indicators_are_on_overview_with_direction_legend():
    root = Path(__file__).resolve().parents[1] / "live" / "web"
    html = (root / "index.html").read_text(encoding="utf-8-sig")
    assert html.index('id="overviewPanel"') < html.index('id="evidenceRows"') < html.index('id="detailsPanel"')
    assert html.count('id="evidenceRows"') == 1
    assert "Long / bullish" in html
    assert "Short / bearish" in html


@pytest.mark.parametrize("value,expected", [(12, "up"), (-2, "down"), (0, ""), (None, ""), ("NaN", "")])
def test_value_tones_keep_zero_and_missing_neutral(value, expected):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute frontend helpers")
    path = Path(__file__).resolve().parents[1] / "live" / "web" / "app.js"
    script = ('global.document={addEventListener:()=>{}};'
              'global.matchMedia=()=>({matches:false,addEventListener:()=>{}});'
              + path.read_text(encoding="utf-8-sig")
              + f'\nconsole.log(JSON.stringify(valueTone({json.dumps(value)})));')
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected