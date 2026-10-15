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
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


@pytest.mark.parametrize("step", [2, 6, 20, 100])
def test_candle_body_tracks_spacing_when_zoomed(step):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute frontend helpers")
    path = Path(__file__).resolve().parents[1] / "live" / "web" / "app.js"
    script = ('global.document={addEventListener:()=>{}};'
              'global.matchMedia=()=>({matches:false,addEventListener:()=>{}});'
              + path.read_text(encoding="utf-8-sig")
              + f'\nconsole.log(JSON.stringify(candleBodyWidth({step})));')
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, check=True)
    width = json.loads(result.stdout)
    assert 0.85 * step <= width < step


def test_account_card_is_on_overview():
    path = Path(__file__).resolve().parents[1] / "live" / "web" / "index.html"
    html = path.read_text(encoding="utf-8-sig")
    assert html.index('id="overviewPanel"') < html.index('id="tradeRows"')
    assert 'id="detailsPanel"' not in html
    assert html.count('id="accountSummary"') == 1
    assert "not lifetime profits" in html
    assert "may double-count" in html


def test_model_is_a_standalone_panel_not_a_chart_sidebar():
    from html.parser import HTMLParser
    root = Path(__file__).resolve().parents[1] / "live" / "web"
    html = (root / "index.html").read_text(encoding="utf-8-sig")

    class Structure(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack = []
            self.model_parent = None
            self.ids = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            self.ids.append(attrs.get("id"))
            if attrs.get("id") == "decisionCard":
                assert tag == "section"
                self.model_parent = self.stack[-1][1].get("id")
            if tag not in {"meta", "link", "input", "br", "hr", "img"}:
                self.stack.append((tag, attrs))

        def handle_endtag(self, tag):
            assert self.stack[-1][0] == tag, (tag, self.stack[-1])
            self.stack.pop()

    parser = Structure()
    parser.feed(html)
    assert parser.model_parent == "overviewPanel"
    for identifier in ("decisionCard", "overviewProbabilityBars", "modelLean", "validationNote"):
        assert parser.ids.count(identifier) == 1
    assert 'Validation &amp; model details</summary>' in html
    assert 'href="/model-panel.css"' in html


def test_workstation_layout_uses_available_width_and_shows_activity():
    root = Path(__file__).resolve().parents[1] / "live" / "web"
    html = (root / "index.html").read_text(encoding="utf-8-sig")
    css = (root / "dashboard-layout.css").read_text(encoding="utf-8-sig")
    assert html.index('href="/model-panel.css"') < html.index('href="/dashboard-layout.css"')
    assert 'max-width: none' in css
    assert '<div class="account-section">' in html
    assert 'account-disclosure' not in html
    assert 'id="detailsTab"' not in html
    assert html.index('id="horizonSelect"') < html.index('id="modelState"')
    assert '100dvh' in css
    # Modal clipping is intentional; the dashboard itself must keep scrolling.
    assert 'body { overflow: hidden' not in css
    assert '.appearance-body {' in css and 'overflow-y: auto' in css


def test_appearance_controls_are_local_and_accessible():
    root = Path(__file__).resolve().parents[1] / "live" / "web"
    html = (root / "index.html").read_text(encoding="utf-8-sig")
    js = (root / "app.js").read_text(encoding="utf-8-sig")
    for name in ("themeSelect", "chartStyle", "chartHeight", "chartGrid", "chartPriceLine", "chartSignals"):
        assert f'id="{name}"' in html
    assert 'aria-controls="appearanceDialog"' in html
    assert '<dialog id="appearanceDialog"' in html
    for name in ("appearanceToggle", "paletteSelect", "upColor", "downColor", "accentColor", "gridColor", "resetAppearance"):
        assert html.count(f'id="{name}"') == 1
    assert 'dialog.showModal()' in js
    assert 'localStorage.setItem("marketBrief.appearance"' in js
    assert 'prefers-color-scheme: dark' in js


@pytest.mark.parametrize("value,expected", [
    ("#12abEF", True), ("#123456", True), (None, False),
    ("red", False), ("#abc", False), ("#123456;background:red", False), (42, False),
])
def test_custom_colors_accept_only_full_hex_values(value, expected):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute frontend helpers")
    path = Path(__file__).resolve().parents[1] / "live" / "web" / "app.js"
    script = ('global.document={addEventListener:()=>{}};'
              'global.matchMedia=()=>({matches:false,addEventListener:()=>{}});'
              + path.read_text(encoding="utf-8-sig")
              + f'\nconsole.log(JSON.stringify(validColor({json.dumps(value)})));')
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


def test_indicators_are_on_overview_with_direction_legend():
    root = Path(__file__).resolve().parents[1] / "live" / "web"
    html = (root / "index.html").read_text(encoding="utf-8-sig")
    assert html.index('id="overviewPanel"') < html.index('id="evidenceRows"')
    assert html.count('id="evidenceRows"') == 1
    assert "Long / bullish" in html
    assert "Short / bearish" in html


def test_dark_palette_covers_tables_controls_and_canvas_labels():
    root = Path(__file__).resolve().parents[1] / "live" / "web"
    html = (root / "index.html").read_text(encoding="utf-8-sig")
    css = (root / "dark.css").read_text(encoding="utf-8-sig")
    js = (root / "app.js").read_text(encoding="utf-8-sig")
    assert html.index('href="/styles.css"') < html.index('href="/dark.css"')
    for selector in (".settings-row label", "td", "input::placeholder", ".summary-chip.down", ".vote-pill.long"):
        assert selector in css
    assert "--chart-ink:" in css
    assert "c.fillStyle=C.ink" in js
    assert "[C.up,C.axis,C.down][index]" in js


def test_dark_theme_uses_true_black_surfaces():
    root = Path(__file__).resolve().parents[1] / "live" / "web"
    css = (root / "dark.css").read_text(encoding="utf-8-sig")
    assert "--bg: #000000;" in css
    assert "--surface: #080808;" in css
    assert "--surface-raised: #141414;" in css
    assert "--chart-ink: #050505;" in css
    assert "#121315" not in css
    assert "#1b1d20" not in css


def test_day_chart_present_and_slice_logic():
    root = Path(__file__).resolve().parents[1] / "live" / "web"
    html = (root / "index.html").read_text(encoding="utf-8-sig")
    js = (root / "app.js").read_text(encoding="utf-8-sig")
    assert 'id="dayChart"' in html and 'role="img"' in html
    assert 'aria-label="SOL price over the last 24 hours"' in html
    assert "function draw24h" in js and "function last24h" in js
    assert "draw24h();" in js  # wired into the data refresh path


def test_last24h_filters_to_rolling_window():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute frontend helpers")
    path = Path(__file__).resolve().parents[1] / "live" / "web" / "app.js"
    now = 1_700_000_000_000
    candles = [{"timestamp": now - 90 * 60_000, "close": 1},      # 1.5h ago: kept
               {"timestamp": now - 30 * 60_000, "close": 2},      # 30m ago: kept
               {"timestamp": now - 23 * 3600_000, "close": 3},    # 23h ago: kept
               {"timestamp": now - 25 * 3600_000, "close": 4}]    # 25h ago: dropped
    import tempfile
    script = ('global.document={addEventListener:()=>{}};'
              'global.matchMedia=()=>({matches:false,addEventListener:()=>{}});'
              + path.read_text(encoding="utf-8-sig")
              + f';console.log(JSON.stringify([last24h({json.dumps(candles)},{now}).length,'
              f'last24h([],{now}).length,last24h(null,{now}).length]));')
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        tmp = fh.name
    try:
        result = subprocess.run([node, tmp], capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"node failed: {result.stderr[:2000]}"
    finally:
        Path(tmp).unlink(missing_ok=True)
    got = json.loads(result.stdout)
    assert got[0] == 3
    assert got[1] == 0 and got[2] == 0


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
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


def frontend_value(expression):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute frontend helpers")
    path = Path(__file__).resolve().parents[1] / "live" / "web" / "app.js"
    script = ('global.document={addEventListener:()=>{}};'
              'global.matchMedia=()=>({matches:false,addEventListener:()=>{}});'
              + path.read_text(encoding="utf-8-sig")
              + f'\nconsole.log(JSON.stringify({expression}));')
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("stored,key,expected", [
    (None, "bodyWidth", 88),
    ({"style": "line"}, "candleStyle", "solid"),
    ({"palette": "removed-palette"}, "palette", "classic"),
    ({"palette": "emerald"}, "palette", "emerald"),
    ({"candleStyle": "hollow-up"}, "candleStyle", "hollow-up"),
    ({"candleStyle": "unknown"}, "candleStyle", "solid"),
    ({"bodyWidth": -100}, "bodyWidth", 40),
    ({"bodyWidth": 120}, "bodyWidth", 95),
    ({"bodyOpacity": "50"}, "bodyOpacity", 100),
    ({"bodyOpacity": 2}, "bodyOpacity", 30),
    ({"wickWidth": 1.6}, "wickWidth", 1.5),
    ({"wickWidth": 1000}, "wickWidth", 3),
    ({"wicks": "false"}, "wicks", True),
    ({"matchWicks": False}, "matchWicks", False),
    ({"wickUp": "#12AbEF"}, "wickUp", "#12abef"),
    ({"wickDown": "#abc"}, "wickDown", None),
])
def test_appearance_normalization_migrates_and_bounds_values(stored, key, expected):
    assert frontend_value(f'normalizeAppearance({json.dumps(stored)})[{json.dumps(key)}]') == expected


@pytest.mark.parametrize("step,percent", [(0.2, 40), (2, 40), (20, 70), (100, 95)])
def test_custom_candle_width_stays_inside_its_slot(step, percent):
    width = frontend_value(f"candleBodyWidth({step},{percent})")
    assert 0 < width <= step
    if step >= 2:
        assert width == pytest.approx(step * percent / 100)


@pytest.mark.parametrize("style,up,fills", [
    ("solid", True, 1), ("solid", False, 1),
    ("hollow-up", True, 0), ("hollow-up", False, 1),
    ("hollow", True, 0), ("hollow", False, 0),
])
def test_candle_painter_obeys_direction_and_body_style(style, up, fills):
    expression = '''(()=>{
      appearance={...appearanceDefaults,candleStyle:STYLE,bodyOpacity:50,matchWicks:false,wickUp:"#123456",wickDown:"#abcdef"};
      const calls={fills:0,borders:0,strokes:[]};
      const ctx={save(){},restore(){},beginPath(){},moveTo(){},lineTo(){},
        stroke(){calls.strokes.push(this.strokeStyle);},
        fillRect(){calls.fills++;calls.alpha=this.globalAlpha;},
        strokeRect(){calls.borders++;}};
      paintCandle(ctx,{open:30,close:CLOSE,high:60,low:10},20,v=>100-v,10);
      return calls;
    })()'''.replace("STYLE", json.dumps(style)).replace("CLOSE", "45" if up else "20")
    result = frontend_value(expression)
    assert result["fills"] == fills
    assert result["borders"] == 1
    assert result["strokes"] == ["#123456" if up else "#abcdef"]
    if fills:
        assert result["alpha"] == 0.5


def test_candle_painter_can_hide_wicks_and_borders():
    assert frontend_value('''(()=>{
      appearance={...appearanceDefaults,wicks:false,borders:false};
      const calls=[];
      const ctx={save(){},restore(){},fillRect(){calls.push("body");},
        stroke(){calls.push("wick");},strokeRect(){calls.push("border");}};
      paintCandle(ctx,{open:30,close:40,high:50,low:20},10,v=>100-v,8);
      return calls;
    })()''') == ["body"]