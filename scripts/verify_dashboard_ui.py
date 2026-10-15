"""Browser smoke tests. Run with the dashboard listening on localhost:8501.

Requires the optional development package playwright and installed Chrome.
API fixtures exercise rendering only; screenshots are never performance evidence.
"""
import json
from urllib.request import urlopen
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8501"


def verify_appearance(page):
    posts = []
    page.on("request", lambda request: posts.append(request.url) if request.method == "POST" else None)
    page.evaluate("window.scrollTo(0,0)")
    page.locator("#appearanceToggle").click()
    assert page.locator("#appearanceDialog").is_visible()
    assert page.locator("#appearanceToggle").get_attribute("aria-expanded") == "true"
    assert page.locator("#closeAppearance").evaluate("el=>el===document.activeElement")
    page.keyboard.press("Shift+Tab")
    assert page.locator("#doneAppearance").evaluate("el=>el===document.activeElement")
    page.keyboard.press("Tab")
    assert page.locator("#closeAppearance").evaluate("el=>el===document.activeElement")
    page.locator("#themeSelect").select_option("dark")
    for name in ("emerald", "mint", "terminal", "forest", "crimson"):
        page.locator("#paletteSelect").select_option(name)
        assert page.evaluate("C.up") == page.evaluate(f'colorPalettes.{name}.dark.up')
        page.locator("#themeSelect").select_option("light")
        assert page.evaluate("C.down") == page.evaluate(f'colorPalettes.{name}.light.down')
        page.locator("#themeSelect").select_option("dark")
    page.locator("#paletteSelect").select_option("ocean")
    assert page.evaluate("C.up") == "#72baff"
    page.locator("#themeSelect").select_option("light")
    assert page.evaluate("C.up") == "#1769b5"
    page.locator("#paletteSelect").select_option("violet")
    assert page.evaluate("C.down") == "#b12e83"
    for selector, value in (("upColor", "#2266cc"), ("downColor", "#dd7744"),
                            ("accentColor", "#7755bb"), ("gridColor", "#333344")):
        page.locator(f"#{selector}").fill(value)
        page.locator(f"#{selector}").dispatch_event("input")
        assert page.locator(f"#{selector}Value").inner_text() == value.upper()
    assert page.locator("#paletteSelect").input_value() == "custom"
    assert page.evaluate("[C.up,C.down,C.grid]") == ["#2266cc", "#dd7744", "#333344"]
    assert page.locator(".preview-up").evaluate("el=>getComputedStyle(el).color") == "rgb(34, 102, 204)"
    assert page.locator("#overviewProbabilityBars .prob-fill").first.evaluate("el=>el.style.background") == "rgb(34, 102, 204)"
    page.locator("#swapColors").click()
    assert page.evaluate("[C.up,C.down]") == ["#dd7744", "#2266cc"]
    page.locator("#swapColors").click()
    renders = []
    for style in ("solid", "hollow-up", "hollow"):
        page.locator("#candleStyle").select_option(style)
        page.wait_for_timeout(40)
        renders.append(page.locator("#chart").evaluate("el=>el.toDataURL()"))
    assert len(set(renders)) == 3
    page.locator("#candleStyle").select_option("hollow-up")
    for name, value in (("bodyWidth", "65"), ("bodyOpacity", "60"), ("wickWidth", "2.5")):
        page.locator(f"#{name}").fill(value)
        page.locator(f"#{name}").dispatch_event("input")
    page.locator("#matchWicks").uncheck()
    page.locator("#wickUpColor").fill("#abcdef")
    page.locator("#wickUpColor").dispatch_event("input")
    page.locator("#wickDownColor").fill("#123456")
    page.locator("#wickDownColor").dispatch_event("input")
    assert page.locator("#wickUpColorValue").inner_text() == "#ABCDEF"
    page.locator("#candleWicks").uncheck()
    assert page.locator("#wickWidth").is_disabled()
    assert page.locator("#wickUpColor").is_disabled()
    page.locator("#candleWicks").check()
    page.locator("#candleBorders").uncheck()
    page.locator("#candlePreview").scroll_into_view_if_needed()
    page.wait_for_timeout(100)
    assert page.locator("#candlePreview").evaluate("el=>el.width>0 && el.getContext('2d').getImageData(0,0,el.width,el.height).data.some(v=>v>0)")
    page.screenshot(path=str(ROOT / "logs" / "appearance-candle-preview.png"))
    page.locator("#chartStyle").select_option("line")
    assert page.locator("#candleStyle").is_disabled()
    assert page.locator("#bodyWidth").is_disabled()
    page.locator("#chartHeight").select_option("tall")
    page.locator("#chartGrid").uncheck()
    page.locator("#chartPriceLine").uncheck()
    page.locator("#chartSignals").uncheck()
    page.locator("#doneAppearance").click()
    page.wait_for_function("!document.getElementById('appearanceDialog').open")
    page.reload()
    page.wait_for_function("S.data !== null")
    assert page.evaluate("[appearance.up,appearance.style,appearance.height,appearance.grid]") == ["#2266cc", "line", "tall", False]
    assert page.evaluate("[appearance.candleStyle,appearance.bodyWidth,appearance.bodyOpacity,appearance.wickWidth,appearance.wickUp,appearance.matchWicks,appearance.borders]") == ["hollow-up", 65, 60, 2.5, "#abcdef", False, False]
    assert page.locator("#chart").evaluate("el=>el.style.height") == "560px"
    page.locator("#chartSettingsToggle").click()
    page.locator("#resetChartAppearance").click()
    assert page.evaluate("[appearance.up,appearance.style,appearance.grid]") == ["#2266cc", "candles", True]
    assert page.evaluate("[appearance.candleStyle,appearance.bodyWidth,appearance.wickWidth,appearance.matchWicks,appearance.wickUp]") == ["solid", 88, 1, True, None]
    page.keyboard.press("Escape")
    page.wait_for_function("document.getElementById('chartSettingsToggle').getAttribute('aria-expanded')==='false'")
    assert page.locator("#chartSettingsToggle").evaluate("el=>el===document.activeElement")
    page.locator("#appearanceToggle").click()
    for theme in ("light", "dark"):
        page.locator("#themeSelect").select_option(theme)
        for width, height in ((320, 568), (390, 844), (1440, 800)):
            page.set_viewport_size({"width": width, "height": height})
            assert page.locator("#appearanceDialog").evaluate("el=>el.scrollWidth<=el.clientWidth")
            box = page.locator("#appearanceDialog").bounding_box()
            assert 0 <= box["x"] and box["x"] + box["width"] <= width
            assert box["height"] <= height
            assert page.locator(".appearance-body").evaluate("el=>el.scrollWidth<=el.clientWidth && el.scrollHeight>el.clientHeight")
            done = page.locator("#doneAppearance").bounding_box()
            assert done["y"] >= 0 and done["y"] + done["height"] <= height
            page.locator("#doneAppearance").scroll_into_view_if_needed()
            page.screenshot(path=str(ROOT / "logs" / f"appearance-{width}-{theme}.png"))
            page.locator("#candleStyle").scroll_into_view_if_needed()
            page.screenshot(path=str(ROOT / "logs" / f"appearance-candles-{width}-{theme}.png"))
    page.set_viewport_size({"width": 1440, "height": 800})
    page.locator("#resetAppearance").click()
    assert page.evaluate("appearance.palette") == "classic"
    assert page.evaluate("appearance.up") is None
    page.emulate_media(color_scheme="dark")
    page.wait_for_function("document.documentElement.dataset.theme==='dark'")
    assert page.evaluate("C.up") == "#7dceb0"
    page.emulate_media(color_scheme="light")
    page.wait_for_function("document.documentElement.dataset.theme==='light'")
    assert page.evaluate("C.up") == "#167d62"
    page.evaluate("() => {window.originalStorageSet=Storage.prototype.setItem;Storage.prototype.setItem=function(){throw Error('Storage disabled');};}")
    page.locator("#paletteSelect").select_option("ocean")
    assert "unavailable" in page.locator("#appearanceStatus").inner_text()
    assert page.evaluate("C.up") == "#1769b5"
    page.evaluate("() => {Storage.prototype.setItem=window.originalStorageSet;}")
    page.locator("#resetAppearance").click()
    page.mouse.click(2, 2)
    page.wait_for_function("!document.getElementById('appearanceDialog').open")
    assert page.locator("#appearanceToggle").evaluate("el=>el===document.activeElement")
    page.evaluate("localStorage.setItem('marketBrief.appearance', '{broken')")
    page.reload()
    page.wait_for_function("S.data !== null")
    assert page.evaluate("appearance.palette") == "classic"
    assert not posts, posts  # Visual preferences must never write wallet/server settings.


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1050})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        with urlopen(BASE + "/api/snapshot", timeout=90) as response:
            snapshot = json.load(response)
        snapshot.update(gate_pass=False, decision="STAND ASIDE", qualifying_trades=0)
        snapshot.pop("model_available", None)
        preferences = {"hz": "15m", "tf": "1m", "sol_wallet": ""}

        def api(route):
            path = route.request.url.split("/api/", 1)[1].split("?", 1)[0]
            if path == "snapshot":
                data = {**snapshot, "tf": preferences["tf"], "wallet": ""}
            elif path == "settings":
                if route.request.method == "POST":
                    preferences.update(route.request.post_data_json)
                data = preferences
            elif path == "details":
                data = {"scored": []}
            else:
                data = {"positions": {"dataList": []}, "trades": {"dataList": []}}
            route.fulfill(json=data)

        page.route("**/api/**", api)
        page.add_init_script("""window.chartLabels=[];
          const fill=CanvasRenderingContext2D.prototype.fillText;
          CanvasRenderingContext2D.prototype.fillText=function(text,...args){
            window.chartLabels.push(String(text));return fill.call(this,text,...args);
          };""")
        page.goto(BASE)
        page.wait_for_function("document.getElementById('priceValue').textContent.startsWith('$')")
        assert page.locator("#verdictValue").inner_text() == "Stand aside"
        assert page.locator("#modelState").inner_text() == "Validation not passed"
        assert page.locator("#overviewProbabilityBars .prob-row").count() == 3
        assert not page.evaluate("chartLabels.some(x=>x.includes('NaN'))")
        assert page.evaluate("chartLabels.some(x=>/^\\d{2}:\\d{2}$/.test(x))")
        page.screenshot(path=str(ROOT / "logs" / "dashboard-desktop.png"), full_page=True)
        chart = page.locator("#chart").bounding_box()
        page.mouse.move(chart["x"] + 200, chart["y"] + 150)
        before = page.evaluate("S.range.slice()")
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(100)
        assert page.evaluate("S.range[1]-S.range[0]") < before[1] - before[0]
        assert page.locator("#followLive").get_attribute("aria-pressed") == "false"
        frozen = page.evaluate("S.range.slice()")
        page.evaluate("update()")
        assert page.evaluate("S.range") == frozen
        page.locator("#chart").focus()
        page.keyboard.press("l")
        assert page.evaluate("S.followLive && S.range[1]===S.data.candles.length-1")
        assert page.evaluate("S.range[1]-S.range[0]") == frozen[1] - frozen[0]
        page.keyboard.press("r")
        assert page.evaluate("S.range[1]-S.range[0]") == min(179, len(snapshot["candles"]) - 1)
        page.mouse.move(chart["x"] + 180, chart["y"] + 90)
        page.wait_for_timeout(50)
        assert "UTC O " in page.locator("#chartReadout").inner_text()
        assert page.evaluate("S.hover !== null")
        page.locator("#chart").dblclick(position={"x": 120, "y": 70})
        assert page.evaluate("S.followLive")
        page.locator("#resetZoom").click()
        page.locator("#viewEvidence").click()
        assert page.locator("#indicatorsCard").is_visible()
        assert page.locator("#tradeRows").is_visible()
        assert page.locator("#detailsTab").count() == 0
        page.locator("#walletInput").fill("invalid")
        page.locator("#applySettings").click()
        assert "valid public" in page.locator("#settingsStatus").inner_text()
        page.locator("#walletInput").fill("")
        page.locator('#chartIntervals button[data-tf="5m"]').click()
        page.wait_for_function("S.tf === '5m' && !S.tfSaving")
        assert preferences["tf"] == "5m"
        page.locator("#horizonSelect").select_option("60m")
        page.wait_for_function("S.h === '60m' && !S.tfSaving")
        assert preferences["hz"] == "60m"
        page.locator("#applySettings").click()
        page.wait_for_function("document.getElementById('settingsStatus').textContent==='Preferences saved.'")
        verify_appearance(page)
        for theme in ("light", "dark"):
            page.locator("#appearanceToggle").click()
            page.locator("#themeSelect").select_option(theme)
            page.locator("#doneAppearance").click()
            for width, height in ((390, 844), (768, 900), (1366, 768), (1920, 1080)):
                page.set_viewport_size({"width": width, "height": height})
                page.wait_for_timeout(250)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (width, theme)
                assert page.locator("#tradeRows").is_visible()
                page.screenshot(path=str(ROOT / "logs" / f"dashboard-single-{width}-{theme}.png"), full_page=True)
        page.emulate_media(reduced_motion="reduce")
        assert page.locator(".panel").first.evaluate("el=>getComputedStyle(el).animationName") == "none"
        page.route("**/api/snapshot**", lambda route: route.fulfill(status=503, json={"error": "Market data temporarily unavailable"}))
        page.evaluate("update()")
        page.wait_for_function("document.getElementById('feedStatus').textContent==='Data unavailable'")
        assert not errors, errors
        print("PASS: appearance modal, keyboard/focus/backdrop, all palettes, candle render styles/controls, live preview, reload persistence, storage failures, resets, system theme, responsive layouts, OHLC crosshair, follow-latest/reset shortcuts and API checks; no browser exceptions")
        context.close()
        browser.close()


if __name__ == "__main__":
    main()