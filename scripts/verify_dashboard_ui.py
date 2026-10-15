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
        assert page.locator("#probabilityBars .prob-row").count() == 3
        assert not page.evaluate("chartLabels.some(x=>x.includes('NaN'))")
        assert page.evaluate("chartLabels.some(x=>/^\\d{2}:\\d{2}$/.test(x))")
        page.screenshot(path=str(ROOT / "logs" / "dashboard-desktop.png"), full_page=True)
        chart = page.locator("#chart").bounding_box()
        page.mouse.move(chart["x"] + 200, chart["y"] + 150)
        before = page.evaluate("S.range.slice()")
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(100)
        assert page.evaluate("S.range[1]-S.range[0]") < before[1] - before[0]
        page.locator("#resetZoom").click()
        page.locator("#viewEvidence").click()
        assert page.locator("#detailsPanel").is_visible()
        page.locator("#walletInput").fill("invalid")
        page.locator("#applySettings").click()
        assert "valid public" in page.locator("#settingsStatus").inner_text()
        page.locator("#walletInput").fill("")
        page.locator("#timeframeSelect").select_option("5m")
        page.locator("#applySettings").click()
        page.wait_for_function("document.getElementById('settingsStatus').textContent==='Preferences saved.'")
        assert preferences["tf"] == "5m"
        page.locator("#detailsTab").focus()
        page.keyboard.press("ArrowLeft")
        assert page.locator("#overviewPanel").is_visible()
        assert page.locator("#overviewTab").get_attribute("tabindex") == "0"
        for width in (390, 768, 1280):
            page.set_viewport_size({"width": width, "height": 900})
            for tab in ("overviewTab", "detailsTab"):
                page.locator("#" + tab).click()
                page.wait_for_timeout(250)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (width, tab)
                page.screenshot(path=str(ROOT / "logs" / f"dashboard-{width}-{tab}.png"), full_page=True)
        page.emulate_media(reduced_motion="reduce")
        assert page.locator(".panel").first.evaluate("el=>getComputedStyle(el).animationName") == "none"
        page.locator("#overviewTab").click()
        page.route("**/api/snapshot**", lambda route: route.fulfill(status=503, json={"error": "Market data temporarily unavailable"}))
        page.evaluate("update()")
        page.wait_for_function("document.getElementById('feedStatus').textContent==='Data unavailable'")
        assert not errors, errors
        print("PASS: rendering, tabs, keyboard, zoom, settings, 3 responsive widths, reduced motion, API failure; no browser exceptions")
        context.close()
        browser.close()


if __name__ == "__main__":
    main()