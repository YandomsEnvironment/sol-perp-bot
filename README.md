# SOL Perp Research Bot

SOL directional forecasting, chronological multi-horizon model evaluation, and a local
read-only dashboard configured for **Lightning** zero-fee perpetual execution.

**This project does not sign transactions or place orders.** Predictions are
model outputs, not proven win probabilities or demonstrated profitability.

## Data and validation

- Binance futures candles provide historical research data.
- Hyperliquid supplies the current live reference candles and market feed.
- Lightning zero-fee structure ($0.00 / 0 bps fee drag) eliminates breakeven hurdles.
- Multi-horizon forecast suite supports 5m (fast), 15m (mid), 1h (slow), 4h, and 1d horizons.
- Native AI Tactical Advisor provides actionable entry verdict, likely direction odds, and exact ATR-scaled TP/SL barriers.
- Live input requires the `closed-ohlcv-v2` feature contract and valid completed
  candles. Legacy model bundles fail closed.
- Candidates are saved separately, with promotion disabled. Research models remain read-only.

See [VENUES.md](VENUES.md) for the data/execution distinction. Execution is
configured for Lightning zero-fee trading ($0.00 / 0 bps fee drag).

## Install (Windows PowerShell)

Use Python 3.10 or newer. Node.js is also needed for frontend tests.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run commands from your clone's root. Optional wallet configuration is a public
Solana address set in the dashboard preferences or `SOL_WALLET` environment
variable. No private key is needed. Local settings are excluded from Git.

## Data and training

```powershell
.\.venv\Scripts\python.exe data\download_binance_vision.py --help
.\.venv\Scripts\python.exe data\download_hist.py --help
.\.venv\Scripts\python.exe train.py --horizon fast --target forward --estimators 30 --max-folds 1
```

Download history before training. Training writes research artifacts under
`models/candidates`, not the dashboard's active model files. No data or model
binaries are included in this repository. `research_models.json` explicitly maps
each horizon to a local candidate path relative to `models`. Update that mapping
after training; the dashboard never picks the newest file automatically or
replaces active models. It displays the underlying raw classifier probabilities,
including when a saved candidate wraps a calibrated classifier. Research loading
always disables the gate and trading threshold. Missing/incompatible artifacts
return a structured unavailable state; a fresh clone needs local candidates.
Do not load untrusted pickle/joblib artifacts.

## Dashboard

**On Windows, double-click `Start Bot.cmd`.** It starts the dashboard in the
background and opens your browser. Running it again reuses the existing instance.
Close the browser whenever you like; double-click `Stop Bot.cmd` to stop the
dashboard itself. Neither launcher places orders, trains models, or starts the
optional data recorder. On this machine, a desktop shortcut is also available.
Errors stay visible in the launcher; startup logs are saved under `logs`.
Install dependencies first on a fresh clone using the Install section above.

Manual alternative:

```powershell
.\.venv\Scripts\python.exe live\dashboard.py --host 127.0.0.1 --port 8501
```

Open `http://127.0.0.1:8501`. Keep the server local: it is not an authenticated
public web service. The legacy startup scripts are machine-specific; use the
command above on another computer.

Use **Settings** in the top-right corner to open Appearance: choose system/light/
dark mode and six red/green variations: Classic, Emerald (vivid), Mint (pastel),
Terminal (electric), Forest (muted), and Crimson (deep red). Ocean (blue/orange)
and Neon (cyan/pink) are also available. Each preset adapts to light/dark, or pick
custom bullish, bearish, accent, and chart-grid colors. Changes preview instantly
across the chart and dashboard and persist in this browser, not on the server.
Chart style, height, gridlines, price line, model markers, and the OHLC crosshair
are in the same popup; the chart's **Customize** button opens it too. Candles
support solid, hollow-bullish, or outline bodies, adjustable width (40–95%), body
opacity (30–100%), optional borders/wicks, wick thickness (0.5–3px), and independent
bullish/bearish wick colors. The illustrative preview uses the same renderer as
the chart and is not market data. **Swap bullish / bearish** swaps body/semantic
colors; independent wick colors stay unchanged. In line mode, candle controls
are disabled but remembered. The header and Done button remain visible while
the settings body scrolls. Close with Done, Escape, or the backdrop.
**Reset chart only** preserves colors; **Reset all appearance** restores all visual
defaults without touching the forecast horizon or wallet address.

Hover over the chart to inspect candle OHLC in UTC. Dragging or zooming pauses
following the latest candle; **Jump to latest** resumes following while preserving
zoom. **Reset view** or double-click restores the default view. With the chart
focused, **R** resets and **L** jumps to latest. These controls affect the view
only; market data continues refreshing every five seconds.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node --check live\web\app.js
```

## Model research workflow

`scripts\experiment.py` evaluates a training recipe (feature set v2/v3, deadband,
class weights, seed bagging, LightGBM overrides) on the walk-forward selection
folds only; results append to `logs\experiments.jsonl`. `scripts\replay.py` is
the one-shot holdout: it trains through a cutoff (default 2026-09-01) and
simulates the unseen window after it through a fixed decisive rule
(`decisive_rule` is stored in the bundle). Iterate on selection folds, replay
once, and do not tune against the holdout. Feature contract `closed-ohlcv-v3`
adds the columns listed in `features.V3_COLUMNS`; the dashboard accepts both
v2 and v3 bundles. `scripts\forward_test.py` goes further: it fetches 1m
candles live from Binance's futures API for dates AFTER the local data file
ends, verifies the overlap matches exactly, and scores the deployed bundles
on that never-seen window (true forward test).

Measured findings (selection folds 2026-07/08, ledger in
`logs/experiments.jsonl`): seed bagging saturates at 3 seeds; the forward-label
deadband optimum is horizon-dependent (13bp at 5m, 20bp at 15m/60m — wider
deadbands degenerate the flat class); barrier-target labels degenerate at all
horizons; meta-label call filtering helps only at the 5m horizon; expected-move
regression gating never fires (predictions shrink below 1x cost); taker-flow
features (contract v4) add nothing at 15m/60m; positioning features (contract
v5: funding + Binance Vision open-interest history via
`data\download_metrics.py`) also add nothing at any horizon (dlog-loss within
noise of v3; gated profit factor worse) — the model is at the information
ceiling of aggregate public data at these horizons.

Simulator correction (2026-10-15): `path_returns` barriers were scaled by the
raw 1-minute ATR, making the take-profit smaller than the fee at 15m/60m
(net win ~+1bp vs net loss ~-20bp — breakeven ~95%). Barriers now scale by
sqrt(horizon), matching the config's documented "TP/SL = mult x ATR(horizon)"
design. Fees are unchanged (13bp taker / 8bp maker round trip; funding adds
~0.1bp/hour). This correction, not any fee change, is why path metrics moved.

## Risk controls

`scripts\risk_report.py` quantifies the four standing risks against the
deployed bundles on the holdout. Overfitting: `--placebo` runs shuffled labels
through the identical pipeline (noise floor measures dlog-loss +0.003 vs the
real models' +0.05) and the multiple-testing floor (1/configs-evaluated, about
2% on the current ledger) gates what counts as an improvement. Regime
concentration: holdout fills are stratified by causal trend/volatility tags;
training is balanced (~43/41/15 bear/bull/chop) but the decisive gates fire
almost exclusively in high-volatility regimes. Transaction costs: profit
factor is swept across 8-20bp round-trip costs plus funding; the holdout
breakeven sits below 8bp on every horizon, so promotion stays disabled.
Effective sample size: overlapping forward labels deflate independence
(60-minute labels share 59/60 bars, roughly 9k effective windows rather than
543k rows), so block-bootstrap confidence intervals on per-trade returns --
not point estimates -- are the acceptance standard.

Local data, models, logs, wallet settings, credentials and virtual environments
are intentionally excluded from version control.