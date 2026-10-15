# SOL Perp Research Bot

SOL directional forecasting, chronological model evaluation, and a local
read-only dashboard for research intended for **Jupiter Perps**.

**This project does not sign transactions or place orders.** Predictions are
model outputs, not proven win probabilities or demonstrated profitability.

## Data and validation

- Binance futures candles provide historical research data.
- Hyperliquid supplies the current live reference candles and market feed.
- Jupiter Perps supplies public wallet positions and recent activity.
- Live input requires the `closed-ohlcv-v2` feature contract and valid completed
  candles. Legacy model bundles fail closed.
- Candidates are saved separately, with promotion disabled. Jupiter-specific
  costs and reference-price transfer are not yet verified.

See [VENUES.md](VENUES.md) for the data/execution distinction. Existing fee
constants are legacy research assumptions, not validated Jupiter costs.

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
binaries are included in this repository. The dashboard currently expects
local `models/model_fast.pkl`, `model_mid.pkl`, and `model_slow.pkl` bundles;
a fresh clone cannot produce predictions until compatible bundles are supplied.
Do not load untrusted pickle/joblib artifacts.

## Dashboard

```powershell
.\.venv\Scripts\python.exe live\dashboard.py --host 127.0.0.1 --port 8501
```

Open `http://127.0.0.1:8501`. Keep the server local: it is not an authenticated
public web service. The legacy startup scripts are machine-specific; use the
command above on another computer.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node --check live\web\app.js
```

Local data, models, logs, wallet settings, credentials and virtual environments
are intentionally excluded from version control.