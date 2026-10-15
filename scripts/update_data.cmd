@echo off
rem One-shot incremental data update: refresh Hyperliquid 1m candles + funding.
rem Run daily (or scheduled) to keep local history current before retraining.
cd /d C:\Users\lumin\Desktop\sol-perp-bot
.venv\Scripts\python.exe data\download_hist.py --days 3
.venv\Scripts\python.exe data\download_binance_vision.py --months 13
