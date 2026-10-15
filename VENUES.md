# Prediction data and trading venue

The intended trading venue is **Jupiter Perps on Solana**. This project is
currently read-only: it does not sign transactions or place orders.

## Reference data is not execution data

- Historical model research uses Binance futures candles.
- Current live candles, ticks, funding and trade flow come from Hyperliquid
  as a reference market. The chart is NOT a Jupiter fill/oracle history.
- Wallet positions and activity use Jupiter Perps public APIs.
- Solana RPC supplies blockchain state and transactions, not a single canonical
  SOL/USD candle history. An oracle/indexer/market feed is needed for prices.

Directional prediction may use reference-market prices; transfer is a testable
assumption, not a requirement that every input come from Jupiter. Differences
in basis, oracle timing and volume can affect features and short-horizon labels.
Model input compatibility remains required even for raw probability display.

## Execution validation

The fee constants in `config.py` are retained legacy research scenarios, NOT
verified Jupiter fees. Jupiter-specific price impact, opening/closing costs,
borrowing, swaps and transaction/priority fees must be measured separately.
Existing reports and saved artifacts are historical experiments, not Jupiter
profitability evidence. New candidates remain blocked from promotion.

Next evaluation: collect timestamped Jupiter prices alongside the reference
feed, compare forecasts against both, and paper-replay Jupiter costs. Do not
replace historical candles with sparse current-price samples or silently mix
sources in existing data files.

Official references:
- https://perps-api.jup.ag/v1/docs
- https://developers.jup.ag/docs/perps

Run checks from `C:\Users\lumin\Desktop\sol-perp-bot`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node --check live\web\app.js
```