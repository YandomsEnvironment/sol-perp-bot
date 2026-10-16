# Prediction data and trading venue

The execution venue is configured for **Lighter** perpetual DEX ([app.lighter.xyz/trade/SOL](https://app.lighter.xyz/trade/SOL)) featuring **0 fees**
($0.00 / 0 bps fee drag). This project is currently read-only / paper-mode: it does
not sign live transactions or place real orders.

## Reference data is not execution data

- Historical model research uses Binance futures candles.
- Current live candles, ticks, funding and trade flow come from Hyperliquid
  as a reference market. The chart is a reference market feed, NOT a venue fill history.
- Wallet positions and activity integrate with Solana public APIs.
- Solana RPC supplies blockchain state and transactions, not a single canonical
  SOL/USD candle history. An oracle/indexer/market feed is needed for prices.

Directional prediction may use reference-market prices; transfer is a testable
assumption, not a requirement that every input come from the execution venue. Differences
in basis, oracle timing and volume can affect features and short-horizon labels.
Model input compatibility remains required even for raw probability display.

## Execution validation & Lighter Zero-Fee Structure

Lighter features 0 execution fees ($0.00 taker / $0.00 maker, 0 bps fee drag).
In paper trading and the live AI advisor, breakeven entry price equals the exact fill
price without hurdle penalties.

Lighter integration details:
- Web App: https://app.lighter.xyz/trade/SOL
- API Documentation: https://apidocs.lighter.xyz
- REST Base URL: `https://mainnet.zklighter.elliot.ai/api/v1/`
- WebSocket Feed: `wss://mainnet.zklighter.elliot.ai/stream`
- Market ID (SOL Perp): `2`
- Helper Client: [`live/lighter_client.py`](live/lighter_client.py)

Historical model research artifacts from earlier experiments are preserved for
methodological comparison. New candidates remain blocked from automated promotion
(`EXECUTION_COSTS_VERIFIED = False`) until live on-chain execution testing is completed.

Run checks from `C:\Users\lumin\Desktop\sol-perp-bot`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node --check live\web\app.js
```