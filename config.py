"""Central configuration — single source of truth for all constants.

Maps directly to plan doc sections: fees (§11), labels (§2/M5), trading and
risk rules (R1/R5/X3), walk-forward (M1/M3), model params (M4).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_STORE = PROJECT_ROOT / "data_store"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
for _d in (DATA_STORE, MODELS_DIR, LOGS_DIR):
    _d.mkdir(exist_ok=True)

# --- Market ---
COIN = "SOL"
EXECUTION_VENUE = "Jupiter Perps"
EXECUTION_ENABLED = False
EXECUTION_COSTS_VERIFIED = False
LIVE_REFERENCE_SOURCE = "Hyperliquid"
HISTORICAL_SOURCE = "Binance futures"
JUPITER_PERPS_API_URL = "https://perps-api.jup.ag/v1"
VENUE_VALIDATION_REASON = (
    "Reference-price forecasting only; Jupiter Perps price transfer and execution costs are unverified"
)
BTC_COIN = "BTC"             # reference market (lead-lag features)
BINANCE_SYMBOL = "SOLUSDT"   # for Binance Vision bulk history
HIST_START = "2025-04-01"    # aim for 12+ months of 1m candles

# --- Jupiter Perps execution fees ---
JUPITER_FEE_USD = 0.65       # $0.65 USD observed execution fee per trade / side
# On a standard ~10 SOL / ~$1,080 position, $0.65 USD = ~0.06% (6 bps) per side.
TAKER_FEE = 0.0006           # 0.06% Jupiter Perps open/close fee
MAKER_FEE = 0.0006           # Jupiter Perps pool execution (symmetric open/close fee)
SLIPPAGE = 0.0002            # planning number 0.02% per side (oracle spread / price impact)
ROUND_TRIP_COST = 2 * (TAKER_FEE + SLIPPAGE)      # ~0.16% (16 bps) round trip
MAKER_ROUND_TRIP_COST = 2 * TAKER_FEE             # ~0.12% (12 bps) base pool fee

# --- Triple-barrier labels (ATR-scaled) ---
HORIZONS_MIN = {"fast": 5, "mid": 15, "slow": 60}   # suite per M5/R6
TP_ATR = 2.0                 # take-profit = 2 x ATR(horizon) — asymmetric:
SL_ATR = 1.0                 # stop = 1 x ATR. With 1:1 barriers, fees+slip
                             # force a ~72% breakeven hit rate (measured
                             # 2026-10); 2:1 cuts it to ~48%.

# --- Trading rules ---
CONF_THRESHOLD = 0.60        # trade only if calibrated P(class) exceeds this
EV_COST_MULT = 3.0           # expected move must clear 3x round-trip cost (R1)
MAX_LEVERAGE = 3             # isolated margin cap (X3)

# --- Volatility regime filter (Chop filter) ---
CHOP_FILTER_ENABLED = True   # Suppress directional entries during low-volatility sideways compression
MIN_ATR_PERCENTILE = 25.0    # Stand aside if 1m ATR percentile is below this threshold (sideways compression)

# --- Paper trading & forward PnL tracking ---
PAPER_TRADING_ENABLED = True
PAPER_STARTING_BALANCE_USD = 10000.0   # $10,000 virtual forward test account
PAPER_POSITION_USD = 1000.0            # $1,000 notional per paper trade (~10 SOL @ $100 or ~5.4 SOL @ $185)
PAPER_FEE_USD = JUPITER_FEE_USD        # $0.65 USD fee per trade / side ($1.30 round trip)
PAPER_SLIPPAGE = SLIPPAGE              # 0.02% per side
PAPER_MAX_HOLD_BARS = 3                # Max trade duration = 3 x horizon bars (e.g. 45 min for 15m horizon)

# --- Risk (R5) ---
RISK_PER_TRADE = 0.005       # 0.5% of account risked per trade
DAILY_LOSS_CAP = 0.02        # kill-switch at -2% day
MONTHLY_LOSS_CAP = 0.10      # kill-switch at -10% month

# --- Walk-forward (M1/M3) ---
TRAIN_MONTHS = 9
TEST_MONTHS = 1
PURGE_MIN = 60               # drop train rows whose label window touches test
EMBARGO_MIN = 60             # extra gap after test start
SAMPLE_HALF_LIFE_DAYS = 150.0  # exponential recency weighting (M2)

# --- LightGBM (M4: shallow + regularized; depth memorizes noise) ---
LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=3,
    learning_rate=0.03,
    n_estimators=2000,
    num_leaves=31,
    max_depth=5,
    min_data_in_leaf=400,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    verbosity=-1,
    random_state=42,
    deterministic=True,
    force_col_wise=True,
    n_jobs=4,
)

# --- Execution (X1) ---
MAKER_PATIENCE_S = 8
MAKER_REPRICES = 4

# --- Dashboard ---
DASHBOARD_REFRESH_S = 5         # model/plan panel cadence
DASHBOARD_CACHE_TTL = 4         # data cache slightly shorter than refresh
TICK_REFRESH_S = 0.5            # realtime price layer cadence

# --- API endpoints ---
HL_API_URL = "https://api.hyperliquid.xyz"
HL_WS_URL = "wss://api.hyperliquid.xyz/ws"
HL_TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"
HL_TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"
