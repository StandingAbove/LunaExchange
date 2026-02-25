# =========================================================
# Global Configuration
# =========================================================

DAYS_PER_YEAR = 365


# =========================================================
# Data
# =========================================================

DATA_PATH = "/Users/samgeng14/PycharmProjects/LunaExchange/Crypto/Data/cleaned_crypto_data.csv"

PRICE_COLUMN_BTC = "BTC-USD_close"
PRICE_COLUMN_ETH = "ETH-USD_close"


# =========================================================
# Transaction Costs (Crypto Spot Baseline)
# =========================================================

FEE_BPS = 10.0          # exchange fee
SLIPPAGE_BPS = 5.0      # assumed slippage
ANNUAL_BORROW_RATE = 0.0

# For perpetual futures (optional)
USE_FUNDING = False


# =========================================================
# Walk-Forward Settings
# =========================================================

TRAIN_YEARS = 3
VALIDATION_YEARS = 1
TEST_YEARS = 1

EXPANDING_WINDOW = False   # False = rolling


# =========================================================
# Mining Model Defaults
# =========================================================

MINING_Z_WINDOW = 180
MINING_ENTRY_Z = 1.0
MINING_EXIT_Z = 0.0
MINING_USE_LOG_EDGE = True


# =========================================================
# OU Model Defaults
# =========================================================

OU_WINDOW = 180
OU_ENTRY_Z = 1.5
OU_EXIT_Z = 0.0
OU_LONG_SHORT = True


# =========================================================
# Z-Score Model Defaults
# =========================================================

ZSCORE_WINDOW = 180
ZSCORE_ENTRY_Z = 2.0
ZSCORE_EXIT_Z = 0.0
ZSCORE_LONG_SHORT = True

USE_VOL_TARGET = False
VOL_TARGET = 0.15
VOL_WINDOW = 30


# =========================================================
# Trend Model Defaults
# =========================================================

TREND_FAST_WINDOW = 20
TREND_SLOW_WINDOW = 128

TREND_AGGRESSIVE = 1.3
TREND_NEUTRAL = 1.0
TREND_DEFENSIVE = 0.7

TREND_LONG_ONLY = True


# =========================================================
# Pair Trading Defaults (BTC–ETH)
# =========================================================

PAIR_BETA_WINDOW = 180
PAIR_Z_WINDOW = 180
PAIR_ENTRY_Z = 2.0
PAIR_EXIT_Z = 0.0


# =========================================================
# Monte Carlo Settings
# =========================================================

MC_SAMPLES = 1000
MC_BLOCK_SIZE = 20


# =========================================================
# Leverage
# =========================================================

LEVERAGE_CAP = 1.5


# =========================================================
# Stability Testing
# =========================================================

PERTURBATION_RANGE = 0.2     # ±20% for local tests