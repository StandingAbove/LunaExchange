import pandas as pd

from config import *
from Data.raw_data_loader import load_raw_crypto_csv
from Models.zscore import zscore_signal
from Models.trend import trend_signal
from Models.ou import ou_signal
from Models.pair_trading import build_spread, pair_signal_zscore
from Backtest.engine import run_backtest
from Backtest.metrics import build_summary_table
from Validation.walk_forward import run_walk_forward
from Validation.stability import evaluate_surface, stability_score
from Validation.monte_carlo import monte_carlo_report


# =========================================================
# Select Model
# =========================================================

MODEL_TO_RUN = "zscore"   # options: zscore, trend, ou, pair


# =========================================================
# Load Data
# =========================================================

df = load_raw_crypto_csv(DATA_PATH)


# =========================================================
# Model Fit + Signal Wrapper
# =========================================================

def fit_static(train_df):
    """
    For now, static parameters from config.
    Later you can optimize inside this function.
    """
    return {}


def signal_wrapper(full_df, params):

    if MODEL_TO_RUN == "zscore":
        return zscore_signal(
            full_df[PRICE_COLUMN_BTC].apply(pd.Series),
        )

    elif MODEL_TO_RUN == "trend":
        return trend_signal(
            full_df,
            price_column=PRICE_COLUMN_BTC,
            fast_window=TREND_FAST_WINDOW,
            slow_window=TREND_SLOW_WINDOW,
            long_only=TREND_LONG_ONLY,
            leverage_aggressive=TREND_AGGRESSIVE,
            leverage_neutral=TREND_NEUTRAL,
            leverage_defensive=TREND_DEFENSIVE,
        )

    elif MODEL_TO_RUN == "ou":
        return ou_signal(
            full_df[PRICE_COLUMN_BTC],
            window=OU_WINDOW,
            entry_z=OU_ENTRY_Z,
            exit_z=OU_EXIT_Z,
            long_short=OU_LONG_SHORT,
        )

    elif MODEL_TO_RUN == "pair":
        spread = build_spread(full_df, window=PAIR_BETA_WINDOW)
        return pair_signal_zscore(
            spread,
            window=PAIR_Z_WINDOW,
            entry_z=PAIR_ENTRY_Z,
            exit_z=PAIR_EXIT_Z,
        )

    else:
        raise ValueError("Invalid MODEL_TO_RUN")


# =========================================================
# In-Sample Backtest
# =========================================================

position = signal_wrapper(df, {})

results = run_backtest(
    price_series=df[PRICE_COLUMN_BTC],
    position=position,
    fee_bps=FEE_BPS,
    slippage_bps=SLIPPAGE_BPS,
    annual_borrow_rate=ANNUAL_BORROW_RATE,
    leverage_cap=LEVERAGE_CAP,
)

summary_table = build_summary_table({
    "Gross": {
        "returns": results["gross_returns"],
        "position": position,
    },
    "Net": {
        "returns": results["net_returns"],
        "position": position,
    },
})

print("==== In-Sample Results ====")
print(summary_table)


# =========================================================
# Walk-Forward
# =========================================================

wf = run_walk_forward(
    df,
    price_column=PRICE_COLUMN_BTC,
    fit_function=fit_static,
    signal_function=signal_wrapper,
    train_years=TRAIN_YEARS,
    test_years=TEST_YEARS,
    expanding=EXPANDING_WINDOW,
)

if wf:
    wf_results = run_backtest(
        price_series=df[PRICE_COLUMN_BTC],
        position=wf["oos_position"],
        fee_bps=FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        annual_borrow_rate=ANNUAL_BORROW_RATE,
        leverage_cap=LEVERAGE_CAP,
    )

    wf_summary = build_summary_table({
        "OOS Net": {
            "returns": wf_results["net_returns"],
            "position": wf["oos_position"],
        }
    })

    print("==== Walk-Forward Results ====")
    print(wf_summary)


# =========================================================
# Stability Test (Example for Z-Score)
# =========================================================

if MODEL_TO_RUN == "zscore":

    param_grid = {
        "window": [140, 160, 180, 200],
    }

    surface = evaluate_surface(
        df,
        price_column=PRICE_COLUMN_BTC,
        param_grid=param_grid,
        signal_function=lambda d, p: zscore_signal(
            d[PRICE_COLUMN_BTC],
            window=p["window"],
            entry_z=ZSCORE_ENTRY_Z,
            exit_z=ZSCORE_EXIT_Z,
            long_short=ZSCORE_LONG_SHORT,
        ),
    )

    stability = stability_score(surface)

    print("==== Stability ====")
    print(stability)


# =========================================================
# Monte Carlo
# =========================================================

mc = monte_carlo_report(results["net_returns"], n_samples=MC_SAMPLES)

print("==== Monte Carlo ====")
print(mc)