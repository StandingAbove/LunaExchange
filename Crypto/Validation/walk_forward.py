import numpy as np
import pandas as pd


DAYS_PER_YEAR = 365


# =========================================================
# 1. Window Generator
# =========================================================

def generate_walk_forward_windows(
    index: pd.DatetimeIndex,
    train_years: int = 3,
    test_years: int = 1,
    expanding: bool = False,
):
    """
    Yield (train_idx, test_idx) for walk-forward.

    expanding=False -> rolling window
    expanding=True  -> expanding window
    """

    dates = pd.Series(index)
    start_date = dates.min()
    end_date = dates.max()

    current_train_start = start_date
    current_train_end = start_date + pd.DateOffset(years=train_years)

    while True:

        current_test_end = current_train_end + pd.DateOffset(years=test_years)

        if current_test_end > end_date:
            break

        train_mask = (index >= current_train_start) & (index < current_train_end)
        test_mask = (index >= current_train_end) & (index < current_test_end)

        train_idx = index[train_mask]
        test_idx = index[test_mask]

        yield train_idx, test_idx

        if expanding:
            current_train_end = current_test_end
        else:
            current_train_start = current_train_start + pd.DateOffset(years=test_years)
            current_train_end = current_train_end + pd.DateOffset(years=test_years)


# =========================================================
# 2. Core Walk-Forward Runner
# =========================================================

def run_walk_forward(
    df: pd.DataFrame,
    price_column: str,
    fit_function,
    signal_function,
    train_years: int = 3,
    test_years: int = 1,
    expanding: bool = False,
):
    """
    Generic walk-forward engine.

    fit_function(train_df) -> dict of fitted parameters
    signal_function(df, params_dict) -> position series
    """

    all_returns = []
    all_positions = []

    for train_idx, test_idx in generate_walk_forward_windows(
        df.index,
        train_years=train_years,
        test_years=test_years,
        expanding=expanding,
    ):

        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]

        if len(train_df) < 50 or len(test_df) < 10:
            continue

        # Fit only on training data
        params = fit_function(train_df)

        # Generate signal using frozen params
        full_signal = signal_function(df, params)

        test_position = full_signal.loc[test_idx]

        price = df[price_column].astype(float)
        ret = price.pct_change().fillna(0.0)

        test_returns = test_position * ret.loc[test_idx]

        all_returns.append(test_returns)
        all_positions.append(test_position)

    if not all_returns:
        return {}

    oos_returns = pd.concat(all_returns).sort_index()
    oos_position = pd.concat(all_positions).sort_index()

    return {
        "oos_returns": oos_returns,
        "oos_position": oos_position,
    }


# =========================================================
# 3. Simple Static Train/Test Split
# =========================================================

def static_split(
    df: pd.DataFrame,
    train_end_date,
    price_column: str,
    fit_function,
    signal_function,
):
    """
    Simple single train/test split.
    """

    train_df = df.loc[:train_end_date]
    test_df = df.loc[train_end_date:]

    params = fit_function(train_df)
    full_signal = signal_function(df, params)

    price = df[price_column].astype(float)
    ret = price.pct_change().fillna(0.0)

    test_position = full_signal.loc[test_df.index]
    test_returns = test_position * ret.loc[test_df.index]

    return {
        "test_returns": test_returns,
        "test_position": test_position,
    }