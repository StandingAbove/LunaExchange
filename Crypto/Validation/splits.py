import pandas as pd


# =========================================================
# 1. Static Date Split
# =========================================================

def static_split(
    df: pd.DataFrame,
    train_end,
    val_end=None,
):
    """
    Split into:
        train
        validation (optional)
        test

    train:      <= train_end
    validation: (train_end, val_end]
    test:       > val_end (or > train_end if no val)
    """

    train = df.loc[:train_end]

    if val_end is not None:
        val = df.loc[train_end:val_end]
        test = df.loc[val_end:]
        return train, val, test

    test = df.loc[train_end:]
    return train, test


# =========================================================
# 2. Expanding Window Generator
# =========================================================

def expanding_splits(
    df: pd.DataFrame,
    initial_train_years: int = 3,
    test_years: int = 1,
):
    """
    Expanding window walk-forward.

    Example:
        Train: 2016–2018 → Test: 2019
        Train: 2016–2019 → Test: 2020
        Train: 2016–2020 → Test: 2021
    """

    index = df.index
    start_date = index.min()
    end_date = index.max()

    train_end = start_date + pd.DateOffset(years=initial_train_years)

    while True:

        test_end = train_end + pd.DateOffset(years=test_years)

        if test_end > end_date:
            break

        train = df.loc[:train_end]
        test = df.loc[train_end:test_end]

        yield train, test

        train_end = test_end


# =========================================================
# 3. Rolling Window Generator
# =========================================================

def rolling_splits(
    df: pd.DataFrame,
    train_years: int = 3,
    test_years: int = 1,
):
    """
    Rolling window walk-forward.

    Example:
        Train: 2016–2018 → Test: 2019
        Train: 2017–2019 → Test: 2020
        Train: 2018–2020 → Test: 2021
    """

    index = df.index
    start_date = index.min()
    end_date = index.max()

    train_start = start_date
    train_end = train_start + pd.DateOffset(years=train_years)

    while True:

        test_end = train_end + pd.DateOffset(years=test_years)

        if test_end > end_date:
            break

        train = df.loc[train_start:train_end]
        test = df.loc[train_end:test_end]

        yield train, test

        train_start = train_start + pd.DateOffset(years=test_years)
        train_end = train_end + pd.DateOffset(years=test_years)


# =========================================================
# 4. Fixed Ratio Split (Time-Based)
# =========================================================

def ratio_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
):
    """
    Time-ordered split based on ratio.
    """

    n = len(df)
    split_point = int(n * train_ratio)

    train = df.iloc[:split_point]
    test = df.iloc[split_point:]

    return train, test


# =========================================================
# 5. Multi-Segment Split
# =========================================================

def multi_segment_split(
    df: pd.DataFrame,
    train_years: int,
    val_years: int,
    test_years: int,
):
    """
    Returns generator of:
        train, validation, test
    """

    index = df.index
    start_date = index.min()
    end_date = index.max()

    train_start = start_date

    while True:

        train_end = train_start + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(years=val_years)
        test_end = val_end + pd.DateOffset(years=test_years)

        if test_end > end_date:
            break

        train = df.loc[train_start:train_end]
        val = df.loc[train_end:val_end]
        test = df.loc[val_end:test_end]

        yield train, val, test

        train_start = train_start + pd.DateOffset(years=test_years)