import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "Date",
    "BTC-USD_open",
    "BTC-USD_high",
    "BTC-USD_low",
    "BTC-USD_close",
    "BTC-USD_volume",
    "ETH-USD_open",
    "ETH-USD_high",
    "ETH-USD_low",
    "ETH-USD_close",
    "ETH-USD_volume",
]


def load_raw_crypto_csv(path: str) -> pd.DataFrame:
    """
    Load raw crypto CSV and return cleaned DataFrame indexed by Date.

    Cleaning steps:
    - Parse dates
    - Sort by date
    - Drop duplicate dates (keep last)
    - Drop exact duplicate rows
    - Enforce numeric columns
    - Remove rows with missing close prices
    """

    df = pd.read_csv(path)

    # --- Column check ---
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Parse dates ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # --- Sort ---
    df = df.sort_values("Date")

    # --- Remove duplicate dates (keep last occurrence) ---
    df = df.drop_duplicates(subset=["Date"], keep="last")

    # --- Remove exact duplicate rows ---
    df = df.drop_duplicates()

    # --- Enforce numeric types ---
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Remove rows missing core price data ---
    df = df.dropna(
        subset=[
            "BTC-USD_close",
            "ETH-USD_close",
        ]
    )

    # --- Set index ---
    df = df.set_index("Date")

    # --- Final sort ---
    df = df.sort_index()

    return df


def check_constant_stretches(
    df: pd.DataFrame,
    column: str,
    min_length: int = 5,
) -> pd.DataFrame:
    """
    Detect stretches where a column stays constant for >= min_length days.
    Useful for spotting bad data.
    """

    s = df[column]
    groups = (s != s.shift()).cumsum()
    counts = s.groupby(groups).transform("count")

    mask = counts >= min_length
    return df.loc[mask, [column]]


def basic_data_diagnostics(df: pd.DataFrame) -> dict:
    """
    Return basic dataset diagnostics.
    """

    diagnostics = {}

    diagnostics["start_date"] = df.index.min()
    diagnostics["end_date"] = df.index.max()
    diagnostics["n_rows"] = len(df)
    diagnostics["n_missing"] = df.isna().sum().sum()

    diagnostics["btc_zero_volume_days"] = int(
        (df["BTC-USD_volume"] == 0).sum()
    )

    diagnostics["eth_zero_volume_days"] = int(
        (df["ETH-USD_volume"] == 0).sum()
    )

    diagnostics["duplicate_index"] = int(
        df.index.duplicated().sum()
    )

    return diagnostics