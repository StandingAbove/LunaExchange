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


def _remove_ohlc_outliers(df: pd.DataFrame, asset_prefix: str, max_close_to_high_ratio: float = 5.0) -> pd.DataFrame:
    """
    Remove rows where close is implausibly far from daily high/low.
    This catches data-entry spikes (e.g., ETH close typo while high/low are normal).
    """
    close_col = f"{asset_prefix}_close"
    high_col = f"{asset_prefix}_high"
    low_col = f"{asset_prefix}_low"

    close = df[close_col].astype(float)
    high = df[high_col].astype(float)
    low = df[low_col].astype(float)

    valid = (close > 0) & (high > 0) & (low > 0)
    valid &= close <= (high * max_close_to_high_ratio)
    valid &= close >= (low / max_close_to_high_ratio)

    return df.loc[valid].copy()


def load_raw_crypto_csv(path: str, start_date: str = "2017-11-01") -> pd.DataFrame:
    """
    Load raw crypto CSV and return cleaned DataFrame indexed by Date.

    Cleaning steps:
    - Parse dates
    - Sort by date
    - Drop duplicate dates (keep last)
    - Drop exact duplicate rows
    - Enforce numeric columns
    - Remove rows with missing close prices
    - Trim to start_date (default: 2017-11-01)
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

    # --- Remove obvious OHLC outliers (bad ticks/typos) ---
    df = _remove_ohlc_outliers(df, "BTC-USD")
    df = _remove_ohlc_outliers(df, "ETH-USD")

    # --- Set index ---
    df = df.set_index("Date")

    # --- Final sort ---
    df = df.sort_index()

    # --- Global backtest start date alignment ---
    if start_date is not None:
        df = df.loc[pd.Timestamp(start_date):].copy()

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

def load_ibit_with_mining_cost(
    ibit_path: str,
    cleaned_crypto_path: str,
    forward_fill_mining_cost: bool = True,
) -> pd.DataFrame:
    """
    Load IBIT close data and align mining cost from cleaned_crypto_data.csv by date.

    Join logic:
    - Build daily IBIT close on date index.
    - Build mining-cost series from cleaned dataset.
    - If `forward_fill_mining_cost` is True, left-join to IBIT dates then forward-fill
      mining cost (useful when mining cost is less frequent).
    - Otherwise, perform strict inner join on exact overlapping dates.
    """
    ibit = pd.read_csv(ibit_path).copy()
    cleaned = pd.read_csv(cleaned_crypto_path).copy()

    # --- IBIT date and close ---
    ibit_date_col = next((c for c in ["Date", "date", "Timestamp", "timestamp"] if c in ibit.columns), None)
    ibit_close_col = next((c for c in ["Close", "close", "Adj Close", "adj_close", "Price", "price"] if c in ibit.columns), None)

    if ibit_date_col is None or ibit_close_col is None:
        raise ValueError("IBIT data must contain date and close/price columns.")

    ibit_df = pd.DataFrame({
        "date": pd.to_datetime(ibit[ibit_date_col], format="%m/%d/%y", errors="coerce"),
        "close": pd.to_numeric(
            ibit[ibit_close_col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False),
            errors="coerce",
        ),
    }).dropna(subset=["date", "close"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")

    # --- cleaned dataset date and mining cost ---
    cleaned_date_col = next((c for c in ["Date", "date", "Timestamp", "timestamp"] if c in cleaned.columns), None)
    if cleaned_date_col is None:
        raise ValueError("cleaned_crypto_data.csv must contain a date column.")

    mining_candidates = [
        "COST_TO_MINE",
        "mining_cost",
        "cost_to_mine",
        "production_cost",
        "cost",
    ]
    mining_col = next((c for c in mining_candidates if c in cleaned.columns), None)
    if mining_col is None:
        raise ValueError(f"Could not find mining cost column in cleaned data. Tried: {mining_candidates}")

    mining_df = pd.DataFrame({
        "date": pd.to_datetime(cleaned[cleaned_date_col], errors="coerce"),
        "mining_cost": pd.to_numeric(cleaned[mining_col], errors="coerce"),
    }).dropna(subset=["date", "mining_cost"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")

    if forward_fill_mining_cost:
        merged = ibit_df.merge(mining_df, on="date", how="left").sort_values("date")
        merged["mining_cost"] = merged["mining_cost"].ffill()
        merged = merged.dropna(subset=["mining_cost"])
    else:
        merged = ibit_df.merge(mining_df, on="date", how="inner").sort_values("date")

    return merged.set_index("date")
