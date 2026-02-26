"""
Data loading utilities for the dashboard.
"""
from datetime import datetime
import json
from google.cloud import storage
import streamlit as st
import pandas as pd
import pandas_market_calendars as mcal
from io import BytesIO
import yfinance as yf


# Constants
BUCKET_NAME = "wsb-hc-qasap-bucket-1"
SIMULATIONS_AUDIT_PREFIX = "hcf/paper/simulations_audit"
PRODUCTION_AUDIT_PREFIX = "hcf/paper/production_audit"
NYSE = mcal.get_calendar('NYSE').schedule(start_date='1900/01/01', end_date=datetime.today().strftime('%Y-%m-%d'))

@st.cache_resource
def get_gcs_bucket() -> storage.Bucket:
    """Return a cached Google Cloud Storage `Bucket` for reuse across the app.

    This returns the configured `BUCKET_NAME` as a cached resource so other
    functions can reuse the same bucket object instead of recreating clients
    or buckets on every call.
    """
    client = storage.Client()
    return client.bucket(BUCKET_NAME)

@st.cache_data(ttl=3600)
def get_latest_simulations_audit() -> datetime.date:
    """
    Get the most recent audit date from the production simulations directory in GCS.
    
    Returns:
        datetime.date: The most recent audit date found in the simulations_audit directory.
    """
    latest_date = datetime.today().date()
    while True:
        prefix = latest_date.strftime('%Y-%m-%d')
        csv_blob_path = f"{SIMULATIONS_AUDIT_PREFIX}/{prefix}/config.json"

        bucket = get_gcs_bucket()
        blob = bucket.blob(csv_blob_path)

        if blob.exists():
            return latest_date

        latest_date -= pd.Timedelta(days=1)


@st.cache_data(ttl=3600)
def get_latest_production_audit() -> datetime.date:
    """
    Get the most recent audit date from the production directory in GCS.
    
    Returns:
        datetime.date: The most recent audit date found in the production_audit directory.
    """
    latest_date = datetime.today().date()
    while True:
        prefix = latest_date.strftime('%Y-%m-%d')
        csv_blob_path = f"{PRODUCTION_AUDIT_PREFIX}/{prefix}/config.json"

        bucket = get_gcs_bucket()
        blob = bucket.blob(csv_blob_path)

        if blob.exists():
            return latest_date

        latest_date -= pd.Timedelta(days=1)


@st.cache_data
def _get_production_audit_config_on_day(date: datetime.date) -> dict:
    """
    Read `config.json` from a specific production audit folder and return its contents as a dictionary.

    Args:
        date: The date of the production audit to read.
    """
    config_blob_path = f"{PRODUCTION_AUDIT_PREFIX}/{date.strftime('%Y-%m-%d')}/config.json"

    bucket = get_gcs_bucket()
    blob = bucket.blob(config_blob_path)

    content = blob.download_as_text()
    data = json.loads(content)

    return data


def get_production_audit_config() -> dict:
    """
    Read `config.json` from the latest production audit folder and return its contents as a dictionary.

    Returns:
        dict: The contents of `config.json` as a dictionary.
    """
    return _get_production_audit_config_on_day(get_latest_production_audit())


def get_production_audit_models() -> list:
    """
    Read `config.json` from the latest production audit folder and return the
    value of its "models" field.

    Returns:
        list: The list from the `models` field in `config.json` (or an empty
        list if the field is missing).
    """
    return get_production_audit_config().get("models", [])


def get_production_audit_optimizers() -> list:
    """
    Read `config.json` from the latest production audit folder and return the
    value of its "optimizers" field.

    Returns:
        list: The list from the `optimizers` field in `config.json` (or an empty
        list if the field is missing).
    """
    return get_production_audit_config().get("optimizers", [])


def align(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align the given DataFrame's 'date' column to only include NYSE trading days.

    Args:
        df: The DataFrame with a 'date' column to align.

    Returns:
        pd.DataFrame: The aligned DataFrame.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # Only filter in dates that intersect with NYSE trading days
    df = df[df["date"].isin(NYSE.index)]
    return df


@st.cache_data
def _get_model_backtest_on_day(model_name: str, date: datetime.date) -> pd.DataFrame:
    """
    Load a model's backtest results CSV from a specific production audit into a
    pandas DataFrame.

    The CSV path format is:
        hcf/paper/production_audit/YYYY-MM-DD/model_backtests_{model_name}_backtest_results.csv
    """
    csv_blob_path = (
        f"{PRODUCTION_AUDIT_PREFIX}/{date.strftime('%Y-%m-%d')}/"
        f"model_backtests_{model_name}_backtest_results.csv"
    )

    bucket = get_gcs_bucket()
    blob = bucket.blob(csv_blob_path)

    data_bytes = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data_bytes))

    return df

def get_model_backtest(model_name: str) -> pd.DataFrame:
    """
    Load a model's backtest results CSV from the latest production audit into a
    pandas DataFrame.

    The CSV path format is:
        hcf/paper/production_audit/YYYY-MM-DD/model_backtests_{model_name}_backtest_results.csv

    Args:
        model_name: Name of the model to load (used in the filename).

    Returns:
        pd.DataFrame: The backtest results as a pandas DataFrame.
    """
    return align(_get_model_backtest_on_day(model_name, get_latest_production_audit()))



@st.cache_data
def _get_reduced_portfolio_backtest_on_day(model_name: str, date: datetime.date) -> pd.DataFrame:
    """
    Load a reduced portfolio backtest CSV for the given model from a specific
    production audit into a pandas DataFrame.

    The CSV path format is:
        hcf/paper/simulations_audit/YYYY-MM-DD/reduced_portfolio_backtests_{model_name}_backtest_results.csv
    """
    opt = get_active_optimizer()
    csv_blob_path = (
        f"{SIMULATIONS_AUDIT_PREFIX}/{date.strftime('%Y-%m-%d')}/"
        f"reduced_portfolio_backtests_{model_name}_{opt}_backtest_results.csv"
    )

    bucket = get_gcs_bucket()
    blob = bucket.blob(csv_blob_path)

    data_bytes = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data_bytes))

    return df

def get_reduced_portfolio_backtest(model_name: str) -> pd.DataFrame:
    """
    Load a reduced portfolio backtest CSV for the given model from the latest
    simulations audit into a pandas DataFrame.

    The CSV path format is:
        hcf/paper/simulations_audit/YYYY-MM-DD/reduced_portfolio_backtests_{model_name}_backtest_results.csv
    """
    return _get_reduced_portfolio_backtest_on_day(model_name, get_latest_simulations_audit())

@st.cache_data
def _get_portfolio_backtest_on_day(optimizer_name: str, date: datetime.date) -> pd.DataFrame:
    """
    Load a portfolio backtest CSV for the given optimizer from a specific
    production audit into a pandas DataFrame.

    The CSV path format is:
        hcf/paper/production_audit/YYYY-MM-DD/portfolio_backtests_{optimizer_name}_backtest_results.csv
    """
    csv_blob_path = (
        f"{PRODUCTION_AUDIT_PREFIX}/{date.strftime('%Y-%m-%d')}/"
        f"portfolio_backtests_{optimizer_name}_backtest_results.csv"
    )

    bucket = get_gcs_bucket()
    blob = bucket.blob(csv_blob_path)

    data_bytes = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data_bytes))

    return df


def get_portfolio_backtest(optimizer_name: str) -> pd.DataFrame:
    """
    Load a portfolio backtest CSV for the given optimizer from the latest
    production audit into a pandas DataFrame.

    The CSV path format is:
        hcf/paper/production_audit/YYYY-MM-DD/portfolio_backtests_{optimizer_name}_backtest_results.csv

    Args:
        optimizer_name: Name of the optimizer to load (used in the filename).

    Returns:
        pd.DataFrame: The backtest results as a pandas DataFrame.
    """
    return align(_get_portfolio_backtest_on_day(optimizer_name, get_latest_production_audit()))



@st.cache_data
def _get_model_backtest_metrics_on_day(model_name: str, date: datetime.date) -> pd.DataFrame:
    """
    Load an individual model's backtest metrics CSV from a specific production audit into a
    pandas DataFrame.

    The CSV path format is:
        hcf/paper/production_audit/YYYY-MM-DD/model_backtests_{model_name}_backtest_metrics.csv
    """
    csv_blob_path = (
        f"{PRODUCTION_AUDIT_PREFIX}/{date.strftime('%Y-%m-%d')}/"
        f"model_backtests_{model_name}_backtest_metrics.csv"
    )

    bucket = get_gcs_bucket()
    blob = bucket.blob(csv_blob_path)

    data_bytes = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data_bytes))

    return df


def get_model_backtest_metrics(model_name: str) -> pd.DataFrame:
    """
    Load an individual model's backtest metrics CSV from the latest production audit into a
    pandas DataFrame.

    The CSV path format is:
        hcf/paper/production_audit/YYYY-MM-DD/model_backtests_{model_name}_backtest_metrics.csv

    Args:
        model_name: Name of the model to load (used in the filename).

    Returns:
        pd.DataFrame: The backtest metrics as a pandas DataFrame with columns 'metric' and 'value'.
    """
    return _get_model_backtest_metrics_on_day(model_name, get_latest_production_audit())


@st.cache_data
def _get_portfolio_backtest_metrics_on_day(optimizer_name: str, date: datetime.date) -> pd.DataFrame:
    """
    Load portfolio backtest metrics CSV for the given optimizer from a specific
    production audit day into a pandas DataFrame.

    The CSV path format is:
        hcf/paper/production_audit/YYYY-MM-DD/portfolio_backtests_{optimizer_name}_backtest_metrics.csv
    """
    csv_blob_path = (
        f"{PRODUCTION_AUDIT_PREFIX}/{date.strftime('%Y-%m-%d')}/"
        f"portfolio_backtests_{optimizer_name}_backtest_metrics.csv"
    )

    bucket = get_gcs_bucket()
    blob = bucket.blob(csv_blob_path)

    data_bytes = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data_bytes))

    return df


def get_portfolio_backtest_metrics(optimizer_name: str) -> pd.DataFrame:
    """
    Portfolio backtest metrics (CSV path format is):
        hcf/paper/production_audit/YYYY-MM-DD/portfolio_backtests_{optimizer_name}_backtest_metrics.csv

    Args:
        optimizer_name: Name of the optimizer to load (used in the filename).

    Returns:
        pd.DataFrame: backtest metrics pandas DataFrame with cols: 'metric' and 'value'
    """
    return _get_portfolio_backtest_metrics_on_day(optimizer_name, get_latest_production_audit())


@st.cache_data
def _get_historical_nav_on_day(date: datetime.date) -> pd.DataFrame:
    """
    Load historical NAV data from a specific production audit into a pandas DataFrame.

    The CSV path format is:
        hcf/paper/production_audit/YYYY-MM-DD/historical_nav.csv
    """
    csv_blob_path = (
        f"{PRODUCTION_AUDIT_PREFIX}/{date.strftime('%Y-%m-%d')}/"
        f"historical_nav.csv"
    )

    bucket = get_gcs_bucket()
    blob = bucket.blob(csv_blob_path)

    data_bytes = blob.download_as_bytes()
    df = pd.read_csv(BytesIO(data_bytes))
    return df


def get_historical_nav() -> pd.DataFrame:
    """
    Load historical NAV data from the latest production audit into a pandas DataFrame.

    The CSV path format is:
        hcf/paper/production_audit/YYYY-MM-DD/historical_nav.csv
    
    Returns:
        pd.DataFrame: Has two columns, date (YYYY-MM-DD) and nav (float).
    """
    return align(_get_historical_nav_on_day(get_latest_production_audit()))


@st.cache_data
def get_spx_prices_from_date(start_date: datetime.date) -> pd.DataFrame:
    """
    Load historical SPX prices from a specific start date to today.

    Args:
        start_date: The start date as a datetime.date object.
    Returns:
        pd.DataFrame: DataFrame with columns 'date' and 'close'.
    """
    # Initialize ticker object
    spx = yf.Ticker("^GSPC")

    # Fetch historical data
    data = spx.history(start=start_date, end=datetime.today().date())

    # Reset index to make the date a column
    data = data.reset_index()

    # Keep only the relevant columns
    data = data[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})

    # Ensure proper dtypes
    data["date"] = pd.to_datetime(data["date"])
    data["date"] = data["date"].dt.tz_localize(None)
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    return data

@st.cache_data
def get_active_optimizer() -> str:
    """
    TODO: This needs to be changed to support multiple optimizers. This is
    (unfortunately) the logic that the paper portfolio uses to get the optimizer name.
    Get the active optimizer name from the latest production audit config.

    Returns:
        str: The active optimizer name.
    """
    return get_production_audit_optimizers().pop()
