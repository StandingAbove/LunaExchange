import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from datetime import datetime, date, timedelta

from src.dashboard.utils import (
    get_latest_production_audit,
    get_production_audit_models,
    get_model_backtest,
    get_portfolio_backtest,
    get_spx_prices_from_date,
    get_active_optimizer,
    get_reduced_portfolio_backtest
)


def get_marginal_return_stream(df_base: pd.DataFrame, df_reduced: pd.DataFrame) -> pd.DataFrame:
    """
    Compute marginal return stream given base and reduced backtest DataFrames.

    Args:
        df_base: DataFrame with base model backtest results.
        df_reduced: DataFrame with reduced portfolio backtest results.
    Returns:
        DataFrame with marginal daily returns of the marginal portfolio.
    """
    df_base = df_base.copy()
    df_reduced = df_reduced.copy()

    df_base["date"] = pd.to_datetime(df_base["date"])
    df_reduced["date"] = pd.to_datetime(df_reduced["date"])

    # Align on date
    merged = pd.merge(
        df_base[['date', 'daily_return']],
        df_reduced[['date', 'daily_return']],
        on='date',
        suffixes=('_base', '_reduced')
    )
    
    # Compute marginal daily return
    merged['daily_return'] = merged['daily_return_base'] - merged['daily_return_reduced']

    # Build cumulative return + drawdown so we can reuse calculate_metrics_from_backtest unchanged
    r = merged["daily_return"].astype(float)
    equity = (1.0 + r).cumprod()
    merged["cumulative_return"] = equity - 1.0
    peak = equity.cummax()
    merged["drawdown"] = (equity / peak) - 1.0
    
    return merged[["date", "daily_return", "cumulative_return", "drawdown"]]

def calculate_metrics_from_backtest(df: pd.DataFrame) -> dict:
    """
    Calculate performance metrics from filtered backtest results DataFrame.
    Not using the backtest metrics df because this dynamically recalculates metrics based on dates

    Args:
        df: DataFrame with columns 'date', 'daily_return', 'cumulative_return', 'drawdown', etc.

    Returns:
        dict: Dictionary of metric names to values
    """
    metrics = {}

    if len(df) == 0:
        return metrics

    # Total return (final cumulative return)
    if 'cumulative_return' in df.columns:
        metrics['total_return'] = df['cumulative_return'].iloc[-1]

    # Calculate time period in years
    if 'date' in df.columns and len(df) > 1:
        days = len(df)
        years = days / 252  # Assuming 252 trading days per year
    else:
        years = 1.0

    # Annualized return
    if 'total_return' in metrics and years > 0:
        total_return = metrics['total_return']
        metrics['annualized_return'] = (1 + total_return) ** (1 / years) - 1

    # Annualized volatility
    if 'daily_return' in df.columns and len(df) > 1:
        daily_vol = df['daily_return'].std()
        metrics['annualized_volatility'] = daily_vol * np.sqrt(252)

    # Sharpe ratio (assuming 0% risk-free rate)
    if 'annualized_return' in metrics and 'annualized_volatility' in metrics:
        if metrics['annualized_volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_volatility']
        else:
            metrics['sharpe_ratio'] = 0.0

    # Sortino ratio (using downside deviation)
    if 'daily_return' in df.columns and len(df) > 1:
        negative_returns = df[df['daily_return'] < 0]['daily_return']
        if len(negative_returns) > 0:
            downside_vol = negative_returns.std() * np.sqrt(252)
            if downside_vol > 0 and 'annualized_return' in metrics:
                metrics['sortino_ratio'] = metrics['annualized_return'] / downside_vol
            else:
                metrics['sortino_ratio'] = 0.0
        else:
            metrics['sortino_ratio'] = metrics.get('sharpe_ratio', 0.0)

    # Maximum drawdown
    if 'drawdown' in df.columns:
        metrics['max_drawdown'] = df['drawdown'].min()

    # Average drawdown
    if 'drawdown' in df.columns:
        metrics['avg_drawdown'] = df['drawdown'].mean()

    # Win rate
    if 'daily_return' in df.columns and len(df) > 1:
        # Count days with positive returns
        returns_series = df['daily_return']
        winning_days = (returns_series > 0).sum()
        # Only count days where there was actual trading (non-zero returns)
        trading_days = (returns_series != 0).sum()
        if trading_days > 0:
            metrics['win_rate'] = winning_days / trading_days
        else:
            metrics['win_rate'] = 0.0

    # Average daily return
    if 'daily_return' in df.columns:
        metrics['avg_daily_return'] = df['daily_return'].mean()

    return metrics


def app():
    # Header
    latest_audit = get_latest_production_audit()

    left, right = st.columns([3, 1])
    with left:
        st.title("Marginal Model Backtests")
        st.caption("Compare model backtests and marginal contributions")
    with right:
        st.metric("Latest Production Audit", str(latest_audit))

    st.divider()

    # Load available models and optimizers from the latest production audit
    models = get_production_audit_models()
    
    # Controls
    selected_models = st.multiselect(
        "Models",
        options=models + [f"{m}_marginal" for m in models]
    )

    with st.expander("Controls", expanded=True):
        st.session_state["selected_models"] = selected_models

        # Date controls
        st.subheader("Date Range")
        preset = st.radio(
            "Preset",
            options=["Custom", "3M", "6M", "1Y", "YTD"],
            horizontal=True,
        )

        # default range (only used if Custom)
        default_start = datetime(2024, 1, 1).date()
        default_end = datetime.today().date()

        # Apply preset logic
        if preset == "3M":
            start_date = (date.today() - timedelta(days=90))
            end_date = date.today()
        elif preset == "6M":
            start_date = (date.today() - timedelta(days=180))
            end_date = date.today()
        elif preset == "1Y":
            start_date = (date.today() - timedelta(days=365))
            end_date = date.today()
        elif preset == "YTD":
            start_date = date(date.today().year, 1, 1)
            end_date = date.today()
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=default_start,
                    key="start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=default_end,
                    key="end_date"
                )

        # Toggles
        colA, colB = st.columns(2)
        with colA:
            show_aggregate_portfolio = st.checkbox(
                "Show Aggregate Portfolio Backtest",
                value=st.session_state.get("show_aggregate_portfolio", False),
                key="show_aggregate_portfolio",
            )
        with colB:
            show_spx = st.checkbox(
                "Show S&P 500 Equity Curve",
                value=st.session_state.get("show_spx", False),
                key="show_spx",
            )
    
    # Guardrail
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    # Build and plot combined equity curves for selected models and optimizers
    combined_series = {}
    metrics_data = {}  # Store metrics for table display

    # Load model backtests
    for model in selected_models:
        try:
            if model.endswith("_marginal"):
                # grab the full portfolio backtest and compute the marginal return series
                base_model = model[:-9]  # remove '_marginal'
                opt = get_active_optimizer()
                df_base = get_portfolio_backtest(opt)
                df_reduced = get_reduced_portfolio_backtest(base_model)
                df = get_marginal_return_stream(df_base, df_reduced)
            else:
                # normal model, not marginal
                df = get_model_backtest(model)
        except Exception as exc:
            st.warning(f"Could not load backtest for model '{model}': {exc}")
            continue

        if 'date' not in df.columns or 'daily_return' not in df.columns:
            print(df)
            st.warning(f"Backtest file for '{model}' missing expected columns")
            continue

        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Filter by date range
        mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        df_filtered = df[mask]

        if len(df_filtered) == 0:
            st.warning(f"No data for model '{model}' in selected date range")
            continue

        daily_returns = df_filtered.set_index('date')['daily_return'].fillna(0)
        daily_returns.iloc[0] = 0 # set initial day to zero for proper cumulative calculation
        series = (1 + daily_returns).cumprod().rename(model)
        df_filtered["cumulative_return"] = series.values - 1.0
        peak = series.cummax()
        df_filtered["drawdown"] = (series / peak).values - 1.0
        combined_series[model] = series

        # Calculate metrics from filtered data
        try:
            metrics = calculate_metrics_from_backtest(df_filtered)
            metrics_data[model] = metrics
        except Exception as exc:
            st.warning(f"Could not calculate metrics for model '{model}': {exc}")

    # Load optimizer (portfolio) backtests
    if show_aggregate_portfolio:
        try:
            opt = get_active_optimizer()
            pdf = get_portfolio_backtest(opt)
        except Exception as exc:
            st.warning(f"Could not load portfolio backtest for optimizer '{opt}': {exc}")

        pdf = pdf.copy()
        pdf['date'] = pd.to_datetime(pdf['date'])
        pdf = pdf.sort_values('date')

        # Filter by date range
        mask = (pdf['date'].dt.date >= start_date) & (pdf['date'].dt.date <= end_date)
        pdf_filtered = pdf[mask]

        daily_returns = pdf_filtered.set_index('date')['daily_return'].fillna(0)
        daily_returns.iloc[0] = 0 # set initial day to zero for proper cumulative calculation
        series_title = f"Aggregate Portfolio"
        series = (1 + daily_returns).cumprod().rename(series_title)
        pdf_filtered["cumulative_return"] = series.values - 1.0
        peak = series.cummax()
        pdf_filtered["drawdown"] = (series / peak).values - 1.0
        combined_series[series_title] = series

        # Calculate metrics from filtered data
        try:
            metrics = calculate_metrics_from_backtest(pdf_filtered)
            metrics_data["Portfolio"] = metrics
        except Exception as exc:
            st.warning(f"Could not calculate metrics for optimizer '{opt}': {exc}")

    # If requested, fetch SPX prices and add its normalized equity series
    if show_spx:
        try:
            # get_spx_prices_from_date expects a start date (datetime.date)
            d = start_date
            spx_df = get_spx_prices_from_date(d)
            spx_df = spx_df.copy()
            spx_df['date'] = pd.to_datetime(spx_df['date'])

            # Filter to user selected date range
            mask = (spx_df['date'].dt.date >= start_date) & (spx_df['date'].dt.date <= end_date)
            spx_filtered = spx_df[mask]

            if len(spx_filtered) == 0:
                st.warning("No SPX data available for the selected date range")
            else:
                # Compute cumulative returns (equity) from SPX adjusted close prices
                spx_prices = spx_filtered.set_index('date')['close'].sort_index()
                # daily pct change, fill first NaN as 0 (no return), then cumulative product to get equity
                spx_returns = spx_prices.pct_change().fillna(0)
                spx_equity = (1 + spx_returns).cumprod().rename('S&P 500')
                combined_series['S&P 500'] = spx_equity
        except Exception as exc:
            st.warning(f"Could not fetch SPX data: {exc}")

    if not combined_series:
        st.caption("Select one or more models or enable overlays to view equity curves")
        return
    
    combined = pd.concat(combined_series.values(), axis=1)
    # Reset index to have `date` as a column and melt to long form
    combined = combined.reset_index()
    long = combined.melt(id_vars='date', var_name='series', value_name='equity')
    long['date'] = pd.to_datetime(long['date'])

    fig = px.line(
        long,
        x='date',
        y='equity',
        color='series',
        labels={'date': 'Date', 'equity': 'Equity', 'series': 'Series'},
        title='Equity Curves',
    )
    fig.update_traces(mode='lines')
    fig.update_layout(legend_title_text='Series')

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    if metrics_data:
        st.subheader("Performance Metrics")
        st.caption("For Selected Date Range")

        # Convert dict-of-dicts into a nice table
        metrics_df = pd.DataFrame(metrics_data).T

        # Friendly formatting (percent columns)
        percent_cols = ["total_return", "annualized_return", "annualized_volatility", "max_drawdown", "avg_drawdown", "win_rate"]
        for c in percent_cols:
            if c in metrics_df.columns:
                metrics_df[c] = metrics_df[c] * 100

        # reorder (only keep those that exist)
        ordered = [
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "avg_drawdown",
            "win_rate",
            "avg_daily_return",
        ]
        cols = [c for c in ordered if c in metrics_df.columns]
        metrics_df = metrics_df[cols]

        st.dataframe(metrics_df, use_container_width=True)


if __name__ == "__main__":
    app()