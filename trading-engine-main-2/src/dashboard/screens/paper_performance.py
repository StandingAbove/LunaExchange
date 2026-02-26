import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

from src.dashboard.utils import (
    get_latest_production_audit,
    get_historical_nav,
    get_spx_prices_from_date,
    get_portfolio_backtest,
    get_active_optimizer
)

TRADING_DAYS = 252

def _equity_from_returns(daily_returns: pd.Series) -> pd.Series:
    """Convert daily returns to an equity curve normalized to 1."""
    daily_returns = daily_returns.fillna(0).copy()
    if len(daily_returns) > 0:
        daily_returns.iloc[0] = 0
    return (1 + daily_returns).cumprod()

def _compute_metrics(equity: pd.Series) -> dict:
    """
    Basic performance metrics from an equity curve (starting at 1).
    - total_return
    - annualized_return
    - annualized_volatility (from daily returns)
    - max_drawdown
    """
    metrics = {
        "total_return": np.nan,
        "annualized_return": np.nan,
        "annualized_volatility": np.nan,
        "max_drawdown": np.nan,
    }
    
    equity = equity.dropna()

    # Total return
    metrics["total_return"] = equity.iloc[-1] - 1

    # Daily returns implied by equity
    daily_returns = equity.pct_change().dropna()

    # Annualized vol
    if len(daily_returns) > 1:
        metrics["annualized_volatility"] = daily_returns.std() * np.sqrt(TRADING_DAYS)

    # Annualized return using time length
    years = len(equity) / TRADING_DAYS
    if years > 0:
        metrics["annualized_return"] = (equity.iloc[-1]) ** (1 / years) - 1

    # Max drawdown
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1
    metrics["max_drawdown"] = drawdown.min()

    return metrics

def app():
    # Header
    latest_audit = get_latest_production_audit()

    left, right = st.columns([3, 1])
    with left:
        st.title("Paper Portfolio Performance")
        st.caption("Equity curves for paper portfolio NAV, simulated portfolio, and SPX benchmark")
    with right:
        st.metric("Latest Production Audit", str(latest_audit))

    st.divider()

    # Load historical NAV
    nav_df = get_historical_nav()
    if nav_df is None or nav_df.empty:
        st.info("No historical NAV data available.")
        return

    if "date" not in nav_df.columns or "nav" not in nav_df.columns:
        st.warning("historical_nav.csv missing expected columns 'date' and 'nav'.")
        return

    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.sort_values("date")

    default_start = nav_df["date"].iloc[0].date()
    default_end = nav_df["date"].iloc[-1].date()

    # Controls
    with st.expander("Controls", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                min_value=default_start,
                max_value=default_end,
                key="pp_start_date",
            )

        with col2:
            end_date = st.date_input(
                "End Date",
                value=default_end,
                min_value=default_start,
                max_value=default_end,
                key="pp_end_date",
            )

        with col3:
            show_spx = st.checkbox("Show SPX", value=True)
            show_sim = st.checkbox("Show Simulated", value=True)

        show_drawdown = st.checkbox("Show Drawdown Chart", value=False)
    
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        return

    # Filter dataframe to selection
    mask = (nav_df['date'].dt.date >= start_date) & (nav_df['date'].dt.date <= end_date)
    plot_df = nav_df.loc[mask]

    if plot_df.empty:
        st.info("No NAV data in the selected date range.")
        return

    # Combined series that get plotted
    combined_series = {}

    # Build equity curve: normalize NAV so the first value in the selection equals 1
    plot_df = plot_df.copy()
    plot_df['daily_returns'] = plot_df['nav'].pct_change().fillna(0)
    plot_df['equity'] = (1 + plot_df['daily_returns']).cumprod()
    combined_series['Actual Portfolio'] = plot_df.set_index('date')['equity']

    sim_df = None
    if show_sim:
        # TODO: This is what the paper pipeline currently does (assumes single optimizer). Change this 
        # if that assumption is no longer valid.
        opt = get_active_optimizer()
        sim_df = get_portfolio_backtest(opt)

        sim_df['date'] = pd.to_datetime(sim_df['date'])
        sim_df = sim_df.sort_values('date')

        mask = (sim_df['date'].dt.date >= start_date) & (sim_df['date'].dt.date <= end_date)
        sim_df = sim_df[mask]

        if len(sim_df) == 0:
            st.warning(f"No data for optimizer '{opt}' in selected date range")

        sim_df['daily_return'].iloc[0] = 0 # set initial day to zero for proper cumulative calculation
        sim_df['equity'] = (1 + sim_df['daily_return']).cumprod()

    # Optionally load SPX prices and normalize to start at 1 over the same date range
    spx_df = None
    if show_spx:
        try:
            spx_df = get_spx_prices_from_date(default_start)
            # Ensure datetime type
            spx_df['date'] = pd.to_datetime(spx_df['date'])
            # Filter to selected range
            spx_mask = (spx_df['date'].dt.date >= start_date) & (spx_df['date'].dt.date <= end_date)
            spx_df = spx_df.loc[spx_mask].sort_values('date')
            if not spx_df.empty:
                spx_df['daily_returns'] = spx_df['close'].pct_change().fillna(0)
                spx_df['equity'] = (1 + spx_df['daily_returns']).cumprod()
            else:
                spx_df = None
        except Exception as exc:
            st.warning(f"Could not load SPX prices: {exc}")
            spx_df = None

    if show_sim and sim_df is not None:
        combined_series['Simulated Portfolio'] = sim_df.set_index('date')['equity']
    if spx_df is not None and not spx_df.empty:
        combined_series['S&P 500'] = spx_df.set_index('date')['equity']

    # Concatenate series and melt to long form for Plotly
    combined = pd.concat(combined_series.values(), axis=1)
    combined.columns = list(combined_series.keys())
    combined = combined.reset_index()
    long = combined.melt(id_vars='date', var_name='series', value_name='equity')

    series_colors = ["#D39338", "#77C0B0", "#EDECEC"]

    # Equity Curves Chart
    fig = px.line(
        long,
        x='date',
        y='equity',
        color='series',
        color_discrete_sequence=series_colors,
        labels={'date': 'Date', 'equity': 'Cumulative Returns', 'series': 'Series'},
        title='Equity Curves',
    )
    
    fig.update_traces(mode='lines')
    st.plotly_chart(fig, use_container_width=True)

    # Drawdown Chart
    if show_drawdown:
        dd_df = combined.copy()
        # Compute drawdown for each series
        dd_long_parts = []
        for col in combined_series.keys():
            s = dd_df.set_index("date")[col].dropna()
            running_max = s.cummax()
            drawdown = (s / running_max) - 1
            dd_long_parts.append(
                pd.DataFrame({"date": drawdown.index, "series": col, "drawdown": drawdown.values})
            )
        dd_long = pd.concat(dd_long_parts, axis=0)

        dd_fig = px.area(
            dd_long,
            x="date",
            y="drawdown",
            color="series",
            color_discrete_sequence=series_colors,
            labels={"date": "Date", "drawdown": "Drawdown", "series": "Series"},
            title="Drawdown",
        )
        dd_fig.update_traces(mode="lines")
        st.plotly_chart(dd_fig, use_container_width=True)

    # Portfolio Stats
    metrics = _compute_metrics(plot_df["equity"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Return", f"{metrics['total_return']*100:.2f}%")
    m2.metric("Annualized Return", f"{metrics['annualized_return']*100:.2f}%")
    m3.metric("Annualized Volatility", f"{metrics['annualized_volatility']*100:.2f}%")
    m4.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")

if __name__ == "__main__":
    app()
