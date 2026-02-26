import streamlit as st

def main():
    st.set_page_config(
        page_title="Capital Fund Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("Capital Fund")
    st.sidebar.caption("Dashboard")

    page = st.sidebar.radio(
        "Navigation",
        ("Paper Portfolio Performance", "Marginal Model Backtests"),
        label_visibility="collapsed"
    )
    
    if page == "Paper Portfolio Performance":
        from src.dashboard.screens import paper_performance
        paper_performance.app()
    elif page == "Marginal Model Backtests":
        from src.dashboard.screens import backtest_charts
        backtest_charts.app()


if __name__ == "__main__":
    main()