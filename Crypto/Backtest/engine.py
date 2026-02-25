import numpy as np
import pandas as pd
from Backtest.costs import apply_costs

DAYS_PER_YEAR = 365


# =========================================================
# 1. Core Return Computation
# =========================================================

def compute_strategy_returns(
    price_series: pd.Series,
    position: pd.Series,
) -> pd.Series:
    """
    Compute raw (gross) strategy returns.
    """

    price_series = price_series.astype(float)
    ret = price_series.pct_change().fillna(0.0)

    position = position.reindex(price_series.index).fillna(0.0)

    strat_ret = position * ret

    return strat_ret


# =========================================================
# 2. Backtest Execution
# =========================================================

def run_backtest(
    price_series: pd.Series,
    position: pd.Series,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    funding_rate_series: pd.Series = None,
    annual_borrow_rate: float = 0.0,
    leverage_cap: float = None,
) -> dict:
    """
    Run full backtest.

    Returns dictionary containing:
        gross_returns
        net_returns
        gross_equity
        net_equity
        metrics_gross
        metrics_net
    """

    position = position.reindex(price_series.index).fillna(0.0)

    # Optional leverage cap
    if leverage_cap is not None:
        position = position.clip(-leverage_cap, leverage_cap)

    # --- Gross returns ---
    gross_returns = compute_strategy_returns(price_series, position)

    # --- Net returns (apply costs) ---
    if fee_bps > 0 or slippage_bps > 0 or funding_rate_series is not None or annual_borrow_rate > 0:
        net_returns = apply_costs(
            gross_returns,
            position,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            funding_rate_series=funding_rate_series,
            annual_borrow_rate=annual_borrow_rate,
        )
    else:
        net_returns = gross_returns.copy()

    # --- Equity curves ---
    gross_equity = (1.0 + gross_returns).cumprod()
    net_equity = (1.0 + net_returns).cumprod()

    # --- Metrics ---
    metrics_gross = performance_summary(gross_returns)
    metrics_net = performance_summary(net_returns)

    return {
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "gross_equity": gross_equity,
        "net_equity": net_equity,
        "metrics_gross": metrics_gross,
        "metrics_net": metrics_net,
        "position": position,
    }


# =========================================================
# 3. Performance Metrics (365 Annualization)
# =========================================================

def performance_summary(strat_ret: pd.Series) -> dict:
    r = strat_ret.dropna()
    if len(r) < 5:
        return {}

    ann_return = r.mean() * DAYS_PER_YEAR
    ann_vol = r.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    equity = (1.0 + r).cumprod()

    years = len(r) / DAYS_PER_YEAR
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = dd.min()

    turnover = annual_turnover_from_position(strat_ret)

    return {
        "Sharpe": float(sharpe),
        "CAGR": float(cagr),
        "MaxDD": float(max_dd),
        "AnnualReturn": float(ann_return),
        "AnnualVol": float(ann_vol),
        "Observations": int(len(r)),
        "AnnualTurnoverProxy": float(turnover),
    }


# =========================================================
# 4. Rolling Sharpe
# =========================================================

def rolling_sharpe(strat_ret: pd.Series, window: int = 365) -> pd.Series:
    def _sharpe(x):
        if x.std() == 0:
            return np.nan
        return np.sqrt(DAYS_PER_YEAR) * x.mean() / x.std()

    return strat_ret.rolling(window).apply(_sharpe, raw=False)


# =========================================================
# 5. Turnover Proxy
# =========================================================

def annual_turnover_from_position(strat_ret: pd.Series) -> float:
    """
    Proxy turnover if only returns are available.
    Proper turnover should be computed from position series externally.
    """
    return np.nan