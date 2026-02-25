import numpy as np
import pandas as pd


DAYS_PER_YEAR = 365


# =========================================================
# 1. Basic Return Utilities
# =========================================================

def equity_curve(returns: pd.Series) -> pd.Series:
    """
    Compute cumulative equity curve.
    """
    returns = returns.fillna(0.0)
    return (1.0 + returns).cumprod()


def annual_return(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return float(r.mean() * DAYS_PER_YEAR)


def annual_volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(DAYS_PER_YEAR))


# =========================================================
# 2. Risk Metrics
# =========================================================

def sharpe_ratio(returns: pd.Series) -> float:
    ann_ret = annual_return(returns)
    ann_vol = annual_volatility(returns)

    if ann_vol == 0 or not np.isfinite(ann_vol):
        return np.nan

    return float(ann_ret / ann_vol)


def sortino_ratio(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return np.nan

    downside = r[r < 0]
    if len(downside) == 0:
        return np.nan

    downside_vol = downside.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)
    ann_ret = r.mean() * DAYS_PER_YEAR

    if downside_vol == 0:
        return np.nan

    return float(ann_ret / downside_vol)


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) == 0:
        return np.nan

    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan

    years = len(equity) / DAYS_PER_YEAR
    if years <= 0:
        return np.nan

    return float(equity.iloc[-1] ** (1 / years) - 1)


def calmar_ratio(equity: pd.Series) -> float:
    dd = abs(max_drawdown(equity))
    growth = cagr(equity)

    if dd == 0 or not np.isfinite(dd):
        return np.nan

    return float(growth / dd)


# =========================================================
# 3. Rolling Metrics
# =========================================================

def rolling_sharpe(returns: pd.Series, window: int = 365) -> pd.Series:
    def _sharpe(x):
        if x.std() == 0:
            return np.nan
        return np.sqrt(DAYS_PER_YEAR) * x.mean() / x.std()

    return returns.rolling(window).apply(_sharpe, raw=False)


def rolling_max_drawdown(equity: pd.Series, window: int = 365) -> pd.Series:
    rolling_peak = equity.rolling(window).max()
    dd = equity / rolling_peak - 1.0
    return dd.rolling(window).min()


# =========================================================
# 4. Turnover
# =========================================================

def annual_turnover(position: pd.Series) -> float:
    if position is None:
        return np.nan

    turnover = position.diff().abs().sum()
    annualized = turnover / len(position) * DAYS_PER_YEAR

    return float(annualized)


# =========================================================
# 5. Full Summary
# =========================================================

def performance_summary(
    returns: pd.Series,
    position: pd.Series = None,
) -> dict:
    """
    Comprehensive performance summary.
    """

    returns = returns.dropna()
    if len(returns) < 5:
        return {}

    eq = equity_curve(returns)

    summary = {
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "CAGR": cagr(eq),
        "MaxDD": max_drawdown(eq),
        "Calmar": calmar_ratio(eq),
        "AnnualReturn": annual_return(returns),
        "AnnualVol": annual_volatility(returns),
        "Observations": int(len(returns)),
    }

    if position is not None:
        summary["AnnualTurnover"] = annual_turnover(position)

    return summary


# =========================================================
# 6. Multi-Strategy Table Builder
# =========================================================

def build_summary_table(results_dict: dict) -> pd.DataFrame:
    """
    results_dict format:
    {
        "StrategyName": {
            "returns": pd.Series,
            "position": pd.Series (optional)
        }
    }
    """

    rows = []

    for name, data in results_dict.items():
        returns = data.get("returns")
        position = data.get("position")

        summary = performance_summary(returns, position)
        summary["Strategy"] = name

        rows.append(summary)

    df = pd.DataFrame(rows)
    df = df.set_index("Strategy")

    return df