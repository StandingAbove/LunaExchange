import numpy as np
import pandas as pd


DAYS_PER_YEAR = 365


# =========================================================
# 1. Turnover
# =========================================================

def compute_turnover(position: pd.Series) -> pd.Series:
    """
    Daily turnover = absolute change in position.
    """
    return position.diff().abs().fillna(0.0)


def annual_turnover(position: pd.Series) -> float:
    """
    Annualized turnover.
    """
    turnover = compute_turnover(position).sum()
    return float(turnover / len(position) * DAYS_PER_YEAR)


# =========================================================
# 2. Transaction Cost Model
# =========================================================

def transaction_cost_from_turnover(
    position: pd.Series,
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
) -> pd.Series:
    """
    Cost per day = turnover * (fee + slippage).

    fee_bps = exchange fee in basis points
    slippage_bps = assumed slippage per trade
    """

    turnover = compute_turnover(position)

    total_bps = fee_bps + slippage_bps
    cost_rate = total_bps / 10000.0

    cost = turnover * cost_rate

    return cost


# =========================================================
# 3. Funding Cost (Perpetual Futures)
# =========================================================

def funding_cost(
    position: pd.Series,
    funding_rate_series: pd.Series,
) -> pd.Series:
    """
    Apply funding cost.

    funding_rate_series should be daily rate (not annualized).
    """

    funding_rate_series = funding_rate_series.reindex(position.index).fillna(0.0)

    cost = position * funding_rate_series

    return cost


# =========================================================
# 4. Borrow Cost (Short Spot)
# =========================================================

def borrow_cost(
    position: pd.Series,
    annual_borrow_rate: float = 0.05,
) -> pd.Series:
    """
    Apply borrow cost to short positions only.
    """

    daily_rate = annual_borrow_rate / DAYS_PER_YEAR

    short_exposure = position.clip(upper=0.0).abs()

    cost = short_exposure * daily_rate

    return cost


# =========================================================
# 5. Apply Costs to Returns
# =========================================================

def apply_costs(
    returns: pd.Series,
    position: pd.Series,
    fee_bps: float = 10.0,
    slippage_bps: float = 0.0,
    funding_rate_series: pd.Series = None,
    annual_borrow_rate: float = 0.0,
) -> pd.Series:
    """
    Apply all transaction costs to strategy returns.

    Returns net returns.
    """

    position = position.reindex(returns.index).fillna(0.0)

    # Transaction cost
    tc = transaction_cost_from_turnover(
        position,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )

    total_cost = tc

    # Funding cost
    if funding_rate_series is not None:
        total_cost += funding_cost(position, funding_rate_series)

    # Borrow cost
    if annual_borrow_rate > 0:
        total_cost += borrow_cost(position, annual_borrow_rate)

    net_returns = returns - total_cost

    return net_returns