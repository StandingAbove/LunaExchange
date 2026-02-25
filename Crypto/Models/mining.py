import numpy as np
import pandas as pd


DAYS_PER_YEAR = 365


def compute_mining_edge(df: pd.DataFrame) -> pd.Series:
    """
    Compute mining edge as log(price / cost_to_mine).

    Positive -> price above production cost
    Negative -> price below production cost
    """

    price = df["BTC-USD_close"].astype(float)
    cost = df["COST_TO_MINE"].astype(float)

    edge = np.log(price / cost)

    return edge.replace([np.inf, -np.inf], np.nan)


def compute_mining_margin(df: pd.DataFrame) -> pd.Series:
    """
    Mining margin = (price - cost) / cost
    """

    price = df["BTC-USD_close"].astype(float)
    cost = df["COST_TO_MINE"].astype(float)

    margin = (price - cost) / cost

    return margin.replace([np.inf, -np.inf], np.nan)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score using simple rolling mean/std.
    """

    mean = series.rolling(window, min_periods=max(30, window // 3)).mean()
    std = series.rolling(window, min_periods=max(30, window // 3)).std()

    z = (series - mean) / std

    return z.replace([np.inf, -np.inf], np.nan)


def mining_signal(
    df: pd.DataFrame,
    z_window: int = 180,
    entry_z: float = 1.0,
    exit_z: float = 0.0,
    use_log_edge: bool = True,
) -> pd.Series:
    """
    Generate mining-based regime signal.

    Logic:
    - Compute edge (log or margin)
    - Standardize with rolling z-score
    - Long when z > entry_z
    - Exit when z < exit_z
    - Long-only model
    - Shift 1 day to avoid lookahead
    """

    if use_log_edge:
        edge = compute_mining_edge(df)
    else:
        edge = compute_mining_margin(df)

    z = rolling_zscore(edge, z_window)

    pos = pd.Series(0.0, index=df.index)
    in_position = 0.0

    for i in range(len(z)):
        zi = z.iloc[i]

        if np.isnan(zi):
            pos.iloc[i] = in_position
            continue

        if in_position == 0.0:
            if zi > entry_z:
                in_position = 1.0
        else:
            if zi < exit_z:
                in_position = 0.0

        pos.iloc[i] = in_position

    return pos.shift(1).fillna(0.0)


def mining_returns(
    df: pd.DataFrame,
    position: pd.Series,
) -> pd.Series:
    """
    Compute daily strategy returns using mining position.
    Annualization uses 365 days.
    """

    price = df["BTC-USD_close"].astype(float)
    ret = price.pct_change().fillna(0.0)

    position = position.reindex(df.index).fillna(0.0)

    strat_ret = position * ret

    return strat_ret


def mining_performance_summary(
    strat_ret: pd.Series,
) -> dict:
    """
    Return summary metrics using 365-day annualization.
    """

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

    turnover = position_turnover(strat_ret)

    return {
        "Sharpe": float(sharpe),
        "CAGR": float(cagr),
        "MaxDD": float(max_dd),
        "AnnualReturn": float(ann_return),
        "AnnualVol": float(ann_vol),
        "Observations": int(len(r)),
        "AnnualTurnover": float(turnover),
    }


def position_turnover(strat_ret: pd.Series) -> float:
    """
    Approximate annual turnover from returns series.
    Assumes binary position (0 or 1).
    """

    # This is a proxy since we don't pass position directly.
    # Better to compute turnover from position series externally.
    return np.nan