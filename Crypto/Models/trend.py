import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365


import pandas as pd

def moving_average(series: pd.Series, window: int) -> pd.Series:
    window = int(window)
    if window <= 1:
        raise ValueError(f"window must be > 1, got {window}")

    minp = max(3, window // 3)   # small floor
    minp = min(minp, window)     # critical: min_periods cannot exceed window

    return series.rolling(window=window, min_periods=minp).mean()

def trend_signal(
    df: pd.DataFrame,
    price_column: str = "BTC-USD_close",
    fast_window: int = 20,
    slow_window: int = 128,
    long_only: bool = True,
    leverage_aggressive: float = 1.3,
    leverage_neutral: float = 1.0,
    leverage_defensive: float = 0.7,
) -> pd.Series:
    fast_window = int(fast_window)
    slow_window = int(slow_window)

    if fast_window >= slow_window:
        raise ValueError(f"fast_window must be < slow_window (got {fast_window} >= {slow_window})")

    price = df[price_column].astype(float)

    ma_fast = moving_average(price, fast_window)
    ma_slow = moving_average(price, slow_window)

    pos = pd.Series(leverage_neutral, index=price.index, dtype=float)

    pos = np.where(ma_fast > ma_slow, leverage_aggressive, pos)
    pos = np.where(ma_fast < ma_slow, leverage_defensive, pos)

    pos = pd.Series(pos, index=price.index)

    if long_only:
        pos = pos.clip(lower=0.0)

    return pos.shift(1).fillna(leverage_neutral)
# =========================================================
# 2. Strategy Returns
# =========================================================

def trend_returns(
    df: pd.DataFrame,
    position: pd.Series,
    price_column: str = "BTC-USD_close",
) -> pd.Series:
    """
    Compute strategy returns.
    """

    price = df[price_column].astype(float)
    ret = price.pct_change().fillna(0.0)

    position = position.reindex(price.index).fillna(0.0)

    strat_ret = position * ret

    return strat_ret


# =========================================================
# 3. Performance Metrics (365 annualization)
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

    return {
        "Sharpe": float(sharpe),
        "CAGR": float(cagr),
        "MaxDD": float(max_dd),
        "AnnualReturn": float(ann_return),
        "AnnualVol": float(ann_vol),
        "Observations": int(len(r)),
    }


# =========================================================
# 4. Rolling Metrics
# =========================================================

def rolling_sharpe(strat_ret: pd.Series, window: int = 365) -> pd.Series:
    """
    Rolling Sharpe using 365-day annualization.
    """

    def _sharpe(x):
        if x.std() == 0:
            return np.nan
        return np.sqrt(DAYS_PER_YEAR) * x.mean() / x.std()

    return strat_ret.rolling(window).apply(_sharpe, raw=False)


# =========================================================
# 5. Turnover
# =========================================================

def annual_turnover(position: pd.Series) -> float:
    """
    Annual turnover based on absolute position changes.
    """

    turnover = position.diff().abs().sum()
    annualized = turnover / len(position) * DAYS_PER_YEAR

    return float(annualized)