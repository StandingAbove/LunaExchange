import numpy as np
import pandas as pd


DAYS_PER_YEAR = 365


# =========================================================
# 1. Rolling Z-Score
# =========================================================

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling z-score.
    """

    mean = series.rolling(window, min_periods=max(30, window // 3)).mean()
    std = series.rolling(window, min_periods=max(30, window // 3)).std()

    z = (series - mean) / std

    return z.replace([np.inf, -np.inf], np.nan)


# =========================================================
# 2. Z-Score Trading Signal
# =========================================================

def zscore_signal(
    price: pd.Series,
    window: int = 180,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    long_short: bool = True,
):
    price = price.astype(float)

    slow_ma = price.rolling(window).mean()
    residual = price - slow_ma

    vol = residual.rolling(window).std()
    z = residual / vol

    pos = pd.Series(0.0, index=price.index)

    pos[z > entry_z] = -1
    pos[z < -entry_z] = 1

    pos[(z.abs() < exit_z)] = 0

    if not long_short:
        pos = pos.clip(lower=0)

    return pos.shift(1).fillna(0)

# =========================================================
# 3. Optional Volatility Targeting
# =========================================================

def apply_vol_target(
    returns: pd.Series,
    position: pd.Series,
    target_vol: float = 0.15,
    vol_window: int = 30,
) -> pd.Series:
    """
    Scale position to target annual volatility.
    """

    realized_vol = (
        returns.rolling(vol_window)
        .std()
        * np.sqrt(DAYS_PER_YEAR)
    )

    scale = target_vol / realized_vol
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scale = scale.clip(0.0, 3.0)

    scaled_pos = position * scale

    return scaled_pos


# =========================================================
# 4. Strategy Returns
# =========================================================

def zscore_returns(
    price_series: pd.Series,
    position: pd.Series,
) -> pd.Series:
    """
    Compute strategy returns.
    """

    ret = price_series.pct_change().fillna(0.0)
    position = position.reindex(price_series.index).fillna(0.0)

    strat_ret = position * ret

    return strat_ret


# =========================================================
# 5. Performance Metrics (365 annualization)
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
# 6. Rolling Sharpe
# =========================================================

def rolling_sharpe(strat_ret: pd.Series, window: int = 365) -> pd.Series:
    """
    Rolling Sharpe with 365-day annualization.
    """

    def _sharpe(x):
        if x.std() == 0:
            return np.nan
        return np.sqrt(DAYS_PER_YEAR) * x.mean() / x.std()

    return strat_ret.rolling(window).apply(_sharpe, raw=False)


# =========================================================
# 7. Turnover
# =========================================================

def annual_turnover(position: pd.Series) -> float:
    turnover = position.diff().abs().sum()
    annualized = turnover / len(position) * DAYS_PER_YEAR
    return float(annualized)