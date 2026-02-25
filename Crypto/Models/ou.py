import numpy as np
import pandas as pd


DAYS_PER_YEAR = 365


# =========================================================
# 1. Spread / Series Preparation
# =========================================================

def log_price_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Return log price series.
    """
    s = np.log(df[column].astype(float))
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def rolling_demean(series: pd.Series, window: int) -> pd.Series:
    """
    Remove rolling mean to help stationarity.
    """
    mean = series.rolling(window, min_periods=max(30, window // 3)).mean()
    return series - mean


# =========================================================
# 2. OU MLE Estimation (Discrete AR(1) form)
# =========================================================

def ou_mle(series: pd.Series, dt: float = 1.0) -> dict:
    """
    Estimate OU parameters using AR(1) equivalence.

    X_{t+1} = phi X_t + eps_t

    Continuous mapping:
    theta = (1 - phi) / dt
    mu    = mean / (1 - phi)
    sigma from residual std
    """

    x = series.dropna().values
    if len(x) < 20:
        return {}

    x_t = x[:-1]
    x_t1 = x[1:]

    # AR(1) coefficient
    phi = np.sum(x_t * x_t1) / np.sum(x_t * x_t)

    # Residuals
    residuals = x_t1 - phi * x_t
    sigma_hat = np.std(residuals, ddof=1)

    theta = (1.0 - phi) / dt

    if theta <= 0:
        half_life = np.inf
    else:
        half_life = np.log(2.0) / theta

    return {
        "phi": float(phi),
        "theta": float(theta),
        "sigma": float(sigma_hat),
        "half_life": float(half_life),
    }


# =========================================================
# 3. OU Z-Score Construction
# =========================================================

def ou_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score approximation of OU deviations.
    """
    mean = series.rolling(window, min_periods=max(30, window // 3)).mean()
    std = series.rolling(window, min_periods=max(30, window // 3)).std()

    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


# =========================================================
# 4. OU Trading Signal
# =========================================================

def ou_signal(
    series: pd.Series,
    window: int = 180,
    entry_z: float = 1.5,
    exit_z: float = 0.0,
    long_short: bool = True,
) -> pd.Series:
    """
    Generate OU-based trading signal.

    If long_short:
        Long when z < -entry_z
        Short when z > entry_z
        Exit when |z| < exit_z

    If not long_short:
        Long-only mean reversion.
    """

    z = ou_zscore(series, window)
    pos = pd.Series(0.0, index=series.index)

    current = 0.0

    for i in range(len(z)):
        zi = z.iloc[i]

        if np.isnan(zi):
            pos.iloc[i] = current
            continue

        if long_short:

            if current == 0.0:
                if zi < -entry_z:
                    current = 1.0
                elif zi > entry_z:
                    current = -1.0
            elif current == 1.0:
                if zi > -exit_z:
                    current = 0.0
            elif current == -1.0:
                if zi < exit_z:
                    current = 0.0

        else:
            if current == 0.0:
                if zi < -entry_z:
                    current = 1.0
            elif current == 1.0:
                if zi > -exit_z:
                    current = 0.0

        pos.iloc[i] = current

    return pos.shift(1).fillna(0.0)


# =========================================================
# 5. Strategy Returns
# =========================================================

def ou_returns(
    price_series: pd.Series,
    position: pd.Series,
) -> pd.Series:
    """
    Compute strategy returns from OU signal.
    """

    ret = price_series.pct_change().fillna(0.0)
    position = position.reindex(price_series.index).fillna(0.0)

    strat_ret = position * ret
    return strat_ret


# =========================================================
# 6. Performance Summary (365 annualization)
# =========================================================

def ou_performance_summary(strat_ret: pd.Series) -> dict:
    """
    Compute performance metrics using 365-day annualization.
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

    return {
        "Sharpe": float(sharpe),
        "CAGR": float(cagr),
        "MaxDD": float(max_dd),
        "AnnualReturn": float(ann_return),
        "AnnualVol": float(ann_vol),
        "Observations": int(len(r)),
    }