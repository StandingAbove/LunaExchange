# models/ou.py
# Replace OU single-asset logic with a *pair/spread OU module*.
# This aligns OU with stationarity: use OU for spread-like series, not raw BTC price.

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365


def ou_mle(series: pd.Series, dt: float = 1.0) -> dict:
    """
    OU MLE via AR(1) mapping on a stationary series.

    X_{t+1} = a + b X_t + eps
    b = exp(-theta*dt)
    """

    x = series.dropna().astype(float).values
    if len(x) < 50:
        return {}

    x_t = x[:-1]
    x_t1 = x[1:]

    # OLS: x_{t+1} = a + b x_t
    b = np.cov(x_t, x_t1, ddof=1)[0, 1] / np.var(x_t, ddof=1)
    a = np.mean(x_t1) - b * np.mean(x_t)

    # Clamp b to avoid nonsense
    b = float(np.clip(b, 1e-6, 0.999999))

    theta = -np.log(b) / dt
    mu = a / (1.0 - b)

    resid = x_t1 - (a + b * x_t)
    sigma_eps = np.std(resid, ddof=1)

    # Continuous-time sigma
    sigma = sigma_eps * np.sqrt(2.0 * theta / (1.0 - b**2))

    half_life = np.log(2.0) / theta if theta > 0 else np.inf

    return {
        "a": float(a),
        "b": float(b),
        "theta": float(theta),
        "mu": float(mu),
        "sigma": float(sigma),
        "half_life": float(half_life),
    }


def ou_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score for spread deviations.
    """
    window = int(window)
    if window <= 2:
        raise ValueError(f"window must be > 2, got {window}")

    minp = min(window, max(10, window // 3))
    mean = series.rolling(window, min_periods=minp).mean()
    std = series.rolling(window, min_periods=minp).std()

    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def ou_signal_on_spread(
    spread: pd.Series,
    window: int = 180,
    entry_z: float = 2.0,
    exit_z: float = 0.2,
) -> pd.Series:
    """
    OU-style mean reversion signal for a stationary spread.

    Long spread when z < -entry_z
    Short spread when z > entry_z
    Exit when |z| < exit_z
    """

    z = ou_zscore(spread.astype(float), window)
    pos = pd.Series(0.0, index=spread.index, dtype=float)

    current = 0.0
    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = current
            continue

        if current == 0.0:
            if zi > entry_z:
                current = -1.0
            elif zi < -entry_z:
                current = 1.0
        else:
            if abs(zi) < exit_z:
                current = 0.0

        pos.iloc[i] = current

    return pos.shift(1).fillna(0.0)