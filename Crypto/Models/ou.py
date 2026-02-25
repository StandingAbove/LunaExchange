# Models/ou.py

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365


def _rolling_mean(x: pd.Series, window: int) -> pd.Series:
    window = int(window)
    minp = min(window, max(3, window // 3))
    return x.rolling(window, min_periods=minp).mean()


def _rolling_std(x: pd.Series, window: int) -> pd.Series:
    window = int(window)
    minp = min(window, max(3, window // 3))
    return x.rolling(window, min_periods=minp).std()


def ou_signal(
    price_series: pd.Series,
    window: int = 180,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
    long_short: bool = True,
    detrend_window: int = 180,
) -> pd.Series:
    price = price_series.astype(float)
    logp = np.log(price.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    detrend_window = int(detrend_window)
    x = logp - _rolling_mean(logp, detrend_window)

    mean = _rolling_mean(x, window)
    std = _rolling_std(x, window)
    z = (x - mean) / std
    z = z.replace([np.inf, -np.inf], np.nan)

    pos = pd.Series(0.0, index=price.index, dtype=float)
    current = 0.0

    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = current
            continue

        if current == 0.0:
            if zi > float(entry_z):
                current = -1.0
            elif zi < -float(entry_z):
                current = 1.0
        else:
            if abs(zi) < float(exit_z):
                current = 0.0

        pos.iloc[i] = current

    if not long_short:
        pos = pos.clip(lower=0.0)

    return pos.shift(1).fillna(0.0)


def build_spread(
    df: pd.DataFrame,
    btc_col: str = "BTC-USD_close",
    eth_col: str = "ETH-USD_close",
    beta_window: int = 180,
) -> pd.Series:
    log_b = np.log(df[btc_col].astype(float))
    log_e = np.log(df[eth_col].astype(float))

    beta_window = int(beta_window)
    minp = min(beta_window, max(30, beta_window // 3))

    cov = log_b.rolling(beta_window, min_periods=minp).cov(log_e)
    var = log_e.rolling(beta_window, min_periods=minp).var()

    beta = (cov / var).replace([np.inf, -np.inf], np.nan)
    spread = log_b - beta * log_e

    return spread.dropna()


def ou_signal_on_spread(
    spread: pd.Series,
    window: int = 180,
    entry_z: float = 2.0,
    exit_z: float = 0.2,
    long_short: bool = True,
    max_leverage: float = 1.0,
    min_std: float = 1e-8,
    z_cap: float = 4.0,
    smooth_sizing: bool = True,
) -> pd.Series:
    """
    Pair-trading OU-style signal on spread.
    Dedicated pair behavior: pure mean reversion with optional smooth sizing.
    """
    s = spread.astype(float)
    mean = _rolling_mean(s, window)
    std = _rolling_std(s, window).clip(lower=float(min_std))
    z = ((s - mean) / std).replace([np.inf, -np.inf], np.nan).clip(-float(z_cap), float(z_cap))

    pos = pd.Series(0.0, index=s.index, dtype=float)
    current = 0.0

    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            pos.iloc[i] = current
            continue

        if current == 0.0:
            if zi > float(entry_z):
                current = -1.0
            elif zi < -float(entry_z):
                current = 1.0
        else:
            if abs(zi) < float(exit_z):
                current = 0.0

        if smooth_sizing and current != 0.0:
            mag = np.tanh(abs(float(zi)) / max(1e-6, float(entry_z)))
            pos.iloc[i] = np.sign(current) * mag
        else:
            pos.iloc[i] = current

    if not long_short:
        pos = pos.clip(lower=0.0)

    pos = pos.clip(-float(max_leverage), float(max_leverage))
    return pos.shift(1).fillna(0.0)


def pair_ensemble_signal(
    spread: pd.Series,
    z_pos: pd.Series,
    trend_pos: pd.Series,
    mining_pos: pd.Series,
    w_ou: float = 0.45,
    w_z: float = 0.45,
    w_trend: float = 0.05,
    w_mining: float = 0.05,
    ou_window: int = 90,
    ou_entry_z: float = 1.5,
    ou_exit_z: float = 0.3,
    leverage_cap: float = 1.0,
) -> pd.Series:
    ou_pos = ou_signal_on_spread(
        spread,
        window=ou_window,
        entry_z=ou_entry_z,
        exit_z=ou_exit_z,
        long_short=True,
        max_leverage=1.0,
    )

    idx = spread.index
    z = z_pos.reindex(idx).fillna(0.0).clip(-1.0, 1.0)
    t = trend_pos.reindex(idx).fillna(0.0).clip(-1.0, 1.0)
    m = mining_pos.reindex(idx).fillna(0.0).clip(-1.0, 1.0)
    o = ou_pos.reindex(idx).fillna(0.0).clip(-1.0, 1.0)

    combo = (w_ou * o) + (w_z * z) + (w_trend * t) + (w_mining * m)
    combo = combo.clip(-float(leverage_cap), float(leverage_cap))
    return combo
