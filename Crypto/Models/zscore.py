# Models/zscore.py

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


def _trend_strength(price: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    Normalized MA distance. Higher => stronger trend.
    """
    fast_ma = _rolling_mean(price, fast)
    slow_ma = _rolling_mean(price, slow)
    strength = (fast_ma - slow_ma).abs() / slow_ma
    return strength.replace([np.inf, -np.inf], np.nan)


def _trend_direction(price: pd.Series, fast: int, slow: int) -> pd.Series:
    fast_ma = _rolling_mean(price, fast)
    slow_ma = _rolling_mean(price, slow)
    direction = np.sign(fast_ma - slow_ma)  # +1 uptrend, -1 downtrend
    return pd.Series(direction, index=price.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def zscore_signal(
    df: pd.DataFrame,
    price_column: str = "BTC-USD_close",
    # residual mean reversion
    resid_window: int = 180,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
    long_short: bool = True,
    # regime detection
    filter_fast: int = 20,
    filter_slow: int = 128,
    trend_thresh: float = 0.03,
    # risk control
    use_vol_target: bool = True,
    vol_target: float = 0.20,
    vol_window: int = 30,
    max_leverage: float = 1.5,
) -> pd.Series:
    """
    Hybrid model:
      - If market is sideways: mean-revert residual (price - slow MA) using z-score
      - If market is trending: follow the trend (MA direction) instead of mean reversion

    This is the cleanest way to make BTC MR stop bleeding.
    """

    price = df[price_column].astype(float)

    resid_window = int(resid_window)
    filter_fast = int(filter_fast)
    filter_slow = int(filter_slow)

    if filter_fast >= filter_slow:
        raise ValueError(f"filter_fast must be < filter_slow (got {filter_fast} >= {filter_slow})")

    # Residual (de-trended)
    slow_ma = _rolling_mean(price, resid_window)
    resid = price - slow_ma

    # Z-score of residual
    resid_mean = _rolling_mean(resid, resid_window)
    resid_std = _rolling_std(resid, resid_window)
    z = (resid - resid_mean) / resid_std
    z = z.replace([np.inf, -np.inf], np.nan)

    # Regime
    strength = _trend_strength(price, filter_fast, filter_slow)
    is_trending = (strength >= float(trend_thresh)).astype(float)
    is_sideways = 1.0 - is_trending

    # Mean reversion position (stateful enter/exit)
    mr_pos = pd.Series(0.0, index=price.index, dtype=float)
    current = 0.0
    for i in range(len(z)):
        zi = z.iloc[i]
        if np.isnan(zi):
            mr_pos.iloc[i] = current
            continue

        if current == 0.0:
            if zi > float(entry_z):
                current = -1.0
            elif zi < -float(entry_z):
                current = 1.0
        else:
            if abs(zi) < float(exit_z):
                current = 0.0

        mr_pos.iloc[i] = current

    if not long_short:
        mr_pos = mr_pos.clip(lower=0.0)

    # Trend-following fallback (when trending)
    tf_pos = _trend_direction(price, filter_fast, filter_slow)
    if not long_short:
        tf_pos = tf_pos.clip(lower=0.0)

    # Combine regimes
    pos = mr_pos * is_sideways + tf_pos * is_trending

    # Vol targeting (scale total position)
    if use_vol_target:
        ret = price.pct_change().fillna(0.0)
        vol_window = int(vol_window)
        minp = min(vol_window, max(3, vol_window // 2))
        realized_vol = ret.rolling(vol_window, min_periods=minp).std() * np.sqrt(DAYS_PER_YEAR)
        scale = (float(vol_target) / realized_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scale = scale.clip(0.0, float(max_leverage))
        pos = (pos * scale).clip(-float(max_leverage), float(max_leverage))
    else:
        pos = pos.clip(-float(max_leverage), float(max_leverage))

    # Avoid lookahead
    return pos.shift(1).fillna(0.0)