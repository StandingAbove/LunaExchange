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
    fast_ma = _rolling_mean(price, fast)
    slow_ma = _rolling_mean(price, slow)
    strength = (fast_ma - slow_ma).abs() / slow_ma
    return strength.replace([np.inf, -np.inf], np.nan)


def _trend_direction(price: pd.Series, fast: int, slow: int) -> pd.Series:
    fast_ma = _rolling_mean(price, fast)
    slow_ma = _rolling_mean(price, slow)
    direction = np.sign(fast_ma - slow_ma)
    return pd.Series(direction, index=price.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _stateful_zscore_position(
    z: pd.Series,
    entry_z: float,
    exit_z: float,
    long_short: bool,
    max_leverage: float,
    use_smooth_sizing: bool = False,
) -> pd.Series:
    pos = pd.Series(0.0, index=z.index, dtype=float)
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

        if use_smooth_sizing and current != 0.0:
            size = np.tanh(abs(float(zi)) / max(1e-6, float(entry_z)))
            current_signed = np.sign(current) * size
            pos.iloc[i] = current_signed
        else:
            pos.iloc[i] = current

    if not long_short:
        pos = pos.clip(lower=0.0)

    pos = pos.clip(-float(max_leverage), float(max_leverage))
    return pos.shift(1).fillna(0.0)


def zscore_signal(
    data,
    price_column: str = "BTC-USD_close",
    resid_window: int = 180,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
    long_short: bool = True,
    filter_fast: int = 20,
    filter_slow: int = 128,
    trend_thresh: float = 0.03,
    use_vol_target: bool = True,
    vol_target: float = 0.20,
    vol_window: int = 30,
    max_leverage: float = 1.5,
) -> pd.Series:
    """
    Single-asset hybrid regime model (kept for BTC/ETH notebooks).
    """

    if isinstance(data, pd.Series):
        price = data.astype(float).copy()
    elif isinstance(data, pd.DataFrame):
        price = data[price_column].astype(float)
    else:
        raise TypeError("data must be a pandas Series or DataFrame")

    resid_window = int(resid_window)
    filter_fast = int(filter_fast)
    filter_slow = int(filter_slow)

    if filter_fast >= filter_slow:
        raise ValueError(f"filter_fast must be < filter_slow (got {filter_fast} >= {filter_slow})")

    slow_ma = _rolling_mean(price, resid_window)
    resid = price - slow_ma

    resid_mean = _rolling_mean(resid, resid_window)
    resid_std = _rolling_std(resid, resid_window)
    z = (resid - resid_mean) / resid_std
    z = z.replace([np.inf, -np.inf], np.nan)

    strength = _trend_strength(price, filter_fast, filter_slow)
    is_trending = (strength >= float(trend_thresh)).astype(float)
    is_sideways = 1.0 - is_trending

    mr_pos = _stateful_zscore_position(
        z=z,
        entry_z=entry_z,
        exit_z=exit_z,
        long_short=long_short,
        max_leverage=max_leverage,
        use_smooth_sizing=False,
    ).shift(-1).fillna(0.0)

    tf_pos = _trend_direction(price, filter_fast, filter_slow)
    if not long_short:
        tf_pos = tf_pos.clip(lower=0.0)

    pos = mr_pos * is_sideways + tf_pos * is_trending

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

    return pos.shift(1).fillna(0.0)


def zscore_signal_on_spread(
    spread: pd.Series,
    window: int = 90,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
    long_short: bool = True,
    use_vol_target: bool = False,
    max_leverage: float = 1.0,
    min_std: float = 1e-8,
    use_smooth_sizing: bool = True,
) -> pd.Series:
    """
    Pair-trading specific z-score signal for stationary spread.
    Uses pure mean-reversion (no trend-following override), which is usually
    more stable on cointegrated spreads than hybrid trend fallback.
    """
    s = spread.astype(float)
    mean = _rolling_mean(s, window)
    std = _rolling_std(s, window).clip(lower=float(min_std))
    z = ((s - mean) / std).replace([np.inf, -np.inf], np.nan)

    pos = _stateful_zscore_position(
        z=z,
        entry_z=entry_z,
        exit_z=exit_z,
        long_short=long_short,
        max_leverage=max_leverage,
        use_smooth_sizing=use_smooth_sizing,
    )

    if use_vol_target:
        r = s.diff().fillna(0.0)
        rv = r.rolling(max(20, window // 2), min_periods=max(10, window // 4)).std()
        scale = (rv.median() / rv).replace([np.inf, -np.inf], np.nan).clip(0.0, 2.0).fillna(1.0)
        pos = (pos * scale).clip(-float(max_leverage), float(max_leverage))

    return pos


def pair_zscore_signal(*args, **kwargs) -> pd.Series:
    """Alias for readability in pair-trading notebooks."""
    return zscore_signal_on_spread(*args, **kwargs)
