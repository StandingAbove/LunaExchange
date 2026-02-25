import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


DAYS_PER_YEAR = 365


# =========================================================
# 1. Build BTC–ETH Spread
# =========================================================

def rolling_beta(y: pd.Series, x: pd.Series, window: int = 180) -> pd.Series:
    """
    Rolling OLS hedge ratio:
    y_t = beta * x_t

    beta = cov(y,x) / var(x)
    """

    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()

    beta = cov / var
    return beta


def build_spread(
    df: pd.DataFrame,
    window: int = 180,
) -> pd.Series:
    """
    Construct log spread:

    spread = log(BTC) - beta * log(ETH)
    """

    log_btc = np.log(df["BTC-USD_close"].astype(float))
    log_eth = np.log(df["ETH-USD_close"].astype(float))

    beta = rolling_beta(log_btc, log_eth, window)

    spread = log_btc - beta * log_eth
    spread = spread.replace([np.inf, -np.inf], np.nan)

    return spread.dropna()


# =========================================================
# 2. Stationarity Test
# =========================================================

def adf_test(series: pd.Series) -> dict:
    """
    Augmented Dickey-Fuller test.
    """

    s = series.dropna()
    if len(s) < 30:
        return {}

    result = adfuller(s)

    return {
        "ADF_stat": float(result[0]),
        "p_value": float(result[1]),
        "n_lags": int(result[2]),
        "n_obs": int(result[3]),
    }


# =========================================================
# 3. OU MLE Estimation on Spread
# =========================================================

def ou_mle(series: pd.Series, dt: float = 1.0) -> dict:
    """
    Estimate OU parameters via AR(1) mapping.
    """

    x = series.dropna().values
    if len(x) < 20:
        return {}

    x_t = x[:-1]
    x_t1 = x[1:]

    phi = np.sum(x_t * x_t1) / np.sum(x_t * x_t)

    residuals = x_t1 - phi * x_t
    sigma = np.std(residuals, ddof=1)

    theta = 1.0 - phi

    if theta <= 0:
        half_life = np.inf
    else:
        half_life = np.log(2.0) / theta

    return {
        "phi": float(phi),
        "theta": float(theta),
        "sigma": float(sigma),
        "half_life": float(half_life),
    }


# =========================================================
# 4. Z-Score Signal (Preferred Robust Method)
# =========================================================

def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def pair_signal_zscore(
    spread: pd.Series,
    window: int = 180,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
) -> pd.Series:
    """
    Long spread when z < -entry_z
    Short spread when z > entry_z
    Exit when |z| < exit_z
    """

    z = rolling_zscore(spread, window)
    pos = pd.Series(0.0, index=spread.index)

    current = 0.0

    for i in range(len(z)):
        zi = z.iloc[i]

        if np.isnan(zi):
            pos.iloc[i] = current
            continue

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

        pos.iloc[i] = current

    return pos.shift(1).fillna(0.0)


# =========================================================
# 5. Pair Returns (Dollar Neutral)
# =========================================================

def pair_returns(
    df: pd.DataFrame,
    position: pd.Series,
    window: int = 180,
) -> pd.Series:
    """
    Compute dollar-neutral pair returns:

    Long spread:
        +BTC
        -beta * ETH

    Short spread:
        -BTC
        +beta * ETH
    """

    log_btc = np.log(df["BTC-USD_close"].astype(float))
    log_eth = np.log(df["ETH-USD_close"].astype(float))

    beta = rolling_beta(log_btc, log_eth, window)

    ret_btc = df["BTC-USD_close"].pct_change().fillna(0.0)
    ret_eth = df["ETH-USD_close"].pct_change().fillna(0.0)

    beta = beta.reindex(df.index).fillna(method="ffill")

    position = position.reindex(df.index).fillna(0.0)

    strat_ret = position * (ret_btc - beta * ret_eth)

    return strat_ret


# =========================================================
# 6. Performance Summary (365 annualization)
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