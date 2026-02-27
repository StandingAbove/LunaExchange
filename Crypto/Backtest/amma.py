from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import pandas as pd


def load_ibit_csv(path: str) -> pd.DataFrame:
    """
    Load IBIT CSV and normalize to ['date', 'close'] (+ optional OHLCV if present).

    - Detects date column from common names.
    - Detects close/price column from common names.
    - Parses date, sorts ascending, and drops invalid rows.
    """
    df = pd.read_csv(path).copy()

    date_candidates = ["Date", "date", "Timestamp", "timestamp", "datetime"]
    close_candidates = ["Close", "close", "Adj Close", "adj_close", "Price", "price"]

    date_col = next((c for c in date_candidates if c in df.columns), None)
    close_col = next((c for c in close_candidates if c in df.columns), None)

    if date_col is None:
        raise ValueError(f"Could not find date column in {path}. Tried: {date_candidates}")
    if close_col is None:
        raise ValueError(f"Could not find close/price column in {path}. Tried: {close_candidates}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], format="%m/%d/%y", errors="coerce")

    close_raw = df[close_col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    out["close"] = pd.to_numeric(close_raw, errors="coerce")

    optional_map = {
        "open": ["Open", "open"],
        "high": ["High", "high"],
        "low": ["Low", "low"],
        "volume": ["Volume", "volume", "CVol"],
    }
    for norm_col, candidates in optional_map.items():
        source_col = next((c for c in candidates if c in df.columns), None)
        if source_col is not None:
            out[norm_col] = pd.to_numeric(
                df[source_col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    out = out.dropna(subset=["date", "close"]).drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").reset_index(drop=True)
    return out


def AMMA(
        ticker: str,
        momentum_weights: Dict[int, float],
        threshold: float = 0.0,
        long_enabled: bool = True,
        short_enabled: bool = False,
) -> Callable[[Any], Any]:
    """Polars-backed AMMA model for the existing model-state pipeline."""

    def run_model(bundle: Any) -> Any:
        import polars as pl

        lf = bundle.model_state.lazy()
        sig_frames = []

        for window, weight in momentum_weights.items():
            colname = f"close_momentum_{window}"
            sig = lf.filter(pl.col("ticker") == ticker).select([pl.col("date"), pl.col(colname).alias("sig")])

            long_cond = (pl.col("sig") > threshold) if long_enabled else None
            short_cond = (pl.col("sig") < -threshold) if short_enabled else None

            expr = pl.lit(0.0)
            if long_enabled and short_enabled:
                expr = (
                    pl.when(long_cond.fill_null(False)).then(pl.lit(1.0) * weight)
                    .when(short_cond.fill_null(False)).then(pl.lit(-1.0) * weight)
                    .otherwise(pl.lit(0.0))
                )
            elif long_enabled:
                expr = pl.when(long_cond.fill_null(False)).then(pl.lit(1.0) * weight).otherwise(pl.lit(0.0))
            elif short_enabled:
                expr = pl.when(short_cond.fill_null(False)).then(pl.lit(-1.0) * weight).otherwise(pl.lit(0.0))

            sig_frames.append(sig.select([pl.col("date"), expr.cast(pl.Float64).alias(f"sig_{window}")]))

        combined = sig_frames[0]
        for frame in sig_frames[1:]:
            combined = combined.join(frame, on="date", how="inner")

        weight_cols = [f"sig_{w}" for w in momentum_weights.keys()]
        return combined.with_columns(sum([pl.col(c) for c in weight_cols]).alias(ticker)).select(["date", ticker])

    return run_model


def amma_from_ibit_csv(
        ibit_csv_path: str,
        momentum_weights: Dict[int, float],
        threshold: float = 0.0,
        long_enabled: bool = True,
        short_enabled: bool = False,
) -> pd.DataFrame:
    """
    Run AMMA directly on IBIT CSV and return backtest-ready columns.

    Returns columns:
    - date
    - close
    - ret
    - signal (position)
    - strategy_ret
    - equity
    """
    ibit = load_ibit_csv(ibit_csv_path)

    for window in momentum_weights:
        ibit[f"momentum_{window}"] = ibit["close"].pct_change(window)

    weighted_signal = pd.Series(0.0, index=ibit.index, dtype=float)
    for window, weight in momentum_weights.items():
        sig = ibit[f"momentum_{window}"]
        component = pd.Series(0.0, index=ibit.index, dtype=float)

        if long_enabled:
            component = component.where(~(sig > threshold), float(weight))
        if short_enabled:
            component = component.where(~(sig < -threshold), -float(weight))

        weighted_signal = weighted_signal.add(component.fillna(0.0), fill_value=0.0)

    # Convert weighted score to position in the strategy convention.
    # Long-only: signal in {0,1}. Long/short: signal in {-1,0,1}.
    if long_enabled and short_enabled:
        signal = np.sign(weighted_signal)
    elif long_enabled:
        signal = (weighted_signal > 0).astype(float)
    elif short_enabled:
        signal = -(weighted_signal < 0).astype(float)
    else:
        signal = pd.Series(0.0, index=ibit.index)

    ibit["ret"] = ibit["close"].pct_change().fillna(0.0)
    ibit["signal"] = pd.Series(signal, index=ibit.index).shift(1).fillna(0.0)
    ibit["strategy_ret"] = ibit["signal"] * ibit["ret"]
    ibit["equity"] = (1.0 + ibit["strategy_ret"]).cumprod()

    return ibit[["date", "close", "ret", "signal", "strategy_ret", "equity"]]
