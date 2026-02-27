from typing import Any, Callable, Dict

import pandas as pd


def AMMA(
        ticker: str,
        momentum_weights: Dict[int, float],  # {window: weight}
        threshold: float = 0.0,
        long_enabled: bool = True,
        short_enabled: bool = False,
) -> Callable[[Any], Any]:
    """
    Adaptive Momentum Model Averaging (AMMA).

    Parameters
    ----------
    ticker : str
        The trade ticker (column name for output weights).
    momentum_weights : dict[int, float]
        Mapping of momentum windows to their assigned weights.
    threshold : float, optional
        Momentum threshold for signal generation.
    long_enabled : bool, optional
        If True, allow long signals when momentum > threshold.
    short_enabled : bool, optional
        If True, allow short signals when momentum < -threshold.

    Returns
    -------
    Callable[[Any], Any]
        Function that takes a model-state bundle and returns a Polars LazyFrame of weights.
    """

    def run_model(bundle: Any) -> Any:
        import polars as pl

        lf = bundle.model_state.lazy()
        sig_frames = []

        for window, weight in momentum_weights.items():
            colname = f"close_momentum_{window}"

            sig = (
                lf.filter(pl.col("ticker") == ticker)
                .select([
                    pl.col("date"),
                    pl.col(colname).alias("sig")
                ])
            )

            long_cond = (pl.col("sig") > threshold) if long_enabled else None
            short_cond = (pl.col("sig") < -threshold) if short_enabled else None
            expr = pl.lit(0.0)

            if long_enabled and short_enabled:
                expr = (
                    pl.when(long_cond.fill_null(False))
                    .then(pl.lit(1.0) * weight)
                    .when(short_cond.fill_null(False))
                    .then(pl.lit(-1.0) * weight)
                    .otherwise(pl.lit(0.0))
                )
            elif long_enabled:
                expr = pl.when(long_cond.fill_null(False)).then(pl.lit(1.0) * weight).otherwise(pl.lit(0.0))
            elif short_enabled:
                expr = pl.when(short_cond.fill_null(False)).then(pl.lit(-1.0) * weight).otherwise(pl.lit(0.0))

            weighted_sig = sig.select([
                pl.col("date"),
                expr.cast(pl.Float64).alias(f"sig_{window}")
            ])
            sig_frames.append(weighted_sig)

        combined = sig_frames[0]
        for frame in sig_frames[1:]:
            combined = combined.join(frame, on="date", how="inner")

        weight_cols = [f"sig_{w}" for w in momentum_weights.keys()]
        final = combined.with_columns(
            sum([pl.col(c) for c in weight_cols]).alias(ticker)
        ).select(["date", ticker])

        return final

    return run_model


def amma_from_ibit_csv(
        ibit_csv_path: str,
        momentum_weights: Dict[int, float],
        threshold: float = 0.0,
        long_enabled: bool = True,
        short_enabled: bool = False,
) -> pd.DataFrame:
    """
    Build daily AMMA weights directly from the IBIT CSV schema.

    The IBIT file is expected to include at least ``Date`` and ``Price`` columns.
    Output is a DataFrame with columns: ``Date``, ``price``, ``amma_weight``,
    and ``amma_position``.
    """
    ibit = pd.read_csv(ibit_csv_path).copy()

    if "Date" not in ibit.columns or "Price" not in ibit.columns:
        raise ValueError("IBIT CSV must contain 'Date' and 'Price' columns.")

    ibit["Date"] = pd.to_datetime(ibit["Date"], format="%m/%d/%y", errors="coerce")
    ibit["price"] = (
        ibit["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .astype(float)
    )
    ibit = ibit.dropna(subset=["Date", "price"]).sort_values("Date").reset_index(drop=True)

    for window in momentum_weights:
        col = f"momentum_{window}"
        ibit[col] = ibit["price"].pct_change(window)

    weighted_signal = pd.Series(0.0, index=ibit.index)
    for window, weight in momentum_weights.items():
        sig = ibit[f"momentum_{window}"]
        component = pd.Series(0.0, index=ibit.index)

        if long_enabled:
            component = component.where(~(sig > threshold), weight)
        if short_enabled:
            component = component.where(~(sig < -threshold), -weight)

        weighted_signal = weighted_signal.add(component.fillna(0.0), fill_value=0.0)

    ibit["amma_weight"] = weighted_signal.fillna(0.0)
    ibit["amma_position"] = ibit["amma_weight"].shift(1).fillna(0.0)
    return ibit[["Date", "price", "amma_weight", "amma_position"]]
