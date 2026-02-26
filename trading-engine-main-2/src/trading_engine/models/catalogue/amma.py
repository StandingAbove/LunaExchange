from typing import Callable, Dict

import polars as pl
from polars import LazyFrame

from common.bundles import ModelStateBundle


def AMMA(
        ticker: str,
        momentum_weights: Dict[int, float],  # {window: weight}
        threshold: float = 0.0,
        long_enabled: bool = True,
        short_enabled: bool = False,
) -> Callable[[ModelStateBundle], LazyFrame]:
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
    Callable[[ModelStateBundle], LazyFrame]
        Function that takes a ModelStateBundle and returns a LazyFrame of weights.
    """

    def run_model(bundle: ModelStateBundle) -> LazyFrame:
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

            # Signal logic
            long_cond = (pl.col("sig") > threshold) if long_enabled else None
            short_cond = (pl.col("sig") < -threshold) if short_enabled else None

            expr = pl.lit(0.0)

            if long_enabled and short_enabled:
                # Long if positive, short if negative
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

        # Join on date
        combined = sig_frames[0]
        for frame in sig_frames[1:]:
            combined = combined.join(frame, on="date", how="inner")

        # Sum across signals
        weight_cols = [f"sig_{w}" for w in momentum_weights.keys()]
        final = combined.with_columns(
            sum([pl.col(c) for c in weight_cols]).alias(ticker)
        ).select(["date", ticker])

        return final

    return run_model
