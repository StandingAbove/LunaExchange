from typing import Callable

import polars as pl
from polars import LazyFrame


def MomentumModel(
        trade_ticker: str,
        signal_ticker: str,
        momentum_column: str,
        inverse: bool = True,  # True => long when signal < 0, else flat
        threshold: float = 0.0,  # optional dead-band around 0 to avoid noise
) -> Callable[[LazyFrame], LazyFrame]:
    """
    Generate weights for `trade_ticker` from the momentum of `signal_ticker`.

    inverse=True  -> long when momentum(signal_ticker) < -threshold, else 0
    inverse=False -> long when momentum(signal_ticker) >  threshold, else 0

    Output columns: ["date", trade_ticker]
    """

    def run_model(lf: LazyFrame) -> LazyFrame:
        # Pull only the signal series (lazy)
        sig = (
            lf.filter(pl.col("ticker") == signal_ticker)
            .select([
                pl.col("date"),
                pl.col(momentum_column).alias("sig")
            ])
        )

        # Condition: long when (sig < -threshold) if inverse else (sig > threshold)
        cond = (pl.col("sig") < -threshold) if inverse else (pl.col("sig") > threshold)

        # Vectorized mapping to {1.0, 0.0}; treat null as False via fill_null(False)
        weights = sig.select([
            pl.col("date"),
            pl.when(cond.fill_null(False))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float64)
            .alias(trade_ticker)
        ])

        return weights  # LazyFrame

    return run_model
