from typing import Callable

import polars as pl
from polars import LazyFrame


def TrendFollowingAllocationModel(
    trade_ticker: str,
    price_column: str = "adjusted_close_1d",
    slow_ma_column: str = "close_ma_128",
    fast_ma_column: str = "close_ma_20",
    aggressive_weight: float = 1.3,
    neutral_weight: float = 1.0,
    defensive_weight: float = 0.7,
) -> Callable[[LazyFrame], LazyFrame]:
    """
    Trend-following allocation model based on macro (slow) trend and short-term momentum.

    Rules:
      - price > slow MA and price > fast MA  -> aggressive_weight
      - price > slow MA and price <= fast MA -> neutral_weight
      - price <= slow MA                     -> defensive_weight

    Output columns: ["date", trade_ticker]
    """

    def run_model(lf: LazyFrame) -> LazyFrame:
        base = (
            lf.filter(pl.col("ticker") == trade_ticker)
            .select([
                pl.col("date"),
                pl.col(price_column).alias("price"),
                pl.col(slow_ma_column).alias("slow_ma"),
                pl.col(fast_ma_column).alias("fast_ma"),
            ])
        )

        above_slow = (pl.col("price") > pl.col("slow_ma")).fill_null(False)
        above_fast = (pl.col("price") > pl.col("fast_ma")).fill_null(False)

        weights = (
            pl.when(above_slow & above_fast)
            .then(pl.lit(aggressive_weight))
            .when(above_slow & ~above_fast)
            .then(pl.lit(neutral_weight))
            .otherwise(pl.lit(defensive_weight))
            .cast(pl.Float64)
            .alias(trade_ticker)
        )

        return base.select([pl.col("date"), weights])

    return run_model