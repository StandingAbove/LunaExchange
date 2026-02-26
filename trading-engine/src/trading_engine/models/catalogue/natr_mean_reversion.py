from typing import Callable

import polars as pl
from polars import LazyFrame


def NATRMeanReversionModel(
        trade_ticker: str,
        *,
        price_col: str = "adjusted_close_1d",
        natr7_col: str = "natr_7",
        natr14_col: str = "natr_14",
        ret1_col: str = "close_momentum_1",
        ret14_col: str = "close_momentum_14",
        ret32_col: str = "close_momentum_32",
        ret64_col: str = "close_momentum_64",
        # parameters
        lookback_window: int = 14,
        lower_mult: float = 1.5,
        upper_mult: float = 3.0,
        use_std_guard: bool = True,
) -> Callable[[LazyFrame], LazyFrame]:
    """
    Mean-reversion using NATR(7,14) and multi-horizon momentum.

    Rules (per date for `trade_ticker`):
      - momentum buildup:   NATR_7(prev) > NATR_14(prev)
      - return band:        1.5*NATR_7(prev) < |ret1| < 3*NATR_7(prev)
      - if ret1 < 0:
            if ret14 < ret32 < ret64 -> no trade
            else -> long  with weight = -(NATR_7(prev) + ret1)
        elif ret1 > 0:
            short with weight = -(ret1 - NATR_7(prev))
        else: 0
      - else (no buildup or out of band): 0

    Output (wide, lazy): LazyFrame with columns ["date", trade_ticker].
    Post-processing (padding, clamping, L1) is done in the orchestrator.
    """

    def run(lf: LazyFrame) -> LazyFrame:
        # Slice to the trade ticker and only required columns (keeps plan lean)
        base = (
            lf.filter(pl.col("ticker") == trade_ticker)
            .select([
                "date",
                price_col,
                natr7_col, natr14_col,
                ret1_col, ret14_col, ret32_col, ret64_col,
            ])
        )

        # Previous-day NATR (your original used [-2] at the end -> shift(1))
        n7_prev = pl.col(natr7_col).shift(1)
        n14_prev = pl.col(natr14_col).shift(1)

        # Returns
        ret1 = pl.col(ret1_col)
        ret14 = pl.col(ret14_col)
        ret32 = pl.col(ret32_col)
        ret64 = pl.col(ret64_col)

        # Guards
        buildup = (n7_prev > n14_prev)
        in_band = (ret1.abs() > (lower_mult * n7_prev)) & (ret1.abs() < (upper_mult * n7_prev))
        recession = (ret14 < ret32) & (ret32 < ret64)

        if use_std_guard:
            std_ok = pl.col(price_col).rolling_std(window_size=lookback_window).gt(0).fill_null(False)
        else:
            std_ok = pl.lit(True)

        # Weight formula
        w_expr = (
            pl.when(std_ok & buildup & in_band & (ret1 < 0) & (~recession))
            .then(-(n7_prev + ret1))
            .when(std_ok & buildup & in_band & (ret1 > 0))
            .then(-(ret1 - n7_prev))
            .otherwise(0.0)
            .fill_null(0.0)
            .cast(pl.Float64)
            .alias(trade_ticker)
        )

        weights = (
            base.select([
                pl.col("date"),
                w_expr,
            ])
            .sort("date")
        )

        return weights  # LazyFrame

    return run
