from typing import Callable, List

import polars as pl
from polars import LazyFrame


def InverseMomentumMeanReversionModel(
        tickers: List[str],
        momentum_column: str,
        threshold: float = 0.05,
) -> Callable[[LazyFrame], LazyFrame]:
    """
    Mean-reversion model using a precomputed momentum feature.

    Rule per ticker t:
      momentum[t] < -threshold  -> weight[t] = +1.0   (oversold -> long)
      momentum[t] > +threshold  -> weight[t] = -1.0   (overbought -> short)
      otherwise                 -> weight[t] =  0.0

    Inputs to `run_model` (from orchestrator):
      lf: LazyFrame with rows only for `tickers` and columns ["date", "ticker", momentum_column]

    Output (wide, lazy):
      LazyFrame with columns ["date", *tickers] containing {-1.0, 0.0, +1.0}
      (No clamping/L1/padding here; the orchestrator will handle that globally.)
    """
    thr = float(threshold)

    def run_model(lf: LazyFrame) -> LazyFrame:
        # 1) Filter to the requested tickers & select only needed columns (stays lazy)
        base = (
            lf.filter(pl.col("ticker").is_in(tickers))
            .select(["date", "ticker", momentum_column])
        )

        # 2) Row-wise signal -> {-1,0,+1} (vectorized)
        #    +1 when momentum < -thr, -1 when momentum > +thr, else 0
        long_short = (
            pl.when(pl.col(momentum_column) < -thr).then(1.0)
            .when(pl.col(momentum_column) > thr).then(-1.0)
            .otherwise(0.0)
            .cast(pl.Float64)
            .alias("w")
        )

        long_df = base.select(["date", "ticker", long_short])

        # 3) "Manual pivot" to wide lazily: one column per ticker
        #    Using conditional sums keeps everything lazy (no DataFrame.pivot())
        aggs = [
            pl.col("w").filter(pl.col("ticker") == pl.lit(t)).sum().alias(t)
            for t in tickers
        ]

        wide = (
            long_df
            .group_by("date")
            .agg(aggs)  # -> ["date", *tickers]
            .with_columns([pl.col(t).fill_null(0.0).alias(t) for t in tickers])
            .sort("date")
        )

        return wide  # LazyFrame

    return run_model
