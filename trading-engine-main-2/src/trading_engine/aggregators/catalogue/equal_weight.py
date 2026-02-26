from typing import Callable, Dict

import polars as pl
from polars import LazyFrame


def EqualWeightAggregator() -> Callable[[Dict[str, LazyFrame], Dict], LazyFrame]:
    """
    Averaging aggregator: equally weights model insights.

    :param model_insights: { model_name: LazyFrame(["date", ...tickers...]) }
    :param backtest_results: ignored here (hook for future aggregators)
    :return: LazyFrame(["date", ...tickers...]) with per-ticker weights (no clamping/padding/L1 scaling).
    """

    def run(model_insights: Dict[str, LazyFrame], backtest_results: Dict) -> LazyFrame:
        if not model_insights:
            # empty portfolio with just date column
            return pl.DataFrame({"date": []}).lazy()

        n = float(len(model_insights))
        longs = []
        for lf in model_insights.values():
            # Melt model's wide weights to long, scale by 1/n
            long = lf.unpivot(
                index="date", variable_name="ticker", value_name="w"
            ).with_columns((pl.col("w") * (1.0 / n)).alias("w"))
            longs.append(long)

        # concat/group_by stays lazy; pivot requires an eager DataFrame in many Polars versions
        combined_long_lf = (
            pl.concat(longs, how="vertical")
            .group_by(["date", "ticker"])
            .agg(pl.col("w").sum().alias("w"))
        )

        combined_wide_df = (
            combined_long_lf.collect(engine="streaming")
            .pivot(
                values="w", index="date", columns="ticker", aggregate_function="first"
            )
            .sort("date")
        )

        return combined_wide_df.lazy()

    return run
