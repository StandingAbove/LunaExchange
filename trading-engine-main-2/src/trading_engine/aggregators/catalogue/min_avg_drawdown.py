import datetime as _dt
from typing import Callable, Dict, List

import polars as pl
from polars import LazyFrame

_EPS = 1e-9


def MinAvgDrawdownAggregator(
    window_days: int = 90,
) -> Callable[[Dict[str, LazyFrame], Dict], LazyFrame]:
    """
    Build a portfolio by weighting model insights inversely to their average drawdown
    over the last `window_days`. Models with lower recent drawdown get higher weight.

    :param model_insights: { model_name: LazyFrame(["date", ...tickers...]) } (wide)
    :param backtest_results: mapping with keys per model; may contain 'backtest_results' DataFrame
    :return: LazyFrame(["date", ...all tickers seen in inputs...]) without padding/clamping/L1 scaling.
    """

    def _avg_drawdown_recent(metrics_df: pl.DataFrame) -> float:
        if metrics_df is None or metrics_df.is_empty():
            return float("inf")

        # Standardize date to pl.Date
        dt = metrics_df.get_column("date")
        if dt.dtype == pl.Utf8:
            metrics_df = metrics_df.with_columns(
                pl.col("date").str.strptime(pl.Date, strict=False).alias("date")
            )
        elif dt.dtype == pl.Datetime:
            metrics_df = metrics_df.with_columns(pl.col("date").dt.date().alias("date"))
        # If already pl.Date, leave as-is.

        # Window: last `window_days` calendar days from max date
        max_d = metrics_df.select(pl.col("date").max()).item()
        if max_d is None:
            return float("inf")
        start = max_d - _dt.timedelta(days=window_days)

        # Drawdown magnitude (use abs in case stored negative)
        win = metrics_df.filter(pl.col("date") >= pl.lit(start))
        if win.is_empty():
            win = metrics_df

        val = win.select(pl.col("drawdown").abs().mean()).item()
        if val is None or not (val == val):  # NaN check
            return float("inf")
        return float(val)

    def _coef_from_avg_dd(avg_dd: float) -> float:
        # Lower avg drawdown => higher coefficient. Guard with epsilon.
        if avg_dd == float("inf"):
            return 0.0
        return 1.0 / (_EPS + max(avg_dd, 0.0))

    def run(model_insights: Dict[str, LazyFrame], backtest_results: Dict) -> LazyFrame:
        if not model_insights:
            return pl.DataFrame({"date": []}).lazy()

        # 1) Compute per-model coefficients from backtest_metrics
        coefs: Dict[str, float] = {}
        for mname, lf in model_insights.items():
            metrics = None
            if mname in backtest_results:
                metrics = backtest_results[mname].get("backtest_results")

            # Accept either Polars or Pandas metrics (convert if needed)
            if metrics is None:
                avg_dd = float("inf")
            elif isinstance(metrics, pl.DataFrame):
                avg_dd = _avg_drawdown_recent(metrics)
            else:
                try:
                    import pandas as pd

                    if isinstance(metrics, pd.DataFrame):
                        avg_dd = _avg_drawdown_recent(pl.from_pandas(metrics))
                    else:
                        avg_dd = float("inf")
                except Exception:
                    avg_dd = float("inf")

            coefs[mname] = _coef_from_avg_dd(avg_dd)

        # Normalize coefficients to sum to 1; if all zero, fall back to equal weight
        total = sum(coefs.values())
        if total <= _EPS:
            n = float(len(model_insights))
            coefs = {k: 1.0 / n for k in model_insights.keys()}
        else:
            coefs = {k: v / total for k, v in coefs.items()}

        # 2) Scale each model's weights by its coefficient and combine
        longs: List[LazyFrame] = []
        for mname, lf in model_insights.items():
            coef = coefs.get(mname, 0.0)

            # Multiply all non-'date' columns by coef
            names = lf.collect_schema().names()
            wcols = [c for c in names if c != "date"]
            if not wcols:
                continue

            scaled = lf.with_columns(
                [(pl.col(c) * pl.lit(coef)).alias(c) for c in wcols]
            )

            long = scaled.melt(id_vars="date", variable_name="ticker", value_name="w")
            longs.append(long)

        if not longs:
            return pl.DataFrame({"date": []}).lazy()

        # 3) Sum across models in long space, pivot to wide
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
