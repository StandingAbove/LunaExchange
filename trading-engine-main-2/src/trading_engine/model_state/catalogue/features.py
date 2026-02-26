from typing import Callable

import polars as pl
from polars import LazyFrame, DataFrame


def moving_average(source_col: str, dest_col: str, window: int) -> Callable[[LazyFrame], LazyFrame]:
    def transform(df: LazyFrame) -> LazyFrame:
        return df.with_columns(
            pl.col(source_col)
            .rolling_mean(window_size=window)
            .over("ticker")
            .alias(dest_col)
        )

    return transform


def momentum(source_col: str, dest_col: str, window: int) -> Callable[[LazyFrame], LazyFrame]:
    def transform(df: LazyFrame) -> LazyFrame:
        return df.with_columns(
            (pl.col(source_col) - pl.col(source_col).shift(window)
             ).over("ticker").alias(dest_col)
        )

    return transform


def rsi(source_col: str, dest_col: str, window: int) -> Callable[[DataFrame], DataFrame]:
    def transform(df: DataFrame) -> DataFrame:
        df = df.sort(["ticker", "date"])
        df = df.with_columns([
            (pl.col(source_col) - pl.col(source_col).shift(1)).alias("delta")
        ])

        df = df.with_columns([
            pl.when(pl.col("delta") > 0).then(
                pl.col("delta")).otherwise(0.0).alias("gain"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")
                                              ).otherwise(0.0).alias("loss"),
        ])

        df = df.with_columns([
            pl.col("gain").rolling_mean(window_size=window).over(
                "ticker").alias("avg_gain"),
            pl.col("loss").rolling_mean(window_size=window).over(
                "ticker").alias("avg_loss"),
        ])

        df = df.with_columns([
            (100 - (100 / (1 + (pl.col("avg_gain") / pl.col("avg_loss"))))).alias(dest_col)
        ])

        return df.drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])

    return transform


def natr(
        high_col: str,
        low_col: str,
        close_col: str,
        dest_col: str,
        period: int,
) -> Callable[[LazyFrame], LazyFrame]:
    """
    Wilder-style NATR using EWM(α=1/period) per ticker (lazy).
    Adds `dest_col` to the frame. Assumes rows are sorted by ['ticker','date'].

    NATR = 100 * EWM(TR, alpha=1/period) / close
    TR   = max( high-low, |high - close.shift(1)|, |low - close.shift(1)| )
    """
    alpha = 1.0 / float(period)

    def transform(lf: LazyFrame) -> LazyFrame:
        tr = pl.max_horizontal([
            pl.col(high_col) - pl.col(low_col),
            (pl.col(high_col) - pl.col(close_col).shift(1)).abs(),
            (pl.col(low_col) - pl.col(close_col).shift(1)).abs(),
        ])

        atr = tr.ewm_mean(alpha=alpha).over("ticker")
        natr = ((atr / pl.col(close_col)) * 100.0).alias(dest_col)

        return lf.with_columns(natr)

    return transform
