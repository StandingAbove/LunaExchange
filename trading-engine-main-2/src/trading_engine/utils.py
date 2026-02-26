import math

from polars import LazyFrame, scan_parquet


def calculate_calendar_lookback(lookback_days: int, cushion_days: int = 2) -> int:
    # 252 trading days ≈ 365 calendar days; add a small cushion for holidays/halts
    return int(math.ceil(lookback_days * 365 / 252)) + cushion_days


def _read_parquet_data_from_gcs(bucket: str, prefix: str) -> LazyFrame:
    parquet_uri = f"gs://{bucket}/{prefix}/parquet/part-*.parquet"
    return scan_parquet(parquet_uri)
