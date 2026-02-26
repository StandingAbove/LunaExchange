from dataclasses import dataclass
from typing import List

import polars as pl


@dataclass(frozen=True)
class RawDataBundle:
    raw_records: pl.LazyFrame
    raw_supplemental_records: pl.LazyFrame

    def require(self, name: str) -> pl.LazyFrame:
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(f"Missing dataset: {name}")


@dataclass(frozen=True)
class ModelStateBundle:
    model_state: pl.DataFrame
    supplemental_model_state: pl.DataFrame

    def require(self, name: str) -> pl.DataFrame:
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(f"Missing dataset: {name}")

    def filter_model_state(
            self, tickers: List[str], columns: List[str]
    ) -> pl.LazyFrame:
        """
        Helper to filter model_state to specific tickers and columns.
        
        Args:
            tickers: List of ticker symbols to include
            columns: List of column names to select (in addition to date and ticker)
            
        Returns:
            Filtered LazyFrame with columns ["date", "ticker", *columns]
        """
        cols = ["date", "ticker", *columns]
        return self.model_state.lazy().filter(pl.col("ticker").is_in(tickers)).select(cols)
