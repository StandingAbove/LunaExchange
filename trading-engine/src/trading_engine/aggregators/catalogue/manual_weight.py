from typing import Callable, Dict

import polars as pl
from polars import LazyFrame

def ManualWeightOptimizer(weights: Dict[str, float] = {}) -> Callable[[Dict[str, LazyFrame], Dict], LazyFrame]:
    """
    Manual weight optimizer: applies user-specified weights to model insights.

    Args:
      weights: Dictionary mapping model names to their weights, e.g., {"model1": 0.6, "model2": 0.4}

    Inputs:
      model_insights: { model_name: LazyFrame(["date", ...tickers...]) }

    Output (RAW, wide):
      LazyFrame(["date", ...tickers...]) with per-ticker weights.
    """
    def run(model_insights: Dict[str, LazyFrame], backtest_results: Dict) -> LazyFrame:
        # Check if model_insights is empty
        if not model_insights:
            return pl.DataFrame({"date": []}).lazy()
        
        # Check if weights is empty
        if not weights:
            raise ValueError("weights dictionary is empty")

        # Check if all models in weights dict are in model_insights
        missing_models = set(weights.keys()) - set(model_insights.keys())
        if missing_models:
            raise ValueError(f"models {missing_models} in weights dict not found in model_insights")

        # Check if all models in model_insights are in weights dict
        unweighted_models = set(model_insights.keys()) - set(weights.keys())
        if unweighted_models:
            raise ValueError(f"models {unweighted_models} in model_insights not found in weights dict")

        # Check if weights sum to 1.0 (with tolerance for floating point)
        weight_sum = sum(weights.values())
        if not (0.999 <= weight_sum <= 1.001):
            raise ValueError(f"weights must sum to 1.0, got {weight_sum}")

        # Create model weights lookup
        model_weights_lf = pl.DataFrame({
            "model": list(weights.keys()),
            "model_weight": list(weights.values())
        }).lazy()

        # Combine model weights with ticker positions and aggregate
        combined_long_lf = (
            pl.concat([
                lf.unpivot(index="date", variable_name="ticker", value_name="position")
                  .with_columns(pl.lit(model).alias("model"))
                for model, lf in model_insights.items()
            ])
            .join(model_weights_lf, on="model", how="left")
            .with_columns((pl.col("position") * pl.col("model_weight")).alias("w"))
            .group_by(["date", "ticker"])
            .agg(pl.col("w").sum())
        )

        combined_wide_df = (
            combined_long_lf
            .collect(engine='streaming')  # materialize before pivot
            .pivot(values="w", index="date", on="ticker", aggregate_function="first")
            .sort("date")
        )

        return combined_wide_df.lazy()

    return run