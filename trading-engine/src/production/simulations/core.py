from trading_engine.core import (
    orchestrate_portfolio_aggregation,
    orchestrate_portfolio_optimizations,
    orchestrate_portfolio_simulations,
)


def orchestrate_marginal_simulations(
        config,
        model_insights,  # Dict[str, pl.LazyFrame] from orchestrate_model_backtests
        model_backtests,  # Dict[str, dict] from orchestrate_model_simulations
        prices  # pl.DataFrame from construct_prices(...)
):
    """
    For each model in config.models, recompute the full portfolio pipeline with that model removed.
    Returns: reduced_backtests
      - reduced_backtests: { removed_model: { optimizer_name: { kind: pl.DataFrame } } }
    """
    all_reduced_backtests = {}

    # quick exits
    if not getattr(config, "models", None) or len(config.models) <= 1:
        return all_reduced_backtests
    if not getattr(config, "aggregators", None):
        return all_reduced_backtests

    # Iterate: remove one model at a time and rebuild portfolio → optimizer → sims
    for removed in config.models:
        reduced_insights = {k: v for k, v in model_insights.items() if k != removed}
        reduced_model_backtests = {k: v for k, v in model_backtests.items() if k != removed}

        if not reduced_model_backtests:
            # No models left, skip this iteration
            continue

        # 1) Aggregate the remaining models (same aggregators as main)
        aggregated_results = orchestrate_portfolio_aggregation(
            model_insights=reduced_insights,
            backtest_results=reduced_model_backtests,
            universe=config.universe,
            aggregators=config.aggregators,
            start_date=config.start_date,
            end_date=config.end_date,
        )

        # 2) Optimize
        portfolio_optimizers = getattr(config, "optimizers", [])
        optimized_insights = {}
        if portfolio_optimizers:
            optimized_insights = orchestrate_portfolio_optimizations(
                prices=prices,
                aggregated_insights=aggregated_results,
                universe=config.universe,
                optimizers=portfolio_optimizers,
            )

        final_insights = optimized_insights if optimized_insights else aggregated_results

        # 3) Simulate the portfolios
        reduced_portfolio_backtests = orchestrate_portfolio_simulations(
            prices=prices,
            portfolio_insights=final_insights,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_value=1_000_000.0,
        )

        # Store the reduced portfolio backtests
        all_reduced_backtests[removed] = reduced_portfolio_backtests

    return all_reduced_backtests
