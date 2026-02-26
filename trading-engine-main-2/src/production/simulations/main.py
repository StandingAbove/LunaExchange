import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from opentelemetry import trace

from common.async_gcs_writer import AsyncGCSWriter
from common.exceptions import NotATradingDayException
from common.logging import setup_logger
from common.model import Config
from common.otel import setup_otel, flush_otel, timed
from common.utils import read_config_yaml
from production.paper.validation import validate_production_config
from production.simulations.core import orchestrate_marginal_simulations
from trading_engine.core import (
    read_data,
    create_model_state,
    orchestrate_model_backtests,
    orchestrate_model_simulations,
    orchestrate_portfolio_aggregation,
    orchestrate_portfolio_optimizations,
    orchestrate_portfolio_simulations,
    calculate_max_lookback,
)

logger = setup_logger(__name__)


async def setup() -> tuple:
    """Setup function to initialize resources if needed."""
    # Config
    config = read_config_yaml("production/simulations/config.yaml")
    validate_production_config(config)
    logger.info(f"Configuration loaded: {config}")

    # Get current date in US Eastern Time
    current_date = datetime.now(ZoneInfo("America/New_York"))
    current_date_str = current_date.strftime("%Y-%m-%d")

    # GCS Writer
    gcs_writer = AsyncGCSWriter(
        bucket_name="wsb-hc-qasap-bucket-1",
        prefix=f"hcf/paper/simulations_audit/{current_date_str}",
    )
    logger.info(f"Using GCS bucket: {gcs_writer.bucket_name}, prefix: {gcs_writer.prefix}")

    return config, current_date_str, gcs_writer


async def run_trading_engine(config: Config, writer: AsyncGCSWriter, current_date: str):
    with timed("simulations.read_data_duration"):
        raw_data_bundle = read_data(include_supplemental=True)

    # ==== validate data (ensure latest data date == current date)
    latest_date = (
        raw_data_bundle.raw_records.select("date").sort("date", descending=True).first().collect().item()
    )

    if latest_date.strftime("%Y-%m-%d") != current_date:
        logger.info(f"Latest date in data: {latest_date}, current date: {current_date}")
        logger.warning("Not a trading day.")
        raise NotATradingDayException()

    # ==== create model state
    with timed("simulations.model_state_duration"):
        # Calculate max lookback across all used components
        total_lookback_days = calculate_max_lookback(
            features=config.model_state_features,
            models=config.models,
            aggregators=config.aggregators,
            optimizers=getattr(config, "optimizers", None),
        )

        model_state_bundle, prices = create_model_state(
            raw_data_bundle=raw_data_bundle,
            features=config.model_state_features,
            start_date=config.start_date,
            end_date=config.end_date,
            universe=config.universe,
            total_lookback_days=total_lookback_days,
            return_bundle=True,
        )
    await writer.save_polars(model_state_bundle.model_state, "model_state.csv")
    await writer.save_polars(model_state_bundle.supplemental_model_state, "supplemental_model_state.csv")
    await writer.save_polars(prices, "prices.csv")

    # ==== orchestrate model backtests
    with timed("simulations.model_backtests_duration"):
        model_insights = orchestrate_model_backtests(
            model_state_bundle=model_state_bundle,
            models=config.models,
            universe=config.universe,
        )
    await asyncio.gather(*(
        writer.save_polars(df, f"model_insights_{name}.csv")
        for name, df in model_insights.items()
    ))

    # ==== orchestrate model simulations
    with timed("simulations.model_simulations_duration"):
        model_backtests = orchestrate_model_simulations(
            prices=prices,
            model_insights=model_insights,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_value=500_000.0,
        )
        
    # Save canonical backtest results to GCS (accurate backtest starting from start_date)
    await asyncio.gather(*(
        writer.save_polars(df, f"model_backtests_{model}_{kind}.csv")
        for model, results in model_backtests.items()
        for kind, df in results["backtest_results"].items()
    ))

    # ==== orchestrate portfolio aggregation
    with timed("simulations.portfolio_aggregation_duration"):
        aggregated_insights = orchestrate_portfolio_aggregation(
            model_insights=model_insights,
            backtest_results=model_backtests,
            universe=config.universe,
            aggregators=config.aggregators,
            start_date=config.start_date,
            end_date=config.end_date,
        )
    await asyncio.gather(*(
        writer.save_polars(df, f"aggregated_insights_{name}.csv")
        for name, df in aggregated_insights.items()
    ))

    # ==== optional: orchestrate asset-level portfolio optimization
    portfolio_optimizers = getattr(config, "optimizers", [])  # optional field
    optimized_insights = {}
    if portfolio_optimizers:
        with timed("simulations.portfolio_optimization_duration"):
            # Optimizers use all available prices/insights from start_date onwards.
            # The lookback calculation ensures sufficient historical data is fetched
            # before start_date for proper feature computation.
            optimized_insights = orchestrate_portfolio_optimizations(
                prices=prices,
                aggregated_insights=aggregated_insights,
                universe=config.universe,
                optimizers=portfolio_optimizers,
            )
        await asyncio.gather(
            *(
                writer.save_polars(df, f"optimized_insights_{name}.csv")
                for name, df in optimized_insights.items()
            )
        )

    # Choose which insights to simulate and return (prefer optimizer if present)
    final_insights = optimized_insights if optimized_insights else aggregated_insights

    # ==== orchestrate portfolio simulations
    with timed("simulations.portfolio_simulations_duration"):
        portfolio_backtests = orchestrate_portfolio_simulations(
            prices=prices,
            portfolio_insights=final_insights,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_value=1_000_000.0,
        )
    await asyncio.gather(*(
        writer.save_polars(df, f"portfolio_backtests_{pname}_{kind}.csv")
        for pname, results in portfolio_backtests.items()
        for kind, df in results.items()
    ))

    # ==== marginal simulations (reduced portfolio backtests)
    with timed("simulations.marginal_simulations_duration"):
        reduced_portfolio_backtests = orchestrate_marginal_simulations(
            config=config,
            model_insights=model_insights,
            model_backtests=model_backtests,
            prices=prices,
        )

    # Save reduced portfolio backtests (full backtest data)
    await asyncio.gather(*(
        writer.save_polars(df, f"reduced_portfolio_backtests_{removed_model}_{optimizer}_{kind}.csv")
        for removed_model, optimizer_results in reduced_portfolio_backtests.items()
        for optimizer, results in optimizer_results.items()
        for kind, df in results.items()
    ))



async def main():
    logger.info("Starting simulations pipeline.")
    # ==== setup
    c, current_date, writer = await setup()

    # ==== trading engine
    await run_trading_engine(c, writer, current_date)

    # ==== Flush write to GCS
    c.dump_to_gcs(f"gs://{writer.bucket_name}/{writer.prefix}/config.json")
    await writer.flush()
    await writer.close()


if __name__ == "__main__":
    setup_otel("simulations_paper")
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("simulations") as span:
        try:
            asyncio.run(main())
        except NotATradingDayException:
            logger.info("Not a trading day, exiting gracefully.")
            span.set_status(trace.StatusCode.UNSET, "Not a trading day")
        except Exception as e:
            logger.error(f"Error running simulations pipeline: {e}")
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.record_exception(e)
            flush_otel()
            raise

    flush_otel()
