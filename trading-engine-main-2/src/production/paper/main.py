import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from opentelemetry import trace
from polars import DataFrame

from common.async_gcs_writer import AsyncGCSWriter
from common.exceptions import NotATradingDayException
from common.interactive_brokers import IBKR
from common.logging import setup_logger
from common.model import Config
from common.otel import setup_otel, flush_otel, timed
from common.utils import read_config_yaml, post_to_teams
from production.paper.core import (
    construct_goal_positions,
    construct_rebalance_orders,
    to_ibkr_basket_csv,
)
from production.paper.validation import validate_production_config
from trading_engine.core import (
    read_data,
    create_model_state,
    orchestrate_model_backtests,
    orchestrate_model_simulations,
    orchestrate_portfolio_simulations,
    orchestrate_portfolio_aggregation,
    orchestrate_portfolio_optimizations,
    calculate_max_lookback,
)

logger = setup_logger(__name__)


async def setup() -> tuple:
    """Setup function to initialize resources if needed."""
    # Config
    config = read_config_yaml("production/paper/config.yaml")
    validate_production_config(config)
    logger.info(f"Configuration loaded: {config}")

    # TODO: Add max(current_date, config.start_date) logic for testing
    current_date = datetime.now(ZoneInfo("America/New_York"))
    current_date_str = current_date.strftime("%Y-%m-%d")

    # GCS Writer
    gcs_writer = AsyncGCSWriter(
        bucket_name="wsb-hc-qasap-bucket-1",
        prefix=f"hcf/paper/production_audit/{current_date_str}",
    )
    logger.info(
        f"Using GCS bucket: {gcs_writer.bucket_name}, prefix: {gcs_writer.prefix}"
    )

    return config, current_date_str, gcs_writer


async def run_trading_engine(config: Config, writer: AsyncGCSWriter, current_date: str):
    with timed("production.read_data_duration"):
        raw_data_bundle = read_data(include_supplemental=True)

    # ==== validate data (check if the latest date in the data is "current" date, otherwise it's not a trading day)
    latest_date = (
        raw_data_bundle.raw_records.select("date").sort("date", descending=True).first().collect().item()
    )

    if latest_date.strftime("%Y-%m-%d") != current_date:
        logger.info(f"Latest date in data: {latest_date}, current date: {current_date}")
        logger.warning("Not a trading day.")
        raise NotATradingDayException()

    # ==== create model state
    with timed("production.model_state_duration"):
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
    with timed("production.model_backtests_duration"):
        model_insights = orchestrate_model_backtests(
            model_state_bundle=model_state_bundle, models=config.models, universe=config.universe
        )
    await asyncio.gather(
        *(
            writer.save_polars(df, f"model_insights_{name}.csv")
            for name, df in model_insights.items()
        )
    )

    # ==== orchestrate model simulations
    with timed("production.model_simulations_duration"):
        model_backtests = orchestrate_model_simulations(
            prices=prices,
            model_insights=model_insights,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_value=500_000.0,
        )

    # Save canonical backtest results to GCS (accurate backtest starting from start_date)
    await asyncio.gather(
        *(
            writer.save_polars(df, f"model_backtests_{model}_{kind}.csv")
            for model, results in model_backtests.items()
            for kind, df in results["backtest_results"].items()
        )
    )

    # ==== orchestrate portfolio aggregation
    with timed("production.portfolio_aggregation_duration"):
        aggregated_insights = orchestrate_portfolio_aggregation(
            model_insights=model_insights,
            backtest_results=model_backtests,
            universe=config.universe,
            aggregators=config.aggregators,
            start_date=config.start_date,
            end_date=config.end_date,
        )
    await asyncio.gather(
        *(
            writer.save_polars(df, f"aggregated_insights_{name}.csv")
            for name, df in aggregated_insights.items()
        )
    )

    # ==== optional: orchestrate asset-level portfolio optimization
    portfolio_optimizers = getattr(config, "optimizers", [])  # optional field
    optimized_insights = {}
    if portfolio_optimizers:
        with timed("production.portfolio_optimization_duration"):
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
    with timed("production.portfolio_simulations_duration"):
        portfolio_backtests = orchestrate_portfolio_simulations(
            prices=prices,
            portfolio_insights=final_insights,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_value=1_000_000.0,
        )
    await asyncio.gather(
        *(
            writer.save_polars(df, f"portfolio_backtests_{pname}_{kind}.csv")
            for pname, results in portfolio_backtests.items()
            for kind, df in results.items()
        )
    )

    return final_insights, prices


async def run_execution_engine(
        config: Config,
        writer: AsyncGCSWriter,
        prices: DataFrame,
        portfolio_insight: DataFrame,
):
    ib_client = await IBKR.create(
        hostname=config.ib_gateway.host,
        port=config.ib_gateway.port,
        flex_web_token=config.ib_gateway.flex_web_token,
        nav_flex_query_id=config.ib_gateway.nav_flex_query_id,
        fund_inception_date=config.ib_gateway.fund_inception_date,
        client_id=config.ib_gateway.client_id,
    )

    # ==== calculate goal positions
    with timed("production.goal_positions_duration"):
        goal_positions = construct_goal_positions(
            ib_client=ib_client,
            insights=portfolio_insight,
            prices=prices,
            universe=config.universe,
            cash_buffer_pct=float(config.cash_buffer),
        )
    await writer.save_polars(goal_positions, "goal_positions.csv")

    # ==== build rebalance orders
    with timed("production.rebalance_orders_duration"):
        rebalance_df = construct_rebalance_orders(
            ib_client=ib_client,
            targets=goal_positions,
            universe=config.universe,
            close_out_outside_universe=True,
        )
    await writer.save_polars(rebalance_df, "rebalance_orders.csv")

    # ==== build BasketTrader CSV (all orders as MOC, SMART routed, TIF=DAY)
    basket_csv = to_ibkr_basket_csv(
        rebalance_df,
        order_type="MOC",
        time_in_force="DAY",
        exchange="SMART",
    )

    # ==== save historical NAV report
    await writer.save_polars(ib_client.get_historical_nav(), "historical_nav.csv")

    # ==== write CSV to GCS and return link
    await writer.save_text(
        basket_csv, "ibkr_basket_moc.csv", content_type="text/csv; charset=utf-8"
    )
    basket_url = f"https://storage.cloud.google.com/{writer.bucket_name}/{writer.prefix}/ibkr_basket_moc.csv"
    return basket_url


async def main():
    logger.info("Starting production pipeline.")
    # ==== setup
    c, current_date, writer = await setup()

    # ==== trading engine
    portfolio_insights, prices = await run_trading_engine(c, writer, current_date)

    # ==== execution engine
    # TODO: This will break if we have multiple optimizers
    portfolio_name, portfolio_insight = portfolio_insights.popitem()

    basket_path = None
    try:
        basket_path = await run_execution_engine(c, writer, prices, portfolio_insight)
    except Exception as e:
        logger.error(f"Failed to run execution engine: {e}")

    # ==== Flush write to GCS
    c.dump_to_gcs(f"gs://{writer.bucket_name}/{writer.prefix}/config.json")
    await writer.flush()
    await writer.close()

    # Post Goal Positions to Teams
    last_row_dict = portfolio_insight[-1].to_dict(as_series=False)
    goal_position_list = [
        f"{k}: {round(v[0] * 100, 2)}%" for k, v in last_row_dict.items() if k != "date"
    ]
    goal_position_string = "<br>".join(goal_position_list)
    message = (
        f"<strong>Goal Positions ({current_date})</strong><br>{goal_position_string}"
    )

    if basket_path:
        message += (
            f"<br><br>IBKR Basket (MOC): <a href='{basket_path}'>Download CSV</a>"
        )

    logger.info(message)

    post_to_teams(webhook_url=c.notifications["msteams_webhook"], message=message)


if __name__ == "__main__":
    setup_otel("production_paper")
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("production") as span:
        try:
            asyncio.run(main())
        except NotATradingDayException:
            logger.info("Not a trading day, exiting gracefully.")
            span.set_status(trace.StatusCode.UNSET, "Not a trading day")
            # We need to find a better way to skip trading days.
        except Exception as e:
            logger.error(f"Error running production pipeline: {e}")
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.record_exception(e)
            flush_otel()
            raise

    flush_otel()
