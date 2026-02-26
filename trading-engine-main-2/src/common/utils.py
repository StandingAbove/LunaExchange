import json
from typing import Dict

import polars as pl

from common.model import Config


def parse_backtest_result(backtest_result: Dict[str, pl.DataFrame]) -> str:
    df = backtest_result.get("backtest_metrics")

    # Handle missing/empty
    if df is None or df.is_empty():
        return "No backtest results available."

    # If there can be duplicate metrics, pick the last occurrence
    # (change to .mean(), .max(), etc. if you prefer)
    if df.select(pl.len()).item() != df.select(pl.col("metric").n_unique()).item():
        df = (
            df.select(["metric", "value"])
            .group_by("metric")
            .agg(pl.col("value").last())
            .rename({"value": "value"})
        )
    else:
        df = df.select(["metric", "value"])

    # Build metric -> value mapping
    metrics = dict(df.iter_rows())

    # If you truly need a string back:
    return json.dumps(metrics, default=float)


def read_config_yaml(config_path: str) -> Config:
    """
    Read a YAML configuration file and return its contents as a dictionary.
    """
    import yaml

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return Config(**config)


def post_to_teams(webhook_url: str, message: str):
    """ Post a message to Microsoft Teams using a webhook URL."""
    import pymsteams
    card = pymsteams.connectorcard(webhook_url)
    card.text(message)
    assert card.send()

    
def _get_metric(df, metric):
    """ Helper to extract metric value from DataFrame. """
    result = df.filter(pl.col("metric") == metric)
    return result["value"][0] if len(result) > 0 else 0.0