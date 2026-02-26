import numpy as np
from scipy.stats import ttest_rel

from common.utils import _get_metric
from trading_engine.core import run_full_backtest


def model_review_tests(baseline_config, candidate_config):
    """Validates if new model improves portfolio performance, risk, and diversification"""
    baseline = run_full_backtest(**baseline_config)
    candidate = run_full_backtest(**candidate_config)

    baseline_models = baseline["model_simulations"]
    candidate_models = candidate["model_simulations"]
    baseline_portfolio = baseline["optimizer_simulations"]
    candidate_portfolio = candidate["optimizer_simulations"]

    baseline_optimizer = baseline_config["optimizers"][0]
    candidate_optimizer = candidate_config["optimizers"][0]

    baseline_results = baseline_portfolio[baseline_optimizer]["backtest_results"]
    candidate_results = candidate_portfolio[candidate_optimizer]["backtest_results"]
    baseline_metrics = baseline_portfolio[baseline_optimizer]["backtest_metrics"]
    candidate_metrics = candidate_portfolio[candidate_optimizer]["backtest_metrics"]

    baseline_returns = np.asarray(baseline_results["daily_return"], dtype=float)
    candidate_returns = np.asarray(candidate_results["daily_return"], dtype=float)

    min_len = min(len(baseline_returns), len(candidate_returns))
    baseline_returns = baseline_returns[-min_len:]
    candidate_returns = candidate_returns[-min_len:]

    # T-test: Statistical test to determine if performance difference is significant
    if np.allclose(candidate_returns, baseline_returns):
        t_stat, p_value = 0.0, 1.0
    else:
        t_stat, p_value = ttest_rel(candidate_returns, baseline_returns)

    # Information Ratio: Risk-adjusted measure of active return (like Sharpe for excess performance)
    active_returns = candidate_returns - baseline_returns
    active_return_ann = np.mean(active_returns) * 252
    tracking_error = np.std(active_returns) * np.sqrt(252)
    information_ratio = (
        active_return_ann / tracking_error if tracking_error > 1e-8 else 0.0
    )

    # Economic Significance: Total return improvement minus additional trading costs
    gross_benefit = _get_metric(candidate_metrics, "annualized_return") - _get_metric(
        baseline_metrics, "annualized_return"
    )
    transaction_cost = _get_metric(
        candidate_metrics, "cumulative_slippage_cost"
    ) - _get_metric(baseline_metrics, "cumulative_slippage_cost")
    net_economic_benefit = gross_benefit - transaction_cost

    # Model Correlation: Measures similarity between new model and existing models (0=uncorrelated, 1=identical)
    correlations = []
    for model_name, model_data in baseline_models.items():
        if isinstance(model_data, dict) and "backtest_results" in model_data:
            # New structure: model_data["backtest_results"] is a dict with "backtest_results", "portfolio_value", etc.
            backtest_dict = model_data["backtest_results"]
            if isinstance(backtest_dict, dict) and "backtest_results" in backtest_dict:
                model_results = backtest_dict["backtest_results"]
                model_returns = np.asarray(model_results["daily_return"], dtype=float)[
                                -min_len:
                                ]
                if len(model_returns) == len(candidate_returns):
                    corr = np.corrcoef(candidate_returns, model_returns)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(float(corr))

    # Max Drawdown Impact: Change in worst peak-to-trough loss when adding new model
    max_drawdown_diff = _get_metric(candidate_metrics, "max_drawdown") - _get_metric(
        baseline_metrics, "max_drawdown"
    )

    return {
        "p_value": float(p_value),
        "information_ratio": float(information_ratio),
        "net_economic_benefit": float(net_economic_benefit),
        "correlations": correlations,
        "max_drawdown_diff": float(max_drawdown_diff),
    }

# to-do: write plotting utils for model backtest performance.
# plot equity curve, with drawdown subplot (smaller)
# plot histogram of returns, skew metrics, kurtosis metrics
# plot correlation matrix of multiple models?
# Plot multiple backtest results -- could be models/portfolios/optimized/aggregated/etc.
