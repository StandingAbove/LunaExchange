import numpy as np
import pandas as pd

from Backtest.metrics import sharpe_ratio


DAYS_PER_YEAR = 365


# =========================================================
# 1. Local Parameter Perturbation
# =========================================================

def local_perturbation_test(
    df: pd.DataFrame,
    price_column: str,
    base_params: dict,
    param_grid: dict,
    signal_function,
):
    """
    Evaluate sensitivity around base parameters.

    param_grid format:
    {
        "window": [160, 180, 200],
        "entry": [1.8, 2.0, 2.2]
    }
    """

    results = []

    price = df[price_column].astype(float)
    ret = price.pct_change().fillna(0.0)

    keys = list(param_grid.keys())

    # Cartesian product of grid
    import itertools
    combinations = list(itertools.product(*param_grid.values()))

    for combo in combinations:
        params = base_params.copy()
        for i, key in enumerate(keys):
            params[key] = combo[i]

        position = signal_function(df, params)
        strat_ret = position * ret

        s = sharpe_ratio(strat_ret)

        row = params.copy()
        row["Sharpe"] = s
        results.append(row)

    return pd.DataFrame(results)


# =========================================================
# 2. Surface Evaluation
# =========================================================

def evaluate_surface(
    df: pd.DataFrame,
    price_column: str,
    param_grid: dict,
    signal_function,
):
    """
    Full grid surface evaluation.
    """

    results = []

    price = df[price_column].astype(float)
    ret = price.pct_change().fillna(0.0)

    keys = list(param_grid.keys())

    import itertools
    combinations = list(itertools.product(*param_grid.values()))

    for combo in combinations:
        params = dict(zip(keys, combo))

        position = signal_function(df, params)
        strat_ret = position * ret

        s = sharpe_ratio(strat_ret)

        row = params.copy()
        row["Sharpe"] = s
        results.append(row)

    return pd.DataFrame(results)


# =========================================================
# 3. Surface Smoothness Metric
# =========================================================

def surface_smoothness(surface_df: pd.DataFrame, param_name: str) -> float:
    """
    Measure stability along one parameter dimension.

    Lower variance of Sharpe differences = smoother surface.
    """

    df = surface_df.sort_values(param_name)

    sharpe_vals = df["Sharpe"].values

    if len(sharpe_vals) < 3:
        return np.nan

    diffs = np.diff(sharpe_vals)

    return float(np.var(diffs))


# =========================================================
# 4. Stability Score
# =========================================================

def stability_score(surface_df: pd.DataFrame) -> dict:
    """
    Evaluate robustness of parameter surface.

    Outputs:
        - best Sharpe
        - median Sharpe
        - std Sharpe
        - peak-to-median ratio
    """

    sharpes = surface_df["Sharpe"].dropna()

    if len(sharpes) == 0:
        return {}

    best = sharpes.max()
    median = sharpes.median()
    std = sharpes.std()

    peak_to_median = best / median if median != 0 else np.nan

    return {
        "BestSharpe": float(best),
        "MedianSharpe": float(median),
        "SharpeStd": float(std),
        "PeakToMedian": float(peak_to_median),
    }


# =========================================================
# 5. Robustness Flag
# =========================================================

def robustness_flag(stability_metrics: dict) -> str:
    """
    Simple heuristic classification.
    """

    if not stability_metrics:
        return "Insufficient Data"

    peak_ratio = stability_metrics.get("PeakToMedian", np.nan)
    sharpe_std = stability_metrics.get("SharpeStd", np.nan)

    if peak_ratio > 2.0:
        return "Likely Overfit"

    if sharpe_std > 1.0:
        return "Unstable Surface"

    return "Stable"