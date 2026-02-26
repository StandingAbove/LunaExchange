from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "research_02"
RNG_SEED = 7
TRADING_DAYS = 252
N_DAYS = 900


@dataclass
class PerformanceSummary:
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float



def performance_stats(returns: pd.Series) -> PerformanceSummary:
    ann_ret = float((1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(TRADING_DAYS))
    curve = (1 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    return PerformanceSummary(ann_ret, ann_vol, sharpe, max_dd, calmar)



def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[idx].mean(axis=1)
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)



def generate_series() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    dates = pd.bdate_range("2021-01-01", periods=N_DAYS)

    # proxy baseline portfolio returns (engine without candidate model)
    market = rng.normal(0.00035, 0.0135, size=N_DAYS)
    baseline_noise = rng.normal(0, 0.0035, size=N_DAYS)
    baseline = 0.7 * market + baseline_noise

    # candidate model: positively skewed alpha with moderate correlation
    model_noise = rng.normal(0, 0.008, size=N_DAYS)
    model_alpha = rng.normal(0.00032, 0.003, size=N_DAYS)
    candidate_model = 0.35 * market + model_noise + model_alpha

    return pd.DataFrame(
        {
            "baseline": baseline,
            "candidate_model": candidate_model,
        },
        index=dates,
    )



def optimize_marginal_weight(df: pd.DataFrame) -> float:
    grid = np.linspace(0.0, 0.6, 121)
    best_weight = 0.0
    best_sharpe = -np.inf

    for w in grid:
        combined = (1 - w) * df["baseline"] + w * df["candidate_model"]
        std = combined.std(ddof=1)
        if std == 0:
            continue
        sharpe = (combined.mean() / std) * np.sqrt(TRADING_DAYS)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weight = float(w)

    return best_weight



def make_plots(df: pd.DataFrame, combined: pd.Series, marginal: pd.Series) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_curve = (1 + df["baseline"]).cumprod()
    combined_curve = (1 + combined).cumprod()

    plt.figure(figsize=(11, 6))
    plt.plot(df.index, baseline_curve, label="Trading Engine (Baseline)", linewidth=2)
    plt.plot(df.index, combined_curve, label="Baseline + Candidate Model", linewidth=2)
    plt.title("Research 02: Portfolio Equity Curve Comparison")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "equity_curve_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(marginal, bins=40, alpha=0.75, color="#2e86de", edgecolor="white")
    plt.axvline(marginal.mean(), color="red", linestyle="--", linewidth=1.8, label=f"Mean={marginal.mean():.5f}")
    plt.title("Distribution of Marginal Return Stream")
    plt.xlabel("Daily Marginal Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "marginal_return_histogram.png", dpi=180)
    plt.close()

    rolling_window = 90
    rolling_mean = marginal.rolling(rolling_window).mean()
    rolling_std = marginal.rolling(rolling_window).std(ddof=1)
    rolling_ir = (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS)

    plt.figure(figsize=(11, 5))
    plt.plot(df.index, rolling_ir, color="#27ae60", linewidth=1.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Rolling Information Ratio ({rolling_window}D) of Marginal Returns")
    plt.ylabel("Information Ratio")
    plt.xlabel("Date")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "rolling_information_ratio.png", dpi=180)
    plt.close()



def main() -> None:
    df = generate_series()
    weight = optimize_marginal_weight(df)
    combined = (1 - weight) * df["baseline"] + weight * df["candidate_model"]
    marginal = combined - df["baseline"]

    base_stats = performance_stats(df["baseline"])
    combo_stats = performance_stats(combined)

    t_stat, t_p_two_sided = stats.ttest_1samp(marginal, popmean=0.0, alternative="greater")
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(marginal)
    ci_low, ci_high = bootstrap_ci(marginal.to_numpy())

    summary = {
        "optimal_candidate_weight": weight,
        "baseline": asdict(base_stats),
        "combined": asdict(combo_stats),
        "uplift": {
            "annual_return_delta": combo_stats.annual_return - base_stats.annual_return,
            "sharpe_delta": combo_stats.sharpe - base_stats.sharpe,
            "max_drawdown_delta": combo_stats.max_drawdown - base_stats.max_drawdown,
        },
        "stat_tests": {
            "one_sample_ttest_greater": {"t_stat": float(t_stat), "p_value": float(t_p_two_sided)},
            "wilcoxon_signed_rank": {"stat": float(wilcoxon_stat), "p_value": float(wilcoxon_p)},
            "bootstrap_mean_ci_95": {"low": ci_low, "high": ci_high},
        },
        "resource_count": {
            "charts_generated": 3,
            "json_tables": 1,
            "csv_rows": len(df),
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out["combined"] = combined
    df_out["marginal"] = marginal
    df_out.to_csv(ARTIFACT_DIR / "returns_timeseries.csv", index=True)

    with open(ARTIFACT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plots(df, combined, marginal)

    print("Research 02 marginal analysis complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
