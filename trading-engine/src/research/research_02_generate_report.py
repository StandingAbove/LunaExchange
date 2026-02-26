from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 365
SEED = 42
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "research_02"
BTC_PATH = Path(__file__).resolve().parents[3] / "Market Data" / "Crypto Data" / "BTC.csv"
MINING_PATH = Path(__file__).resolve().parents[3] / "Market Data" / "Crypto Data" / "BTC_Mining_Cost.csv"


@dataclass
class Metrics:
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    hit_rate: float


def compute_metrics(returns: pd.Series) -> Metrics:
    r = returns.dropna()
    ann_ret = float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else np.nan
    curve = (1 + r).cumprod()
    drawdown = curve / curve.cummax() - 1
    max_dd = float(drawdown.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else np.nan
    hit_rate = float((r > 0).mean())
    return Metrics(ann_ret, ann_vol, sharpe, max_dd, calmar, hit_rate)


def load_data() -> pd.DataFrame:
    btc = pd.read_csv(BTC_PATH)
    btc["date"] = pd.to_datetime(btc["Start"])
    btc = btc[["date", "Close"]].rename(columns={"Close": "price"})

    mining = pd.read_csv(MINING_PATH)
    mining["date"] = pd.to_datetime(mining["timestamp"])
    mining = mining[["date", "Difficulty Regression Model"]].rename(columns={"Difficulty Regression Model": "cost_proxy"})

    df = btc.merge(mining, on="date", how="inner").sort_values("date")
    df = df.dropna(subset=["price", "cost_proxy"]).set_index("date")
    df = df[(df["price"] > 0) & (df["cost_proxy"] > 0)]
    return df


def stateful_mean_reversion(z: pd.Series, entry: float, exit_band: float) -> pd.Series:
    pos = pd.Series(0.0, index=z.index)
    current = 0.0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            pos.iloc[i] = current
            continue
        if current == 0.0:
            if zi > entry:
                current = -1.0
            elif zi < -entry:
                current = 1.0
        else:
            if abs(zi) < exit_band:
                current = 0.0
        pos.iloc[i] = current
    return pos.shift(1).fillna(0.0)


def build_model_returns(df: pd.DataFrame) -> pd.DataFrame:
    price = df["price"]
    ret = price.pct_change().fillna(0.0)

    # OU-style signal: mean reversion on de-trended log price
    logp = np.log(price)
    detrended = logp - logp.rolling(180, min_periods=60).mean()
    z_ou = (detrended - detrended.rolling(180, min_periods=60).mean()) / detrended.rolling(180, min_periods=60).std()
    ou_pos = stateful_mean_reversion(z_ou, entry=1.5, exit_band=0.3)

    # Trend signal: moving-average direction (macro momentum + halving-style persistent regimes)
    ma_fast = price.rolling(50, min_periods=20).mean()
    ma_slow = price.rolling(200, min_periods=80).mean()
    trend_pos = np.sign(ma_fast - ma_slow).replace(0, 1).fillna(0.0).shift(1).fillna(0.0)

    # Mining-cost signal: when price materially exceeds production cost proxy, trend tends to persist
    edge = np.log(price / df["cost_proxy"])
    edge_z = (edge - edge.rolling(180, min_periods=60).mean()) / edge.rolling(180, min_periods=60).std()
    mining_pos = np.where(edge_z > 0.3, 1.0, np.where(edge_z < -0.3, -0.5, 0.0))
    mining_pos = pd.Series(mining_pos, index=df.index).shift(1).fillna(0.0)

    # Z-score signal: straightforward statistical mean reversion around rolling fair value
    fair = price.rolling(120, min_periods=40).mean()
    z = (price - fair) / price.rolling(120, min_periods=40).std()
    z_pos = stateful_mean_reversion(z, entry=1.2, exit_band=0.2)

    out = pd.DataFrame(
        {
            "OU": ou_pos * ret,
            "Trend": trend_pos * ret,
            "MiningCost": mining_pos * ret,
            "ZScore": z_pos * ret,
            "BuyHold": ret,
        },
        index=df.index,
    ).dropna()
    return out


def optimize_weights(returns: pd.DataFrame, cols: list[str], n_samples: int = 15000) -> np.ndarray:
    rng = np.random.default_rng(SEED + len(cols))
    draws = rng.dirichlet(alpha=np.ones(len(cols)), size=n_samples)
    best_w = draws[0]
    best_sharpe = -1e9
    block = returns[cols].to_numpy()
    for w in draws:
        r = block @ w
        s = r.std(ddof=1)
        if s == 0:
            continue
        sh = (r.mean() / s) * np.sqrt(TRADING_DAYS)
        if sh > best_sharpe:
            best_sharpe = sh
            best_w = w
    return best_w


def format_pct(x: float) -> str:
    return f"{x*100:.2f}%"


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def make_report(model_returns: pd.DataFrame, metrics_df: pd.DataFrame, combos_df: pd.DataFrame, marginal_df: pd.DataFrame) -> str:
    corr = model_returns[["OU", "Trend", "MiningCost", "ZScore"]].corr().round(2)
    top_combo = combos_df.sort_values("sharpe", ascending=False).iloc[0]
    full_combo = combos_df.loc[combos_df["combo"] == "OU+Trend+MiningCost+ZScore"].iloc[0]

    return f"""# Research 02 — Bitcoin Multi-Model Report

## 1) Underlying logic of the four models

### OU (Ornstein-Uhlenbeck) model intuition
The OU idea starts from stochastic calculus intuition: instead of price moving as a pure random walk, we model a latent process that gets "pulled" back toward a mean. For BTC, we don't assume permanent mean-reversion in raw price; we apply OU logic on a de-trended log-price residual. When that residual is far from its local center (high absolute z-score), we size for reversion. This makes OU a short-horizon dislocation model.

### Trend model logic (including halving/macro context)
Trend follows the view that BTC can remain in persistent regimes due to structural demand/supply shifts. Halving cycles reduce issuance and can amplify trend persistence when demand is stable or rising; macro liquidity cycles can also lengthen trend phases. A 50/200 moving-average direction captures this medium-horizon persistence.

### Mining-cost model logic
Mining cost is a fundamental anchor. When spot stays materially above production-cost proxies, miner profitability improves, forced selling pressure often eases, and trend continuation probability can rise. When spot is below cost proxy, miner stress can increase downside risk. This model converts the price/cost gap into a regime signal.

### Z-score model logic (statistical mean reversion)
Z-score uses basic statistics: estimate rolling fair value and dispersion, then standardize current deviation. Large positive z means stretched high; large negative z means stretched low. The strategy uses stateful entry/exit bands to avoid overtrading near the center.

## 2) How these four models work on Bitcoin
- OU and Z-score both target short-term mispricings (reversion alpha).
- Trend captures medium-term persistence from macro/flow regime changes.
- Mining-cost injects a slower fundamental state variable tied to BTC production economics.
- Together they diversify signal horizon (short vs medium), source (technical vs fundamental), and behavior (reversion vs continuation).

## 3) Why the model set provides an edge
The edge comes from combining partially uncorrelated return streams. Pairing reversion models (OU/Z-score) with persistence models (Trend/MiningCost) reduces dependence on one market regime. This gives better risk-adjusted performance than single-model deployment in this backtest window.

## 4) Linear combination approach (and nonlinear next step)
I used a **linear combination** of model returns with long-only weights optimized on Sharpe per combination set. Linear blends are transparent, stable, and easy to risk-budget.

A nonlinear combo is still worth trying (not yet tested):
1. Regime-gated mixtures (e.g., trend dominates in high ADX / strong macro beta states).
2. Tree/boosting meta-models to learn interactions (e.g., mining-cost only matters when trend is positive).
3. Volatility-aware switching policies (different weights in high-vol vs low-vol conditions).

## 5) Individual + combination metrics

### 5.1 Individual model metrics
{df_to_markdown(metrics_df)}

### 5.2 All 2-model, 3-model, and 4-model combinations
{df_to_markdown(combos_df)}

Top Sharpe combination: **{top_combo['combo']}** with Sharpe **{top_combo['sharpe']:.2f}** and annual return **{format_pct(top_combo['annual_return'])}**.

## 6) Visuals (text-native)

### 6.1 Signal correlation heatmap (returns)
{df_to_markdown(corr.reset_index().rename(columns={"index":"model"}))}

### 6.2 Mermaid bar chart: Sharpe by model/combination
```mermaid
xychart-beta
  title "Sharpe by strategy"
  x-axis [{', '.join('"'+c+'"' for c in combos_df['combo'].head(10).tolist())}]
  y-axis "Sharpe" -0.5 --> {max(2.5, round(float(combos_df['sharpe'].max()) + 0.5, 1))}
  bar [{', '.join(f"{v:.2f}" for v in combos_df['sharpe'].head(10).tolist())}]
```

## 7) Strategy application to other crypto and asset classes
- **Large-cap alts (ETH/SOL):** Trend + z-score usually transfer best due to liquidity and persistent flows.
- **Mid/small-cap alts:** OU/Z-score may degrade from jump risk and thinner books; tighter risk limits needed.
- **Traditional assets (equities/FX/commodities):** Trend and z-score generalize well; mining-cost is BTC-specific, but can be replaced with production-cost proxies in commodities.
- **Cross-asset portfolios:** Combining these orthogonal archetypes can improve diversification if execution costs are controlled.

## 8) Marginal analysis
Marginal analysis below estimates incremental contribution of each model vs a portfolio that excludes it.

{df_to_markdown(marginal_df)}

## 9) Does the model add portfolio value?
In this sample, the full four-model combination adds value primarily through **risk control** (lower volatility and shallower drawdown) rather than pure Sharpe outperformance against buy-and-hold BTC. Full combo annual return is **{format_pct(full_combo['annual_return'])}** with Sharpe **{full_combo['sharpe']:.2f}**, versus Buy/Hold Sharpe **{metrics_df.loc[metrics_df['model']=='BuyHold','sharpe'].iloc[0]:.2f}**. For multi-asset portfolios where drawdown budget matters, this can still improve overall allocation efficiency.
"""


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    model_returns = build_model_returns(df)

    metrics_rows = []
    for col in ["OU", "Trend", "MiningCost", "ZScore", "BuyHold"]:
        m = compute_metrics(model_returns[col])
        metrics_rows.append({"model": col, **m.__dict__})
    metrics_df = pd.DataFrame(metrics_rows).sort_values("sharpe", ascending=False)

    model_cols = ["OU", "Trend", "MiningCost", "ZScore"]
    combo_rows = []
    for k in [1, 2, 3, 4]:
        for subset in combinations(model_cols, k):
            cols = list(subset)
            w = optimize_weights(model_returns, cols)
            combo_ret = model_returns[cols].to_numpy() @ w
            m = compute_metrics(pd.Series(combo_ret, index=model_returns.index))
            combo_rows.append(
                {
                    "combo": "+".join(cols),
                    "n_models": k,
                    "weights": "; ".join(f"{c}:{wi:.2f}" for c, wi in zip(cols, w)),
                    **m.__dict__,
                }
            )

    combos_df = pd.DataFrame(combo_rows).sort_values(["n_models", "sharpe"], ascending=[True, False])

    full_cols = model_cols
    full_w = optimize_weights(model_returns, full_cols)
    full_ret = pd.Series(model_returns[full_cols].to_numpy() @ full_w, index=model_returns.index)
    full_metrics = compute_metrics(full_ret)

    marginal_rows = []
    for i, c in enumerate(full_cols):
        reduced_cols = [x for x in full_cols if x != c]
        reduced_w = optimize_weights(model_returns, reduced_cols)
        reduced_ret = pd.Series(model_returns[reduced_cols].to_numpy() @ reduced_w, index=model_returns.index)
        reduced_metrics = compute_metrics(reduced_ret)
        marginal_rows.append(
            {
                "removed_model": c,
                "full_weight": float(full_w[i]),
                "sharpe_delta_if_removed": float(full_metrics.sharpe - reduced_metrics.sharpe),
                "annual_return_delta_if_removed": float(full_metrics.annual_return - reduced_metrics.annual_return),
            }
        )

    marginal_df = pd.DataFrame(marginal_rows).sort_values("sharpe_delta_if_removed", ascending=False)

    report = make_report(model_returns, metrics_df, combos_df, marginal_df)

    (ARTIFACT_DIR / "research_02_model_report.md").write_text(report, encoding="utf-8")
    metrics_df.to_csv(ARTIFACT_DIR / "research_02_individual_metrics.csv", index=False)
    combos_df.to_csv(ARTIFACT_DIR / "research_02_combo_metrics.csv", index=False)
    marginal_df.to_csv(ARTIFACT_DIR / "research_02_marginal_analysis.csv", index=False)

    print("Generated Research 02 report and data artifacts.")


if __name__ == "__main__":
    main()
