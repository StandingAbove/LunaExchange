import numpy as np
import pandas as pd

from Backtest.metrics import sharpe_ratio


DAYS_PER_YEAR = 365


# =========================================================
# 1. IID Bootstrap
# =========================================================

def iid_bootstrap_returns(
    returns: pd.Series,
    n_samples: int = 1000,
) -> np.ndarray:
    """
    Resample daily returns with replacement.
    Returns array of Sharpe ratios.
    """

    r = returns.dropna().values
    n = len(r)

    sharpes = []

    for _ in range(n_samples):
        sample = np.random.choice(r, size=n, replace=True)
        sharpes.append(sharpe_ratio(pd.Series(sample)))

    return np.array(sharpes)


# =========================================================
# 2. Block Bootstrap
# =========================================================

def block_bootstrap_returns(
    returns: pd.Series,
    block_size: int = 20,
    n_samples: int = 1000,
) -> np.ndarray:
    """
    Block bootstrap preserves short-term autocorrelation.
    """

    r = returns.dropna().values
    n = len(r)

    sharpes = []

    for _ in range(n_samples):
        sample = []

        while len(sample) < n:
            start = np.random.randint(0, n - block_size)
            block = r[start:start + block_size]
            sample.extend(block)

        sample = np.array(sample[:n])
        sharpes.append(sharpe_ratio(pd.Series(sample)))

    return np.array(sharpes)


# =========================================================
# 3. Return Shuffling Test
# =========================================================

def shuffled_sharpe_distribution(
    returns: pd.Series,
    n_samples: int = 1000,
) -> np.ndarray:
    """
    Shuffle returns to destroy time structure.
    """

    r = returns.dropna().values
    sharpes = []

    for _ in range(n_samples):
        shuffled = np.random.permutation(r)
        sharpes.append(sharpe_ratio(pd.Series(shuffled)))

    return np.array(sharpes)


# =========================================================
# 4. Noise Injection Test
# =========================================================

def noise_injection_test(
    returns: pd.Series,
    noise_std: float = 0.001,
    n_samples: int = 500,
) -> np.ndarray:
    """
    Add Gaussian noise to returns.
    """

    r = returns.dropna().values
    sharpes = []

    for _ in range(n_samples):
        noise = np.random.normal(0, noise_std, size=len(r))
        noisy = r + noise
        sharpes.append(sharpe_ratio(pd.Series(noisy)))

    return np.array(sharpes)


# =========================================================
# 5. Probability of Overfitting Estimate
# =========================================================

def probability_of_overfitting(
    actual_sharpe: float,
    simulated_sharpes: np.ndarray,
) -> float:
    """
    Probability that simulated Sharpe >= actual Sharpe.
    """

    if len(simulated_sharpes) == 0:
        return np.nan

    count = np.sum(simulated_sharpes >= actual_sharpe)
    return float(count / len(simulated_sharpes))


# =========================================================
# 6. Full Monte Carlo Report
# =========================================================

def monte_carlo_report(
    returns: pd.Series,
    n_samples: int = 1000,
    block_size: int = 20,
) -> dict:
    """
    Run multiple robustness tests and summarize.
    """

    actual = sharpe_ratio(returns)

    iid_dist = iid_bootstrap_returns(returns, n_samples)
    block_dist = block_bootstrap_returns(returns, block_size, n_samples)
    shuffle_dist = shuffled_sharpe_distribution(returns, n_samples)

    return {
        "ActualSharpe": float(actual),
        "IID_MeanSharpe": float(np.nanmean(iid_dist)),
        "Block_MeanSharpe": float(np.nanmean(block_dist)),
        "Shuffle_MeanSharpe": float(np.nanmean(shuffle_dist)),
        "ProbOverfit_IID": probability_of_overfitting(actual, iid_dist),
        "ProbOverfit_Block": probability_of_overfitting(actual, block_dist),
        "ProbOverfit_Shuffle": probability_of_overfitting(actual, shuffle_dist),
    }