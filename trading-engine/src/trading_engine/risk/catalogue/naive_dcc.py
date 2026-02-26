from typing import Callable

import numpy as np


def NaiveDCC(
    half_life_vol: float = 11.6,
    half_life_corr: float = 23.4,
    ridge: float = 1e-3,
    center: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable that estimates covariance using a naïve DCC-style approach.

    The estimator standardizes returns by per-asset EWMA volatility, smooths the
    outer-products of standardized returns with an EWMA to form a correlation
    estimate, and rescales by the most recent EWMA volatilities.

    :param half_life_vol: Half-life (in periods) for EWMA volatility estimation.
                         After this many periods, a volatility shock retains 50%
                         of its impact. Must be > 0.
    :param half_life_corr: Half-life (in periods) for EWMA correlation estimation.
                          After this many periods, a correlation shock retains 50%
                          of its impact. Must be > 0.
    :param ridge: Non-negative scalar ε to add to the diagonal of the result
    :param center: If True, subtract per-asset mean before volatility estimation
    :return: function(window_returns) -> covariance matrix (N x N)
    """

    if half_life_vol <= 0.0:  # pragma: no cover - defensive guard
        raise ValueError("half_life_vol must be positive")
    if half_life_corr <= 0.0:  # pragma: no cover - defensive guard
        raise ValueError("half_life_corr must be positive")
    if ridge < 0.0:  # pragma: no cover - defensive guard
        raise ValueError("ridge must be non-negative")

    # Convert half-lives to beta parameters for internal EWMA calculations
    beta_vol: float = 0.5 ** (1.0 / half_life_vol)
    beta_corr: float = 0.5 ** (1.0 / half_life_corr)

    one_minus_beta_vol: float = 1.0 - beta_vol
    one_minus_beta_corr: float = 1.0 - beta_corr
    numerical_epsilon: float = 1e-10

    def run(window_returns: np.ndarray) -> np.ndarray:
        """
        Estimate an asset covariance matrix from a returns window.

        :param window_returns: Array of shape (T, N) with asset returns
        :return: Covariance estimate of shape (N, N)
        """
        returns: np.ndarray = np.asarray(window_returns, dtype=float)
        if returns.ndim != 2:
            raise ValueError("window_returns must be a 2D array of shape (T, N)")

        num_periods, num_assets = returns.shape
        if num_assets == 0:
            return np.zeros((0, 0), dtype=float)

        if center:
            returns = returns - returns.mean(axis=0, keepdims=True)

        # EWMA volatility per asset across time
        ewma_variance: np.ndarray = np.empty((num_periods, num_assets), dtype=float)
        if num_periods > 1:
            initial_variance: np.ndarray = returns.var(axis=0, ddof=1)
        else:
            initial_variance = returns.var(axis=0, ddof=0)
        initial_variance = np.maximum(initial_variance, numerical_epsilon)
        ewma_variance[0, :] = initial_variance

        for t in range(1, num_periods):
            previous_squared_returns: np.ndarray = returns[t - 1, :] ** 2
            ewma_variance[t, :] = (
                beta_vol * ewma_variance[t - 1, :]
                + one_minus_beta_vol * previous_squared_returns
            )

        per_time_volatility: np.ndarray = np.sqrt(ewma_variance)
        latest_volatility: np.ndarray = np.maximum(
            per_time_volatility[-1, :], np.sqrt(numerical_epsilon)
        )

        # Standardize by time-varying volatility to form z_t
        safe_denominator: np.ndarray = np.maximum(
            per_time_volatility, np.sqrt(numerical_epsilon)
        )
        standardized_returns: np.ndarray = returns / safe_denominator

        # EWMA of outer-products of standardized returns
        smoothed_outer_products: np.ndarray = np.zeros(
            (num_assets, num_assets), dtype=float
        )
        for t in range(num_periods):
            z_t: np.ndarray = standardized_returns[t, :]
            outer_product: np.ndarray = np.outer(z_t, z_t)
            smoothed_outer_products = (
                beta_corr * smoothed_outer_products
                + one_minus_beta_corr * outer_product
            )

        # Convert to correlation by normalizing with its diagonal
        diagonal_elements: np.ndarray = np.diag(smoothed_outer_products).copy()
        diagonal_elements = np.maximum(diagonal_elements, numerical_epsilon)
        scaling_matrix: np.ndarray = np.sqrt(
            np.outer(diagonal_elements, diagonal_elements)
        )
        correlation_matrix: np.ndarray = smoothed_outer_products / scaling_matrix
        np.fill_diagonal(correlation_matrix, 1.0)
        correlation_matrix = 0.5 * (correlation_matrix + correlation_matrix.T)

        # Rescale by the latest EWMA volatilities
        covariance_matrix: np.ndarray = correlation_matrix * np.outer(
            latest_volatility, latest_volatility
        )

        if ridge > 0.0:
            covariance_matrix = covariance_matrix + ridge * np.eye(num_assets)

        covariance_matrix = 0.5 * (covariance_matrix + covariance_matrix.T)
        return covariance_matrix

    return run
