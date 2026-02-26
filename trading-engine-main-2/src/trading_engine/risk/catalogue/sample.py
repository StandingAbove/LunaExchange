from typing import Callable

import numpy as np


def SampleCovariance() -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable that computes the sample covariance of window returns.

    :return: function(window_returns) -> cov
    """

    def run(window_returns: np.ndarray) -> np.ndarray:
        return np.cov(window_returns, rowvar=False)

    return run


def SampleCovarianceWithRidge(
    ridge: float = 1e-2,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable that computes sample covariance with ridge ε on the diagonal.

    :param ridge: Non-negative scalar ε to add to the diagonal
    :return: function(window_returns) -> cov + ε I
    """

    def run(window_returns: np.ndarray) -> np.ndarray:
        cov = np.cov(window_returns, rowvar=False)
        if ridge > 0.0:
            n = cov.shape[0]
            cov = cov + (ridge * np.eye(n))
        return cov

    return run
