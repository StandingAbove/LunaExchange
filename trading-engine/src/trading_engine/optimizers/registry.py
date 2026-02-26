from trading_engine.optimizers.catalogue.mean_variance import MeanVarianceOptimizer
from trading_engine.optimizers.catalogue.miqp_mean_variance import (
    MIQPMeanVarianceOptimizer,
)
from trading_engine.risk.registry import RISK_MODELS

OPTIMIZERS = {
    "mean_variance": {
        "function": MeanVarianceOptimizer(
            cov_window_days=240,
            gamma=1.0,
            lambda_te=0.5,
            risk_model=RISK_MODELS["naive_dcc"]["function"],
            kappa=1.0,
            turnover_lambda=0.1,  # Moderate turnover penalty to reduce transaction costs
        ),
        "lookback": 240,
    },
    "mean_variance_constrained": {
        "function": MeanVarianceOptimizer(
            cov_window_days=240,
            gamma=1.0,
            lambda_te=1.0,
            risk_model=RISK_MODELS["naive_dcc"]["function"],
            kappa=1.0,
            min_position_delta=0.00,  # Turned off this constraint.
            turnover_lambda=1.0,  # Turnover penalty to reduce transaction costs
            asset_weight_bounds={
                # US Equity - Allow concentrated positions
                "SPY-US": {"min": -0.5, "max": 0.5},
                # Fixed Income - Conservative bounds
                "TLT-US": {"min": -0.3, "max": 0.5},
                "IEI-US": {"min": -0.2, "max": 0.5},
                "SHY-US": {"min": -0.2, "max": 0.5},
                "BIL-US": {"min": -0.2, "max": 0.5},
                # International Equity - Moderate
                "EWJ-US": {"min": -0.2, "max": 0.3},
                "INDA-US": {"min": -0.2, "max": 0.3},
                "MCHI-US": {"min": -0.2, "max": 0.3},
                "EZU-US": {"min": -0.2, "max": 0.3},
                # Precious Metals
                "GLD-US": {"min": -0.2, "max": 0.3},
                "SLV-US": {"min": -0.15, "max": 0.25},
                # Commodities - Tighter bounds due to volatility
                "USO-US": {"min": -0.15, "max": 0.2},
                "UNG-US": {"min": -0.1, "max": 0.15},
                # Volatility - Very tight
                "VIXY-US": {"min": -0.1, "max": 0.1},
                # Crypto - Emerging
                "IBIT-US": {"min": -0.15, "max": 0.2},
                "ETHA-US": {"min": -0.15, "max": 0.2},
            },
        ),
        "lookback": 240,
    },
    "miqp_mean_variance": {
        "function": MIQPMeanVarianceOptimizer(
            cov_window_days=240,
            gamma=1.0,
            lambda_te=0.5,
            risk_model=RISK_MODELS["naive_dcc"]["function"],
            kappa=1.0,
            # 3% minimum trade size (enforced exactly via MIQP)
            min_position_delta=0.03,
            turnover_lambda=0.1,  # Turnover penalty to reduce transaction costs
            big_m=10.0,  # Big-M constant for MIQP formulation
            asset_weight_bounds={
                # Same bounds as constrained version for fair comparison
                "SPY-US": {"min": -0.5, "max": 0.5},
                "TLT-US": {"min": -0.3, "max": 0.5},
                "IEI-US": {"min": -0.2, "max": 0.5},
                "SHY-US": {"min": -0.2, "max": 0.5},
                "BIL-US": {"min": -0.2, "max": 0.5},
                "EWJ-US": {"min": -0.2, "max": 0.3},
                "INDA-US": {"min": -0.2, "max": 0.3},
                "MCHI-US": {"min": -0.2, "max": 0.3},
                "EZU-US": {"min": -0.2, "max": 0.3},
                "GLD-US": {"min": -0.2, "max": 0.3},
                "SLV-US": {"min": -0.15, "max": 0.25},
                "USO-US": {"min": -0.15, "max": 0.2},
                "UNG-US": {"min": -0.1, "max": 0.15},
                "VIXY-US": {"min": -0.1, "max": 0.1},
                "IBIT-US": {"min": -0.15, "max": 0.2},
                "ETHA-US": {"min": -0.15, "max": 0.2},
            },
        ),
        "lookback": 240,
    },
}
