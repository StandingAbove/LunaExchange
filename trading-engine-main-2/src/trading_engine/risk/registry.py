from trading_engine.risk.catalogue.sample import (
    SampleCovariance,
    SampleCovarianceWithRidge,
)
from trading_engine.risk.catalogue.naive_dcc import NaiveDCC

RISK_MODELS = {
    "sample": {"function": SampleCovariance(), "lookback": 0},
    "sample_with_ridge": {"function": SampleCovarianceWithRidge(ridge=1e-3), "lookback": 0},
    "naive_dcc": {
        "function": NaiveDCC(
            half_life_vol=21,
            half_life_corr=120,
            ridge=1e-3,
            center=True,
        ),
        "lookback": 0,
    },
}
