from trading_engine.models.catalogue.amma import AMMA
from trading_engine.models.catalogue.inverse_momentum_mean_reversion import (
    InverseMomentumMeanReversionModel,
)
from trading_engine.models.catalogue.momentum import MomentumModel
from trading_engine.models.catalogue.natr_mean_reversion import NATRMeanReversionModel

MODELS = {
    "RXI_TLT_pml_10": {
        "tickers": ["RXI-US", "TLT-US"],
        "columns": ["close_momentum_10"],
        "function": MomentumModel(
            trade_ticker="RXI-US",
            signal_ticker="TLT-US",
            momentum_column="close_momentum_10",
            inverse=False,
        ),
        "lookback": 0,
    },
    "GLD_USO_nml_10": {
        "tickers": ["GLD-US", "USO-US"],
        "columns": ["close_momentum_10"],
        "function": MomentumModel(
            trade_ticker="GLD-US",
            signal_ticker="USO-US",
            momentum_column="close_momentum_10",
            inverse=True,
        ),
        "lookback": 0,
    },
    "IXJ_USO_pml_10": {
        "tickers": ["IXJ-US", "USO-US"],
        "columns": ["close_momentum_10"],
        "function": MomentumModel(
            trade_ticker="IXJ-US",
            signal_ticker="USO-US",
            momentum_column="close_momentum_10",
            inverse=False,
        ),
        "lookback": 0,
    },
    "TLT_IXC_nml_10": {
        "tickers": ["TLT-US", "IXC-US"],
        "columns": ["close_momentum_10"],
        "function": MomentumModel(
            trade_ticker="TLT-US",
            signal_ticker="IXC-US",
            momentum_column="close_momentum_10",
            inverse=True,
        ),
        "lookback": 0,
    },
    "IXJ_KXI_pml_10": {
        "tickers": ["IXJ-US", "KXI-US"],
        "columns": ["close_momentum_10"],
        "function": MomentumModel(
            trade_ticker="IXJ-US",
            signal_ticker="KXI-US",
            momentum_column="close_momentum_10",
            inverse=False,
        ),
        "lookback": 0,
    },
    "etf_mr_gld_10_0.1": {
        "tickers": ["GLD-US"],
        "columns": ["close_momentum_10"],
        "function": InverseMomentumMeanReversionModel(
            tickers=["GLD-US"], momentum_column="close_momentum_10", threshold=0.1
        ),
        "lookback": 0,
    },
    "etf_mr_gld_60_0.1": {
        "tickers": ["GLD-US"],
        "columns": ["close_momentum_60"],
        "function": InverseMomentumMeanReversionModel(
            tickers=["GLD-US"], momentum_column="close_momentum_60", threshold=0.1
        ),
        "lookback": 0,
    },
    "etf_mr_ixn_10_0.1": {
        "tickers": ["IXN-US"],
        "columns": ["close_momentum_10"],
        "function": InverseMomentumMeanReversionModel(
            tickers=["IXN-US"], momentum_column="close_momentum_10", threshold=0.1
        ),
        "lookback": 0,
    },
    "etf_mr_jxi_60_0.001": {
        "tickers": ["JXI-US"],
        "columns": ["close_momentum_60"],
        "function": InverseMomentumMeanReversionModel(
            tickers=["JXI-US"], momentum_column="close_momentum_60", threshold=0.001
        ),
        "lookback": 0,
    },
    "etf_mr_jxi_240_0.05": {
        "tickers": ["JXI-US"],
        "columns": ["close_momentum_240"],
        "function": InverseMomentumMeanReversionModel(
            tickers=["JXI-US"], momentum_column="close_momentum_240", threshold=0.05
        ),
        "lookback": 0,
    },
    "etf_mr_jxi_240_0.01": {
        "tickers": ["JXI-US"],
        "columns": ["close_momentum_240"],
        "function": InverseMomentumMeanReversionModel(
            tickers=["JXI-US"], momentum_column="close_momentum_240", threshold=0.01
        ),
        "lookback": 0,
    },
    "etf_mr_jxi_240_0.001": {
        "tickers": ["JXI-US"],
        "columns": ["close_momentum_240"],
        "function": InverseMomentumMeanReversionModel(
            tickers=["JXI-US"], momentum_column="close_momentum_240", threshold=0.001
        ),
        "lookback": 0,
    },
    "etf_mr_jxi_240_0.005": {
        "tickers": ["JXI-US"],
        "columns": ["close_momentum_240"],
        "function": InverseMomentumMeanReversionModel(
            tickers=["JXI-US"], momentum_column="close_momentum_240", threshold=0.005
        ),
        "lookback": 0,
    },
    "natr_7_14": {
        "tickers": ["IXN-US"],
        "columns": [
            "adjusted_close_1d",
            "natr_7",
            "natr_14",
            "close_momentum_1",
            "close_momentum_14",
            "close_momentum_32",
            "close_momentum_64",
        ],
        "function": NATRMeanReversionModel(trade_ticker="IXN-US"),
        "lookback": 0,
    },
    "TLT_AMMA": {
        "tickers": ["TLT-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="TLT-US",
            momentum_weights={
                10: 0.145,
                20: 0.221,
                30: 0,
                60: 0.252,
                90: 0,
                120: 0.195,
                240: 0.187,
            },
        ),
        "lookback": 0,
    },
    "IEI_AMMA": {
        "tickers": ["IEI-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="IEI-US",
            momentum_weights={
                10: 0.14,
                20: 0.176,
                30: 0,
                60: 0.222,
                90: 0,
                120: 0.236,
                240: 0.226,
            },
        ),
        "lookback": 0,
    },
    "SHY_AMMA": {
        "tickers": ["SHY-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="SHY-US",
            momentum_weights={
                10: 0,
                20: 0.1527,
                30: 0,
                60: 0,
                90: 0,
                120: 0.2304,
                240: 0.6169,
            },
        ),
        "lookback": 0,
    },
    "BIL_AMMA": {
        "tickers": ["BIL-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="BIL-US",
            momentum_weights={
                10: 0,
                20: 0.1,
                30: 0,
                60: 0.2,
                90: 0,
                120: 0.3,
                240: 0.4,
            },
        ),
        "lookback": 0,
    },
    "SLV_AMMA": {
        "tickers": ["SLV-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="SLV-US",
            momentum_weights={
                10: 0.33,
                20: 0,
                30: 0.33,
                60: 0,
                90: 0,
                120: 0,
                240: 0.34,
            },
        ),
        "lookback": 0,
    },
    "GLD_AMMA": {
        "tickers": ["GLD-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="GLD-US",
            momentum_weights={
                10: 0.15,
                20: 0,
                30: 0,
                60: 0.2,
                90: 0.35,
                120: 0.4,
                240: 0,
            },
        ),
        "lookback": 0,
    },
    "USO_AMMA": {
        "tickers": ["USO-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="USO-US",
            momentum_weights={10: 0, 20: 0.61, 30: 0, 60: 0.39, 90: 0, 120: 0, 240: 0},
        ),
        "lookback": 0,
    },
    "UNG_AMMA": {
        "tickers": ["UNG-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="UNG-US",
            momentum_weights={
                10: 0,
                20: 0.15,
                30: 0,
                60: 0,
                90: 0,
                120: 0.57,
                240: 0.28,
            },
        ),
        "lookback": 0,
    },
    "SPY_AMMA": {
        "tickers": ["SPY-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="SPY-US",
            momentum_weights={
                10: 0,
                20: 0,
                30: 0,
                60: 0.98888462,
                90: 0,
                120: 0,
                240: 0.01111538,
            },
        ),
        "lookback": 0,
    },
    "EWJ_AMMA": {
        "tickers": ["EWJ-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="EWJ-US",
            momentum_weights={
                10: 0,
                20: 0.6906,
                30: 0,
                60: 0.1527,
                90: 0,
                120: 0,
                240: 0.1567,
            },
        ),
        "lookback": 0,
    },
    "INDA_AMMA": {
        "tickers": ["INDA-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="INDA-US",
            momentum_weights={
                10: 0.45,
                20: 0.45,
                30: 0,
                60: 0,
                90: 0,
                120: 0,
                240: 0.1,
            },
        ),
        "lookback": 0,
    },
    "MCHI_AMMA": {
        "tickers": ["MCHI-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="MCHI-US",
            momentum_weights={
                10: 0,
                20: 0.35,
                30: 0,
                60: 0.35,
                90: 0,
                120: 0.05,
                240: 0.25,
            },
        ),
        "lookback": 0,
    },
    "EZU_AMMA": {
        "tickers": ["EZU-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="EZU-US",
            momentum_weights={
                10: 0,
                20: 0.0332,
                30: 0,
                60: 0,
                90: 0,
                120: 0.9668,
                240: 0,
            },
        ),
        "lookback": 0,
    },
    "VIXY_AMMA": {
        "tickers": ["VIXY-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="VIXY-US",
            momentum_weights={10: 0, 20: 0, 30: 0, 60: 1, 90: 0, 120: 0, 240: 0},
        ),
        "lookback": 0,
    },
    "IBIT_AMMA": {
        "tickers": ["IBIT-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="IBIT-US",
            momentum_weights={
                10: 0.045,
                20: 0.271,
                30: 0,
                60: 0.232,
                90: 0,
                120: 0.406,
                240: 0.045,
            },
        ),
        "lookback": 0,
    },
    "ETHA_AMMA": {
        "tickers": ["ETHA-US"],
        "columns": [
            "close_momentum_10",
            "close_momentum_20",
            "close_momentum_30",
            "close_momentum_60",
            "close_momentum_90",
            "close_momentum_120",
            "close_momentum_240",
        ],
        "function": AMMA(
            ticker="ETHA-US",
            momentum_weights={
                10: 0.062,
                20: 0.75,
                30: 0,
                60: 0.062,
                90: 0,
                120: 0.062,
                240: 0.062,
            },
        ),
        "lookback": 0,
    },
}
