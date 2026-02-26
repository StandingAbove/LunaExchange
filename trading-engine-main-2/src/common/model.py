import dataclasses
import datetime
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class IBGatewayConfig:
    host: str
    port: int
    client_id: int
    flex_web_token: int
    nav_flex_query_id: int
    fund_inception_date: datetime.date

    def __post_init__(self):
        if os.environ.get("IB_HOST"):
            self.host = os.environ["IB_HOST"]
        if os.environ.get("IB_PORT"):
            self.port = int(os.environ["IB_PORT"])


@dataclass
class Config:
    start_date: datetime.date
    end_date: datetime.date
    universe: list[str]
    model_state_features: list[str]
    models: list[str]
    aggregators: list[str]
    ib_gateway: IBGatewayConfig
    notifications: dict
    optimizers: Optional[list[str]] = None
    cash_buffer: float = 0.0

    def __post_init__(self):
        if isinstance(self.ib_gateway, dict):
            self.ib_gateway = IBGatewayConfig(**self.ib_gateway)
        elif not isinstance(self.ib_gateway, IBGatewayConfig):
            raise TypeError("ib_gateway must be IBGatewayConfig or dict")

    def dump_to_gcs(self, gcs_url: str):
        """
        Serialize the Config instance to a JSON string and save it to a GCS URL.
        """
        import json
        from gcsfs import GCSFileSystem

        fs = GCSFileSystem()
        json_string = json.dumps(dataclasses.asdict(self), default=str, indent=4)

        with fs.open(gcs_url, "w") as f:
            f.write(json_string)
