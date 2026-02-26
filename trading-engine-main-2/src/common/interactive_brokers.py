from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import polars as pl
from dataclasses import is_dataclass, fields

from ib_async import IB, Order, Stock, FlexReport

@dataclass
class AccountData:
    nav: float
    positions: dict


class IBKR:
    def __init__(self):
        self.ib: Optional[IB] = None
        self.account = AccountData(-1, {})

    @classmethod
    async def create(
        cls, 
        hostname: str, 
        port: int, 
        flex_web_token: int, 
        nav_flex_query_id: int, 
        fund_inception_date: datetime.date, 
        client_id: int = 1
    ):
        self = cls()
        self.ib = IB()
        self.flex_web_token = flex_web_token
        self.nav_flex_query_id = nav_flex_query_id
        self.fund_inception_date = fund_inception_date

        await self.ib.connectAsync(hostname, port, clientId=client_id)
        await self.__fetch_initial_data()
        return self

    async def __fetch_initial_data(self):
        await self.__fetch_nav()
        await self.__fetch_positions()

    async def __fetch_nav(self):
        for v in self.ib.accountValues():
            if v.tag == "NetLiquidation":
                self.account.nav = float(v.value)
                break

    async def __fetch_positions(self):
        for p in self.ib.positions():
            self.account.positions[p.contract.symbol] = p

    def __objs_to_polars_df(self, objs, labels=None):
        # see https://github.com/ib-api-reloaded/ib_async/blob/2afdae44a47f067bf3d1a23a4840234f2906ef64/ib_async/util.py#L45
        # for context. (this is a polars port of the equivalent util from ib_async)
        if objs:
            objs = list(objs)
            obj = objs[0]

            if is_dataclass(obj):
                df = pl.DataFrame(
                    [tuple(getattr(obj, field.name) for field in fields(obj)) for o in objs],
                    schema=[field.name for field in fields(obj)],
                )
            else:
                df = pl.DataFrame([o.__dict__ for o in objs])

            if isinstance(obj, tuple):
                _fields = getattr(obj, "_fields", None)
                if _fields:
                    # assume it's a namedtuple
                    df = df.rename(dict(zip(df.columns, _fields)))
        else:
            df = None

        if df is not None and labels:
            exclude = [label for label in df.columns if label not in labels]
            df = df.drop(exclude)

        return df

    def get_nav(self) -> float:
        return self.account.nav

    def get_position(self, ticker: str):
        sym = ticker.split("-")[0]
        return self.account.positions.get(sym, 0)

    async def get_latest_positions(self):
        return self.ib.positions()

    async def get_open_trades(self):
        return self.ib.openTrades()  # local cache; no await

    async def get_contract(self, ticker: str):
        sym = ticker.split("-")[0]
        c = Stock(sym, "SMART", "USD")
        await self.ib.qualifyContractsAsync(c)
        return c
    
    def get_historical_nav(self):
        """
        Returns a polars dataframe with date column and nav column.
        """
        flex_report = FlexReport(self.flex_web_token, self.nav_flex_query_id)
        report = flex_report.extract("EquitySummaryByReportDateInBase")
        df = self.__objs_to_polars_df(report)
        df = df.with_columns(
            pl.col("reportDate").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date"),
            pl.col("total").alias("nav")
        ).filter(pl.col("date") >= self.fund_inception_date)
        return df["date", "nav"]

    def generate_order_ref(self, ticker: str) -> str:
        return f"{datetime.now().strftime('%Y%m%d')}.{ticker}"

    def construct_order(self, ticker: str, qty: int):
        if qty == 0:
            return None
        o = Order()
        o.action = "BUY" if qty > 0 else "SELL"
        o.totalQuantity = abs(qty)
        o.orderType = "MOC"  # ensure venue supports MOC; SMART often doesn’t
        o.orderRef = self.generate_order_ref(ticker)
        return o

    def cleanup(self):
        self.ib.disconnect()
