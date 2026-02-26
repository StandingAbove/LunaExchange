import math
import re
from typing import Iterable, Optional, Dict, Any

import polars as pl
from polars import DataFrame

from common.interactive_brokers import IBKR


def construct_goal_positions(
        ib_client: IBKR,
        insights: DataFrame,
        prices: DataFrame,
        universe: Optional[Iterable[str]] = None,
        cash_buffer_pct: float = 0.0,
) -> DataFrame:
    """
    Build per-ticker targets using latest insight weights and latest prices,
    accounting for a cash buffer and estimated transaction costs.

    Cost Model: max($1.00, $0.01 per share)

    Inputs:
      - cash_buffer_pct: Fraction of NAV (0–1) to explicitly reserve as cash.
                          Example: 0.05 -> keep 5% of NAV in cash.
    """
    if insights.is_empty() or prices.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "weight": pl.Float64,
                "price": pl.Float64,
                "target_value": pl.Float64,
                "target_shares": pl.Int64,
            }
        )

    def _latest_row(df: DataFrame) -> Dict[str, Any]:
        if "date" in df.columns:
            df = df.sort("date")
        return df.tail(1).to_dicts()[0]

    def _num(x: Any) -> bool:
        return isinstance(x, (int, float)) and not (
                isinstance(x, float) and math.isnan(x)
        )

    def _canon(sym: str) -> str:
        return sym.split("-")[0]

    latest_w = _latest_row(insights)
    latest_p = _latest_row(prices)

    weights = {k: float(v) for k, v in latest_w.items() if k != "date" and _num(v)}
    px = {
        k: float(v)
        for k, v in latest_p.items()
        if k != "date" and _num(v) and v > 0.0
    }

    tickers = sorted(set(weights) & set(px))

    if universe:
        u_full = set(universe)
        u_canon = {_canon(u) for u in universe}
        tickers = [t for t in tickers if t in u_full or _canon(t) in u_canon]

    # 1. Get Base NAV and apply Cash Buffer as a % of NAV
    raw_nav = float(ib_client.get_nav())
    buffer_frac = max(0.0, min(1.0, float(cash_buffer_pct)))  # clamp to [0, 1]
    investable_capital = raw_nav * (1.0 - buffer_frac)

    rows = []
    for t in tickers:
        w = weights[t]
        p = px[t]

        # 2. Allocate capital based on investable amount (post-buffer)
        gross_target_value = w * investable_capital

        # 3. Calculate Shares accounting for Transaction Costs
        # Formula: Shares * Price + Max(1.0, 0.01 * Shares) <= Gross_Allocated_Value
        # We solve for abs(Shares) then re-apply sign.
        abs_val = abs(gross_target_value)

        # Scenario A: Commission is likely > $1.00 (Variable cost dominates)
        # S * P + 0.01 * S <= Val  ->  S * (P + 0.01) <= Val
        shares_scenario_a = abs_val / (p + 0.01)

        # Scenario B: Commission is likely $1.00 (Min ticket cost dominates)
        # S * P + 1.0 <= Val
        shares_scenario_b = (abs_val - 1.0) / p

        # Determine which scenario applies based on the calculated shares
        if shares_scenario_a * 0.01 >= 1.0:
            raw_shares = shares_scenario_a
        else:
            # If variable cost < 1, we use the fixed cost scenario
            raw_shares = max(0.0, shares_scenario_b)

        # Floor to integer
        target_shares = int(raw_shares)

        # Apply correct sign (if shorting)
        if w < 0:
            target_shares = -target_shares

        # Recalculate actual target value (Shares * Price) for reporting
        final_value = target_shares * p

        rows.append(
            {
                "ticker": t,
                "weight": w,
                "price": p,
                "target_value": final_value,
                "target_shares": target_shares,
            }
        )

    print(rows)

    return pl.DataFrame(
        rows,
        schema={
            "ticker": pl.Utf8,
            "weight": pl.Float64,
            "price": pl.Float64,
            "target_value": pl.Float64,
            "target_shares": pl.Int64,
        },
    )


def construct_rebalance_orders(
        ib_client: IBKR,
        targets: DataFrame,  # output of construct_goal_positions()
        universe: Optional[Iterable[str]] = None,
        close_out_outside_universe: bool = True,  # if True, flatten any positions not in universe/targets
) -> DataFrame:
    """
    Compare current holdings vs target_shares and produce a rebalance plan (no order objects).

    Returns a Polars DataFrame:
      ["ticker","current_shares","target_shares","delta_shares","action","order_quantity"]

    Notes:
      - Negative shares are shorts and are preserved.
      - If `universe` is provided and `close_out_outside_universe=True`, any held name
        outside the universe/targets is targeted to 0 (i.e., closed).
    """
    required_cols = {"ticker", "target_shares"}
    missing = required_cols - set(targets.columns)
    if missing:
        raise ValueError(f"'targets' is missing columns: {sorted(missing)}")

    def _canon(sym: str) -> str:
        return sym.split("-")[0]

    # Map canonical -> full ticker and target shares
    target_rows = targets.select(["ticker", "target_shares"]).to_dicts()
    canonical_to_full: Dict[str, str] = {}
    target_shares_map: Dict[str, int] = {}
    for r in target_rows:
        full = r["ticker"]
        canon = _canon(full)
        canonical_to_full[canon] = full
        target_shares_map[canon] = int(r["target_shares"])

    # Current positions from IB
    current_positions_map: Dict[str, int] = {}
    for sym, pos in ib_client.account.positions.items():
        current_positions_map[sym] = int(pos.position)

    # Build canonical universe (if provided)
    if universe:
        u_canon = {_canon(u) for u in universe}
    else:
        u_canon = None

    # Determine the set to evaluate
    base_set = set(target_shares_map.keys())
    if u_canon:
        base_set |= u_canon  # ensure all universe names are considered (even if target 0)
    all_canonicals = base_set | set(current_positions_map.keys())

    # If we are NOT closing outside universe, drop anything not in (targets ∪ universe)
    if u_canon and not close_out_outside_universe:
        all_canonicals = {
            c for c in all_canonicals if (c in target_shares_map or c in u_canon)
        }

    rows = []
    for canon in sorted(all_canonicals):
        current = current_positions_map.get(canon, 0)
        target = target_shares_map.get(canon, 0)

        # If outside universe and not explicitly targeted, optionally close out to 0
        if u_canon and (canon not in u_canon) and (canon not in target_shares_map):
            target = 0 if close_out_outside_universe else current

        delta = target - current  # >0 BUY, <0 SELL (short more)

        full_ticker = canonical_to_full.get(canon, canon)
        rows.append(
            {
                "ticker": full_ticker,
                "current_shares": current,
                "target_shares": target,
                "delta_shares": delta,
                "action": ("BUY" if delta > 0 else ("SELL" if delta < 0 else "HOLD")),
                "order_quantity": abs(delta),
            }
        )

    df = pl.DataFrame(
        rows,
        schema={
            "ticker": pl.Utf8,
            "current_shares": pl.Int64,
            "target_shares": pl.Int64,
            "delta_shares": pl.Int64,
            "action": pl.Utf8,
            "order_quantity": pl.Int64,
        },
    ).filter(pl.col("delta_shares") != 0)

    return df


def to_ibkr_basket_csv(
        df: DataFrame,
        *,
        order_type: str = "MOC",
        time_in_force: str = "DAY",
        exchange: str = "SMART",
) -> str:
    """
    Convert rebalance orders to an IBKR BasketTrader CSV.

    Input df schema (from construct_rebalance_orders):
      ["ticker","current_shares","target_shares","delta_shares","action","order_quantity"]

    Output CSV columns (header order is flexible in TWS):
      Symbol,SecType,Currency,Exchange,Action,Quantity,OrderType,LmtPrice,AuxPrice,TimeInForce

    Notes:
      - ETFs go as SecType=STK, Currency=USD.
      - For MOC: LmtPrice and AuxPrice are blank.
    """
    header = [
        "Symbol",
        "SecType",
        "Currency",
        "Exchange",
        "Action",
        "Quantity",
        "OrderType",
        "LmtPrice",
        "AuxPrice",
        "TimeInForce",
    ]

    if df.is_empty():
        return ",".join(header) + "\n"

    def _canon(sym: str) -> str:
        # strip region suffixes like '-US' / '.US' if present
        s = sym.split("-")[0]
        s = re.sub(r"\.US$", "", s, flags=re.IGNORECASE)
        return s

    lines = [",".join(header)]

    for row in df.select(["ticker", "action", "order_quantity"]).to_dicts():
        symbol = _canon(row["ticker"])
        action = str(row["action"]).upper()  # BUY or SELL
        qty = int(row["order_quantity"])
        # For MOC, leave LmtPrice and AuxPrice blank
        line = f"{symbol},STK,USD,{exchange},{action},{qty},{order_type},,,{time_in_force}"
        lines.append(line)

    return "\n".join(lines) + "\n"
