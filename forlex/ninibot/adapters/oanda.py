"""Runs the bot against an OANDA v20 account."""

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from forlex.models import Account, Direction, Quote
from forlex.models.currency import Currency, CurrencyPair
from forlex.ninibot.api.oanda_api import OandaApi
from forlex.ninibot.ports import (
    BrokerError,
    ClosedTrade,
    OpenTrade,
    OrderRejected,
    TradeId,
)

_CSV_COLUMNS = {"time": "timestamp"} | {
    f"{side}_{short}": f"{side}_{name}"
    for side in ("mid", "bid", "ask")
    for short, name in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"))
}


class OandaAdapter:
    """Market data and brokerage backed by the OANDA REST API."""

    def __init__(
        self, api: OandaApi, pairs: list[CurrencyPair], granularity: str
    ) -> None:
        self._api = api
        self._pairs = pairs
        self._granularity = granularity

    def now(self) -> datetime:
        return datetime.now(UTC)

    def candles(self, pair: CurrencyPair, count: int) -> pd.DataFrame:
        # One extra, because OANDA's newest candle is usually still forming
        candles = self._api.get_candles_df(
            pair.value, count=count + 1, granularity=self._granularity
        )
        if candles is None or candles.empty:
            return pd.DataFrame()
        return candles.rename(columns=_CSV_COLUMNS).tail(count).reset_index(drop=True)

    def quotes(self) -> dict[CurrencyPair, Quote]:
        prices = self._api.get_prices([pair.value for pair in self._pairs])
        if prices is None:
            raise BrokerError("OANDA pricing request failed")
        return {
            CurrencyPair(price.instrument): Quote(ask=price.ask, bid=price.bid)
            for price in prices
        }

    def account(self) -> Account:
        summary = self._api.get_account_summary()
        if summary is None:
            raise BrokerError("OANDA account summary request failed")
        return Account(
            currency=Currency(summary["currency"]),  # BUGFIX-6: needed for sizing
            balance=float(summary["balance"]),
            nav=float(summary["NAV"]),
            position=float(summary["positionValue"]),
            unrealized_pl=float(summary["unrealizedPL"]),
            realized_pl=float(summary["pl"]),
            margin_used=float(summary["marginUsed"]),
            margin_available=float(summary["marginAvailable"]),
        )

    def open_trades(self) -> list[OpenTrade]:
        trades = self._api.get_open_trades()
        if trades is None:
            raise BrokerError("OANDA open trades request failed")
        # Trades on instruments CurrencyPair doesn't model aren't the bot's to manage
        return [_open_trade(t) for t in trades if t["instrument"] in CurrencyPair]

    def closed_trade(self, trade_id: TradeId) -> ClosedTrade | None:
        trade = self._api.get_trade(trade_id)
        if trade is None or trade["state"] != "CLOSED":
            return None
        # BUGFIX-1 (OANDA half): the old OandaApi.get_trade_close_price read
        # "closePrice", which v20 doesn't have; the field is "averageClosePrice"
        return ClosedTrade(
            id=trade_id,
            close_price=float(trade["averageClosePrice"]),
            close_time=datetime.fromisoformat(trade["closeTime"]),
            realized_pl=float(trade["realizedPL"]),
        )

    def place_market_order(
        self,
        pair: CurrencyPair,
        units: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> TradeId:
        direction = Direction.LONG if units > 0 else Direction.SHORT
        trade_id = self._api.place_trade(
            pair.value, abs(units), direction, stop_loss, take_profit
        )
        if trade_id is None:
            raise OrderRejected(f"OANDA rejected {units:+g} {pair}")
        return trade_id

    def set_stop_loss(self, trade_id: TradeId, price: float) -> None:
        # BUGFIX-2: the fixed request needs the instrument to round the price
        trade = self._api.get_trade(trade_id)
        if trade is None:
            raise BrokerError(f"OANDA could not find trade {trade_id}")
        if not self._api.update_trade_sl(trade["instrument"], trade_id, price):
            raise OrderRejected(f"OANDA refused stop loss {price} on trade {trade_id}")

    def close_trade(self, trade_id: TradeId) -> None:
        if not self._api.close_trade(trade_id):
            raise BrokerError(f"OANDA could not close trade {trade_id}")


def _open_trade(trade: dict[str, Any]) -> OpenTrade:
    stop_loss = trade.get("stopLossOrder", {}).get("price")
    return OpenTrade(
        id=trade["id"],
        pair=CurrencyPair(trade["instrument"]),
        units=float(trade["currentUnits"]),
        entry_price=float(trade["price"]),
        open_time=datetime.fromisoformat(trade["openTime"]),
        unrealized_pl=float(trade["unrealizedPL"]),
        stop_loss=None if stop_loss is None else float(stop_loss),
    )
