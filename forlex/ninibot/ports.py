"""What the bot needs from a market and a broker, whether live or simulated."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd

from forlex.models import Account, Quote
from forlex.models.currency import CurrencyPair

type TradeId = str


class BrokerError(Exception):
    """The broker could not answer or carry out a request."""


class OrderRejected(BrokerError):
    """The broker refused an order."""


@dataclass(frozen=True)
class OpenTrade:
    id: TradeId
    pair: CurrencyPair
    units: float  # negative when short
    entry_price: float
    open_time: datetime
    unrealized_pl: float  # in the account currency
    stop_loss: float | None


@dataclass(frozen=True)
class ClosedTrade:
    id: TradeId
    close_price: float
    close_time: datetime
    realized_pl: float


class MarketData(Protocol):
    def now(self) -> datetime: ...

    def candles(self, pair: CurrencyPair, count: int) -> pd.DataFrame:
        """Last `count` complete candles up to now(), oldest first.

        Columns match the CSV files (timestamp, ask_open ... mid_close, volume).
        Empty when no candles are available.
        """
        ...

    def quotes(self) -> dict[CurrencyPair, Quote]: ...


class Brokerage(Protocol):
    def account(self) -> Account: ...

    def open_trades(self) -> list[OpenTrade]: ...

    def closed_trade(self, trade_id: TradeId) -> ClosedTrade | None:
        """The trade once it has closed, or None while it is still open."""
        ...

    def place_market_order(
        self,
        pair: CurrencyPair,
        units: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> TradeId:
        """Opens a trade now at the current quote, or raises OrderRejected."""
        ...

    def set_stop_loss(self, trade_id: TradeId, price: float) -> None: ...

    def close_trade(self, trade_id: TradeId) -> None: ...
