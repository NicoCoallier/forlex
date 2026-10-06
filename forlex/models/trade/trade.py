from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import count

from forlex.models import Direction, Quote
from forlex.models.currency import CurrencyPair

_trade_ids = count(1)


@dataclass
class Trade:
    pair: CurrencyPair
    units: float
    entry_price: float
    entry_time: datetime
    close_price: float | None = None
    close_time: datetime | None = None
    # In account currency, fixed at close with that moment's conversion rate
    realized_pl: float = 0.0
    id: int = field(default_factory=lambda: next(_trade_ids))

    @property
    def closed(self) -> bool:
        return self.close_price is not None

    @property
    def direction(self) -> Direction:
        return Direction.SHORT if self.units < 0 else Direction.LONG

    def close(
        self, close_price: float, close_time: datetime, quote_to_account_rate: float
    ) -> None:
        if self.closed:
            return
        self.close_price = close_price
        self.close_time = close_time
        self.realized_pl = (
            self.units * (close_price - self.entry_price) * quote_to_account_rate
        )

    def margin(self, quote: Quote) -> float:
        return abs(self.position(quote)) * self.pair.margin_rate

    def position(self, quote: Quote) -> float:
        current_price = self.current_price(quote)
        return self.units * current_price

    def unrealized_pl(self, quote: Quote) -> float:
        current_price = self.current_price(quote)
        return self.units * (current_price - self.entry_price)

    def current_price(self, quote: Quote) -> float:
        return quote.ask if self.units < 0 else quote.bid
