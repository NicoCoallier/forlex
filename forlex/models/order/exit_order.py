from __future__ import annotations

from dataclasses import dataclass

from forlex.models import Candle
from forlex.models.simulation import PriceState

from .order import Order


@dataclass
class ExitOrder(Order):
    """Closes trade `trade_id` once the price reaches `price`.

    `units` are the trade's units, so `direction` is the trade's direction.
    Levels are checked on the candle of the closing side: a long closes by
    selling at bid, a short by buying at ask.
    """

    trade_id: int
    price: float

    def closing_candle(self, state: PriceState) -> Candle:
        return state.bid_candle if self.units > 0 else state.ask_candle

    def _fill_when_falls_to_price(self, candle: Candle) -> float | None:
        if candle.low > self.price:
            return None
        return min(self.price, candle.open)

    def _fill_when_rises_to_price(self, candle: Candle) -> float | None:
        if candle.high < self.price:
            return None
        return max(self.price, candle.open)
