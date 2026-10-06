from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from forlex.models.simulation import PriceState

from .order import Order


# Its an order but lets call it MarketOrder to remember that its not a limit order or stop order
@dataclass
class MarketOrder(Order):
    """Opens a trade at the current price, with optional exit levels."""

    priority: ClassVar[int] = 2

    stop_loss: float | None = None
    take_profit: float | None = None

    def fill_price(self, state: PriceState) -> float:
        # buy at ask price, sell at bid price
        return state.quote.ask if self.units > 0 else state.quote.bid
