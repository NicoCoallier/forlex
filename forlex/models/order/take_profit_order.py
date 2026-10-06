from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from forlex.models.simulation import PriceState

from .exit_order import ExitOrder


@dataclass
class TakeProfitOrder(ExitOrder):
    priority: ClassVar[int] = 1

    def fill_price(self, state: PriceState) -> float | None:
        candle = self.closing_candle(state)
        if self.units > 0:
            return self._fill_when_rises_to_price(candle)
        return self._fill_when_falls_to_price(candle)
