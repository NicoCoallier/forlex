from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from forlex.models.simulation import PriceState

from .exit_order import ExitOrder


@dataclass
class StopLossOrder(ExitOrder):
    priority: ClassVar[int] = 0

    def fill_price(self, state: PriceState) -> float | None:
        candle = self.closing_candle(state)
        if self.units > 0:
            return self._fill_when_falls_to_price(candle)
        return self._fill_when_rises_to_price(candle)
