from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import count
from typing import ClassVar

from forlex.models import Direction
from forlex.models.currency import CurrencyPair
from forlex.models.simulation import PriceState

_order_ids = count(1)


@dataclass
class Order(ABC):
    priority: ClassVar[int]

    pair: CurrencyPair
    units: float
    id: int = field(default_factory=lambda: next(_order_ids), kw_only=True)

    @property
    def direction(self) -> Direction:
        return Direction.SHORT if self.units < 0 else Direction.LONG

    @abstractmethod
    def fill_price(self, state: PriceState) -> float | None:
        """Returns the price this order fills at, or None if it does not fill yet."""
