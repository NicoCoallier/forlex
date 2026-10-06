import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from forlex.models.currency import CurrencyPair
from forlex.ninibot.models.trade_settings import TradeSettings


@dataclass(frozen=True)
class BotSettings:
    trade_settings: dict[CurrencyPair, TradeSettings]
    risk_percent: float

    @classmethod
    def load(cls, path: Path) -> Self:
        with open(path) as f:
            data = json.load(f)
        return cls(
            trade_settings={
                CurrencyPair(k): TradeSettings(v) for k, v in data["pairs"].items()
            },
            risk_percent=data["trade_risk"] / 100,
        )

    @property
    def pairs(self) -> list[CurrencyPair]:
        return list(self.trade_settings)
