from abc import ABC, abstractmethod

from forlex.models.currency import CurrencyPair
from forlex.ninibot.models.trade_decision import TradeDecision
from forlex.ninibot.ports import OpenTrade, TradeId


class TradingStrategy(ABC):
    @abstractmethod
    def get_decision(self, pair: CurrencyPair,
                     open_trades: list[OpenTrade]) -> TradeDecision | None:
        """Returns the trade to open on `pair`, or None."""

    def initialize(self) -> None:
        """Called once, before the first decision."""

    def on_trade_placed(self, trade_id: TradeId, decision: TradeDecision) -> None:
        """Called after the broker opens the trade a decision asked for."""

    def save_model(self) -> None:
        """Persists whatever the strategy has learned."""
