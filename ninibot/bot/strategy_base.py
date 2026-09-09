from abc import ABC, abstractmethod
from models.trade_decision import TradeDecision

class TradingStrategy(ABC):
    @abstractmethod
    def get_decision(self, candle_time, pair, granularity, api, trade_settings, log_message) -> TradeDecision:
        """
        Must return a TradeDecision object or None.
        """
        pass   