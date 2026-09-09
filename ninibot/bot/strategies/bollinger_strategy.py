
from bot.strategy_base import TradingStrategy
from bot.technicals_manager import get_trade_decision as old_get_decision
from models.trade_decision import TradeDecision

class BollingerBandsStrategy(TradingStrategy):
    def get_decision(self, candle_time, pair, granularity, api, trade_settings, log_message, open_trades=None):
        # Extract single pair settings from dict
        pair_settings = trade_settings.get(pair) if isinstance(trade_settings, dict) else trade_settings
        return old_get_decision(candle_time, pair, granularity, api, pair_settings, log_message)   