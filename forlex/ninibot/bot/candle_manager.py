from datetime import datetime

from forlex.models.currency import CurrencyPair
from forlex.ninibot.infrastructure.log_wrapper import LogMessage
from forlex.ninibot.models.candle_timing import CandleTiming
from forlex.ninibot.ports import MarketData


class CandleManager:
    """Polls the market and reports which pairs have a newly closed candle."""

    def __init__(self, market: MarketData, pairs: list[CurrencyPair],
                 log_message: LogMessage):
        self.market = market
        self.log_message = log_message
        self.pairs_list = pairs
        self.timings = { p: CandleTiming(self.last_complete_candle(p)) for p in self.pairs_list }
        for p, t in self.timings.items():
            if t.last_time is not None:  # ✅ Only log if initialized
                self.log_message(f"CandleManager() init last_candle:{t}", p)
            else:
                self.log_message(f"CandleManager() init: no candle data for {p} yet", p)

    def last_complete_candle(self, pair: CurrencyPair) -> datetime | None:
        candles = self.market.candles(pair, 1)
        return None if candles.empty else candles["timestamp"].iloc[-1]

    def update_timings(self) -> list[CurrencyPair]:
        triggered = []
        for pair in self.pairs_list:
            current = self.last_complete_candle(pair)
            if current is None:
                self.log_message("Unable to get candle", pair)
                continue
            self.timings[pair].is_ready = False
            if self.timings[pair].last_time is not None and current > self.timings[pair].last_time:
                self.timings[pair].is_ready = True
                self.timings[pair].last_time = current
                self.log_message(f"CandleManager() new candle:{self.timings[pair]}", pair)
                triggered.append(pair)
        return triggered
