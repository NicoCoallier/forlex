from dataclasses import dataclass

from forlex.models import Candle, Quote


@dataclass
class PriceState:
    ask_candle: Candle
    bid_candle: Candle
    quote: Quote

    @property
    def mid_candle(self) -> Candle:
        return (self.ask_candle + self.bid_candle) / 2
