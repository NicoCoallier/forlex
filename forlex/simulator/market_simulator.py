from pandas import DataFrame

from forlex.models import Candle, Quote
from forlex.models.currency import CurrencyPair
from forlex.models.simulation import PriceState
from forlex.utils import DataFrameTimeStampCursor


class MarketSimulator:
    def __init__(self, pair_to_history: dict[CurrencyPair, DataFrame]):
        self.pair_to_history = pair_to_history
        self.pair_to_latest = {
            pair: DataFrameTimeStampCursor(history)
            for pair, history in self.pair_to_history.items()
        }

    def get_latest_state(self, current_time) -> dict[CurrencyPair, PriceState]:
        return {
            pair: MarketSimulator._state_from_df_row(latest.get_latest(current_time))
            for pair, latest in self.pair_to_latest.items()
        }

    def get_latest_currency_pair_state(
        self, current_time, pair: CurrencyPair
    ) -> PriceState:
        return MarketSimulator._state_from_df_row(
            self.pair_to_latest[pair].get_latest(current_time)
        )

    @staticmethod
    def _state_from_df_row(row) -> PriceState:
        ask_candle = Candle(
            open=row.ask_open, low=row.ask_low, high=row.ask_high, close=row.ask_close
        )
        bid_candle = Candle(
            open=row.bid_open, low=row.bid_low, high=row.bid_high, close=row.bid_close
        )
        quote = Quote(ask_candle.close, bid=bid_candle.close)
        return PriceState(ask_candle, bid_candle, quote)
