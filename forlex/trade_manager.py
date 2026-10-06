from datetime import datetime
from math import fsum

from forlex.models import Quote
from forlex.models.currency import Currency, CurrencyConverter, CurrencyPair
from forlex.models.trade import Trade


class TradeManager:
    def __init__(self, account_currency: Currency):
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.account_currency = account_currency
        # Running total, so reading it never walks closed_trades
        self._realized_pl = 0.0

    def open_instruments(self):
        return {trade.pair for trade in self.open_trades}

    def position(self, pair_to_quote: dict[CurrencyPair, Quote]):
        return self._sum_in_account_currency(Trade.position, pair_to_quote)

    def unrealized_pl(self, pair_to_quote: dict[CurrencyPair, Quote]):
        return self._sum_in_account_currency(Trade.unrealized_pl, pair_to_quote)

    def realized_pl(self) -> float:
        return self._realized_pl

    def get_open_trade(self, trade_id: int) -> Trade:
        trade = next((t for t in self.open_trades if t.id == trade_id), None)
        if trade is None:
            raise KeyError(f"No open trade with id {trade_id}")
        return trade

    def close_trade(
        self,
        trade_id: int,
        pair_to_quote: dict[CurrencyPair, Quote],
        close_time: datetime,
        close_price: float | None = None,
    ) -> Trade:
        trade = self.get_open_trade(trade_id)

        quote = pair_to_quote[trade.pair]
        rate = CurrencyConverter(pair_to_quote).rate(
            trade.pair.quote, self.account_currency
        )
        if close_price is None:
            close_price = trade.current_price(quote)
        trade.close(close_price, close_time, rate)

        self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        self._realized_pl += trade.realized_pl
        return trade

    def margin_used(self, pair_to_quote: dict[CurrencyPair, Quote]):
        return self._sum_in_account_currency(Trade.margin, pair_to_quote)

    def _sum_in_account_currency(self, amount_of, pair_to_quote):
        self._assert_contains_all_quotes(pair_to_quote)
        converter = CurrencyConverter(pair_to_quote)
        return fsum(
            converter.convert(
                amount_of(trade, pair_to_quote[trade.pair]),
                trade.pair.quote,
                self.account_currency,
            )
            for trade in self.open_trades
        )

    def _assert_contains_all_quotes(self, pair_to_quote: dict[CurrencyPair, Quote]):
        open_instruments = self.open_instruments()
        given_pairs = pair_to_quote.keys()
        assert open_instruments <= given_pairs, (
            f"Missing following pairs : {open_instruments - given_pairs}"
        )
