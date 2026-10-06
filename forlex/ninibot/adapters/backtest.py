"""Runs the bot against the forlex simulator, for backtesting."""

from dataclasses import replace
from datetime import datetime

import pandas as pd

from forlex.broker import Broker
from forlex.models import Account, Quote
from forlex.models.currency import CurrencyConverter, CurrencyPair
from forlex.models.order import StopLossOrder
from forlex.ninibot.ports import BrokerError, ClosedTrade, OpenTrade, TradeId


class BacktestAdapter:
    """Market data and brokerage backed by the forlex simulator.

    `history` must hold the frames the MarketSimulator replays, so the candles
    the bot reads always agree with the broker's prices.
    """

    def __init__(
        self, broker: Broker, history: dict[CurrencyPair, pd.DataFrame]
    ) -> None:
        self._broker = broker
        self._history = history

    def now(self) -> datetime:
        # The broker's time is the bar it is on. BackTestingEngine.current_time
        # has already moved one bar ahead, so reading it would leak the future.
        return self._broker.current_time

    def candles(self, pair: CurrencyPair, count: int) -> pd.DataFrame:
        history = self._history.get(pair)
        if history is None:
            return pd.DataFrame()
        end = history["timestamp"].searchsorted(self.now(), side="right")
        return history.iloc[max(0, end - count) : end]

    def quotes(self) -> dict[CurrencyPair, Quote]:
        return self._broker.get_prices()

    def account(self) -> Account:
        # A copy, because the broker keeps updating its own as prices move
        return replace(self._broker.account)

    def open_trades(self) -> list[OpenTrade]:
        quotes = self.quotes()
        converter = CurrencyConverter(quotes)
        currency = self._broker.account.currency
        stop_losses = self._pending_stop_losses()
        return [
            OpenTrade(
                id=str(trade.id),
                pair=trade.pair,
                units=trade.units,
                entry_price=trade.entry_price,
                open_time=trade.entry_time,
                unrealized_pl=converter.convert(
                    trade.unrealized_pl(quotes[trade.pair]), trade.pair.quote, currency
                ),
                stop_loss=stop_losses.get(trade.id),
            )
            for trade in self._broker.trade_manager.open_trades
        ]

    def closed_trade(self, trade_id: TradeId) -> ClosedTrade | None:
        # Newest first: the bot asks about a trade right after it closes
        trade = next(
            (
                t
                for t in reversed(self._broker.trade_manager.closed_trades)
                if str(t.id) == trade_id
            ),
            None,
        )
        if trade is None or trade.close_price is None or trade.close_time is None:
            return None
        return ClosedTrade(
            id=trade_id,
            close_price=trade.close_price,
            close_time=trade.close_time,
            realized_pl=trade.realized_pl,
        )

    def place_market_order(
        self,
        pair: CurrencyPair,
        units: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> TradeId:
        trade = self._broker.place_market_order(pair, units, stop_loss, take_profit)
        return str(trade.id)

    def set_stop_loss(self, trade_id: TradeId, price: float) -> None:
        try:
            self._broker.set_stop_loss(int(trade_id), price)
        except KeyError as error:
            raise BrokerError(f"No open trade {trade_id}") from error

    def close_trade(self, trade_id: TradeId) -> None:
        try:
            self._broker.close_trade(int(trade_id))
        except KeyError as error:
            raise BrokerError(f"No open trade {trade_id}") from error

    def _pending_stop_losses(self) -> dict[int, float]:
        return {
            order.trade_id: order.price
            for order in self._broker.order_manager.pending_orders
            if isinstance(order, StopLossOrder)
        }
