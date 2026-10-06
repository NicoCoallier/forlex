from dataclasses import dataclass
from datetime import datetime

from forlex.models.currency import CurrencyPair
from forlex.models.order import (
    ExitOrder,
    MarketOrder,
    Order,
    StopLossOrder,
    TakeProfitOrder,
)
from forlex.models.simulation import PriceState
from forlex.models.trade import Trade


@dataclass(frozen=True)
class TradeExit:
    trade_id: int
    price: float


@dataclass(frozen=True)
class OrderFills:
    opened_trades: list[Trade]
    trade_exits: list[TradeExit]


class OrderManager:


    def __init__(self) -> None:
        self.pending_orders: list[Order] = []
        self.filled_orders: list[Order] = []
        self.cancelled_orders: list[Order] = []

    def place[O: Order](self, order: O) -> O:
        self.pending_orders.append(order)
        return order

    def process(
        self, market_state: dict[CurrencyPair, PriceState], current_time: datetime
    ) -> OrderFills:
        fills = OrderFills(opened_trades=[], trade_exits=[])
        for order in sorted(self.pending_orders, key=lambda o: o.priority):
            if order not in self.pending_orders:
                continue  # cancelled earlier in this tick
            state = market_state.get(order.pair)
            if state is None:
                continue
            price = order.fill_price(state)
            if price is None:
                continue
            self._move(order, self.filled_orders)
            if isinstance(order, MarketOrder):
                fills.opened_trades.append(self._open_trade(order, price, current_time))
            elif isinstance(order, ExitOrder):
                self.cancel_trade_orders(order.trade_id)
                fills.trade_exits.append(TradeExit(order.trade_id, price))
        return fills

    def fill(self, order: MarketOrder, state: PriceState, fill_time: datetime) -> Trade:
        self.filled_orders.append(order)
        return self._open_trade(order, order.fill_price(state), fill_time)

    def replace_stop_loss(self, trade: Trade, price: float) -> None:
        for order in list(self.pending_orders):
            if isinstance(order, StopLossOrder) and order.trade_id == trade.id:
                self._move(order, self.cancelled_orders)
        self.place(StopLossOrder(trade.pair, trade.units, trade.id, price))

    def cancel_trade_orders(self, trade_id: int) -> None:
        for order in list(self.pending_orders):
            if isinstance(order, ExitOrder) and order.trade_id == trade_id:
                self._move(order, self.cancelled_orders)

    def _open_trade(
        self, order: MarketOrder, entry_price: float, entry_time: datetime
    ) -> Trade:
        trade = Trade(
            pair=order.pair,
            units=order.units,
            entry_price=entry_price,
            entry_time=entry_time,
        )
        if order.stop_loss is not None:
            self.place(
                StopLossOrder(trade.pair, trade.units, trade.id, order.stop_loss)
            )
        if order.take_profit is not None:
            self.place(
                TakeProfitOrder(trade.pair, trade.units, trade.id, order.take_profit)
            )
        return trade

    def _move(self, order: Order, to: list[Order]) -> None:
        self.pending_orders.remove(order)
        to.append(order)
