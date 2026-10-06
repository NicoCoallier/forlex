from datetime import datetime

from forlex.models import Account, Quote
from forlex.models.currency import Currency, CurrencyPair
from forlex.models.order import MarketOrder
from forlex.models.simulation import PriceState
from forlex.models.trade import Trade
from forlex.order_manager import OrderManager
from forlex.trade_manager import TradeManager


class Broker:
    def __init__(self, balance: float, account_currency: Currency) -> None:
        self.initial_balance = balance
        self.deposits = 0.0
        self.account = Account(
            currency=account_currency,
            balance=balance,
            nav=balance,
            position=0.0,
            unrealized_pl=0.0,
            realized_pl=0.0,
            margin_used=0.0,
            margin_available=balance,
        )
        self.trade_manager = TradeManager(account_currency)
        self.order_manager = OrderManager()
        self.market_state: dict[CurrencyPair, PriceState] = {}
        self._last_bars: dict[CurrencyPair, PriceState] = {}

    def get_prices(self) -> dict[CurrencyPair, Quote]:
        return {pair: state.quote for pair, state in self.market_state.items()}

    def update(
        self, market_state: dict[CurrencyPair, PriceState], current_time: datetime
    ) -> None:
        self.market_state = market_state
        self.current_time = current_time
        self.update_order_and_trade(self._new_bars(market_state))
        self.update_account()

    def _new_bars(
        self, market_state: dict[CurrencyPair, PriceState]
    ) -> dict[CurrencyPair, PriceState]:
        # The simulator repeats a pair's last bar until a new one prints. An exit
        # order placed after that bar must not fill on prices from before it.
        new_bars = {
            pair: state
            for pair, state in market_state.items()
            if state != self._last_bars.get(pair)
        }
        self._last_bars = market_state
        return new_bars

    def update_order_and_trade(self, new_bars: dict[CurrencyPair, PriceState]) -> None:
        fills = self.order_manager.process(new_bars, self.current_time)
        pair_to_quote = self.get_prices()
        for trade_exit in fills.trade_exits:
            self.trade_manager.close_trade(
                trade_exit.trade_id, pair_to_quote, self.current_time, trade_exit.price
            )
        self.trade_manager.open_trades.extend(fills.opened_trades)

    def close_trade(self, trade_id: int) -> Trade:
        trade = self.trade_manager.close_trade(
            trade_id, self.get_prices(), self.current_time
        )
        self.order_manager.cancel_trade_orders(trade_id)
        self.update_account()
        return trade

    def set_stop_loss(self, trade_id: int, price: float) -> None:
        trade = self.trade_manager.get_open_trade(trade_id)
        self.order_manager.replace_stop_loss(trade, price)

    def deposit(self, amount: float) -> None:
        self.deposits += amount
        self.update_account()

    def update_account(self) -> None:
        pair_to_quote = self.get_prices()
        self.account.position = self.trade_manager.position(pair_to_quote)
        self.account.unrealized_pl = self.trade_manager.unrealized_pl(pair_to_quote)
        self.account.realized_pl = self.trade_manager.realized_pl()
        self.account.balance = (
            self.initial_balance + self.deposits + self.account.realized_pl
        )
        self.account.nav = self.account.balance + self.account.unrealized_pl
        self.account.margin_used = self.trade_manager.margin_used(pair_to_quote)
        self.account.margin_available = self.account.nav - self.account.margin_used

    def place_market_order(
        self,
        pair: CurrencyPair,
        units: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Trade:
        order = MarketOrder(
            pair=pair,
            units=units,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        trade = self.order_manager.fill(
            order, self.market_state[pair], self.current_time
        )
        self.trade_manager.open_trades.append(trade)
        self.update_account()
        return trade
