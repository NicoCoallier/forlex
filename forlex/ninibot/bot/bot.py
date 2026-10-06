from forlex.models import Account, Direction
from forlex.models.currency import CurrencyPair
from forlex.ninibot.bot.strategy_base import TradingStrategy
from forlex.ninibot.bot.trade_manager import place_trade
from forlex.ninibot.infrastructure.log_wrapper import ERROR_LOG, MAIN_LOG, LogMessage
from forlex.ninibot.ports import Brokerage, BrokerError, MarketData, OpenTrade


class Bot:
    """Runs a strategy on each closed candle, against whichever broker it is given."""

    MIN_MARGIN = 10.0
    MAX_OPEN_TRADES = 20
    MAX_TRADE_AGE_HOURS = 5.0

    def __init__(
        self,
        market: MarketData,
        broker: Brokerage,
        strategy: TradingStrategy,
        risk_percent: float,
        log_message: LogMessage,
    ) -> None:
        self.market = market
        self.broker = broker
        self.strategy = strategy
        self.risk_percent = risk_percent
        self.log_message = log_message

        self.strategy.initialize()
        self.log_to_main(f"Bot started with strategy: {strategy.__class__.__name__}")

    def log_to_main(self, msg: str) -> None:
        self.log_message(msg, MAIN_LOG)

    def log_to_error(self, msg: str) -> None:
        self.log_message(msg, ERROR_LOG)

    def on_candle_close(self, pairs: list[CurrencyPair]) -> None:
        """Lets the strategy act on each pair whose candle just closed."""
        if not pairs:
            return
        try:
            self._trade(pairs)
        except BrokerError as error:
            self.log_to_error(f"Skipping cycle, broker unavailable: {error}")

    def _trade(self, pairs: list[CurrencyPair]) -> None:
        account = self.broker.account()
        if account.margin_available < self.MIN_MARGIN:
            self.log_to_main(
                f"⚠️ Margin too low: {account.margin_available:.2f} < {self.MIN_MARGIN}. "
                "Skipping cycle."
            )
            return

        open_trades = self.broker.open_trades()
        if len(open_trades) >= self.MAX_OPEN_TRADES:
            self.log_to_main(
                f"⚠️ Max trades reached: {len(open_trades)}/{self.MAX_OPEN_TRADES}. "
                "Skipping."
            )
            return

        self.log_to_main(
            f"Account: NAV={account.nav:.2f}, "
            f"FreeMargin={account.margin_available:.2f}, OpenTrades={len(open_trades)}"
        )

        self._close_stale_trades(open_trades)
        open_trades = self.broker.open_trades()

        for pair in pairs:
            try:
                self._act_on(pair, open_trades, account)
            except Exception as e:
                self.log_to_error(f"⚠️ Error processing {pair}: {e}")

    def _close_stale_trades(self, open_trades: list[OpenTrade]) -> None:
        now = self.market.now()
        for trade in open_trades:
            age_hours = (now - trade.open_time).total_seconds() / 3600.0
            if age_hours <= self.MAX_TRADE_AGE_HOURS:
                continue
            self.log_to_main(
                f"⏰ Closing stale trade: {trade.pair} (id={trade.id}, "
                f"age={age_hours:.1f}h > {self.MAX_TRADE_AGE_HOURS}h)"
            )
            try:
                self.broker.close_trade(trade.id)
            except BrokerError as e:
                self.log_to_main(f"⚠️ Time-based exit error: {e}")

    def _act_on(
        self, pair: CurrencyPair, open_trades: list[OpenTrade], account: Account
    ) -> None:
        decision = self.strategy.get_decision(pair, open_trades)
        if decision is None or decision.signal == Direction.NONE:
            return

        self.log_message(f"Place Trade: {decision}", pair)
        trade_id = place_trade(
            decision,
            self.market,
            self.broker,
            account,
            self.risk_percent,
            self.log_message,
            self.log_to_error,
        )
        if trade_id:
            self.strategy.on_trade_placed(trade_id, decision)
