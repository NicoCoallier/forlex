from datetime import datetime, timedelta

from forlex.broker import Broker

from .market_simulator import MarketSimulator


class BackTestingEngine:
    def __init__(
        self,
        initial_time: datetime,
        time_delta: timedelta,
        market: MarketSimulator,
        broker: Broker,
    ) -> None:
        self.initial_time = initial_time
        self.current_time = self.initial_time
        self.time_delta = time_delta
        self.market = market
        self.broker = broker

    def step(self) -> None:
        market_state = self.market.get_latest_state(self.current_time)
        self.broker.update(market_state, self.current_time)

        # Eventually have the bot here
        self.current_time += self.time_delta
