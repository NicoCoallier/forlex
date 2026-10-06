# ticks depending on the granularity of the engine
# define the instruments you want to fetch from the market
# fetch instruments every minute from the market
# Tick and step
from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd

from forlex.broker import Broker
from forlex.models.currency import Currency, CurrencyPair
from forlex.simulator import BackTestingEngine, MarketSimulator


def main():
    currency_to_backtest = [CurrencyPair.USD_CAD, CurrencyPair.EUR_USD]
    pair_to_history = _load_csv_from_pairs(currency_to_backtest)

    market_simulator = MarketSimulator(pair_to_history)
    broker = Broker(10000, Currency.USD)
    engine = BackTestingEngine(
        initial_time=datetime.fromisoformat("2026-08-02 21:04:00+00:00"),
        time_delta=timedelta(minutes=1),
        market=market_simulator,
        broker=broker,
    )

    engine.step()
    engine.broker.place_market_order(CurrencyPair.EUR_USD, 100000)  # negative units to short
    print(engine.broker.order_manager.pending_orders)
    engine.step()
    print(engine.broker.trade_manager.open_trades)
    print("#")

    print(engine.broker.account)


def _load_csv_from_pairs(pairs: Sequence[CurrencyPair]):
    # lots of hardcoded stuff in here
    pair_to_history = {}
    for pair in pairs:
        df = pd.read_csv(f"./data/{pair.value}_M1_2026_present.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        pair_to_history[pair] = df
    return pair_to_history


if __name__ == "__main__":
    main()
