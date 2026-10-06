"""Replays the bot over the CSV history in data/ and writes daily HTML replays.

Run from the project root: poetry run python -m forlex.ninibot.run_backtest

Each run starts from an empty Q-table with a fixed seed, so runs repeat exactly,
and saves what it learned to bot/models/backtest/, apart from the live model.
A replayed day takes about 15 s.

REPLAYS writes one HTML chart per replayed day to bot/models/backtest/replays/,
named after the day's first minute (UTC); days with no candles, such as
weekends, are skipped. A "day" is 24 h from START, not a calendar day.

START picks the first replayed minute (UTC); None starts at the beginning of the
data. LOGS writes the bot's logs to bot/models/backtest/logs/, one file per
pair plus main and error, each line stamped with the replay time. They take
about 14 MB and a few extra seconds per replayed day.

At the end of each replayed day, if the balance is below STARTING_BALANCE, the
difference is deposited. The summary reports the deposits and the net result.
"""

import contextlib
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from forlex.broker import Broker
from forlex.models.currency import Currency, CurrencyPair
from forlex.ninibot.adapters.backtest import BacktestAdapter
from forlex.ninibot.bot.bot import Bot
from forlex.ninibot.bot.candle_manager import CandleManager
from forlex.ninibot.bot.strategies.q_learning_portfolio import (
    PortfolioQLearningStrategy,
)
from forlex.ninibot.infrastructure.log_wrapper import MAIN_LOG, BotLogs, LogMessage
from forlex.ninibot.models.bot_settings import BotSettings
from forlex.simulator import BackTestingEngine, MarketSimulator
from forlex.visualization import ReplayRecorder

SETTINGS_FILE = Path(__file__).parent / "bot" / "settings.json"
DATA_DIR = Path("data")
MODEL_DIR = Path("bot/models/backtest")
STARTING_BALANCE = 100_000
START: str | None = None  # e.g. "2026-03-02 00:00"
DAYS = 128
LOGS = False
SEED = 0
REPLAYS = True
MINUTES_PER_DAY = 24 * 60


def main() -> None:
    settings = BotSettings.load(SETTINGS_FILE)
    history = {pair: load_history(pair) for pair in settings.pairs}
    start = replay_start(history)
    for previous_run in ("q_table.json", "training_data.jsonl"):
        (MODEL_DIR / previous_run).unlink(missing_ok=True)
    for previous_replay in (MODEL_DIR / "replays").glob("*.html"):
        previous_replay.unlink()

    print(f"Replaying {DAYS} day(s) from {start:%Y-%m-%d %H:%M} UTC...")
    # The strategy prints a line per Q-update; keep them out of the console
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (
        open(MODEL_DIR / "strategy_output.log", "w") as output,
        contextlib.redirect_stdout(output),
    ):
        broker = backtest(
            settings, history, start, DAYS * MINUTES_PER_DAY, MODEL_DIR, LOGS, REPLAYS
        )

    print(summary(broker))
    print(f"Model and strategy output written to {MODEL_DIR}/")
    if LOGS:
        print(f"Bot logs written to {MODEL_DIR / 'logs'}/")
    if REPLAYS:
        print(f"Daily replays written to {MODEL_DIR / 'replays'}/")


def load_history(pair: CurrencyPair) -> pd.DataFrame:
    bars = pd.read_csv(DATA_DIR / f"{pair.value}_M1_2026_present.csv")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    return bars


def replay_start(history: dict[CurrencyPair, pd.DataFrame]) -> datetime:
    # The simulator can only price a pair once it has a bar
    data_start = max(bars["timestamp"].iloc[0] for bars in history.values())
    if START is None:
        return data_start
    start = pd.Timestamp(START, tz="UTC")
    if start < data_start:
        raise ValueError(f"START {start} is before the data begins ({data_start})")
    return start


def backtest(
    settings: BotSettings,
    history: dict[CurrencyPair, pd.DataFrame],
    start: datetime,
    minutes: int,
    model_dir: Path,
    logs: bool = False,
    replays: bool = False,
) -> Broker:
    """Replays `minutes` one-minute bars of history through the bot, from `start`."""
    broker = Broker(STARTING_BALANCE, Currency.USD)
    log_message = _replay_logs(broker, settings, model_dir / "logs") if logs else _quiet
    market = MarketSimulator(history)
    engine = BackTestingEngine(start, timedelta(minutes=1), market, broker)
    replay_dir = model_dir / "replays"
    recorder = ReplayRecorder(broker) if replays else None
    sim = BacktestAdapter(broker, history)
    strategy = PortfolioQLearningStrategy(
        sim,
        sim,
        settings.trade_settings,
        log_message,
        rng=np.random.default_rng(SEED),
        model_path=str(model_dir / "q_table.json"),
        training_data_path=str(model_dir / "training_data.jsonl"),
    )

    # The bot reads the broker's current bar, so the replay needs one first
    engine.step()
    if recorder is not None:
        recorder.record()
    day_start = broker.current_time
    bot = Bot(sim, sim, strategy, settings.risk_percent, log_message)
    candles = CandleManager(sim, settings.pairs, log_message)

    for minute in range(1, minutes):
        engine.step()
        bot.on_candle_close(candles.update_timings())
        if recorder is not None:
            recorder.record()
        if minute % MINUTES_PER_DAY == 0:
            top_up(broker, log_message)
            _report_progress(broker)
            if recorder is not None:
                write_replay(recorder, replay_dir, day_start)
                recorder = ReplayRecorder(broker)
            day_start = broker.current_time

    if recorder is not None:
        write_replay(recorder, replay_dir, day_start)
    strategy.save_model()
    return broker


def write_replay(recorder: ReplayRecorder, directory: Path, day_start: datetime) -> None:
    if not recorder.has_candles:
        return
    directory.mkdir(parents=True, exist_ok=True)
    # plotly.min.js is written once beside the replays instead of into each one
    recorder.figure().write_html(
        directory / f"{day_start:%Y-%m-%d_%H%M}.html", include_plotlyjs="directory"
    )


def top_up(broker: Broker, log_message: LogMessage) -> None:
    """Deposits enough to bring the balance back to STARTING_BALANCE, so a losing
    run keeps trading at full size instead of shrinking toward zero."""
    shortfall = STARTING_BALANCE - broker.account.balance
    if shortfall > 0:
        broker.deposit(shortfall)
        log_message(f"Topped up {shortfall:,.2f} to the starting balance", MAIN_LOG)


def summary(broker: Broker) -> str:
    account = broker.account
    closed = broker.trade_manager.closed_trades
    lines = [
        f"{'pair':8} {'trades':>6} {'won':>5} {'P/L':>10}",
    ]
    for pair in sorted({trade.pair for trade in closed}):
        trades = [trade for trade in closed if trade.pair == pair]
        won = sum(trade.realized_pl > 0 for trade in trades)
        pl = sum(trade.realized_pl for trade in trades)
        lines.append(f"{pair.value:8} {len(trades):6} {won:5} {pl:10,.2f}")
    lines += [
        f"Closed trades: {len(closed)}, still open: {len(broker.trade_manager.open_trades)}",
        f"Realized P/L:  {account.realized_pl:,.2f} {account.currency.value}",
        f"NAV:           {account.nav:,.2f} (started at {STARTING_BALANCE:,.2f})",
        f"Deposits:      {broker.deposits:,.2f}",
        # NAV alone hides losses once top-ups have refilled the account
        f"Net result:    {account.nav - STARTING_BALANCE - broker.deposits:,.2f}",
    ]
    return "\n".join(lines)


def _report_progress(broker: Broker) -> None:
    closed = len(broker.trade_manager.closed_trades)
    print(
        f"{broker.current_time:%Y-%m-%d %H:%M}  NAV {broker.account.nav:,.2f}  "
        f"{closed} trades closed",
        file=sys.stderr,
    )


def _replay_logs(broker: Broker, settings: BotSettings, directory: Path) -> LogMessage:
    logs = BotLogs(settings.pairs, str(directory))

    def log_message(msg: str, key: str) -> None:
        # The file timestamps are wall-clock time; the replay time is what matters
        logs.message(f"[{broker.current_time:%Y-%m-%d %H:%M}] {msg}", key)

    return log_message


def _quiet(msg: str, key: str) -> None:
    """Drops the bot's log lines; writing one per pair per bar would dominate."""


if __name__ == "__main__":
    main()
