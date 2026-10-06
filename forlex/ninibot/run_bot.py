"""Runs the bot live against the OANDA account configured in .env.

Run from the project root: poetry run python -m forlex.ninibot.run_bot
"""

import signal
import time
from pathlib import Path
from types import FrameType

from forlex.ninibot.adapters.oanda import OandaAdapter
from forlex.ninibot.api.oanda_api import OandaApi
from forlex.ninibot.bot.bot import Bot
from forlex.ninibot.bot.candle_manager import CandleManager
from forlex.ninibot.bot.strategies.q_learning_portfolio import (
    PortfolioQLearningStrategy,
)
from forlex.ninibot.infrastructure.instrument_collection import instrumentCollection
from forlex.ninibot.infrastructure.log_wrapper import ERROR_LOG, MAIN_LOG, BotLogs
from forlex.ninibot.models.bot_settings import BotSettings
from forlex.ninibot.models.trade_settings import TradeSettings

SETTINGS_FILE = Path(__file__).parent / "bot" / "settings.json"
GRANULARITY = "M1"
POLL_SECONDS = 1


def main() -> None:
    instrumentCollection.LoadInstruments("./data")
    settings = BotSettings.load(SETTINGS_FILE)
    logs = _start_logs(settings)

    oanda = OandaAdapter(OandaApi.from_env(), settings.pairs, GRANULARITY)
    strategy = PortfolioQLearningStrategy(
        oanda, oanda, settings.trade_settings, logs.message
    )
    bot = Bot(oanda, oanda, strategy, settings.risk_percent, logs.message)
    candles = CandleManager(oanda, settings.pairs, logs.message)

    signal.signal(signal.SIGINT, _shut_down)
    signal.signal(signal.SIGTERM, _shut_down)
    try:
        while True:
            time.sleep(POLL_SECONDS)
            closed = candles.update_timings()
            if closed:
                bot.on_candle_close(closed)
                strategy.save_model()
    except Exception as error:
        logs.message(f"CRASH: {error}", ERROR_LOG)
        raise
    finally:
        strategy.save_model()
        logs.message("Strategy model saved successfully.", MAIN_LOG)
        logs.message("Bot shutdown complete.", MAIN_LOG)


def _start_logs(settings: BotSettings) -> BotLogs:
    logs = BotLogs(settings.pairs)
    for pair, pair_settings in settings.trade_settings.items():
        logs.message(f"{pair_settings}", pair)
    trade_settings = TradeSettings.settings_to_str(settings.trade_settings)
    logs.message(f"Bot started with {trade_settings}", MAIN_LOG)
    return logs


def _shut_down(signum: int, frame: FrameType | None) -> None:
    print("\n🛑 Shutdown signal received. Saving model...")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
