import json
import time
import atexit
import signal
import sys

from bot.candle_manager import CandleManager
from bot.trade_manager import place_trade
from bot.strategy_base import TradingStrategy
from bot.strategies.bollinger_strategy import BollingerBandsStrategy
from bot.strategies.q_learning_strategy import QLearningStrategy
from bot.strategies.q_learning_portfolio import PortfolioQLearningStrategy

from infrastructure.log_wrapper import LogWrapper
from models.trade_settings import TradeSettings
from api.oanda_api import OandaApi
import constants.defs as defs

class Bot:
    ERROR_LOG = "error"
    MAIN_LOG = "main"
    GRANULARITY = "M1"
    SLEEP = 1

    def __init__(self):
        self.load_settings()
        self.setup_logs()
        
        self.api = OandaApi()
        self.candle_manager = CandleManager(self.api, self.trade_settings, self.log_message, Bot.GRANULARITY)
        self.strategy = self._init_strategy()

        if hasattr(self.strategy, 'initialize'):
            self.strategy.initialize(self.api)
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.log_to_main(f"Bot started with strategy: {self.strategy.__class__.__name__}")

    def _signal_handler(self, sig, frame):
        print("\n🛑 Shutdown signal received. Saving model...")
        self._cleanup()
        # Don't sys.exit — let atexit handle cleanup once
        # OR: set a flag and let the main loop break naturally
        raise SystemExit(0)  # atexit fires _cleanup once   

    def _cleanup(self):
        if hasattr(self.strategy, 'save_model'):
            self.strategy.save_model()
            self.log_to_main("Strategy model saved successfully.")
        self.log_to_main("Bot shutdown complete.")

    def _init_strategy(self):
        strategy_name = self.settings.get('strategy', 'bollinger').lower()
        
        if strategy_name == 'bollinger':
            return BollingerBandsStrategy()
        elif strategy_name == 'q_learning':
            return QLearningStrategy()
        elif strategy_name == 'portfolio_q_learning':
            pairs_list = list(self.trade_settings.keys())
            return PortfolioQLearningStrategy(pairs_list)
        else:
            self.log_to_error(f"Unknown strategy: {strategy_name}. Defaulting to Bollinger.")
            return BollingerBandsStrategy()

    def load_settings(self):
        with open("ninibot/bot/settings.json", "r") as f:
            data = json.loads(f.read())
            self.settings = data
            self.trade_settings = { k: TradeSettings(v, k) for k, v in data['pairs'].items() }
            self.trade_risk = data['trade_risk']

    def setup_logs(self):
        self.logs = {}
        for k in self.trade_settings.keys():
            self.logs[k] = LogWrapper(k)
            self.log_message(f"{self.trade_settings[k]}", k)
        self.logs[Bot.ERROR_LOG] = LogWrapper(Bot.ERROR_LOG)
        self.logs[Bot.MAIN_LOG] = LogWrapper(Bot.MAIN_LOG)
        self.log_to_main(f"Bot started with {TradeSettings.settings_to_str(self.trade_settings)}")

    def log_message(self, msg, key):
        self.logs[key].logger.debug(msg)

    def log_to_main(self, msg):
        self.log_message(msg, Bot.MAIN_LOG)

    def log_to_error(self, msg):
        self.log_message(msg, Bot.ERROR_LOG)

    def _get_pair_features(self, pair, api, trade_settings):
        """Return raw (unhashed) features for a single pair."""
        try:
            max_rows = trade_settings.n_ma + 20
            df = api.get_candles_df(pair, count=max_rows, granularity="M1")
            if df is None or df.shape[0] < 20:
                return {}

            close = df['mid_c'].values if 'mid_c' in df.columns else df['c'].values
            high = df['h'].values if 'h' in df.columns else close
            low = df['l'].values if 'l' in df.columns else close
            open_ = df['mid_o'].values if 'mid_o' in df.columns else close

            ema9 = pd.Series(close).ewm(span=9).mean().iloc[-1]
            atr14 = self._calc_atr(high, low, close, 14)
            price_vs_ema = (close[-1] - ema9) / atr14 if atr14 > 0 else 0
            rsi14 = self._calc_rsi(close, 14)
            atr_norm = (atr14 / close[-1]) * 10000 if close[-1] > 0 else 0
            body = abs(close[-1] - open_[-1])
            range_ = high[-1] - low[-1]
            body_ratio = body / range_ if range_ > 0 else 0.5

            return {
                'ema_pos': round(price_vs_ema, 4),
                'rsi': round(rsi14, 2),
                'vol': int(np.clip((atr_norm / 2.0), 0, 3)),
                'body': round(body_ratio, 4),
                'atr_norm': round(atr_norm, 4),
            }
        except:
            return {}   

    def process_candles(self, triggered):
        if len(triggered) == 0:
            return

        # ✅ Per-pair dedup
        if not hasattr(self, '_last_processed_candles'):
            self._last_processed_candles = {}
        now_candles = {p: self.candle_manager.timings[p].last_time for p in triggered}
        all_same = all(c == self._last_processed_candles.get(p) for p, c in now_candles.items())
        if all_same:
            return
        self._last_processed_candles.update(now_candles)

        account = self.api.get_account_summary()
        if not account:
            return

        margin_available = float(account['marginAvailable'])
        min_margin_required = 10.0
        if margin_available < min_margin_required:
            self.log_message(
                f"⚠️ Margin too low: {margin_available:.2f} < {min_margin_required}. Skipping cycle.", "main"
            )
            return

        open_count = int(account['openTradeCount'])
        max_concurrent = 20
        if open_count >= max_concurrent:
            self.log_message(
                f"⚠️ Max trades reached: {open_count}/{max_concurrent}. Skipping.", "main"
            )
            return

        nav = float(account['NAV'])
        self.log_message(
            f"Account: NAV={nav:.2f}, FreeMargin={margin_available:.2f}, OpenTrades={open_count}", "main"
        )

        open_trades = self.api.get_open_trades()

        # --- Stale trade cleanup ---
        MAX_AGE_HOURS = 5.0
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        for t in open_trades:
            try:
                open_time_str = t.get('openTime') if isinstance(t, dict) else getattr(t, 'openTime', None)
                if not open_time_str:
                    continue
                open_time_str_clean = open_time_str.replace('Z', '+00:00')
                if '.' in open_time_str_clean:
                    base, frac = open_time_str_clean.split('.', 1)
                    frac = frac[:6]
                    open_time_str_clean = f"{base}.{frac}"
                open_time = datetime.fromisoformat(open_time_str_clean)
                age_hours = (now - open_time).total_seconds() / 3600.0

                if age_hours > MAX_AGE_HOURS:
                    trade_id = t.get('id') if isinstance(t, dict) else getattr(t, 'id', '')
                    instrument = t.get('instrument') if isinstance(t, dict) else getattr(t, 'instrument', '')
                    self.log_message(
                        f"⏰ Closing stale trade: {instrument} (id={trade_id}, age={age_hours:.1f}h > {MAX_AGE_HOURS}h)", "main"
                    )
                    self.api.close_trade(trade_id)
            except Exception as e:
                self.log_message(f"⚠️ Time-based exit error: {e}", "main")
                continue

        # Refresh after stale closes
        open_trades = self.api.get_open_trades()

        # --- Trade decisions ---
        for p in triggered:
            try:
                last_time = self.candle_manager.timings[p].last_time
                trade_decision = self.strategy.get_decision(
                    last_time, p, Bot.GRANULARITY, self.api,
                    self.trade_settings, self.log_message, open_trades
                )
                if trade_decision is None or trade_decision.signal == defs.NONE:
                    continue

                self.log_message(f"Place Trade: {trade_decision}", p)
                trade_id = place_trade(
                    trade_decision, self.api, self.log_message, self.log_to_error,
                    account_equity=nav, margin_available=margin_available,
                    risk_percent=self.trade_risk / 100
                )
                if trade_id and hasattr(self.strategy, 'on_trade_placed'):
                    # ✅ State and features come from the decision (computed inside get_decision)
                    state = getattr(trade_decision, 'state', 0)
                    tier = getattr(trade_decision, 'sl_tp_tier', 1)

                    self.strategy.on_trade_placed(
                        trade_id, p, trade_decision.signal, state,
                        trade_decision.entry_price, tier,
                        sl_price=trade_decision.sl, tp_price=trade_decision.tp,
                        state_features=getattr(trade_decision, 'state_features', {}),
                        open_trades_count=getattr(trade_decision, 'open_trades_count', len(open_trades)),
                        exposure_ratio=getattr(trade_decision, 'exposure_ratio', 0),
                        corr_feature=getattr(trade_decision, 'corr_feature', 0),
                        spread=getattr(trade_decision, 'spread', 0),
                    )
            except Exception as e:
                self.log_to_error(f"⚠️ Error processing {p}: {e}")
                continue     

    def run(self):
        try:
            while True:
                time.sleep(Bot.SLEEP)
                self.process_candles(self.candle_manager.update_timings())
        except Exception as error:
            self.log_to_error(f"CRASH: {error}")
            self._cleanup()
            raise   