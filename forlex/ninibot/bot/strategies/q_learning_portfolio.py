import json
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from forlex.models import Direction
from forlex.models.currency import CurrencyPair
from forlex.ninibot.bot.strategy_base import TradingStrategy
from forlex.ninibot.bot.technicals_manager import get_trade_decision
from forlex.ninibot.infrastructure.log_wrapper import LogMessage
from forlex.ninibot.models.trade_decision import TradeDecision
from forlex.ninibot.models.trade_settings import TradeSettings
from forlex.ninibot.ports import Brokerage, MarketData, OpenTrade, TradeId


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")   

class PortfolioQLearningStrategy(TradingStrategy):
    def __init__(
        self,
        market: MarketData,
        broker: Brokerage,
        trade_settings: dict[CurrencyPair, TradeSettings],
        log_message: LogMessage,
        rng: np.random.Generator | None = None,
        model_path: str = "bot/models/q_table_portfolio.json",
        training_data_path: str = "bot/models/training_data.jsonl",
    ):
        self.market = market
        self.broker = broker
        self.trade_settings = trade_settings
        self.log_message = log_message
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pairs = list(trade_settings)
        self.n_states = 1000
        self.n_actions = len(self.pairs) * 11

        self.profit_take_threshold = 1.5     
                
        # BUGFIX-5: get_decision compares against min_profit_to_close but it was never
        # defined, so closing a winning trade older than 15 min raised AttributeError
        # (the bot logged an error and did nothing). 1.0 pip is my pick, tune it: below
        # it, the 0.7-pip spread cost in the reward makes the close a net loss, so the
        # bot keeps trailing instead.
        self.min_profit_to_close = 1.0

        self.epsilon = 0.5
        self.epsilon_decay = 0.999  # Decay slowly
        self.epsilon_min = 0.25    # Floor
         
        self.gamma = 0.95
        self.alpha = 0.3
        self.pair_index = {pair: i for i, pair in enumerate(self.pairs)}
        self.model_path = model_path
        self.open_trades: dict[str, dict[str, Any]] = {}
        # Dont move this 
        self.q_table2 = np.zeros((self.n_states, self.n_actions)) 
        self.q_table = self._load_model()
        # BUG-10 (not fixed yet): _load_model() on the line above has just restored
        # trade_history from the model file (see "✅ Restore" in _load_model), and
        # this line wipes it again. So the repeat-trade penalty in _calculate_reward
        # restarts from zero after every bot restart, and the trade_history written
        # by save_model() is never actually used. Likely fix: set trade_history to {}
        # before calling _load_model(), the way open_trades already is.
        self.trade_history: dict[str, tuple[int, float]] = {}
        self.training_data_path = training_data_path
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)

    def _record_training_data(self, info, reward, next_state, tier, age_seconds, open_trades):
        """Append one trade's features + outcome to the training data file (JSONL)."""
        pair = info['pair']

        # Reconstruct raw features from the state that was stored at entry
        # (or recompute if you prefer — but stored is faster and consistent)
        state_features = info.get('state_features', {})
        now = self.market.now()
        entry_time = datetime.fromtimestamp(info.get('entry_time', now.timestamp()), UTC)

        record = {
            # --- Timestamp ---
            'close_time': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'entry_time': entry_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'age_seconds': round(age_seconds, 1),
            'hour_of_day': entry_time.hour,

            # --- Trade identity ---
            'pair': pair,
            'direction': 1 if info['action'] == Direction.LONG else -1,
            'tier': tier,
            'entry_price': info.get('entry_price', 0),
            'spread': info.get('spread', 0),

            # --- Outcome ---
            'reward': round(reward, 4),
            'profit_pips': info.get('profit_pips', 0),  # ← must be set in _calculate_reward
            'close_reason': info.get('close_reason', 'unknown'),  # 'SL', 'TP', 'manual', 'mgmt_close'
            'peak_pl': info.get('peak_pl', 0),
            'trail_in_profit': info.get('trail_in_profit', False),
            'mgmt_action_at_close': info.get('mgmt_action_at_close'),

            # --- Market state at ENTRY (raw features, not the hash) ---
            'state_hash': info.get('state', 0),
            'next_state_hash': next_state,
            'ema_pos': state_features.get('ema_pos', 0),
            'rsi': state_features.get('rsi', 0),
            'vol_bucket': state_features.get('vol', 0),
            'body_ratio': state_features.get('body', 0),
            'atr_norm': state_features.get('atr_norm', 0),

            # --- Portfolio context at entry ---
            'open_trades_count': info.get('open_trades_count', len(open_trades)),
            'exposure_ratio': info.get('exposure_ratio', 0),
            'corr_feature': info.get('corr_feature', 0),
        }

        # Append to JSONL file
        try:
            with open(self.training_data_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"⚠️ Failed to write training data: {e}")   

    def update_q_value(self, state, action, reward, next_state):
        best_action = np.argmax(self.q_table[next_state])
        target_value = self.q_table2[next_state][best_action]
        
        td_target = reward + self.gamma * target_value
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        
        # ANSI colors
        green = '\033[32m'
        red = '\033[31m'
        reset = '\033[0m'
        color = green if reward >= 0 else red
        
        print(f"📊 Q-UPDATE: state={state}, action={action}, "
              f"{color}reward={reward:.4f}{reset}, "
              f"td_error={td_error:.6f}, new_Q={self.q_table[state][action]:.6f}, "
              f"target_Q2={target_value:.6f}")
        
        self._update_count = getattr(self, '_update_count', 0) + 1
        if self._update_count % 10 == 0:
            self.q_table2 = self.q_table.copy()     


    def initialize(self):
        """
        Called by Bot once the broker is ready.
        Restores awareness of any inherited open trades.
        """
        open_trades = self.broker.open_trades()
        if not open_trades:
            return

        # BUGFIX-9: only adopt trades the bot has no record of. Trades it placed itself
        # are already in self.open_trades, reloaded from the model file with their
        # action, state and tier. This used to overwrite them with the blank entry
        # below, so the bot never learned from a trade that stayed open across a
        # restart. The message now counts only the trades adopted here.
        adopted = [
            t for t in open_trades
            if t.pair in self.pair_index and t.id not in self.open_trades
        ]
        for t in adopted:
            self.open_trades[t.id] = {
                'pair': t.pair,
                'action': None,
                'state': 0,
                'entry_price': t.entry_price,
                'sl_tp_tier': 1
            }

        if adopted:
            print(f"✅ Adopted {len(adopted)} open trades the bot has no record of")

    def _load_model(self):
        try:
            with open(self.model_path, 'r') as f:
                data = json.load(f)
                loaded_table = np.array(data['q_table'])
                if loaded_table.shape != (self.n_states, self.n_actions):
                    return np.zeros((self.n_states, self.n_actions))
                self.open_trades = data.get('open_trades', {})
                self.trade_history = data.get('trade_history', {})  # ✅ Restore
                self.q_table2 = np.array(data.get('q_table2', self.q_table2)) 
                return loaded_table
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return np.zeros((self.n_states, self.n_actions))

    def save_model(self):
        if not hasattr(self, 'q_table') or not hasattr(self, 'open_trades'):
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        data = {
            'q_table': self.q_table.tolist(),
            'q_table2': self.q_table2.tolist(),
            'open_trades': self.open_trades,
            'trade_history': self.trade_history
        }
        with open(self.model_path, 'w') as f:
            json.dump(data, f, indent=2, default=_json_default)     

    def _get_correlation_feature(self, pairs):
        """
        Compute average pairwise correlation of last 20 candles.
        High correlation → agent should avoid adding to the same direction.
        """
        if len(pairs) < 2:
            return 0

        # Get recent returns for each pair
        returns = {}
        for pair in pairs[:4]:  # Limit to 4 pairs for speed
            try:
                df = self.market.candles(pair, 21)
                if df.shape[0] >= 20:
                    col = 'mid_close' if 'mid_close' in df.columns else 'c'
                    close = df[col].values
                    returns[pair] = np.diff(np.log(close))[-19:]
            except:
                continue
        
        if len(returns) < 2:
            return 0
        
        # Compute average pairwise correlation
        pair_names = list(returns.keys())
        correlations = []
        for i in range(len(pair_names)):
            for j in range(i + 1, len(pair_names)):
                r1, r2 = returns[pair_names[i]], returns[pair_names[j]]
                if len(r1) == len(r2) and np.std(r1) > 0 and np.std(r2) > 0:
                    correlations.append(np.corrcoef(r1, r2)[0, 1])
        
        if not correlations:
            return 0
        
        avg_corr = np.mean(correlations)
        
        # Bucket: -1 to +1 → 0,1,2,3,4
        # 0 = strongly negative, 2 = uncorrelated, 4 = strongly positive
        return int(np.clip((avg_corr + 1) / 0.5, 0, 4))    

    def _get_global_state(self, current_positions):
        # Only recompute if we're on a new candle
        now_min = int(self.market.now().timestamp() // 60)
        if hasattr(self, '_state_cache_time') and self._state_cache_time == now_min:
            return self._state_cache

        features = []
        for pair in self.pairs:
            ts = self.trade_settings.get(pair)
            if not ts:
                features.append(0)
                continue
            try:
                max_rows = ts.n_ma + 20
                df = self.market.candles(pair, max_rows)
                if df.shape[0] < 20:
                    features.append(0)
                    continue

                close = df['mid_close'].values if 'mid_close' in df.columns else df['c'].values
                # BUGFIX-7: candles have no 'h'/'l' columns, so high and low silently fell
                # back to close: ATR became close-to-close only, and the body feature
                # (f4 below) was always 2. Use the real mid high/low.
                high = df['mid_high'].values
                low = df['mid_low'].values
                open_ = df['mid_open'].values if 'mid_open' in df.columns else close

                ema9 = pd.Series(close).ewm(span=9).mean().iloc[-1]
                atr14 = self._calc_atr(high, low, close, 14)
                price_vs_ema = (close[-1] - ema9) / atr14 if atr14 > 0 else 0
                f1 = int(np.clip((price_vs_ema + 2) / 1.0, 0, 4))

                rsi14 = self._calc_rsi(close, 14)
                f2 = int(np.clip(rsi14 / 25.0, 0, 3))

                atr_norm = (atr14 / close[-1]) * 10000 if close[-1] > 0 else 0
                f3 = int(np.clip(atr_norm / 2.0, 0, 3))

                body = abs(close[-1] - open_[-1])
                range_ = high[-1] - low[-1]
                body_ratio = body / range_ if range_ > 0 else 0.5
                f4 = int(np.clip(body_ratio * 4, 0, 3))

                pair_feature = f1 * 81 + f2 * 27 + f3 * 9 + f4

                self.log_message(
                    f"State[{pair}]: ema_pos={f1}, rsi={f2}(rsi={rsi14:.0f}), "
                    f"vol={f3}(atr={atr14:.5f}), body={f4} → {pair_feature}", pair
                )
                features.append(pair_feature)

            except Exception as e:
                self.log_message(f"⚠️ State feature error {pair}: {e}", pair)
                features.append(0)

        corr_feature = self._get_correlation_feature(self.pairs)
        features.append(corr_feature)

        total_positions = len(current_positions)
        max_positions = len(self.pairs)
        exposure_ratio = int((total_positions / max_positions) * 4) if max_positions > 0 else 0
        h = 0
        for i, f in enumerate(features):
            h = (h * 31 + f * (i + 1)) % (2**32)
        h = (h + exposure_ratio * 17 + corr_feature * 7) % (2**32)
        state = h % self.n_states

        # Cache for this minute
        self._state_cache = state
        self._state_cache_time = now_min
        return state   

    def _calc_atr(self, high, low, close, period=14):
        if len(close) < period + 1:
            return 0.0
        
        # ✅ All arrays must be the same length (period elements)
        h = high[-period:]            # 14 elements
        l = low[-period:]             # 14 elements
        prev_c = close[-period-1:-1]  # 14 elements (previous closes)
        
        tr = np.maximum(
            h - l,
            np.maximum(abs(h - prev_c), abs(l - prev_c))
        )
        return np.mean(tr)   

    def _calc_rsi(self, close, period=14):
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))    

    def _move_sl_to_breakeven(self, pair):
        for t in self.broker.open_trades():
            if t.pair == pair:
                # Get current price to check direction
                quote = self.market.quotes()[pair]
                bid = quote.bid
                ask = quote.ask

                is_long = t.units > 0
                # BUGFIX-8: a long is closed by selling at the bid and a short by buying
                # at the ask, so that is the price to judge profit by. This used the
                # opposite side (ask for longs, bid for shorts), overstating profit by
                # the spread.
                current_price = bid if is_long else ask

                # Add half-spread so it's actually breakeven (not a tiny loss)
                spread = ask - bid
                be_price = t.entry_price + (spread / 2) if is_long else t.entry_price - (spread / 2)

                # ✅ Only move to breakeven if trade is in profit
                # BUGFIX-8: compare with the breakeven stop itself, not with the entry.
                # A long whose bid sat between its entry and be_price got a stop above
                # the bid, on the wrong side of the market: that closes the trade (or
                # gets rejected) instead of protecting it. Moving the be_price
                # calculation above this check is part of the same fix.
                if is_long and current_price <= be_price:
                    self.log_message(f"⏭️ BE skipped: {pair} not far enough in profit (long, bid={current_price:.5f} ≤ breakeven={be_price:.5f})", pair)
                    return
                if not is_long and current_price >= be_price:
                    self.log_message(f"⏭️ BE skipped: {pair} not far enough in profit (short, ask={current_price:.5f} ≥ breakeven={be_price:.5f})", pair)
                    return

                # ✅ Don't move SL backwards (wider)
                if t.stop_loss is not None:
                    if is_long and t.stop_loss >= t.entry_price:
                        return  # already at/past breakeven
                    if not is_long and t.stop_loss <= t.entry_price:
                        return  # already at/past breakeven

                try:
                    self.broker.set_stop_loss(t.id, be_price)
                    self.log_message(f"🛡️ SL→BE: {pair} @ {be_price:.5f} (entry={t.entry_price:.5f})", pair)
                except Exception as e:
                    self.log_message(f"❌ SL→BE failed: {pair} — {e}", pair)
                return

    def _trail_sl(self, pair):
        """Trail SL by 1 ATR from current price."""
        for t in self.broker.open_trades():
            if t.pair == pair:
                is_long = t.units > 0

                # Get current price and ATR
                quote = self.market.quotes()[pair]
                df = self.market.candles(pair, 20)
                if not df.empty:
                    col = 'mid_close' if 'mid_close' in df.columns else 'c'
                    close = df[col].values
                    high = df['mid_high'].values  # BUGFIX-7: was 'h', which never exists
                    low = df['mid_low'].values  # BUGFIX-7: was 'l', which never exists
                    atr = self._calc_atr(high, low, close, 14)

                    if is_long:
                        new_sl = quote.bid - atr  # Trail 1 ATR below bid
                    else:
                        new_sl = quote.ask + atr  # Trail 1 ATR above ask

                    # BUGFIX-2 (follow-on): now that stop moves really reach OANDA, a
                    # trail must only ever tighten the stop. On a losing trade, 1 ATR
                    # from the price is often further away than the current stop,
                    # which would add risk the position size never accounted for.
                    if t.stop_loss is not None and (
                        (is_long and new_sl <= t.stop_loss)
                        or (not is_long and new_sl >= t.stop_loss)
                    ):
                        self.log_message(f"⏭️ Trail skipped: {pair} {new_sl:.5f} would not tighten SL {t.stop_loss:.5f}", pair)
                        return

                    self.broker.set_stop_loss(t.id, new_sl)
                    self.log_message(f"📐 SL trailed: {pair} → {new_sl:.5f} (1 ATR)", pair)
                return

    def _close_trade(self, pair):
        """Close the open trade on this pair. Store P/L for reward calculation."""
        for t in self.broker.open_trades():
            if t.pair == pair:
                # ✅ Store P/L at close time (before the trade disappears)
                # BUG-11 (not fixed yet): account-currency P/L, not pips, unless the pair
                # is quoted in USD; see the note in get_decision's management block
                close_pl_pips = (t.unrealized_pl / abs(t.units)) / pair.pip_size

                # ✅ Store in open_trades so _calculate_reward can use it
                if t.id in self.open_trades:
                    self.open_trades[t.id]['manual_close_pl'] = close_pl_pips
                    self.open_trades[t.id]['manual_close_time'] = self.market.now().timestamp()

                self.broker.close_trade(t.id)
                self.log_message(
                    f"🚪 Trade closed manually: {pair} (P/L={close_pl_pips:+.1f}p)",
                    pair
                )
                return

    def _get_pair_features(self, pair, trade_settings):
        """Return raw (unhashed) features for a single pair."""
        try:
            max_rows = trade_settings.n_ma + 20
            df = self.market.candles(pair, max_rows)
            if df.shape[0] < 20:
                return {}

            close = df['mid_close'].values if 'mid_close' in df.columns else df['c'].values
            high = df['mid_high'].values  # BUGFIX-7: was 'h', which never exists
            low = df['mid_low'].values  # BUGFIX-7: was 'l', which never exists
            open_ = df['mid_open'].values if 'mid_open' in df.columns else close

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

    def get_decision(self, pair: CurrencyPair,
                     open_trades: list[OpenTrade]) -> TradeDecision | None:
        # Decay epsilon at the START of the decision
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._check_closed_trades(open_trades)
        state = self._get_global_state(open_trades)

        # --- Trade Probability Gate ---
        trade_probability = 0.985

        if self.rng.random() > trade_probability:
            return None

        pair_idx = self.pair_index.get(pair, 0)
        has_open = any(t.pair == pair for t in open_trades)

        if has_open:
            # ═══════════════════════════════════════════════
            # MANAGEMENT BLOCK (unchanged)
            # ═══════════════════════════════════════════════
            base_mgmt = pair_idx * 11 + 8

            trade_age_hours = 0.0
            unrealized_pl = 0.0
            trade_id_found = None
            for t in open_trades:
                if t.pair == pair:
                    trade_id_found = t.id
                    trade_age_hours = (self.market.now() - t.open_time).total_seconds() / 3600.0
                    # BUG-11 (not fixed yet): t.unrealized_pl is in the account currency
                    # (USD), but dividing it by units and pip size only gives pips for
                    # pairs quoted in USD. Elsewhere it is scaled by the quote-to-USD
                    # rate: about 1.2x on the CHF pairs, 1.3x on EUR_GBP, 0.7x on the CAD
                    # pairs and 0.006x on GBP_JPY. So profit_take_threshold,
                    # min_profit_to_close and the fading check all fire at the wrong
                    # profit. Likely fix: compute pips from prices instead, e.g.
                    # (quote.bid - t.entry_price) / pair.pip_size for a long. Same in
                    # _close_trade.
                    unrealized_pl = (t.unrealized_pl / abs(t.units)) / pair.pip_size
                    break

            if trade_id_found and trade_id_found in self.open_trades:
                try:
                    self.open_trades[trade_id_found]['peak_pl'] = max(self.open_trades[trade_id_found]['peak_pl'], unrealized_pl)
                    peak_pl = self.open_trades[trade_id_found]['peak_pl']
                except:
                    peak_pl = unrealized_pl
            else:
                peak_pl = unrealized_pl

            profit_fading = False
            if peak_pl > 0.5 and unrealized_pl < peak_pl * 0.5:
                profit_fading = True

            if profit_fading:
                time_bias = np.array([0.1, 0.1, 0.8])
            elif unrealized_pl > 0:
                if trade_age_hours < 0.1:
                    time_bias = np.array([0.2, 0.3, 0.5])
                elif trade_age_hours < 0.25:
                    time_bias = np.array([0.1, 0.4, 0.5])
                else:
                    time_bias = np.array([0.1, 0.1, 0.8])
            else:
                if trade_age_hours < 0.05:
                    time_bias = np.array([0.2, 0.7, 0.1])
                elif trade_age_hours < 0.1:
                    time_bias = np.array([0.1, 0.7, 0.2])
                elif trade_age_hours < 0.25:
                    time_bias = np.array([0.5, 0.2, 0.3])
                else:
                    time_bias = np.array([0.2, 0.0, 0.8])

            if self.rng.random() < self.epsilon:
                action_idx = int(self.rng.integers(base_mgmt, base_mgmt + 3))
            else:
                q_mgmt = self.q_table[state][base_mgmt:base_mgmt + 3]
                if q_mgmt.size == 0:
                    return None
                tau = 0.3
                exp_q = np.exp(q_mgmt / tau)
                probs = exp_q / np.sum(exp_q)
                probs = 0.5 * probs + 0.5 * time_bias
                probs = probs / np.sum(probs)
                best_local = int(self.rng.choice(3, p=probs))
                action_idx = base_mgmt + best_local

            mgmt_action = action_idx - base_mgmt
            self.log_message(
                f"🕐 MGMT: {pair} age={trade_age_hours*60:.0f}m "
                f"P/L={unrealized_pl:+.1f}p peak={peak_pl:+.1f}p "
                f"fading={'✅' if profit_fading else '❌'} "
                f"action={['breakeven','trail','close'][mgmt_action]}", pair
            )

            if mgmt_action == 0:
                self._move_sl_to_breakeven(pair)

            elif mgmt_action == 1:
                if unrealized_pl > 0 and trade_id_found and trade_id_found in self.open_trades:
                    self.open_trades[trade_id_found]['trail_in_profit'] = True
                    self.open_trades[trade_id_found]['trail_pl'] = unrealized_pl

                    if unrealized_pl >= self.profit_take_threshold:
                        self.open_trades[trade_id_found]['trail_in_profit'] = False
                        self.open_trades[trade_id_found]['manual_close_pl'] = unrealized_pl
                        self._close_trade(pair)
                        return None

                self._trail_sl(pair)

            elif mgmt_action == 2:
                if unrealized_pl < 0:
                    if trade_id_found and trade_id_found in self.open_trades:
                        self.open_trades[trade_id_found]['mgmt_action_at_close'] = 2
                    self._close_trade(pair)
                else:
                    if profit_fading:
                        if trade_id_found and trade_id_found in self.open_trades:
                            self.open_trades[trade_id_found]['manual_close_pl'] = unrealized_pl
                            self.open_trades[trade_id_found]['mgmt_action_at_close'] = 2
                        self._close_trade(pair)
                    else:
                        if trade_id_found and trade_id_found in self.open_trades:
                            self.open_trades[trade_id_found]['trail_in_profit'] = True
                            self.open_trades[trade_id_found]['trail_pl'] = unrealized_pl

                            if trade_age_hours < 0.25 or unrealized_pl < self.min_profit_to_close:
                                self._trail_sl(pair)
                            else:
                                self.open_trades[trade_id_found]['trail_in_profit'] = False
                                self.open_trades[trade_id_found]['manual_close_pl'] = unrealized_pl
                                self.open_trades[trade_id_found]['mgmt_action_at_close'] = 2
                                self._close_trade(pair)
                                return None

            return None

        # ═══════════════════════════════════════════════
        # ENTRY BLOCK — signal-constrained
        # ═══════════════════════════════════════════════

        # ✅ GET SIGNAL FROM TECH MANAGER
        pair_settings = self.trade_settings[pair]
        decision_row = get_trade_decision(self.market, pair, pair_settings, self.log_message)
        if decision_row is None:
            return None

        signal = decision_row.signal  # Direction.LONG, Direction.SHORT, or Direction.NONE
        
        if signal == Direction.NONE:
            self.log_message(f"⏭️ NO TRADE {pair}: signal=NONE "
                            f"(mid_c={decision_row.mid_c:.5f})", pair)
            return None

        # ✅ CONSTRAIN TO 4 VALID TIERS
        # BUGFIX-3: each pair owns 11 Q-table columns (BUY tiers 0-3, SELL tiers 4-7,
        # management 8-10, see base_mgmt), but entries used pair_idx * 8, so from the
        # second pair on, entry actions shared cells with the previous pair's
        # management actions
        base = pair_idx * 11
        offset = 0 if signal == Direction.LONG else 4

        if self.rng.random() < self.epsilon:
            local = int(self.rng.integers(4))
        else:
            q_values = self.q_table[state][base + offset : base + offset + 4]
            if q_values.size == 0:
                self.log_message(f"Q-table slice empty! state={state}, base={base+offset}, shape={self.q_table.shape}", pair)
                return None
            tau = 0.65
            exp_q = np.exp(q_values / tau)
            probs = exp_q / np.sum(exp_q)
            local = int(self.rng.choice(4, p=probs))

        action_idx = base + offset + local
        sl_tp_tier = local  # 0-3

        # --- Get Price ---
        quote = self.market.quotes()[pair]
        bid = quote.bid
        ask = quote.ask

        entry_price = ask if signal == Direction.LONG else bid
        spread = ask - bid

        # --- PIP-SCALED SL/TP ---
        pip_location = pair.pip_size

        max_allowed_spread = 12 * pip_location
        if spread > max_allowed_spread:
            self.log_message(f"⚠️ Spread too wide: {spread:.5f} ({spread/pip_location:.1f} pips). Skipping.", pair)
            return None

        pip_distances = {
            0: (1.5, 3.5),
            1: (2.0, 4.0),
            2: (3.0, 4.5),
            3: (5.0, 7.0),
        }
        sl_pips, tp_pips = pip_distances[sl_tp_tier]
        sl_distance = sl_pips * pip_location
        tp_distance = tp_pips * pip_location

        min_sl = spread * 3.0
        sl_distance = max(sl_distance, min_sl)

        if tp_distance <= sl_distance:
            tp_distance = sl_distance * 2

        sl = entry_price - sl_distance if signal == Direction.LONG else entry_price + sl_distance
        tp = entry_price + tp_distance if signal == Direction.LONG else entry_price - tp_distance

        # VALIDATION
        if signal == Direction.LONG and sl >= entry_price:
            self.log_message(f"❌ SL ({sl:.5f}) >= entry ({entry_price:.5f}) for BUY. Skipping.", pair)
            return None
        if signal == Direction.SHORT and sl <= entry_price:
            self.log_message(f"❌ SL ({sl:.5f}) <= entry ({entry_price:.5f}) for SELL. Skipping.", pair)
            return None
        if signal == Direction.LONG and tp <= entry_price:
            self.log_message(f"❌ TP ({tp:.5f}) <= entry ({entry_price:.5f}) for BUY. Skipping.", pair)
            return None
        if signal == Direction.SHORT and tp >= entry_price:
            self.log_message(f"❌ TP ({tp:.5f}) >= entry ({entry_price:.5f}) for SELL. Skipping.", pair)
            return None

        self.log_message(
            f"Q-Action: State={state}, Signal={signal}, Tier={sl_tp_tier}, "
            f"(SL={sl_distance:.5f}/{sl_pips}pips, TP={tp_distance:.5f}/{tp_pips}pips), "
            f"Spread={spread:.5f}, Entry={entry_price:.5f}, "
            f"ε={self.epsilon:.3f}, "
            f"Q={[f'{v:.4f}' for v in self.q_table[state][base+offset:base+offset+4]]}", pair
        )

        # --- Calculate Gain from Candle Data ---
        ts = self.trade_settings.get(pair)
        gain = 0.0
        if ts:
            max_rows = ts.n_ma + 20
            df = self.market.candles(pair, max_rows)
            if not df.empty:
                col = 'mid_close' if 'mid_close' in df.columns else 'c'
                mid_c = df[col].iloc[-1]
                if len(df) >= ts.n_ma:
                    ma = df[col].iloc[-ts.n_ma:].mean()
                    gain = abs(mid_c - ma)

        decision = TradeDecision(
            pair=pair, signal=signal, sl=sl, tp=tp, gain=gain,
            spread=spread, mid_c=(ask + bid) / 2, time=self.market.now()
        )
        decision.details = {
            'entry_price': entry_price,
            'sl_tp_tier': sl_tp_tier,
            'state': state,
            'state_features': self._get_pair_features(pair, pair_settings),
            'open_trades_count': len(open_trades),
            'exposure_ratio': len(open_trades) / len(self.pairs) if self.pairs else 0,
            'corr_feature': self._get_correlation_feature(self.pairs),
        }
        decision.spread = spread
        self.log_message(f"Place Trade: {decision} | Tier={sl_tp_tier}", pair)

        return decision

    def on_trade_placed(self, trade_id: TradeId, decision: TradeDecision) -> None:
        details = decision.details
        self.open_trades[str(trade_id)] = {
            'pair': decision.pair,
            'action': decision.signal,
            'state': details['state'],
            'entry_price': details['entry_price'],
            'entry_time': self.market.now().timestamp(),
            'sl_price': decision.sl,
            'tp_price': decision.tp,
            'sl_tp_tier': details['sl_tp_tier'],
            'peak_pl': 0.0,
            'trail_in_profit': False,
            'trail_pl': 0.0,
            # ✅ Raw features for supervised training
            'state_features': details['state_features'] or {},
            'open_trades_count': details['open_trades_count'],
            'exposure_ratio': details['exposure_ratio'],
            'corr_feature': details['corr_feature'],
            'spread': decision.spread,
        }

    def _calculate_reward(self, trade_id, trade_info):
        """Reward for microtrading. Rebalanced for stable Q-learning convergence."""
        pair = trade_info['pair']
        entry = trade_info['entry_price']
        action = trade_info['action']
        sl_price = trade_info.get('sl_price')
        # tp_price = trade_info.get('tp_price')
        # BUGFIX-1: trade_id used to come from trade_info.get('trade_id'), a key
        # on_trade_placed never stores, so the broker was never asked for the close
        # price and every non-manual close (take profits included) was scored as if
        # it had hit the stop loss. The caller now passes the id it tracks.

        if action is None:
            print(f"⏭️ Skipping inherited trade (no action data): {pair}")
            return 0.0

        # --- Holding time ---
        now = self.market.now().timestamp()
        entry_time = trade_info.get('entry_time', now)
        hours_held = max((now - entry_time) / 3600.0, 2.0 / 60.0)

        pip_location = CurrencyPair(pair).pip_size

        # --- Resolve close price (priority: API > manual > SL fallback) ---
        close_price = None
        close_source = None

        if trade_id:
            closed = self.broker.closed_trade(trade_id)
            if closed is not None:
                close_price = closed.close_price
                close_source = "API"

        if close_price is None:
            manual_close_pl = trade_info.get('manual_close_pl')
            if manual_close_pl is not None:
                if action == Direction.LONG:
                    close_price = entry + (manual_close_pl * pip_location)
                else:
                    close_price = entry - (manual_close_pl * pip_location)
                close_source = "manual"

        if close_price is None:
            if sl_price is not None:
                close_price = sl_price
                close_source = "SL_fallback"
            else:
                close_price = entry
                close_source = "unknown"
                print(f"⚠️ No close price or SL for {pair}, defaulting to entry (0 pips)")

        # --- P/L in pips ---
        if action == Direction.LONG:
            profit_pips = (close_price - entry) / pip_location
        else:
            profit_pips = (entry - close_price) / pip_location

        # --- FIX 6: Subtract spread cost (adjust per instrument) ---
        spread_pips = 0.7  # ~0.5-1.0 typical
        net_pips = profit_pips - spread_pips

        # --- FIX 3: Rebalanced base reward (mild asymmetry, higher cap) ---
        if net_pips > 0:
            # Cap at 10 pips instead of 5 — still scalper-oriented but not too tight
            effective_pips = min(net_pips, 10.0)
            base_reward = effective_pips * 1.0
        else:
            # 1.2× loss multiplier instead of 2× — still loss-averse but not extreme
            base_reward = net_pips * 1.2

        # --- Speed bonus: proportional, not a flat cliff ---
        speed_bonus = 0.0
        if net_pips > 0:
            if hours_held < 0.1:       # < 6 min
                speed_bonus = 1.5
            elif hours_held < 0.25:    # < 15 min
                speed_bonus = 0.5

        # --- FIX 5: Gentler holding penalty ---
        holding_penalty = 0.0
        if hours_held > 0.5:  # start after 30 min (was 45 min)
            holding_penalty = (hours_held - 0.5) * 5.0  # slope 5 (was 30)
            # At 1h: -2.5 | At 2h: -7.5 | At 3h: -12.5

        # --- FIX 1: Escalating repeat penalty with cooldown ---
        repeat_key = f"{pair}_{action}"
        history_entry = self.trade_history.get(repeat_key, (0, 0.0))
        repeat_count, last_trade_time = history_entry

        # Cooldown: reset counter if > 30 min since last trade on this combo
        if now - last_trade_time > 1800:
            repeat_count = 0

        # Escalate: 0, 0.3, 0.6, 0.9, 1.2... capped at 2.0
        repeat_penalty = min(repeat_count * 0.3, 2.0)

        self.trade_history[repeat_key] = (repeat_count + 1, now)

        # Prune history
        if len(self.trade_history) > 10000:
            # Keep only entries where last_trade_time is within 1 hour
            self.trade_history = {
                k: v for k, v in self.trade_history.items()
                if now - v[1] < 3600
            }

        # --- FIX 7: Capped regret penalty ---
        trail_in_profit = trade_info.get('trail_in_profit', False)
        trail_pl = trade_info.get('trail_pl', 0.0)
        regret_penalty = 0.0
        if trail_in_profit and net_pips < 0:
            regret_penalty = min((trail_pl - net_pips) * 1.0, 2.0)  # cap at 2.0
            if regret_penalty > 0:
                print(f"😞 REGRET: {pair} trailed at +{trail_pl:.1f}p, "
                    f"closed at {net_pips:+.1f}p. Penalty={regret_penalty:.2f}")

        # --- FIX 2: Directional balance — REMOVED from reward ---
        # Instead, fold long/short count into your state representation:
        #   e.g., state_features = [..., long_count, short_count]
        # The Q-table will learn the interaction naturally.

        # --- Final reward ---
        reward = base_reward + speed_bonus - holding_penalty - repeat_penalty - regret_penalty

        # Clamp (tighter range for stable learning)
        reward = max(min(reward, 20.0), -20.0)

        # --- Store metadata ---
        trade_info['profit_pips'] = round(net_pips, 4)
        trade_info['close_reason'] = close_source

        # --- Debug ---
        print(f"📊 REWARD: {pair} {action} | entry={entry:.5f} close={close_price:.5f} "
            f"[{close_source}] | net_pips={net_pips:+.2f} (gross={profit_pips:+.2f}) "
            f"| hours={hours_held:.2f} | base={base_reward:.2f} bonus={speed_bonus:.1f} "
            f"hold_pen={holding_penalty:.2f} repeat_pen={repeat_penalty:.2f} "
            f"regret={regret_penalty:.2f} | FINAL={reward:.4f}")

        return reward                         

    def _check_closed_trades(self, open_trades):
        if not self.open_trades:
            return

        open_ids = {t.id for t in open_trades}

        for trade_id, info in list(self.open_trades.items()):
            trade_id_str = str(trade_id)
            if trade_id_str not in open_ids:
                # BUGFIX-9: a trade adopted in initialize() has no recorded action, state
                # or tier, so there is nothing to learn from it. When it closed, it used
                # to get a zero-reward Q-update on an action the bot never took (state 0,
                # SELL tier 1) and a fake row in the training data. Now it is forgotten.
                if info['action'] is None:
                    self.log_message(f"Adopted trade {trade_id} closed, nothing to learn from it", info['pair'])
                    del self.open_trades[trade_id]
                    continue

                # 90-SECOND MINIMUM AGE
                # BUGFIX-4: this check used to sit inside the try below, whose finally
                # deletes the trade, so `continue` dropped every trade that closed
                # within 90 s instead of retrying it on a later call
                entry_time = info.get('entry_time', 0)
                age_seconds = self.market.now().timestamp() - entry_time if entry_time else 9999

                if age_seconds < 90:
                    continue

                try:
                    # 1. Calculate Reward
                    reward = self._calculate_reward(trade_id, info)  # BUGFIX-1: pass the id

                    # 2. Get Next State
                    next_state = self._get_global_state(open_trades)

                    if next_state >= self.q_table.shape[0]:
                        self.log_message(f"ERROR: next_state {next_state} out of bounds!", "main")
                        next_state = 0

                    # 3. Map Action to Index (FIX: 0-indexed tier to avoid collision)
                    pair_idx = self.pair_index.get(info['pair'], 0)
                    is_buy = 0 if info['action'] == Direction.LONG else 4
                    # BUGFIX-3: sl_tp_tier is already stored 0-3, so the old "- 1" updated
                    # the neighbouring action (tier 0 of the first pair hit index -1, the
                    # table's last column), and the pair stride is 11 as in get_decision
                    tier = info.get('sl_tp_tier', 1)
                    action_idx = (pair_idx * 11) + is_buy + tier

                    if action_idx >= self.q_table.shape[1]:
                        self.log_message(f"ERROR: action_idx {action_idx} out of bounds (max={self.q_table.shape[1]-1})!", "main")
                        # BUGFIX-4 (same cause): the finally below already deletes the
                        # trade, so deleting it here as well raised KeyError
                        continue

                    # 4. Update Q-value (entry decision)
                    self.update_q_value(info['state'], action_idx, reward, next_state)

                    # 4b. Update Q-value (management decision) — FIX: use mgmt_state
                    mgmt_action_taken = info.get('mgmt_action_at_close')
                    if mgmt_action_taken is not None:
                        mgmt_idx = (pair_idx * 11 + 8) + mgmt_action_taken
                        if mgmt_idx < self.q_table.shape[1]:
                            mgmt_reward = reward * 0.5
                            # Use state at time of management decision, not entry state
                            mgmt_state = info.get('mgmt_state', info['state'])
                            self.update_q_value(mgmt_state, mgmt_idx, mgmt_reward, next_state)

                    # 5. Record training data for supervised model
                    # BUGFIX-3: tier is no longer shifted down, so record it as-is (still 0-3)
                    self._record_training_data(info, reward, next_state, tier, age_seconds, open_trades)

                    self.log_message(
                        f"Q-UPDATE: trade={trade_id} pair={info['pair']} "
                        f"state={info['state']} action={action_idx} reward={reward:.2f} "
                        f"age={age_seconds:.0f}s",
                        info['pair']
                    )

                except Exception as e:
                    self.log_message(f"ERROR processing trade {trade_id}: {e}", "main")

                finally:
                    del self.open_trades[trade_id]
 