import numpy as np
import pandas as pd 
import os
import json
from bot.strategy_base import TradingStrategy
from models.trade_decision import TradeDecision
from infrastructure.instrument_collection import instrumentCollection as ic  
from bot.technicals_manager import get_trade_decision
import constants.defs as defs
import time


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")   

class PortfolioQLearningStrategy(TradingStrategy):
    def __init__(self, pairs_list, model_path="bot/models/q_table_portfolio.json"):
        self.pairs = pairs_list
        self.n_states = 1000
        self.n_actions = len(pairs_list) * 11  

        self.profit_take_threshold = 1.5     
                
        self.epsilon = 0.2
        self.epsilon_decay = 0.999  # Decay slowly
        self.epsilon_min = 0.05    # Floor
         
        self.gamma = 0.95
        self.alpha = 0.3
        self.pair_index = {pair: i for i, pair in enumerate(self.pairs)}
        self.model_path = model_path
        self.open_trades = {}
        # Dont move this 
        self.q_table2 = np.zeros((self.n_states, self.n_actions)) 
        self.q_table = self._load_model()
        self.trade_history = {}
        self.training_data_path = "bot/models/training_data.jsonl"
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)   

    def _record_training_data(self, info, reward, next_state, tier, age_seconds, open_trades):
        """Append one trade's features + outcome to the training data file (JSONL)."""
        pair = info['pair']

        # Reconstruct raw features from the state that was stored at entry
        # (or recompute if you prefer — but stored is faster and consistent)
        state_features = info.get('state_features', {})

        record = {
            # --- Timestamp ---
            'close_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'entry_time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(info.get('entry_time', time.time()))),
            'age_seconds': round(age_seconds, 1),
            'hour_of_day': time.gmtime(info.get('entry_time', time.time())).tm_hour,

            # --- Trade identity ---
            'pair': pair,
            'direction': 1 if info['action'] == defs.BUY else -1,
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


    def initialize(self, api):
        """
        Called by Bot after API is ready.
        Restores awareness of any inherited open trades.
        """
        open_trades = api.get_open_trades()
        if not open_trades:
            return
        
        for t in open_trades:
            # ✅ Handle both object and dict access
            if isinstance(t, dict):
                trade_id = str(t.get('id', ''))
                pair = t.get('instrument', '')
                price = float(t.get('price', 0))
            else:
                # OpenTrade object: use attribute access
                trade_id = str(getattr(t, 'id', ''))
                pair = getattr(t, 'instrument', '')
                price = float(getattr(t, 'price', 0))
            
            if pair in self.pair_index:
                self.open_trades[trade_id] = {
                    'pair': pair,
                    'action': None,
                    'state': 0,
                    'entry_price': price,
                    'sl_tp_tier': 1
                }
        
        if self.open_trades:
            print(f"✅ Restored {len(self.open_trades)} inherited open trades")
            self.save_model()   

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

    def _get_correlation_feature(self, api, pairs):
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
                df = api.get_candles_df(pair, count=21, granularity="M1")
                if df is not None and df.shape[0] >= 20:
                    col = 'mid_c' if 'mid_c' in df.columns else 'c'
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

    def _get_global_state(self, api, trade_settings_dict, current_positions, log_message=None):
        # Only recompute if we're on a new candle
        now_min = int(time.time() // 60)
        if hasattr(self, '_state_cache_time') and self._state_cache_time == now_min:
            return self._state_cache

        features = []
        for pair in self.pairs:
            ts = trade_settings_dict.get(pair)
            if not ts:
                features.append(0)
                continue
            try:
                max_rows = ts.n_ma + 20
                df = api.get_candles_df(pair, count=max_rows, granularity="M1")
                if df is None or df.shape[0] < 20:
                    features.append(0)
                    continue

                close = df['mid_c'].values if 'mid_c' in df.columns else df['c'].values
                high = df['h'].values if 'h' in df.columns else close
                low = df['l'].values if 'l' in df.columns else close
                open_ = df['mid_o'].values if 'mid_o' in df.columns else close

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

                if log_message:
                    log_message(
                        f"State[{pair}]: ema_pos={f1}, rsi={f2}(rsi={rsi14:.0f}), "
                        f"vol={f3}(atr={atr14:.5f}), body={f4} → {pair_feature}", pair
                    )
                features.append(pair_feature)

            except Exception as e:
                log_message(f"⚠️ State feature error {pair}: {e}", pair)
                features.append(0)

        corr_feature = self._get_correlation_feature(api, self.pairs)
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

    def _move_sl_to_breakeven(self, api, pair, log_message):
        open_trades = api.get_open_trades()
        for t in open_trades:
            inst = t.get('instrument') if isinstance(t, dict) else getattr(t, 'instrument', '')
            if inst == pair:
                trade_id = t.get('id') if isinstance(t, dict) else getattr(t, 'id', '')
                entry_price = float(t.get('price', 0) if isinstance(t, dict) else getattr(t, 'price', 0))
                units = float(t.get('currentUnits', 1) if isinstance(t, dict) else getattr(t, 'currentUnits', 1))

                # Get current price to check direction
                prices = api.get_prices([pair])
                if not prices:
                    return
                p = prices[0]
                bid = float(p['bids'][0]['price']) if isinstance(p, dict) else float(p.bid)
                ask = float(p['asks'][0]['price']) if isinstance(p, dict) else float(p.ask)

                is_long = units > 0
                current_price = ask if is_long else bid

                # ✅ Only move to breakeven if trade is in profit
                if is_long and current_price <= entry_price:
                    log_message(f"⏭️ BE skipped: {pair} not in profit (long, px={current_price:.5f} ≤ entry={entry_price:.5f})", pair)
                    return
                if not is_long and current_price >= entry_price:
                    log_message(f"⏭️ BE skipped: {pair} not in profit (short, px={current_price:.5f} ≥ entry={entry_price:.5f})", pair)
                    return

                # ✅ Don't move SL backwards (wider)
                current_sl = t.get('stopLoss', {}).get('price') if isinstance(t, dict) else getattr(t, 'stopLoss', {}).get('price')
                if current_sl:
                    current_sl = float(current_sl)
                    if is_long and current_sl >= entry_price:
                        return  # already at/past breakeven
                    if not is_long and current_sl <= entry_price:
                        return  # already at/past breakeven

                # Add half-spread so it's actually breakeven (not a tiny loss)
                spread = ask - bid
                be_price = entry_price + (spread / 2) if is_long else entry_price - (spread / 2)

                try:
                    api.update_trade_sl(trade_id, be_price)
                    log_message(f"🛡️ SL→BE: {pair} @ {be_price:.5f} (entry={entry_price:.5f})", pair)
                except Exception as e:
                    log_message(f"❌ SL→BE failed: {pair} — {e}", pair)
                return   

    def _trail_sl(self, api, pair, log_message):
        """Trail SL by 1 ATR from current price."""
        open_trades = api.get_open_trades()
        for t in open_trades:
            inst = t.get('instrument') if isinstance(t, dict) else getattr(t, 'instrument', '')
            if inst == pair:
                trade_id = t.get('id') if isinstance(t, dict) else getattr(t, 'id', '')
                units = float(t.get('currentUnits', 0) if isinstance(t, dict) else getattr(t, 'currentUnits', 0))
                is_long = units > 0
                
                # Get current price and ATR
                prices = api.get_prices([pair])
                df = api.get_candles_df(pair, count=20, granularity="M1")
                if prices and df is not None:
                    col = 'mid_c' if 'mid_c' in df.columns else 'c'
                    close = df[col].values
                    high = df['h'].values if 'h' in df.columns else close
                    low = df['l'].values if 'l' in df.columns else close
                    atr = self._calc_atr(high, low, close, 14)
                    
                    p = prices[0]
                    bid = float(p.bid if isinstance(p.bid, (int, float)) else p.bid.price)
                    ask = float(p.ask if isinstance(p.ask, (int, float)) else p.ask.price)
                    
                    if is_long:
                        new_sl = bid - atr  # Trail 1 ATR below bid
                    else:
                        new_sl = ask + atr  # Trail 1 ATR above ask
                    
                    api.update_trade_sl(trade_id, new_sl)
                    log_message(f"📐 SL trailed: {pair} → {new_sl:.5f} (1 ATR)", pair)
                return

    def _close_trade(self, api, pair, log_message):
        """Close the open trade on this pair. Store P/L for reward calculation."""
        open_trades = api.get_open_trades()
        for t in open_trades:
            inst = t.get('instrument') if isinstance(t, dict) else getattr(t, 'instrument', '')
            if inst == pair:
                trade_id = str(t.get('id') if isinstance(t, dict) else getattr(t, 'id', ''))

                # ✅ Store P/L at close time (before the trade disappears)
                try:
                    pl_raw = t.get('unrealizedPL', '0') if isinstance(t, dict) else getattr(t, 'unrealizedPL', '0')
                    instrument = ic.instruments_dict[pair]
                    pip_location = float(instrument.pipLocation)
                    units = float(t.get('currentUnits', 1) if isinstance(t, dict) else getattr(t, 'currentUnits', 1))
                    close_pl_pips = (float(pl_raw) / abs(units)) / pip_location
                except:
                    close_pl_pips = 0.0

                # ✅ Store in open_trades so _calculate_reward can use it
                if trade_id in self.open_trades:
                    self.open_trades[trade_id]['manual_close_pl'] = close_pl_pips
                    self.open_trades[trade_id]['manual_close_time'] = time.time()

                api.close_trade(trade_id)
                log_message(
                    f"🚪 Trade closed manually: {pair} (P/L={close_pl_pips:+.1f}p)",
                    pair
                )
                return   

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

    def get_decision(self, candle_time, pair, granularity, api, trade_settings, log_message, open_trades=None):
        if open_trades is None:
            open_trades = api.get_open_trades()

        # Decay epsilon at the START of the decision
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._check_closed_trades(api, log_message, trade_settings, open_trades)
        state = self._get_global_state(api, trade_settings, open_trades, log_message)

        # --- Trade Probability Gate ---
        trade_probability = 0.95

        if np.random.random() > trade_probability:
            return None

        pair_idx = self.pair_index.get(pair, 0)
        has_open = any(
            (t.get('instrument') if isinstance(t, dict) else getattr(t, 'instrument', '')) == pair
            for t in open_trades
        )

        if has_open:
            # ═══════════════════════════════════════════════
            # MANAGEMENT BLOCK (unchanged)
            # ═══════════════════════════════════════════════
            base_mgmt = pair_idx * 11 + 8

            trade_age_hours = 0.0
            unrealized_pl = 0.0
            trade_id_found = None
            for t in open_trades:
                inst = t.get('instrument') if isinstance(t, dict) else getattr(t, 'instrument', '')
                if inst == pair:
                    trade_id_found = str(t.get('id') if isinstance(t, dict) else getattr(t, 'id', ''))
                    open_time_str = t.get('openTime') if isinstance(t, dict) else getattr(t, 'openTime', None)
                    if open_time_str:
                        try:
                            from datetime import datetime, timezone
                            ots = open_time_str.replace('Z', '+00:00')
                            if '.' in ots:
                                base_t, frac = ots.split('.', 1)
                                ots = f"{base_t}.{frac[:6]}"
                            open_time = datetime.fromisoformat(ots)
                            trade_age_hours = (datetime.now(timezone.utc) - open_time).total_seconds() / 3600.0
                        except:
                            trade_age_hours = 0.0

                    pl_raw = t.get('unrealizedPL', '0') if isinstance(t, dict) else getattr(t, 'unrealizedPL', '0')
                    try:
                        instrument = ic.instruments_dict[pair]
                        pip_location = float(instrument.pipLocation)
                        units = float(t.get('currentUnits', 1) if isinstance(t, dict) else getattr(t, 'currentUnits', 1))
                        unrealized_pl = (float(pl_raw) / abs(units)) / pip_location
                    except:
                        unrealized_pl = 0.0
                    break

            if trade_id_found and trade_id_found in self.open_trades:
                try:
                    if unrealized_pl > self.open_trades[trade_id_found]['peak_pl']:
                        self.open_trades[trade_id_found]['peak_pl'] = unrealized_pl
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

            if np.random.random() < self.epsilon:
                action_idx = np.random.randint(base_mgmt, base_mgmt + 3)
            else:
                q_mgmt = self.q_table[state][base_mgmt:base_mgmt + 3]
                if q_mgmt.size == 0:
                    return None
                tau = 0.3
                exp_q = np.exp(q_mgmt / tau)
                probs = exp_q / np.sum(exp_q)
                probs = 0.5 * probs + 0.5 * time_bias
                probs = probs / np.sum(probs)
                best_local = np.random.choice(3, p=probs)
                action_idx = base_mgmt + best_local

            mgmt_action = action_idx - base_mgmt
            log_message(
                f"🕐 MGMT: {pair} age={trade_age_hours*60:.0f}m "
                f"P/L={unrealized_pl:+.1f}p peak={peak_pl:+.1f}p "
                f"fading={'✅' if profit_fading else '❌'} "
                f"action={['breakeven','trail','close'][mgmt_action]}", pair
            )

            if mgmt_action == 0:
                self._move_sl_to_breakeven(api, pair, log_message)

            elif mgmt_action == 1:
                if unrealized_pl > 0 and trade_id_found and trade_id_found in self.open_trades:
                    self.open_trades[trade_id_found]['trail_in_profit'] = True
                    self.open_trades[trade_id_found]['trail_pl'] = unrealized_pl

                    if unrealized_pl >= self.profit_take_threshold:
                        self.open_trades[trade_id_found]['trail_in_profit'] = False
                        self.open_trades[trade_id_found]['manual_close_pl'] = unrealized_pl
                        self._close_trade(api, pair, log_message)
                        return None

                self._trail_sl(api, pair, log_message)

            elif mgmt_action == 2:
                if unrealized_pl < 0:
                    if trade_id_found and trade_id_found in self.open_trades:
                        self.open_trades[trade_id_found]['mgmt_action_at_close'] = 2
                    self._close_trade(api, pair, log_message)
                else:
                    if profit_fading:
                        if trade_id_found and trade_id_found in self.open_trades:
                            self.open_trades[trade_id_found]['manual_close_pl'] = unrealized_pl
                            self.open_trades[trade_id_found]['mgmt_action_at_close'] = 2
                        self._close_trade(api, pair, log_message)
                    else:
                        if trade_id_found and trade_id_found in self.open_trades:
                            self.open_trades[trade_id_found]['trail_in_profit'] = True
                            self.open_trades[trade_id_found]['trail_pl'] = unrealized_pl

                            if trade_age_hours < 0.25 or unrealized_pl < self.min_profit_to_close:
                                self._trail_sl(api, pair, log_message)
                            else:
                                self.open_trades[trade_id_found]['trail_in_profit'] = False
                                self.open_trades[trade_id_found]['manual_close_pl'] = unrealized_pl
                                self.open_trades[trade_id_found]['mgmt_action_at_close'] = 2
                                self._close_trade(api, pair, log_message)
                                return None

            return None

        # ═══════════════════════════════════════════════
        # ENTRY BLOCK — signal-constrained
        # ═══════════════════════════════════════════════

        # ✅ GET SIGNAL FROM TECH MANAGER
        pair_settings = trade_settings.get(pair) if isinstance(trade_settings, dict) else trade_settings
        decision_row = get_trade_decision(candle_time, pair, granularity, api, pair_settings, log_message)   
        if decision_row is None:
            return None

        signal = decision_row.signal  # defs.BUY, defs.SELL, or defs.NONE
        
        if signal == defs.NONE:
            log_message(f"⏭️ NO TRADE {pair}: signal=NONE "
                            f"(mid_c={decision_row.mid_c:.5f})", pair)
            return None

        # ✅ CONSTRAIN TO 4 VALID TIERS
        base = pair_idx * 8
        offset = 0 if signal == defs.BUY else 4

        if np.random.random() < self.epsilon:
            local = np.random.randint(4)
        else:
            q_values = self.q_table[state][base + offset : base + offset + 4]
            if q_values.size == 0:
                log_message(f"Q-table slice empty! state={state}, base={base+offset}, shape={self.q_table.shape}", pair)
                return None
            tau = 0.65
            exp_q = np.exp(q_values / tau)
            probs = exp_q / np.sum(exp_q)
            local = np.random.choice(4, p=probs)

        action_idx = base + offset + local
        sl_tp_tier = local  # 0-3

        # --- Get Price ---
        prices = api.get_prices([pair])
        if not prices:
            log_message("No price", pair)
            return None
        p = prices[0]
        try:
            if isinstance(p, dict):
                bid = float(p['bids'][0]['price'])
                ask = float(p['asks'][0]['price'])
            else:
                first_bid = p.bid
                first_ask = p.ask
                if isinstance(first_bid, (int, float)):
                    bid = float(first_bid)
                    ask = float(first_ask)
                elif isinstance(first_bid, dict):
                    bid = float(first_bid['price'])
                    ask = float(first_ask['price'])
                elif hasattr(first_bid, 'price'):
                    bid = float(first_bid.price)
                    ask = float(first_ask.price)
                else:
                    raise TypeError(f"Unknown price format: {type(first_bid)}")
        except Exception as e:
            log_message(f"fail parse price: {e}", pair)
            return None

        entry_price = ask if signal == defs.BUY else bid
        spread = ask - bid

        # --- PIP-SCALED SL/TP ---
        instrument = ic.instruments_dict[pair]
        pip_location = float(instrument.pipLocation)

        max_allowed_spread = 12 * pip_location
        if spread > max_allowed_spread:
            log_message(f"⚠️ Spread too wide: {spread:.5f} ({spread/pip_location:.1f} pips). Skipping.", pair)
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

        sl = entry_price - sl_distance if signal == defs.BUY else entry_price + sl_distance
        tp = entry_price + tp_distance if signal == defs.BUY else entry_price - tp_distance

        # VALIDATION
        if signal == defs.BUY and sl >= entry_price:
            log_message(f"❌ SL ({sl:.5f}) >= entry ({entry_price:.5f}) for BUY. Skipping.", pair)
            return None
        if signal == defs.SELL and sl <= entry_price:
            log_message(f"❌ SL ({sl:.5f}) <= entry ({entry_price:.5f}) for SELL. Skipping.", pair)
            return None
        if signal == defs.BUY and tp <= entry_price:
            log_message(f"❌ TP ({tp:.5f}) <= entry ({entry_price:.5f}) for BUY. Skipping.", pair)
            return None
        if signal == defs.SELL and tp >= entry_price:
            log_message(f"❌ TP ({tp:.5f}) >= entry ({entry_price:.5f}) for SELL. Skipping.", pair)
            return None

        log_message(
            f"Q-Action: State={state}, Signal={signal}, Tier={sl_tp_tier}, "
            f"(SL={sl_distance:.5f}/{sl_pips}pips, TP={tp_distance:.5f}/{tp_pips}pips), "
            f"Spread={spread:.5f}, Entry={entry_price:.5f}, "
            f"ε={self.epsilon:.3f}, "
            f"Q={[f'{v:.4f}' for v in self.q_table[state][base+offset:base+offset+4]]}", pair
        )

        # --- Calculate Gain from Candle Data ---
        ts = trade_settings.get(pair) if hasattr(trade_settings, 'get') else trade_settings
        gain = 0.0
        if ts:
            max_rows = ts.n_ma + 20
            df = api.get_candles_df(pair, count=max_rows, granularity="M1")
            if df is not None and df.shape[0] > 0:
                col = 'mid_c' if 'mid_c' in df.columns else 'c'
                mid_c = df[col].iloc[-1]
                if len(df) >= ts.n_ma:
                    ma = df[col].iloc[-ts.n_ma:].mean()
                    gain = abs(mid_c - ma)

        decision = TradeDecision(
            pair=pair, signal=signal, sl=sl, tp=tp, gain=gain,
            spread=spread, mid_c=(ask + bid) / 2, time=candle_time
        )
        decision.entry_price = entry_price
        decision.sl_tp_tier = sl_tp_tier
        decision.state = state
        decision.state_features = self._get_pair_features(pair, api, pair_settings)
        decision.open_trades_count = len(open_trades)
        decision.exposure_ratio = len(open_trades) / len(self.pairs) if self.pairs else 0
        decision.corr_feature = self._get_correlation_feature(api, self.pairs)
        decision.spread = spread   
        log_message(f"Place Trade: {decision} | Tier={sl_tp_tier}", pair)

        return decision   
                   
    def on_trade_placed(self, trade_id, pair, signal, state, entry_price, tier,
                        sl_price=None, tp_price=None, state_features=None,
                        open_trades_count=0, exposure_ratio=0, corr_feature=0,
                        spread=0):
        self.open_trades[str(trade_id)] = {
            'pair': pair,
            'action': signal,
            'state': state,
            'entry_price': entry_price,
            'entry_time': time.time(),
            'sl_price': sl_price,
            'tp_price': tp_price,
            'sl_tp_tier': tier,
            'peak_pl': 0.0,
            'trail_in_profit': False,
            'trail_pl': 0.0,
            # ✅ Raw features for supervised training
            'state_features': state_features or {},
            'open_trades_count': open_trades_count,
            'exposure_ratio': exposure_ratio,
            'corr_feature': corr_feature,
            'spread': spread,
        }              

    def _calculate_reward(self, api, trade_info, open_trades=None):
        """Reward for microtrading. Uses actual close price from OANDA, or stored manual close P/L, or SL fallback."""
        pair = trade_info['pair']
        entry = trade_info['entry_price']
        action = trade_info['action']
        sl_price = trade_info.get('sl_price')
        tp_price = trade_info.get('tp_price')
        trade_id = trade_info.get('trade_id')

        if action is None:
            print(f"⏭️ Skipping inherited trade (no action data): {pair}")
            return 0.0

        # Holding time
        entry_time = trade_info.get('entry_time', time.time())
        hours_held = max((time.time() - entry_time) / 3600.0, 2.0 / 60.0)

        instrument = ic.instruments_dict[pair]
        pip_location = float(instrument.pipLocation)

        # ✅ PRIORITY 1: Fetch actual close price from OANDA
        close_price = None
        close_source = None

        if trade_id:
            close_price = api.get_trade_close_price(trade_id)
            if close_price is not None:
                close_source = "API"

        # ✅ PRIORITY 2: If manually closed, use stored P/L
        if close_price is None:
            manual_close_pl = trade_info.get('manual_close_pl')
            if manual_close_pl is not None:
                if action == defs.BUY:
                    close_price = entry + (manual_close_pl * pip_location)
                else:
                    close_price = entry - (manual_close_pl * pip_location)
                close_source = "manual"

        # ✅ PRIORITY 3: Fallback — assume SL was hit
        if close_price is None:
            close_price = sl_price if sl_price is not None else entry
            close_source = "SL_fallback"

        # Calculate P/L in pips
        if action == defs.BUY:
            profit_pips = (close_price - entry) / pip_location
        else:
            profit_pips = (entry - close_price) / pip_location

        # --- Scalper Reward Logic ---
        if profit_pips > 0:
            effective_pips = min(profit_pips, 5.0)
            base_reward = effective_pips * 1.0
        else:
            base_reward = profit_pips * 2.0

        # Speed Bonus: only for profitable trades
        if hours_held < 0.1 and profit_pips > 0:
            speed_bonus = 3.0
        elif hours_held < 0.25 and profit_pips > 0:
            speed_bonus = 1.0
        else:
            speed_bonus = 0.0

        # Holding Penalty
        holding_penalty = 0.0
        if hours_held > 0.75:
            holding_penalty = (hours_held - 0.75) * 30.0

        # Repeat Penalty — only on actual repeats
        repeat_key = f"{pair}_{action}"
        repeat_count = self.trade_history.get(repeat_key, 0)
        repeat_penalty = 1.0 if repeat_count > 0 else 0.0
        self.trade_history[repeat_key] = repeat_count + 1

        # Prune history to prevent unbounded growth
        if len(self.trade_history) > 10000:
            self.trade_history = dict(list(self.trade_history.items())[-5000:])

        # ✅ REGRET PENALTY: Agent chose "trail" while in profit, trade ended in loss
        trail_in_profit = trade_info.get('trail_in_profit', False)
        trail_pl = trade_info.get('trail_pl', 0.0)
        regret_penalty = 0.0
        if trail_in_profit and profit_pips < 0:
            regret_penalty = (trail_pl - profit_pips) * 1.5
            print(f"😞 REGRET: {pair} chose trail at +{trail_pl:.1f}p, "
                f"closed at {profit_pips:+.1f}p. Penalty={regret_penalty:.2f}")

        reward = base_reward + speed_bonus - holding_penalty - repeat_penalty - regret_penalty

        # Directional Balance Penalty
        if open_trades:
            long_count = 0
            short_count = 0
            for t in open_trades:
                units = float(t.get('currentUnits', 0) if isinstance(t, dict) else getattr(t, 'currentUnits', 0))
                if units > 0:
                    long_count += 1
                else:
                    short_count += 1

            if long_count > short_count * 2 and action == defs.BUY:
                if reward > 0:
                    reward *= 0.5
                else:
                    reward -= 1.0
            elif short_count > long_count * 2 and action == defs.SELL:
                if reward > 0:
                    reward *= 0.5
                else:
                    reward -= 1.0

        # Clamp
        reward = max(min(reward, 50.0), -50.0)
        # At the end of _calculate_reward, before return:
        trade_info['profit_pips'] = round(profit_pips, 4)
        trade_info['close_reason'] = close_source  # "API", "manual", "SL_fallback"   

        # DEBUG
        print(f"📊 REWARD: {pair} {action} | entry={entry:.5f} close={close_price:.5f} [{close_source}] "
            f"| pips={profit_pips:+.2f} hours={hours_held:.2f} "
            f"| base={base_reward:.2f} bonus={speed_bonus:.2f} "
            f"hold_pen={holding_penalty:.2f} repeat_pen={repeat_penalty:.2f} "
            f"regret={regret_penalty:.2f} "
            f"FINAL={reward:.4f}")

        return reward                       

    def _check_closed_trades(self, api, log_message, trade_settings, open_trades=None):
        if not self.open_trades:
            return

        if open_trades is None:
            open_trades = api.get_open_trades()

        open_ids = set()
        for t in (open_trades or []):
            try:
                tid = t.get('id') if isinstance(t, dict) else getattr(t, 'id', None)
                if tid:
                    open_ids.add(str(tid))
            except:
                continue

        for trade_id, info in list(self.open_trades.items()):
            trade_id_str = str(trade_id)
            if trade_id_str not in open_ids:

                # 90-SECOND MINIMUM AGE
                entry_time = info.get('entry_time', 0)
                age_seconds = time.time() - entry_time if entry_time else 9999

                if age_seconds < 90:
                    continue

                # 1. Calculate Reward
                reward = self._calculate_reward(api, info, open_trades)

                # 2. Get Next State
                next_state = 0
                if hasattr(self, '_get_global_state'):
                    next_state = self._get_global_state(api, trade_settings, open_trades, log_message)

                if next_state >= self.q_table.shape[0]:
                    log_message(f"ERROR: next_state {next_state} out of bounds!", "main")
                    next_state = 0

                # 3. Map Action to Index
                pair_idx = self.pair_index.get(info['pair'], 0)
                is_buy = 0 if info['action'] == defs.BUY else 4
                tier = info.get('sl_tp_tier', 1)
                action_idx = (pair_idx * 8) + is_buy + tier

                if action_idx >= self.q_table.shape[1]:
                    log_message(f"ERROR: action_idx {action_idx} out of bounds (max={self.q_table.shape[1]-1})!", "main")
                    continue

                # 4. Update Q-value (entry decision)
                self.update_q_value(info['state'], action_idx, reward, next_state)

                # 4b. Update Q-value (management decision)
                mgmt_action_taken = info.get('mgmt_action_at_close')
                if mgmt_action_taken is not None:
                    mgmt_idx = (pair_idx * 11 + 8) + mgmt_action_taken
                    if mgmt_idx < self.q_table.shape[1]:
                        mgmt_reward = reward * 0.5
                        self.update_q_value(info['state'], mgmt_idx, mgmt_reward, next_state)

                # ✅ 5. Record training data for supervised model
                self._record_training_data(info, reward, next_state, tier, age_seconds, open_trades)

                log_message(
                    f"Q-UPDATE: trade={trade_id} pair={info['pair']} "
                    f"state={info['state']} action={action_idx} reward={reward:.2f} "
                    f"age={age_seconds:.0f}s",
                    info['pair']
                )
                del self.open_trades[trade_id]

        self.save_model()       
 