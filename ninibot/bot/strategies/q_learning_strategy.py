import numpy as np
import os
import json
from bot.strategy_base import TradingStrategy
from models.trade_decision import TradeDecision
import constants.defs as defs
import time

class QLearningStrategy(TradingStrategy):
    def __init__(self, model_path="bot/models/q_table_single.json"):
        self.model_path = model_path
        self.n_states = 50  # Discretized price buckets
        self.n_actions = 3  # 0=BUY, 1=SELL, 2=NONE (mapped from defs)
        
        self.epsilon = 0.1
        self.gamma = 0.95
        self.alpha = 0.1  # Learning rate
        
        # Track open trades to calculate reward when they close
        self.open_trades = {}  # {trade_id: {state, action, entry_price, pair}}
        
        # Load or initialize model
        self.q_table = self._load_model()
        
        # Map defs constants to action indices
        self.action_map = {defs.BUY: 0, defs.SELL: 1, defs.NONE: 2}
        self.action_reverse = {0: defs.BUY, 1: defs.SELL, 2: defs.NONE}

    def _load_model(self):
        """Load Q-table and open trades from disk."""
        try:
            with open(self.model_path, 'r') as f:
                data = json.load(f)
                self.q_table = np.array(data['q_table'])
                self.open_trades = data.get('open_trades', {})
                print(f"✅ Q-Learning model loaded: {self.model_path} (Shape: {self.q_table.shape})")
            return self.q_table
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            print(f"⚠️ No model found at {self.model_path}. Initializing new Q-table.")
            return np.zeros((self.n_states, self.n_actions))

    def save_model(self):
        """Save Q-table and open trades to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        data = {
            'q_table': self.q_table.tolist(),
            'open_trades': self.open_trades
        }
        with open(self.model_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Q-Learning model saved to {self.model_path}")

    def _get_state(self, api, pair, trade_settings, log_message=None):
        """
        Build state from recent candles with logging.
        """
        try:
            max_rows = trade_settings.n_ma + 20
            df = api.get_candles_df(pair, count=max_rows, granularity="M1")
            
            if df is None or df.shape[0] < 10:
                return 0
                
            # --- ADD LOGGING (Same as Bollinger) ---
            if log_message:
                log_cols = ['PAIR', 'time', 'mid_c', 'mid_o', 'SPREAD', 'GAIN', 'LOSS', 'SIGNAL']
                # Filter to only columns that exist in df
                available_cols = [c for c in log_cols if c in df.columns]
                if available_cols:
                    log_message(f"Q-Learning Candles:\n{df[available_cols].tail()}", pair)
            # ----------------------------------------
            
            close = df['mid_c'].values if 'mid_c' in df.columns else df['c'].values
            
            price_range = close[-1] - close[0]
            price_pos = (close[-1] - close[0]) / price_range if price_range != 0 else 0.5
            momentum = close[-1] - close[-4] if len(close) >= 4 else 0
            volatility = np.std(np.diff(close[-6:]) / close[-6:-1]) if len(close) >= 6 else 0
            
            state_features = [
                int(price_pos * 10),
                int(momentum * 10000),
                int(volatility * 10000)
            ]
            
            state = (state_features[0] * 7 + state_features[1] * 13 + state_features[2] * 31) % self.n_states
            return state
            
        except Exception as e:
            if log_message:
                log_message(f"⚠️ State calc error: {e}", pair)
            return 0     

    def get_decision(self, candle_time, pair, granularity, api, trade_settings, log_message, open_trades=None):
        # trade_settings is now a DICT
        pair_settings = trade_settings.get(pair) if isinstance(trade_settings, dict) else trade_settings
        
        if open_trades is None:
            open_trades = api.get_open_trades()
            
        self._check_closed_trades(api, log_message, open_trades)
        state = self._get_state(api, pair, pair_settings)
        
        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.n_actions)
        else:
            action_idx = int(np.argmax(self.q_table[state]))
            
        action = self.action_reverse[action_idx]
        
        if action == defs.NONE:
            return None
            
        # Get current price for SL/TP
        prices = api.get_prices([pair])
        if not prices:
            return None
            
        p = prices[0]
        try:
            if isinstance(p, dict):
                bid = float(p['bids'][0]['price'])
                ask = float(p['asks'][0]['price'])
            else:
                first_bid = p.bids[0]
                first_ask = p.asks[0]
                bid = float(first_bid.price if hasattr(first_bid, 'price') else first_bid['price'])
                ask = float(first_ask.price if hasattr(first_ask, 'price') else first_ask['price'])
        except Exception:
            return None
            
        entry_price = ask if action == defs.BUY else bid
        sl_distance = 0.002
        tp_distance = 0.004
        
        sl = entry_price - sl_distance if action == defs.BUY else entry_price + sl_distance
        tp = entry_price + tp_distance if action == defs.BUY else entry_price - tp_distance
        
        # Calculate gain from candle data (same logic as Bollinger: distance from MA)
        max_rows = pair_settings.n_ma + 20
        df = api.get_candles_df(pair, count=max_rows, granularity="M1")
        gain = 0.0
        if df is not None and df.shape[0] > 0:
            mid_c = df['mid_c'].iloc[-1] if 'mid_c' in df.columns else df['c'].iloc[-1]
            ma = mid_c  # Simplified: use current price as MA proxy
            gain = abs(mid_c - ma)
            # Better: use rolling MA
            if len(df) >= pair_settings.n_ma:
                ma = df['mid_c'].iloc[-pair_settings.n_ma:].mean() if 'mid_c' in df.columns else df['c'].iloc[-pair_settings.n_ma:].mean()
                gain = abs(mid_c - ma)
        
        decision = TradeDecision(pair=pair, signal=action, sl=sl, tp=tp, 
                                  gain=gain, spread=ask - bid, mid_c=(ask + bid) / 2, time=candle_time)   
        log_message(f"Place Trade: {decision}", pair)
        return decision     

    def on_trade_placed(self, trade_id, pair, action, state, entry_price):
        """Called by Bot after successful trade placement."""
        self.open_trades[trade_id] = {
            'pair': pair,
            'action': action,
            'state': state,
            'entry_price': entry_price,
            'entry_time': time.time() ,
        }
        self.save_model()  # Persist immediately

    def _check_closed_trades(self, api, log_message):
        """Check if any tracked trades have closed and update Q-values."""
        if not self.open_trades:
            return
            
        open_trades = api.get_open_trades()
        open_ids = set(t['id'] for t in open_trades)
        
        for trade_id, info in list(self.open_trades.items()):
            if trade_id not in open_ids:
                # Trade is closed - calculate reward
                reward = self._calculate_reward(api, info)
                next_state = self._get_state(api, info['pair'])
                
                self.update_q_value(info['state'], self.action_map[info['action']], reward, next_state)
                del self.open_trades[trade_id]
                log_message(f"Q-Update: Trade {trade_id} closed. Reward={reward:.4f}", info['pair'])
                
        if self.open_trades:
            self.save_model()


    def _calculate_reward(self, api, trade_info):
        """
        Reward = (profit_in_pips / hours_held) - time_penalty
        Maximizes profit/hour. Holding long positions is penalized.
        """
        pair = trade_info['pair']
        entry = trade_info['entry_price']
        action = trade_info['action']
        
        # ✅ Calculate holding time
        entry_time = trade_info.get('entry_time', time.time())
        hours_held = max((time.time() - entry_time) / 3600.0, 0.01)  # Min 1 min
        
        # Get current price
        prices = api.get_prices([pair])
        if not prices:
            return 0.0
            
        p = prices[0]
        try:
            if isinstance(p, dict):
                current = float(p['bids'][0]['price'])
            else:
                first_bid = p.bids[0]
                current = float(first_bid.price if hasattr(first_bid, 'price') else first_bid['price'])
        except Exception:
            return 0.0
            
        # Calculate P/L in pips (not raw price diff)
        instrument = ic.instruments_dict[pair]
        pip_location = float(instrument.pipLocation)
        
        if action == defs.BUY:
            profit_pips = (current - entry) / pip_location
        else:
            profit_pips = (entry - current) / pip_location
        
        # ✅ PRIMARY REWARD: Profit per hour
        profit_per_hour = profit_pips / hours_held
        
        # ✅ TIME PENALTY: Subtract cost per hour held
        # Makes holding a flat/losing position increasingly negative
        time_penalty_per_hour = 0.5  # 0.5 pips/hour (tune this)
        penalty = time_penalty_per_hour * hours_held
        
        reward = profit_per_hour - penalty
        
        # ✅ BONUS: Quick wins (< 1 hour) get 50% extra reward
        if profit_pips > 0 and hours_held < 1.0:
            reward *= 1.5
        
        # ✅ PENALTY: Slow losses (> 4 hours) get 50% extra punishment
        if profit_pips < 0 and hours_held > 4.0:
            reward *= 1.5
        
        return reward   

    def update_q_value(self, state, action, reward, next_state):
        """Standard Q-Learning update rule."""
        best_next = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error   