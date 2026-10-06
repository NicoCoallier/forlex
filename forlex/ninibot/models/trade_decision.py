from typing import Any

from forlex.models import Direction


class TradeDecision:
    def __init__(self, row=None, **kwargs):
        # Whatever the strategy wants handed back to it in on_trade_placed()
        self.details: dict[str, Any] = {}
        if row is not None:
            # Bollinger-style: accept a pandas row or dict
            self.pair = row['PAIR'] if 'PAIR' in row else row.get('pair')
            signal = row['SIGNAL'] if 'SIGNAL' in row else row.get('signal')
            self.signal = Direction(signal)
            self.sl = row['SL'] if 'SL' in row else row.get('sl', 0.0)
            self.tp = row['TP'] if 'TP' in row else row.get('tp', 0.0)
            self.gain = row['GAIN'] if 'GAIN' in row else row.get('gain', 0.0)
            self.loss = row['LOSS'] if 'LOSS' in row else row.get('loss', 0.0)
            self.spread = row['SPREAD'] if 'SPREAD' in row else row.get('spread', 0.0)
            self.mid_c = row['mid_close'] if 'mid_close' in row else row.get('mid_c', 0.0)
            self.mid_o = row['mid_open'] if 'mid_open' in row else row.get('mid_o', 0.0)
            self.time = row['timestamp'] if 'timestamp' in row else row.get('time', '')
        else:
            # Q-learning-style: accept keyword arguments
            self.pair = kwargs.get('pair', '')
            self.signal = kwargs.get('signal', Direction.NONE)
            self.sl = kwargs.get('sl', 0.0)
            self.tp = kwargs.get('tp', 0.0)
            self.gain = kwargs.get('gain', 0.0)
            self.loss = kwargs.get('loss', 0.0)
            self.spread = kwargs.get('spread', 0.0)
            self.mid_c = kwargs.get('mid_c', 0.0)
            self.mid_o = kwargs.get('mid_o', 0.0)
            self.time = kwargs.get('time', '')

    def __str__(self):
        return (f"TradeDecision(): {self.pair} "
                f"dir:{self.signal} "
                f"gain:{self.gain:.4f} "
                f"sl:{self.sl:.4f} "
                f"tp:{self.tp:.4f}")   