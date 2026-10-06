import datetime as dt


class CandleTiming:

    def __init__(self, last_time=None):
        self.last_time = last_time
        self.last_time = last_time  # Can be None initially
        self.is_ready = last_time is not None

    def __repr__(self):
        if self.last_time is None:
            return "last_candle:N/A is_ready:False"
        return f"last_candle:{dt.datetime.strftime(self.last_time, '%y-%m-%d %H:%M')} is_ready:{self.is_ready}"   