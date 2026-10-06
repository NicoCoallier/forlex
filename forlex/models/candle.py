from dataclasses import dataclass


@dataclass
class Candle:
    open: float
    low: float
    high: float
    close: float

    def __add__(self, other):
        return Candle(
            open=self.open + other.open,
            low=self.low + other.low,
            high=self.high + other.high,
            close=self.close + other.close,
        )

    def __sub__(self, other):
        return Candle(
            open=self.open - other.open,
            low=self.low - other.low,
            high=self.high - other.high,
            close=self.close - other.close,
        )

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplementedError
        if other == 0:
            raise ZeroDivisionError
        return Candle(
            open=self.open / other,
            low=self.low / other,
            high=self.high / other,
            close=self.close / other,
        )
