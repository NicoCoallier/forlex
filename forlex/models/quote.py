from dataclasses import dataclass


@dataclass
class Quote:
    ask: float
    bid: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.ask + self.bid) / 2
