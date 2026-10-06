from forlex.models import Quote

from .currency import Currency
from .currency_pair import CurrencyPair


class CurrencyConverter:
    def __init__(self, pair_to_quote: dict[CurrencyPair, Quote]):
        self.pair_to_quote = pair_to_quote

    def convert(self, amount: float, from_: Currency, to: Currency) -> float:
        return amount * self.rate(from_, to)

    def rate(self, from_: Currency, to: Currency) -> float:
        rate = self._direct_rate(from_, to)
        if rate is not None:
            return rate

        for intermediate in Currency:
            first_leg = self._direct_rate(from_, intermediate)
            second_leg = self._direct_rate(intermediate, to)
            if first_leg is not None and second_leg is not None:
                return first_leg * second_leg

        raise KeyError(
            f"No quote available to convert {from_.value} to {to.value}, "
            f"stream a pair linking them (given: {list(self.pair_to_quote)})"
        )

    def _direct_rate(self, from_: Currency, to: Currency) -> float | None:
        if from_ == to:
            return 1.0
        for pair, quote in self.pair_to_quote.items():
            if pair.base == from_ and pair.quote == to:
                return quote.mid
            if pair.base == to and pair.quote == from_:
                return 1 / quote.mid
        return None
