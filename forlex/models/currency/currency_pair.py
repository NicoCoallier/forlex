from enum import StrEnum

from forlex.const import MARGIN_RATES

from .currency import Currency


class CurrencyPair(StrEnum):
    AUD_CAD = "AUD_CAD"
    AUD_JPY = "AUD_JPY"
    AUD_USD = "AUD_USD"
    EUR_CHF = "EUR_CHF"
    EUR_GBP = "EUR_GBP"
    EUR_JPY = "EUR_JPY"
    EUR_USD = "EUR_USD"
    GBP_CHF = "GBP_CHF"
    GBP_JPY = "GBP_JPY"
    GBP_USD = "GBP_USD"
    USD_CAD = "USD_CAD"
    USD_CHF = "USD_CHF"
    USD_JPY = "USD_JPY"

    @property
    def base(self) -> Currency:
        return Currency(self.value.split("_")[0])

    @property
    def quote(self) -> Currency:
        return Currency(self.value.split("_")[1])

    @property
    def pip_size(self) -> float:
        return 0.01 if self.quote == Currency.JPY else 0.0001

    @property
    def margin_rate(self) -> float:
        return MARGIN_RATES[self]