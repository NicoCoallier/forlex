from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forlex.models.currency import Currency


# All amounts are in the account currency
@dataclass
class Account:
    currency: Currency
    balance: float
    nav: float
    position: float
    unrealized_pl: float
    realized_pl: float
    margin_used: float
    margin_available: float
