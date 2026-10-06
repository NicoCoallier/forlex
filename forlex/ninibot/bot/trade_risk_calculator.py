from forlex.models import Account, Direction, Quote
from forlex.models.currency import CurrencyConverter, CurrencyPair
from forlex.ninibot.infrastructure.log_wrapper import LogMessage
from forlex.ninibot.models.trade_decision import TradeDecision


def get_trade_units(decision: TradeDecision, quotes: dict[CurrencyPair, Quote],
                    account: Account, risk_percent: float,
                    log_message: LogMessage) -> int:
    """
    Sizes a trade by the equity it risks, capped by the available margin.
    """
    pair = decision.pair
    quote = quotes[pair]

    # 1. Determine Entry Price
    entry_price = quote.ask if decision.signal == Direction.LONG else quote.bid

    # 2. Calculate Stop Loss Distance
    sl_distance = abs(entry_price - decision.sl)
    if sl_distance == 0:
        log_message("get_trade_units() SL distance is 0", pair)
        return 0

    # BUGFIX-6: prices move in the pair's quote currency (JPY for GBP_JPY) while risk
    # and margin are budgeted in the account currency. Without this rate, a JPY pair
    # spent its risk budget in yen, and the margin cap ignored the pair's price.
    to_account = CurrencyConverter(quotes).rate(pair.quote, account.currency)

    # 3. Calculate Units based on RISK
    risk_amount = account.nav * risk_percent
    units_by_risk = int(risk_amount / (sl_distance * to_account))  # BUGFIX-6

    # 4. Margin-based units: CAP at 10% of available margin
    margin_cap = account.margin_available * 0.10
    # BUGFIX-6: margin per unit is margin rate x price in the account currency
    # (OANDA's formula); the old code divided by the margin rate alone
    margin_per_unit = pair.margin_rate * entry_price * to_account
    max_units_by_margin = int(margin_cap / margin_per_unit)

    log_message(
        f"Margin: avail={account.margin_available:.2f}, cap(10%)={margin_cap:.2f}, "
        f"rate={pair.margin_rate}, to_account={to_account:.6f}, max={max_units_by_margin}",
        pair
    )

    # 5. Final: smallest of risk and margin cap
    final_units = min(units_by_risk, max_units_by_margin)

    if final_units < 1:
        log_message(f"Units ({final_units}) below min. risk={units_by_risk}, margin={max_units_by_margin}", pair)
        return 0

    log_message(f"FINAL: Units={final_units} (risk={units_by_risk}, margin_cap={max_units_by_margin})", pair)
    return final_units
