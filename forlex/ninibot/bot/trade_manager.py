from collections.abc import Callable

from forlex.models import Account, Direction
from forlex.ninibot.bot.trade_risk_calculator import get_trade_units
from forlex.ninibot.infrastructure.log_wrapper import LogMessage
from forlex.ninibot.models.trade_decision import TradeDecision
from forlex.ninibot.ports import Brokerage, MarketData, OrderRejected, TradeId


def place_trade(decision: TradeDecision, market: MarketData, broker: Brokerage,
                account: Account, risk_percent: float, log_message: LogMessage,
                log_error: Callable[[str], None]) -> TradeId | None:
    # ✅ ALLOW MICROTRADES: Only block if SAME direction is already open on this pair
    for trade in broker.open_trades():
        if trade.pair == decision.pair:
            existing_is_buy = trade.units > 0
            new_is_buy = decision.signal == Direction.LONG

            # Same direction already open → skip
            if existing_is_buy == new_is_buy:
                log_message(
                    f"⏭️ Same trade already open: {decision.pair} "
                    f"{'LONG' if new_is_buy else 'SHORT'}. Skipping duplicate.",
                    decision.pair
                )
                return None
            # Opposite direction → allow (hedge/microtrade)
            break

    # 2. Calculate dynamic units
    # BUGFIX-6: sizing needs every quote, to convert into the account currency
    trade_units = get_trade_units(decision, market.quotes(), account, risk_percent, log_message)

    if trade_units == 0:
        log_error(f"Calculated units is 0. Skipping trade for {decision.pair}")
        return None

    # 3. Place the trade
    units = trade_units if decision.signal == Direction.LONG else -trade_units
    try:
        trade_id = broker.place_market_order(decision.pair, units, decision.sl, decision.tp)
    except OrderRejected:
        log_error(f"ERROR placing {decision}")
        log_message(f"ERROR placing {decision} for these {trade_units}", decision.pair)
        return None

    log_message(f"placed trade_id:{trade_id} for {decision} | Units: {trade_units}", decision.pair)
    return trade_id
