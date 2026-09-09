
from api.oanda_api import OandaApi
from bot.trade_risk_calculator import get_trade_units
from models.trade_decision import TradeDecision
import constants.defs as defs 

def trade_is_open(pair, api: OandaApi):

    open_trades = api.get_open_trades()

    for ot in open_trades:
        if ot.instrument == pair:
            return ot

    return None


def place_trade(trade_decision: TradeDecision, api: OandaApi, log_message, log_error,
                account_equity, margin_available, risk_percent):
    # ✅ ALLOW MICROTRADES: Only block if SAME direction is already open on this pair
    open_trades = api.get_open_trades()
    for t in open_trades:
        inst = t.get('instrument') if isinstance(t, dict) else getattr(t, 'instrument', '')
        if inst == trade_decision.pair:
            units = float(t.get('currentUnits', 0) if isinstance(t, dict) else getattr(t, 'currentUnits', 0))
            existing_is_buy = units > 0
            new_is_buy = trade_decision.signal == defs.BUY

            # Same direction already open → skip
            if existing_is_buy == new_is_buy:
                log_message(
                    f"⏭️ Same trade already open: {trade_decision.pair} "
                    f"{'LONG' if new_is_buy else 'SHORT'}. Skipping duplicate.",
                    trade_decision.pair
                )
                return None
            # Opposite direction → allow (hedge/microtrade)
            break

    # 2. Calculate dynamic units
    trade_units = get_trade_units(
        api,
        trade_decision.pair,
        trade_decision.signal,
        trade_decision.sl,
        account_equity,
        margin_available,
        risk_percent,
        log_message
    )

    if trade_units == 0:
        log_error(f"Calculated units is 0. Skipping trade for {trade_decision.pair}")
        return None

    # 3. Place the trade
    trade_id = api.place_trade(
        trade_decision.pair,
        trade_units,
        trade_decision.signal,
        trade_decision.sl,
        trade_decision.tp
    )

    if trade_id is None:
        log_error(f"ERROR placing {trade_decision}")
        log_message(f"ERROR placing {trade_decision} for these {trade_units}", trade_decision.pair)
    else:
        log_message(f"placed trade_id:{trade_id} for {trade_decision} | Units: {trade_units}", trade_decision.pair)

    return trade_id      
