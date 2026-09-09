from api.oanda_api import OandaApi
import constants.defs as defs
from infrastructure.instrument_collection import instrumentCollection as ic

def get_trade_units(api, pair, signal, sl_price, account_equity, margin_available,
                    risk_percent, log_message):
    """
    Calculates dynamic trade units based on equity, risk, AND available margin.
    Uses OANDA's official formula: Margin_Used = Margin_Rate × Units × Conversion
    """
    
    # 1. Fetch Current Price
    prices_data = api.get_prices([pair])
    if not prices_data:
        log_message("get_trade_units() Prices is none", pair)
        return 0

    current_price_obj = None
    for p in prices_data:
        inst_name = p.instrument if hasattr(p, 'instrument') else p.get('instrument') if isinstance(p, dict) else None
        if inst_name == pair:
            current_price_obj = p
            break
    
    if not current_price_obj:
        log_message(f"get_trade_units() price object not found for {pair}", pair)
        return 0
        
    # 2. Extract Bid/Ask Prices
    bid_price = None
    ask_price = None
    
    try:
        if hasattr(current_price_obj, 'bid') and hasattr(current_price_obj, 'ask'):
            bid_price = float(current_price_obj.bid)
            ask_price = float(current_price_obj.ask)
        elif isinstance(current_price_obj, dict) and 'bid' in current_price_obj and 'ask' in current_price_obj:
            bid_price = float(current_price_obj['bid'])
            ask_price = float(current_price_obj['ask'])
        elif hasattr(current_price_obj, 'bids'):
            first_bid = current_price_obj.bids[0]
            first_ask = current_price_obj.asks[0]
            bid_price = float(first_bid.price if hasattr(first_bid, 'price') else first_bid['price'])
            ask_price = float(first_ask.price if hasattr(first_ask, 'price') else first_ask['price'])
        elif isinstance(current_price_obj, dict) and 'bids' in current_price_obj:
            bid_price = float(current_price_obj['bids'][0]['price'])
            ask_price = float(current_price_obj['asks'][0]['price'])
        else:
            log_message(f"DEBUG: Unknown price structure. Type={type(current_price_obj)}, Content={str(current_price_obj)}", pair)
            return 0
    except (IndexError, KeyError, AttributeError, TypeError, ValueError) as e:
        log_message(f"get_trade_units() Failed to parse price: {e}", pair)
        return 0

    if bid_price is None or ask_price is None:
        log_message("get_trade_units() Could not extract bid/ask prices", pair)
        return 0

    # 3. Determine Entry Price
    entry_price = ask_price if signal == defs.BUY else bid_price

    # 4. Calculate Stop Loss Distance
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        log_message("get_trade_units() SL distance is 0", pair)
        return 0

    # 5. Calculate Units based on RISK
    risk_amount = account_equity * risk_percent
    units_by_risk = int(risk_amount / sl_distance)

    # 2. Margin-based units: CAP at 20% of available margin
    try:
        instrument = ic.instruments_dict[pair]
        margin_rate = float(instrument.marginRate)
        conversion = _get_currency_conversion(api, pair)

        # ✅ 20% of marginAvailable is the max this trade can consume
        margin_cap = margin_available * 0.10

        # Required margin per unit = marginRate × conversion
        # Max units = margin_cap / (marginRate × conversion)
        max_units_by_margin = int(margin_cap / (margin_rate * conversion))

        log_message(
            f"Margin: avail={margin_available:.2f}, cap(20%)={margin_cap:.2f}, "
            f"rate={margin_rate}, conv={conversion:.4f}, max={max_units_by_margin}",
            pair
        )
    except Exception as e:
        log_message(f"Margin calc failed: {e}", pair)
        max_units_by_margin = units_by_risk

    # 3. Final: smallest of risk and margin cap
    final_units = min(units_by_risk, max_units_by_margin)

    if final_units < 1:
        log_message(f"Units ({final_units}) below min. risk={units_by_risk}, margin={max_units_by_margin}", pair)
        return 0

    log_message(f"FINAL: Units={final_units} (risk={units_by_risk}, margin_cap={max_units_by_margin})", pair)
    return final_units


def _get_currency_conversion(api, pair):
    """
    Get the conversion rate from the pair's quote currency to the account's home currency.
    For EUR_USD on USD account → 1.0
    For EUR_CHF on USD account → 1 / CHF_USD_price
    For GBP_JPY on USD account → 1 / JPY_USD_price
    """
    # Extract quote currency from pair name (e.g., "EUR_CHF" → "CHF")
    quote_ccy = pair.split('_')[1]
    
    # If account is USD and quote is USD, no conversion needed
    if quote_ccy == 'USD':
        return 1.0
    
    # Fetch the cross rate: {quote_ccy}_USD
    cross_pair = f"{quote_ccy}_USD"
    try:
        prices = api.get_prices([cross_pair])
        if prices:
            p = prices[0]
            if isinstance(p, dict):
                return 1.0 / float(p['bids'][0]['price'])
            else:
                first_bid = p.bids[0]
                bid = float(first_bid.price if hasattr(first_bid, 'price') else first_bid['price'])
                return 1.0 / bid
    except Exception:
        pass
    
    # Fallback: assume 1.0 (may cause margin issues for cross pairs)
    return 1.0   