import requests
import pandas as pd
import json
import constants.defs as defs

from dateutil import parser
from datetime import datetime as dt
from infrastructure.instrument_collection import instrumentCollection as ic
from models.api_price import ApiPrice
from models.open_trade import OpenTrade
import ninibot.constants.defs as defs

class OandaApi:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {defs.API_KEY}",
            "Content-Type": "application/json"
        })

    def get_open_trades(self):
        """
        Fetches the list of currently open trades for the account.
        Returns a list of trade dictionaries, or an empty list if none/error.
        """
        endpoint = f"accounts/{defs.ACCOUNT_ID}/openTrades"
        
        try:
            ok, response = self.make_request(endpoint, verb="get", code=200)
            
            if ok:
                # The response structure is: { "trades": [ ... ], "lastTransactionID": "..." }
                trades = response.get('trades', [])
                return trades
            else:
                print(f"Error fetching open trades: {response}")
                return []
                
        except Exception as e:
            print(f"Exception fetching open trades: {e}")
            return []   

    def get_account_equity(self):
        """Fetches current account NAV (equity) to calculate dynamic position size."""
        endpoint = f"{defs.OANDA_URL}/accounts/{defs.ACCOUNT_ID}"
        try:
            response = self.session.get(endpoint)
            
            # Check for successful HTTP status (200 OK)
            if response.status_code == 200:
                data = response.json()
                # Navigate the JSON structure: { "account": { "NAV": "..." } }
                nav = float(data['account']['NAV'])
                return nav
            else:
                # Log or handle specific errors (401 Unauthorized, 404 Not Found, etc.)
                print(f"Error fetching equity: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Exception fetching equity: {e}")
            return None

    def make_request(self, url, verb='get', code=200, params=None, data=None, headers=None):
        full_url = f"{defs.OANDA_URL}/{url}"

        if data is not None:
            data = json.dumps(data)

        try:
            response = None
            if verb == "get":
                response = self.session.get(full_url, params=params, data=data, headers=headers)
            if verb == "post":
                response = self.session.post(full_url, params=params, data=data, headers=headers)
            if verb == "put":
                response = self.session.put(full_url, params=params, data=data, headers=headers)
            
            if response == None:
                return False, {'error': 'verb not found'}

            if response.status_code == code:
                return True, response.json()
            else:
                return False, response.json()
            
        except Exception as error:
            return False, {'Exception': error}

    def get_account_ep(self, ep, data_key):
        url = f"accounts/{defs.ACCOUNT_ID}/{ep}"
        ok, data = self.make_request(url);

        if ok == True and data_key in data:
            return data[data_key]
        else:
            print("ERROR get_account_ep()", data)
            return None

    def get_account_summary(self):
        return self.get_account_ep("summary", "account")

    def get_account_instruments(self):
        return self.get_account_ep("instruments", "instruments")

    def fetch_candles(self, pair_name, count=10, granularity="H1",
                            price="MBA", date_f=None, date_t=None):
        url = f"instruments/{pair_name}/candles"
        params = dict(
            granularity = granularity,
            price = price
        )

        if date_f is not None and date_t is not None:
            date_format = "%Y-%m-%dT%H:%M:%SZ"
            params["from"] = dt.strftime(date_f, date_format)
            params["to"] = dt.strftime(date_t, date_format)
        else:
            params["count"] = count

        ok, data = self.make_request(url, params=params)
    
        if ok == True and 'candles' in data:
            return data['candles']
        else:
            print("ERROR fetch_candles()", params, data)
            return None

    def get_candles_df(self, pair_name, **kwargs):

        data = self.fetch_candles(pair_name, **kwargs)

        if data is None:
            return None
        if len(data) == 0:
            return pd.DataFrame()
        
        prices = ['mid', 'bid', 'ask']
        ohlc = ['o', 'h', 'l', 'c']
        
        final_data = []
        for candle in data:
            if candle['complete'] == False:
                continue
            new_dict = {}
            new_dict['time'] = parser.parse(candle['time'])
            new_dict['volume'] = candle['volume']
            for p in prices:
                if p in candle:
                    for o in ohlc:
                        new_dict[f"{p}_{o}"] = float(candle[p][o])
            final_data.append(new_dict)
        df = pd.DataFrame.from_dict(final_data)
        return df

    def last_complete_candle(self, pair_name, granularity):
        df = self.get_candles_df(pair_name, granularity=granularity, count=10)
        
        if df is None or df.shape[0] == 0:
            return None
            
        if 'complete' in df.columns and not df['complete'].iloc[-1]:
            if df.shape[0] >= 2:
                return df.iloc[-2].time
            return None
            
        return df.iloc[-1].time   

    def place_trade(self, pair_name: str, units: float, direction: int,
                        stop_loss: float=None, take_profit: float=None):

        instrument = ic.instruments_dict[pair_name]
        units = round(units, instrument.tradeUnitsPrecision)
        try:
            summary = self.get_account_summary()
            if summary:
                margin_available = float(summary['marginAvailable'])
                margin_rate = float(instrument.marginRate)
                required_margin = abs(int(units)) * margin_rate
                if required_margin > margin_available:
                    print(f"⚠️ MARGIN CHECK FAILED: required={required_margin:.2f}, available={margin_available:.2f}, using 10% of available margin for trade")
                    required_margin = 0.1*margin_available
        except Exception as e:
            print(f"⚠️ Pre-flight margin check error: {e}")

        url = f"accounts/{defs.ACCOUNT_ID}/orders"
        
        instrument = ic.instruments_dict[pair_name]
        
        # 1. Handle Units
        units = round(units, instrument.tradeUnitsPrecision)
        if direction == defs.SELL:
            units = units * -1
            
        # Ensure units is a non-zero integer string
        if int(units) == 0:
            print(f"ERROR: Units rounded to 0 for {pair_name}")
            return None

        # 2. Build Order Data
        data = {
            "order": {
                "units": str(int(units)),
                "instrument": pair_name,
                "type": "MARKET",
                "timeInForce": "FOK",
                "positionFill": "DEFAULT"
            }
        }

        # 3. Stop Loss (CRITICAL: Use f-string for exact precision)
        if stop_loss is not None:
            # f"{value:.{precision}f}" ensures exact decimal places, avoiding float errors
            sl_price = f"{stop_loss:.{instrument.displayPrecision}f}"
            data["order"]["stopLossOnFill"] = {
                "price": sl_price,
                "timeInForce": "GTC"
            }

        # 4. Take Profit (CRITICAL: Use f-string for exact precision)
        if take_profit is not None:
            tp_price = f"{take_profit:.{instrument.displayPrecision}f}"
            data["order"]["takeProfitOnFill"] = {
                "price": tp_price,
                "timeInForce": "GTC"
            }

        # 5. Execute
        ok, response = self.make_request(url, verb="post", data=data, code=201)

        if ok and response and 'orderFillTransaction' in response:
            fill = response['orderFillTransaction']
            trade_id = fill.get('tradeOpened', {}).get('tradeID', fill['id'])
            return trade_id
        else:
            # 6. Detailed Error Logging
            if response:
                # Check for Order Reject
                if 'orderRejectTransaction' in response:
                    rej = response['orderRejectTransaction']
                    reason = rej.get('rejectReason', 'Unknown')
                    error_msg = rej.get('errorMessage', '')
                    print(f"❌ ORDER REJECTED: {reason} - {error_msg}")
                    print(f"   Attempted: Units={units}, SL={stop_loss}, TP={take_profit}")
                    print(f"   Sent Prices: SL={data['order'].get('stopLossOnFill', {}).get('price')}, "
                        f"TP={data['order'].get('takeProfitOnFill', {}).get('price')}")
                
                # Check for generic error
                elif 'errorMessage' in response:
                    print(f"❌ API ERROR: {response['errorMessage']}")
                    print(f"   Full Response: {response}")
                else:
                    print(f"❌ UNKNOWN RESPONSE: {response}")
            else:
                print(f"❌ NO RESPONSE from API for {pair_name}")
                
            return None  

    def update_trade_sl(self, trade_id, new_sl_price):
        """Update the stop loss on an open trade."""
        url = f"accounts/{defs.ACCOUNT_ID}/trades/{trade_id}"
        data = {
            "order": {
                "type": "TAKE_PROFIT" if False else "STOP_LOSS",
                # Actually use the trade update endpoint:
            }
        }
        # OANDA v20: PUT /v3/accounts/{accountID}/trades/{tradeID}
        # Body: {"stopLossOnFill": {"price": "1.23456"}}
        payload = {"stopLossOnFill": {"price": str(new_sl_price)}}
        ok, response = self.make_request(url, verb="put", code=200, data=payload)
        return ok    

    def get_trade_close_price(self, trade_id):
        """
        Fetch the actual close price for a closed trade.
        Uses the transactions endpoint to find the TRADE_CLOSE transaction.
        """
        url = f"accounts/{defs.ACCOUNT_ID}/trades/{trade_id}"
        ok, data = self.make_request(url, verb="get", code=200)
        
        if ok and 'trade' in data:
            trade = data['trade']
            # Closed trades have 'closePrice' field
            close_price = trade.get('closePrice')
            if close_price:
                return float(close_price)
        
        # Fallback: check transactions
        url2 = f"accounts/{defs.ACCOUNT_ID}/transactions"
        params = {"type": "TRADE_CLOSE", "count": "20"}
        ok2, data2 = self.make_request(url2, verb="get", code=200, params=params)
        
        if ok2 and 'transactions' in data2:
            for t in data2['transactions']:
                if t.get('type') == 'TRADE_CLOSE' and t.get('tradeID') == str(trade_id):
                    return float(t.get('price', 0))
        
        return None     
            
    def close_trade(self, trade_id):
        url = f"accounts/{defs.ACCOUNT_ID}/trades/{trade_id}/close"
        ok, _ = self.make_request(url, verb="put", code=200)

        if ok == True:
            print(f"Closed {trade_id} successfully")
        else:
            print(f"Failed to close {trade_id}")

        return ok

    def get_open_trade(self, trade_id):
        url = f"accounts/{defs.ACCOUNT_ID}/trades/{trade_id}"
        ok, response = self.make_request(url)

        if ok == True and 'trade' in response:
            return OpenTrade(response['trade'])

    def get_open_trades(self):
        url = f"accounts/{defs.ACCOUNT_ID}/openTrades"
        ok, response = self.make_request(url)

        if ok == True and 'trades' in response:
            return [OpenTrade(x) for x in response['trades']]

    def get_prices(self, instruments_list):
        url = f"accounts/{defs.ACCOUNT_ID}/pricing"

        params = dict(
            instruments=','.join(instruments_list),
            includeHomeConversions=True
        )

        ok, response = self.make_request(url, params=params)

        if ok == True and 'prices' in response and 'homeConversions' in response:
            return [ApiPrice(x, response['homeConversions']) for x in response['prices']]

        return None

    













