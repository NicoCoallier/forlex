import json
import os
from datetime import datetime as dt
from typing import Self

import pandas as pd
import requests
from dateutil import parser
from dotenv import load_dotenv

from forlex.models import Direction
from forlex.ninibot.infrastructure.instrument_collection import (
    instrumentCollection as ic,
)
from forlex.ninibot.models.api_price import ApiPrice

PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"


class OandaApi:
    def __init__(self, api_key: str, account_id: str, url: str = PRACTICE_URL):
        self.account_id = account_id
        self.url = url
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    @classmethod
    def from_env(cls) -> Self:
        """Builds a client from the OANDA_* variables in the environment or .env."""
        load_dotenv()
        return cls(
            os.environ["OANDA_API_KEY"],
            os.environ["OANDA_ACCOUNT_ID"],
            os.getenv("OANDA_URL", PRACTICE_URL),
        )

    def get_open_trades(self):
        """
        Fetches the list of currently open trades for the account.
        Returns a list of trade dictionaries, or None on error.
        """
        endpoint = f"accounts/{self.account_id}/openTrades"
        ok, response = self.make_request(endpoint, verb="get", code=200)

        if ok:
            # The response structure is: { "trades": [ ... ], "lastTransactionID": "..." }
            return response.get("trades", [])

        print(f"Error fetching open trades: {response}")
        return None

    def make_request(
        self, url, verb="get", code=200, params=None, data=None, headers=None
    ):
        full_url = f"{self.url}/{url}"

        if data is not None:
            data = json.dumps(data)

        try:
            response = None
            if verb == "get":
                response = self.session.get(
                    full_url, params=params, data=data, headers=headers
                )
            if verb == "post":
                response = self.session.post(
                    full_url, params=params, data=data, headers=headers
                )
            if verb == "put":
                response = self.session.put(
                    full_url, params=params, data=data, headers=headers
                )

            if response == None:
                return False, {"error": "verb not found"}

            if response.status_code == code:
                return True, response.json()
            else:
                return False, response.json()

        except Exception as error:
            return False, {"Exception": error}

    def get_account_ep(self, ep, data_key):
        url = f"accounts/{self.account_id}/{ep}"
        ok, data = self.make_request(url)
        if ok == True and data_key in data:
            return data[data_key]
        else:
            print("ERROR get_account_ep()", data)
            return None

    def get_account_summary(self):
        return self.get_account_ep("summary", "account")

    def get_account_instruments(self):
        return self.get_account_ep("instruments", "instruments")

    def fetch_candles(
        self,
        pair_name,
        count=10,
        granularity="H1",
        price="MBA",
        date_f=None,
        date_t=None,
    ):
        url = f"instruments/{pair_name}/candles"
        params = {granularity: granularity, price: price}

        if date_f is not None and date_t is not None:
            date_format = "%Y-%m-%dT%H:%M:%SZ"
            params["from"] = dt.strftime(date_f, date_format)
            params["to"] = dt.strftime(date_t, date_format)
        else:
            params["count"] = count

        ok, data = self.make_request(url, params=params)

        if ok == True and "candles" in data:
            return data["candles"]
        else:
            print("ERROR fetch_candles()", params, data)
            return None

    def get_candles_df(self, pair_name, **kwargs):

        data = self.fetch_candles(pair_name, **kwargs)

        if data is None:
            return None
        if len(data) == 0:
            return pd.DataFrame()

        prices = ["mid", "bid", "ask"]
        ohlc = ["o", "h", "l", "c"]

        final_data = []
        for candle in data:
            if candle["complete"] == False:
                continue
            new_dict = {}
            new_dict["time"] = parser.parse(candle["time"])
            new_dict["volume"] = candle["volume"]
            for p in prices:
                if p in candle:
                    for o in ohlc:
                        new_dict[f"{p}_{o}"] = float(candle[p][o])
            final_data.append(new_dict)
        df = pd.DataFrame.from_dict(final_data)
        return df

    def place_trade(
        self,
        pair_name: str,
        units: float,
        direction: Direction,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ):

        instrument = ic.instruments_dict[pair_name]
        units = round(units, instrument.tradeUnitsPrecision)
        try:
            summary = self.get_account_summary()
            if summary:
                margin_available = float(summary["marginAvailable"])
                margin_rate = float(instrument.marginRate)
                required_margin = abs(int(units)) * margin_rate
                if required_margin > margin_available:
                    print(
                        f"⚠️ MARGIN CHECK FAILED: required={required_margin:.2f}, available={margin_available:.2f}, using 10% of available margin for trade"
                    )
                    required_margin = 0.1 * margin_available
        except Exception as e:
            print(f"⚠️ Pre-flight margin check error: {e}")

        url = f"accounts/{self.account_id}/orders"

        instrument = ic.instruments_dict[pair_name]

        # 1. Handle Units
        units = round(units, instrument.tradeUnitsPrecision)
        if direction == Direction.SHORT:
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
                "positionFill": "DEFAULT",
            }
        }

        # 3. Stop Loss (CRITICAL: Use f-string for exact precision)
        if stop_loss is not None:
            # f"{value:.{precision}f}" ensures exact decimal places, avoiding float errors
            sl_price = f"{stop_loss:.{instrument.displayPrecision}f}"
            data["order"]["stopLossOnFill"] = {"price": sl_price, "timeInForce": "GTC"}

        # 4. Take Profit (CRITICAL: Use f-string for exact precision)
        if take_profit is not None:
            tp_price = f"{take_profit:.{instrument.displayPrecision}f}"
            data["order"]["takeProfitOnFill"] = {
                "price": tp_price,
                "timeInForce": "GTC",
            }

        # 5. Execute
        ok, response = self.make_request(url, verb="post", data=data, code=201)

        if ok and response and "orderFillTransaction" in response:
            fill = response["orderFillTransaction"]
            trade_id = fill.get("tradeOpened", {}).get("tradeID", fill["id"])
            return trade_id
        else:
            # 6. Detailed Error Logging
            if response:
                # Check for Order Reject
                if "orderRejectTransaction" in response:
                    rej = response["orderRejectTransaction"]
                    reason = rej.get("rejectReason", "Unknown")
                    error_msg = rej.get("errorMessage", "")
                    print(f"❌ ORDER REJECTED: {reason} - {error_msg}")
                    print(
                        f"   Attempted: Units={units}, SL={stop_loss}, TP={take_profit}"
                    )
                    print(
                        f"   Sent Prices: SL={data['order'].get('stopLossOnFill', {}).get('price')}, "
                        f"TP={data['order'].get('takeProfitOnFill', {}).get('price')}"
                    )

                # Check for generic error
                elif "errorMessage" in response:
                    print(f"❌ API ERROR: {response['errorMessage']}")
                    print(f"   Full Response: {response}")
                else:
                    print(f"❌ UNKNOWN RESPONSE: {response}")
            else:
                print(f"❌ NO RESPONSE from API for {pair_name}")

            return None

    def update_trade_sl(self, pair_name, trade_id, new_sl_price):
        """Update the stop loss on an open trade."""
        # BUGFIX-2: this used PUT /trades/{id}, which v20 doesn't have, so every
        # breakeven/trailing move failed and nobody checked the result. v20 replaces a
        # trade's stop loss through PUT /trades/{id}/orders with a "stopLoss" body (the
        # take profit is left alone when it isn't in the body), and the price must be
        # rounded to the instrument's display precision, same as in place_trade
        instrument = ic.instruments_dict[pair_name]
        url = f"accounts/{self.account_id}/trades/{trade_id}/orders"
        payload = {
            "stopLoss": {
                "price": f"{new_sl_price:.{instrument.displayPrecision}f}",
                "timeInForce": "GTC",
            }
        }
        ok, response = self.make_request(url, verb="put", code=200, data=payload)
        if not ok:
            print(f"❌ STOP LOSS UPDATE REJECTED for trade {trade_id}: {response}")
        return ok

    def close_trade(self, trade_id):
        url = f"accounts/{self.account_id}/trades/{trade_id}/close"
        ok, _ = self.make_request(url, verb="put", code=200)

        if ok == True:
            print(f"Closed {trade_id} successfully")
        else:
            print(f"Failed to close {trade_id}")

        return ok

    def get_trade(self, trade_id):
        """Returns the v20 trade (open or closed) as a dict, or None on error."""
        url = f"accounts/{self.account_id}/trades/{trade_id}"
        ok, response = self.make_request(url)

        if ok and "trade" in response:
            return response["trade"]
        return None

    def get_prices(self, instruments_list):
        url = f"accounts/{self.account_id}/pricing"

        params = {
            "instruments": ",".join(instruments_list),
            "includeHomeConversions": True,
        }

        ok, response = self.make_request(url, params=params)

        if ok == True and "prices" in response and "homeConversions" in response:
            return [
                ApiPrice(x, response["homeConversions"]) for x in response["prices"]
            ]

        return None
