import pandas as pd
import numpy as np 
from models.trade_decision import TradeDecision
from technicals.indicators import BollingerBands

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

from api.oanda_api import OandaApi
from models.trade_settings import TradeSettings
import constants.defs as defs

ADDROWS = 20


def apply_signal(row, trade_settings: TradeSettings):
    if np.isnan(row.BB_UP) or np.isnan(row.BB_LW):
        return defs.NONE
    if row.SPREAD > trade_settings.maxspread:
        return defs.NONE
    if row.GAIN < trade_settings.mingain:
        return defs.NONE

    # Fire at 1.0σ from mean (40% of the way to the 2.5σ band edge)
    band_half = (row.BB_UP - row.BB_LW) / 2
    threshold = band_half * 0.15  # = 1.0σ with n_std=2.5

    if row.mid_c > row.BB_MA + threshold:
        return defs.SELL
    elif row.mid_c < row.BB_MA - threshold:
        return defs.BUY

    return defs.NONE                 


def apply_TP(row):
    if row.SIGNAL == defs.BUY:
        tp_distance = max(row.GAIN * 1.5, row.SPREAD * 2)
        return row.ask_c + tp_distance
    elif row.SIGNAL == defs.SELL:
        tp_distance = max(row.GAIN * 1.5, row.SPREAD * 2)
        return row.bid_c - tp_distance
    return 0.0

def apply_SL(row, trade_settings: TradeSettings):
    if row.SIGNAL == defs.BUY:
        sl_distance = max(row.GAIN, row.SPREAD * 2)
        return row.ask_c - sl_distance
    elif row.SIGNAL == defs.SELL:
        sl_distance = max(row.GAIN, row.SPREAD * 2)
        return row.bid_c + sl_distance
    return 0.0   


def process_candles(df: pd.DataFrame, pair, trade_settings: TradeSettings, log_message):
    df.reset_index(drop=True, inplace=True)
    df['PAIR'] = pair
    df['SPREAD'] = df.ask_c - df.bid_c

    # ✅ ADD THIS:
    log_message(f"BB input: n_ma={trade_settings.n_ma}, n_std={trade_settings.n_std}, "
                f"cols={list(df.columns)}, rows={df.shape[0]}", pair)

    df = BollingerBands(df, trade_settings.n_ma, trade_settings.n_std)

    # ✅ ADD THIS:
    log_message(f"BB output: BB_MA={df['BB_MA'].tail(3).tolist()}, "
                f"BB_UP={df['BB_UP'].tail(3).tolist()}, "
                f"BB_LW={df['BB_LW'].tail(3).tolist()}", pair)

    df['GAIN'] = (df.mid_c - df.BB_MA).abs()
    df['SIGNAL'] = df.apply(apply_signal, axis=1, trade_settings=trade_settings)
    df['TP'] = df.apply(apply_TP, axis=1)
    df['SL'] = df.apply(apply_SL, axis=1, trade_settings=trade_settings)
    df['LOSS'] = (df.mid_c - df.SL).abs()

    log_cols = ['PAIR', 'time', 'mid_c', 'mid_o', 'SL', 'TP', 'SPREAD', 'GAIN', 'LOSS', 'SIGNAL']
    log_message(f"process_candles:\n{df[log_cols].tail()}", pair)

    return df[log_cols].iloc[-1]   


def fetch_candles(pair, row_count, candle_time, granularity,
                  api: OandaApi, log_message):
    df = api.get_candles_df(pair, count=row_count, granularity=granularity)

    if df is None or df.shape[0] == 0:
        log_message("tech_manager fetch_candles failed to get candles", pair)
        return None

    if df.iloc[-1].time != candle_time:
        log_message(f"tech_manager fetch_candles {df.iloc[-1].time} not correct", pair)
        return None

    return df


def get_trade_decision(candle_time, pair, granularity, api: OandaApi,
                       trade_settings: TradeSettings, log_message):
    max_rows = trade_settings.n_ma + ADDROWS

    log_message(f"tech_manager: max_rows:{max_rows} candle_time:{candle_time} granularity:{granularity}", pair)

    df = fetch_candles(pair, max_rows, candle_time, granularity, api, log_message)

    if df is not None:
        last_row = process_candles(df, pair, trade_settings, log_message)
        return TradeDecision(last_row)

    return None   