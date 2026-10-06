import math

import pandas as pd

from forlex.ninibot.models.trade_decision import TradeDecision
from forlex.ninibot.technicals.indicators import BollingerBands

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

from forlex.models import Direction
from forlex.models.currency import CurrencyPair
from forlex.ninibot.infrastructure.log_wrapper import LogMessage
from forlex.ninibot.models.trade_settings import TradeSettings
from forlex.ninibot.ports import MarketData

ADDROWS = 20


def apply_signal(row: dict, trade_settings: TradeSettings) -> Direction:
    if math.isnan(row['BB_UP']) or math.isnan(row['BB_LW']):
        return Direction.NONE
    if row['SPREAD'] > trade_settings.maxspread:
        return Direction.NONE
    if row['GAIN'] < trade_settings.mingain:
        return Direction.NONE

    # Fire at 1.0σ from mean (40% of the way to the 2.5σ band edge)
    band_half = (row['BB_UP'] - row['BB_LW']) / 2
    threshold = band_half * 0.15  # = 1.0σ with n_std=2.5

    if row['mid_close'] > row['BB_MA'] + threshold:
        return Direction.SHORT
    elif row['mid_close'] < row['BB_MA'] - threshold:
        return Direction.LONG

    return Direction.NONE


def apply_TP(row: dict) -> float:
    if row['SIGNAL'] == Direction.LONG:
        tp_distance = max(row['GAIN'] * 1.5, row['SPREAD'] * 2)
        return row['ask_close'] + tp_distance
    elif row['SIGNAL'] == Direction.SHORT:
        tp_distance = max(row['GAIN'] * 1.5, row['SPREAD'] * 2)
        return row['bid_close'] - tp_distance
    return 0.0


def apply_SL(row: dict) -> float:
    if row['SIGNAL'] == Direction.LONG:
        sl_distance = max(row['GAIN'], row['SPREAD'] * 2)
        return row['ask_close'] - sl_distance
    elif row['SIGNAL'] == Direction.SHORT:
        sl_distance = max(row['GAIN'], row['SPREAD'] * 2)
        return row['bid_close'] + sl_distance
    return 0.0


def process_candles(df: pd.DataFrame, pair, trade_settings: TradeSettings, log_message) -> dict:
    # ✅ ADD THIS:
    log_message(f"BB input: n_ma={trade_settings.n_ma}, n_std={trade_settings.n_std}, "
                f"cols={list(df.columns)}, rows={df.shape[0]}", pair)

    df = BollingerBands(df, trade_settings.n_ma, trade_settings.n_std)

    # ✅ ADD THIS:
    log_message(f"BB output: BB_MA={df['BB_MA'].tail(3).tolist()}, "
                f"BB_UP={df['BB_UP'].tail(3).tolist()}, "
                f"BB_LW={df['BB_LW'].tail(3).tolist()}", pair)

    # Only the latest candle drives the decision, and pandas' per-call
    # overhead on a frame this small costs more than the arithmetic, so the
    # rest is plain floats on that one row.
    row = df.iloc[-1].to_dict()
    row['PAIR'] = pair
    row['SPREAD'] = row['ask_close'] - row['bid_close']
    row['GAIN'] = abs(row['mid_close'] - row['BB_MA'])
    row['SIGNAL'] = apply_signal(row, trade_settings)
    row['TP'] = apply_TP(row)
    row['SL'] = apply_SL(row)
    row['LOSS'] = abs(row['mid_close'] - row['SL'])

    log_cols = ['PAIR', 'timestamp', 'mid_close', 'mid_open', 'SL', 'TP', 'SPREAD', 'GAIN', 'LOSS', 'SIGNAL']
    last_row = {col: row[col] for col in log_cols}
    log_message(f"process_candles: {last_row}", pair)

    return last_row


def fetch_candles(pair: CurrencyPair, row_count: int, market: MarketData,
                  log_message: LogMessage) -> pd.DataFrame | None:
    df = market.candles(pair, row_count)

    if df.empty:
        log_message("tech_manager fetch_candles failed to get candles", pair)
        return None

    return df


def get_trade_decision(market: MarketData, pair: CurrencyPair,
                       trade_settings: TradeSettings,
                       log_message: LogMessage) -> TradeDecision | None:
    max_rows = trade_settings.n_ma + ADDROWS

    log_message(f"tech_manager: max_rows:{max_rows}", pair)

    df = fetch_candles(pair, max_rows, market, log_message)

    if df is not None:
        last_row = process_candles(df, pair, trade_settings, log_message)
        return TradeDecision(last_row)

    return None   