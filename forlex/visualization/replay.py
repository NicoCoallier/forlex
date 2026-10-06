from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from forlex.broker import Broker
from forlex.models import Candle, Direction
from forlex.models.currency import CurrencyPair
from forlex.models.order import ExitOrder, StopLossOrder
from forlex.models.simulation import PriceState
from forlex.models.trade import Trade

STOP_LOSS_COLOR = "#d62728"
TAKE_PROFIT_COLOR = "#2ca02c"
EXIT_COLOR = "#555555"


@dataclass(frozen=True)
class _Level:
    """An exit order's price, drawn over the time it was pending."""

    order: ExitOrder
    start: datetime
    end: datetime


class ReplayRecorder:
    """Records what the broker sees each step, then draws it as a replay figure.

    Call `record()` after every `BackTestingEngine.step()`. Candles are drawn at
    mid, while exit orders trigger on bid or ask, so a level can fill half a
    spread away from where the drawn candle touches it.
    """

    def __init__(self, broker: Broker) -> None:
        self._broker = broker
        self._times: dict[CurrencyPair, list[datetime]] = defaultdict(list)
        self._candles: dict[CurrencyPair, list[Candle]] = defaultdict(list)
        # Bars and trades the broker already has belong to an earlier replay
        self._last_state: dict[CurrencyPair, PriceState] = dict(broker.market_state)
        self._pending_levels: dict[int, tuple[ExitOrder, datetime]] = {}
        self._levels: list[_Level] = []
        self._trades: dict[int, Trade] = {}
        self._closed_seen = len(broker.trade_manager.closed_trades)
        self._last_time: datetime | None = None

    def record(self) -> None:
        time = _plot_time(self._broker.current_time)
        self._record_candles(time)
        self._record_levels(time)
        self._record_trades()
        self._last_time = time

    @property
    def has_candles(self) -> bool:
        return any(self._candles.values())

    def figure(self, window: int = 120, max_steps: int = 300) -> go.Figure:
        """Builds one candlestick chart per instrument with a scrubbing slider.

        Each slider step shows the last `window` bars up to that point.
        """
        pairs = sorted(self._candles)
        fig = make_subplots(
            rows=len(pairs),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[pair.value for pair in pairs],
        )
        levels = self._all_levels()
        for row, pair in enumerate(pairs, start=1):
            for trace in self._pair_traces(pair, levels):
                fig.add_trace(trace, row=row, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        fig.update_layout(
            height=max(400, 320 * len(pairs)),
            showlegend=False,
            sliders=[self._slider(pairs, levels, window, max_steps)],
        )
        steps = fig.layout.sliders[0].steps
        if steps:
            first_view = steps[0].args[0]
            fig.update_layout(**{k.replace(".", "_"): v for k, v in first_view.items()})
        return fig

    def _record_candles(self, time: datetime) -> None:
        for pair, state in self._broker.market_state.items():
            # The simulator repeats the last bar when no new one exists
            if self._last_state.get(pair) == state:
                continue
            self._last_state[pair] = state
            self._times[pair].append(time)
            self._candles[pair].append(state.mid_candle)

    def _record_levels(self, time: datetime) -> None:
        pending = {
            order.id: order
            for order in self._broker.order_manager.pending_orders
            if isinstance(order, ExitOrder)
        }
        for order_id in self._pending_levels.keys() - pending.keys():
            order, start = self._pending_levels.pop(order_id)
            self._levels.append(_Level(order, start, time))
        for order_id, order in pending.items():
            self._pending_levels.setdefault(order_id, (order, time))

    def _record_trades(self) -> None:
        trade_manager = self._broker.trade_manager
        for trade in trade_manager.open_trades:
            self._trades[trade.id] = trade
        for trade in trade_manager.closed_trades[self._closed_seen :]:
            self._trades[trade.id] = trade
        self._closed_seen = len(trade_manager.closed_trades)

    def _all_levels(self) -> list[_Level]:
        still_pending = [
            _Level(order, start, self._last_time)
            for order, start in self._pending_levels.values()
            if self._last_time is not None
        ]
        return self._levels + still_pending

    def _pair_traces(self, pair: CurrencyPair, levels: list[_Level]) -> list:
        candles = self._candles[pair]
        traces = [
            go.Candlestick(
                x=self._times[pair],
                open=[c.open for c in candles],
                high=[c.high for c in candles],
                low=[c.low for c in candles],
                close=[c.close for c in candles],
                name=pair.value,
            )
        ]
        pair_levels = [level for level in levels if level.order.pair == pair]
        for is_stop, color, name in (
            (True, STOP_LOSS_COLOR, "Stop loss"),
            (False, TAKE_PROFIT_COLOR, "Take profit"),
        ):
            kind = [
                level
                for level in pair_levels
                if isinstance(level.order, StopLossOrder) == is_stop
            ]
            traces.append(_level_trace(kind, color, name))
        trades = [trade for trade in self._trades.values() if trade.pair == pair]
        traces.extend(_trade_traces(trades))
        return traces

    def _slider(
        self,
        pairs: list[CurrencyPair],
        levels: list[_Level],
        window: int,
        max_steps: int,
    ) -> dict:
        times = sorted({time for pair in pairs for time in self._times[pair]})
        if not times:
            return {"steps": []}
        # Start once a full window exists, and end exactly on the last bar
        first = min(window, len(times) - 1)
        stride = max(1, -(-(len(times) - first) // max_steps))
        indices = list(range(first, len(times), stride))
        if indices[-1] != len(times) - 1:
            indices.append(len(times) - 1)

        steps = []
        for index in indices:
            start, end = times[max(0, index - window)], times[index]
            layout: dict = {"xaxis.range": [start, end]}
            for row, pair in enumerate(pairs, start=1):
                y_range = self._y_range(pair, levels, start, end)
                if y_range is not None:
                    axis = "yaxis" if row == 1 else f"yaxis{row}"
                    layout[f"{axis}.range"] = y_range
            steps.append(
                {
                    "method": "relayout",
                    "args": [layout],
                    "label": end.strftime("%m-%d %H:%M"),
                }
            )
        return {"steps": steps, "currentvalue": {"prefix": "UTC "}, "pad": {"t": 40}}

    def _y_range(
        self,
        pair: CurrencyPair,
        levels: list[_Level],
        start: datetime,
        end: datetime,
    ) -> list[float] | None:
        times = self._times[pair]
        visible = self._candles[pair][
            bisect_left(times, start) : bisect_right(times, end)
        ]
        prices = [c.low for c in visible] + [c.high for c in visible]
        prices += [
            level.order.price
            for level in levels
            if level.order.pair == pair and level.start <= end and level.end >= start
        ]
        if not prices:
            return None
        low, high = min(prices), max(prices)
        padding = (high - low) * 0.05 or pair.pip_size
        return [low - padding, high + padding]


def _level_trace(levels: list[_Level], color: str, name: str) -> go.Scatter:
    x: list[datetime | None] = []
    y: list[float | None] = []
    for level in levels:
        x += [level.start, level.end, None]
        y += [level.order.price, level.order.price, None]
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"color": color, "width": 1.5, "dash": "dash"},
        name=name,
        hovertemplate=f"{name} %{{y}}<extra></extra>",
    )


def _trade_traces(trades: list[Trade]) -> list[go.Scatter]:
    entries = go.Scatter(
        x=[_plot_time(trade.entry_time) for trade in trades],
        y=[trade.entry_price for trade in trades],
        mode="markers",
        marker={
            "symbol": [
                "triangle-up" if trade.direction == Direction.LONG else "triangle-down"
                for trade in trades
            ],
            "color": [
                TAKE_PROFIT_COLOR
                if trade.direction == Direction.LONG
                else STOP_LOSS_COLOR
                for trade in trades
            ],
            "size": 11,
        },
        text=[
            f"#{trade.id} {trade.direction.name} {trade.units:g}" for trade in trades
        ],
        hovertemplate="Entry %{text} @ %{y}<extra></extra>",
        name="Entries",
    )
    closed = [
        (trade, _plot_time(trade.close_time))
        for trade in trades
        if trade.close_time is not None
    ]
    exits = go.Scatter(
        x=[time for _, time in closed],
        y=[trade.close_price for trade, _ in closed],
        mode="markers",
        marker={"symbol": "x", "color": EXIT_COLOR, "size": 10},
        text=[f"#{trade.id} P/L {trade.realized_pl:.2f}" for trade, _ in closed],
        hovertemplate="Exit %{text} @ %{y}<extra></extra>",
        name="Exits",
    )
    return [entries, exits]


def _plot_time(time: datetime) -> datetime:
    # Plotly drops timezones; simulation times are UTC
    return time.replace(tzinfo=None)
