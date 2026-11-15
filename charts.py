"""
charts.py
=========
Plotly figure builders for the research dashboard (dark theme).

Pure functions: DataFrames / Series in, ``plotly.graph_objects.Figure`` out.
Candlestick figures use a *category* x-axis so weekends, holidays and overnight
gaps do not leave empty space between bars.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics import drawdown_series, monthly_returns
from validation import MonteCarloResult

BG = "#0b0f17"
PANEL = "#131a26"
GRID = "#243044"
TEXT = "#e6edf3"
C_STRAT = "#4cc9f0"
C_BENCH = "#9aa4b2"
C_UP = "#2ecc71"
C_DOWN = "#ff5d5d"
C_WARN = "#f5a623"


def _layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", title=dict(text=title, x=0.01, font=dict(size=15)),
        paper_bgcolor=BG, plot_bgcolor=PANEL, font=dict(color=TEXT),
        height=height, margin=dict(l=50, r=20, t=50 if title else 20, b=40),
        legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _xs(index: pd.DatetimeIndex) -> pd.Index:
    """Index -> display strings (category axis labels)."""
    daily = (index.normalize() == index).all()
    return pd.Index(index.strftime("%Y-%m-%d" if daily else "%Y-%m-%d %H:%M"))


def _x(ts, daily: bool) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d" if daily else "%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------- #
# Performance overview
# --------------------------------------------------------------------------- #
def equity_chart(equity: pd.Series, benchmark: Optional[pd.Series], initial_capital: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Strategy",
                             line=dict(color=C_STRAT, width=2)))
    if benchmark is not None and len(benchmark):
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, name="Buy & Hold",
                                 line=dict(color=C_BENCH, width=1.5, dash="dot")))
    fig.add_hline(y=initial_capital, line=dict(color=GRID, width=1))
    fig.update_yaxes(title="Equity")
    return _layout(fig, "Equity Curve vs Buy & Hold", 420)


def drawdown_chart(equity: pd.Series, initial_capital: float) -> go.Figure:
    dd = drawdown_series(equity, initial_capital) * 100.0
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", fill="tozeroy",
                               line=dict(color=C_DOWN, width=1), fillcolor="rgba(255,93,93,0.25)",
                               hovertemplate="%{y:.2f}%"))
    fig.update_yaxes(title="Drawdown (%)", ticksuffix="%")
    return _layout(fig, "Underwater Curve", 280)


def monthly_heatmap(equity: pd.Series, initial_capital: float) -> go.Figure:
    table = monthly_returns(equity, initial_capital) * 100.0
    z = table.to_numpy(dtype=float)
    text = np.where(np.isnan(z), "", np.char.add(np.round(z, 1).astype(str), "%"))
    bound = max(1.0, float(np.nanmax(np.abs(z)))) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=list(table.columns), y=[str(y) for y in table.index], text=text, texttemplate="%{text}",
        colorscale=[[0, "#c0392b"], [0.5, "#1c2433"], [1, "#27ae60"]], zmin=-bound, zmax=bound, zmid=0,
        colorbar=dict(title="%"), hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>", xgap=2, ygap=2,
    ))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(hovermode="closest")
    return _layout(fig, "Monthly Returns", max(260, 60 + 34 * len(table)))


# --------------------------------------------------------------------------- #
# Trade inspector
# --------------------------------------------------------------------------- #
def trade_chart(df: pd.DataFrame, trades: pd.DataFrame, i0: int, i1: int,
                show_levels: bool = True, height: int = 620) -> go.Figure:
    """Candles for ``df.iloc[i0:i1]`` with entry/exit markers and SL/TP segments for trades in view."""
    view = df.iloc[i0:i1]
    daily = (view.index.normalize() == view.index).all()
    xs = _xs(view.index)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=xs, open=view["open"], high=view["high"], low=view["low"], close=view["close"], name="Price",
        increasing_line_color=C_UP, decreasing_line_color=C_DOWN,
        increasing_fillcolor=C_UP, decreasing_fillcolor=C_DOWN,
    ), row=1, col=1)
    colors = np.where(view["close"] >= view["open"], C_UP, C_DOWN)
    fig.add_trace(go.Bar(x=xs, y=view["volume"], marker_color=colors, name="Volume", opacity=0.5,
                         showlegend=False), row=2, col=1)

    if len(trades):
        lo_t, hi_t = view.index[0], view.index[-1]
        sub = trades[(trades["exit_time"] >= lo_t) & (trades["entry_time"] <= hi_t)]
        in_view = set(xs)
        longs = sub[sub["direction"] == "Long"]
        shorts = sub[sub["direction"] == "Short"]
        for part, symbol, name, col in ((longs, "triangle-up", "Long entry", C_UP),
                                         (shorts, "triangle-down", "Short entry", C_DOWN)):
            part = part[part["entry_time"] >= lo_t]
            if len(part):
                fig.add_trace(go.Scatter(
                    x=[_x(t, daily) for t in part["entry_time"]], y=part["entry_price"], mode="markers", name=name,
                    marker=dict(symbol=symbol, size=13, color=col, line=dict(color="white", width=1)),
                    customdata=np.c_[part["size"], part["sl_price"], part["tp_price"].fillna(np.nan)],
                    hovertemplate="%{x}<br>%{y:.4f}<br>size %{customdata[0]:.0f}<br>SL %{customdata[1]:.4f}"
                                  "<br>TP %{customdata[2]:.4f}<extra>" + name + "</extra>"), row=1, col=1)
        ex = sub[sub["exit_time"] <= hi_t]
        if len(ex):
            fig.add_trace(go.Scatter(
                x=[_x(t, daily) for t in ex["exit_time"]], y=ex["exit_price"], mode="markers", name="Exit",
                marker=dict(symbol="x", size=11, line=dict(width=2, color="white"),
                            color=[C_UP if p > 0 else C_DOWN for p in ex["net_pnl"]]),
                customdata=np.c_[ex["net_pnl"], ex["exit_reason"]],
                hovertemplate="%{x}<br>%{y:.4f}<br>net %{customdata[0]:,.2f}<br>%{customdata[1]}<extra>Exit</extra>"),
                row=1, col=1)
        if show_levels and len(sub) <= 60:
            first = True
            for _, t in sub.iterrows():
                xa = _x(max(t["entry_time"], lo_t), daily)
                xb = _x(min(t["exit_time"], hi_t), daily)
                if xa not in in_view or xb not in in_view:
                    continue
                fig.add_trace(go.Scatter(x=[xa, xb], y=[t["sl_price"]] * 2, mode="lines", name="Initial SL",
                                         line=dict(color=C_DOWN, width=1.5, dash="dash"),
                                         legendgroup="sl", showlegend=first, hoverinfo="skip"), row=1, col=1)
                if pd.notna(t["tp_price"]):
                    fig.add_trace(go.Scatter(x=[xa, xb], y=[t["tp_price"]] * 2, mode="lines", name="TP",
                                             line=dict(color=C_UP, width=1.5, dash="dash"),
                                             legendgroup="tp", showlegend=first, hoverinfo="skip"), row=1, col=1)
                first = False
    fig.update_xaxes(type="category", nticks=12, rangeslider_visible=False)
    fig.update_yaxes(title="Price", row=1, col=1)
    fig.update_yaxes(title="Vol", row=2, col=1)
    fig.update_layout(hovermode="closest")
    return _layout(fig, "", height)


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def wfo_equity_chart(oos_equity: pd.Series, windows: pd.DataFrame, initial_capital: float,
                     benchmark: Optional[pd.Series] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=oos_equity.index, y=oos_equity.values, name="Walk-forward OOS",
                             line=dict(color=C_STRAT, width=2)))
    if benchmark is not None and len(benchmark):
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values, name="Buy & Hold",
                                 line=dict(color=C_BENCH, width=1.5, dash="dot")))
    for ts in windows["test_start"]:
        fig.add_shape(type="line", xref="x", yref="paper", x0=ts, x1=ts, y0=0, y1=1,
                      line=dict(color=GRID, width=1, dash="dot"))
    fig.add_hline(y=initial_capital, line=dict(color=GRID, width=1))
    fig.update_yaxes(title="Equity")
    return _layout(fig, "Out-of-Sample Equity (stitched walk-forward test windows)", 400)


def wfo_scores_chart(windows: pd.DataFrame, objective: str) -> go.Figure:
    fig = go.Figure()
    labels = [f"W{w}" for w in windows["window"]]
    fig.add_trace(go.Bar(x=labels, y=windows["is_score"], name=f"In-sample {objective}", marker_color=C_BENCH))
    fig.add_trace(go.Bar(x=labels, y=windows["oos_score"], name=f"Out-of-sample {objective}", marker_color=C_STRAT))
    fig.update_layout(barmode="group", hovermode="x unified")
    fig.update_yaxes(title=objective.capitalize())
    return _layout(fig, "In-Sample vs Out-of-Sample Score per Window", 320)


def mc_fan_chart(mc: MonteCarloResult) -> go.Figure:
    p = mc.percentiles
    x = p.index.to_numpy()
    fig = go.Figure()
    for path in mc.sample_paths[:25]:
        fig.add_trace(go.Scatter(x=x, y=path, mode="lines", line=dict(color="rgba(154,164,178,0.18)", width=1),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=p["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=p["p05"], fill="tonexty", fillcolor="rgba(76,201,240,0.15)",
                             line=dict(width=0), name="5th-95th pct"))
    fig.add_trace(go.Scatter(x=x, y=p["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=p["p25"], fill="tonexty", fillcolor="rgba(76,201,240,0.30)",
                             line=dict(width=0), name="25th-75th pct"))
    fig.add_trace(go.Scatter(x=x, y=p["p50"], name="Median", line=dict(color=C_STRAT, width=2.5)))
    fig.add_hline(y=mc.initial_capital, line=dict(color=GRID, width=1))
    fig.add_hline(y=mc.ruin_level * mc.initial_capital, line=dict(color=C_DOWN, width=1, dash="dash"),
                  annotation_text="ruin level", annotation_position="bottom right")
    fig.update_xaxes(title="Trade #")
    fig.update_yaxes(title="Equity")
    return _layout(fig, f"Monte Carlo Equity Fan ({mc.n_sims:,} {mc.method} paths)", 400)


def mc_drawdown_hist(mc: MonteCarloResult) -> go.Figure:
    dd = mc.max_dd * 100.0
    fig = go.Figure(go.Histogram(x=dd, nbinsx=50, marker_color=C_STRAT, opacity=0.85, name="Max drawdown"))
    fig.add_vline(x=mc.p95_max_dd * 100, line=dict(color=C_DOWN, width=2, dash="dash"),
                  annotation_text=f"95th pct {mc.p95_max_dd * 100:.1f}%", annotation_position="top right")
    fig.add_vline(x=mc.observed_max_dd * 100, line=dict(color=C_WARN, width=2),
                  annotation_text=f"observed {mc.observed_max_dd * 100:.1f}%", annotation_position="top left")
    fig.update_xaxes(title="Max drawdown (%)", ticksuffix="%")
    fig.update_yaxes(title="Simulations")
    fig.update_layout(hovermode="closest")
    return _layout(fig, "Monte Carlo Max Drawdown Distribution", 340)
