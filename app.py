"""
app.py
======
Interactive research dashboard.  Run with:  ``streamlit run app.py``

Ties every module together: data_loader -> strategies -> execution_engine ->
analytics -> validation, rendered with the Plotly figures from charts.py.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import analytics
import charts
import data_loader as dl
import validation as val
from execution_engine import BacktestEngine, EngineConfig
from strategies import STRATEGY_REGISTRY, get_strategy, list_strategies

st.set_page_config(page_title="QuantLab Research", page_icon="📈", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stMetric"]{background:#131a26;border:1px solid #243044;
        padding:12px 14px;border-radius:10px;}
    div[data-testid="stMetricLabel"] p{color:#9aa4b2;font-size:0.82rem;}
    .block-container{padding-top:1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

INTERVALS = ["1d", "1h", "15m", "5m"]
# (train_days, test_days) defaults for walk-forward, per timeframe
WFO_DEFAULTS = {"1d": (730, 180), "1h": (180, 30), "15m": (30, 7), "5m": (30, 7)}


# --------------------------------------------------------------------------- #
# Data / backtest plumbing
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading market data...", ttl=3600)
def load_data(ticker: str, start: str, end: str, interval: str, source: str, auto_adjust: bool,
              extended: bool = False):
    """Cached wrapper around :func:`data_loader.load_market_data` (Parquet cache sits underneath)."""
    df, rep = dl.load_market_data(ticker, start, end, interval, source=source, auto_adjust=auto_adjust,
                                  extended=extended)
    return df, rep.to_dict()


def execute(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Load data, generate causal signals and run the engine for the submitted settings."""
    strategy = get_strategy(inp["strategy"], **inp["params"])
    interval = inp["interval"]
    start, end = pd.Timestamp(inp["start"]), pd.Timestamp(inp["end"])
    if end <= start:
        raise ValueError("End date must be after start date.")
    minutes = dl.INTERVAL_MINUTES[interval]
    bars_per_day = 1 if interval == "1d" else max(1, 390 // minutes)
    pad_days = int(math.ceil(strategy.warmup_bars / bars_per_day) * 1.6) + 4
    load_start = (start - pd.Timedelta(days=pad_days)).strftime("%Y-%m-%d")

    extended = STRATEGY_REGISTRY[inp["strategy"]].data_session == "extended" and interval != "1d"
    df, rep = load_data(inp["ticker"], load_start, end.strftime("%Y-%m-%d"), interval,
                        inp["source"], inp["auto_adjust"], extended)
    signals = strategy.generate_signals(df)
    cfg = EngineConfig(
        initial_capital=inp["capital"], risk_per_trade=inp["risk_pct"] / 100.0,
        commission_per_trade=inp["comm_order"], commission_per_unit=inp["comm_unit"],
        slippage_bps=inp["slip_bps"], slippage_vol_coeff=inp["slip_vol"],
        stop_slippage_mult=inp["stop_slip_mult"], tp_penetration_bps=inp["tp_pen_bps"],
        multiplier=inp["multiplier"], max_leverage=inp["max_leverage"], integer_size=inp["integer_size"],
    )
    res = BacktestEngine(cfg).run(df, signals, start=start,
                                  end=end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    metrics = analytics.compute_metrics(res.equity, res.trades_df, res.in_market, cfg.initial_capital)
    bench = analytics.buy_and_hold_equity(df.loc[res.equity.index], cfg.initial_capital)
    return {"df": df, "report": rep, "cfg": cfg, "res": res, "metrics": metrics, "bench": bench,
            "inp": inp, "strategy_label": strategy.label}


def cards(items: List[tuple], ncols: int = 6) -> None:
    """Render (label, value) pairs as metric cards in rows of ``ncols``."""
    for r in range(0, len(items), ncols):
        cols = st.columns(ncols)
        for col, (label, value) in zip(cols, items[r:r + ncols]):
            col.metric(label, value)


def render_param_widgets(strategy_key: str) -> Dict[str, Any]:
    """Build one widget per declared strategy parameter."""
    out: Dict[str, Any] = {}
    for spec in STRATEGY_REGISTRY[strategy_key].PARAM_SPECS:
        key = f"p_{strategy_key}_{spec.name}"
        if spec.slider and spec.kind in ("int", "float"):
            cast = int if spec.kind == "int" else float
            out[spec.name] = cast(st.slider(spec.label, cast(spec.min), cast(spec.max), cast(spec.default),
                                            cast(spec.step or (1 if spec.kind == "int" else 0.1)),
                                            help=spec.help or None, key=key))
        elif spec.kind == "int":
            out[spec.name] = int(st.number_input(spec.label, int(spec.min), int(spec.max), int(spec.default),
                                                 int(spec.step or 1), help=spec.help or None, key=key))
        elif spec.kind == "float":
            out[spec.name] = float(st.number_input(spec.label, float(spec.min), float(spec.max),
                                                   float(spec.default), float(spec.step or 0.1),
                                                   help=spec.help or None, key=key))
        elif spec.kind == "bool":
            out[spec.name] = bool(st.checkbox(spec.label, bool(spec.default), help=spec.help or None, key=key))
        else:
            out[spec.name] = str(st.text_input(spec.label, str(spec.default), help=spec.help or None, key=key))
    return out


def round_floats(frame: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """Round float columns only (datetime columns must not be passed to DataFrame.round)."""
    out = frame.copy()
    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(digits)
    return out


def parse_numbers(text: str, kind: str) -> List[Any]:
    """'1.5, 2, 2.5' -> [1.5, 2.0, 2.5] (ints for int params). Raises ValueError on junk."""
    vals = [float(t) for t in text.replace(";", ",").split(",") if t.strip()]
    if not vals:
        raise ValueError("empty list")
    return [int(round(v)) for v in vals] if kind == "int" else vals


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
today = pd.Timestamp.today().normalize()
sb = st.sidebar
sb.title("📈 QuantLab")
sb.caption("Transparent, auditable backtesting and robustness research.")

source_label = sb.radio("Data source", ["Yahoo Finance", "Synthetic demo"], horizontal=True,
                        help="Synthetic data lets you test the plumbing offline. It says nothing about real markets.")
source = "yahoo" if source_label == "Yahoo Finance" else "synthetic"
ticker = sb.text_input("Ticker", "QQQ", help="e.g. QQQ, SPY, AAPL, MNQ=F, ES=F").strip().upper()
interval = sb.selectbox("Timeframe", INTERVALS,
                        help="Yahoo serves ~58 days of 5m/15m, ~2 years of 1h, and full history for 1d.")
strategies = list_strategies()
strategy_key = sb.selectbox("Strategy", list(strategies), format_func=lambda k: strategies[k])
sb.caption(STRATEGY_REGISTRY[strategy_key].description)
_supported = STRATEGY_REGISTRY[strategy_key].supported_intervals
if _supported and interval not in _supported:
    sb.warning(f"This strategy is designed for {' / '.join(_supported)} bars; you selected {interval}.")

default_lookback = {"1d": 365 * 5, "1h": 600, "15m": 55, "5m": 55}[interval]
with sb.form("controls"):
    start_date = st.date_input("Start date", (today - pd.Timedelta(days=default_lookback)).date(),
                               key=f"start_{interval}")
    end_date = st.date_input("End date", today.date(), key=f"end_{interval}")
    capital = st.number_input("Initial capital", 1_000.0, 1e9, 100_000.0, 1_000.0)
    risk_pct = st.number_input("Risk per trade (% of equity)", 0.05, 10.0, 1.0, 0.05)
    slip_bps = st.number_input("Slippage (bps)", 0.0, 100.0, 1.0, 0.25)
    comm_unit = st.number_input("Commission per share / contract (per side)", 0.0, 50.0, 0.0, 0.005, format="%.4f")
    comm_order = st.number_input("Commission per order (flat, per side)", 0.0, 500.0, 0.0, 0.25)
    multiplier = st.number_input("Contract multiplier", 0.01, 100_000.0, 1.0, 0.5,
                                 help="1 for stocks/ETFs, 2 for MNQ, 20 for NQ, 5 for MES, 50 for ES.")
    with st.expander("Advanced execution"):
        max_leverage = st.number_input("Max leverage (notional / equity)", 0.1, 50.0, 2.0, 0.1)
        stop_slip_mult = st.number_input("Stop-order slippage multiple", 0.0, 10.0, 1.5, 0.25)
        slip_vol = st.number_input("Volatility slippage coeff.", 0.0, 5.0, 0.0, 0.05,
                                   help="Extra bps = coeff x (prior ATR / prior close) in bps.")
        tp_pen_bps = st.number_input("TP must trade through by (bps)", 0.0, 50.0, 0.0, 0.5)
        integer_size = st.checkbox("Whole shares / contracts", True)
        auto_adjust = st.checkbox("Adjust for splits & dividends", True)
    st.markdown("**Strategy parameters**")
    params = render_param_widgets(strategy_key)
    submitted = st.form_submit_button("Run backtest", type="primary")

if submitted:
    for k in ("wfo", "mc", "oos", "audit"):
        st.session_state.pop(k, None)
    payload = {
        "ticker": ticker, "interval": interval, "source": source, "strategy": strategy_key, "params": params,
        "start": str(start_date), "end": str(end_date), "capital": float(capital), "risk_pct": float(risk_pct),
        "slip_bps": float(slip_bps), "comm_unit": float(comm_unit), "comm_order": float(comm_order),
        "multiplier": float(multiplier), "max_leverage": float(max_leverage),
        "stop_slip_mult": float(stop_slip_mult), "slip_vol": float(slip_vol), "tp_pen_bps": float(tp_pen_bps),
        "integer_size": bool(integer_size), "auto_adjust": bool(auto_adjust),
    }
    try:
        with st.spinner("Running backtest..."):
            st.session_state["bt"] = execute(payload)
        st.session_state.pop("error", None)
    except Exception as exc:  # surface any data / parameter problem to the user
        st.session_state["bt"] = None
        st.session_state["error"] = f"{type(exc).__name__}: {exc}"

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("Quant Research Dashboard")
bt: Optional[Dict[str, Any]] = st.session_state.get("bt")

if st.session_state.get("error"):
    st.error(st.session_state["error"])
if bt is None:
    st.info(
        "Choose a ticker, timeframe and strategy in the sidebar, adjust the settings, then press "
        "**Run backtest**. Signals decided on bar *t* are filled no earlier than the open of bar *t+1*; "
        "if both SL and TP are touched within one bar, the stop is assumed to hit first."
    )
    st.stop()

df: pd.DataFrame = bt["df"]
res = bt["res"]
cfg: EngineConfig = bt["cfg"]
metrics = bt["metrics"]
inp = bt["inp"]
trades = res.trades_df
rep = bt["report"]

st.caption(
    f"**{inp['ticker']}** · {inp['interval']} · {bt['strategy_label']} · "
    f"{len(res.equity):,} bars traded ({res.equity.index[0]:%Y-%m-%d} to {res.equity.index[-1]:%Y-%m-%d})"
    + (" · synthetic data" if inp["source"] == "synthetic" else "")
)
with st.expander("Data validation report"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bars loaded", f"{rep['rows_out']:,}")
    c2.metric("Missing bars (est.)", f"{rep['missing_bars_est']:,}")
    c3.metric("OHLC repaired", f"{rep['ohlc_repaired']:,}")
    c4.metric("Timezone", str(rep["timezone"]))
    st.write(f"Dropped: {rep['dropped_nan']} NaN, {rep['dropped_non_positive']} non-positive, "
             f"{rep['dropped_duplicates']} duplicate timestamps · zero-volume bars: {rep['zero_volume_bars']} · "
             f"adjusted prices: {rep['auto_adjusted']} · from cache: {rep['from_cache']}")
    for note in rep["notes"]:
        st.caption(f"• {note}")

tab_perf, tab_trades, tab_robust = st.tabs(["Performance Overview", "Trade Inspector", "Robustness & Walk-Forward"])

# ------------------------------- Performance -------------------------------- #
with tab_perf:
    if res.skipped_entries:
        st.warning(f"{res.skipped_entries} entry signal(s) skipped (size rounded to zero or stop below zero).")
    if len(trades) == 0:
        hint = ""
        if STRATEGY_REGISTRY[inp["strategy"]].data_session == "extended":
            hint = (" This strategy needs overnight bars before the 08:30 NY open (e.g. MNQ=F, or an equity "
                    "with extended-hours data); check the data report for the bar range.")
        st.warning("No trades were generated. Loosen the entry conditions or lengthen the date range." + hint)
    fm = analytics.format_metrics(metrics)
    cards([(k, fm[k]) for k in ("Total Return", "CAGR", "Ann. Volatility", "Sharpe", "Sortino", "Calmar",
                                 "Max Drawdown", "Avg DD Duration", "Win Rate", "Profit Factor",
                                 "Expectancy / Trade", "Exposure")], 6)
    cards([(k, fm[k]) for k in ("# Trades", "Net Profit", "Fees Paid", "Slippage Cost", "Payoff Ratio",
                                 "Expectancy (R)")], 6)
    st.plotly_chart(charts.equity_chart(res.equity, bt["bench"], cfg.initial_capital))
    st.plotly_chart(charts.drawdown_chart(res.equity, cfg.initial_capital))
    st.plotly_chart(charts.monthly_heatmap(res.equity, cfg.initial_capital))
    with st.expander("Full metrics table"):
        st.dataframe(analytics.metrics_table(metrics), hide_index=True)

# ----------------------------- Trade inspector ------------------------------ #
with tab_trades:
    if len(trades) == 0:
        st.info("No trades to inspect.")
    else:
        ledger = trades.copy()
        ledger.insert(0, "trade", np.arange(1, len(ledger) + 1))
        f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
        query = f1.text_input("Search the trade log", "", placeholder="e.g. 2024-03, Short, SL ...")
        dirs = f2.multiselect("Direction", ["Long", "Short"], ["Long", "Short"])
        reasons_all = sorted(ledger["exit_reason"].unique())
        reasons = f3.multiselect("Exit reason", reasons_all, reasons_all)
        outcome = f4.selectbox("Outcome", ["All", "Winners", "Losers"])
        view = ledger[ledger["direction"].isin(dirs) & ledger["exit_reason"].isin(reasons)]
        if outcome == "Winners":
            view = view[view["net_pnl"] > 0]
        elif outcome == "Losers":
            view = view[view["net_pnl"] <= 0]
        if query.strip():
            hit = view.astype(str).apply(lambda s: s.str.contains(query.strip(), case=False, regex=False)).any(axis=1)
            view = view[hit]

        shown_cols = ["trade", "entry_time", "exit_time", "direction", "entry_price", "exit_price", "size",
                      "gross_pnl", "fees", "net_pnl", "exit_reason", "r_multiple", "bars_held"]
        st.dataframe(round_floats(view[shown_cols]), hide_index=True, height=300)
        st.download_button("Download full trade ledger (CSV)", ledger.to_csv(index=False).encode(),
                           file_name=f"trades_{inp['ticker']}_{inp['strategy']}.csv", mime="text/csv")

        mode = st.radio("Chart view", ["Selected trade", "Date window"], horizontal=True)
        if mode == "Selected trade":
            if len(view) == 0:
                st.info("No trades match the current filters.")
            else:
                labels = {int(r.trade): f"#{int(r.trade)} · {r.direction} · {r.entry_time:%Y-%m-%d %H:%M} · "
                                        f"{r.net_pnl:+,.2f} · {r.exit_reason}" for r in view.itertuples()}
                pick = st.selectbox("Trade", list(labels), format_func=lambda k: labels[k])
                ctx = st.slider("Context bars either side", 10, 300, 60)
                t = ledger[ledger["trade"] == pick].iloc[0]
                i_in, i_out = df.index.get_loc(t["entry_time"]), df.index.get_loc(t["exit_time"])
                i0, i1 = max(0, i_in - ctx), min(len(df), i_out + ctx + 1)
                cards([("Net PnL", f"{t['net_pnl']:+,.2f}"), ("R multiple", f"{t['r_multiple']:+.2f}"),
                       ("Size", f"{t['size']:,.0f}"), ("Bars held", f"{int(t['bars_held'])}"),
                       ("Fees", f"{t['fees']:,.2f}"), ("Slippage", f"{t['slippage_cost']:,.2f}")], 6)
                st.plotly_chart(charts.trade_chart(df, view, i0, i1))
                st.caption("Dashed lines show the initial SL / TP. Trailing stops move after entry; "
                           "the exit marker shows the actual fill.")
        else:
            n_bars = st.slider("Bars to display", 50, min(2000, len(df)), min(300, len(df)), 10)
            max_start = max(0, len(df) - n_bars)
            first_trade_pos = int(df.index.get_loc(trades["entry_time"].iloc[0]))
            begin = st.slider("Window start (bar #)", 0, max_start, min(max_start, max(0, first_trade_pos - 20)))
            st.plotly_chart(charts.trade_chart(df, view, begin, begin + n_bars))

# ------------------------------- Robustness --------------------------------- #
with tab_robust:
    strat_cls = STRATEGY_REGISTRY[inp["strategy"]]
    st.subheader("Optimisation setup")
    st.caption("Parameters you edit here are swept on each in-sample window; all other parameters stay at the "
               "sidebar values. Comma-separated lists.")
    grid: Dict[str, List[Any]] = {}
    grid_ok = True
    spec_by_name = {s.name: s for s in strat_cls.PARAM_SPECS}
    gcols = st.columns(max(1, len(strat_cls.OPT_GRID)))
    for col, (pname, defaults) in zip(gcols, strat_cls.OPT_GRID.items()):
        spec = spec_by_name[pname]
        text = col.text_input(spec.label, ", ".join(str(v) for v in defaults),
                              key=f"grid_{inp['strategy']}_{pname}")
        try:
            grid[pname] = parse_numbers(text, spec.kind)
        except ValueError:
            grid_ok = False
            col.error("Enter numbers separated by commas.")
    n_combos = int(np.prod([len(v) for v in grid.values()])) if grid else 1
    o1, o2, o3 = st.columns(3)
    objective = o1.selectbox("Optimise for", ["sharpe", "calmar"])
    min_trades = int(o2.number_input("Min trades per window", 1, 200, 5))
    o3.metric("Grid combinations", f"{n_combos:,}")
    if n_combos > 400:
        st.error("Grid too large (> 400 combinations). Trim some lists.")
        grid_ok = False
    base_params = {k: v for k, v in inp["params"].items() if k not in grid}

    # ---- look-ahead audit ----
    with st.expander("Look-ahead bias audit"):
        st.caption("Re-runs the strategy and engine on truncated histories and checks that nothing already "
                   "decided ever changes when future bars are removed.")
        if st.button("Run causality audit"):
            with st.spinner("Auditing..."):
                st.session_state["audit"] = val.causality_audit(df, get_strategy(inp["strategy"], **inp["params"]),
                                                                cfg, n_checks=5)
        audit = st.session_state.get("audit")
        if audit:
            if audit["passed"]:
                st.success(f"PASSED: {audit['n_checks']} random cut points, signals and fills identical.")
            else:
                st.error("FAILED:\n\n" + "\n".join(f"- {f}" for f in audit["failures"]))

    # ---- single OOS split ----
    st.subheader("Single out-of-sample split")
    s1, s2 = st.columns([3, 1])
    train_frac = s1.slider("Train fraction", 0.5, 0.85, 0.7, 0.05)
    if s2.button("Run OOS split", disabled=not grid_ok):
        with st.spinner("Sweeping in-sample window..."):
            st.session_state["oos"] = val.oos_split(df, inp["strategy"], grid, cfg, train_frac, objective,
                                                    min_trades, base_params)
    oos = st.session_state.get("oos")
    if oos is not None:
        if oos.best_params is None:
            st.warning("No parameter set reached the minimum trade count in-sample.")
        else:
            st.write(f"Split at **{oos.split_time:%Y-%m-%d}** · locked parameters: `{oos.best_params}` · "
                     f"OOS/IS {oos.objective} ratio: "
                     f"**{oos.degradation:.2f}**" if oos.degradation == oos.degradation else
                     f"Split at **{oos.split_time:%Y-%m-%d}** · locked parameters: `{oos.best_params}`")
            rows = []
            for label, m in (("In-sample", oos.is_metrics), ("Out-of-sample", oos.oos_metrics)):
                rows.append({"Window": label, "CAGR": analytics._fmt(m["cagr"], "pct"),
                             "Sharpe": analytics._fmt(m["sharpe"], "num"),
                             "Calmar": analytics._fmt(m["calmar"], "num"),
                             "Max DD": analytics._fmt(m["max_drawdown"], "pct"),
                             "Win rate": analytics._fmt(m["win_rate"], "pct"),
                             "Profit factor": analytics._fmt(m["profit_factor"], "num"),
                             "Trades": int(m["n_trades"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            with st.expander("Top in-sample parameter sets"):
                st.dataframe(round_floats(oos.sweep.head(15)), hide_index=True)

    # ---- walk-forward ----
    st.subheader("Walk-forward optimisation")
    td_def, te_def = WFO_DEFAULTS[inp["interval"]]
    w1, w2, w3 = st.columns(3)
    train_days = int(w1.number_input("Train window (days)", 5, 3650, td_def, 5, key=f"trd_{inp['interval']}"))
    test_days = int(w2.number_input("Test window (days)", 2, 1000, te_def, 1, key=f"ted_{inp['interval']}"))
    anchored = w3.checkbox("Anchored (expanding train window)", False)
    if st.button("Run walk-forward", disabled=not grid_ok):
        bar = st.progress(0.0, text="Starting...")

        def _progress(frac: float, msg: str) -> None:
            bar.progress(min(1.0, max(0.0, frac)), text=msg)

        try:
            st.session_state["wfo"] = val.walk_forward(
                df, inp["strategy"], grid, cfg, train_days=train_days, test_days=test_days, anchored=anchored,
                objective=objective, min_trades=min_trades, base_params=base_params, progress=_progress)
            st.session_state["wfo_objective"] = objective
        except ValueError as exc:
            st.session_state.pop("wfo", None)
            st.error(str(exc))
        bar.empty()
    wfo = st.session_state.get("wfo")
    if wfo is not None and wfo.oos_metrics is not None:
        om = wfo.oos_metrics
        sm = wfo.summary
        cards([
            ("OOS CAGR", analytics._fmt(om["cagr"], "pct")), ("OOS Sharpe", analytics._fmt(om["sharpe"], "num")),
            ("OOS Max DD", analytics._fmt(om["max_drawdown"], "pct")),
            ("Walk-forward efficiency", analytics._fmt(sm.get("efficiency_ratio"), "num")),
            ("Profitable windows", analytics._fmt(sm.get("pct_profitable_windows"), "pct")),
            ("Param stability", analytics._fmt(sm.get("param_stability"), "pct")),
        ], 6)
        st.caption("Efficiency = mean OOS score / mean IS score (close to 1 is healthy; near 0 or negative "
                   "suggests the optimiser is fitting noise). Param stability = share of windows that chose the "
                   "most common parameter set.")
        oos_window_bench = bt["bench"].reindex(wfo.oos_equity.index).dropna()
        if len(oos_window_bench):
            oos_window_bench = oos_window_bench / oos_window_bench.iloc[0] * cfg.initial_capital
        st.plotly_chart(charts.wfo_equity_chart(wfo.oos_equity, wfo.windows, cfg.initial_capital, oos_window_bench))
        st.plotly_chart(charts.wfo_scores_chart(wfo.windows, st.session_state.get("wfo_objective", objective)))
        with st.expander("Window-by-window detail"):
            st.dataframe(round_floats(wfo.windows), hide_index=True)
    elif wfo is not None:
        st.warning("Walk-forward produced no evaluable out-of-sample data.")

    # ---- Monte Carlo ----
    st.subheader("Monte Carlo risk simulation")
    m1, m2, m3, m4, m5 = st.columns(5)
    sources = ["Backtest trades"]
    if wfo is not None and len(wfo.oos_trades) >= 5:
        sources.append("Walk-forward OOS trades")
    mc_source = m1.selectbox("Trade source", sources)
    n_sims = int(m2.number_input("Simulations", 100, 20_000, 1000, 100))
    mc_method = m3.selectbox("Method", ["shuffle", "bootstrap"],
                             help="Shuffle re-orders the same trades; bootstrap resamples with replacement.")
    ruin_pct = m4.number_input("Ruin = equity falls to (% of start)", 5.0, 95.0, 50.0, 5.0)
    dd_lim = m5.number_input("Drawdown limit (%)", 1.0, 90.0, 25.0, 1.0)
    if st.button("Run Monte Carlo"):
        src = wfo.oos_trades if mc_source.startswith("Walk") else trades
        try:
            st.session_state["mc"] = val.monte_carlo(src, cfg.initial_capital, n_sims, mc_method,
                                                     ruin_level=ruin_pct / 100.0, dd_limit=dd_lim / 100.0)
        except ValueError as exc:
            st.session_state.pop("mc", None)
            st.warning(str(exc))
    mc = st.session_state.get("mc")
    if mc is not None:
        cards([
            ("95th pct worst Max DD", f"{mc.p95_max_dd * 100:.1f}%"), ("Median Max DD", f"{mc.median_max_dd * 100:.1f}%"),
            ("Observed Max DD", f"{mc.observed_max_dd * 100:.1f}%"),
            ("Probability of ruin", f"{mc.prob_ruin * 100:.2f}%"),
            (f"P(DD ≥ {mc.dd_limit * 100:.0f}%)", f"{mc.prob_dd_breach * 100:.1f}%"),
            ("P(finish below start)", f"{mc.prob_loss * 100:.1f}%"),
        ], 6)
        st.plotly_chart(charts.mc_fan_chart(mc))
        st.plotly_chart(charts.mc_drawdown_hist(mc))
        st.caption(f"{mc.n_trades} trades x {mc.n_sims:,} {mc.method} paths. Ruin = equity touching "
                   f"{mc.ruin_level * 100:.0f}% of starting capital at any point.")
