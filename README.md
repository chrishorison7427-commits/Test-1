# QuantLab: transparent backtesting and research framework

Event-driven, bar-by-bar backtester written from scratch (NumPy / pandas only). No backtrader, no vectorbt:
every fill, fee, slippage charge and stop decision lives in one readable loop in `execution_engine.py`.

## Setup

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python test_engine.py              # 27 invariant tests, should all PASS
streamlit run app.py               # opens the dark-mode dashboard
```

Use the **Synthetic demo** data source in the sidebar to check the plumbing offline.
Synthetic data says nothing about real markets.

## Layout

| File | Role |
|---|---|
| `data_loader.py` | yfinance download, Parquet cache in `./data_cache/`, validation (timezone alignment, duplicates, OHLC repair, missing-bar and split-jump detection), synthetic data generator |
| `indicators.py` | Causal ATR, rolling z-score, SMA, Bollinger, rolling percentile rank |
| `strategies.py` | `BaseStrategy`, `VolFilteredMeanReversion`, `OpeningRangeBreakout`, strategy registry |
| `execution_engine.py` | `EngineConfig`, `BacktestEngine`, trade ledger `trades_df` |
| `analytics.py` | CAGR, vol, Sharpe, Sortino, Calmar, max / average drawdown duration, win rate, profit factor, expectancy, exposure |
| `validation.py` | OOS split, walk-forward optimisation, Monte Carlo, look-ahead audit, parameter sweep |
| `charts.py` | Plotly figures used by the dashboard |
| `app.py` | Streamlit dashboard |
| `test_engine.py` | Executable specification of the engine's conventions |

## Execution conventions (the contract)

1. **No look-ahead.** A decision made at the close of bar `t` can only fill at the **open of bar `t+1`**
   (market orders), or via intrabar stop / limit triggers on bars `>= t+1`.
2. **Sizing.** `size = equity * risk_per_trade / (stop_distance * multiplier)`, floored to whole units,
   capped by `max_leverage`. The stop distance is anchored on the actual fill price.
3. **Pessimistic intrabar rules.** If a bar touches both SL and TP, the **SL fills first**. Stops that gap
   through fill at the open (never better than the stop). Take-profits fill at their limit and never improve on gaps
   (unless `tp_gap_improvement=True`).
4. **Friction.** Slippage in bps on market and stop fills (stops use `stop_slippage_mult`, plus an optional
   volatility term), commission per order, per unit, and in bps of notional. All are charged on both sides.
5. **Trailing stops** update at the close of each bar and are active from the next bar.
6. **Ledger.** Each trade records entry/exit time, direction, entry/exit price, size, gross PnL, fees, net PnL,
   exit reason (`SL`, `TP`, `Signal`, or `EndOfData` for a forced liquidation at the last bar), plus
   initial SL/TP, bars held, R multiple and slippage cost.
7. **Equity** is marked to market at each bar close. `initial + sum(net_pnl) == final equity`
   (asserted in the tests).

`validation.causality_audit` empirically tests the contract: it re-runs strategy and engine on truncated
histories and requires that signals and already-closed trades never change when future bars are removed.

## Built-in strategies

| Key | Strategy | Data |
|---|---|---|
| `mean_reversion` | Bollinger z-score fade + 200-bar regime filter + ATR volatility gate, ATR SL/TP | daily or intraday |
| `orb` | Opening-range breakout, volume confirmation, trailing ATR stop | intraday (5m / 15m / 1h) |
| `ict_sweep` | ICT liquidity sweep in the NY killzone (below) | intraday **with overnight bars** |
| `vwap_orb` | VWAP Opening Range Momentum (below) | 5m / 15m intraday, needs volume |
| `rsi2_pullback` | Regime-filtered RSI(2) pullback (below) | 1h / 1d |

### VWAP Opening Range Momentum (`vwap_orb`)

All clocks are `America/New_York`. Anchored VWAP = cumulative volume-weighted typical price (H+L+C)/3 from 09:30,
restarted each day. Opening range = high / low of the bars completely finished inside 09:30-10:00.
Signal bars start at or after 10:00 and the entry bar must open before 14:30 (slider). **Max 1 trade per day**
(first trigger). **Long:** close > ORH and close > VWAP. **Short:** close < ORL and close < VWAP.
Stop = 1.5 x ATR, target = 1.5 x the stop distance (both sliders), measured from the real next-open fill.

### Regime-Filtered RSI(2) Pullback (`rsi2_pullback`)

Long-only. **Entry:** close > 200 SMA and Wilder RSI(2) < 10, filled at the next open. **Exit:** close above the
5 SMA (filled at the next open). **Safety stop:** 3.0 x ATR(14) below the actual fill, no profit target.
All windows and thresholds are sliders. The dashboard warns when the timeframe is not the one a strategy was
designed for.

### Force-flat time

For intraday strategies the engine exits at the open of the **last bar that starts at or before the flat time**,
so a position is never held past it: 5m bars and 15:55 give an exit at 15:55, 15m bars give 15:45.
Early-close days (half-days) are not detected.

### ICT Liquidity Sweep (`ict_sweep`)

All clocks are `America/New_York`. Naive timestamps are treated as UTC and converted, tz-aware ones are converted.

1. **Range:** high / low of the bars from 00:00 to 08:29 NY = buy-side / sell-side liquidity (all times are inputs).
2. **Window:** the entry bar must open inside 08:30 to `kz_end_hour` (slider, default 11:00). Max 1 trade per NY day
   (first trigger wins). Flat at the open of the first bar starting at or after 15:55.
3. **Long:** a bar trades below the range low (sweep), then a bar closes back above it (reclaim, within
   `max_reclaim_bars`). **Confirmation toggle ON:** a *later* bar must show a bullish displacement candle
   (body >= `disp_atr` x ATR) **or** a bullish FVG (`low[t] > high[t-2]`, bullish middle candle, formed after the sweep)
   **or** an MSS (close above the highest high since the sweep began), within `confirm_bars`.
   **OFF:** the reclaim close is the signal. Shorts mirror around the range high.
4. **Stop:** sweep extreme -/+ `sl_buffer_atr` x ATR (slider). **Target:** `rr` x the real stop distance at the fill
   (slider), or the opposing range extreme (`tp_use_liquidity`, skipped when it offers less than `min_rr`).
5. **Fills:** signal on bar `t` close, fill at bar `t+1` open. Size = 1% risk / |fill - stop|. If the next open
   already gaps through the stop or beyond the target, the trade is skipped (counted in the dashboard).

**Data:** `MNQ=F` already includes the overnight session. For an equity or ETF such as QQQ the app requests
Yahoo's extended-hours bars automatically, but those only start around 04:00 NY, so the "00:00" range is really
04:00 to 08:29. Yahoo keeps only about 58 days of 5m / 15m bars, so a longer history needs another data source.

**Interpretation caveats:** FVG / MSS / displacement are rule-based approximations of discretionary concepts
(the code definitions above are explicit so you can audit and change them). Only the first trigger per day is
traded, so expect roughly 20 trades a month at most; with so few trades, treat any Sharpe as very noisy.

## Adding a strategy

```python
from strategies import BaseStrategy, ParamSpec, STRATEGY_REGISTRY
from indicators import atr, sma

class SmaCross(BaseStrategy):
    name, label = "sma_cross", "SMA Cross"
    PARAM_SPECS = (ParamSpec("fast", "Fast", "int", 20, 2, 100, 1),
                   ParamSpec("slow", "Slow", "int", 100, 20, 400, 5),
                   ParamSpec("sl_atr", "Stop (x ATR)", "float", 2.0, 0.5, 6.0, 0.25))
    OPT_GRID = {"fast": [10, 20, 30], "slow": [100, 150, 200]}

    def generate_signals(self, df):
        p, a = self.params, atr(df, 14)
        f, s = sma(df["close"], p["fast"]), sma(df["close"], p["slow"])
        up = (f > s) & (f.shift(1) <= s.shift(1))
        dn = (f < s) & (f.shift(1) >= s.shift(1))
        signal = up.astype(int) - dn.astype(int)
        return self.build_signal_frame(df, signal, a, (p["sl_atr"] * a).where(signal != 0),
                                       exit_long=dn, exit_short=up)

STRATEGY_REGISTRY[SmaCross.name] = SmaCross
```

Rule: use only data up to and including bar `t` when producing the signal for bar `t`. Run the audit to check.

## Data notes

* Yahoo serves roughly 58 days of `5m` / `15m` data, about 2 years of `1h`, and full history for `1d`. Requests are
  clipped automatically (the report says so).
* Daily and equity data are split/dividend adjusted by default. Futures (`MNQ=F`) come as a continuous series with
  roll gaps. Set the contract multiplier (MNQ = 2) in the sidebar.
* Missing bars are reported, never forward-filled, so the backtest never trades invented prices.
* The ORB strategy expects bars stamped in exchange-local time with the session defined by `session_open` /
  `session_close` (default 09:30-16:00; adjust for futures).

## Interpreting robustness output

* **Walk-forward efficiency** (mean OOS score / mean IS score): near 1 is healthy; near 0 or negative means the
  optimiser is fitting noise.
* **Monte Carlo shuffle** keeps the same trades, so terminal equity is unchanged and only drawdown path risk varies.
  Use **bootstrap** to also vary terminal wealth.
* Parameter sweeps are subject to selection bias. Prefer parameter regions that are stable across windows over
  a single best point.
