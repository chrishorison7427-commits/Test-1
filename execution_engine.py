"""
execution_engine.py
===================
Transparent, event-driven, bar-by-bar execution simulator.

No third-party backtesting library is used: every fill, fee, slippage charge and
stop decision happens in the single loop in :meth:`BacktestEngine.run`.

Causality contract (no look-ahead)
----------------------------------
* A strategy emits its decision on bar ``t`` using data up to and including the
  *close* of ``t``.
* That decision can only be executed at the **open of bar t+1** (market orders)
  or by **intrabar stop / limit triggers on bars >= t+1** (SL / TP).
* Stop distances are measured in price units on the signal bar (e.g. ``k * ATR_t``)
  and then *anchored on the actual fill price* at the t+1 open.
* Trailing stops are updated at the *close* of each bar and only become active
  on the following bar.

Intrabar conventions (pessimistic)
----------------------------------
* If a single bar's range touches both SL and TP, the **SL is assumed to hit first**.
* Stop orders fill at the stop price, or at the open if the bar gaps through the stop
  (never better), then pay stop-order slippage.
* Take-profits are limit orders: fill at the limit price (no slippage, no price
  improvement on gaps unless ``tp_gap_improvement=True``).
* Entries and signal exits are market orders at the open, paying slippage.

Signal frame expected from a strategy (index must equal the price index)
------------------------------------------------------------------------
``signal``      int   +1 enter long / -1 enter short / 0 none (decided at bar close)
``sl_dist``     float stop distance in price units (required for entries)
``tp_dist``     float take-profit distance in price units (NaN = no TP)
``trail_mult``  float ATR multiple for the trailing stop (NaN = none)
``sl_price``    float (optional) ABSOLUTE stop level; overrides ``sl_dist``. Size is risk-based on
                |fill - sl_price|; the entry is skipped if the next open already gaps through it.
``tp_price``    float (optional) ABSOLUTE target level; overrides ``tp_rr`` / ``tp_dist``. The entry
                is skipped if the fill is already beyond the target.
``tp_rr``       float (optional) target = fill +/- tp_rr * (actual stop distance at the fill)
``atr``         float ATR on each bar (used by trailing stop and slippage model)
``exit_long``   bool  close an open long at next open
``exit_short``  bool  close an open short at next open
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import List, Optional

import numpy as np
import pandas as pd

from indicators import atr as _atr

TRADE_COLUMNS = [
    "entry_time", "exit_time", "direction", "entry_price", "exit_price", "size",
    "gross_pnl", "fees", "net_pnl", "exit_reason",
    "sl_price", "tp_price", "bars_held", "equity_before", "risk_amount",
    "r_multiple", "return_pct", "slippage_cost",
]


@dataclass
class EngineConfig:
    """All knobs that affect execution. Every cost is explicit and auditable."""

    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01          # fraction of current equity risked to the initial stop
    commission_per_trade: float = 0.0     # flat currency amount per order (each side)
    commission_per_unit: float = 0.0      # per share / contract per side
    commission_bps: float = 0.0           # of notional per side
    slippage_bps: float = 1.0             # base adverse slippage on market/stop fills
    slippage_vol_coeff: float = 0.0       # extra bps = coeff * (prior-bar ATR / prior close) in bps
    stop_slippage_mult: float = 1.5       # stop orders slip this multiple of the market slippage
    tp_gap_improvement: bool = False      # False: TP never fills better than its limit price
    tp_penetration_bps: float = 0.0       # price must trade this far THROUGH the TP to fill it
    multiplier: float = 1.0               # contract multiplier (MNQ = 2, NQ = 20, equities = 1)
    integer_size: bool = True             # whole shares / contracts
    max_leverage: float = 2.0             # cap on notional / equity at entry
    allow_shorts: bool = True
    reverse_on_opposite_signal: bool = False
    close_at_end: bool = True             # liquidate any open position at the final close ("EndOfData")

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in (0, 1].")
        if min(self.commission_per_trade, self.commission_per_unit, self.commission_bps,
               self.slippage_bps, self.slippage_vol_coeff, self.tp_penetration_bps) < 0:
            raise ValueError("Costs must be non-negative.")
        if self.multiplier <= 0 or self.max_leverage <= 0 or self.stop_slippage_mult < 0:
            raise ValueError("multiplier / max_leverage must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)

    def with_capital(self, capital: float) -> "EngineConfig":
        return replace(self, initial_capital=float(capital))


@dataclass
class BacktestResult:
    """Output of one simulation run."""

    trades_df: pd.DataFrame
    equity: pd.Series                 # mark-to-market equity at each bar close
    positions: pd.Series              # direction (+1/-1/0) held at each bar close
    in_market: pd.Series              # True if a position existed at any point during the bar
    config: EngineConfig
    skipped_entries: int = 0
    signals: Optional[pd.DataFrame] = None

    @property
    def trades(self) -> pd.DataFrame:
        return self.trades_df

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1]) if len(self.equity) else self.config.initial_capital


def _align_ts(ts, index: pd.DatetimeIndex) -> pd.Timestamp:
    """Make a user timestamp comparable with ``index`` (handles tz-aware vs naive)."""
    t = pd.Timestamp(ts)
    if index.tz is not None:
        t = t.tz_localize(index.tz) if t.tzinfo is None else t.tz_convert(index.tz)
    elif t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


class BacktestEngine:
    """Bar-by-bar simulator. Instantiate with an :class:`EngineConfig` and call :meth:`run`."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.cfg = config or EngineConfig()

    # ------------------------------------------------------------------ utils
    def _fee(self, price: float, size: float) -> float:
        c = self.cfg
        return c.commission_per_trade + c.commission_per_unit * size + c.commission_bps * 1e-4 * price * size * c.multiplier

    @staticmethod
    def _col(signals: pd.DataFrame, name: str, default: float) -> list:
        if name in signals.columns:
            return signals[name].astype(float).to_numpy().tolist()
        return [default] * len(signals)

    def _prepare(self, df: pd.DataFrame, signals: pd.DataFrame, start, end):
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Price frame missing '{col}'.")
        if "signal" not in signals.columns:
            raise ValueError("Signal frame must contain a 'signal' column.")
        if not df.index.equals(signals.index):
            signals = signals.reindex(df.index)
        if start is not None or end is not None:
            mask = np.ones(len(df), dtype=bool)
            if start is not None:
                mask &= df.index >= _align_ts(start, df.index)
            if end is not None:
                mask &= df.index <= _align_ts(end, df.index)
            df, signals = df[mask], signals[mask]
        if len(df) < 2:
            raise ValueError("Need at least two bars to simulate.")
        return df, signals

    def _slip(self, i: int, c_prev: float, a_prev: float, is_stop: bool) -> float:
        cfg = self.cfg
        bps = cfg.slippage_bps
        if cfg.slippage_vol_coeff > 0 and i > 0 and a_prev == a_prev and c_prev > 0:
            bps += cfg.slippage_vol_coeff * a_prev / c_prev * 1e4
        if is_stop:
            bps *= cfg.stop_slippage_mult
        return bps * 1e-4

    # -------------------------------------------------------------------- run
    def run(self, df: pd.DataFrame, signals: pd.DataFrame, start=None, end=None) -> BacktestResult:
        """
        Simulate. ``signals`` must have been computed causally. If ``start`` / ``end``
        are given, trading is restricted to that window (signals computed on earlier
        history are still valid, which is how out-of-sample windows are evaluated).
        """
        cfg = self.cfg
        df, signals = self._prepare(df, signals, start, end)
        n = len(df)
        idx = df.index

        o = df["open"].astype(float).to_numpy().tolist()
        h = df["high"].astype(float).to_numpy().tolist()
        lo = df["low"].astype(float).to_numpy().tolist()
        c = df["close"].astype(float).to_numpy().tolist()
        if "atr" in signals.columns:
            atr_l = signals["atr"].astype(float).to_numpy().tolist()
        else:
            atr_l = _atr(df, 14).to_numpy().tolist()
        sig = signals["signal"].fillna(0).astype(int).to_numpy().tolist()
        sl_dist = self._col(signals, "sl_dist", float("nan"))
        tp_dist = self._col(signals, "tp_dist", float("nan"))
        trail_m = self._col(signals, "trail_mult", float("nan"))
        sl_px_l = self._col(signals, "sl_price", float("nan"))
        tp_px_l = self._col(signals, "tp_price", float("nan"))
        tp_rr_l = self._col(signals, "tp_rr", float("nan"))
        ex_long = signals["exit_long"].fillna(False).astype(bool).to_numpy().tolist() if "exit_long" in signals else [False] * n
        ex_short = signals["exit_short"].fillna(False).astype(bool).to_numpy().tolist() if "exit_short" in signals else [False] * n

        mult = cfg.multiplier
        cash = float(cfg.initial_capital)
        pos = 0
        st: dict = {}
        trades: List[dict] = []
        equity = np.empty(n)
        position = np.zeros(n, dtype=np.int8)
        in_mkt = np.zeros(n, dtype=bool)
        p_entry = None
        p_exit = False
        skipped = 0

        def book(exit_px: float, ref_px: float, i: int, reason: str, cash_in: float) -> float:
            """Realise the open position; append to the ledger; return new cash."""
            gross = st["dir"] * (exit_px - st["entry_px"]) * st["size"] * mult
            exit_fee = self._fee(exit_px, st["size"])
            fees = st["entry_fee"] + exit_fee
            net = gross - fees
            slip_cost = st["entry_slip_cost"] + abs(exit_px - ref_px) * st["size"] * mult
            trades.append({
                "entry_time": idx[st["entry_i"]], "exit_time": idx[i],
                "direction": "Long" if st["dir"] == 1 else "Short",
                "entry_price": st["entry_px"], "exit_price": exit_px, "size": st["size"],
                "gross_pnl": gross, "fees": fees, "net_pnl": net, "exit_reason": reason,
                "sl_price": st["sl0"], "tp_price": st["tp0"],
                "bars_held": i - st["entry_i"] + 1,
                "equity_before": st["equity_before"], "risk_amount": st["risk_amt"],
                "r_multiple": net / st["risk_amt"] if st["risk_amt"] > 0 else float("nan"),
                "return_pct": net / st["equity_before"],
                "slippage_cost": slip_cost,
            })
            return cash_in + gross - exit_fee

        for i in range(n):
            oi, hi, li, ci = o[i], h[i], lo[i], c[i]
            c_prev = c[i - 1] if i > 0 else ci
            a_prev = atr_l[i - 1] if i > 0 else float("nan")

            # (1) Orders decided on the previous bar's close execute at this open.
            if p_exit and pos != 0:
                s = self._slip(i, c_prev, a_prev, False)
                px = oi * (1 - s) if pos == 1 else oi * (1 + s)
                cash = book(px, oi, i, "Signal", cash)
                pos, st = 0, {}
            if p_entry is not None and pos == 0:
                d, sd, td, tm, sl_px, tp_px, tp_rr = p_entry
                s = self._slip(i, c_prev, a_prev, False)
                fill = oi * (1 + s) if d == 1 else oi * (1 - s)
                bad_levels = False
                if sl_px is not None:            # absolute stop: distance measured from the real fill
                    sd = d * (fill - sl_px)
                    sl0 = sl_px
                    bad_levels = sd <= 0         # next open already gapped through the stop
                else:
                    sl0 = fill - d * sd
                tp_abs = None
                if tp_px is not None:            # absolute target must still lie ahead of the fill
                    tp_abs = tp_px
                    bad_levels = bad_levels or d * (tp_px - fill) <= 0
                elif tp_rr is not None:
                    tp_abs = fill + d * tp_rr * sd if sd > 0 else None
                elif td is not None:
                    tp_abs = fill + d * td
                equity_now = cash
                unit_risk = sd * mult + 2.0 * cfg.commission_per_unit
                size = (equity_now * cfg.risk_per_trade) / unit_risk if unit_risk > 0 else 0.0
                size = min(size, equity_now * cfg.max_leverage / (fill * mult))
                if cfg.integer_size:
                    size = math.floor(size)
                if bad_levels or sl0 <= 0 or size <= 0 or not math.isfinite(size):
                    skipped += 1
                else:
                    entry_fee = self._fee(fill, size)
                    cash -= entry_fee
                    pos = d
                    tp0 = tp_abs
                    st = {
                        "dir": d, "entry_i": i, "entry_px": fill, "size": float(size),
                        "sl": sl0, "sl0": sl0, "tp": tp0, "tp0": tp0, "trail": tm, "ext": fill,
                        "entry_fee": entry_fee, "equity_before": equity_now,
                        "risk_amt": size * sd * mult, "entry_slip_cost": abs(fill - oi) * size * mult,
                    }
            p_entry, p_exit = None, False
            if pos != 0:
                in_mkt[i] = True

            # (2) Intrabar stop / take-profit using this bar's high & low.
            if pos != 0:
                sl, tp = st["sl"], st["tp"]
                pen = cfg.tp_penetration_bps * 1e-4
                if pos == 1:
                    sl_hit = li <= sl
                    tp_hit = tp is not None and hi >= tp * (1 + pen)
                    sl_ref = min(oi, sl)
                    tp_ref = max(oi, tp) if (tp is not None and cfg.tp_gap_improvement) else tp
                else:
                    sl_hit = hi >= sl
                    tp_hit = tp is not None and li <= tp * (1 - pen)
                    sl_ref = max(oi, sl)
                    tp_ref = min(oi, tp) if (tp is not None and cfg.tp_gap_improvement) else tp
                if sl_hit:  # pessimistic: SL wins whenever both levels are touched
                    s = self._slip(i, c_prev, a_prev, True)
                    px = sl_ref * (1 - s) if pos == 1 else sl_ref * (1 + s)
                    cash = book(px, sl_ref, i, "SL", cash)
                    pos, st = 0, {}
                elif tp_hit:
                    cash = book(tp_ref, tp_ref, i, "TP", cash)
                    pos, st = 0, {}

            # (3) End-of-bar: trailing stop update (active from the next bar) and marking.
            if pos != 0:
                tm = st["trail"]
                if tm is not None:
                    a = atr_l[i]
                    if pos == 1:
                        st["ext"] = max(st["ext"], hi)
                        if a == a and st["ext"] - tm * a > st["sl"]:
                            st["sl"] = st["ext"] - tm * a
                    else:
                        st["ext"] = min(st["ext"], li)
                        if a == a and st["ext"] + tm * a < st["sl"]:
                            st["sl"] = st["ext"] + tm * a
                equity[i] = cash + pos * (ci - st["entry_px"]) * st["size"] * mult
            else:
                equity[i] = cash
            position[i] = pos

            # (4) Decisions at this close -> pending orders for the next open.
            if i < n - 1:
                if pos != 0:
                    if (pos == 1 and ex_long[i]) or (pos == -1 and ex_short[i]):
                        p_exit = True
                    if cfg.reverse_on_opposite_signal and sig[i] == -pos:
                        ent = self._entry_order(sig[i], sl_dist[i], tp_dist[i], trail_m[i],
                                                sl_px_l[i], tp_px_l[i], tp_rr_l[i])
                        if ent is not None:
                            p_exit, p_entry = True, ent
                elif sig[i] != 0:
                    p_entry = self._entry_order(sig[i], sl_dist[i], tp_dist[i], trail_m[i],
                                                sl_px_l[i], tp_px_l[i], tp_rr_l[i])

        # (5) Optional liquidation at the last close so the ledger and equity agree.
        if pos != 0 and cfg.close_at_end:
            i = n - 1
            s = self._slip(i, c[i - 1], atr_l[i - 1], False)
            px = c[i] * (1 - s) if pos == 1 else c[i] * (1 + s)
            cash = book(px, c[i], i, "EndOfData", cash)
            equity[i] = cash
            position[i] = 0
            pos, st = 0, {}

        tdf = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        return BacktestResult(
            trades_df=tdf,
            equity=pd.Series(equity, index=idx, name="equity"),
            positions=pd.Series(position, index=idx, name="position"),
            in_market=pd.Series(in_mkt, index=idx, name="in_market"),
            config=cfg,
            skipped_entries=skipped,
            signals=signals,
        )

    def _entry_order(self, direction: int, sl_d: float, tp_d: float, tr_m: float,
                     sl_px: float = float("nan"), tp_px: float = float("nan"), tp_rr: float = float("nan")):
        """
        Validate a raw signal and normalise it into an order tuple, or None.

        Tuple: (direction, sl_distance, tp_distance, trail_mult, sl_price, tp_price, tp_rr).
        An absolute ``sl_price`` takes precedence over ``sl_dist``; target precedence is
        ``tp_price`` > ``tp_rr`` > ``tp_dist``.
        """
        if direction not in (1, -1):
            return None
        if direction == -1 and not self.cfg.allow_shorts:
            return None
        has_px = sl_px == sl_px and sl_px > 0
        has_dist = sl_d == sl_d and sl_d > 0
        if not (has_px or has_dist):
            return None
        td = tp_d if (tp_d == tp_d and tp_d > 0) else None
        tm = tr_m if (tr_m == tr_m and tr_m > 0) else None
        tpx = tp_px if (tp_px == tp_px and tp_px > 0) else None
        trr = tp_rr if (tp_rr == tp_rr and tp_rr > 0) else None
        return (direction, sl_d if has_dist else None, td, tm, sl_px if has_px else None, tpx, trr)


def run_backtest(df: pd.DataFrame, strategy, config: Optional[EngineConfig] = None,
                 start=None, end=None) -> BacktestResult:
    """Convenience wrapper: generate causal signals, then simulate."""
    signals = strategy.generate_signals(df)
    return BacktestEngine(config).run(df, signals, start=start, end=end)
