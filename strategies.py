"""
strategies.py
=============
Strategy interface plus two built-in example strategies.

A strategy is a *pure, causal* function ``DataFrame -> signal frame``. It must
produce, for every bar ``t``, only decisions that could have been made at the
close of ``t`` (rolling windows that look backwards, no negative shifts).
The execution engine then fills those decisions no earlier than the open of
bar ``t+1``. ``validation.causality_audit`` can empirically test this contract.

To add a strategy: subclass :class:`BaseStrategy`, declare ``PARAM_SPECS`` (drives
the dashboard widgets), optionally ``OPT_GRID`` (default optimisation grid),
implement ``generate_signals`` and register the class in ``STRATEGY_REGISTRY``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from indicators import atr as atr_indicator
from indicators import anchored_vwap, rolling_percentile_rank, rolling_zscore, rsi_wilder, sma

SIGNAL_COLUMNS = ["signal", "atr", "sl_dist", "tp_dist", "trail_mult", "exit_long", "exit_short"]


@dataclass(frozen=True)
class ParamSpec:
    """Declarative description of one strategy parameter (used by the UI and validation)."""

    name: str
    label: str
    kind: str                      # 'int' | 'float' | 'bool' | 'text'
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    help: str = ""
    slider: bool = False           # render as a slider instead of a number box (int / float only)


def _hhmm_to_minutes(text: str) -> int:
    try:
        hh, mm = str(text).strip().split(":")
        value = int(hh) * 60 + int(mm)
    except ValueError as exc:
        raise ValueError(f"Time '{text}' must look like HH:MM.") from exc
    if not 0 <= value < 24 * 60:
        raise ValueError(f"Time '{text}' is out of range.")
    return value


def infer_bar_minutes(index: pd.DatetimeIndex) -> int:
    """Median spacing (minutes) between consecutive bars on the same calendar day."""
    if len(index) < 3:
        raise ValueError("Not enough bars to infer the bar size.")
    diffs = pd.Series(index[1:] - index[:-1])
    same_day = index[1:].normalize() == index[:-1].normalize()
    within = diffs[same_day]
    if within.empty:
        return 1440
    return max(1, int(round(within.median().total_seconds() / 60.0)))


class BaseStrategy(ABC):
    """Abstract base class: validates parameters and standardises the signal frame."""

    name: str = "base"
    label: str = "Base strategy"
    description: str = ""
    supported_intervals: Optional[Tuple[str, ...]] = None   # timeframes it is designed for (dashboard hint)
    data_session: str = "rth"      # "extended" = needs overnight / pre-market bars (futures, or equities with prepost)
    PARAM_SPECS: Tuple[ParamSpec, ...] = ()
    OPT_GRID: Dict[str, List[Any]] = {}

    def __init__(self, **params: Any) -> None:
        specs = {p.name: p for p in self.PARAM_SPECS}
        unknown = set(params) - set(specs)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        merged: Dict[str, Any] = {}
        for pname, spec in specs.items():
            value = params.get(pname, spec.default)
            if spec.kind == "int":
                value = int(round(float(value)))
            elif spec.kind == "float":
                value = float(value)
            elif spec.kind == "bool":
                value = bool(value)
            else:
                value = str(value)
            if spec.kind in ("int", "float"):
                if spec.min is not None and value < spec.min:
                    raise ValueError(f"{pname}={value} below minimum {spec.min}")
                if spec.max is not None and value > spec.max:
                    raise ValueError(f"{pname}={value} above maximum {spec.max}")
            merged[pname] = value
        self.params: Dict[str, Any] = merged
        self._validate()

    # -- hooks --------------------------------------------------------------
    def _validate(self) -> None:
        """Cross-parameter validation (override as needed)."""

    @property
    def warmup_bars(self) -> int:
        """Bars needed before the first possible signal."""
        return 0

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a signal frame indexed like ``df`` (see :func:`build_signal_frame`)."""

    # -- helpers ------------------------------------------------------------
    @classmethod
    def defaults(cls) -> Dict[str, Any]:
        return {p.name: p.default for p in cls.PARAM_SPECS}

    def get_params(self) -> Dict[str, Any]:
        return dict(self.params)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({args})"

    @staticmethod
    def build_signal_frame(
        df: pd.DataFrame,
        signal: pd.Series,
        atr: pd.Series,
        sl_dist: pd.Series,
        tp_dist: Optional[pd.Series] = None,
        trail_mult: Optional[pd.Series] = None,
        exit_long: Optional[pd.Series] = None,
        exit_short: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Assemble the standard signal frame the engine consumes."""
        nan = pd.Series(np.nan, index=df.index)
        false = pd.Series(False, index=df.index)
        out = pd.DataFrame(
            {
                "signal": signal.reindex(df.index).fillna(0).astype(np.int8),
                "atr": atr.reindex(df.index).astype(float),
                "sl_dist": sl_dist.reindex(df.index).astype(float),
                "tp_dist": (tp_dist if tp_dist is not None else nan).reindex(df.index).astype(float),
                "trail_mult": (trail_mult if trail_mult is not None else nan).reindex(df.index).astype(float),
                "exit_long": (exit_long if exit_long is not None else false).reindex(df.index).fillna(False).astype(bool),
                "exit_short": (exit_short if exit_short is not None else false).reindex(df.index).fillna(False).astype(bool),
            },
            index=df.index,
        )
        return out[SIGNAL_COLUMNS]


# =========================================================================== #
# (a) Volatility-filtered mean reversion
# =========================================================================== #
class VolFilteredMeanReversion(BaseStrategy):
    """
    Bollinger z-score mean reversion, gated by a trend regime and a volatility regime.

    Entries (decided at the close, filled next open)
        Long : z <= -z_entry  AND close > SMA(regime_window)  AND vol-percentile <= max_vol_pct
        Short: z >=  z_entry  AND close < SMA(regime_window)  AND vol-percentile <= max_vol_pct
    The regime filter only buys dips inside uptrends and only sells rallies inside
    downtrends; the volatility filter stands aside when ATR% is in the top of its
    trailing distribution (where mean reversion tends to fail).
    Exits: ATR stop, optional ATR target, or z-score reverting to ``z_exit``.
    """

    name = "mean_reversion"
    label = "Volatility-Filtered Mean Reversion"
    description = "Bollinger z-score fade + 200-bar regime filter + ATR volatility gate, ATR-based SL/TP."
    PARAM_SPECS = (
        ParamSpec("bb_window", "Bollinger window", "int", 20, 5, 100, 1),
        ParamSpec("z_entry", "Entry |z|", "float", 2.0, 0.5, 4.0, 0.1),
        ParamSpec("z_exit", "Exit z (mean target)", "float", 0.0, -1.0, 2.0, 0.1,
                  "Long exits when z >= value; short exits when z <= -value."),
        ParamSpec("regime_window", "Regime SMA window", "int", 200, 20, 400, 5),
        ParamSpec("atr_window", "ATR window", "int", 14, 5, 50, 1),
        ParamSpec("sl_atr", "Stop (x ATR)", "float", 2.0, 0.5, 6.0, 0.25),
        ParamSpec("tp_atr", "Target (x ATR, 0 = none)", "float", 3.0, 0.0, 10.0, 0.25),
        ParamSpec("vol_window", "Vol-rank lookback", "int", 100, 20, 500, 10),
        ParamSpec("max_vol_pct", "Max vol percentile (1 = off)", "float", 0.85, 0.3, 1.0, 0.05),
        ParamSpec("allow_short", "Allow shorts", "bool", True),
    )
    OPT_GRID = {
        "z_entry": [1.5, 2.0, 2.5],
        "bb_window": [10, 20, 30],
        "sl_atr": [1.5, 2.5],
        "tp_atr": [2.0, 4.0],
    }

    @property
    def warmup_bars(self) -> int:
        p = self.params
        return max(p["regime_window"], p["bb_window"], p["atr_window"] + p["vol_window"])

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        atr_s = atr_indicator(df, p["atr_window"])
        z = rolling_zscore(close, p["bb_window"])
        regime = sma(close, p["regime_window"])
        if p["max_vol_pct"] >= 1.0:
            vol_ok = atr_s.notna()
        else:
            vol_rank = rolling_percentile_rank(atr_s / close, p["vol_window"])
            vol_ok = vol_rank <= p["max_vol_pct"]

        long_sig = (z <= -p["z_entry"]) & (close > regime) & vol_ok
        short_sig = (z >= p["z_entry"]) & (close < regime) & vol_ok & p["allow_short"]
        signal = long_sig.astype(int) - short_sig.astype(int)

        sl = (p["sl_atr"] * atr_s).where(signal != 0)
        tp = (p["tp_atr"] * atr_s).where(signal != 0) if p["tp_atr"] > 0 else None
        return self.build_signal_frame(
            df, signal, atr_s, sl, tp_dist=tp,
            exit_long=(z >= p["z_exit"]), exit_short=(z <= -p["z_exit"]),
        )


# =========================================================================== #
# (b) Intraday opening-range breakout
# =========================================================================== #
class OpeningRangeBreakout(BaseStrategy):
    """
    Opening Range Breakout with volume confirmation and a trailing ATR stop.

    * Opening range = high/low of the first ``or_minutes`` after ``session_open``.
    * Long on the first bar that *closes* above the range high (short: below the low),
      only if bar volume >= ``vol_mult`` x trailing average volume.
    * Initial stop ``sl_atr`` x ATR; chandelier-style trailing stop ``trail_atr`` x ATR
      measured from the best price since entry (updated at each close).
    * No new entries within ``no_entry_minutes_before_close`` of the close; positions are
      flattened at the open of the last bar of the session (clock-based, hence causal).
    * At most one long and one short signal per day.
    Requires intraday bars stamped in the exchange's local time.
    """

    name = "orb"
    supported_intervals = ("5m", "15m", "1h")
    label = "Intraday Opening Range Breakout"
    description = "Opening-range breakout + volume confirmation + trailing ATR stop, flat by the close."
    PARAM_SPECS = (
        ParamSpec("or_minutes", "Opening range (min)", "int", 30, 5, 120, 5),
        ParamSpec("session_open", "Session open (HH:MM)", "text", "09:30"),
        ParamSpec("session_close", "Session close (HH:MM)", "text", "16:00"),
        ParamSpec("vol_mult", "Volume multiple", "float", 1.2, 0.0, 5.0, 0.1),
        ParamSpec("vol_window", "Volume avg bars", "int", 20, 5, 200, 1),
        ParamSpec("use_volume", "Require volume confirmation", "bool", True),
        ParamSpec("atr_window", "ATR window", "int", 14, 5, 50, 1),
        ParamSpec("sl_atr", "Initial stop (x ATR)", "float", 1.5, 0.5, 6.0, 0.25),
        ParamSpec("trail_atr", "Trailing stop (x ATR, 0 = off)", "float", 2.5, 0.0, 8.0, 0.25),
        ParamSpec("tp_atr", "Target (x ATR, 0 = none)", "float", 0.0, 0.0, 15.0, 0.25),
        ParamSpec("no_entry_minutes_before_close", "No entries last (min)", "int", 60, 0, 240, 5),
        ParamSpec("allow_long", "Allow longs", "bool", True),
        ParamSpec("allow_short", "Allow shorts", "bool", True),
    )
    OPT_GRID = {
        "or_minutes": [15, 30, 60],
        "sl_atr": [1.0, 1.5, 2.5],
        "trail_atr": [1.5, 2.5, 4.0],
        "vol_mult": [1.0, 1.5],
    }

    def _validate(self) -> None:
        o = _hhmm_to_minutes(self.params["session_open"])
        c = _hhmm_to_minutes(self.params["session_close"])
        if c <= o:
            raise ValueError("session_close must be later than session_open.")
        if self.params["or_minutes"] >= c - o:
            raise ValueError("Opening range must be shorter than the session.")

    @property
    def warmup_bars(self) -> int:
        return max(self.params["atr_window"], self.params["vol_window"]) + 1

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("ORB needs a DatetimeIndex.")
        bar_min = infer_bar_minutes(idx)
        if bar_min >= 1440:
            raise ValueError("Opening Range Breakout needs intraday bars (5m / 15m / 1h), not daily data.")

        open_m = _hhmm_to_minutes(p["session_open"])
        close_m = _hhmm_to_minutes(p["session_close"])
        mod = np.asarray(idx.hour * 60 + idx.minute)
        day_key = idx.normalize().asi8

        in_or = (mod >= open_m) & (mod < open_m + p["or_minutes"])
        after_or = (mod >= open_m + p["or_minutes"]) & (mod < close_m)
        mins_to_close = close_m - mod

        or_high = df["high"].where(in_or).groupby(day_key).transform("max")
        or_low = df["low"].where(in_or).groupby(day_key).transform("min")
        close = df["close"]
        atr_s = atr_indicator(df, p["atr_window"])

        if p["use_volume"]:
            avg_vol = df["volume"].rolling(p["vol_window"], min_periods=p["vol_window"]).mean().shift(1)
            vol_ok = df["volume"] >= p["vol_mult"] * avg_vol
        else:
            vol_ok = pd.Series(True, index=idx)

        can_enter = after_or & (mins_to_close > p["no_entry_minutes_before_close"])
        can_enter_s = pd.Series(can_enter, index=idx)
        long_cond = can_enter_s & (close > or_high) & vol_ok & p["allow_long"]
        short_cond = can_enter_s & (close < or_low) & vol_ok & p["allow_short"]
        long_first = long_cond & (long_cond.astype(int).groupby(day_key).cumsum() == 1)
        short_first = short_cond & (short_cond.astype(int).groupby(day_key).cumsum() == 1)
        signal = long_first.astype(int) - short_first.astype(int)

        sl = (p["sl_atr"] * atr_s).where(signal != 0)
        tp = (p["tp_atr"] * atr_s).where(signal != 0) if p["tp_atr"] > 0 else None
        trail = pd.Series(p["trail_atr"], index=idx).where(signal != 0) if p["trail_atr"] > 0 else None

        # Flatten: flag the bar *before* the last bar so the exit fills at the last bar's open.
        flatten = pd.Series((mins_to_close <= 2 * bar_min) & (mod >= open_m) & (mod < close_m), index=idx)
        return self.build_signal_frame(
            df, signal, atr_s, sl, tp_dist=tp, trail_mult=trail,
            exit_long=flatten, exit_short=flatten,
        )


# --------------------------------------------------------------------------- #
# (c) ICT liquidity sweep in the New York killzone
# =========================================================================== #
def _to_new_york(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Wall-clock New York index. Naive timestamps are assumed to be UTC (as the loader does)."""
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert("America/New_York")


class ICTLiquiditySweep(BaseStrategy):
    """
    Pre-market liquidity sweep + reclaim, traded inside the NY Open killzone (all clocks America/New_York).

    Liquidity range : high / low of the bars in [range_start, range_end) (default 00:00-08:29) = BSL / SSL.
    Entries         : only when the signal bar closes so that the NEXT bar opens inside
                      [kz_start, kz_end) (default 08:30-11:00). Max one trade per NY day.
    Long            : a bar trades below the range low (sweep), then a bar closes back above it (reclaim,
                      within ``max_reclaim_bars``). Confirmation (if enabled) must come on a LATER bar within
                      ``confirm_bars`` and be any of:
                        * displacement - bullish candle with body >= disp_atr x ATR
                        * bullish FVG  - low[t] > high[t-2] with a bullish middle candle (pattern after the sweep)
                        * MSS          - close above the highest high since the sweep began
                      With ``use_confirmation=False`` the reclaim close itself is the signal.
    Short           : exact mirror around the range high.
    Stop            : sweep extreme (lowest low / highest high since the sweep began) -/+ sl_buffer_atr x ATR.
    Target          : ``rr`` x the actual stop distance at the fill, or (tp_use_liquidity) the opposing range
                      extreme, skipped if it offers less than ``min_rr`` at the signal close.
    Exit            : stop / target, or flat at the open of the first bar that starts at/after flat_time
                      (default 15:55). No overnight holds.

    Causality: the range level is a running max / min that is final before the killzone, every condition uses
    bars <= t, and the engine fills at the open of t+1. A same-bar double sweep (both sides trigger together)
    stands aside for the day. Needs intraday bars that include the overnight session (``MNQ=F``, or equities
    fetched with extended hours).
    """

    name = "ict_sweep"
    supported_intervals = ("5m", "15m")
    label = "ICT Liquidity Sweep (NY Killzone)"
    description = ("Pre-08:30 NY range sweep + reclaim, optional FVG / MSS / displacement confirmation, "
                   "SL beyond the sweep, R-multiple or liquidity target, 1 trade/day, flat 15:55.")
    data_session = "extended"
    PARAM_SPECS = (
        ParamSpec("range_start", "Liquidity range start (NY HH:MM)", "text", "00:00"),
        ParamSpec("range_end", "Liquidity range end (NY HH:MM)", "text", "08:30",
                  help="The range covers [start, end), i.e. 00:00-08:29 by default."),
        ParamSpec("kz_start", "Killzone start (NY HH:MM)", "text", "08:30"),
        ParamSpec("kz_end_hour", "Killzone end hour (NY)", "float", 11.0, 9.0, 15.0, 0.25,
                  "11.0 = 11:00, 10.5 = 10:30. The entry bar must open before this time.", slider=True),
        ParamSpec("flat_time", "Force-flat time (NY HH:MM)", "text", "15:55"),
        ParamSpec("use_confirmation", "Require FVG / MSS / displacement confirmation", "bool", True,
                  help="Off = trade the sweep-reclaim close directly."),
        ParamSpec("max_reclaim_bars", "Max bars from sweep to reclaim", "int", 6, 1, 40, 1),
        ParamSpec("confirm_bars", "Max bars from reclaim to confirmation", "int", 8, 1, 40, 1),
        ParamSpec("disp_atr", "Displacement body (x ATR)", "float", 1.0, 0.3, 4.0, 0.1),
        ParamSpec("atr_window", "ATR window", "int", 14, 5, 50, 1),
        ParamSpec("sl_buffer_atr", "SL buffer beyond sweep (x ATR)", "float", 0.25, 0.0, 2.0, 0.05, slider=True),
        ParamSpec("rr", "Target R multiple", "float", 2.0, 0.5, 6.0, 0.25, slider=True),
        ParamSpec("tp_use_liquidity", "Target the opposing liquidity pool instead", "bool", False),
        ParamSpec("min_rr", "Min R to liquidity target (0 = off)", "float", 1.0, 0.0, 5.0, 0.25),
        ParamSpec("allow_long", "Allow longs", "bool", True),
        ParamSpec("allow_short", "Allow shorts", "bool", True),
    )
    OPT_GRID = {
        "rr": [1.5, 2.0, 3.0],
        "sl_buffer_atr": [0.1, 0.25, 0.5],
        "kz_end_hour": [10.0, 11.0],
        "confirm_bars": [4, 8, 12],
    }

    def _validate(self) -> None:
        p = self.params
        rs, re_ = _hhmm_to_minutes(p["range_start"]), _hhmm_to_minutes(p["range_end"])
        kz_s, flat = _hhmm_to_minutes(p["kz_start"]), _hhmm_to_minutes(p["flat_time"])
        kz_e = int(round(p["kz_end_hour"] * 60))
        if re_ <= rs:
            raise ValueError("range_end must be later than range_start.")
        if kz_e <= kz_s:
            raise ValueError("Killzone end must be later than killzone start.")
        if flat <= kz_e:
            raise ValueError("flat_time must be later than the killzone end.")

    @property
    def warmup_bars(self) -> int:
        return self.params["atr_window"] + 2

    # -- long-side scan (the short side is run on mirrored prices) -----------------
    def _confirmed(self, o, h, l, c, atr, i: int, start: int, lvl: float) -> bool:
        """Displacement / FVG / MSS confirmation of a long on bar ``i`` (uses bars <= i only)."""
        if c[i] <= lvl:
            return False
        a = atr[i]
        if a == a and c[i] > o[i] and c[i] - o[i] >= self.params["disp_atr"] * a:
            return True                                              # displacement candle
        if i - 2 >= start and l[i] > h[i - 2] and c[i - 1] > o[i - 1]:
            return True                                              # bullish FVG formed after the sweep
        if i - 1 >= start and c[i] > max(h[start:i]):
            return True                                              # market-structure shift
        return False

    def _scan_long(self, o, h, l, c, lvl, atr, mod, day_key, bar_min: int, re_: int, kz_s: int, kz_e: int):
        """Per-bar state machine; returns (trigger flags, sweep extreme at the trigger bar)."""
        p = self.params
        n = len(c)
        trig = [False] * n
        ext_out = [float("nan")] * n
        cur_day = None
        done = active = False
        start = reclaim = -1
        extreme = 0.0
        for i in range(n):
            if day_key[i] != cur_day:
                cur_day, done, active, start, reclaim = day_key[i], False, False, -1, -1
            lv, m = lvl[i], mod[i]
            if done or m < re_ or m >= kz_e or lv != lv:
                continue
            if active and reclaim >= 0 and c[i] <= lv:        # reclaim failed: close back through the level
                active, reclaim = False, -1
            if active:
                extreme = min(extreme, l[i])
            elif l[i] < lv:                                    # new sweep of the liquidity level
                active, start, extreme, reclaim = True, i, l[i], -1
            if not active:
                continue
            if reclaim < 0:
                if c[i] > lv:
                    reclaim = i
                elif i - start >= p["max_reclaim_bars"]:
                    active = False
                    continue
            if reclaim < 0:
                continue
            if not p["use_confirmation"]:
                hit = reclaim == i
            else:
                if i - reclaim > p["confirm_bars"]:
                    active, reclaim = False, -1
                    continue
                hit = i > reclaim and self._confirmed(o, h, l, c, atr, i, start, lv)
            if hit and m >= kz_s and m + bar_min < kz_e and atr[i] == atr[i]:
                trig[i], ext_out[i], done = True, extreme, True
        return trig, ext_out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("ICT Liquidity Sweep needs a DatetimeIndex.")
        bar_min = infer_bar_minutes(idx)
        if bar_min >= 1440:
            raise ValueError("ICT Liquidity Sweep needs intraday bars (5m / 15m / 1h), not daily data.")

        ny = _to_new_york(idx)
        mod = np.asarray(ny.hour, dtype=np.int64) * 60 + np.asarray(ny.minute, dtype=np.int64)
        day_key = (np.asarray(ny.year, dtype=np.int64) * 10000
                   + np.asarray(ny.month, dtype=np.int64) * 100 + np.asarray(ny.day, dtype=np.int64))
        rs, re_ = _hhmm_to_minutes(p["range_start"]), _hhmm_to_minutes(p["range_end"])
        kz_s, flat_m = _hhmm_to_minutes(p["kz_start"]), _hhmm_to_minutes(p["flat_time"])
        kz_e = int(round(p["kz_end_hour"] * 60))

        # Running (causal) extremes of the liquidity range, carried forward through the rest of the NY day.
        in_range = (mod >= rs) & (mod < re_)
        pre_hi = df["high"].where(in_range).groupby(day_key).cummax().groupby(day_key).ffill()
        pre_lo = df["low"].where(in_range).groupby(day_key).cummin().groupby(day_key).ffill()
        atr_s = atr_indicator(df, p["atr_window"])

        o = df["open"].to_numpy(float).tolist()
        h = df["high"].to_numpy(float).tolist()
        l = df["low"].to_numpy(float).tolist()
        c = df["close"].to_numpy(float).tolist()
        atr_l = atr_s.to_numpy(float).tolist()
        hi_l, lo_l = pre_hi.to_numpy(float).tolist(), pre_lo.to_numpy(float).tolist()
        n = len(df)

        zero = [False] * n
        nan = [float("nan")] * n
        long_trig, long_ext = (self._scan_long(o, h, l, c, lo_l, atr_l, mod, day_key, bar_min, re_, kz_s, kz_e)
                               if p["allow_long"] else (zero, nan))
        if p["allow_short"]:   # mirror prices: a sweep of the high is a sweep of the (negated) low
            short_trig, short_ext_m = self._scan_long([-x for x in o], [-x for x in l], [-x for x in h],
                                                      [-x for x in c], [-x for x in hi_l], atr_l, mod, day_key,
                                                      bar_min, re_, kz_s, kz_e)
        else:
            short_trig, short_ext_m = zero, nan

        signal = np.zeros(n, dtype=np.int8)
        sl_px, tp_px, tp_rr = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
        buf, taken_days = p["sl_buffer_atr"], set()
        for i in sorted(i for i in range(n) if long_trig[i] or short_trig[i]):
            if day_key[i] in taken_days:
                continue                                   # one trade per NY day: first trigger wins
            taken_days.add(day_key[i])
            if long_trig[i] and short_trig[i]:
                continue                                   # same-bar double sweep: stand aside
            if long_trig[i]:
                sl = long_ext[i] - buf * atr_l[i]
                risk, target = c[i] - sl, hi_l[i]
                side = 1
            else:
                sl = -short_ext_m[i] + buf * atr_l[i]
                risk, target = sl - c[i], lo_l[i]
                side = -1
            if not risk > 0:
                continue
            if p["tp_use_liquidity"]:
                reward = side * (target - c[i])
                if reward <= 0 or (p["min_rr"] > 0 and reward / risk < p["min_rr"]):
                    continue
                tp_px[i] = target
            else:
                tp_rr[i] = p["rr"]
            signal[i], sl_px[i] = side, sl

        flat = pd.Series(_flat_flags(mod, bar_min, flat_m), index=idx)   # exit fills at the next bar's open
        frame = self.build_signal_frame(df, pd.Series(signal, index=idx), atr_s,
                                        pd.Series(np.nan, index=idx), exit_long=flat, exit_short=flat)
        frame["sl_price"], frame["tp_price"], frame["tp_rr"] = sl_px, tp_px, tp_rr
        return frame


# --------------------------------------------------------------------------- #
# (d) VWAP opening-range momentum and (e) RSI(2) pullback
# =========================================================================== #
def _ny_clock(index: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    """(minute-of-day, NY calendar-date key) arrays on the America/New_York wall clock."""
    ny = _to_new_york(index)
    mod = np.asarray(ny.hour, dtype=np.int64) * 60 + np.asarray(ny.minute, dtype=np.int64)
    day_key = (np.asarray(ny.year, dtype=np.int64) * 10000 + np.asarray(ny.month, dtype=np.int64) * 100
               + np.asarray(ny.day, dtype=np.int64))
    return mod, day_key


def _flat_flags(mod: np.ndarray, bar_min: int, flat_m: int) -> np.ndarray:
    """
    Bars after whose close the engine must exit, so that the fill lands on the open of the LAST bar that
    starts at or before ``flat_m`` (never later). 5m bars + 15:55 -> fill at 15:55; 15m bars -> fill at 15:45.
    Early-close days (half-days) are not detected.
    """
    return (mod + 2 * bar_min) > flat_m


class VWAPMomentumStrategy(BaseStrategy):
    """
    VWAP Opening Range Momentum (all clocks America/New_York, 5m or 15m bars).

    * Opening range : high / low of the bars that lie completely inside [or_start, or_end) (09:30-10:00).
    * VWAP          : volume-weighted average of the typical price (H+L+C)/3, anchored at ``vwap_anchor``
                      (09:30) and restarted every NY day.
    * Window        : signal bars start at/after or_end and the entry bar must open before ``entry_end_hour``
                      (14:30). At most ONE trade per NY day (the first trigger).
    * Long          : close > opening-range high AND close > VWAP.
    * Short         : close < opening-range low AND close < VWAP.
    * Stop / target : stop = sl_atr x ATR (1.5), target = rr x the stop distance (1.5R), both anchored on
                      the actual next-open fill by the engine.
    * Exit          : flat at the open of the last bar starting at or before flat_time (15:55).

    Causality: the opening range and VWAP only use bars <= t; a bar that is not finished by ``or_end`` never
    contributes to the range. The anchor time must fall on the same NY calendar day (use 00:00 for a
    whole-day VWAP). Needs real volume.
    """

    name = "vwap_orb"
    label = "VWAP Opening Range Momentum"
    description = ("Opening-range (09:30-10:00 NY) breakout confirmed by session VWAP; 1 trade/day, "
                   "1.5 ATR stop, 1.5R target, flat 15:55.")
    supported_intervals = ("5m", "15m")
    PARAM_SPECS = (
        ParamSpec("or_start", "Opening range start (NY HH:MM)", "text", "09:30"),
        ParamSpec("or_end", "Opening range end (NY HH:MM)", "text", "10:00"),
        ParamSpec("entry_end_hour", "Entry window end hour (NY)", "float", 14.5, 10.5, 15.5, 0.25,
                  "14.5 = 14:30. The entry bar must open before this time.", slider=True),
        ParamSpec("vwap_anchor", "VWAP anchor (NY HH:MM)", "text", "09:30"),
        ParamSpec("atr_window", "ATR window", "int", 14, 5, 50, 1, slider=True),
        ParamSpec("sl_atr", "Stop (x ATR)", "float", 1.5, 0.5, 4.0, 0.1, slider=True),
        ParamSpec("rr", "Target (R multiple of the stop)", "float", 1.5, 0.5, 4.0, 0.25, slider=True),
        ParamSpec("flat_time", "Force-flat time (NY HH:MM)", "text", "15:55"),
        ParamSpec("allow_long", "Allow longs", "bool", True),
        ParamSpec("allow_short", "Allow shorts", "bool", True),
    )
    OPT_GRID = {
        "sl_atr": [1.0, 1.5, 2.0],
        "rr": [1.0, 1.5, 2.0],
        "entry_end_hour": [12.0, 14.5],
    }

    def _validate(self) -> None:
        p = self.params
        or_s, or_e = _hhmm_to_minutes(p["or_start"]), _hhmm_to_minutes(p["or_end"])
        _hhmm_to_minutes(p["vwap_anchor"])
        if or_e <= or_s:
            raise ValueError("or_end must be later than or_start.")
        if int(round(p["entry_end_hour"] * 60)) <= or_e:
            raise ValueError("Entry window end must be later than the opening range end.")
        if _hhmm_to_minutes(p["flat_time"]) <= int(round(p["entry_end_hour"] * 60)):
            raise ValueError("flat_time must be later than the entry window end.")

    @property
    def warmup_bars(self) -> int:
        return self.params["atr_window"] + 2

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("VWAP Opening Range Momentum needs a DatetimeIndex.")
        bar_min = infer_bar_minutes(idx)
        if bar_min >= 1440:
            raise ValueError("VWAP Opening Range Momentum needs intraday bars (5m / 15m), not daily data.")
        or_s, or_e = _hhmm_to_minutes(p["or_start"]), _hhmm_to_minutes(p["or_end"])
        if bar_min > or_e - or_s:
            raise ValueError(f"{bar_min}-minute bars are too coarse for a {or_e - or_s}-minute opening range.")
        vwap_m, flat_m = _hhmm_to_minutes(p["vwap_anchor"]), _hhmm_to_minutes(p["flat_time"])
        entry_end = int(round(p["entry_end_hour"] * 60))
        mod, day_key = _ny_clock(idx)

        in_or = (mod >= or_s) & (mod + bar_min <= or_e)          # only bars fully finished by or_end
        or_hi = df["high"].where(in_or).groupby(day_key).cummax().groupby(day_key).ffill()
        or_lo = df["low"].where(in_or).groupby(day_key).cummin().groupby(day_key).ffill()
        vwap = anchored_vwap(df, day_key, mod >= vwap_m)
        if vwap.notna().sum() == 0:
            raise ValueError("VWAP is undefined: the data has no volume (or no bars after the anchor time).")
        atr_s = atr_indicator(df, p["atr_window"])
        close = df["close"]

        can_enter = (pd.Series((mod >= or_e) & (mod + bar_min < entry_end), index=idx)
                     & atr_s.notna() & or_hi.notna() & vwap.notna())
        long_c = can_enter & (close > or_hi) & (close > vwap) & p["allow_long"]
        short_c = can_enter & (close < or_lo) & (close < vwap) & p["allow_short"]
        any_c = long_c | short_c
        first = any_c & (any_c.astype(int).groupby(day_key).cumsum() == 1)       # max one trade per day
        signal = (first & long_c).astype(int) - (first & short_c).astype(int)

        sl = (p["sl_atr"] * atr_s).where(signal != 0)
        tp = (p["rr"] * sl).where(signal != 0)
        flat = pd.Series(_flat_flags(mod, bar_min, flat_m), index=idx)
        return self.build_signal_frame(df, signal, atr_s, sl, tp_dist=tp, exit_long=flat, exit_short=flat)


class RSI2MeanReversionStrategy(BaseStrategy):
    """
    Regime-Filtered RSI(2) Pullback (long-only, 1h or 1d bars).

    * Regime : close above the ``regime_window`` SMA (200).
    * Entry  : RSI(``rsi_window`` = 2, Wilder) below ``rsi_entry`` (10) on a bar close; the engine buys the
               next bar's open.
    * Exit   : close above the ``exit_window`` SMA (5) -> sells the next bar's open.
    * Safety : catastrophic stop ``sl_atr`` x ATR (3.0 x ATR(14)) below the actual fill. No profit target.

    Only one position at a time (signals while in a trade are ignored by the engine). Everything uses
    bars <= t.
    """

    name = "rsi2_pullback"
    label = "Regime-Filtered RSI(2) Pullback"
    description = ("Long-only: buy RSI(2) < 10 above the 200 SMA, exit on a close above the 5 SMA, "
                   "3 ATR catastrophic stop.")
    supported_intervals = ("1h", "1d")
    PARAM_SPECS = (
        ParamSpec("regime_window", "Regime SMA window", "int", 200, 50, 400, 5, slider=True),
        ParamSpec("exit_window", "Exit SMA window", "int", 5, 2, 20, 1, slider=True),
        ParamSpec("rsi_window", "RSI window", "int", 2, 2, 10, 1, slider=True),
        ParamSpec("rsi_entry", "Entry: RSI below", "float", 10.0, 1.0, 40.0, 1.0, slider=True),
        ParamSpec("atr_window", "ATR window", "int", 14, 5, 50, 1, slider=True),
        ParamSpec("sl_atr", "Catastrophic stop (x ATR)", "float", 3.0, 1.0, 8.0, 0.25, slider=True),
    )
    OPT_GRID = {
        "rsi_entry": [5.0, 10.0, 15.0],
        "exit_window": [3, 5, 7],
        "sl_atr": [2.0, 3.0, 4.0],
    }

    @property
    def warmup_bars(self) -> int:
        p = self.params
        return max(p["regime_window"], p["atr_window"] + 1, p["rsi_window"] + 1, p["exit_window"])

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = df["close"]
        atr_s = atr_indicator(df, p["atr_window"])
        regime = sma(close, p["regime_window"])
        exit_ma = sma(close, p["exit_window"])
        rsi = rsi_wilder(close, p["rsi_window"])
        # NaN comparisons are False, so nothing fires during indicator warm-up.
        signal = ((close > regime) & (rsi < p["rsi_entry"]) & atr_s.notna()).astype(int)
        sl = (p["sl_atr"] * atr_s).where(signal != 0)
        return self.build_signal_frame(df, signal, atr_s, sl, exit_long=(close > exit_ma))


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
STRATEGY_REGISTRY: Dict[str, type] = {
    cls.name: cls for cls in (VolFilteredMeanReversion, OpeningRangeBreakout, ICTLiquiditySweep,
                              VWAPMomentumStrategy, RSI2MeanReversionStrategy)
}


def list_strategies() -> Dict[str, str]:
    """Mapping ``key -> human label``."""
    return {k: v.label for k, v in STRATEGY_REGISTRY.items()}


def get_strategy(name: str, **params: Any) -> BaseStrategy:
    """Instantiate a registered strategy by key."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](**params)
