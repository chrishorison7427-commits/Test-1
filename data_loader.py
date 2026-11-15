"""
data_loader.py
==============
Historical OHLCV acquisition (yfinance), local Parquet caching and validation.

Public API
----------
fetch_ohlcv(...)            -> (DataFrame, ValidationReport)   Yahoo Finance + cache
generate_synthetic_ohlcv()  -> DataFrame                       offline demo / test data
load_market_data(...)       -> (DataFrame, ValidationReport)   source switch
validate_ohlcv(...)         -> (DataFrame, ValidationReport)   cleaning + diagnostics

All frames returned have a sorted, de-duplicated DatetimeIndex and lower-case
float columns: open, high, low, close, volume.

* Daily data  : tz-naive, normalized to midnight (the exchange-local trading date).
* Intraday    : tz-aware (exchange-local timezone as delivered by Yahoo, or the
                ``tz`` argument if given). Bar timestamps are bar *start* times.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("./data_cache")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
DateLike = Union[str, date, datetime, pd.Timestamp]

_INTERVAL_ALIASES = {
    "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
    "60m": "1h", "1h": "1h", "90m": "90m", "1d": "1d", "d": "1d", "daily": "1d",
}
INTERVAL_MINUTES = {
    "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "90m": 90, "1h": 60, "1d": 1440,
}
# Yahoo Finance look-back ceilings (days) for intraday requests, with a safety margin.
YF_LOOKBACK_LIMIT_DAYS = {
    "1m": 6, "2m": 58, "5m": 58, "15m": 58, "30m": 58, "90m": 58, "1h": 728,
}


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    """Diagnostics produced while cleaning a raw OHLCV frame."""

    ticker: str = ""
    interval: str = "1d"
    rows_in: int = 0
    rows_out: int = 0
    dropped_nan: int = 0
    dropped_non_positive: int = 0
    dropped_duplicates: int = 0
    ohlc_repaired: int = 0
    zero_volume_bars: int = 0
    missing_bars_est: int = 0
    largest_gap_minutes: float = 0.0
    suspected_split_jumps: int = 0
    timezone: str = "naive"
    first_bar: str = ""
    last_bar: str = ""
    auto_adjusted: bool = True
    from_cache: bool = False
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.ticker or 'data'} [{self.interval}] {self.rows_out} bars "
            f"({self.first_bar} -> {self.last_bar}), tz={self.timezone}, "
            f"missing~{self.missing_bars_est}, repaired={self.ohlc_repaired}, "
            f"dropped={self.dropped_nan + self.dropped_non_positive + self.dropped_duplicates}"
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_interval(interval: str) -> str:
    key = str(interval).strip().lower()
    if key not in _INTERVAL_ALIASES:
        raise ValueError(f"Unsupported interval '{interval}'. Use one of {sorted(set(_INTERVAL_ALIASES.values()))}.")
    return _INTERVAL_ALIASES[key]


def _to_naive_ts(x: DateLike) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _today() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None).normalize()


def _estimate_missing_bars(idx: pd.DatetimeIndex, interval: str) -> Tuple[int, float]:
    """Return (estimated missing bars, largest gap in minutes)."""
    if len(idx) < 3:
        return 0, 0.0
    if interval == "1d":
        expected = pd.bdate_range(idx[0], idx[-1])
        missing = len(expected.difference(idx))
        largest = float((pd.Series(idx[1:] - idx[:-1]).max()).total_seconds() / 60.0)
        return int(missing), largest
    step = pd.Timedelta(minutes=INTERVAL_MINUTES[interval])
    diffs = pd.Series(idx[1:] - idx[:-1])
    same_day = (idx[1:].normalize() == idx[:-1].normalize())
    within = diffs[same_day]
    gaps = within[within > step * 1.5]
    missing = int(((gaps / step).round() - 1).clip(lower=0).sum())
    largest = float(within.max().total_seconds() / 60.0) if len(within) else 0.0
    return missing, largest


# --------------------------------------------------------------------------- #
# Validation / cleaning
# --------------------------------------------------------------------------- #
def validate_ohlcv(
    df: pd.DataFrame,
    interval: str = "1d",
    tz: Optional[str] = None,
    ticker: str = "",
    auto_adjusted: bool = True,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Clean a raw OHLCV frame and report what was found.

    Steps: column normalization -> timezone alignment -> sort/de-duplicate ->
    numeric coercion -> drop NaN / non-positive rows -> repair impossible
    OHLC relationships -> missing-bar estimation -> split-jump detection.

    Missing bars are *reported, never forward-filled*: inventing bars would
    fabricate fills in a backtest.
    """
    interval = normalize_interval(interval)
    rep = ValidationReport(ticker=ticker, interval=interval, rows_in=len(df), auto_adjusted=auto_adjusted)
    if df is None or len(df) == 0:
        raise ValueError("Empty OHLCV frame supplied to validate_ohlcv().")

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "adj_close" in out.columns and not auto_adjusted:
        rep.notes.append("Raw prices requested: splits/dividends are NOT adjusted; expect artificial jumps.")
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise ValueError(f"Missing required column '{col}'.")
    if "volume" not in out.columns:
        out["volume"] = 0.0
        rep.notes.append("No volume column supplied; filled with zeros.")
    out = out[OHLCV_COLUMNS]

    # --- timezone alignment -------------------------------------------------
    idx = pd.DatetimeIndex(out.index)
    if interval == "1d":
        if idx.tz is not None:
            idx = idx.tz_localize(None)  # keep exchange-local wall date
        idx = idx.normalize()
    else:
        if idx.tz is None:
            idx = idx.tz_localize(tz or "UTC", ambiguous="NaT", nonexistent="NaT")
            if tz is None:
                rep.notes.append("Naive intraday timestamps assumed to be UTC.")
        elif tz is not None:
            idx = idx.tz_convert(tz)
    out.index = idx
    out = out[~out.index.isna()]
    out.index.name = "datetime"

    # --- ordering / duplicates ---------------------------------------------
    out = out.sort_index()
    dup = out.index.duplicated(keep="last")
    rep.dropped_duplicates = int(dup.sum())
    out = out[~dup]

    # --- numeric sanity -----------------------------------------------------
    out = out.apply(pd.to_numeric, errors="coerce")
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    nan_mask = out[["open", "high", "low", "close"]].isna().any(axis=1)
    rep.dropped_nan = int(nan_mask.sum())
    out = out[~nan_mask]
    nonpos = (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
    rep.dropped_non_positive = int(nonpos.sum())
    out = out[~nonpos]

    # --- OHLC consistency ---------------------------------------------------
    hi = out[["open", "high", "low", "close"]].max(axis=1)
    lo = out[["open", "high", "low", "close"]].min(axis=1)
    bad = (out["high"] < hi) | (out["low"] > lo)
    rep.ohlc_repaired = int(bad.sum())
    if rep.ohlc_repaired:
        out.loc[bad, "high"] = hi[bad]
        out.loc[bad, "low"] = lo[bad]
        rep.notes.append(f"Repaired {rep.ohlc_repaired} bars where high/low did not bound open/close.")

    rep.zero_volume_bars = int((out["volume"] == 0).sum())
    if len(out) == 0:
        raise ValueError("No valid bars remain after validation.")

    # --- gaps & split-like jumps -------------------------------------------
    rep.missing_bars_est, rep.largest_gap_minutes = _estimate_missing_bars(out.index, interval)
    if interval == "1d":
        rep.notes.append("Missing business days include exchange holidays.")
    else:
        rep.notes.append("Intraday gap estimate counts only within-day holes; overnight/weekend breaks excluded.")
    jump_thr = 0.35 if interval == "1d" else 0.20
    rep.suspected_split_jumps = int((out["close"].pct_change().abs() > jump_thr).sum())
    if rep.suspected_split_jumps:
        rep.notes.append(
            f"{rep.suspected_split_jumps} close-to-close moves exceed {jump_thr:.0%}: "
            "possible unadjusted split or bad tick. Inspect before trusting results."
        )

    rep.rows_out = len(out)
    rep.timezone = str(out.index.tz) if out.index.tz is not None else "naive"
    rep.first_bar = str(out.index[0])
    rep.last_bar = str(out.index[-1])
    return out.astype(float), rep


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, interval: str, start: pd.Timestamp, end: pd.Timestamp,
                auto_adjust: bool, cache_dir: Path, prepost: bool = False) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)
    tag = ("adj" if auto_adjust else "raw") + ("_pp" if prepost else "")
    name = f"{safe}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}_{tag}.parquet"
    return Path(cache_dir) / name


def _is_stale(path: Path, end: pd.Timestamp, interval: str, max_age_hours: Optional[float]) -> bool:
    """Windows ending before today are immutable; open-ended windows expire by TTL."""
    if end < _today():
        return False
    ttl = max_age_hours if max_age_hours is not None else (6.0 if interval != "1d" else 12.0)
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h > ttl


def _read_cache(path: Path) -> Tuple[pd.DataFrame, ValidationReport]:
    df = pd.read_parquet(path)
    rep = ValidationReport()
    meta = path.with_suffix(".json")
    if meta.exists():
        try:
            rep = ValidationReport(**json.loads(meta.read_text()))
        except (TypeError, ValueError):
            rep = ValidationReport()
    rep.from_cache = True
    return df, rep


def _write_cache(path: Path, df: pd.DataFrame, rep: ValidationReport) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        path.with_suffix(".json").write_text(json.dumps(rep.to_dict(), default=str))
    except ImportError:
        logger.warning("pyarrow/fastparquet not installed: Parquet cache disabled (pip install pyarrow).")
    except Exception as exc:  # pragma: no cover - disk problems must not break research
        logger.warning("Could not write cache %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Yahoo Finance
# --------------------------------------------------------------------------- #
def fetch_ohlcv(
    ticker: str,
    start: DateLike,
    end: Optional[DateLike] = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    max_age_hours: Optional[float] = None,
    tz: Optional[str] = None,
    retries: int = 3,
    prepost: bool = False,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Download (or load from ``./data_cache``) validated OHLCV bars.

    ``auto_adjust=True`` returns split- and dividend-adjusted OHLC, which is
    what you want for equity/ETF backtests. For futures tickers (``MNQ=F``)
    Yahoo supplies a continuous, unadjusted-for-roll series: expect roll gaps.
    Intraday requests are clipped to Yahoo's look-back ceilings (noted in the report).
    """
    interval = normalize_interval(interval)
    start_ts = _to_naive_ts(start)
    end_ts = _to_naive_ts(end) if end is not None else _today()
    if end_ts <= start_ts:
        raise ValueError("`end` must be after `start`.")
    notes: list = []

    limit = YF_LOOKBACK_LIMIT_DAYS.get(interval)
    if limit is not None:
        earliest = _today() - pd.Timedelta(days=limit)
        if start_ts < earliest:
            notes.append(f"Start clipped to {earliest:%Y-%m-%d}: Yahoo only serves ~{limit} days of {interval} data.")
            start_ts = earliest
        if end_ts <= start_ts:
            raise ValueError(f"Requested window is outside Yahoo's {interval} look-back range.")

    cache_dir = Path(cache_dir)
    prepost = bool(prepost) and interval != "1d"   # pre/post-market bars exist only intraday
    path = _cache_path(ticker, interval, start_ts, end_ts, auto_adjust, cache_dir, prepost)
    if use_cache and not force_refresh and path.exists() and not _is_stale(path, end_ts, interval, max_age_hours):
        try:
            df, rep = _read_cache(path)
            rep.notes = list(dict.fromkeys(notes + rep.notes))
            logger.info("Loaded %s from cache.", path.name)
            return df, rep
        except Exception as exc:
            logger.warning("Cache read failed (%s); refetching.", exc)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required: pip install yfinance") from exc

    raw = None
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.Ticker(ticker).history(
                start=start_ts.strftime("%Y-%m-%d"),
                end=(end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),  # Yahoo's end is exclusive
                interval=interval,
                auto_adjust=auto_adjust,
                actions=False,
                prepost=prepost,
            )
            if raw is not None and len(raw) > 0:
                break
        except Exception as exc:  # network / throttling
            last_err = exc
        time.sleep(1.5 * attempt)
    if raw is None or len(raw) == 0:
        raise ValueError(
            f"No data returned for {ticker} [{interval}] {start_ts:%Y-%m-%d}..{end_ts:%Y-%m-%d}. "
            f"Check the ticker and interval limits. Last error: {last_err}"
        )

    df, rep = validate_ohlcv(raw, interval=interval, tz=tz, ticker=ticker, auto_adjusted=auto_adjust)
    # Yahoo's end is exclusive; trim any extra trailing day we asked for.
    cutoff = end_ts + pd.Timedelta(days=1)
    if df.index.tz is not None:
        cutoff = cutoff.tz_localize(df.index.tz)
    df = df[df.index < cutoff]
    rep.rows_out = len(df)
    rep.notes = list(dict.fromkeys(notes + rep.notes))
    if use_cache:
        _write_cache(path, df, rep)
    return df, rep


# --------------------------------------------------------------------------- #
# Synthetic data (offline demos, unit tests, CI)
# --------------------------------------------------------------------------- #
def generate_synthetic_ohlcv(
    start: DateLike = "2022-01-03",
    end: DateLike = "2024-12-31",
    interval: str = "1d",
    seed: int = 7,
    start_price: float = 100.0,
    annual_vol: float = 0.22,
    annual_drift: float = 0.08,
    tz: str = "America/New_York",
    session: str = "rth",
) -> pd.DataFrame:
    """
    Regime-switching, session-structured synthetic OHLCV.

    Two volatility regimes (Markov), overnight gaps, trending/mean-reverting
    day-level drift, U-shaped intraday volatility and volume. Intraday bars span
    09:30-16:00 in ``tz``. For plumbing tests and demos only: it says nothing
    about real markets.
    """
    interval = normalize_interval(interval)
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(_to_naive_ts(start), _to_naive_ts(end))
    d = len(days)
    if d < 5:
        raise ValueError("Synthetic range too short.")

    if session not in ("rth", "24h"):
        raise ValueError("session must be 'rth' (09:30-16:00) or '24h' (futures-like, round the clock).")
    span_min, offset0 = (1440, 0) if session == "24h" else (390, 570)
    if interval == "1d":
        bars, minutes = 1, 1440
    else:
        minutes = INTERVAL_MINUTES[interval]
        if minutes > span_min:
            raise ValueError("Interval too long for an intraday session.")
        bars = int(np.ceil(span_min / minutes))

    # volatility regimes
    regime = np.zeros(d, dtype=int)
    for i in range(1, d):
        stay = 0.97 if regime[i - 1] == 0 else 0.93
        regime[i] = regime[i - 1] if rng.random() < stay else 1 - regime[i - 1]
    sig_day = annual_vol / np.sqrt(252) * np.where(regime == 0, 0.8, 1.9)

    gap = rng.normal(0.0, 0.35 * sig_day)
    trend = rng.normal(annual_drift / 252, 0.6 * sig_day)
    k = np.arange(bars)
    if session == "24h" and interval != "1d":
        hrs = k * minutes / 60.0   # wall-clock hour of each bar: quiet Asia, London bump, NY-open burst
        w = (0.35 + 0.65 * np.exp(-((hrs - 3.0) / 1.0) ** 2) + 1.3 * np.exp(-((hrs - 8.5) / 0.5) ** 2)
             + 2.2 * np.exp(-((hrs - 9.6) / 1.4) ** 2) + 0.9 * np.exp(-((hrs - 14.5) / 2.5) ** 2))
    else:
        w = 1.0 + 1.5 * (np.exp(-(k / max(1.0, 0.12 * bars)) ** 2)
                         + np.exp(-((bars - 1 - k) / max(1.0, 0.12 * bars)) ** 2))
    w = w / w.sum()
    bar_sig = 0.94 * sig_day[:, None] * np.sqrt(w)[None, :]
    r = trend[:, None] / bars + bar_sig * rng.normal(size=(d, bars))

    day_ret = gap + r.sum(axis=1)
    day_close_log = np.log(start_price) + np.cumsum(day_ret)
    day_open_log = day_close_log - r.sum(axis=1)
    close_log = day_open_log[:, None] + np.cumsum(r, axis=1)
    open_log = np.concatenate([day_open_log[:, None], close_log[:, :-1]], axis=1)

    o, c = np.exp(open_log), np.exp(close_log)
    wick_up = np.abs(rng.normal(size=(d, bars))) * bar_sig * 0.6
    wick_dn = np.abs(rng.normal(size=(d, bars))) * bar_sig * 0.6
    h = np.maximum(o, c) * np.exp(wick_up)
    lo = np.minimum(o, c) * np.exp(-wick_dn)
    base_vol = 2_000_000.0 / bars
    shape = w * bars
    v = base_vol * shape[None, :] * np.exp(rng.normal(0, 0.35, size=(d, bars))) * (1 + 3 * np.abs(r) / (bar_sig + 1e-12) / 10)

    if interval == "1d":
        index = pd.DatetimeIndex(days)
    else:
        offsets = pd.to_timedelta(offset0 + minutes * np.arange(bars), unit="m")
        stamps = (days.values[:, None] + offsets.values[None, :]).ravel()
        # 24h sessions cross the 02:00 DST transitions: bars that do not exist / are ambiguous become NaT
        index = pd.DatetimeIndex(stamps).tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    frame = pd.DataFrame(
        {"open": o.ravel(), "high": h.ravel(), "low": lo.ravel(), "close": c.ravel(), "volume": np.round(v.ravel())},
        index=index,
    )
    frame = frame[~frame.index.isna()]
    frame.index.name = "datetime"
    return frame


def load_market_data(
    ticker: str,
    start: DateLike,
    end: Optional[DateLike],
    interval: str = "1d",
    source: str = "yahoo",
    **kwargs,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Unified entry point: ``source`` is ``'yahoo'`` or ``'synthetic'``."""
    extended = bool(kwargs.pop("extended", False))   # strategy needs overnight / pre-market bars
    if source == "synthetic":
        raw = generate_synthetic_ohlcv(start=start, end=end or _today(), interval=interval,
                                       seed=int(kwargs.get("seed", 7)),
                                       session="24h" if extended else "rth")
        df, rep = validate_ohlcv(raw, interval=interval, ticker=f"SYNTH-{ticker}")
        rep.notes.append("Synthetic data: for plumbing tests only.")
        return df, rep
    kwargs.pop("seed", None)
    return fetch_ohlcv(ticker, start, end, interval=interval, prepost=extended, **kwargs)
