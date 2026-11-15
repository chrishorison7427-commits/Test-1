"""
indicators.py
=============
Vectorized, strictly causal technical indicators shared by the strategies and
the execution engine.

Every function here uses only information available at or before bar ``t``
when producing the value stamped on bar ``t`` (rolling / expanding windows
that look backwards only). No centered windows, no negative shifts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's True Range. The first bar falls back to ``high - low``."""
    prev_close = df["close"].shift(1)
    parts = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing (EWM, alpha = 1 / window)."""
    return true_range(df).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average (NaN until ``window`` observations exist)."""
    return series.rolling(window, min_periods=window).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of ``series`` against its trailing mean / std."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    z = (series - mean) / std.replace(0.0, np.nan)
    return z


def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger mid / upper / lower bands."""
    mid = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + n_std * std, "lower": mid - n_std * std})


def rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Percentile rank (0..1) of the current value within the trailing ``window``
    observations (current value included). Fully vectorized via a strided view.
    Windows that contain NaN yield NaN.
    """
    x = series.to_numpy(dtype=float)
    out = np.full(x.shape, np.nan)
    if window < 2 or len(x) < window:
        return pd.Series(out, index=series.index)
    win = np.lib.stride_tricks.sliding_window_view(x, window)
    last = win[:, -1][:, None]
    with np.errstate(invalid="ignore"):
        rank = (win <= last).mean(axis=1)
    rank[np.isnan(win).any(axis=1)] = np.nan
    out[window - 1:] = rank
    return pd.Series(out, index=series.index)
