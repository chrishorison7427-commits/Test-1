"""
analytics.py
============
Performance analytics for equity curves and trade ledgers.

Conventions
-----------
* Return-based ratios (volatility, Sharpe, Sortino) are computed on **daily**
  equity returns. Intraday equity is collapsed to the last mark of each calendar
  day first, so 5-minute seasonality does not distort annualisation.
* ``periods_per_year`` defaults to 252 (use 365 for 24/7 markets).
* Drawdowns are measured on bar-close marks, relative to the running peak (the
  starting capital counts as the first peak).
* Trade statistics use *net* PnL (after commissions; slippage is already in fills).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- #
# Series helpers
# --------------------------------------------------------------------------- #
def _day_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local calendar date of each timestamp as a naive, normalised index."""
    naive = index.tz_localize(None) if index.tz is not None else index
    return naive.normalize()


def daily_equity(equity: pd.Series) -> pd.Series:
    """Last equity mark of each calendar day."""
    if equity.empty:
        return equity
    out = equity.groupby(_day_key(equity.index)).last()
    out.index.name = "date"
    return out


def daily_returns(equity: pd.Series, initial_capital: Optional[float] = None) -> pd.Series:
    """Simple daily returns; the first day is measured against ``initial_capital``."""
    de = daily_equity(equity)
    if de.empty:
        return de
    prev = de.shift(1)
    prev.iloc[0] = float(initial_capital) if initial_capital else float(de.iloc[0])
    return (de / prev - 1.0).replace([np.inf, -np.inf], np.nan).dropna()


def drawdown_series(equity: pd.Series, initial_capital: Optional[float] = None) -> pd.Series:
    """Fractional drawdown (<= 0) from the running peak."""
    peak = equity.cummax()
    if initial_capital:
        peak = peak.clip(lower=float(initial_capital))
    return equity / peak - 1.0


def drawdown_durations(equity: pd.Series, initial_capital: Optional[float] = None) -> np.ndarray:
    """
    Duration in calendar days of every drawdown episode (peak -> recovery).
    An episode still underwater at the end is measured to the last bar.
    """
    if len(equity) < 2:
        return np.array([])
    dd = drawdown_series(equity, initial_capital).to_numpy()
    under = (dd < 0).astype(int)
    edges = np.diff(np.r_[0, under, 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # first bar back at/above the peak (may equal len)
    idx = equity.index
    durations = []
    for s, e in zip(starts, ends):
        t0 = idx[s - 1] if s > 0 else idx[0]
        t1 = idx[e] if e < len(idx) else idx[-1]
        durations.append((t1 - t0).total_seconds() / 86400.0)
    return np.asarray(durations, dtype=float)


def buy_and_hold_equity(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    """Benchmark: invest everything at the first bar's open and hold (no costs)."""
    return initial_capital * df["close"] / float(df["open"].iloc[0])


def monthly_returns(equity: pd.Series, initial_capital: Optional[float] = None) -> pd.DataFrame:
    """Year x month table of compounded monthly returns (fractions)."""
    de = daily_equity(equity)
    if de.empty:
        return pd.DataFrame(columns=MONTH_LABELS)
    month_end = de.groupby([de.index.year, de.index.month]).last()
    prev = month_end.shift(1)
    prev.iloc[0] = float(initial_capital) if initial_capital else float(de.iloc[0])
    ret = month_end / prev - 1.0
    table = ret.unstack(level=1).reindex(columns=range(1, 13))
    table.columns = MONTH_LABELS
    table.index.name = "year"
    return table


# --------------------------------------------------------------------------- #
# Trade statistics
# --------------------------------------------------------------------------- #
def trade_stats(trades_df: pd.DataFrame) -> Dict[str, float]:
    """Win rate, profit factor, expectancy and friends, from the trade ledger."""
    n = len(trades_df)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": float("nan"), "profit_factor": float("nan"),
            "expectancy": float("nan"), "expectancy_r": float("nan"), "avg_win": float("nan"),
            "avg_loss": float("nan"), "payoff_ratio": float("nan"), "avg_bars_held": float("nan"),
            "net_profit": 0.0, "total_fees": 0.0, "total_slippage": 0.0,
        }
    pnl = trades_df["net_pnl"].to_numpy(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    if gross_loss > 0:
        pf = gross_win / gross_loss
    else:
        pf = float("inf") if gross_win > 0 else float("nan")
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    payoff = avg_win / abs(avg_loss) if avg_loss < 0 else float("nan")
    r = trades_df["r_multiple"].replace([np.inf, -np.inf], np.nan)
    return {
        "n_trades": int(n),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": float(pf),
        "expectancy": float(pnl.mean()),
        "expectancy_r": float(r.mean()) if r.notna().any() else float("nan"),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "payoff_ratio": float(payoff),
        "avg_bars_held": float(trades_df["bars_held"].mean()),
        "net_profit": float(pnl.sum()),
        "total_fees": float(trades_df["fees"].sum()),
        "total_slippage": float(trades_df["slippage_cost"].sum()) if "slippage_cost" in trades_df else 0.0,
    }


# --------------------------------------------------------------------------- #
# Master metrics
# --------------------------------------------------------------------------- #
def compute_metrics(
    equity: pd.Series,
    trades_df: Optional[pd.DataFrame] = None,
    in_market: Optional[pd.Series] = None,
    initial_capital: Optional[float] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Full metric set: CAGR, annualised volatility, Sharpe, Sortino, Calmar,
    max drawdown, average / max drawdown duration, win rate, profit factor,
    expectancy per trade and exposure (time in market).
    """
    if equity is None or len(equity) < 2:
        raise ValueError("Equity curve needs at least two points.")
    initial = float(initial_capital) if initial_capital else float(equity.iloc[0])
    end_eq = float(equity.iloc[-1])
    span_days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    years = max(span_days / 365.25, 1.0 / 365.25)
    total_return = end_eq / initial - 1.0
    cagr = (end_eq / initial) ** (1.0 / years) - 1.0 if end_eq > 0 else -1.0

    rets = daily_returns(equity, initial)
    ann = math.sqrt(periods_per_year)
    excess = rets - risk_free_rate / periods_per_year
    std = rets.std(ddof=1) if len(rets) > 1 else float("nan")
    vol = std * ann if std == std else float("nan")
    sharpe = excess.mean() / std * ann if (std == std and std > 0) else float("nan")
    downside = np.minimum(excess.to_numpy(), 0.0)
    dd_dev = math.sqrt(float(np.mean(downside ** 2))) if len(downside) else float("nan")
    sortino = excess.mean() / dd_dev * ann if (dd_dev == dd_dev and dd_dev > 0) else float("nan")

    dd = drawdown_series(equity, initial)
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    durations = drawdown_durations(equity, initial)

    if in_market is not None and len(in_market):
        exposure = float(in_market.astype(float).mean())
    else:
        exposure = float("nan")

    metrics: Dict[str, float] = {
        "start": str(equity.index[0]), "end": str(equity.index[-1]),
        "years": years, "initial_capital": initial, "final_equity": end_eq,
        "total_return": total_return, "cagr": cagr, "ann_volatility": vol,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_drawdown": max_dd,
        "avg_dd_duration_days": float(durations.mean()) if len(durations) else 0.0,
        "max_dd_duration_days": float(durations.max()) if len(durations) else 0.0,
        "exposure": exposure,
    }
    metrics.update(trade_stats(trades_df if trades_df is not None else pd.DataFrame(columns=["net_pnl"])))
    return metrics


def objective_value(metrics: Dict[str, float], objective: str = "sharpe") -> float:
    """Scalar score for optimisation; NaN becomes -inf so degenerate runs never win."""
    mapping = {
        "sharpe": "sharpe", "calmar": "calmar", "sortino": "sortino",
        "profit_factor": "profit_factor", "expectancy": "expectancy", "cagr": "cagr",
    }
    if objective not in mapping:
        raise ValueError(f"objective must be one of {sorted(mapping)}")
    v = metrics.get(mapping[objective], float("nan"))
    if v is None or v != v:
        return float("-inf")
    if math.isinf(v):
        return 1e9 if v > 0 else float("-inf")
    return float(v)


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def _fmt(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    if kind == "pct":
        return f"{v * 100:,.2f}%"
    if kind == "num":
        return f"{v:,.2f}"
    if kind == "money":
        return f"{v:,.2f}"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "days":
        return f"{v:,.1f} d"
    return str(v)


def format_metrics(m: Dict[str, float]) -> Dict[str, str]:
    """Human-readable strings for dashboard cards / tables."""
    spec = [
        ("Total Return", "total_return", "pct"), ("CAGR", "cagr", "pct"),
        ("Ann. Volatility", "ann_volatility", "pct"), ("Sharpe", "sharpe", "num"),
        ("Sortino", "sortino", "num"), ("Calmar", "calmar", "num"),
        ("Max Drawdown", "max_drawdown", "pct"), ("Avg DD Duration", "avg_dd_duration_days", "days"),
        ("Max DD Duration", "max_dd_duration_days", "days"), ("Win Rate", "win_rate", "pct"),
        ("Profit Factor", "profit_factor", "num"), ("Expectancy / Trade", "expectancy", "money"),
        ("Expectancy (R)", "expectancy_r", "num"), ("Exposure", "exposure", "pct"),
        ("# Trades", "n_trades", "int"), ("Net Profit", "net_profit", "money"),
        ("Fees Paid", "total_fees", "money"), ("Slippage Cost", "total_slippage", "money"),
        ("Avg Win", "avg_win", "money"), ("Avg Loss", "avg_loss", "money"),
        ("Payoff Ratio", "payoff_ratio", "num"), ("Avg Bars Held", "avg_bars_held", "num"),
    ]
    return {label: _fmt(m.get(key), kind) for label, key, kind in spec}


def metrics_table(m: Dict[str, float]) -> pd.DataFrame:
    """Two-column DataFrame (Metric, Value) for display or export."""
    fm = format_metrics(m)
    return pd.DataFrame({"Metric": list(fm.keys()), "Value": list(fm.values())})
