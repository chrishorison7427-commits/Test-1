"""
validation.py
=============
Overfitting and robustness tooling.

* ``oos_split``       one-shot In-Sample / Out-of-Sample test with locked parameters
* ``walk_forward``    rolling or anchored Walk-Forward Optimisation (WFO)
* ``monte_carlo``     trade-order shuffle (or bootstrap) risk simulation
* ``causality_audit`` empirical look-ahead-bias detector for strategy + engine
* ``parameter_sweep`` grid search on a window, scored by Sharpe / Calmar / ...

Method notes
------------
Signals are strictly causal, so computing them once on the full history and then
slicing is *identical* to recomputing on truncated data (``causality_audit``
verifies this). That is how indicator warm-up is handled honestly: an
out-of-sample window starts with fully warmed-up indicators but can only trade
inside the window. Each OOS window is simulated with the equity carried over from
the previous window, so the stitched curve compounds like a live account.
Open positions are liquidated at a window's last close (exit reason ``EndOfData``).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analytics import compute_metrics, objective_value
from execution_engine import BacktestEngine, BacktestResult, EngineConfig
from strategies import BaseStrategy, get_strategy

ProgressFn = Optional[Callable[[float, str], None]]
SWEEP_METRICS = ["sharpe", "sortino", "calmar", "cagr", "max_drawdown", "n_trades",
                 "win_rate", "profit_factor", "expectancy"]


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def expand_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of a ``{param: [values]}`` grid."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(list(grid[k]) for k in keys))]


class _SignalBank:
    """Computes (and memoises) causal signals once per parameter set."""

    def __init__(self, df: pd.DataFrame, strategy_name: str, base_params: Optional[Dict[str, Any]] = None):
        self.df, self.name, self.base = df, strategy_name, dict(base_params or {})
        self._cache: Dict[Tuple, Optional[pd.DataFrame]] = {}

    def get(self, combo: Dict[str, Any]) -> Optional[pd.DataFrame]:
        key = tuple(sorted(combo.items()))
        if key not in self._cache:
            try:
                strat = get_strategy(self.name, **{**self.base, **combo})
                self._cache[key] = strat.generate_signals(self.df)
            except ValueError:
                self._cache[key] = None  # invalid combination (e.g. OR longer than session)
        return self._cache[key]


def _evaluate_slice(bank: _SignalBank, combo: Dict[str, Any], i0: int, i1: int, cfg: EngineConfig,
                    objective: str, min_trades: int):
    sig = bank.get(combo)
    if sig is None or i1 - i0 < 5:
        return None
    res = BacktestEngine(cfg).run(bank.df.iloc[i0:i1], sig.iloc[i0:i1])
    m = compute_metrics(res.equity, res.trades_df, res.in_market, cfg.initial_capital)
    score = objective_value(m, objective) if m["n_trades"] >= min_trades else float("-inf")
    return score, m, res


def _sweep_slice(bank: _SignalBank, combos: List[Dict[str, Any]], i0: int, i1: int, cfg: EngineConfig,
                 objective: str, min_trades: int) -> pd.DataFrame:
    rows = []
    for combo in combos:
        out = _evaluate_slice(bank, combo, i0, i1, cfg, objective, min_trades)
        if out is None:
            continue
        score, m, _ = out
        rows.append({**combo, **{k: m[k] for k in SWEEP_METRICS}, "score": score})
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    return table


def parameter_sweep(df: pd.DataFrame, strategy_name: str, grid: Dict[str, Sequence[Any]], cfg: EngineConfig,
                    start=None, end=None, objective: str = "sharpe", min_trades: int = 10,
                    base_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Score every grid point on ``[start, end]`` (inclusive); best first."""
    idx = df.index
    i0 = 0 if start is None else int(idx.searchsorted(_ts(start, idx), "left"))
    i1 = len(df) if end is None else int(idx.searchsorted(_ts(end, idx), "right"))
    bank = _SignalBank(df, strategy_name, base_params)
    return _sweep_slice(bank, expand_grid(grid), i0, i1, cfg, objective, min_trades)


def _ts(x, index: pd.DatetimeIndex) -> pd.Timestamp:
    t = pd.Timestamp(x)
    if index.tz is not None:
        return t.tz_localize(index.tz) if t.tzinfo is None else t.tz_convert(index.tz)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _best_params(table: pd.DataFrame, grid: Dict[str, Sequence[Any]]) -> Optional[Dict[str, Any]]:
    """Top-ranked combo, with each value cast back to the type used in ``grid``."""
    if table.empty or not np.isfinite(table["score"].iloc[0]):
        return None
    row = table.iloc[0]
    out: Dict[str, Any] = {}
    for k, values in grid.items():
        v = row[k]
        v = v.item() if hasattr(v, "item") else v
        first = list(values)[0]
        if isinstance(first, bool):
            v = bool(v)
        elif isinstance(first, (int, np.integer)):
            v = int(round(float(v)))
        elif isinstance(first, (float, np.floating)):
            v = float(v)
        out[k] = v
    return out


# --------------------------------------------------------------------------- #
# One-shot OOS split
# --------------------------------------------------------------------------- #
@dataclass
class OOSSplitResult:
    split_time: pd.Timestamp
    best_params: Optional[Dict[str, Any]]
    sweep: pd.DataFrame
    is_metrics: Optional[Dict[str, float]]
    oos_metrics: Optional[Dict[str, float]]
    is_result: Optional[BacktestResult]
    oos_result: Optional[BacktestResult]
    degradation: float           # OOS score / IS score (1.0 = no decay)
    objective: str


def oos_split(df: pd.DataFrame, strategy_name: str, grid: Dict[str, Sequence[Any]], cfg: EngineConfig,
              train_frac: float = 0.7, objective: str = "sharpe", min_trades: int = 10,
              base_params: Optional[Dict[str, Any]] = None) -> OOSSplitResult:
    """Optimise on the first ``train_frac`` of bars, lock parameters, test on the rest."""
    if not 0.2 <= train_frac <= 0.9:
        raise ValueError("train_frac must be between 0.2 and 0.9.")
    n = len(df)
    cut = int(n * train_frac)
    bank = _SignalBank(df, strategy_name, base_params)
    combos = expand_grid(grid)
    table = _sweep_slice(bank, combos, 0, cut, cfg, objective, min_trades)
    best = _best_params(table, grid)
    split_time = df.index[cut]
    if best is None:
        return OOSSplitResult(split_time, None, table, None, None, None, None, float("nan"), objective)
    is_score, is_m, is_res = _evaluate_slice(bank, best, 0, cut, cfg, objective, min_trades)
    oos_score, oos_m, oos_res = _evaluate_slice(bank, best, cut, n, cfg, objective, 0)
    deg = oos_score / is_score if (np.isfinite(is_score) and np.isfinite(oos_score) and is_score > 0) else float("nan")
    return OOSSplitResult(split_time, best, table, is_m, oos_m, is_res, oos_res, deg, objective)


# --------------------------------------------------------------------------- #
# Walk-forward optimisation
# --------------------------------------------------------------------------- #
@dataclass
class WalkForwardResult:
    windows: pd.DataFrame
    oos_equity: pd.Series
    oos_trades: pd.DataFrame
    oos_in_market: pd.Series
    oos_metrics: Optional[Dict[str, float]]
    summary: Dict[str, float] = field(default_factory=dict)


def walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    grid: Dict[str, Sequence[Any]],
    cfg: EngineConfig,
    train_days: int = 365,
    test_days: int = 90,
    step_days: Optional[int] = None,
    anchored: bool = False,
    objective: str = "sharpe",
    min_trades: int = 5,
    base_params: Optional[Dict[str, Any]] = None,
    progress: ProgressFn = None,
) -> WalkForwardResult:
    """
    Rolling (or anchored) Walk-Forward Optimisation.

    For each window: sweep ``grid`` on the train span, lock the best parameters
    (must reach ``min_trades``, otherwise the strategy stays flat), then trade the
    following test span with those parameters only. Test spans never overlap when
    ``step_days == test_days`` (the default), so the stitched curve is purely OOS.
    """
    step_days = step_days or test_days
    idx = df.index
    t0, t_last = idx[0], idx[-1]
    train_td, test_td, step_td = (pd.Timedelta(days=d) for d in (train_days, test_days, step_days))
    bank = _SignalBank(df, strategy_name, base_params)
    combos = expand_grid(grid)
    keys = list(grid)

    spans = []
    cursor = t0
    while True:
        tr_start = t0 if anchored else cursor
        tr_end = cursor + train_td
        te_end = tr_end + test_td
        if tr_end >= t_last:
            break
        if te_end > t_last and (t_last - tr_end) < 0.5 * test_td:
            break
        spans.append((tr_start, tr_end, tr_end, min(te_end, t_last + pd.Timedelta(days=1))))
        cursor += step_td
    if not spans:
        raise ValueError("Data span too short for the requested train/test windows.")

    rows, eq_parts, im_parts, trade_parts = [], [], [], []
    capital = cfg.initial_capital
    for w, (tr_s, tr_e, te_s, te_e) in enumerate(spans, start=1):
        if progress:
            progress((w - 1) / len(spans), f"Window {w}/{len(spans)}: optimising")
        a0, a1 = int(idx.searchsorted(tr_s, "left")), int(idx.searchsorted(tr_e, "left"))
        b0, b1 = int(idx.searchsorted(te_s, "left")), int(idx.searchsorted(te_e, "left"))
        if b1 - b0 < 2:
            continue
        table = _sweep_slice(bank, combos, a0, a1, cfg, objective, min_trades)
        best = _best_params(table, grid)
        row = {"window": w, "train_start": idx[a0], "train_end": idx[a1 - 1], "test_start": idx[b0],
               "test_end": idx[b1 - 1], "params": str(best) if best else "none (flat)"}
        row.update({f"p_{k}": (best[k] if best else np.nan) for k in keys})
        if best is None:
            row.update({"is_score": np.nan, "oos_score": np.nan, "is_trades": 0, "oos_trades": 0, "oos_return": 0.0})
            window_idx = idx[b0:b1]
            eq_parts.append(pd.Series(capital, index=window_idx))
            im_parts.append(pd.Series(False, index=window_idx))
        else:
            is_score, is_m, _ = _evaluate_slice(bank, best, a0, a1, cfg, objective, min_trades)
            run_cfg = cfg.with_capital(capital)
            res = BacktestEngine(run_cfg).run(df.iloc[b0:b1], bank.get(best).iloc[b0:b1])
            m = compute_metrics(res.equity, res.trades_df, res.in_market, capital)
            row.update({
                "is_score": is_score, "oos_score": objective_value(m, objective),
                "is_trades": is_m["n_trades"], "oos_trades": m["n_trades"],
                "oos_return": res.final_equity / capital - 1.0,
            })
            eq_parts.append(res.equity)
            im_parts.append(res.in_market)
            if len(res.trades_df):
                trade_parts.append(res.trades_df)
            capital = res.final_equity
        rows.append(row)
    if progress:
        progress(1.0, "Walk-forward complete")

    windows = pd.DataFrame(rows)
    oos_equity = pd.concat(eq_parts) if eq_parts else pd.Series(dtype=float)
    oos_in_market = pd.concat(im_parts) if im_parts else pd.Series(dtype=bool)
    oos_trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    oos_metrics = None
    if len(oos_equity) >= 2:
        oos_metrics = compute_metrics(oos_equity, oos_trades if len(oos_trades) else None,
                                      oos_in_market, cfg.initial_capital)

    summary: Dict[str, float] = {"n_windows": float(len(windows))}
    if len(windows):
        both = windows[np.isfinite(windows["is_score"].astype(float)) & np.isfinite(windows["oos_score"].astype(float))]
        mean_is = float(both["is_score"].mean()) if len(both) else float("nan")
        mean_oos = float(both["oos_score"].mean()) if len(both) else float("nan")
        summary.update({
            "mean_is_score": mean_is, "mean_oos_score": mean_oos,
            "efficiency_ratio": mean_oos / mean_is if (len(both) and mean_is > 0) else float("nan"),
            "pct_profitable_windows": float((windows["oos_return"] > 0).mean()),
            "param_stability": float(windows["params"].value_counts(normalize=True).iloc[0]),
        })
    return WalkForwardResult(windows, oos_equity, oos_trades, oos_in_market, oos_metrics, summary)


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
@dataclass
class MonteCarloResult:
    n_sims: int
    n_trades: int
    method: str
    initial_capital: float
    max_dd: np.ndarray                 # positive fractions, one per simulation
    final_equity: np.ndarray
    p95_max_dd: float                  # 95th percentile worst-case max drawdown
    median_max_dd: float
    mean_max_dd: float
    worst_max_dd: float
    observed_max_dd: float             # drawdown of the actual trade order
    prob_ruin: float                   # P(equity ever <= ruin_level * initial)
    prob_dd_breach: float              # P(max drawdown >= dd_limit)
    prob_loss: float                   # P(final equity < initial)
    ruin_level: float
    dd_limit: float
    percentiles: pd.DataFrame          # equity percentiles per trade number
    sample_paths: np.ndarray           # a few raw paths for plotting


def monte_carlo(trades_df: pd.DataFrame, initial_capital: float, n_sims: int = 1000,
                method: str = "shuffle", ruin_level: float = 0.5, dd_limit: float = 0.25,
                seed: Optional[int] = 42) -> MonteCarloResult:
    """
    Re-order (``shuffle``) or resample with replacement (``bootstrap``) the *fractional*
    trade returns (net PnL / equity at entry) and compound them from ``initial_capital``.
    Using fractional returns preserves the fixed-fractional sizing logic.

    Note: shuffling keeps the same set of trades, so every path ends at the same equity;
    only the *path* (hence drawdown and ruin risk) changes. Use ``bootstrap`` to also
    vary terminal wealth.
    """
    if method not in ("shuffle", "bootstrap"):
        raise ValueError("method must be 'shuffle' or 'bootstrap'.")
    if trades_df is None or len(trades_df) < 5:
        raise ValueError("Monte Carlo needs at least 5 trades.")
    rets = trades_df["return_pct"].to_numpy(dtype=float)
    n = len(rets)
    rng = np.random.default_rng(seed)
    if method == "shuffle":
        order = rng.random((n_sims, n)).argsort(axis=1)
        sampled = rets[order]
    else:
        sampled = rets[rng.integers(0, n, size=(n_sims, n))]
    growth = np.cumprod(np.clip(1.0 + sampled, 0.0, None), axis=1)
    paths = initial_capital * np.concatenate([np.ones((n_sims, 1)), growth], axis=1)
    peak = np.maximum.accumulate(paths, axis=1)
    max_dd = (1.0 - paths / peak).max(axis=1)

    obs = initial_capital * np.concatenate([[1.0], np.cumprod(np.clip(1.0 + rets, 0.0, None))])
    observed_dd = float((1.0 - obs / np.maximum.accumulate(obs)).max())
    pct = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    percentiles = pd.DataFrame(pct.T, columns=["p05", "p25", "p50", "p75", "p95"])
    percentiles.index.name = "trade"
    return MonteCarloResult(
        n_sims=n_sims, n_trades=n, method=method, initial_capital=float(initial_capital),
        max_dd=max_dd, final_equity=paths[:, -1],
        p95_max_dd=float(np.percentile(max_dd, 95)), median_max_dd=float(np.median(max_dd)),
        mean_max_dd=float(max_dd.mean()), worst_max_dd=float(max_dd.max()), observed_max_dd=observed_dd,
        prob_ruin=float((paths.min(axis=1) <= ruin_level * initial_capital).mean()),
        prob_dd_breach=float((max_dd >= dd_limit).mean()),
        prob_loss=float((paths[:, -1] < initial_capital).mean()),
        ruin_level=ruin_level, dd_limit=dd_limit, percentiles=percentiles,
        sample_paths=paths[: min(40, n_sims)],
    )


# --------------------------------------------------------------------------- #
# Look-ahead audit
# --------------------------------------------------------------------------- #
def causality_audit(df: pd.DataFrame, strategy: BaseStrategy, cfg: Optional[EngineConfig] = None,
                    n_checks: int = 5, seed: int = 0) -> Dict[str, Any]:
    """
    Empirical look-ahead test. For random cut points ``k`` we re-run the strategy and
    the engine on ``df[:k]`` only and require that (1) every signal up to ``k`` equals the
    full-history signal and (2) every trade and equity mark that was already closed by
    ``k`` is bit-for-bit what the full run produced. Any dependence on future bars breaks
    one of these equalities.
    """
    cfg = cfg or EngineConfig()
    engine = BacktestEngine(cfg)
    full_sig = strategy.generate_signals(df)
    full = engine.run(df, full_sig)
    n = len(df)
    lo = min(max(strategy.warmup_bars + 50, n // 4), n - 10)
    rng = np.random.default_rng(seed)
    cuts = sorted(set(int(c) for c in rng.integers(lo, n - 2, size=n_checks)))
    keys = ["entry_time", "exit_time", "entry_price", "exit_price", "size", "exit_reason"]
    failures: List[str] = []
    for k in cuts:
        sub = df.iloc[:k]
        sub_sig = strategy.generate_signals(sub)
        a, b = full_sig.iloc[:k], sub_sig
        for col in a.columns:
            if a[col].dtype == bool or col == "signal":
                same = bool((a[col].to_numpy() == b[col].to_numpy()).all())
            else:
                same = bool(np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True, atol=1e-9))
            if not same:
                failures.append(f"cut@{k}: signal column '{col}' changes when future bars are removed")
        sub_run = engine.run(sub, sub_sig)
        ft = full.trades_df[full.trades_df["exit_time"] <= sub.index[-1]].reset_index(drop=True)
        st = sub_run.trades_df[sub_run.trades_df["exit_reason"] != "EndOfData"].reset_index(drop=True)
        if len(ft) != len(st) or (len(ft) and not (
            (ft[keys[:2] + keys[5:]].to_numpy() == st[keys[:2] + keys[5:]].to_numpy()).all()
            and np.allclose(ft[keys[2:5]].to_numpy(float), st[keys[2:5]].to_numpy(float), atol=1e-9)
        )):
            failures.append(f"cut@{k}: closed trades differ ({len(ft)} full vs {len(st)} truncated)")
        if not np.allclose(full.equity.iloc[: k - 1].to_numpy(), sub_run.equity.iloc[: k - 1].to_numpy(), atol=1e-6):
            failures.append(f"cut@{k}: equity path differs")
    return {"passed": not failures, "n_checks": len(cuts), "cut_points": cuts, "failures": failures}


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def run_strategy(df: pd.DataFrame, strategy_name: str, params: Dict[str, Any], cfg: EngineConfig,
                 start=None, end=None) -> Tuple[BacktestResult, Dict[str, float]]:
    """Run one strategy with fixed parameters and return ``(result, metrics)``."""
    strat = get_strategy(strategy_name, **params)
    sig = strat.generate_signals(df)
    res = BacktestEngine(cfg).run(df, sig, start=start, end=end)
    return res, compute_metrics(res.equity, res.trades_df, res.in_market, cfg.initial_capital)
