"""
test_engine.py
==============
Hand-built scenarios that pin down the execution engine's conventions.
Run with:  ``python test_engine.py``  (or ``pytest -q``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_loader as dl
import validation as val
from analytics import compute_metrics
from execution_engine import BacktestEngine, EngineConfig
from strategies import get_strategy

CFG = EngineConfig(initial_capital=100_000, risk_per_trade=0.01, slippage_bps=0.0, max_leverage=10.0)


def make(bars, signal_bar=1, direction=1, sl=2.0, tp=2.0, trail=np.nan, atr=1.0):
    """bars: list of (open, high, low, close). One entry signal at ``signal_bar``."""
    idx = pd.date_range("2024-01-01", periods=len(bars), freq="D")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1_000.0
    sig = pd.DataFrame(index=idx)
    sig["signal"] = 0
    sig.iloc[signal_bar, 0] = direction
    sig["atr"] = atr
    sig["sl_dist"] = np.where(sig["signal"] != 0, sl, np.nan)
    sig["tp_dist"] = np.where(sig["signal"] != 0, tp, np.nan) if tp == tp else np.nan
    sig["trail_mult"] = np.where(sig["signal"] != 0, trail, np.nan)
    sig["exit_long"] = False
    sig["exit_short"] = False
    return df, sig


def run(df, sig, cfg=CFG):
    return BacktestEngine(cfg).run(df, sig)


def test_entry_fills_at_next_open_not_signal_close():
    df, sig = make([(100, 101, 99, 100), (100, 101, 99, 100.5), (101, 101.5, 100.5, 101),
                    (101, 101.5, 100.5, 101), (101, 101.5, 100.5, 101)])
    t = run(df, sig).trades_df.iloc[0]
    assert t["entry_time"] == df.index[2] and t["entry_price"] == 101.0


def test_sl_wins_when_sl_and_tp_both_touched():
    df, sig = make([(100, 100, 100, 100), (100, 100, 100, 100), (100, 103, 97, 100), (100, 100, 100, 100)])
    t = run(df, sig).trades_df.iloc[0]
    assert t["exit_reason"] == "SL" and t["exit_price"] == 98.0
    assert t["size"] == 500 and abs(t["net_pnl"] + 1000.0) < 1e-9


def test_take_profit_fills_at_limit():
    df, sig = make([(100, 100, 100, 100), (100, 100, 100, 100), (100, 103, 99.5, 102.5), (100, 100, 100, 100)])
    t = run(df, sig).trades_df.iloc[0]
    assert t["exit_reason"] == "TP" and t["exit_price"] == 102.0 and abs(t["net_pnl"] - 1000.0) < 1e-9


def test_stop_gap_fills_at_open_not_at_stop():
    df, sig = make([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100.5, 99.5, 100),
                    (95, 96, 94, 95), (95, 95, 95, 95)])
    t = run(df, sig).trades_df.iloc[0]
    assert t["exit_reason"] == "SL" and t["exit_price"] == 95.0


def test_tp_gap_never_improves_fill_by_default():
    df, sig = make([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100.5, 99.5, 100),
                    (105, 106, 104, 105), (105, 105, 105, 105)])
    t = run(df, sig).trades_df.iloc[0]
    assert t["exit_reason"] == "TP" and t["exit_price"] == 102.0


def test_short_mirrors_long():
    df, sig = make([(100, 100, 100, 100), (100, 100, 100, 100), (100, 103, 97, 100), (100, 100, 100, 100)],
                   direction=-1)
    t = run(df, sig).trades_df.iloc[0]
    assert t["direction"] == "Short" and t["exit_reason"] == "SL" and t["exit_price"] == 102.0


def test_trailing_stop_is_not_applied_intrabar_on_entry_bar():
    df, sig = make([(100, 100, 100, 100), (100, 100, 100, 100), (100, 104, 99, 103),
                    (103.5, 104, 102.5, 103), (103, 103, 103, 103)], tp=np.nan, trail=1.0)
    t = run(df, sig).trades_df.iloc[0]
    assert t["exit_time"] == df.index[3], "trail must only act from the bar after it was raised"
    assert t["exit_reason"] == "SL" and t["exit_price"] == 103.0


def test_slippage_and_commission_are_charged_adversely():
    cfg = EngineConfig(initial_capital=100_000, risk_per_trade=0.01, slippage_bps=10.0, commission_per_unit=0.01,
                       max_leverage=10.0)
    df, sig = make([(100, 100, 100, 100), (100, 100, 100, 100), (100, 100.5, 99.5, 100),
                    (100, 100, 100, 100), (100, 100, 100, 100)])
    t = run(df, sig, cfg).trades_df.iloc[0]
    assert abs(t["entry_price"] - 100.1) < 1e-9          # long pays up 10 bps
    assert t["fees"] > 0 and t["net_pnl"] < t["gross_pnl"]


def test_equity_reconciles_with_ledger_on_real_strategies():
    raw = dl.generate_synthetic_ohlcv("2016-01-04", "2024-12-31", "1d", seed=11)
    df, _ = dl.validate_ohlcv(raw, "1d")
    cfg = EngineConfig(initial_capital=50_000, slippage_bps=2.0, commission_per_unit=0.005, stop_slippage_mult=2.0)
    res = BacktestEngine(cfg).run(df, get_strategy("mean_reversion", z_entry=1.5).generate_signals(df))
    assert len(res.trades_df) > 5
    assert abs(50_000 + res.trades_df["net_pnl"].sum() - res.final_equity) < 1e-6
    assert (res.trades_df["gross_pnl"] - res.trades_df["fees"] - res.trades_df["net_pnl"]).abs().max() < 1e-9
    assert set(res.trades_df["exit_reason"]) <= {"SL", "TP", "Signal", "EndOfData"}


def test_no_lookahead_in_built_in_strategies():
    for name, interval, start, end in (("mean_reversion", "1d", "2014-01-02", "2024-12-31"),
                                       ("orb", "15m", "2024-01-02", "2024-06-28")):
        raw = dl.generate_synthetic_ohlcv(start, end, interval, seed=2)
        df, _ = dl.validate_ohlcv(raw, interval)
        audit = val.causality_audit(df, get_strategy(name), CFG, n_checks=4)
        assert audit["passed"], audit["failures"]


def test_monte_carlo_shuffle_preserves_terminal_equity():
    raw = dl.generate_synthetic_ohlcv("2014-01-02", "2024-12-31", "1d", seed=3)
    df, _ = dl.validate_ohlcv(raw, "1d")
    res = BacktestEngine(CFG).run(df, get_strategy("mean_reversion", z_entry=1.5).generate_signals(df))
    mc = val.monte_carlo(res.trades_df, CFG.initial_capital, n_sims=1000, method="shuffle")
    assert mc.n_sims == 1000 and np.ptp(mc.final_equity) < 1e-6 * CFG.initial_capital
    assert 0 <= mc.p95_max_dd <= 1 and 0 <= mc.prob_ruin <= 1


def test_absolute_stop_and_r_multiple_target_anchor_on_the_fill():
    df, sig = make([(100, 100, 100, 100), (100, 100, 100, 100), (101, 101.5, 100.5, 101),
                    (101, 104, 100.5, 103), (103, 103, 103, 103)])
    sig["sl_price"] = np.where(sig["signal"] != 0, 99.0, np.nan)   # absolute stop, 2.0 below the 101 fill
    sig["tp_rr"] = np.where(sig["signal"] != 0, 1.5, np.nan)
    sig["sl_dist"] = np.nan
    sig["tp_dist"] = np.nan
    t = run(df, sig).trades_df.iloc[0]
    assert t["entry_price"] == 101.0 and t["sl_price"] == 99.0 and t["tp_price"] == 104.0   # 101 + 1.5 * 2
    assert t["exit_reason"] == "TP" and abs(t["r_multiple"] - 1.5) < 1e-9


def test_absolute_stop_already_gapped_through_skips_the_entry():
    df, sig = make([(100, 100, 100, 100), (100, 100, 100, 100), (98, 99, 97, 98), (98, 98, 98, 98)])
    sig["sl_price"] = np.where(sig["signal"] != 0, 99.0, np.nan)
    res = run(df, sig)
    assert len(res.trades_df) == 0 and res.skipped_entries == 1


def _ict_frame(interval="5m", start="2024-01-02", end="2024-04-30", seed=4):
    raw = dl.generate_synthetic_ohlcv(start, end, interval, seed=seed, session="24h")
    df, _ = dl.validate_ohlcv(raw, interval)
    return df


def test_ict_rules_killzone_one_trade_per_day_and_flat_by_1555():
    df = _ict_frame()
    cfg = EngineConfig(initial_capital=100_000, slippage_bps=1.0, max_leverage=10.0)
    for conf in (True, False):
        res = BacktestEngine(cfg).run(df, get_strategy("ict_sweep", use_confirmation=conf).generate_signals(df))
        tr = res.trades_df
        assert len(tr) > 10
        e = tr["entry_time"].dt.tz_convert("America/New_York")
        x = tr["exit_time"].dt.tz_convert("America/New_York")
        e_min, x_min = e.dt.hour * 60 + e.dt.minute, x.dt.hour * 60 + x.dt.minute
        assert e_min.min() >= 8 * 60 + 30 and e_min.max() < 11 * 60, "entries must open inside 08:30-11:00 NY"
        assert x_min.max() <= 15 * 60 + 55, "no position may survive past 15:55 NY"
        assert (e.dt.date == x.dt.date).all(), "no overnight holds"
        assert e.dt.date.value_counts().max() == 1, "max one trade per day"


def test_ict_stop_sits_beyond_the_sweep_and_target_is_exact_r():
    df = _ict_frame()
    cfg = EngineConfig(initial_capital=100_000, slippage_bps=0.0, max_leverage=10.0)
    sig = get_strategy("ict_sweep", use_confirmation=False, rr=2.0, sl_buffer_atr=0.25).generate_signals(df)
    tr = BacktestEngine(cfg).run(df, sig).trades_df
    longs, shorts = tr[tr.direction == "Long"], tr[tr.direction == "Short"]
    assert (longs["sl_price"] < longs["entry_price"]).all() and (shorts["sl_price"] > shorts["entry_price"]).all()
    risk = (tr["entry_price"] - tr["sl_price"]).abs()
    reward = (tr["tp_price"] - tr["entry_price"]).abs()
    assert np.allclose(reward, 2.0 * risk, rtol=1e-9)


def test_ict_liquidity_target_uses_opposing_range_extreme():
    df = _ict_frame()
    cfg = EngineConfig(initial_capital=100_000, slippage_bps=0.0, max_leverage=10.0)
    sig = get_strategy("ict_sweep", use_confirmation=False, tp_use_liquidity=True, min_rr=1.0).generate_signals(df)
    tr = BacktestEngine(cfg).run(df, sig).trades_df
    assert len(tr) > 3
    longs = tr[tr.direction == "Long"]
    assert (longs["tp_price"] > longs["entry_price"]).all()


def test_ict_naive_utc_and_new_york_timestamps_give_identical_signals():
    df = _ict_frame(end="2024-03-15")
    naive = df.copy()
    naive.index = df.index.tz_convert("UTC").tz_localize(None)
    s = get_strategy("ict_sweep")
    a, b = s.generate_signals(df), s.generate_signals(naive)
    assert (a["signal"].to_numpy() == b["signal"].to_numpy()).all() and (a["signal"] != 0).any()
    assert np.allclose(a["sl_price"].to_numpy(float), b["sl_price"].to_numpy(float), equal_nan=True)
    res = BacktestEngine(CFG).run(naive, b)           # the engine must also accept the naive index
    assert len(res.trades_df) == int((b["signal"] != 0).sum()) - res.skipped_entries


def test_ict_has_no_lookahead():
    df = _ict_frame(end="2024-03-28")
    for conf in (True, False):
        audit = val.causality_audit(df, get_strategy("ict_sweep", use_confirmation=conf),
                                    EngineConfig(slippage_bps=1.0, max_leverage=10.0), n_checks=4)
        assert audit["passed"], audit["failures"]


def test_ict_rejects_daily_bars():
    raw = dl.generate_synthetic_ohlcv("2020-01-02", "2020-12-31", "1d")
    df, _ = dl.validate_ohlcv(raw, "1d")
    try:
        get_strategy("ict_sweep").generate_signals(df)
    except ValueError as exc:
        assert "intraday" in str(exc)
    else:
        raise AssertionError("daily data must be rejected")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n{len(tests)} tests passed.")
