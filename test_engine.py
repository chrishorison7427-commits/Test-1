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


# --------------------------------------------------------------------------- #
# VWAP opening-range momentum and RSI(2) pullback
# --------------------------------------------------------------------------- #
def _ny_day(date, base=100.0, volume=1000.0):
    """One NY session of 5m bars (08:00-15:55) hovering around ``base`` with an opening range of 99.5-100.5."""
    idx = pd.date_range(f"{date} 08:00", f"{date} 15:55", freq="5min", tz="America/New_York")
    c = base + 0.02 * np.sin(np.arange(len(idx)))
    df = pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c, "volume": volume}, index=idx)
    _setbar(df, date, "09:35", h=100.5)
    _setbar(df, date, "09:45", l=99.5)
    return df


def _setbar(df, date, hhmm, o=None, h=None, l=None, c=None, v=None):
    ts = pd.Timestamp(f"{date} {hhmm}", tz="America/New_York")
    for col, val in (("open", o), ("high", h), ("low", l), ("close", c), ("volume", v)):
        if val is not None:
            df.loc[ts, col] = val
    return ts


def _vwap_days():
    d1, d2, d3, d4, d5 = (_ny_day(f"2024-03-0{i}") for i in range(4, 9))
    t1 = _setbar(d1, "2024-03-04", "10:00", o=100.0, h=100.45, l=99.95, c=100.4)          # inside range: no signal
    t1 = _setbar(d1, "2024-03-04", "10:05", o=100.4, h=100.8, l=100.35, c=100.7)           # breakout above ORH and VWAP
    _setbar(d1, "2024-03-04", "10:30", o=100.7, h=101.0, l=100.6, c=100.9)                 # 2nd breakout: ignored (1/day)
    _setbar(d2, "2024-03-05", "14:25", o=100.0, h=100.9, l=99.95, c=100.8)                 # entry bar would open 14:30
    t3 = _setbar(d3, "2024-03-06", "14:20", o=100.0, h=100.9, l=99.95, c=100.8)            # last bar allowed
    t4 = _setbar(d4, "2024-03-07", "10:15", o=100.0, h=100.05, l=99.2, c=99.3)             # breakdown below ORL and VWAP
    _setbar(d5, "2024-03-08", "10:00", o=100.4, h=103.0, l=100.3, c=100.4, v=100_000.0)    # spike drags VWAP to ~101
    _setbar(d5, "2024-03-08", "10:05", o=100.4, h=100.9, l=100.35, c=100.8)                # > ORH but < VWAP: no signal
    t5 = _setbar(d5, "2024-03-08", "10:10", o=100.8, h=101.7, l=100.8, c=101.5)            # above both: signal
    return pd.concat([d1, d2, d3, d4, d5]), (t1, t3, t4, t5)


def test_anchored_vwap_matches_manual_calculation():
    from indicators import anchored_vwap
    df = pd.concat([_ny_day("2024-03-04"), _ny_day("2024-03-05", base=105.0)])
    df["volume"] = np.arange(1, len(df) + 1, dtype=float)
    day = np.asarray(df.index.year * 10000 + df.index.month * 100 + df.index.day)
    mod = np.asarray(df.index.hour * 60 + df.index.minute)
    got = anchored_vwap(df, day, mod >= 570)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    for d in np.unique(day):
        m = (day == d) & (mod >= 570)
        want = (typical[m] * df["volume"][m]).cumsum() / df["volume"][m].cumsum()
        assert np.allclose(got[m].to_numpy(), want.to_numpy())
        assert got[(day == d) & (mod < 570)].isna().all()


def test_vwap_orb_signal_rules_on_hand_built_days():
    df, (t1, t3, t4, t5) = _vwap_days()
    sig = get_strategy("vwap_orb").generate_signals(df)
    hit = sig[sig["signal"] != 0]
    assert list(hit.index) == [t1, t3, t4, t5], "wrong set of signal bars"
    assert hit["signal"].tolist() == [1, 1, -1, 1]
    assert t1 == pd.Timestamp("2024-03-04 10:05", tz="America/New_York")          # not the 10:00 bar, one per day
    assert t5 == pd.Timestamp("2024-03-08 10:10", tz="America/New_York")          # VWAP filter delayed it a bar
    assert np.allclose(hit["sl_dist"], 1.5 * hit["atr"]) and np.allclose(hit["tp_dist"], 1.5 * hit["sl_dist"])


def test_vwap_orb_engine_fills_next_open_with_1p5r_target_and_flat_1555():
    df, (t1, *_rest) = _vwap_days()
    d1_spike = df[df.index.date == t1.date()]
    cfg0 = EngineConfig(initial_capital=100_000, slippage_bps=0.0, max_leverage=50.0)
    sig_spike = get_strategy("vwap_orb").generate_signals(df).loc[d1_spike.index]
    tp_trade = BacktestEngine(cfg0).run(d1_spike, sig_spike).trades_df.iloc[0]
    assert tp_trade["exit_reason"] == "TP" and tp_trade["exit_time"] == t1 + pd.Timedelta(minutes=25)  # 10:30 spike
    assert abs(tp_trade["exit_price"] - tp_trade["tp_price"]) < 1e-12, "TP fills exactly at the target"
    d1 = d1_spike.copy()      # calm the 10:30 spike so the position runs to the flatten time
    ohlc = ["open", "high", "low", "close"]
    d1.loc[t1 + pd.Timedelta(minutes=25), ohlc] = d1.loc[t1 + pd.Timedelta(minutes=20), ohlc].to_numpy()
    sig = get_strategy("vwap_orb").generate_signals(d1)
    tr = BacktestEngine(EngineConfig(initial_capital=100_000, slippage_bps=0.0, max_leverage=50.0)).run(d1, sig).trades_df
    t = tr.iloc[0]
    assert t["entry_time"] == t1 + pd.Timedelta(minutes=5) and t["entry_price"] == d1.loc[t["entry_time"], "open"]
    assert abs((t["entry_price"] - t["sl_price"]) - 1.5 * sig.loc[t1, "atr"]) < 1e-9
    assert abs((t["tp_price"] - t["entry_price"]) - 1.5 * (t["entry_price"] - t["sl_price"])) < 1e-9
    assert t["exit_reason"] == "Signal" and t["exit_time"] == pd.Timestamp("2024-03-04 15:55", tz="America/New_York")


def test_vwap_orb_15m_bars_exit_before_the_close_never_after():
    raw = dl.generate_synthetic_ohlcv("2024-01-02", "2024-04-30", "15m", seed=6)
    df, _ = dl.validate_ohlcv(raw, "15m")
    tr = BacktestEngine(EngineConfig(slippage_bps=0.0, max_leverage=50.0)).run(
        df, get_strategy("vwap_orb").generate_signals(df)).trades_df
    x = tr["exit_time"].dt.tz_convert("America/New_York")
    assert (x.dt.hour * 60 + x.dt.minute).max() <= 15 * 60 + 55
    e = tr["entry_time"].dt.tz_convert("America/New_York")
    assert (e.dt.hour * 60 + e.dt.minute).min() >= 10 * 60 and (e.dt.hour * 60 + e.dt.minute).max() < 14 * 60 + 30
    assert e.dt.date.value_counts().max() == 1


def test_vwap_orb_rejects_daily_and_too_coarse_bars():
    raw = dl.generate_synthetic_ohlcv("2020-01-02", "2020-12-31", "1d")
    daily, _ = dl.validate_ohlcv(raw, "1d")
    hourly, _ = dl.validate_ohlcv(dl.generate_synthetic_ohlcv("2024-01-02", "2024-02-29", "1h"), "1h")
    for frame in (daily, hourly):
        try:
            get_strategy("vwap_orb").generate_signals(frame)
        except ValueError:
            continue
        raise AssertionError("daily / 1h bars must be rejected")


def test_vwap_orb_has_no_lookahead():
    for interval in ("5m", "15m"):
        raw = dl.generate_synthetic_ohlcv("2024-01-02", "2024-03-28", interval, seed=3)
        df, _ = dl.validate_ohlcv(raw, interval)
        audit = val.causality_audit(df, get_strategy("vwap_orb"), EngineConfig(slippage_bps=1.0, max_leverage=50.0),
                                    n_checks=4)
        assert audit["passed"], audit["failures"]


def _manual_wilder_rsi(close: pd.Series, n: int) -> np.ndarray:
    c, out = close.to_numpy(float), np.full(len(close), np.nan)
    ag = al = 0.0
    for i in range(1, len(c)):
        g, l = max(c[i] - c[i - 1], 0.0), max(c[i - 1] - c[i], 0.0)
        ag, al = (g, l) if i == 1 else ((1 - 1 / n) * ag + g / n, (1 - 1 / n) * al + l / n)
        if i >= n:
            out[i] = (100.0 if ag > 0 else 50.0) if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def test_rsi2_pullback_signals_exits_and_stop_match_independent_calculation():
    from indicators import atr as atr_fn, rsi_wilder
    raw = dl.generate_synthetic_ohlcv("2008-01-02", "2024-12-31", "1d", seed=12, annual_vol=0.30)
    df, _ = dl.validate_ohlcv(raw, "1d")
    close = df["close"]
    rsi_manual = _manual_wilder_rsi(close, 2)
    assert np.allclose(rsi_wilder(close, 2).to_numpy(), rsi_manual, equal_nan=True)
    sma200, sma5 = close.rolling(200).mean(), close.rolling(5).mean()
    sig = get_strategy("rsi2_pullback").generate_signals(df)
    want = ((close > sma200) & (rsi_manual < 10) & atr_fn(df, 14).notna()).astype(int)
    assert (sig["signal"].to_numpy() == want.to_numpy()).all() and want.sum() > 20
    assert (sig["exit_long"].to_numpy() == (close > sma5).to_numpy()).all()
    assert sig["signal"].iloc[:199].eq(0).all(), "no trades before the 200 SMA exists"
    res = BacktestEngine(EngineConfig(slippage_bps=0.0, max_leverage=10.0)).run(df, sig)
    tr = res.trades_df
    assert len(tr) > 20 and set(tr["direction"]) == {"Long"}
    atr_at_signal = sig["atr"].shift(1).reindex(tr["entry_time"]).to_numpy()
    assert np.allclose(tr["entry_price"] - tr["sl_price"], 3.0 * atr_at_signal), "stop = fill - 3 ATR"
    assert tr["tp_price"].isna().all(), "no profit target"
    entry_open = df["open"].reindex(tr["entry_time"]).to_numpy()
    assert np.allclose(tr["entry_price"], entry_open), "entries fill at the next bar's open"
    sig_exits = tr[tr["exit_reason"] == "Signal"]
    prev_close = close.shift(1).reindex(sig_exits["exit_time"]).to_numpy()
    prev_sma5 = sma5.shift(1).reindex(sig_exits["exit_time"]).to_numpy()
    assert (prev_close > prev_sma5).all(), "exit only after a close above the 5 SMA, filled next open"


def test_rsi2_pullback_has_no_lookahead():
    for interval, start in (("1d", "2012-01-02"), ("1h", "2023-01-02")):
        raw = dl.generate_synthetic_ohlcv(start, "2024-12-31", interval, seed=5)
        df, _ = dl.validate_ohlcv(raw, interval)
        audit = val.causality_audit(df, get_strategy("rsi2_pullback"), EngineConfig(slippage_bps=1.0, max_leverage=10.0),
                                    n_checks=4)
        assert audit["passed"], audit["failures"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n{len(tests)} tests passed.")
