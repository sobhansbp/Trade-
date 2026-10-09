"""Stops, targets, trade simulation and position sizing.

The same ``simulate_trade`` routine labels ML training data and runs the
backtest, so what the model learns is exactly how trades are managed:
  entry at next bar open (+ half spread + slippage), structural/ATR stop,
  partial profit at +1R with stop to breakeven, final target at ``rr_target``,
  time stop. If stop and target are touched in the same bar the stop wins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Instrument, StrategyConfig


def stop_distances(f: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Per-bar stop distance (price units) for a long and a short entered at the
    next open. Prefers the nearest *structural* invalidation (swing, order block,
    sweep wick) that is at least ``min_stop_atr`` away; falls back to ATR."""
    c, a = f["close"], f["atr"]
    buf = 0.2 * a
    lo_cands = [f["last_sl"] - buf, f["low"].rolling(3).min() - buf]
    hi_cands = [f["last_sh"] + buf, f["high"].rolling(3).max() + buf]
    if "pdl" in f:
        lo_cands.append(f["pdl"] - buf)
        hi_cands.append(f["pdh"] + buf)
    atr_stop = cfg.atr_stop_mult * a
    mn, mx = cfg.min_stop_atr * a, cfg.max_stop_atr * a

    def pick(dists):
        D = np.vstack([d.values for d in dists]).T
        D = np.where((D >= mn.values[:, None]) & (D <= mx.values[:, None]), D, np.nan)
        best = np.nanmin(np.where(np.isnan(D), np.inf, D), axis=1)
        best = np.where(np.isfinite(best), best, atr_stop.values)
        return np.maximum(best, mn.values)

    long_d = pick([c - x for x in lo_cands])
    short_d = pick([x - c for x in hi_cands])
    return pd.DataFrame({"stop_long": long_d, "stop_short": short_d}, index=f.index)


@dataclass
class TradeResult:
    side: int
    entry_i: int
    exit_i: int
    entry: float
    exit: float
    stop: float
    target: float
    r_multiple: float
    reason: str
    partial: bool


def simulate_trade(o, h, l, c, t: int, side: int, stop_dist: float, inst: Instrument,
                   cfg: StrategyConfig, horizon: int | None = None) -> TradeResult | None:
    """Simulate a trade decided at the close of bar ``t``."""
    n = len(o)
    if t + 1 >= n or not np.isfinite(stop_dist) or stop_dist <= 0:
        return None
    horizon = horizon or cfg.time_stop_bars
    cost = inst.spread / 2 + inst.slippage
    entry = o[t + 1] + side * cost
    stop = entry - side * stop_dist
    target = entry + side * cfg.rr_target * stop_dist
    p_lvl = entry + side * cfg.partial_at_r * stop_dist
    partial = False
    realized = 0.0
    remaining = 1.0
    end = min(n - 1, t + horizon)
    for i in range(t + 1, end + 1):
        hi, lo = h[i], l[i]
        # worst case first: stop
        if (side > 0 and lo <= stop) or (side < 0 and hi >= stop):
            px = stop
            if i > t + 1 and ((side > 0 and o[i] < stop) or (side < 0 and o[i] > stop)):
                px = o[i]  # gapped through
            px -= side * inst.slippage
            realized += remaining * side * (px - entry) / stop_dist
            return TradeResult(side, t + 1, i, entry, px, stop, target, realized,
                               "breakeven" if partial else "stop", partial)
        if (side > 0 and hi >= target) or (side < 0 and lo <= target):
            realized += remaining * side * (target - entry) / stop_dist
            return TradeResult(side, t + 1, i, entry, target, stop, target, realized, "target", partial)
        if not partial and cfg.partial_fraction > 0 and ((side > 0 and hi >= p_lvl) or (side < 0 and lo <= p_lvl)):
            realized += cfg.partial_fraction * side * (p_lvl - entry) / stop_dist
            remaining -= cfg.partial_fraction
            partial = True
            stop = entry + side * cost  # breakeven incl. costs
    px = c[end] - side * (inst.spread / 2)
    realized += remaining * side * (px - entry) / stop_dist
    return TradeResult(side, t + 1, end, entry, px, stop, target, realized, "time", partial)


def label_outcomes(f: pd.DataFrame, stops: pd.DataFrame, inst: Instrument, cfg: StrategyConfig,
                   step: int = 1) -> pd.DataFrame:
    """R-multiple of a hypothetical long and short trade at every ``step`` bar."""
    o, h, l, c = (f[k].values for k in ("open", "high", "low", "close"))
    n = len(f)
    rl = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    el = np.full(n, -1)
    es = np.full(n, -1)
    sl, ss = stops["stop_long"].values, stops["stop_short"].values
    for t in range(0, n - 2, step):
        tr = simulate_trade(o, h, l, c, t, 1, sl[t], inst, cfg)
        if tr:
            rl[t], el[t] = tr.r_multiple, tr.exit_i
        tr = simulate_trade(o, h, l, c, t, -1, ss[t], inst, cfg)
        if tr:
            rs[t], es[t] = tr.r_multiple, tr.exit_i
    return pd.DataFrame({"r_long": rl, "r_short": rs, "exit_long": el, "exit_short": es}, index=f.index)


def position_size(equity: float, risk_frac: float, stop_dist: float, inst: Instrument) -> float:
    """Lots so that hitting the stop loses ``risk_frac`` of equity."""
    if stop_dist <= 0:
        return 0.0
    loss_per_lot = stop_dist * inst.contract_size
    return max(0.0, equity * risk_frac / loss_per_lot)


def risk_fraction(cfg: StrategyConfig, p_win: float | None, drawdown: float, rr_eff: float = 1.6) -> float:
    """Base risk scaled by a fractional-Kelly read of the meta-model probability
    and throttled by the current drawdown."""
    base = cfg.risk_per_trade
    if p_win is not None and np.isfinite(p_win):
        kelly = p_win - (1 - p_win) / rr_eff
        scale = np.clip(0.5 + 2.0 * kelly, 0.5, 1.5)  # quarter-Kelly-like bounded multiplier
        base *= scale
    if drawdown >= cfg.dd_halt:
        return 0.0
    if drawdown >= cfg.dd_throttle:
        base *= 0.5
    return float(min(base, cfg.max_risk_per_trade))
