"""Event-driven backtester + performance statistics.

One position per instrument at a time, signals evaluated on bar close,
fills at next open with spread and slippage, trade management identical to
``risk.manager.simulate_trade``. Equity compounds by ``risk_fraction * R``.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Instrument, StrategyConfig
from ..risk.manager import risk_fraction, simulate_trade


def signal_mask(f: pd.DataFrame, ens: pd.DataFrame, cfg: StrategyConfig,
                meta_p: pd.Series | None = None) -> pd.Series:
    """Production rule: trade only in the direction of a strong daily strategic
    bias, and only while the H1 analyst desk agrees with it."""
    bias = f["sb_bias"].fillna(0.0)
    side = np.sign(bias)
    ok = (bias.abs() >= cfg.bias_min) & (ens["composite"] * side > cfg.comp_align)
    ok &= ~f["hour"].isin(cfg.avoid_hours_utc)
    if meta_p is not None and cfg.use_meta:
        ok &= meta_p.fillna(0) >= cfg.meta_threshold
    return pd.Series(np.where(ok, side, 0.0), index=f.index)


def desk_only_signal(f: pd.DataFrame, ens: pd.DataFrame, cfg: StrategyConfig,
                     meta_p: pd.Series | None = None) -> pd.Series:
    """Legacy rule (v1): H1 desk composite alone. Kept as a research baseline -
    it lost money out-of-sample."""
    comp = ens["composite"]
    side = np.sign(comp)
    ok = (comp.abs() >= ens["threshold"]) & (ens["agreement"] >= cfg.min_agreement)
    ok &= ~f["hour"].isin(cfg.avoid_hours_utc)
    # do not fight aligned H4 + D1 structure unless the reversal evidence is strong
    rev = (ens["a_liquidity"] + ens["a_smc"] + ens["a_patterns"]) * side
    htf_against = (f["h4_ms_trend"] * side < 0) & (f["d1_ms_trend"] * side < 0)
    ok &= ~(htf_against & (rev < 0.8))
    if meta_p is not None and cfg.use_meta:
        ok &= meta_p.fillna(0) >= cfg.meta_threshold
    return pd.Series(np.where(ok, side, 0.0), index=f.index)


def run_backtest(f: pd.DataFrame, signals: pd.Series, stops: pd.DataFrame, inst: Instrument,
                 cfg: StrategyConfig, meta_p: pd.Series | None = None, start: int = 0,
                 initial_equity: float = 10_000.0) -> dict:
    o, h, l, c = (f[k].values for k in ("open", "high", "low", "close"))
    sig = signals.values
    sl, ss = stops["stop_long"].values, stops["stop_short"].values
    mp = meta_p.values if meta_p is not None else None
    eq = initial_equity
    peak = eq
    trades = []
    t = start
    n = len(f)
    while t < n - 2:
        s = sig[t]
        if s == 0:
            t += 1
            continue
        dd = 1 - eq / peak
        p = mp[t] if mp is not None and np.isfinite(mp[t]) else None
        rf = risk_fraction(cfg, p, dd)
        if rf <= 0:
            t += 1
            continue
        tr = simulate_trade(o, h, l, c, t, int(s), sl[t] if s > 0 else ss[t], inst, cfg)
        if tr is None:
            break
        pnl = eq * rf * tr.r_multiple
        eq += pnl
        peak = max(peak, eq)
        d = asdict(tr)
        d.update(signal_time=f.index[t], entry_time=f.index[tr.entry_i], exit_time=f.index[tr.exit_i],
                 risk_frac=rf, pnl=pnl, equity=eq, meta_p=p, symbol=inst.symbol)
        trades.append(d)
        t = tr.exit_i + cfg.cooldown_bars
    tdf = pd.DataFrame(trades)
    return {"trades": tdf, "stats": performance(tdf, f.index[start], f.index[-1], initial_equity),
            "equity": equity_curve(tdf, f.index[start:], initial_equity)}


def equity_curve(trades: pd.DataFrame, index: pd.DatetimeIndex, initial: float) -> pd.Series:
    eq = pd.Series(np.nan, index=index)
    eq.iloc[0] = initial
    if len(trades):
        last = trades.groupby("exit_time")["equity"].last()
        last = last[last.index.isin(index)]
        eq.loc[last.index] = last.values
    return eq.ffill()


def performance(trades: pd.DataFrame, start, end, initial: float) -> dict:
    years = max((end - start).total_seconds() / (365.25 * 86400), 1e-9)
    if trades is None or len(trades) == 0:
        return {"trades": 0, "years": round(years, 2)}
    r = trades["r_multiple"]
    wins, losses = r[r > 0], r[r <= 0]
    final = trades["equity"].iloc[-1]
    eq = pd.concat([pd.Series([initial], index=[start]), trades.set_index("exit_time")["equity"]])
    eq = eq[~eq.index.duplicated(keep="last")]
    daily = eq.resample("1D").last().ffill()
    dr = daily.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * math.sqrt(252) if dr.std() > 0 else np.nan
    downside = dr[dr < 0].std()
    sortino = dr.mean() / downside * math.sqrt(252) if downside and downside > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    cagr = (final / initial) ** (1 / years) - 1
    # probabilistic Sharpe ratio (Bailey & Lopez de Prado) vs SR*=0, per-trade
    n = len(r)
    sr_t = r.mean() / r.std() if r.std() > 0 else 0.0
    sk, ku = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - sk * sr_t + (ku - 1) / 4 * sr_t ** 2))
    psr = float(stats.norm.cdf(sr_t * math.sqrt(n - 1) / denom)) if n > 2 else np.nan
    streak = mx = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        mx = max(mx, streak)
    return {
        "trades": int(n), "years": round(years, 2), "trades_per_month": round(n / (years * 12), 1),
        "win_rate": round(float((r > 0).mean()), 3),
        "avg_r": round(float(r.mean()), 3), "median_r": round(float(r.median()), 3),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else np.inf,
        "total_return": round(float(final / initial - 1), 4), "cagr": round(float(cagr), 4),
        "sharpe": round(float(sharpe), 2), "sortino": round(float(sortino), 2),
        "max_drawdown": round(float(dd), 4),
        "calmar": round(float(cagr / -dd), 2) if dd < 0 else np.inf,
        "psr": round(psr, 3), "max_losing_streak": int(mx),
        "long_trades": int((trades["side"] > 0).sum()), "short_trades": int((trades["side"] < 0).sum()),
        "exit_reasons": trades["reason"].value_counts().to_dict(),
    }


def combine_portfolio(results: dict[str, dict], instruments: dict[str, Instrument],
                      initial: float = 10_000.0, corr_haircut: float = 0.5) -> dict:
    """Merge per-symbol trade lists into one account. A new trade that adds to an
    open trade with the same USD direction gets its risk cut by ``corr_haircut``."""
    rows = []
    for sym, res in results.items():
        t = res["trades"]
        if len(t):
            rows.append(t.assign(symbol=sym))
    if not rows:
        return {"trades": pd.DataFrame(), "stats": {"trades": 0}}
    allt = pd.concat(rows).sort_values("entry_time").reset_index(drop=True)
    eq, open_pos, out = initial, [], []
    for _, tr in allt.iterrows():
        open_pos = [p for p in open_pos if p["exit_time"] > tr["entry_time"]]
        usd_dir = tr["side"] * instruments[tr["symbol"]].usd_side
        rf = tr["risk_frac"]
        if any(p["usd_dir"] == usd_dir for p in open_pos):
            rf *= corr_haircut
        out.append({**tr.to_dict(), "port_risk": rf, "usd_dir": usd_dir})
        open_pos.append({"exit_time": tr["exit_time"], "usd_dir": usd_dir})
    port = pd.DataFrame(out).sort_values("exit_time").reset_index(drop=True)
    eqs = []
    for _, tr in port.iterrows():
        eq += eq * tr["port_risk"] * tr["r_multiple"]
        eqs.append(eq)
    port["equity"] = eqs
    start = min(r["equity"].index[0] for r in results.values())
    end = max(r["equity"].index[-1] for r in results.values())
    return {"trades": port, "stats": performance(port, start, end, initial)}
