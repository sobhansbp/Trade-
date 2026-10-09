"""Leverage vs. outcome Monte Carlo (block bootstrap of daily strategy returns).

Answers "what risk level would a monthly return target require, and what does
it cost?" Each scenario rescales the strategy's daily returns to a target
annual volatility and simulates many 12-month paths by resampling 20-day blocks
(keeps volatility clustering and losing streaks together).

A ``haircut`` scenario halves the mean return, because live performance of a
backtested strategy is usually worse than the backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(daily: pd.Series, target_vols=(0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60),
             months: int = 12, paths: int = 4000, block: int = 20, haircut: float = 0.0,
             ruin_level: float = 0.5, seed: int = 11) -> pd.DataFrame:
    r = daily.dropna().values.astype(float)
    if len(r) < block * 3:
        raise ValueError("not enough history")
    mu, sd = r.mean(), r.std()
    base = r - haircut * mu  # remove part of the edge, keep the noise
    rng = np.random.default_rng(seed)
    days = months * 21
    n_blocks = int(np.ceil(days / block))
    starts = rng.integers(0, len(base) - block, size=(paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(paths, -1)[:, :days]
    sample = base[idx]
    rows = []
    for tv in target_vols:
        lev = tv / (sd * np.sqrt(252))
        rr = np.clip(sample * lev, -0.95, None)
        eq = np.cumprod(1 + rr, axis=1)
        dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
        m_end = eq[:, 20::21]
        m_prev = np.concatenate([np.ones((paths, 1)), m_end[:, :-1]], axis=1)
        mret = m_end / m_prev - 1
        final = eq[:, -1]
        ruined = (eq.min(axis=1) <= (1 - ruin_level))
        rows.append({
            "target_vol": tv, "leverage_x": round(lev, 2),
            "median_month": float(np.median(mret)), "mean_month": float(mret.mean()),
            "p_month_loss": float((mret < 0).mean()), "p_month_ge_10pct": float((mret >= 0.10).mean()),
            "median_year": float(np.median(final) - 1), "p_year_loss": float((final < 1).mean()),
            "median_max_dd": float(np.median(dd)), "p95_max_dd": float(np.quantile(dd, 0.05)),
            "p_drawdown_50pct": float(ruined.mean()),
        })
    return pd.DataFrame(rows)


def kelly_ceiling(sharpe: float) -> dict:
    """Max expected log-growth with perfectly known Sharpe (full Kelly): SR^2/2 per year."""
    g = sharpe ** 2 / 2
    return {"sharpe": sharpe, "max_log_growth_year": g, "max_growth_month": float(np.exp(g / 12) - 1),
            "sharpe_needed_for_10pct_month": float(np.sqrt(2 * 12 * np.log(1.10)))}
