"""Account-level view: H1 tactical portfolio + daily swing book, monthly table,
and the leverage Monte Carlo used to answer return-target questions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest.engine import combine_portfolio
from .config import INSTRUMENTS
from .pipeline import swing_returns
from .risk.montecarlo import kelly_ceiling, simulate


def _daily_from_trades(port: pd.DataFrame, initial: float = 10_000.0) -> pd.Series:
    eq = port.set_index("exit_time")["equity"]
    eq.index = eq.index.tz_convert(None).normalize()
    eq = eq.groupby(level=0).last()
    start = port["entry_time"].min().tz_convert(None).normalize()
    idx = pd.bdate_range(start, eq.index.max())
    full = pd.concat([pd.Series([initial], index=[start - pd.Timedelta(days=1)]), eq]).groupby(level=0).last()
    full = full.reindex(idx.union([start - pd.Timedelta(days=1)])).ffill()
    return full.pct_change().dropna().reindex(idx).fillna(0.0)


def account_view(results: dict) -> dict:
    """results: {symbol: research()/analyze_live() output}."""
    runs = {s: r["_runs"]["production_oos"] for s, r in results.items()}
    port = combine_portfolio(runs, {s: INSTRUMENTS[s] for s in runs})
    out = {"portfolio_stats": port["stats"]}
    if not len(port["trades"]):
        return out
    h1 = _daily_from_trades(port["trades"])
    sw_all = [swing_returns(r["_bundle"]) for r in results.values()]
    sw_long = pd.concat(sw_all, axis=1).fillna(0).mean(axis=1)
    sw = sw_long.reindex(h1.index).fillna(0)
    comb = (h1 * (sw.std() / h1.std()) + sw) / 2 if h1.std() > 0 else sw

    def st(r):
        eq = (1 + r).cumprod()
        return {"sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else None,
                "ann_return": round(float(r.mean() * 252), 4), "vol": round(float(r.std() * np.sqrt(252)), 4),
                "max_dd": round(float((eq / eq.cummax() - 1).min()), 4)}

    t = port["trades"]
    eq_m = t.set_index("exit_time")["equity"].groupby(t["exit_time"].dt.to_period("M").values).last()
    full = pd.period_range(eq_m.index.min(), eq_m.index.max(), freq="M")
    eq_m = eq_m.reindex(full).ffill()
    monthly = eq_m.pct_change()
    monthly.iloc[0] = eq_m.iloc[0] / 10_000 - 1
    out.update({
        "window": [str(h1.index[0].date()), str(h1.index[-1].date())],
        "h1": st(h1), "swing": st(sw), "combined": st(comb),
        "corr_h1_swing": round(float(h1.corr(sw)), 2),
        "swing_20y": st(sw_long["2005":"2023"]),
        "monthly_h1_portfolio": {str(k): round(float(v), 4) for k, v in monthly.items()},
        "mc_backtested": simulate(comb).to_dict("records"),
        "mc_haircut": simulate(comb, haircut=0.5).to_dict("records"),
        "kelly": {"backtested": kelly_ceiling(float(comb.mean() / comb.std() * np.sqrt(252))),
                  "conservative": kelly_ceiling(0.9)},
        "_combined_daily": comb,
    })
    return out
