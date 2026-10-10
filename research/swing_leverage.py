"""The one layer with long-run evidence: the daily swing book (bias x vol-target leverage).

EURUSD + XAUUSD with equal risk, 2005 to today, at several portfolio volatility targets.
Leverage is set from trailing volatility only (no look-ahead); costs are charged on turnover.

    PYTHONPATH=. python3 research/swing_leverage.py
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aitrader.config import INSTRUMENTS  # noqa: E402
from aitrader.data.loader import load_macro_daily  # noqa: E402
from aitrader.strategic import daily_bias, load_daily  # noqa: E402


def book(sym: str, macro) -> pd.Series:
    inst = INSTRUMENTS[sym]
    close = load_daily(sym, inst.yahoo)
    b = daily_bias(sym, close, macro, target_vol=0.10)
    c = close.copy()
    idx = pd.DatetimeIndex(c.index)
    c.index = (idx.tz_convert(None) if idx.tz is not None else idx).normalize()
    c = c[~c.index.duplicated(keep="last")]
    r = c.pct_change().reindex(b.index)
    pos = (b["bias"] * b["leverage"]).fillna(0.0)
    return (pos * r - pos.diff().abs() * inst.spread / c.reindex(b.index)).dropna()


def stats(x: pd.Series) -> dict:
    eq = (1 + x).cumprod()
    m = (1 + x).groupby(x.index.to_period("M")).prod() - 1
    years = len(x) / 252
    roll12 = (1 + m).rolling(12).apply(np.prod, raw=True) - 1
    return {"cagr": round(float(eq.iloc[-1] ** (1 / years) - 1), 4), "vol": round(float(x.std() * np.sqrt(252)), 4),
            "sharpe": round(float(x.mean() / x.std() * np.sqrt(252)), 2),
            "max_drawdown": round(float((eq / eq.cummax() - 1).min()), 4),
            "median_month": round(float(m.median()), 4), "mean_month": round(float(m.mean()), 4),
            "positive_months": round(float((m > 0).mean()), 3), "worst_month": round(float(m.min()), 4),
            "best_month": round(float(m.max()), 4), "worst_12m": round(float(roll12.min()), 4),
            "months_ge_10pct": int((m >= 0.10).sum()), "months": int(len(m)), "years": round(years, 1)}


if __name__ == "__main__":
    macro = load_macro_daily()
    books = pd.concat({s: book(s, macro) for s in ("EURUSD", "XAUUSD")}, axis=1).dropna()
    port = books.mean(axis=1)                          # equal capital, each book already vol-targeted to 10%
    out = {"period": [str(port.index[0].date()), str(port.index[-1].date())],
           "books_sharpe": {s: round(float(books[s].mean() / books[s].std() * np.sqrt(252)), 2) for s in books},
           "corr": round(float(books.corr().iloc[0, 1]), 2), "targets": {}}
    # scale to a portfolio vol target using trailing 60-day realised vol of the portfolio itself (lagged)
    rv = port.rolling(60).std().shift(1) * np.sqrt(252)
    for tv in (0.10, 0.15, 0.20, 0.30, 0.40):
        lev = (tv / rv).clip(upper=8).fillna(0)
        x = (port * lev).iloc[60:]
        out["targets"][f"{tv:.2f}"] = {"all": stats(x), "since_2014": stats(x["2014":])}
        a = out["targets"][f"{tv:.2f}"]["all"]
        print(f"vol target {tv:.0%}: CAGR {a['cagr']:+.1%} sharpe {a['sharpe']} maxDD {a['max_drawdown']:+.1%} "
              f"median month {a['median_month']:+.2%} worst month {a['worst_month']:+.1%} worst 12m {a['worst_12m']:+.1%} "
              f"months>=10%: {a['months_ge_10pct']}/{a['months']}")
    json.dump(out, open("research/results/swing_leverage.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "targets"}))
