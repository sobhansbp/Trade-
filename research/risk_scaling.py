"""What would more risk per trade have done over the 16-year out-of-sample run?

Re-compounds the combined EURUSD + XAUUSD trade list from long_history_backtest.py
with the per-trade risk scaled up (0.5% is the bot's default). The drawdown
throttle is not re-simulated, so higher-risk rows are, if anything, optimistic.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aitrader.config import CACHE_DIR  # noqa: E402


def path(port: pd.DataFrame, k: float, initial=10_000.0):
    eq = initial * np.cumprod(1 + k * port["port_risk"].values * port["r_multiple"].values)
    s = pd.Series(eq, index=pd.DatetimeIndex(port["exit_time"]))
    return s


def stats(s: pd.Series, initial=10_000.0) -> dict:
    m = s.groupby(s.index.tz_convert(None).to_period("M")).last()
    m = m.reindex(pd.period_range(m.index.min(), m.index.max(), freq="M")).ffill()
    ret = m.pct_change()
    ret.iloc[0] = m.iloc[0] / initial - 1
    years = (s.index[-1] - s.index[0]).days / 365.25
    peak = np.maximum.accumulate(np.r_[initial, s.values])
    dd = (np.r_[initial, s.values] / peak - 1).min()
    roll12 = (1 + ret).rolling(12).apply(np.prod, raw=True) - 1
    return {"cagr": round(float((s.iloc[-1] / initial) ** (1 / years) - 1), 4),
            "mean_month": round(float(ret.mean()), 4), "median_month": round(float(ret.median()), 4),
            "positive_months": round(float((ret > 0).mean()), 3), "best_month": round(float(ret.max()), 4),
            "worst_month": round(float(ret.min()), 4), "worst_12m": round(float(roll12.min()), 4),
            "max_drawdown": round(float(dd), 4), "final_equity": round(float(s.iloc[-1]), 0),
            "months_ge_10pct": int((ret >= 0.10).sum()), "months": int(len(ret))}


if __name__ == "__main__":
    d = pd.read_pickle(CACHE_DIR / "long_history_runs.pkl")
    port = d["port"].sort_values("exit_time")
    out = {}
    for risk in (0.005, 0.01, 0.02, 0.03, 0.05):
        k = risk / 0.005
        out[f"{risk:.3f}"] = stats(path(port, k))
        print(f"risk {risk * 100:.1f}%/trade:", out[f"{risk:.3f}"])
    json.dump(out, open("research/results/risk_scaling.json", "w"), indent=1)
