"""Re-validate the production strategy on 16+ years of H1 data (HistData, UTC-corrected).

Same code path as `python -m aitrader backtest` (features, purged walk-forward ML,
12-analyst desk, daily strategic bias, trade management), only the H1 source is
the long HistData history instead of Yahoo's 730-day window. Walk-forward ML
starts after the first 25% of the data, so roughly 2014-2026 is out-of-sample.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from aitrader.config import CACHE_DIR, INSTRUMENTS, StrategyConfig  # noqa: E402
from aitrader.backtest.engine import combine_portfolio  # noqa: E402
from aitrader.features import build_features  # noqa: E402
from aitrader.pipeline import Bundle, load_inputs, research, swing_stats  # noqa: E402
from aitrader.risk.manager import label_outcomes, stop_distances  # noqa: E402
from aitrader.strategic import attach_bias, daily_bias, load_daily  # noqa: E402

CFG = StrategyConfig(wf_initial_frac=0.25, wf_step_bars=2500, embargo_bars=72)


def bundle_for(sym: str) -> Bundle:
    inst, _, peer, macro, cot = load_inputs(sym)
    df = pd.read_pickle(CACHE_DIR / f"hd_{sym}_H1.pkl")
    df = df[df.index.dayofweek < 5 | ((df.index.dayofweek == 6) & (df.index.hour >= 21))]
    t = time.time()
    f, state = build_features(df, inst, peer, macro, cot, return_state=True)
    daily = load_daily(sym, inst.yahoo)
    bias = daily_bias(sym, daily, macro, target_vol=CFG.swing_target_vol)
    f = f.join(attach_bias(f.index, bias.drop(columns=["close"])))
    stops = stop_distances(f, CFG)
    labels = label_outcomes(f, stops, inst, CFG)
    print(f"{sym}: {len(f):,} H1 bars {f.index[0]} -> {f.index[-1]} features+labels {time.time() - t:.0f}s", flush=True)
    return Bundle(inst, df, f, state, stops, labels, extras={"daily_bias": bias, "daily_close": daily, "macro": macro})


def monthly(trades: pd.DataFrame, initial=10_000.0) -> pd.Series:
    eq = trades.set_index("exit_time")["equity"]
    m = eq.groupby(eq.index.tz_convert(None).to_period("M")).last()
    full = pd.period_range(m.index.min(), m.index.max(), freq="M")
    m = m.reindex(full).ffill()
    ret = m.pct_change()
    ret.iloc[0] = m.iloc[0] / initial - 1
    return ret


if __name__ == "__main__":
    out, runs = {}, {}
    for sym in ("EURUSD", "XAUUSD"):
        t = time.time()
        b = bundle_for(sym)
        r = research(sym, CFG, bundle=b)
        runs[sym] = r["_runs"]["production_oos"]
        tr = runs[sym]["trades"]
        mret = monthly(tr) if len(tr) else pd.Series(dtype=float)
        out[sym] = {k: v for k, v in r.items() if not k.startswith("_")}
        out[sym]["monthly"] = {str(k): round(float(v), 5) for k, v in mret.items()}
        out[sym]["by_year"] = {str(y): round(float((1 + g).prod() - 1), 4)
                               for y, g in mret.groupby(mret.index.year)}
        print(f"{sym} done in {time.time() - t:.0f}s | ML OOS from {r['ml_oos_start']}", flush=True)
        for k, s in r["runs"].items():
            print(f"  {k:22s} trades={s.get('trades', 0):5d} win={s.get('win_rate', 0):.2f} "
                  f"avgR={s.get('avg_r', 0):+.3f} PF={s.get('profit_factor', 0)} sharpe={s.get('sharpe', 0)} "
                  f"dd={s.get('max_drawdown', 0)} ret={s.get('total_return', 0):+.4f} cagr={s.get('cagr', 0)}", flush=True)
        print("  by year:", out[sym]["by_year"], flush=True)
    port = combine_portfolio(runs, INSTRUMENTS)
    pt = port["trades"]
    pm = monthly(pt)
    out["portfolio"] = {"stats": port["stats"], "monthly": {str(k): round(float(v), 5) for k, v in pm.items()},
                        "by_year": {str(y): round(float((1 + g).prod() - 1), 4) for y, g in pm.groupby(pm.index.year)}}
    print("PORTFOLIO", json.dumps(port["stats"], default=str))
    print("PORTFOLIO by year", out["portfolio"]["by_year"])
    print(f"PORTFOLIO monthly: mean {pm.mean() * 100:+.2f}% median {pm.median() * 100:+.2f}% "
          f"positive {(pm > 0).mean() * 100:.0f}% best {pm.max() * 100:+.2f}% worst {pm.min() * 100:+.2f}%")
    json.dump(out, open("research/results/long_history_backtest.json", "w"), indent=1, default=str)
    pd.to_pickle({"runs": runs, "port": pt}, CACHE_DIR / "long_history_runs.pkl")
