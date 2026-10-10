"""Charts + compact exports for the round-3 strategy battery.

Reads research/results/round3_results.csv, hour_profile_*.csv and the per-trade
curves in data_cache/round3_curves.pkl; writes PNGs to reports/ and a compact
JSON (monthly cumulative curves) that the dashboard can embed without the cache.

    PYTHONPATH=. python3 research/round3_charts.py
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
from aitrader.config import CACHE_DIR, REPORT_DIR  # noqa: E402

RES = "research/results"
SPLIT = pd.Timestamp("2019-01-01", tz="UTC")
PICK = {
    "EURUSD": ["W3 pre-ECB fix: USD up (08:00 London -> 14:15 Frankfurt)",
               "W1 pre-Tokyo fix: USD up (18:00 ET -> 10:00 Tokyo)",
               "London breakout of Asian range (stop opposite, exit 16:00 London)",
               "5-min ORB (Zarattini-Aziz) at 08:20 ET",
               "ICT Judas swing (Asian range sweep fade, London KZ)",
               "BASELINE: long every day (close to close)"],
    "XAUUSD": ["Gold overnight long (13:30 ET -> 08:20 ET)",
               "Gold COMEX day short (08:20 -> 13:30 ET)",
               "London breakout of Asian range (stop opposite, exit 16:00 London)",
               "Williams volatility breakout k=0.5",
               "Pre-FOMC drift: long 24h before statement",
               "BASELINE: long every day (close to close)"],
}


def cum_monthly(tr: pd.DataFrame, col: str) -> pd.Series:
    s = tr[col].groupby(tr.index.tz_convert(None).to_period("M")).sum().cumsum() * 100
    return s


def main():
    r = pd.read_csv(f"{RES}/round3_results.csv")
    curves = pd.read_pickle(CACHE_DIR / "round3_curves.pkl")

    # 1) equity curves (cumulative % return per trade, summed; retail costs)
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), dpi=100)
    compact = {}
    for ax, sym in zip(axes, ("EURUSD", "XAUUSD")):
        for name in PICK[sym]:
            tr = curves[(sym, name)]
            net, gross = cum_monthly(tr, "net_retail"), cum_monthly(tr, "gross")
            compact[f"{sym}|{name}"] = {"t": [str(p) for p in net.index], "net": net.round(3).tolist(),
                                        "gross": gross.round(3).tolist()}
            ax.plot(net.index.to_timestamp(), net.values, lw=1.3, label=name[:58])
        ax.axvline(SPLIT.tz_convert(None), color="#555", ls="--", lw=1)
        ax.text(SPLIT.tz_convert(None), ax.get_ylim()[1], "  out-of-sample ->", va="top", fontsize=8)
        ax.axhline(0, color="#999", lw=0.7)
        ax.set_title(f"{sym}: cumulative return per strategy after retail costs (sum of trade returns, %)", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "round3_equity.png")
    plt.close(fig)

    # 2) hour-of-day profile, in-sample vs out-of-sample
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=100)
    hours = {}
    for ax, sym in zip(axes, ("EURUSD", "XAUUSD")):
        h = pd.read_csv(f"{RES}/hour_profile_{sym}.csv")
        hours[sym] = {}
        for k, (flag, col) in enumerate(((False, "#7f8c8d"), (True, "#c0392b"))):
            d = h[h.oos == flag].set_index("hour").reindex(range(24))
            hours[sym]["oos" if flag else "is"] = {"mean_bp": d["mean"].round(3).tolist(),
                                                   "t": d["t"].round(2).tolist()}
            ax.bar(np.arange(24) + (k - 0.5) * 0.4, d["mean"], width=0.4, color=col,
                   label=("2019-2026 (out-of-sample)" if flag else "2010-2018 (in-sample)"))
            for x, (m, t) in enumerate(zip(d["mean"], d["t"])):
                if abs(t) > 2:
                    ax.text(x + (k - 0.5) * 0.4, m, "*", ha="center", va="bottom" if m > 0 else "top", color=col)
        ax.axhline(0, color="#333", lw=0.7)
        ax.set_xticks(range(24))
        ax.set_title(f"{sym}: mean return by UTC hour (bp), * = |t| > 2", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "round3_hours.png")
    plt.close(fig)

    # 3) in-sample vs out-of-sample net bp per trade (all strategies)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=100)
    for sym, col in (("EURUSD", "#2980b9"), ("XAUUSD", "#b07d2b")):
        d = r[r.symbol == sym]
        ax.scatter(d.is_net_bp, d.oos_net_bp, s=28, color=col, label=sym, alpha=0.85)
    lim = max(r.is_net_bp.abs().max(), r.oos_net_bp.abs().max()) * 1.05
    lim = min(lim, 25)
    ax.plot([-lim, lim], [-lim, lim], color="#999", ls="--", lw=0.8)
    ax.axhline(0, color="#333", lw=0.7)
    ax.axvline(0, color="#333", lw=0.7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("in-sample 2010-2018, net bp per trade")
    ax.set_ylabel("out-of-sample 2019-2026, net bp per trade")
    ax.set_title("60 published / popular rules: does the in-sample edge survive? (retail costs)", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "round3_is_vs_oos.png")
    plt.close(fig)

    table = r[["strategy", "symbol", "family", "source", "trades_per_year", "is_gross_bp", "is_net_bp", "is_t",
               "oos_gross_bp", "oos_net_bp", "oos_t", "oos_t_gross", "oos_q", "same_sign", "passes_retail"]]
    json.dump({"curves": compact, "hours": hours, "table": table.round(4).to_dict(orient="records")},
              open(f"{RES}/round3_compact.json", "w"), indent=0)
    print("wrote", REPORT_DIR / "round3_equity.png", REPORT_DIR / "round3_hours.png",
          REPORT_DIR / "round3_is_vs_oos.png", f"{RES}/round3_compact.json")


if __name__ == "__main__":
    main()
