"""Multi-timeframe chart pack (D1 / H4 / H1 / M15) with structure, zones and levels."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
from aitrader.config import INSTRUMENTS, REPORT_DIR  # noqa: E402
from aitrader.data.loader import fetch_yahoo, resample_ohlc  # noqa: E402
from aitrader.indicators import core as ta  # noqa: E402
from aitrader.indicators.structure import market_structure  # noqa: E402

COL = {"bull_ob": "#26a65b", "bear_ob": "#d64541", "bull_fvg": "#3498db", "bear_fvg": "#9b59b6"}


def candles(ax, d):
    x = np.arange(len(d))
    up = d["close"] >= d["open"]
    col = np.where(up, "#26a65b", "#d64541")
    ax.vlines(x, d["low"], d["high"], color=col, lw=0.7)
    ax.bar(x, (d["close"] - d["open"]).abs(), bottom=np.minimum(d["open"], d["close"]), color=col, width=0.7)
    return x


def panel(ax, df, title, n, emas=(20, 50, 200), zones=True, swings=True, extra=None):
    full = df.copy()
    a = ta.atr(full, 14)
    ms, st = market_structure(full, a, left=3, right=3)
    d = full.tail(n)
    x = candles(ax, d)
    off = len(full) - n
    for p, c in zip(emas, ("#f39c12", "#2980b9", "#7f8c8d")):
        e = ta.ema(full["close"], p).tail(n)
        ax.plot(x, e.values, color=c, lw=1.1, label=f"EMA{p}")
    lo_y, hi_y = d["low"].min(), d["high"].max()
    pad = (hi_y - lo_y) * 0.06
    if zones:
        for z in st.zones:
            if not z.active or z.bottom > hi_y + pad or z.top < lo_y - pad:
                continue
            xs = max(0, z.start - off)
            ax.fill_between([xs, n + 2], z.bottom, z.top, color=COL[z.kind], alpha=0.16, lw=0)
            ax.text(n + 2, (z.top + z.bottom) / 2, z.kind.replace("_", " ").upper(), fontsize=6.5, color=COL[z.kind],
                    clip_on=True)
    if swings:
        for pos, kind, lvl, lab in st.swings:
            if pos - off < 0:
                continue
            ax.annotate(lab, (pos - off, lvl), fontsize=6.5, ha="center",
                        va="bottom" if kind == "H" else "top", color="#34495e")
    if extra:
        extra(ax, d, x)
    ax.set_ylim(lo_y - pad, hi_y + pad)
    ax.set_xlim(-1, n + 10)
    ticks = np.linspace(0, n - 1, 6).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([d.index[i].strftime("%y-%m-%d %H:%M" if "D1" not in title else "%Y-%m-%d") for i in ticks],
                       fontsize=7)
    trend = {1: "bullish", -1: "bearish", 0: "n/a"}[int(ms["ms_trend"].iloc[-1])]
    ax.set_title(f"{title} | structure {trend} | close {d['close'].iloc[-1]:.5g} | ATR {a.iloc[-1]:.4g}",
                 fontsize=9)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=7)
    return {"trend": trend, "close": float(d["close"].iloc[-1]), "atr": float(a.iloc[-1]),
            "swings": [(str(full.index[p]), k, float(v), lab) for p, k, v, lab in st.swings[-8:]],
            "zones": [(z.kind, float(z.bottom), float(z.top)) for z in st.zones if z.active][-8:]}


def fib_overlay(ax, d, x):
    hi_i, lo_i = int(np.argmax(d["high"].values)), int(np.argmin(d["low"].values))
    hi, lo = d["high"].iloc[hi_i], d["low"].iloc[lo_i]
    up = lo_i < hi_i
    for r in (0.236, 0.382, 0.5, 0.618, 0.786):
        lv = hi - r * (hi - lo) if up else lo + r * (hi - lo)
        ax.axhline(lv, color="#b07d2b", lw=0.6, ls="--", alpha=0.7)
        ax.text(0, lv, f"fib {r}", fontsize=6.5, color="#b07d2b", va="bottom")


def pack(sym):
    inst = INSTRUMENTS[sym]
    d1 = fetch_yahoo(inst.yahoo, "1d", "5y", max_age_s=600)
    h1 = fetch_yahoo(inst.yahoo, "1h", max_age_s=600)
    h4 = resample_ohlc(h1, "4h")
    m15 = fetch_yahoo(inst.yahoo, "15m", max_age_s=600)
    fig, axes = plt.subplots(4, 1, figsize=(14, 22), dpi=100)
    info = {
        "D1": panel(axes[0], d1, f"{sym} D1", 260, extra=fib_overlay),
        "H4": panel(axes[1], h4, f"{sym} H4", 180),
        "H1": panel(axes[2], h1, f"{sym} H1", 200, emas=(20, 50, 200)),
        "M15": panel(axes[3], m15, f"{sym} M15", 260, emas=(20, 50, 200)),
    }
    fig.tight_layout()
    path = REPORT_DIR / f"{sym}_mtf.png"
    fig.savefig(path)
    plt.close(fig)
    return path, info


if __name__ == "__main__":
    import json
    allinfo = {}
    for s in ("EURUSD", "XAUUSD"):
        p, info = pack(s)
        allinfo[s] = info
        print(p)
    json.dump(allinfo, open("research/results/mtf_info.json", "w"), indent=1)
    print(json.dumps(allinfo, indent=1)[:6000])
