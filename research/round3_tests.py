"""Round-3 strategy test battery on 16+ years of 1-minute data (HistData, UTC-corrected).

Every strategy uses parameters taken from its source (paper / common definition),
NOT optimised. In-sample = 2010-2018, out-of-sample = 2019-2026 (for most of the
cited papers this is after publication). Costs are charged per round trip at two
levels (retail and raw-spread). Multiple testing is handled with Benjamini-Hochberg
over all out-of-sample p-values.

    PYTHONPATH=. python3 research/round3_tests.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")
from aitrader.config import CACHE_DIR  # noqa: E402

OUT = Path("research/results")
OUT.mkdir(parents=True, exist_ok=True)
IS_END = pd.Timestamp("2019-01-01", tz="UTC")
COST = {  # round-trip cost in price units: (retail, raw)
    "EURUSD": (0.00012, 0.00006),
    "XAUUSD": (0.40, 0.20),
}
FOMC = [pd.Timestamp(d) for d in json.load(open("research/fomc_dates.json"))]


# ----------------------------------------------------------------- helpers

class Px:
    """Fast price lookups on an M5 frame (UTC index)."""

    def __init__(self, m5: pd.DataFrame):
        self.m5 = m5
        self.t = m5.index.values.astype("datetime64[ns]").astype(np.int64)
        self.o, self.h = m5["open"].values, m5["high"].values
        self.l, self.c = m5["low"].values, m5["close"].values
        self.tol = pd.Timedelta("15min").value
        self.bar = pd.Timedelta("5min").value

    def at(self, ts: pd.DatetimeIndex, tol: pd.Timedelta | None = None) -> np.ndarray:
        """Open of the first bar starting at/after ts (within tol, default 15 min)."""
        q = ts.values.astype("datetime64[ns]").astype(np.int64)
        tl = self.tol if tol is None else tol.value
        pos = np.searchsorted(self.t, q)
        out = np.full(len(q), np.nan)
        ok = pos < len(self.t)
        ok[ok] &= (self.t[pos[ok]] - q[ok]) <= tl
        out[ok] = self.o[pos[ok]]
        return out

    def before(self, ts: pd.DatetimeIndex) -> np.ndarray:
        """Close of the last bar ending at/before ts (within 15 min)."""
        q = ts.values.astype("datetime64[ns]").astype(np.int64) - self.bar
        pos = np.searchsorted(self.t, q, side="right") - 1
        out = np.full(len(q), np.nan)
        ok = pos >= 0
        ok[ok] &= (q[ok] - self.t[pos[ok]]) <= self.tol
        out[ok] = self.c[pos[ok]]
        return out

    def span(self, t0: pd.Timestamp, t1: pd.Timestamp) -> slice:
        a = np.searchsorted(self.t, t0.value)
        b = np.searchsorted(self.t, t1.value)
        return slice(a, b)


def local_ts(days: pd.DatetimeIndex, hm: str, tz: str, day_offset: int = 0) -> pd.DatetimeIndex:
    d = days + pd.Timedelta(days=day_offset)
    naive = pd.DatetimeIndex(d.strftime("%Y-%m-%d") + " " + hm)
    return naive.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")


def trading_days(start="2010-01-05", end="2026-09-30") -> pd.DatetimeIndex:
    d = pd.bdate_range(start, end)
    return d[~((d.month == 12) & (d.day.isin([25, 26]))) & ~((d.month == 1) & (d.day == 1))]


def summarize(name: str, sym: str, trades: pd.DataFrame, family: str, source: str) -> dict:
    """trades: index = entry time (UTC), columns gross (return), net_retail, net_raw."""
    rows = {"strategy": name, "symbol": sym, "family": family, "source": source}
    trades = trades.dropna()
    if len(trades) < 30:
        rows["n"] = len(trades)
        return rows
    yrs = (trades.index[-1] - trades.index[0]).days / 365.25
    tpy = len(trades) / max(yrs, 0.5)

    def block(tr, tag):
        if len(tr) < 20:
            return {}
        g, n1, n2 = tr["gross"], tr["net_retail"], tr["net_raw"]
        t, p = stats.ttest_1samp(n1, 0.0)
        tg, pg = stats.ttest_1samp(g, 0.0)
        sh = n1.mean() / n1.std() * math.sqrt(tpy) if n1.std() > 0 else np.nan
        sh_raw = n2.mean() / n2.std() * math.sqrt(tpy) if n2.std() > 0 else np.nan
        return {f"{tag}_n": len(tr), f"{tag}_gross_bp": g.mean() * 1e4, f"{tag}_net_bp": n1.mean() * 1e4,
                f"{tag}_net_raw_bp": n2.mean() * 1e4, f"{tag}_hit": (n1 > 0).mean(), f"{tag}_t": t, f"{tag}_p": p,
                f"{tag}_t_gross": tg, f"{tag}_p_gross": pg,
                f"{tag}_sharpe": sh, f"{tag}_sharpe_raw": sh_raw, f"{tag}_ann_net": n1.mean() * tpy,
                f"{tag}_ann_raw": n2.mean() * tpy}
    rows.update({"n": len(trades), "trades_per_year": tpy})
    rows.update(block(trades[trades.index < IS_END], "is"))
    rows.update(block(trades[trades.index >= IS_END], "oos"))
    rows.update(block(trades, "all"))
    return rows


def make_trades(entry_ts, exit_ts, p0, p1, side, sym) -> pd.DataFrame:
    side = np.asarray(side, dtype=float)
    ok = np.isfinite(p0) & np.isfinite(p1) & (side != 0)
    r = side * (p1 - p0) / p0
    cr, cw = COST[sym]
    df = pd.DataFrame({"gross": r, "net_retail": r - cr / p0, "net_raw": r - cw / p0,
                       "exit": exit_ts}, index=entry_ts)
    return df[ok]


# --------------------------------------------------------- strategy families

def window_strategy(px: Px, days, sym, start, end, side, label, family, source, filt=None):
    """Hold from local start to local end. start/end = (hm, tz, day_offset)."""
    t0 = local_ts(days, *start)
    t1 = local_ts(days, *end)
    keep = ~(t0.isna() | t1.isna())
    if filt is not None:
        keep &= filt
    t0, t1, d = t0[keep], t1[keep], days[keep]
    ok = (t1 > t0) & ((t1 - t0) < pd.Timedelta(hours=30))
    t0, t1 = t0[ok], t1[ok]
    p0 = px.at(t0)
    miss = ~np.isfinite(p0)
    p0[miss] = px.before(t0[miss])          # e.g. window starting inside gold's daily break
    p1 = px.before(t1)
    tr = make_trades(t0, t1, p0, p1, np.full(len(t0), side), sym)
    return summarize(label, sym, tr, family, source), tr


def breakout_strategy(px: Px, days, sym, rng, trade_until, exit_at, label, family, source,
                      target_mult=None, width_filter=False):
    """Range [rng] -> first break in (range end, trade_until] -> stop at opposite side,
    optional target = target_mult x range, else exit at exit_at."""
    r0, r1 = local_ts(days, *rng[0]), local_ts(days, *rng[1])
    tu, ex = local_ts(days, *trade_until), local_ts(days, *exit_at)
    cr, cw = COST[sym]
    rows, widths = [], []
    for a, b, u, e in zip(r0, r1, tu, ex):
        if pd.isna(a) or pd.isna(e):
            continue
        s = px.span(a, b)
        if s.stop - s.start < 6:
            continue
        hi, lo = px.h[s].max(), px.l[s].min()
        width = hi - lo
        widths.append(width)
        if width <= 0:
            continue
        if width_filter and len(widths) > 20 and width > 2 * np.median(widths[-21:-1]):
            continue
        w = px.span(b, e)
        if w.stop - w.start < 3:
            continue
        hh, ll, cc, tt = px.h[w], px.l[w], px.c[w], px.t[w]
        u_ns = u.value
        side = 0
        entry = stop = tgt = None
        res = None
        for i in range(len(hh)):
            if side == 0:
                if tt[i] >= u_ns:
                    break
                up, dn = hh[i] > hi, ll[i] < lo
                if up and dn:
                    break  # ambiguous bar: skip day
                if up:
                    side, entry, stop = 1, hi, lo
                elif dn:
                    side, entry, stop = -1, lo, hi
                if side and target_mult:
                    tgt = entry + side * target_mult * width
                if side:
                    t_entry = tt[i]
                    # same-bar stop check is skipped (entry at the level inside this bar)
                    continue
            else:
                if (side > 0 and ll[i] <= stop) or (side < 0 and hh[i] >= stop):
                    res = stop
                    break
                if tgt is not None and ((side > 0 and hh[i] >= tgt) or (side < 0 and ll[i] <= tgt)):
                    res = tgt
                    break
        if side == 0:
            continue
        if res is None:
            res = cc[-1]
        g = side * (res - entry) / entry
        rows.append((pd.Timestamp(t_entry, tz="UTC"), g, g - cr / entry, g - cw / entry))
    tr = pd.DataFrame(rows, columns=["t", "gross", "net_retail", "net_raw"]).set_index("t")
    return summarize(label, sym, tr, family, source), tr


def judas_strategy(px: Px, days, sym, label):
    """ICT 'Judas swing': Asian range 20:00-24:00 ET; in the London killzone (02:00-05:00 ET)
    a sweep beyond the range that closes back inside is faded toward the opposite side.
    Stop = sweep extreme; target = opposite side of the Asian range; time exit 12:00 ET."""
    a0, a1 = local_ts(days, "20:00", "America/New_York", -1), local_ts(days, "00:00", "America/New_York")
    k0, k1 = local_ts(days, "02:00", "America/New_York"), local_ts(days, "05:00", "America/New_York")
    ex = local_ts(days, "12:00", "America/New_York")
    cr, cw = COST[sym]
    rows = []
    for A0, A1, K0, K1, E in zip(a0, a1, k0, k1, ex):
        s = px.span(A0, A1)
        if s.stop - s.start < 6:
            continue
        hi, lo = px.h[s].max(), px.l[s].min()
        w = px.span(K0, E)
        if w.stop - w.start < 10:
            continue
        hh, ll, cc, tt = px.h[w], px.l[w], px.c[w], px.t[w]
        side, entry, stop, tgt, res = 0, None, None, None, None
        ext_hi, ext_lo = -np.inf, np.inf
        for i in range(len(hh)):
            if side == 0:
                if tt[i] >= K1.value:
                    break
                ext_hi, ext_lo = max(ext_hi, hh[i]), min(ext_lo, ll[i])
                if ext_hi > hi and cc[i] < hi and ext_lo >= lo:
                    side, entry, stop, tgt, t_entry = -1, cc[i], ext_hi, lo, tt[i]
                elif ext_lo < lo and cc[i] > lo and ext_hi <= hi:
                    side, entry, stop, tgt, t_entry = 1, cc[i], ext_lo, hi, tt[i]
                continue
            if (side > 0 and ll[i] <= stop) or (side < 0 and hh[i] >= stop):
                res = stop
                break
            if (side > 0 and hh[i] >= tgt) or (side < 0 and ll[i] <= tgt):
                res = tgt
                break
        if side == 0:
            continue
        res = cc[-1] if res is None else res
        g = side * (res - entry) / entry
        rows.append((pd.Timestamp(t_entry, tz="UTC"), g, g - cr / entry, g - cw / entry))
    tr = pd.DataFrame(rows, columns=["t", "gross", "net_retail", "net_raw"]).set_index("t")
    return summarize(label, sym, tr, "ICT / price action", "ICT (vendor claims 70-80% win, unaudited)"), tr


def williams_breakout(px: Px, days, sym, k=0.5):
    """Daily volatility breakout: NY day 17:00->17:00 ET; stops at open +/- k x prior range."""
    t0 = local_ts(days, "17:00", "America/New_York", -1)
    t1 = local_ts(days, "17:00", "America/New_York")
    cr, cw = COST[sym]
    rows, prev_rng = [], None
    for a, b in zip(t0, t1):
        s = px.span(a, b)
        if s.stop - s.start < 50:
            prev_rng = None
            continue
        o = px.o[s.start]
        hh, ll, cc, tt = px.h[s], px.l[s], px.c[s], px.t[s]
        if prev_rng:
            up, dn = o + k * prev_rng, o - k * prev_rng
            side, entry = 0, None
            for i in range(len(hh)):
                if hh[i] >= up and ll[i] <= dn:
                    break
                if hh[i] >= up:
                    side, entry, t_e = 1, up, tt[i]
                    break
                if ll[i] <= dn:
                    side, entry, t_e = -1, dn, tt[i]
                    break
            if side:
                g = side * (cc[-1] - entry) / entry
                rows.append((pd.Timestamp(t_e, tz="UTC"), g, g - cr / entry, g - cw / entry))
        prev_rng = hh.max() - ll.min()
    tr = pd.DataFrame(rows, columns=["t", "gross", "net_retail", "net_raw"]).set_index("t")
    return summarize(f"Williams volatility breakout k={k}", sym, tr, "breakout", "Larry Williams"), tr


def orb5_zarattini(px: Px, days, sym):
    """Zarattini-Aziz 5-min ORB adapted to COMEX open 08:20 ET: trade in the direction of the
    first 5-min bar, stop at its opposite extreme, target 10R, exit 16:55 ET."""
    t0 = local_ts(days, "08:20", "America/New_York")
    ex = local_ts(days, "16:55", "America/New_York")
    cr, cw = COST[sym]
    rows = []
    for a, e in zip(t0, ex):
        s = px.span(a, e)
        if s.stop - s.start < 20:
            continue
        o1, c1, h1, l1 = px.o[s.start], px.c[s.start], px.h[s.start], px.l[s.start]
        if c1 == o1:
            continue
        side = 1 if c1 > o1 else -1
        entry, stop = c1, (l1 if side > 0 else h1)
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        tgt = entry + side * 10 * risk
        res = None
        for i in range(s.start + 1, s.stop):
            if (side > 0 and px.l[i] <= stop) or (side < 0 and px.h[i] >= stop):
                res = stop
                break
            if (side > 0 and px.h[i] >= tgt) or (side < 0 and px.l[i] <= tgt):
                res = tgt
                break
        res = px.c[s.stop - 1] if res is None else res
        g = side * (res - entry) / entry
        rows.append((pd.Timestamp(px.t[s.start], tz="UTC"), g, g - cr / entry, g - cw / entry))
    tr = pd.DataFrame(rows, columns=["t", "gross", "net_retail", "net_raw"]).set_index("t")
    return summarize("5-min ORB (Zarattini-Aziz) at 08:20 ET", sym, tr, "breakout",
                     "Zarattini & Aziz 2023 (QQQ)"), tr


def intraday_momentum(px: Px, days, sym):
    """Gao-Han-Li-Zhou: first-half-hour return (from prior 17:00 ET to 08:50 ET) predicts the
    last hour of the COMEX day session (12:30-13:30 ET)."""
    prev = local_ts(days, "17:00", "America/New_York", -1)
    f1 = local_ts(days, "08:50", "America/New_York")
    l0, l1 = local_ts(days, "12:30", "America/New_York"), local_ts(days, "13:30", "America/New_York")
    first = np.log(px.at(f1) / px.before(prev))
    side = np.sign(first)
    p0, p1 = px.at(l0), px.before(l1)
    tr = make_trades(l0, l1, p0, p1, side, sym)
    return summarize("Intraday momentum (first 30m -> last hour)", sym, tr, "momentum",
                     "Gao, Han, Li & Zhou 2018 (SPY)"), tr


def gap_fade(px: Px, days, sym):
    """Fade the weekend gap from Friday close to the Sunday open until Monday 12:00 London."""
    mondays = days[days.dayofweek == 0]
    fri = local_ts(mondays, "16:55", "America/New_York", -3)
    sun = local_ts(mondays, "17:00", "America/New_York", -1)
    ex = local_ts(mondays, "12:00", "Europe/London")
    pf, ps = px.before(fri + pd.Timedelta("5min")), px.at(sun, pd.Timedelta("75min"))
    gap = (ps - pf) / pf
    side = np.where(np.abs(gap) > 0.001, -np.sign(gap), 0.0)
    tr = make_trades(sun + pd.Timedelta("1h"), ex, ps, px.before(ex), side, sym)
    return summarize("Weekend gap fade (>0.1%) to Mon 12:00 London", sym, tr, "calendar",
                     "practitioner lore"), tr


def event_drift(px: Px, sym, dates, start_off_h, end_off_h, label, family, source, side=1, anchor="14:00"):
    """Hold from FOMC statement time + start_off_h to + end_off_h (hours)."""
    st = pd.DatetimeIndex([pd.Timestamp(f"{d:%Y-%m-%d} {anchor}") for d in dates]).tz_localize(
        "America/New_York").tz_convert("UTC")
    t0, t1 = st + pd.Timedelta(hours=start_off_h), st + pd.Timedelta(hours=end_off_h)
    p0, p1 = px.at(t0), px.before(t1)
    tr = make_trades(t0, t1, p0, p1, np.full(len(t0), side), sym)
    return summarize(label, sym, tr, family, source), tr


def fomc_reversal(px: Px, sym, dates):
    """Lee & Wang: fade the first 30 minutes after the statement from 14:30 ET to 14:00 ET next day."""
    st = pd.DatetimeIndex([pd.Timestamp(f"{d:%Y-%m-%d} 14:00") for d in dates]).tz_localize(
        "America/New_York").tz_convert("UTC")
    a, b, c = st, st + pd.Timedelta("30min"), st + pd.Timedelta(hours=24)
    first = px.before(b) / px.at(a) - 1
    side = -np.sign(first)
    tr = make_trades(b, c, px.at(b), px.before(c), side, sym)
    return summarize("Post-FOMC reversal (fade first 30m, hold 24h)", sym, tr, "event",
                     "Lee & Wang (pre/post-FOMC reversal)"), tr


def nfp_days(px: Px) -> list:
    """Detect NFP release days: the Friday among days 1-10 with the largest 08:30 ET 5-min move."""
    fr = pd.bdate_range("2010-01-01", "2026-09-30")
    fr = fr[(fr.dayofweek == 4) & (fr.day <= 10)]
    out = {}
    t = local_ts(fr, "08:30", "America/New_York")
    mv = np.abs(px.before(t + pd.Timedelta("5min")) / px.at(t) - 1)
    for d, m in zip(fr, mv):
        if np.isfinite(m):
            key = (d.year, d.month)
            if key not in out or m > out[key][1]:
                out[key] = (d, m)
    return sorted(v[0] for v in out.values())


def nfp_continuation(px: Px, sym, days_nfp):
    t = local_ts(pd.DatetimeIndex(days_nfp), "08:30", "America/New_York")
    first = px.before(t + pd.Timedelta("15min")) / px.at(t) - 1
    a, b = t + pd.Timedelta("15min"), t + pd.Timedelta(hours=3)
    tr = make_trades(a, b, px.at(a), px.before(b), np.sign(first), sym)
    s1 = summarize("NFP continuation (first 15m direction, hold 3h)", sym, tr, "event", "FXStreet NFP studies")
    c = local_ts(pd.DatetimeIndex(days_nfp), "16:55", "America/New_York")
    tr2 = make_trades(a, c, px.at(a), px.before(c), -np.sign(first), sym)
    s2 = summarize("NFP reversal (fade first 15m, hold to NY close)", sym, tr2, "event", "FXStreet NFP studies")
    return (s1, tr), (s2, tr2)


# ------------------------------------------------------------------- main

def hour_profile(px: Px, sym: str) -> pd.DataFrame:
    """Mean 1-hour log return by London hour, in-sample vs out-of-sample (bp)."""
    h1 = px.m5["close"].resample("1h").last().dropna()
    r = np.log(h1).diff().dropna()
    lon = r.index.tz_convert("Europe/London")
    df = pd.DataFrame({"r": r.values * 1e4, "hour": lon.hour, "oos": r.index >= IS_END})
    g = df.groupby(["oos", "hour"])["r"].agg(["mean", "std", "count"])
    g["t"] = g["mean"] / (g["std"] / np.sqrt(g["count"]))
    return g.reset_index()


def run():
    results, curves = [], {}
    for sym in ("EURUSD", "XAUUSD"):
        m5 = pd.read_pickle(CACHE_DIR / f"hd_{sym}_M5.pkl")
        px = Px(m5)
        days = trading_days()
        print(f"== {sym}: {len(m5):,} M5 bars", flush=True)
        ET, LON, TKY, CET = "America/New_York", "Europe/London", "Asia/Tokyo", "Europe/Berlin"
        KMW = "Krohn, Mueller & Whelan (fixing reversals, 1999-2018)"
        # USD-sign: for EURUSD and gold, 'USD up' means short the instrument
        windows = [
            ("W1 pre-Tokyo fix: USD up (18:00 ET -> 10:00 Tokyo)", ("18:00", ET, -1), ("10:00", TKY, 0), -1, KMW),
            ("W2 post-Tokyo fix: USD down (10:00 Tokyo -> 08:00 London)", ("10:00", TKY, 0), ("08:00", LON, 0), 1, KMW),
            ("W3 pre-ECB fix: USD up (08:00 London -> 14:15 Frankfurt)", ("08:00", LON, 0), ("14:15", CET, 0), -1, KMW),
            ("W4 ECB -> 1h before London fix: USD down", ("14:15", CET, 0), ("15:00", LON, 0), 1, KMW),
            ("W5 into London 4pm fix: USD up (15:00 -> 16:00 London)", ("15:00", LON, 0), ("16:00", LON, 0), -1, KMW),
            ("W6 post London fix: USD down (16:00 London -> 16:50 ET)", ("16:00", LON, 0), ("16:50", ET, 0), 1, KMW),
            ("Breedon-Ranaldo: European morning (08:00-12:00 London)", ("08:00", LON, 0), ("12:00", LON, 0),
             -1 if sym == "EURUSD" else 0, "Breedon & Ranaldo 2013 (EUR weak in EU hours)"),
            ("Breedon-Ranaldo: US afternoon (12:00 ET -> 16:00 ET)", ("12:00", ET, 0), ("16:00", ET, 0),
             1 if sym == "EURUSD" else 0, "Breedon & Ranaldo 2013 (EUR strong in US hours)"),
        ]
        if sym == "XAUUSD":
            windows += [
                ("Gold Asian hours long (19:00 ET -> 03:00 ET)", ("19:00", ET, -1), ("03:00", ET, 0), 1,
                 "Donati & Jung (CBS) Asian-hours appreciation"),
                ("Gold overnight long (13:30 ET -> 08:20 ET)", ("13:30", ET, -1), ("08:20", ET, 0), 1,
                 "Blose et al. 2018 (overnight +)"),
                ("Gold COMEX day short (08:20 -> 13:30 ET)", ("08:20", ET, 0), ("13:30", ET, 0), -1,
                 "Blose et al. 2018 (daytime -)"),
                ("Gold into PM fix short (14:00 -> 15:00 London)", ("14:00", LON, 0), ("15:00", LON, 0), -1,
                 "London fix lore / Caminschi & Heaney"),
                ("Gold after PM fix long (15:00 -> 17:00 London)", ("15:00", LON, 0), ("17:00", LON, 0), 1,
                 "London fix lore / Caminschi & Heaney"),
            ]
        for label, a, b, side, src in windows:
            if side == 0:
                continue
            s, tr = window_strategy(px, days, sym, a, b, side, label, "session / fix", src)
            results.append(s)
            curves[(sym, label)] = tr
        # month-end variants of the London fix windows (Evans: stronger on the last day)
        last_day = pd.Series(days, index=days).groupby([days.year, days.month]).transform("max").values == days.values
        for label, a, b, side in [("W5 into London fix, month-end only", ("15:00", LON, 0), ("16:00", LON, 0), -1),
                                  ("W6 post London fix, month-end only", ("16:00", LON, 0), ("16:50", ET, 0), 1)]:
            s, tr = window_strategy(px, days, sym, a, b, side, label, "session / fix",
                                    "Evans 2018 (fix reversals strongest at month-end)", filt=last_day)
            results.append(s)
            curves[(sym, label)] = tr
        # calendar: daily NY-close returns
        day_ret = []
        for label, filt, side, src in [
            ("Last trading day of month long", last_day, 1, "month-end rebalancing lore"),
            ("Friday long", days.dayofweek == 4, 1, "Weekday effects (gold) 2016"),
            ("Tuesday short", days.dayofweek == 1, -1, "Weekday effects (gold) 2016"),
        ]:
            s, tr = window_strategy(px, days, sym, ("16:50", ET, -1), ("16:50", ET, 0), side, label, "calendar",
                                    src, filt=filt)
            results.append(s)
            curves[(sym, label)] = tr
        s, tr = window_strategy(px, days, sym, ("16:50", ET, -1), ("16:50", ET, 0), 1,
                                "BASELINE: long every day (close to close)", "baseline", "buy & hold")
        results.append(s)
        curves[(sym, s["strategy"])] = tr
        if sym == "XAUUSD":
            sepnov = days.month.isin([9, 11])
            s, tr = window_strategy(px, days, sym, ("16:50", ET, -1), ("16:50", ET, 0), 1,
                                    "Gold long in September & November", "calendar",
                                    "Baur 2013 (1980-2010; 2011+ is out-of-sample)", filt=sepnov)
            results.append(s)
            curves[(sym, s["strategy"])] = tr
        # breakouts / price action
        for label, rng, until, ex, tm, wf in [
            ("London breakout of Asian range (stop opposite, exit 16:00 London)",
             (("00:00", LON, 0), ("07:00", LON, 0)), ("11:00", LON, 0), ("16:00", LON, 0), None, False),
            ("London breakout, target 1x range, width filter",
             (("00:00", LON, 0), ("07:00", LON, 0)), ("11:00", LON, 0), ("16:00", LON, 0), 1.0, True),
            ("NY 30-min ORB 08:20-08:50 ET (exit 13:30 ET)",
             (("08:20", ET, 0), ("08:50", ET, 0)), ("11:00", ET, 0), ("13:30", ET, 0), None, False),
        ]:
            s, tr = breakout_strategy(px, days, sym, rng, until, ex, label, "breakout",
                                      "practitioner standard definitions", tm, wf)
            results.append(s)
            curves[(sym, label)] = tr
        for fn in (lambda: judas_strategy(px, days, sym, "ICT Judas swing (Asian range sweep fade, London KZ)"),
                   lambda: williams_breakout(px, days, sym, 0.5),
                   lambda: orb5_zarattini(px, days, sym),
                   lambda: intraday_momentum(px, days, sym),
                   lambda: gap_fade(px, days, sym)):
            s, tr = fn()
            results.append(s)
            curves[(sym, s["strategy"])] = tr
        # events
        fomc = [d for d in FOMC]
        for label, a, b, side, src in [
            ("Pre-FOMC drift: long 24h before statement", -24, 0, 1, "Lucca & Moench (equities); gold untested"),
            ("Pre-FOMC USD-up: short 48h before statement", -48, 0, -1, "Karnaukh (dollar ahead of FOMC)"),
            ("Post-FOMC 2-day drift long", 0.5, 48, 1, "Sunshine Profits (gold up after FOMC, anecdotal)"),
        ]:
            s, tr = event_drift(px, sym, fomc, a, b, label, "event", src, side)
            results.append(s)
            curves[(sym, label)] = tr
        s, tr = fomc_reversal(px, sym, fomc)
        results.append(s)
        curves[(sym, s["strategy"])] = tr
        for s, tr in nfp_continuation(px, sym, nfp_days(px)):
            results.append(s)
            curves[(sym, s["strategy"])] = tr
        hp = hour_profile(px, sym)
        hp.to_csv(OUT / f"hour_profile_{sym}.csv", index=False)
        print(f"   {sym}: {sum(1 for r in results if r['symbol'] == sym)} strategies tested", flush=True)

    res = pd.DataFrame(results)
    # Benjamini-Hochberg over all OOS p-values (net of retail costs) and gross p-values
    for col in ("oos_p", "oos_p_gross"):
        p = res[col].values
        m = np.isfinite(p)
        q = np.full(len(p), np.nan)
        order = np.argsort(p[m])
        ranked = p[m][order] * m.sum() / (np.arange(m.sum()) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        tmp = np.empty(m.sum())
        tmp[order] = np.minimum(ranked, 1.0)
        q[m] = tmp
        res[col.replace("_p", "_q")] = q
    res["same_sign"] = np.sign(res["is_net_bp"]) == np.sign(res["oos_net_bp"])
    res["passes_retail"] = (res["oos_q"] < 0.10) & (res["oos_net_bp"] > 0) & res["same_sign"]
    res["gross_real_cost_killed"] = (res["oos_q_gross"] < 0.10) & (res["oos_gross_bp"] > 0) & ~res["passes_retail"]
    res.to_csv(OUT / "round3_results.csv", index=False)
    pd.to_pickle(curves, CACHE_DIR / "round3_curves.pkl")
    return res


if __name__ == "__main__":
    r = run()
    cols = ["symbol", "strategy", "n", "is_gross_bp", "is_net_bp", "oos_gross_bp", "oos_net_bp", "oos_net_raw_bp",
            "oos_t", "oos_q", "oos_sharpe", "oos_sharpe_raw", "passes_retail", "gross_real_cost_killed"]
    pd.set_option("display.width", 250)
    print(r[cols].round(3).to_string())
