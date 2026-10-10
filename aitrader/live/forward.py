"""Forward test of research hypotheses that did not clear the round-3 bar.

Registered on 2026-10-10, before any of the data it will be judged on existed. Each
cycle the rules are evaluated on completed days only (Yahoo 5-minute bars) and the
hypothetical trade is appended to journal/forward_tests.csv with gross and
net-of-retail-cost returns. Nothing is traded: the aim is a clean out-of-sample
record that decides later whether a rule earns a place in the bot.

A third candidate, fading the weekend gap in EURUSD, was dropped before registration:
its backtest profit (+4.3bp net out of sample) existed only when entering at the
exact Sunday re-open print and vanished with a 15-minute delay (-0.1bp).
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import INSTRUMENTS, JOURNAL_DIR

START = pd.Timestamp("2026-10-12")          # first trading day after registration
PATH = JOURNAL_DIR / "forward_tests.csv"
COST = {"EURUSD": 0.00012, "XAUUSD": 0.40}  # retail round trip, price units
LON, ET = "Europe/London", "America/New_York"
BAR = pd.Timedelta("5min")
FIELDS = ["hypothesis", "symbol", "day", "side", "entry_time", "entry", "exit_time", "exit",
          "gross_bp", "net_bp", "note", "logged_at"]

HYPOTHESES = {
    "gold_london_breakout_short": (
        "XAUUSD", "فروش طلا با شکست کف رنج ۰۰:۰۰ تا ۰۷:۰۰ لندن پیش از ساعت ۱۱، حد ضرر سقف رنج، خروج ۱۶:۰۰ لندن "
                  "(درون‌نمونه +۳٫۶ و برون‌نمونه +۳٫۰ واحد پایه خالص، غیرمعنادار)"),
    "gold_overnight_long": (
        "XAUUSD", "خرید شبانه طلا از ۱۳:۳۰ تا ۰۸:۲۰ نیویورک (برون‌نمونه +۵٫۰ واحد پایه خالص، عمدتاً روند صعودی طلا)"),
    "gold_day_long": (
        "XAUUSD", "گروه کنترل: خرید طلا در جلسه روزانه کومکس ۰۸:۲۰ تا ۱۳:۳۰ نیویورک؛ اگر سود شبانه فقط روند کلی طلا باشد، "
                  "این دو باید مشابه باشند"),
}


def _ts(day, hm: str, tz: str, offset: int = 0) -> pd.Timestamp:
    d = (pd.Timestamp(day) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
    return pd.Timestamp(f"{d} {hm}").tz_localize(tz).tz_convert("UTC")


def _open_at(df: pd.DataFrame, t: pd.Timestamp, tol: str = "20min"):
    i = df.index.searchsorted(t)
    if i < len(df) and df.index[i] - t <= pd.Timedelta(tol):
        return df.index[i], float(df["open"].iloc[i])
    return None, None


def _close_before(df: pd.DataFrame, t: pd.Timestamp, tol: str = "20min"):
    i = df.index.searchsorted(t) - 1
    if i >= 0 and t - df.index[i] <= pd.Timedelta(tol):
        return df.index[i] + BAR, float(df["close"].iloc[i])
    return None, None


def _trade(name, sym, day, side, te, pe, tx, px, note=""):
    g = side * (px - pe) / pe
    return {"hypothesis": name, "symbol": sym, "day": pd.Timestamp(day).strftime("%Y-%m-%d"), "side": side,
            "entry_time": str(te), "entry": round(pe, 5), "exit_time": str(tx), "exit": round(px, 5),
            "gross_bp": round(g * 1e4, 3), "net_bp": round((g - COST[sym] / pe) * 1e4, 3), "note": note}


def _none(name, sym, day, note):
    return {"hypothesis": name, "symbol": sym, "day": pd.Timestamp(day).strftime("%Y-%m-%d"), "side": 0,
            "entry_time": "", "entry": "", "exit_time": "", "exit": "", "gross_bp": 0.0, "net_bp": 0.0, "note": note}


def gold_london_breakout_short(df: pd.DataFrame, day, name="gold_london_breakout_short", sym="XAUUSD"):
    r0, r1 = _ts(day, "00:00", LON), _ts(day, "07:00", LON)
    tu, ex = _ts(day, "11:00", LON), _ts(day, "16:00", LON)
    rng = df[(df.index >= r0) & (df.index < r1)]
    win = df[(df.index >= r1) & (df.index < ex)]
    if len(rng) < 30 or len(win) < 30:
        return _none(name, sym, day, "no data")
    hi, lo = float(rng["high"].max()), float(rng["low"].min())
    side, t_entry = 0, None
    for t, h, low in zip(win.index, win["high"].values, win["low"].values):
        if side == 0:
            if t >= tu:
                return _none(name, sym, day, "no break before 11:00")
            up, dn = h > hi, low < lo
            if up and dn:
                return _none(name, sym, day, "ambiguous bar")
            if up:
                return _none(name, sym, day, "first break was upward")
            if dn:
                side, t_entry = -1, t
            continue   # the entry bar itself is not checked for the stop (as in the research test)
        if h >= hi:
            return _trade(name, sym, day, -1, t_entry, lo, t + BAR, hi, "stopped")
    return _trade(name, sym, day, -1, t_entry, lo, win.index[-1] + BAR, float(win["close"].iloc[-1]), "time exit")


def gold_overnight_long(df: pd.DataFrame, day, name="gold_overnight_long", sym="XAUUSD"):
    if pd.Timestamp(day).dayofweek == 0:      # the research rule had no Sunday entry
        return _none(name, sym, day, "monday: no entry the previous day")
    te, pe = _open_at(df, _ts(day, "13:30", ET, -1))
    tx, px = _close_before(df, _ts(day, "08:20", ET))
    if pe is None or px is None:
        return _none(name, sym, day, "no data")
    return _trade(name, sym, day, 1, te, pe, tx, px)


def gold_day_long(df: pd.DataFrame, day, name="gold_day_long", sym="XAUUSD"):
    te, pe = _open_at(df, _ts(day, "08:20", ET))
    tx, px = _close_before(df, _ts(day, "13:30", ET))
    if pe is None or px is None:
        return _none(name, sym, day, "no data")
    return _trade(name, sym, day, 1, te, pe, tx, px)


RULES = {"gold_london_breakout_short": (gold_london_breakout_short, ("16:00", LON)),
         "gold_overnight_long": (gold_overnight_long, ("08:20", ET)),
         "gold_day_long": (gold_day_long, ("13:30", ET))}


def _logged() -> set:
    if not PATH.exists():
        return set()
    with PATH.open(encoding="utf-8") as fh:
        return {(r["hypothesis"], r["day"]) for r in csv.DictReader(fh)}


def update(now: pd.Timestamp | None = None, data: dict | None = None) -> list[str]:
    """Evaluate every completed, not yet logged day since START; returns log lines."""
    from ..data.loader import fetch_yahoo

    now = now or pd.Timestamp.now(tz="UTC")
    done = _logged()
    days = pd.bdate_range(START, now.tz_convert(None).normalize())
    todo = [(n, d) for n in RULES for d in days
            if (n, d.strftime("%Y-%m-%d")) not in done
            and now >= _ts(d, *RULES[n][1]) + pd.Timedelta("30min")]
    if not todo:
        return []
    data = data or {}
    for sym in {HYPOTHESES[n][0] for n, _ in todo} - set(data):
        data[sym] = fetch_yahoo(INSTRUMENTS[sym].yahoo, "5m", "60d", max_age_s=900)
    rows = []
    for n, d in todo:
        rec = RULES[n][0](data[HYPOTHESES[n][0]], d)
        if rec is not None:
            rec["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rows.append(rec)
    if rows:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        new = not PATH.exists()
        with PATH.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
    return [f"forward test {r['hypothesis']} {r['day']}: "
            + (f"side {r['side']:+d} net {r['net_bp']:+.1f}bp" if r["side"] else r["note"]) for r in rows]


def summary() -> list[dict]:
    if not PATH.exists():
        return []
    df = pd.read_csv(PATH)
    out = []
    for n, (sym, desc) in HYPOTHESES.items():
        x = df[(df.hypothesis == n) & (df.side != 0)]["net_bp"].astype(float)
        sd = x.std(ddof=1) if len(x) > 2 else np.nan
        out.append({"hypothesis": n, "symbol": sym, "desc_fa": desc, "n": int(len(x)),
                    "mean_net_bp": float(x.mean()) if len(x) else 0.0,
                    "sum_net_bp": float(x.sum()), "hit": float((x > 0).mean()) if len(x) else 0.0,
                    "t": float(x.mean() / sd * np.sqrt(len(x))) if sd and np.isfinite(sd) and sd > 0 else np.nan})
    return out
