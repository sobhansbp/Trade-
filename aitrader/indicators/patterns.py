"""Candlestick, divergence, classical chart, harmonic and Elliott-wave patterns.

Pivot-based detectors only fire on the bar where the last pivot is confirmed,
so nothing peeks into the future. Signals decay geometrically afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------- candlesticks


def candlesticks(df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    po, pc, ph, pl = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    pbody = (pc - po).abs()

    out = pd.DataFrame(index=df.index)
    out["bull_engulf"] = ((c > o) & (pc < po) & (c >= po) & (o <= pc) & (body > pbody)).astype(float)
    out["bear_engulf"] = ((c < o) & (pc > po) & (c <= po) & (o >= pc) & (body > pbody)).astype(float)
    out["hammer"] = ((lower >= 2 * body) & (lower >= 0.55 * rng) & ((c - l) / rng >= 0.6)).astype(float)
    out["shooting_star"] = ((upper >= 2 * body) & (upper >= 0.55 * rng) & ((h - c) / rng >= 0.6)).astype(float)
    out["doji"] = (body <= 0.1 * rng).astype(float)
    out["inside_bar"] = ((h <= ph) & (l >= pl)).astype(float)
    # morning / evening star (3 candles)
    o2, c2 = o.shift(2), c.shift(2)
    small_mid = pbody <= 0.35 * (h.shift(1) - l.shift(1))
    out["morning_star"] = ((c2 < o2) & small_mid & (c > o) & (c > (o2 + c2) / 2)).astype(float)
    out["evening_star"] = ((c2 > o2) & small_mid & (c < o) & (c < (o2 + c2) / 2)).astype(float)
    up3 = (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) & (c > pc) & (pc > c2)
    dn3 = (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2)) & (c < pc) & (pc < c2)
    out["three_soldiers"] = up3.astype(float)
    out["three_crows"] = dn3.astype(float)
    # displacement candle: large body relative to ATR, closes near extreme
    out["displacement"] = np.where((body > 1.3 * atr) & (body / rng > 0.7), np.sign(c - o), 0.0)

    score = (0.8 * out["bull_engulf"] - 0.8 * out["bear_engulf"]
             + 0.6 * out["hammer"] - 0.6 * out["shooting_star"]
             + 0.7 * out["morning_star"] - 0.7 * out["evening_star"]
             + 0.4 * out["three_soldiers"] - 0.4 * out["three_crows"]
             + 0.3 * out["displacement"])
    out["candle_score"] = score.clip(-1, 1).fillna(0)
    return out


# ---------------------------------------------------------------- zigzag

def _zigzag_events(swings: pd.DataFrame):
    """Yield (t, pivots) where pivots is the alternating list of
    (pos, kind, level) known at confirmation bar t."""
    piv: list = []
    ph, pl = swings["ph"].values, swings["pl"].values
    php, plp = swings["ph_pos"].values, swings["pl_pos"].values
    for t in range(len(swings)):
        new = []
        if not np.isnan(ph[t]):
            new.append((int(php[t]), "H", ph[t]))
        if not np.isnan(pl[t]):
            new.append((int(plp[t]), "L", pl[t]))
        if not new:
            continue
        for p in sorted(new):
            if piv and piv[-1][1] == p[1]:
                # same side twice: keep the more extreme one
                if (p[1] == "H" and p[2] >= piv[-1][2]) or (p[1] == "L" and p[2] <= piv[-1][2]):
                    piv[-1] = p
            else:
                piv.append(p)
        piv = piv[-12:]
        yield t, list(piv)


def _decay(events: np.ndarray, half_life: float = 6.0, horizon: int = 24) -> np.ndarray:
    """Spread sparse event values forward with geometric decay."""
    out = np.zeros_like(events)
    k = 0.5 ** (1.0 / half_life)
    cur, age = 0.0, horizon + 1
    for i, e in enumerate(events):
        if e != 0:
            cur, age = e, 0
        else:
            age += 1
            cur = cur * k if age <= horizon else 0.0
        out[i] = cur
    return out


# ---------------------------------------------------------------- harmonics

HARMONICS = {
    # name: (AB/XA, BC/AB, CD/BC, AD/XA) ranges
    "Gartley":   ((0.58, 0.66), (0.38, 0.89), (1.13, 1.618), (0.75, 0.82)),
    "Bat":       ((0.38, 0.52), (0.38, 0.89), (1.618, 2.618), (0.85, 0.92)),
    "Butterfly": ((0.75, 0.82), (0.38, 0.89), (1.618, 2.24), (1.20, 1.65)),
    "Crab":      ((0.38, 0.65), (0.38, 0.89), (2.24, 3.618), (1.55, 1.70)),
    "Cypher":    ((0.38, 0.62), (1.13, 1.414), (1.27, 2.0), (0.75, 0.82)),
}


def match_harmonic(X, A, B, C, D) -> str | None:
    XA, AB, BC, CD = abs(A - X), abs(B - A), abs(C - B), abs(D - C)
    if min(XA, AB, BC) <= 0:
        return None
    r = (AB / XA, BC / AB, CD / BC, abs(A - D) / XA)
    for name, rngs in HARMONICS.items():
        if all(lo <= v <= hi for v, (lo, hi) in zip(r, rngs)):
            return name
    return None


def harmonic_prz(X, A, B, C) -> dict:
    """Projected D completion zones for an XABC structure (live use)."""
    XA = A - X
    BC = C - B
    out = {}
    for name, (ab, bc, cd, ad) in HARMONICS.items():
        r_ab = abs(B - A) / abs(XA) if XA else 0
        r_bc = abs(BC) / abs(B - A) if B != A else 0
        if not (ab[0] <= r_ab <= ab[1] and bc[0] <= r_bc <= bc[1]):
            continue
        d1 = A - ad[0] * XA
        d2 = A - ad[1] * XA
        out[name] = (min(d1, d2), max(d1, d2))
    return out


# ----------------------------------------------------------- pivot patterns

def pivot_patterns(df: pd.DataFrame, swings: pd.DataFrame, osc: pd.Series, atr: pd.Series) -> pd.DataFrame:
    N = len(df)
    c = df["close"].values
    a = atr.bfill().values
    oscv = osc.values
    div_reg = np.zeros(N)
    div_hid = np.zeros(N)
    harm = np.zeros(N)
    harm_name = np.array([""] * N, dtype=object)
    ell = np.zeros(N)
    ell_label = np.array([""] * N, dtype=object)
    dbl = np.zeros(N)
    hs = np.zeros(N)
    pending: list = []  # (kind, neckline, sign, expiry)

    events = dict(_zigzag_events(swings))
    for t in range(N):
        piv = events.get(t)
        if piv is not None and len(piv) >= 2:
            last = piv[-1]
            same = [p for p in piv[:-1] if p[1] == last[1]]
            # ---- divergences (price vs oscillator at pivot bars)
            if same:
                prev = same[-1]
                po, lo_ = oscv[prev[0]], oscv[last[0]]
                if last[1] == "L":
                    if last[2] < prev[2] and lo_ > po + 2:
                        div_reg[t] = 1.0
                    elif last[2] > prev[2] and lo_ < po - 2:
                        div_hid[t] = 0.6
                else:
                    if last[2] > prev[2] and lo_ < po - 2:
                        div_reg[t] = -1.0
                    elif last[2] < prev[2] and lo_ > po + 2:
                        div_hid[t] = -0.6
            # ---- double top / bottom (arm with neckline, trigger on break)
            if len(piv) >= 3:
                p1, mid, p2 = piv[-3], piv[-2], piv[-1]
                tol = 0.35 * a[t]
                if p1[1] == p2[1] == "L" and abs(p1[2] - p2[2]) <= tol:
                    pending.append(("dbl", mid[2], 1.0, t + 30))
                if p1[1] == p2[1] == "H" and abs(p1[2] - p2[2]) <= tol:
                    pending.append(("dbl", mid[2], -1.0, t + 30))
            # ---- head & shoulders
            if len(piv) >= 5:
                s1, n1, hd, n2, s2 = piv[-5:]
                if s1[1] == hd[1] == s2[1] == "H" and hd[2] > max(s1[2], s2[2]) \
                        and abs(s1[2] - s2[2]) <= 0.6 * a[t] and hd[2] - max(s1[2], s2[2]) > 0.5 * a[t]:
                    pending.append(("hs", min(n1[2], n2[2]), -1.0, t + 30))
                if s1[1] == hd[1] == s2[1] == "L" and hd[2] < min(s1[2], s2[2]) \
                        and abs(s1[2] - s2[2]) <= 0.6 * a[t] and min(s1[2], s2[2]) - hd[2] > 0.5 * a[t]:
                    pending.append(("hs", max(n1[2], n2[2]), 1.0, t + 30))
            # ---- harmonics: D is the newest pivot
            if len(piv) >= 5:
                X, A, B, C, D = (p[2] for p in piv[-5:])
                name = match_harmonic(X, A, B, C, D)
                if name:
                    harm[t] = 1.0 if piv[-1][1] == "L" else -1.0
                    harm_name[t] = name
            # ---- Elliott impulse rules
            if len(piv) >= 6:
                p = [x[2] for x in piv[-6:]]
                kinds = "".join(x[1] for x in piv[-6:])
                w1, w3, w5 = abs(p[1] - p[0]), abs(p[3] - p[2]), abs(p[5] - p[4])
                if kinds == "LHLHLH" and p[2] > p[0] and p[4] > p[1] and p[5] > p[3] \
                        and w3 > min(w1, w5):
                    ell[t], ell_label[t] = -0.6, "impulse-up complete (5) -> ABC correction risk"
                elif kinds == "HLHLHL" and p[2] < p[0] and p[4] < p[1] and p[5] < p[3] \
                        and w3 > min(w1, w5):
                    ell[t], ell_label[t] = 0.6, "impulse-down complete (5) -> ABC rebound risk"
            if len(piv) >= 5 and ell[t] == 0:
                p = [x[2] for x in piv[-5:]]
                kinds = "".join(x[1] for x in piv[-5:])
                w1, w3 = abs(p[1] - p[0]), abs(p[3] - p[2])
                if kinds == "LHLHL" and p[2] > p[0] and p[4] > p[1] and w3 > w1:
                    ell[t], ell_label[t] = 0.7, "wave-4 low in place -> wave 5 up"
                elif kinds == "HLHLH" and p[2] < p[0] and p[4] < p[1] and w3 > w1:
                    ell[t], ell_label[t] = -0.7, "wave-4 high in place -> wave 5 down"
        # ---- trigger armed neckline patterns
        keep = []
        for kind, neck, sign, exp in pending:
            if t > exp:
                continue
            if (sign > 0 and c[t] > neck) or (sign < 0 and c[t] < neck):
                if kind == "dbl":
                    dbl[t] = sign
                else:
                    hs[t] = sign
                continue
            keep.append((kind, neck, sign, exp))
        pending = keep[-6:]

    out = pd.DataFrame(index=df.index)
    out["div_reg"] = _decay(div_reg, 4, 12)
    out["div_hid"] = _decay(div_hid, 4, 12)
    out["harmonic"] = _decay(harm, 4, 12)
    out["harmonic_name"] = harm_name
    out["elliott"] = _decay(ell, 8, 30)
    out["elliott_label"] = ell_label
    out["double_tb"] = _decay(dbl, 6, 24)
    out["head_shoulders"] = _decay(hs, 6, 24)
    return out


def describe_geometry(swings_list: list, atr_val: float) -> list[str]:
    """Human description of compression patterns from the last pivots (live)."""
    notes = []
    highs = [s for s in swings_list if s[1] == "H"][-3:]
    lows = [s for s in swings_list if s[1] == "L"][-3:]
    if len(highs) == 3 and len(lows) == 3:
        hs = np.polyfit([h[0] for h in highs], [h[2] for h in highs], 1)[0]
        ls = np.polyfit([lw[0] for lw in lows], [lw[2] for lw in lows], 1)[0]
        tol = 0.02 * atr_val
        if hs < -tol and ls > tol:
            notes.append("symmetrical triangle (compression) - expect expansion")
        elif abs(hs) <= tol and ls > tol:
            notes.append("ascending triangle - bullish bias on break of flat top")
        elif hs < -tol and abs(ls) <= tol:
            notes.append("descending triangle - bearish bias on break of flat bottom")
        elif hs > tol and ls > tol and ls > hs:
            notes.append("rising wedge - bearish reversal risk")
        elif hs < -tol and ls < -tol and hs < ls:
            notes.append("falling wedge - bullish reversal risk")
        elif hs > tol and ls > tol:
            notes.append("ascending channel")
        elif hs < -tol and ls < -tol:
            notes.append("descending channel")
    return notes
