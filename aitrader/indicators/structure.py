"""Price-action / Smart-Money-Concepts structure engine.

Implements, causally (a swing is only known ``right`` bars after it printed):
  * fractal swing highs/lows
  * market structure: HH/HL/LH/LL, Break of Structure (BOS), Change of Character (CHoCH)
  * order blocks (last opposing candle before the displacement that broke structure)
  * fair value gaps (3-candle imbalances) with mitigation tracking
  * equal highs/lows (resting liquidity) and liquidity sweeps
  * premium/discount of the dealing range and Fibonacci/OTE position of the last leg
  * clustered support/resistance levels
  * session levels (Asian range, previous day/week high/low) and prior-day value area
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- swings

def swing_points(df: pd.DataFrame, left: int = 5, right: int = 5) -> pd.DataFrame:
    """Columns (indexed at *confirmation* bar):
    ph / pl  : level of the swing high/low confirmed on this bar (NaN otherwise)
    ph_pos / pl_pos : integer position of the swing bar itself
    """
    n = left + right + 1
    hi, lo = df["high"], df["low"]
    is_ph = hi.eq(hi.rolling(n).max().shift(-right)) & hi.notna()
    is_pl = lo.eq(lo.rolling(n).min().shift(-right)) & lo.notna()
    # avoid flat duplicates: keep first in window
    is_ph &= ~is_ph.shift(1, fill_value=False) | hi.ne(hi.shift(1))
    is_pl &= ~is_pl.shift(1, fill_value=False) | lo.ne(lo.shift(1))
    pos = np.arange(len(df))
    out = pd.DataFrame(index=df.index)
    out["ph"] = hi.where(is_ph).shift(right)
    out["pl"] = lo.where(is_pl).shift(right)
    out["ph_pos"] = pd.Series(np.where(is_ph, pos, np.nan), index=df.index).shift(right)
    out["pl_pos"] = pd.Series(np.where(is_pl, pos, np.nan), index=df.index).shift(right)
    return out


@dataclass
class Zone:
    kind: str          # 'bull_ob', 'bear_ob', 'bull_fvg', 'bear_fvg'
    top: float
    bottom: float
    start: int         # bar position where zone originates
    created: int       # bar position when it became known
    strength: float = 1.0
    touched: int = 0
    active: bool = True

    def as_dict(self, index) -> dict:
        return {"kind": self.kind, "top": self.top, "bottom": self.bottom,
                "start": index[self.start], "created": index[self.created],
                "strength": round(self.strength, 2), "touched": self.touched}


@dataclass
class StructureState:
    zones: list = field(default_factory=list)
    swings: list = field(default_factory=list)   # (pos, kind, level, label)
    events: list = field(default_factory=list)   # (pos, kind, level)
    eqh: float = np.nan
    eql: float = np.nan


def market_structure(df: pd.DataFrame, atr: pd.Series, left: int = 5, right: int = 5,
                     fvg_min_atr: float = 0.15, max_zones: int = 12):
    """Bar-by-bar structure engine. Returns (features DataFrame, final StructureState)."""
    sw = swing_points(df, left, right)
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    a = atr.bfill().values
    N = len(df)
    ph, pl = sw["ph"].values, sw["pl"].values
    php, plp = sw["ph_pos"].values, sw["pl_pos"].values

    cols = {k: np.full(N, np.nan) for k in (
        "ms_trend", "last_sh", "last_sl", "prev_sh", "prev_sl", "bos_up", "bos_dn",
        "choch_up", "choch_dn", "in_bull_ob", "in_bear_ob", "dist_bull_ob", "dist_bear_ob",
        "in_bull_fvg", "in_bear_fvg", "dist_bull_fvg", "dist_bear_fvg", "eqh", "eql",
        "sh_label", "sl_label", "range_pos", "leg_retr", "leg_dir", "ob_strength")}

    st = StructureState()
    trend = 0
    last_sh = last_sl = prev_sh = prev_sl = np.nan
    last_sh_pos = last_sl_pos = -1
    sh_broken = sl_broken = True
    sh_label = sl_label = 0.0
    recent_highs: list = []
    recent_lows: list = []
    last_kind = None

    for t in range(N):
        # 1) newly confirmed swings
        if not np.isnan(ph[t]):
            prev_sh, last_sh, last_sh_pos, sh_broken = last_sh, ph[t], int(php[t]), False
            sh_label = 1.0 if (np.isnan(prev_sh) or last_sh > prev_sh) else -1.0  # HH / LH
            st.swings.append((last_sh_pos, "H", last_sh, "HH" if sh_label > 0 else "LH"))
            recent_highs = (recent_highs + [last_sh])[-3:]
            last_kind = "H"
        if not np.isnan(pl[t]):
            prev_sl, last_sl, last_sl_pos, sl_broken = last_sl, pl[t], int(plp[t]), False
            sl_label = 1.0 if (np.isnan(prev_sl) or last_sl > prev_sl) else -1.0  # HL / LL
            st.swings.append((last_sl_pos, "L", last_sl, "HL" if sl_label > 0 else "LL"))
            recent_lows = (recent_lows + [last_sl])[-3:]
            last_kind = "L"

        # equal highs / lows = resting liquidity
        tol = 0.15 * a[t]
        if len(recent_highs) >= 2 and np.isnan(st.eqh):
            rh = recent_highs
            for i in range(len(rh) - 1):
                if abs(rh[i] - rh[-1]) <= tol and max(rh[i], rh[-1]) >= h[t]:
                    st.eqh = max(rh[i], rh[-1])
        if len(recent_lows) >= 2 and np.isnan(st.eql):
            rl = recent_lows
            for i in range(len(rl) - 1):
                if abs(rl[i] - rl[-1]) <= tol and min(rl[i], rl[-1]) <= l[t]:
                    st.eql = min(rl[i], rl[-1])
        if not np.isnan(st.eqh) and h[t] > st.eqh:
            st.events.append((t, "eqh_swept", st.eqh))
            st.eqh = np.nan
            recent_highs = []
        if not np.isnan(st.eql) and l[t] < st.eql:
            st.events.append((t, "eql_swept", st.eql))
            st.eql = np.nan
            recent_lows = []

        # 2) structure breaks on close
        if not sh_broken and not np.isnan(last_sh) and c[t] > last_sh:
            kind = "bos_up" if trend == 1 else "choch_up"
            cols[kind][t] = 1.0
            st.events.append((t, kind, last_sh))
            trend, sh_broken = 1, True
            lo_i = max(0, (last_sl_pos if last_sl_pos >= 0 else t - 10) - 2)
            seg = range(lo_i, t)
            bears = [i for i in seg if c[i] < o[i]]
            ob_i = min(bears, key=lambda i: l[i]) if bears else (last_sl_pos if last_sl_pos >= 0 else t - 1)
            disp = (c[t] - l[ob_i]) / a[t] if a[t] > 0 else 1.0
            st.zones.append(Zone("bull_ob", h[ob_i], l[ob_i], ob_i, t, strength=min(disp / 3, 2.0)))
        if not sl_broken and not np.isnan(last_sl) and c[t] < last_sl:
            kind = "bos_dn" if trend == -1 else "choch_dn"
            cols[kind][t] = 1.0
            st.events.append((t, kind, last_sl))
            trend, sl_broken = -1, True
            lo_i = max(0, (last_sh_pos if last_sh_pos >= 0 else t - 10) - 2)
            seg = range(lo_i, t)
            bulls = [i for i in seg if c[i] > o[i]]
            ob_i = max(bulls, key=lambda i: h[i]) if bulls else (last_sh_pos if last_sh_pos >= 0 else t - 1)
            disp = (h[ob_i] - c[t]) / a[t] if a[t] > 0 else 1.0
            st.zones.append(Zone("bear_ob", h[ob_i], l[ob_i], ob_i, t, strength=min(disp / 3, 2.0)))

        # 3) fair value gaps (known at close of candle t)
        if t >= 2:
            if l[t] > h[t - 2] and (l[t] - h[t - 2]) >= fvg_min_atr * a[t]:
                st.zones.append(Zone("bull_fvg", l[t], h[t - 2], t - 1, t,
                                     strength=min((l[t] - h[t - 2]) / a[t], 2.0)))
            if h[t] < l[t - 2] and (l[t - 2] - h[t]) >= fvg_min_atr * a[t]:
                st.zones.append(Zone("bear_fvg", l[t - 2], h[t], t - 1, t,
                                     strength=min((l[t - 2] - h[t]) / a[t], 2.0)))

        # 4) zone interaction / mitigation (zones created this bar are skipped)
        in_bob = in_sob = in_bfvg = in_sfvg = 0.0
        d_bob = d_sob = d_bfvg = d_sfvg = np.nan
        ob_str = 0.0
        alive = []
        for z in st.zones:
            if not z.active:
                continue
            if z.created < t:
                if z.kind.startswith("bull"):
                    if c[t] < z.bottom:                       # invalidated
                        z.active = False
                        continue
                    if z.kind == "bull_fvg" and l[t] <= z.bottom:   # fully filled
                        z.active = False
                        continue
                    if l[t] <= z.top:
                        z.touched += 1
                        if z.kind == "bull_ob":
                            in_bob, ob_str = 1.0, max(ob_str, z.strength)
                        else:
                            in_bfvg = 1.0
                    dist = (c[t] - z.top) / a[t]
                    if z.kind == "bull_ob":
                        d_bob = dist if np.isnan(d_bob) else min(d_bob, dist)
                    else:
                        d_bfvg = dist if np.isnan(d_bfvg) else min(d_bfvg, dist)
                else:
                    if c[t] > z.top:
                        z.active = False
                        continue
                    if z.kind == "bear_fvg" and h[t] >= z.top:
                        z.active = False
                        continue
                    if h[t] >= z.bottom:
                        z.touched += 1
                        if z.kind == "bear_ob":
                            in_sob, ob_str = 1.0, max(ob_str, z.strength)
                        else:
                            in_sfvg = 1.0
                    dist = (z.bottom - c[t]) / a[t]
                    if z.kind == "bear_ob":
                        d_sob = dist if np.isnan(d_sob) else min(d_sob, dist)
                    else:
                        d_sfvg = dist if np.isnan(d_sfvg) else min(d_sfvg, dist)
                # stale zones: touched many times lose meaning
                if z.touched > 6 or (t - z.created) > 500:
                    z.active = False
                    continue
            alive.append(z)
        st.zones = alive[-max_zones * 4:]

        # 5) dealing range / leg position
        rp = lr = ld = np.nan
        if not np.isnan(last_sh) and not np.isnan(last_sl) and last_sh > last_sl:
            rng = last_sh - last_sl
            rp = (c[t] - last_sl) / rng
            if last_kind == "H":      # most recent leg was up: measure pullback from high
                lr, ld = (last_sh - c[t]) / rng, 1.0
            else:
                lr, ld = (c[t] - last_sl) / rng, -1.0

        cols["ms_trend"][t] = trend
        cols["last_sh"][t], cols["last_sl"][t] = last_sh, last_sl
        cols["prev_sh"][t], cols["prev_sl"][t] = prev_sh, prev_sl
        cols["in_bull_ob"][t], cols["in_bear_ob"][t] = in_bob, in_sob
        cols["dist_bull_ob"][t], cols["dist_bear_ob"][t] = d_bob, d_sob
        cols["in_bull_fvg"][t], cols["in_bear_fvg"][t] = in_bfvg, in_sfvg
        cols["dist_bull_fvg"][t], cols["dist_bear_fvg"][t] = d_bfvg, d_sfvg
        cols["eqh"][t], cols["eql"][t] = st.eqh, st.eql
        cols["sh_label"][t], cols["sl_label"][t] = sh_label, sl_label
        cols["range_pos"][t], cols["leg_retr"][t], cols["leg_dir"][t] = rp, lr, ld
        cols["ob_strength"][t] = ob_str

    out = pd.DataFrame(cols, index=df.index)
    for k in ("bos_up", "bos_dn", "choch_up", "choch_dn"):
        out[k] = out[k].fillna(0.0)
    out = pd.concat([out, sw], axis=1)
    return out, st


# --------------------------------------------------------- liquidity sweeps

def sweeps(df: pd.DataFrame, lows: dict[str, pd.Series], highs: dict[str, pd.Series]) -> pd.DataFrame:
    """Wick through a prior level that closes back inside = stop hunt.

    Levels must already be known *before* the bar. Positive value = bullish
    sweep (sell-side liquidity taken below a low), negative = bearish sweep
    (buy-side liquidity taken above a high).
    """
    out = pd.DataFrame(index=df.index)
    for name, lvl in lows.items():
        out[f"sweep_{name}"] = ((df["low"] < lvl) & (df["close"] > lvl)).astype(float)
    for name, lvl in highs.items():
        out[f"sweep_{name}"] = -((df["high"] > lvl) & (df["close"] < lvl)).astype(float)
    return out


# --------------------------------------------------------- support/resistance

def cluster_levels(prices: np.ndarray, weights: np.ndarray, tol: float) -> list[tuple[float, float]]:
    """1-D agglomerative clustering. Returns [(level, strength)] sorted by level."""
    if len(prices) == 0:
        return []
    order = np.argsort(prices)
    p, w = prices[order], weights[order]
    levels, cur_p, cur_w = [], [p[0]], [w[0]]
    for x, wx in zip(p[1:], w[1:]):
        if x - np.average(cur_p, weights=cur_w) <= tol:
            cur_p.append(x)
            cur_w.append(wx)
        else:
            levels.append((float(np.average(cur_p, weights=cur_w)), float(np.sum(cur_w))))
            cur_p, cur_w = [x], [wx]
    levels.append((float(np.average(cur_p, weights=cur_w)), float(np.sum(cur_w))))
    return levels


def sr_levels_at(df: pd.DataFrame, swings: pd.DataFrame, t: int, atr_val: float,
                 lookback: int = 720) -> list[tuple[float, float]]:
    lo = max(0, t - lookback)
    seg = swings.iloc[lo:t + 1]
    pts = pd.concat([seg["ph"].dropna(), seg["pl"].dropna()])
    if pts.empty or not np.isfinite(atr_val):
        return []
    age = (t - np.array([swings.index.get_loc(i) for i in pts.index])) / lookback
    weights = 1.0 + (1.0 - age)          # recent touches count more
    return cluster_levels(pts.values, weights, tol=0.4 * atr_val)


def sr_features(df: pd.DataFrame, swings: pd.DataFrame, atr: pd.Series,
                every: int = 12, lookback: int = 720) -> pd.DataFrame:
    N = len(df)
    c = df["close"].values
    a = atr.values
    out = {k: np.full(N, np.nan) for k in ("sup", "res", "sup_str", "res_str")}
    levels: list = []
    for t in range(N):
        if t % every == 0:
            levels = sr_levels_at(df, swings, t, a[t], lookback)
        if not levels:
            continue
        below = [(lv, s) for lv, s in levels if lv < c[t]]
        above = [(lv, s) for lv, s in levels if lv >= c[t]]
        if below:
            lv, s = max(below, key=lambda x: x[0])
            out["sup"][t], out["sup_str"][t] = lv, s
        if above:
            lv, s = min(above, key=lambda x: x[0])
            out["res"][t], out["res_str"][t] = lv, s
    res = pd.DataFrame(out, index=df.index)
    res["dist_sup"] = (df["close"] - res["sup"]) / atr
    res["dist_res"] = (res["res"] - df["close"]) / atr
    return res


# ------------------------------------------------------------ session levels

def session_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Previous day/week extremes, day open and the Asian range (00-07 UTC).

    Asian range values only appear once the Asian session has closed.
    """
    out = pd.DataFrame(index=df.index)
    day = df.index.floor("D")
    daily = df.groupby(day).agg(high=("high", "max"), low=("low", "min"),
                                close=("close", "last"), open=("open", "first"))
    prev = daily.shift(1)
    out["pdh"] = prev["high"].reindex(day).values
    out["pdl"] = prev["low"].reindex(day).values
    out["pdc"] = prev["close"].reindex(day).values
    out["day_open"] = daily["open"].reindex(day).values
    wk = df.index.to_period("W-SUN").start_time.tz_localize("UTC") if df.index.tz is not None \
        else df.index.to_period("W-SUN").start_time
    weekly = df.groupby(wk).agg(high=("high", "max"), low=("low", "min"), open=("open", "first"))
    pw = weekly.shift(1)
    out["pwh"] = pw["high"].reindex(wk).values
    out["pwl"] = pw["low"].reindex(wk).values
    out["week_open"] = weekly["open"].reindex(wk).values

    hour = df.index.hour
    asia = df[(hour >= 0) & (hour < 7)]
    ar = asia.groupby(asia.index.floor("D")).agg(high=("high", "max"), low=("low", "min"))
    ah = ar["high"].reindex(day).values
    al = ar["low"].reindex(day).values
    after = hour >= 7
    out["asia_high"] = np.where(after, ah, np.nan)
    out["asia_low"] = np.where(after, al, np.nan)

    # classic floor pivots from previous day
    p = (out["pdh"] + out["pdl"] + out["pdc"]) / 3
    out["pivot"] = p
    out["r1"] = 2 * p - out["pdl"]
    out["s1"] = 2 * p - out["pdh"]
    out["r2"] = p + (out["pdh"] - out["pdl"])
    out["s2"] = p - (out["pdh"] - out["pdl"])

    out["hour"] = hour
    out["dow"] = df.index.dayofweek
    out["sess_asia"] = ((hour >= 0) & (hour < 7)).astype(float)
    out["sess_london"] = ((hour >= 7) & (hour < 16)).astype(float)
    out["sess_ny"] = ((hour >= 12) & (hour < 21)).astype(float)
    out["killzone"] = (((hour >= 7) & (hour < 10)) | ((hour >= 12) & (hour < 15))).astype(float)
    return out


# --------------------------------------------------------- value area profile

def _profile(bars: pd.DataFrame, use_volume: bool, bins: int = 48):
    lo, hi = bars["low"].min(), bars["high"].max()
    if not np.isfinite(lo) or hi <= lo:
        return np.nan, np.nan, np.nan
    edges = np.linspace(lo, hi, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    hist = np.zeros(bins)
    vols = bars["volume"].values if use_volume else np.ones(len(bars))
    for bl, bh, v in zip(bars["low"].values, bars["high"].values, vols):
        m = (mids >= bl) & (mids <= bh)
        k = m.sum()
        if k:
            hist[m] += v / k
    if hist.sum() <= 0:
        return np.nan, np.nan, np.nan
    poc_i = int(hist.argmax())
    total, acc = hist.sum(), hist[poc_i]
    lo_i = hi_i = poc_i
    while acc < 0.7 * total and (lo_i > 0 or hi_i < bins - 1):
        up = hist[hi_i + 1] if hi_i < bins - 1 else -1
        dn = hist[lo_i - 1] if lo_i > 0 else -1
        if up >= dn:
            hi_i += 1
            acc += up
        else:
            lo_i -= 1
            acc += dn
    return mids[poc_i], mids[hi_i], mids[lo_i]


def value_area(df: pd.DataFrame, use_volume: bool) -> pd.DataFrame:
    """Prior UTC-day volume profile (or TPO profile when no real volume)."""
    day = df.index.floor("D")
    rows = {}
    for d, g in df.groupby(day):
        rows[d] = _profile(g, use_volume)
    prof = pd.DataFrame.from_dict(rows, orient="index", columns=["poc", "vah", "val"]).shift(1)
    out = prof.reindex(day)
    out.index = df.index
    return out
