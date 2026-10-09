"""Causal feature matrix: H1 primary timeframe + H4/D1 context + intermarket,
macro, positioning, sessions and seasonality.

Higher-timeframe bars are attached only after they have *closed* (merge on the
completion time), so the backtest never sees an unfinished H4 or daily candle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Instrument
from .data.loader import resample_ohlc
from .indicators import core as ta
from .indicators.patterns import candlesticks, pivot_patterns
from .indicators.structure import (market_structure, session_levels, sr_features,
                                   sweeps, value_area)


def _htf_context(df_h1: pd.DataFrame, rule: str, prefix: str, bar: pd.Timedelta) -> pd.DataFrame:
    h = resample_ohlc(df_h1, rule)
    ctx = pd.DataFrame(index=h.index)
    c = h["close"]
    a = ta.atr(h, 14)
    e20, e50 = ta.ema(c, 20), ta.ema(c, 50)
    ctx["ema_fast_slope"] = (e20 - e20.shift(3)) / a
    ctx["ema_stack"] = np.sign(e20 - e50)
    ctx["px_vs_ema50"] = (c - e50) / a
    _, st_dir = ta.supertrend(h, 10, 3.0)
    ctx["st_dir"] = st_dir
    ctx["rsi"] = ta.rsi(c, 14)
    _, _, hist = ta.macd(c)
    ctx["macd_hist"] = hist / a
    adx, pdi, mdi = ta.adx(h, 14)
    ctx["adx"] = adx
    ctx["di_diff"] = (pdi - mdi) / 100
    ms, _ = market_structure(h, a, left=3, right=3)
    ctx["ms_trend"] = ms["ms_trend"]
    ctx["range_pos"] = ms["range_pos"]
    ctx["atr"] = a
    # attach on completion time, aligned to H1 bar close time
    ctx.index = ctx.index + bar
    ctx = ctx.add_prefix(prefix)
    ctx.index = ctx.index.as_unit("ns")
    left = pd.DataFrame({"t_close": (df_h1.index + pd.Timedelta("1h")).as_unit("ns")}, index=df_h1.index)
    merged = pd.merge_asof(left.sort_values("t_close"), ctx.sort_index(),
                           left_on="t_close", right_index=True, direction="backward")
    merged.index = df_h1.index
    return merged.drop(columns="t_close")


def _seasonality(df: pd.DataFrame, window: int = 260, ahead: int = 6) -> pd.DataFrame:
    """t-stat of mean log return for the next ``ahead`` hours, estimated from
    each hour's previous ``window`` occurrences (strictly before now)."""
    r = np.log(df["close"]).diff()
    hour = df.index.hour
    g = r.groupby(hour)
    mean = g.transform(lambda x: x.shift(1).rolling(window, min_periods=60).mean())
    var = g.transform(lambda x: x.shift(1).rolling(window, min_periods=60).var())
    cnt = g.transform(lambda x: x.shift(1).rolling(window, min_periods=60).count())
    m_sum = sum(mean.shift(-k) for k in range(1, ahead + 1))
    v_sum = sum((var / cnt).shift(-k) for k in range(1, ahead + 1))
    out = pd.DataFrame(index=df.index)
    out["seas_t"] = (m_sum / np.sqrt(v_sum)).clip(-6, 6)
    # day-of-week drift of the full next day (weak, low weight)
    dow = df.index.dayofweek
    dr = r.groupby([df.index.floor("D")]).sum()
    dmean = dr.groupby(dr.index.dayofweek).transform(lambda x: x.shift(1).rolling(104, min_periods=30).mean())
    dstd = dr.groupby(dr.index.dayofweek).transform(lambda x: x.shift(1).rolling(104, min_periods=30).std())
    dcnt = dr.groupby(dr.index.dayofweek).transform(lambda x: x.shift(1).rolling(104, min_periods=30).count())
    dt = (dmean / (dstd / np.sqrt(dcnt))).reindex(df.index.floor("D")).values
    out["dow_t"] = np.clip(np.nan_to_num(dt), -4, 4)
    _ = dow
    # the last ``ahead`` rows look up hours that have not printed yet; repeat
    # yesterday's estimate for those hours (same hour, still causal)
    tail = out["seas_t"].isna() & out["seas_t"].shift(24).notna()
    out.loc[tail, "seas_t"] = out["seas_t"].shift(24)[tail]
    return out


def _smt(df: pd.DataFrame, peer: pd.DataFrame | None, inverse: bool, n: int = 20) -> pd.Series:
    """Smart-Money-Technique divergence with a correlated market.

    +1: instrument sweeps a fresh n-bar low but the peer refuses to confirm.
    -1: instrument prints a fresh n-bar high but the peer does not.
    """
    if peer is None or peer.empty:
        return pd.Series(0.0, index=df.index)
    p = peer.reindex(df.index).ffill(limit=3)
    if inverse:
        p_hi, p_lo = -p["low"], -p["high"]
    else:
        p_hi, p_lo = p["high"], p["low"]
    new_low = df["low"] < df["low"].shift(1).rolling(n).min()
    new_high = df["high"] > df["high"].shift(1).rolling(n).max()
    peer_new_low = p_lo < p_lo.shift(1).rolling(n).min()
    peer_new_high = p_hi > p_hi.shift(1).rolling(n).max()
    sig = np.where(new_low & ~peer_new_low & p_lo.notna(), 1.0,
                   np.where(new_high & ~peer_new_high & p_hi.notna(), -1.0, 0.0))
    from .indicators.patterns import _decay
    return pd.Series(_decay(sig, 3, 8), index=df.index)


def _macro(df: pd.DataFrame, macro: pd.DataFrame | None, inst: Instrument) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if macro is None or macro.empty:
        return out
    m = macro.copy()
    feats = pd.DataFrame(index=m.index)

    def chg(col, n):
        return m[col].diff(n) if col in m else None

    def ret(col, n):
        return np.log(m[col]).diff(n) if col in m else None

    for n in (5, 20):
        for col in ("dxy", "silver", "spx", "eurusd", "gold"):
            r = ret(col, n)
            if r is not None:
                feats[f"{col}_ret{n}"] = r / (np.log(m[col]).diff().rolling(250, min_periods=60).std() * np.sqrt(n))
        for col in ("real10y", "us2y", "us10y", "breakeven10y"):
            d = chg(col, n)
            if d is not None:
                feats[f"{col}_chg{n}"] = d / (m[col].diff().rolling(250, min_periods=60).std() * np.sqrt(n))
    if "vix" in m:
        feats["vix_z"] = ta.zscore(m["vix"], 120)
        feats["vix_chg5"] = np.log(m["vix"]).diff(5)
    if "gold" in m and "silver" in m:
        feats["gsr_z"] = ta.zscore(m["gold"] / m["silver"], 250)
    # known only after the daily close -> use from next day
    feats.index = feats.index + pd.Timedelta(days=1)
    feats = feats.reindex(feats.index.union(df.index.floor("D").unique())).sort_index().ffill()
    out = feats.reindex(df.index.floor("D"))
    out.index = df.index
    return out.add_prefix("mac_")


def _cot(df: pd.DataFrame, cot: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if cot is None or cot.empty:
        return out
    left = pd.DataFrame({"t": df.index.as_unit("ns")}, index=df.index)
    merged = pd.merge_asof(left, cot[["cot_index", "spec_net_chg4"]].sort_index(),
                           left_on="t", right_index=True, direction="backward")
    merged.index = df.index
    return merged.drop(columns="t").add_prefix("cot_")


def build_features(df: pd.DataFrame, inst: Instrument, peer: pd.DataFrame | None = None,
                   macro: pd.DataFrame | None = None, cot: pd.DataFrame | None = None,
                   return_state: bool = False):
    f = pd.DataFrame(index=df.index)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    a = ta.atr(df, 14)
    f["atr"] = a
    f["close"] = c
    f["open"], f["high"], f["low"] = o, h, l
    f["ret1"] = np.log(c).diff()

    # ---- trend
    for n in (20, 50, 100, 200):
        f[f"ema{n}"] = ta.ema(c, n)
    f["ema_stack"] = (np.sign(f["ema20"] - f["ema50"]) + np.sign(f["ema50"] - f["ema200"])) / 2
    f["px_vs_ema20"] = (c - f["ema20"]) / a
    f["px_vs_ema200"] = (c - f["ema200"]) / a
    f["ema20_slope"] = (f["ema20"] - f["ema20"].shift(5)) / a
    f["st_line"], f["st_dir"] = ta.supertrend(df, 10, 3.0)
    tk, kj, sa, sb = ta.ichimoku(df)
    f["tenkan"], f["kijun"], f["span_a"], f["span_b"] = tk, kj, sa, sb
    cloud_top, cloud_bot = np.maximum(sa, sb), np.minimum(sa, sb)
    f["ichi_pos"] = np.where(c > cloud_top, 1.0, np.where(c < cloud_bot, -1.0, 0.0))
    f["ichi_tk"] = np.sign(tk - kj)
    f["psar_dir"] = np.sign(c - ta.psar(df))
    f["adx"], f["pdi"], f["mdi"] = ta.adx(df, 14)
    f["di_diff"] = (f["pdi"] - f["mdi"]) / 100
    f["kama"] = ta.kama(c)
    f["kama_slope"] = (f["kama"] - f["kama"].shift(5)) / a
    f["lr_slope"] = ta.linreg_slope(c, 24) / a
    f["tsmom_120"] = np.log(c).diff(120) / (f["ret1"].rolling(500, min_periods=100).std() * np.sqrt(120))
    f["tsmom_480"] = np.log(c).diff(480) / (f["ret1"].rolling(500, min_periods=100).std() * np.sqrt(480))

    # ---- momentum / oscillators
    f["rsi"] = ta.rsi(c, 14)
    f["rsi3"] = ta.rsi(c, 3)
    m, s, hist = ta.macd(c)
    f["macd"], f["macd_sig"], f["macd_hist"] = m / a, s / a, hist / a
    f["macd_hist_slope"] = f["macd_hist"].diff(2)
    f["stoch_k"], f["stoch_d"] = ta.stochastic(df)
    f["willr"] = ta.williams_r(df)
    f["cci"] = ta.cci(df)
    f["roc10"] = np.log(c).diff(10) / (f["ret1"].rolling(200, min_periods=50).std() * np.sqrt(10))

    # ---- volatility / mean reversion
    bm, bu, bl = ta.bollinger(c, 20, 2.0)
    f["bb_mid"], f["bb_up"], f["bb_lo"] = bm, bu, bl
    f["bb_z"] = (c - bm) / ((bu - bm) / 2).replace(0, np.nan)
    f["bb_width"] = (bu - bl) / bm
    km, ku, kl = ta.keltner(df, 20, 1.5)
    f["squeeze"] = ((bu < ku) & (bl > kl)).astype(float)
    f["squeeze_release"] = ((f["squeeze"].shift(1) == 1) & (f["squeeze"] == 0)).astype(float)
    dh, dl = ta.donchian(df, 20)
    f["don_hi"], f["don_lo"] = dh.shift(1), dl.shift(1)
    f["don_break"] = np.where(c > f["don_hi"], 1.0, np.where(c < f["don_lo"], -1.0, 0.0))
    f["atr_pct"] = ta.pct_rank(a / c, 500)
    f["bbw_pct"] = ta.pct_rank(f["bb_width"], 500)
    f["er"] = ta.efficiency_ratio(c, 20)
    f["chop"] = ta.choppiness(df, 14)
    f["hurst"] = ta.hurst(c, 128)
    f["z50"] = ta.zscore(c, 50)
    f["vwap"] = ta.session_vwap(df)
    f["px_vs_vwap"] = (c - f["vwap"]) / a

    # ---- volume (only meaningful for exchange-traded gold)
    if inst.has_volume and df["volume"].sum() > 0:
        v = df["volume"].replace(0, np.nan)
        f["rel_vol"] = v / v.groupby(df.index.hour).transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
        f["mfi"] = ta.mfi(df)
        ob = ta.obv(df)
        f["obv_slope"] = (ob - ob.shift(20)) / v.rolling(100, min_periods=20).mean() / 20
    else:
        f["rel_vol"] = np.nan
        f["mfi"] = np.nan
        f["obv_slope"] = np.nan

    # ---- structure / SMC
    ms, state = market_structure(df, a, left=5, right=5)
    f = pd.concat([f, ms], axis=1)
    for k in ("bos_up", "bos_dn", "choch_up", "choch_dn"):
        from .indicators.patterns import _decay
        f[f"{k}_d"] = _decay(f[k].values, 8, 30)
    sess = session_levels(df)
    f = pd.concat([f, sess], axis=1)
    sw = sweeps(df,
                lows={"swing_low": f["last_sl"].shift(1), "eql": f["eql"].shift(1),
                      "pdl": f["pdl"], "asia_low": f["asia_low"].shift(1), "pwl": f["pwl"]},
                highs={"swing_high": f["last_sh"].shift(1), "eqh": f["eqh"].shift(1),
                       "pdh": f["pdh"], "asia_high": f["asia_high"].shift(1), "pwh": f["pwh"]})
    f = pd.concat([f, sw], axis=1)
    sweep_w = {"swing_low": 0.6, "eql": 1.0, "pdl": 0.8, "asia_low": 0.6, "pwl": 1.0,
               "swing_high": 0.6, "eqh": 1.0, "pdh": 0.8, "asia_high": 0.6, "pwh": 1.0}
    raw = sum(sw[f"sweep_{k}"] * w for k, w in sweep_w.items()).clip(-1.5, 1.5) / 1.5
    from .indicators.patterns import _decay
    f["sweep_score"] = _decay(raw.values, 3, 10)
    srf = sr_features(df, ms[["ph", "pl"]], a)
    f = pd.concat([f, srf], axis=1)
    va = value_area(df, use_volume=inst.has_volume and df["volume"].sum() > 0)
    f = pd.concat([f, va], axis=1)
    f["va_pos"] = np.where(c > f["vah"], 1.0, np.where(c < f["val"], -1.0, 0.0))
    f["dist_poc"] = (c - f["poc"]) / a
    for k in ("pdh", "pdl", "asia_high", "asia_low", "pivot", "r1", "s1", "day_open", "week_open"):
        f[f"d_{k}"] = (c - f[k]) / a

    # ---- patterns
    cs = candlesticks(df, a)
    f = pd.concat([f, cs], axis=1)
    pp = pivot_patterns(df, ms[["ph", "pl", "ph_pos", "pl_pos"]], f["rsi"], a)
    f = pd.concat([f, pp], axis=1)

    # ---- higher timeframes
    f = pd.concat([f, _htf_context(df, "4h", "h4_", pd.Timedelta("4h"))], axis=1)
    f = pd.concat([f, _htf_context(df, "1D", "d1_", pd.Timedelta("1D"))], axis=1)

    # ---- intermarket / macro / positioning / seasonality
    f["smt"] = _smt(df, peer, inst.smt_inverse)
    if peer is not None and not peer.empty:
        pc_ = peer["close"].reindex(df.index).ffill(limit=3)
        pr = np.log(pc_).diff()
        f["peer_corr"] = f["ret1"].rolling(120, min_periods=60).corr(pr)
        f["peer_ret24"] = np.log(pc_).diff(24) / (pr.rolling(500, min_periods=100).std() * np.sqrt(24))
    else:
        f["peer_corr"] = np.nan
        f["peer_ret24"] = np.nan
    f = pd.concat([f, _macro(df, macro, inst), _cot(df, cot), _seasonality(df)], axis=1)
    f = f.loc[:, ~f.columns.duplicated()]
    if return_state:
        return f, state
    return f


# features fed to ML (stationary / scale-free only)
NON_FEATURES = {
    "open", "high", "low", "close", "atr", "ema20", "ema50", "ema100", "ema200", "st_line",
    "tenkan", "kijun", "span_a", "span_b", "kama", "bb_mid", "bb_up", "bb_lo", "don_hi",
    "don_lo", "vwap", "last_sh", "last_sl", "prev_sh", "prev_sl", "eqh", "eql", "ph", "pl",
    "ph_pos", "pl_pos", "pdh", "pdl", "pdc", "day_open", "pwh", "pwl", "week_open",
    "asia_high", "asia_low", "pivot", "r1", "s1", "r2", "s2", "sup", "res", "poc", "vah",
    "val", "harmonic_name", "elliott_label", "h4_atr", "d1_atr", "bos_up", "bos_dn",
    "choch_up", "choch_dn",
}


def ml_columns(f: pd.DataFrame) -> list[str]:
    cols = []
    for col in f.columns:
        if col in NON_FEATURES or (col.startswith("sweep_") and col != "sweep_score") \
                or col in ("ml_p_long", "ml_p_short", "ml_score"):
            continue
        if f[col].dtype == object:
            continue
        cols.append(col)
    return cols
